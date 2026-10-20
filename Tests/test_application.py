from dataclasses import replace
import json
from pathlib import Path
import random
import tempfile
import unittest
import uuid

from pet_daemon.application import Application
from pet_daemon.attention import Attention
from pet_daemon.commentary import Cognition
from pet_daemon.config import Config
from pet_daemon.delivery import DeliveryOutcome
from pet_daemon.episodes import Episodes
from pet_daemon.sessions import Sessions
from pet_daemon.sources import RepositoryTracker
from pet_daemon.sources.hooks import normalize_hook
from pet_daemon.storage import Memory
from pet_daemon.work import WorkView
from pet_daemon.work.observations import GitObservation, PullRequest, RepositoryObservation


class FixtureProvider:
    def __init__(self):
        self.emit = None
        self.targets = ()
        self.paused = False

    def publish(self, observation):
        self.emit(observation)

    def associate(self, session, path, *, refresh=False):
        pass

    def reconcile_targets(self, targets):
        self.targets = tuple(targets)

    def tick(self):
        pass

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def clear(self):
        self.targets = ()

    async def close(self):
        pass


class FixtureModel:
    def __init__(self):
        self.requests = []
        self.text = "The parser pull request requests your review."

    async def complete(self, request, reason):
        self.requests.append((request, reason))
        return {"speak": self.text}


class FixtureBehavior:
    sleeping = False
    status = True
    care_remark = None

    def wake(self, reason):
        pass

    def stop_movement(self):
        pass

    def tick(self, activity, sessions):
        pass

    def react(self, kind, sessions, session=None):
        pass


class FixtureProcesses:
    async def refresh(self, identities):
        return {}


class ApplicationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.now = 1000.0
        self.clock = lambda: self.now
        self.config = Config(pet_dir=Path(self.temporary.name), summarize_on=[])
        self.config.behavior.sentiment_reactions = False
        self.memory = Memory(Path(self.temporary.name) / "memory.sqlite3", "fixture-pet", clock=self.clock)
        self.model = FixtureModel()
        self.provider = FixtureProvider()
        self.packets = []
        self.attention = Attention(self.config, self.clock, random.Random(0))
        self.app = Application(self.config, sessions=Sessions(self.clock), episodes=Episodes(),
                               attention=self.attention, behavior=FixtureBehavior(),
                               cognition=Cognition(self.model, self.config, memory=self.memory, clock=self.clock),
                               providers=(self.provider,), processes=FixtureProcesses(), memory=self.memory,
                               emit=self.packets.append, clock=self.clock, rng=random.Random(0))
        self.provider.emit = self.app.ingest
        hook = normalize_hook({"kind": "event", "harness": "codex", "session": "parser",
                               "event": "intent", "goal": "Improve quoted-field parsing", "repo": "/fixture/parser"})
        self.app.ingest_hook(hook)
        self.observation = RepositoryObservation(
            "fixture:parser", "/fixture/parser", "parser", sessions=("codex:parser",),
            local=GitObservation("/fixture/parser", "parser", "head-one", 0, 0, 0, 0, "origin/main", 1, 0,
                                 "example.test/team/parser"),
            pr=PullRequest(23, "https://example.test/team/parser/pull/23", "OPEN", None, "REVIEW_REQUIRED", (),
                           head="head-one", head_repository="example.test/team/parser", head_branch="parser",
                           repository="example.test/team/parser", check_revision="head-one", review_requested=True,
                           title="Handle quoted fields"),
            revision=1, changed_at=self.now, local_checked_at=self.now, github_checked_at=self.now,
            repository="example.test/team/parser", source="fixture")
        self.provider.publish(self.observation)

    async def asyncTearDown(self):
        await self.app.close()
        self.memory.close()
        self.temporary.cleanup()

    async def submit_review_commentary(self):
        snapshot = self.app.snapshot()
        self.assertIsInstance(snapshot["work"][0].view, WorkView)
        selection = self.attention.select(snapshot, "routine")
        self.assertEqual(selection.action.kind, "review_output")
        await self.app.observe("routine", snapshot=snapshot, selection=selection)
        packet = next(packet for packet in reversed(self.packets) if "delivery_id" in packet)
        self.assertEqual(str(uuid.UUID(packet["delivery_id"])), packet["delivery_id"])
        return packet, selection.action

    def accept(self, packet):
        self.app.receive_renderer({"kind": "renderer", "event": "speech_accepted", "delivery_id": packet["delivery_id"]})

    async def test_fixture_observation_reaches_model_renderer_and_accepted_surfacing(self):
        packet, action = await self.submit_review_commentary()
        request, reason = self.model.requests[0]
        payload = json.loads(request["messages"][0]["content"])
        self.assertEqual(reason, "routine")
        self.assertEqual(payload["focus_opportunity"]["kind"], "review_output")
        self.assertIn("Handle quoted fields", json.dumps(payload))
        self.assertEqual(packet["text"], self.model.text)
        self.assertEqual(self.memory.speech(), [])
        self.assertIsNone(self.memory.message_outcome(packet["delivery_id"]))
        self.accept(packet)
        self.accept(packet)
        outcome = self.memory.message_outcome(packet["delivery_id"])
        self.assertEqual((outcome["outcome"], outcome["opportunity_id"], outcome["version"]),
                         ("accepted", action.id, action.version))
        self.assertEqual([item["text"] for item in self.memory.speech()], [self.model.text])
        self.assertEqual(self.memory.work.opportunities(), ())
        surfacings = self.memory.work.repository.load().surfacings
        self.assertEqual([(item.opportunity_id, item.version, item.outcome) for item in surfacings],
                         [(action.id, action.version, "accepted")])
        self.assertTrue(any(target.kind == "pull_request" and target.ref == self.observation.pr.url
                            for target in self.provider.targets))

    async def test_late_renderer_acceptance_records_the_message_after_preemption(self):
        packet, action = await self.submit_review_commentary()
        self.app.preempt()
        self.accept(packet)
        outcome = self.memory.message_outcome(packet["delivery_id"])
        self.assertEqual((outcome["outcome"], outcome["version"]), ("accepted", action.version))
        self.assertEqual(len(self.memory.speech()), 1)
        self.assertEqual(self.memory.work.opportunities(), ())

    async def test_late_acceptance_of_an_earlier_version_keeps_changed_work_eligible(self):
        packet, action = await self.submit_review_commentary()
        self.now += 1
        updated = replace(self.observation, revision=2, changed_at=self.now, local_checked_at=self.now,
                          github_checked_at=self.now, local=replace(self.observation.local, head="head-two"),
                          pr=replace(self.observation.pr, head="head-two", check_revision="head-two"))
        self.provider.publish(updated)
        changed = self.memory.work.opportunity(action.id)
        self.assertGreater(changed.version, action.version)
        self.app.preempt()
        self.accept(packet)
        self.assertEqual(self.memory.message_outcome(packet["delivery_id"])["version"], action.version)
        self.assertEqual(self.memory.work.opportunity(action.id).version, changed.version)
        surfacings = self.memory.work.repository.load().surfacings
        self.assertEqual([(item.version, item.outcome) for item in surfacings], [(action.version, "accepted")])

    async def test_memory_reset_rejects_old_renderer_acknowledgments_and_callbacks(self):
        packet, _ = await self.submit_review_commentary()
        message = self.app.delivery.deliveries[packet["delivery_id"]].message
        self.app.clear_memory()
        self.accept(packet)
        self.app.delivery_outcome(DeliveryOutcome(message, "accepted"))
        self.assertIsNone(self.memory.message_outcome(packet["delivery_id"]))
        self.assertEqual(self.memory.speech(), [])
        self.assertEqual(self.memory.work.work_items(), ())

    async def test_failed_acceptance_transaction_preserves_retry_and_work_eligibility(self):
        packet, action = await self.submit_review_commentary()
        self.memory.db.execute("""CREATE TRIGGER reject_surfacing BEFORE INSERT ON work_surfacings
                                  BEGIN SELECT RAISE(ABORT, 'fixture surfacing failure'); END""")
        with self.assertLogs("pet-daemon", level="WARNING"):
            self.accept(packet)
        self.assertIsNone(self.memory.message_outcome(packet["delivery_id"]))
        self.assertEqual(self.memory.speech(), [])
        self.assertEqual(self.memory.work.opportunity(action.id).version, action.version)
        self.memory.db.execute("DROP TRIGGER reject_surfacing")
        self.accept(packet)
        self.assertEqual(self.memory.message_outcome(packet["delivery_id"])["outcome"], "accepted")
        self.assertEqual(len(self.memory.speech()), 1)
        self.assertEqual(self.memory.work.opportunities(), ())

    async def test_reconciling_collection_targets_preserves_failed_observation_evidence(self):
        failed = replace(self.observation, github_error="GitHub unavailable")
        self.provider.publish(failed)
        source = RepositoryTracker(self.app.ingest, clock=self.clock, checkpoints=(failed,))
        self.app.providers = (source,)
        self.app.reconcile_targets()
        artifacts = self.memory.work.session_context("codex:parser")["artifacts"]
        branch = next(item for item in artifacts if item["kind"] == "branch")
        availability = next(item for item in branch["evidence"] if item["source"] == "observer")
        self.assertEqual(availability["data"]["github_error"], "GitHub unavailable")
        self.assertEqual(len(self.memory.load_repositories()), 1)
        self.assertGreater(len(source.snapshot()), 1)

    async def test_failed_reset_preserves_work_history_and_running_state(self):
        self.memory.record_speech("Earlier parser observation", "routine")
        work = self.memory.work.work_items()
        checkpoints = self.memory.load_repositories()
        targets = self.provider.targets
        sessions = self.app.sessions
        self.memory.db.execute("""CREATE TRIGGER reject_memory_reset BEFORE DELETE ON events
                                  BEGIN SELECT RAISE(ABORT, 'fixture reset failure'); END""")
        with self.assertLogs("pet-daemon", level="WARNING"):
            self.app.clear_memory()
        self.assertEqual(self.memory.work.work_items(), work)
        self.assertEqual(self.memory.load_repositories(), checkpoints)
        self.assertEqual([item["text"] for item in self.memory.speech()], ["Earlier parser observation"])
        self.assertEqual(self.provider.targets, targets)
        self.assertIs(self.app.sessions, sessions)
        self.assertEqual(self.app.history_epoch, 0)

    async def test_failed_delivery_diagnostics_do_not_prevent_late_acceptance(self):
        packet, _ = await self.submit_review_commentary()
        self.memory.db.execute("""CREATE TRIGGER reject_delivery_failure BEFORE INSERT ON message_outcomes
                                  WHEN NEW.outcome = 'failed'
                                  BEGIN SELECT RAISE(ABORT, 'fixture diagnostic failure'); END""")
        self.now += 2
        self.app.delivery.tick()
        self.now += 2
        with self.assertLogs("pet-daemon", level="WARNING"):
            self.app.delivery.tick()
        self.assertEqual(self.app.delivery.deliveries[packet["delivery_id"]].status, "failed")
        self.assertIsNone(self.memory.message_outcome(packet["delivery_id"]))
        self.accept(packet)
        self.assertEqual(self.memory.message_outcome(packet["delivery_id"])["outcome"], "accepted")
        self.assertEqual(len(self.memory.speech()), 1)

    async def test_silent_repetition_retry_records_the_refreshed_work_revision(self):
        self.memory.record_speech(self.model.text, "fixture")
        requests = self.model.requests
        initial_text = self.model.text
        async def complete(request, reason):
            requests.append((request, reason))
            if len(requests) == 1:
                self.now += 1
                updated = replace(self.observation, revision=2, github_checked_at=self.now,
                                  pr=replace(self.observation.pr, body="Explain quoted-field cases."))
                self.provider.publish(updated)
                return {"speak": initial_text}
            return {"speak": None}
        self.model.complete = complete
        await self.app.observe("routine")
        self.assertEqual(len(requests), 2)
        self.assertFalse(self.attention.is_new(self.app.snapshot()["work"][0]))
        self.assertFalse(any("delivery_id" in packet for packet in self.packets))
