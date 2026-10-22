import asyncio
import base64
import json
import random
import tempfile
import unittest
import uuid
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from pet_daemon.application import Application
from pet_daemon.attention import Attention, AttentionSelection
from pet_daemon.behavior.controller import Behavior
from pet_daemon.behavior.sprites import Pack
from pet_daemon.commentary import Cognition
from pet_daemon.commentary.contracts import ModelRequest
from pet_daemon.config import Config
from pet_daemon.delivery import DeliveryOutcome, SpeechPacket
from pet_daemon.domain import (
    ArtifactKind,
    DeliveryStatus,
    EvidenceSource,
    ExecutionOutcome,
    LinkRelationship,
    ModelPurpose,
    PullRequestState,
    SpeechReason,
)
from pet_daemon.runtime import HookEvent, OverlayPacket
from pet_daemon.sessions.processes import ProcessTracker
from pet_daemon.sessions.reactions import SessionReactions
from pet_daemon.sessions.tracking import Sessions
from pet_daemon.sources import RepositoryTracker
from pet_daemon.sources.hooks import normalize_hook
from pet_daemon.storage import Memory
from pet_daemon.work import WorkStory
from pet_daemon.work.models import RelatedLink
from pet_daemon.work.observations import (
    BranchObservation,
    CheckoutObservation,
    GitObservation,
    PullRequest,
    RepositoryObservation,
)
from Tests.fixtures import require


class FixtureProvider:
    def __init__(self) -> None:
        self.emit: Callable[[HookEvent | RepositoryObservation], object] | None = None
        self.paused = False

    def publish(self, observation: HookEvent | RepositoryObservation) -> None:
        assert self.emit is not None
        self.emit(observation)

    def tick(self) -> None:
        pass

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False

    def clear(self) -> None:
        pass

    async def close(self) -> None:
        pass


class FixtureModel:
    def __init__(self) -> None:
        self.requests = []
        self.text: str | None = (
            "The parser preserves escaped delimiters inside quoted fields."
        )

    async def complete(
        self, request: ModelRequest, reason: SpeechReason | ModelPurpose
    ) -> dict[str, object]:
        self.requests.append((request, reason))
        return {"speak": self.text}


class ApplicationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.now = 1000.0
        self.clock = lambda: self.now
        self.config = Config(pet_dir=Path(self.temporary.name), summarize_on=[])
        self.config.behavior.sentiment_reactions = False
        self.memory = Memory(
            Path(self.temporary.name) / "memory.sqlite3",
            "fixture-pet",
            clock=self.clock,
        )
        self.model = FixtureModel()
        self.provider = FixtureProvider()
        self.packets: list[OverlayPacket] = []
        self.attention = Attention(self.config, self.clock, random.Random(0))
        directory = Path(self.temporary.name)
        (directory / "pixel.png").write_bytes(
            base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII="
            )
        )
        (directory / "manifest.json").write_text(
            json.dumps(
                {
                    "pet": "fixture",
                    "rest": "idle",
                    "clips": {
                        "idle": {"frames": ["pixel.png"], "fps": 1, "role": "rest"}
                    },
                }
            )
        )
        behavior = Behavior(
            Pack(directory),
            self.config.behavior,
            self.clock,
            random.Random(0),
            self.packets.append,
        )
        behavior.renderer_status(
            {
                "x": 100,
                "y": 100,
                "width": 200,
                "height": 160,
                "screen": {"x": 0, "y": 0, "width": 600, "height": 400},
                "pointer": {"x": 2000, "y": 2000},
                "dragging": False,
                "menu": False,
                "speech": False,
                "sleeping": False,
                "system_sleeping": False,
            }
        )
        self.app = Application(
            self.config,
            sessions=Sessions(self.clock),
            reactions=SessionReactions(),
            attention=self.attention,
            behavior=behavior,
            cognition=Cognition(
                self.model, self.config, memory=self.memory, clock=self.clock
            ),
            providers=(self.provider,),
            processes=ProcessTracker(clock=self.clock),
            memory=self.memory,
            emit=self.packets.append,
            clock=self.clock,
            rng=random.Random(0),
        )
        self.provider.emit = self.app.ingest
        hook = normalize_hook(
            {
                "kind": "event",
                "harness": "codex",
                "session": "parser",
                "event": "intent",
                "goal": "Improve quoted-field parsing",
                "repo": "/fixture/parser",
            }
        )
        self.provider.publish(require(hook))
        self.checkout = CheckoutObservation(
            "/fixture/parser",
            GitObservation(
                "/fixture/parser",
                "parser",
                "head-one",
                0,
                0,
                0,
                0,
                "origin/main",
                1,
                0,
                "example.test/team/parser",
            ),
            self.now,
        )
        self.observation = BranchObservation(
            "example.test/team/parser",
            "parser",
            PullRequest(
                23,
                "https://example.test/team/parser/pull/23",
                PullRequestState.OPEN,
                None,
                head="head-one",
                head_repository="example.test/team/parser",
                head_branch="parser",
                repository="example.test/team/parser",
                title="Handle quoted fields",
            ),
            self.now,
        )
        self.provider.publish(self.checkout)
        self.provider.publish(self.observation)
        self.attention.pending = None

    async def asyncTearDown(self) -> None:
        await self.app.close()
        self.memory.close()
        self.temporary.cleanup()

    def report(self, text: str, *, session: str = "parser") -> None:
        self.provider.publish(HookEvent("codex", session, "progress", progress=text))

    async def submit_commentary(self) -> tuple[SpeechPacket, AttentionSelection]:
        snapshot = self.app.context()
        selection = self.attention.select(snapshot, SpeechReason.ROUTINE)
        await self.app.comment(
            SpeechReason.ROUTINE, snapshot=snapshot, selection=selection
        )
        packet = next(
            packet for packet in reversed(self.packets) if "delivery_id" in packet
        )
        uuid.UUID(packet["delivery_id"])
        return {
            "text": packet["text"],
            "ttl": packet["ttl"],
            "delivery_id": packet["delivery_id"],
        }, selection

    def accept(self, packet: SpeechPacket) -> None:
        self.app.receive_renderer(
            {
                "kind": "renderer",
                "event": "speech_accepted",
                "delivery_id": packet["delivery_id"],
            }
        )

    async def test_failed_intent_write_keeps_the_original_story_after_the_next_report(
        self,
    ) -> None:
        failed_intent = "Replace the parser with an unrelated formatter"
        self.memory.db.execute(
            "CREATE TRIGGER reject_session_write BEFORE UPDATE ON work_sessions BEGIN SELECT RAISE(ABORT, 'fixture session failure'); END"
        )
        with self.assertLogs("pet-daemon", level="WARNING"):
            self.provider.publish(
                HookEvent(
                    "codex",
                    "parser",
                    "intent",
                    goal=failed_intent,
                    repo="/fixture/rejected-parser",
                )
            )
        self.memory.db.execute("DROP TRIGGER reject_session_write")
        self.report("Quoted fields preserve escaped delimiters.")
        await self.submit_commentary()
        payload = json.loads(self.model.requests[-1][0]["messages"][0]["content"])
        work = payload["focus_work"]
        self.assertEqual(work["intent"], "Improve quoted-field parsing")
        self.assertIn("Quoted fields preserve escaped delimiters.", json.dumps(work))
        self.assertNotIn(failed_intent, json.dumps(work))

    async def test_late_pr_resolution_keeps_a_report_on_its_work_after_branch_reuse(
        self,
    ) -> None:
        self.now += 1
        self.provider.publish(replace(self.observation, pr=None, checked_at=self.now))
        self.provider.publish(
            HookEvent(
                "codex",
                "parser",
                "turn_done",
                outcome=ExecutionOutcome.SUCCESS,
                progress="The parser changes are ready.",
            )
        )
        reported_pr = replace(
            require(self.observation.pr),
            number=24,
            url="https://example.test/team/parser/pull/24",
        )
        self.now += 1
        self.provider.publish(
            replace(self.observation, pr=reported_pr, checked_at=self.now)
        )
        unrelated_pr = replace(
            reported_pr,
            number=25,
            url="https://example.test/team/parser/pull/25",
            head="unrelated-head",
            title="Add unrelated import support",
        )
        self.now += 1
        self.provider.publish(
            replace(
                self.checkout,
                local=replace(require(self.checkout.local), head=unrelated_pr.head),
                checked_at=self.now,
            )
        )
        self.provider.publish(
            replace(self.observation, pr=unrelated_pr, checked_at=self.now)
        )
        await self.submit_commentary()
        payload = json.loads(self.model.requests[-1][0]["messages"][0]["content"])
        work = payload["focus_work"]
        report = next(
            report
            for report in work["activity"]
            if report.get("reported_progress") == "The parser changes are ready."
        )
        self.assertEqual(report["context"]["pull_request"], reported_pr.url)
        artifacts = {artifact["ref"] for artifact in work["artifacts"]}
        self.assertIn(reported_pr.url, artifacts)
        self.assertNotIn(unrelated_pr.url, artifacts)

    async def test_repeated_progress_about_another_branch_becomes_commentary_for_its_pr(
        self,
    ) -> None:
        self.report("Tests pass")
        packet, _ = await self.submit_commentary()
        first_payload = json.loads(self.model.requests[-1][0]["messages"][0]["content"])
        first_report = next(
            report
            for report in first_payload["focus_work"]["activity"]
            if report.get("reported_progress") == "Tests pass"
        )
        self.assertEqual(
            first_report["context"]["pull_request"],
            require(self.observation.pr).url,
        )
        self.accept(packet)
        self.now += 1
        next_pr = replace(
            require(self.observation.pr),
            number=24,
            url="https://example.test/team/parser/pull/24",
            head_branch="parser-examples",
            title="Document quoted-field examples",
        )
        self.provider.publish(
            replace(
                self.checkout,
                local=replace(require(self.checkout.local), branch=next_pr.head_branch),
                checked_at=self.now,
            )
        )
        self.provider.publish(
            BranchObservation(
                self.observation.repository, next_pr.head_branch, next_pr, self.now
            )
        )
        self.attention.next_observation = self.now
        self.assertIsNone(self.attention.request(self.app.context()))
        self.report("Tests pass")
        self.attention.next_observation = self.now
        snapshot = self.app.context()
        request = require(self.attention.request(snapshot))
        selection = self.attention.select(snapshot, request.reason)
        self.model.text = "The quoted-field examples pass their validation."
        await self.app.comment(request.reason, snapshot=snapshot, selection=selection)
        payload = json.loads(self.model.requests[-1][0]["messages"][0]["content"])
        latest_report = next(
            report
            for report in payload["focus_work"]["activity"]
            if report.get("reported_progress") == "Tests pass"
        )
        self.assertEqual(
            latest_report["context"],
            {"branch": next_pr.head_branch, "pull_request": next_pr.url},
        )

    async def test_repeated_progress_about_the_same_work_keeps_automatic_speech_quiet(
        self,
    ) -> None:
        self.report("Tests pass")
        packet, _ = await self.submit_commentary()
        self.accept(packet)
        self.now += 1
        self.report("Tests pass")
        self.attention.next_observation = self.now
        self.assertIsNone(self.attention.request(self.app.context()))

    async def test_parent_and_child_reports_share_stack_context_and_one_accepted_speech(
        self,
    ) -> None:
        parent_pr = require(self.observation.pr)
        child_pr = replace(
            parent_pr,
            number=24,
            url="https://example.test/team/parser/pull/24",
            head="child-head",
            head_branch="parser-examples",
            title="Document escaped delimiter examples",
            related_links=(
                RelatedLink(
                    ArtifactKind.PULL_REQUEST,
                    parent_pr.url,
                    LinkRelationship.STACK_ENTRY,
                    EvidenceSource.GITHUB_STACK,
                    stack=4,
                    position=1,
                ),
            ),
        )
        self.provider.publish(
            replace(
                self.observation,
                pr=replace(
                    parent_pr,
                    related_links=(
                        RelatedLink(
                            ArtifactKind.PULL_REQUEST,
                            child_pr.url,
                            LinkRelationship.STACK_ENTRY,
                            EvidenceSource.GITHUB_STACK,
                            stack=4,
                            position=2,
                        ),
                    ),
                ),
            )
        )
        parent_report = "Quoted fields preserve escaped delimiters. Next I will review the examples."
        child_report = (
            "Documented escaped delimiters. Next I will add a multiline example."
        )
        self.report(parent_report)
        self.provider.publish(
            HookEvent(
                "codex",
                "examples",
                "subagent_start",
                parent="parser",
                goal="Document quoted-field examples",
                repo="/fixture/parser-examples",
            )
        )
        self.provider.publish(
            CheckoutObservation(
                "/fixture/parser-examples",
                replace(
                    require(self.checkout.local),
                    worktree="/fixture/parser-examples",
                    branch="parser-examples",
                    head="child-head",
                ),
                self.now,
            )
        )
        self.provider.publish(
            BranchObservation(
                self.observation.repository,
                "parser-examples",
                child_pr,
                self.now,
            )
        )
        self.report(child_report, session="examples")
        subjects = self.app.context().work
        self.assertEqual(len(subjects), 1)
        self.assertEqual(subjects[0].sessions, ("codex:examples", "codex:parser"))
        self.assertIsInstance(subjects[0].story, WorkStory)
        self.assertEqual(subjects[0].story.intent, "Improve quoted-field parsing")
        packet, selection = await self.submit_commentary()
        payload = json.loads(self.model.requests[0][0]["messages"][0]["content"])
        work = payload["focus_work"]
        self.assertIn(parent_report, json.dumps(work["activity"]))
        self.assertNotIn(child_report, json.dumps(work["activity"]))
        child_session = next(
            session
            for session in work["sessions"]
            if session["scope"] == "Document quoted-field examples"
        )
        self.assertIn(child_report, json.dumps(child_session["activity"]))
        self.assertNotIn(parent_report, json.dumps(child_session["activity"]))
        self.assertIn("Handle quoted fields", json.dumps(work))
        self.assertIn("Document escaped delimiter examples", json.dumps(work))
        self.assertIn("stack", json.dumps(work).lower())
        self.assertEqual(self.memory.speech(), [])
        self.assertTrue(self.attention.is_new(require(selection.subject)))
        self.assertTrue(self.app.delivery.has_pending())
        self.accept(packet)
        self.accept(packet)
        outcome = require(self.memory.message_outcome(packet["delivery_id"]))
        self.assertEqual(
            (outcome.outcome, outcome.subject_id, outcome.revision),
            ("accepted", selection.id, require(selection.subject).revision),
        )
        self.assertEqual(
            [item.text for item in self.memory.speech()], [self.model.text]
        )
        self.assertFalse(self.app.delivery.has_pending())
        self.assertFalse(self.attention.is_new(self.app.context().work[0]))

    async def test_child_location_comes_from_observed_hooks_and_parent_is_canonical(
        self,
    ) -> None:
        self.provider.publish(
            HookEvent("codex", "parser", "progress", aliases=("parser-alias",))
        )
        self.provider.publish(
            HookEvent(
                "codex",
                "examples",
                "subagent_start",
                parent="parser-alias",
                goal="Document quoted-field examples",
            )
        )
        self.assertEqual(
            self.app.sessions.by_id["codex:examples"].repo, "/fixture/parser"
        )
        subject = self.app.context().work[0]
        self.assertEqual(subject.id, "codex:parser")
        self.assertTrue({"codex:examples", "codex:parser"} <= set(subject.sessions))
        self.assertEqual(require(subject.story.sessions[0].context).checkout, "")
        self.provider.publish(
            HookEvent(
                "codex",
                "examples",
                "progress",
                repo="/fixture/parser-examples",
                progress="Documented the quoted-field examples.",
            )
        )
        child = self.app.context().work[0].story.sessions[0]
        self.assertEqual(require(child.context).checkout, "/fixture/parser-examples")
        self.assertEqual(len(self.app.context().work), 1)

    async def test_pending_delivery_defers_other_work_and_queued_automatic_generation(
        self,
    ) -> None:
        self.config.behavior.automatic_speech_cooldown_seconds = 0
        self.config.observation_seconds = (1, 1)
        self.config.idle_message_seconds = 1
        await self.submit_commentary()
        self.provider.publish(
            HookEvent("codex", "other", "intent", goal="Document the renderer")
        )
        self.app.queue(SpeechReason.PROMPT)
        self.attention.next_observation = self.now
        self.now += 1
        self.app.tick()
        await asyncio.sleep(0)
        self.assertEqual(len(self.model.requests), 1)
        self.assertIsNone(self.app.task)
        self.assertIsNotNone(self.attention.pending)

    async def test_cancelled_delivery_stays_eligible_after_attention_restore(
        self,
    ) -> None:
        packet, selection = await self.submit_commentary()
        self.app.preempt()
        restored = Attention(
            self.config,
            self.clock,
            random.Random(0),
            self.memory.load_attention_history(),
        )
        self.now += max(self.config.observation_seconds)
        restored.next_observation = self.now
        self.assertTrue(restored.is_new(require(selection.subject)))
        self.assertIsNotNone(
            restored.request(
                self.app.context(), delivery_pending=self.app.delivery.has_pending()
            )
        )
        self.assertIsNone(self.memory.message_outcome(packet["delivery_id"]))
        self.assertEqual(self.memory.speech(), [])

    async def test_silence_and_exhausted_delivery_persist_consideration(self) -> None:
        self.model.text = None
        await self.app.comment(SpeechReason.ROUTINE)
        restored = Attention(
            self.config,
            self.clock,
            random.Random(0),
            self.memory.load_attention_history(),
        )
        self.assertFalse(restored.is_new(self.app.context().work[0]))
        self.report("Added examples for escaped delimiters.")
        self.model.text = "The examples cover escaped delimiters."
        packet, _ = await self.submit_commentary()
        self.now += 2
        self.app.delivery.tick()
        self.now += 2
        self.app.delivery.tick()
        self.assertFalse(self.app.delivery.has_pending())
        self.assertEqual(
            require(self.memory.message_outcome(packet["delivery_id"])).outcome,
            "failed",
        )
        restored = Attention(
            self.config,
            self.clock,
            random.Random(0),
            self.memory.load_attention_history(),
        )
        self.assertFalse(restored.is_new(self.app.context().work[0]))

    async def test_late_acceptance_records_captured_story_without_suppressing_new_report(
        self,
    ) -> None:
        packet, selection = await self.submit_commentary()
        self.app.preempt()
        self.report("The parser examples are documented; next I will update the guide.")
        self.accept(packet)
        outcome = require(self.memory.message_outcome(packet["delivery_id"]))
        self.assertEqual(outcome.revision, require(selection.subject).revision)
        self.assertTrue(self.attention.is_new(self.app.context().work[0]))
        self.assertEqual(len(self.memory.speech()), 1)

    async def test_reset_rejects_old_acceptance_and_callbacks(self) -> None:
        packet, _ = await self.submit_commentary()
        message = self.app.delivery.deliveries[packet["delivery_id"]].message
        self.app.clear_memory()
        self.accept(packet)
        self.app.delivery_outcome(DeliveryOutcome(message, DeliveryStatus.ACCEPTED))
        self.assertIsNone(self.memory.message_outcome(packet["delivery_id"]))
        self.assertEqual(self.memory.speech(), [])
        self.assertEqual(self.app.context().work, ())

    async def test_failed_acceptance_transaction_keeps_delivery_and_attention_retryable(
        self,
    ) -> None:
        packet, _ = await self.submit_commentary()
        self.memory.db.execute(
            "CREATE TRIGGER reject_speech BEFORE INSERT ON speech BEGIN SELECT RAISE(ABORT, 'fixture speech failure'); END"
        )
        with self.assertLogs("pet-daemon", level="WARNING"):
            self.accept(packet)
        self.assertIsNone(self.memory.message_outcome(packet["delivery_id"]))
        self.assertEqual(self.memory.speech(), [])
        self.assertTrue(self.attention.is_new(self.app.context().work[0]))
        self.assertTrue(self.app.delivery.has_pending())
        self.memory.db.execute("DROP TRIGGER reject_speech")
        self.accept(packet)
        self.assertFalse(self.attention.is_new(self.app.context().work[0]))

    async def test_basic_pr_state_refresh_preserves_consideration(self) -> None:
        packet, _ = await self.submit_commentary()
        self.accept(packet)
        self.provider.publish(
            replace(
                self.observation,
                pr=replace(
                    require(self.observation.pr),
                    state="MERGED",
                    merged_at="2026-10-20T00:00:00Z",
                ),
            )
        )
        subject = self.app.context().work[0]
        self.assertFalse(self.attention.is_new(subject))
        self.attention.next_observation = self.now
        self.assertIsNone(self.attention.request(self.app.context()))
        self.assertIn("MERGED", str(subject.story))

    async def test_reconciliation_retains_successful_pr_facts_after_source_failure(
        self,
    ) -> None:
        failed = replace(
            self.observation,
            pr=None,
            checked_at=self.now + 1,
            error="GitHub unavailable",
        )
        self.provider.publish(failed)
        source = RepositoryTracker(
            self.app.ingest,
            clock=self.clock,
            checkpoints=(self.checkout, failed),
        )
        self.app.providers = (source,)
        self.app.reconcile_work = lambda: source.reconcile_targets(
            self.memory.collection_targets()
        )
        self.app.reconcile_targets()
        self.assertIn("Handle quoted fields", str(self.app.context().work[0].story))
        checkpoints = self.memory.load_repositories()
        branch = next(
            item for item in checkpoints if isinstance(item, BranchObservation)
        )
        self.assertEqual(branch.error, "GitHub unavailable")
        self.assertEqual(require(branch.pr).title, "Handle quoted fields")
        checkout = next(
            item for item in checkpoints if isinstance(item, CheckoutObservation)
        )
        self.assertEqual(checkout.local, self.checkout.local)
        self.assertEqual(checkout.last_success_at, self.checkout.checked_at)
        self.assertEqual(branch.last_success_at, self.observation.checked_at)

    async def test_failed_reset_preserves_history_and_sessions(self) -> None:
        self.memory.record_speech("Earlier parser observation", SpeechReason.ROUTINE)
        work = self.app.context().work
        sessions = self.app.sessions
        self.memory.db.execute(
            "CREATE TRIGGER reject_memory_reset BEFORE DELETE ON events BEGIN SELECT RAISE(ABORT, 'fixture reset failure'); END"
        )
        with self.assertLogs("pet-daemon", level="WARNING"):
            self.app.clear_memory()
        self.assertEqual(self.app.context().work, work)
        self.assertEqual(
            [item.text for item in self.memory.speech()], ["Earlier parser observation"]
        )
        self.assertIs(self.app.sessions, sessions)
        self.assertEqual(self.app.history_epoch, 0)

    async def test_failed_delivery_diagnostics_do_not_prevent_late_acceptance(
        self,
    ) -> None:
        packet, _ = await self.submit_commentary()
        self.memory.db.execute(
            "CREATE TRIGGER reject_delivery_failure BEFORE INSERT ON message_outcomes WHEN NEW.outcome = 'failed' BEGIN SELECT RAISE(ABORT, 'fixture diagnostic failure'); END"
        )
        self.now += 2
        self.app.delivery.tick()
        self.now += 2
        with self.assertLogs("pet-daemon", level="WARNING"):
            self.app.delivery.tick()
        self.assertFalse(self.app.delivery.has_pending())
        self.accept(packet)
        self.assertEqual(
            require(self.memory.message_outcome(packet["delivery_id"])).outcome,
            "accepted",
        )
        self.assertEqual(len(self.memory.speech()), 1)
