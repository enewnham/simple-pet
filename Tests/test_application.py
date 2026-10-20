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
from pet_daemon.behavior import Behavior
from pet_daemon.commentary import Cognition
from pet_daemon.commentary.contracts import ModelRequest
from pet_daemon.config import Config
from pet_daemon.delivery import DeliveryOutcome, SpeechPacket
from pet_daemon.episodes import Episodes
from pet_daemon.processes import ProcessTracker
from pet_daemon.runtime import HookEvent, OverlayPacket
from pet_daemon.sessions import Sessions
from pet_daemon.sources import RepositoryTracker
from pet_daemon.sources.hooks import normalize_hook
from pet_daemon.sprites import Pack
from pet_daemon.storage import Memory
from pet_daemon.work import WorkView
from pet_daemon.work.models import AvailabilityEvidence
from pet_daemon.work.observations import (
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

    async def complete(self, request: ModelRequest, reason: str) -> dict[str, object]:
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
            episodes=Episodes(),
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
        self.observation = RepositoryObservation(
            "fixture:parser",
            "/fixture/parser",
            "parser",
            sessions=("codex:parser",),
            pr_sessions=("codex:parser",),
            local=GitObservation(
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
            pr=PullRequest(
                23,
                "https://example.test/team/parser/pull/23",
                "OPEN",
                None,
                head="head-one",
                head_repository="example.test/team/parser",
                head_branch="parser",
                repository="example.test/team/parser",
                title="Handle quoted fields",
            ),
            revision=1,
            changed_at=self.now,
            local_checked_at=self.now,
            github_checked_at=self.now,
            repository="example.test/team/parser",
            source="fixture",
        )
        self.provider.publish(self.observation)
        self.attention.pending = None

    async def asyncTearDown(self) -> None:
        await self.app.close()
        self.memory.close()
        self.temporary.cleanup()

    def report(self, text: str, *, session: str = "parser") -> None:
        self.provider.publish(HookEvent("codex", session, "progress", progress=text))

    async def submit_commentary(self) -> tuple[SpeechPacket, AttentionSelection]:
        snapshot = self.app.snapshot()
        selection = self.attention.select(snapshot, "routine")
        await self.app.observe("routine", snapshot=snapshot, selection=selection)
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

    async def test_agent_report_reaches_connected_work_and_accepted_speech(
        self,
    ) -> None:
        self.report(
            "Quoted fields preserve escaped delimiters. Next I will document the cases."
        )
        packet, selection = await self.submit_commentary()
        payload = json.loads(self.model.requests[0][0]["messages"][0]["content"])
        self.assertIsInstance(require(selection.subject).view, WorkView)
        self.assertIn("Next I will document the cases", json.dumps(payload))
        self.assertIn("Handle quoted fields", json.dumps(payload))
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
        self.assertFalse(self.attention.is_new(self.app.snapshot().work[0]))

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
        self.app.queue("prompt")
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
                self.app.snapshot(), delivery_pending=self.app.delivery.has_pending()
            )
        )
        self.assertIsNone(self.memory.message_outcome(packet["delivery_id"]))
        self.assertEqual(self.memory.speech(), [])

    async def test_silence_and_exhausted_delivery_persist_consideration(self) -> None:
        self.model.text = None
        await self.app.observe("routine")
        restored = Attention(
            self.config,
            self.clock,
            random.Random(0),
            self.memory.load_attention_history(),
        )
        self.assertFalse(restored.is_new(self.app.snapshot().work[0]))
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
        self.assertFalse(restored.is_new(self.app.snapshot().work[0]))

    async def test_late_acceptance_records_captured_story_without_suppressing_new_report(
        self,
    ) -> None:
        packet, selection = await self.submit_commentary()
        self.app.preempt()
        self.report("The parser examples are documented; next I will update the guide.")
        self.accept(packet)
        outcome = require(self.memory.message_outcome(packet["delivery_id"]))
        self.assertEqual(outcome.revision, require(selection.subject).revision)
        self.assertTrue(self.attention.is_new(self.app.snapshot().work[0]))
        self.assertEqual(len(self.memory.speech()), 1)

    async def test_reset_rejects_old_acceptance_and_callbacks(self) -> None:
        packet, _ = await self.submit_commentary()
        message = self.app.delivery.deliveries[packet["delivery_id"]].message
        self.app.clear_memory()
        self.accept(packet)
        self.app.delivery_outcome(DeliveryOutcome(message, "accepted"))
        self.assertIsNone(self.memory.message_outcome(packet["delivery_id"]))
        self.assertEqual(self.memory.speech(), [])
        self.assertEqual(self.memory.work.work_items(), ())

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
        self.assertTrue(self.attention.is_new(self.app.snapshot().work[0]))
        self.assertTrue(self.app.delivery.has_pending())
        self.memory.db.execute("DROP TRIGGER reject_speech")
        self.accept(packet)
        self.assertFalse(self.attention.is_new(self.app.snapshot().work[0]))

    async def test_basic_pr_state_refresh_preserves_consideration(self) -> None:
        packet, _ = await self.submit_commentary()
        self.accept(packet)
        self.provider.publish(
            replace(
                self.observation,
                revision=2,
                pr=replace(
                    require(self.observation.pr),
                    state="MERGED",
                    merged_at="2026-10-20T00:00:00Z",
                ),
            )
        )
        subject = self.app.snapshot().work[0]
        self.assertFalse(self.attention.is_new(subject))
        self.attention.next_observation = self.now
        self.assertIsNone(self.attention.request(self.app.snapshot()))
        self.assertIn("MERGED", str(subject.view))

    async def test_reconciliation_retains_last_successful_repository_evidence(
        self,
    ) -> None:
        failed = replace(self.observation, github_error="GitHub unavailable")
        self.provider.publish(failed)
        source = RepositoryTracker(
            self.app.ingest, clock=self.clock, checkpoints=(failed,)
        )
        self.app.providers = (source,)
        self.app.reconcile_work = lambda: source.reconcile_targets(
            self.memory.work.collection_targets()
        )
        self.app.reconcile_targets()
        account = require(self.memory.work.session_context("codex:parser"))
        branch = next(
            item for item in account.artifacts if item.artifact.kind == "branch"
        )
        availability = next(
            item.payload
            for item in branch.evidence
            if isinstance(item.payload, AvailabilityEvidence)
        )
        self.assertEqual(availability.github_error, "GitHub unavailable")
        self.assertEqual(len(self.memory.load_repositories()), 1)

    async def test_failed_reset_preserves_history_and_sessions(self) -> None:
        self.memory.record_speech("Earlier parser observation", "routine")
        work = self.memory.work.work_items()
        sessions = self.app.sessions
        self.memory.db.execute(
            "CREATE TRIGGER reject_memory_reset BEFORE DELETE ON events BEGIN SELECT RAISE(ABORT, 'fixture reset failure'); END"
        )
        with self.assertLogs("pet-daemon", level="WARNING"):
            self.app.clear_memory()
        self.assertEqual(self.memory.work.work_items(), work)
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
