import asyncio
import base64
import json
import random
import tempfile
import unittest
import uuid
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel

from pet_daemon.application import Application
from pet_daemon.attention import Attention, AttentionSelection
from pet_daemon.behavior.controller import Behavior
from pet_daemon.behavior.sprites import Pack
from pet_daemon.commentary import Cognition
from pet_daemon.commentary.context import CommentaryContext
from pet_daemon.commentary.contracts import ModelRequest
from pet_daemon.commentary.request import SpeechConditions
from pet_daemon.config import Config
from pet_daemon.delivery import (
    DeliveryConditions,
    DeliveryOutcome,
    SpeechMessage,
    SpeechPacket,
)
from pet_daemon.domain import (
    ArtifactKind,
    DeliveryStatus,
    EvidenceSource,
    ExecutionOutcome,
    LinkRelationship,
    ModelPurpose,
    ProcessStatus,
    PullRequestState,
    SessionSource,
    SpeechReason,
)
from pet_daemon.identity import SessionIdentity
from pet_daemon.repository_identity import RemoteRepository
from pet_daemon.runtime import (
    HookEvent,
    HookEventKind,
    OverlayPacket,
    ProcessIdentity,
    SessionTrigger,
)
from pet_daemon.sessions.processes import ProcessTracker
from pet_daemon.sessions.reactions import SessionReactions
from pet_daemon.sessions.tracking import Sessions
from pet_daemon.sources import RepositoryTracker
from pet_daemon.sources.hooks import normalize_hook
from pet_daemon.storage import Memory
from pet_daemon.types import JSONObject
from pet_daemon.work import WorkStory
from pet_daemon.work.evidence import ReportedProgress, WorkAssociation
from pet_daemon.work.models import RelatedLink
from pet_daemon.work.observations import (
    BranchObservation,
    CheckoutObservation,
    GitObservation,
    PullRequest,
    RepositoryObservation,
)
from Tests.fixtures import require, story_response


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
        self.story_requests = []
        self.exchange_choices: tuple[str, ...] = ("context",)
        self.choice_labels: dict[str, str] = {}
        self.headline = "Escaped delimiters hold"
        self.exchange_context = "The agent verified escaped delimiters while improving quoted-field parsing."
        self.before_response: Callable[[SpeechReason | ModelPurpose], None] | None = (
            None
        )
        self.text: str | None = (
            "The parser preserves escaped delimiters inside quoted fields."
        )
        self.response: (
            Callable[[ModelRequest, SpeechReason], dict[str, object]] | None
        ) = None
        self.narrative_response: Callable[[ModelRequest], dict[str, object]] | None = (
            None
        )

    async def complete(
        self,
        request: ModelRequest,
        reason: SpeechReason | ModelPurpose,
        *,
        response_type: type[BaseModel],
    ) -> dict[str, object]:
        if reason == ModelPurpose.STORY:
            self.story_requests.append(request)
            if self.before_response:
                self.before_response(reason)
            if self.narrative_response:
                return self.narrative_response(request)
            return story_response(request)
        self.requests.append((request, reason))
        if self.before_response:
            self.before_response(reason)
        result = (
            self.response(request, reason) if self.response else {"speak": self.text}
        )
        schema = request["output_config"]["format"]["schema"]
        properties = schema.get("properties", {})
        if isinstance(properties, dict) and "headline" in properties:
            payload = json.loads(request["messages"][0]["content"])
            available = {
                action["id"]: action for action in payload.get("available_actions", [])
            }
            speaking = result.get("speak") is not None
            result.setdefault("headline", self.headline if speaking else "")
            result.setdefault("context", self.exchange_context if speaking else "")
            result.setdefault(
                "choices",
                [
                    {
                        "id": identifier,
                        "label": self.choice_labels.get(
                            identifier, available[identifier]["label"]
                        ),
                    }
                    for identifier in self.exchange_choices
                    if speaking and identifier in available
                ],
            )
        return result


class ApplicationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.now = 1000.0
        self.clock = lambda: self.now
        self.config = Config(pet_dir=Path(self.temporary.name))
        self.config.behavior.sentiment_reactions = False
        self.memory = Memory(
            Path(self.temporary.name) / "memory.sqlite3",
            "fixture-pet",
            clock=self.clock,
        )
        self.model = FixtureModel()
        self.provider = FixtureProvider()
        self.packets: list[OverlayPacket] = []
        self.attention = Attention(
            self.config, self.clock, random.Random(0), wall_clock=self.clock
        )
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
        behavior.update_renderer_status(
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
            sessions=Sessions(self.clock, wall_clock=self.clock),
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
                "source": "codex",
                "session": "parser",
                "event": "intent",
                "goal": "Improve quoted-field parsing",
                "repo": "/fixture/parser",
            }
        )
        self.provider.publish(require(hook))
        self.checkout = CheckoutObservation(
            worktree="/fixture/parser",
            local=GitObservation(
                worktree="/fixture/parser",
                branch="parser",
                head="head-one",
                staged=0,
                unstaged=0,
                untracked=0,
                conflicts=0,
                upstream="origin/main",
                ahead=1,
                behind=0,
                repository=RemoteRepository(value="example.test/team/parser"),
            ),
            checked_at=self.now,
        )
        self.observation = BranchObservation(
            repository=RemoteRepository(value="example.test/team/parser"),
            branch="parser",
            pr=PullRequest(
                number=23,
                url="https://example.test/team/parser/pull/23",
                state=PullRequestState.OPEN,
                merged_at=None,
                head="head-one",
                head_repository=RemoteRepository(value="example.test/team/parser"),
                head_branch="parser",
                repository=RemoteRepository(value="example.test/team/parser"),
                title="Handle quoted fields",
            ),
            checked_at=self.now,
        )
        self.provider.publish(self.checkout)
        self.provider.publish(self.observation)
        self.attention.pending = None

    async def asyncTearDown(self) -> None:
        await self.app.close()
        self.memory.close()
        self.temporary.cleanup()

    def report(self, text: str, *, session: str = "parser") -> None:
        self.provider.publish(
            HookEvent(
                source=SessionSource.CODEX,
                session=session,
                event=HookEventKind.PROGRESS,
                progress=text,
            )
        )

    def test_context_enrichment_preserves_earlier_snapshots(self) -> None:
        earlier = self.app.context()
        captured = earlier.model_dump()
        self.provider.publish(
            HookEvent(
                source=SessionSource.CODEX,
                session="parser",
                event=HookEventKind.INTENT,
                goal="Also handle escaped delimiters",
                repo="/fixture/parser",
            )
        )
        current = self.app.context()
        self.assertEqual(earlier.model_dump(), captured)
        self.assertNotEqual(
            current.sessions[0].request_count, earlier.sessions[0].request_count
        )
        current.sessions[0].summary = "A local annotation"
        self.assertNotEqual(
            self.app.context().sessions[0].summary, "A local annotation"
        )

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

    async def test_closure_before_generation_discards_the_scheduled_snapshot(
        self,
    ) -> None:
        snapshot = self.app.context()
        selection = self.attention.select(snapshot)
        self.provider.publish(
            HookEvent(
                source=SessionSource.CODEX,
                session="parser",
                event=HookEventKind.SESSION_END,
            )
        )
        await self.app.comment(
            SpeechReason.ROUTINE, snapshot=snapshot, selection=selection
        )
        self.assertEqual(self.model.story_requests, [])
        self.assertEqual(self.model.requests, [])
        self.assertFalse(self.app.delivery.has_pending())

    async def test_current_waits_receive_periodic_check_ins_until_liveness_expires(
        self,
    ) -> None:
        self.config.idle_message_seconds = 10
        self.config.behavior.automatic_speech_cooldown_seconds = 0
        self.provider.publish(
            HookEvent(
                source=SessionSource.CODEX,
                session="parser",
                event=HookEventKind.WAITING,
                progress="Choose a format",
            )
        )
        self.attention.pending = None
        self.now += 10
        self.app.tick()
        self.assertEqual(self.app.reason, SpeechReason.CHECK_IN)
        self.assertEqual(
            self.app.sessions.by_id[
                SessionIdentity(source=SessionSource.CODEX, native_id="parser")
            ].last_check_in,
            self.now,
        )
        await require(self.app.task)
        if self.packets and "delivery_id" in self.packets[-1]:
            self.app.receive_renderer(
                {
                    "kind": "renderer",
                    "event": "speech_accepted",
                    "delivery_id": self.packets[-1]["delivery_id"],
                }
            )
        self.now += self.config.activity_fresh_seconds
        self.app.tick()
        self.assertEqual(self.app.sessions.check_in_due(10), [])

    async def test_child_lifecycle_retains_reports_without_interrupting_the_root(
        self,
    ) -> None:
        self.packets.clear()
        for event in (
            HookEvent(
                source=SessionSource.CODEX,
                session="examples",
                event=HookEventKind.SUBAGENT_START,
                parent="parser",
                goal="Document quoted-field examples",
            ),
            HookEvent(
                source=SessionSource.CODEX,
                session="examples",
                event=HookEventKind.TOOL_START,
                call="check",
                tool="pytest",
            ),
            HookEvent(
                source=SessionSource.CODEX,
                session="examples",
                event=HookEventKind.TOOL_ERROR,
                call="check",
            ),
            HookEvent(
                source=SessionSource.CODEX,
                session="examples",
                event=HookEventKind.TOOL_START,
                call="retry",
                tool="pytest",
            ),
            HookEvent(
                source=SessionSource.CODEX,
                session="examples",
                event=HookEventKind.TOOL_DONE,
                call="retry",
            ),
            HookEvent(
                source=SessionSource.CODEX,
                session="examples",
                event=HookEventKind.SUBAGENT_DONE,
                progress="Documented escaped delimiter examples.",
            ),
        ):
            self.provider.publish(event)
            self.assertIsNone(self.attention.pending)
        self.assertFalse(any("animation" in packet for packet in self.packets))
        self.assertEqual(self.app.agent_activity.activity(), "working")
        self.assertTrue(
            self.app.sessions.by_id[
                SessionIdentity(source=SessionSource.CODEX, native_id="parser")
            ].active
        )
        child = self.app.sessions.by_id[
            SessionIdentity(source=SessionSource.CODEX, native_id="examples")
        ]
        self.assertTrue(child.finished)
        self.assertEqual(child.outcome, ExecutionOutcome.SUCCESS)
        reports = self.memory.recent_events(
            session=SessionIdentity(source=SessionSource.CODEX, native_id="examples"),
            limit=10,
        )
        self.assertTrue(any(row.event_type == "subagent_done" for row in reports))
        _, selection = await self.submit_commentary()
        self.assertEqual(
            require(selection.focus_session).id,
            SessionIdentity(source=SessionSource.CODEX, native_id="parser"),
        )
        payload = json.loads(self.model.requests[-1][0]["messages"][0]["content"])
        self.assertIn("Documented escaped delimiter examples.", json.dumps(payload))

    async def test_child_input_request_receives_attention_within_the_root_work(
        self,
    ) -> None:
        self.provider.publish(
            HookEvent(
                source=SessionSource.CODEX,
                session="examples",
                event=HookEventKind.SUBAGENT_START,
                parent="parser",
            )
        )
        self.provider.publish(
            HookEvent(
                source=SessionSource.CODEX,
                session="examples",
                event=HookEventKind.WAITING,
                call="question",
                progress="Which quoted-field examples should I document?",
            )
        )
        self.app.tick()
        await require(self.app.task)
        self.assertEqual(self.model.requests[-1][1], SpeechReason.CHECK_IN)
        payload = json.loads(self.model.requests[-1][0]["messages"][0]["content"])
        self.assertEqual(
            payload["current_moment"]["event"], "The work is waiting for input."
        )
        self.assertEqual(
            payload["current_moment"]["reported_progress"],
            "Which quoted-field examples should I document?",
        )
        self.assertEqual(
            payload["current_work"]["request"], "Improve quoted-field parsing"
        )

    async def test_quiet_children_do_not_create_independent_timed_check_ins(
        self,
    ) -> None:
        self.config.idle_message_seconds = 5
        self.attention.next_observation = self.now + 3600
        for child in ("examples", "escapes"):
            self.provider.publish(
                HookEvent(
                    source=SessionSource.CODEX,
                    session=child,
                    event=HookEventKind.SUBAGENT_START,
                    parent="parser",
                )
            )
        self.now += 6
        self.report("Reviewing the delegated findings.")
        self.app.tick()
        self.assertIsNone(self.app.task)
        self.assertEqual(self.model.requests, [])
        self.now += 6
        self.app.tick()
        await require(self.app.task)
        self.assertEqual(self.model.requests[-1][1], SpeechReason.CHECK_IN)
        self.assertEqual(
            self.app.sessions.by_id[
                SessionIdentity(source=SessionSource.CODEX, native_id="parser")
            ].last_check_in,
            self.now,
        )
        for child in ("examples", "escapes"):
            self.assertIsNone(
                self.app.sessions.by_id[
                    SessionIdentity(source=SessionSource.CODEX, native_id=child)
                ].last_check_in
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
                    source=SessionSource.CODEX,
                    session="parser",
                    event=HookEventKind.INTENT,
                    goal=failed_intent,
                    repo="/fixture/rejected-parser",
                )
            )
        self.memory.db.execute("DROP TRIGGER reject_session_write")
        self.report("Quoted fields preserve escaped delimiters.")
        await self.submit_commentary()
        payload = json.loads(self.model.requests[-1][0]["messages"][0]["content"])
        self.assertEqual(
            payload["current_work"]["request"], "Improve quoted-field parsing"
        )
        self.assertIn("Quoted fields preserve escaped delimiters.", json.dumps(payload))
        self.assertNotIn(failed_intent, json.dumps(payload))

    async def test_late_pr_resolution_keeps_a_report_on_its_work_after_branch_reuse(
        self,
    ) -> None:
        self.now += 1
        self.provider.publish(
            self.observation.model_copy(update={"pr": None, "checked_at": self.now})
        )
        self.provider.publish(
            HookEvent(
                source=SessionSource.CODEX,
                session="parser",
                event=HookEventKind.TURN_DONE,
                outcome=ExecutionOutcome.SUCCESS,
                progress="The parser changes are ready.",
            )
        )
        reported_pr = require(self.observation.pr).model_copy(
            update={"number": 24, "url": "https://example.test/team/parser/pull/24"}
        )
        self.now += 1
        self.provider.publish(
            self.observation.model_copy(
                update={"pr": reported_pr, "checked_at": self.now}
            )
        )
        unrelated_pr = reported_pr.model_copy(
            update={
                "number": 25,
                "url": "https://example.test/team/parser/pull/25",
                "head": "unrelated-head",
                "title": "Add unrelated import support",
            }
        )
        self.now += 1
        self.provider.publish(
            self.checkout.model_copy(
                update={
                    "local": require(self.checkout.local).model_copy(
                        update={"head": unrelated_pr.head}
                    ),
                    "checked_at": self.now,
                }
            )
        )
        self.provider.publish(
            self.observation.model_copy(
                update={"pr": unrelated_pr, "checked_at": self.now}
            )
        )
        _, selection = await self.submit_commentary()
        payload = json.loads(self.model.requests[-1][0]["messages"][0]["content"])
        work = require(selection.subject).story
        report = next(
            report
            for report in work.activity
            if isinstance(report.details, ReportedProgress)
            and report.details.reported_progress == "The parser changes are ready."
        )
        self.assertEqual(require(report.context).pull_request, reported_pr.url)
        self.assertNotIn(reported_pr.url, json.dumps(payload))
        self.assertNotIn(unrelated_pr.url, json.dumps(payload))
        artifacts = {artifact.ref for artifact in work.artifacts}
        self.assertIn(reported_pr.url, artifacts)
        self.assertNotIn(unrelated_pr.url, artifacts)

    async def test_repeated_progress_preserves_report_context_without_repeating_speech(
        self,
    ) -> None:
        self.report("Tests pass")
        packet, first_selection = await self.submit_commentary()
        first_report = next(
            report
            for report in require(first_selection.subject).story.activity
            if isinstance(report.details, ReportedProgress)
            and report.details.reported_progress == "Tests pass"
        )
        self.assertEqual(
            require(first_report.context).pull_request,
            require(self.observation.pr).url,
        )
        self.accept(packet)
        self.now += 1
        next_pr = require(self.observation.pr).model_copy(
            update={
                "number": 24,
                "url": "https://example.test/team/parser/pull/24",
                "head_branch": "parser-examples",
                "title": "Document quoted-field examples",
            }
        )
        self.provider.publish(
            self.checkout.model_copy(
                update={
                    "local": require(self.checkout.local).model_copy(
                        update={"branch": next_pr.head_branch}
                    ),
                    "checked_at": self.now,
                }
            )
        )
        self.provider.publish(
            BranchObservation(
                repository=self.observation.repository,
                branch=next_pr.head_branch,
                pr=next_pr,
                checked_at=self.now,
            )
        )
        self.attention.next_observation = self.now
        self.assertIsNone(self.attention.request(self.app.context()))
        self.report("Tests pass")
        self.attention.next_observation = self.now
        snapshot = self.app.context()
        self.assertIsNone(self.attention.request(snapshot))
        latest_report = next(
            report
            for report in snapshot.work[0].story.activity
            if isinstance(report.details, ReportedProgress)
            and report.details.reported_progress == "Tests pass"
        )
        self.assertEqual(
            (
                require(latest_report.context).branch,
                require(latest_report.context).pull_request,
            ),
            (next_pr.head_branch, next_pr.url),
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
        child_pr = parent_pr.model_copy(
            update={
                "number": 24,
                "url": "https://example.test/team/parser/pull/24",
                "head": "child-head",
                "head_branch": "parser-examples",
                "title": "Document escaped delimiter examples",
                "related_links": (
                    RelatedLink(
                        kind=ArtifactKind.PULL_REQUEST,
                        ref=parent_pr.url,
                        relationship=LinkRelationship.STACK_ENTRY,
                        source=EvidenceSource.GITHUB_STACK,
                        stack=4,
                        position=1,
                    ),
                ),
            }
        )
        self.provider.publish(
            self.observation.model_copy(
                update={
                    "pr": parent_pr.model_copy(
                        update={
                            "related_links": (
                                RelatedLink(
                                    kind=ArtifactKind.PULL_REQUEST,
                                    ref=child_pr.url,
                                    relationship=LinkRelationship.STACK_ENTRY,
                                    source=EvidenceSource.GITHUB_STACK,
                                    stack=4,
                                    position=2,
                                ),
                            )
                        }
                    )
                }
            )
        )
        parent_report = "Quoted fields preserve escaped delimiters. Next I will review the examples."
        child_report = (
            "Documented escaped delimiters. Next I will add a multiline example."
        )
        self.report(parent_report)
        self.provider.publish(
            HookEvent(
                source=SessionSource.CODEX,
                session="examples",
                event=HookEventKind.SUBAGENT_START,
                parent="parser",
                goal="Document quoted-field examples",
                repo="/fixture/parser-examples",
            )
        )
        self.provider.publish(
            CheckoutObservation(
                worktree="/fixture/parser-examples",
                local=require(self.checkout.local).model_copy(
                    update={
                        "worktree": "/fixture/parser-examples",
                        "branch": "parser-examples",
                        "head": "child-head",
                    }
                ),
                checked_at=self.now,
            )
        )
        self.provider.publish(
            BranchObservation(
                repository=self.observation.repository,
                branch="parser-examples",
                pr=child_pr,
                checked_at=self.now,
            )
        )
        self.now += 1
        self.report(child_report, session="examples")
        subjects = self.app.context().work
        self.assertEqual(len(subjects), 2)
        child_subject = next(
            item for item in subjects if item.id.native_id == "examples"
        )
        self.assertEqual(child_subject.sessions, (child_subject.id,))
        self.assertIsInstance(child_subject.story, WorkStory)
        self.assertEqual(child_subject.story.intent, "Document quoted-field examples")
        artifact_titles = {artifact.title for artifact in child_subject.story.artifacts}
        self.assertIn(parent_pr.title, artifact_titles)
        self.assertIn(child_pr.title, artifact_titles)
        self.assertTrue(
            any(
                isinstance(evidence.details, WorkAssociation)
                and "stack" in evidence.details.association
                for artifact in child_subject.story.artifacts
                for evidence in artifact.evidence
            )
        )
        packet, selection = await self.submit_commentary()
        payload = json.loads(self.model.requests[0][0]["messages"][0]["content"])
        self.assertIn(parent_report, json.dumps(payload["earlier_context"]))
        self.assertIn(child_report, json.dumps(payload["current_moment"]))
        story_payload = json.loads(
            self.model.story_requests[0]["messages"][0]["content"]
        )
        child_sources = [
            item
            for item in story_payload["current_moment"]["evidence"]
            if item["scope"] == "current task"
        ]
        self.assertIn(child_report, json.dumps(child_sources))
        self.assertNotIn(parent_report, json.dumps(child_sources))
        self.assertNotIn("artifacts", json.dumps(payload))
        self.assertNotIn(child_pr.url, json.dumps(payload))
        self.assertEqual(self.memory.speech(), [])
        self.assertTrue(
            self.attention.has_unconsidered_changes(require(selection.subject))
        )
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
        self.assertFalse(
            self.attention.has_unconsidered_changes(self.app.context().work[0])
        )

    async def test_child_location_comes_from_observed_hooks_and_parent_is_canonical(
        self,
    ) -> None:
        self.provider.publish(
            HookEvent(
                source=SessionSource.CODEX,
                session="parser",
                event=HookEventKind.PROGRESS,
                aliases=("parser-alias",),
            )
        )
        self.provider.publish(
            HookEvent(
                source=SessionSource.CODEX,
                session="examples",
                event=HookEventKind.SUBAGENT_START,
                parent="parser-alias",
                goal="Document quoted-field examples",
            )
        )
        self.assertEqual(
            self.app.sessions.by_id[
                SessionIdentity(source=SessionSource.CODEX, native_id="examples")
            ].repo,
            "/fixture/parser",
        )
        subject = next(
            item for item in self.app.context().work if item.id.native_id == "examples"
        )
        self.assertEqual(subject.sessions, (subject.id,))
        parent = require(self.app.sessions.by_id[subject.id].parent)
        self.assertEqual(parent.native_id, "parser")
        self.provider.publish(
            HookEvent(
                source=SessionSource.CODEX,
                session="examples",
                event=HookEventKind.PROGRESS,
                repo="/fixture/parser-examples",
                progress="Documented the quoted-field examples.",
            )
        )
        child = next(
            item for item in self.app.context().work if item.id.native_id == "examples"
        )
        self.assertEqual(
            require(child.story.activity[0].context).checkout,
            "/fixture/parser-examples",
        )
        self.assertEqual(len(self.app.context().work), 2)

    def complete_turn(self, session: str = "parser") -> None:
        self.provider.publish(
            HookEvent(
                source=SessionSource.CODEX,
                session=session,
                event=HookEventKind.TURN_DONE,
                outcome=ExecutionOutcome.SUCCESS,
            )
        )

    def restart_attention_and_sessions(self) -> None:
        self.attention = Attention(
            self.config,
            self.clock,
            random.Random(0),
            self.memory.load_attention_history(),
            wall_clock=self.clock,
        )
        self.app.attention = self.attention
        self.app.sessions = Sessions(self.clock, wall_clock=self.clock)
        self.app.reactions = SessionReactions()
        self.app.agent_activity.sessions = self.app.sessions
        self.app.agent_activity.reactions = self.app.reactions

    async def test_accepted_completion_stays_consumed_through_restart_and_closure(
        self,
    ) -> None:
        self.complete_turn()
        packet, _ = await self.submit_commentary()
        self.accept(packet)
        self.assertEqual(len(self.memory.speech()), 1)
        self.restart_attention_and_sessions()
        self.assertFalse(
            self.attention.is_routine_eligible(
                self.app.context().work[0], self.app.context()
            )
        )
        self.now += 10
        self.provider.publish(
            HookEvent(
                source=SessionSource.CODEX,
                session="parser",
                event=HookEventKind.SESSION_END,
            )
        )
        snapshot = self.app.context()
        self.assertFalse(self.attention.is_routine_eligible(snapshot.work[0], snapshot))
        await self.app.comment(SpeechReason.COMPLETION)
        self.assertEqual(len(self.model.requests), 1)
        self.now += 7200
        await self.app.comment(SpeechReason.ROUTINE)
        self.assertEqual(len(self.model.requests), 1)
        self.assertEqual(
            require(self.memory.message_outcome(packet["delivery_id"])).outcome,
            "accepted",
        )

    async def test_unspoken_completion_expires_at_original_grace_after_restart(
        self,
    ) -> None:
        self.complete_turn()
        self.restart_attention_and_sessions()
        self.now += 299
        snapshot = self.app.context()
        selection = self.attention.select(snapshot)
        self.assertIsNotNone(selection.update)
        self.assertTrue(self.attention.is_selection_eligible(selection, snapshot))
        self.provider.publish(
            HookEvent(
                source=SessionSource.CODEX,
                session="parser",
                event=HookEventKind.SESSION_END,
            )
        )
        self.now += 1
        await self.app.comment(SpeechReason.ROUTINE)
        self.assertEqual(self.model.story_requests, [])
        self.assertEqual(self.model.requests, [])

    async def test_verified_active_work_remains_eligible_beyond_completion_grace(
        self,
    ) -> None:
        owner = ProcessIdentity(pid=42, started="fixture-owner")
        self.provider.publish(
            HookEvent(
                source=SessionSource.CODEX,
                session="parser",
                event=HookEventKind.PROGRESS,
                progress="Checking parser cases",
                owner=owner,
            )
        )
        self.app.sessions.apply_process_status({owner: ProcessStatus.ALIVE})
        self.now += 7200
        await self.app.comment(SpeechReason.ROUTINE)
        self.assertEqual(len(self.model.requests), 1)
        self.assertTrue(self.app.delivery.has_pending())

    async def test_completion_expiring_during_story_generation_stops_commentary(
        self,
    ) -> None:
        self.complete_turn()
        wall = self.now
        self.attention.wall_clock = lambda: wall
        self.app.sessions.wall_clock = lambda: wall

        def sleep_during_story(reason: SpeechReason | ModelPurpose) -> None:
            nonlocal wall
            if reason == ModelPurpose.STORY:
                wall += 300

        self.model.before_response = sleep_during_story
        await self.app.comment(SpeechReason.ROUTINE)
        self.assertEqual(len(self.model.story_requests), 1)
        self.assertEqual(self.model.requests, [])
        self.assertFalse(self.app.delivery.has_pending())

    async def test_completion_expiring_during_commentary_stops_submission(self) -> None:
        self.complete_turn()
        wall = self.now
        self.attention.wall_clock = lambda: wall
        self.app.sessions.wall_clock = lambda: wall

        def sleep_during_commentary(reason: SpeechReason | ModelPurpose) -> None:
            nonlocal wall
            if reason == SpeechReason.ROUTINE:
                wall += 300

        self.model.before_response = sleep_during_commentary
        await self.app.comment(SpeechReason.ROUTINE)
        self.assertEqual(len(self.model.requests), 1)
        self.assertFalse(self.app.delivery.has_pending())
        self.assertFalse(any("delivery_id" in packet for packet in self.packets))
        self.assertEqual(self.memory.speech(), [])

    async def test_completion_delivery_retries_stop_at_grace_expiry(self) -> None:
        self.complete_turn()
        self.now += 299
        packet, _ = await self.submit_commentary()
        emitted = len([item for item in self.packets if "delivery_id" in item])
        self.now += 1
        self.app.delivery.tick()
        self.assertFalse(self.app.delivery.has_pending())
        self.assertEqual(
            len([item for item in self.packets if "delivery_id" in item]), emitted
        )
        self.assertIsNone(self.memory.message_outcome(packet["delivery_id"]))

    async def test_failed_completion_delivery_remains_eligible_before_expiry(
        self,
    ) -> None:
        self.complete_turn()
        packet, selection = await self.submit_commentary()
        for _ in range(2):
            self.now += 2
            self.app.delivery.tick()
        self.assertEqual(
            require(self.memory.message_outcome(packet["delivery_id"])).outcome,
            "failed",
        )
        self.assertTrue(
            self.attention.is_selection_eligible(selection, self.app.context())
        )
        self.assertEqual(self.memory.speech(), [])

    async def test_deliberate_silence_consumes_queued_completion(self) -> None:
        self.complete_turn()
        trigger = require(self.attention.pending)[2]
        self.model.text = None
        await self.app.comment(SpeechReason.ROUTINE)
        self.now += 2
        await self.app.comment(SpeechReason.COMPLETION, trigger=trigger)
        self.assertEqual(len(self.model.requests), 1)
        self.assertEqual(self.memory.speech(), [])

    async def test_live_child_cannot_revive_consumed_parent_completion(self) -> None:
        self.complete_turn()
        packet, _ = await self.submit_commentary()
        self.accept(packet)
        self.provider.publish(
            HookEvent(
                source=SessionSource.CODEX,
                session="child",
                event=HookEventKind.SUBAGENT_START,
                parent="parser",
                goal="Check examples",
            )
        )
        snapshot = self.app.context(
            SessionTrigger(
                session_id=SessionIdentity(
                    source=SessionSource.CODEX, native_id="parser"
                )
            )
        )
        selection = self.attention.select(snapshot, SpeechReason.COMPLETION)
        self.assertIsNone(selection.update)
        self.assertFalse(self.attention.is_selection_eligible(selection, snapshot))
        child_snapshot = self.app.context(
            SessionTrigger(
                session_id=SessionIdentity(
                    source=SessionSource.CODEX, native_id="child"
                )
            )
        )
        child_selection = self.attention.select(child_snapshot, SpeechReason.PROMPT)
        self.assertEqual(
            require(child_selection.update).session,
            SessionIdentity(source=SessionSource.CODEX, native_id="child"),
        )
        self.assertTrue(
            self.attention.is_selection_eligible(child_selection, child_snapshot)
        )

    async def test_live_parent_cannot_revive_consumed_child_completion(self) -> None:
        self.provider.publish(
            HookEvent(
                source=SessionSource.CODEX,
                session="child",
                event=HookEventKind.SUBAGENT_START,
                parent="parser",
                goal="Check examples",
            )
        )
        self.complete_turn("child")
        self.model.text = None
        await self.app.comment(
            SpeechReason.COMPLETION,
            trigger=SessionTrigger(
                session_id=SessionIdentity(
                    source=SessionSource.CODEX, native_id="child"
                )
            ),
        )
        self.report("The parent is still checking the parser")
        snapshot = self.app.context(
            SessionTrigger(
                session_id=SessionIdentity(
                    source=SessionSource.CODEX, native_id="child"
                )
            )
        )
        selection = self.attention.select(snapshot, SpeechReason.COMPLETION)
        self.assertIsNone(selection.update)
        self.assertFalse(self.attention.is_selection_eligible(selection, snapshot))

    async def test_pending_delivery_defers_other_work_and_queued_automatic_generation(
        self,
    ) -> None:
        self.config.behavior.automatic_speech_cooldown_seconds = 0
        self.config.observation_seconds = (1, 1)
        self.config.idle_message_seconds = 1
        await self.submit_commentary()
        self.provider.publish(
            HookEvent(
                source=SessionSource.CODEX,
                session="other",
                event=HookEventKind.INTENT,
                goal="Document the renderer",
            )
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
            wall_clock=self.clock,
        )
        self.now += max(self.config.observation_seconds)
        restored.next_observation = self.now
        self.assertTrue(restored.has_unconsidered_changes(require(selection.subject)))
        self.assertIsNotNone(
            restored.request(
                self.app.context(), delivery_pending=self.app.delivery.has_pending()
            )
        )
        self.assertIsNone(self.memory.message_outcome(packet["delivery_id"]))
        self.assertEqual(self.memory.speech(), [])

    async def test_silence_persists_consideration_and_failed_delivery_stays_retryable(
        self,
    ) -> None:
        self.model.text = None
        await self.app.comment(SpeechReason.ROUTINE)
        restored = Attention(
            self.config,
            self.clock,
            random.Random(0),
            self.memory.load_attention_history(),
            wall_clock=self.clock,
        )
        self.assertFalse(restored.has_unconsidered_changes(self.app.context().work[0]))
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
            wall_clock=self.clock,
        )
        self.assertTrue(restored.has_unconsidered_changes(self.app.context().work[0]))
        self.assertTrue(
            self.attention.has_unconsidered_changes(self.app.context().work[0])
        )
        self.assertEqual(self.memory.speech(), [])

    async def test_late_acceptance_records_captured_story_without_suppressing_new_report(
        self,
    ) -> None:
        packet, selection = await self.submit_commentary()
        self.app.preempt()
        self.report("The parser examples are documented; next I will update the guide.")
        self.accept(packet)
        outcome = require(self.memory.message_outcome(packet["delivery_id"]))
        self.assertEqual(outcome.revision, require(selection.subject).revision)
        self.assertTrue(
            self.attention.has_unconsidered_changes(self.app.context().work[0])
        )
        self.assertEqual(len(self.memory.speech()), 1)

    async def test_background_report_remains_eligible_after_selected_speech_is_accepted(
        self,
    ) -> None:
        peer_id = SessionIdentity(source=SessionSource.CODEX, native_id="restoration")
        selected_id = SessionIdentity(source=SessionSource.CODEX, native_id="storage")
        peer_report = "Restoration checks passed."
        selected_report = "Storage writes retain session context."
        for session, goal, report in (
            (peer_id, "Restore session records", peer_report),
            (selected_id, "Implement session storage", selected_report),
        ):
            self.now += 1
            self.provider.publish(
                HookEvent(
                    source=session.source,
                    session=session.native_id,
                    event=HookEventKind.INTENT,
                    goal=goal,
                    repo="/fixture/parser",
                )
            )
            self.provider.publish(
                HookEvent(
                    source=session.source,
                    session=session.native_id,
                    event=HookEventKind.TURN_DONE,
                    progress=report,
                    outcome=ExecutionOutcome.SUCCESS,
                )
            )
        snapshot = self.app.context()
        selection = self.attention.select(snapshot, eligible=(selected_id,))
        self.assertEqual(require(selection.update).session, selected_id)
        self.assertEqual(
            [item.session for item in require(selection.subject).story.sessions],
            [peer_id],
        )
        peer = next(subject for subject in snapshot.work if subject.id == peer_id)
        peer_history = self.memory.load_attention_history().get(peer_id)
        self.assertTrue(self.attention.is_update_eligible(peer.updates[0], snapshot))
        self.model.text = "The storage checks passed."
        await self.app.comment(
            SpeechReason.ROUTINE, snapshot=snapshot, selection=selection
        )
        payload = json.loads(self.model.requests[-1][0]["messages"][0]["content"])
        self.assertEqual(
            payload["current_moment"]["reported_progress"], selected_report
        )
        self.assertIn(peer_report, json.dumps(payload["earlier_context"]))
        emitted = next(
            packet for packet in reversed(self.packets) if "delivery_id" in packet
        )
        packet: SpeechPacket = {
            "text": emitted["text"],
            "ttl": emitted["ttl"],
            "delivery_id": emitted["delivery_id"],
        }
        self.assertEqual(packet["text"], "The storage checks passed.")
        self.accept(packet)
        self.assertEqual(
            require(self.memory.message_outcome(packet["delivery_id"])).outcome,
            DeliveryStatus.ACCEPTED,
        )
        self.assertEqual(
            self.attention.history(selected_id).considered_session_revision,
            require(selection.update).revision,
        )
        self.assertIsNone(self.attention.history(peer_id).considered_session_revision)
        self.assertEqual(
            self.memory.load_attention_history().get(peer_id), peer_history
        )
        current = self.app.context()
        self.assertTrue(self.attention.is_update_eligible(peer.updates[0], current))
        next_selection = self.attention.select(current, eligible=(peer_id,))
        self.assertEqual(require(next_selection.update).session, peer_id)

    async def test_acceptance_keeps_uncaptured_related_reports_eligible(self) -> None:
        self.provider.publish(
            HookEvent(
                source=SessionSource.CODEX,
                session="child",
                event=HookEventKind.INTENT,
                repo="/fixture/parser",
                goal="Check parser restoration",
            )
        )
        self.report("The parser preserves escaped delimiters.")
        selected_id = SessionIdentity(source=SessionSource.CODEX, native_id="parser")
        selection = self.attention.select(self.app.context(), eligible=(selected_id,))
        message = SpeechMessage(
            text="The parser keeps escaped delimiters intact.",
            ttl=5,
            reason=SpeechReason.ROUTINE,
            selection_id="captured-related-context",
            subject_id=selection.id,
            revision=require(selection.subject).revision,
            history_epoch=self.app.history_epoch,
            conditions=DeliveryConditions(
                commentary=SpeechConditions(
                    selection=selection, context=CommentaryContext()
                ),
                trigger=None,
                request_epoch=self.app.revision,
            ),
        )
        self.assertEqual(require(selection.subject).story.sessions, ())
        self.report("Restoration handles escaped delimiters.", session="child")
        snapshot = self.app.context()
        parent = next(
            subject for subject in snapshot.work if subject.id.native_id == "parser"
        )
        child = next(
            subject for subject in snapshot.work if subject.id.native_id == "child"
        )
        self.assertEqual(parent.revision, require(selection.subject).revision)
        self.assertEqual([value.session for value in parent.story.sessions], [child.id])
        self.app.delivery_outcome(
            DeliveryOutcome(message=message, status=DeliveryStatus.ACCEPTED)
        )
        self.assertIsNone(self.attention.history(child.id).considered_session_revision)
        self.assertTrue(
            self.attention.is_update_eligible(child.updates[0], self.app.context())
        )
        self.assertIsNone(self.memory.load_attention_history().get(child.id))

    async def test_reset_rejects_old_acceptance_and_callbacks(self) -> None:
        packet, _ = await self.submit_commentary()
        message = self.app.delivery.deliveries[packet["delivery_id"]].message
        self.app.clear_memory()
        self.accept(packet)
        self.app.delivery_outcome(
            DeliveryOutcome(message=message, status=DeliveryStatus.ACCEPTED)
        )
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
        self.assertTrue(
            self.attention.has_unconsidered_changes(self.app.context().work[0])
        )
        self.assertTrue(self.app.delivery.has_pending())
        self.memory.db.execute("DROP TRIGGER reject_speech")
        self.accept(packet)
        self.assertFalse(
            self.attention.has_unconsidered_changes(self.app.context().work[0])
        )

    async def test_basic_pr_state_refresh_preserves_consideration(self) -> None:
        packet, _ = await self.submit_commentary()
        self.accept(packet)
        self.provider.publish(
            self.observation.model_copy(
                update={
                    "pr": require(self.observation.pr).model_copy(
                        update={
                            "state": PullRequestState.MERGED,
                            "merged_at": "2026-10-20T00:00:00Z",
                        }
                    )
                }
            )
        )
        subject = self.app.context().work[0]
        self.assertFalse(self.attention.has_unconsidered_changes(subject))
        self.attention.next_observation = self.now
        self.assertIsNone(self.attention.request(self.app.context()))
        self.assertIn("MERGED", str(subject.story))

    async def test_reconciliation_retains_successful_pr_facts_after_source_failure(
        self,
    ) -> None:
        failed = self.observation.model_copy(
            update={
                "pr": None,
                "checked_at": self.now + 1,
                "error": "GitHub unavailable",
            }
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

    async def test_new_related_work_recalls_an_earlier_report_without_a_recall_request(
        self,
    ) -> None:
        earlier = "The quoted-field parser preserves escaped delimiters."
        self.report(earlier)
        self.complete_turn()
        self.attention.pending = None
        self.now += 3600
        self.provider.publish(
            HookEvent(
                source=SessionSource.CODEX,
                session="verification",
                event=HookEventKind.INTENT,
                goal="Verify escaped delimiters in quoted fields",
                repo="/fixture/parser",
            )
        )
        self.attention.next_observation = self.now
        self.app.tick()
        await require(self.app.task)

        self.assertEqual(self.model.requests[-1][1], SpeechReason.PROMPT)
        story_payload = json.loads(
            self.model.story_requests[-1]["messages"][0]["content"]
        )
        evidence = story_payload["current_moment"]["evidence"]
        self.assertTrue(any(item.get("text") == earlier for item in evidence))
        speech_payload = json.loads(
            self.model.requests[-1][0]["messages"][0]["content"]
        )
        self.assertIn(earlier, json.dumps(speech_payload))

    async def test_new_request_in_another_conversation_preserves_queued_completion(
        self,
    ) -> None:
        self.report("The quoted-field parser preserves escaped delimiters.")
        self.complete_turn()
        snapshot = self.app.context()
        selected = self.attention.select(snapshot, SpeechReason.COMPLETION)
        self.assertEqual(require(selected.update).session.native_id, "parser")

        self.now += 1
        self.provider.publish(
            HookEvent(
                source=SessionSource.CODEX,
                session="billing",
                event=HookEventKind.INTENT,
                goal="Investigate billing totals",
                repo="/fixture/billing",
            )
        )
        await self.app.comment(
            SpeechReason.COMPLETION, snapshot=snapshot, selection=selected
        )

        emitted = require(
            next(
                (
                    packet
                    for packet in reversed(self.packets)
                    if "delivery_id" in packet
                ),
                None,
            )
        )
        packet: SpeechPacket = {
            "text": emitted["text"],
            "ttl": emitted["ttl"],
            "delivery_id": emitted["delivery_id"],
        }
        self.assertEqual(self.model.requests[-1][1], SpeechReason.COMPLETION)
        self.accept(packet)
        self.assertEqual(
            require(self.memory.message_outcome(packet["delivery_id"])).outcome,
            DeliveryStatus.ACCEPTED,
        )

    async def test_unrelated_text_match_does_not_cancel_parser_delivery(self) -> None:
        peer = SessionIdentity(source=SessionSource.CODEX, native_id="telemetry")
        repository = "https://example.test/team/parser"
        self.provider.publish(
            HookEvent(
                source=peer.source,
                session=peer.native_id,
                event=HookEventKind.INTENT,
                goal="Investigate quoted field telemetry",
                repo="/fixture/telemetry",
                repository=repository,
            )
        )
        self.provider.publish(
            HookEvent(
                source=peer.source,
                session=peer.native_id,
                event=HookEventKind.PROGRESS,
                progress="The quoted field telemetry samples are indexed.",
                repo="/fixture/telemetry",
                repository=repository,
            )
        )
        self.report("The quoted field parser preserves escaped delimiters.")
        snapshot = self.app.context()
        parser = SessionIdentity(source=SessionSource.CODEX, native_id="parser")
        subject = next(item for item in snapshot.work if item.id == parser)
        self.assertTrue(
            any(
                item.payload.session == peer
                for item in self.memory.context_reports(subject)
            )
        )
        selection = self.attention.select(snapshot, eligible=(parser,))
        await self.app.comment(
            SpeechReason.ROUTINE, snapshot=snapshot, selection=selection
        )
        emitted = next(item for item in reversed(self.packets) if "delivery_id" in item)
        packet: SpeechPacket = {
            "text": emitted["text"],
            "ttl": emitted["ttl"],
            "delivery_id": emitted["delivery_id"],
        }
        message = self.app.delivery.deliveries[packet["delivery_id"]].message

        self.now += 1
        self.provider.publish(
            HookEvent(
                source=peer.source,
                session=peer.native_id,
                event=HookEventKind.PROGRESS,
                progress="The quoted field telemetry chart updates correctly.",
                repo="/fixture/telemetry",
                repository=repository,
            )
        )
        self.assertTrue(self.app.is_delivery_valid(message))
        self.app.delivery.tick()
        self.assertTrue(self.app.delivery.has_pending())
        self.accept(packet)
        self.assertEqual(
            require(self.memory.message_outcome(packet["delivery_id"])).outcome,
            DeliveryStatus.ACCEPTED,
        )

    async def test_any_report_on_a_cited_request_scope_cancels_pending_delivery(
        self,
    ) -> None:
        peer = SessionIdentity(source=SessionSource.CODEX, native_id="telemetry")
        repository = "https://example.test/team/parser"
        reported = "The quoted field parser still needs a multiline escape check."
        self.provider.publish(
            HookEvent(
                source=peer.source,
                session=peer.native_id,
                event=HookEventKind.INTENT,
                goal="Check quoted field telemetry",
                repo="/fixture/telemetry",
                repository=repository,
            )
        )
        self.provider.publish(
            HookEvent(
                source=peer.source,
                session=peer.native_id,
                event=HookEventKind.PROGRESS,
                progress=reported,
                repo="/fixture/telemetry",
                repository=repository,
            )
        )
        self.report("The quoted field parser handles escaped delimiters.")

        def cite_peer(request: ModelRequest) -> dict[str, object]:
            payload = json.loads(request["messages"][0]["content"])
            evidence = payload["current_moment"]["evidence"]
            latest = payload["current_work"]["latest_request_source"]
            intent = next(item for item in evidence if item["id"] == latest)
            source = next(item for item in evidence if item.get("text") == reported)
            self.assertNotIn("native_id", json.dumps(payload))
            return {
                "intent": {"text": intent["text"], "sources": [latest]},
                "decisions": [],
                "progress": [{"text": reported, "sources": [source["id"]]}],
            }

        def mention_peer(
            request: ModelRequest, reason: SpeechReason
        ) -> dict[str, object]:
            payload = json.loads(request["messages"][0]["content"])
            source = next(
                item
                for item in payload["earlier_context"]
                if item.get("reported_progress") == reported
            )
            return {
                "speak": "The multiline escape check is still open.",
                "mentions": source["source_ids"],
            }

        self.model.narrative_response = cite_peer
        self.model.response = mention_peer
        snapshot = self.app.context()
        parser = SessionIdentity(source=SessionSource.CODEX, native_id="parser")
        selection = self.attention.select(snapshot, eligible=(parser,))
        await self.app.comment(
            SpeechReason.ROUTINE, snapshot=snapshot, selection=selection
        )
        emitted = next(item for item in reversed(self.packets) if "delivery_id" in item)
        message = self.app.delivery.deliveries[emitted["delivery_id"]].message
        self.assertEqual(len(message.mentioned_sources), 1)
        narrative = require(self.memory.load_narrative(parser))
        self.assertEqual(
            narrative.sources[message.mentioned_sources[0]]["text"], reported
        )
        self.assertTrue(self.app.is_delivery_valid(message))

        self.now += 1
        self.provider.publish(
            HookEvent(
                source=peer.source,
                session=peer.native_id,
                event=HookEventKind.PROGRESS,
                progress="Telemetry sampling frequency is five seconds.",
                repo="/fixture/telemetry",
                repository=repository,
            )
        )
        current = next(item for item in self.app.context().work if item.id == parser)
        self.assertEqual(current.revision, require(selection.subject).revision)
        self.assertFalse(self.app.is_delivery_valid(message))
        self.app.delivery.tick()
        self.assertFalse(self.app.delivery.has_pending())
        self.assertIsNone(self.memory.message_outcome(emitted["delivery_id"]))

    async def test_idle_reflection_records_the_selected_report_and_silence(
        self,
    ) -> None:
        self.config.behavior.naps = False
        self.config.behavior.care = False
        report = "The quoted-field parser still needs a multiline escape check."
        self.report(report)
        self.complete_turn()
        self.attention.pending = None
        self.attention.next_observation = self.now + 10_000
        selected: list[str] = []

        def recall(request: ModelRequest, reason: SpeechReason) -> dict[str, object]:
            self.assertEqual(reason, SpeechReason.CHECK_IN)
            payload = json.loads(request["messages"][0]["content"])
            candidates = [
                item["memory_candidate"]
                for item in payload.get("earlier_context", ())
                if "memory_candidate" in item
            ]
            candidate = next(item for item in candidates if item["text"] == report)
            identifier = candidate["id"]
            selected.append(identifier)
            if len(selected) == 1:
                return {
                    "speak": "The multiline escape check is still open.",
                    "mentions": [identifier],
                    "memory": identifier,
                    "ongoing": True,
                }
            return {"speak": None, "memory": identifier, "ongoing": True}

        self.model.response = recall
        self.now += 299
        self.app.tick()
        self.assertIsNone(self.app.task)
        self.now += 1
        self.app.tick()
        await require(self.app.task)
        emitted = next(
            packet for packet in reversed(self.packets) if "delivery_id" in packet
        )
        packet: SpeechPacket = {
            "text": emitted["text"],
            "ttl": emitted["ttl"],
            "delivery_id": emitted["delivery_id"],
        }
        self.assertEqual(packet["text"], "The multiline escape check is still open.")
        message = self.app.delivery.deliveries[emitted["delivery_id"]].message
        source = message.mentioned_sources[0]
        self.assertNotEqual(source, selected[0])
        self.assertIsNone(self.memory.load_source_attention().get(source))
        self.accept(packet)
        recorded = require(self.memory.load_source_attention().get(source))
        self.assertEqual(recorded.considered_at, self.now)
        self.assertEqual(recorded.mentioned_at, self.now)

        self.now += 899
        self.app.tick()
        self.assertEqual(len(selected), 1)
        self.now += 1
        self.app.tick()
        await require(self.app.task)
        self.assertEqual(len(selected), 2)
        self.assertEqual(len(self.memory.speech()), 1)
        recorded = require(self.memory.load_source_attention().get(source))
        self.assertEqual(recorded.considered_at, self.now)
        self.assertEqual(recorded.mentioned_at, self.now - 900)
        self.assertEqual(set(self.memory.load_source_attention()), {source})

        self.restart_attention_and_sessions()
        self.attention.restore_sources(self.memory.load_source_attention())
        self.assertEqual(self.attention.sources[source], recorded)

    async def test_idle_reflection_uses_older_work_when_preferred_has_no_memory(
        self,
    ) -> None:
        self.config.behavior.naps = False
        self.config.behavior.care = False
        remaining = "The docs still need an accessibility example."
        self.now += 1
        self.provider.publish(
            HookEvent(
                source=SessionSource.CODEX,
                session="docs",
                event=HookEventKind.INTENT,
                goal="Document accessibility examples",
                repo="/fixture/docs",
            )
        )
        self.report(remaining, session="docs")
        self.now += 1
        self.provider.publish(
            HookEvent(
                source=SessionSource.CODEX,
                session="parser",
                event=HookEventKind.INTENT,
                goal="Review quoted-field parser behavior",
                repo="/fixture/parser",
            )
        )
        self.attention.pending = None
        self.now += 3600
        self.attention.next_observation = self.now + 10_000
        selected: list[str] = []
        older_candidates: list[list[str]] = []

        def respond(request: ModelRequest, reason: SpeechReason) -> dict[str, object]:
            self.assertEqual(reason, SpeechReason.CHECK_IN)
            payload = json.loads(request["messages"][0]["content"])
            work = payload["current_work"]["request"]
            if work == "Review quoted-field parser behavior":
                return {"speak": None, "memory": None}
            self.assertEqual(work, "Document accessibility examples")
            if not older_candidates:
                self.assertEqual(self.memory.load_source_attention(), {})
            candidates = [
                item["memory_candidate"]
                for item in payload.get("earlier_context", ())
                if "memory_candidate" in item
            ]
            older_candidates.append([item["text"] for item in candidates])
            selected_report = next(
                (item for item in candidates if item["text"] == remaining), None
            )
            if selected_report is None:
                return {"speak": None, "memory": None}
            selected.append(selected_report["id"])
            return {
                "speak": "The accessibility example is still on the docs list.",
                "mentions": [selected_report["id"]],
                "memory": selected_report["id"],
                "ongoing": True,
            }

        self.model.response = respond
        self.app.tick()
        await require(self.app.task)
        requests = [
            json.loads(request["messages"][0]["content"])
            for request, _ in self.model.requests
        ]
        self.assertEqual(
            [item["current_work"]["request"] for item in requests],
            ["Review quoted-field parser behavior", "Document accessibility examples"],
        )
        self.assertEqual(len(selected), 1)
        emitted = next(item for item in reversed(self.packets) if "delivery_id" in item)
        packet: SpeechPacket = {
            "text": emitted["text"],
            "ttl": emitted["ttl"],
            "delivery_id": emitted["delivery_id"],
        }
        self.assertEqual(self.memory.load_source_attention(), {})
        message = self.app.delivery.deliveries[emitted["delivery_id"]].message
        source = message.mentioned_sources[0]
        self.assertNotEqual(source, selected[0])
        self.accept(packet)
        considered = require(self.memory.load_source_attention().get(source))
        self.assertEqual(considered.considered_at, self.now)
        self.assertEqual(set(self.memory.load_source_attention()), {source})

        self.now += 900
        self.app.tick()
        await require(self.app.task)
        self.assertEqual(len(self.model.requests), 4)
        self.assertNotIn(remaining, older_candidates[-1])
        self.assertEqual(len(self.memory.speech()), 1)
        self.assertEqual(self.memory.load_source_attention()[source], considered)

    async def test_new_activity_during_idle_generation_discards_the_reflection(
        self,
    ) -> None:
        report = "The quoted-field parser still needs a multiline escape check."
        self.report(report)
        self.complete_turn()
        self.attention.pending = None
        self.attention.next_observation = self.now + 10_000
        self.now += 300

        def publish_progress(reason: SpeechReason | ModelPurpose) -> None:
            if reason == SpeechReason.CHECK_IN:
                self.report("The multiline escape check passed.")

        self.model.before_response = publish_progress
        self.model.response = lambda request, reason: {"speak": None}
        self.app.tick()
        await require(self.app.task)
        self.assertFalse(any("delivery_id" in packet for packet in self.packets))
        self.assertEqual(self.memory.load_source_attention(), {})

    async def test_later_completion_excludes_an_uncited_older_remaining_step(
        self,
    ) -> None:
        self.config.behavior.naps = False
        remaining = "The parser still needs the multiline escape check."
        completed = "The multiline escape check passed and parser work is complete."
        self.report(remaining)
        self.now += 1
        self.report(completed)
        self.complete_turn()
        self.attention.pending = None
        self.attention.next_observation = self.now + 10_000
        older_source: list[str] = []

        def corrected_story(request: ModelRequest) -> dict[str, object]:
            payload = json.loads(request["messages"][0]["content"])
            evidence = payload["current_moment"]["evidence"]
            self.assertTrue(any(item.get("text") == remaining for item in evidence))
            later = next(item for item in evidence if item.get("text") == completed)
            latest = payload["current_work"]["latest_request_source"]
            intent = next(item for item in evidence if item["id"] == latest)
            return {
                "intent": {"text": intent["text"], "sources": [latest]},
                "decisions": [],
                "progress": [{"text": completed, "sources": [later["id"]]}],
            }

        def cite_old(request: ModelRequest, reason: SpeechReason) -> dict[str, object]:
            self.assertEqual(reason, SpeechReason.CHECK_IN)
            payload = json.loads(request["messages"][0]["content"])
            candidates = [
                item["memory_candidate"]
                for item in payload.get("earlier_context", ())
                if "memory_candidate" in item
            ]
            self.assertTrue(any(item["text"] == completed for item in candidates))
            self.assertFalse(any(item["text"] == remaining for item in candidates))
            parser = SessionIdentity(source=SessionSource.CODEX, native_id="parser")
            narrative = require(self.memory.load_narrative(parser))
            older_source.append(
                next(
                    identifier
                    for identifier, source in narrative.sources.items()
                    if source.get("text") == remaining
                )
            )
            return {
                "speak": "The multiline escape check still needs work.",
                "mentions": [older_source[0]],
                "memory": older_source[0],
                "ongoing": True,
            }

        self.model.narrative_response = corrected_story
        self.model.response = cite_old
        self.now += 300
        self.app.tick()
        await require(self.app.task)

        self.assertEqual(len(older_source), 1)
        self.assertFalse(any("delivery_id" in packet for packet in self.packets))
        self.assertEqual(self.memory.load_source_attention(), {})
        parser = SessionIdentity(source=SessionSource.CODEX, native_id="parser")
        narrative = require(self.memory.load_narrative(parser))
        self.assertIn(older_source[0], narrative.sources)
        self.assertEqual(narrative.progress[0].text, completed)

    async def test_explicit_idle_click_does_not_rearm_an_old_report_for_automatic_speech(
        self,
    ) -> None:
        self.config.behavior.naps = False
        self.config.behavior.care = False
        report = "The parser still needs the multiline escape check."
        self.report(report)
        self.complete_turn()
        self.attention.pending = None
        self.attention.next_observation = self.now + 200_000
        self.now += 86_400
        selected: list[str] = []
        automatic_candidates: list[list[dict[str, object]]] = []

        def respond(request: ModelRequest, reason: SpeechReason) -> dict[str, object]:
            payload = json.loads(request["messages"][0]["content"])
            candidates = [
                item["memory_candidate"]
                for item in payload.get("earlier_context", ())
                if "memory_candidate" in item
            ]
            if reason == SpeechReason.CLICK:
                candidate = next(item for item in candidates if item["text"] == report)
                identifier = candidate["id"]
                selected.append(identifier)
                return {
                    "speak": "The multiline escape check is still on the list.",
                    "mentions": [identifier],
                    "memory": identifier,
                    "ongoing": True,
                }
            self.assertEqual(reason, SpeechReason.CHECK_IN)
            automatic_candidates.append(candidates)
            return {"speak": None}

        self.model.response = respond
        self.app.click()
        self.app.tick()
        await require(self.app.task)
        emitted = next(
            packet for packet in reversed(self.packets) if "delivery_id" in packet
        )
        packet: SpeechPacket = {
            "text": emitted["text"],
            "ttl": emitted["ttl"],
            "delivery_id": emitted["delivery_id"],
        }
        message = self.app.delivery.deliveries[emitted["delivery_id"]].message
        source = message.mentioned_sources[0]
        self.accept(packet)
        mentioned = require(self.memory.load_source_attention().get(source))
        self.assertEqual(mentioned.mentioned_at, self.now)

        conversation = emitted["conversation"]
        self.app.receive_renderer(
            {
                "event": "interaction_dismissed",
                "interaction_id": conversation["id"],
                "revision": conversation["revision"],
            }
        )

        self.now += 900
        self.app.tick()
        await require(self.app.task)
        self.assertEqual(len(automatic_candidates), 1)
        self.assertNotIn(report, [item["text"] for item in automatic_candidates[0]])
        self.assertEqual(len(self.memory.speech()), 1)
        self.assertEqual(self.memory.load_source_attention()[source], mentioned)

    async def test_silent_reflection_keeps_its_interval_after_restart(self) -> None:
        self.config.behavior.naps = False
        self.config.behavior.care = False
        self.report("The quoted-field parser check passed.")
        self.complete_turn()
        self.attention.pending = None
        self.attention.next_observation = self.now + 10_000
        self.model.response = lambda request, reason: {"speak": None, "memory": None}

        self.now += 300
        self.app.tick()
        await require(self.app.task)
        self.assertEqual(self.model.requests[-1][1], SpeechReason.CHECK_IN)
        self.assertEqual(len(self.model.requests), 1)
        self.assertEqual(self.memory.load_source_attention(), {})
        parser = SessionIdentity(source=SessionSource.CODEX, native_id="parser")
        history = require(self.memory.load_attention_history().get(parser))
        self.assertEqual(history.last_reflection_at, self.now)

        self.restart_attention_and_sessions()
        self.attention.next_observation = self.now + 10_000
        self.now += 899
        self.app.tick()
        self.assertEqual(len(self.model.requests), 1)
        self.now += 1
        self.app.tick()
        await require(self.app.task)
        self.assertEqual(len(self.model.requests), 2)
        self.assertEqual(self.model.requests[-1][1], SpeechReason.CHECK_IN)
        self.assertEqual(self.memory.load_source_attention(), {})

    async def test_failed_idle_delivery_keeps_the_selected_report_retryable(
        self,
    ) -> None:
        self.config.behavior.naps = False
        self.config.behavior.care = False
        report = "The parser still needs the multiline escape check."
        self.report(report)
        self.complete_turn()
        self.attention.pending = None
        self.attention.next_observation = self.now + 10_000
        selected: list[str] = []

        def recall(request: ModelRequest, reason: SpeechReason) -> dict[str, object]:
            payload = json.loads(request["messages"][0]["content"])
            candidate = next(
                item["memory_candidate"]
                for item in payload.get("earlier_context", ())
                if "memory_candidate" in item
                and item["memory_candidate"]["text"] == report
            )
            selected.append(candidate["id"])
            return {
                "speak": "The multiline escape check remains open.",
                "mentions": [candidate["id"]],
                "memory": candidate["id"],
                "ongoing": True,
            }

        self.model.response = recall
        self.now += 300
        self.app.tick()
        await require(self.app.task)
        first = next(
            packet for packet in reversed(self.packets) if "delivery_id" in packet
        )
        first_message = self.app.delivery.deliveries[first["delivery_id"]].message
        for _ in range(2):
            self.now += 2
            self.app.delivery.tick()
        self.assertEqual(
            require(self.memory.message_outcome(first["delivery_id"])).outcome,
            DeliveryStatus.FAILED,
        )
        self.assertEqual(self.memory.load_source_attention(), {})

        self.now += 896
        self.app.tick()
        await require(self.app.task)
        second = next(
            packet for packet in reversed(self.packets) if "delivery_id" in packet
        )
        self.assertNotEqual(second["delivery_id"], first["delivery_id"])
        self.assertEqual(len(selected), 2)
        self.assertEqual(self.memory.load_source_attention(), {})
        message = self.app.delivery.deliveries[second["delivery_id"]].message
        source = message.mentioned_sources[0]
        self.assertEqual(message.mentioned_sources, first_message.mentioned_sources)
        self.accept(
            {
                "text": second["text"],
                "ttl": second["ttl"],
                "delivery_id": second["delivery_id"],
            }
        )
        attention = require(self.memory.load_source_attention().get(source))
        self.assertEqual(attention.considered_at, self.now)
        self.assertEqual(attention.mentioned_at, self.now)

    def interaction_snapshot(self) -> JSONObject:
        snapshot = json.loads(
            json.dumps(
                next(
                    packet["conversation"]
                    for packet in reversed(self.packets)
                    if "conversation" in packet
                )
            )
        )
        assert isinstance(snapshot, dict)
        return snapshot

    def interaction_item(self, snapshot: JSONObject) -> JSONObject:
        items = snapshot["items"]
        assert isinstance(items, list)
        item = items[0]
        assert isinstance(item, dict)
        return item

    def interaction_choice(
        self, item: JSONObject, kind: str, label: str | None = None
    ) -> JSONObject:
        actions = item["actions"]
        assert isinstance(actions, list)
        return next(
            action
            for action in actions
            if isinstance(action, dict)
            and action["kind"] == kind
            and (label is None or action["label"] == label)
        )

    def interaction_action(self, snapshot: JSONObject, action: JSONObject) -> None:
        self.app.receive_renderer(
            {
                "kind": "renderer",
                "event": "interaction_action",
                "interaction_id": snapshot["id"],
                "revision": snapshot["revision"],
                "action_id": action["id"],
            }
        )

    def interaction_model_response(
        self, request: ModelRequest, reason: SpeechReason
    ) -> dict[str, object]:
        payload = json.loads(request["messages"][0]["content"])
        candidates = [
            item["memory_candidate"]
            for item in payload.get("earlier_context", [])
            if "memory_candidate" in item
        ]
        if candidates:
            candidate = next(
                (
                    item
                    for item in reversed(candidates)
                    if item["kind"] == "reported_progress"
                ),
                candidates[-1],
            )
            return {
                "speak": self.model.text,
                "mentions": [candidate["id"]],
                "memory": candidate["id"],
                "ongoing": candidate["eligible_ongoing"],
            }
        sources = payload.get("current_moment", {}).get("source_ids", [])
        reports = [
            item["source_ids"]
            for item in payload.get("earlier_context", [])
            if "reported_progress" in item and item.get("source_ids")
        ]
        return {
            "speak": self.model.text,
            "mentions": sources or (reports[-1] if reports else []),
        }

    async def test_active_work_automatic_commentary_keeps_the_short_speech_schema(
        self,
    ) -> None:
        self.report("The parser preserves escaped delimiters.")
        await self.app.comment(SpeechReason.ROUTINE)
        packet = next(
            packet for packet in reversed(self.packets) if "delivery_id" in packet
        )
        self.assertEqual(packet["text"], self.model.text)
        self.assertNotIn("conversation", packet)
        request = self.model.requests[-1][0]
        schema = request["output_config"]["format"]["schema"]
        properties = schema["properties"]
        assert isinstance(properties, dict)
        self.assertEqual(set(properties), {"speak", "mentions"})
        payload = json.loads(request["messages"][0]["content"])
        self.assertNotIn("available_actions", payload)

    async def test_click_during_active_work_opens_a_conversation(self) -> None:
        self.report("The parser preserves escaped delimiters.")
        self.model.response = self.interaction_model_response
        await self.app.comment(SpeechReason.CLICK)
        snapshot = self.interaction_snapshot()
        self.assertEqual(snapshot["headline"], self.model.headline)
        self.assertEqual(self.interaction_item(snapshot)["text"], self.model.text)
        self.assertTrue(snapshot["persistent"])

    async def test_interaction_renders_model_choices_from_available_destinations(
        self,
    ) -> None:
        self.model.exchange_choices = ("worktree", "context")
        self.model.choice_labels = {
            "worktree": "Take me there",
            "context": "How did we get here?",
        }
        self.now += 1
        self.report("The parser preserves escaped delimiters.")
        self.complete_turn()
        self.model.response = self.interaction_model_response
        await self.app.comment(SpeechReason.CLICK)
        snapshot = self.interaction_snapshot()
        item = self.interaction_item(snapshot)
        actions = item["actions"]
        assert isinstance(actions, list)
        self.assertEqual(
            [
                (action["kind"], action["label"])
                for action in actions
                if isinstance(action, dict)
            ],
            [
                ("navigate", "Take me there"),
                ("context", "How did we get here?"),
            ],
        )
        payload = json.loads(self.model.requests[-1][0]["messages"][0]["content"])
        self.assertEqual(
            {action["id"] for action in payload["available_actions"]},
            {
                "worktree",
                "pull_request",
                "context",
                "tomorrow",
                "handled",
                "more",
                "rounds",
            },
        )
        self.assertEqual(snapshot["headline"], self.model.headline)
        self.assertEqual(snapshot["mode"], "thread")
        self.assertTrue(snapshot["persistent"])
        self.assertEqual(item["text"], self.model.text)
        self.assertIsInstance(item["id"], str)

    async def test_interaction_context_expands_the_attributed_narrative(self) -> None:
        self.now += 1
        self.report("The agent verified escaped delimiters inside quoted fields.")
        self.complete_turn()
        self.model.response = self.interaction_model_response
        await self.app.comment(SpeechReason.CLICK)
        snapshot = self.interaction_snapshot()
        item = self.interaction_item(snapshot)
        self.interaction_action(snapshot, self.interaction_choice(item, "context"))
        expanded = self.interaction_snapshot()
        self.assertEqual(expanded["mode"], "context")
        self.assertEqual(expanded["id"], snapshot["id"])
        before_revision, after_revision = snapshot["revision"], expanded["revision"]
        assert isinstance(before_revision, int) and isinstance(after_revision, int)
        self.assertGreater(after_revision, before_revision)
        context = self.interaction_item(expanded)["context"]
        self.assertIn(self.model.exchange_context, str(context))
        self.assertIn("Agent report", str(context))
        self.assertEqual(expanded["headline"], self.model.headline)
        actions = item["actions"]
        assert isinstance(actions, list)
        self.assertEqual(
            [action["kind"] for action in actions if isinstance(action, dict)],
            ["context"],
        )
        self.assertEqual(self.memory.load_source_attention(), {})

    async def test_interaction_tomorrow_defers_only_the_selected_source(self) -> None:
        from pet_daemon.navigation import next_morning

        self.model.exchange_choices = ("context", "tomorrow")
        self.now += 1
        self.report("The multiline escape check remains unverified.")
        self.complete_turn()
        self.model.response = self.interaction_model_response
        await self.app.comment(SpeechReason.CLICK)
        snapshot = self.interaction_snapshot()
        item = self.interaction_item(snapshot)
        source = str(item["id"])
        deadline = next_morning(self.now)
        self.interaction_action(snapshot, self.interaction_choice(item, "tomorrow"))
        feedback = self.memory.load_source_attention()
        self.assertEqual(set(feedback), {source})
        self.assertEqual(feedback[source].deferred_until, deadline)
        self.assertFalse(feedback[source].handled)
        self.assertTrue(self.packets[-1].get("dismiss_conversation"))
        self.assertFalse(self.attention.is_source_available(source, self.now))
        self.now = deadline
        self.assertTrue(self.attention.is_source_available(source, 1001))

    async def test_interaction_handled_source_survives_application_restart(
        self,
    ) -> None:
        self.model.exchange_choices = ("handled",)
        self.now += 1
        self.report("The multiline escape check remains unverified.")
        self.complete_turn()
        self.model.response = self.interaction_model_response
        await self.app.comment(SpeechReason.CLICK)
        snapshot = self.interaction_snapshot()
        item = self.interaction_item(snapshot)
        source = str(item["id"])
        self.interaction_action(snapshot, self.interaction_choice(item, "handled"))
        self.assertTrue(self.memory.load_source_attention()[source].handled)
        behavior = self.app.behavior
        await self.app.close()
        self.memory.close()
        self.memory = Memory(
            Path(self.temporary.name) / "memory.sqlite3",
            "fixture-pet",
            clock=self.clock,
        )
        self.attention = Attention(
            self.config, self.clock, random.Random(0), wall_clock=self.clock
        )
        self.app = Application(
            self.config,
            sessions=Sessions(self.clock, wall_clock=self.clock),
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
        self.model.response = self.interaction_model_response
        self.assertTrue(self.memory.load_source_attention()[source].handled)
        self.assertFalse(self.attention.is_source_available(source, self.now))
        self.app.receive_renderer({"kind": "renderer", "event": "rounds"})
        await require(self.app.task)
        items = self.interaction_snapshot()["items"]
        assert isinstance(items, list)
        self.assertGreater(len(items), 0)
        self.assertNotIn(
            source, [item["id"] for item in items if isinstance(item, dict)]
        )

    async def test_interaction_navigation_records_engagement_and_finite_spacing(
        self,
    ) -> None:
        from unittest.mock import patch

        from pet_daemon.navigation import Navigator, OpenResult, Target

        self.model.exchange_choices = ("pull_request",)
        self.now += 1
        reported_at = self.now
        self.report("The parser preserves escaped delimiters.")
        self.complete_turn()
        self.model.response = self.interaction_model_response
        await self.app.comment(SpeechReason.CLICK)
        snapshot = self.interaction_snapshot()
        item = self.interaction_item(snapshot)
        source = str(item["id"])
        opened = asyncio.Event()
        targets: list[Target] = []

        async def open_target(navigator: Navigator, target: Target) -> OpenResult:
            targets.append(target)
            opened.set()
            return OpenResult(True)

        with patch.object(Navigator, "open", open_target):
            self.interaction_action(
                snapshot, self.interaction_choice(item, "navigate", "See PR")
            )
            await asyncio.wait_for(opened.wait(), 1)
        self.assertEqual(
            [target.value for target in targets], [require(self.observation.pr).url]
        )
        feedback = self.memory.load_source_attention()[source]
        self.assertEqual(feedback.engaged_at, self.now)
        self.assertFalse(feedback.handled)
        self.assertIsNone(feedback.deferred_until)
        self.assertTrue(self.packets[-1].get("dismiss_conversation"))
        self.assertFalse(self.attention.is_source_available(source, reported_at))
        await self.app.comment(SpeechReason.CLICK)
        remaining = self.interaction_snapshot()["items"]
        assert isinstance(remaining, list)
        self.assertNotIn(
            source, [item["id"] for item in remaining if isinstance(item, dict)]
        )
        self.now += 900
        self.assertTrue(self.attention.is_source_available(source, reported_at))
        await self.app.comment(SpeechReason.CLICK)
        self.assertEqual(
            self.interaction_item(self.interaction_snapshot())["id"], source
        )

    async def test_interaction_navigation_failure_preserves_the_thread(self) -> None:
        from unittest.mock import patch

        from pet_daemon.navigation import Navigator, OpenResult, Target

        self.model.exchange_choices = ("worktree",)
        self.now += 1
        self.report("The parser preserves escaped delimiters.")
        self.complete_turn()
        self.model.response = self.interaction_model_response
        await self.app.comment(SpeechReason.CLICK)
        snapshot = self.interaction_snapshot()
        item = self.interaction_item(snapshot)
        opened = asyncio.Event()
        targets: list[Target] = []

        async def open_target(navigator: Navigator, target: Target) -> OpenResult:
            targets.append(target)
            opened.set()
            return OpenResult(False, "Checkout is unavailable")

        with patch.object(Navigator, "open", open_target):
            self.interaction_action(
                snapshot, self.interaction_choice(item, "navigate", "Show worktree")
            )
            await asyncio.wait_for(opened.wait(), 1)
        failed = self.interaction_snapshot()
        self.assertEqual([target.value for target in targets], [self.checkout.worktree])
        self.assertEqual(failed["id"], snapshot["id"])
        self.assertEqual(failed["mode"], "thread")
        self.assertEqual(self.interaction_item(failed)["id"], item["id"])
        self.assertIn("isn't available", str(failed["error"]))
        self.assertEqual(self.memory.load_source_attention(), {})

    async def test_interaction_stale_and_unknown_actions_have_no_effect(self) -> None:
        from unittest.mock import patch

        from pet_daemon.navigation import Navigator, OpenResult, Target

        self.model.exchange_choices = ("worktree", "handled")
        self.now += 1
        self.report("The parser preserves escaped delimiters.")
        self.complete_turn()
        self.model.response = self.interaction_model_response
        await self.app.comment(SpeechReason.CLICK)
        snapshot = self.interaction_snapshot()
        item = self.interaction_item(snapshot)
        opened: list[Target] = []

        async def open_target(navigator: Navigator, target: Target) -> OpenResult:
            opened.append(target)
            return OpenResult(True)

        with patch.object(Navigator, "open", open_target):
            stale = dict(snapshot)
            revision = snapshot["revision"]
            assert isinstance(revision, int)
            stale["revision"] = revision - 1
            self.interaction_action(stale, self.interaction_choice(item, "navigate"))
            self.interaction_action(
                self.interaction_snapshot(), {"id": str(uuid.uuid4())}
            )
            replaced = dict(snapshot)
            replaced["id"] = str(uuid.uuid4())
            self.interaction_action(replaced, self.interaction_choice(item, "handled"))
            await asyncio.sleep(0)
        self.assertEqual(opened, [])
        self.assertEqual(self.memory.load_source_attention(), {})
        refreshed = self.interaction_snapshot()
        self.assertEqual(refreshed["id"], snapshot["id"])
        self.assertEqual(self.interaction_item(refreshed)["id"], item["id"])
        self.assertIn("choose again", str(refreshed["error"]))

    async def test_interaction_rounds_include_informative_and_remaining_reports(
        self,
    ) -> None:
        self.now += 1
        self.report("Escaped delimiters are preserved inside quoted fields.")
        self.provider.publish(
            HookEvent(
                source=SessionSource.CODEX,
                session="renderer",
                event=HookEventKind.INTENT,
                goal="Document renderer compatibility",
                repo="/fixture/renderer",
            )
        )
        self.now += 1
        self.report("Desktop compatibility remains unverified.", session="renderer")
        self.model.response = self.interaction_model_response
        self.app.receive_renderer({"kind": "renderer", "event": "rounds"})
        await require(self.app.task)
        snapshot = self.interaction_snapshot()
        self.assertEqual(snapshot["mode"], "rounds")
        items = snapshot["items"]
        assert isinstance(items, list)
        self.assertEqual(len(items), 2)
        texts = [item["text"] for item in items if isinstance(item, dict)]
        self.assertIn(
            "Last report: Escaped delimiters are preserved inside quoted fields.",
            texts,
        )
        self.assertIn("Last report: Desktop compatibility remains unverified.", texts)
        source_ids = [item["id"] for item in items if isinstance(item, dict)]
        self.assertEqual(len(set(source_ids)), 2)
        self.assertEqual(self.memory.load_source_attention(), {})

    async def test_interaction_rounds_reuse_memory_and_return_to_a_single_row(
        self,
    ) -> None:
        self.now += 1
        self.report("Desktop validation remains unverified.")
        self.app.rounds()
        await require(self.app.task)
        snapshot = self.interaction_snapshot()
        item = self.interaction_item(snapshot)
        self.interaction_action(snapshot, self.interaction_choice(item, "context"))
        expanded = self.interaction_snapshot()
        self.interaction_action(expanded, self.interaction_choice(expanded, "back"))
        self.assertEqual(self.interaction_snapshot()["mode"], "rounds")
        self.assertEqual(self.model.requests, [])

    async def test_interaction_context_stops_the_earlier_delivery_retry(self) -> None:
        self.now += 1
        self.report("The parser preserves escaped delimiters.")
        self.complete_turn()
        self.model.response = self.interaction_model_response
        await self.app.comment(SpeechReason.CLICK)
        snapshot = self.interaction_snapshot()
        self.interaction_action(
            snapshot,
            self.interaction_choice(self.interaction_item(snapshot), "context"),
        )
        count = len(self.packets)
        self.now += 2
        self.app.delivery.tick()
        self.assertEqual(len(self.packets), count)
        self.assertEqual(self.interaction_snapshot()["mode"], "context")

    async def test_interaction_explicit_speech_releases_the_open_conversation(
        self,
    ) -> None:
        self.now += 1
        self.report("The parser preserves escaped delimiters.")
        self.complete_turn()
        self.model.response = self.interaction_model_response
        await self.app.comment(SpeechReason.CLICK)
        self.assertTrue(self.app.interactions.is_interacting)
        self.app.speak("React to the next parser check.")
        self.assertFalse(self.app.interactions.is_interacting)
        self.assertTrue(self.packets[-1].get("dismiss_conversation"))

    async def test_interaction_navigation_uses_the_displayed_report_checkout(
        self,
    ) -> None:
        from unittest.mock import patch

        from pet_daemon.navigation import Navigator, OpenResult, Target

        self.model.exchange_choices = ("worktree",)
        self.now += 1
        self.report("The parser preserves escaped delimiters.")
        self.complete_turn()
        self.model.response = self.interaction_model_response
        await self.app.comment(SpeechReason.CLICK)
        snapshot = self.interaction_snapshot()
        item = self.interaction_item(snapshot)
        self.provider.publish(
            HookEvent(
                source=SessionSource.CODEX,
                session="parser",
                event=HookEventKind.INTENT,
                goal="Inspect another checkout",
                repo="/fixture/other-checkout",
            )
        )
        opened: list[Target] = []
        finished = asyncio.Event()

        async def open_target(navigator: Navigator, target: Target) -> OpenResult:
            opened.append(target)
            finished.set()
            return OpenResult(True)

        with patch.object(Navigator, "open", open_target):
            self.interaction_action(
                snapshot, self.interaction_choice(item, "navigate", "Show worktree")
            )
            await asyncio.wait_for(finished.wait(), 1)
        self.assertEqual([target.value for target in opened], ["/fixture/parser"])

    async def test_interaction_dismissal_discards_a_pending_rounds_response(
        self,
    ) -> None:
        from unittest.mock import patch

        self.now += 1
        self.report("The parser preserves escaped delimiters.")
        self.model.response = self.interaction_model_response
        started, release = asyncio.Event(), asyncio.Event()
        complete = self.model.complete

        async def held_response(
            request: ModelRequest,
            reason: SpeechReason | ModelPurpose,
            *,
            response_type: type[BaseModel],
        ) -> dict[str, object]:
            if reason == ModelPurpose.STORY:
                started.set()
                await release.wait()
            return await complete(request, reason, response_type=response_type)

        with patch.object(self.model, "complete", held_response):
            self.app.receive_renderer({"kind": "renderer", "event": "rounds"})
            await asyncio.wait_for(started.wait(), 1)
            loading = self.interaction_snapshot()
            self.assertEqual(loading["mode"], "loading")
            self.app.receive_renderer(
                {
                    "kind": "renderer",
                    "event": "interaction_dismissed",
                    "interaction_id": loading["id"],
                    "revision": loading["revision"],
                }
            )
            count = len(self.packets)
            release.set()
            await asyncio.gather(require(self.app.task), return_exceptions=True)
        self.assertEqual(len(self.packets), count)
        self.assertTrue(self.packets[-1].get("dismiss_conversation"))
        self.assertEqual(self.memory.speech(), [])
        self.assertEqual(self.memory.load_source_attention(), {})

    async def test_interaction_another_thread_reports_exhaustion(self) -> None:
        self.model.exchange_choices = ("more",)
        self.now += 1
        self.report("The parser preserves escaped delimiters.")
        self.complete_turn()
        self.model.response = self.interaction_model_response
        await self.app.comment(SpeechReason.CLICK)
        snapshot = self.interaction_snapshot()
        requests = len(self.model.requests)
        self.interaction_action(snapshot, self.interaction_choice(snapshot, "more"))
        self.app.tick()
        exhausted = self.interaction_snapshot()
        self.assertEqual(exhausted["mode"], "empty")
        self.assertEqual(exhausted["items"], [])
        self.assertEqual(len(self.model.requests), requests)
        self.assertEqual(self.memory.load_source_attention(), {})

    async def test_interaction_dismissal_cancels_one_pending_navigation(self) -> None:
        from unittest.mock import patch

        from pet_daemon.navigation import Navigator, OpenResult, Target

        self.model.exchange_choices = ("worktree",)
        self.now += 1
        self.report("The parser preserves escaped delimiters.")
        self.complete_turn()
        self.model.response = self.interaction_model_response
        await self.app.comment(SpeechReason.CLICK)
        snapshot = self.interaction_snapshot()
        item = self.interaction_item(snapshot)
        started, cancelled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        targets: list[Target] = []

        async def open_target(navigator: Navigator, target: Target) -> OpenResult:
            targets.append(target)
            started.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise
            return OpenResult(True)

        with patch.object(Navigator, "open", open_target):
            action = self.interaction_choice(item, "navigate")
            self.interaction_action(snapshot, action)
            self.interaction_action(snapshot, action)
            await asyncio.wait_for(started.wait(), 1)
            current = self.interaction_snapshot()
            self.app.receive_renderer(
                {
                    "kind": "renderer",
                    "event": "interaction_dismissed",
                    "interaction_id": current["id"],
                    "revision": current["revision"],
                }
            )
            await asyncio.wait_for(cancelled.wait(), 1)
        self.assertEqual(len(targets), 1)
        self.assertEqual(self.memory.load_source_attention(), {})
        self.assertTrue(self.packets[-1].get("dismiss_conversation"))
