import json
import random
import tempfile
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Unpack

from pet_daemon.attention import Attention
from pet_daemon.commentary import Cognition
from pet_daemon.commentary.context import CommentaryPayload, PreparedCommentary
from pet_daemon.commentary.contracts import ModelRequest
from pet_daemon.config import Config
from pet_daemon.domain import (
    ArtifactKind,
    CareKind,
    ExecutionOutcome,
    ModelPurpose,
    ProcessStatus,
    PullRequestState,
    SpeechReason,
)
from pet_daemon.runtime import CareTrigger, HookEvent, PetContext
from pet_daemon.sessions.tracking import Sessions
from pet_daemon.storage import Memory
from pet_daemon.work import WorkSubject
from pet_daemon.work.history import build_session_entry
from pet_daemon.work.observations import (
    BranchObservation,
    CheckoutObservation,
    GitObservation,
    PullRequest,
    PullRequestObservation,
)
from pet_daemon.work.views import ArtifactContext
from Tests.fixtures import (
    SessionOptions,
    capture_snapshot,
    session_snapshot,
    sessions_snapshot,
)


class NoRequestsModel:
    async def complete(
        self, request: ModelRequest, reason: SpeechReason | ModelPurpose
    ) -> dict[str, object]:
        raise AssertionError("PromptBuilder preparation must not request model output")


class ContextTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = Config(pet_dir=Path(self.temp.name))
        self.memory = Memory(
            Path(self.temp.name) / "memory.sqlite3", "pet", clock=lambda: 1000
        )
        self.addCleanup(self.memory.close)
        self.cognition = Cognition(
            NoRequestsModel(), self.config, clock=lambda: 1000, memory=self.memory
        )
        self.attention = Attention(self.config, lambda: 1000, random.Random(0))
        self.snapshot = sessions_snapshot(
            session_snapshot(
                id="cli:parser",
                turn_active=True,
                generation=1,
                recent_requests=(
                    "Explore parsing options",
                    "Keep the public API stable",
                ),
                progress="Checking parser compatibility",
                summary="Implement the parser.",
            )
        )
        self.work: tuple[WorkSubject, ...] = ()
        self.sessions = Sessions(lambda: 1000)

    def record_hook(self, hook: HookEvent) -> None:
        result = self.sessions.apply(hook)
        assert result is not None
        with self.memory.transaction():
            self.memory.work.record_session(result.observation)
            self.memory.observe_session_context(result.observation)
            self.memory.record_event(build_session_entry(hook))

    def update_session(self, index: int = 0, **values: Unpack[SessionOptions]) -> None:
        sessions = list(self.snapshot.sessions)
        sessions[index] = replace(sessions[index], **values)
        self.snapshot = replace(self.snapshot, sessions=tuple(sessions))

    def current_snapshot(self) -> PetContext:
        return capture_snapshot(self.snapshot, work=self.work)

    def prepare(
        self, reason: SpeechReason = SpeechReason.ROUTINE, context: str | None = None
    ) -> PreparedCommentary:
        selection = self.attention.select(self.current_snapshot(), reason, context)
        return self.cognition.prompts.prepare(
            self.current_snapshot(), selection, self.cognition.request
        )

    def payload(self, prepared: PreparedCommentary) -> CommentaryPayload:
        return json.loads(prepared.request["messages"][0]["content"])

    def test_small_budget_keeps_intent_latest_request_and_progress(self) -> None:
        self.config.prompt_target_bytes = 1
        before = deepcopy(self.snapshot)
        prepared = self.prepare()
        focus = self.payload(prepared)["focus_session"]
        assert focus is not None
        self.assertEqual(focus["recent_requests"], ["Keep the public API stable"])
        self.assertEqual(focus["summary"], "Implement the parser.")
        self.assertEqual(focus["progress"], "Checking parser compatibility")
        self.assertEqual(self.snapshot, before)

    def test_available_budget_keeps_requests_in_chronological_order(self) -> None:
        focus = self.payload(self.prepare())["focus_session"]
        assert focus is not None
        self.assertEqual(
            focus["recent_requests"], list(self.snapshot.sessions[0].recent_requests)
        )

    def test_manifest_ignores_an_omitted_session_changing(self) -> None:
        self.config.prompt_target_bytes = 1
        self.snapshot = replace(
            self.snapshot,
            sessions=(
                *self.snapshot.sessions,
                session_snapshot(
                    id="cli:renderer", turn_active=True, progress="Checking layout"
                ),
            ),
        )
        prepared = self.prepare()
        self.assertEqual(prepared.support.session_ids, ("cli:parser",))
        self.update_session(1, progress="Renderer completed", turn_active=False)
        self.assertTrue(prepared.support.valid(self.current_snapshot()))

    def test_manifest_ignores_a_request_omitted_by_the_budget(self) -> None:
        self.config.prompt_target_bytes = 1
        prepared = self.prepare()
        self.update_session(
            recent_requests=(
                "Discarded exploration",
                *self.snapshot.sessions[0].recent_requests[1:],
            )
        )
        self.assertTrue(prepared.support.valid(self.current_snapshot()))

    def test_manifest_rejects_changed_included_progress(self) -> None:
        prepared = self.prepare()
        self.update_session(progress="Parser verification complete")
        self.assertFalse(prepared.support.valid(self.current_snapshot()))

    def test_manifest_ignores_elapsed_time_and_metadata(self) -> None:
        prepared = self.prepare()
        self.update_session(duration_seconds=300, quiet_seconds=150, changed_at=1200)
        self.assertTrue(prepared.support.valid(self.current_snapshot()))

    def test_process_presence_metadata_does_not_supersede_ongoing_work(self) -> None:
        self.update_session(presence=ProcessStatus.UNKNOWN)
        prepared = self.prepare()
        self.update_session(presence=ProcessStatus.ALIVE)
        self.assertTrue(prepared.support.valid(self.current_snapshot()))

    def test_quiet_connected_work_keeps_its_intent_without_a_live_session(self) -> None:
        self.record_hook(
            HookEvent("cli", "parser", "intent", goal="Preserve parser compatibility")
        )
        self.snapshot = replace(self.snapshot, sessions=())
        self.work = self.memory.connected_work(self.snapshot)
        prepared = self.prepare()
        payload = self.payload(prepared)
        self.assertIsNone(payload["focus_session"])
        self.assertEqual(
            payload["focus_work"]["intent"], "Preserve parser compatibility"
        )
        self.assertTrue(prepared.support.valid(self.current_snapshot()))

    def test_manifest_rejects_a_superseded_runtime_transition(self) -> None:
        prepared = self.prepare()
        self.update_session(turn_active=False, outcome=ExecutionOutcome.SUCCESS)
        self.assertFalse(prepared.support.valid(self.current_snapshot()))

    def test_quiet_greeting_remains_valid_without_session_context(self) -> None:
        self.snapshot = replace(self.snapshot, sessions=())
        prepared = self.prepare(SpeechReason.CLICK)
        self.assertTrue(prepared.support.valid(self.current_snapshot()))

    def test_explicit_context_stays_in_data_and_keeps_work_out_of_the_prompt(
        self,
    ) -> None:
        prepared = self.prepare(context="Inspect the parser result")
        payload = self.payload(prepared)
        self.assertEqual(payload["context"], "Inspect the parser result")
        self.assertIsNone(payload["focus_session"])
        self.assertNotIn(
            "Inspect the parser result", json.dumps(prepared.request["system"])
        )

    def output(
        self,
        *,
        head: str = "one",
        state: PullRequestState = PullRequestState.OPEN,
        is_draft: bool = True,
    ) -> None:
        if not self.memory.recent(session="cli:parser", event_types=("intent",)):
            self.record_hook(
                HookEvent(
                    "cli",
                    "parser",
                    "intent",
                    goal="Make session tracking survive restarts",
                )
            )
        self.memory.work.observe(
            CheckoutObservation(
                "/parser",
                GitObservation(
                    "/parser",
                    "persistence",
                    head,
                    0,
                    7,
                    0,
                    0,
                    "",
                    0,
                    0,
                    "example.test/team/parser",
                ),
                1000,
            )
        )
        self.record_hook(HookEvent("cli", "parser", "session_start", repo="/parser"))
        self.memory.work.observe(
            BranchObservation(
                "example.test/team/parser",
                "persistence",
                PullRequest(
                    42,
                    "https://example.test/team/parser/pull/42",
                    state,
                    None,
                    is_draft=is_draft,
                    title="Persist coding session context",
                    head=head,
                    head_repository="example.test/team/parser",
                    head_branch="persistence",
                ),
                1000,
            )
        )
        self.work = self.memory.connected_work(self.snapshot)

    def test_presented_work_omits_checkout_paths_and_internal_session_names(
        self,
    ) -> None:
        self.output()
        encoded = json.dumps(self.payload(self.prepare()))
        self.assertNotIn('"checkout"', encoded)
        self.assertNotIn('"/parser"', encoded)
        self.assertNotIn("cli:parser", encoded)
        self.assertIn("https://example.test/team/parser/pull/42", encoded)

    def test_head_metadata_does_not_invalidate_the_agent_story(self) -> None:
        self.output()
        prepared = self.prepare()
        self.output(head="two")
        self.assertTrue(prepared.support.valid(self.current_snapshot()))

    def test_supplied_pull_request_state_invalidates_changed_context(self) -> None:
        self.output()
        prepared = self.prepare()
        self.output(state=PullRequestState.MERGED)
        self.assertFalse(prepared.support.valid(self.current_snapshot()))

    def test_exact_pull_request_refresh_supplies_one_current_state(self) -> None:
        self.output()
        pr = PullRequest(
            42,
            "https://example.test/team/parser/pull/42",
            PullRequestState.MERGED,
            "2026-10-21T00:00:00Z",
            title="Persist coding session context",
        )
        self.memory.work.observe(PullRequestObservation(pr.url, pr, 1001))
        self.work = self.memory.connected_work(self.snapshot)
        encoded = json.dumps(self.payload(self.prepare()))
        self.assertIn(PullRequestState.MERGED, encoded)
        self.assertNotIn("OPEN", encoded)

    def test_new_artifact_context_does_not_invalidate_captured_story(self) -> None:
        self.output()
        prepared = self.prepare()
        self.work = tuple(
            replace(
                subject,
                story=replace(
                    subject.story,
                    artifacts=(
                        *subject.story.artifacts,
                        ArtifactContext(
                            ArtifactKind.TICKET, "https://example.test/tickets/123"
                        ),
                    ),
                ),
            )
            for subject in self.work
        )
        self.assertTrue(prepared.support.valid(self.current_snapshot()))

    def test_disconnected_focus_invalidates_the_captured_subject(self) -> None:
        self.output()
        prepared = self.prepare()
        self.work = tuple(replace(subject, sessions=()) for subject in self.work)
        self.assertFalse(prepared.support.valid(self.current_snapshot()))

    def test_selected_work_context_is_supplied_once(self) -> None:
        self.output()
        self.update_session(connected_work=self.work[0].story)
        payload = self.payload(self.prepare())
        self.assertIn("focus_work", payload)
        focus = payload["focus_session"]
        assert focus is not None
        self.assertNotIn("connected_work", focus)

    def test_resumed_work_invalidates_a_quiet_care_remark(self) -> None:
        self.snapshot = sessions_snapshot()
        snapshot = replace(
            self.current_snapshot(), trigger=CareTrigger(CareKind.EXERCISE)
        )
        selection = self.attention.select(snapshot, SpeechReason.CARE)
        prepared = self.cognition.prompts.prepare(
            snapshot, selection, self.cognition.request
        )
        self.assertTrue(prepared.support.valid(snapshot))
        resumed = capture_snapshot(
            sessions_snapshot(session_snapshot(turn_active=True))
        )
        self.assertFalse(prepared.support.valid(resumed))

    def test_new_input_wait_invalidates_a_quiet_greeting(self) -> None:
        self.snapshot = sessions_snapshot()
        prepared = self.prepare(SpeechReason.CLICK)
        waiting = capture_snapshot(
            sessions_snapshot(session_snapshot(waiting_for_input=True))
        )
        self.assertFalse(prepared.support.valid(waiting))

    def test_summary_saved_from_the_same_result_preserves_commentary(self) -> None:
        prepared = self.prepare()
        self.update_session(summary="The parser preserves its public API.")
        self.assertTrue(prepared.support.valid(self.current_snapshot()))
