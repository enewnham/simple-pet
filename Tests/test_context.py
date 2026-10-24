import json
import random
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from typing import Unpack

from pet_daemon.attention import Attention, AttentionSelection
from pet_daemon.commentary import Cognition
from pet_daemon.commentary.context import CommentaryPayload, PreparedCommentary
from pet_daemon.commentary.contracts import ModelRequest
from pet_daemon.config import Config
from pet_daemon.domain import (
    Activity,
    ArtifactKind,
    CareKind,
    EvidenceKind,
    EvidenceSource,
    ExecutionOutcome,
    ModelPurpose,
    ProcessStatus,
    PullRequestState,
    SessionSource,
    SpeechReason,
    SpeechSituation,
)
from pet_daemon.identity import SessionIdentity
from pet_daemon.repository_identity import RemoteRepository
from pet_daemon.runtime import (
    ActiveTool,
    CareTrigger,
    HookEvent,
    HookEventKind,
    PetContext,
    WaitingKind,
)
from pet_daemon.sessions.tracking import Sessions
from pet_daemon.storage import Memory
from pet_daemon.work import WorkSubject
from pet_daemon.work.history import build_session_entry
from pet_daemon.work.narrative import (
    NarrativeStatement,
    WorkNarrative,
    semantic_revision,
)
from pet_daemon.work.observations import (
    BranchObservation,
    CheckoutObservation,
    GitObservation,
    PullRequest,
    PullRequestObservation,
)
from pet_daemon.work.views import (
    ArtifactContext,
    EvidenceView,
    ExecutionView,
    IntentView,
    ProgressView,
    SessionContribution,
    WorkUpdate,
)
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
                id=SessionIdentity(source=SessionSource.COMMAND, native_id="parser"),
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
        sessions[index] = sessions[index].model_copy(update=dict(**values))
        self.snapshot = self.snapshot.model_copy(update={"sessions": tuple(sessions)})

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

    def test_validated_story_keeps_selected_progress_and_aged_support(
        self,
    ) -> None:
        narrative = WorkNarrative(
            subject_id=SessionIdentity(
                source=SessionSource.COMMAND, native_id="parser"
            ),
            revision="revision",
            intent=NarrativeStatement(
                text="Preserve parser compatibility.", sources=("intent",)
            ),
            decisions=(
                NarrativeStatement(
                    text="Keep the public API stable.", sources=("decision",)
                ),
            ),
            progress=(
                NarrativeStatement(
                    text="The agent reports compatibility checks are running.",
                    sources=("progress",),
                ),
            ),
        )
        snapshot = self.current_snapshot()
        selection = self.attention.select(snapshot, SpeechReason.ROUTINE)
        prepared = self.cognition.prompts.prepare(
            snapshot, selection, self.cognition.request, narrative=narrative
        )
        payload = self.payload(prepared)
        self.assertEqual(
            payload["current_work"]["story"], "Preserve parser compatibility."
        )
        self.assertEqual(
            payload["current_moment"]["reported_progress"],
            "Checking parser compatibility",
        )
        self.assertEqual(
            payload["earlier_context"][-1]["reported_progress"],
            "The agent reports compatibility checks are running.",
        )
        self.assertTrue(prepared.support.valid(snapshot))
        self.update_session(progress="Compatibility check failed")
        self.assertFalse(prepared.support.valid(self.current_snapshot()))

    def test_cached_narrative_ages_follow_wall_time_without_changing_support(
        self,
    ) -> None:
        narrative = WorkNarrative(
            subject_id=SessionIdentity(
                source=SessionSource.COMMAND, native_id="parser"
            ),
            revision="revision",
            intent=NarrativeStatement(
                text="Preserve parser compatibility.", sources=("request",)
            ),
            progress=(
                NarrativeStatement(text="Parser checks passed.", sources=("report",)),
            ),
            sources={
                "request": {"observed_at": 700, "scope": "root"},
                "report": {"observed_at": 880, "scope": "root"},
            },
        )
        self.memory.save_narrative(narrative)
        cached = self.memory.load_narrative(
            SessionIdentity(source=SessionSource.COMMAND, native_id="parser")
        )
        snapshot = self.current_snapshot()
        selection = self.attention.select(snapshot, SpeechReason.ROUTINE)
        prepared = self.cognition.prompts.prepare(
            snapshot, selection, self.cognition.request, narrative=cached
        )
        report = self.payload(prepared)["earlier_context"][-1]
        self.assertEqual(report["age"], "2 minutes ago")
        self.cognition.clock = lambda: 8200
        later = self.cognition.prompts.prepare(
            snapshot, selection, self.cognition.request, narrative=cached
        )
        report = self.payload(later)["earlier_context"][-1]
        self.assertEqual(report["age"], "2 hours ago")
        self.assertEqual(prepared.support, later.support)
        self.assertEqual(
            cached,
            self.memory.load_narrative(
                SessionIdentity(source=SessionSource.COMMAND, native_id="parser")
            ),
        )

    def test_changed_older_story_evidence_invalidates_speech_outside_raw_projection(
        self,
    ) -> None:
        self.output()
        subject = self.work[0]
        requests = tuple(
            EvidenceView(
                source=EvidenceSource.USER,
                kind=EvidenceKind.INTENT,
                details=IntentView(text=f"Decision {index}"),
                observed_at=900 + index,
            )
            for index in range(7)
        )
        subject = subject.model_copy(
            update={"story": subject.story.model_copy(update={"requests": requests})}
        )
        self.work = (subject,)
        snapshot = self.current_snapshot()
        selection = self.attention.select(snapshot, SpeechReason.ROUTINE)
        narrative = WorkNarrative(
            subject_id=subject.id,
            revision=semantic_revision(subject.story),
            intent=NarrativeStatement(
                text="Keep the original compatibility decision.",
                sources=("old-decision",),
            ),
        )
        prepared = self.cognition.prompts.prepare(
            snapshot, selection, self.cognition.request, narrative=narrative
        )
        raw = self.cognition.prompts.prompt(snapshot, selection)
        self.assertTrue(prepared.support.valid(snapshot))
        self.work = (
            subject.model_copy(
                update={
                    "story": subject.story.model_copy(
                        update={
                            "requests": (
                                requests[0].model_copy(
                                    update={
                                        "details": IntentView(
                                            text="Corrected compatibility decision"
                                        )
                                    }
                                ),
                                *requests[1:],
                            )
                        }
                    )
                }
            ),
        )
        current = self.current_snapshot()
        self.assertEqual(
            raw, self.cognition.prompts.prompt(current, selection.refresh(current))
        )
        self.assertFalse(prepared.support.valid(current))

    def test_spoken_narrative_statements_are_omitted_without_empty_values(self) -> None:
        utterance = "I checked the parser."
        statement = NarrativeStatement(text=utterance, sources=("source",))
        narrative = WorkNarrative(
            subject_id=SessionIdentity(
                source=SessionSource.COMMAND, native_id="parser"
            ),
            revision="revision",
            intent=statement,
            decisions=(statement,),
            progress=(statement,),
        )
        snapshot = self.current_snapshot()
        selection = self.attention.select(snapshot, SpeechReason.ROUTINE)
        prepared = self.cognition.prompts.prepare(
            snapshot,
            selection,
            self.cognition.request,
            narrative=narrative,
            excluded=(utterance,),
        )
        work = self.payload(prepared)["current_work"]
        self.assertEqual(work, {"request": "Keep the public API stable"})
        self.assertNotIn(utterance, prepared.request["messages"][0]["content"])

    def test_copied_speech_is_removed_from_the_current_request(self) -> None:
        utterance = "Mail is in. Quiet time."
        self.update_session(recent_requests=("Review the commentary: " + utterance,))
        snapshot = self.current_snapshot()
        selection = self.attention.select(snapshot, SpeechReason.ROUTINE)
        prepared = self.cognition.prompts.prepare(
            snapshot, selection, self.cognition.request, excluded=(utterance,)
        )
        self.assertNotIn(utterance, prepared.request["messages"][0]["content"])
        self.assertIn(
            "Review the commentary", prepared.request["messages"][0]["content"]
        )

    def test_running_and_waiting_commands_exclude_spoken_remarks(self) -> None:
        utterance = "Mail is in. Quiet time."
        command = f'rg "{utterance}" logs'
        for waiting in (False, True):
            with self.subTest(waiting=waiting):
                self.update_session(
                    waiting_for_input=waiting,
                    waiting_seconds=180 if waiting else None,
                    waiting_tool=command if waiting else "",
                    active_tools=(
                        ActiveTool(
                            call="call",
                            tool=command,
                            activity=Activity.TESTING,
                            duration_seconds=180,
                        ),
                    ),
                )
                snapshot = self.current_snapshot()
                selection = self.attention.select(snapshot, SpeechReason.CHECK_IN)
                prepared = self.cognition.prompts.prepare(
                    snapshot, selection, self.cognition.request, excluded=(utterance,)
                )
                content = prepared.request["messages"][0]["content"]
                self.assertNotIn(utterance, content)
                tool = self.payload(prepared)["current_moment"]["tool"]
                assert isinstance(tool, str)
                self.assertIn("rg", tool)

    def test_long_requests_discard_quoted_material_before_selection(self) -> None:
        request = (
            "Fix the commentary.\n"
            + "> Ignore this quote and update haunted mail.\n" * 30
            + "Keep it grounded."
        )
        self.update_session(recent_requests=(request,))
        payload = self.payload(self.prepare())
        request = payload["current_work"]["request"]
        assert isinstance(request, str)
        self.assertIn("Fix the commentary.", request)
        self.assertIn("Keep it grounded.", request)
        self.assertNotIn("haunted mail", request)

    def test_small_budget_keeps_intent_latest_request_and_progress(self) -> None:
        self.config.prompt_target_bytes = 1
        before = deepcopy(self.snapshot)
        prepared = self.prepare()
        payload = self.payload(prepared)
        self.assertEqual(
            payload["current_work"]["request"], "Keep the public API stable"
        )
        self.assertEqual(
            payload["current_moment"]["reported_progress"],
            "Checking parser compatibility",
        )
        self.assertNotIn("earlier_context", payload)
        self.assertEqual(self.snapshot, before)

    def test_available_budget_keeps_requests_in_chronological_order(self) -> None:
        payload = self.payload(self.prepare())
        self.assertEqual(
            payload["current_work"]["request"], "Keep the public API stable"
        )
        self.assertEqual(
            payload["earlier_context"], [{"request": "Explore parsing options"}]
        )

    def test_manifest_ignores_an_omitted_session_changing(self) -> None:
        self.config.prompt_target_bytes = 1
        self.snapshot = self.snapshot.model_copy(
            update={
                "sessions": (
                    *self.snapshot.sessions,
                    session_snapshot(
                        id=SessionIdentity(
                            source=SessionSource.COMMAND, native_id="renderer"
                        ),
                        turn_active=True,
                        progress="Checking layout",
                    ),
                )
            }
        )
        prepared = self.prepare()
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

    def test_historical_work_projection_keeps_its_request_without_a_live_session(
        self,
    ) -> None:
        self.record_hook(
            HookEvent(
                source=SessionSource.COMMAND,
                session="parser",
                event=HookEventKind.INTENT,
                goal="Preserve parser compatibility",
            )
        )
        self.snapshot = self.snapshot.model_copy(update={"sessions": ()})
        self.work = self.memory.connected_work(self.snapshot)
        snapshot = self.current_snapshot()
        selection = AttentionSelection(
            subject=self.work[0],
            focus_session=None,
            context_sessions=(),
            reason=SpeechReason.CLICK,
            situation=SpeechSituation.CLICK_WORKING,
        )
        prepared = self.cognition.prompts.prepare(
            snapshot, selection, self.cognition.request
        )
        payload = self.payload(prepared)
        self.assertEqual(
            payload["current_work"]["request"], "Preserve parser compatibility"
        )

    def test_manifest_rejects_a_superseded_runtime_transition(self) -> None:
        prepared = self.prepare()
        self.update_session(turn_active=False, outcome=ExecutionOutcome.SUCCESS)
        self.assertFalse(prepared.support.valid(self.current_snapshot()))

    def test_quiet_greeting_remains_valid_without_session_context(self) -> None:
        self.snapshot = self.snapshot.model_copy(update={"sessions": ()})
        prepared = self.prepare(SpeechReason.CLICK)
        self.assertTrue(prepared.support.valid(self.current_snapshot()))

    def test_explicit_context_stays_in_data_and_keeps_work_out_of_the_prompt(
        self,
    ) -> None:
        prepared = self.prepare(context="Inspect the parser result")
        payload = self.payload(prepared)
        self.assertEqual(
            payload,
            {
                "current_moment": {
                    "kind": "explicit_context",
                    "text": "Inspect the parser result",
                }
            },
        )
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
        if not self.memory.recent(
            session=SessionIdentity(source=SessionSource.COMMAND, native_id="parser"),
            event_types=("intent",),
        ):
            self.record_hook(
                HookEvent(
                    source=SessionSource.COMMAND,
                    session="parser",
                    event=HookEventKind.INTENT,
                    goal="Make session tracking survive restarts",
                )
            )
        self.memory.work.observe(
            CheckoutObservation(
                worktree="/parser",
                local=GitObservation(
                    worktree="/parser",
                    branch="persistence",
                    head=head,
                    staged=0,
                    unstaged=7,
                    untracked=0,
                    conflicts=0,
                    upstream="",
                    ahead=0,
                    behind=0,
                    repository=RemoteRepository(value="example.test/team/parser"),
                ),
                checked_at=1000,
            )
        )
        self.record_hook(
            HookEvent(
                source=SessionSource.COMMAND,
                session="parser",
                event=HookEventKind.SESSION_START,
                repo="/parser",
            )
        )
        self.memory.work.observe(
            BranchObservation(
                repository=RemoteRepository(value="example.test/team/parser"),
                branch="persistence",
                pr=PullRequest(
                    number=42,
                    url="https://example.test/team/parser/pull/42",
                    state=state,
                    merged_at=None,
                    is_draft=is_draft,
                    title="Persist coding session context",
                    head=head,
                    head_repository=RemoteRepository(value="example.test/team/parser"),
                    head_branch="persistence",
                    repository=RemoteRepository(value="example.test/team/parser"),
                ),
                checked_at=1000,
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
        self.assertNotIn("native_id", encoded)
        self.assertNotIn("https://example.test/team/parser/pull/42", encoded)

    def test_head_metadata_does_not_invalidate_the_agent_story(self) -> None:
        self.output()
        prepared = self.prepare()
        self.output(head="two")
        self.assertTrue(prepared.support.valid(self.current_snapshot()))

    def test_omitted_pull_request_state_does_not_invalidate_commentary(self) -> None:
        self.output()
        prepared = self.prepare()
        self.output(state=PullRequestState.MERGED)
        self.assertTrue(prepared.support.valid(self.current_snapshot()))

    def test_pull_request_refresh_keeps_artifact_metadata_out_of_commentary(
        self,
    ) -> None:
        self.output()
        pr = PullRequest(
            number=42,
            url="https://example.test/team/parser/pull/42",
            state=PullRequestState.MERGED,
            merged_at="2026-10-21T00:00:00Z",
            title="Persist coding session context",
            repository=RemoteRepository(value="example.test/team/parser"),
        )
        self.memory.work.observe(
            PullRequestObservation(ref=pr.url, pr=pr, checked_at=1001)
        )
        self.work = self.memory.connected_work(self.snapshot)
        encoded = json.dumps(self.payload(self.prepare()))
        self.assertNotIn(PullRequestState.MERGED, encoded)
        self.assertNotIn("OPEN", encoded)

    def test_new_artifact_context_does_not_invalidate_captured_story(self) -> None:
        self.output()
        prepared = self.prepare()
        self.work = tuple(
            subject.model_copy(
                update={
                    "story": subject.story.model_copy(
                        update={
                            "artifacts": (
                                *subject.story.artifacts,
                                ArtifactContext(
                                    kind=ArtifactKind.TICKET,
                                    ref="https://example.test/tickets/123",
                                ),
                            )
                        }
                    )
                }
            )
            for subject in self.work
        )
        self.assertTrue(prepared.support.valid(self.current_snapshot()))

    def test_disconnected_focus_invalidates_the_captured_subject(self) -> None:
        self.output()
        prepared = self.prepare()
        self.work = tuple(
            subject.model_copy(update={"sessions": ()}) for subject in self.work
        )
        self.assertFalse(prepared.support.valid(self.current_snapshot()))

    def test_selected_work_context_is_supplied_once(self) -> None:
        self.output()
        self.update_session(connected_work=self.work[0].story)
        payload = self.payload(self.prepare())
        self.assertEqual(
            payload["current_work"]["request"], "Make session tracking survive restarts"
        )
        self.assertNotIn("focus_work", payload)
        self.assertNotIn("focus_session", payload)

    def test_resumed_work_invalidates_a_quiet_care_remark(self) -> None:
        self.snapshot = sessions_snapshot()
        snapshot = self.current_snapshot().model_copy(
            update={"trigger": CareTrigger(care=CareKind.EXERCISE)}
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

    def test_commentary_omits_cached_summaries_empty_fields_and_artifacts(self) -> None:
        self.output()
        self.update_session(
            summary="No implementation results have been reported yet.",
            progress="",
            recent_requests=(),
        )
        payload = self.payload(self.prepare())
        self.assertEqual(set(payload), {"current_work", "current_moment"})
        encoded = json.dumps(payload)
        for excluded in (
            "summary",
            "observed_at",
            "artifacts",
            "branch",
            "other_sessions",
            "No implementation results",
        ):
            self.assertNotIn(excluded, encoded)
        self.assertNotIn(': ""', encoded)
        self.assertNotIn("null", encoded)
        self.assertNotIn("[]", encoded)

    def test_earlier_reports_do_not_describe_the_current_request(self) -> None:
        self.output()
        self.work = tuple(
            subject.model_copy(
                update={
                    "story": subject.story.model_copy(
                        update={
                            "requests": (
                                EvidenceView(
                                    source=EvidenceSource.USER,
                                    kind=EvidenceKind.INTENT,
                                    details=IntentView(text="Build a log explorer"),
                                    observed_at=900,
                                ),
                            ),
                            "activity": (
                                EvidenceView(
                                    source=EvidenceSource.SOURCE,
                                    kind=EvidenceKind.PROGRESS,
                                    details=ProgressView(
                                        reported_progress="Finished simplifying log labels"
                                    ),
                                    observed_at=800,
                                ),
                            ),
                            "sessions": (
                                SessionContribution(
                                    scope="Audit tests",
                                    session=SessionIdentity(
                                        source=SessionSource.COMMAND, native_id="audit"
                                    ),
                                    requests=(
                                        EvidenceView(
                                            source=EvidenceSource.SOURCE,
                                            kind=EvidenceKind.INTENT,
                                            details=IntentView(text="Audit tests"),
                                            observed_at=750,
                                        ),
                                    ),
                                    activity=(
                                        EvidenceView(
                                            source=EvidenceSource.SOURCE,
                                            kind=EvidenceKind.PROGRESS,
                                            details=ProgressView(
                                                reported_progress="Found two redundant assertions"
                                            ),
                                            observed_at=850,
                                        ),
                                    ),
                                ),
                            ),
                        }
                    )
                }
            )
            for subject in self.work
        )
        payload = self.payload(self.prepare())
        self.assertEqual(payload["current_work"]["request"], "Build a log explorer")
        self.assertNotIn("reported_progress", payload["current_moment"])
        self.assertEqual(
            payload["earlier_context"],
            [
                {
                    "reported_progress": "Finished simplifying log labels",
                    "age": "3 minutes ago",
                },
                {
                    "reported_progress": "Found two redundant assertions",
                    "age": "2 minutes ago",
                    "scope": "Audit tests",
                },
            ],
        )
        self.assertNotIn("observed_at", json.dumps(payload))

    def test_waiting_reports_only_the_current_input_request(self) -> None:
        self.update_session(
            waiting_for_input=True,
            waiting_seconds=5,
            waiting_kind=WaitingKind.QUESTION,
            waiting_prompt="Which examples should I use?",
            progress="Finished an older task",
        )
        moment = self.payload(self.prepare())["current_moment"]
        self.assertEqual(moment["kind"], "input_requested")
        self.assertEqual(moment["reported_progress"], "Which examples should I use?")
        self.update_session(
            waiting_kind=WaitingKind.PERMISSION,
            waiting_prompt=None,
            waiting_tool="Run the local browser check",
        )
        moment = self.payload(self.prepare())["current_moment"]
        self.assertEqual(moment["kind"], "permission_request")
        self.assertEqual(moment["tool"], "Run the local browser check")
        self.assertNotIn("reported_progress", moment)

    def test_running_check_in_rejects_a_replacement_call_with_the_same_command(
        self,
    ) -> None:
        self.update_session(
            active_tools=(
                ActiveTool(
                    call="first",
                    tool="Run tests",
                    activity=Activity.TESTING,
                    duration_seconds=180,
                ),
            )
        )
        snapshot = self.current_snapshot()
        selection = self.attention.select(snapshot, SpeechReason.ROUTINE).model_copy(
            update={"situation": SpeechSituation.CHECK_IN_RUNNING}
        )
        prepared = self.cognition.prompts.prepare(
            snapshot, selection, self.cognition.request
        )
        self.update_session(
            active_tools=(
                ActiveTool(
                    call="second",
                    tool="Run tests",
                    activity=Activity.TESTING,
                    duration_seconds=10,
                ),
            )
        )
        self.assertFalse(prepared.support.valid(self.current_snapshot()))

    def test_selected_terminal_failure_preserves_the_failed_call(self) -> None:
        self.output()
        subject = self.work[0]
        story = subject.story.model_copy(
            update={
                "activity": (
                    EvidenceView(
                        source=EvidenceSource.SOURCE,
                        kind=EvidenceKind.EXECUTION,
                        details=ExecutionView(
                            execution="turn ended",
                            reported_progress=None,
                            reported_execution_outcome=ExecutionOutcome.ERROR,
                        ),
                        observed_at=1000,
                    ),
                )
            }
        )
        self.work = (subject.model_copy(update={"story": story}),)
        snapshot = self.current_snapshot()
        selection = AttentionSelection(
            subject=self.work[0],
            focus_session=snapshot.sessions[0],
            context_sessions=(),
            reason=SpeechReason.FAILURE,
            situation=SpeechSituation.FAILURE,
            update=WorkUpdate(
                session=subject.id,
                revision="terminal-revision",
                observed_at=1000,
                ended_at=1000,
            ),
        )
        moment = self.cognition.prompts.prompt(snapshot, selection)["current_moment"]
        self.assertEqual(moment, {"kind": "failed_call"})

    def test_selected_completion_preserves_outcome_without_runtime_after_restart(
        self,
    ) -> None:
        self.output()
        subject = self.work[0]
        story = subject.story.model_copy(
            update={
                "activity": (
                    EvidenceView(
                        source=EvidenceSource.SOURCE,
                        kind=EvidenceKind.EXECUTION,
                        details=ExecutionView(
                            execution="turn ended",
                            reported_progress="Parser checks passed",
                            reported_execution_outcome=ExecutionOutcome.SUCCESS,
                        ),
                        observed_at=1000,
                    ),
                )
            }
        )
        self.work = (subject.model_copy(update={"story": story}),)
        self.snapshot = self.snapshot.model_copy(update={"sessions": ()})
        snapshot = self.current_snapshot()
        selection = AttentionSelection(
            subject=self.work[0],
            focus_session=None,
            context_sessions=(),
            reason=SpeechReason.ROUTINE,
            situation=SpeechSituation.COMPLETION,
            update=WorkUpdate(
                session=subject.id,
                revision="terminal-revision",
                observed_at=1000,
                ended_at=1000,
            ),
        )
        moment = self.cognition.prompts.prompt(snapshot, selection)["current_moment"]
        self.assertEqual(moment["kind"], "turn_ended")
        self.assertEqual(moment["outcome"], "success")
        self.assertEqual(moment["reported_progress"], "Parser checks passed")

    def test_selected_child_update_keeps_parent_result_in_historical_context(
        self,
    ) -> None:
        self.output()
        subject = self.work[0]
        story = subject.story.model_copy(
            update={
                "activity": (
                    EvidenceView(
                        source=EvidenceSource.SOURCE,
                        kind=EvidenceKind.EXECUTION,
                        details=ExecutionView(
                            execution="turn ended",
                            reported_progress="Parent checks passed",
                            reported_execution_outcome=ExecutionOutcome.SUCCESS,
                        ),
                        observed_at=700,
                    ),
                ),
                "sessions": (
                    SessionContribution(
                        scope="Inspect the parser",
                        session=SessionIdentity(
                            source=SessionSource.COMMAND, native_id="child"
                        ),
                        activity=(
                            EvidenceView(
                                source=EvidenceSource.SOURCE,
                                kind=EvidenceKind.PROGRESS,
                                details=ProgressView(
                                    reported_progress="Child is inspecting escape handling"
                                ),
                                observed_at=940,
                            ),
                        ),
                    ),
                ),
            }
        )
        self.work = (subject.model_copy(update={"story": story}),)
        snapshot = self.current_snapshot()
        selection = self.attention.select(snapshot, SpeechReason.ROUTINE).model_copy(
            update={
                "subject": self.work[0],
                "update": WorkUpdate(
                    session=SessionIdentity(
                        source=SessionSource.COMMAND, native_id="child"
                    ),
                    revision="child-revision",
                    observed_at=940,
                    ended_at=None,
                ),
            }
        )
        payload = self.cognition.prompts.prompt(snapshot, selection)
        self.assertEqual(payload["current_work"]["request"], "Inspect the parser")
        self.assertEqual(
            payload["current_moment"]["reported_progress"],
            "Child is inspecting escape handling",
        )
        self.assertEqual(payload["current_moment"]["age"], "1 minute ago")
        self.assertTrue(
            any(
                item.get("reported_progress") == "Parent checks passed"
                and item["age"] == "5 minutes ago"
                for item in payload["earlier_context"]
            )
        )

    def test_selected_related_task_preserves_the_scope_of_broader_results(self) -> None:
        self.output()
        subject = self.work[0]
        related = SessionIdentity(source=SessionSource.COMMAND, native_id="coverage")
        story = subject.story.model_copy(
            update={
                "intent": "Build parser",
                "activity": (
                    EvidenceView(
                        source=EvidenceSource.SOURCE,
                        kind=EvidenceKind.PROGRESS,
                        details=ProgressView(
                            reported_progress="Parser compatibility checks passed"
                        ),
                        observed_at=700,
                    ),
                ),
                "sessions": (
                    SessionContribution(
                        scope="current task",
                        session=related,
                        activity=(
                            EvidenceView(
                                source=EvidenceSource.SOURCE,
                                kind=EvidenceKind.PROGRESS,
                                details=ProgressView(
                                    reported_progress="Escape checks passed"
                                ),
                                observed_at=940,
                            ),
                        ),
                    ),
                ),
            }
        )
        self.work = (subject.model_copy(update={"story": story}),)
        snapshot = self.current_snapshot()
        selection = self.attention.select(snapshot, SpeechReason.ROUTINE).model_copy(
            update={
                "update": WorkUpdate(
                    session=related, revision="coverage-result", observed_at=940
                )
            }
        )
        narrative = WorkNarrative(
            subject_id=subject.id,
            revision=semantic_revision(story),
            intent=NarrativeStatement(
                text="I am building the parser", sources=("request",)
            ),
            progress=(
                NarrativeStatement(
                    text="I checked parser compatibility", sources=("parser",)
                ),
                NarrativeStatement(text="I checked escapes", sources=("coverage",)),
            ),
            sources={
                "request": {
                    "scope": "current task",
                    "origin": {"source": "cli", "native_id": "parser"},
                },
                "parser": {
                    "scope": "current task",
                    "origin": {"source": "cli", "native_id": "parser"},
                },
                "coverage": {
                    "scope": "current task",
                    "origin": {"source": "cli", "native_id": "coverage"},
                },
            },
        )
        payload = self.cognition.prompts.prompt(snapshot, selection, narrative)
        self.assertEqual(payload["current_work"]["request"], "current task")
        self.assertEqual(
            payload["current_moment"]["reported_progress"], "Escape checks passed"
        )
        history = {
            item.get("request", item.get("reported_progress")): item.get("scope")
            for item in payload["earlier_context"]
        }
        self.assertEqual(history["Parser compatibility checks passed"], "Build parser")
        self.assertEqual(history["I am building the parser"], "Build parser")
        self.assertEqual(history["I checked parser compatibility"], "Build parser")
        self.assertEqual(history["I checked escapes"], "current task")

    def test_child_report_keeps_the_assignment_observed_before_its_result(self) -> None:
        self.output()
        report = EvidenceView(
            source=EvidenceSource.SOURCE,
            kind=EvidenceKind.PROGRESS,
            details=ProgressView(reported_progress="Found two redundant assertions"),
            observed_at=850,
        )
        earlier_assignment = EvidenceView(
            source=EvidenceSource.SOURCE,
            kind=EvidenceKind.INTENT,
            details=IntentView(text="Audit tests"),
            observed_at=800,
        )
        current_assignment = EvidenceView(
            source=EvidenceSource.SOURCE,
            kind=EvidenceKind.INTENT,
            details=IntentView(text="Check the explorer website"),
            observed_at=900,
        )
        child = SessionContribution(
            scope="Check the explorer website",
            session=SessionIdentity(source=SessionSource.COMMAND, native_id="explorer"),
            requests=(earlier_assignment, current_assignment),
            activity=(report,),
        )
        self.work = tuple(
            subject.model_copy(
                update={
                    "story": subject.story.model_copy(update={"sessions": (child,)})
                }
            )
            for subject in self.work
        )
        payload = self.payload(self.prepare())
        self.assertEqual(payload["earlier_context"][-1]["scope"], "Audit tests")
        child_without_earlier_assignment = child.model_copy(
            update={"requests": (current_assignment,)}
        )
        self.work = tuple(
            subject.model_copy(
                update={
                    "story": subject.story.model_copy(
                        update={"sessions": (child_without_earlier_assignment,)}
                    )
                }
            )
            for subject in self.work
        )
        payload = self.payload(self.prepare())
        self.assertEqual(payload["earlier_context"][-1]["scope"], "related task")

    def test_cancelled_routine_work_preserves_the_terminal_outcome(self) -> None:
        self.update_session(
            turn_active=False,
            outcome=ExecutionOutcome.CANCELLED,
            progress="Started checking parser behavior",
        )
        snapshot = self.current_snapshot()
        selection = AttentionSelection(
            subject=None,
            focus_session=snapshot.sessions[0],
            context_sessions=(),
            reason=SpeechReason.CLICK,
            situation=SpeechSituation.ROUTINE,
        )
        prepared = self.cognition.prompts.prepare(
            snapshot, selection, self.cognition.request
        )
        moment = self.payload(prepared)["current_moment"]
        self.assertEqual(moment["kind"], "work_update")
        self.assertEqual(moment["state"], "quiet")
        self.assertEqual(moment["outcome"], "cancelled")

    def test_running_check_in_omits_an_unknown_tool_name(self) -> None:
        self.update_session(
            active_tools=(
                ActiveTool(
                    call="first",
                    tool="",
                    activity=Activity.COMMAND,
                    duration_seconds=180,
                ),
            )
        )
        snapshot = self.current_snapshot()
        selection = self.attention.select(snapshot, SpeechReason.ROUTINE).model_copy(
            update={"situation": SpeechSituation.CHECK_IN_RUNNING}
        )
        prepared = self.cognition.prompts.prepare(
            snapshot, selection, self.cognition.request
        )
        moment = self.payload(prepared)["current_moment"]
        self.assertNotIn("tool", moment)
        self.assertEqual(moment["elapsed"], "3 minutes")

    def test_historical_reports_preserve_terminal_outcomes_without_duplicates(
        self,
    ) -> None:
        self.output()
        progress = EvidenceView(
            source=EvidenceSource.SOURCE,
            kind=EvidenceKind.PROGRESS,
            details=ProgressView(reported_progress="Checked parser compatibility"),
            observed_at=800,
        )
        ended = EvidenceView(
            source=EvidenceSource.SOURCE,
            kind=EvidenceKind.EXECUTION,
            details=ExecutionView(
                execution="turn ended",
                reported_progress="Checked parser compatibility",
                reported_execution_outcome=ExecutionOutcome.CANCELLED,
            ),
            observed_at=800,
        )
        request = EvidenceView(
            source=EvidenceSource.USER,
            kind=EvidenceKind.INTENT,
            details=IntentView(text="Build a log explorer"),
            observed_at=900,
        )
        self.work = tuple(
            subject.model_copy(
                update={
                    "story": subject.story.model_copy(
                        update={
                            "requests": (request,),
                            "activity": (progress, ended),
                            "sessions": (
                                SessionContribution(
                                    scope="Audit tests",
                                    session=SessionIdentity(
                                        source=SessionSource.COMMAND, native_id="audit"
                                    ),
                                    activity=(ended, progress),
                                ),
                            ),
                        }
                    )
                }
            )
            for subject in self.work
        )
        earlier = self.payload(self.prepare())["earlier_context"]
        self.assertEqual(len(earlier), 2)
        self.assertEqual(earlier[0]["outcome"], "cancelled")
        self.assertEqual(earlier[1]["outcome"], "cancelled")
        self.assertEqual(earlier[1]["scope"], "related task")

    def test_followup_keeps_its_prior_request_ahead_of_optional_child_reports(
        self,
    ) -> None:
        self.output()
        original_request = EvidenceView(
            source=EvidenceSource.USER,
            kind=EvidenceKind.INTENT,
            details=IntentView(text="Build a log explorer"),
            observed_at=700,
        )
        followup = EvidenceView(
            source=EvidenceSource.USER,
            kind=EvidenceKind.INTENT,
            details=IntentView(text="proceed"),
            observed_at=900,
        )
        children = tuple(
            SessionContribution(
                scope=f"Assignment {index}",
                session=SessionIdentity(
                    source=SessionSource.COMMAND, native_id=f"child-{index}"
                ),
                activity=(
                    EvidenceView(
                        source=EvidenceSource.SOURCE,
                        kind=EvidenceKind.PROGRESS,
                        details=ProgressView(
                            reported_progress=f"Report {index}: "
                            + "Checked parser behavior. " * 8
                        ),
                        observed_at=950,
                    ),
                ),
            )
            for index in range(6)
        )
        self.work = tuple(
            subject.model_copy(
                update={
                    "story": subject.story.model_copy(
                        update={
                            "requests": (original_request, followup),
                            "sessions": children,
                        }
                    )
                }
            )
            for subject in self.work
        )
        prepared = self.prepare()
        payload = self.payload(prepared)
        self.assertEqual(payload["current_work"]["request"], "proceed")
        self.assertEqual(len(payload["earlier_context"]), 6)
        self.assertEqual(
            payload["earlier_context"][0]["request"], "Build a log explorer"
        )
        self.assertEqual(
            payload["earlier_context"][1]["reported_progress"],
            "Report 1: " + "Checked parser behavior. " * 7 + "Checked parser behavior.",
        )
        self.assertEqual(
            payload["earlier_context"][-1]["reported_progress"],
            "Report 5: " + "Checked parser behavior. " * 7 + "Checked parser behavior.",
        )
        one_request_payload = deepcopy(payload)
        one_request_payload["earlier_context"] = payload["earlier_context"][:1]
        one_request = self.cognition.request(
            one_request_payload, prepared.request["system"]
        )
        self.config.prompt_target_bytes = (
            len(json.dumps(one_request, ensure_ascii=False).encode("utf-8")) + 1
        )
        limited = self.payload(self.prepare())
        self.assertEqual(limited["earlier_context"], payload["earlier_context"][:1])
