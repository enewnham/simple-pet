import json
import random
import tempfile
import unittest
from copy import deepcopy
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Unpack
from zoneinfo import ZoneInfo

from pydantic import BaseModel

from pet_daemon.attention import Attention, AttentionSelection, SelectedWorkState
from pet_daemon.commentary import Cognition
from pet_daemon.commentary.context import (
    ContextFact,
    Event,
    FactKind,
    Moment,
    SupportedText,
    WorkState,
)
from pet_daemon.commentary.contracts import ModelRequest
from pet_daemon.commentary.request import (
    CommentaryRequest,
    _record_source_ids,
    encoded_size,
    evidence_age,
    select_prompt,
)
from pet_daemon.commentary.sources import NarrativeSource, SourceKind
from pet_daemon.commentary.story import prepare as prepare_story
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
from pet_daemon.identity import SessionIdentity, session_data
from pet_daemon.repository_identity import RemoteRepository
from pet_daemon.runtime import (
    ActiveTool,
    CareTrigger,
    FailureSnapshot,
    HookEvent,
    HookEventKind,
    PetContext,
    WaitingKind,
)
from pet_daemon.sessions.tracking import Sessions
from pet_daemon.storage import Memory
from pet_daemon.work import WorkSubject
from pet_daemon.work.evidence import (
    CommentaryEvidence,
    ExecutionStatus,
    ReportedProgress,
    UserIntent,
)
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
from pet_daemon.work.shared_work import WorkUpdate
from pet_daemon.work.stories import ArtifactContext, SessionContribution
from Tests.fixtures import (
    SessionOptions,
    capture_snapshot,
    session_snapshot,
    sessions_snapshot,
)


class NoRequestsModel:
    async def complete(
        self,
        request: ModelRequest,
        reason: SpeechReason | ModelPurpose,
        *,
        response_type: type[BaseModel],
    ) -> dict[str, object]:
        raise AssertionError("PromptBuilder preparation must not request model output")


class EvidenceAgeTests(unittest.TestCase):
    def test_ages_use_local_calendar_days_and_omit_recent_times(self) -> None:
        timezone = ZoneInfo("America/New_York")
        timestamp = datetime(2026, 11, 4, 12, tzinfo=timezone).timestamp()
        for observed, expected in (
            (datetime(2026, 11, 4, 11, 58, tzinfo=timezone), None),
            (datetime(2026, 11, 4, 11, 0, 1, tzinfo=timezone), None),
            (datetime(2026, 11, 4, 11, tzinfo=timezone), "today"),
            (datetime(2026, 11, 4, 8, tzinfo=timezone), "today"),
            (datetime(2026, 11, 3, 23, tzinfo=timezone), "yesterday"),
            (datetime(2026, 11, 2, 23, tzinfo=timezone), "earlier"),
        ):
            with self.subTest(observed=observed):
                self.assertEqual(
                    evidence_age(timestamp, observed.timestamp(), timezone), expected
                )
        self.assertIsNone(evidence_age(timestamp, None, timezone))

    def test_midnight_and_daylight_saving_use_calendar_dates(self) -> None:
        timezone = ZoneInfo("America/New_York")
        for current, observed in (
            (
                datetime(2026, 11, 4, 0, 1, tzinfo=timezone),
                datetime(2026, 11, 3, 23, 59, tzinfo=timezone),
            ),
            (
                datetime(2026, 11, 1, 23, 30, tzinfo=timezone),
                datetime(2026, 10, 31, 23, 45, tzinfo=timezone),
            ),
            (
                datetime(2026, 3, 9, 0, 15, tzinfo=timezone),
                datetime(2026, 3, 8, 0, 0, tzinfo=timezone),
            ),
        ):
            with self.subTest(current=current):
                self.assertEqual(
                    evidence_age(current.timestamp(), observed.timestamp(), timezone),
                    "yesterday",
                )


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
        self.sessions = Sessions(lambda: 1000, wall_clock=lambda: 1000)

    def record_hook(self, hook: HookEvent) -> None:
        hook = hook.model_copy(update={"capture_at": self.memory.clock()})
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

    def finished_work(
        self, outcome: ExecutionOutcome = ExecutionOutcome.SUCCESS
    ) -> None:
        self.attention.wall_clock = lambda: 1000
        self.record_hook(
            HookEvent(
                source=SessionSource.COMMAND,
                session="parser",
                event=HookEventKind.INTENT,
                goal="Check parser compatibility",
            )
        )
        self.record_hook(
            HookEvent(
                source=SessionSource.COMMAND,
                session="parser",
                event=HookEventKind.TURN_DONE,
                progress="Parser checks passed; desktop behavior remains unverified.",
                outcome=outcome,
            )
        )
        self.snapshot = self.sessions.activity()
        self.work = self.memory.connected_work(self.snapshot)

    def test_execution_descriptions_preserve_results_and_validity_facts(self) -> None:
        cases = (
            (ExecutionOutcome.SUCCESS, "without a reported execution error"),
            (ExecutionOutcome.ERROR, "with a reported execution error"),
            (ExecutionOutcome.CANCELLED, "was cancelled"),
            (ExecutionOutcome.STALE, "its result is unknown"),
        )
        for outcome, meaning in cases:
            with self.subTest(outcome=outcome):
                self.finished_work(outcome)
                prepared = self.prepare(SpeechReason.COMPLETION)
                moment = prepared.context.moment
                assert moment is not None
                self.assertEqual(
                    moment.event,
                    Event.TURN_ENDED,
                )
                self.assertEqual(moment.outcome, outcome)
                self.assertIn(meaning, prepared.request["messages"][0]["content"])
                if outcome == ExecutionOutcome.SUCCESS:
                    self.assertIn(
                        "does not establish task completion",
                        prepared.request["messages"][0]["content"],
                    )
                self.assertNotIn(
                    '"outcome"', prepared.request["messages"][0]["content"]
                )
                conditions_moment = prepared.conditions.context.moment
                assert conditions_moment is not None
                self.assertEqual(conditions_moment.outcome, outcome)
                self.assertTrue(prepared.conditions.is_valid(self.current_snapshot()))
                self.update_session(outcome=None, turn_active=True)
                self.assertFalse(prepared.conditions.is_valid(self.current_snapshot()))

    def test_failed_operations_describe_the_activity_without_diagnosing_a_cause(
        self,
    ) -> None:
        for activity, meaning in (
            (Activity.TESTING, "test execution"),
            (Activity.SOURCE, "coding tool execution"),
        ):
            with self.subTest(activity=activity):
                self.update_session(
                    failure=FailureSnapshot(
                        activity=activity,
                        count=1,
                        goal="Check compatibility",
                        resolved=False,
                        current_turn=True,
                        seconds_ago=0,
                    )
                )
                prepared = self.prepare(SpeechReason.FAILURE)
                moment = prepared.context.moment
                assert moment is not None
                self.assertEqual(moment.event, Event.FAILED_CALL)
                self.assertEqual(moment.operation, activity)
                self.assertIn(meaning, prepared.request["messages"][0]["content"])
                self.assertNotIn(
                    "cause",
                    prepared.request["messages"][0]["content"].split("</observations>")[
                        0
                    ],
                )

    def test_current_report_and_its_cached_paraphrase_are_sent_once(self) -> None:
        self.finished_work()
        snapshot = self.current_snapshot()
        selection = self.attention.select(snapshot, SpeechReason.COMPLETION)
        story_request = prepare_story(
            snapshot, selection, self.memory, self.cognition.request
        )
        assert story_request is not None
        report_ids = tuple(
            identifier
            for identifier, source in story_request.sources.items()
            if source["kind"] in ("reported_progress", "turn_ended")
        )
        self.assertEqual(len(report_ids), 2)
        narrative = WorkNarrative(
            subject_id=story_request.subject_id,
            revision=story_request.revision,
            intent=NarrativeStatement(
                text="Check parser compatibility.",
                sources=(story_request.latest_request_source,),
            ),
            progress=(
                NarrativeStatement(
                    text="I verified compatibility; the desktop remains unverified.",
                    sources=report_ids,
                ),
            ),
            sources=story_request.sources,
        )
        prepared = self.cognition.prompts.prepare(
            snapshot, selection, self.cognition.request, narrative=narrative
        )
        context = prepared.context
        assert context.moment is not None
        assert context.moment.progress is not None
        self.assertEqual(
            context.moment.progress.text,
            "Parser checks passed; desktop behavior remains unverified.",
        )
        self.assertFalse(
            any(
                isinstance(item, ContextFact) and item.kind == FactKind.PROGRESS
                for item in context.earlier
            )
        )
        self.assertTrue(prepared.conditions.is_valid(snapshot))

    def test_waiting_question_does_not_cite_an_earlier_result(self) -> None:
        self.finished_work()
        question = "Should we keep the legacy endpoint?"
        self.update_session(
            waiting_for_input=True,
            waiting_kind=WaitingKind.QUESTION,
            waiting_prompt=question,
            waiting_seconds=10,
        )
        snapshot = self.current_snapshot()
        selection = self.attention.select(snapshot, SpeechReason.CLICK)
        story_request = prepare_story(
            snapshot, selection, self.memory, self.cognition.request
        )
        assert story_request is not None
        narrative = WorkNarrative(
            subject_id=story_request.subject_id,
            revision=story_request.revision,
            intent=NarrativeStatement(
                text="Check parser compatibility.",
                sources=(story_request.latest_request_source,),
            ),
            sources=story_request.sources,
        )
        prepared = self.cognition.prompts.prepare(
            snapshot, selection, self.cognition.request, narrative=narrative
        )
        moment = prepared.context.moment
        assert moment is not None
        assert moment.progress is not None
        self.assertEqual(moment.progress.text, question)
        self.assertEqual(moment.sources, ())

    def test_truncated_report_keeps_a_summary_of_the_omitted_result(self) -> None:
        self.finished_work()
        self.record_hook(
            HookEvent(
                source=SessionSource.COMMAND,
                session="parser",
                event=HookEventKind.INTENT,
                goal="Check desktop compatibility",
            )
        )
        report = "Checked parser compatibility. " * 30 + "Desktop check failed."
        self.record_hook(
            HookEvent(
                source=SessionSource.COMMAND,
                session="parser",
                event=HookEventKind.TURN_DONE,
                progress=report,
            )
        )
        self.snapshot = self.sessions.activity()
        self.work = self.memory.connected_work(self.snapshot)
        snapshot = self.current_snapshot()
        selection = self.attention.select(snapshot, SpeechReason.COMPLETION)
        story_request = prepare_story(
            snapshot, selection, self.memory, self.cognition.request
        )
        assert story_request is not None
        report_ids = tuple(
            identifier
            for identifier, source in story_request.sources.items()
            if source.get("text") == report
        )
        self.assertEqual(len(report_ids), 2)
        narrative = WorkNarrative(
            subject_id=story_request.subject_id,
            revision=story_request.revision,
            intent=NarrativeStatement(
                text="Check desktop compatibility.",
                sources=(story_request.latest_request_source,),
            ),
            progress=(
                NarrativeStatement(text="Desktop check failed.", sources=report_ids),
            ),
            sources=story_request.sources,
        )
        prepared = self.cognition.prompts.prepare(
            snapshot, selection, self.cognition.request, narrative=narrative
        )
        context = prepared.context
        assert context.moment is not None
        assert context.moment.progress is not None
        excerpt = context.moment.progress.text
        assert isinstance(excerpt, str)
        self.assertNotIn("Desktop check failed.", excerpt)
        self.assertEqual(set(context.moment.sources), set(report_ids))
        self.assertNotIn("native_id", prepared.request["messages"][0]["content"])
        for identifier in prepared.source_ids:
            self.assertNotIn(identifier, prepared.request["messages"][0]["content"])
        self.assertEqual(set(report_ids) & set(prepared.source_ids), set(report_ids))
        self.assertTrue(
            any(
                isinstance(item, ContextFact)
                and item.statement.text == "Desktop check failed."
                for item in context.earlier
            )
        )
        self.assertTrue(prepared.conditions.is_valid(snapshot))

    def test_raw_records_keep_citations_without_a_summary_catalog(self) -> None:
        self.finished_work()
        snapshot = self.current_snapshot()
        selection = self.attention.select(snapshot, SpeechReason.COMPLETION)
        subject = selection.subject
        assert subject is not None
        request_key = subject.story.requests[-1].source_key
        report_key = subject.story.activity[-1].source_key
        assert request_key is not None and report_key is not None
        narrative = WorkNarrative(
            subject_id=subject.id,
            revision=semantic_revision(subject.story),
            intent=NarrativeStatement(
                text="Check compatibility.", sources=(request_key,)
            ),
            sources={},
        )
        for summary in (None, narrative):
            with self.subTest(summary=summary is not None):
                prepared = self.cognition.prompts.prepare(
                    snapshot, selection, self.cognition.request, narrative=summary
                )
                context = prepared.context
                assert context.work is not None
                assert context.work.request is not None
                self.assertEqual(
                    context.work.request.sources,
                    (request_key,),
                )
                assert context.moment is not None
                self.assertEqual(
                    context.moment.sources,
                    (report_key,),
                )
                self.assertEqual(
                    prepared.citations.resolve(prepared.citations.label(report_key)),
                    report_key,
                )
                self.assertTrue(prepared.conditions.is_valid(snapshot))
                changed_story = subject.story.model_copy(
                    update={
                        "activity": tuple(
                            record.model_copy(
                                update={
                                    "details": ReportedProgress(
                                        reported_progress="Compatibility checks failed."
                                    )
                                }
                            )
                            if record.source_key == report_key
                            else record
                            for record in subject.story.activity
                        )
                    }
                )
                self.work = (subject.model_copy(update={"story": changed_story}),)
                self.assertFalse(prepared.conditions.is_valid(self.current_snapshot()))
                self.work = (subject,)

    def test_keyed_raw_report_keeps_its_identity_outside_the_summary_catalog(
        self,
    ) -> None:
        root = SessionIdentity(source=SessionSource.COMMAND, native_id="child")
        record = CommentaryEvidence(
            source=EvidenceSource.SOURCE,
            kind=EvidenceKind.PROGRESS,
            details=ReportedProgress(reported_progress="Checked parser compatibility."),
            source_key="actual-child-report",
        )
        unrelated = NarrativeSource(
            identifier="different-report",
            kind=SourceKind.PROGRESS,
            text="Checked parser compatibility.",
            full_text="Checked parser compatibility.",
            scope="related task",
            outcome=None,
            observed_at=None,
            origin=root,
        )
        for catalog in ((), (unrelated,)):
            with self.subTest(catalog=bool(catalog)):
                self.assertEqual(
                    _record_source_ids(
                        record, root, ("reported_progress",), root, catalog, ()
                    ),
                    ["actual-child-report"],
                )

    def test_source_ids_keep_equal_reports_separate_by_reporter_and_time(self) -> None:
        root = SessionIdentity(source=SessionSource.COMMAND, native_id="parser")
        peer = SessionIdentity(source=SessionSource.COMMAND, native_id="peer")
        record = CommentaryEvidence(
            source=EvidenceSource.SOURCE,
            kind=EvidenceKind.PROGRESS,
            details=ReportedProgress(reported_progress="Compatibility checks passed."),
            observed_at=1000,
        )
        report = {
            "kind": "reported_progress",
            "text": "Compatibility checks passed.",
            "scope": "current task",
            "observed_at": 1000,
            "origin": session_data(root),
        }
        narrative = WorkNarrative(
            subject_id=root,
            revision="revision",
            intent=NarrativeStatement(text="Check compatibility.", sources=("root",)),
            sources={
                "root": report,
                "peer": {**report, "origin": session_data(peer)},
                "earlier": {**report, "observed_at": 900},
            },
        )
        self.assertEqual(
            _record_source_ids(
                record,
                root,
                ("reported_progress",),
                narrative.subject_id,
                tuple(
                    NarrativeSource.from_record(identifier, value)
                    for identifier, value in narrative.sources.items()
                ),
                (),
            ),
            ["root"],
        )

    def prepare(
        self, reason: SpeechReason = SpeechReason.ROUTINE, context: str | None = None
    ) -> CommentaryRequest:
        selection = self.attention.select(self.current_snapshot(), reason, context)
        return self.cognition.prompts.prepare(
            self.current_snapshot(), selection, self.cognition.request
        )

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
        context = prepared.context
        assert context.work is not None
        assert context.work.story is not None
        self.assertEqual(context.work.story.text, "Preserve parser compatibility.")
        assert context.moment is not None
        assert context.moment.progress is not None
        self.assertEqual(
            context.moment.progress.text,
            "Checking parser compatibility",
        )
        assert isinstance(context.earlier[-1], ContextFact)
        self.assertEqual(
            context.earlier[-1].statement.text,
            "The agent reports compatibility checks are running.",
        )
        self.assertTrue(prepared.conditions.is_valid(snapshot))
        self.update_session(progress="Compatibility check failed")
        self.assertFalse(prepared.conditions.is_valid(self.current_snapshot()))

    def test_covered_quiet_check_ins_omit_findings_after_narrative_enrichment(
        self,
    ) -> None:
        self.finished_work()
        self.update_session(quiet_seconds=21600)
        snapshot = self.current_snapshot()
        selection = AttentionSelection(
            subject=self.work[0],
            focus_session=snapshot.sessions[0],
            context_sessions=(),
            reason=SpeechReason.CHECK_IN,
            situation=SpeechSituation.CHECK_IN_QUIET,
            selected_state=SelectedWorkState.capture(
                self.work[0], snapshot.sessions[0]
            ),
        )
        subject = selection.subject
        assert subject is not None
        narrative = WorkNarrative(
            subject_id=subject.id,
            revision="narrative",
            intent=NarrativeStatement(
                text="The stale overlays need a refresh.", sources=("intent",)
            ),
            decisions=(
                NarrativeStatement(
                    text="Refresh the stale overlays.", sources=("decision",)
                ),
            ),
            progress=(
                NarrativeStatement(
                    text="The overlay source file is gone.", sources=("progress",)
                ),
            ),
        )
        for revision, covered in (
            (None, False),
            (semantic_revision(subject.story), True),
            ("different-content", False),
        ):
            with self.subTest(covered=covered, revision=revision):
                prepared = self.cognition.prompts.prepare(
                    snapshot,
                    selection,
                    self.cognition.request,
                    narrative=narrative,
                    spoken_content_revision=revision,
                )
                context = prepared.context
                assert context.moment is not None
                self.assertEqual(context.moment.elapsed, "6 hours")
                self.assertIsNone(context.moment.state)
                self.assertEqual(
                    context.moment.event,
                    Event.QUIET_CHECK_IN,
                )
                self.assertTrue(prepared.conditions.is_valid(snapshot))
                if covered:
                    self.assertIsNone(context.work)
                    self.assertEqual(prepared.citations.sources, ())
                    self.assertNotIn(
                        "overlay", prepared.request["messages"][0]["content"]
                    )
                    self.assertEqual(context.earlier, ())
                    self.assertIsNone(context.moment.progress)
                    self.assertNotIn(
                        "Check parser compatibility",
                        prepared.request["messages"][0]["content"],
                    )
                else:
                    assert context.work is not None
                    assert context.work.request is not None
                    self.assertEqual(
                        context.work.request.text, "Check parser compatibility"
                    )
                    assert context.work.story is not None
                    self.assertEqual(
                        context.work.story.text,
                        "The stale overlays need a refresh.",
                    )
                    self.assertTrue(context.earlier)
                    self.assertIsNotNone(context.moment.progress)

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
                item.identifier: item.record()
                for item in (
                    NarrativeSource(
                        "request",
                        SourceKind.REQUEST,
                        "Preserve parser compatibility.",
                        "root",
                        None,
                        None,
                        700,
                        "Preserve parser compatibility.",
                    ),
                    NarrativeSource(
                        "report",
                        SourceKind.PROGRESS,
                        "Parser checks passed.",
                        "root",
                        None,
                        None,
                        880,
                        "Parser checks passed.",
                    ),
                )
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
        report = prepared.context.earlier[-1]
        self.assertIsNone(report.age)
        self.cognition.clock = lambda: 8200
        later = self.cognition.prompts.prepare(
            snapshot, selection, self.cognition.request, narrative=cached
        )
        report = later.context.earlier[-1]
        self.assertEqual(report.age, "today")
        self.assertEqual(prepared.conditions, later.conditions)
        self.assertEqual(
            cached,
            self.memory.load_narrative(
                SessionIdentity(source=SessionSource.COMMAND, native_id="parser")
            ),
        )

    def test_short_followup_keeps_the_supported_intent_in_the_rendered_request(
        self,
    ) -> None:
        self.update_session(
            recent_requests=("Keep the existing retry mechanism", "yes, do that")
        )
        snapshot = self.current_snapshot()
        selection = self.attention.select(snapshot, SpeechReason.ROUTINE)
        narrative = WorkNarrative(
            subject_id=snapshot.sessions[0].id,
            revision="retry-intent",
            intent=NarrativeStatement(
                text="Keep the existing retry mechanism and close the obsolete retry-queue PR.",
                sources=("intent-source",),
            ),
        )
        prepared = self.cognition.prompts.prepare(
            snapshot, selection, self.cognition.request, narrative=narrative
        )
        text = prepared.request["messages"][0]["content"]
        self.assertIn("yes, do that", text)
        self.assertIn("Supported work intent [s1]:", text)
        self.assertIn(
            "Keep the existing retry mechanism and close the obsolete retry-queue PR.",
            text,
        )
        self.assertEqual(prepared.citations.resolve("s1"), "intent-source")

    def test_identical_supported_intent_is_rendered_once(self) -> None:
        request = "Preserve parser compatibility."
        self.update_session(recent_requests=(request,))
        snapshot = self.current_snapshot()
        selection = self.attention.select(snapshot, SpeechReason.ROUTINE)
        narrative = WorkNarrative(
            subject_id=snapshot.sessions[0].id,
            revision="parser-intent",
            intent=NarrativeStatement(text=request, sources=("intent-source",)),
        )
        prepared = self.cognition.prompts.prepare(
            snapshot, selection, self.cognition.request, narrative=narrative
        )
        text = prepared.request["messages"][0]["content"]
        self.assertEqual(text.count(request), 1)
        self.assertNotIn("Supported work intent", text)

    def test_request_age_follows_its_observation_despite_recent_activity(self) -> None:
        self.output()
        snapshot = self.current_snapshot()
        selection = self.attention.select(snapshot, SpeechReason.ROUTINE)
        prepared = self.cognition.prompts.prepare(
            snapshot, selection, self.cognition.request
        )
        assert prepared.context.work is not None
        self.assertIsNone(prepared.context.work.request_age)
        self.cognition.clock = lambda: 15400
        update = selection.update
        assert update is not None
        later_selection = selection.model_copy(
            update={"update": update.model_copy(update={"observed_at": 15400})}
        )
        later = self.cognition.prompts.prepare(
            snapshot, later_selection, self.cognition.request
        )
        assert later.context.work is not None
        self.assertEqual(later.context.work.request_age, "today")
        self.assertEqual(
            prepared.conditions.context.stable(), later.conditions.context.stable()
        )
        self.assertTrue(later.conditions.is_valid(snapshot))

    def test_undated_request_does_not_inherit_the_update_time(self) -> None:
        self.output()
        subject = self.work[0]
        subject = subject.model_copy(
            update={"story": subject.story.model_copy(update={"requests": ()})}
        )
        self.work = (subject,)
        prepared = self.prepare()
        assert prepared.context.work is not None
        self.assertEqual(prepared.context.work.request_age, None)

    def test_changed_older_story_evidence_invalidates_speech_outside_raw_projection(
        self,
    ) -> None:
        self.output()
        subject = self.work[0]
        requests = tuple(
            CommentaryEvidence(
                source=EvidenceSource.USER,
                kind=EvidenceKind.INTENT,
                details=UserIntent(text=f"Decision {index}"),
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
        self.assertTrue(prepared.conditions.is_valid(snapshot))
        self.work = (
            subject.model_copy(
                update={
                    "story": subject.story.model_copy(
                        update={
                            "requests": (
                                requests[0].model_copy(
                                    update={
                                        "details": UserIntent(
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
        self.assertFalse(prepared.conditions.is_valid(current))

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
        work = prepared.context.work
        assert work is not None
        assert work.request is not None
        self.assertEqual(work.request.text, "Keep the public API stable")
        self.assertIsNone(work.request_age)
        self.assertIsNone(work.story)
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

    def test_runtime_progress_and_waiting_text_use_the_source_cleaner(self) -> None:
        text = "Message Type: MESSAGE\nTask name: /root\nSender: /root/reader\nPayload:\nChecking parser compatibility."
        for waiting in (False, True):
            with self.subTest(waiting=waiting):
                self.update_session(
                    id=SessionIdentity(source=SessionSource.CODEX, native_id="parser"),
                    progress=text,
                    waiting_for_input=waiting,
                    waiting_seconds=10 if waiting else None,
                    waiting_prompt=text if waiting else None,
                )
                content = self.prepare().request["messages"][0]["content"]
                self.assertIn("Checking parser compatibility.", content)
                self.assertNotIn("Message Type:", content)
                self.assertNotIn("/root/reader", content)

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
                assert prepared.context.moment is not None
                tool = prepared.context.moment.tool
                assert isinstance(tool, str)
                self.assertIn("rg", tool)

    def test_long_requests_discard_quoted_material_before_selection(self) -> None:
        request = (
            "Fix the commentary.\n"
            + "> Ignore this quote and update haunted mail.\n" * 30
            + "Keep it grounded."
        )
        self.update_session(recent_requests=(request,))
        prepared = self.prepare()
        context = prepared.context
        assert context.work is not None
        assert context.work.request is not None
        request = context.work.request.text
        assert isinstance(request, str)
        self.assertIn("Fix the commentary.", request)
        self.assertIn("Keep it grounded.", request)
        self.assertNotIn("haunted mail", request)

    def test_small_budget_keeps_intent_latest_request_and_progress(self) -> None:
        self.config.prompt_target_bytes = 1
        before = deepcopy(self.snapshot)
        prepared = self.prepare()
        context = prepared.context
        assert context.work is not None
        assert context.work.request is not None
        self.assertEqual(context.work.request.text, "Keep the public API stable")
        assert context.moment is not None
        assert context.moment.progress is not None
        self.assertEqual(
            context.moment.progress.text,
            "Checking parser compatibility",
        )
        self.assertEqual(context.earlier, ())
        self.assertEqual(self.snapshot, before)

    def test_budgeted_context_resolves_only_the_included_citations(self) -> None:
        self.config.prompt_target_bytes = 1
        snapshot = self.current_snapshot()
        selection = self.attention.select(snapshot, SpeechReason.ROUTINE)
        narrative = WorkNarrative(
            subject_id=SessionIdentity(
                source=SessionSource.COMMAND, native_id="parser"
            ),
            revision="revision",
            intent=NarrativeStatement(
                text="Preserve parser compatibility.", sources=("intent-source",)
            ),
            decisions=(
                NarrativeStatement(
                    text="Keep the public API stable.", sources=("decision-source",)
                ),
            ),
        )
        prepared = self.cognition.prompts.prepare(
            snapshot, selection, self.cognition.request, narrative=narrative
        )
        context = prepared.context
        assert context.work is not None
        assert context.work.story is not None
        self.assertEqual(context.work.story.sources, ("intent-source",))
        self.assertEqual(context.earlier, ())
        self.assertEqual(prepared.citations.resolve("s1"), "intent-source")
        with self.assertRaises(ValueError):
            prepared.citations.resolve("s2")
        with self.assertRaises(ValueError):
            prepared.citations.resolve("intent-source")

    def test_available_budget_keeps_requests_in_chronological_order(self) -> None:
        prepared = self.prepare()
        context = prepared.context
        assert context.work is not None
        assert context.work.request is not None
        self.assertEqual(context.work.request.text, "Keep the public API stable")
        self.assertEqual(
            context.earlier,
            (ContextFact(FactKind.REQUEST, SupportedText("Explore parsing options")),),
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
        self.assertTrue(prepared.conditions.is_valid(self.current_snapshot()))

    def test_manifest_ignores_a_request_omitted_by_the_budget(self) -> None:
        self.config.prompt_target_bytes = 1
        prepared = self.prepare()
        self.update_session(
            recent_requests=(
                "Discarded exploration",
                *self.snapshot.sessions[0].recent_requests[1:],
            )
        )
        self.assertTrue(prepared.conditions.is_valid(self.current_snapshot()))

    def test_manifest_rejects_changed_included_progress(self) -> None:
        prepared = self.prepare()
        self.update_session(progress="Parser verification complete")
        self.assertFalse(prepared.conditions.is_valid(self.current_snapshot()))

    def test_manifest_ignores_elapsed_time_and_metadata(self) -> None:
        prepared = self.prepare()
        self.update_session(duration_seconds=300, quiet_seconds=150, changed_at=1200)
        self.assertTrue(prepared.conditions.is_valid(self.current_snapshot()))

    def test_process_presence_metadata_does_not_supersede_ongoing_work(self) -> None:
        self.update_session(presence=ProcessStatus.UNKNOWN)
        prepared = self.prepare()
        self.update_session(presence=ProcessStatus.ALIVE)
        self.assertTrue(prepared.conditions.is_valid(self.current_snapshot()))

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
        context = prepared.context
        assert context.work is not None
        assert context.work.request is not None
        self.assertEqual(context.work.request.text, "Preserve parser compatibility")

    def test_manifest_rejects_a_superseded_runtime_transition(self) -> None:
        prepared = self.prepare()
        self.update_session(turn_active=False, outcome=ExecutionOutcome.SUCCESS)
        self.assertFalse(prepared.conditions.is_valid(self.current_snapshot()))

    def test_quiet_greeting_remains_valid_without_session_context(self) -> None:
        self.snapshot = self.snapshot.model_copy(update={"sessions": ()})
        prepared = self.prepare(SpeechReason.CLICK)
        self.assertTrue(prepared.conditions.is_valid(self.current_snapshot()))

    def test_explicit_context_stays_in_data_and_keeps_work_out_of_the_prompt(
        self,
    ) -> None:
        prepared = self.prepare(context="Inspect the parser result")
        context = prepared.context
        self.assertIsNone(context.work)
        self.assertEqual(context.earlier, ())
        self.assertEqual(
            context.moment,
            Moment(Event.EXPLICIT_CONTEXT, text="Inspect the parser result"),
        )
        self.assertIn(
            "Inspect the parser result", prepared.request["messages"][0]["content"]
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
        if not self.memory.recent_events(
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
        encoded = self.prepare().request["messages"][0]["content"]
        self.assertNotIn('"checkout"', encoded)
        self.assertNotIn('"/parser"', encoded)
        self.assertNotIn("native_id", encoded)
        self.assertNotIn("https://example.test/team/parser/pull/42", encoded)

    def test_head_metadata_does_not_invalidate_the_agent_story(self) -> None:
        self.output()
        prepared = self.prepare()
        self.output(head="two")
        self.assertTrue(prepared.conditions.is_valid(self.current_snapshot()))

    def test_omitted_pull_request_state_does_not_invalidate_commentary(self) -> None:
        self.output()
        prepared = self.prepare()
        self.output(state=PullRequestState.MERGED)
        self.assertTrue(prepared.conditions.is_valid(self.current_snapshot()))

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
        encoded = self.prepare().request["messages"][0]["content"]
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
        self.assertTrue(prepared.conditions.is_valid(self.current_snapshot()))

    def test_disconnected_focus_invalidates_the_captured_subject(self) -> None:
        self.output()
        prepared = self.prepare()
        self.work = tuple(
            subject.model_copy(update={"sessions": ()}) for subject in self.work
        )
        self.assertFalse(prepared.conditions.is_valid(self.current_snapshot()))

    def test_selected_work_context_is_supplied_once(self) -> None:
        self.output()
        self.update_session(connected_work=self.work[0].story)
        prepared = self.prepare()
        context = prepared.context
        assert context.work is not None
        assert context.work.request is not None
        self.assertEqual(
            context.work.request.text, "Make session tracking survive restarts"
        )
        self.assertEqual(
            self.prepare()
            .request["messages"][0]["content"]
            .count("Make session tracking survive restarts"),
            1,
        )

    def test_resumed_work_invalidates_a_quiet_care_remark(self) -> None:
        self.snapshot = sessions_snapshot()
        snapshot = self.current_snapshot().model_copy(
            update={"trigger": CareTrigger(care=CareKind.EXERCISE)}
        )
        selection = self.attention.select(snapshot, SpeechReason.CARE)
        prepared = self.cognition.prompts.prepare(
            snapshot, selection, self.cognition.request
        )
        self.assertTrue(prepared.conditions.is_valid(snapshot))
        resumed = capture_snapshot(
            sessions_snapshot(session_snapshot(turn_active=True))
        )
        self.assertFalse(prepared.conditions.is_valid(resumed))

    def test_new_input_wait_invalidates_a_quiet_greeting(self) -> None:
        self.snapshot = sessions_snapshot()
        prepared = self.prepare(SpeechReason.CLICK)
        waiting = capture_snapshot(
            sessions_snapshot(session_snapshot(waiting_for_input=True))
        )
        self.assertFalse(prepared.conditions.is_valid(waiting))

    def test_summary_saved_from_the_same_result_preserves_commentary(self) -> None:
        prepared = self.prepare()
        self.update_session(summary="The parser preserves its public API.")
        self.assertTrue(prepared.conditions.is_valid(self.current_snapshot()))

    def test_commentary_omits_cached_summaries_empty_fields_and_artifacts(self) -> None:
        self.output()
        self.update_session(
            summary="No implementation results have been reported yet.",
            progress="",
            recent_requests=(),
        )
        prepared = self.prepare()
        context = prepared.context
        self.assertIsNotNone(context.work)
        self.assertIsNotNone(context.moment)
        self.assertEqual(context.earlier, ())
        encoded = prepared.request["messages"][0]["content"]
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
                                CommentaryEvidence(
                                    source=EvidenceSource.USER,
                                    kind=EvidenceKind.INTENT,
                                    details=UserIntent(text="Build a log explorer"),
                                    observed_at=900,
                                ),
                            ),
                            "activity": (
                                CommentaryEvidence(
                                    source=EvidenceSource.SOURCE,
                                    kind=EvidenceKind.PROGRESS,
                                    details=ReportedProgress(
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
                                        CommentaryEvidence(
                                            source=EvidenceSource.SOURCE,
                                            kind=EvidenceKind.INTENT,
                                            details=UserIntent(text="Audit tests"),
                                            observed_at=750,
                                        ),
                                    ),
                                    activity=(
                                        CommentaryEvidence(
                                            source=EvidenceSource.SOURCE,
                                            kind=EvidenceKind.PROGRESS,
                                            details=ReportedProgress(
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
        prepared = self.prepare()
        context = prepared.context
        assert context.work is not None
        assert context.work.request is not None
        self.assertEqual(context.work.request.text, "Build a log explorer")
        assert context.moment is not None
        self.assertIsNone(context.moment.progress)
        self.assertEqual(
            context.earlier,
            (
                ContextFact(
                    FactKind.PROGRESS,
                    SupportedText("Finished simplifying log labels"),
                ),
                ContextFact(
                    FactKind.PROGRESS,
                    SupportedText("Found two redundant assertions"),
                    scope="Audit tests",
                ),
            ),
        )
        self.assertNotIn("observed_at", prepared.request["messages"][0]["content"])

    def test_waiting_reports_only_the_current_input_request(self) -> None:
        self.update_session(
            waiting_for_input=True,
            waiting_seconds=5,
            waiting_kind=WaitingKind.QUESTION,
            waiting_prompt="Which examples should I use?",
            progress="Finished an older task",
        )
        moment = self.prepare().context.moment
        assert moment is not None
        self.assertEqual(moment.event, Event.INPUT_REQUESTED)
        assert moment.progress is not None
        self.assertEqual(moment.progress.text, "Which examples should I use?")
        self.update_session(
            waiting_kind=WaitingKind.PERMISSION,
            waiting_prompt=None,
            waiting_tool="Run the local browser check",
        )
        moment = self.prepare().context.moment
        assert moment is not None
        self.assertEqual(
            moment.event,
            Event.PERMISSION_REQUEST,
        )
        self.assertEqual(moment.tool, "Run the local browser check")
        self.assertIsNone(moment.progress)

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
        self.assertFalse(prepared.conditions.is_valid(self.current_snapshot()))

    def test_selected_terminal_failure_preserves_the_failed_call(self) -> None:
        self.output()
        subject = self.work[0]
        story = subject.story.model_copy(
            update={
                "activity": (
                    CommentaryEvidence(
                        source=EvidenceSource.SOURCE,
                        kind=EvidenceKind.EXECUTION,
                        details=ExecutionStatus(
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
        moment = self.cognition.prompts.prompt(snapshot, selection).moment
        self.assertEqual(moment, Moment(Event.FAILED_CALL))

    def test_selected_completion_preserves_outcome_without_runtime_after_restart(
        self,
    ) -> None:
        self.output()
        subject = self.work[0]
        story = subject.story.model_copy(
            update={
                "activity": (
                    CommentaryEvidence(
                        source=EvidenceSource.SOURCE,
                        kind=EvidenceKind.EXECUTION,
                        details=ExecutionStatus(
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
        moment = self.cognition.prompts.prompt(snapshot, selection).moment
        assert moment is not None and moment.progress is not None
        assert moment is not None
        self.assertEqual(moment.event, Event.TURN_ENDED)
        self.assertEqual(moment.outcome, ExecutionOutcome.SUCCESS)
        assert moment.progress is not None
        self.assertEqual(moment.progress.text, "Parser checks passed")

    def test_selected_child_update_keeps_parent_result_in_historical_context(
        self,
    ) -> None:
        self.output()
        subject = self.work[0]
        story = subject.story.model_copy(
            update={
                "activity": (
                    CommentaryEvidence(
                        source=EvidenceSource.SOURCE,
                        kind=EvidenceKind.EXECUTION,
                        details=ExecutionStatus(
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
                            CommentaryEvidence(
                                source=EvidenceSource.SOURCE,
                                kind=EvidenceKind.PROGRESS,
                                details=ReportedProgress(
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
        context = self.cognition.prompts.prompt(snapshot, selection)
        assert context.work is not None
        assert context.work.request is not None
        self.assertEqual(context.work.request.text, "Inspect the parser")
        assert context.moment is not None
        assert context.moment.progress is not None
        self.assertEqual(
            context.moment.progress.text,
            "Child is inspecting escape handling",
        )
        self.assertIsNone(context.moment.age)
        self.assertTrue(
            any(
                isinstance(item, ContextFact)
                and item.kind == FactKind.PROGRESS
                and item.statement.text == "Parent checks passed"
                and item.age is None
                for item in context.earlier
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
                    CommentaryEvidence(
                        source=EvidenceSource.SOURCE,
                        kind=EvidenceKind.PROGRESS,
                        details=ReportedProgress(
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
                            CommentaryEvidence(
                                source=EvidenceSource.SOURCE,
                                kind=EvidenceKind.PROGRESS,
                                details=ReportedProgress(
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
                identifier: NarrativeSource(
                    identifier=identifier,
                    kind=kind,
                    text=text,
                    full_text=text,
                    scope="current task",
                    outcome=None,
                    origin=origin,
                    observed_at=None,
                ).record()
                for identifier, kind, text, origin in (
                    (
                        "request",
                        SourceKind.REQUEST,
                        "I am building the parser",
                        subject.id,
                    ),
                    (
                        "parser",
                        SourceKind.PROGRESS,
                        "I checked parser compatibility",
                        subject.id,
                    ),
                    ("coverage", SourceKind.PROGRESS, "I checked escapes", related),
                )
            },
        )
        context = self.cognition.prompts.prompt(snapshot, selection, narrative)
        assert context.work is not None
        assert context.work.request is not None
        self.assertEqual(context.work.request.text, "current task")
        assert context.moment is not None
        assert context.moment.progress is not None
        self.assertEqual(context.moment.progress.text, "Escape checks passed")
        history = {
            item.statement.text: item.scope
            for item in context.earlier
            if isinstance(item, ContextFact)
        }
        self.assertEqual(history["Parser compatibility checks passed"], "Build parser")
        self.assertEqual(history["I am building the parser"], "Build parser")
        self.assertEqual(history["I checked parser compatibility"], "Build parser")
        self.assertEqual(history["I checked escapes"], "current task")

    def test_child_report_keeps_the_assignment_observed_before_its_result(self) -> None:
        self.output()
        report = CommentaryEvidence(
            source=EvidenceSource.SOURCE,
            kind=EvidenceKind.PROGRESS,
            details=ReportedProgress(
                reported_progress="Found two redundant assertions"
            ),
            observed_at=850,
        )
        earlier_assignment = CommentaryEvidence(
            source=EvidenceSource.SOURCE,
            kind=EvidenceKind.INTENT,
            details=UserIntent(text="Audit tests"),
            observed_at=800,
        )
        current_assignment = CommentaryEvidence(
            source=EvidenceSource.SOURCE,
            kind=EvidenceKind.INTENT,
            details=UserIntent(text="Check the explorer website"),
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
        prepared = self.prepare()
        context = prepared.context
        self.assertEqual(context.earlier[-1].scope, "Audit tests")
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
        prepared = self.prepare()
        context = prepared.context
        self.assertEqual(context.earlier[-1].scope, "related task")

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
        moment = prepared.context.moment
        assert moment is not None
        self.assertEqual(moment.event, Event.WORK_UPDATE)
        self.assertEqual(moment.state, WorkState.QUIET)
        self.assertEqual(
            moment.outcome,
            ExecutionOutcome.CANCELLED,
        )

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
        moment = prepared.context.moment
        assert moment is not None
        self.assertIsNone(moment.tool)
        self.assertEqual(moment.elapsed, "3 minutes")

    def test_historical_reports_preserve_terminal_outcomes_without_duplicates(
        self,
    ) -> None:
        self.output()
        progress = CommentaryEvidence(
            source=EvidenceSource.SOURCE,
            kind=EvidenceKind.PROGRESS,
            details=ReportedProgress(reported_progress="Checked parser compatibility"),
            observed_at=800,
        )
        ended = CommentaryEvidence(
            source=EvidenceSource.SOURCE,
            kind=EvidenceKind.EXECUTION,
            details=ExecutionStatus(
                execution="turn ended",
                reported_progress="Checked parser compatibility",
                reported_execution_outcome=ExecutionOutcome.CANCELLED,
            ),
            observed_at=800,
        )
        request = CommentaryEvidence(
            source=EvidenceSource.USER,
            kind=EvidenceKind.INTENT,
            details=UserIntent(text="Build a log explorer"),
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
        earlier = self.prepare().context.earlier
        self.assertEqual(len(earlier), 2)
        self.assertEqual(
            earlier[0].outcome,
            ExecutionOutcome.CANCELLED,
        )
        self.assertEqual(earlier[1].outcome, earlier[0].outcome)
        self.assertEqual(earlier[1].scope, "related task")

    def test_followup_keeps_its_prior_request_ahead_of_optional_child_reports(
        self,
    ) -> None:
        self.output()
        original_request = CommentaryEvidence(
            source=EvidenceSource.USER,
            kind=EvidenceKind.INTENT,
            details=UserIntent(text="Build a log explorer"),
            observed_at=700,
        )
        followup = CommentaryEvidence(
            source=EvidenceSource.USER,
            kind=EvidenceKind.INTENT,
            details=UserIntent(text="proceed"),
            observed_at=900,
        )
        children = tuple(
            SessionContribution(
                scope=f"Assignment {index}",
                session=SessionIdentity(
                    source=SessionSource.COMMAND, native_id=f"child-{index}"
                ),
                activity=(
                    CommentaryEvidence(
                        source=EvidenceSource.SOURCE,
                        kind=EvidenceKind.PROGRESS,
                        details=ReportedProgress(
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
        context = prepared.context
        assert context.work is not None
        assert context.work.request is not None
        self.assertEqual(context.work.request.text, "proceed")
        self.assertEqual(len(context.earlier), 6)
        assert isinstance(context.earlier[0], ContextFact)
        self.assertEqual(context.earlier[0].statement.text, "Build a log explorer")
        assert isinstance(context.earlier[1], ContextFact)
        self.assertEqual(
            context.earlier[1].statement.text,
            "Report 1: " + "Checked parser behavior. " * 7 + "Checked parser behavior.",
        )
        assert isinstance(context.earlier[-1], ContextFact)
        self.assertEqual(
            context.earlier[-1].statement.text,
            "Report 5: " + "Checked parser behavior. " * 7 + "Checked parser behavior.",
        )
        one_request = deepcopy(prepared.request)
        selection = self.attention.select(self.current_snapshot(), SpeechReason.ROUTINE)
        one_request["messages"][0]["content"] = self.cognition.prompts.payload(
            replace(context, earlier=context.earlier[:1]),
            select_prompt(selection),
            (),
            None,
        )
        self.config.prompt_target_bytes = encoded_size(one_request) + 1
        limited = self.prepare().context
        self.assertEqual(limited.earlier, context.earlier[:1])
