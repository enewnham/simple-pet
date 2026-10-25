import json
import tempfile
import unittest
from pathlib import Path

from pet_daemon.attention import AttentionSelection
from pet_daemon.commentary.cognition import Cognition
from pet_daemon.commentary.contracts import ModelRequest
from pet_daemon.commentary.story import (
    WorkNarrativeRequest,
    prepare,
    validate,
)
from pet_daemon.config import Config
from pet_daemon.domain import (
    EvidenceKind,
    EvidenceSource,
    ExecutionOutcome,
    SessionSource,
    SpeechReason,
    SpeechSituation,
    WorkSubjectKind,
)
from pet_daemon.identity import SessionIdentity, session_data
from pet_daemon.storage import Memory
from pet_daemon.work.evidence import CommentaryEvidence, UserIntent
from pet_daemon.work.models import ExecutionEvidence, IntentEvidence, ProgressEvidence
from pet_daemon.work.narrative import (
    NarrativeStatement,
    WorkNarrative,
    semantic_revision,
)
from pet_daemon.work.observations import SessionObservation
from pet_daemon.work.shared_work import WorkSubject
from pet_daemon.work.stories import SessionContribution, WorkStory
from Tests.fixtures import (
    capture_snapshot,
    require,
    session_snapshot,
    sessions_snapshot,
)


class NoModel:
    async def complete(self, request: ModelRequest, reason: str) -> dict[str, object]:
        raise AssertionError("Preparing a story must not call the model")


class StoryTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.memory = Memory(
            Path(directory.name) / "memory.sqlite3", "pet", clock=lambda: 1000
        )
        self.addCleanup(self.memory.close)
        self.cognition = Cognition(
            NoModel(), Config(pet_dir=Path(directory.name)), memory=self.memory
        )

    def add(
        self,
        *reports: IntentEvidence | ProgressEvidence | ExecutionEvidence,
        session: SessionIdentity | None = None,
        parent: SessionIdentity | None = None,
    ) -> None:
        identity = (
            session
            if session is not None
            else SessionIdentity(source=SessionSource.CODEX, native_id="root")
        )
        self.memory.work.record_session(
            SessionObservation(session=identity, parent=parent, reports=reports)
        )

    def prepared(self, *, target_bytes: int = 12000) -> WorkNarrativeRequest:
        focus = session_snapshot(
            id=SessionIdentity(source=SessionSource.CODEX, native_id="root")
        )
        swarm = sessions_snapshot(focus)
        subject = self.memory.connected_work(swarm)[0]
        snapshot = capture_snapshot(swarm, work=(subject,))
        selection = AttentionSelection(
            subject=subject,
            focus_session=focus,
            context_sessions=(),
            reason=SpeechReason.ROUTINE,
            situation=SpeechSituation.ROUTINE,
        )
        return require(
            prepare(
                snapshot,
                selection,
                self.memory,
                self.cognition.request,
                target_bytes=target_bytes,
            )
        )

    def test_story_sources_preserve_observation_time_across_later_requests(
        self,
    ) -> None:
        self.memory.work.clock = lambda: 700
        self.add(
            IntentEvidence(
                session=SessionIdentity(source=SessionSource.CODEX, native_id="root"),
                text="Inspect parser",
            )
        )
        self.memory.work.clock = lambda: 880
        self.add(
            ProgressEvidence(
                session=SessionIdentity(source=SessionSource.CODEX, native_id="root"),
                text="Parser checks passed",
            )
        )
        self.memory.clock = lambda: 1000
        first = self.prepared()
        report = next(
            source
            for source in first.sources.values()
            if source["kind"] == "reported_progress"
        )
        self.assertEqual(report["observed_at"], 880)
        self.assertEqual(
            report["origin"],
            session_data(SessionIdentity(source=SessionSource.CODEX, native_id="root")),
        )
        payload = json.loads(first.request["messages"][0]["content"])
        evidence = payload["current_moment"]["evidence"]
        self.assertEqual(evidence[-1]["age"], "2 minutes ago")
        self.memory.clock = lambda: 8200
        later = self.prepared()
        payload = json.loads(later.request["messages"][0]["content"])
        self.assertEqual(
            payload["current_moment"]["evidence"][-1]["age"], "2 hours ago"
        )
        self.assertEqual(first.sources, later.sources)
        self.assertEqual(first.revision, later.revision)

    def test_latest_followup_keeps_original_intent_beyond_recent_requests(self) -> None:
        self.add(
            IntentEvidence(
                session=SessionIdentity(source=SessionSource.CODEX, native_id="root"),
                text="Implement quoted-field parsing",
            )
        )
        for text in (
            "Keep quotes",
            "Also escapes",
            "Check nesting",
            "Test it",
            "Proceed",
        ):
            self.add(
                IntentEvidence(
                    session=SessionIdentity(
                        source=SessionSource.CODEX, native_id="root"
                    ),
                    text=text,
                )
            )
        prepared = self.prepared()
        payload = json.loads(prepared.request["messages"][0]["content"])
        self.assertIn("Implement quoted-field parsing", json.dumps(payload))
        self.assertEqual(
            prepared.sources[prepared.latest_request_source]["text"], "Proceed"
        )

    def test_speech_quotes_code_and_tables_do_not_become_work_evidence(self) -> None:
        self.memory.record_speech("Mail's in. Quiet now.", SpeechReason.ROUTINE)
        self.add(
            IntentEvidence(
                session=SessionIdentity(source=SessionSource.CODEX, native_id="root"),
                text="Review commentary",
            )
        )
        self.add(
            ProgressEvidence(
                session=SessionIdentity(source=SessionSource.CODEX, native_id="root"),
                text="Checked the parser.\n\n> Mail's in. Quiet now.\n\n```python\nsecret_table = 1\n```\n\n| status | note |\n|---|---|\n| bad | invented |",
            )
        )
        request = self.prepared().request
        text = json.dumps(request)
        self.assertIn("Checked the parser", text)
        for excluded in ("Mail's in", "secret_table", "invented"):
            self.assertNotIn(excluded, request["messages"][0]["content"])

    def test_established_decision_survives_a_long_session_within_the_prompt_budget(
        self,
    ) -> None:
        self.add(
            IntentEvidence(
                session=SessionIdentity(source=SessionSource.CODEX, native_id="root"),
                text="Implement the parser",
            )
        )
        self.add(
            IntentEvidence(
                session=SessionIdentity(source=SessionSource.CODEX, native_id="root"),
                text="Keep the public parser interface compatible",
            )
        )
        initial = self.prepared()
        decision = NarrativeStatement(
            text="Preserve the public parser interface.",
            sources=(initial.latest_request_source,),
        )
        self.memory.save_narrative(
            WorkNarrative(
                subject_id=initial.subject_id,
                revision=initial.revision,
                intent=NarrativeStatement(
                    text="Implement the parser",
                    sources=(initial.latest_request_source,),
                ),
                decisions=(decision,),
                sources=initial.sources,
            )
        )
        for index in range(50):
            self.add(
                ProgressEvidence(
                    session=SessionIdentity(
                        source=SessionSource.CODEX, native_id="root"
                    ),
                    text=f"Inspected case {index}. "
                    + "Each nested escaped quote needs a separate case. " * 10,
                )
            )
        self.add(
            IntentEvidence(
                session=SessionIdentity(source=SessionSource.CODEX, native_id="root"),
                text="Proceed",
            )
        )
        prepared = self.prepared()
        payload = json.loads(prepared.request["messages"][0]["content"])
        self.assertIn(initial.latest_request_source, prepared.sources)
        self.assertIn(decision.text, json.dumps(payload["earlier_context"]))
        self.assertLessEqual(
            len(prepared.request["messages"][0]["content"].encode()), 12000
        )

    def test_spoken_story_paraphrases_are_excluded_from_later_story_inputs(
        self,
    ) -> None:
        self.add(
            IntentEvidence(
                session=SessionIdentity(source=SessionSource.CODEX, native_id="root"),
                text="Review quoted-field parsing",
            )
        )
        prepared = self.prepared()
        spoken = "I am inspecting the parser's quoted fields."
        self.memory.save_narrative(
            WorkNarrative(
                subject_id=prepared.subject_id,
                revision=prepared.revision,
                intent=NarrativeStatement(
                    text=spoken, sources=(prepared.latest_request_source,)
                ),
                sources=prepared.sources,
            )
        )
        self.memory.record_speech(spoken, SpeechReason.ROUTINE)
        self.add(
            IntentEvidence(
                session=SessionIdentity(source=SessionSource.CODEX, native_id="root"),
                text="Check escaped quotes too",
            )
        )
        payload = json.loads(self.prepared().request["messages"][0]["content"])
        self.assertNotIn(spoken, json.dumps(payload))
        self.assertNotIn("earlier_context", payload)

    def test_latest_report_survives_a_budget_filled_by_prior_citations(self) -> None:
        for budget, count in ((12000, 8), (16000, 11)):
            with self.subTest(budget=budget):
                self.memory.clear()
                self.add(
                    IntentEvidence(
                        session=SessionIdentity(
                            source=SessionSource.CODEX, native_id="root"
                        ),
                        text="Implement parser",
                    )
                )
                for index in range(count):
                    self.add(
                        ProgressEvidence(
                            session=SessionIdentity(
                                source=SessionSource.CODEX, native_id="root"
                            ),
                            text=f"Inspected case {index}. "
                            + "Each nested escaped quote needs a separate case. " * 25,
                        )
                    )
                initial = self.prepared(target_bytes=budget)
                reports = tuple(
                    key
                    for key, source in initial.sources.items()
                    if source["kind"] == "reported_progress"
                )
                self.assertGreaterEqual(len(reports), 4)
                self.memory.save_narrative(
                    WorkNarrative(
                        subject_id=initial.subject_id,
                        revision=initial.revision,
                        intent=NarrativeStatement(
                            text="Implement parser",
                            sources=(initial.latest_request_source,),
                        ),
                        decisions=tuple(
                            NarrativeStatement(
                                text="Keep escaped quote cases distinct.",
                                sources=reports[index::4],
                            )
                            for index in range(4)
                        ),
                        sources=initial.sources,
                    )
                )
                self.add(
                    ProgressEvidence(
                        session=SessionIdentity(
                            source=SessionSource.CODEX, native_id="root"
                        ),
                        text="Critical regression in the latest parser case. "
                        + "A newly found failure needs investigation. " * 30,
                    )
                )
                prepared = self.prepared(target_bytes=budget)
                payload = prepared.request["messages"][0]["content"]
                self.assertIn("Critical regression", payload)
                self.assertLessEqual(len(payload.encode()), budget)

    def test_child_alias_keeps_assignment_at_report_time(self) -> None:
        self.add(
            IntentEvidence(
                session=SessionIdentity(source=SessionSource.CODEX, native_id="root"),
                text="Improve parser",
            )
        )
        self.add(
            IntentEvidence(
                session=SessionIdentity(source=SessionSource.CODEX, native_id="old"),
                text="Audit escapes",
            ),
            session=SessionIdentity(source=SessionSource.CODEX, native_id="old"),
            parent=SessionIdentity(source=SessionSource.CODEX, native_id="root"),
        )
        self.memory.work.record_session(
            SessionObservation(
                session=SessionIdentity(source=SessionSource.CODEX, native_id="native"),
                aliases=(SessionIdentity(source=SessionSource.CODEX, native_id="old"),),
                reports=(
                    ProgressEvidence(
                        session=SessionIdentity(
                            source=SessionSource.CODEX, native_id="native"
                        ),
                        text="Escape checks passed",
                    ),
                ),
            )
        )
        self.add(
            IntentEvidence(
                session=SessionIdentity(source=SessionSource.CODEX, native_id="native"),
                text="Audit cancellation",
            ),
            ProgressEvidence(
                session=SessionIdentity(source=SessionSource.CODEX, native_id="native"),
                text="Cancellation checks passed",
            ),
            session=SessionIdentity(source=SessionSource.CODEX, native_id="native"),
        )
        reports = {
            source["text"]: source["scope"]
            for source in self.prepared().sources.values()
            if source["kind"] == "reported_progress"
        }
        self.assertEqual(reports["Escape checks passed"], "Audit escapes")
        self.assertEqual(
            reports["Cancellation checks passed"],
            "Audit cancellation",
        )

    def test_child_report_keeps_assignment_and_failed_outcome(self) -> None:
        self.add(
            IntentEvidence(
                session=SessionIdentity(source=SessionSource.CODEX, native_id="root"),
                text="Improve parser",
            )
        )
        self.add(
            IntentEvidence(
                session=SessionIdentity(source=SessionSource.CODEX, native_id="child"),
                text="Audit escaped quotes",
            ),
            session=SessionIdentity(source=SessionSource.CODEX, native_id="child"),
            parent=SessionIdentity(source=SessionSource.CODEX, native_id="root"),
        )
        self.add(
            ExecutionEvidence(
                session=SessionIdentity(source=SessionSource.CODEX, native_id="child"),
                event="turn_error",
                outcome=ExecutionOutcome.ERROR,
                tool_name=None,
                progress="Found one failing escaped quote case",
            ),
            session=SessionIdentity(source=SessionSource.CODEX, native_id="child"),
            parent=SessionIdentity(source=SessionSource.CODEX, native_id="root"),
        )
        sources = list(self.prepared().sources.values())
        report = next(source for source in sources if source["kind"] == "turn_ended")
        self.assertIn("Audit escaped quotes", str(report["scope"]))
        self.assertEqual(report["outcome"], "error")

    def test_related_task_without_an_assignment_keeps_its_result_scoped(self) -> None:
        self.add(
            IntentEvidence(
                session=SessionIdentity(source=SessionSource.CODEX, native_id="root"),
                text="Improve parser",
            )
        )
        self.add(
            ProgressEvidence(
                session=SessionIdentity(source=SessionSource.CODEX, native_id="child"),
                text="Escape checks passed",
            ),
            session=SessionIdentity(source=SessionSource.CODEX, native_id="child"),
            parent=SessionIdentity(source=SessionSource.CODEX, native_id="root"),
        )
        prepared = self.prepared()
        report = next(
            source
            for source in prepared.sources.values()
            if source["kind"] == "reported_progress"
        )
        self.assertEqual(report["scope"], "related task")
        payload = json.loads(prepared.request["messages"][0]["content"])
        self.assertEqual(
            {source["scope"] for source in payload["current_moment"]["evidence"]},
            {"current task", "related task"},
        )

    def test_related_assignment_name_does_not_select_the_current_request(self) -> None:
        self.add(
            IntentEvidence(
                session=SessionIdentity(source=SessionSource.CODEX, native_id="root"),
                text="Improve parser",
            )
        )
        self.add(
            IntentEvidence(
                session=SessionIdentity(source=SessionSource.CODEX, native_id="child"),
                text="current task",
            ),
            session=SessionIdentity(source=SessionSource.CODEX, native_id="child"),
            parent=SessionIdentity(source=SessionSource.CODEX, native_id="root"),
        )
        prepared = self.prepared()
        self.assertEqual(
            prepared.sources[prepared.latest_request_source]["text"], "Improve parser"
        )
        self.assertEqual(
            prepared.sources[prepared.latest_request_source]["origin"],
            session_data(SessionIdentity(source=SessionSource.CODEX, native_id="root")),
        )
        child_request = next(
            source
            for source in prepared.sources.values()
            if source.get("text") == "current task"
        )
        self.assertEqual(child_request["scope"], "current task")

    def test_related_task_view_keeps_the_current_request_selected(self) -> None:
        root = SessionIdentity(source=SessionSource.CODEX, native_id="root")
        story = WorkStory(
            id=root,
            intent="Improve parser",
            revision="fallback",
            requests=(
                CommentaryEvidence(
                    source=EvidenceSource.USER,
                    kind=EvidenceKind.INTENT,
                    details=UserIntent(text="Improve parser"),
                ),
            ),
            sessions=(
                SessionContribution(
                    scope="Audit escapes",
                    session=SessionIdentity(
                        source=SessionSource.CODEX, native_id="coverage"
                    ),
                    requests=(
                        CommentaryEvidence(
                            source=EvidenceSource.USER,
                            kind=EvidenceKind.INTENT,
                            details=UserIntent(text="Audit escapes"),
                        ),
                    ),
                ),
            ),
        )
        subject = WorkSubject(
            id=root,
            kind=WorkSubjectKind.SESSION,
            changed_at=1000,
            revision="fallback",
            story=story,
        )
        focus = session_snapshot(id=root)
        snapshot = capture_snapshot(sessions_snapshot(focus), work=(subject,))
        selection = AttentionSelection(
            subject=subject,
            focus_session=focus,
            context_sessions=(),
            reason=SpeechReason.ROUTINE,
            situation=SpeechSituation.ROUTINE,
        )
        prepared = require(
            prepare(snapshot, selection, self.memory, self.cognition.request)
        )
        self.assertEqual(
            prepared.sources[prepared.latest_request_source]["text"], "Improve parser"
        )
        self.assertTrue(
            any(
                source.get("text") == "Audit escapes"
                for source in prepared.sources.values()
            )
        )

    def test_validation_rejects_unknown_sources_and_stale_intent(self) -> None:
        self.add(
            IntentEvidence(
                session=SessionIdentity(source=SessionSource.CODEX, native_id="root"),
                text="Improve parser",
            ),
            IntentEvidence(
                session=SessionIdentity(source=SessionSource.CODEX, native_id="root"),
                text="Proceed",
            ),
        )
        prepared = self.prepared()
        old = next(
            source
            for source in prepared.sources
            if source != prepared.latest_request_source
        )
        for refs in (["unknown"], [old]):
            with self.subTest(refs=refs), self.assertRaises(ValueError):
                validate(
                    {
                        "intent": {"text": "Improve parser", "sources": refs},
                        "decisions": [],
                        "progress": [],
                    },
                    prepared,
                )
        narrative = validate(
            {
                "intent": {
                    "text": "Improve parser",
                    "sources": [prepared.latest_request_source],
                },
                "decisions": [],
                "progress": [],
            },
            prepared,
        )
        self.assertEqual(narrative.revision, prepared.revision)

    def test_repository_refresh_and_running_events_do_not_change_story_revision(
        self,
    ) -> None:
        self.add(
            IntentEvidence(
                session=SessionIdentity(source=SessionSource.CODEX, native_id="root"),
                text="Improve parser",
            )
        )
        original = self.memory.connected_work(sessions_snapshot())[0].story
        self.add(
            ExecutionEvidence(
                session=SessionIdentity(source=SessionSource.CODEX, native_id="root"),
                event="tool_start",
                outcome=None,
                tool_name="Bash",
                progress=None,
            )
        )
        running = self.memory.connected_work(sessions_snapshot())[0].story
        self.assertEqual(semantic_revision(original), semantic_revision(running))
        self.assertEqual(
            semantic_revision(original),
            semantic_revision(
                original.model_copy(
                    update={"revision": "git refreshed", "artifacts": ()}
                )
            ),
        )
        self.add(
            ExecutionEvidence(
                session=SessionIdentity(source=SessionSource.CODEX, native_id="root"),
                event="turn_done",
                outcome=ExecutionOutcome.SUCCESS,
                tool_name=None,
                progress="Parser checked",
            )
        )
        ended = self.memory.connected_work(sessions_snapshot())[0].story
        self.assertNotEqual(semantic_revision(original), semantic_revision(ended))

    def test_matching_terminal_and_progress_reports_share_text_and_keep_citations(
        self,
    ) -> None:
        root = SessionIdentity(source=SessionSource.CODEX, native_id="root")
        self.add(
            IntentEvidence(session=root, text="Check parser"),
            ExecutionEvidence(
                session=root,
                event="turn_done",
                outcome=ExecutionOutcome.SUCCESS,
                tool_name=None,
                progress="Parser checks passed",
            ),
            ProgressEvidence(session=root, text="Parser checks passed"),
        )
        prepared = self.prepared()
        payload = json.loads(prepared.request["messages"][0]["content"])
        reports = [
            item
            for item in payload["current_moment"]["evidence"]
            if item.get("text") == "Parser checks passed"
        ]
        self.assertEqual(len(reports), 1)
        report_ids = {
            key
            for key, source in prepared.sources.items()
            if source.get("text") == "Parser checks passed"
        }
        self.assertEqual(set(reports[0]["supporting_ids"]), report_ids)
        self.assertIn("execution", reports[0])
        narrative = validate(
            {
                "intent": {
                    "text": "Check parser",
                    "sources": [prepared.latest_request_source],
                },
                "decisions": [],
                "progress": [
                    {"text": "Parser checks passed", "sources": list(report_ids)}
                ],
            },
            prepared,
        )
        self.assertEqual(set(narrative.progress[0].sources), report_ids)

    def test_similar_reports_keep_distinct_observation_times(self) -> None:
        root = SessionIdentity(source=SessionSource.CODEX, native_id="root")
        self.add(
            IntentEvidence(session=root, text="Check parser"),
            ExecutionEvidence(
                session=root,
                event="turn_done",
                outcome=ExecutionOutcome.SUCCESS,
                tool_name=None,
                progress="Parser checks passed",
            ),
        )
        self.memory.work.clock = lambda: 1100
        self.add(ProgressEvidence(session=root, text="Parser checks passed"))
        payload = json.loads(self.prepared().request["messages"][0]["content"])
        self.assertEqual(
            sum(
                item.get("text") == "Parser checks passed"
                for item in payload["current_moment"]["evidence"]
            ),
            2,
        )

    def test_repeated_progress_keeps_distinct_times_and_stable_citations(self) -> None:
        root = SessionIdentity(source=SessionSource.CODEX, native_id="root")
        self.memory.work.clock = lambda: 700
        self.add(IntentEvidence(session=root, text="Check parser"))
        self.memory.work.clock = lambda: 800
        self.add(ProgressEvidence(session=root, text="Parser checks passed"))
        self.memory.work.clock = lambda: 850
        self.add(ProgressEvidence(session=root, text="Checking another parser case"))
        self.memory.work.clock = lambda: 900
        self.add(ProgressEvidence(session=root, text="Parser checks passed"))
        first = self.prepared()
        reports = {
            key: source
            for key, source in first.sources.items()
            if source.get("text") == "Parser checks passed"
        }
        self.assertEqual(len(reports), 2)
        self.assertEqual(
            [source["observed_at"] for source in reports.values()], [800, 900]
        )
        first_payload = json.loads(first.request["messages"][0]["content"])
        first_evidence = [
            item
            for item in first_payload["current_moment"]["evidence"]
            if item.get("text") == "Parser checks passed"
        ]
        self.assertEqual([item["id"] for item in first_evidence], list(reports))
        self.assertEqual(
            [item["age"] for item in first_evidence], ["3 minutes ago", "1 minute ago"]
        )
        self.memory.clock = lambda: 8200
        later = self.prepared()
        later_payload = json.loads(later.request["messages"][0]["content"])
        later_evidence = [
            item
            for item in later_payload["current_moment"]["evidence"]
            if item.get("text") == "Parser checks passed"
        ]
        self.assertEqual(later.sources, first.sources)
        self.assertEqual([item["id"] for item in later_evidence], list(reports))
        self.assertEqual(
            [item["age"] for item in later_evidence], ["2 hours ago", "2 hours ago"]
        )

    def test_matching_report_text_from_related_tasks_keeps_both_observations(
        self,
    ) -> None:
        root = SessionIdentity(source=SessionSource.CODEX, native_id="root")
        child = SessionIdentity(source=SessionSource.CODEX, native_id="child")
        self.add(
            IntentEvidence(session=root, text="Check parser"),
            ExecutionEvidence(
                session=root,
                event="turn_done",
                outcome=ExecutionOutcome.SUCCESS,
                tool_name=None,
                progress="Checks passed",
            ),
        )
        self.add(
            IntentEvidence(session=child, text="current task"),
            ProgressEvidence(session=child, text="Checks passed"),
            session=child,
            parent=root,
        )
        payload = json.loads(self.prepared().request["messages"][0]["content"])
        reports = [
            item
            for item in payload["current_moment"]["evidence"]
            if item.get("text") == "Checks passed"
        ]
        self.assertEqual(len(reports), 2)
        self.assertTrue(all("supporting_ids" not in item for item in reports))

    def test_reports_with_matching_excerpts_keep_different_full_texts(self) -> None:
        root = SessionIdentity(source=SessionSource.CODEX, native_id="root")
        prefix = "Checked a shared parser case. " * 50
        self.add(
            IntentEvidence(session=root, text="Check parser"),
            ExecutionEvidence(
                session=root,
                event="turn_done",
                outcome=ExecutionOutcome.SUCCESS,
                tool_name=None,
                progress=prefix + "Found a failure.",
            ),
            ProgressEvidence(session=root, text=prefix + "Found no failures."),
        )
        payload = json.loads(self.prepared().request["messages"][0]["content"])
        reports = [
            item
            for item in payload["current_moment"]["evidence"]
            if str(item.get("text", "")).startswith("Checked a shared parser case.")
        ]
        self.assertEqual(len(reports), 2)
        self.assertEqual(reports[0]["text"], reports[1]["text"])

    def test_same_kind_reports_with_matching_excerpts_keep_distinct_citations(
        self,
    ) -> None:
        root = SessionIdentity(source=SessionSource.CODEX, native_id="root")
        prefix = "Checked a shared parser case. " * 50
        self.add(
            IntentEvidence(session=root, text="Check parser"),
            ProgressEvidence(session=root, text=prefix + "Found a failure."),
            ProgressEvidence(session=root, text=prefix + "Found no failures."),
        )
        first = self.prepared()
        reports = {
            key: source
            for key, source in first.sources.items()
            if source["kind"] == "reported_progress"
        }
        self.assertEqual(len(reports), 2)
        self.assertEqual(len({str(source["text"]) for source in reports.values()}), 1)
        payload = json.loads(first.request["messages"][0]["content"])
        evidence = [
            item
            for item in payload["current_moment"]["evidence"]
            if str(item.get("text", "")).startswith("Checked a shared parser case.")
        ]
        self.assertEqual([item["id"] for item in evidence], list(reports))
        self.assertTrue(all("supporting_ids" not in item for item in evidence))
        self.memory.clock = lambda: 8200
        later = self.prepared()
        self.assertEqual(later.sources, first.sources)

    def test_prior_memory_keeps_intent_decision_and_progress_categories(self) -> None:
        root = SessionIdentity(source=SessionSource.CODEX, native_id="root")
        self.add(IntentEvidence(session=root, text="Check parser"))
        initial = self.prepared()
        refs = (initial.latest_request_source,)
        self.memory.save_narrative(
            WorkNarrative(
                subject_id=initial.subject_id,
                revision=initial.revision,
                intent=NarrativeStatement(text="Check parser", sources=refs),
                decisions=(
                    NarrativeStatement(text="Keep the interface", sources=refs),
                ),
                progress=(NarrativeStatement(text="Inspected one case", sources=refs),),
                sources=initial.sources,
            )
        )
        payload = json.loads(self.prepared().request["messages"][0]["content"])
        self.assertEqual(
            [(item["category"], item["text"]) for item in payload["earlier_context"]],
            [
                ("intent", "Check parser"),
                ("decision", "Keep the interface"),
                ("progress", "Inspected one case"),
            ],
        )

    def test_short_followup_retains_the_preceding_exchange_within_a_small_budget(
        self,
    ) -> None:
        root = SessionIdentity(source=SessionSource.CODEX, native_id="root")
        self.add(IntentEvidence(session=root, text="Improve parser"))
        for index in range(12):
            self.add(
                ProgressEvidence(
                    session=root,
                    text=f"Inspected old case {index}. "
                    + "Checked nested quotes. " * 25,
                )
            )
        self.add(
            IntentEvidence(session=root, text="Can this be simpler?"),
            ProgressEvidence(
                session=root, text="I recommend removing the duplicate type."
            ),
            IntentEvidence(session=root, text="ok"),
        )
        prepared = self.prepared(target_bytes=3500)
        content = prepared.request["messages"][0]["content"]
        self.assertIn("Can this be simpler?", content)
        self.assertIn("I recommend removing the duplicate type.", content)
        self.assertEqual(prepared.sources[prepared.latest_request_source]["text"], "ok")
        self.assertLessEqual(len(content.encode()), 3500)

    def test_many_short_reports_fit_the_model_facing_budget(self) -> None:
        root = SessionIdentity(source=SessionSource.CODEX, native_id="root")
        self.add(IntentEvidence(session=root, text="Check parser"))
        for index in range(100):
            self.memory.work.clock = lambda index=index: 1100 + index
            self.add(ProgressEvidence(session=root, text=f"Checked case {index}."))
        prepared = self.prepared(target_bytes=12000)
        content = prepared.request["messages"][0]["content"]
        payload = json.loads(content)
        evidence = payload["current_moment"]["evidence"]
        self.assertIn("Check parser", content)
        self.assertIn("Checked case 99.", content)
        self.assertLess(len(evidence), 101)
        self.assertLessEqual(len(content.encode()), 12000)

    def test_repeated_acknowledgment_follows_its_latest_preceding_report(self) -> None:
        root = SessionIdentity(source=SessionSource.CODEX, native_id="root")
        self.memory.work.clock = lambda: 700
        self.add(
            IntentEvidence(session=root, text="Improve parser"),
            IntentEvidence(session=root, text="ok"),
            ProgressEvidence(session=root, text="Checked the first case."),
        )
        self.memory.work.clock = lambda: 800
        self.add(
            IntentEvidence(session=root, text="Can we simplify the next case?"),
            ProgressEvidence(
                session=root, text="I recommend removing the extra branch."
            ),
        )
        self.memory.work.clock = lambda: 900
        self.add(IntentEvidence(session=root, text="ok"))
        prepared = self.prepared(target_bytes=3500)
        payload = json.loads(prepared.request["messages"][0]["content"])
        evidence = payload["current_moment"]["evidence"]
        self.assertEqual(sum(item.get("text") == "ok" for item in evidence), 2)
        self.assertEqual(evidence[-1]["id"], prepared.latest_request_source)
        self.assertEqual(evidence[-2]["text"], "I recommend removing the extra branch.")
        self.assertEqual(evidence[-3]["text"], "Can we simplify the next case?")
