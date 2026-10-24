import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from pet_daemon.attention import AttentionSelection
from pet_daemon.commentary.cognition import Cognition
from pet_daemon.commentary.contracts import ModelRequest
from pet_daemon.commentary.story import PreparedStory, prepare, validate
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
from pet_daemon.work.models import ExecutionEvidence, IntentEvidence, ProgressEvidence
from pet_daemon.work.narrative import (
    NarrativeStatement,
    WorkNarrative,
    semantic_revision,
)
from pet_daemon.work.observations import SessionObservation
from pet_daemon.work.views import (
    EvidenceView,
    IntentView,
    SessionContribution,
    WorkStory,
    WorkSubject,
)
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
            else SessionIdentity(SessionSource.CODEX, "root")
        )
        self.memory.work.record_session(
            SessionObservation(identity, parent=parent, reports=reports)
        )

    def prepared(self, *, target_bytes: int = 12000) -> PreparedStory:
        focus = session_snapshot(id=SessionIdentity(SessionSource.CODEX, "root"))
        swarm = sessions_snapshot(focus)
        subject = self.memory.connected_work(swarm)[0]
        snapshot = capture_snapshot(swarm, work=(subject,))
        selection = AttentionSelection(
            subject, focus, (), SpeechReason.ROUTINE, SpeechSituation.ROUTINE
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
                SessionIdentity(SessionSource.CODEX, "root"), "Inspect parser"
            )
        )
        self.memory.work.clock = lambda: 880
        self.add(
            ProgressEvidence(
                SessionIdentity(SessionSource.CODEX, "root"), "Parser checks passed"
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
            report["origin"], session_data(SessionIdentity(SessionSource.CODEX, "root"))
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
                SessionIdentity(SessionSource.CODEX, "root"),
                "Implement quoted-field parsing",
            )
        )
        for text in (
            "Keep quotes",
            "Also escapes",
            "Check nesting",
            "Test it",
            "Proceed",
        ):
            self.add(IntentEvidence(SessionIdentity(SessionSource.CODEX, "root"), text))
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
                SessionIdentity(SessionSource.CODEX, "root"), "Review commentary"
            )
        )
        self.add(
            ProgressEvidence(
                SessionIdentity(SessionSource.CODEX, "root"),
                "Checked the parser.\n\n> Mail's in. Quiet now.\n\n```python\nsecret_table = 1\n```\n\n| status | note |\n|---|---|\n| bad | invented |",
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
                SessionIdentity(SessionSource.CODEX, "root"), "Implement the parser"
            )
        )
        self.add(
            IntentEvidence(
                SessionIdentity(SessionSource.CODEX, "root"),
                "Keep the public parser interface compatible",
            )
        )
        initial = self.prepared()
        decision = NarrativeStatement(
            "Preserve the public parser interface.", (initial.latest_request_source,)
        )
        self.memory.save_narrative(
            WorkNarrative(
                initial.subject_id,
                initial.revision,
                NarrativeStatement(
                    "Implement the parser", (initial.latest_request_source,)
                ),
                decisions=(decision,),
                sources=initial.sources,
            )
        )
        for index in range(50):
            self.add(
                ProgressEvidence(
                    SessionIdentity(SessionSource.CODEX, "root"),
                    f"Inspected case {index}. "
                    + "Each nested escaped quote needs a separate case. " * 10,
                )
            )
        self.add(
            IntentEvidence(SessionIdentity(SessionSource.CODEX, "root"), "Proceed")
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
                SessionIdentity(SessionSource.CODEX, "root"),
                "Review quoted-field parsing",
            )
        )
        prepared = self.prepared()
        spoken = "I am inspecting the parser's quoted fields."
        self.memory.save_narrative(
            WorkNarrative(
                prepared.subject_id,
                prepared.revision,
                NarrativeStatement(spoken, (prepared.latest_request_source,)),
                sources=prepared.sources,
            )
        )
        self.memory.record_speech(spoken, SpeechReason.ROUTINE)
        self.add(
            IntentEvidence(
                SessionIdentity(SessionSource.CODEX, "root"), "Check escaped quotes too"
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
                        SessionIdentity(SessionSource.CODEX, "root"), "Implement parser"
                    )
                )
                for index in range(count):
                    self.add(
                        ProgressEvidence(
                            SessionIdentity(SessionSource.CODEX, "root"),
                            f"Inspected case {index}. "
                            + "Each nested escaped quote needs a separate case. " * 25,
                        )
                    )
                initial = self.prepared(target_bytes=budget)
                reports = tuple(
                    key
                    for key, source in initial.sources.items()
                    if source["kind"] == "reported_progress"
                )
                self.assertEqual(len(reports), count)
                self.memory.save_narrative(
                    WorkNarrative(
                        initial.subject_id,
                        initial.revision,
                        NarrativeStatement(
                            "Implement parser", (initial.latest_request_source,)
                        ),
                        decisions=tuple(
                            NarrativeStatement(
                                "Keep escaped quote cases distinct.", reports[index::4]
                            )
                            for index in range(4)
                        ),
                        sources=initial.sources,
                    )
                )
                self.add(
                    ProgressEvidence(
                        SessionIdentity(SessionSource.CODEX, "root"),
                        "Critical regression in the latest parser case. "
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
                SessionIdentity(SessionSource.CODEX, "root"), "Improve parser"
            )
        )
        self.add(
            IntentEvidence(
                SessionIdentity(SessionSource.CODEX, "old"), "Audit escapes"
            ),
            session=SessionIdentity(SessionSource.CODEX, "old"),
            parent=SessionIdentity(SessionSource.CODEX, "root"),
        )
        self.memory.work.record_session(
            SessionObservation(
                SessionIdentity(SessionSource.CODEX, "native"),
                aliases=(SessionIdentity(SessionSource.CODEX, "old"),),
                reports=(
                    ProgressEvidence(
                        SessionIdentity(SessionSource.CODEX, "native"),
                        "Escape checks passed",
                    ),
                ),
            )
        )
        self.add(
            IntentEvidence(
                SessionIdentity(SessionSource.CODEX, "native"), "Audit cancellation"
            ),
            ProgressEvidence(
                SessionIdentity(SessionSource.CODEX, "native"),
                "Cancellation checks passed",
            ),
            session=SessionIdentity(SessionSource.CODEX, "native"),
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
                SessionIdentity(SessionSource.CODEX, "root"), "Improve parser"
            )
        )
        self.add(
            IntentEvidence(
                SessionIdentity(SessionSource.CODEX, "child"), "Audit escaped quotes"
            ),
            session=SessionIdentity(SessionSource.CODEX, "child"),
            parent=SessionIdentity(SessionSource.CODEX, "root"),
        )
        self.add(
            ExecutionEvidence(
                SessionIdentity(SessionSource.CODEX, "child"),
                "turn_error",
                ExecutionOutcome.ERROR,
                None,
                "Found one failing escaped quote case",
            ),
            session=SessionIdentity(SessionSource.CODEX, "child"),
            parent=SessionIdentity(SessionSource.CODEX, "root"),
        )
        sources = list(self.prepared().sources.values())
        report = next(source for source in sources if source["kind"] == "turn_ended")
        self.assertIn("Audit escaped quotes", str(report["scope"]))
        self.assertEqual(report["outcome"], "error")

    def test_related_task_without_an_assignment_keeps_its_result_scoped(self) -> None:
        self.add(
            IntentEvidence(
                SessionIdentity(SessionSource.CODEX, "root"), "Improve parser"
            )
        )
        self.add(
            ProgressEvidence(
                SessionIdentity(SessionSource.CODEX, "child"), "Escape checks passed"
            ),
            session=SessionIdentity(SessionSource.CODEX, "child"),
            parent=SessionIdentity(SessionSource.CODEX, "root"),
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
                SessionIdentity(SessionSource.CODEX, "root"), "Improve parser"
            )
        )
        self.add(
            IntentEvidence(
                SessionIdentity(SessionSource.CODEX, "child"), "current task"
            ),
            session=SessionIdentity(SessionSource.CODEX, "child"),
            parent=SessionIdentity(SessionSource.CODEX, "root"),
        )
        prepared = self.prepared()
        self.assertEqual(
            prepared.sources[prepared.latest_request_source]["text"], "Improve parser"
        )
        self.assertEqual(
            prepared.sources[prepared.latest_request_source]["origin"],
            session_data(SessionIdentity(SessionSource.CODEX, "root")),
        )
        child_request = next(
            source
            for source in prepared.sources.values()
            if source.get("text") == "current task"
        )
        self.assertEqual(child_request["scope"], "current task")

    def test_related_task_view_keeps_the_current_request_selected(self) -> None:
        root = SessionIdentity(SessionSource.CODEX, "root")
        story = WorkStory(
            root,
            "Improve parser",
            "fallback",
            requests=(
                EvidenceView(
                    EvidenceSource.USER,
                    EvidenceKind.INTENT,
                    IntentView("Improve parser"),
                ),
            ),
            sessions=(
                SessionContribution(
                    "Audit escapes",
                    session=SessionIdentity(SessionSource.CODEX, "coverage"),
                    requests=(
                        EvidenceView(
                            EvidenceSource.USER,
                            EvidenceKind.INTENT,
                            IntentView("Audit escapes"),
                        ),
                    ),
                ),
            ),
        )
        subject = WorkSubject(root, WorkSubjectKind.SESSION, 1000, "fallback", story)
        focus = session_snapshot(id=root)
        snapshot = capture_snapshot(sessions_snapshot(focus), work=(subject,))
        selection = AttentionSelection(
            subject, focus, (), SpeechReason.ROUTINE, SpeechSituation.ROUTINE
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
                SessionIdentity(SessionSource.CODEX, "root"), "Improve parser"
            ),
            IntentEvidence(SessionIdentity(SessionSource.CODEX, "root"), "Proceed"),
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
                SessionIdentity(SessionSource.CODEX, "root"), "Improve parser"
            )
        )
        original = self.memory.connected_work(sessions_snapshot())[0].story
        self.add(
            ExecutionEvidence(
                SessionIdentity(SessionSource.CODEX, "root"),
                "tool_start",
                None,
                "Bash",
                None,
            )
        )
        running = self.memory.connected_work(sessions_snapshot())[0].story
        self.assertEqual(semantic_revision(original), semantic_revision(running))
        self.assertEqual(
            semantic_revision(original),
            semantic_revision(
                replace(original, revision="git refreshed", artifacts=())
            ),
        )
        self.add(
            ExecutionEvidence(
                SessionIdentity(SessionSource.CODEX, "root"),
                "turn_done",
                ExecutionOutcome.SUCCESS,
                None,
                "Parser checked",
            )
        )
        ended = self.memory.connected_work(sessions_snapshot())[0].story
        self.assertNotEqual(semantic_revision(original), semantic_revision(ended))
