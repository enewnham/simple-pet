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
from pet_daemon.domain import ExecutionOutcome, SpeechReason, SpeechSituation
from pet_daemon.storage import Memory
from pet_daemon.work.models import ExecutionEvidence, IntentEvidence, ProgressEvidence
from pet_daemon.work.narrative import (
    NarrativeStatement,
    WorkNarrative,
    semantic_revision,
)
from pet_daemon.work.observations import SessionObservation
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
        session: str = "codex:root",
        parent: str | None = None,
    ) -> None:
        self.memory.work.record_session(
            SessionObservation(session, parent=parent, reports=reports)
        )

    def prepared(self, *, target_bytes: int = 12000) -> PreparedStory:
        focus = session_snapshot(id="codex:root")
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

    def test_latest_followup_keeps_original_intent_beyond_recent_requests(self) -> None:
        self.add(IntentEvidence("codex:root", "Implement quoted-field parsing"))
        for text in (
            "Keep quotes",
            "Also escapes",
            "Check nesting",
            "Test it",
            "Proceed",
        ):
            self.add(IntentEvidence("codex:root", text))
        prepared = self.prepared()
        payload = json.loads(prepared.request["messages"][0]["content"])
        self.assertIn("Implement quoted-field parsing", json.dumps(payload))
        self.assertEqual(
            prepared.sources[prepared.latest_request_source]["text"], "Proceed"
        )

    def test_speech_quotes_code_and_tables_do_not_become_work_evidence(self) -> None:
        self.memory.record_speech("Mail's in. Quiet now.", SpeechReason.ROUTINE)
        self.add(IntentEvidence("codex:root", "Review commentary"))
        self.add(
            ProgressEvidence(
                "codex:root",
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
        self.add(IntentEvidence("codex:root", "Implement the parser"))
        self.add(
            IntentEvidence("codex:root", "Keep the public parser interface compatible")
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
                    "codex:root",
                    f"Inspected case {index}. "
                    + "Each nested escaped quote needs a separate case. " * 10,
                )
            )
        self.add(IntentEvidence("codex:root", "Proceed"))
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
        self.add(IntentEvidence("codex:root", "Review quoted-field parsing"))
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
        self.add(IntentEvidence("codex:root", "Check escaped quotes too"))
        payload = json.loads(self.prepared().request["messages"][0]["content"])
        self.assertNotIn(spoken, json.dumps(payload))
        self.assertNotIn("earlier_context", payload)

    def test_latest_report_survives_a_budget_filled_by_prior_citations(self) -> None:
        for budget, count in ((12000, 8), (16000, 11)):
            with self.subTest(budget=budget):
                self.memory.clear()
                self.add(IntentEvidence("codex:root", "Implement parser"))
                for index in range(count):
                    self.add(
                        ProgressEvidence(
                            "codex:root",
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
                        "codex:root",
                        "Critical regression in the latest parser case. "
                        + "A newly found failure needs investigation. " * 30,
                    )
                )
                prepared = self.prepared(target_bytes=budget)
                payload = prepared.request["messages"][0]["content"]
                self.assertIn("Critical regression", payload)
                self.assertLessEqual(len(payload.encode()), budget)

    def test_child_alias_keeps_assignment_at_report_time(self) -> None:
        self.add(IntentEvidence("codex:root", "Improve parser"))
        self.add(
            IntentEvidence("codex:old", "Audit escapes"),
            session="codex:old",
            parent="codex:root",
        )
        self.memory.work.record_session(
            SessionObservation(
                "codex:native",
                aliases=("codex:old",),
                reports=(ProgressEvidence("codex:native", "Escape checks passed"),),
            )
        )
        self.add(
            IntentEvidence("codex:native", "Audit cancellation"),
            ProgressEvidence("codex:native", "Cancellation checks passed"),
            session="codex:native",
        )
        reports = {
            source["text"]: source["scope"]
            for source in self.prepared().sources.values()
            if source["kind"] == "reported_progress"
        }
        self.assertEqual(
            reports["Escape checks passed"], "child assignment: Audit escapes"
        )
        self.assertEqual(
            reports["Cancellation checks passed"],
            "child assignment: Audit cancellation",
        )

    def test_child_report_keeps_assignment_and_failed_outcome(self) -> None:
        self.add(IntentEvidence("codex:root", "Improve parser"))
        self.add(
            IntentEvidence("codex:child", "Audit escaped quotes"),
            session="codex:child",
            parent="codex:root",
        )
        self.add(
            ExecutionEvidence(
                "codex:child",
                "turn_error",
                ExecutionOutcome.ERROR,
                None,
                "Found one failing escaped quote case",
            ),
            session="codex:child",
            parent="codex:root",
        )
        sources = list(self.prepared().sources.values())
        report = next(source for source in sources if source["kind"] == "turn_ended")
        self.assertIn("Audit escaped quotes", str(report["scope"]))
        self.assertEqual(report["outcome"], "error")

    def test_validation_rejects_unknown_sources_and_stale_intent(self) -> None:
        self.add(
            IntentEvidence("codex:root", "Improve parser"),
            IntentEvidence("codex:root", "Proceed"),
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
        self.add(IntentEvidence("codex:root", "Improve parser"))
        original = self.memory.connected_work(sessions_snapshot())[0].story
        self.add(ExecutionEvidence("codex:root", "tool_start", None, "Bash", None))
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
                "codex:root",
                "turn_done",
                ExecutionOutcome.SUCCESS,
                None,
                "Parser checked",
            )
        )
        ended = self.memory.connected_work(sessions_snapshot())[0].story
        self.assertNotEqual(semantic_revision(original), semantic_revision(ended))
