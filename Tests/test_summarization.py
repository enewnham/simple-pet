import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from pet_daemon.cognition import Cognition
from pet_daemon.config import Config
from pet_daemon.memory import Memory
from pet_daemon.summarization import summary_request, summary_text


class SummarizationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = Config(pet_dir=Path(self.temp.name), summarize_first_request_only=False)
        self.memory = Memory(Path(self.temp.name) / "memory.sqlite3", "pet", clock=lambda: 1000)
        self.addCleanup(self.memory.close)
        self.context = Cognition(None, self.config, SimpleNamespace(pet="pet"), clock=lambda: 1000, memory=self.memory)
        self.focus = {"id": "codex:main", "first_prompt": "Plan: [ ] Implement parser. [ ] Verify parser.",
                      "recent_requests": ["Implement parser"], "session_context": "Parser work remains pending.",
                      "progress": "Parser implemented and tests passed", "outcome": "success", "turn_active": False}

    def event(self, kind, session="main", **values):
        self.memory.record_event({"harness": "codex", "session": session, "event": kind,
                                  "repo": "/repo", **values})

    def payload(self, request=None):
        request = request or summary_request(self.context, self.focus)
        return json.loads(request["messages"][0]["content"])

    def test_initial_checklist_and_later_completion_keep_their_provenance(self):
        self.event("turn_done", outcome="success", progress="Parser implemented and tests passed")
        request = summary_request(self.context, self.focus)
        payload = self.payload(request)
        self.assertIn("[ ] Implement parser", payload["initial_intent"])
        self.assertEqual(payload["historical_session_context"], "Parser work remains pending.")
        self.assertEqual(payload["current_report"]["outcome"], "success")
        self.assertIn("Parser implemented and tests passed", payload["reported_results"][0]["report"])
        self.assertIn("agent reported success", payload["reported_results"][0]["report"])
        self.assertEqual(payload["reported_results"][0]["age"], "less than a minute ago")
        self.assertNotIn("turn_done", json.dumps(payload))
        self.assertNotIn("reported_at", json.dumps(payload))
        self.assertIn("unchecked checklist", request["system"][0]["text"])
        self.assertEqual(request["output_config"]["format"]["schema"]["required"], ["session_context"])

    def test_next_request_retains_its_session_completion_after_later_tools(self):
        self.event("turn_done", outcome="success", progress="Parser delivered")
        for index in range(10):
            self.event("progress", progress=f"Follow-up inspection {index}")
        self.event("turn_done", session="other", outcome="success", progress="Unrelated migration delivered")
        self.focus.update(progress="Inspecting a follow-up", outcome=None, turn_active=True)
        payload = self.payload()
        self.assertEqual(len(payload["reported_results"]), 1)
        self.assertIn("Parser delivered", payload["reported_results"][0]["report"])
        self.assertNotIn("Unrelated migration", json.dumps(payload))
        self.assertEqual(payload["current_report"], {"progress": "Inspecting a follow-up", "outcome": None})

    def test_first_request_only_excludes_followups_history_and_results(self):
        self.config.summarize_first_request_only = True
        self.event("turn_done", outcome="success", progress="Parser delivered")
        payload = self.payload()
        self.assertEqual(payload, {"initial_intent": self.focus["first_prompt"]})

    def test_missing_results_preserve_an_unknown_outcome(self):
        self.focus.update(progress="", outcome=None, turn_active=True)
        payload = self.payload()
        self.assertEqual(payload["reported_results"], [])
        self.assertIsNone(payload["current_report"]["outcome"])
        self.assertEqual(payload["current_report"]["progress"], "")

    def test_budget_reduces_initial_history_before_recent_report(self):
        self.config.prompt_max_bytes = 4000
        self.focus.update(first_prompt="Implement 猫 parser. " * 500,
                          recent_requests=["Review 猫. " * 500] * 4,
                          session_context="Historical 猫 plan. " * 100,
                          progress="Current 猫 progress. " * 100)
        self.event("turn_done", outcome="success", progress="Parser delivered after verification")
        request = summary_request(self.context, self.focus)
        payload = self.payload(request)
        self.assertLessEqual(len(json.dumps(request, ensure_ascii=False).encode()), self.config.prompt_max_bytes)
        self.assertEqual(payload["historical_session_context"], "")
        self.assertLess(len(payload["initial_intent"]), 120)
        self.assertIn("Parser delivered after verification", payload["reported_results"][0]["report"])


class SummaryTextTests(unittest.TestCase):
    def test_keeps_complete_sentences_within_the_storage_limit(self):
        first = "Keep the parser API stable."
        second = "The agent reported that parser tests passed."
        text = first + "\n" + second + " " + "Additional details " * 30 + "."
        self.assertEqual(summary_text(text), first + " " + second)

    def test_preserves_short_summaries_without_truncation(self):
        self.assertEqual(summary_text("  Keep the parser API stable.\nTests passed. "),
                         "Keep the parser API stable. Tests passed.")

    def test_rejects_an_overlong_first_sentence(self):
        self.assertEqual(summary_text("The agent reported " + "parser details " * 40 + "."), "")

    def test_abbreviations_and_decimals_do_not_end_an_overlong_sentence(self):
        for beginning in ("Dr. Example selected version 1.2 of the parser ",
                          "Keep aliases, e.g. the parser names ",
                          "Keep aliases (e.g. the parser names) ",
                          "The U.S. parser configuration "):
            with self.subTest(beginning=beginning):
                self.assertEqual(summary_text(beginning + "details " * 60 + "."), "")

    def test_preserves_sentence_quotes_without_cutting_the_next_sentence(self):
        complete = 'The agent reported "Checks passed."'
        self.assertEqual(summary_text(complete + " " + "Details " * 60 + "."), complete)
