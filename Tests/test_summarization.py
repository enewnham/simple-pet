from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from pet_daemon.commentary.cognition import Cognition
from pet_daemon.config import Config
from pet_daemon.work.history import project_event
from pet_daemon.storage import Memory
from pet_daemon.prompts import bounded, select_prompt


class SummarizationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = Config(pet_dir=Path(self.temp.name), summarize_first_request_only=False)
        self.memory = Memory(Path(self.temp.name) / "memory.sqlite3", "pet", clock=lambda: 1000)
        self.addCleanup(self.memory.close)
        self.context = Cognition(None, self.config, clock=lambda: 1000, memory=self.memory)
        self.focus = {"id": "codex:main", "first_prompt": "Plan: [ ] Implement parser. [ ] Verify parser.",
                      "recent_requests": ["Implement parser"], "session_context": "Parser work remains pending.",
                      "progress": "Parser implemented and tests passed", "outcome": "success", "turn_active": False}

    def event(self, kind, session="main", **values):
        self.memory.record_event(project_event({"harness": "codex", "session": session, "event": kind,
                                  "repo": "/repo", **values}))

    def payload(self, request=None):
        request = request or self.context.summary_request(self.focus)
        return json.loads(request["messages"][0]["content"])

    def test_initial_checklist_and_later_completion_keep_their_provenance(self):
        self.event("turn_done", outcome="success", progress="Parser implemented and tests passed")
        request = self.context.summary_request(self.focus)
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

    def test_tiny_target_preserves_initial_intent_latest_request_and_current_report(self):
        self.config.prompt_target_bytes = 1
        self.focus.update(first_prompt="Implement 猫 parser. " * 500,
                          recent_requests=["Earlier request", "Review 猫. " * 500],
                          progress="Current 猫 progress. " * 100)
        self.event("turn_done", outcome="success", progress="Parser delivered after verification")
        request = self.context.summary_request(self.focus)
        payload = self.payload(request)
        self.assertGreater(len(json.dumps(request, ensure_ascii=False).encode()), self.config.prompt_target_bytes)
        self.assertEqual(payload["initial_intent"], select_prompt(self.focus["first_prompt"]))
        self.assertEqual(payload["recent_requests"], [select_prompt(self.focus["recent_requests"][-1])])
        self.assertEqual(payload["current_report"], {"progress": bounded(self.focus["progress"], 800), "outcome": "success"})
        self.assertEqual(payload["historical_session_context"], "")
        self.assertEqual(payload["reported_results"], [])

    def test_whole_latest_report_can_cross_target_before_other_context_is_added(self):
        self.event("turn_done", outcome="success", progress="Earlier parser result")
        self.event("turn_done", outcome="success", progress="Latest parser verification " * 20)
        self.config.prompt_target_bytes = 1
        essential = self.context.summary_request(self.focus)
        self.config.prompt_target_bytes = len(json.dumps(essential, ensure_ascii=False).encode()) + 1
        request = self.context.summary_request(self.focus)
        payload = self.payload(request)
        self.assertGreater(len(json.dumps(request, ensure_ascii=False).encode()), self.config.prompt_target_bytes)
        self.assertEqual(len(payload["reported_results"]), 1)
        row = self.memory.recent(limit=1, session=self.focus["id"], event_types=("turn_done",))[0]
        self.assertEqual(payload["reported_results"][0]["report"],
                         bounded(row["summary"].removeprefix("turn_done").removeprefix(" | "), 320))
        self.assertEqual(payload["historical_session_context"], "")

    def test_optional_requests_retain_chronology_without_mutating_session_input(self):
        self.focus["recent_requests"] = ["Implement parser", "Keep aliases", "Verify compatibility"]
        before = deepcopy(self.focus)
        payload = self.payload()
        self.assertEqual(payload["recent_requests"], self.focus["recent_requests"])
        self.assertEqual(payload["historical_session_context"], self.focus["session_context"])
        self.assertEqual(self.focus, before)
