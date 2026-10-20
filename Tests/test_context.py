import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from pet_daemon.cognition import Cognition
from pet_daemon.config import Config
from pet_daemon.memory import Memory
from pet_daemon.sessions import Sessions
from pet_daemon.work import build_work


class ContextTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = Config(pet_dir=Path(self.temp.name))
        self.memory = Memory(Path(self.temp.name) / "memory.sqlite3", "pet", clock=lambda: 1000)
        self.addCleanup(self.memory.close)
        self.cognition = Cognition(None, self.config, SimpleNamespace(pet="pet"), clock=lambda: 1000, memory=self.memory)
        self.snapshot = {"sessions": [], "work": ()}

    def output(self):
        self.memory.work.session_intent("codex:author", "Make session tracking survive restarts")
        self.memory.work.session_event("codex:author", {"event": "turn_done", "outcome": "success",
                                                       "progress": "Implemented persistence; validation passed"})
        self.memory.work.observe_repository({
            "repository": "example.test/team/parser", "worktree": "/parser", "branch": "persistence",
            "sessions": ["codex:author"], "current": True, "local_checked_at": 1000, "github_checked_at": 1000,
            "local": {"head": "one", "unstaged": 7, "ahead": 3},
            "pr": {"url": "https://example.test/team/parser/pull/42", "state": "OPEN", "is_draft": True,
                   "title": "Persist coding session context", "head": "one", "check_revision": "one",
                   "head_repository": "example.test/team/parser", "head_branch": "persistence",
                   "review_requested": True, "checks": []},
        })
        self.snapshot["work"] = build_work(self.snapshot, self.memory.work, now=1000)

    def test_explicit_and_care_requests_keep_their_directions_with_opportunities(self):
        self.output()
        for reason, context, trigger, expected in (
            ("explicit", "Comment on the test run", None, "supplied context"),
            ("care", None, {"care": "exercise"}, "stretch break"),
        ):
            with self.subTest(reason=reason):
                self.snapshot["trigger"] = trigger
                payload = self.cognition.prompt(self.snapshot, reason, context)
                self.assertFalse(payload.get("focus_opportunity"))
                self.assertIn(expected, self.cognition.system(self.snapshot, reason, context)[1]["text"])

    def test_explicit_context_keeps_its_subject_during_quiet_requests(self):
        for reason in ("routine", "click", "care"):
            with self.subTest(reason=reason):
                payload = self.cognition.prompt({"sessions": []}, reason, "Inspect the parser result")
                self.assertEqual(payload["context"], "Inspect the parser result")
                self.assertIsNone(payload["focus_session"])
                self.assertIsNone(payload["focus_opportunity"])
                self.assertIn("supplied context", self.cognition.system({"sessions": []}, reason, "Inspect the parser result")[1]["text"])

    def test_opportunity_context_preserves_provenance_and_hides_tracking_identifiers(self):
        self.output()
        payload = self.cognition.prompt(self.snapshot, "routine")
        focus = payload["focus_opportunity"]
        self.assertEqual(focus["intent"], "Make session tracking survive restarts")
        self.assertEqual(focus["action"], "review_output")
        self.assertEqual(focus["artifact"]["subject"], "Persist coding session context")
        self.assertNotIn("revision", focus["artifact"])
        reports = [item for item in focus["connected_work"]["evidence"] if item["source"] == "agent"]
        self.assertTrue(reports)
        self.assertIn("validation passed", json.dumps(reports))
        encoded = json.dumps(focus)
        for internal in ("codex:author", '"unstaged"', '"ahead"', '"head"', '"revision"', '"number"'):
            self.assertNotIn(internal, encoded)
        self.assertIn("review_requested_from_you", encoded)
        self.assertIsNone(payload["focus_session"])
        self.assertIn("do not establish readiness", self.cognition.system(self.snapshot, "routine")[0]["text"])

    def test_required_opportunity_survives_a_small_request_target(self):
        self.output()
        self.config.prompt_target_bytes = 1
        payload = self.cognition.prompt(self.snapshot, "routine")
        request = self.cognition.commentary_request(payload, self.cognition.system(self.snapshot, "routine"))
        self.assertEqual(json.loads(request["messages"][0]["content"]), payload)
        self.assertEqual(payload["focus_opportunity"]["action"], "review_output")

    def test_speech_history_remains_local_and_expired_remarks_are_unavailable(self):
        now = [1000]
        self.memory.clock = lambda: now[0]
        self.cognition.clock = lambda: now[0]
        self.memory.record_speech("Old remark", "routine")
        now[0] += 1800
        self.memory.record_speech("Fresh remark", "routine")
        payload = self.cognition.prompt(self.snapshot, "routine")
        self.assertEqual([row["text"] for row in self.memory.speech()], ["Fresh remark"])
        self.assertNotIn("Old remark", json.dumps(payload))
        self.assertNotIn("Fresh remark", json.dumps(payload))

    def test_greetings_exclude_old_plans_without_an_opportunity(self):
        sessions = Sessions(lambda: 1000)
        sessions.handle({"event": "intent", "harness": "cli", "session": "one", "goal": "Plan: [ ] P5 [ ] P9"})
        sessions.handle({"event": "turn_done", "harness": "cli", "session": "one", "progress": "P0 through P9 complete"})
        self.snapshot.update(sessions.snapshot())
        self.snapshot["work"] = build_work(self.snapshot, self.memory.work, now=1000)
        payload = self.cognition.prompt(self.snapshot, "click")
        self.assertEqual(set(payload), {"activity", "sleeping", "time"})
        self.assertNotIn("P5", json.dumps(payload))

    def test_click_can_surface_an_opportunity_after_all_sessions_stop(self):
        self.output()
        payload = self.cognition.prompt(self.snapshot, "click")
        self.assertEqual(payload["focus_opportunity"]["action"], "review_output")
        self.assertIn("possible next action", self.cognition.system(self.snapshot, "click")[1]["text"])

    def test_prompt_uses_latest_request_without_unrelated_opportunities(self):
        self.output()
        self.snapshot["sessions"] = [{"id": "cli:one", "turn_active": True,
                                      "recent_requests": ["Old plan", "Explain the counter limits"]}]
        self.snapshot["trigger"] = {"target_session": "cli:one"}
        self.snapshot["work"] = build_work(self.snapshot, self.memory.work, now=1000)
        payload = self.cognition.prompt(self.snapshot, "prompt")
        self.assertEqual(payload["focus_session"]["recent_requests"], ["Explain the counter limits"])
        self.assertIsNone(payload["focus_opportunity"])
        self.assertNotIn("Persist coding session context", json.dumps(payload))

    def test_session_commentary_uses_summary_with_latest_requests_and_progress(self):
        snapshot = {"sessions": [{"id": "cli:one", "repo": "/parser", "turn_active": True,
                                  "recent_requests": ["Keep the public parser API stable"],
                                  "progress": "Checking parser compatibility",
                                  "session_context": "Implement the parser and preserve its public API. " * 30}]}
        for reason in ("prompt", "routine", "check_in"):
            with self.subTest(reason=reason):
                payload = self.cognition.prompt(snapshot, reason)
                focus = payload["focus_session"]
                self.assertTrue(focus["session_context"].startswith("Implement the parser"))
                self.assertLessEqual(len(focus["session_context"]), 400)
                self.assertEqual(focus["recent_requests"], ["Keep the public parser API stable"])
                if reason != "prompt":
                    self.assertEqual(focus["progress"], "Checking parser compatibility")
                self.assertIn("Later requests and current progress", self.cognition.system(snapshot, reason)[0]["text"])

    def test_small_target_preserves_summary_latest_request_and_current_progress(self):
        self.config.prompt_target_bytes = 1
        self.config.validate()
        snapshot = {"sessions": [{"id": "cli:one", "turn_active": True,
                                  "recent_requests": ["Explore parser options", "Keep the public API stable"],
                                  "progress": "Checking parser compatibility",
                                  "session_context": "Historical parser intent."}]}
        payload = self.cognition.prompt(snapshot, "routine")
        original = json.dumps(payload)
        request = self.cognition.commentary_request(payload, self.cognition.system(snapshot, "routine"))
        focus = json.loads(request["messages"][0]["content"])["focus_session"]
        self.assertEqual(focus["recent_requests"], ["Keep the public API stable"])
        self.assertEqual(focus["progress"], "Checking parser compatibility")
        self.assertEqual(focus["session_context"], "Historical parser intent.")
        self.assertEqual(json.dumps(payload), original)

    def test_complete_request_can_cross_target_before_other_sessions_are_added(self):
        snapshot = {"sessions": [
            {"id": "cli:focus", "turn_active": True,
             "recent_requests": ["Explore parser options", "Keep the public API stable"],
             "session_context": "Implement the parser while preserving its public API."},
            {"id": "cli:other", "turn_active": True, "recent_requests": ["Review the renderer"]},
        ]}
        payload = self.cognition.prompt(snapshot, "routine")
        original = json.dumps(payload)
        system = self.cognition.system(snapshot, "routine")
        self.config.prompt_target_bytes = 1
        essential = self.cognition.commentary_request(payload, system)
        self.config.prompt_target_bytes = len(json.dumps(essential, ensure_ascii=False).encode()) + 1
        request = self.cognition.commentary_request(payload, system)
        selected = json.loads(request["messages"][0]["content"])
        self.assertEqual(selected["focus_session"]["session_context"], payload["focus_session"]["session_context"])
        self.assertEqual(selected["focus_session"]["recent_requests"], ["Explore parser options", "Keep the public API stable"])
        self.assertEqual(selected["other_sessions"], [])
        self.assertEqual(selected["omitted_sessions"], {"active": 1, "total": 1})
        self.assertGreater(len(json.dumps(request, ensure_ascii=False).encode()), self.config.prompt_target_bytes)
        self.assertEqual(json.dumps(payload), original)

    def test_available_target_keeps_complete_context_in_chronological_order(self):
        snapshot = {"sessions": [
            {"id": "cli:focus", "turn_active": True,
             "recent_requests": ["Explore parser options", "Keep the public API stable"],
             "session_context": "Implement the parser while preserving its public API."},
            {"id": "cli:other", "turn_active": True, "recent_requests": ["Review the renderer"]},
        ]}
        payload = self.cognition.prompt(snapshot, "routine")
        request = self.cognition.commentary_request(payload, self.cognition.system(snapshot, "routine"))
        self.assertEqual(json.loads(request["messages"][0]["content"]), payload)

    def test_polling_a_session_does_not_change_its_work_recency(self):
        now = [1.0]
        sessions = Sessions(lambda: now[0])
        sessions.handle({"event": "intent", "harness": "cli", "session": "one", "goal": "Review parser"})
        before = build_work(sessions.snapshot(), self.memory.work, now=1000)[0]
        now[0] += 0.4
        after = build_work(sessions.snapshot(), self.memory.work, now=1000.4)[0]
        self.assertEqual(before.changed_at, after.changed_at)

    def test_quiet_activity_transition_invalidates_an_observation(self):
        now = [1.0]
        sessions = Sessions(lambda: now[0])
        sessions.handle({"event": "intent", "harness": "cli", "session": "one", "goal": "Review parser"})
        before = sessions.observation_context()
        now[0] += 300
        self.assertNotEqual(sessions.observation_context(), before)
        self.assertFalse(sessions.working)
        self.assertEqual(sessions.check_in_due(300), [])
