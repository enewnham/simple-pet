from dataclasses import asdict
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from pet_daemon.cognition import Cognition
from pet_daemon.config import Config
from pet_daemon.memory import Memory
from pet_daemon.repositories.git import GitObservation
from pet_daemon.repositories.github import PullRequest
from pet_daemon.repositories.tracker import RepositoryObservation
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
        local = GitObservation("/repo", "feature", "abc", 0, 1, 0, 0, "origin/feature", 0, 0)
        pr = PullRequest(7, "https://example.test/pr/7", "OPEN", None, "REVIEW_REQUIRED", (("ci", "SUCCESS"),))
        self.repository = RepositoryObservation("/repo:feature", "/repo", "feature", local=local, pr=pr, revision=2, changed_at=990)
        self.snapshot = {"sessions": [], "repositories": [{**asdict(self.repository), "pending": True}]}
        self.snapshot["work"] = build_work(self.snapshot, (self.repository,), now=1000)

    def test_explicit_and_care_requests_keep_their_directions_with_pending_work(self):
        for reason, context, trigger, expected in (
            ("explicit", "Comment on the test run", None, "supplied context"),
            ("care", None, {"care": "exercise"}, "stretch break"),
        ):
            with self.subTest(reason=reason):
                self.snapshot["trigger"] = trigger
                payload = self.cognition.prompt(self.snapshot, reason, context)
                if reason == "care":
                    self.assertNotIn("focus_repository", payload)
                else:
                    self.assertIsNone(payload["focus_repository"])
                    self.assertIsNone(payload["focus_session"])
                    self.assertEqual(payload["context"], context)
                self.assertIn(expected, self.cognition.system(self.snapshot, reason, context)[1]["text"])

    def test_explicit_context_keeps_its_subject_during_quiet_requests(self):
        for reason in ("routine", "click", "care"):
            with self.subTest(reason=reason):
                payload = self.cognition.prompt({"sessions": []}, reason, "Inspect the parser result")
                self.assertEqual(payload["context"], "Inspect the parser result")
                self.assertIsNone(payload["focus_session"])
                self.assertIsNone(payload["focus_repository"])
                direction = self.cognition.system({"sessions": []}, reason, "Inspect the parser result")[1]["text"]
                self.assertIn("supplied context", direction)

    def test_repository_evidence_fits_the_request_budget_with_large_check_names(self):
        self.snapshot["repositories"][0]["pr"]["checks"] = [("猫" * 2000, "FAILURE")] * 100
        payload = self.cognition.prompt(self.snapshot, "routine", None)
        request = self.cognition.bounded_request(payload, self.cognition.system(self.snapshot, "routine"))
        self.assertLessEqual(len(json.dumps(request, ensure_ascii=False).encode()), self.config.prompt_max_bytes)
        evidence = json.loads(request["messages"][0]["content"])["focus_repository"]
        self.assertEqual(evidence["local"]["unstaged"], 1)
        self.assertEqual(evidence["pr"]["checks"][0]["result"], "FAILURE")

    def test_speech_history_remains_local_and_expired_remarks_are_unavailable(self):
        now = [1000]
        self.memory.clock = lambda: now[0]
        self.cognition.clock = lambda: now[0]
        self.memory.record_speech("Old remark", "routine")
        now[0] += 1800
        self.memory.record_speech("Fresh remark", "routine")
        payload = self.cognition.prompt(self.snapshot, "routine", None)
        self.assertEqual([row["text"] for row in self.memory.speech()], ["Fresh remark"])
        self.assertNotIn("repetition", payload)
        self.assertNotIn("Old remark", json.dumps(payload))
        self.assertNotIn("Fresh remark", json.dumps(payload))

    def test_greetings_exclude_stale_plans_and_pending_repositories(self):
        sessions = Sessions(lambda: 1000)
        sessions.handle({"event": "intent", "harness": "cli", "session": "one",
                         "repo": "/repo", "goal": "Plan: [ ] P5 [ ] P9"})
        sessions.handle({"event": "turn_done", "harness": "cli", "session": "one",
                         "progress": "P0 through P9 complete"})
        self.snapshot.update(sessions.snapshot())
        self.snapshot["work"] = build_work(self.snapshot, (self.repository,), now=1000)
        self.memory.save_context("cli:one", "Plan: [ ] P5 [ ] P9", "P5 through P9 pending", [], 1, True)
        payload = self.cognition.prompt(self.snapshot, "click", None)
        self.assertEqual(set(payload), {"activity", "sleeping", "time"})
        self.assertNotIn("P5", json.dumps(payload))
        self.assertIn("casual greeting", self.cognition.system(self.snapshot, "click")[1]["text"])

    def test_repository_projection_keeps_facts_and_uncertainty(self):
        item = self.snapshot["repositories"][0]
        item.update(local_checked_at=700, github_checked_at=400, github_error="Command failed")
        projected = self.cognition.repository_record(item)
        self.assertEqual(projected["project"], "repo")
        self.assertEqual(projected["branch"], "feature")
        self.assertEqual(projected["local"]["unstaged"], 1)
        self.assertEqual(projected["pr"]["state"], "OPEN")
        self.assertEqual(projected["github_observed"], "10 minutes ago")
        self.assertTrue(projected["uncertainty"])
        for name in ("head", "subject", "worktree", "revision", "changed_at", "local_checked_at", "github_checked_at"):
            self.assertNotIn(name, projected)
        self.assertNotIn("url", projected["pr"])
        self.assertNotIn("number", projected["pr"])

    def test_repository_commentary_excludes_unrelated_history_and_duplicate_evidence(self):
        self.memory.record_event({"harness": "cli", "session": "one", "event": "progress",
                                  "repo": "/repo", "progress": "Parser delivered"})
        self.memory.record_event({"harness": "cli", "session": "other", "event": "progress",
                                  "repo": "/unrelated", "progress": "Unrelated plan still pending"})
        payload = self.cognition.prompt(self.snapshot, "routine", None)
        self.assertEqual(payload["focus_repository"]["project"], "repo")
        self.assertEqual(payload["repository_work"], [])
        self.assertEqual(payload["memory"]["recent_events"], [])
        self.assertIsNone(payload["focus_session"])
        self.assertEqual(payload["other_sessions"], [])
        self.assertNotIn("Parser delivered", json.dumps(payload))
        self.assertNotIn("Unrelated plan", json.dumps(payload))

    def test_missing_pull_request_preserves_query_uncertainty(self):
        item = {**self.snapshot["repositories"][0], "pr": None, "github_checked_at": 900}
        self.assertEqual(self.cognition.repository_record(item)["pr_status"], "none found")
        item["github_error"] = "Query unavailable"
        self.assertEqual(self.cognition.repository_record(item)["pr_status"], "unknown")
        item.update(github_error=None, github_checked_at=0)
        self.assertEqual(self.cognition.repository_record(item)["pr_status"], "unknown")

    def test_small_budget_preserves_focused_repository_evidence_without_supporting_facts(self):
        self.config.prompt_max_bytes = 8000
        self.snapshot["repositories"][0]["pr"]["checks"] = [("check" * 23, "FAILURE")] * 8
        payload = self.cognition.prompt(self.snapshot, "routine", None)
        self.assertEqual(payload["repository_work"], [])
        self.assertEqual(payload["memory"]["recent_events"], [])
        self.assertIsNone(payload["focus_session"])
        request = self.cognition.bounded_request(payload, self.cognition.system(self.snapshot, "routine"))
        self.assertLessEqual(len(json.dumps(request, ensure_ascii=False).encode()), 8000)
        evidence = json.loads(request["messages"][0]["content"])
        self.assertEqual(evidence["repository_work"], [])
        self.assertEqual(evidence["focus_repository"]["pr"]["checks"][0]["result"], "FAILURE")

    def test_prompt_acknowledges_only_the_latest_request_without_old_progress_or_repository_facts(self):
        sessions = Sessions(lambda: 1000)
        sessions.handle({"event": "intent", "harness": "cli", "session": "one", "repo": "/repo", "goal": "Old unchecked deployment plan"})
        sessions.handle({"event": "progress", "harness": "cli", "session": "one", "progress": "Old progress waiting for deployment"})
        sessions.handle({"event": "intent", "harness": "cli", "session": "one", "repo": "/repo", "goal": "Explain the counter limits"})
        sessions.handle({"event": "progress", "harness": "cli", "session": "one", "progress": "Old progress waiting for deployment"})
        sessions.handle({"event": "intent", "harness": "cli", "session": "other", "repo": "/elsewhere", "goal": "Unrelated billing review"})
        self.snapshot.update(sessions.snapshot())
        self.snapshot["work"] = build_work(self.snapshot, (self.repository,), now=1000)
        self.snapshot["trigger"] = {"target_session": "cli:one"}
        self.memory.record_event({"event": "progress", "harness": "cli", "session": "one", "repo": "/repo", "progress": "Historical parser result"})
        payload = self.cognition.prompt(self.snapshot, "prompt", None)
        self.assertEqual(payload["focus_session"]["recent_requests"], ["Explain the counter limits"])
        self.assertNotIn("progress", payload["focus_session"])
        self.assertEqual(payload["other_sessions"], [])
        self.assertIsNone(payload["focus_repository"])
        self.assertEqual(payload["repository_work"], [])
        self.assertEqual(payload["memory"]["recent_events"], [])
        for stale in ("Old unchecked", "Old progress", "Unrelated billing", "Historical parser", "REVIEW_REQUIRED", "feature"):
            self.assertNotIn(stale, json.dumps(payload))
        self.assertIn("latest request", self.cognition.system(self.snapshot, "prompt")[1]["text"])

    def test_session_situations_exclude_repository_evidence_and_historical_reports(self):
        sessions = Sessions(lambda: 1000)
        sessions.handle({"event": "intent", "harness": "cli", "session": "one", "repo": "/repo", "goal": "Review parser"})
        self.snapshot.update(sessions.snapshot())
        self.snapshot["work"] = build_work(self.snapshot, (self.repository,), now=1000)
        self.snapshot["trigger"] = {"target_session": "cli:one"}
        self.memory.record_event({"event": "progress", "harness": "cli", "session": "one", "repo": "/repo", "progress": "Historical delivery claim"})
        for reason in ("routine", "completion", "failure", "check_in", "click"):
            with self.subTest(reason=reason):
                payload = self.cognition.prompt(self.snapshot, reason, None)
                self.assertEqual(payload["focus_session"]["recent_requests"], ["Review parser"])
                self.assertIsNone(payload["focus_repository"])
                self.assertEqual(payload["repository_work"], [])
                self.assertEqual(payload["memory"]["recent_events"], [])
                self.assertNotIn("Historical delivery claim", json.dumps(payload))
                self.assertNotIn("REVIEW_REQUIRED", json.dumps(payload))

    def test_polling_a_session_does_not_change_its_work_recency(self):
        now = [1.0]
        sessions = Sessions(lambda: now[0])
        sessions.handle({"event": "intent", "harness": "cli", "session": "one", "goal": "Review parser"})
        before = build_work(sessions.snapshot(), (), now=1000)[0]
        now[0] += 0.4
        after = build_work(sessions.snapshot(), (), now=1000.4)[0]
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
