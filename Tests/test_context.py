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
                self.assertEqual(payload["focus_repository"]["pr"]["state"], "OPEN")
                self.assertIn(expected, self.cognition.system(payload)[1]["text"])

    def test_repository_evidence_fits_the_request_budget_with_large_check_names(self):
        self.snapshot["repositories"][0]["pr"]["checks"] = [("猫" * 2000, "FAILURE")] * 100
        payload = self.cognition.prompt(self.snapshot, "routine", None)
        request = self.cognition.bounded_request(payload)
        self.assertLessEqual(len(json.dumps(request, ensure_ascii=False).encode()), self.config.prompt_max_bytes)
        evidence = json.loads(request["messages"][0]["content"])["focus_repository"]
        self.assertEqual(evidence["local"]["unstaged"], 1)
        self.assertEqual(evidence["pr"]["checks"][0][1], "FAILURE")

    def test_expired_remarks_do_not_enter_repetition_context(self):
        now = [1000]
        self.memory.clock = lambda: now[0]
        self.cognition.clock = lambda: now[0]
        self.memory.record_speech("Old remark", "routine")
        now[0] += 1800
        self.memory.record_speech("Fresh remark", "routine")
        payload = self.cognition.prompt(self.snapshot, "routine", None)
        self.assertEqual([row["text"] for row in payload["repetition"]["recent_speech"]], ["Fresh remark"])
        self.assertNotIn("Old remark", json.dumps(payload))
        self.assertNotIn("Fresh remark", json.dumps(payload["memory"]))

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
