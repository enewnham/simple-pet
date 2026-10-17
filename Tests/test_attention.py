import random
from pathlib import Path
import unittest

from pet_daemon.attention import Attention, selected_sessions, selected_subject
from pet_daemon.config import Config
from pet_daemon.work import WorkSubject


class AttentionTests(unittest.TestCase):
    def setUp(self):
        self.now = 0
        self.config = Config(pet_dir=Path("."), observation_seconds=(30, 30))
        self.attention = Attention(self.config, lambda: self.now, random.Random(0))

    def test_rotates_among_eligible_sessions(self):
        snapshot = {"sessions": [
            {"id": "codex:a", "turn_active": True, "quiet_seconds": 100},
            {"id": "codex:b", "turn_active": True, "quiet_seconds": 50},
        ]}
        chosen = []
        for _ in range(4):
            session = self.attention.choose(snapshot, "check_in", {"codex:a", "codex:b"})
            chosen.append(session["id"])
            self.attention.scheduled(session["id"])
        self.assertEqual(chosen, ["codex:a", "codex:b", "codex:a", "codex:b"])

    def test_waiting_session_preserves_other_routine_attention(self):
        snapshot = {"sessions": [
            {"id": "codex:waiting", "turn_active": True, "waiting_for_input": True},
            {"id": "codex:working", "turn_active": True},
        ]}
        self.assertEqual(self.attention.choose(snapshot)["id"], "codex:working")
        self.assertEqual(self.attention.choose(snapshot, "click")["id"], "codex:waiting")

    def test_completion_uses_most_recent_completed_session(self):
        snapshot = {"sessions": [
            {"id": "codex:old", "turn_active": False, "quiet_seconds": 100},
            {"id": "codex:recent", "turn_active": False, "quiet_seconds": 5},
        ]}
        self.assertEqual(selected_sessions(snapshot, "completion")[0]["id"], "codex:recent")

    def test_explicit_target_keeps_its_completion_context(self):
        snapshot = {"trigger": {"target_session": "codex:done"}, "sessions": [
            {"id": "codex:working", "turn_active": True},
            {"id": "codex:done", "turn_active": False},
        ]}
        self.assertEqual(selected_sessions(snapshot, "completion")[0]["id"], "codex:done")

    def test_pending_failure_survives_lower_priority_prompt(self):
        self.attention.queue("failure", {"target_session": "codex:failed"})
        self.attention.queue("prompt", {"target_session": "codex:working"}, failed=True)
        self.assertEqual(self.attention.pending, ("failure", 1, {"target_session": "codex:failed"}))

    def test_cooldown_preserves_waiting_and_user_requests(self):
        self.attention.scheduled()
        self.assertFalse(self.attention.allowed("routine"))
        self.assertTrue(self.attention.allowed("click"))
        self.assertTrue(self.attention.allowed("check_in", {"episode": "waiting"}))
        self.now = self.config.behavior.automatic_speech_cooldown_seconds
        self.assertTrue(self.attention.allowed("routine"))

    def test_pending_repository_precedes_historical_goal(self):
        history = WorkSubject("codex:quiet", "session", 4, 100, 1, ())
        repository = WorkSubject("repo:branch", "repository", 2, 10, 1, ())
        self.assertEqual(selected_subject({"work": (history, repository)}), repository)

    def test_new_pending_evidence_precedes_older_repository(self):
        older = WorkSubject("repo:older", "repository", 2, 10, 1, ())
        newer = WorkSubject("repo:newer", "repository", 2, 20, 1, ())
        snapshot = {"work": (older, newer)}
        self.attention.scheduled(newer.id)
        self.assertEqual(self.attention.choose_work(snapshot), newer)

    def test_explicit_repository_target_keeps_its_subject(self):
        repository = WorkSubject("repo:branch", "repository", 2, 10, 1, ())
        waiting = WorkSubject("codex:waiting", "session", 0, 20, 1, ())
        snapshot = {"work": (waiting, repository), "trigger": {"target_repository": repository.id}}
        self.assertEqual(selected_subject(snapshot), repository)

    def test_equally_recent_subjects_receive_fair_attention(self):
        first = WorkSubject("repo:a", "repository", 2, 10, 1, ())
        second = WorkSubject("repo:b", "repository", 2, 10, 1, ())
        snapshot = {"work": (first, second)}
        self.assertEqual(self.attention.choose_work(snapshot), first)
        self.attention.scheduled(first.id)
        self.assertEqual(self.attention.choose_work(snapshot), second)


if __name__ == "__main__":
    unittest.main()
