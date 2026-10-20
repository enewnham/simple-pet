import random
from pathlib import Path
import unittest

from pet_daemon.attention import Attention
from pet_daemon.config import Config
from pet_daemon.work import Opportunity, WorkSubject, WorkView


class AttentionTests(unittest.TestCase):
    def setUp(self):
        self.now = 0
        self.config = Config(pet_dir=Path("."), observation_seconds=(30, 30))
        self.attention = Attention(self.config, lambda: self.now, random.Random(0))

    def snapshot(self, *sessions, actions=()):
        subjects = tuple(
            WorkSubject(
                session["id"],
                "session",
                session.get("changed_at", 0),
                "one",
                WorkView(session["id"], "", "one"),
                (session["id"],),
            )
            for session in sessions
        )
        for action in actions:
            item = Opportunity(
                action["id"],
                1,
                action["kind"],
                action["id"],
                "",
                {},
                (),
                (),
                "",
                action.get("changed_at", 0),
                (),
            )
            subjects += (
                WorkSubject(
                    item.id,
                    "work",
                    item.changed_at,
                    "one",
                    WorkView(item.id, "", "one", opportunities=(item,)),
                ),
            )
        return {"sessions": list(sessions), "work": subjects}

    def test_rotates_within_a_tier_before_preferring_recency(self):
        snapshot = self.snapshot(
            {"id": "cli:older", "turn_active": True, "changed_at": 10},
            {"id": "cli:newer", "turn_active": True, "changed_at": 20},
        )
        chosen = []
        for _ in range(4):
            selected = self.attention.select(snapshot)
            chosen.append(selected.id)
            self.attention.scheduled(selected.id)
        self.assertEqual(chosen, ["cli:newer", "cli:older", "cli:newer", "cli:older"])

    def test_explicit_input_precedes_current_checks_and_active_work(self):
        waiting = {"id": "cli:input", "turn_active": True, "waiting_for_input": True}
        snapshot = self.snapshot(
            waiting,
            {"id": "cli:active", "turn_active": True},
            actions=({"id": "checks", "kind": "resolve_checks"},),
        )
        self.assertEqual(self.attention.select(snapshot).id, "cli:input")

    def test_checks_precede_review_and_active_work(self):
        snapshot = self.snapshot(
            {"id": "cli:active", "turn_active": True},
            actions=(
                {"id": "review", "kind": "review_output", "changed_at": 10},
                {"id": "checks", "kind": "resolve_checks", "changed_at": 1},
            ),
        )
        self.assertEqual(self.attention.select(snapshot).action.id, "checks")

    def test_tool_failure_alone_does_not_outrank_an_active_session(self):
        snapshot = self.snapshot(
            {
                "id": "cli:failed",
                "failure": {"current_turn": True, "resolved": False},
                "outcome": "error",
                "changed_at": 20,
            },
            {"id": "cli:active", "turn_active": True, "changed_at": 1},
        )
        self.assertEqual(self.attention.select(snapshot).id, "cli:active")

    def test_event_target_keeps_its_focus_and_direction(self):
        snapshot = self.snapshot(
            {"id": "cli:finished", "outcome": "success"},
            actions=({"id": "checks", "kind": "resolve_checks"},),
        )
        snapshot["trigger"] = {"target_session": "cli:finished"}
        selected = self.attention.select(snapshot, "completion")
        self.assertEqual(selected.focus_session["id"], "cli:finished")
        self.assertIsNone(selected.action)
        self.assertEqual(selected.situation, "completion")

    def test_manual_context_and_care_do_not_select_work(self):
        snapshot = self.snapshot({"id": "cli:active", "turn_active": True})
        explicit = self.attention.select(snapshot, context="Inspect the result")
        self.assertIsNone(explicit.subject)
        self.assertIsNone(explicit.focus_session)
        self.assertEqual(explicit.situation, "explicit")
        snapshot["trigger"] = {"care": "exercise"}
        self.assertEqual(
            self.attention.select(snapshot, "care").situation, "care_exercise"
        )

    def test_click_prioritizes_the_oldest_explicit_input_wait(self):
        snapshot = self.snapshot(
            {
                "id": "cli:recent",
                "waiting_for_input": True,
                "waiting_seconds": 5,
                "changed_at": 20,
            },
            {
                "id": "cli:old",
                "waiting_for_input": True,
                "waiting_seconds": 30,
                "changed_at": 10,
            },
        )
        selected = self.attention.select(snapshot, "click")
        self.assertEqual(selected.focus_session["id"], "cli:old")
        self.assertEqual(selected.situation, "click_waiting")

    def test_pending_failure_survives_a_lower_priority_prompt(self):
        self.attention.queue("failure", {"target_session": "cli:failed"})
        self.attention.queue("prompt", {"target_session": "cli:working"}, failed=True)
        self.assertEqual(
            self.attention.pending, ("failure", 1, {"target_session": "cli:failed"})
        )

    def test_cooldown_preserves_waiting_and_user_requests(self):
        self.attention.scheduled()
        self.assertFalse(self.attention.allowed("routine"))
        self.assertTrue(self.attention.allowed("click"))
        self.assertTrue(self.attention.allowed("check_in", {"episode": "waiting"}))
        self.now = self.config.behavior.automatic_speech_cooldown_seconds
        self.assertTrue(self.attention.allowed("routine"))

    def test_considered_subjects_stay_quiet_after_restoring_history(self):
        snapshot = self.snapshot({"id": "cli:parser", "turn_active": True})
        selected = self.attention.select(snapshot)
        self.attention.scheduled(selected.id)
        self.attention.mark_considered(selected.subject)
        last_selected, revision = self.attention.history(selected.id)
        restored = Attention(self.config, lambda: self.now, random.Random(0))
        restored.restore(
            {
                selected.id: {
                    "last_selected": last_selected,
                    "considered_revision": revision,
                }
            }
        )
        self.now = 60
        self.assertFalse(restored.is_new(selected.subject))
        self.assertIsNone(restored.request(snapshot))

    def test_request_precedence_keeps_manual_and_click_ahead_of_due_events(self):
        self.attention.queue("prompt")
        snapshot = self.snapshot({"id": "cli:active", "turn_active": True})
        self.assertEqual(
            self.attention.request(
                snapshot, manual=("Inspect parser", 4), clicked=True
            ).reason,
            "explicit",
        )
        self.assertEqual(self.attention.request(snapshot, clicked=True).reason, "click")
        self.assertEqual(self.attention.request(snapshot).reason, "prompt")

    def test_session_presence_without_work_does_not_request_commentary(self):
        snapshot = self.snapshot({"id": "cli:present", "turn_active": False})
        self.now = 30
        self.assertIsNone(self.attention.request(snapshot))

    def test_nap_pauses_automatic_requests_and_preserves_the_pending_trigger(self):
        snapshot = self.snapshot({"id": "cli:parser", "turn_active": True})
        snapshot["sleeping"] = True
        self.now = 60
        self.attention.queue("prompt", {"target_session": "cli:parser"})
        pending = self.attention.pending
        self.assertIsNone(
            self.attention.request(snapshot, check_in=("cli:parser",), care="exercise")
        )
        self.assertEqual(self.attention.pending, pending)
        self.assertEqual(
            self.attention.request(
                snapshot, manual=("Inspect parser", 4), clicked=True
            ).reason,
            "explicit",
        )
        self.assertEqual(self.attention.request(snapshot, clicked=True).reason, "click")
        self.assertEqual(self.attention.pending, pending)
        snapshot["sleeping"] = False
        self.assertEqual(self.attention.request(snapshot).reason, "prompt")

    def test_check_in_keeps_the_eligible_sessions(self):
        snapshot = self.snapshot(
            {"id": "cli:first", "turn_active": True},
            {"id": "cli:second", "turn_active": True},
        )
        request = self.attention.request(snapshot, check_in=("cli:second",))
        selected = self.attention.select(
            snapshot, request.reason, eligible=request.eligible
        )
        self.assertEqual(selected.focus_session["id"], "cli:second")
