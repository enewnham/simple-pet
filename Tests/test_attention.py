import random
import unittest
from dataclasses import replace
from pathlib import Path

from pet_daemon.attention import Attention
from pet_daemon.config import Config
from pet_daemon.runtime import (
    CareTrigger,
    EpisodeTrigger,
    FailureSnapshot,
    PetContext,
    SessionContext,
    SessionTrigger,
)
from pet_daemon.sessions.episodes import Episode
from pet_daemon.storage.memory import HistoryRecord
from pet_daemon.work import WorkStory, WorkSubject
from Tests.fixtures import (
    capture_snapshot,
    require,
    session_snapshot,
    sessions_snapshot,
)


class AttentionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.now = 0
        self.config = Config(pet_dir=Path("."), observation_seconds=(30, 30))
        self.attention = Attention(self.config, lambda: self.now, random.Random(0))

    def snapshot(self, *sessions: SessionContext) -> PetContext:
        members = sessions
        subjects = tuple(
            WorkSubject(
                session.id,
                "session",
                session.changed_at,
                "one",
                WorkStory(session.id, "", "one"),
                (session.id,),
            )
            for session in members
        )
        return capture_snapshot(sessions_snapshot(*members), work=subjects)

    def test_rotates_within_a_tier_before_preferring_recency(self) -> None:
        snapshot = self.snapshot(
            session_snapshot(id="cli:older", turn_active=True, changed_at=10),
            session_snapshot(id="cli:newer", turn_active=True, changed_at=20),
        )
        chosen = []
        for _ in range(4):
            selected = self.attention.select(snapshot)
            chosen.append(selected.id)
            self.attention.scheduled(selected.id)
        self.assertEqual(chosen, ["cli:newer", "cli:older", "cli:newer", "cli:older"])

    def test_explicit_input_precedes_active_work(self) -> None:
        waiting = session_snapshot(
            id="cli:input", turn_active=True, waiting_for_input=True, waiting_seconds=0
        )
        snapshot = self.snapshot(
            waiting,
            session_snapshot(id="cli:active", turn_active=True),
        )
        self.assertEqual(self.attention.select(snapshot).id, "cli:input")

    def test_tool_failure_alone_does_not_outrank_an_active_session(self) -> None:
        snapshot = self.snapshot(
            session_snapshot(
                id="cli:failed",
                failure=FailureSnapshot(
                    activity="testing",
                    count=1,
                    goal="",
                    current_turn=True,
                    resolved=False,
                    seconds_ago=0,
                ),
                outcome="error",
                changed_at=20,
            ),
            session_snapshot(id="cli:active", turn_active=True, changed_at=1),
        )
        self.assertEqual(self.attention.select(snapshot).id, "cli:active")

    def test_event_target_keeps_its_focus_and_direction(self) -> None:
        snapshot = self.snapshot(
            session_snapshot(id="cli:finished", outcome="success"),
        )
        snapshot = replace(snapshot, trigger=SessionTrigger("cli:finished"))
        selected = self.attention.select(snapshot, "completion")
        self.assertEqual(require(selected.focus_session).id, "cli:finished")
        self.assertEqual(selected.situation, "completion")

    def test_manual_context_and_care_do_not_select_work(self) -> None:
        snapshot = self.snapshot(session_snapshot(id="cli:active", turn_active=True))
        explicit = self.attention.select(snapshot, context="Inspect the result")
        self.assertIsNone(explicit.subject)
        self.assertIsNone(explicit.focus_session)
        self.assertEqual(explicit.situation, "explicit")
        snapshot = replace(snapshot, trigger=CareTrigger("exercise"))
        self.assertEqual(
            self.attention.select(snapshot, "care").situation, "care_exercise"
        )

    def test_click_prioritizes_the_oldest_explicit_input_wait(self) -> None:
        snapshot = self.snapshot(
            session_snapshot(
                id="cli:recent",
                waiting_for_input=True,
                waiting_seconds=5,
                changed_at=20,
            ),
            session_snapshot(
                id="cli:old", waiting_for_input=True, waiting_seconds=30, changed_at=10
            ),
        )
        selected = self.attention.select(snapshot, "click")
        self.assertEqual(require(selected.focus_session).id, "cli:old")
        self.assertEqual(selected.situation, "click_waiting")

    def test_pending_failure_survives_a_lower_priority_prompt(self) -> None:
        self.attention.queue("failure", SessionTrigger("cli:failed"))
        self.attention.queue("prompt", SessionTrigger("cli:working"), failed=True)
        self.assertEqual(
            self.attention.pending, ("failure", 1, SessionTrigger("cli:failed"))
        )

    def test_cooldown_preserves_waiting_and_user_requests(self) -> None:
        self.attention.scheduled()
        self.assertFalse(self.attention.allowed("routine"))
        self.assertTrue(self.attention.allowed("click"))
        self.assertTrue(
            self.attention.allowed(
                "check_in",
                EpisodeTrigger(
                    Episode(
                        kind="waiting",
                        session="cli:parser",
                        generation=1,
                        activity="command",
                        serial=1,
                    )
                ),
            )
        )
        self.now = self.config.behavior.automatic_speech_cooldown_seconds
        self.assertTrue(self.attention.allowed("routine"))

    def test_considered_subjects_stay_quiet_after_restoring_history(self) -> None:
        snapshot = self.snapshot(session_snapshot(id="cli:parser", turn_active=True))
        selected = self.attention.select(snapshot)
        self.attention.scheduled(selected.id)
        self.attention.mark_considered(require(selected.subject))
        last_selected, revision = self.attention.history(require(selected.id))
        restored = Attention(self.config, lambda: self.now, random.Random(0))
        restored.restore(
            {
                require(selected.id): HistoryRecord(
                    last_selected=last_selected, considered_revision=revision
                )
            }
        )
        self.now = 60
        self.assertFalse(restored.is_new(require(selected.subject)))
        self.assertIsNone(restored.request(snapshot))

    def test_request_precedence_keeps_manual_and_click_ahead_of_due_events(
        self,
    ) -> None:
        self.attention.queue("prompt")
        snapshot = self.snapshot(session_snapshot(id="cli:active", turn_active=True))
        self.assertEqual(
            require(
                self.attention.request(
                    snapshot, manual=("Inspect parser", 4), clicked=True
                )
            ).reason,
            "explicit",
        )
        self.assertEqual(
            require(self.attention.request(snapshot, clicked=True)).reason, "click"
        )
        self.assertEqual(require(self.attention.request(snapshot)).reason, "prompt")

    def test_session_presence_without_work_does_not_request_commentary(self) -> None:
        snapshot = self.snapshot(session_snapshot(id="cli:present", turn_active=False))
        self.now = 30
        self.assertIsNone(self.attention.request(snapshot))

    def test_nap_pauses_automatic_requests_and_preserves_the_pending_trigger(
        self,
    ) -> None:
        snapshot = self.snapshot(session_snapshot(id="cli:parser", turn_active=True))
        snapshot = replace(snapshot, sleeping=True)
        self.now = 60
        self.attention.queue("prompt", SessionTrigger("cli:parser"))
        pending = self.attention.pending
        self.assertIsNone(
            self.attention.request(snapshot, check_in=("cli:parser",), care="exercise")
        )
        self.assertEqual(self.attention.pending, pending)
        self.assertEqual(
            require(
                self.attention.request(
                    snapshot, manual=("Inspect parser", 4), clicked=True
                )
            ).reason,
            "explicit",
        )
        self.assertEqual(
            require(self.attention.request(snapshot, clicked=True)).reason, "click"
        )
        self.assertEqual(self.attention.pending, pending)
        snapshot = replace(snapshot, sleeping=False)
        self.assertEqual(require(self.attention.request(snapshot)).reason, "prompt")

    def test_check_in_keeps_the_eligible_sessions(self) -> None:
        snapshot = self.snapshot(
            session_snapshot(id="cli:first", turn_active=True),
            session_snapshot(id="cli:second", turn_active=True),
        )
        request = self.attention.request(snapshot, check_in=("cli:second",))
        selected = self.attention.select(
            snapshot, require(request).reason, eligible=require(request).eligible
        )
        self.assertEqual(require(selected.focus_session).id, "cli:second")

    def test_pending_delivery_preserves_every_automatic_request(self) -> None:
        snapshot = self.snapshot(session_snapshot(id="cli:parser", turn_active=True))
        self.now = 60
        self.attention.queue("prompt", SessionTrigger("cli:parser"))
        pending = self.attention.pending
        next_observation = self.attention.next_observation
        self.assertIsNone(
            self.attention.request(
                snapshot,
                check_in=("cli:parser",),
                care="exercise",
                delivery_pending=True,
            )
        )
        self.assertEqual(self.attention.pending, pending)
        self.assertEqual(self.attention.next_observation, next_observation)
        self.assertEqual(
            require(
                self.attention.request(
                    snapshot, manual=("Inspect parser", 4), delivery_pending=True
                )
            ).reason,
            "explicit",
        )
        self.assertEqual(
            require(
                self.attention.request(snapshot, clicked=True, delivery_pending=True)
            ).reason,
            "click",
        )
        self.assertEqual(require(self.attention.request(snapshot)).reason, "prompt")

    def test_pending_delivery_preserves_due_observation_and_care(self) -> None:
        snapshot = self.snapshot(session_snapshot(id="cli:parser", turn_active=True))
        self.now = 60
        next_observation = self.attention.next_observation
        self.assertIsNone(self.attention.request(snapshot, delivery_pending=True))
        self.assertEqual(self.attention.next_observation, next_observation)
        self.assertEqual(require(self.attention.request(snapshot)).reason, "routine")
        idle = self.snapshot()
        self.assertIsNone(
            self.attention.request(idle, care="exercise", delivery_pending=True)
        )
        self.assertEqual(
            require(self.attention.request(idle, care="exercise")).reason, "care"
        )
