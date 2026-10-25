import random
import unittest
from pathlib import Path

from pet_daemon.attention import Attention
from pet_daemon.config import Config
from pet_daemon.domain import (
    Activity,
    ArtifactKind,
    CareKind,
    EvidenceKind,
    EvidenceSource,
    ExecutionOutcome,
    ProcessStatus,
    SessionSource,
    SpeechReason,
    WorkSubjectKind,
)
from pet_daemon.identity import SessionIdentity
from pet_daemon.runtime import (
    ActiveTool,
    AgentSessionStatus,
    CareTrigger,
    FailureSnapshot,
    PetContext,
    ReactionTrigger,
    SessionTrigger,
)
from pet_daemon.sessions.reactions import ReactionKind, SessionReaction
from pet_daemon.storage.memory import SourceAttention
from pet_daemon.work import WorkStory, WorkSubject
from pet_daemon.work.evidence import CommentaryEvidence, UserIntent
from pet_daemon.work.narrative import SelectedMemory
from pet_daemon.work.shared_work import WorkUpdate
from pet_daemon.work.stories import ArtifactContext, SessionContribution
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
        self.attention = Attention(
            self.config, lambda: self.now, random.Random(0), wall_clock=lambda: self.now
        )

    def snapshot(self, *sessions: AgentSessionStatus) -> PetContext:
        members = sessions
        subjects = tuple(
            WorkSubject(
                id=session.id,
                kind=WorkSubjectKind.SESSION,
                changed_at=session.changed_at,
                revision="one",
                story=WorkStory(id=session.id, intent="", revision="one"),
                sessions=(session.id,),
            )
            for session in members
        )
        return capture_snapshot(sessions_snapshot(*members), work=subjects)

    def update_snapshot(
        self, *sessions: AgentSessionStatus, updates: tuple[WorkUpdate, ...]
    ) -> PetContext:
        snapshot = self.connected_snapshot(*sessions)
        return snapshot.model_copy(
            update={"work": (snapshot.work[0].model_copy(update={"updates": updates}),)}
        )

    def test_completed_update_expires_at_its_original_deadline_without_runtime(
        self,
    ) -> None:
        update = WorkUpdate(
            session=SessionIdentity(source=SessionSource.COMMAND, native_id="root"),
            revision="result",
            observed_at=10,
            ended_at=10,
        )
        snapshot = self.update_snapshot(
            session_snapshot(
                id=SessionIdentity(source=SessionSource.COMMAND, native_id="root")
            ),
            updates=(update,),
        )
        snapshot = snapshot.model_copy(update={"sessions": ()})
        self.now = 309
        self.assertTrue(self.attention.is_routine_eligible(snapshot.work[0], snapshot))
        selected = self.attention.select(snapshot)
        self.assertEqual(selected.update, update)
        self.assertTrue(self.attention.is_selection_eligible(selected, snapshot))
        self.now = 310
        self.assertFalse(self.attention.is_routine_eligible(snapshot.work[0], snapshot))
        self.assertFalse(self.attention.is_selection_eligible(selected, snapshot))

    def test_considered_related_update_remains_available_as_background(self) -> None:
        root = SessionIdentity(source=SessionSource.COMMAND, native_id="root")
        peer = SessionIdentity(source=SessionSource.COMMAND, native_id="peer")
        own_update = WorkUpdate(session=root, revision="own", observed_at=0)
        peer_update = WorkUpdate(session=peer, revision="peer", observed_at=0)
        snapshot = self.snapshot(session_snapshot(id=root, turn_active=True))
        subject = snapshot.work[0].model_copy(
            update={
                "updates": (own_update,),
                "story": snapshot.work[0].story.model_copy(
                    update={
                        "sessions": (
                            SessionContribution(scope="Check storage", session=peer),
                        )
                    }
                ),
            }
        )
        snapshot = snapshot.model_copy(update={"work": (subject,)})
        self.attention.mark_considered(None, peer_update)
        selected = self.attention.select(snapshot)
        self.assertEqual(
            [item.session for item in require(selected.subject).story.sessions], [peer]
        )
        refreshed = selected.refresh(snapshot)
        self.assertEqual(
            require(refreshed.subject).story.sessions,
            require(selected.subject).story.sessions,
        )

    def test_click_consideration_keeps_pending_member_updates_eligible(self) -> None:
        update = WorkUpdate(
            session=SessionIdentity(source=SessionSource.COMMAND, native_id="root"),
            revision="result",
            observed_at=10,
            ended_at=10,
        )
        snapshot = self.update_snapshot(
            session_snapshot(
                id=SessionIdentity(source=SessionSource.COMMAND, native_id="root")
            ),
            updates=(update,),
        )
        self.attention.mark_considered(snapshot.work[0])
        self.assertTrue(self.attention.is_routine_eligible(snapshot.work[0], snapshot))
        self.attention.mark_considered(snapshot.work[0], update)
        self.assertFalse(self.attention.is_routine_eligible(snapshot.work[0], snapshot))

    def test_consumed_result_follows_a_session_alias(self) -> None:
        old = WorkUpdate(
            session=SessionIdentity(source=SessionSource.COMMAND, native_id="old"),
            revision="result",
            observed_at=10,
            ended_at=10,
        )
        subject = self.snapshot(
            session_snapshot(
                id=SessionIdentity(source=SessionSource.COMMAND, native_id="old")
            )
        ).work[0]
        self.attention.mark_considered(subject, old)
        renamed = old.model_copy(
            update={
                "session": SessionIdentity(
                    source=SessionSource.COMMAND, native_id="root"
                ),
                "aliases": (
                    SessionIdentity(source=SessionSource.COMMAND, native_id="old"),
                ),
            }
        )
        snapshot = self.update_snapshot(
            session_snapshot(
                id=SessionIdentity(source=SessionSource.COMMAND, native_id="root")
            ),
            updates=(renamed,),
        )
        self.assertFalse(self.attention.is_routine_eligible(snapshot.work[0], snapshot))

    def test_active_update_uses_owner_or_recent_hooks_beyond_five_minutes(self) -> None:
        update = WorkUpdate(
            session=SessionIdentity(source=SessionSource.COMMAND, native_id="root"),
            revision="progress",
            observed_at=0,
        )
        session = session_snapshot(
            id=SessionIdentity(source=SessionSource.COMMAND, native_id="root"),
            turn_active=True,
            last_hook_at=0,
            presence=ProcessStatus.ALIVE,
        )
        snapshot = self.update_snapshot(session, updates=(update,))
        self.now = 1000
        self.assertTrue(self.attention.is_routine_eligible(snapshot.work[0], snapshot))
        snapshot = snapshot.model_copy(
            update={
                "sessions": (
                    session.model_copy(update={"presence": ProcessStatus.UNKNOWN}),
                )
            }
        )
        self.assertFalse(self.attention.is_routine_eligible(snapshot.work[0], snapshot))
        snapshot = snapshot.model_copy(
            update={
                "sessions": (
                    session.model_copy(
                        update={"presence": ProcessStatus.UNKNOWN, "last_hook_at": 999}
                    ),
                )
            }
        )
        self.assertTrue(self.attention.is_routine_eligible(snapshot.work[0], snapshot))

    def test_live_child_selects_its_assignment_after_parent_result_is_consumed(
        self,
    ) -> None:
        parent = WorkUpdate(
            session=SessionIdentity(source=SessionSource.COMMAND, native_id="root"),
            revision="result",
            observed_at=10,
            ended_at=10,
        )
        child = WorkUpdate(
            session=SessionIdentity(source=SessionSource.COMMAND, native_id="child"),
            revision="assignment",
            observed_at=20,
        )
        snapshot = self.update_snapshot(
            session_snapshot(
                id=SessionIdentity(source=SessionSource.COMMAND, native_id="root"),
                outcome=ExecutionOutcome.SUCCESS,
            ),
            session_snapshot(
                id=SessionIdentity(source=SessionSource.COMMAND, native_id="child"),
                parent=SessionIdentity(source=SessionSource.COMMAND, native_id="root"),
                turn_active=True,
            ),
            updates=(parent, child),
        )
        self.attention.mark_considered(snapshot.work[0], parent)
        selected = self.attention.select(snapshot)
        self.assertEqual(selected.update, child)
        self.assertEqual(
            require(selected.focus_session).id,
            SessionIdentity(source=SessionSource.COMMAND, native_id="child"),
        )
        self.attention.mark_considered(snapshot.work[0], child)
        self.assertFalse(self.attention.is_routine_eligible(snapshot.work[0], snapshot))
        targeted = snapshot.model_copy(
            update={
                "trigger": SessionTrigger(
                    session_id=SessionIdentity(
                        source=SessionSource.COMMAND, native_id="root"
                    )
                )
            }
        )
        self.assertFalse(
            self.attention.is_selection_eligible(
                self.attention.select(targeted, SpeechReason.COMPLETION), targeted
            )
        )

    def test_live_parent_does_not_revive_consumed_child_result(self) -> None:
        parent = WorkUpdate(
            session=SessionIdentity(source=SessionSource.COMMAND, native_id="root"),
            revision="assignment",
            observed_at=10,
        )
        child = WorkUpdate(
            session=SessionIdentity(source=SessionSource.COMMAND, native_id="child"),
            revision="result",
            observed_at=20,
            ended_at=20,
        )
        snapshot = self.update_snapshot(
            session_snapshot(
                id=SessionIdentity(source=SessionSource.COMMAND, native_id="root"),
                turn_active=True,
            ),
            session_snapshot(
                id=SessionIdentity(source=SessionSource.COMMAND, native_id="child"),
                parent=SessionIdentity(source=SessionSource.COMMAND, native_id="root"),
                outcome=ExecutionOutcome.SUCCESS,
            ),
            updates=(parent, child),
        )
        self.attention.mark_considered(snapshot.work[0], child)
        self.assertEqual(self.attention.select(snapshot).update, parent)
        targeted = snapshot.model_copy(
            update={
                "trigger": SessionTrigger(
                    session_id=SessionIdentity(
                        source=SessionSource.COMMAND, native_id="child"
                    )
                )
            }
        )
        self.assertFalse(
            self.attention.is_selection_eligible(
                self.attention.select(targeted, SpeechReason.COMPLETION), targeted
            )
        )

    def test_check_ins_require_live_work_even_after_an_update_is_consumed(self) -> None:
        update = WorkUpdate(
            session=SessionIdentity(source=SessionSource.COMMAND, native_id="root"),
            revision="progress",
            observed_at=10,
        )
        snapshot = self.update_snapshot(
            session_snapshot(
                id=SessionIdentity(source=SessionSource.COMMAND, native_id="root"),
                turn_active=True,
            ),
            updates=(update,),
        )
        self.attention.mark_considered(snapshot.work[0], update)
        routine = self.attention.select(snapshot)
        self.assertFalse(self.attention.is_selection_eligible(routine, snapshot))
        selected = self.attention.select(snapshot, SpeechReason.CHECK_IN)
        self.assertTrue(self.attention.is_selection_eligible(selected, snapshot))
        snapshot = snapshot.model_copy(
            update={
                "sessions": (
                    snapshot.sessions[0].model_copy(update={"turn_active": False}),
                )
            }
        )
        self.assertFalse(self.attention.is_selection_eligible(selected, snapshot))

    def test_rotates_within_a_tier_before_preferring_recency(self) -> None:
        snapshot = self.snapshot(
            session_snapshot(
                id=SessionIdentity(source=SessionSource.COMMAND, native_id="older"),
                turn_active=True,
                changed_at=10,
            ),
            session_snapshot(
                id=SessionIdentity(source=SessionSource.COMMAND, native_id="newer"),
                turn_active=True,
                changed_at=20,
            ),
        )
        chosen = []
        for _ in range(4):
            selected = self.attention.select(snapshot)
            chosen.append(selected.id)
            self.attention.mark_scheduled(selected.id)
        self.assertEqual(
            chosen,
            [
                SessionIdentity(source=SessionSource.COMMAND, native_id="newer"),
                SessionIdentity(source=SessionSource.COMMAND, native_id="older"),
                SessionIdentity(source=SessionSource.COMMAND, native_id="newer"),
                SessionIdentity(source=SessionSource.COMMAND, native_id="older"),
            ],
        )

    def connected_snapshot(self, *sessions: AgentSessionStatus) -> PetContext:
        root, *children = sessions
        subject = WorkSubject(
            id=root.id,
            kind=WorkSubjectKind.WORK,
            changed_at=max(session.changed_at for session in sessions),
            revision="one",
            story=WorkStory(
                id=root.id,
                intent="Complete the request",
                revision="one",
                sessions=tuple(
                    SessionContribution(scope=child.goal, session=child.id)
                    for child in children
                ),
            ),
            sessions=tuple(session.id for session in sessions),
        )
        return capture_snapshot(sessions_snapshot(*sessions), work=(subject,))

    def test_routine_focuses_the_root_with_descendant_activity_and_reports(
        self,
    ) -> None:
        snapshot = self.connected_snapshot(
            session_snapshot(
                id=SessionIdentity(source=SessionSource.COMMAND, native_id="root"),
                quiet_seconds=60,
            ),
            session_snapshot(
                id=SessionIdentity(source=SessionSource.COMMAND, native_id="child"),
                parent=SessionIdentity(source=SessionSource.COMMAND, native_id="root"),
                turn_active=True,
                goal="Review code",
            ),
            session_snapshot(
                id=SessionIdentity(source=SessionSource.COMMAND, native_id="nested"),
                parent=SessionIdentity(source=SessionSource.COMMAND, native_id="child"),
                turn_active=True,
                goal="Run tests",
            ),
        )
        selected = self.attention.select(snapshot)
        self.assertEqual(
            require(selected.focus_session).id,
            SessionIdentity(source=SessionSource.COMMAND, native_id="root"),
        )
        self.assertFalse(require(selected.focus_session).turn_active)
        self.assertEqual(
            {session.id for session in selected.context_sessions},
            {
                SessionIdentity(source=SessionSource.COMMAND, native_id="child"),
                SessionIdentity(source=SessionSource.COMMAND, native_id="nested"),
            },
        )
        self.assertEqual(
            [item.scope for item in require(selected.subject).story.sessions],
            ["Review code", "Run tests"],
        )
        self.assertTrue(selected.is_valid(snapshot))

    def test_click_follows_active_descendants_with_a_finished_root(self) -> None:
        snapshot = self.connected_snapshot(
            session_snapshot(
                id=SessionIdentity(source=SessionSource.COMMAND, native_id="root"),
                outcome=ExecutionOutcome.SUCCESS,
            ),
            session_snapshot(
                id=SessionIdentity(source=SessionSource.COMMAND, native_id="child"),
                parent=SessionIdentity(source=SessionSource.COMMAND, native_id="root"),
                turn_active=True,
            ),
            session_snapshot(
                id=SessionIdentity(source=SessionSource.COMMAND, native_id="nested"),
                parent=SessionIdentity(source=SessionSource.COMMAND, native_id="child"),
                turn_active=True,
            ),
        )
        selected = self.attention.select(snapshot, SpeechReason.CLICK)
        self.assertEqual(
            require(selected.focus_session).id,
            SessionIdentity(source=SessionSource.COMMAND, native_id="root"),
        )
        self.assertEqual(selected.situation, "click_working")
        self.assertFalse(require(selected.focus_session).turn_active)
        self.assertEqual(
            require(selected.focus_session).outcome, ExecutionOutcome.SUCCESS
        )
        self.assertTrue(selected.is_valid(snapshot))
        finished = snapshot.model_copy(
            update={
                "sessions": tuple(
                    session.model_copy(update={"turn_active": False})
                    for session in snapshot.sessions
                )
            }
        )
        self.assertFalse(selected.is_valid(finished))
        self.assertEqual(
            self.attention.select(finished, SpeechReason.CLICK).situation,
            "idle_reflection",
        )

    def test_click_focuses_the_oldest_waiting_descendant(self) -> None:
        snapshot = self.connected_snapshot(
            session_snapshot(
                id=SessionIdentity(source=SessionSource.COMMAND, native_id="root"),
                turn_active=True,
            ),
            session_snapshot(
                id=SessionIdentity(source=SessionSource.COMMAND, native_id="child"),
                parent=SessionIdentity(source=SessionSource.COMMAND, native_id="root"),
                waiting_for_input=True,
                waiting_seconds=30,
            ),
            session_snapshot(
                id=SessionIdentity(source=SessionSource.COMMAND, native_id="nested"),
                parent=SessionIdentity(source=SessionSource.COMMAND, native_id="child"),
                waiting_for_input=True,
                waiting_seconds=5,
            ),
        )
        self.attention.mark_scheduled(
            SessionIdentity(source=SessionSource.COMMAND, native_id="child")
        )
        selected = self.attention.select(snapshot, SpeechReason.CLICK)
        self.assertEqual(
            require(selected.focus_session).id,
            SessionIdentity(source=SessionSource.COMMAND, native_id="child"),
        )
        self.assertEqual(selected.situation, "click_waiting")
        self.assertTrue(selected.is_valid(snapshot))
        targeted = snapshot.model_copy(
            update={
                "trigger": SessionTrigger(
                    session_id=SessionIdentity(
                        source=SessionSource.COMMAND, native_id="nested"
                    )
                )
            }
        )
        selected = self.attention.select(targeted, SpeechReason.CHECK_IN)
        self.assertEqual(
            require(selected.focus_session).id,
            SessionIdentity(source=SessionSource.COMMAND, native_id="nested"),
        )
        self.assertEqual(selected.situation, "check_in_waiting")

    def test_check_in_retains_descendant_context_for_an_eligible_root(self) -> None:
        snapshot = self.connected_snapshot(
            session_snapshot(
                id=SessionIdentity(source=SessionSource.COMMAND, native_id="root"),
                turn_active=True,
            ),
            session_snapshot(
                id=SessionIdentity(source=SessionSource.COMMAND, native_id="child"),
                parent=SessionIdentity(source=SessionSource.COMMAND, native_id="root"),
                waiting_for_input=True,
                waiting_seconds=30,
            ),
        )
        selected = self.attention.select(
            snapshot,
            SpeechReason.CHECK_IN,
            eligible=(SessionIdentity(source=SessionSource.COMMAND, native_id="root"),),
        )
        self.assertEqual(
            require(selected.focus_session).id,
            SessionIdentity(source=SessionSource.COMMAND, native_id="root"),
        )
        self.assertEqual(
            [session.id for session in selected.context_sessions],
            [SessionIdentity(source=SessionSource.COMMAND, native_id="child")],
        )

    def test_an_unavailable_parent_does_not_promote_child_activity_to_focus(
        self,
    ) -> None:
        child = session_snapshot(
            id=SessionIdentity(source=SessionSource.COMMAND, native_id="child"),
            parent=SessionIdentity(source=SessionSource.COMMAND, native_id="missing"),
            turn_active=True,
        )
        snapshot = capture_snapshot(sessions_snapshot(child))
        self.assertIsNone(self.attention.select(snapshot).focus_session)
        self.assertEqual(
            self.attention.select(snapshot, SpeechReason.CLICK).situation, "click_idle"
        )
        waiting = child.model_copy(
            update={"waiting_for_input": True, "waiting_seconds": 10}
        )
        snapshot = snapshot.model_copy(update={"sessions": (waiting,)})
        self.assertEqual(
            require(self.attention.select(snapshot).focus_session).id,
            SessionIdentity(source=SessionSource.COMMAND, native_id="child"),
        )

    def test_explicit_input_precedes_active_work(self) -> None:
        waiting = session_snapshot(
            id=SessionIdentity(source=SessionSource.COMMAND, native_id="input"),
            turn_active=True,
            waiting_for_input=True,
            waiting_seconds=0,
        )
        snapshot = self.snapshot(
            waiting,
            session_snapshot(
                id=SessionIdentity(source=SessionSource.COMMAND, native_id="active"),
                turn_active=True,
            ),
        )
        self.assertEqual(
            self.attention.select(snapshot).id,
            SessionIdentity(source=SessionSource.COMMAND, native_id="input"),
        )

    def test_tool_failure_alone_does_not_outrank_an_active_session(self) -> None:
        snapshot = self.snapshot(
            session_snapshot(
                id=SessionIdentity(source=SessionSource.COMMAND, native_id="failed"),
                failure=FailureSnapshot(
                    activity=Activity.TESTING,
                    count=1,
                    goal="",
                    current_turn=True,
                    resolved=False,
                    seconds_ago=0,
                ),
                outcome=ExecutionOutcome.ERROR,
                changed_at=20,
            ),
            session_snapshot(
                id=SessionIdentity(source=SessionSource.COMMAND, native_id="active"),
                turn_active=True,
                changed_at=1,
            ),
        )
        self.assertEqual(
            self.attention.select(snapshot).id,
            SessionIdentity(source=SessionSource.COMMAND, native_id="active"),
        )

    def test_event_target_keeps_its_focus_and_direction(self) -> None:
        snapshot = self.snapshot(
            session_snapshot(
                id=SessionIdentity(source=SessionSource.COMMAND, native_id="finished"),
                outcome=ExecutionOutcome.SUCCESS,
            ),
        )
        snapshot = snapshot.model_copy(
            update={
                "trigger": SessionTrigger(
                    session_id=SessionIdentity(
                        source=SessionSource.COMMAND, native_id="finished"
                    )
                )
            }
        )
        selected = self.attention.select(snapshot, SpeechReason.COMPLETION)
        self.assertEqual(
            require(selected.focus_session).id,
            SessionIdentity(source=SessionSource.COMMAND, native_id="finished"),
        )
        self.assertEqual(selected.situation, "completion")

    def test_manual_context_recalls_work_without_active_focus(self) -> None:
        snapshot = self.snapshot(
            session_snapshot(
                id=SessionIdentity(source=SessionSource.COMMAND, native_id="active"),
                turn_active=True,
            )
        )
        explicit = self.attention.select(snapshot, context="Inspect the result")
        self.assertEqual(require(explicit.subject).id.native_id, "active")
        self.assertIsNone(explicit.focus_session)
        self.assertEqual(explicit.situation, "explicit")
        snapshot = snapshot.model_copy(
            update={"trigger": CareTrigger(care=CareKind.EXERCISE)}
        )
        self.assertEqual(
            self.attention.select(snapshot, SpeechReason.CARE).situation,
            "care_exercise",
        )

    def test_click_prioritizes_the_oldest_explicit_input_wait(self) -> None:
        snapshot = self.snapshot(
            session_snapshot(
                id=SessionIdentity(source=SessionSource.COMMAND, native_id="recent"),
                waiting_for_input=True,
                waiting_seconds=5,
                changed_at=20,
            ),
            session_snapshot(
                id=SessionIdentity(source=SessionSource.COMMAND, native_id="old"),
                waiting_for_input=True,
                waiting_seconds=30,
                changed_at=10,
            ),
        )
        selected = self.attention.select(snapshot, SpeechReason.CLICK)
        self.assertEqual(
            require(selected.focus_session).id,
            SessionIdentity(source=SessionSource.COMMAND, native_id="old"),
        )
        self.assertEqual(selected.situation, "click_waiting")

    def test_pending_failure_survives_a_lower_priority_prompt(self) -> None:
        self.attention.queue(
            SpeechReason.FAILURE,
            SessionTrigger(
                session_id=SessionIdentity(
                    source=SessionSource.COMMAND, native_id="failed"
                )
            ),
        )
        self.attention.queue(
            SpeechReason.PROMPT,
            SessionTrigger(
                session_id=SessionIdentity(
                    source=SessionSource.COMMAND, native_id="working"
                )
            ),
            failed=True,
        )
        self.assertEqual(
            self.attention.pending,
            (
                "failure",
                1,
                SessionTrigger(
                    session_id=SessionIdentity(
                        source=SessionSource.COMMAND, native_id="failed"
                    )
                ),
            ),
        )

    def test_cooldown_preserves_waiting_and_user_requests(self) -> None:
        self.attention.mark_scheduled()
        self.assertFalse(self.attention.can_speak(SpeechReason.ROUTINE))
        self.assertTrue(self.attention.can_speak(SpeechReason.CLICK))
        self.assertTrue(
            self.attention.can_speak(
                SpeechReason.CHECK_IN,
                ReactionTrigger(
                    reaction=SessionReaction(
                        kind=ReactionKind.WAITING,
                        session=SessionIdentity(
                            source=SessionSource.COMMAND, native_id="parser"
                        ),
                        generation=1,
                        activity=Activity.COMMAND,
                        serial=1,
                    )
                ),
            )
        )
        self.now = self.config.behavior.automatic_speech_cooldown_seconds
        self.assertTrue(self.attention.can_speak(SpeechReason.ROUTINE))

    def test_considered_subjects_stay_quiet_after_restoring_history(self) -> None:
        snapshot = self.snapshot(
            session_snapshot(
                id=SessionIdentity(source=SessionSource.COMMAND, native_id="parser"),
                turn_active=True,
            )
        )
        selected = self.attention.select(snapshot)
        self.attention.mark_scheduled(selected.id)
        self.attention.mark_considered(require(selected.subject))
        history = self.attention.history(require(selected.id))
        restored = Attention(self.config, lambda: self.now, random.Random(0))
        restored.restore({require(selected.id): history})
        self.now = 60
        self.assertFalse(restored.has_unconsidered_changes(require(selected.subject)))
        self.assertIsNone(restored.request(snapshot))

    def test_request_precedence_keeps_manual_and_click_ahead_of_due_events(
        self,
    ) -> None:
        self.attention.queue(SpeechReason.PROMPT)
        snapshot = self.snapshot(
            session_snapshot(
                id=SessionIdentity(source=SessionSource.COMMAND, native_id="active"),
                turn_active=True,
            )
        )
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
        snapshot = self.snapshot(
            session_snapshot(
                id=SessionIdentity(source=SessionSource.COMMAND, native_id="present"),
                turn_active=False,
            )
        )
        self.now = 30
        self.assertIsNone(self.attention.request(snapshot))

    def test_nap_pauses_automatic_requests_and_preserves_the_pending_trigger(
        self,
    ) -> None:
        snapshot = self.snapshot(
            session_snapshot(
                id=SessionIdentity(source=SessionSource.COMMAND, native_id="parser"),
                turn_active=True,
            )
        )
        snapshot = snapshot.model_copy(update={"sleeping": True})
        self.now = 60
        self.attention.queue(
            SpeechReason.PROMPT,
            SessionTrigger(
                session_id=SessionIdentity(
                    source=SessionSource.COMMAND, native_id="parser"
                )
            ),
        )
        pending = self.attention.pending
        self.assertIsNone(
            self.attention.request(
                snapshot,
                check_in=(
                    SessionIdentity(source=SessionSource.COMMAND, native_id="parser"),
                ),
                care=CareKind.EXERCISE,
            )
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
        snapshot = snapshot.model_copy(update={"sleeping": False})
        self.assertEqual(require(self.attention.request(snapshot)).reason, "prompt")

    def test_check_in_keeps_the_eligible_sessions(self) -> None:
        snapshot = self.snapshot(
            session_snapshot(
                id=SessionIdentity(source=SessionSource.COMMAND, native_id="first"),
                turn_active=True,
            ),
            session_snapshot(
                id=SessionIdentity(source=SessionSource.COMMAND, native_id="second"),
                turn_active=True,
            ),
        )
        request = self.attention.request(
            snapshot,
            check_in=(
                SessionIdentity(source=SessionSource.COMMAND, native_id="second"),
            ),
        )
        selected = self.attention.select(
            snapshot, require(request).reason, eligible=require(request).eligible
        )
        self.assertEqual(
            require(selected.focus_session).id,
            SessionIdentity(source=SessionSource.COMMAND, native_id="second"),
        )

    def test_pending_delivery_preserves_every_automatic_request(self) -> None:
        snapshot = self.snapshot(
            session_snapshot(
                id=SessionIdentity(source=SessionSource.COMMAND, native_id="parser"),
                turn_active=True,
            )
        )
        self.now = 60
        self.attention.queue(
            SpeechReason.PROMPT,
            SessionTrigger(
                session_id=SessionIdentity(
                    source=SessionSource.COMMAND, native_id="parser"
                )
            ),
        )
        pending = self.attention.pending
        next_observation = self.attention.next_observation
        self.assertIsNone(
            self.attention.request(
                snapshot,
                check_in=(
                    SessionIdentity(source=SessionSource.COMMAND, native_id="parser"),
                ),
                care=CareKind.EXERCISE,
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
        snapshot = self.snapshot(
            session_snapshot(
                id=SessionIdentity(source=SessionSource.COMMAND, native_id="parser"),
                turn_active=True,
            )
        )
        self.now = 60
        next_observation = self.attention.next_observation
        self.assertIsNone(self.attention.request(snapshot, delivery_pending=True))
        self.assertEqual(self.attention.next_observation, next_observation)
        self.assertEqual(require(self.attention.request(snapshot)).reason, "routine")
        idle = self.snapshot()
        self.assertIsNone(
            self.attention.request(idle, care=CareKind.EXERCISE, delivery_pending=True)
        )
        self.assertEqual(
            require(self.attention.request(idle, care=CareKind.EXERCISE)).reason, "care"
        )

    def test_quiet_conversation_reflects_after_five_minutes_and_every_fifteen_minutes(
        self,
    ) -> None:
        root = SessionIdentity(source=SessionSource.COMMAND, native_id="parser")
        snapshot = self.snapshot(session_snapshot(id=root, last_hook_at=0))
        self.attention.next_observation = 10_000

        self.now = 299
        self.assertIsNone(self.attention.request(snapshot))
        self.now = 300
        request = require(self.attention.request(snapshot))
        self.assertEqual(request.reason, SpeechReason.CHECK_IN)
        self.assertTrue(request.reflection)
        selected = self.attention.select(snapshot, request.reason, reflection=True)
        self.assertEqual(require(selected.subject).id, root)
        self.assertTrue(self.attention.is_selection_eligible(selected, snapshot))

        self.now = 1199
        self.assertIsNone(self.attention.request(snapshot))
        self.now = 1200
        self.assertTrue(require(self.attention.request(snapshot)).reflection)

        changed = self.snapshot(session_snapshot(id=root, last_hook_at=1200))
        self.assertFalse(selected.is_valid(changed))

    def test_active_tool_with_unknown_presence_blocks_idle_reflection_at_five_minutes(
        self,
    ) -> None:
        root = SessionIdentity(source=SessionSource.COMMAND, native_id="parser")
        running = session_snapshot(
            id=root,
            last_hook_at=0,
            turn_active=True,
            presence=ProcessStatus.UNKNOWN,
            active_tools=(
                ActiveTool(
                    call="test-suite",
                    tool="pytest",
                    activity=Activity.TESTING,
                    duration_seconds=300,
                ),
            ),
        )
        snapshot = self.snapshot(running)
        self.attention.next_observation = 10_000
        self.now = 300

        self.assertFalse(self.attention.is_quiet(snapshot))
        check_in = require(self.attention.request(snapshot, check_in=(root,)))
        self.assertEqual(check_in.reason, SpeechReason.CHECK_IN)
        self.assertFalse(check_in.reflection)
        self.assertIsNone(self.attention.request(snapshot))
        reflection = self.attention.select(
            snapshot, SpeechReason.CHECK_IN, reflection=True
        )
        self.assertFalse(self.attention.is_selection_eligible(reflection, snapshot))

    def test_report_memory_spacing_uses_current_age_and_last_consideration(
        self,
    ) -> None:
        recent = SelectedMemory(key="report", observed_at=0)
        ongoing = recent.model_copy(update={"is_ongoing": True})

        self.now = 899
        self.assertTrue(self.attention.is_memory_eligible(recent))
        self.attention.sources["report"] = SourceAttention(considered_at=899)
        self.assertFalse(self.attention.is_memory_eligible(recent))
        self.now = 900
        self.assertFalse(self.attention.is_memory_eligible(recent))

        self.attention.sources["report"] = SourceAttention(considered_at=1000)
        self.now = 1899
        self.assertFalse(self.attention.is_memory_eligible(ongoing))
        self.now = 1900
        self.assertTrue(self.attention.is_memory_eligible(ongoing))

        self.attention.sources["report"] = SourceAttention(
            considered_at=3500, mentioned_at=3501
        )
        self.now = 7100
        self.assertFalse(self.attention.is_memory_eligible(ongoing))
        self.now = 7101
        self.assertTrue(self.attention.is_memory_eligible(ongoing))

        self.attention.sources["report"] = SourceAttention(mentioned_at=86_390)
        self.now = 172_789
        self.assertFalse(self.attention.is_memory_eligible(ongoing))
        self.now = 172_790
        self.assertTrue(self.attention.is_memory_eligible(ongoing))

    def test_other_conversations_do_not_expire_parser_completion(
        self,
    ) -> None:
        parser = SessionIdentity(source=SessionSource.COMMAND, native_id="parser")
        billing = SessionIdentity(source=SessionSource.COMMAND, native_id="billing")
        parser_session = session_snapshot(id=parser, outcome=ExecutionOutcome.SUCCESS)
        billing_session = session_snapshot(id=billing, last_hook_at=20)
        base = self.snapshot(parser_session, billing_session)
        parser_update = WorkUpdate(
            session=parser, revision="parser-result", observed_at=10, ended_at=10
        )
        parser_subject = base.work[0].model_copy(update={"updates": (parser_update,)})
        billing_subject = base.work[1].model_copy(
            update={
                "story": base.work[1].story.model_copy(
                    update={
                        "requests": (
                            CommentaryEvidence(
                                source=EvidenceSource.USER,
                                kind=EvidenceKind.INTENT,
                                details=UserIntent(text="Investigate billing totals"),
                                observed_at=20,
                            ),
                        )
                    }
                )
            }
        )
        self.now = 20
        separate_roots = base.model_copy(
            update={"work": (parser_subject, billing_subject)}
        )
        self.assertTrue(
            self.attention.is_update_eligible(parser_update, separate_roots)
        )

        shared_pr = ArtifactContext(
            kind=ArtifactKind.PULL_REQUEST,
            ref="https://example.test/team/parser/pull/23",
        )
        parser_with_pr = parser_subject.model_copy(
            update={
                "story": parser_subject.story.model_copy(
                    update={"artifacts": (shared_pr,)}
                )
            }
        )
        billing_with_pr = billing_subject.model_copy(
            update={
                "story": billing_subject.story.model_copy(
                    update={"artifacts": (shared_pr,)}
                )
            }
        )
        linked_roots = separate_roots.model_copy(
            update={"work": (parser_with_pr, billing_with_pr)}
        )
        self.assertTrue(self.attention.is_update_eligible(parser_update, linked_roots))

        child_assignment = billing_with_pr.model_copy(update={"parent": parser})
        child_work = base.model_copy(
            update={"work": (parser_with_pr, child_assignment)}
        )
        self.assertTrue(self.attention.is_update_eligible(parser_update, child_work))

        newer_parser_request = CommentaryEvidence(
            source=EvidenceSource.USER,
            kind=EvidenceKind.INTENT,
            details=UserIntent(text="Check parser diagnostics"),
            observed_at=20,
        )
        renewed_parser = parser_subject.model_copy(
            update={
                "story": parser_subject.story.model_copy(
                    update={"requests": (newer_parser_request,)}
                )
            }
        )
        parser_restarted = base.model_copy(
            update={"work": (renewed_parser, billing_subject)}
        )
        self.assertFalse(
            self.attention.is_update_eligible(parser_update, parser_restarted)
        )

        newer_completion = WorkUpdate(
            session=parser, revision="parser-new-result", observed_at=21, ended_at=21
        )
        newer = linked_roots.model_copy(
            update={
                "work": (
                    parser_with_pr.model_copy(update={"updates": (newer_completion,)}),
                    billing_with_pr,
                )
            }
        )
        self.now = 21
        self.assertTrue(self.attention.is_update_eligible(newer_completion, newer))

        fresh_update = WorkUpdate(
            session=parser, revision="parser-progress", observed_at=21
        )
        parser_active = parser_session.model_copy(
            update={"turn_active": True, "last_hook_at": 21}
        )
        fresh = separate_roots.model_copy(
            update={
                "sessions": (parser_active, billing_session),
                "work": (
                    parser_subject.model_copy(update={"updates": (fresh_update,)}),
                    billing_subject,
                ),
            }
        )
        self.now = 21
        self.assertTrue(self.attention.is_update_eligible(fresh_update, fresh))
