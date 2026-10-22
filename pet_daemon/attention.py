import json
from collections.abc import Callable, Collection, Mapping
from copy import deepcopy
from dataclasses import dataclass, replace

from .config import Config
from .domain import (
    CareKind,
    ExecutionOutcome,
    SpeechReason,
    SpeechSituation,
    WorkSubjectKind,
)
from .logging import get_logger
from .runtime import (
    CareTrigger,
    PetContext,
    RandomSource,
    ReactionTrigger,
    SessionContext,
    SessionTrigger,
    SubjectTrigger,
    Trigger,
)
from .sessions.reactions import SessionReaction
from .storage.memory import HistoryRecord
from .work import WorkSubject

log = get_logger("pet-daemon")


@dataclass(frozen=True)
class AttentionRequest:
    reason: SpeechReason
    context: str | None = None
    ttl: float | None = None
    trigger: Trigger | None = None
    eligible: frozenset[str] | None = None


@dataclass(frozen=True)
class SelectionSupport:
    subject_revision: str | None = None
    session_transition: tuple[int, bool, bool, ExecutionOutcome | None] | tuple[()] = ()

    @classmethod
    def capture(
        cls,
        subject: WorkSubject | None,
        focus: SessionContext | None,
    ) -> "SelectionSupport":
        return cls(
            subject.revision if subject else None,
            (
                focus.generation,
                focus.turn_active,
                focus.waiting_for_input,
                focus.outcome,
            )
            if focus
            else (),
        )


@dataclass(frozen=True)
class AttentionSelection:
    subject: WorkSubject | None
    focus_session: SessionContext | None
    context_sessions: tuple[SessionContext, ...]
    reason: SpeechReason
    situation: SpeechSituation
    context: str | None = None
    supporting_facts: SelectionSupport = SelectionSupport()

    @property
    def id(self) -> str | None:
        return (
            self.subject.id
            if self.subject is not None
            else self.focus_session.id
            if self.focus_session
            else None
        )

    def supported(self, snapshot: PetContext) -> bool:
        current = self.refresh(snapshot)
        if self.subject is not None and current.subject is None:
            return False
        if self.focus_session is not None:
            if current.focus_session is None:
                return False
            if (
                self.supporting_facts.session_transition
                != current.supporting_facts.session_transition
            ):
                return False
        if current.subject is not None and current.focus_session is not None:
            members = set(current.subject.sessions)
            if current.subject.kind == WorkSubjectKind.SESSION:
                members.add(current.subject.id)
            if current.focus_session.id not in members:
                return False
        if self.situation == SpeechSituation.CLICK_WORKING:
            return any(
                session.turn_active and not session.activity_uncertain
                for session in (
                    *((current.focus_session,) if current.focus_session else ()),
                    *current.context_sessions,
                )
            )
        return True

    def refresh(self, snapshot: PetContext) -> "AttentionSelection":
        subjects = {subject.id: subject for subject in snapshot.work}
        sessions = {session.id: session for session in snapshot.sessions}
        subject = subjects.get(self.subject.id) if self.subject is not None else None
        focus = sessions.get(self.focus_session.id) if self.focus_session else None
        return replace(
            self,
            subject=subject,
            supporting_facts=SelectionSupport.capture(subject, focus),
            focus_session=deepcopy(focus),
            context_sessions=tuple(
                deepcopy(sessions[item.id])
                for item in self.context_sessions
                if item.id in sessions
            ),
        )


def _sessions(subject: WorkSubject, snapshot: PetContext) -> list[SessionContext]:
    identifiers = set(subject.sessions)
    if subject.kind == WorkSubjectKind.SESSION:
        identifiers.add(subject.id)
    return [session for session in snapshot.sessions if session.id in identifiers]


def _session_tier(session: SessionContext) -> int:
    if session.waiting_for_input:
        return 0
    if session.turn_active and not session.activity_uncertain:
        return 3
    if session.outcome and session.quiet_seconds < 300:
        return 4
    return 5


def _target_session(trigger: Trigger | None) -> str | None:
    if isinstance(trigger, SessionTrigger):
        return trigger.session_id
    if isinstance(trigger, ReactionTrigger):
        return trigger.reaction.session
    return None


class Attention:
    def __init__(
        self,
        config: Config,
        clock: Callable[[], float],
        rng: RandomSource,
        history: Mapping[str, HistoryRecord] | None = None,
    ) -> None:
        self.config = config
        self.clock = clock
        self.rng = rng
        self.pending: tuple[SpeechReason, float, Trigger | None] | None = None
        self.last_attempt = float("-inf")
        self.next_observation = clock() + rng.uniform(*config.observation_seconds)
        self.selections: dict[str, float] = {}
        self.considered: dict[str, str | None] = {}
        self.sequence: float = 0
        if history:
            self.restore(history)

    def reset(self) -> None:
        self.pending = None
        self.last_attempt = float("-inf")
        self.next_observation = self.clock() + self.rng.uniform(
            *self.config.observation_seconds
        )
        self.selections.clear()
        self.considered.clear()
        self.sequence = 0

    def queue(
        self, reason: SpeechReason, trigger: Trigger | None = None, failed: bool = False
    ) -> None:
        now = self.clock()
        if reason == SpeechReason.FAILURE:
            self.pending = (reason, now + 1, trigger)
        elif (
            reason == SpeechReason.CHECK_IN
            or not self.pending
            or self.pending[0] != SpeechReason.FAILURE
            or not failed
        ):
            delay = {
                SpeechReason.PROMPT: 0,
                SpeechReason.COMPLETION: 1,
                SpeechReason.CHECK_IN: 0,
            }[reason]
            self.pending = (reason, now + delay, trigger)
        log.debug(
            "observation.queued",
            reason=self.pending[0],
            scheduled_at=self.pending[1],
            session=_target_session(self.pending[2]),
            scheduled=self.pending,
        )

    def routine_eligible(self, subject: WorkSubject, snapshot: PetContext) -> bool:
        if not self.is_new(subject):
            return False
        story = subject.story
        if (
            story.intent
            or story.requests
            or story.activity
            or any(
                session.scope or session.requests or session.activity
                for session in story.sessions
            )
        ):
            return True
        return any(
            session.turn_active or session.waiting_for_input or session.outcome
            for session in _sessions(subject, snapshot)
        )

    def select(
        self,
        snapshot: PetContext,
        reason: SpeechReason = SpeechReason.ROUTINE,
        context: str | None = None,
        eligible: Collection[str] | None = None,
    ) -> AttentionSelection:
        trigger = snapshot.trigger
        subject = None
        focus = None
        others: tuple[SessionContext, ...] = ()
        if context is None and reason not in (SpeechReason.EXPLICIT, SpeechReason.CARE):
            target_session = (
                _target_session(trigger) if reason != SpeechReason.CLICK else None
            )
            target_subject = (
                trigger.subject_id if isinstance(trigger, SubjectTrigger) else None
            )
            subjects = [
                subject
                for subject in snapshot.work
                if reason != SpeechReason.ROUTINE
                or self.routine_eligible(subject, snapshot)
            ]
            candidates = []
            for item in subjects:
                sessions = _sessions(item, snapshot)
                eligible_sessions = sessions
                if eligible is not None:
                    eligible_sessions = [
                        session for session in sessions if session.id in eligible
                    ]
                    if not eligible_sessions and item.id not in eligible:
                        continue
                tier = min(
                    (_session_tier(session) for session in eligible_sessions), default=5
                )
                candidates.append((item, sessions, tier))
            targeted = [
                item
                for item in candidates
                if item[0].id == target_subject
                or target_session
                and any(session.id == target_session for session in item[1])
            ]
            if targeted:
                candidates = targeted
            elif target_session:
                candidates = []
            if candidates:
                subject, sessions, tier = min(
                    candidates,
                    key=lambda item: (
                        item[2],
                        self.selections.get(item[0].id, -1),
                        -max(
                            (
                                session.input_wait_seconds
                                for session in item[1]
                                if session.waiting_for_input
                            ),
                            default=0,
                        )
                        if reason == SpeechReason.CLICK and item[2] == 0
                        else -item[0].changed_at,
                        item[0].id,
                    ),
                )
                preferred = [
                    session
                    for session in sessions
                    if session.id == target_session
                    and (session.parent is None or session.waiting_for_input)
                ]
                if not preferred and not target_session:
                    preferred = [
                        session
                        for session in sessions
                        if session.waiting_for_input
                        and (eligible is None or session.id in eligible)
                    ]
                if not preferred:
                    preferred = [
                        session for session in sessions if session.id == subject.id
                    ]
                if preferred:
                    focus = min(
                        preferred,
                        key=lambda session: (
                            _session_tier(session),
                            -session.input_wait_seconds
                            if reason == SpeechReason.CLICK
                            and session.waiting_for_input
                            else 0,
                            self.selections.get(session.id, -1),
                            -session.input_wait_seconds
                            if session.waiting_for_input
                            else session.quiet_seconds
                            if not session.turn_active
                            else -session.quiet_seconds,
                            session.id,
                        ),
                    )
                others = tuple(
                    sorted(
                        (
                            session
                            for session in sessions
                            if focus is None or session.id != focus.id
                        ),
                        key=lambda session: (_session_tier(session), session.id),
                    )
                )
            else:
                sessions = [
                    session
                    for session in snapshot.sessions
                    if (eligible is None or session.id in eligible)
                    and (session.parent is None or session.waiting_for_input)
                ]
                preferred = (
                    [session for session in sessions if session.id == target_session]
                    if target_session
                    else sessions
                )
                if preferred:
                    focus = min(
                        preferred,
                        key=lambda session: (
                            _session_tier(session),
                            self.selections.get(session.id, -1),
                            -session.input_wait_seconds
                            if session.waiting_for_input
                            else session.quiet_seconds
                            if not session.turn_active
                            else -session.quiet_seconds,
                            session.id,
                        ),
                    )
                    others = tuple(
                        sorted(
                            (session for session in sessions if session.id != focus.id),
                            key=lambda session: (_session_tier(session), session.id),
                        )
                    )
            if reason == SpeechReason.CLICK and not any(
                session.waiting_for_input
                or (session.turn_active and not session.activity_uncertain)
                for session in ((focus, *others) if focus else ())
            ):
                subject, focus, others = None, None, ()
        situation: SpeechSituation
        if context is not None:
            situation = SpeechSituation.EXPLICIT
        elif reason == SpeechReason.CLICK:
            situation = (
                SpeechSituation.CLICK_WAITING
                if focus and focus.waiting_for_input
                else SpeechSituation.CLICK_WORKING
                if focus
                else SpeechSituation.CLICK_IDLE
            )
        elif reason == SpeechReason.CARE:
            assert isinstance(trigger, CareTrigger)
            situation = {
                CareKind.EXERCISE: SpeechSituation.CARE_EXERCISE,
                CareKind.ATTENTION: SpeechSituation.CARE_ATTENTION,
                CareKind.CONSTERNATION: SpeechSituation.CARE_CONSTERNATION,
            }[trigger.care]
        elif reason == SpeechReason.CHECK_IN:
            situation = (
                SpeechSituation.CHECK_IN_WAITING
                if focus and focus.waiting_for_input
                else SpeechSituation.CHECK_IN_RUNNING
                if focus and focus.active_tools
                else SpeechSituation.CHECK_IN_QUIET
            )
        else:
            situation = {
                SpeechReason.PROMPT: SpeechSituation.PROMPT,
                SpeechReason.COMPLETION: SpeechSituation.COMPLETION,
                SpeechReason.FAILURE: SpeechSituation.FAILURE,
                SpeechReason.ROUTINE: SpeechSituation.ROUTINE,
                SpeechReason.EXPLICIT: SpeechSituation.EXPLICIT,
            }[reason]
        return AttentionSelection(
            subject,
            deepcopy(focus),
            deepcopy(others),
            reason,
            situation,
            context,
            SelectionSupport.capture(subject, focus),
        )

    def request(
        self,
        snapshot: PetContext,
        *,
        check_in: Collection[str] = (),
        manual: tuple[str, float | None] | None = None,
        clicked: bool = False,
        care: CareKind | None = None,
        reaction_relevant: Callable[[SessionReaction], bool] = lambda trigger: True,
        delivery_pending: bool = False,
    ) -> AttentionRequest | None:
        now = self.clock()
        if manual is not None:
            return AttentionRequest(SpeechReason.EXPLICIT, manual[0], manual[1])
        if clicked:
            return AttentionRequest(SpeechReason.CLICK)
        if delivery_pending or snapshot.sleeping:
            return None
        if self.pending and now >= self.pending[1]:
            reason, _, trigger = self.pending
            self.pending = None
            if not isinstance(trigger, ReactionTrigger) or reaction_relevant(
                trigger.reaction
            ):
                return AttentionRequest(reason, trigger=trigger)
        if check_in:
            return AttentionRequest(SpeechReason.CHECK_IN, eligible=frozenset(check_in))
        if now >= self.next_observation:
            self.next_observation = now + self.rng.uniform(
                *self.config.observation_seconds
            )
            if any(
                self.routine_eligible(subject, snapshot) for subject in snapshot.work
            ):
                return AttentionRequest(SpeechReason.ROUTINE)
        if care is not None and not any(
            session.turn_active for session in snapshot.sessions
        ):
            return AttentionRequest(SpeechReason.CARE, trigger=CareTrigger(care))
        return None

    def restore(self, history: Mapping[str, HistoryRecord]) -> None:
        self.selections = {
            identifier: row.last_selected for identifier, row in history.items()
        }
        self.considered = {
            identifier: row.considered_revision for identifier, row in history.items()
        }
        self.sequence = max(self.selections.values(), default=0)

    def is_new(self, subject: WorkSubject) -> bool:
        revision = json.dumps(subject.revision, sort_keys=True, separators=(",", ":"))
        return self.considered.get(subject.id) != revision

    def mark_considered(self, subject: WorkSubject) -> None:
        self.considered[subject.id] = json.dumps(
            subject.revision, sort_keys=True, separators=(",", ":")
        )

    def history(self, subject_id: str) -> tuple[float, str | None]:
        return self.selections.get(subject_id, 0), self.considered.get(subject_id)

    def mark_selected(self, subject_id: str) -> None:
        self.sequence = max(self.clock(), self.sequence + 0.000001)
        self.selections[subject_id] = self.sequence
        log.debug("attention.selected", session=subject_id)

    def allowed(self, reason: SpeechReason, trigger: Trigger | None = None) -> bool:
        if (
            reason in (SpeechReason.CLICK, SpeechReason.EXPLICIT)
            or isinstance(trigger, ReactionTrigger)
            and trigger.reaction.kind == "waiting"
        ):
            return True
        return (
            self.clock() - self.last_attempt
            >= self.config.behavior.automatic_speech_cooldown_seconds
        )

    def scheduled(self, subject_id: str | None = None) -> None:
        self.last_attempt = self.clock()
        self.next_observation = self.last_attempt + self.rng.uniform(
            *self.config.observation_seconds
        )
        if subject_id is not None:
            self.mark_selected(subject_id)
