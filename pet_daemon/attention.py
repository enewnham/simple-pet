import time
from collections.abc import Callable, Collection, Mapping
from copy import deepcopy
from dataclasses import dataclass, replace

from .config import Config
from .domain import (
    CareKind,
    ExecutionOutcome,
    ProcessStatus,
    SpeechReason,
    SpeechSituation,
    WorkSubjectKind,
)
from .identity import SessionIdentity
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
from .sessions.reactions import ReactionKind, SessionReaction
from .storage.memory import HistoryRecord
from .work import WorkSubject
from .work.views import WorkUpdate

log = get_logger("pet-daemon")

COMPLETION_GRACE_SECONDS = 300


@dataclass(frozen=True)
class AttentionRequest:
    reason: SpeechReason
    context: str | None = None
    ttl: float | None = None
    trigger: Trigger | None = None
    eligible: frozenset[SessionIdentity] | None = None


@dataclass(frozen=True)
class SelectionSupport:
    subject_revision: str | None = None
    session_transition: tuple[int, bool, bool, ExecutionOutcome | None] | tuple[()] = ()

    @classmethod
    def capture(
        cls,
        subject: WorkSubject | None,
        focus: SessionContext | None,
    ) -> SelectionSupport:
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
    update: WorkUpdate | None = None

    @property
    def id(self) -> SessionIdentity | None:
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

    def refresh(self, snapshot: PetContext) -> AttentionSelection:
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


def _target_session(trigger: Trigger | None) -> SessionIdentity | None:
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
        history: Mapping[SessionIdentity, HistoryRecord] | None = None,
        *,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self.config = config
        self.wall_clock = wall_clock
        self.clock = clock
        self.rng = rng
        self.pending: tuple[SpeechReason, float, Trigger | None] | None = None
        self.last_attempt = float("-inf")
        self.next_observation = clock() + rng.uniform(*config.observation_seconds)
        self.histories: dict[SessionIdentity, HistoryRecord] = {}
        self.sequence: float = 0
        if history:
            self.restore(history)

    def reset(self) -> None:
        self.pending = None
        self.last_attempt = float("-inf")
        self.next_observation = self.clock() + self.rng.uniform(
            *self.config.observation_seconds
        )
        self.histories.clear()
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

    def live(self, session: SessionContext) -> bool:
        if session.retired or session.presence in (
            ProcessStatus.EXITED,
            ProcessStatus.REPLACED,
        ):
            return False
        if not session.turn_active:
            return False
        recent = (
            self.wall_clock() - session.last_hook_at
            < self.config.activity_fresh_seconds
            if session.last_hook_at is not None
            else session.quiet_seconds < self.config.activity_fresh_seconds
        )
        return session.presence == ProcessStatus.ALIVE or recent

    def update_eligible(
        self, update: WorkUpdate, snapshot: PetContext, *, check_in: bool = False
    ) -> bool:
        session = next(
            (item for item in snapshot.sessions if item.id == update.session), None
        )
        if session and (
            session.retired
            or session.presence in (ProcessStatus.EXITED, ProcessStatus.REPLACED)
        ):
            decision = "retired"
        elif check_in:
            if session and self.live(session):
                return True
            decision = "insufficient_liveness"
        elif any(
            self.history(identifier).considered_session_revision == update.revision
            for identifier in (update.session, *update.aliases)
        ):
            decision = "consumed"
        elif update.ended_at is not None:
            if self.wall_clock() < update.ended_at + COMPLETION_GRACE_SECONDS:
                return True
            decision = "expired"
        elif session and self.live(session):
            return True
        else:
            decision = "insufficient_liveness"
        log.debug("attention.ineligible", session=update.session, decision=decision)
        return False

    def selection_eligible(
        self, selection: AttentionSelection, snapshot: PetContext
    ) -> bool:
        if selection.reason in (
            SpeechReason.EXPLICIT,
            SpeechReason.CLICK,
            SpeechReason.CARE,
        ):
            return True
        if selection.update is not None:
            subject = next(
                (item for item in snapshot.work if item.id == selection.id), None
            )
            return bool(
                subject
                and selection.update in subject.updates
                and self.update_eligible(
                    selection.update,
                    snapshot,
                    check_in=selection.reason == SpeechReason.CHECK_IN,
                )
            )
        if selection.subject and selection.subject.updates:
            return False
        session = selection.focus_session
        return bool(session and self.live(session))

    def routine_eligible(self, subject: WorkSubject, snapshot: PetContext) -> bool:
        if subject.updates:
            return any(
                self.update_eligible(update, snapshot) for update in subject.updates
            )
        if not self.is_new(subject):
            return False
        return any(self.live(session) for session in _sessions(subject, snapshot))

    def select(
        self,
        snapshot: PetContext,
        reason: SpeechReason = SpeechReason.ROUTINE,
        context: str | None = None,
        eligible: Collection[SessionIdentity] | None = None,
    ) -> AttentionSelection:
        trigger = snapshot.trigger
        subject = None
        focus = None
        others: tuple[SessionContext, ...] = ()
        update = None
        automatic = reason not in (
            SpeechReason.EXPLICIT,
            SpeechReason.CLICK,
            SpeechReason.CARE,
        )
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
            updates: dict[SessionIdentity, WorkUpdate] = {}
            for item in subjects:
                sessions = _sessions(item, snapshot)
                if automatic and item.updates:
                    available = [
                        value
                        for value in item.updates
                        if (target_session is None or value.session == target_session)
                        and (
                            eligible is None
                            or value.session in eligible
                            or item.id in eligible
                        )
                        and self.update_eligible(
                            value, snapshot, check_in=reason == SpeechReason.CHECK_IN
                        )
                    ]
                    if not available:
                        continue
                    tiers = {session.id: _session_tier(session) for session in sessions}
                    updates[item.id] = min(
                        available,
                        key=lambda value: (
                            tiers.get(value.session, 5),
                            -value.observed_at,
                            value.session,
                        ),
                    )
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
                        self.history(item[0].id).last_selected,
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
                update = updates.get(subject.id)
                preferred = [
                    session
                    for session in sessions
                    if update and session.id == update.session
                ]
                preferred = preferred or [
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
                            self.history(session.id).last_selected,
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
                    if (not automatic or self.live(session))
                    and (
                        not automatic
                        or not any(
                            value.session == session.id
                            for subject in snapshot.work
                            for value in subject.updates
                        )
                    )
                    and (eligible is None or session.id in eligible)
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
                            self.history(session.id).last_selected,
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
            update,
        )

    def request(
        self,
        snapshot: PetContext,
        *,
        check_in: Collection[SessionIdentity] = (),
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

    def restore(self, history: Mapping[SessionIdentity, HistoryRecord]) -> None:
        self.histories = dict(history)
        self.sequence = max((row.last_selected for row in history.values()), default=0)

    def is_new(self, subject: WorkSubject) -> bool:
        return self.history(subject.id).considered_subject_revision != subject.revision

    def consideration(
        self, subject: WorkSubject | None, update: WorkUpdate | None = None
    ) -> dict[SessionIdentity, HistoryRecord]:
        records = {}
        if subject:
            records[subject.id] = replace(
                self.history(subject.id), considered_subject_revision=subject.revision
            )
        if update:
            member = records.get(update.session, self.history(update.session))
            records[update.session] = replace(
                member, considered_session_revision=update.revision
            )
        return records

    def mark_considered(
        self, subject: WorkSubject | None, update: WorkUpdate | None = None
    ) -> None:
        self.histories.update(self.consideration(subject, update))

    def history(self, identifier: SessionIdentity) -> HistoryRecord:
        return self.histories.get(identifier, HistoryRecord())

    def mark_selected(self, subject_id: SessionIdentity) -> None:
        self.sequence = max(self.clock(), self.sequence + 0.000001)
        self.histories[subject_id] = replace(
            self.history(subject_id), last_selected=self.sequence
        )
        log.debug("attention.selected", session=subject_id)

    def allowed(self, reason: SpeechReason, trigger: Trigger | None = None) -> bool:
        if (
            reason in (SpeechReason.CLICK, SpeechReason.EXPLICIT)
            or isinstance(trigger, ReactionTrigger)
            and trigger.reaction.kind == ReactionKind.WAITING
        ):
            return True
        return (
            self.clock() - self.last_attempt
            >= self.config.behavior.automatic_speech_cooldown_seconds
        )

    def scheduled(self, subject_id: SessionIdentity | None = None) -> None:
        self.last_attempt = self.clock()
        self.next_observation = self.last_attempt + self.rng.uniform(
            *self.config.observation_seconds
        )
        if subject_id is not None:
            self.mark_selected(subject_id)
