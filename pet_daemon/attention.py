import time
from collections.abc import Callable, Collection, Mapping
from copy import deepcopy

from pydantic import BaseModel

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
    AgentSessionStatus,
    CareTrigger,
    PetContext,
    RandomSource,
    ReactionTrigger,
    SessionTrigger,
    SubjectTrigger,
    Trigger,
)
from .sessions.reactions import ReactionKind, SessionReaction
from .storage.memory import AttentionHistory, SourceAttention
from .work import WorkSubject
from .work.narrative import SelectedMemory
from .work.shared_work import WorkUpdate

log = get_logger("pet-daemon")

COMPLETION_GRACE_SECONDS = 300
IDLE_REFLECTION_DELAY_SECONDS = 300
IDLE_REFLECTION_INTERVAL_SECONDS = 900
COMPLETION_REFLECTION_SECONDS = 900


def memory_spacing(age: float) -> float:
    if age < 3600:
        return 900
    if age < 86400:
        return 3600
    return 86400


def _latest_request(subject: WorkSubject) -> float:
    return max(
        (record.observed_at or 0 for record in subject.story.requests), default=0
    )


def _conversation(snapshot: PetContext) -> WorkSubject | None:
    roots = [subject for subject in snapshot.work if subject.parent is None]
    if not roots:
        return None
    return max(
        roots,
        key=lambda subject: (_latest_request(subject), subject.changed_at, subject.id),
    )


class AttentionRequest(BaseModel):
    reason: SpeechReason
    context: str | None = None
    ttl: float | None = None
    trigger: Trigger | None = None
    eligible: frozenset[SessionIdentity] | None = None
    reflection: bool = False


class SelectedWorkState(BaseModel):
    subject_revision: str | None = None
    session_state: tuple[int, bool, bool, ExecutionOutcome | None] | tuple[()] = ()

    @classmethod
    def capture(
        cls,
        subject: WorkSubject | None,
        focus: AgentSessionStatus | None,
    ) -> SelectedWorkState:
        return cls(
            subject_revision=subject.revision if subject else None,
            session_state=(
                focus.generation,
                focus.turn_active,
                focus.waiting_for_input,
                focus.outcome,
            )
            if focus
            else (),
        )


class AttentionSelection(BaseModel):
    subject: WorkSubject | None
    focus_session: AgentSessionStatus | None
    context_sessions: tuple[AgentSessionStatus, ...]
    reason: SpeechReason
    situation: SpeechSituation
    context: str | None = None
    selected_state: SelectedWorkState = SelectedWorkState()
    update: WorkUpdate | None = None
    memory: SelectedMemory | None = None
    quiet_revision: tuple[tuple[SessionIdentity, float], ...] = ()

    @property
    def id(self) -> SessionIdentity | None:
        return (
            self.subject.id
            if self.subject is not None
            else self.focus_session.id
            if self.focus_session
            else None
        )

    def is_valid(self, snapshot: PetContext) -> bool:
        if self.quiet_revision and self.quiet_revision != activity_revision(snapshot):
            return False
        current = self.refresh(snapshot)
        if self.subject is not None and current.subject is None:
            return False
        if self.focus_session is not None:
            if current.focus_session is None:
                return False
            if (
                self.selected_state.session_state
                != current.selected_state.session_state
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
        return self.model_copy(
            update={
                "subject": subject,
                "selected_state": SelectedWorkState.capture(subject, focus),
                "focus_session": deepcopy(focus),
                "context_sessions": tuple(
                    deepcopy(sessions[item.id])
                    for item in self.context_sessions
                    if item.id in sessions
                ),
            }
        )


def _sessions(subject: WorkSubject, snapshot: PetContext) -> list[AgentSessionStatus]:
    identifiers = set(subject.sessions)
    if subject.kind == WorkSubjectKind.SESSION:
        identifiers.add(subject.id)
    return [session for session in snapshot.sessions if session.id in identifiers]


def activity_revision(
    snapshot: PetContext,
) -> tuple[tuple[SessionIdentity, float], ...]:
    times = {
        subject.id: max(
            (
                record.observed_at or 0
                for record in (*subject.story.requests, *subject.story.activity)
            ),
            default=0,
        )
        for subject in snapshot.work
    }
    for session in snapshot.sessions:
        times[session.id] = max(times.get(session.id, 0), session.last_hook_at)
    return tuple(sorted(times.items()))


def _session_tier(session: AgentSessionStatus) -> int:
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
        history: Mapping[SessionIdentity, AttentionHistory] | None = None,
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
        self.histories: dict[SessionIdentity, AttentionHistory] = {}
        self.sources: dict[str, SourceAttention] = {}
        self.last_reflection_at = float("-inf")
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
        self.sources.clear()
        self.last_reflection_at = float("-inf")
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

    def is_working(self, session: AgentSessionStatus) -> bool:
        if session.retired or session.presence in (
            ProcessStatus.EXITED,
            ProcessStatus.REPLACED,
        ):
            return False
        if not session.turn_active:
            return False
        has_recent_activity = (
            self.wall_clock() - session.last_hook_at
            < self.config.activity_fresh_seconds
            if session.last_hook_at is not None
            else session.quiet_seconds < self.config.activity_fresh_seconds
        )
        return session.presence == ProcessStatus.ALIVE or has_recent_activity

    def is_update_eligible(
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
            if session and self.is_working(session):
                return True
            decision = "insufficient_liveness"
        elif any(
            self.history(identifier).considered_session_revision == update.revision
            for identifier in (update.session, *update.aliases)
        ):
            decision = "consumed"
        elif update.ended_at is not None:
            subject = next(
                (item for item in snapshot.work if update in item.updates), None
            )
            if subject is not None and update.ended_at < _latest_request(subject):
                return False
            if self.wall_clock() < update.ended_at + COMPLETION_GRACE_SECONDS:
                return True
            decision = "expired"
        elif session and self.is_working(session):
            return True
        else:
            decision = "insufficient_liveness"
        log.debug(
            "attention.ineligible",
            detail="payload",
            session=update.session,
            decision=decision,
            check_in=check_in,
            runtime_session_present=session is not None,
            presence=session.presence if session else None,
            retired=session.retired if session else None,
            turn_active=session.turn_active if session else None,
            update_observed_at=update.observed_at,
            update_ended_at=update.ended_at,
        )
        return False

    def is_selection_eligible(
        self, selection: AttentionSelection, snapshot: PetContext
    ) -> bool:
        if selection.reason in (
            SpeechReason.EXPLICIT,
            SpeechReason.CLICK,
            SpeechReason.CARE,
        ):
            return True
        if selection.situation == SpeechSituation.IDLE_REFLECTION:
            return (
                selection.subject is not None
                and self.is_quiet(snapshot)
                and selection.is_valid(snapshot)
                and (
                    selection.memory is None
                    or self.is_memory_eligible(selection.memory)
                )
            )
        if selection.update is not None:
            subject = next(
                (item for item in snapshot.work if item.id == selection.id), None
            )
            return bool(
                subject
                and selection.update in subject.updates
                and self.is_update_eligible(
                    selection.update,
                    snapshot,
                    check_in=selection.reason == SpeechReason.CHECK_IN,
                )
            )
        if selection.subject and selection.subject.updates:
            return False
        session = selection.focus_session
        return bool(session and self.is_working(session))

    def is_routine_eligible(self, subject: WorkSubject, snapshot: PetContext) -> bool:
        if subject.updates:
            return any(
                self.is_update_eligible(update, snapshot) for update in subject.updates
            )
        if not self.has_unconsidered_changes(subject):
            return False
        return any(self.is_working(session) for session in _sessions(subject, snapshot))

    def select(
        self,
        snapshot: PetContext,
        reason: SpeechReason = SpeechReason.ROUTINE,
        context: str | None = None,
        eligible: Collection[SessionIdentity] | None = None,
        *,
        reflection: bool = False,
    ) -> AttentionSelection:
        if reflection or context is not None:
            return self.select_recollection(snapshot, reason, context)
        trigger = snapshot.trigger
        subject = None
        focus = None
        others: tuple[AgentSessionStatus, ...] = ()
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
                or self.is_routine_eligible(subject, snapshot)
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
                        and self.is_update_eligible(
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
                    if (not automatic or self.is_working(session))
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
                return self.select_recollection(snapshot, reason)
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
            subject=subject,
            focus_session=deepcopy(focus),
            context_sessions=deepcopy(others),
            reason=reason,
            situation=situation,
            context=context,
            selected_state=SelectedWorkState.capture(subject, focus),
            update=update,
        )

    def select_recollection(
        self,
        snapshot: PetContext,
        reason: SpeechReason,
        context: str | None = None,
        *,
        subject: WorkSubject | None = None,
    ) -> AttentionSelection:
        subject = subject or _conversation(snapshot)
        return AttentionSelection(
            subject=subject,
            focus_session=None,
            context_sessions=(),
            reason=reason,
            context=context,
            situation=SpeechSituation.EXPLICIT
            if context is not None
            else SpeechSituation.IDLE_REFLECTION
            if subject
            else SpeechSituation.CLICK_IDLE,
            selected_state=SelectedWorkState.capture(subject, None),
            quiet_revision=() if context is not None else activity_revision(snapshot),
        )

    def recollection_alternatives(
        self, snapshot: PetContext, selection: AttentionSelection
    ) -> tuple[AttentionSelection, ...]:
        subjects = sorted(
            (
                subject
                for subject in snapshot.work
                if subject.parent is None and subject.id != selection.id
            ),
            key=lambda subject: (
                self.history(subject.id).last_selected,
                -subject.changed_at,
                subject.id,
            ),
        )
        return tuple(
            self.select_recollection(snapshot, selection.reason, subject=subject)
            for subject in subjects
        )

    def is_quiet(self, snapshot: PetContext) -> bool:
        if any(
            (session.active_tools or session.waiting_for_input)
            and not session.retired
            and session.presence not in (ProcessStatus.EXITED, ProcessStatus.REPLACED)
            for session in snapshot.sessions
        ):
            return False
        activity = activity_revision(snapshot)
        return (
            bool(activity)
            and self.wall_clock() - max(at for _, at in activity)
            >= IDLE_REFLECTION_DELAY_SECONDS
        )

    def is_memory_eligible(self, memory: SelectedMemory) -> bool:
        age = self.wall_clock() - memory.observed_at
        history = self.sources.get(memory.key, SourceAttention())
        if not memory.is_ongoing:
            return (
                age < COMPLETION_REFLECTION_SECONDS
                and history.considered_at is None
                and history.mentioned_at is None
            )
        previous = max(
            (
                value
                for value in (history.considered_at, history.mentioned_at)
                if value is not None
            ),
            default=None,
        )
        return previous is None or self.wall_clock() - previous >= memory_spacing(age)

    def restore_sources(self, histories: Mapping[str, SourceAttention]) -> None:
        self.sources = {key: history.model_copy() for key, history in histories.items()}

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
            return AttentionRequest(
                reason=SpeechReason.EXPLICIT, context=manual[0], ttl=manual[1]
            )
        if clicked:
            return AttentionRequest(reason=SpeechReason.CLICK)
        if delivery_pending or snapshot.sleeping:
            return None
        if self.pending and now >= self.pending[1]:
            reason, _, trigger = self.pending
            self.pending = None
            if not isinstance(trigger, ReactionTrigger) or reaction_relevant(
                trigger.reaction
            ):
                return AttentionRequest(reason=reason, trigger=trigger)
        live_check_in = any(
            session.id in check_in
            and (session.waiting_for_input or session.active_tools)
            for session in snapshot.sessions
        )
        if live_check_in or check_in and not self.is_quiet(snapshot):
            return AttentionRequest(
                reason=SpeechReason.CHECK_IN, eligible=frozenset(check_in)
            )
        if now >= self.next_observation:
            self.next_observation = now + self.rng.uniform(
                *self.config.observation_seconds
            )
            if any(
                self.is_routine_eligible(subject, snapshot) for subject in snapshot.work
            ):
                return AttentionRequest(reason=SpeechReason.ROUTINE)
        if (
            self.is_quiet(snapshot)
            and _conversation(snapshot) is not None
            and self.wall_clock() - self.last_reflection_at
            >= IDLE_REFLECTION_INTERVAL_SECONDS
        ):
            self.last_reflection_at = self.wall_clock()
            conversation = _conversation(snapshot)
            assert conversation is not None
            history = self.histories.setdefault(conversation.id, AttentionHistory())
            history.last_reflection_at = self.last_reflection_at
            return AttentionRequest(reason=SpeechReason.CHECK_IN, reflection=True)
        if care is not None and not any(
            session.turn_active for session in snapshot.sessions
        ):
            return AttentionRequest(
                reason=SpeechReason.CARE, trigger=CareTrigger(care=care)
            )
        return None

    def restore(self, history: Mapping[SessionIdentity, AttentionHistory]) -> None:
        self.histories = {
            identifier: record.model_copy() for identifier, record in history.items()
        }
        self.sequence = max((row.last_selected for row in history.values()), default=0)
        self.last_reflection_at = max(
            (
                row.last_reflection_at
                for row in history.values()
                if row.last_reflection_at is not None
            ),
            default=float("-inf"),
        )

    def has_unconsidered_changes(self, subject: WorkSubject) -> bool:
        return self.history(subject.id).considered_subject_revision != subject.revision

    def consideration(
        self, subject: WorkSubject | None, update: WorkUpdate | None = None
    ) -> dict[SessionIdentity, AttentionHistory]:
        records = {}
        if subject:
            record = self.history(subject.id).model_copy()
            record.considered_subject_revision = subject.revision
            records[subject.id] = record
        if update:
            record = records.get(update.session)
            if record is None:
                record = self.history(update.session).model_copy()
                records[update.session] = record
            record.considered_session_revision = update.revision
        return records

    def mark_considered(
        self, subject: WorkSubject | None, update: WorkUpdate | None = None
    ) -> None:
        self.histories.update(self.consideration(subject, update))

    def history(self, identifier: SessionIdentity) -> AttentionHistory:
        return self.histories.get(identifier, AttentionHistory())

    def mark_selected(self, subject_id: SessionIdentity) -> None:
        self.sequence = max(self.clock(), self.sequence + 0.000001)
        record = self.histories.setdefault(subject_id, AttentionHistory())
        record.last_selected = self.sequence
        log.debug("attention.selected", session=subject_id)

    def can_speak(self, reason: SpeechReason, trigger: Trigger | None = None) -> bool:
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

    def mark_scheduled(self, subject_id: SessionIdentity | None = None) -> None:
        self.last_attempt = self.clock()
        self.next_observation = self.last_attempt + self.rng.uniform(
            *self.config.observation_seconds
        )
        if subject_id is not None:
            self.mark_selected(subject_id)
