from __future__ import annotations

import time
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from uuid import UUID

from ..domain import Activity, ExecutionOutcome, PetState, ProcessStatus, SessionSource
from ..identity import SessionIdentity
from ..logging import get_logger
from ..prompts import select_prompt
from ..prose import prose_text
from ..repository_identity import parse_remote_repository
from ..runtime import (
    ActiveTool,
    ConvergenceItem,
    FailureSnapshot,
    HookEventKind,
    ProcessIdentity,
    SessionContext,
    SwarmActivity,
    UserSentiment,
    WaitingKind,
)
from ..work.models import ExecutionEvidence, IntentEvidence, ProgressEvidence
from ..work.observations import SessionObservation
from .activity import activity

log = get_logger("pet-daemon")

if TYPE_CHECKING:
    from ..runtime import HookEvent

SENTIMENT_CONTEXT_SECONDS = 300


@dataclass
class Failure:
    tool: str
    activity: Activity
    at: float
    count: int
    resolved: bool
    generation: int
    goal: str


@dataclass(frozen=True)
class SessionUpdate:
    failure: bool
    convergence: bool
    session: SessionIdentity
    activity: Activity
    observation: SessionObservation


@dataclass
class Call:
    tool: str
    started: float
    activity: Activity = Activity.COMMAND


@dataclass
class AgentSession:
    id: SessionIdentity
    started: float
    last_seen: float
    changed_at: float = field(default_factory=time.time)
    last_hook_at: float = 0
    retired: bool = False
    last_event: HookEventKind | None = None
    parent: SessionIdentity | None = None
    goal: str = ""
    intent_received: bool = False
    user_sentiment: float | None = None
    sentiment_seen: float = float("-inf")
    sentiment_identity: tuple[str | None, str] | None = None
    repo: str = ""
    phase: Activity | None = None
    last_tool: str = ""
    progress: str = ""
    failure: Failure | None = None
    failures: dict[Activity, Failure] = field(default_factory=dict)
    calls: dict[str, Call] = field(default_factory=dict)
    active: bool = False
    finished: bool = False
    turn: str | None = None
    generation: int = 0
    outcome: ExecutionOutcome | None = None
    ended: float | None = None
    cancelled: bool = False
    waiting: bool = False
    waiting_call: str | None = None
    waiting_since: float | None = None
    waiting_kind: WaitingKind | None = None
    waiting_tool: str | None = None
    waiting_prompt: str | None = None
    last_check_in: float | None = None
    owner: ProcessIdentity | None = None
    presence: ProcessStatus = ProcessStatus.UNKNOWN
    owner_checked: float | None = None
    owner_last_alive: float | None = None
    activity_uncertain: bool = False

    @property
    def source(self) -> SessionSource:
        return self.id.source

    @property
    def waiting_for_input(self) -> bool:
        return self.waiting and self.waiting_kind != WaitingKind.PERMISSION_CHECK

    @property
    def waiting_started(self) -> float:
        assert self.waiting_since is not None
        return self.waiting_since

    @property
    def working(self) -> bool:
        return (
            self.active
            and not self.activity_uncertain
            and (
                not self.waiting
                or any(call != self.waiting_call for call in self.calls)
            )
        )


class Sessions:
    def __init__(
        self,
        clock: Callable[[], float],
        stale_seconds: float = 3600,
        retention_seconds: float = 7200,
        fresh_seconds: float = 300,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self.clock = clock
        self.wall_clock = wall_clock
        self.quiet_seconds = fresh_seconds
        self.stale_seconds = stale_seconds
        self.retention_seconds = retention_seconds
        self.by_id: dict[SessionIdentity, AgentSession] = {}
        self.aliases: dict[SessionIdentity, SessionIdentity] = {}
        self.child_calls: dict[tuple[SessionIdentity, str], SessionIdentity] = {}
        self.members: set[SessionIdentity] = set()
        self.convergence: tuple[ConvergenceItem, ...] | None = None

    @property
    def working(self) -> bool:
        self._refresh_activity()
        return any(session.working for session in self.by_id.values())

    @property
    def engaged(self) -> bool:
        self._refresh_activity()
        return any(
            session.active and not session.activity_uncertain
            for session in self.by_id.values()
        )

    @property
    def active(self) -> bool:
        return any(session.active for session in self.by_id.values())

    @property
    def failed(self) -> bool:
        return any(self._current_failure(session) for session in self.by_id.values())

    def _current_failure(self, session: AgentSession) -> bool:
        return any(
            failure.generation == session.generation and not failure.resolved
            for failure in session.failures.values()
        )

    def _failure(
        self,
        session: AgentSession,
        category: Activity,
        now: float,
        tool: str = "",
    ) -> Failure:
        previous = session.failures.get(category)
        count = previous.count + 1 if previous and not previous.resolved else 1
        failure = Failure(
            tool=tool,
            activity=category,
            at=now,
            count=count,
            resolved=False,
            generation=session.generation,
            goal=session.goal,
        )
        session.failures[category] = failure
        return failure

    @property
    def clean_convergence(self) -> bool:
        return (
            self.convergence is not None
            and not self.active
            and all(
                item.outcome == ExecutionOutcome.SUCCESS for item in self.convergence
            )
        )

    def _key(self, source: SessionSource, identifier: str) -> SessionIdentity:
        identity = SessionIdentity(source, identifier)
        return self.aliases.get(identity, identity)

    def _session(
        self, event: HookEvent, now: float
    ) -> tuple[SessionIdentity, AgentSession] | None:
        source = event.source
        identifier = event.session
        key = self._key(source, identifier)
        routed = False
        parent_call = event.parent_call
        if parent_call and event.event in (
            HookEventKind.TOOL_START,
            HookEventKind.TOOL_DONE,
            HookEventKind.TOOL_ERROR,
            HookEventKind.WAITING,
        ):
            target = self.child_calls.get(
                (SessionIdentity(source, identifier), parent_call)
            )
            if target and target != key:
                key = target
                routed = True
        alternate = event.aliases
        matches = [self._key(source, alias) for alias in alternate]
        if key not in self.by_id:
            key = next((match for match in matches if match in self.by_id), key)
        parent = event.parent
        if parent and self._key(source, parent) in {key, *matches}:
            return self._ignored(event, "invalid_parent")
        created = key not in self.by_id
        session = self.by_id.setdefault(
            key, AgentSession(key, now, now, self.wall_clock())
        )
        for alias in alternate if routed else [identifier, *alternate]:
            old_key = self._key(source, alias)
            if old_key != key and old_key in self.by_id:
                other = self.by_id.pop(old_key)
                if not session.active and (
                    other.active or (other.finished and not session.finished)
                ):
                    session, other = other, session
                    session.id = key
                    self.by_id[key] = session
                session.calls.update(other.calls)
                session.active = session.active or other.active
                session.goal = session.goal or other.goal
                session.intent_received = (
                    session.intent_received or other.intent_received
                )
                if other.sentiment_seen > session.sentiment_seen:
                    session.user_sentiment = other.user_sentiment
                    session.sentiment_seen = other.sentiment_seen
                    session.sentiment_identity = other.sentiment_identity
                session.repo = session.repo or other.repo
                session.owner = session.owner or other.owner
                if session.owner == other.owner:
                    session.presence = other.presence
                if other.active:
                    adopt_waiting = other.waiting and (
                        not session.waiting
                        or other.waiting_started >= session.waiting_started
                    )
                    if adopt_waiting:
                        session.waiting = True
                        session.waiting_call = other.waiting_call
                        session.waiting_since = other.waiting_since
                        session.waiting_kind = other.waiting_kind
                        session.waiting_tool = other.waiting_tool
                        session.waiting_prompt = other.waiting_prompt
                        session.last_check_in = other.last_check_in
                    if adopt_waiting or other.last_seen > session.last_seen:
                        if other.phase:
                            session.phase = other.phase
                        session.last_tool = other.last_tool or session.last_tool
                        session.progress = other.progress or session.progress
                    if other.last_seen > session.last_seen:
                        session.last_event = other.last_event
                        session.changed_at = other.changed_at
                    session.last_seen = max(session.last_seen, other.last_seen)
                    for category, failure in other.failures.items():
                        if failure.generation == other.generation:
                            failure.generation = session.generation
                            existing = session.failures.get(category)
                            if existing is None or failure.at > existing.at:
                                session.failures[category] = failure
                    unresolved = [
                        item for item in session.failures.values() if not item.resolved
                    ]
                    if unresolved:
                        session.failure = max(unresolved, key=lambda item: item.at)
                session.started = min(session.started, other.started)
                self.aliases = {
                    name: key if target == old_key else target
                    for name, target in self.aliases.items()
                }
                self.child_calls = {
                    name: key if target == old_key else target
                    for name, target in self.child_calls.items()
                }
                for child in self.by_id.values():
                    if child.parent == old_key:
                        child.parent = key
                if old_key in self.members:
                    self.members.remove(old_key)
                    self.members.add(key)
                log.info(
                    "session.merged",
                    session=key,
                    merged_session=old_key,
                    active=session.active,
                    active_calls=list(session.calls),
                    parent=session.parent,
                )
            alias_key = SessionIdentity(source, alias)
            if alias_key != key and self.aliases.get(alias_key) != key:
                log.info("session.aliased", session=key, alias=alias_key)
            self.aliases[alias_key] = key
        if parent:
            session.parent = self._key(source, parent)
            if parent_call:
                self.child_calls[(SessionIdentity(source, parent), parent_call)] = key
            parent_session = self.by_id.get(session.parent)
            if parent_session and not session.goal:
                session.goal = parent_session.goal
            if parent_session and not session.repo:
                session.repo = parent_session.repo
            if parent_session and not session.owner:
                session.owner = parent_session.owner
                session.presence = parent_session.presence
        if created:
            log.info(
                "session.created",
                session=key,
                parent=session.parent,
                source_event=event.event,
                repo=event.repo or session.repo,
                active_sessions=sum(item.active for item in self.by_id.values()),
            )
        return key, session

    def _open(
        self, key: SessionIdentity, session: AgentSession, event: HookEvent, now: float
    ) -> None:
        previous_turn = session.turn
        replaced_calls = list(session.calls)
        if not self.active:
            self.members.clear()
            self.convergence = None
        session.active = True
        session.retired = False
        session.activity_uncertain = False
        session.finished = False
        session.outcome = None
        session.cancelled = False
        session.ended = None
        session.generation += 1
        session.failures.clear()
        session.turn = (
            None if event.event == HookEventKind.SUBAGENT_START else event.turn
        )
        session.started = now
        session.calls.clear()
        session.last_tool = ""
        session.phase = None
        session.progress = ""
        session.waiting = False
        session.waiting_call = None
        session.waiting_since = None
        session.waiting_kind = None
        session.waiting_tool = None
        session.waiting_prompt = None
        session.last_check_in = None
        self.members.add(key)
        log.info(
            "session.turn_started",
            session=key,
            parent=session.parent,
            turn=session.turn,
            generation=session.generation,
            source_event=event.event,
            repo=session.repo or (event.repo if event.repo is not None else ""),
            previous_turn=previous_turn,
            replaced_calls=replaced_calls,
            active_sessions=sum(item.active for item in self.by_id.values()),
        )

    def _finish(
        self,
        key: SessionIdentity,
        session: AgentSession,
        outcome: ExecutionOutcome,
        now: float,
        reason: str,
    ) -> None:
        log.info(
            "session.turn_finished",
            session=key,
            parent=session.parent,
            turn=session.turn,
            generation=session.generation,
            outcome=outcome,
            reason=reason,
            duration_seconds=round(now - session.started, 2),
            active_calls=list(session.calls),
            active_sessions=sum(item.active for item in self.by_id.values())
            - int(session.active),
            active_children=[
                child_key
                for child_key, child in self.by_id.items()
                if child.parent == key and child.active
            ],
        )
        session.calls.clear()
        session.active = False
        session.finished = True
        session.ended = now
        session.changed_at = self.wall_clock()
        session.outcome = outcome
        if outcome == ExecutionOutcome.CANCELLED:
            for failure in session.failures.values():
                failure.resolved = True
        session.waiting = False
        session.waiting_call = None
        session.waiting_since = None
        session.waiting_kind = None
        session.waiting_tool = None
        session.waiting_prompt = None

    def _retire(
        self, key: SessionIdentity, session: AgentSession, now: float, reason: str
    ) -> None:
        log.info("session.activity_closed", session=key, reason=reason)
        session.active = False
        session.finished = True
        if session.ended is None:
            session.ended = now
        session.calls.clear()
        session.waiting = False
        session.waiting_call = None
        session.waiting_since = None
        session.waiting_kind = None
        session.waiting_tool = None
        session.waiting_prompt = None
        session.retired |= reason != HookEventKind.SESSION_END

    def _ignored(
        self,
        event: HookEvent,
        reason: str,
        key: SessionIdentity | None = None,
        session: AgentSession | None = None,
    ) -> None:
        log.debug(
            "session.event_ignored",
            source_event=event.event,
            reason=reason,
            session=key or event.subject,
            turn=event.turn,
            current_turn=session.turn if session else None,
            call=event.call,
        )

    def _accepted(
        self, key: SessionIdentity, session: AgentSession, event: HookEvent
    ) -> None:
        session.activity_uncertain = False
        session.last_event = event.event
        session.last_hook_at = self.wall_clock()
        if event.event not in (HookEventKind.SESSION_END, HookEventKind.SESSION_START):
            session.changed_at = session.last_hook_at
        log.debug(
            "session.event_accepted",
            detail="payload",
            session=key,
            source_session=event.session,
            source_event=event.event,
            turn=session.turn,
            generation=session.generation,
            call=event.call,
            tool=event.tool,
            active=session.active,
            waiting=session.waiting,
            active_calls=list(session.calls),
        )

    def _resumed(
        self,
        key: SessionIdentity,
        session: AgentSession,
        event: HookEvent,
        now: float,
        waiting_since: float,
    ) -> None:
        log.info(
            "session.resumed",
            session=key,
            parent=session.parent,
            turn=session.turn,
            source_event=event.event,
            waiting_seconds=round(now - waiting_since, 2),
            call=event.call,
        )

    def handle(self, event: object) -> SessionUpdate | None:
        from ..sources.hooks import normalize_hook

        hook = normalize_hook(event)
        if hook is None:
            log.debug("session.event_ignored", reason="invalid_event")
            return None
        return self.apply(hook)

    @staticmethod
    def _resumes_codex_child(session: AgentSession, event: HookEvent) -> bool:
        if not (
            session.source == SessionSource.CODEX
            and session.parent
            and session.finished
            and event.event == HookEventKind.TOOL_START
            and event.call
            and event.turn
            and session.turn
        ):
            return False
        try:
            previous, current = UUID(session.turn), UUID(event.turn)
        except ValueError:
            return False
        return previous.version == current.version == 7 and current.time > previous.time

    def apply(self, event: HookEvent) -> SessionUpdate | None:
        kind = event.event
        now = self.clock()
        previously_active = self.active
        identity = self._session(event, now)
        if identity is None:
            return None
        key, session = identity
        turn = event.turn
        resumes_child = self._resumes_codex_child(session, event)
        if (
            turn
            and session.turn
            and turn != session.turn
            and not resumes_child
            and kind
            not in (
                HookEventKind.INTENT,
                HookEventKind.SESSION_START,
                HookEventKind.SUBAGENT_START,
                HookEventKind.SUBAGENT_DONE,
                HookEventKind.SESSION_END,
            )
        ):
            return self._ignored(event, "turn_mismatch", key, session)
        owner = event.owner
        if owner and owner != session.owner:
            session.owner = owner
            session.presence = ProcessStatus.UNKNOWN
            session.owner_checked = None
            session.owner_last_alive = None
        waiting_since = session.waiting_since
        waiting_call = session.waiting_call
        waiting_kind = session.waiting_kind
        was_active = session.active
        was_finished = session.finished
        if event.repo:
            session.repo = event.repo
        if (
            kind == HookEventKind.SUBAGENT_START
            and session.active
            and session.intent_received
        ):
            return self._ignored(event, "duplicate_subagent_start", key, session)
        failure = False
        event_activity = session.phase or Activity.COMMAND
        if kind in (HookEventKind.INTENT, HookEventKind.SUBAGENT_START):
            new_native_turn = kind == HookEventKind.INTENT and not event.manual
            if (
                not session.active
                or new_native_turn
                or (kind == HookEventKind.INTENT and turn and turn != session.turn)
            ):
                self._open(key, session, event, now)
            if event.goal:
                request = select_prompt(prose_text(event.goal))
                if not session.intent_received:
                    session.goal = request
                session.intent_received = True
            elif kind == HookEventKind.SUBAGENT_START and session.parent in self.by_id:
                session.goal = self.by_id[session.parent].goal
                session.intent_received = bool(session.goal)
        elif kind == HookEventKind.TOOL_START:
            if not event.call:
                return self._ignored(event, "missing_call", key, session)
            if session.finished and not event.manual and not resumes_child:
                return self._ignored(event, "finished_session", key, session)
            if not session.active:
                self._open(key, session, event, now)
            if turn and session.turn is None:
                session.turn = turn
            call_id = event.call
            if call_id in session.calls:
                return self._ignored(event, "duplicate_call", key, session)
            tool = event.tool if event.tool is not None else ""
            event_activity = activity(
                tool, (event.tool_name if event.tool_name is not None else "")
            )
            session.calls[call_id] = Call(tool, now, event_activity)
            session.last_tool = tool
            session.phase = event_activity
            session.progress = event.progress if event.progress is not None else ""
        elif kind in (HookEventKind.TOOL_DONE, HookEventKind.TOOL_ERROR):
            call = session.calls.pop(event.call, None)
            if call is None:
                if (
                    session.waiting
                    and event.call
                    and event.call == session.waiting_call
                ):
                    tool = event.tool if event.tool is not None else session.last_tool
                    assert waiting_since is not None
                    call = Call(
                        tool,
                        waiting_since,
                        activity(
                            tool,
                            (event.tool_name if event.tool_name is not None else ""),
                        ),
                    )
                else:
                    return self._ignored(event, "unknown_call", key, session)
            event_activity = call.activity
            session.progress = (
                ""
                if kind == HookEventKind.TOOL_ERROR
                or event.outcome == ExecutionOutcome.ERROR
                else (
                    event.progress if event.progress is not None else session.progress
                )
            )
            if (
                kind == HookEventKind.TOOL_ERROR
                or event.outcome == ExecutionOutcome.ERROR
            ):
                session.failure = self._failure(session, call.activity, now, call.tool)
                failure = True
            elif event.outcome == ExecutionOutcome.CANCELLED:
                session.cancelled = True
            elif call.activity in session.failures:
                recovered = session.failures[call.activity]
                recovered.resolved = True
                unresolved = [
                    item for item in session.failures.values() if not item.resolved
                ]
                session.failure = (
                    max(unresolved, key=lambda item: item.at)
                    if unresolved
                    else recovered
                )
            if event.manual and not session.calls:
                outcome = (
                    ExecutionOutcome.ERROR
                    if self._current_failure(session)
                    else ExecutionOutcome.CANCELLED
                    if session.cancelled
                    else ExecutionOutcome.SUCCESS
                )
                self._finish(key, session, outcome, now, kind)
        elif kind == HookEventKind.SESSION_END:
            if session.last_event == HookEventKind.SESSION_END:
                return self._ignored(event, "duplicate_session_end", key, session)
            self._retire(key, session, now, kind)
            descendants = {key}
            for _ in range(len(self.by_id)):
                descendants.update(
                    child_key
                    for child_key, child in self.by_id.items()
                    if child.parent in descendants
                )
            for child_key in descendants - {key}:
                child = self.by_id[child_key]
                if child.active:
                    self._retire(child_key, child, now, "parent_session_end")
        elif kind in (
            HookEventKind.TURN_DONE,
            HookEventKind.SUBAGENT_DONE,
            HookEventKind.TURN_ERROR,
            HookEventKind.INTERRUPT,
        ):
            if session.finished:
                return self._ignored(event, "finished_session", key, session)
            outcome = (
                event.outcome if event.outcome is not None else ExecutionOutcome.SUCCESS
            )
            if kind == HookEventKind.TURN_ERROR or outcome == ExecutionOutcome.ERROR:
                outcome = ExecutionOutcome.ERROR
                if kind == HookEventKind.TURN_ERROR or not self._current_failure(
                    session
                ):
                    session.failure = self._failure(session, Activity.SOURCE, now)
                assert session.failure is not None
                event_activity = session.failure.activity
                failure = True
            elif kind == HookEventKind.INTERRUPT:
                outcome = event.outcome or (
                    ExecutionOutcome.CANCELLED
                    if session.active
                    else session.outcome or ExecutionOutcome.CANCELLED
                )
            elif self._current_failure(session):
                outcome = ExecutionOutcome.ERROR
            elif session.cancelled:
                outcome = ExecutionOutcome.CANCELLED
            if outcome == ExecutionOutcome.ERROR and session.failure:
                event_activity = session.failure.activity
            self._finish(key, session, outcome, now, kind)
            session.progress = (
                ""
                if outcome == ExecutionOutcome.ERROR
                else (
                    event.progress if event.progress is not None else session.progress
                )
            )
            if kind == HookEventKind.INTERRUPT:
                descendants = {key}
                for _ in range(len(self.by_id)):
                    descendants.update(
                        child_key
                        for child_key, child in self.by_id.items()
                        if child.parent in descendants
                    )
                for child_key in descendants - {key}:
                    child = self.by_id[child_key]
                    if child.active:
                        self._finish(
                            child_key,
                            child,
                            ExecutionOutcome.CANCELLED,
                            now,
                            f"parent_{kind}",
                        )
        elif kind in (HookEventKind.PROGRESS, HookEventKind.WAITING):
            if session.finished:
                return self._ignored(event, "finished_session", key, session)
            if kind == HookEventKind.WAITING and not session.active:
                self._open(key, session, event, now)
            if event.waiting_kind != WaitingKind.PERMISSION_CHECK:
                session.progress = (
                    event.progress if event.progress is not None else session.progress
                )
            if kind == HookEventKind.WAITING:
                if (
                    not session.waiting
                    or session.waiting_call != event.call
                    or session.waiting_kind != event.waiting_kind
                ):
                    session.waiting_since = now
                session.waiting = True
                session.waiting_call = event.call
                session.waiting_kind = event.waiting_kind
                session.waiting_tool = event.tool
                session.waiting_prompt = event.progress
                session.last_tool = (
                    event.tool if event.tool is not None else session.last_tool
                )
        if kind in (
            HookEventKind.TOOL_START,
            HookEventKind.TOOL_DONE,
            HookEventKind.TOOL_ERROR,
            HookEventKind.PROGRESS,
        ):
            session.waiting = False
            session.waiting_call = None
            session.waiting_since = None
            session.waiting_kind = None
            session.waiting_tool = None
            session.waiting_prompt = None
        if session.waiting and (
            waiting_since is None or waiting_call != session.waiting_call
        ):
            log.info(
                "session.waiting",
                session=key,
                parent=session.parent,
                turn=session.turn,
                call=session.waiting_call,
                tool=session.last_tool,
                active_calls=list(session.calls),
            )
        elif waiting_since is not None and not session.waiting and session.active:
            self._resumed(key, session, event, now, waiting_since)
        self._accepted(key, session, event)
        session.last_seen = now
        convergence = previously_active and not self.active
        if convergence:
            self.convergence = tuple(
                ConvergenceItem(member, self.by_id[member].outcome)
                for member in sorted(self.members)
                if member in self.by_id
            )
        return SessionUpdate(
            failure=failure,
            convergence=convergence,
            session=key,
            activity=event_activity,
            observation=self._observation(
                key,
                session,
                event,
                resumed=(
                    (
                        waiting_since is not None
                        and waiting_kind != WaitingKind.PERMISSION_CHECK
                    )
                    or was_finished
                )
                and session.active
                and not session.waiting,
                opened=not was_active and session.active,
            ),
        )

    def _observation(
        self,
        key: SessionIdentity,
        session: AgentSession,
        event: HookEvent,
        *,
        resumed: bool,
        opened: bool,
    ) -> SessionObservation:
        reports: list[IntentEvidence | ProgressEvidence | ExecutionEvidence] = []
        if event.goal and event.event in ("intent", "subagent_start"):
            reports.append(IntentEvidence(key, select_prompt(prose_text(event.goal))))
        if session.finished and event.event != "session_start":
            kind = "turn_done" if event.event.startswith("tool_") else event.event
            reports.append(
                ExecutionEvidence(key, kind, session.outcome, None, event.progress)
            )
        elif session.waiting:
            if session.waiting_for_input:
                reports.append(ExecutionEvidence(key, "waiting", None, None, None))
        elif resumed:
            reports.append(ExecutionEvidence(key, "resumed", None, None, None))
        elif opened or event.event in ("intent", "subagent_start"):
            reports.append(ExecutionEvidence(key, "turn_start", None, None, None))
        if (
            event.progress
            and not event.event.startswith("tool_")
            and event.waiting_kind != WaitingKind.PERMISSION_CHECK
        ):
            reports.append(ProgressEvidence(key, event.progress))
        return SessionObservation(
            session=key,
            aliases=tuple(
                sorted(
                    alias
                    for alias, target in self.aliases.items()
                    if target == key and alias != key
                )
            ),
            parent=session.parent,
            location=event.repo,
            reports=tuple(reports),
            repository=parse_remote_repository(event.repository)
            if event.repository
            else None,
            branch=event.branch,
            pull_request=event.pull_request,
        )

    def _refresh_activity(self) -> None:
        now = self.wall_clock()
        for session in self.by_id.values():
            session.activity_uncertain = bool(
                session.active
                and session.presence != ProcessStatus.ALIVE
                and now - session.last_hook_at >= self.quiet_seconds
            )

    def process_owners(self) -> set[ProcessIdentity]:
        return {
            session.owner
            for session in self.by_id.values()
            if session.owner
            and session.presence not in (ProcessStatus.EXITED, ProcessStatus.REPLACED)
        }

    def apply_process_status(
        self, results: Mapping[ProcessIdentity, ProcessStatus]
    ) -> list[SessionIdentity]:
        now = self.clock()
        retired = []
        for key, session in self.by_id.items():
            owner = session.owner
            if owner is None:
                continue
            status = results.get(owner)
            if status is None:
                continue
            session.presence = status
            session.owner_checked = now
            if status == ProcessStatus.ALIVE:
                session.owner_last_alive = now
            if status in (ProcessStatus.EXITED, ProcessStatus.REPLACED):
                log.info(
                    "session.owner_retired",
                    session=key,
                    owner=owner.packet(),
                    status=status,
                )
                self._retire(key, session, now, f"owner_{status}")
                retired.append(key)
            elif status == ProcessStatus.UNKNOWN:
                log.debug(
                    "session.owner_unknown",
                    session=key,
                    owner=owner.packet(),
                )
        if retired:
            self.convergence = None
        return retired

    def prune(self) -> bool:
        self._refresh_activity()
        now = self.clock()
        changed = False
        for key, session in self.by_id.items():
            if (
                session.active
                and session.presence != ProcessStatus.ALIVE
                and now - session.last_seen >= self.stale_seconds
            ):
                log.warning(
                    "session.expired",
                    session=key,
                    parent=session.parent,
                    turn=session.turn,
                    generation=session.generation,
                    last_event=session.last_event,
                    quiet_seconds=round(now - session.last_seen, 2),
                    stale_seconds=self.stale_seconds,
                    waiting=session.waiting,
                    waiting_call=session.waiting_call,
                    active_tools=[
                        {
                            "call": call_id,
                            "tool": call.tool,
                            "duration_seconds": round(now - call.started, 2),
                        }
                        for call_id, call in session.calls.items()
                    ],
                )
                self._retire(key, session, now, "stale_timeout")
                changed = True
        obsolete = {
            key
            for key, session in self.by_id.items()
            if not session.active
            and now
            - (session.ended if session.ended is not None else session.last_seen)
            >= self.retention_seconds
        }
        for key in obsolete:
            session = self.by_id[key]
            log.info(
                "session.removed",
                session=key,
                parent=session.parent,
                outcome=session.outcome,
                quiet_seconds=round(now - session.last_seen, 2),
                retention_seconds=self.retention_seconds,
                aliases=[
                    alias
                    for alias, target in self.aliases.items()
                    if target == key and alias != key
                ],
            )
            del self.by_id[key]
        self.aliases = {
            name: key for name, key in self.aliases.items() if key not in obsolete
        }
        self.child_calls = {
            name: key for name, key in self.child_calls.items() if key not in obsolete
        }
        self.members.difference_update(obsolete)
        if changed:
            self.convergence = None
        return changed

    def check_in_due(self, seconds: float) -> list[SessionIdentity]:
        self._refresh_activity()
        now = self.clock()
        return [
            key
            for key, session in sorted(self.by_id.items())
            if session.active
            and not session.activity_uncertain
            and now
            - max(
                session.waiting_started if session.waiting else session.last_seen,
                session.last_check_in
                if session.last_check_in is not None
                else session.started,
            )
            >= seconds
        ]

    def activity_view(
        self, keys: Collection[SessionIdentity] | None = None
    ) -> SwarmActivity:
        self._refresh_activity()
        now = self.clock()
        members = [
            (key, session)
            for key, session in sorted(self.by_id.items())
            if keys is None or key in keys
        ]
        return SwarmActivity(
            activity=PetState.WORKING
            if any(session.working for _, session in members)
            else PetState.IDLE,
            sessions=tuple(
                SessionContext(
                    id=key,
                    changed_at=session.changed_at,
                    last_hook_at=session.last_hook_at,
                    retired=session.retired,
                    parent=session.parent,
                    goal=session.goal,
                    repo=session.repo,
                    phase=session.phase,
                    initial_request="",
                    summary="",
                    recent_requests=(),
                    generation=session.generation,
                    request_count=0,
                    first_summary_attempted=False,
                    user_sentiment=UserSentiment(
                        compound=session.user_sentiment,
                        seconds_ago=round(now - session.sentiment_seen),
                    )
                    if session.user_sentiment is not None
                    and now - session.sentiment_seen < SENTIMENT_CONTEXT_SECONDS
                    else None,
                    duration_seconds=round(now - session.started),
                    quiet_seconds=round(now - session.last_seen),
                    turn_active=session.active,
                    last_tool=session.last_tool,
                    presence=session.presence,
                    owner=session.owner,
                    owner_last_alive_seconds=None
                    if session.owner_last_alive is None
                    else round(now - session.owner_last_alive),
                    activity_uncertain=session.activity_uncertain,
                    waiting_for_input=session.waiting_for_input,
                    waiting_seconds=None
                    if session.waiting_since is None
                    else round(now - session.waiting_since),
                    waiting_kind=session.waiting_kind,
                    waiting_tool=session.waiting_tool,
                    waiting_prompt=session.waiting_prompt,
                    active_tools=tuple(
                        ActiveTool(
                            call=call_id,
                            tool=call.tool,
                            activity=call.activity,
                            duration_seconds=round(now - call.started),
                        )
                        for call_id, call in session.calls.items()
                    ),
                    progress=session.progress,
                    failure=None
                    if session.failure is None
                    else FailureSnapshot(
                        activity=session.failure.activity,
                        count=session.failure.count,
                        goal=session.failure.goal,
                        resolved=session.failure.resolved,
                        current_turn=session.failure.generation == session.generation,
                        seconds_ago=round(now - session.failure.at),
                    ),
                    outcome=session.outcome,
                )
                for key, session in members
            ),
            convergence=self.convergence if keys is None else None,
        )
