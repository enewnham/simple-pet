from __future__ import annotations

import logging
import time
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .. import telemetry
from ..prompts import select_prompt
from ..runtime import (
    ActiveTool,
    ConvergenceItem,
    FailureSnapshot,
    ProcessIdentity,
    SessionContext,
    SwarmActivity,
    UserSentiment,
)
from ..work.models import ExecutionEvidence, IntentEvidence, ProgressEvidence
from ..work.observations import SessionObservation
from .activity import activity
from .processes import ProcessStatus

if TYPE_CHECKING:
    from ..runtime import HookEvent

SENTIMENT_CONTEXT_SECONDS = 300


@dataclass
class Failure:
    tool: str
    activity: str
    at: float
    count: int
    resolved: bool
    generation: int
    goal: str


@dataclass(frozen=True)
class SessionUpdate:
    failure: bool
    convergence: bool
    session: str
    activity: str
    observation: SessionObservation


@dataclass
class Call:
    tool: str
    started: float
    activity: str = "command"


@dataclass
class AgentSession:
    harness: str
    id: str
    started: float
    last_seen: float
    changed_at: float = field(default_factory=time.time)
    last_event: str | None = None
    parent: str | None = None
    goal: str = ""
    intent_received: bool = False
    user_sentiment: float | None = None
    sentiment_seen: float = float("-inf")
    sentiment_identity: tuple[str | None, str] | None = None
    repo: str = ""
    phase: str = ""
    last_tool: str = ""
    progress: str = ""
    failure: Failure | None = None
    failures: dict[str, Failure] = field(default_factory=dict)
    calls: dict[str, Call] = field(default_factory=dict)
    active: bool = False
    finished: bool = False
    turn: str | None = None
    generation: int = 0
    outcome: str | None = None
    ended: float | None = None
    cancelled: bool = False
    waiting: bool = False
    waiting_call: str | None = None
    waiting_since: float | None = None
    last_check_in: float | None = None
    owner: ProcessIdentity | None = None
    presence: str = "unknown"
    owner_checked: float | None = None
    owner_last_alive: float | None = None
    activity_uncertain: bool = False

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
    ) -> None:
        self.clock = clock
        self.quiet_seconds = fresh_seconds
        self.stale_seconds = stale_seconds
        self.retention_seconds = retention_seconds
        self.by_id: dict[str, AgentSession] = {}
        self.aliases: dict[str, str] = {}
        self.child_calls: dict[tuple[str, str, str], str] = {}
        self.members: set[str] = set()
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
        self, session: AgentSession, category: str, now: float, tool: str = ""
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
            and all(item.outcome == "success" for item in self.convergence)
        )

    def _key(self, harness: str, identifier: str) -> str:
        key = (
            identifier
            if identifier.startswith(f"{harness}:")
            else f"{harness}:{identifier}"
        )
        return self.aliases.get(key, key)

    def _session(self, event: HookEvent, now: float) -> tuple[str, AgentSession] | None:
        harness = event.harness
        identifier = event.session
        key = self._key(harness, identifier)
        routed = False
        parent_call = event.parent_call
        if parent_call and event.event in (
            "tool_start",
            "tool_done",
            "tool_error",
            "waiting",
        ):
            target = self.child_calls.get((harness, identifier, parent_call))
            if target and target != key:
                key = target
                routed = True
        alternate = event.aliases
        matches = [self._key(harness, alias) for alias in alternate]
        if key not in self.by_id:
            key = next((match for match in matches if match in self.by_id), key)
        parent = event.parent
        if parent and self._key(harness, parent) in {key, *matches}:
            return self._ignored(event, "invalid_parent")
        created = key not in self.by_id
        session = self.by_id.setdefault(
            key, AgentSession(harness, identifier, now, now)
        )
        for alias in alternate if routed else [identifier, *alternate]:
            old_key = self._key(harness, alias)
            if old_key != key and old_key in self.by_id:
                other = self.by_id.pop(old_key)
                if not session.active and (
                    other.active or (other.finished and not session.finished)
                ):
                    session, other = other, session
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
                        session.last_check_in = other.last_check_in
                    if adopt_waiting or other.last_seen > session.last_seen:
                        session.phase = other.phase or session.phase
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
                telemetry.event(
                    "session.merged",
                    session=key,
                    merged_session=old_key,
                    active=session.active,
                    active_calls=list(session.calls),
                    parent=session.parent,
                )
            alias_key = (
                alias if alias.startswith(f"{harness}:") else f"{harness}:{alias}"
            )
            if alias_key != key and self.aliases.get(alias_key) != key:
                telemetry.event("session.aliased", session=key, alias=alias_key)
            self.aliases[alias_key] = key
        if parent:
            session.parent = self._key(harness, parent)
            if parent_call:
                self.child_calls[(harness, parent, parent_call)] = key
            parent_session = self.by_id.get(session.parent)
            if parent_session and not session.goal:
                session.goal = parent_session.goal
            if parent_session and not session.repo:
                session.repo = parent_session.repo
            if parent_session and not session.owner:
                session.owner = parent_session.owner
                session.presence = parent_session.presence
        if created:
            telemetry.event(
                "session.created",
                session=key,
                parent=session.parent,
                source_event=event.event,
                repo=event.repo or session.repo,
                active_sessions=sum(item.active for item in self.by_id.values()),
            )
        return key, session

    def _open(
        self, key: str, session: AgentSession, event: HookEvent, now: float
    ) -> None:
        previous_turn = session.turn
        replaced_calls = list(session.calls)
        if not self.active:
            self.members.clear()
            self.convergence = None
        session.active = True
        session.activity_uncertain = False
        session.finished = False
        session.outcome = None
        session.cancelled = False
        session.ended = None
        session.generation += 1
        session.failures.clear()
        session.turn = None if event.event == "subagent_start" else event.turn
        session.started = now
        session.calls.clear()
        session.last_tool = ""
        session.phase = ""
        session.progress = ""
        session.waiting = False
        session.waiting_call = None
        session.waiting_since = None
        session.last_check_in = None
        self.members.add(key)
        telemetry.event(
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
        self, key: str, session: AgentSession, outcome: str, now: float, reason: str
    ) -> None:
        telemetry.event(
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
        session.changed_at = time.time()
        session.outcome = outcome
        if outcome == "cancelled":
            for failure in session.failures.values():
                failure.resolved = True
        session.waiting = False
        session.waiting_call = None
        session.waiting_since = None

    def _ignored(
        self,
        event: HookEvent,
        reason: str,
        key: str | None = None,
        session: AgentSession | None = None,
    ) -> None:
        telemetry.trace(
            "session.event_ignored",
            source_event=event.event,
            reason=reason,
            session=key or event.subject,
            turn=event.turn,
            current_turn=session.turn if session else None,
            call=event.call,
        )

    def _accepted(self, key: str, session: AgentSession, event: HookEvent) -> None:
        session.activity_uncertain = False
        session.last_event = event.event
        session.changed_at = time.time()
        telemetry.verbose(
            "session.event_accepted",
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
        key: str,
        session: AgentSession,
        event: HookEvent,
        now: float,
        waiting_since: float,
    ) -> None:
        telemetry.event(
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
            telemetry.trace("session.event_ignored", reason="invalid_event")
            return None
        return self.apply(hook)

    def apply(self, event: HookEvent) -> SessionUpdate | None:
        kind = event.event
        now = self.clock()
        previously_active = self.active
        identity = self._session(event, now)
        if identity is None:
            return None
        key, session = identity
        turn = event.turn
        if (
            turn
            and session.turn
            and turn != session.turn
            and kind
            not in (
                "intent",
                "session_start",
                "subagent_start",
                "subagent_done",
                "session_end",
            )
        ):
            return self._ignored(event, "turn_mismatch", key, session)
        owner = event.owner
        if owner and owner != session.owner:
            session.owner = owner
            session.presence = "unknown"
            session.owner_checked = None
            session.owner_last_alive = None
        waiting_since = session.waiting_since
        waiting_call = session.waiting_call
        was_active = session.active
        was_finished = session.finished
        if event.repo:
            session.repo = event.repo
        if kind == "subagent_start" and session.active and session.intent_received:
            return self._ignored(event, "duplicate_subagent_start", key, session)
        failure = False
        event_activity = session.phase or "command"
        if kind == "session_start":
            if session.finished:
                session.finished = False
        elif kind in ("intent", "subagent_start"):
            new_native_turn = kind == "intent" and not event.manual
            if (
                not session.active
                or new_native_turn
                or (kind == "intent" and turn and turn != session.turn)
            ):
                self._open(key, session, event, now)
            if event.goal:
                request = select_prompt(event.goal)
                if not session.intent_received:
                    session.goal = request
                session.intent_received = True
            elif kind == "subagent_start" and session.parent in self.by_id:
                session.goal = self.by_id[session.parent].goal
                session.intent_received = bool(session.goal)
        elif kind == "tool_start":
            if not event.call:
                return self._ignored(event, "missing_call", key, session)
            if session.finished and not event.manual:
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
        elif kind in ("tool_done", "tool_error"):
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
                if kind == "tool_error" or event.outcome == "error"
                else (
                    event.progress if event.progress is not None else session.progress
                )
            )
            if kind == "tool_error" or event.outcome == "error":
                session.failure = self._failure(session, call.activity, now, call.tool)
                failure = True
            elif event.outcome == "cancelled":
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
                    "error"
                    if self._current_failure(session)
                    else "cancelled"
                    if session.cancelled
                    else "success"
                )
                self._finish(key, session, outcome, now, kind)
        elif kind in (
            "turn_done",
            "subagent_done",
            "turn_error",
            "interrupt",
            "session_end",
        ):
            if session.finished and kind != "session_end":
                return self._ignored(event, "finished_session", key, session)
            outcome = event.outcome if event.outcome is not None else "success"
            duplicate_end = kind == "session_end" and session.finished
            if (kind == "turn_error" or outcome == "error") and not duplicate_end:
                outcome = "error"
                if kind == "turn_error" or not self._current_failure(session):
                    session.failure = self._failure(session, "harness", now)
                assert session.failure is not None
                event_activity = session.failure.activity
                failure = True
            elif kind in ("interrupt", "session_end"):
                outcome = event.outcome or (
                    "cancelled" if session.active else session.outcome or "cancelled"
                )
            elif self._current_failure(session):
                outcome = "error"
            elif session.cancelled:
                outcome = "cancelled"
            if outcome == "error" and session.failure:
                event_activity = session.failure.activity
            self._finish(key, session, outcome, now, kind)
            session.progress = (
                ""
                if outcome == "error"
                else (
                    event.progress if event.progress is not None else session.progress
                )
            )
            if kind in ("session_end", "interrupt"):
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
                            child_key, child, "cancelled", now, f"parent_{kind}"
                        )
        elif kind in ("progress", "waiting"):
            if session.finished:
                return self._ignored(event, "finished_session", key, session)
            if kind == "waiting" and not session.active:
                self._open(key, session, event, now)
            session.progress = (
                event.progress if event.progress is not None else session.progress
            )
            if kind == "waiting":
                if not session.waiting or session.waiting_call != event.call:
                    session.waiting_since = now
                session.waiting = True
                session.waiting_call = event.call
                session.last_tool = (
                    event.tool if event.tool is not None else session.last_tool
                )
        if kind in ("tool_start", "tool_done", "tool_error", "progress"):
            session.waiting = False
            session.waiting_call = None
            session.waiting_since = None
        if session.waiting and (
            waiting_since is None or waiting_call != session.waiting_call
        ):
            telemetry.event(
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
                resumed=(waiting_since is not None or was_finished)
                and session.active
                and not session.waiting,
                opened=not was_active and session.active,
            ),
        )

    def _observation(
        self,
        key: str,
        session: AgentSession,
        event: HookEvent,
        *,
        resumed: bool,
        opened: bool,
    ) -> SessionObservation:
        reports: list[IntentEvidence | ProgressEvidence | ExecutionEvidence] = []
        if event.goal and event.event in ("intent", "subagent_start"):
            reports.append(IntentEvidence(key, select_prompt(event.goal)))
        if session.finished:
            kind = "turn_done" if event.event.startswith("tool_") else event.event
            reports.append(
                ExecutionEvidence(key, kind, session.outcome, None, event.progress)
            )
        elif session.waiting:
            reports.append(ExecutionEvidence(key, "waiting", None, None, None))
        elif resumed:
            reports.append(ExecutionEvidence(key, "resumed", None, None, None))
        elif opened or event.event in ("intent", "subagent_start"):
            reports.append(ExecutionEvidence(key, "turn_start", None, None, None))
        if event.progress and not event.event.startswith("tool_"):
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
            repository=event.repository,
            branch=event.branch,
            pull_request=event.pull_request,
        )

    def _refresh_activity(self) -> None:
        now = self.clock()
        for session in self.by_id.values():
            session.activity_uncertain = bool(
                session.active
                and not session.calls
                and not session.waiting
                and now - session.last_seen >= self.quiet_seconds
            )

    def process_owners(self) -> set[ProcessIdentity]:
        return {
            session.owner
            for session in self.by_id.values()
            if session.owner and session.presence not in ("exited", "replaced")
        }

    def apply_process_status(
        self, results: Mapping[ProcessIdentity, ProcessStatus]
    ) -> list[str]:
        now = self.clock()
        retired = []
        for key, session in self.by_id.items():
            owner = session.owner
            if owner is None:
                continue
            status = results.get(owner)
            if status not in ("alive", "exited", "replaced", "unknown"):
                continue
            session.presence = status
            session.owner_checked = now
            if status == "alive":
                session.owner_last_alive = now
            if status in ("exited", "replaced") and session.active:
                telemetry.event(
                    "session.owner_retired",
                    session=key,
                    owner=owner.packet(),
                    status=status,
                )
                self._finish(key, session, "stale", now, f"owner_{status}")
                retired.append(key)
            elif status == "unknown":
                telemetry.trace(
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
            if session.active and now - session.last_seen >= self.stale_seconds:
                telemetry.event(
                    "session.expired",
                    level=logging.WARNING,
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
                self._finish(key, session, "stale", now, "stale_timeout")
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
            telemetry.event(
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

    def check_in_due(self, seconds: float) -> list[str]:
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

    def activity_view(self, keys: Collection[str] | None = None) -> SwarmActivity:
        self._refresh_activity()
        now = self.clock()
        members = [
            (key, session)
            for key, session in sorted(self.by_id.items())
            if keys is None or key in keys
        ]
        return SwarmActivity(
            activity="working"
            if any(session.working for _, session in members)
            else "idle",
            sessions=tuple(
                SessionContext(
                    id=key,
                    changed_at=session.changed_at,
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
                    waiting_for_input=session.waiting,
                    waiting_seconds=None
                    if session.waiting_since is None
                    else round(now - session.waiting_since),
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
