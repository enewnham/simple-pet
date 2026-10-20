from __future__ import annotations

import logging
import time
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from . import telemetry
from .activity import activity
from .processes import ProcessStatus
from .prompts import select_prompt
from .runtime import (
    ConvergenceItem,
    FailureSnapshot,
    ProcessIdentity,
    SentimentSnapshot,
    SessionSnapshot,
    SessionsSnapshot,
    ToolSnapshot,
)

if TYPE_CHECKING:
    from .runtime import HookEvent

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
class SessionResult:
    failure: bool
    convergence: bool
    session: str
    activity: str


@dataclass
class Call:
    tool: str
    started: float
    activity: str = "command"


@dataclass
class Lane:
    harness: str
    id: str
    started: float
    last_seen: float
    changed_at: float = field(default_factory=time.time)
    last_event: str | None = None
    parent: str | None = None
    goal: str = ""
    first_prompt: str = ""
    recent_requests: list[str] = field(default_factory=list)
    session_context: str = ""
    context_loaded: bool = False
    request_count: int = 0
    first_summary_attempted: bool = False
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
        self.lanes: dict[str, Lane] = {}
        self.aliases: dict[str, str] = {}
        self.child_calls: dict[tuple[str, str, str], str] = {}
        self.members: set[str] = set()
        self.convergence: tuple[ConvergenceItem, ...] | None = None

    @property
    def working(self) -> bool:
        self._refresh_activity()
        return any(lane.working for lane in self.lanes.values())

    @property
    def engaged(self) -> bool:
        self._refresh_activity()
        return any(
            lane.active and not lane.activity_uncertain for lane in self.lanes.values()
        )

    @property
    def active(self) -> bool:
        return any(lane.active for lane in self.lanes.values())

    @property
    def failed(self) -> bool:
        return any(self._current_failure(lane) for lane in self.lanes.values())

    def _current_failure(self, lane: Lane) -> bool:
        return any(
            failure.generation == lane.generation and not failure.resolved
            for failure in lane.failures.values()
        )

    def _failure(
        self, lane: Lane, category: str, now: float, tool: str = ""
    ) -> Failure:
        previous = lane.failures.get(category)
        count = previous.count + 1 if previous and not previous.resolved else 1
        failure = Failure(
            tool=tool,
            activity=category,
            at=now,
            count=count,
            resolved=False,
            generation=lane.generation,
            goal=lane.goal,
        )
        lane.failures[category] = failure
        return failure

    @property
    def clean_convergence(self) -> bool:
        return (
            self.convergence is not None
            and not self.active
            and all(item.outcome == "success" for item in self.convergence)
        )

    def _key(self, harness: str, identifier: str) -> str:
        key = f"{harness}:{identifier}"
        return self.aliases.get(key, key)

    def _lane(self, event: HookEvent, now: float) -> tuple[str, Lane] | None:
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
        if key not in self.lanes:
            key = next((match for match in matches if match in self.lanes), key)
        parent = event.parent
        if parent and self._key(harness, parent) in {key, *matches}:
            return self._ignored(event, "invalid_parent")
        created = key not in self.lanes
        lane = self.lanes.setdefault(key, Lane(harness, identifier, now, now))
        for alias in alternate if routed else [identifier, *alternate]:
            old_key = self._key(harness, alias)
            if old_key != key and old_key in self.lanes:
                other = self.lanes.pop(old_key)
                if not lane.active and (
                    other.active or (other.finished and not lane.finished)
                ):
                    lane, other = other, lane
                    self.lanes[key] = lane
                lane.calls.update(other.calls)
                lane.active = lane.active or other.active
                lane.goal = lane.goal or other.goal
                lane.first_prompt = lane.first_prompt or other.first_prompt
                lane.session_context = lane.session_context or other.session_context
                lane.recent_requests = lane.recent_requests or other.recent_requests
                lane.request_count += other.request_count
                lane.first_summary_attempted = (
                    lane.first_summary_attempted or other.first_summary_attempted
                )
                if other.sentiment_seen > lane.sentiment_seen:
                    lane.user_sentiment = other.user_sentiment
                    lane.sentiment_seen = other.sentiment_seen
                    lane.sentiment_identity = other.sentiment_identity
                lane.context_loaded = lane.context_loaded or other.context_loaded
                lane.repo = lane.repo or other.repo
                lane.owner = lane.owner or other.owner
                if lane.owner == other.owner:
                    lane.presence = other.presence
                if other.active:
                    adopt_waiting = other.waiting and (
                        not lane.waiting
                        or other.waiting_started >= lane.waiting_started
                    )
                    if adopt_waiting:
                        lane.waiting = True
                        lane.waiting_call = other.waiting_call
                        lane.waiting_since = other.waiting_since
                        lane.last_check_in = other.last_check_in
                    if adopt_waiting or other.last_seen > lane.last_seen:
                        lane.phase = other.phase or lane.phase
                        lane.last_tool = other.last_tool or lane.last_tool
                        lane.progress = other.progress or lane.progress
                    if other.last_seen > lane.last_seen:
                        lane.last_event = other.last_event
                        lane.changed_at = other.changed_at
                    lane.last_seen = max(lane.last_seen, other.last_seen)
                    for category, failure in other.failures.items():
                        if failure.generation == other.generation:
                            failure.generation = lane.generation
                            existing = lane.failures.get(category)
                            if existing is None or failure.at > existing.at:
                                lane.failures[category] = failure
                    unresolved = [
                        item for item in lane.failures.values() if not item.resolved
                    ]
                    if unresolved:
                        lane.failure = max(unresolved, key=lambda item: item.at)
                lane.started = min(lane.started, other.started)
                self.aliases = {
                    name: key if target == old_key else target
                    for name, target in self.aliases.items()
                }
                self.child_calls = {
                    name: key if target == old_key else target
                    for name, target in self.child_calls.items()
                }
                for child in self.lanes.values():
                    if child.parent == old_key:
                        child.parent = key
                if old_key in self.members:
                    self.members.remove(old_key)
                    self.members.add(key)
                telemetry.event(
                    "session.merged",
                    session=key,
                    merged_session=old_key,
                    active=lane.active,
                    active_calls=list(lane.calls),
                    parent=lane.parent,
                )
            alias_key = f"{harness}:{alias}"
            if alias_key != key and self.aliases.get(alias_key) != key:
                telemetry.event("session.aliased", session=key, alias=alias_key)
            self.aliases[alias_key] = key
        if parent:
            lane.parent = self._key(harness, parent)
            if parent_call:
                self.child_calls[(harness, parent, parent_call)] = key
            parent_lane = self.lanes.get(lane.parent)
            if parent_lane and not lane.goal:
                lane.goal = parent_lane.goal
            if parent_lane and not lane.repo:
                lane.repo = parent_lane.repo
            if parent_lane and not lane.owner:
                lane.owner = parent_lane.owner
                lane.presence = parent_lane.presence
        if created:
            telemetry.event(
                "session.created",
                session=key,
                parent=lane.parent,
                source_event=event.event,
                repo=event.repo or lane.repo,
                active_sessions=sum(item.active for item in self.lanes.values()),
            )
        return key, lane

    def _open(self, key: str, lane: Lane, event: HookEvent, now: float) -> None:
        previous_turn = lane.turn
        replaced_calls = list(lane.calls)
        if not self.active:
            self.members.clear()
            self.convergence = None
        lane.active = True
        lane.activity_uncertain = False
        lane.finished = False
        lane.outcome = None
        lane.cancelled = False
        lane.ended = None
        lane.generation += 1
        lane.failures.clear()
        lane.turn = None if event.event == "subagent_start" else event.turn
        lane.started = now
        lane.calls.clear()
        lane.last_tool = ""
        lane.phase = ""
        lane.progress = ""
        lane.waiting = False
        lane.waiting_call = None
        lane.waiting_since = None
        lane.last_check_in = None
        self.members.add(key)
        telemetry.event(
            "session.turn_started",
            session=key,
            parent=lane.parent,
            turn=lane.turn,
            generation=lane.generation,
            source_event=event.event,
            repo=lane.repo or (event.repo if event.repo is not None else ""),
            previous_turn=previous_turn,
            replaced_calls=replaced_calls,
            active_sessions=sum(item.active for item in self.lanes.values()),
        )

    def _finish(
        self, key: str, lane: Lane, outcome: str, now: float, reason: str
    ) -> None:
        telemetry.event(
            "session.turn_finished",
            session=key,
            parent=lane.parent,
            turn=lane.turn,
            generation=lane.generation,
            outcome=outcome,
            reason=reason,
            duration_seconds=round(now - lane.started, 2),
            active_calls=list(lane.calls),
            active_sessions=sum(item.active for item in self.lanes.values())
            - int(lane.active),
            active_children=[
                child_key
                for child_key, child in self.lanes.items()
                if child.parent == key and child.active
            ],
        )
        lane.calls.clear()
        lane.active = False
        lane.finished = True
        lane.ended = now
        lane.changed_at = time.time()
        lane.outcome = outcome
        if outcome == "cancelled":
            for failure in lane.failures.values():
                failure.resolved = True
        lane.waiting = False
        lane.waiting_call = None
        lane.waiting_since = None

    def _ignored(
        self,
        event: HookEvent,
        reason: str,
        key: str | None = None,
        lane: Lane | None = None,
    ) -> None:
        telemetry.trace(
            "session.event_ignored",
            source_event=event.event,
            reason=reason,
            session=key or event.subject,
            turn=event.turn,
            current_turn=lane.turn if lane else None,
            call=event.call,
        )

    def _accepted(self, key: str, lane: Lane, event: HookEvent) -> None:
        lane.activity_uncertain = False
        lane.last_event = event.event
        lane.changed_at = time.time()
        telemetry.verbose(
            "session.event_accepted",
            session=key,
            source_session=event.session,
            source_event=event.event,
            turn=lane.turn,
            generation=lane.generation,
            call=event.call,
            tool=event.tool,
            active=lane.active,
            waiting=lane.waiting,
            active_calls=list(lane.calls),
        )

    def _resumed(
        self, key: str, lane: Lane, event: HookEvent, now: float, waiting_since: float
    ) -> None:
        telemetry.event(
            "session.resumed",
            session=key,
            parent=lane.parent,
            turn=lane.turn,
            source_event=event.event,
            waiting_seconds=round(now - waiting_since, 2),
            call=event.call,
        )

    def handle(self, event: object) -> SessionResult | None:
        from .sources.hooks import normalize_hook

        hook = normalize_hook(event)
        if hook is None:
            telemetry.trace("session.event_ignored", reason="invalid_event")
            return None
        return self.apply(hook)

    def apply(self, event: HookEvent) -> SessionResult | None:
        kind = event.event
        now = self.clock()
        previously_active = self.active
        identity = self._lane(event, now)
        if identity is None:
            return None
        key, lane = identity
        turn = event.turn
        if (
            turn
            and lane.turn
            and turn != lane.turn
            and kind
            not in (
                "intent",
                "session_start",
                "subagent_start",
                "subagent_done",
                "session_end",
            )
        ):
            return self._ignored(event, "turn_mismatch", key, lane)
        owner = event.owner
        if owner and owner != lane.owner:
            lane.owner = owner
            lane.presence = "unknown"
            lane.owner_checked = None
            lane.owner_last_alive = None
        waiting_since = lane.waiting_since
        waiting_call = lane.waiting_call
        if event.repo:
            lane.repo = event.repo
        if kind == "subagent_start" and lane.active and lane.first_prompt:
            return self._ignored(event, "duplicate_subagent_start", key, lane)
        failure = False
        event_activity = lane.phase or "command"
        if kind == "session_start":
            if lane.finished:
                lane.finished = False
        elif kind in ("intent", "subagent_start"):
            new_native_turn = kind == "intent" and not event.manual
            if (
                not lane.active
                or new_native_turn
                or (kind == "intent" and turn and turn != lane.turn)
            ):
                self._open(key, lane, event, now)
            if event.goal:
                request = select_prompt(event.goal)
                lane.first_prompt = (
                    request if lane.request_count == 0 else lane.first_prompt
                )
                lane.recent_requests = [*lane.recent_requests, request][-4:]
                lane.request_count += 1
                lane.goal = lane.first_prompt
            elif (
                kind == "subagent_start"
                and not lane.first_prompt
                and lane.parent in self.lanes
            ):
                parent = self.lanes[lane.parent]
                lane.first_prompt = parent.first_prompt
                lane.session_context = parent.session_context
                lane.goal = lane.first_prompt
        elif kind == "tool_start":
            if not event.call:
                return self._ignored(event, "missing_call", key, lane)
            if lane.finished and not event.manual:
                return self._ignored(event, "finished_session", key, lane)
            if not lane.active:
                self._open(key, lane, event, now)
            if turn and lane.turn is None:
                lane.turn = turn
            call_id = event.call
            if call_id in lane.calls:
                return self._ignored(event, "duplicate_call", key, lane)
            tool = event.tool if event.tool is not None else ""
            event_activity = activity(
                tool, (event.tool_name if event.tool_name is not None else "")
            )
            lane.calls[call_id] = Call(tool, now, event_activity)
            lane.last_tool = tool
            lane.phase = event_activity
            lane.progress = event.progress if event.progress is not None else ""
        elif kind in ("tool_done", "tool_error"):
            call = lane.calls.pop(event.call, None)
            if call is None:
                if lane.waiting and event.call and event.call == lane.waiting_call:
                    tool = event.tool if event.tool is not None else lane.last_tool
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
                    return self._ignored(event, "unknown_call", key, lane)
            event_activity = call.activity
            lane.progress = (
                ""
                if kind == "tool_error" or event.outcome == "error"
                else (event.progress if event.progress is not None else lane.progress)
            )
            if kind == "tool_error" or event.outcome == "error":
                lane.failure = self._failure(lane, call.activity, now, call.tool)
                failure = True
            elif event.outcome == "cancelled":
                lane.cancelled = True
            elif call.activity in lane.failures:
                recovered = lane.failures[call.activity]
                recovered.resolved = True
                unresolved = [
                    item for item in lane.failures.values() if not item.resolved
                ]
                lane.failure = (
                    max(unresolved, key=lambda item: item.at)
                    if unresolved
                    else recovered
                )
            if event.manual and not lane.calls:
                outcome = (
                    "error"
                    if self._current_failure(lane)
                    else "cancelled"
                    if lane.cancelled
                    else "success"
                )
                self._finish(key, lane, outcome, now, kind)
        elif kind in (
            "turn_done",
            "subagent_done",
            "turn_error",
            "interrupt",
            "session_end",
        ):
            if lane.finished and kind != "session_end":
                return self._ignored(event, "finished_session", key, lane)
            outcome = event.outcome if event.outcome is not None else "success"
            duplicate_end = kind == "session_end" and lane.finished
            if (kind == "turn_error" or outcome == "error") and not duplicate_end:
                outcome = "error"
                if kind == "turn_error" or not self._current_failure(lane):
                    lane.failure = self._failure(lane, "harness", now)
                assert lane.failure is not None
                event_activity = lane.failure.activity
                failure = True
            elif kind in ("interrupt", "session_end"):
                outcome = event.outcome or (
                    "cancelled" if lane.active else lane.outcome or "cancelled"
                )
            elif self._current_failure(lane):
                outcome = "error"
            elif lane.cancelled:
                outcome = "cancelled"
            if outcome == "error" and lane.failure:
                event_activity = lane.failure.activity
            self._finish(key, lane, outcome, now, kind)
            lane.progress = (
                ""
                if outcome == "error"
                else (event.progress if event.progress is not None else lane.progress)
            )
            if kind in ("session_end", "interrupt"):
                descendants = {key}
                for _ in range(len(self.lanes)):
                    descendants.update(
                        child_key
                        for child_key, child in self.lanes.items()
                        if child.parent in descendants
                    )
                for child_key in descendants - {key}:
                    child = self.lanes[child_key]
                    if child.active:
                        self._finish(
                            child_key, child, "cancelled", now, f"parent_{kind}"
                        )
        elif kind in ("progress", "waiting"):
            if lane.finished:
                return self._ignored(event, "finished_session", key, lane)
            if kind == "waiting" and not lane.active:
                self._open(key, lane, event, now)
            lane.progress = (
                event.progress if event.progress is not None else lane.progress
            )
            if kind == "waiting":
                if not lane.waiting or lane.waiting_call != event.call:
                    lane.waiting_since = now
                lane.waiting = True
                lane.waiting_call = event.call
                lane.last_tool = (
                    event.tool if event.tool is not None else lane.last_tool
                )
        if kind in ("tool_start", "tool_done", "tool_error", "progress"):
            lane.waiting = False
            lane.waiting_call = None
            lane.waiting_since = None
        if lane.waiting and (
            waiting_since is None or waiting_call != lane.waiting_call
        ):
            telemetry.event(
                "session.waiting",
                session=key,
                parent=lane.parent,
                turn=lane.turn,
                call=lane.waiting_call,
                tool=lane.last_tool,
                active_calls=list(lane.calls),
            )
        elif waiting_since is not None and not lane.waiting and lane.active:
            self._resumed(key, lane, event, now, waiting_since)
        self._accepted(key, lane, event)
        lane.last_seen = now
        convergence = previously_active and not self.active
        if convergence:
            self.convergence = tuple(
                ConvergenceItem(member, self.lanes[member].outcome)
                for member in sorted(self.members)
                if member in self.lanes
            )
        return SessionResult(
            failure=failure,
            convergence=convergence,
            session=key,
            activity=event_activity,
        )

    def _refresh_activity(self) -> None:
        now = self.clock()
        for lane in self.lanes.values():
            lane.activity_uncertain = bool(
                lane.active
                and not lane.calls
                and not lane.waiting
                and now - lane.last_seen >= self.quiet_seconds
            )

    def process_owners(self) -> set[ProcessIdentity]:
        return {
            lane.owner
            for lane in self.lanes.values()
            if lane.owner and lane.presence not in ("exited", "replaced")
        }

    def apply_process_status(
        self, results: Mapping[ProcessIdentity, ProcessStatus]
    ) -> list[str]:
        now = self.clock()
        retired = []
        for key, lane in self.lanes.items():
            owner = lane.owner
            if owner is None:
                continue
            status = results.get(owner)
            if status not in ("alive", "exited", "replaced", "unknown"):
                continue
            lane.presence = status
            lane.owner_checked = now
            if status == "alive":
                lane.owner_last_alive = now
            if status in ("exited", "replaced") and lane.active:
                telemetry.event(
                    "session.owner_retired",
                    session=key,
                    owner=owner.packet(),
                    status=status,
                )
                self._finish(key, lane, "stale", now, f"owner_{status}")
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
        for key, lane in self.lanes.items():
            if lane.active and now - lane.last_seen >= self.stale_seconds:
                telemetry.event(
                    "session.expired",
                    level=logging.WARNING,
                    session=key,
                    parent=lane.parent,
                    turn=lane.turn,
                    generation=lane.generation,
                    last_event=lane.last_event,
                    quiet_seconds=round(now - lane.last_seen, 2),
                    stale_seconds=self.stale_seconds,
                    waiting=lane.waiting,
                    waiting_call=lane.waiting_call,
                    active_tools=[
                        {
                            "call": call_id,
                            "tool": call.tool,
                            "duration_seconds": round(now - call.started, 2),
                        }
                        for call_id, call in lane.calls.items()
                    ],
                )
                self._finish(key, lane, "stale", now, "stale_timeout")
                changed = True
        obsolete = {
            key
            for key, lane in self.lanes.items()
            if not lane.active
            and now - (lane.ended if lane.ended is not None else lane.last_seen)
            >= self.retention_seconds
        }
        for key in obsolete:
            lane = self.lanes[key]
            telemetry.event(
                "session.removed",
                session=key,
                parent=lane.parent,
                outcome=lane.outcome,
                quiet_seconds=round(now - lane.last_seen, 2),
                retention_seconds=self.retention_seconds,
                aliases=[
                    alias
                    for alias, target in self.aliases.items()
                    if target == key and alias != key
                ],
            )
            del self.lanes[key]
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
            for key, lane in sorted(self.lanes.items())
            if lane.active
            and not lane.activity_uncertain
            and now
            - max(
                lane.waiting_started if lane.waiting else lane.last_seen,
                lane.last_check_in if lane.last_check_in is not None else lane.started,
            )
            >= seconds
        ]

    def snapshot(self, keys: Collection[str] | None = None) -> SessionsSnapshot:
        self._refresh_activity()
        now = self.clock()
        lanes = [
            (key, lane)
            for key, lane in sorted(self.lanes.items())
            if keys is None or key in keys
        ]
        return SessionsSnapshot(
            activity="working" if any(lane.working for _, lane in lanes) else "idle",
            sessions=tuple(
                SessionSnapshot(
                    id=key,
                    changed_at=lane.changed_at,
                    parent=lane.parent,
                    goal=lane.goal,
                    repo=lane.repo,
                    phase=lane.phase,
                    first_prompt=lane.first_prompt,
                    session_context=lane.session_context,
                    recent_requests=tuple(lane.recent_requests),
                    generation=lane.generation,
                    request_count=lane.request_count,
                    first_summary_attempted=lane.first_summary_attempted,
                    user_sentiment=SentimentSnapshot(
                        compound=lane.user_sentiment,
                        seconds_ago=round(now - lane.sentiment_seen),
                    )
                    if lane.user_sentiment is not None
                    and now - lane.sentiment_seen < SENTIMENT_CONTEXT_SECONDS
                    else None,
                    duration_seconds=round(now - lane.started),
                    quiet_seconds=round(now - lane.last_seen),
                    turn_active=lane.active,
                    last_tool=lane.last_tool,
                    presence=lane.presence,
                    owner=lane.owner,
                    owner_last_alive_seconds=None
                    if lane.owner_last_alive is None
                    else round(now - lane.owner_last_alive),
                    activity_uncertain=lane.activity_uncertain,
                    waiting_for_input=lane.waiting,
                    waiting_seconds=None
                    if lane.waiting_since is None
                    else round(now - lane.waiting_since),
                    active_tools=tuple(
                        ToolSnapshot(
                            call=call_id,
                            tool=call.tool,
                            activity=call.activity,
                            duration_seconds=round(now - call.started),
                        )
                        for call_id, call in lane.calls.items()
                    ),
                    progress=lane.progress,
                    failure=None
                    if lane.failure is None
                    else FailureSnapshot(
                        activity=lane.failure.activity,
                        count=lane.failure.count,
                        goal=lane.failure.goal,
                        resolved=lane.failure.resolved,
                        current_turn=lane.failure.generation == lane.generation,
                        seconds_ago=round(now - lane.failure.at),
                    ),
                    outcome=lane.outcome,
                )
                for key, lane in lanes
            ),
            convergence=self.convergence if keys is None else None,
        )
