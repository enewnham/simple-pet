from dataclasses import dataclass, field
from enum import StrEnum
from typing import assert_never

from ..domain import Activity, ExecutionOutcome
from ..identity import SessionIdentity
from ..runtime import HookEvent, HookEventKind, WaitingKind
from .tracking import AgentSession, Sessions, SessionUpdate


class ReactionKind(StrEnum):
    START = "start"
    WAITING = "waiting"
    INVESTIGATION = "investigation"
    TROUBLE = "trouble"
    RECOVERY = "recovery"
    FAILURE = "failure"
    CANCELLED = "cancelled"
    SUCCESS = "success"


@dataclass(frozen=True)
class SessionReaction:
    kind: ReactionKind
    session: SessionIdentity
    generation: int
    activity: Activity
    serial: int


@dataclass
class ReactionState:
    session: AgentSession
    generation: int
    started: bool = False
    waiting: bool = False
    waiting_serial: int = 0
    waiting_identity: tuple[str | None, float | None, WaitingKind | None] | None = None
    trouble: set[Activity] = field(default_factory=set)
    unresolved: set[Activity] = field(default_factory=set)
    category_serials: dict[Activity, int] = field(default_factory=dict)
    terminal: bool = False
    investigating: bool = False

    def merge(self, other: ReactionState) -> None:
        self.started |= other.started
        self.waiting |= other.waiting
        waiting_identity = (
            (
                self.session.waiting_call,
                self.session.waiting_since,
                self.session.waiting_kind,
            )
            if self.session.waiting
            else None
        )
        if other.waiting_identity == waiting_identity and (
            self.waiting_identity != waiting_identity
            or other.waiting_serial > self.waiting_serial
        ):
            self.waiting_serial = other.waiting_serial
            self.waiting_identity = other.waiting_identity
        self.trouble.update(other.trouble)
        self.unresolved.update(other.unresolved)
        for category, serial in other.category_serials.items():
            self.category_serials[category] = max(
                serial, self.category_serials.get(category, 0)
            )
        self.terminal |= other.terminal
        self.investigating |= other.investigating


class SessionReactions:
    def __init__(self) -> None:
        self.states: dict[tuple[SessionIdentity, int], ReactionState] = {}
        self.serial = 0

    def _reconcile_sessions(self, sessions: Sessions) -> None:
        reconciled: dict[tuple[SessionIdentity, int], ReactionState] = {}
        for (key, generation), state in self.states.items():
            if state.session.generation != state.generation:
                continue
            canonical = sessions.aliases.get(key, key)
            session = sessions.by_id.get(canonical)
            if session is None or (
                canonical == key and generation != session.generation
            ):
                continue
            state.session = session
            state.generation = session.generation
            identity = (
                canonical,
                session.generation if canonical != key else generation,
            )
            previous = reconciled.get(identity)
            if previous:
                previous.merge(state)
            else:
                reconciled[identity] = state
        self.states = reconciled

    def handle(
        self, packet: HookEvent, result: SessionUpdate, sessions: Sessions
    ) -> list[SessionReaction]:
        self._reconcile_sessions(sessions)
        key = result.session
        session = sessions.by_id[key]
        state = self.states.setdefault(
            (key, session.generation), ReactionState(session, session.generation)
        )
        emitted = []

        def add(kind: ReactionKind, activity: Activity = result.activity) -> None:
            self.serial += 1
            emitted.append(
                SessionReaction(kind, key, session.generation, activity, self.serial)
            )
            if kind in (ReactionKind.TROUBLE, ReactionKind.RECOVERY):
                state.category_serials[activity] = self.serial

        event = packet.event
        if session.active and (
            not state.started
            or (event == HookEventKind.INTENT and packet.manual and packet.goal)
        ):
            state.started = True
            add(ReactionKind.START)
        waiting_identity = (
            (session.waiting_call, session.waiting_since, session.waiting_kind)
            if session.waiting
            else None
        )
        if session.waiting and (
            not state.waiting or waiting_identity != state.waiting_identity
        ):
            add(ReactionKind.WAITING)
            state.waiting_serial = self.serial
        state.waiting = session.waiting
        state.waiting_identity = waiting_identity
        unresolved = {
            category
            for category, failure in session.failures.items()
            if failure.generation == session.generation and not failure.resolved
        }
        recovered = state.unresolved - unresolved
        if session.active:
            for category in sorted(recovered):
                add(ReactionKind.RECOVERY, category)
        state.trouble.difference_update(recovered)
        state.unresolved = unresolved
        terminal = session.finished and event in (
            HookEventKind.TURN_DONE,
            HookEventKind.TURN_ERROR,
            HookEventKind.SUBAGENT_DONE,
            HookEventKind.INTERRUPT,
            HookEventKind.TOOL_DONE,
            HookEventKind.TOOL_ERROR,
        )
        if terminal and not state.terminal:
            state.terminal = True
            add(
                ReactionKind.FAILURE
                if session.outcome == ExecutionOutcome.ERROR
                else ReactionKind.CANCELLED
                if session.outcome == ExecutionOutcome.CANCELLED
                else ReactionKind.SUCCESS
            )
        elif result.failure:
            failure = session.failures[result.activity]
            if failure.count >= 3 and result.activity not in state.trouble:
                state.trouble.add(result.activity)
                add(ReactionKind.TROUBLE)
            elif result.activity not in state.trouble:
                add(ReactionKind.INVESTIGATION)
        elif session.phase == Activity.INVESTIGATING and not state.investigating:
            add(ReactionKind.INVESTIGATION)
        state.investigating = session.phase == Activity.INVESTIGATING
        return emitted

    def relevant(self, reaction: SessionReaction, sessions: Sessions) -> bool:
        key = sessions.aliases.get(reaction.session, reaction.session)
        session = sessions.by_id.get(key)
        if (
            session is None
            or session.retired
            or session.generation != reaction.generation
        ):
            return False
        kind = reaction.kind
        state = self.states.get((key, session.generation))
        if kind == ReactionKind.WAITING:
            return bool(
                session.waiting and state and state.waiting_serial == reaction.serial
            )
        if kind in (ReactionKind.TROUBLE, ReactionKind.RECOVERY) and (
            not state
            or state.category_serials.get(reaction.activity) != reaction.serial
        ):
            return False
        if kind == ReactionKind.TROUBLE:
            failure = session.failures.get(reaction.activity)
            return bool(
                failure
                and not failure.resolved
                and failure.generation == session.generation
            )
        if kind == ReactionKind.FAILURE:
            return session.finished and session.outcome == ExecutionOutcome.ERROR
        if kind in (ReactionKind.SUCCESS, ReactionKind.CANCELLED):
            return session.finished and session.outcome == (
                ExecutionOutcome.SUCCESS
                if kind == ReactionKind.SUCCESS
                else ExecutionOutcome.CANCELLED
            )
        if kind == ReactionKind.RECOVERY:
            failure = session.failures.get(reaction.activity)
            return bool(session.active and failure and failure.resolved)
        if kind in (ReactionKind.START, ReactionKind.INVESTIGATION):
            return session.active
        assert_never(kind)
