from dataclasses import dataclass, field

from ..runtime import HookEvent
from .tracking import AgentSession, Sessions, SessionUpdate


@dataclass(frozen=True)
class Episode:
    kind: str
    session: str
    generation: int
    activity: str
    serial: int


@dataclass
class EpisodeState:
    session: AgentSession
    generation: int
    started: bool = False
    waiting: bool = False
    waiting_serial: int = 0
    waiting_identity: tuple[str | None, float | None] | None = None
    trouble: set[str] = field(default_factory=set)
    unresolved: set[str] = field(default_factory=set)
    category_serials: dict[str, int] = field(default_factory=dict)
    terminal: bool = False
    investigating: bool = False


class Episodes:
    def __init__(self) -> None:
        self.states: dict[tuple[str, int], EpisodeState] = {}
        self.serial = 0

    def handle(
        self, packet: HookEvent, result: SessionUpdate, sessions: Sessions
    ) -> list[Episode]:
        reconciled: dict[tuple[str, int], EpisodeState] = {}
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
                previous.started |= state.started
                previous.waiting |= state.waiting
                waiting_identity = (
                    (session.waiting_call, session.waiting_since)
                    if session.waiting
                    else None
                )
                if state.waiting_identity == waiting_identity and (
                    previous.waiting_identity != waiting_identity
                    or state.waiting_serial > previous.waiting_serial
                ):
                    previous.waiting_serial = state.waiting_serial
                    previous.waiting_identity = state.waiting_identity
                previous.trouble.update(state.trouble)
                previous.unresolved.update(state.unresolved)
                for category, serial in state.category_serials.items():
                    previous.category_serials[category] = max(
                        serial, previous.category_serials.get(category, 0)
                    )
                previous.terminal |= state.terminal
                previous.investigating |= state.investigating
            else:
                reconciled[identity] = state
        self.states = reconciled
        key = result.session
        session = sessions.by_id[key]
        state = self.states.setdefault(
            (key, session.generation), EpisodeState(session, session.generation)
        )
        emitted = []

        def add(kind: str, activity: str = result.activity) -> None:
            self.serial += 1
            emitted.append(
                Episode(kind, key, session.generation, activity, self.serial)
            )
            if kind in ("trouble", "recovery"):
                state.category_serials[activity] = self.serial

        event = packet.event
        if session.active and (
            not state.started or (event == "intent" and packet.manual and packet.goal)
        ):
            state.started = True
            add("start")
        waiting_identity = (
            (session.waiting_call, session.waiting_since) if session.waiting else None
        )
        if session.waiting and (
            not state.waiting or waiting_identity != state.waiting_identity
        ):
            add("waiting")
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
                add("recovery", category)
        state.trouble.difference_update(recovered)
        state.unresolved = unresolved
        terminal = session.finished and event in (
            "turn_done",
            "turn_error",
            "subagent_done",
            "interrupt",
            "session_end",
            "tool_done",
            "tool_error",
        )
        if terminal and not state.terminal:
            state.terminal = True
            add(
                "failure"
                if session.outcome == "error"
                else "cancelled"
                if session.outcome == "cancelled"
                else "success"
            )
        elif result.failure:
            failure = session.failures[result.activity]
            if failure.count >= 3 and result.activity not in state.trouble:
                state.trouble.add(result.activity)
                add("trouble")
            elif result.activity not in state.trouble:
                add("investigation")
        elif session.phase == "investigating" and not state.investigating:
            add("investigation")
        state.investigating = session.phase == "investigating"
        return emitted

    def relevant(self, episode: Episode, sessions: Sessions) -> bool:
        key = sessions.aliases.get(episode.session, episode.session)
        session = sessions.by_id.get(key)
        if session is None or session.generation != episode.generation:
            return False
        kind = episode.kind
        state = self.states.get((key, session.generation))
        if kind == "waiting":
            return bool(
                session.waiting and state and state.waiting_serial == episode.serial
            )
        if kind in ("trouble", "recovery") and (
            not state or state.category_serials.get(episode.activity) != episode.serial
        ):
            return False
        if kind == "trouble":
            failure = session.failures.get(episode.activity)
            return bool(
                failure
                and not failure.resolved
                and failure.generation == session.generation
            )
        if kind == "failure":
            return session.finished and session.outcome == "error"
        if kind in ("success", "cancelled"):
            return session.finished and session.outcome == (
                "success" if kind == "success" else "cancelled"
            )
        if kind == "recovery":
            failure = session.failures.get(episode.activity)
            return bool(session.active and failure and failure.resolved)
        return session.active
