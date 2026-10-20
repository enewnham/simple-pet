from dataclasses import dataclass, field


@dataclass(frozen=True)
class Episode:
    kind: str
    session: str
    generation: int
    activity: str
    serial: int


@dataclass
class EpisodeState:
    lane: object
    generation: int
    started: bool = False
    waiting: bool = False
    waiting_serial: int = 0
    waiting_identity: tuple | None = None
    trouble: set = field(default_factory=set)
    unresolved: set = field(default_factory=set)
    category_serials: dict = field(default_factory=dict)
    terminal: bool = False
    investigating: bool = False


class Episodes:
    def __init__(self):
        self.states = {}
        self.serial = 0

    def handle(self, packet, result, sessions):
        reconciled = {}
        for (key, generation), state in self.states.items():
            if state.lane.generation != state.generation:
                continue
            canonical = sessions.aliases.get(key, key)
            lane = sessions.lanes.get(canonical)
            if lane is None or (canonical == key and generation != lane.generation):
                continue
            state.lane = lane
            state.generation = lane.generation
            identity = (canonical, lane.generation if canonical != key else generation)
            previous = reconciled.get(identity)
            if previous:
                previous.started |= state.started
                previous.waiting |= state.waiting
                waiting_identity = (
                    (lane.waiting_call, lane.waiting_since) if lane.waiting else None
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
        key = result["session"]
        lane = sessions.lanes[key]
        state = self.states.setdefault(
            (key, lane.generation), EpisodeState(lane, lane.generation)
        )
        emitted = []

        def add(kind, activity=result["activity"]):
            self.serial += 1
            emitted.append(Episode(kind, key, lane.generation, activity, self.serial))
            if kind in ("trouble", "recovery"):
                state.category_serials[activity] = self.serial

        event = packet["event"]
        if lane.active and (
            not state.started
            or (event == "intent" and packet.get("manual") and packet.get("goal"))
        ):
            state.started = True
            add("start")
        waiting_identity = (
            (lane.waiting_call, lane.waiting_since) if lane.waiting else None
        )
        if lane.waiting and (
            not state.waiting or waiting_identity != state.waiting_identity
        ):
            add("waiting")
            state.waiting_serial = self.serial
        state.waiting = lane.waiting
        state.waiting_identity = waiting_identity
        unresolved = {
            category
            for category, failure in lane.failures.items()
            if failure["generation"] == lane.generation and not failure["resolved"]
        }
        recovered = state.unresolved - unresolved
        if lane.active:
            for category in sorted(recovered):
                add("recovery", category)
        state.trouble.difference_update(recovered)
        state.unresolved = unresolved
        terminal = lane.finished and event in (
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
                if lane.outcome == "error"
                else "cancelled"
                if lane.outcome == "cancelled"
                else "success"
            )
        elif result["failure"]:
            failure = lane.failures[result["activity"]]
            if failure["count"] >= 3 and result["activity"] not in state.trouble:
                state.trouble.add(result["activity"])
                add("trouble")
            elif result["activity"] not in state.trouble:
                add("investigation")
        elif lane.phase == "investigating" and not state.investigating:
            add("investigation")
        state.investigating = lane.phase == "investigating"
        return emitted

    def relevant(self, trigger, sessions):
        key = sessions.aliases.get(trigger["target_session"], trigger["target_session"])
        lane = sessions.lanes.get(key)
        if lane is None or lane.generation != trigger["generation"]:
            return False
        kind = trigger["episode"]
        state = self.states.get((key, lane.generation))
        if kind == "waiting":
            return bool(
                lane.waiting and state and state.waiting_serial == trigger["serial"]
            )
        if kind in ("trouble", "recovery"):
            if (
                not state
                or state.category_serials.get(trigger["activity"]) != trigger["serial"]
            ):
                return False
        if kind == "trouble":
            failure = lane.failures.get(trigger["activity"])
            return bool(
                failure
                and not failure["resolved"]
                and failure["generation"] == lane.generation
            )
        if kind == "failure":
            return lane.finished and lane.outcome == "error"
        if kind in ("success", "cancelled"):
            return lane.finished and lane.outcome == (
                "success" if kind == "success" else "cancelled"
            )
        if kind == "recovery":
            failure = lane.failures.get(trigger["activity"])
            return bool(lane.active and failure and failure["resolved"])
        return lane.active
