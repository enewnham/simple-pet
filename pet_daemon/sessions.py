from dataclasses import dataclass, field


EVENTS = {
    "session_start", "session_end", "intent", "tool_start", "tool_done",
    "tool_error", "turn_done", "turn_error", "interrupt", "subagent_start",
    "subagent_done", "progress", "waiting",
}
HARNESS = {"cli", "claude", "codex", "cursor"}


@dataclass
class Call:
    tool: str
    started: float


@dataclass
class Lane:
    harness: str
    id: str
    started: float
    last_seen: float
    parent: str | None = None
    goal: str = ""
    last_tool: str = ""
    progress: str = ""
    failure: dict | None = None
    calls: dict = field(default_factory=dict)
    active: bool = False
    finished: bool = False
    turn: str | None = None
    generation: int = 0
    outcome: str | None = None
    ended: float | None = None
    cancelled: bool = False
    waiting: bool = False
    waiting_call: str | None = None


class Sessions:
    def __init__(self, clock, stale_seconds=3600, retention_seconds=7200):
        self.clock = clock
        self.stale_seconds = stale_seconds
        self.retention_seconds = retention_seconds
        self.lanes = {}
        self.aliases = {}
        self.child_calls = {}
        self.revision = 0
        self.members = set()
        self.convergence = None

    @property
    def working(self):
        return any(lane.calls for lane in self.lanes.values())

    @property
    def active(self):
        return any(lane.active for lane in self.lanes.values())

    @property
    def failed(self):
        return any(self._current_failure(lane) for lane in self.lanes.values())

    def _current_failure(self, lane):
        return lane.failure and lane.failure["generation"] == lane.generation and not lane.failure["resolved"]

    def _failure(self, lane, detail, now, tool=None):
        return {"tool": lane.last_tool if tool is None else tool, "detail": detail, "at": now,
                "resolved": False, "generation": lane.generation, "goal": lane.goal}

    @property
    def clean_convergence(self):
        return self.convergence is not None and not self.active and all(item["outcome"] == "success" for item in self.convergence)

    def _key(self, harness, identifier):
        key = f"{harness}:{identifier}"
        return self.aliases.get(key, key)

    def _lane(self, event, now):
        harness = event["harness"]
        identifier = event["session"]
        key = self._key(harness, identifier)
        routed = False
        parent_call = event.get("parent_call")
        if parent_call and event["event"] in ("tool_start", "tool_done", "tool_error", "waiting"):
            target = self.child_calls.get((harness, identifier, parent_call))
            if target and target != key:
                key = target
                routed = True
        alternate = event.get("aliases", [])
        if not isinstance(alternate, list) or not all(isinstance(alias, str) and alias for alias in alternate):
            raise ValueError("Invalid aliases")
        matches = [self._key(harness, alias) for alias in alternate]
        if key not in self.lanes:
            key = next((match for match in matches if match in self.lanes), key)
        lane = self.lanes.setdefault(key, Lane(harness, identifier, now, now))
        for alias in alternate if routed else [identifier, *alternate]:
            old_key = self._key(harness, alias)
            if old_key != key and old_key in self.lanes:
                other = self.lanes.pop(old_key)
                lane.calls.update(other.calls)
                lane.active = lane.active or other.active
                lane.goal = lane.goal or other.goal
                lane.failure = lane.failure or other.failure
                lane.started = min(lane.started, other.started)
                self.aliases = {name: key if target == old_key else target for name, target in self.aliases.items()}
                self.child_calls = {name: key if target == old_key else target for name, target in self.child_calls.items()}
                for child in self.lanes.values():
                    if child.parent == old_key:
                        child.parent = key
                if old_key in self.members:
                    self.members.remove(old_key)
                    self.members.add(key)
            self.aliases[f"{harness}:{alias}"] = key
        parent = event.get("parent")
        if parent:
            lane.parent = self._key(harness, parent)
            if lane.parent == key:
                raise ValueError("A lane cannot parent itself")
            if parent_call:
                self.child_calls[(harness, parent, parent_call)] = key
            parent_lane = self.lanes.get(lane.parent)
            if parent_lane and not lane.goal:
                lane.goal = parent_lane.goal
        return key, lane

    def _open(self, key, lane, event, now):
        if not self.active:
            self.members.clear()
            self.convergence = None
        lane.active = True
        lane.finished = False
        lane.outcome = None
        lane.cancelled = False
        lane.ended = None
        lane.generation += 1
        lane.turn = None if event["event"] == "subagent_start" else event.get("turn")
        lane.started = now
        lane.calls.clear()
        lane.last_tool = ""
        lane.progress = ""
        lane.waiting = False
        lane.waiting_call = None
        self.members.add(key)

    def _finish(self, lane, outcome, now):
        lane.calls.clear()
        lane.active = False
        lane.finished = True
        lane.ended = now
        lane.outcome = outcome
        lane.waiting = False
        lane.waiting_call = None

    def handle(self, event):
        kind = event.get("event")
        if event.get("harness") not in HARNESS or kind not in EVENTS or not isinstance(event.get("session"), str) or not event["session"]:
            return None
        for name in ("parent", "parent_call", "call", "turn", "tool", "goal", "detail", "progress"):
            if name in event and not isinstance(event[name], str):
                return None
        now = self.clock()
        previously_active = self.active
        previously_working = self.working
        key, lane = self._lane(event, now)
        turn = event.get("turn")
        if turn and lane.turn and turn != lane.turn and kind not in ("intent", "session_start", "subagent_start", "subagent_done", "session_end"):
            return None
        failure = False
        changed = False
        if kind == "session_start":
            if lane.finished:
                lane.finished = False
        elif kind in ("intent", "subagent_start"):
            new_native_turn = kind == "intent" and not event.get("manual")
            if not lane.active or new_native_turn or (kind == "intent" and turn and turn != lane.turn):
                self._open(key, lane, event, now)
            if "goal" in event:
                lane.goal = event["goal"]
            changed = True
        elif kind == "tool_start":
            if not event.get("call"):
                return None
            if lane.finished and not event.get("manual"):
                return None
            if not lane.active:
                self._open(key, lane, event, now)
            if turn and lane.turn is None:
                lane.turn = turn
            call_id = event["call"]
            if call_id in lane.calls:
                return None
            tool = event.get("tool", "")
            lane.calls[call_id] = Call(tool, now)
            lane.last_tool = tool
            lane.waiting = False
            lane.waiting_call = None
            lane.progress = event.get("progress", "")
            changed = True
        elif kind in ("tool_done", "tool_error"):
            call = lane.calls.pop(event.get("call"), None)
            if call is None:
                if lane.waiting and event.get("call") and event["call"] == lane.waiting_call:
                    lane.waiting = False
                    lane.waiting_call = None
                    lane.progress = event.get("progress", lane.progress)
                    lane.last_seen = now
                    self.revision += 1
                    return {"failure": False, "convergence": False}
                return None
            lane.waiting = False
            lane.waiting_call = None
            lane.progress = event.get("progress", lane.progress)
            if kind == "tool_error":
                lane.failure = self._failure(lane, event.get("detail", ""), now, call.tool)
                failure = True
                changed = True
            elif event.get("outcome") == "cancelled":
                lane.cancelled = True
            elif lane.failure and lane.failure["tool"] == call.tool:
                lane.failure["resolved"] = True
                changed = True
            if event.get("manual") and not lane.calls:
                outcome = "error" if self._current_failure(lane) else "cancelled" if lane.cancelled else "success"
                self._finish(lane, outcome, now)
                changed = True
        elif kind in ("turn_done", "subagent_done", "turn_error", "interrupt", "session_end"):
            if lane.finished and kind != "session_end":
                return None
            outcome = event.get("outcome", "success")
            if outcome not in ("success", "error", "cancelled"):
                return None
            duplicate_end = kind == "session_end" and lane.finished
            if (kind == "turn_error" or outcome == "error") and not duplicate_end:
                outcome = "error"
                if event.get("detail") or not self._current_failure(lane):
                    lane.failure = self._failure(lane, event.get("detail", ""), now)
                failure = True
            elif kind in ("interrupt", "session_end"):
                outcome = event.get("outcome") or ("cancelled" if lane.active else lane.outcome or "cancelled")
            elif self._current_failure(lane):
                outcome = "error"
            elif lane.cancelled:
                outcome = "cancelled"
            self._finish(lane, outcome, now)
            lane.progress = event.get("progress", lane.progress)
            if kind in ("session_end", "interrupt"):
                descendants = {key}
                for _ in range(len(self.lanes)):
                    descendants.update(child_key for child_key, child in self.lanes.items() if child.parent in descendants)
                for child_key in descendants - {key}:
                    child = self.lanes[child_key]
                    if child.active:
                        self._finish(child, "cancelled", now)
            changed = True
        elif kind in ("progress", "waiting"):
            if lane.finished:
                return None
            lane.progress = event.get("progress", event.get("detail", lane.progress))
            if kind == "waiting":
                lane.waiting = True
                lane.waiting_call = event.get("call")
        lane.last_seen = now
        if changed or previously_working != self.working:
            self.revision += 1
        convergence = previously_active and not self.active
        if convergence:
            self.convergence = [{"id": member, "outcome": self.lanes[member].outcome} for member in sorted(self.members) if member in self.lanes]
        return {"failure": failure, "convergence": convergence}

    def prune(self):
        now = self.clock()
        changed = False
        for lane in self.lanes.values():
            if lane.active and now - lane.last_seen >= self.stale_seconds:
                self._finish(lane, "stale", now)
                changed = True
        obsolete = {key for key, lane in self.lanes.items() if not lane.active and now - (lane.ended if lane.ended is not None else lane.last_seen) >= self.retention_seconds}
        for key in obsolete:
            del self.lanes[key]
        self.aliases = {name: key for name, key in self.aliases.items() if key not in obsolete}
        self.child_calls = {name: key for name, key in self.child_calls.items() if key not in obsolete}
        self.members.difference_update(obsolete)
        if changed:
            self.revision += 1
            self.convergence = None
        return changed

    def snapshot(self):
        now = self.clock()
        return {
            "activity": "working" if self.working else "idle",
            "sessions": [{
                "id": key, "parent": lane.parent, "goal": lane.goal,
                "duration_seconds": round(now - lane.started),
                "turn_active": lane.active, "last_tool": lane.last_tool,
                "waiting_for_input": lane.waiting,
                "active_tools": [{"call": call_id, "tool": call.tool, "duration_seconds": round(now - call.started)} for call_id, call in lane.calls.items()],
                "progress": lane.progress,
                "failure": None if not lane.failure else {
                    "tool": lane.failure["tool"], "detail": lane.failure["detail"],
                    "goal": lane.failure["goal"], "resolved": lane.failure["resolved"],
                    "current_turn": lane.failure["generation"] == lane.generation,
                    "seconds_ago": round(now - lane.failure["at"]),
                },
                "outcome": lane.outcome,
            } for key, lane in sorted(self.lanes.items())],
            "convergence": self.convergence,
        }
