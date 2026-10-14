from dataclasses import dataclass, field

from .activity import activity


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
    activity: str = "command"


@dataclass
class Lane:
    harness: str
    id: str
    started: float
    last_seen: float
    parent: str | None = None
    goal: str = ""
    first_prompt: str = ""
    recent_requests: list = field(default_factory=list)
    session_context: str = ""
    context_loaded: bool = False
    repo: str = ""
    phase: str = ""
    last_tool: str = ""
    progress: str = ""
    failure: dict | None = None
    failures: dict = field(default_factory=dict)
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
    waiting_since: float | None = None
    last_check_in: float | None = None

    @property
    def working(self):
        return self.active and (not self.waiting or any(call != self.waiting_call for call in self.calls))


class Sessions:
    def __init__(self, clock, stale_seconds=3600, retention_seconds=7200):
        self.clock = clock
        self.stale_seconds = stale_seconds
        self.retention_seconds = retention_seconds
        self.lanes = {}
        self.aliases = {}
        self.child_calls = {}
        self.members = set()
        self.convergence = None

    @property
    def working(self):
        return any(lane.working for lane in self.lanes.values())

    @property
    def active(self):
        return any(lane.active for lane in self.lanes.values())

    @property
    def failed(self):
        return any(self._current_failure(lane) for lane in self.lanes.values())

    def _current_failure(self, lane):
        return any(failure["generation"] == lane.generation and not failure["resolved"] for failure in lane.failures.values())

    def _failure(self, lane, category, now, tool=""):
        previous = lane.failures.get(category)
        repeated = previous and not previous["resolved"]
        count = previous["count"] + 1 if repeated else 1
        failure = {"tool": tool, "activity": category, "at": now, "count": count,
                   "resolved": False, "generation": lane.generation, "goal": lane.goal}
        lane.failures[category] = failure
        return failure

    def reaction_state(self):
        return tuple((key, lane.generation, lane.active, lane.first_prompt, tuple(lane.recent_requests), lane.outcome, lane.waiting,
                      lane.phase, lane.progress, bool(self._current_failure(lane)))
                     for key, lane in sorted(self.lanes.items()) if lane.active or key in self.members)

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
                if not lane.active and (other.active or (other.finished and not lane.finished)):
                    lane, other = other, lane
                    self.lanes[key] = lane
                lane.calls.update(other.calls)
                lane.active = lane.active or other.active
                lane.goal = lane.goal or other.goal
                lane.first_prompt = lane.first_prompt or other.first_prompt
                lane.session_context = lane.session_context or other.session_context
                lane.recent_requests = lane.recent_requests or other.recent_requests
                lane.repo = lane.repo or other.repo
                if other.active:
                    for category, failure in other.failures.items():
                        if failure["generation"] == other.generation:
                            failure["generation"] = lane.generation
                            existing = lane.failures.get(category)
                            if existing is None or failure["at"] > existing["at"]:
                                lane.failures[category] = failure
                    unresolved = [item for item in lane.failures.values() if not item["resolved"]]
                    if unresolved:
                        lane.failure = max(unresolved, key=lambda item: item["at"])
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
            if parent_lane and not lane.repo:
                lane.repo = parent_lane.repo
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
        lane.failures.clear()
        lane.turn = None if event["event"] == "subagent_start" else event.get("turn")
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

    def _finish(self, lane, outcome, now):
        lane.calls.clear()
        lane.active = False
        lane.finished = True
        lane.ended = now
        lane.outcome = outcome
        if outcome == "cancelled":
            for failure in lane.failures.values():
                failure["resolved"] = True
        lane.waiting = False
        lane.waiting_call = None
        lane.waiting_since = None

    def handle(self, event):
        kind = event.get("event")
        if event.get("harness") not in HARNESS or kind not in EVENTS or not isinstance(event.get("session"), str) or not event["session"]:
            return None
        for name in ("parent", "parent_call", "call", "turn", "tool", "goal", "progress", "repo", "tool_name"):
            if name in event and not isinstance(event[name], str):
                return None
        now = self.clock()
        previously_active = self.active
        key, lane = self._lane(event, now)
        turn = event.get("turn")
        if turn and lane.turn and turn != lane.turn and kind not in ("intent", "session_start", "subagent_start", "subagent_done", "session_end"):
            return None
        if event.get("repo"):
            lane.repo = event["repo"]
        if kind == "subagent_start" and lane.active and lane.first_prompt:
            return None
        failure = False
        event_activity = lane.phase or "command"
        if kind == "session_start":
            if lane.finished:
                lane.finished = False
        elif kind in ("intent", "subagent_start"):
            new_native_turn = kind == "intent" and not event.get("manual")
            if not lane.active or new_native_turn or (kind == "intent" and turn and turn != lane.turn):
                self._open(key, lane, event, now)
            if event.get("goal"):
                request = " ".join(event["goal"].split())[:240]
                lane.first_prompt = lane.first_prompt or request
                lane.recent_requests = [*lane.recent_requests, request][-4:]
                lane.goal = lane.session_context or lane.first_prompt
            elif kind == "subagent_start" and not lane.first_prompt and lane.parent in self.lanes:
                parent = self.lanes[lane.parent]
                lane.first_prompt = parent.first_prompt
                lane.session_context = parent.session_context
                lane.goal = lane.session_context or lane.first_prompt
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
            event_activity = activity(tool, event.get("tool_name", ""))
            lane.calls[call_id] = Call(tool, now, event_activity)
            lane.last_tool = tool
            lane.phase = event_activity
            if not lane.waiting or call_id == lane.waiting_call:
                lane.waiting = False
                lane.waiting_call = None
                lane.waiting_since = None
            lane.progress = event.get("progress", "")
        elif kind in ("tool_done", "tool_error"):
            call = lane.calls.pop(event.get("call"), None)
            if call is None:
                if lane.waiting and event.get("call") and event["call"] == lane.waiting_call:
                    lane.waiting = False
                    lane.waiting_call = None
                    lane.waiting_since = None
                    lane.progress = "" if kind == "tool_error" else event.get("progress", lane.progress)
                    lane.last_seen = now
                    return {"failure": False, "convergence": False, "session": key, "activity": event_activity}
                return None
            event_activity = call.activity
            if not lane.waiting or event.get("call") == lane.waiting_call:
                lane.waiting = False
                lane.waiting_call = None
                lane.waiting_since = None
            lane.progress = "" if kind == "tool_error" or event.get("outcome") == "error" else event.get("progress", lane.progress)
            if kind == "tool_error":
                lane.failure = self._failure(lane, call.activity, now, call.tool)
                failure = True
            elif event.get("outcome") == "cancelled":
                lane.cancelled = True
            elif call.activity in lane.failures:
                recovered = lane.failures[call.activity]
                recovered["resolved"] = True
                unresolved = [item for item in lane.failures.values() if not item["resolved"]]
                lane.failure = max(unresolved, key=lambda item: item["at"]) if unresolved else recovered
            if event.get("manual") and not lane.calls:
                outcome = "error" if self._current_failure(lane) else "cancelled" if lane.cancelled else "success"
                self._finish(lane, outcome, now)
        elif kind in ("turn_done", "subagent_done", "turn_error", "interrupt", "session_end"):
            if lane.finished and kind != "session_end":
                return None
            outcome = event.get("outcome", "success")
            if outcome not in ("success", "error", "cancelled"):
                return None
            duplicate_end = kind == "session_end" and lane.finished
            if (kind == "turn_error" or outcome == "error") and not duplicate_end:
                outcome = "error"
                if kind == "turn_error" or not self._current_failure(lane):
                    lane.failure = self._failure(lane, "harness", now)
                event_activity = lane.failure["activity"]
                failure = True
            elif kind in ("interrupt", "session_end"):
                outcome = event.get("outcome") or ("cancelled" if lane.active else lane.outcome or "cancelled")
            elif self._current_failure(lane):
                outcome = "error"
            elif lane.cancelled:
                outcome = "cancelled"
            if outcome == "error" and lane.failure:
                event_activity = lane.failure["activity"]
            self._finish(lane, outcome, now)
            lane.progress = "" if outcome == "error" else event.get("progress", lane.progress)
            if kind in ("session_end", "interrupt"):
                descendants = {key}
                for _ in range(len(self.lanes)):
                    descendants.update(child_key for child_key, child in self.lanes.items() if child.parent in descendants)
                for child_key in descendants - {key}:
                    child = self.lanes[child_key]
                    if child.active:
                        self._finish(child, "cancelled", now)
        elif kind in ("progress", "waiting"):
            if lane.finished:
                return None
            if kind == "waiting" and not lane.active:
                self._open(key, lane, event, now)
            lane.progress = event.get("progress", lane.progress)
            if kind == "waiting":
                if not lane.waiting or lane.waiting_call != event.get("call"):
                    lane.waiting_since = now
                lane.waiting = True
                lane.waiting_call = event.get("call")
                lane.last_tool = event.get("tool", lane.last_tool)
        lane.last_seen = now
        convergence = previously_active and not self.active
        if convergence:
            self.convergence = [{"id": member, "outcome": self.lanes[member].outcome} for member in sorted(self.members) if member in self.lanes]
        return {"failure": failure, "convergence": convergence, "session": key, "activity": event_activity}

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
            self.convergence = None
        return changed

    def check_in_due(self, seconds):
        now = self.clock()
        return [key for key, lane in sorted(self.lanes.items()) if lane.active
                and now - max(lane.waiting_since if lane.waiting else lane.last_seen,
                              lane.last_check_in if lane.last_check_in is not None else lane.started) >= seconds]

    def observation_context(self, include_activity=False, keys=None):
        return {key: (lane.generation, lane.active, lane.outcome, lane.goal, bool(self._current_failure(lane)),
                      lane.waiting, lane.waiting_call,
                      (tuple(lane.calls), lane.waiting_since if lane.waiting else lane.last_seen) if include_activity else None)
                for key, lane in self.lanes.items() if keys is None or key in keys}

    def snapshot(self, keys=None):
        now = self.clock()
        lanes = [(key, lane) for key, lane in sorted(self.lanes.items()) if keys is None or key in keys]
        return {
            "activity": "working" if any(lane.working for _, lane in lanes) else "idle",
            "sessions": [{
                "id": key, "parent": lane.parent, "goal": lane.goal, "repo": lane.repo, "phase": lane.phase,
                "first_prompt": lane.first_prompt, "session_context": lane.session_context,
                "recent_requests": list(lane.recent_requests), "generation": lane.generation,
                "duration_seconds": round(now - lane.started),
                "quiet_seconds": round(now - lane.last_seen),
                "turn_active": lane.active, "last_tool": lane.last_tool,
                "waiting_for_input": lane.waiting,
                "waiting_seconds": None if lane.waiting_since is None else round(now - lane.waiting_since),
                "active_tools": [{"call": call_id, "tool": call.tool, "activity": call.activity, "duration_seconds": round(now - call.started)} for call_id, call in lane.calls.items()],
                "progress": lane.progress,
                "failure": None if not lane.failure else {
                    "activity": lane.failure["activity"], "count": lane.failure["count"],
                    "goal": lane.failure["goal"], "resolved": lane.failure["resolved"],
                    "current_turn": lane.failure["generation"] == lane.generation,
                    "seconds_ago": round(now - lane.failure["at"]),
                },
                "outcome": lane.outcome,
            } for key, lane in lanes],
            "convergence": self.convergence if keys is None else None,
        }
