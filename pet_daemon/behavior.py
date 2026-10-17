import math

from .execution import Action, Execution
from .policy import ANIMATION_ROLES, Observation, Policy, REACTIONS
from .telemetry import event, trace


class Behavior:
    def __init__(self, pack, config, clock, rng, emit):
        self.pack = pack
        self.config = config
        self.clock = clock
        self.rng = rng
        self.emit = emit
        self.policy = Policy()
        self.execution = Execution()
        self.status = None
        self.state = None
        self.clip = None
        self.command = 0
        self.idle_since = None
        self.inactive_since = None
        self.greeted_idle = False
        self.gesture_after = 0
        self.clear_since = None
        self.pointer_moved = -math.inf
        self.last_pointer = None
        self.preferred_position = None
        self.last_movement = clock()
        self.nap_until = None
        self.next_nap = None
        self.suspension = None
        self.observation = None
        self.next_care = None
        self.next_care_speech = None
        self.care_remark = None
        self.care_choices = []

    def satisfy(self):
        self.next_care = self.clock() + self.rng.uniform(*self.config.care_interval)
        self.next_care_speech = self.clock() + self.rng.uniform(*self.config.care_speech_interval)
        self.care_remark = None
        self.care_choices = []

    def _care(self, sessions):
        if not self.config.care or sessions.working or self.clock() < self.next_care or self._blocked():
            return False
        choices = [("exercise", "celebrate", "jumping"), ("attention", "greeting", "waving"),
                   ("consternation", "error", "failed")]
        available = [item for item in choices if self.pack.role(item[1])]
        if sessions.engaged:
            available = [item for item in available if item[0] == "consternation"]
        self.next_care = self.clock() + self.rng.uniform(*self.config.care_interval)
        if not available:
            return False
        if sessions.engaged:
            kind, role, preferred = self.rng.choice(available)
        else:
            if not self.care_choices:
                self.care_choices = [item for item in available for _ in range(1 if item[0] == "consternation" else 2)]
            selected = self.rng.choice(self.care_choices)
            self.care_choices.remove(selected)
            kind, role, preferred = selected
        clip = self._resolve_clip(role, preferred, "idle")
        self._play(clip, "care", "reaction")
        if not sessions.engaged and self.clock() >= self.next_care_speech:
            self.care_remark = kind
            self.next_care_speech = self.clock() + self.rng.uniform(*self.config.care_speech_interval)
        event("behavior.care", activity=kind)
        return True

    @property
    def pending(self):
        action = self.execution.action
        if action is None or action.event is None:
            return None
        return {"id": action.command_id, "event": action.event, "kind": action.kind,
                "role": action.role, "until": action.until}

    @property
    def sleeping(self):
        return self.nap_until is not None

    def _next_id(self):
        self.command += 1
        return self.command

    def _pause(self):
        return self.config.working_pause if self.observation and self.observation.working else self.config.idle_pause

    def _rest(self, clip_id=None):
        clip_id = clip_id or self.pack.rest
        if self.clip != clip_id:
            self.clip = clip_id
            self.emit({"animation": {"id": clip_id, "repeats": None, "command_id": self._next_id()}})

    def stop_movement(self):
        if self.execution.action and self.execution.action.kind == "walk":
            self._interrupt("interaction")
        self.emit({"stop_motion": True})

    def _interrupt(self, reason):
        trace("behavior.interrupted", reason=reason, pending=self.pending)
        self.execution.interrupt()
        self.clip = None
        self.emit({"stop_motion": True})

    def suspend(self, reason):
        self.care_remark = None
        self.wake(reason)
        self.suspension = reason
        self.execution.suspend()
        self.clip = None
        self.emit({"stop_motion": True})
        self._rest()

    def resume(self):
        if self.suspension == "system_sleep":
            self.satisfy()
        if self.suspension == "interaction":
            self.wake("interaction_finished")
        self.suspension = None
        self.execution.resume()

    def wake(self, reason):
        if reason != "nap_finished" and self.inactive_since is not None:
            self.inactive_since = self.clock()
        self.next_nap = self.clock() + self.rng.uniform(*self.config.nap_interval)
        if not self.sleeping:
            return
        self.nap_until = None
        self.emit({"sleeping": False})
        event("behavior.woke", reason=reason)
        self._interrupt(reason)
        self._rest()

    def renderer_status(self, packet):
        for name in ("x", "y", "width", "height"):
            if type(packet.get(name)) not in (int, float) or not math.isfinite(packet[name]):
                return False
        if packet["width"] <= 0 or packet["height"] <= 0:
            return False
        screen, pointer = packet.get("screen"), packet.get("pointer")
        if not isinstance(screen, dict) or not isinstance(pointer, dict):
            return False
        for item, keys in ((screen, ("x", "y", "width", "height")), (pointer, ("x", "y"))):
            if any(type(item.get(name)) not in (int, float) or not math.isfinite(item[name]) for name in keys):
                return False
        if screen["width"] <= 0 or screen["height"] <= 0:
            return False
        for name in ("dragging", "menu", "speech"):
            if type(packet.get(name)) is not bool:
                return False
        position = (pointer["x"], pointer["y"])
        if position != self.last_pointer:
            self.pointer_moved = self.clock()
            self.last_pointer = position
        if self.status is None:
            self.state = None
        if self.status is None or any(self.status[name] != packet[name] for name in ("screen", "dragging", "menu", "speech")):
            trace("renderer.status", packet=packet)
        if self.preferred_position is None or (self.status and self.status["dragging"] and not packet["dragging"]):
            self.preferred_position = (packet["x"], packet["y"])
            self.last_movement = self.clock()
        screen = packet["screen"]
        home_x, home_y = self.preferred_position
        self.preferred_position = (
            max(screen["x"], min(home_x, max(screen["x"], screen["x"] + screen["width"] - packet["width"]))),
            max(screen["y"], min(home_y, max(screen["y"], screen["y"] + screen["height"] - packet["height"]))),
        )
        self.status = packet
        return True

    def completion(self, packet):
        walking = self.execution.action and self.execution.action.kind == "walk"
        if self.execution.completion(packet, self.clock(), self.rng.uniform(*self._pause())):
            if walking:
                self.last_movement = self.clock()
            self.clip = None
            self._rest()

    def _near_pointer(self):
        status = self.status
        return math.hypot(status["pointer"]["x"] - (status["x"] + status["width"] / 2),
                          status["pointer"]["y"] - (status["y"] + 52)) <= self.config.pointer_radius

    def _blocked(self):
        return not self.status or any(self.status[name] for name in ("speech", "dragging", "menu")) or self._near_pointer()

    def _play(self, clip, name, kind="gesture", seconds=None):
        if kind in ("error", "success", "reaction"):
            repeats = 1
        elif seconds is not None:
            repeats = max(1, math.ceil(seconds / clip.seconds))
            if clip.max_repeats is not None:
                repeats = min(repeats, clip.max_repeats)
        else:
            minimum, maximum = self.config.repeat_range
            maximum = min(maximum, clip.max_repeats or math.inf,
                          max(1, math.floor(self.config.repeat_budget / clip.seconds)))
            minimum = min(minimum, maximum)
            repeats = self.rng.randint(minimum, maximum)
        command = self._next_id()
        duration = clip.seconds * repeats
        self.execution.start(Action(name, command, "animation", self.clock() + duration + 1, clip.role, kind))
        self.clip = clip.id
        self.emit({"stop_motion": True, "animation": {"id": clip.id, "repeats": repeats, "command_id": command}})
        self.gesture_after = self.clock() + duration + self.config.gesture_cooldown

    def _resolve_clip(self, role, preferred, fallback):
        for candidate_role, candidate_id in ((role, preferred), (ANIMATION_ROLES[fallback], fallback)):
            clips = self.pack.role(candidate_role)
            preferred_clips = [clip for clip in clips if clip.id == candidate_id]
            custom_clips = [clip for clip in clips if clip.id not in ANIMATION_ROLES]
            candidates = preferred_clips or custom_clips
            if candidates:
                return self.rng.choice(candidates)
        return self.pack.clips[self.pack.rest]

    def _reaction_clip(self, kind):
        spec = REACTIONS[kind]
        return self._resolve_clip(spec.role, spec.preferred, spec.fallback)

    def react(self, kind, sessions=None, key=None):
        clip = self._reaction_clip(kind)
        if not self.status or self.suspension or self.status["dragging"] or self.status["menu"]:
            return False
        current = self.execution.action
        if current and current.name in REACTIONS:
            if current.name == kind or REACTIONS[kind].priority < REACTIONS[current.name].priority:
                return False
        self.wake(kind)
        self._interrupt(kind)
        if clip.id == self.pack.rest:
            self._rest()
            self.execution.start(Action(kind, self.command, None, self.clock() + self.config.baseline_dwell,
                                        "rest", REACTIONS[kind].kind))
        else:
            self._play(clip, kind, REACTIONS[kind].kind)
        return True

    def success(self):
        return self.react("success")

    def attention(self, sessions, key):
        lane = sessions.lanes[key]
        return self.react("waiting" if lane.waiting else "investigation", sessions, key) if lane.waiting or lane.phase == "investigating" else False

    def gesture(self, clip_id, sessions):
        clip = self.pack.clips.get(clip_id)
        if (not clip or clip_id not in self.pack.gesture_ids or not self.status or self.suspension
                or self.status["dragging"] or self.status["menu"] or self.clock() < self.gesture_after):
            return False
        if ((clip.role == "celebrate" and not sessions.clean_convergence)
                or (clip.role == "error" and not sessions.failed)
                or (clip.role == "work" and not sessions.working)
                or (clip.role == "attention" and not sessions.engaged)):
            return False
        self.wake("gesture")
        self._interrupt("gesture")
        self._play(clip, "gesture")
        return True

    def greet(self, sessions):
        if (not self.status or sessions.engaged or self.idle_since is None
                or self.clock() - self.idle_since < self.config.idle_greeting_seconds):
            return False
        accepted = self.react("click", sessions)
        self.greeted_idle = accepted
        return accepted

    def _walk(self):
        status = self.status
        seconds = self.rng.uniform(*self.config.walk_seconds)
        angle = self.rng.uniform(0, 2 * math.pi)
        x = status["x"] + math.cos(angle) * self.config.walk_speed * seconds
        y = status["y"] + math.sin(angle) * self.config.walk_speed * seconds
        home_x, home_y = self.preferred_position
        distance = math.hypot(x - home_x, y - home_y)
        if distance > self.config.roaming_radius:
            scale = self.config.roaming_radius / distance
            x, y = home_x + (x - home_x) * scale, home_y + (y - home_y) * scale
        screen = status["screen"]
        x = max(screen["x"], min(x, max(screen["x"], screen["x"] + screen["width"] - status["width"])))
        y = max(screen["y"], min(y, max(screen["y"], screen["y"] + screen["height"] - status["height"])))
        distance = math.hypot(x - status["x"], y - status["y"])
        if distance < 1:
            return False
        seconds = min(seconds, distance / self.config.walk_speed)
        clip = self.rng.choice(self.pack.role("walk-left" if x < status["x"] else "walk-right"))
        if clip.max_repeats is not None and seconds > clip.seconds * clip.max_repeats:
            scale = clip.seconds * clip.max_repeats / seconds
            x = status["x"] + (x - status["x"]) * scale
            y = status["y"] + (y - status["y"]) * scale
            seconds = clip.seconds * clip.max_repeats
        command = self._next_id()
        self.last_movement = self.clock()
        self.execution.start(Action("roam", command, "motion", self.clock() + seconds + 1, clip.role, "walk"))
        self.clip = clip.id
        self.emit({"animation": {"id": clip.id, "repeats": clip.max_repeats, "command_id": command},
                   "motion": {"x": x, "y": y, "seconds": seconds, "command_id": command}})
        return True

    def _gaze_clip(self):
        if not self.pack.gaze:
            return None
        status = self.status
        anchor = self.pack.gaze_anchor
        dx = status["pointer"]["x"] - (status["x"] + 52 + 96 * anchor["x"])
        dy = status["pointer"]["y"] - (status["y"] + 4 + 96 * (1 - anchor["y"]))
        if math.hypot(dx, dy) <= self.config.gaze_deadzone:
            return None
        index = round(math.atan2(dx, dy) * 16 / (2 * math.pi)) % 16
        return self.pack.gaze[index]

    def _available(self, candidate):
        if candidate.name == "roam":
            return bool(self.pack.role("walk-left") and self.pack.role("walk-right"))
        if candidate.name == "observe":
            return self._gaze_clip() is not None
        return True

    def tick(self, state, sessions):
        if not self.status:
            return
        now = self.clock()
        status = self.status
        if sessions.working:
            self.next_care = None
            self.next_care_speech = None
            self.care_remark = None
        elif self.next_care is None:
            self.satisfy()
        if status["dragging"] or status["menu"]:
            if self.suspension is None:
                self.suspend("interaction")
            return
        if self.suspension == "interaction":
            self.resume()
        elif self.suspension:
            return
        starting = state == "working" and self.state in (None, "idle")
        if state != self.state:
            self.state = state
            event("behavior.state_changed", state=state)
            self.emit({"state": state})
        if sessions.engaged:
            self.inactive_since = None
            self.next_nap = None
            self.idle_since = None
            self.greeted_idle = False
        elif self.inactive_since is None:
            self.inactive_since = now
            self.idle_since = now
            self.next_nap = now + self.rng.uniform(*self.config.nap_interval)
        blocked = self._blocked()
        if blocked:
            if not sessions.engaged:
                self.inactive_since = now
                self.next_nap = now + self.rng.uniform(*self.config.nap_interval)
            self.clear_since = None
            if self.execution.action and self.execution.action.kind == "walk":
                self._interrupt("interaction")
        elif self.clear_since is None:
            self.clear_since = now
        inactive_seconds = 0 if self.inactive_since is None else now - self.inactive_since
        self.observation = Observation(
            now=now,
            active=sessions.engaged,
            working=sessions.working,
            investigating=any(lane.working and lane.phase == "investigating" for lane in sessions.lanes.values()),
            waiting=any(lane.waiting for lane in sessions.lanes.values()),
            interacting=status["speech"],
            pointer_recent=now - self.pointer_moved < self.config.gaze_reset_seconds,
            can_roam=(self.config.roaming and not blocked and self.clear_since is not None
                      and now - self.clear_since >= self.config.resume_seconds),
            can_nap=(self.config.naps and not sessions.engaged and not blocked and inactive_seconds >= 120
                     and self.next_nap is not None and now >= self.next_nap),
            restlessness=min(1, (now - self.last_movement) / 300),
            rest_pressure=min(1, inactive_seconds / 240),
            position=(status["x"], status["y"], status["width"], status["height"]),
            pointer=(status["pointer"]["x"], status["pointer"]["y"]),
            screen=tuple(status["screen"][name] for name in ("x", "y", "width", "height")),
            sessions=tuple((key, lane.generation, lane.working, lane.waiting, lane.phase,
                            tuple((call, item.activity) for call, item in lane.calls.items()))
                           for key, lane in sorted(sessions.lanes.items())),
        )
        if self.sleeping:
            if sessions.engaged or blocked or now >= self.nap_until:
                self.wake("activity_or_interaction" if sessions.engaged or blocked else "nap_finished")
                self._rest()
                return
            return
        if starting:
            current = self.execution.action
            if current is None or current.name not in REACTIONS:
                self.react("start", sessions)
        current = self.execution.action
        if (self.observation.pointer_recent and not status["speech"] and self._gaze_clip()
                and (current is None or current.name in ("rest", "observe"))):
            if current is None or current.name != "observe":
                self._interrupt("pointer")
                self._rest(self._gaze_clip())
                self.execution.start(Action("observe", self.command, None,
                                            now + self.config.gaze_reset_seconds, "gaze", "rest"))
            else:
                self._rest(self._gaze_clip())
                current.until = self.pointer_moved + self.config.gaze_reset_seconds
        if self.execution.advance(now, self.rng.uniform(*self._pause())):
            if current and current.event is not None:
                self.clip = None
            self._rest()
        if self.execution.phase == "settling":
            self._rest()
            return
        if self.execution.phase != "idle":
            return
        if state != "error" and self._care(sessions):
            return
        candidate = self.policy.select(self.observation, self.execution.last_completed, self.rng, self._available)
        if candidate.name == "roam" and self._walk():
            return
        if candidate.name == "nap":
            self.nap_until = now + self.rng.uniform(*self.config.nap_seconds)
            event("behavior.napping", duration_seconds=round(self.nap_until - now, 2))
            self.emit({"sleeping": True, "stop_motion": True})
            self._rest()
            self.execution.start(Action("nap", self.command, None, self.nap_until, "rest", "nap"))
            return
        if candidate.name == "focus":
            preferred, fallback = ("review", "running") if self.observation.investigating else ("running", "review")
            clip = self._resolve_clip(candidate.role, preferred, fallback)
            if clip.id == self.pack.rest:
                self._rest()
                duration = self.rng.uniform(*self._pause())
                self.execution.start(Action("focus", self.command, None, now + duration, "rest", "rest"))
            else:
                self._play(clip, "focus", "focus", self.rng.uniform(*self.config.work_burst_seconds))
            return
        clip_id = self._gaze_clip() if candidate.name == "observe" else self.pack.rest
        self._rest(clip_id)
        duration = self.config.gaze_reset_seconds if candidate.name == "observe" else self.rng.uniform(*self._pause())
        self.execution.start(Action(candidate.name, self.command, None, now + duration, "rest", "rest"))
