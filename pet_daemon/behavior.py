import math

from .telemetry import trace


class Behavior:
    def __init__(self, pack, config, clock, rng, emit):
        self.pack = pack
        self.config = config
        self.clock = clock
        self.rng = rng
        self.emit = emit
        self.status = None
        self.state = None
        self.clip = None
        self.command = 0
        self.pending = None
        self.baseline_started = 0
        self.baseline_until = None
        self.idle_since = None
        self.next_action = 0
        self.gesture_after = 0
        self.clear_since = None
        self.pointer_moved = -math.inf
        self.last_pointer = None
        self.blocked = None
        self.nap_until = None
        self.next_nap = None

    @property
    def sleeping(self):
        return self.nap_until is not None

    def wake(self, reason):
        self.next_nap = self.clock() + self.rng.uniform(*self.config.nap_interval)
        if not self.sleeping:
            return
        self.nap_until = None
        self.emit({"sleeping": False})
        self._baseline(self.pack.rest)
        self.next_action = self.clock() + self.rng.uniform(*self._pause())
        trace("behavior.woke", reason=reason, next_nap=self.next_nap)

    def _nap(self):
        now = self.clock()
        clips = self.pack.role("sleep")
        clip_id = self.rng.choice(clips).id if clips else self.pack.rest
        self.nap_until = now + self.rng.uniform(*self.config.nap_seconds)
        self.emit({"sleeping": True, "stop_motion": True})
        self._baseline(clip_id, source="nap")
        trace("behavior.napping", clip=clip_id, until=self.nap_until)

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
        self.status = packet
        return True

    def completion(self, packet):
        if not self.pending or packet.get("command_id") != self.pending["id"]:
            return
        if packet.get("event") != self.pending["event"] and packet.get("result") != "interrupted":
            return
        if packet.get("result") not in ("finished", "interrupted"):
            return
        trace("behavior.completed", pending=self.pending, packet=packet, position={"x": self.status["x"], "y": self.status["y"]})
        self.pending = None
        self.clip = None
        self.next_action = self.clock() + self.rng.uniform(*self._pause())
        trace("behavior.pause", next_action=self.next_action)

    def _pause(self):
        return self.config.working_pause if self.state == "working" else self.config.idle_pause

    def _next_id(self):
        self.command += 1
        return self.command

    def _baseline(self, clip_id, source="rest"):
        if self.clip == clip_id and not self.pending:
            return
        command = self._next_id()
        self.clip = clip_id
        self.baseline_started = self.clock()
        self.baseline_until = None
        trace("behavior.animation", source=source, clip=clip_id, repeats=None, command_id=command)
        self.emit({"animation": {"id": clip_id, "repeats": None, "command_id": command}})

    def _stop(self, reason):
        trace("behavior.interrupted", reason=reason, pending=self.pending, clip=self.clip)
        self.pending = None
        self.clip = None
        self.baseline_until = None
        self.emit({"stop_motion": True})

    def _play(self, clip, kind="gesture", source="routine"):
        minimum, maximum = self.config.repeat_range
        maximum = min(maximum, clip.max_repeats or maximum, max(1, math.floor(self.config.repeat_budget / clip.seconds)))
        minimum = min(minimum, maximum)
        repeats = 1 if kind == "error" else self.rng.randint(minimum, maximum)
        command = self._next_id()
        duration = clip.seconds * repeats
        self.pending = {"id": command, "event": "animation", "kind": kind, "role": clip.role, "until": self.clock() + duration + 1}
        self.clip = clip.id
        self.baseline_until = None
        trace("behavior.animation", source=source, kind=kind, clip=clip.id, repeats=repeats, duration_seconds=duration, command_id=command)
        self.emit({"stop_motion": True, "animation": {"id": clip.id, "repeats": repeats, "command_id": command}})
        if kind == "gesture":
            self.gesture_after = self.clock() + duration + self.config.gesture_cooldown

    def _blocked(self):
        if not self.status:
            return True
        status = self.status
        distance = math.hypot(status["pointer"]["x"] - (status["x"] + status["width"] / 2),
                              status["pointer"]["y"] - (status["y"] + 52))
        return self.state == "error" or status["speech"] or status["dragging"] or status["menu"] or distance <= self.config.pointer_radius

    def gesture(self, clip_id, sessions):
        clip = self.pack.clips.get(clip_id)
        reason = None
        if not clip or clip_id not in self.pack.gesture_ids:
            reason = "unavailable_clip"
        elif not self.status or self.state == "error":
            reason = "renderer_unavailable_or_error"
        elif self.status["dragging"] or self.status["menu"]:
            reason = "interaction"
        elif self.clock() < self.gesture_after:
            reason = "cooldown"
        elif clip.role == "celebrate" and not sessions.clean_convergence:
            reason = "no_clean_convergence"
        elif clip.role == "error" and not sessions.failed:
            reason = "no_unresolved_failure"
        elif clip.role == "work" and not sessions.working:
            reason = "no_active_work"
        elif clip.role == "attention" and not sessions.active:
            reason = "no_active_turn"
        if reason:
            trace("behavior.gesture_rejected", clip=clip_id, reason=reason)
            return False
        self.wake("gesture")
        self._play(clip, source="observation")
        return True

    def _walk(self):
        left, right = self.pack.role("walk-left"), self.pack.role("walk-right")
        if not self.config.roaming or not left or not right or self._blocked():
            return False
        now = self.clock()
        if self.clear_since is None or now - self.clear_since < self.config.resume_seconds:
            return False
        status = self.status
        seconds = self.rng.uniform(*self.config.walk_seconds)
        angle = self.rng.uniform(0, 2 * math.pi)
        x = status["x"] + math.cos(angle) * self.config.walk_speed * seconds
        y = status["y"] + math.sin(angle) * self.config.walk_speed * seconds
        screen = status["screen"]
        x = max(screen["x"], min(x, max(screen["x"], screen["x"] + screen["width"] - status["width"])))
        y = max(screen["y"], min(y, max(screen["y"], screen["y"] + screen["height"] - status["height"])))
        if math.hypot(x - status["x"], y - status["y"]) < 1:
            return False
        clip = self.rng.choice(left if x < status["x"] else right)
        command = self._next_id()
        self.pending = {"id": command, "event": "motion", "kind": "walk", "role": clip.role, "until": now + seconds + 1}
        self.clip = clip.id
        self.baseline_until = None
        trace("behavior.walk", clip=clip.id, command_id=command, seconds=seconds,
              origin={"x": status["x"], "y": status["y"]}, target={"x": x, "y": y})
        self.emit({"animation": {"id": clip.id, "repeats": None, "command_id": command},
                   "motion": {"x": x, "y": y, "seconds": seconds, "command_id": command}})
        return True

    def _gaze(self):
        if not self.pack.gaze or self.state != "idle" or self.status["dragging"] or self.status["menu"] or self.status["speech"]:
            return False
        if self.clock() - self.pointer_moved >= self.config.gaze_reset_seconds:
            return False
        status = self.status
        anchor = self.pack.gaze_anchor
        dx = status["pointer"]["x"] - (status["x"] + 52 + 96 * anchor["x"])
        dy = status["pointer"]["y"] - (status["y"] + 4 + 96 * (1 - anchor["y"]))
        if math.hypot(dx, dy) <= self.config.gaze_deadzone:
            return False
        index = round(math.atan2(dx, dy) * 16 / (2 * math.pi)) % 16
        self._baseline(self.pack.gaze[index], source="gaze")
        return True

    def tick(self, state, sessions):
        if not self.status:
            return
        now = self.clock()
        nap_allowed = self.config.naps and state == "idle" and not sessions.active and not sessions.failed and not self._blocked()
        if not nap_allowed:
            if self.sleeping:
                self.wake("activity_or_interaction")
            self.next_nap = None
        elif self.sleeping:
            if now < self.nap_until:
                return
            self.wake("nap_finished")
        elif self.next_nap is None:
            self.next_nap = now + self.rng.uniform(*self.config.nap_interval)
        changed = state != self.state
        if changed:
            trace("behavior.state", previous=self.state, state=state)
            self.state = state
            self.emit({"state": state})
            self.idle_since = now if state == "idle" else None
            if state == "error":
                self._stop("error")
                candidates = self.pack.role("error")
                if candidates:
                    self._play(self.rng.choice(candidates), "error", source="failure")
                else:
                    self._baseline(self.pack.rest)
                return
        blocked = self._blocked()
        if blocked != self.blocked:
            trace("behavior.roaming_blocked", blocked=blocked, state=state, status=self.status)
            self.blocked = blocked
        if blocked:
            self.clear_since = None
            if self.pending and self.pending["kind"] == "walk":
                self._stop("roaming_blocked")
        elif self.clear_since is None:
            self.clear_since = now
        if state == "error":
            return
        if self.pending:
            if self.pending["kind"] == "error" or (self.pending["role"] == "celebrate" and not sessions.clean_convergence) or (self.pending["role"] == "error" and not sessions.failed):
                self._stop("activity_changed")
            elif self.pending["kind"] == "gesture" and (self.status["dragging"] or self.status["menu"]):
                self._stop("interaction")
            elif now >= self.pending["until"]:
                trace("behavior.timed_out", pending=self.pending)
                self.pending = None
                self.clip = None
                self.next_action = now + self.rng.uniform(*self._pause())
            else:
                return
        if state == "idle" and self.clip and self.clip != self.pack.rest and self.clip not in self.pack.gaze:
            if now < max((self.idle_since or now) + self.config.idle_debounce, self.baseline_started + self.config.baseline_dwell):
                return
            self.baseline_until = None
        if self.baseline_until is not None and now < self.baseline_until:
            return
        if self.baseline_until is not None:
            self.clip = None
            self.baseline_until = None
            self.next_action = now + self.rng.uniform(*self._pause())
        if state == "working" and (changed or now >= self.next_action) and not self.status["dragging"] and not self.status["menu"]:
            if not changed and self.rng.random() < 0.5 and self._walk():
                return
            candidates = self.pack.role("work") or self.pack.role("attention")
            if candidates:
                self._baseline(self.rng.choice(candidates).id, source="work_burst")
                self.baseline_until = now + self.rng.uniform(*self.config.work_burst_seconds)
                trace("behavior.work_burst", clip=self.clip, until=self.baseline_until)
            else:
                self.next_action = now + self.rng.uniform(*self._pause())
                self._baseline(self.pack.rest)
            return
        if nap_allowed and now >= self.next_nap:
            self._nap()
            return
        if now >= self.next_action and not self.status["dragging"] and not self.status["menu"]:
            self.next_action = now + self.rng.uniform(*self._pause())
            trace("behavior.routine", state=state, next_action=self.next_action, gesture_after=self.gesture_after)
            if self.rng.random() < 0.5 and self._walk():
                return
            candidates = self.pack.role("greeting", "gesture")
            if candidates and now >= self.gesture_after:
                self._play(self.rng.choice(candidates))
                return
        if self._gaze():
            return
        self._baseline(self.pack.rest)
