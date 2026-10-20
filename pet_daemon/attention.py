from .telemetry import trace


def selected_sessions(snapshot, reason=None):
    trigger = snapshot.get("trigger") or {}
    target = trigger.get("target_session") or f"{trigger.get('harness', '')}:{trigger.get('session', '')}"

    def priority(session):
        waiting = reason == "click" and session.get("waiting_for_input", False)
        active = session.get("turn_active", False)
        quiet = session.get("quiet_seconds", 0)
        return (not waiting, -(session.get("waiting_seconds") or 0) if waiting else 0,
                session["id"] != target,
                not active, -quiet if active else quiet)

    return sorted(snapshot.get("sessions", []), key=priority)


def selected_subject(snapshot):
    subjects = snapshot.get("work", ())
    trigger = snapshot.get("trigger") or {}
    target = trigger.get("target_opportunity") or trigger.get("target_session")
    if target:
        match = next((subject for subject in subjects if subject.id == target), None)
        if match:
            return match
    return min(subjects, key=lambda subject: (subject.priority, -subject.changed_at, subject.id), default=None)


class Attention:
    def __init__(self, config, clock, rng):
        self.config = config
        self.clock = clock
        self.rng = rng
        self.pending = None
        self.last_attempt = float("-inf")
        self.next_observation = clock() + rng.uniform(*config.observation_seconds)
        self.selections = {}
        self.sequence = 0

    def reset(self):
        self.pending = None
        self.last_attempt = float("-inf")
        self.next_observation = self.clock() + self.rng.uniform(*self.config.observation_seconds)
        self.selections.clear()
        self.sequence = 0

    def queue(self, reason, trigger=None, failed=False):
        now = self.clock()
        if reason == "failure":
            self.pending = (reason, now + 1, trigger)
        elif reason == "check_in" or not self.pending or self.pending[0] != "failure" or not failed:
            delay = {"prompt": 0, "completion": 1, "check_in": 0}[reason]
            self.pending = (reason, now + delay, trigger)
        trace("observation.queued", reason=self.pending[0], scheduled_at=self.pending[1],
              session=(self.pending[2] or {}).get("target_session"), scheduled=self.pending)

    def choose(self, snapshot, reason="routine", eligible=None):
        candidates = selected_sessions(snapshot, reason)
        if eligible is not None:
            candidates = [session for session in candidates if session["id"] in eligible]
        if reason == "routine":
            candidates = [session for session in candidates
                          if session.get("turn_active") and not session.get("waiting_for_input") and not session.get("activity_uncertain")]
        if not candidates:
            return None
        trigger = snapshot.get("trigger") or {}
        target = trigger.get("target_session")
        if target:
            match = next((session for session in candidates if session["id"] == target), None)
            if match:
                return match
        if reason not in ("routine", "check_in", "click"):
            return candidates[0]
        if reason == "click":
            waiting = [session for session in candidates if session.get("waiting_for_input")]
            if waiting:
                candidates = waiting
        active = [session for session in candidates if session.get("turn_active")]
        if active:
            candidates = active
        return min(candidates, key=lambda session: self.selections.get(session["id"], -1))

    def mark_selected(self, session_id):
        self.sequence += 1
        self.selections[session_id] = self.sequence
        trace("attention.selected", session=session_id)

    def choose_work(self, snapshot):
        selected = selected_subject(snapshot)
        trigger = snapshot.get("trigger") or {}
        if selected is None or trigger.get("target_opportunity") or trigger.get("target_session"):
            return selected
        eligible = [subject for subject in snapshot.get("work", ())
                    if (subject.priority, subject.changed_at) == (selected.priority, selected.changed_at)]
        return min(eligible, key=lambda subject: self.selections.get(subject.id, -1))

    def allowed(self, reason, trigger=None):
        if reason in ("click", "explicit") or trigger and trigger.get("episode") == "waiting":
            return True
        return self.clock() - self.last_attempt >= self.config.behavior.automatic_speech_cooldown_seconds

    def scheduled(self, session_id=None):
        self.last_attempt = self.clock()
        self.next_observation = self.last_attempt + self.rng.uniform(*self.config.observation_seconds)
        if session_id is not None:
            self.mark_selected(session_id)
