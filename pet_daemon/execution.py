from dataclasses import dataclass


@dataclass
class Action:
    name: str
    command_id: int
    event: str | None
    until: float
    role: str
    kind: str


class Execution:
    transitions = {
        ("idle", "start"): "executing",
        ("executing", "finish"): "settling",
        ("settling", "ready"): "idle",
        ("idle", "suspend"): "suspended",
        ("executing", "suspend"): "suspended",
        ("settling", "suspend"): "suspended",
        ("suspended", "resume"): "idle",
    }

    def __init__(self):
        self.phase = "idle"
        self.action = None
        self.last_completed = None
        self.settle_until = 0

    def transition(self, event):
        self.phase = self.transitions[(self.phase, event)]

    def start(self, action):
        self.transition("start")
        self.action = action

    def finish(self, now, pause, interrupted=False):
        if not interrupted:
            self.last_completed = self.action.name
        self.action = None
        self.transition("finish")
        self.settle_until = now + pause

    def completion(self, packet, now, pause):
        action = self.action
        if (self.phase != "executing" or action.event is None
                or packet.get("command_id") != action.command_id
                or packet.get("event") != action.event
                or packet.get("result") not in ("finished", "interrupted")):
            return False
        self.finish(now, pause, packet["result"] == "interrupted")
        return True

    def advance(self, now, pause):
        if self.phase == "executing" and now >= self.action.until:
            self.finish(now, pause)
            return True
        if self.phase == "settling" and now >= self.settle_until:
            self.transition("ready")
        return False

    def interrupt(self):
        self.action = None
        self.phase = "idle"

    def suspend(self):
        if self.phase != "suspended":
            self.action = None
            self.transition("suspend")

    def resume(self):
        if self.phase == "suspended":
            self.transition("resume")
