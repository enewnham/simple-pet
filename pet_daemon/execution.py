from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar


@dataclass
class Action:
    name: str
    command_id: int
    event: str | None
    until: float
    role: str
    kind: str


class Execution:
    transitions: ClassVar[dict[tuple[str, str], str]] = {
        ("idle", "start"): "executing",
        ("executing", "finish"): "settling",
        ("settling", "ready"): "idle",
        ("idle", "suspend"): "suspended",
        ("executing", "suspend"): "suspended",
        ("settling", "suspend"): "suspended",
        ("suspended", "resume"): "idle",
    }

    def __init__(self) -> None:
        self.phase = "idle"
        self.action: Action | None = None
        self.last_completed: str | None = None
        self.settle_until = 0

    def transition(self, event: str) -> None:
        self.phase = self.transitions[(self.phase, event)]

    def start(self, action: Action) -> None:
        self.transition("start")
        self.action = action

    def finish(self, now: float, pause: float, interrupted: bool = False) -> None:
        action = self.action
        assert action is not None
        if not interrupted:
            self.last_completed = action.name
        self.action = None
        self.transition("finish")
        self.settle_until = now + pause

    def completion(
        self, packet: Mapping[str, object], now: float, pause: float
    ) -> bool:
        action = self.action
        if self.phase != "executing":
            return False
        assert action is not None
        if (
            action.event is None
            or packet.get("command_id") != action.command_id
            or packet.get("event") != action.event
            or packet.get("result") not in ("finished", "interrupted")
        ):
            return False
        self.finish(now, pause, packet["result"] == "interrupted")
        return True

    def advance(self, now: float, pause: float) -> bool:
        if self.phase == "executing":
            action = self.action
            assert action is not None
            if now >= action.until:
                self.finish(now, pause)
                return True
        if self.phase == "settling" and now >= self.settle_until:
            self.transition("ready")
        return False

    def interrupt(self) -> None:
        self.action = None
        self.phase = "idle"

    def suspend(self) -> None:
        if self.phase != "suspended":
            self.action = None
            self.transition("suspend")

    def resume(self) -> None:
        if self.phase == "suspended":
            self.transition("resume")
