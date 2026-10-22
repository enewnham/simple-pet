from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar

from ..domain import ActionKind, ExecutionPhase, ExecutionTransition


@dataclass
class Action:
    name: str
    command_id: int
    event: str | None
    until: float
    role: str
    kind: ActionKind


class Execution:
    transitions: ClassVar[
        dict[tuple[ExecutionPhase, ExecutionTransition], ExecutionPhase]
    ] = {
        (ExecutionPhase.IDLE, ExecutionTransition.START): ExecutionPhase.EXECUTING,
        (ExecutionPhase.EXECUTING, ExecutionTransition.FINISH): ExecutionPhase.SETTLING,
        (ExecutionPhase.SETTLING, ExecutionTransition.READY): ExecutionPhase.IDLE,
        (ExecutionPhase.IDLE, ExecutionTransition.SUSPEND): ExecutionPhase.SUSPENDED,
        (
            ExecutionPhase.EXECUTING,
            ExecutionTransition.SUSPEND,
        ): ExecutionPhase.SUSPENDED,
        (
            ExecutionPhase.SETTLING,
            ExecutionTransition.SUSPEND,
        ): ExecutionPhase.SUSPENDED,
        (ExecutionPhase.SUSPENDED, ExecutionTransition.RESUME): ExecutionPhase.IDLE,
    }

    def __init__(self) -> None:
        self.phase = ExecutionPhase.IDLE
        self.action: Action | None = None
        self.last_completed: str | None = None
        self.settle_until = 0

    def transition(self, event: ExecutionTransition) -> None:
        self.phase = self.transitions[(self.phase, event)]

    def start(self, action: Action) -> None:
        self.transition(ExecutionTransition.START)
        self.action = action

    def finish(self, now: float, pause: float, interrupted: bool = False) -> None:
        action = self.action
        assert action is not None
        if not interrupted:
            self.last_completed = action.name
        self.action = None
        self.transition(ExecutionTransition.FINISH)
        self.settle_until = now + pause

    def completion(
        self, packet: Mapping[str, object], now: float, pause: float
    ) -> bool:
        action = self.action
        if self.phase != ExecutionPhase.EXECUTING:
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
        if self.phase == ExecutionPhase.EXECUTING:
            action = self.action
            assert action is not None
            if now >= action.until:
                self.finish(now, pause)
                return True
        if self.phase == ExecutionPhase.SETTLING and now >= self.settle_until:
            self.transition(ExecutionTransition.READY)
        return False

    def interrupt(self) -> None:
        self.action = None
        self.phase = ExecutionPhase.IDLE

    def suspend(self) -> None:
        if self.phase != ExecutionPhase.SUSPENDED:
            self.action = None
            self.transition(ExecutionTransition.SUSPEND)

    def resume(self) -> None:
        if self.phase == ExecutionPhase.SUSPENDED:
            self.transition(ExecutionTransition.RESUME)
