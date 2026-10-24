from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, Literal, Protocol, TypedDict, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .domain import (
    Activity,
    CareKind,
    ExecutionOutcome,
    PetState,
    ProcessStatus,
    SessionSource,
)
from .identity import SessionIdentity


class HookEventKind(StrEnum):
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    INTENT = "intent"
    TOOL_START = "tool_start"
    TOOL_DONE = "tool_done"
    TOOL_ERROR = "tool_error"
    TURN_DONE = "turn_done"
    TURN_ERROR = "turn_error"
    INTERRUPT = "interrupt"
    SUBAGENT_START = "subagent_start"
    SUBAGENT_DONE = "subagent_done"
    PROGRESS = "progress"
    WAITING = "waiting"


class WaitingKind(StrEnum):
    PERMISSION_CHECK = "permission_check"
    PERMISSION = "permission"
    QUESTION = "question"


type ProcessPacket = dict[str, object]
type HookPacket = dict[str, object]


class AnimationPacket(TypedDict):
    id: str
    repeats: int | None
    command_id: int


class MotionPacket(TypedDict):
    x: float
    y: float
    seconds: float
    command_id: int


class OverlayPacket(TypedDict, total=False):
    text: str
    ttl: float
    delivery_id: str
    animation: AnimationPacket
    motion: MotionPacket
    stop_motion: bool
    sleeping: bool
    state: PetState


class Point(BaseModel):
    x: float = Field(strict=True, allow_inf_nan=False)
    y: float = Field(strict=True, allow_inf_nan=False)


class SessionTrigger(BaseModel):
    kind: Literal["session"] = "session"
    session_id: SessionIdentity


class SubjectTrigger(BaseModel):
    kind: Literal["subject"] = "subject"
    subject_id: SessionIdentity


class ReactionKind(StrEnum):
    START = "start"
    WAITING = "waiting"
    INVESTIGATION = "investigation"
    TROUBLE = "trouble"
    RECOVERY = "recovery"
    FAILURE = "failure"
    CANCELLED = "cancelled"
    SUCCESS = "success"


class SessionReaction(BaseModel):
    kind: ReactionKind
    session: SessionIdentity
    generation: int
    activity: Activity
    serial: int


class ReactionTrigger(BaseModel):
    kind: Literal["reaction"] = "reaction"
    model_config = ConfigDict(defer_build=True)
    reaction: SessionReaction


class CareTrigger(BaseModel):
    kind: Literal["care"] = "care"
    care: CareKind


type Trigger = Annotated[
    SessionTrigger | SubjectTrigger | ReactionTrigger | CareTrigger,
    Field(discriminator="kind"),
]


class RandomSource(Protocol):
    def random(self) -> float: ...
    def uniform(self, a: float, b: float) -> float: ...
    def choice[T](self, seq: Sequence[T]) -> T: ...


class UserSentiment(BaseModel):
    compound: float
    seconds_ago: float


class ConvergenceItem(BaseModel):
    id: SessionIdentity
    outcome: ExecutionOutcome | None


class ProcessIdentity(BaseModel):
    model_config = ConfigDict(frozen=True)
    pid: int = Field(strict=True, gt=0)
    started: str = Field(min_length=1, max_length=128)

    @classmethod
    def parse(cls, value: object) -> ProcessIdentity | None:
        try:
            identity = cls.model_validate(value)
            return identity if identity.pid > 1 else None
        except ValidationError:
            return None

    def packet(self) -> ProcessPacket:
        return cast(ProcessPacket, self.model_dump(mode="json"))


class HookEvent(BaseModel):
    kind: Literal["event"] = "event"
    source: SessionSource
    session: str = Field(min_length=1)
    event: HookEventKind
    aliases: tuple[Annotated[str, Field(min_length=1)], ...] = ()
    parent: str | None = None
    parent_call: str | None = None
    call: str | None = None
    turn: str | None = None
    tool: str | None = None
    goal: str | None = None
    progress: str | None = None
    repo: str | None = None
    tool_name: str | None = None
    sentiment_text: str | None = None
    outcome: ExecutionOutcome | None = None
    manual: bool | None = None
    owner: ProcessIdentity | None = None
    exit_code: int | None = None
    repository: str | None = None
    branch: str | None = None
    pull_request: str | None = None
    waiting_kind: WaitingKind | None = None

    @property
    def subject(self) -> SessionIdentity:
        return SessionIdentity(source=self.source, native_id=self.session)

    def packet(self) -> HookPacket:
        packet = self.model_dump(mode="json", exclude_none=True)
        if not self.aliases:
            packet.pop("aliases")
        return cast(HookPacket, packet)


if TYPE_CHECKING:
    from .work.shared_work import WorkSubject
    from .work.stories import WorkStory


class ActiveTool(BaseModel):
    call: str
    tool: str
    activity: Activity
    duration_seconds: float


class FailureSnapshot(BaseModel):
    activity: Activity
    count: int
    goal: str
    resolved: bool
    current_turn: bool
    seconds_ago: float


class AgentSessionStatus(BaseModel):
    model_config = ConfigDict(defer_build=True)
    id: SessionIdentity
    changed_at: float
    parent: SessionIdentity | None
    goal: str
    repo: str
    phase: Activity | None
    initial_request: str
    summary: str
    recent_requests: tuple[str, ...]
    generation: int
    request_count: int
    first_summary_attempted: bool
    duration_seconds: float
    quiet_seconds: float
    turn_active: bool
    last_tool: str
    presence: ProcessStatus
    owner: ProcessIdentity | None
    owner_last_alive_seconds: float | None
    activity_uncertain: bool
    waiting_for_input: bool
    waiting_seconds: float | None
    active_tools: tuple[ActiveTool, ...]
    progress: str
    failure: FailureSnapshot | None
    outcome: ExecutionOutcome | None
    last_hook_at: float = 0
    retired: bool = False
    user_sentiment: UserSentiment | None = None
    connected_work: WorkStory | None = None
    waiting_kind: WaitingKind | None = None
    waiting_tool: str | None = None
    waiting_prompt: str | None = None

    @property
    def input_wait_seconds(self) -> float:
        assert self.waiting_seconds is not None
        return self.waiting_seconds


class SwarmActivity(BaseModel):
    model_config = ConfigDict(defer_build=True)
    activity: PetState
    sessions: tuple[AgentSessionStatus, ...]
    convergence: tuple[ConvergenceItem, ...] | None = None


class PetContext(BaseModel):
    model_config = ConfigDict(defer_build=True)
    activity: PetState
    sessions: tuple[AgentSessionStatus, ...]
    sleeping: bool
    trigger: Trigger | None = None
    work: tuple[WorkSubject, ...] = ()
    convergence: tuple[ConvergenceItem, ...] | None = None
