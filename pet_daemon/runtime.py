from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, NotRequired, Protocol, TypedDict

type ToolActivity = Literal[
    "investigating", "editing", "testing", "building", "checking", "command"
]
type SessionActivity = ToolActivity | Literal["harness"]


class ProcessPacket(TypedDict):
    pid: int
    started: str


class HookTextFields(TypedDict, total=False):
    parent: str
    parent_call: str
    call: str
    turn: str
    tool: str
    goal: str
    progress: str
    repo: str
    repository: str
    branch: str
    pull_request: str
    tool_name: str
    sentiment_text: str
    outcome: str


class HookPacket(HookTextFields):
    harness: str
    session: str
    event: str
    kind: NotRequired[str]
    aliases: NotRequired[list[str]]
    manual: NotRequired[bool]
    owner: NotRequired[ProcessPacket]
    exit_code: NotRequired[int]


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
    state: str


@dataclass(frozen=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True)
class SessionTrigger:
    session_id: str


@dataclass(frozen=True)
class SubjectTrigger:
    subject_id: str


@dataclass(frozen=True)
class ReactionTrigger:
    reaction: SessionReaction


@dataclass(frozen=True)
class CareTrigger:
    care: str


type Trigger = SessionTrigger | SubjectTrigger | ReactionTrigger | CareTrigger


class RandomSource(Protocol):
    def random(self) -> float: ...
    def uniform(self, a: float, b: float) -> float: ...
    def choice[T](self, seq: Sequence[T]) -> T: ...


@dataclass(frozen=True)
class UserSentiment:
    compound: float
    seconds_ago: float


@dataclass(frozen=True)
class ConvergenceItem:
    id: str
    outcome: str | None


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    started: str

    @classmethod
    def parse(cls, value: object) -> ProcessIdentity | None:
        if (
            isinstance(value, dict)
            and type(value.get("pid")) is int
            and value["pid"] > 1
            and isinstance(value.get("started"), str)
            and 0 < len(value["started"]) <= 128
        ):
            return cls(value["pid"], value["started"])
        return None

    def packet(self) -> ProcessPacket:
        return {"pid": self.pid, "started": self.started}


@dataclass(frozen=True)
class HookEvent:
    harness: str
    session: str
    event: str
    aliases: tuple[str, ...] = ()
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
    outcome: str | None = None
    manual: bool | None = None
    owner: ProcessIdentity | None = None
    exit_code: int | None = None
    repository: str | None = None
    branch: str | None = None
    pull_request: str | None = None

    @property
    def source(self) -> str:
        return f"hook:{self.harness}"

    @property
    def subject(self) -> str:
        return f"{self.harness}:{self.session}"

    def packet(self) -> HookPacket:
        packet: HookPacket = {
            "kind": "event",
            "harness": self.harness,
            "session": self.session,
            "event": self.event,
        }
        if self.parent is not None:
            packet["parent"] = self.parent
        if self.parent_call is not None:
            packet["parent_call"] = self.parent_call
        if self.call is not None:
            packet["call"] = self.call
        if self.turn is not None:
            packet["turn"] = self.turn
        if self.tool is not None:
            packet["tool"] = self.tool
        if self.goal is not None:
            packet["goal"] = self.goal
        if self.progress is not None:
            packet["progress"] = self.progress
        if self.repo is not None:
            packet["repo"] = self.repo
        if self.repository is not None:
            packet["repository"] = self.repository
        if self.branch is not None:
            packet["branch"] = self.branch
        if self.pull_request is not None:
            packet["pull_request"] = self.pull_request
        if self.tool_name is not None:
            packet["tool_name"] = self.tool_name
        if self.sentiment_text is not None:
            packet["sentiment_text"] = self.sentiment_text
        if self.outcome is not None:
            packet["outcome"] = self.outcome
        if self.manual is not None:
            packet["manual"] = self.manual
        if self.exit_code is not None:
            packet["exit_code"] = self.exit_code
        if self.aliases:
            packet["aliases"] = list(self.aliases)
        if self.owner is not None:
            packet["owner"] = self.owner.packet()
        return packet


if TYPE_CHECKING:
    from .sessions.reactions import SessionReaction
    from .work.views import WorkStory, WorkSubject


@dataclass(frozen=True)
class ActiveTool:
    call: str
    tool: str
    activity: ToolActivity
    duration_seconds: float


@dataclass(frozen=True)
class FailureSnapshot:
    activity: SessionActivity
    count: int
    goal: str
    resolved: bool
    current_turn: bool
    seconds_ago: float


@dataclass(frozen=True)
class SessionContext:
    id: str
    changed_at: float
    parent: str | None
    goal: str
    repo: str
    phase: ToolActivity | Literal[""]
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
    presence: str
    owner: ProcessIdentity | None
    owner_last_alive_seconds: float | None
    activity_uncertain: bool
    waiting_for_input: bool
    waiting_seconds: float | None
    active_tools: tuple[ActiveTool, ...]
    progress: str
    failure: FailureSnapshot | None
    outcome: str | None
    user_sentiment: UserSentiment | None = None
    connected_work: WorkStory | None = None

    @property
    def input_wait_seconds(self) -> float:
        assert self.waiting_seconds is not None
        return self.waiting_seconds


@dataclass(frozen=True)
class SwarmActivity:
    activity: str
    sessions: tuple[SessionContext, ...]
    convergence: tuple[ConvergenceItem, ...] | None = None


@dataclass(frozen=True)
class PetContext:
    activity: str
    sessions: tuple[SessionContext, ...]
    sleeping: bool
    trigger: Trigger | None = None
    work: tuple[WorkSubject, ...] = ()
    convergence: tuple[ConvergenceItem, ...] | None = None
