from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from dataclasses import dataclass


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    started: str

    @classmethod
    def parse(cls, value):
        if (
            isinstance(value, dict)
            and type(value.get("pid")) is int
            and value["pid"] > 1
            and isinstance(value.get("started"), str)
            and 0 < len(value["started"]) <= 128
        ):
            return cls(value["pid"], value["started"])
        return None

    def packet(self):
        return {"pid": self.pid, "started": self.started}


if TYPE_CHECKING:
    from .work.views import WorkSubject, WorkView


class Record(Mapping):
    def __getitem__(self, name):
        if name not in self.__dataclass_fields__:
            raise KeyError(name)
        return getattr(self, name)

    def __iter__(self):
        return iter(self.__dataclass_fields__)

    def __len__(self):
        return len(self.__dataclass_fields__)


@dataclass(frozen=True)
class ToolSnapshot(Record):
    call: str
    tool: str
    activity: str
    duration_seconds: float


@dataclass(frozen=True)
class FailureSnapshot(Record):
    activity: str
    count: int
    goal: str
    resolved: bool
    current_turn: bool
    seconds_ago: float


@dataclass(frozen=True)
class SessionSnapshot(Record):
    id: str
    changed_at: float
    parent: str | None
    goal: str
    repo: str
    phase: str
    first_prompt: str
    session_context: str
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
    active_tools: tuple[ToolSnapshot, ...]
    progress: str
    failure: FailureSnapshot | None
    outcome: str | None
    user_sentiment: dict | None = None
    connected_work: WorkView | None = None

    @classmethod
    def capture(cls, values, connected_work):
        return cls(
            **{
                **values,
                "recent_requests": tuple(values["recent_requests"]),
                "active_tools": tuple(
                    ToolSnapshot(**tool) for tool in values["active_tools"]
                ),
                "failure": FailureSnapshot(**values["failure"])
                if values["failure"]
                else None,
                "owner": ProcessIdentity.parse(values["owner"]),
                "connected_work": connected_work,
            }
        )


@dataclass(frozen=True)
class WorkSnapshot(Record):
    activity: str
    sessions: tuple[SessionSnapshot, ...]
    sleeping: bool
    trigger: dict | None = None
    work: tuple[WorkSubject, ...] = ()
    convergence: object = None
