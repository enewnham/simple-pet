from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Literal

from ..domain import Activity, ExecutionOutcome
from ..types import JSONObject
from .citations import Citations
from .observations import outcome_text
from .sources import SourceKind


class Event(StrEnum):
    WORK_UPDATE = "work_update"
    PERMISSION_REQUEST = "permission_request"
    INPUT_REQUESTED = "input_requested"
    WAITING = "waiting"
    FAILED_CALL = "failed_call"
    TURN_ENDED = "turn_ended"
    REQUEST_RECEIVED = "request_received"
    QUIET_TIME = "quiet_time"
    QUIET_CHECK_IN = "quiet_check_in"
    TOOL_RUNNING = "tool_running"
    PET_CLICKED = "pet_clicked"
    PET_CLICKED_IDLE = "pet_clicked_idle"
    RECOLLECTION = "recollection"
    EXPLICIT_CONTEXT = "explicit_context"


class WorkState(StrEnum):
    WORKING = "working"
    QUIET = "quiet"


class FactKind(StrEnum):
    REQUEST = "request"
    DECISION = "decision"
    PROGRESS = "reported_progress"


@dataclass(frozen=True)
class SupportedText:
    text: str
    sources: tuple[str, ...] = ()


@dataclass(frozen=True)
class WorkContext:
    request: SupportedText | None = None
    request_age: str | None = None
    story: SupportedText | None = None

    @property
    def sources(self) -> tuple[str, ...]:
        return (
            *(self.request.sources if self.request else ()),
            *(self.story.sources if self.story else ()),
        )


@dataclass(frozen=True)
class Moment:
    event: Event
    text: str | None = None
    progress: SupportedText | None = None
    state: WorkState | None = None
    outcome: ExecutionOutcome | None = None
    operation: Activity | None = None
    tool: str | None = None
    elapsed: str | None = None
    age: str | None = None
    sleeping: bool = False

    @property
    def sources(self) -> tuple[str, ...]:
        return self.progress.sources if self.progress else ()


@dataclass(frozen=True)
class ContextFact:
    kind: FactKind
    statement: SupportedText
    age: str | None = None
    scope: str | None = None
    outcome: ExecutionOutcome | None = None

    @property
    def sources(self) -> tuple[str, ...]:
        return self.statement.sources

    def payload(self, citations: Citations) -> JSONObject:
        result: JSONObject = {self.kind.value: self.statement.text}
        if self.sources:
            result["source_ids"] = tuple(citations.labels(self.sources))
        if self.age is not None:
            result["age"] = self.age
        if self.scope is not None:
            result["scope"] = self.scope
        if self.outcome is not None:
            result["execution"] = outcome_text(self.outcome)
        return result


@dataclass(frozen=True)
class Interpretation:
    category: Literal["intent", "decision", "progress"]
    text: str


@dataclass(frozen=True)
class MemoryCandidate:
    source_id: str
    kind: SourceKind
    text: str
    scope: str
    eligible_recent: bool
    eligible_ongoing: bool
    interpretations: tuple[Interpretation, ...] = ()
    age: str | None = None
    outcome: ExecutionOutcome | None = None

    @property
    def sources(self) -> tuple[str, ...]:
        return (self.source_id,)

    def payload(self, citations: Citations) -> JSONObject:
        candidate: JSONObject = {
            "id": citations.label(self.source_id),
            "kind": self.kind,
            "text": self.text,
            "scope": self.scope,
            "eligible_recent": self.eligible_recent,
            "eligible_ongoing": self.eligible_ongoing,
            "interpretations": [
                {"category": item.category, "text": item.text}
                for item in self.interpretations
            ],
        }
        if self.age is not None:
            candidate["age"] = self.age
        if self.outcome is not None:
            candidate["outcome"] = self.outcome
        return {"memory_candidate": candidate}


type ContextItem = ContextFact | MemoryCandidate


@dataclass(frozen=True)
class CommentaryContext:
    work: WorkContext | None = None
    moment: Moment | None = None
    earlier: tuple[ContextItem, ...] = ()

    @property
    def citations(self) -> Citations:
        return Citations.from_sources(
            (
                *(self.work.sources if self.work else ()),
                *(self.moment.sources if self.moment else ()),
                *(source for item in self.earlier for source in item.sources),
            )
        )

    def stable(self) -> CommentaryContext:
        return replace(
            self,
            work=replace(self.work, request_age=None) if self.work else None,
            moment=replace(self.moment, elapsed=None, age=None)
            if self.moment
            else None,
            earlier=(),
        )
