from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..identity import SessionIdentity
from ..types import JSONObject
from .evidence import CommentaryEvidence, ExecutionStatus, ReportedProgress, UserIntent
from .stories import WorkStory, revision_key


class NarrativeStatement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=400)
    sources: tuple[str, ...] = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def bounded_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Story statement requires bounded text")
        return value.strip()

    @field_validator("sources")
    @classmethod
    def distinct_sources(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(value))


class WorkNarrative(BaseModel):
    subject_id: SessionIdentity
    revision: str
    context_revision: str = ""
    intent: NarrativeStatement
    decisions: tuple[NarrativeStatement, ...] = ()
    progress: tuple[NarrativeStatement, ...] = ()
    sources: dict[str, JSONObject] = Field(default_factory=dict)


class SelectedMemory(BaseModel):
    key: str
    observed_at: float
    is_ongoing: bool = False


def semantic_revision(story: WorkStory, *, related: bool = True) -> str:
    def facts(
        records: tuple[CommentaryEvidence, ...],
    ) -> tuple[tuple[object, ...], ...]:
        result = []
        for record in records:
            match record.details:
                case UserIntent(text=text) | ReportedProgress(reported_progress=text):
                    result.append((record.kind, text))
                case ExecutionStatus(
                    execution="turn ended",
                    reported_progress=progress,
                    reported_execution_outcome=outcome,
                ):
                    result.append(("turn ended", progress, outcome))
        return tuple(result)

    return revision_key(
        (
            story.intent,
            facts(story.requests),
            facts(story.activity),
            tuple(
                (child.scope, facts(child.requests), facts(child.activity))
                for child in story.sessions
                if related
            ),
        )
    )
