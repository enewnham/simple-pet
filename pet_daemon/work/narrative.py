from dataclasses import dataclass, field

from ..types import JSONObject
from .views import (
    EvidenceView,
    ExecutionView,
    IntentView,
    ProgressView,
    WorkStory,
    revision_key,
)


@dataclass(frozen=True)
class NarrativeStatement:
    text: str
    sources: tuple[str, ...]


@dataclass(frozen=True)
class WorkNarrative:
    subject_id: str
    revision: str
    intent: NarrativeStatement
    decisions: tuple[NarrativeStatement, ...] = ()
    progress: tuple[NarrativeStatement, ...] = ()
    sources: dict[str, JSONObject] = field(default_factory=dict)


def semantic_revision(story: WorkStory) -> str:
    def facts(records: tuple[EvidenceView, ...]) -> tuple[tuple[object, ...], ...]:
        result = []
        for record in records:
            match record.details:
                case IntentView(text) | ProgressView(text):
                    result.append((record.kind, text))
                case ExecutionView("turn ended", progress, outcome):
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
            ),
        )
    )
