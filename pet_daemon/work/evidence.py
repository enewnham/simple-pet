from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

from ..domain import (
    EvidenceKind,
    EvidenceSource,
    ExecutionOutcome,
    PullRequestState,
)
from ..repository_identity import (
    RemoteRepository,
)
from .models import (
    AvailabilityEvidence,
    CheckoutAssociation,
    EvidenceRecord,
    ExecutionEvidence,
    GitHubEvidence,
    IntentEvidence,
    ProgressEvidence,
    PullRequestAssociation,
    ReferenceAssociation,
    ReportContext,
    StackAssociation,
)
from .observations import GitObservation


class PullRequestStatus(BaseModel):
    type: Literal["PullRequestStatus"] = "PullRequestStatus"
    url: str
    title: str
    state: PullRequestState
    is_draft: bool | None


class BranchCondition(BaseModel):
    type: Literal["BranchCondition"] = "BranchCondition"
    repository: str | None
    branch: str
    output_present: bool
    conflicts: bool
    behind: bool


class PullRequestNotFound(BaseModel):
    type: Literal["PullRequestNotFound"] = "PullRequestNotFound"
    pull_request_found: bool = False


class ExecutionStatus(BaseModel):
    type: Literal["ExecutionStatus"] = "ExecutionStatus"
    execution: str
    reported_progress: str | None
    reported_execution_outcome: ExecutionOutcome | None


class ObservationGaps(BaseModel):
    type: Literal["ObservationGaps"] = "ObservationGaps"
    uncertainty: tuple[str, ...]


class UserIntent(BaseModel):
    type: Literal["UserIntent"] = "UserIntent"
    text: str


class ReportedProgress(BaseModel):
    type: Literal["ReportedProgress"] = "ReportedProgress"
    reported_progress: str


class WorkAssociation(BaseModel):
    type: Literal["WorkAssociation"] = "WorkAssociation"
    association: str
    authorship: str = "Association does not establish authorship."


type EvidenceDetails = Annotated[
    (
        BranchCondition
        | PullRequestStatus
        | PullRequestNotFound
        | ExecutionStatus
        | ObservationGaps
        | UserIntent
        | ReportedProgress
        | WorkAssociation
    ),
    Field(discriminator="type"),
]


class CommentaryEvidence(BaseModel):
    source: EvidenceSource
    kind: EvidenceKind
    details: EvidenceDetails
    observed_at: float | None = None
    context: ReportContext | None = None


def commentary_evidence(record: EvidenceRecord) -> CommentaryEvidence | None:
    source, kind, payload = record.source, record.kind, record.payload
    details: EvidenceDetails
    match payload:
        case GitObservation():
            details = BranchCondition(
                repository=payload.repository.value
                if isinstance(payload.repository, RemoteRepository)
                else None,
                branch=payload.branch,
                output_present=bool(
                    payload.staged
                    or payload.unstaged
                    or payload.untracked
                    or payload.ahead
                ),
                conflicts=bool(payload.conflicts),
                behind=bool(payload.behind),
            )
        case GitHubEvidence():
            pr = payload.pr
            if pr:
                details = PullRequestStatus(
                    url=pr.url, title=pr.title, state=pr.state, is_draft=pr.is_draft
                )
            else:
                details = PullRequestNotFound()
        case ExecutionEvidence() if not payload.event.startswith("tool_"):
            ended = payload.event in {
                "turn_done",
                "turn_error",
                "subagent_done",
                "interrupt",
                "session_end",
            }
            execution = (
                "waiting"
                if payload.event == "waiting"
                else "turn ended"
                if ended
                else "in progress"
            )
            details = ExecutionStatus(
                execution=execution,
                reported_progress=" ".join(payload.progress.split())[:800]
                if payload.progress
                else None,
                reported_execution_outcome=payload.outcome
                if ended and payload.outcome
                else None,
            )
        case AvailabilityEvidence():
            unavailable = tuple(
                name
                for name, error in (
                    ("local", payload.local_error),
                    ("github", payload.github_error),
                )
                if error
            )
            if not unavailable:
                return None
            details = ObservationGaps(
                uncertainty=tuple(
                    f"{name} observation unavailable" for name in unavailable
                )
            )
        case IntentEvidence():
            details = UserIntent(text=" ".join(payload.text.split())[:1000])
        case ProgressEvidence():
            details = ReportedProgress(
                reported_progress=" ".join(payload.text.split())[:800]
            )
        case CheckoutAssociation():
            details = WorkAssociation(
                association="The agent session used this checkout."
            )
        case PullRequestAssociation():
            details = WorkAssociation(
                association="The pull request matches the observed branch."
            )
        case ReferenceAssociation():
            details = WorkAssociation(
                association="The reported text explicitly references this URL."
            )
        case StackAssociation():
            details = WorkAssociation(
                association="The pull requests share an observed stack."
            )
        case _:
            return None
    return CommentaryEvidence(
        source=source,
        kind=kind,
        details=details,
        observed_at=record.observed_at,
        context=record.context,
    )
