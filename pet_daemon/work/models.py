from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import BaseModel, Field

from ..domain import (
    ArtifactKind,
    EvidenceKind,
    EvidenceSource,
    ExecutionOutcome,
    LinkRelationship,
    PullRequestState,
)
from ..identity import SessionIdentity
from ..repository_identity import RepositoryIdentity

if TYPE_CHECKING:
    from .observations import (
        BranchObservation,
        CheckoutObservation,
        GitObservation,
        PullRequest,
        PullRequestObservation,
    )


class RelatedLink(BaseModel):
    kind: ArtifactKind
    ref: str
    relationship: LinkRelationship
    source: EvidenceSource
    stack: int | None = None
    position: int | None = None
    state: PullRequestState | None = None
    is_draft: bool | None = None
    head: str = ""
    head_branch: str = ""
    title: str = ""


class ArtifactReference(BaseModel):
    kind: ArtifactKind
    ref: str
    repository: RepositoryIdentity | None = None
    revision: str | None = None
    checkout: str | None = None
    title: str | None = None


class IntentEvidence(BaseModel):
    type: Literal["IntentEvidence"] = "IntentEvidence"
    source: Literal[EvidenceSource.USER] = EvidenceSource.USER
    kind: Literal[EvidenceKind.INTENT] = EvidenceKind.INTENT

    session: SessionIdentity
    text: str


class ProgressEvidence(BaseModel):
    type: Literal["ProgressEvidence"] = "ProgressEvidence"
    source: Literal[EvidenceSource.SOURCE] = EvidenceSource.SOURCE
    kind: Literal[EvidenceKind.PROGRESS] = EvidenceKind.PROGRESS

    session: SessionIdentity
    text: str


class ExecutionEvidence(BaseModel):
    type: Literal["ExecutionEvidence"] = "ExecutionEvidence"
    source: Literal[EvidenceSource.SOURCE] = EvidenceSource.SOURCE
    kind: Literal[EvidenceKind.EXECUTION] = EvidenceKind.EXECUTION

    session: SessionIdentity
    event: str
    outcome: ExecutionOutcome | None
    tool_name: str | None
    progress: str | None


class CheckoutAssociation(BaseModel):
    type: Literal["CheckoutAssociation"] = "CheckoutAssociation"
    source: Literal[EvidenceSource.SOURCE] = EvidenceSource.SOURCE
    kind: Literal[EvidenceKind.RELATIONSHIP] = EvidenceKind.RELATIONSHIP

    checkout: str
    branch: str


class PullRequestAssociation(BaseModel):
    type: Literal["PullRequestAssociation"] = "PullRequestAssociation"
    source: Literal[EvidenceSource.GITHUB] = EvidenceSource.GITHUB
    kind: Literal[EvidenceKind.RELATIONSHIP] = EvidenceKind.RELATIONSHIP

    checkout: str
    branch: str


class ReferenceAssociation(BaseModel):
    type: Literal["ReferenceAssociation"] = "ReferenceAssociation"
    kind: Literal[EvidenceKind.RELATIONSHIP] = EvidenceKind.RELATIONSHIP

    ref: str
    source: EvidenceSource


class StackAssociation(BaseModel):
    type: Literal["StackAssociation"] = "StackAssociation"
    source: Literal[EvidenceSource.GITHUB_STACK] = EvidenceSource.GITHUB_STACK
    kind: Literal[EvidenceKind.RELATIONSHIP] = EvidenceKind.RELATIONSHIP

    link: RelatedLink
    from_ref: str


class GitHubEvidence(BaseModel):
    type: Literal["GitHubEvidence"] = "GitHubEvidence"
    source: Literal[EvidenceSource.GITHUB] = EvidenceSource.GITHUB
    kind: Literal[EvidenceKind.OBSERVATION] = EvidenceKind.OBSERVATION

    pr: PullRequest | None
    repository: RepositoryIdentity | None


class AvailabilityEvidence(BaseModel):
    type: Literal["AvailabilityEvidence"] = "AvailabilityEvidence"
    source: Literal[EvidenceSource.OBSERVER] = EvidenceSource.OBSERVER
    kind: Literal[EvidenceKind.AVAILABILITY] = EvidenceKind.AVAILABILITY

    local_error: str | None
    github_error: str | None


type ReportPayload = Annotated[
    IntentEvidence | ProgressEvidence | ExecutionEvidence, Field(discriminator="type")
]


type EvidencePayload = Annotated[
    IntentEvidence
    | ProgressEvidence
    | ExecutionEvidence
    | CheckoutAssociation
    | PullRequestAssociation
    | StackAssociation
    | ReferenceAssociation
    | "GitObservation"
    | GitHubEvidence
    | AvailabilityEvidence,
    Field(discriminator="type"),
]


class EvidenceRecord(BaseModel):
    payload: EvidencePayload
    observed_at: float | None = None
    context: ReportContext | None = None

    @property
    def source(self) -> EvidenceSource:
        return self.payload.source

    @property
    def kind(self) -> EvidenceKind:
        return self.payload.kind


class ReportContext(BaseModel):
    checkout: str = ""
    repository: RepositoryIdentity | None = None
    branch: str = ""
    pull_request: str = ""
    references: tuple[ArtifactReference, ...] = ()
    # Capture the next branch resolution when the report has no PR.
    branch_resolution: int = 0


class SessionReport(BaseModel):
    payload: ReportPayload
    context: ReportContext
    observed_at: float

    def record(self) -> EvidenceRecord:
        return EvidenceRecord(
            payload=self.payload, observed_at=self.observed_at, context=self.context
        )


class SessionRecord(BaseModel):
    id: SessionIdentity
    goal: str = ""
    parent: SessionIdentity | None = None
    location: str = ""
    location_at: float = 0
    reports: tuple[SessionReport, ...] = ()
    execution: SessionReport | None = None
    changed_at: float = 0
    aliases: tuple[SessionIdentity, ...] = ()
    closure: SessionReport | None = None


@dataclass
class WorkFacts:
    aliases: dict[SessionIdentity, SessionIdentity] = field(default_factory=dict)
    sessions: dict[SessionIdentity, SessionRecord] = field(default_factory=dict)
    checkouts: dict[str, CheckoutObservation] = field(default_factory=dict)
    branches: dict[tuple[RepositoryIdentity, str], BranchObservation] = field(
        default_factory=dict
    )
    pull_requests: dict[str, PullRequestObservation] = field(default_factory=dict)
