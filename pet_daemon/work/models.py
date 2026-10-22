from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

from ..domain import (
    ArtifactKind,
    EvidenceKind,
    EvidenceSource,
    ExecutionOutcome,
    LinkRelationship,
    PullRequestState,
)

if TYPE_CHECKING:
    from .observations import (
        BranchObservation,
        CheckoutObservation,
        GitObservation,
        PullRequest,
        PullRequestObservation,
    )


@dataclass(frozen=True)
class RelatedLink:
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


@dataclass(frozen=True)
class ArtifactReference:
    kind: ArtifactKind
    ref: str
    repository: str = ""
    revision: str | None = None
    checkout: str | None = None
    title: str | None = None


@dataclass(frozen=True)
class IntentEvidence:
    source: ClassVar[EvidenceSource] = EvidenceSource.USER
    kind: ClassVar[EvidenceKind] = EvidenceKind.INTENT

    session: str
    text: str


@dataclass(frozen=True)
class ProgressEvidence:
    source: ClassVar[EvidenceSource] = EvidenceSource.HARNESS
    kind: ClassVar[EvidenceKind] = EvidenceKind.PROGRESS

    session: str
    text: str


@dataclass(frozen=True)
class ExecutionEvidence:
    source: ClassVar[EvidenceSource] = EvidenceSource.HARNESS
    kind: ClassVar[EvidenceKind] = EvidenceKind.EXECUTION

    session: str
    event: str
    outcome: ExecutionOutcome | None
    tool_name: str | None
    progress: str | None


@dataclass(frozen=True)
class CheckoutAssociation:
    source: ClassVar[EvidenceSource] = EvidenceSource.HARNESS
    kind: ClassVar[EvidenceKind] = EvidenceKind.RELATIONSHIP

    checkout: str
    branch: str


@dataclass(frozen=True)
class PullRequestAssociation:
    source: ClassVar[EvidenceSource] = EvidenceSource.GITHUB
    kind: ClassVar[EvidenceKind] = EvidenceKind.RELATIONSHIP

    checkout: str
    branch: str


@dataclass(frozen=True)
class ReferenceAssociation:
    kind: ClassVar[EvidenceKind] = EvidenceKind.RELATIONSHIP

    ref: str
    source: EvidenceSource


@dataclass(frozen=True)
class StackAssociation:
    source: ClassVar[EvidenceSource] = EvidenceSource.GITHUB_STACK
    kind: ClassVar[EvidenceKind] = EvidenceKind.RELATIONSHIP

    link: RelatedLink
    from_ref: str


@dataclass(frozen=True)
class GitHubEvidence:
    source: ClassVar[EvidenceSource] = EvidenceSource.GITHUB
    kind: ClassVar[EvidenceKind] = EvidenceKind.OBSERVATION

    pr: PullRequest | None
    repository: str


@dataclass(frozen=True)
class AvailabilityEvidence:
    source: ClassVar[EvidenceSource] = EvidenceSource.OBSERVER
    kind: ClassVar[EvidenceKind] = EvidenceKind.AVAILABILITY

    local_error: str | None
    github_error: str | None


type EvidencePayload = (
    IntentEvidence
    | ProgressEvidence
    | ExecutionEvidence
    | CheckoutAssociation
    | PullRequestAssociation
    | StackAssociation
    | ReferenceAssociation
    | GitObservation
    | GitHubEvidence
    | AvailabilityEvidence
)


@dataclass(frozen=True)
class EvidenceRecord:
    payload: EvidencePayload
    observed_at: float | None = None
    context: ReportContext | None = None

    @property
    def source(self) -> EvidenceSource:
        return self.payload.source

    @property
    def kind(self) -> EvidenceKind:
        return self.payload.kind


@dataclass(frozen=True)
class ReportContext:
    checkout: str = ""
    repository: str = ""
    branch: str = ""
    pull_request: str = ""
    references: tuple[ArtifactReference, ...] = ()
    # Capture the next branch resolution when the report has no PR.
    branch_resolution: int = 0


@dataclass(frozen=True)
class SessionReport:
    payload: IntentEvidence | ProgressEvidence | ExecutionEvidence
    context: ReportContext
    observed_at: float

    def record(self) -> EvidenceRecord:
        return EvidenceRecord(self.payload, self.observed_at, self.context)


@dataclass(frozen=True)
class SessionRecord:
    id: str
    goal: str = ""
    parent: str | None = None
    location: str = ""
    location_at: float = 0
    reports: tuple[SessionReport, ...] = ()
    execution: SessionReport | None = None
    changed_at: float = 0
    aliases: tuple[str, ...] = ()


@dataclass
class WorkFacts:
    aliases: dict[str, str] = field(default_factory=dict)
    sessions: dict[str, SessionRecord] = field(default_factory=dict)
    checkouts: dict[str, CheckoutObservation] = field(default_factory=dict)
    branches: dict[tuple[str, str], BranchObservation] = field(default_factory=dict)
    pull_requests: dict[str, PullRequestObservation] = field(default_factory=dict)
