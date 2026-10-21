from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

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
    kind: str
    ref: str
    relationship: str
    source: str
    stack: int | None = None
    position: int | None = None
    state: str | None = None
    is_draft: bool | None = None
    head: str = ""
    head_branch: str = ""
    title: str = ""


@dataclass(frozen=True)
class ArtifactReference:
    kind: str
    ref: str
    repository: str = ""
    revision: str | None = None
    checkout: str | None = None
    title: str | None = None


@dataclass(frozen=True)
class IntentEvidence:
    session: str
    text: str


@dataclass(frozen=True)
class ProgressEvidence:
    session: str
    text: str


@dataclass(frozen=True)
class ExecutionEvidence:
    session: str
    event: str
    outcome: str | None
    tool_name: str | None
    progress: str | None


@dataclass(frozen=True)
class CheckoutAssociation:
    checkout: str
    branch: str


@dataclass(frozen=True)
class PullRequestAssociation:
    checkout: str
    branch: str


@dataclass(frozen=True)
class ReferenceAssociation:
    ref: str


@dataclass(frozen=True)
class StackAssociation:
    link: RelatedLink
    from_ref: str


@dataclass(frozen=True)
class GitHubEvidence:
    pr: PullRequest | None
    repository: str


@dataclass(frozen=True)
class AvailabilityEvidence:
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
    source: str
    kind: str
    payload: EvidencePayload
    observed_at: float | None = None
    context: ReportContext | None = None


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
        source = "user" if isinstance(self.payload, IntentEvidence) else "harness"
        kind = (
            "intent"
            if isinstance(self.payload, IntentEvidence)
            else "progress"
            if isinstance(self.payload, ProgressEvidence)
            else "execution"
        )
        return EvidenceRecord(
            source, kind, self.payload, self.observed_at, self.context
        )


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
