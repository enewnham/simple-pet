from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .observations import GitObservation, PullRequest


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
class ArtifactDescriptor:
    kind: str
    ref: str
    repository: str = ""
    revision: str | None = None
    checkout: str | None = None
    title: str | None = None


@dataclass(frozen=True, kw_only=True)
class Artifact(ArtifactDescriptor):
    id: str
    version: int = 1

    def descriptor(self) -> ArtifactDescriptor:
        return ArtifactDescriptor(
            self.kind,
            self.ref,
            self.repository,
            self.revision,
            self.checkout,
            self.title,
        )


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


@dataclass(frozen=True)
class ArtifactAccount:
    artifact: ArtifactDescriptor
    evidence: tuple[EvidenceRecord, ...]


@dataclass(frozen=True)
class WorkAccount:
    intent: str
    revision: str
    requests: tuple[EvidenceRecord, ...]
    activity: tuple[EvidenceRecord, ...]
    artifacts: tuple[ArtifactAccount, ...]


@dataclass
class WorkItem:
    id: str
    intent: str
    status: str
    changed_at: float


@dataclass
class Evidence:
    id: int
    work_id: str
    artifact_id: str
    source: str
    kind: str
    payload: EvidencePayload
    timestamp: float

    def record(self) -> EvidenceRecord:
        return EvidenceRecord(self.source, self.kind, self.payload, self.timestamp)


@dataclass
class WorkState:
    next_evidence_id: int = 1
    items: dict[str, WorkItem] = field(default_factory=dict)
    sessions: dict[str, str] = field(default_factory=dict)
    memberships: set[tuple[str, str]] = field(default_factory=set)
    aliases: dict[str, str] = field(default_factory=dict)
    artifacts: dict[str, Artifact] = field(default_factory=dict)
    links: set[tuple[str, str]] = field(default_factory=set)
    evidence: list[Evidence] = field(default_factory=list)

    def current(self, session: str) -> WorkItem | None:
        identifier = self.sessions.get(session)
        return self.items.get(identifier) if identifier is not None else None

    def latest(self, work_id: str, artifact_id: str) -> dict[tuple[str, str], Evidence]:
        records = {}
        for record in self.evidence:
            if record.work_id == work_id and record.artifact_id == artifact_id:
                records[record.source, record.kind] = record
        return records

    def linked_work(self, artifact_id: str) -> set[str]:
        return {
            work_id for work_id, linked_id in self.links if linked_id == artifact_id
        }
