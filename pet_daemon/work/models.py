from dataclasses import dataclass, field
from typing import Any


@dataclass
class WorkItem:
    id: str
    intent: str
    status: str
    changed_at: float


@dataclass
class Artifact:
    id: str
    data: dict[str, Any]
    version: int = 1


@dataclass
class Evidence:
    id: int
    work_id: str
    artifact_id: str
    source: str
    kind: str
    data: dict[str, Any]
    timestamp: float

    def record(self):
        return {"source": self.source, "kind": self.kind, "data": self.data, "observed_at": self.timestamp}


@dataclass
class OpportunityState:
    id: str
    work_id: str
    kind: str
    artifact_id: str
    version: int
    fingerprint: str
    data: dict[str, Any]
    active: bool
    surfaced_version: int
    changed_at: float


@dataclass(frozen=True)
class Opportunity:
    id: str
    version: int
    kind: str
    work_id: str
    intent: str
    artifact: dict
    evidence: tuple
    uncertainty: tuple
    next_step: str
    changed_at: float
    sessions: tuple


@dataclass
class Surfacing:
    id: int
    opportunity_id: str
    version: int
    outcome: str
    timestamp: float


@dataclass
class WorkState:
    next_evidence_id: int = 1
    next_surfacing_id: int = 1
    items: dict[str, WorkItem] = field(default_factory=dict)
    sessions: dict[str, str] = field(default_factory=dict)
    memberships: set[tuple[str, str]] = field(default_factory=set)
    aliases: dict[str, str] = field(default_factory=dict)
    artifacts: dict[str, Artifact] = field(default_factory=dict)
    links: set[tuple[str, str]] = field(default_factory=set)
    evidence: list[Evidence] = field(default_factory=list)
    opportunities: dict[str, OpportunityState] = field(default_factory=dict)
    surfacings: list[Surfacing] = field(default_factory=list)

    def current(self, session):
        return self.items.get(self.sessions.get(session))

    def latest(self, work_id, artifact_id):
        records = {}
        for record in self.evidence:
            if record.work_id == work_id and record.artifact_id == artifact_id:
                records[record.source, record.kind] = record
        return records

    def linked_work(self, artifact_id):
        return {work_id for work_id, linked_id in self.links if linked_id == artifact_id}
