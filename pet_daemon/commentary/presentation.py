from __future__ import annotations

from typing import NotRequired, TypedDict

from ..domain import ArtifactKind, EvidenceKind, EvidenceSource, ExecutionOutcome
from ..identity import SessionIdentity
from ..types import JSONObject
from ..work.models import ReportContext
from ..work.views import EvidenceView, ExecutionView, WorkStory


class ReferenceRecord(TypedDict):
    kind: ArtifactKind
    ref: str


class ReportContextRecord(TypedDict, total=False):
    branch: str
    pull_request: str
    references: tuple[ReferenceRecord, ...]
    uncertainty: tuple[str, ...]


class EvidenceFields(TypedDict, total=False):
    repository: str | None
    branch: str
    output_present: bool
    conflicts: bool
    behind: bool
    output: JSONObject
    pull_request_found: bool
    execution: str
    reported_progress: str
    reported_execution_outcome: ExecutionOutcome
    uncertainty: tuple[str, ...]
    text: str
    association: str
    authorship: str
    context: ReportContextRecord


class PresentedEvidence(EvidenceFields):
    source: EvidenceSource
    kind: EvidenceKind
    observed_at: NotRequired[float]


class PresentedArtifact(TypedDict):
    kind: ArtifactKind
    ref: str
    title: NotRequired[str]
    evidence: tuple[PresentedEvidence, ...]


class WorkRecord(TypedDict):
    intent: str
    requests: tuple[PresentedEvidence, ...]
    activity: tuple[PresentedEvidence, ...]
    artifacts: tuple[PresentedArtifact, ...]
    sessions: NotRequired[tuple[AgentSessionRecord, ...]]


class AgentSessionRecord(TypedDict):
    session: SessionIdentity
    scope: str
    requests: tuple[PresentedEvidence, ...]
    activity: tuple[PresentedEvidence, ...]
    context: ReportContextRecord


def context_record(context: ReportContext) -> ReportContextRecord:
    result: ReportContextRecord = {}
    if context.branch:
        result["branch"] = context.branch
    if context.pull_request:
        result["pull_request"] = context.pull_request
    if context.references:
        result["references"] = tuple(
            {"kind": item.kind, "ref": item.ref} for item in context.references
        )
    if context.checkout and not (context.repository and context.branch):
        result["uncertainty"] = ("The report's checkout context was not resolved.",)
    return result


def evidence_record(record: EvidenceView) -> PresentedEvidence:
    details = record.details
    result: PresentedEvidence = {
        "source": record.source,
        "kind": record.kind,
        **details.model_dump(
            exclude={"type"}, exclude_none=isinstance(details, ExecutionView)
        ),
    }
    if record.observed_at is not None:
        result["observed_at"] = record.observed_at
    if record.context is not None:
        result["context"] = context_record(record.context)
    return result


def work_record(work: WorkStory) -> WorkRecord:
    result: WorkRecord = {
        "intent": work.intent,
        "requests": tuple(evidence_record(record) for record in work.requests),
        "activity": tuple(evidence_record(record) for record in work.activity),
        "artifacts": tuple(
            {
                "kind": item.kind,
                "ref": item.ref,
                **({"title": item.title} if item.title else {}),
                "evidence": tuple(evidence_record(record) for record in item.evidence),
            }
            for item in work.artifacts
        ),
    }
    if work.sessions:
        result["sessions"] = tuple(
            {
                "session": agent.session,
                "scope": agent.scope,
                "requests": tuple(evidence_record(record) for record in agent.requests),
                "activity": tuple(evidence_record(record) for record in agent.activity),
                "context": context_record(agent.context) if agent.context else {},
            }
            for agent in work.sessions
        )
    return result
