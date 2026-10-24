from __future__ import annotations

from typing import NotRequired, TypedDict

from ..domain import (
    ArtifactKind,
    EvidenceKind,
    EvidenceSource,
    ExecutionOutcome,
    PullRequestState,
)
from ..identity import SessionIdentity
from ..work.models import ReportContext
from ..work.views import (
    AssociationView,
    AvailabilityView,
    EvidenceView,
    ExecutionView,
    GitEvidenceView,
    IntentView,
    MissingPullRequestView,
    ProgressView,
    PullRequestEvidenceView,
    WorkStory,
)


class PullRequestRecord(TypedDict):
    url: str
    title: str
    state: PullRequestState
    is_draft: bool | None


class EvidenceFields(TypedDict, total=False):
    repository: str | None
    branch: str
    output_present: bool
    conflicts: bool
    behind: bool
    output: PullRequestRecord
    pull_request_found: bool
    execution: str
    reported_progress: str
    reported_execution_outcome: ExecutionOutcome
    uncertainty: tuple[str, ...]
    text: str
    association: str
    authorship: str
    context: ReportContextRecord


class ReferenceRecord(TypedDict):
    kind: ArtifactKind
    ref: str


class ReportContextRecord(TypedDict, total=False):
    branch: str
    pull_request: str
    references: tuple[ReferenceRecord, ...]
    uncertainty: tuple[str, ...]


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
    result: PresentedEvidence = {"source": record.source, "kind": record.kind}
    match record.details:
        case GitEvidenceView() as details:
            result.update(
                repository=details.repository,
                branch=details.branch,
                output_present=details.output_present,
                conflicts=details.conflicts,
                behind=details.behind,
            )
        case PullRequestEvidenceView() as details:
            output = details.output
            result.update(
                output={
                    "url": output.url,
                    "title": output.title,
                    "state": output.state,
                    "is_draft": output.is_draft,
                },
            )
        case MissingPullRequestView() as details:
            result["pull_request_found"] = details.pull_request_found
        case ExecutionView() as details:
            result["execution"] = details.execution
            if details.reported_progress is not None:
                result["reported_progress"] = details.reported_progress
            if details.reported_execution_outcome is not None:
                result["reported_execution_outcome"] = (
                    details.reported_execution_outcome
                )
        case AvailabilityView() as details:
            result["uncertainty"] = details.uncertainty
        case IntentView() as details:
            result["text"] = details.text
        case ProgressView() as details:
            result["reported_progress"] = details.reported_progress
        case AssociationView() as details:
            result.update(
                association=details.association, authorship=details.authorship
            )
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
