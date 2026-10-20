from typing import NotRequired, TypedDict

from .models import Opportunity
from .views import (
    AssociationView,
    AvailabilityView,
    EvidenceView,
    ExecutionView,
    GitEvidenceView,
    IntentView,
    MissingPullRequestView,
    ProgressView,
    PullRequestEvidenceView,
    WorkView,
    project_evidence,
)


class PullRequestRecord(TypedDict):
    url: str
    title: str
    state: str
    is_draft: bool | None
    review: str
    review_requested: bool | None


class EvidenceFields(TypedDict, total=False):
    repository: str | None
    branch: str
    output_present: bool
    conflicts: bool
    behind: bool
    output: PullRequestRecord
    checks: tuple[tuple[str, str], ...]
    checks_current: bool
    pull_request_found: bool
    execution: str
    reported_progress: str
    reported_execution_outcome: str
    uncertainty: tuple[str, ...]
    text: str
    association: str
    authorship: str


class PresentedEvidence(EvidenceFields):
    source: str
    kind: str
    observed_at: NotRequired[float]


class PresentedArtifact(TypedDict):
    kind: str
    ref: str
    title: NotRequired[str]
    evidence: tuple[PresentedEvidence, ...]


class WorkRecord(TypedDict):
    intent: str
    requests: tuple[PresentedEvidence, ...]
    activity: tuple[PresentedEvidence, ...]
    artifacts: tuple[PresentedArtifact, ...]


class OpportunityRecord(TypedDict):
    kind: str
    intent: str
    artifact: dict[str, str]
    evidence: tuple[PresentedEvidence, ...]
    uncertainty: tuple[str, ...]
    next_step: str


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
                    "review": output.review,
                    "review_requested": output.review_requested,
                },
                checks=details.checks,
                checks_current=details.checks_current,
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
    return result


def work_record(work: WorkView) -> WorkRecord:
    return {
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


def opportunity_record(action: Opportunity) -> OpportunityRecord:
    artifact = action.artifact
    return {
        "kind": action.kind,
        "intent": action.intent[:1000],
        "artifact": {
            "kind": artifact.kind,
            "ref": artifact.ref,
            **({"title": artifact.title[:240]} if artifact.title is not None else {}),
        },
        "evidence": tuple(
            evidence_record(value)
            for record in action.evidence
            if (value := project_evidence(record)) is not None
        ),
        "uncertainty": action.uncertainty,
        "next_step": action.next_step,
    }
