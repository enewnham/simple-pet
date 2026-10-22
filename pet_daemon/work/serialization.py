from dataclasses import asdict
from typing import TypedDict

from ..domain import ArtifactKind, EvidenceKind, ExecutionOutcome
from .models import (
    ArtifactReference,
    AvailabilityEvidence,
    CheckoutAssociation,
    EvidencePayload,
    EvidenceRecord,
    ExecutionEvidence,
    GitHubEvidence,
    IntentEvidence,
    ProgressEvidence,
    PullRequestAssociation,
    ReferenceAssociation,
    ReportContext,
    SessionReport,
    StackAssociation,
)
from .observations import (
    GitObservation,
    related_link_data,
)


class StoredPayload(TypedDict, total=False):
    session: str
    text: str
    event: str
    outcome: str | None
    tool_name: str | None
    progress: str | None


class StoredArtifactReference(TypedDict):
    kind: str
    ref: str
    repository: str
    revision: str | None
    checkout: str | None
    title: str | None


class StoredContext(TypedDict):
    checkout: str
    repository: str
    branch: str
    pull_request: str
    references: list[StoredArtifactReference]
    branch_resolution: int


class StoredSessionReport(TypedDict):
    source: str
    kind: str
    data: StoredPayload
    observed_at: float
    context: StoredContext


def payload_record(payload: EvidencePayload) -> dict[str, object]:
    match payload:
        case IntentEvidence() | ProgressEvidence():
            return {"session": payload.session, "text": payload.text}
        case ExecutionEvidence():
            return {
                "session": payload.session,
                "event": payload.event,
                "outcome": payload.outcome,
                "tool_name": payload.tool_name,
                "progress": payload.progress,
            }
        case CheckoutAssociation():
            return {
                "relationship": "session_checkout",
                "checkout": payload.checkout,
                "branch": payload.branch,
                "association_only": True,
            }
        case PullRequestAssociation():
            return {
                "relationship": "checkout_pull_request",
                "checkout": payload.checkout,
                "branch": payload.branch,
                "association_only": True,
            }
        case ReferenceAssociation():
            return {"ref": payload.ref}
        case StackAssociation():
            return {
                **related_link_data(payload.link),
                "from": payload.from_ref,
                "association_only": True,
            }
        case GitObservation():
            return dict(payload.data())
        case GitHubEvidence():
            return {
                "pr": payload.pr.data() if payload.pr else None,
                "repository": payload.repository,
            }
        case AvailabilityEvidence():
            return {
                "local_error": payload.local_error,
                "github_error": payload.github_error,
            }


def report_payload(
    kind: EvidenceKind, data: StoredPayload
) -> IntentEvidence | ProgressEvidence | ExecutionEvidence:
    match kind:
        case EvidenceKind.INTENT:
            return IntentEvidence(data["session"], data["text"])
        case EvidenceKind.PROGRESS:
            return ProgressEvidence(data["session"], data["text"])
        case EvidenceKind.EXECUTION:
            return ExecutionEvidence(
                data["session"],
                data["event"],
                ExecutionOutcome(data["outcome"])
                if data["outcome"] is not None
                else None,
                data["tool_name"],
                data["progress"],
            )
        case _:
            raise ValueError(f"Unsupported session report: {kind}")


def evidence_record(record: EvidenceRecord) -> dict[str, object]:
    return {
        "source": record.source,
        "kind": record.kind,
        "data": payload_record(record.payload),
        "observed_at": record.observed_at,
    }


def report_data(value: SessionReport) -> dict[str, object]:
    return {**evidence_record(value.record()), "context": asdict(value.context)}


def report_from_data(value: StoredSessionReport) -> SessionReport:
    context = value["context"]
    return SessionReport(
        report_payload(EvidenceKind(value["kind"]), value["data"]),
        ReportContext(
            context["checkout"],
            context["repository"],
            context["branch"],
            context["pull_request"],
            tuple(
                ArtifactReference(
                    kind=ArtifactKind(item["kind"]),
                    ref=item["ref"],
                    repository=item["repository"],
                    revision=item.get("revision"),
                    checkout=item.get("checkout"),
                    title=item.get("title"),
                )
                for item in context["references"]
            ),
            context["branch_resolution"],
        ),
        value["observed_at"],
    )
