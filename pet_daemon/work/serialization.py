from dataclasses import asdict
from typing import TypedDict

from ..domain import ArtifactKind, EvidenceKind, ExecutionOutcome
from ..identity import SessionData, session_data, session_from_data
from ..repository_identity import RepositoryData, repository_data, repository_from_data
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
    session: SessionData
    text: str
    event: str
    outcome: str | None
    tool_name: str | None
    progress: str | None


class StoredArtifactReference(TypedDict):
    kind: str
    ref: str
    repository: RepositoryData | None
    revision: str | None
    checkout: str | None
    title: str | None


class StoredContext(TypedDict):
    checkout: str
    repository: RepositoryData | None
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
            return {"session": session_data(payload.session), "text": payload.text}
        case ExecutionEvidence():
            return {
                "session": session_data(payload.session),
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
                "repository": repository_data(payload.repository),
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
            return IntentEvidence(session_from_data(data["session"]), data["text"])
        case EvidenceKind.PROGRESS:
            return ProgressEvidence(session_from_data(data["session"]), data["text"])
        case EvidenceKind.EXECUTION:
            return ExecutionEvidence(
                session_from_data(data["session"]),
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
    context = value.context
    return {
        **evidence_record(value.record()),
        "context": {
            "checkout": context.checkout,
            "repository": repository_data(context.repository),
            "branch": context.branch,
            "pull_request": context.pull_request,
            "references": [
                {**asdict(item), "repository": repository_data(item.repository)}
                for item in context.references
            ],
            "branch_resolution": context.branch_resolution,
        },
    }


def report_from_data(value: StoredSessionReport) -> SessionReport:
    context = value["context"]
    return SessionReport(
        report_payload(EvidenceKind(value["kind"]), value["data"]),
        ReportContext(
            context["checkout"],
            repository_from_data(context["repository"]),
            context["branch"],
            context["pull_request"],
            tuple(
                ArtifactReference(
                    kind=ArtifactKind(item["kind"]),
                    ref=item["ref"],
                    repository=repository_from_data(item["repository"]),
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
