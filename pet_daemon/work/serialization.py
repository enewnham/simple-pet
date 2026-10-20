import json
from typing import TypedDict

from .models import (
    ArtifactDescriptor,
    AvailabilityEvidence,
    CheckoutAssociation,
    EvidencePayload,
    EvidenceRecord,
    ExecutionEvidence,
    GitHubEvidence,
    IntentEvidence,
    OpportunityContent,
    OpportunityState,
    ProgressEvidence,
    PullRequestAssociation,
    RelatedLink,
    StackAssociation,
)
from .observations import (
    GitObservation,
    pull_request_from_data,
    related_link_data,
)
from .validation import canonical
from .wire import PullRequestData

StoredOrigin = TypedDict("StoredOrigin", {"from": str}, total=False)


class StoredPayload(StoredOrigin, total=False):
    session: str
    text: str
    event: str
    outcome: str | None
    tool_name: str | None
    progress: str | None
    worktree: str
    branch: str
    head: str
    staged: int
    unstaged: int
    untracked: int
    conflicts: int
    upstream: str
    ahead: int
    behind: int
    repository: str
    pr: PullRequestData | None
    local_error: str | None
    github_error: str | None
    checkout: str
    kind: str
    ref: str
    relationship: str
    source: str
    stack: int | None
    position: int | None
    state: str | None
    is_draft: bool | None
    head_branch: str
    title: str


class StoredEvidence(TypedDict):
    source: str
    kind: str
    data: StoredPayload
    observed_at: float | None


class StoredOpportunity(TypedDict):
    intent: str
    artifact: dict[str, str]
    evidence: list[StoredEvidence]
    uncertainty: list[str]
    next_step: str
    sessions: list[str]


def artifact_record(artifact: ArtifactDescriptor) -> dict[str, str]:
    result = {
        "kind": artifact.kind,
        "ref": artifact.ref,
        "repository": artifact.repository,
    }
    if artifact.revision is not None:
        result["revision"] = artifact.revision
    if artifact.checkout is not None:
        result["checkout"] = artifact.checkout
    if artifact.title is not None:
        result["title"] = artifact.title
    return result


def encode_artifact(artifact: ArtifactDescriptor) -> str:
    return canonical(artifact_record(artifact))


def decode_artifact(encoded: str) -> ArtifactDescriptor:
    return ArtifactDescriptor(**json.loads(encoded))


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


def encode_evidence(payload: EvidencePayload) -> str:
    return canonical(payload_record(payload))


def decode_evidence(source: str, kind: str, encoded: str) -> EvidencePayload:
    data: StoredPayload = json.loads(encoded)
    return _decode_payload(source, kind, data)


def _decode_payload(source: str, kind: str, data: StoredPayload) -> EvidencePayload:
    if source == "user" and kind == "intent":
        return IntentEvidence(data["session"], data["text"])
    if source == "harness" and kind == "progress":
        return ProgressEvidence(data["session"], data["text"])
    if source == "harness" and kind == "execution":
        return ExecutionEvidence(
            data["session"],
            data["event"],
            data["outcome"],
            data["tool_name"],
            data["progress"],
        )
    if source == "git" and kind == "observation":
        return GitObservation(
            data["worktree"],
            data["branch"],
            data["head"],
            data["staged"],
            data["unstaged"],
            data["untracked"],
            data["conflicts"],
            data["upstream"],
            data["ahead"],
            data["behind"],
            data["repository"],
        )
    if source == "github" and kind == "observation":
        pr = data["pr"]
        return GitHubEvidence(
            pull_request_from_data(pr) if pr else None, data["repository"]
        )
    if source == "observer" and kind == "availability":
        return AvailabilityEvidence(data["local_error"], data["github_error"])
    if kind == "relationship":
        if source == "harness":
            return CheckoutAssociation(data["checkout"], data["branch"])
        if source == "github":
            return PullRequestAssociation(data["checkout"], data["branch"])
        if source == "github_stack":
            return StackAssociation(
                RelatedLink(
                    data["kind"],
                    data["ref"],
                    data["relationship"],
                    data["source"],
                    data["stack"],
                    data["position"],
                    data["state"],
                    data["is_draft"],
                    data["head"],
                    data["head_branch"],
                    data["title"],
                ),
                data["from"],
            )
    raise ValueError(f"Unsupported evidence: {source}/{kind}")


def evidence_record(record: EvidenceRecord) -> dict[str, object]:
    return {
        "source": record.source,
        "kind": record.kind,
        "data": payload_record(record.payload),
        "observed_at": record.observed_at,
    }


def encode_opportunity(opportunity: OpportunityState) -> str:
    return canonical(
        {
            "intent": opportunity.intent,
            "artifact": artifact_record(opportunity.artifact),
            "evidence": tuple(
                evidence_record(record) for record in opportunity.evidence
            ),
            "uncertainty": opportunity.uncertainty,
            "next_step": opportunity.next_step,
            "sessions": opportunity.sessions,
        }
    )


def decode_opportunity(encoded: str) -> OpportunityContent:
    data: StoredOpportunity = json.loads(encoded)
    return OpportunityContent(
        intent=data["intent"],
        artifact=ArtifactDescriptor(**data["artifact"]),
        evidence=tuple(
            EvidenceRecord(
                record["source"],
                record["kind"],
                _decode_payload(record["source"], record["kind"], record["data"]),
                record["observed_at"],
            )
            for record in data["evidence"]
        ),
        uncertainty=tuple(data["uncertainty"]),
        next_step=data["next_step"],
        sessions=tuple(data["sessions"]),
    )
