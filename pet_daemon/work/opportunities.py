import hashlib
from collections.abc import Callable, Iterable
from typing import TypedDict

from .models import (
    AvailabilityEvidence,
    EvidenceRecord,
    ExecutionEvidence,
    GitHubEvidence,
    Opportunity,
    OpportunityState,
    WorkAccount,
    WorkState,
)
from .observations import GitObservation
from .serialization import artifact_record, payload_record
from .validation import FAILURES, canonical


class EvidenceGround(TypedDict):
    source: str
    kind: str
    data: dict[str, object]


def evidence_grounds(records: Iterable[EvidenceRecord]) -> list[EvidenceGround]:
    result: list[EvidenceGround] = []
    for record in records:
        payload = record.payload
        data: dict[str, object]
        match payload:
            case ExecutionEvidence():
                ended = payload.event in {
                    "turn_done",
                    "turn_error",
                    "subagent_done",
                    "interrupt",
                    "session_end",
                }
                data = {
                    "execution": "turn ended"
                    if ended
                    else "input requested"
                    if payload.event == "waiting"
                    else "in progress"
                }
                if payload.progress:
                    data["reported_progress"] = payload.progress
                if ended and payload.outcome:
                    data["reported_execution_outcome"] = payload.outcome
            case GitObservation():
                data = {
                    "repository": payload.repository,
                    "branch": payload.branch,
                    "head": payload.head,
                    "upstream": payload.upstream,
                    "dirty": bool(
                        payload.staged or payload.unstaged or payload.untracked
                    ),
                    "conflicts": bool(payload.conflicts),
                    "ahead": bool(payload.ahead),
                    "behind": bool(payload.behind),
                }
            case AvailabilityEvidence():
                data = {
                    "local_error": bool(payload.local_error),
                    "github_error": bool(payload.github_error),
                }
            case GitHubEvidence():
                data = {
                    "pr": {
                        key: value
                        for key, value in payload.pr.data().items()
                        if key != "related_links_error"
                    }
                    if payload.pr
                    else None,
                    "repository": payload.repository,
                }
            case _:
                data = {
                    key: value
                    for key, value in payload_record(payload).items()
                    if key != "session"
                }
        result.append({"source": record.source, "kind": record.kind, "data": data})
    return sorted(result, key=lambda item: (item["source"], item["kind"]))


def derive_opportunities(
    state: WorkState, now: float, context_for_work: Callable[[str], WorkAccount | None]
) -> None:
    candidates = set()
    for work in state.items.values():
        if work.status != "active":
            continue
        progress = state.latest(work.id, "").get(("harness", "progress"))
        sessions = tuple(
            sorted(
                session for work_id, session in state.memberships if work_id == work.id
            )
        )
        account = context_for_work(work.id)
        assert account is not None
        artifacts = (
            state.artifacts[artifact_id]
            for work_id, artifact_id in sorted(state.links)
            if work_id == work.id
        )
        for artifact in artifacts:
            latest = state.latest(work.id, artifact.id)
            github = next(
                (
                    item.payload
                    for item in latest.values()
                    if isinstance(item.payload, GitHubEvidence)
                ),
                None,
            )
            pr = github.pr if github else None
            availability = next(
                (
                    item.payload
                    for item in latest.values()
                    if isinstance(item.payload, AvailabilityEvidence)
                ),
                None,
            )
            local = next(
                (
                    item.payload
                    for item in latest.values()
                    if isinstance(item.payload, GitObservation)
                ),
                None,
            )
            kinds = []
            uncertainty = [
                "Checkout association does not establish who authored the output."
            ]
            if availability and availability.github_error:
                uncertainty.append(
                    "GitHub refresh failed. Status comes from the last successful observation."
                )
            if availability and availability.local_error:
                uncertainty.append(
                    "Local repository refresh failed. Output information comes from the last successful observation."
                )
            if artifact.kind == "pull_request" and pr and pr.state == "OPEN":
                if pr.review_requested is True:
                    kinds.append(
                        (
                            "review_output",
                            "Review the pull request that requests your review.",
                        )
                    )
                current_checks = pr.head and pr.check_revision == pr.head
                failing = any(
                    len(check) >= 2
                    and check[1] in FAILURES
                    and (len(check) < 4 or not check[3] or check[3] == pr.head)
                    for check in pr.checks
                )
                if current_checks and failing:
                    kinds.append(
                        (
                            "resolve_checks",
                            "Inspect the failing checks on this pull request.",
                        )
                    )
            if (
                artifact.kind == "branch"
                and artifact.checkout
                and github is not None
                and pr is None
                and progress
                and not (
                    availability
                    and (availability.github_error or availability.local_error)
                )
                and local
                and (local.staged or local.unstaged or local.untracked or local.ahead)
            ):
                kinds.append(
                    (
                        "prepare_pr",
                        "The branch output may benefit from review and pull request preparation.",
                    )
                )
                uncertainty.append(
                    "Agent progress does not establish that the work is complete."
                )
            evidence_records = list(latest.values())
            if progress:
                evidence_records.append(progress)
            evidence = tuple(record.record() for record in evidence_records)
            for kind, next_step in kinds:
                identifier = hashlib.sha256(
                    f"{work.id}:{artifact.id}:{kind}".encode()
                ).hexdigest()[:24]
                output = artifact.descriptor()
                fingerprint = canonical(
                    {
                        "intent": work.intent,
                        "artifact": artifact_record(output),
                        "evidence": evidence_grounds(evidence),
                        "kind": kind,
                        "next_step": next_step,
                        "connected_work": account.revision,
                    }
                )
                previous = state.opportunities.get(identifier)
                if previous:
                    changed = previous.fingerprint != fingerprint or not previous.active
                    previous.version += int(changed)
                    previous.fingerprint = fingerprint
                    previous.intent, previous.artifact, previous.evidence = (
                        work.intent,
                        output,
                        evidence,
                    )
                    previous.uncertainty, previous.next_step, previous.sessions = (
                        tuple(uncertainty),
                        next_step,
                        sessions,
                    )
                    previous.active = True
                    if changed:
                        previous.changed_at = now
                else:
                    state.opportunities[identifier] = OpportunityState(
                        identifier,
                        work.id,
                        kind,
                        artifact.id,
                        1,
                        fingerprint,
                        work.intent,
                        output,
                        evidence,
                        tuple(uncertainty),
                        next_step,
                        sessions,
                        True,
                        0,
                        now,
                    )
                candidates.add(identifier)
    for record in state.opportunities.values():
        if record.active and record.id not in candidates:
            record.active = False
            record.version += 1
            record.changed_at = now


def eligible(record: OpportunityState) -> bool:
    return record.active and record.surfaced_version != record.version


def project(record: OpportunityState) -> Opportunity:
    return Opportunity(
        record.id,
        record.version,
        record.kind,
        record.work_id,
        record.intent,
        record.artifact,
        record.evidence,
        record.uncertainty,
        record.next_step,
        record.changed_at,
        record.sessions,
    )
