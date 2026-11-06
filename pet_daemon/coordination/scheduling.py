from collections import defaultdict
from dataclasses import dataclass, replace

from .config import ScheduleConfig
from .models import (
    Candidate,
    Consideration,
    DelegatedTask,
    InquiryAttempt,
    Operation,
    OperationKind,
    OperationOutcome,
    Responsibility,
    ResponsibilityStatus,
)


@dataclass(frozen=True)
class Selection:
    candidates: tuple[Candidate, ...]
    focus_id: str | None = None
    override: str | None = None


def task_meaning(task: DelegatedTask) -> DelegatedTask:
    return replace(task, revision=1, observed_at=None, adapter_revision=None)


def related_responsibilities(
    candidates: tuple[Candidate, ...], records: tuple[Responsibility, ...]
) -> tuple[Responsibility, ...]:
    identifiers = {item.responsibility_id for item in candidates}
    pending = list(identifiers)
    by_id = {record.id: record for record in records}
    while pending:
        record = by_id.get(pending.pop() or "")
        if record is None:
            continue
        for dependency in record.dependencies:
            if dependency not in identifiers:
                identifiers.add(dependency)
                pending.append(dependency)
    return tuple(record for record in records if record.id in identifiers)


def review_due(candidate: Candidate, config: ScheduleConfig) -> float:
    origin = (
        candidate.first_queued_at
        if candidate.last_serviced_at is None
        else candidate.last_serviced_at
    )
    requested = candidate.next_review_at
    return min(
        requested if requested is not None else origin + config.review_seconds,
        origin + config.max_deferral_seconds,
    )


def has_new_evidence(candidate: Candidate) -> bool:
    return candidate.evidence_revision != candidate.serviced_evidence_revision


def latest_consideration(
    candidate: Candidate, records: tuple[Consideration, ...]
) -> Consideration | None:
    matching = [record for record in records if record.candidate_id == candidate.id]
    return (
        max(matching, key=lambda record: record.candidate_revision)
        if matching
        else None
    )


def due_request(
    candidate: Candidate, responsibility: Responsibility | None, now: float
) -> float | None:
    requested = [candidate.deferred_until]
    if responsibility is not None:
        requested.extend((responsibility.due_at, responsibility.followup_at))
    return max(
        (value for value in requested if value is not None and value <= now),
        default=None,
    )


def can_attempt(
    candidate: Candidate,
    responsibility: Responsibility | None,
    prior: Consideration | None,
    attempts: tuple[InquiryAttempt, ...],
    instruction_revision: int,
    now: float,
    config: ScheduleConfig,
    last_instruction_at: float | None = None,
) -> bool:
    due = due_request(candidate, responsibility, now)
    failures = []
    for attempt in attempts:
        if not attempt.error or candidate.id not in attempt.candidate_ids:
            continue
        if attempt.selected_candidate_id not in (None, candidate.id):
            continue
        legacy = not attempt.evidence_revisions
        if (
            legacy
            and last_instruction_at is not None
            and last_instruction_at > attempt.created_at
        ):
            continue
        if not legacy and attempt.instruction_revision != instruction_revision:
            continue
        evidence = dict(attempt.evidence_revisions).get(candidate.id)
        if evidence is None:
            if candidate.evidence_revision != 1:
                continue
        elif evidence != candidate.evidence_revision:
            continue
        if prior is not None and attempt.created_at <= prior.considered_at:
            continue
        if due is not None and attempt.created_at < due:
            continue
        failures.append(attempt)
    if not failures:
        return True
    if len(failures) >= 2:
        return False
    last = max(attempt.completed_at or attempt.created_at for attempt in failures)
    return now >= last + config.failure_retry_seconds


def provider_failed(
    attempts: tuple[InquiryAttempt, ...], last_instruction_at: float | None
) -> bool:
    return any(
        attempt.error_kind
        in (
            "billing_error",
            "authentication_failed",
            "authentication_error",
            "invalid_api_key",
        )
        and (last_instruction_at is None or last_instruction_at <= attempt.created_at)
        for attempt in attempts
    )


def eligible_candidates(
    candidates: tuple[Candidate, ...],
    responsibilities: tuple[Responsibility, ...],
    tasks: tuple[DelegatedTask, ...],
    operations: tuple[Operation, ...],
    now: float,
    config: ScheduleConfig,
    considerations: tuple[Consideration, ...] = (),
) -> tuple[Candidate, ...]:
    responsibility_by_id = {record.id: record for record in responsibilities}
    task_by_id = {task.id: task for task in tasks}
    blocked = {
        task.responsibility_id
        for operation in operations
        if operation.outcome in (OperationOutcome.PENDING, OperationOutcome.UNKNOWN)
        and operation.kind
        in (OperationKind.LAUNCH, OperationKind.CONTINUE, OperationKind.STOP)
        and (task := task_by_id.get(operation.task_id)) is not None
    }
    eligible = []
    for candidate in candidates:
        responsibility = responsibility_by_id.get(candidate.responsibility_id or "")
        if responsibility is not None and responsibility.status in (
            ResponsibilityStatus.PAUSED,
            ResponsibilityStatus.COMPLETED,
        ):
            continue
        if candidate.responsibility_id in blocked:
            continue
        prior = latest_consideration(candidate, considerations)
        considered = prior.evidence_revision if prior is not None else None
        changed = (
            candidate.evidence_revision != considered
            if considered is not None
            else has_new_evidence(candidate)
        )
        if candidate.waiting_on in ("human", "handled", "not_relevant") and not changed:
            continue
        if candidate.deferred_until is not None and now < candidate.deferred_until:
            continue
        due = due_request(candidate, responsibility, now)
        considered_at = (
            prior.considered_at if prior is not None else candidate.last_serviced_at
        )
        timed = due is not None and (considered_at is None or considered_at < due)
        if changed or timed:
            eligible.append(candidate)
    return tuple(eligible)


def select(
    candidates: tuple[Candidate, ...],
    now: float,
    config: ScheduleConfig,
    history: tuple[Candidate, ...] = (),
) -> Selection:
    if not candidates:
        return Selection(())
    grouped: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.project_id or candidate.scope].append(candidate)
    records = history or candidates
    served = {
        project: max(
            (
                candidate.last_serviced_at
                for candidate in records
                if (candidate.project_id or candidate.scope) == project
                and candidate.last_serviced_at is not None
            ),
            default=float("-inf"),
        )
        for project, peers in grouped.items()
    }

    def order(candidate: Candidate) -> tuple[float, float, float, str]:
        return (
            -candidate.priority,
            candidate.due_at if candidate.due_at is not None else float("inf"),
            candidate.first_queued_at,
            candidate.id,
        )

    for peers in grouped.values():
        peers.sort(key=order)
    projects = sorted(grouped, key=lambda project: (served[project], project))
    routine = []
    while projects:
        remaining = []
        for project in projects:
            peers = grouped[project]
            routine.append(peers.pop(0))
            if peers:
                remaining.append(project)
        projects = remaining
    overdue = tuple(
        candidate for candidate in candidates if now >= review_due(candidate, config)
    )
    priority = tuple(
        candidate
        for candidate in candidates
        if candidate.exclusive
        or candidate.due_at is not None
        and candidate.due_at <= now
    )
    focus = None
    reason = None
    if priority:
        focus = min(priority, key=order)
        if overdue and focus not in overdue:
            reason = "Explicit user priority or a due obligation takes precedence; overdue work remains queued"
    elif overdue:
        focus = min(
            overdue,
            key=lambda candidate: (
                served[candidate.project_id or candidate.scope],
                review_due(candidate, config),
                candidate.first_queued_at,
                candidate.id,
            ),
        )
    ordered = ([focus] if focus is not None else []) + [
        candidate for candidate in routine if candidate != focus
    ]
    return Selection(
        tuple(ordered[: config.max_candidates]), focus.id if focus else None, reason
    )
