from collections import defaultdict
from dataclasses import dataclass

from .config import ScheduleConfig
from .models import (
    Candidate,
    DelegatedTask,
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


def eligible_candidates(
    candidates: tuple[Candidate, ...],
    responsibilities: tuple[Responsibility, ...],
    tasks: tuple[DelegatedTask, ...],
    operations: tuple[Operation, ...],
    now: float,
    config: ScheduleConfig,
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
        changed = has_new_evidence(candidate)
        if candidate.waiting_on in ("human", "handled", "not_relevant") and not changed:
            continue
        if candidate.deferred_until is not None and now < candidate.deferred_until:
            continue
        if now < review_due(candidate, config) and not changed:
            continue
        if changed or now >= review_due(candidate, config):
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
