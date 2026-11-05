import uuid
from dataclasses import dataclass
from datetime import datetime

from .coordination.models import (
    Candidate,
    ChatAction,
    DelegatedTask,
    NoteKind,
    OperationKind,
    SourceReference,
)
from .storage.coordination import SQLiteCoordinationRepository


@dataclass(frozen=True)
class ActivityEntry:
    id: str
    category: str
    title: str
    text: str
    state: str
    created_at: float
    reason: str = ""
    revisit_at: float | None = None
    sources: tuple[SourceReference, ...] = ()
    actions: tuple[ChatAction, ...] = ()


def action(
    kind: str,
    target: str,
    label: str,
    revision: int | None = None,
    source: SourceReference | None = None,
) -> ChatAction:
    identifier = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"activity:{kind}:{target}:{revision}:{source.revision if source else ''}",
        )
    )
    return ChatAction(identifier, label, kind, target, revision, source)


def feedback_actions(target: str, revision: int) -> tuple[ChatAction, ...]:
    return tuple(
        action(kind, target, label, revision)
        for kind, label in (
            ("later", "Later (1 hour)"),
            ("tomorrow", "Tomorrow"),
            ("handled", "Handled"),
            ("not_relevant", "Not relevant"),
        )
    )


def source_detail(source: SourceReference) -> str:
    original = (
        stamp(source.original_at) if source.original_at is not None else "unknown"
    )
    return (
        f"{source.provider} · {'available' if source.available else 'unavailable'} · revision {source.revision}\n"
        f"Source date: {original}. Observed: {stamp(source.observed_at)}. Scope: {source.scope or 'unspecified'}."
    )


def stamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).astimezone().isoformat(timespec="minutes")


class Activity:
    def __init__(
        self,
        store: SQLiteCoordinationRepository,
        capabilities: tuple[OperationKind, ...] = (),
    ) -> None:
        self.store = store
        self.capabilities = capabilities

    def entries(self) -> tuple[ActivityEntry, ...]:
        entries = []
        for responsibility in self.store.responsibilities():
            entries.append(
                ActivityEntry(
                    "responsibility:" + responsibility.id,
                    "responsibility",
                    responsibility.purpose,
                    f"Scope: {responsibility.scope}\nExpected outcome: {responsibility.completion_condition or responsibility.purpose}",
                    responsibility.status.value,
                    responsibility.created_at,
                    responsibility.waiting_on
                    or "Adopted by the user. Completion requires supporting evidence.",
                    responsibility.followup_at,
                    responsibility.sources,
                    feedback_actions(responsibility.id, responsibility.revision),
                )
            )
        entries.extend(
            self._candidate(candidate) for candidate in self.store.candidates()
        )
        entries.extend(self._task(task) for task in self.store.tasks())
        notices = {notice.activity_id: notice for notice in self.store.notices()}
        for record in self.store.activity():
            notice = notices.get(record.id)
            delivery = (
                " Report awaits a quiet delivery opportunity."
                if notice is not None and notice.delivered_at is None
                else " Report delivered at " + stamp(notice.delivered_at) + "."
                if notice is not None and notice.delivered_at is not None
                else ""
            )
            entries.append(
                ActivityEntry(
                    "record:" + record.id,
                    "result"
                    if "complete" in record.kind or "result" in record.kind
                    else "decision",
                    record.title,
                    record.detail,
                    record.kind,
                    record.created_at,
                    (
                        "Recorded outcome"
                        if record.sources
                        else "Recorded event. Inspect its origin before inferring verification."
                    )
                    + delivery,
                    sources=record.sources,
                )
            )
        for note in self.store.notes():
            actions = feedback_actions(note.id, note.revision)
            if note.kind == NoteKind.HYPOTHESIS:
                actions += tuple(
                    action(kind, note.id, label, note.revision)
                    for kind, label in (
                        ("selected_yes", "Yes"),
                        ("selected_no", "No"),
                        ("selected_unsure", "Not sure"),
                    )
                )
            entries.append(
                ActivityEntry(
                    "note:" + note.id,
                    "investigation",
                    note.kind.value.capitalize(),
                    note.text,
                    note.status.value,
                    note.created_at,
                    f"Origin: {note.origin.value} ({note.actor}). Support: {note.support.value}. Scope: {note.scope}. "
                    f"Source date: {stamp(note.original_at) if note.original_at is not None else 'unknown'}. "
                    f"Observed: {stamp(note.observed_at)}."
                    + (
                        f" Supersedes: {', '.join(note.supersedes)}."
                        if note.supersedes
                        else ""
                    ),
                    sources=note.sources,
                    actions=actions,
                )
            )
        for feedback in self.store.feedback():
            entries.append(
                ActivityEntry(
                    "feedback:" + feedback.id,
                    "deferred" if feedback.deferred_until else "decision",
                    feedback.kind.value.replace("_", " ").capitalize(),
                    f"User feedback on {feedback.target_id}. Scope: {feedback.scope}.",
                    feedback.kind.value,
                    feedback.created_at,
                    "Direct user decision at the selected scope.",
                    feedback.deferred_until,
                )
            )
        for selected in self.store.user_actions():
            entries.append(
                ActivityEntry(
                    "user-action:" + selected.id,
                    "decision",
                    selected.label,
                    f"User requested {selected.kind} for {selected.target_id or 'coordination'}. Scope: {selected.scope or 'coordination'}.",
                    selected.outcome.value,
                    selected.created_at,
                    selected.error
                    or "An accepted user selection is separate from the operation receipt and observed task outcome.",
                )
            )
        for operation in self.store.operations():
            entries.append(
                ActivityEntry(
                    "operation:" + operation.id,
                    "decision",
                    f"{operation.kind.value.capitalize()} request",
                    operation.instruction or f"Task: {operation.task_id}",
                    operation.outcome.value,
                    operation.created_at,
                    operation.error
                    or "Operation acceptance and observed execution are separate facts.",
                    sources=operation.evidence_sources,
                )
            )
        for attempt in self.store.attempts():
            if attempt.error:
                entries.append(
                    ActivityEntry(
                        "attempt:" + attempt.id,
                        "deferred",
                        "Investigation pending",
                        attempt.error,
                        "retry pending",
                        attempt.created_at,
                        "This review did not complete. It remains eligible for reconsideration.",
                    )
                )
        return tuple(
            sorted(
                entries, key=lambda entry: (entry.created_at, entry.id), reverse=True
            )
        )

    def _candidate(self, candidate: Candidate) -> ActivityEntry:
        considerations = self.store.considerations(candidate.id)
        latest = considerations[-1] if considerations else None
        waiting = (
            candidate.deferred_until or candidate.next_review_at or candidate.due_at
        )
        state = latest.disposition.value if latest else "queued"
        reason = latest.reason if latest else "Waiting for a coordination review."
        if candidate.waiting_on:
            reason += f" Reconsider when: {candidate.waiting_on}."
        reason += f" Priority: {candidate.priority}. First queued: {stamp(candidate.first_queued_at)}."
        return ActivityEntry(
            "candidate:" + candidate.id,
            "deferred"
            if state in ("wait", "defer") or candidate.deferred_until
            else "investigation",
            candidate.question,
            f"Scope: {candidate.scope}",
            state,
            candidate.first_queued_at,
            reason,
            waiting,
            latest.sources if latest else (),
            feedback_actions(candidate.id, candidate.revision),
        )

    def _task(self, task: DelegatedTask) -> ActivityEntry:
        actions = []
        if task.session is not None:
            if OperationKind.OPEN in self.capabilities:
                actions.append(
                    action("open_task", task.id, "Open native session", task.revision)
                )
            if OperationKind.STOP in self.capabilities and task.status.value not in (
                "stopped",
                "ended",
            ):
                actions.append(
                    action("stop_task", task.id, "Stop this task", task.revision)
                )
        if OperationKind.INSPECT in self.capabilities:
            actions.append(
                action("inspect_task", task.id, "Refresh task", task.revision)
            )
        text = f"{task.objective}\nExpected result: {task.expected_result}\nRepository: {task.repository}"
        if task.session is not None:
            text += f"\nNative session: {task.session.harness} / {task.session.session}\nWorkspace: {task.session.workspace}"
        if task.observed_input is not None:
            text += f"\nObserved native input: {task.observed_input}"
        reason = f"Observed readiness: {task.readiness.value}. "
        reason += (
            task.last_error
            or "A session report or ended turn does not establish the adopted outcome."
        )
        if OperationKind.STOP not in self.capabilities:
            reason += " This adapter does not support stopping tasks."
        return ActivityEntry(
            "task:" + task.id,
            "session",
            task.objective,
            text,
            task.status.value,
            task.observed_at or task.created_at,
            reason,
            sources=task.sources,
            actions=tuple(actions),
        )
