import re
from collections.abc import Callable
from dataclasses import dataclass

from ..coordination.models import (
    ChatRole,
    Consideration,
    DelegatedTask,
    Instruction,
    Note,
    NoteKind,
    NoteOrigin,
    NoteStatus,
    Operation,
    Project,
    Responsibility,
    SourceReference,
)
from ..storage.memory import Memory
from ..work.history import report_prose, report_source_key
from ..work.models import IntentEvidence
from ..work.observations import CheckoutObservation
from .config import RetrievalConfig
from .index import SourceChange, SourceIndex
from .providers import Document, SourceStatus, revision
from .retrieval import SearchContext


@dataclass(frozen=True)
class UserInput:
    id: str
    text: str
    scope: str
    created_at: float
    source: SourceReference


@dataclass(frozen=True)
class WorkBrief:
    purpose: str
    scope: str
    instruction_revision: int
    instructions: tuple[Instruction, ...]
    recent_user_inputs: tuple[UserInput, ...]
    notes: tuple[Note, ...]
    projects: tuple[Project, ...]
    responsibilities: tuple[Responsibility, ...]
    tasks: tuple[DelegatedTask, ...]
    operations: tuple[Operation, ...]
    considerations: tuple[Consideration, ...]
    recent_work: tuple[Document, ...]
    changes: tuple[SourceChange, ...]
    sources: tuple[SourceStatus, ...]
    aliases: tuple[str, ...]
    human_questions: tuple[str, ...]
    unknowns: tuple[str, ...]
    checkpoint: int


def applies(scope: str, selected: str) -> bool:
    return not scope or not selected or scope == selected


class WorkContext:
    def __init__(
        self,
        memory: Memory,
        index: SourceIndex,
        config: RetrievalConfig,
        clock: Callable[[], float],
    ) -> None:
        self.memory = memory
        self.index = index
        self.config = config
        self.clock = clock

    def recent_user_inputs(self, scope: str) -> tuple[UserInput, ...]:
        store = self.memory.coordination
        result = []
        for instruction in reversed(store.instructions()):
            if (
                not instruction.active
                or not instruction.context_only
                or not applies(instruction.scope, scope)
            ):
                continue
            message = store.message(instruction.message_id or "")
            if (
                message is None
                or message.role != ChatRole.USER
                or message.text != instruction.text
            ):
                continue
            source = SourceReference(
                instruction.id,
                "user",
                str(instruction.revision),
                message.created_at,
                uri=f"conversation://message/{message.id}",
                scope=instruction.scope,
                original_at=message.created_at,
            )
            result.append(
                UserInput(
                    message.id,
                    message.text,
                    instruction.scope,
                    message.created_at,
                    source,
                )
            )
            if len(result) >= self.config.user_inputs:
                break
        return tuple(reversed(result))

    def note_documents(self) -> tuple[Document, ...]:
        return tuple(
            Document(
                SourceReference(
                    id=f"notes:{note.id}",
                    provider="notes",
                    revision=str(note.revision),
                    observed_at=note.observed_at,
                    original_at=note.original_at,
                    scope=note.scope,
                ),
                f"{note.origin.value} {note.kind.value}",
                f"Support: {note.support.value}. Status: {note.status.value}.\n\n{note.text}",
                kind=f"note:{note.origin.value}:{note.kind.value}",
                links=tuple(source.id for source in note.sources),
                superseded=note.status
                in (NoteStatus.SUPERSEDED, NoteStatus.ARCHIVED, NoteStatus.REJECTED),
                leaf_sources=self.note_sources(note),
            )
            for note in self.memory.coordination.notes()
        )

    def note_sources(self, note: Note) -> tuple[SourceReference, ...]:
        if note.origin != NoteOrigin.USER:
            return note.sources
        instruction = self.memory.coordination.instruction(note.instruction_id or "")
        if instruction is None:
            return ()
        return (
            SourceReference(
                instruction.id,
                "user",
                str(instruction.revision),
                instruction.created_at,
                uri=f"conversation://message/{instruction.message_id}",
                scope=note.scope,
                original_at=instruction.created_at,
            ),
        )

    def history(self) -> tuple[Document, ...]:
        reports = self.memory.recent_work_reports(self.config.bootstrap_reports)
        documents = []
        for report in reports:
            text = report_prose(report)
            if not text:
                continue
            references = tuple(reference.ref for reference in report.context.references)
            if report.context.pull_request:
                references = tuple(
                    dict.fromkeys((*references, report.context.pull_request))
                )
            session = report.payload.session
            documents.append(
                Document(
                    SourceReference(
                        id=f"work:{report_source_key(report)}",
                        provider="work",
                        revision=revision(report.model_dump_json()),
                        observed_at=report.observed_at,
                        scope=report.context.checkout,
                        uri=f"agent://{session.source}/{session.native_id}",
                        original_at=report.capture_at,
                    ),
                    "User request reported by harness"
                    if isinstance(report.payload, IntentEvidence)
                    else "Harness work report",
                    text[: self.config.source_characters],
                    kind="reported_intent"
                    if isinstance(report.payload, IntentEvidence)
                    else "harness_report",
                    topics=(report.context.branch,) if report.context.branch else (),
                    links=references,
                )
            )
        facts = self.memory.work_facts()
        for observation in (*facts.checkouts.values(), *facts.pull_requests.values()):
            text = observation.model_dump_json()
            identifier = (
                observation.worktree
                if isinstance(observation, CheckoutObservation)
                else observation.ref
            )
            documents.append(
                Document(
                    SourceReference(
                        id=f"work:observation:{identifier}",
                        provider="work",
                        revision=revision(
                            observation.model_dump_json(
                                exclude={"checked_at", "last_success_at"}
                            )
                        ),
                        observed_at=observation.checked_at,
                        uri=identifier,
                        scope=getattr(observation, "worktree", ""),
                        available=not bool(observation.error),
                    ),
                    "Repository observation",
                    text[: self.config.source_characters],
                    kind="observation",
                    links=(identifier,),
                    checked_at=observation.checked_at,
                )
            )
        return tuple(documents)

    def brief(self, purpose: str, scope: str = "", checkpoint: int = 0) -> WorkBrief:
        store = self.memory.coordination
        notes = tuple(
            note
            for note in store.notes()
            if applies(note.scope, scope)
            and note.status in (NoteStatus.ACTIVE, NoteStatus.NEEDS_RECHECK)
        )
        corrections = tuple(
            note
            for note in notes
            if note.origin == NoteOrigin.USER and note.kind == NoteKind.CORRECTION
        )
        remaining = tuple(note for note in notes if note not in corrections)
        tokens = set(re.findall(r"\w+", purpose.casefold()))
        ordered_notes = sorted(
            remaining,
            key=lambda note: len(
                tokens & set(re.findall(r"\w+", note.text.casefold()))
            ),
            reverse=True,
        )
        projects = tuple(
            project for project in store.projects() if applies(project.scope, scope)
        )
        responsibilities = tuple(
            item for item in store.responsibilities() if applies(item.scope, scope)
        )
        responsibility_ids = {item.id for item in responsibilities}
        tasks = tuple(
            task
            for task in store.tasks()
            if task.responsibility_id in responsibility_ids
        )
        task_ids = {task.id for task in tasks}
        recent = tuple(
            document
            for document in (
                *self.history(),
                *(
                    document
                    for document in self.index.documents()
                    if document.source.provider.startswith("history:")
                ),
            )
            if applies(document.source.scope, scope)
        )
        recent = tuple(
            sorted(
                recent,
                key=lambda document: (
                    len(tokens & set(re.findall(r"\w+", document.text.casefold()))),
                    document.source.original_at or document.source.observed_at,
                ),
                reverse=True,
            )
        )
        statuses = self.index.statuses()
        unknowns = tuple(
            f"{status.provider}: {status.availability.value}: {status.detail}"
            for status in statuses
            if status.availability.value != "available"
        )
        if not recent:
            unknowns = (
                *unknowns,
                "No configured historical work reports are available.",
            )
        return WorkBrief(
            purpose=purpose,
            scope=scope,
            instruction_revision=store.state().instruction_revision,
            recent_user_inputs=self.recent_user_inputs(scope),
            instructions=tuple(
                instruction
                for instruction in store.instructions()
                if instruction.active
                and not instruction.context_only
                and applies(instruction.scope, scope)
            ),
            notes=(*corrections, *ordered_notes[: self.config.results]),
            projects=projects,
            responsibilities=responsibilities,
            tasks=tasks,
            operations=tuple(
                operation
                for operation in store.operations()
                if operation.task_id in task_ids
            ),
            considerations=tuple(
                record
                for record in store.considerations()
                if any(
                    candidate.id == record.candidate_id
                    and applies(candidate.scope, scope)
                    for candidate in store.candidates()
                )
            )[-self.config.results :],
            recent_work=recent[: self.config.bootstrap_reports],
            changes=self.index.changes(checkpoint),
            sources=statuses,
            aliases=tuple(
                dict.fromkeys(
                    alias for project in projects for alias in project.aliases
                )
            ),
            human_questions=tuple(
                dict.fromkeys(
                    (
                        *(
                            item.waiting_on
                            for item in responsibilities
                            if item.waiting_on
                        ),
                        *(
                            candidate.waiting_on
                            for candidate in store.candidates()
                            if applies(candidate.scope, scope) and candidate.waiting_on
                        ),
                    )
                )
            ),
            unknowns=unknowns,
            checkpoint=self.index.checkpoint(),
        )

    def search_context(self, brief: WorkBrief) -> SearchContext:
        references = tuple(
            dict.fromkeys(
                reference
                for document in brief.recent_work
                for reference in document.links
            )
        )
        return SearchContext(
            scope=brief.scope,
            references=references,
            aliases=brief.aliases,
            excluded_references=tuple(
                dict.fromkeys(
                    reference
                    for note in brief.notes
                    if note.origin == NoteOrigin.USER
                    and note.kind == NoteKind.CORRECTION
                    for reference in note.excluded_references
                )
            ),
            confirmed_links=tuple(
                dict.fromkeys(
                    source.uri
                    for project in brief.projects
                    for source in project.sources
                    if source.uri
                )
            ),
            corrections=tuple(
                note.text
                for note in brief.notes
                if note.origin == NoteOrigin.USER and note.kind == NoteKind.CORRECTION
            ),
        )
