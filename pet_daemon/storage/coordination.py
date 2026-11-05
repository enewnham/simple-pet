import sqlite3
from collections.abc import Callable
from dataclasses import replace

from pydantic import TypeAdapter

from ..coordination.models import (
    ActivityRecord,
    Candidate,
    ChatMessage,
    ChatRetry,
    ChatRole,
    Consideration,
    Conversation,
    CoordinationState,
    DelegatedTask,
    Instruction,
    Note,
    NoteKind,
    NoteOrigin,
    NoteStatus,
    NoteSupport,
    Operation,
    Project,
    Responsibility,
)
from .transactions import transaction


class RevisionConflict(ValueError):
    pass


class SQLiteCoordinationRepository:
    def __init__(
        self, db: sqlite3.Connection, pet: str, clock: Callable[[], float]
    ) -> None:
        self.db = db
        self.pet = pet
        self.clock = clock

    def _get[T](self, kind: str, identifier: str, record: type[T]) -> T | None:
        row = self.db.execute(
            "SELECT data FROM coordination_records WHERE pet=? AND kind=? AND id=?",
            (self.pet, kind, identifier),
        ).fetchone()
        return TypeAdapter(record).validate_json(row["data"]) if row else None

    def _all[T](self, kind: str, record: type[T]) -> tuple[T, ...]:
        rows = self.db.execute(
            "SELECT data FROM coordination_records WHERE pet=? AND kind=? ORDER BY rowid",
            (self.pet, kind),
        )
        adapter = TypeAdapter(record)
        return tuple(adapter.validate_json(row["data"]) for row in rows)

    def _write(self, kind: str, identifier: str, revision: int, record: object) -> None:
        data = TypeAdapter(type(record)).dump_json(record).decode()
        self.db.execute(
            """INSERT INTO coordination_records(pet,kind,id,revision,data)
            VALUES (?,?,?,?,?) ON CONFLICT(pet,kind,id) DO UPDATE SET
            revision=excluded.revision,data=excluded.data""",
            (self.pet, kind, identifier, revision, data),
        )
        self.db.execute(
            "INSERT INTO coordination_history(pet,kind,id,revision,data,recorded_at) VALUES (?,?,?,?,?,?)",
            (self.pet, kind, identifier, revision, data, self.clock()),
        )

    def state(self) -> CoordinationState:
        return (
            self._get("state", "coordination", CoordinationState) or CoordinationState()
        )

    def set_paused(self, paused: bool) -> CoordinationState:
        with transaction(self.db):
            state = self.state()
            if state.paused == paused:
                return state
            state = replace(state, revision=state.revision + 1, paused=paused)
            self._write("state", "coordination", state.revision, state)
            return state

    def conversation(self, identifier: str) -> Conversation | None:
        return self._get("conversation", identifier, Conversation)

    def ensure_conversation(self, identifier: str) -> Conversation:
        with transaction(self.db):
            conversation = self.conversation(identifier)
            if conversation is None:
                now = self.clock()
                conversation = Conversation(identifier, now, now)
                self._write("conversation", identifier, 1, conversation)
            return conversation

    def conversations(self) -> tuple[Conversation, ...]:
        return self._all("conversation", Conversation)

    def messages(self, conversation_id: str) -> tuple[ChatMessage, ...]:
        return tuple(
            message
            for message in self._all("message", ChatMessage)
            if message.conversation_id == conversation_id
        )

    def begin_turn(
        self, message: ChatMessage, expected_revision: int | None = None
    ) -> bool:
        if message.role != ChatRole.USER or not message.text.strip():
            raise ValueError("A user turn requires user prose")
        with transaction(self.db):
            if self._get("message", message.id, ChatMessage) is not None:
                return False
            conversation = self.ensure_conversation(message.conversation_id)
            if (
                expected_revision is not None
                and conversation.revision != expected_revision
            ):
                return False
            if conversation.pending_message_id is not None:
                return False
            self._write("message", message.id, 1, message)
            self._write(
                "conversation",
                conversation.id,
                conversation.revision + 1,
                replace(
                    conversation,
                    revision=conversation.revision + 1,
                    updated_at=self.clock(),
                    pending_message_id=message.id,
                    error=None,
                ),
            )
            return True

    def finish_turn(self, message: ChatMessage, expected_revision: int) -> bool:
        if message.role != ChatRole.ASSISTANT or not message.text.strip():
            raise ValueError("An assistant reply requires assistant prose")
        with transaction(self.db):
            conversation = self.conversation(message.conversation_id)
            if (
                conversation is None
                or conversation.revision != expected_revision
                or conversation.pending_message_id != message.reply_to
                or message.reply_to is None
                or self._get("message", message.id, ChatMessage) is not None
            ):
                return False
            self._write("message", message.id, 1, message)
            self._write(
                "conversation",
                conversation.id,
                conversation.revision + 1,
                replace(
                    conversation,
                    revision=conversation.revision + 1,
                    updated_at=self.clock(),
                    pending_message_id=None,
                    error=None,
                ),
            )
            return True

    def fail_turn(
        self,
        conversation_id: str,
        message_id: str,
        error: str,
        *,
        expected_revision: int | None = None,
    ) -> bool:
        with transaction(self.db):
            conversation = self.conversation(conversation_id)
            if conversation is None or conversation.pending_message_id != message_id:
                return False
            if (
                expected_revision is not None
                and conversation.revision != expected_revision
            ):
                return False
            failed = replace(
                conversation,
                revision=conversation.revision + 1,
                updated_at=self.clock(),
                pending_message_id=None,
                error=error,
            )
            self._write("conversation", conversation_id, failed.revision, failed)
            return True

    def retry_turn(
        self, conversation_id: str, message_id: str, command_id: str | None = None
    ) -> bool:
        with transaction(self.db):
            if command_id is not None and self._get(
                "chat_retry", command_id, ChatRetry
            ):
                return False
            conversation = self.conversation(conversation_id)
            messages = self.messages(conversation_id)
            if (
                conversation is None
                or conversation.pending_message_id is not None
                or conversation.error is None
                or not messages
                or messages[-1].id != message_id
                or messages[-1].role != ChatRole.USER
            ):
                return False
            pending = replace(
                conversation,
                revision=conversation.revision + 1,
                updated_at=self.clock(),
                pending_message_id=message_id,
                error=None,
            )
            if command_id is not None:
                self._write(
                    "chat_retry",
                    command_id,
                    1,
                    ChatRetry(command_id, conversation_id, message_id, self.clock()),
                )
            self._write("conversation", conversation_id, pending.revision, pending)
            return True

    def instruction(self, identifier: str) -> Instruction | None:
        return self._get("instruction", identifier, Instruction)

    def instructions(self) -> tuple[Instruction, ...]:
        return self._all("instruction", Instruction)

    def save_instruction(self, instruction: Instruction) -> None:
        if not instruction.text.strip():
            raise ValueError("A user instruction requires text")
        with transaction(self.db):
            previous = self.instruction(instruction.id)
            self._check_revision(
                previous.revision if previous else 0, instruction.revision
            )
            self._write(
                "instruction", instruction.id, instruction.revision, instruction
            )
            state = self.state()
            state = replace(
                state,
                revision=state.revision + 1,
                instruction_revision=state.instruction_revision + 1,
            )
            self._write("state", "coordination", state.revision, state)

    def note(self, identifier: str) -> Note | None:
        return self._get("note", identifier, Note)

    def notes(self, scope: str | None = None) -> tuple[Note, ...]:
        return tuple(
            note
            for note in self._all("note", Note)
            if scope is None or note.scope == scope
        )

    def note_history(self, identifier: str) -> tuple[Note, ...]:
        rows = self.db.execute(
            "SELECT data FROM coordination_history WHERE pet=? AND kind='note' AND id=? ORDER BY revision",
            (self.pet, identifier),
        )
        adapter = TypeAdapter(Note)
        return tuple(adapter.validate_json(row["data"]) for row in rows)

    def save_note(self, note: Note, expected_revision: int | None = None) -> None:
        with transaction(self.db):
            previous = self.note(note.id)
            revision = previous.revision if previous else 0
            if expected_revision is not None and revision != expected_revision:
                raise RevisionConflict("The note changed during consideration")
            self._check_revision(revision, note.revision)
            if previous is not None and (
                previous.origin != note.origin or previous.kind != note.kind
            ):
                raise ValueError("A note's origin and kind remain part of its identity")
            if previous is not None and note.created_at != previous.created_at:
                raise ValueError("A note retains its original creation date")
            if previous is not None and note.actor != previous.actor:
                raise ValueError("A note retains its originating actor")
            if (
                previous is not None
                and note.sources == previous.sources
                and (
                    note.observed_at != previous.observed_at
                    or note.original_at != previous.original_at
                )
            ):
                raise ValueError("Repeated notes cannot refresh evidence dates")
            self._validate_note(note)
            self._write("note", note.id, note.revision, note)

    def _validate_note(self, note: Note) -> None:
        if not note.text.strip():
            raise ValueError("A note requires text")
        if note.origin == NoteOrigin.USER:
            instruction = self.instruction(note.instruction_id or "")
            if (
                instruction is None
                or instruction.text != note.text
                or instruction.scope != note.scope
            ):
                raise ValueError("A user note requires its canonical instruction")
            if instruction.active and note.status not in (
                NoteStatus.ACTIVE,
                NoteStatus.NEEDS_RECHECK,
            ):
                raise ValueError("An active user instruction remains applicable")
            if note.support != NoteSupport.CONFIRMED:
                raise ValueError("A direct user statement retains confirmed origin")
        elif note.kind in (NoteKind.INSTRUCTION, NoteKind.CORRECTION):
            raise ValueError(
                "Imported and derived notes cannot supply user instructions"
            )
        if note.origin == NoteOrigin.HARNESS and note.kind != NoteKind.REPORT:
            raise ValueError("A harness statement remains a report")
        if note.origin == NoteOrigin.HARNESS and note.support != NoteSupport.REPORTED:
            raise ValueError("A harness report retains reported support")
        if note.kind == NoteKind.HYPOTHESIS and note.support != NoteSupport.TENTATIVE:
            raise ValueError("A hypothesis remains tentative")
        if (
            note.kind in (NoteKind.DECISION, NoteKind.OBSERVATION, NoteKind.SYNTHESIS)
            and not note.sources
        ):
            raise ValueError("A supported note requires leaf source references")

    def projects(self) -> tuple[Project, ...]:
        return self._all("project", Project)

    def save_project(self, project: Project) -> None:
        with transaction(self.db):
            previous = self._get("project", project.id, Project)
            self._check_revision(previous.revision if previous else 0, project.revision)
            self._write("project", project.id, project.revision, project)

    def responsibilities(self) -> tuple[Responsibility, ...]:
        return self._all("responsibility", Responsibility)

    def responsibility(self, identifier: str) -> Responsibility | None:
        return self._get("responsibility", identifier, Responsibility)

    def save_responsibility(self, responsibility: Responsibility) -> None:
        with transaction(self.db):
            instruction = self.instruction(responsibility.instruction_id)
            if instruction is None or not instruction.active:
                raise ValueError(
                    "Responsibility adoption requires an active user instruction"
                )
            previous = self.responsibility(responsibility.id)
            self._check_revision(
                previous.revision if previous else 0, responsibility.revision
            )
            self._write(
                "responsibility",
                responsibility.id,
                responsibility.revision,
                responsibility,
            )

    def activity(self) -> tuple[ActivityRecord, ...]:
        return self._all("activity", ActivityRecord)

    def record_activity(self, activity: ActivityRecord) -> bool:
        with transaction(self.db):
            if self._get("activity", activity.id, ActivityRecord) is not None:
                return False
            self._write("activity", activity.id, 1, activity)
            return True

    def tasks(self) -> tuple[DelegatedTask, ...]:
        return self._all("task", DelegatedTask)

    def task(self, identifier: str) -> DelegatedTask | None:
        return self._get("task", identifier, DelegatedTask)

    def save_task(self, task: DelegatedTask) -> None:
        with transaction(self.db):
            responsibility = self.responsibility(task.responsibility_id)
            if responsibility is None:
                raise ValueError("A task requires its adopted responsibility")
            previous = self.task(task.id)
            self._check_revision(previous.revision if previous else 0, task.revision)
            self._write("task", task.id, task.revision, task)

    def operations(self) -> tuple[Operation, ...]:
        return self._all("operation", Operation)

    def operation(self, identifier: str) -> Operation | None:
        return self._get("operation", identifier, Operation)

    def save_operation(self, operation: Operation) -> None:
        with transaction(self.db):
            if self.task(operation.task_id) is None:
                raise ValueError("An operation requires its delegated task")
            previous = self.operation(operation.id)
            self._check_revision(
                previous.revision if previous else 0, operation.revision
            )
            self._write("operation", operation.id, operation.revision, operation)

    def candidates(self) -> tuple[Candidate, ...]:
        return self._all("candidate", Candidate)

    def candidate(self, identifier: str) -> Candidate | None:
        return self._get("candidate", identifier, Candidate)

    def save_candidate(self, candidate: Candidate) -> None:
        with transaction(self.db):
            previous = self.candidate(candidate.id)
            self._check_revision(
                previous.revision if previous else 0, candidate.revision
            )
            self._write("candidate", candidate.id, candidate.revision, candidate)

    def considerations(
        self, candidate_id: str | None = None
    ) -> tuple[Consideration, ...]:
        return tuple(
            record
            for record in self._all("consideration", Consideration)
            if candidate_id is None or record.candidate_id == candidate_id
        )

    def record_consideration(self, record: Consideration) -> bool:
        with transaction(self.db):
            if self._get("consideration", record.id, Consideration) is not None:
                return False
            candidate = self.candidate(record.candidate_id)
            if candidate is None or candidate.revision != record.candidate_revision:
                raise RevisionConflict("The candidate changed during consideration")
            if self.state().instruction_revision != record.instruction_revision:
                raise RevisionConflict("Instructions changed during consideration")
            self._write("consideration", record.id, 1, record)
            updated = replace(
                candidate,
                revision=candidate.revision + 1,
                last_serviced_at=record.serviced_at or candidate.last_serviced_at,
                next_review_at=record.next_review_at,
                waiting_on=record.wake_condition,
            )
            self._write("candidate", updated.id, updated.revision, updated)
            return True

    def clear(self) -> None:
        with transaction(self.db):
            for table in ("coordination_records", "coordination_history"):
                self.db.execute(f"DELETE FROM {table} WHERE pet=?", (self.pet,))

    @staticmethod
    def _check_revision(previous: int, revision: int) -> None:
        if revision != previous + 1:
            raise RevisionConflict("A revision must follow the current record")
