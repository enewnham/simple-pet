import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import Literal

from claude_agent_sdk import SdkMcpTool, tool
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from ..coordination.models import (
    Note,
    NoteKind,
    NoteOrigin,
    NoteStatus,
    NoteSupport,
    SourceReference,
)
from ..storage.memory import Memory
from .context import WorkBrief, WorkContext
from .providers import Passage, SourceResult
from .retrieval import ContextRetrieval, SearchContext

DIRECT_SOURCE_KINDS = {
    "observation",
    "meeting_export",
    "ticket",
    "mcp_result",
    "user_instruction",
}


@dataclass(frozen=True)
class ToolAuthority:
    instruction_revision: int
    conversation_id: str | None = None
    user_message_id: str | None = None
    pass_id: str | None = None
    candidates: tuple[tuple[str, int], ...] = ()
    responsibilities: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class WorkflowTool:
    name: str
    description: str
    input_schema: dict[str, object]
    handler: Callable[[dict[str, object], ToolAuthority], Awaitable[object]]


class ToolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SearchInput(ToolInput):
    question: str = Field(min_length=1)


class PassageInput(ToolInput):
    source_id: str


class ChangesInput(ToolInput):
    checkpoint: int = Field(default=0, ge=0)


class RecordNoteInput(ToolInput):
    kind: Literal["hypothesis", "synthesis"]
    text: str = Field(min_length=1)
    source_ids: tuple[str, ...] = ()
    scope: str = ""
    supersedes: tuple[str, ...] = ()


class ReviseNoteInput(ToolInput):
    id: str
    expected_revision: int = Field(gt=0)
    operation: Literal["revise", "recheck", "refute", "supersede", "archive"]
    text: str | None = None
    source_ids: tuple[str, ...] = ()
    replacement_id: str | None = None


def encoded(value: object) -> str:
    return TypeAdapter(type(value)).dump_json(value).decode()


class CompanionTools:
    def __init__(
        self,
        memory: Memory,
        context: WorkContext,
        retrieval: ContextRetrieval,
        brief: WorkBrief,
        search_context: SearchContext,
        clock: Callable[[], float],
        guard: Callable[[], None],
        workflow: tuple[WorkflowTool, ...] = (),
        authority: ToolAuthority | None = None,
    ) -> None:
        self.memory = memory
        self.context = context
        self.retrieval = retrieval
        self.brief = brief
        self.search_context = search_context
        self.clock = clock
        self.guard = guard
        self.workflow = workflow
        self.authority = authority or ToolAuthority(brief.instruction_revision)
        self.sources = {
            document.source.id: document.source for document in brief.recent_work
        }
        self.source_kinds = {
            document.source.id: document.kind for document in brief.recent_work
        }
        for note in brief.notes:
            sources = context.note_sources(note)
            for source in sources:
                self.sources[source.id] = source
                self.source_kinds[source.id] = self.source_kind(source)
                if source.provider == "user":
                    self.sources[f"notes:{note.id}"] = source
                    self.source_kinds[f"notes:{note.id}"] = "user_instruction"
        for record in (*brief.projects, *brief.responsibilities, *brief.tasks):
            for source in record.sources:
                self.sources[source.id] = source
                self.source_kinds[source.id] = self.source_kind(source)

    def source_kind(self, source: SourceReference) -> str:
        if source.provider == "user":
            return "user_instruction"
        document = self.retrieval.index.document(
            source.id.split("#L", 1)[0], source.revision
        )
        return document.kind if document is not None else "retained_evidence"

    def remember(self, passage: Passage) -> None:
        self.sources[passage.source.id] = passage.source
        self.source_kinds[passage.source.id] = passage.kind
        document = self.retrieval.index.document(
            passage.document_id, passage.source.revision
        )
        if document is not None:
            self.sources[document.source.id] = replace(
                document.source, available=passage.source.available
            )
            self.source_kinds[document.source.id] = passage.kind
        for source in passage.leaf_sources:
            self.sources[source.id] = source
            self.source_kinds[source.id] = self.source_kind(source)
            if source.provider == "user":
                self.sources[passage.source.id] = source
                self.sources[passage.document_id] = source
                self.source_kinds[passage.source.id] = "user_instruction"
                self.source_kinds[passage.document_id] = "user_instruction"

    def is_current(self, source: SourceReference) -> bool:
        if source.provider == "user":
            instruction = self.memory.coordination.instruction(source.id)
            return (
                instruction is not None
                and instruction.active
                and str(instruction.revision) == source.revision
            )
        return self.retrieval.index.current(source)

    def references(self, identifiers: tuple[str, ...]) -> tuple[SourceReference, ...]:
        if any(identifier not in self.sources for identifier in identifiers):
            raise ValueError(
                "A source reference must come from supplied or retrieved evidence"
            )
        return tuple(
            dict.fromkeys(self.sources[identifier] for identifier in identifiers)
        )

    def leaf_references(
        self, references: tuple[SourceReference, ...]
    ) -> tuple[SourceReference, ...]:
        if any(source.provider == "notes" for source in references):
            raise ValueError(
                "Follow a note's leaf evidence instead of citing the note as independent support"
            )
        return references

    async def dispatch(self, name: str, arguments: dict[str, object]) -> object:
        self.guard()
        match name:
            case "current_work_context":
                ToolInput.model_validate(arguments)
                return self.context.brief(
                    self.brief.purpose, self.brief.scope, self.brief.checkpoint
                )
            case "context_search":
                request = SearchInput.model_validate(arguments)
                result = await self.retrieval.search(
                    request.question, self.search_context
                )
                self.guard()
                for hit in result.hits:
                    self.remember(hit.passage)
                return result
            case "read_passage":
                request = PassageInput.model_validate(arguments)
                reference = self.references((request.source_id,))[0]
                result = await self.retrieval.read_passage(reference)
                self.guard()
                for passage in result:
                    self.remember(passage)
                return result
            case "relevant_changes":
                request = ChangesInput.model_validate(arguments)
                return self.retrieval.changes(request.checkpoint)
            case "record_note":
                return self._record(RecordNoteInput.model_validate(arguments))
            case "revise_note":
                return self._revise(ReviseNoteInput.model_validate(arguments))
        selected = next((item for item in self.workflow if item.name == name), None)
        if selected is None:
            raise ValueError("The tool is unavailable")
        result = await selected.handler(arguments, self.authority)
        self.guard()
        if isinstance(result, SourceResult):
            self.retrieval.index.capture(result)
            for document in result.documents:
                self.sources[document.source.id] = document.source
                self.source_kinds[document.source.id] = document.kind
        return result

    def _record(self, request: RecordNoteInput) -> Note:
        references = self.leaf_references(self.references(request.source_ids))
        if request.kind == "synthesis" and not references:
            raise ValueError("A synthesis requires underlying leaf evidence")
        if any(not self.is_current(reference) for reference in references):
            raise ValueError(
                "Recheck changed or unavailable evidence before recording a finding"
            )
        now = self.clock()
        for identifier in request.supersedes:
            previous = self.memory.coordination.note(identifier)
            if previous is None or previous.origin != NoteOrigin.AGENT:
                raise ValueError(
                    "Only an existing agent note can be superseded by the agent"
                )
        note = Note(
            id=str(uuid.uuid4()),
            origin=NoteOrigin.AGENT,
            kind=NoteKind(request.kind),
            text=request.text,
            scope=request.scope or self.brief.scope,
            created_at=now,
            observed_at=min((source.observed_at for source in references), default=now),
            sources=references,
            support=NoteSupport.TENTATIVE,
            actor="companion",
            supersedes=request.supersedes,
        )
        self.memory.coordination.save_note(note)
        return note

    def _revise(self, request: ReviseNoteInput) -> Note:
        previous = self.memory.coordination.note(request.id)
        if previous is None or previous.origin != NoteOrigin.AGENT:
            raise ValueError("The agent can revise only its own derived notes")
        if previous.revision != request.expected_revision:
            raise ValueError("The note changed during consideration")
        references = self.leaf_references(
            self.references(request.source_ids)
            if request.source_ids
            else previous.sources
        )
        status = previous.status
        checked_at = previous.checked_at
        if request.operation == "recheck":
            if not references or any(
                not self.is_current(source) for source in references
            ):
                raise ValueError("A recheck requires current underlying evidence")
            checked_at = self.clock()
            status = NoteStatus.ACTIVE
        elif request.operation == "refute":
            if not request.source_ids:
                raise ValueError("Refutation requires contrary evidence")
            status = NoteStatus.REJECTED
        elif request.operation == "archive":
            status = NoteStatus.ARCHIVED
        elif request.operation == "supersede":
            replacement = self.memory.coordination.note(request.replacement_id or "")
            if replacement is None or previous.id not in replacement.supersedes:
                raise ValueError(
                    "Supersession requires a recorded replacement and lineage"
                )
            status = NoteStatus.SUPERSEDED
        updated = replace(
            previous,
            text=request.text if request.text is not None else previous.text,
            sources=references,
            status=status,
            checked_at=checked_at,
            revision=previous.revision + 1,
        )
        self.memory.coordination.save_note(updated, expected_revision=previous.revision)
        return updated

    def definitions(self) -> list[SdkMcpTool[dict[str, object]]]:
        specifications: list[tuple[str, str, type[BaseModel] | dict[str, object]]] = [
            (
                "current_work_context",
                "Read the current work brief, scoped instructions, corrections, active and deferred work, source map, freshness, and unknowns.",
                ToolInput,
            ),
            (
                "context_search",
                "Search enabled sources and notes for a focused question. Expand aliases and references, return ranked cited passages and explicit source failures. Follow relevant references beyond the initial brief.",
                SearchInput,
            ),
            (
                "read_passage",
                "Read a retrieved source with neighboring passages and its original provenance. Use this to understand a decision in context.",
                PassageInput,
            ),
            (
                "relevant_changes",
                "Read source changes since a checkpoint, including edits and removals. An empty result means no recorded changes.",
                ChangesInput,
            ),
            (
                "record_note",
                "Record a tentative hypothesis or synthesis with underlying evidence IDs. This never creates user instructions, ownership, or authority.",
                RecordNoteInput,
            ),
            (
                "revise_note",
                "Revise, recheck, refute, supersede, or archive an agent note while preserving origin, support, evidence dates, and lineage.",
                ReviseNoteInput,
            ),
        ]
        specifications.extend(
            (item.name, item.description, item.input_schema) for item in self.workflow
        )
        definitions = []
        for name, description, schema in specifications:

            async def invoke(
                arguments: dict[str, object], selected: str = name
            ) -> dict[str, object]:
                try:
                    value = await self.dispatch(selected, arguments)
                    return {"content": [{"type": "text", "text": encoded(value)}]}
                except (ValueError, OSError, RuntimeError, TimeoutError) as error:
                    return {
                        "isError": True,
                        "content": [{"type": "text", "text": str(error)}],
                    }

            input_schema = (
                schema.model_json_schema() if isinstance(schema, type) else schema
            )
            definitions.append(tool(name, description, input_schema)(invoke))
        return definitions
