import asyncio
import json
import socket
import sys
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
from mcp import ClientSession
from mcp.types import CallToolResult, ListToolsResult, Tool, ToolAnnotations

from pet_daemon.companion.config import (
    CoordinatorConfig,
    HttpMcp,
    LinearConfig,
    MeetingConfig,
    NativeHistoryConfig,
    RetrievalConfig,
    SelectedMcp,
    StdioMcp,
)
from pet_daemon.companion.context import WorkContext
from pet_daemon.companion.index import SourceChange, SourceIndex
from pet_daemon.companion.mcp import McpGateway, McpSource
from pet_daemon.companion.providers import (
    Availability,
    Document,
    SourceResult,
    SourceStatus,
    WorkHistory,
)
from pet_daemon.companion.retrieval import (
    ContextRetrieval,
    expanded_queries,
)
from pet_daemon.companion.sdk import SdkCompanion
from pet_daemon.companion.tickets import (
    LinearTickets,
    McpLinearTickets,
    TicketCapability,
    TicketSource,
)
from pet_daemon.companion.tools import CompanionTools
from pet_daemon.config import Config
from pet_daemon.coordination.models import (
    ChatMessage,
    ChatRole,
    Note,
    NoteKind,
    NoteOrigin,
    NoteSupport,
    Project,
    SourceReference,
)
from pet_daemon.daemon import Daemon
from pet_daemon.storage import Memory
from pet_daemon.work.observations import CheckoutObservation


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def mcp_selection(*tools: str, **options: object) -> StdioMcp:
    return StdioMcp.model_validate(
        {
            "name": "fixture",
            "read_only": True,
            "command": sys.executable,
            "args": [str(Path(__file__).parent / "fixtures" / "mcp_source.py")],
            "tools": tools,
            **options,
        }
    )


@pytest.mark.anyio
async def test_real_stdio_mcp_exposes_only_selected_read_only_operations_and_retains_query(
    tmp_path: Path,
) -> None:
    memory = Memory(tmp_path / "memory.sqlite3", "pet", clock=lambda: 100)
    index = SourceIndex(memory.db, memory.pet)
    selection = mcp_selection("search", search_tool="search")
    gateway = McpGateway((selection,), lambda: 100, 10, cached=index.document)
    statuses = await gateway.discover()
    assert statuses[0].availability == Availability.AVAILABLE
    definitions = gateway.tools()
    assert len(definitions) == 1
    assert "search" in definitions[0].description
    assert "write" not in definitions[0].description
    result = await gateway.fetch("fixture", "search", {"query": "rollback"})
    assert "rollback" in result.documents[0].text
    index.capture(result)
    with pytest.raises(ValueError, match="explicitly selected"):
        await gateway.fetch("fixture", "write", {})
    restored = McpGateway((selection,), lambda: 200, 10, cached=index.document)
    source = McpSource(restored, "fixture")
    reread = await source.read(result.documents[0].source.id)
    assert reread.documents[0].source.id == result.documents[0].source.id
    assert reread.documents[0].source.revision == result.documents[0].source.revision
    memory.close()


@pytest.mark.anyio
@pytest.mark.parametrize("name", ["write", "undeclared", "contradictory"])
async def test_mcp_rejects_untrusted_effect_declarations(name: str) -> None:
    gateway = McpGateway((mcp_selection(name),), lambda: 100, 10)
    statuses = await gateway.discover()
    assert statuses[0].availability == Availability.FAILED
    assert gateway.tools() == ()
    assert "read-only" in statuses[0].detail or "destructive" in statuses[0].detail


@pytest.mark.anyio
async def test_real_http_mcp_uses_the_selected_read_transport() -> None:
    with socket.socket() as bound:
        bound.bind(("127.0.0.1", 0))
        port = bound.getsockname()[1]
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        str(Path(__file__).parent / "fixtures" / "mcp_source.py"),
        str(port),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        async with asyncio.timeout(10):
            while True:
                try:
                    _, writer = await asyncio.open_connection("127.0.0.1", port)
                    writer.close()
                    await writer.wait_closed()
                    break
                except OSError:
                    await asyncio.sleep(0.02)
        selection = HttpMcp(
            name="fixture",
            read_only=True,
            tools=("search",),
            url=f"http://127.0.0.1:{port}/mcp",
        )
        gateway = McpGateway((selection,), lambda: 100, 10)
        assert (await gateway.discover())[0].availability == Availability.AVAILABLE
        result = await gateway.fetch("fixture", "search", {"query": "owner"})
        assert result.status.availability == Availability.AVAILABLE
        assert "owner" in result.documents[0].text
    finally:
        process.terminate()
        await process.wait()


@pytest.mark.anyio
async def test_linear_api_and_mcp_share_identity_revision_and_explicit_capabilities() -> (
    None
):
    selection = mcp_selection(
        "search", provider="linear", search_tool="search", lookup_tool="search"
    )
    gateway = McpGateway((selection,), lambda: 100, 10)
    await gateway.discover()
    mcp_tickets = McpLinearTickets(
        selection, gateway.fetch, lambda: 100, gateway.statuses
    )
    mcp_result = await mcp_tickets.lookup("fixture-issue")
    assert mcp_result.status.availability == Availability.AVAILABLE
    assert TicketCapability.LOOKUP in mcp_result.capabilities
    captured: list[tuple[str, dict[str, object]]] = []
    payload = json.loads(
        (await gateway.fetch("fixture", "search", {})).documents[0].text
    )
    issue = {
        "id": "fixture-issue",
        "identifier": payload["identifier"],
        "title": payload["title"],
        "description": payload["body"],
        "url": "https://linear.app/fixture/issue/PET-123",
        "updatedAt": "2026-11-05T12:00:00Z",
        "state": {"name": "In Progress", "type": "started"},
        "assignee": None,
        "project": None,
        "team": {"key": "PET"},
    }

    async def request(query: str, variables: dict[str, object]) -> dict[str, object]:
        captured.append((query, variables))
        assert query.startswith("query ")
        return {"data": {"issue": issue}}

    api = LinearTickets(
        LinearConfig(enabled=True, linked_ids=("fixture-issue",)),
        lambda: 200,
        request=request,
    )
    api_result = await api.lookup("fixture-issue")
    assert api_result.tickets[0].id == mcp_result.tickets[0].id
    assert api_result.tickets[0].revision == mcp_result.tickets[0].revision
    assert api_result.tickets[0].project is None
    assert TicketCapability.PROJECT in api_result.capabilities
    source = TicketSource(api)
    result = await source.read("linear:fixture-issue")
    assert result.documents[0].source.id == "linear:fixture-issue"
    assert captured[-1][1] == {"id": "fixture-issue"}
    assert "project" in result.status.capabilities
    unsupported = McpLinearTickets(
        mcp_selection("search", provider="linear", search_tool="search"),
        gateway.fetch,
        lambda: 100,
    )
    missing_capability = await unsupported.lookup("fixture-issue")
    assert missing_capability.status.availability == Availability.UNAVAILABLE
    assert TicketCapability.LOOKUP not in missing_capability.capabilities


@pytest.mark.anyio
async def test_linear_failures_are_distinct_from_empty_scoped_results() -> None:
    calls: list[dict[str, object]] = []

    async def empty(query: str, variables: dict[str, object]) -> dict[str, object]:
        calls.append(variables)
        return {"data": {"issues": {"nodes": [], "pageInfo": {"hasNextPage": False}}}}

    tickets = LinearTickets(
        LinearConfig(enabled=True, assignee_ids=(str(uuid.uuid4()),)),
        lambda: 100,
        request=empty,
    )
    result = await tickets.search_tickets("rollback", 10)
    assert result.status.availability == Availability.AVAILABLE
    assert result.tickets == ()
    scoped_filter = calls[0]["filter"]
    assert isinstance(scoped_filter, dict)
    assert "assignee" in scoped_filter

    async def failed(query: str, variables: dict[str, object]) -> dict[str, object]:
        return {
            "data": {"issues": {"nodes": []}},
            "errors": [{"message": "permission denied"}],
        }

    tickets.request = failed
    failure = await tickets.search_tickets("rollback", 10)
    assert failure.status.availability == Availability.FAILED
    assert failure.tickets == ()
    unconfigured = LinearTickets(LinearConfig(enabled=True), lambda: 100, request=empty)
    assert (
        await unconfigured.search_tickets("rollback", 10)
    ).status.availability == Availability.UNAVAILABLE


@pytest.mark.anyio
async def test_full_meeting_refresh_reports_edits_removal_and_wake_reconciliation(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "meetings"
    folder.mkdir()
    export = folder / "migration.md"
    export.write_text("# Migration\n\nProposal: choose a rollback owner.")
    memory = Memory(tmp_path / "memory.sqlite3", "pet", clock=lambda: 100)
    changes: list[SourceChange] = []
    config = CoordinatorConfig(
        working_directory=tmp_path / "sdk", meetings=MeetingConfig(folders=(folder,))
    )
    companion = SdkCompanion(
        config, memory, lambda: 100, on_source_changes=changes.extend
    )
    await companion.refresh_sources()
    document = next(
        document
        for document in companion.index.documents()
        if document.source.provider == "meetings"
    )
    assert document.source.original_at is None
    assert "Proposal" in document.text
    assert companion.index.current(document.source, now=100, max_age=10)
    export.write_text("# Migration\n\nConfirmed: Ari owns rollback.")
    assert await companion.refresh_sources() == ()
    await companion.suspend()
    await companion.resume()
    assert any(
        change.kind == "updated" and change.source_id == document.source.id
        for change in changes
    )
    assert not companion.index.current(document.source)
    export.unlink()
    await companion.refresh_sources(force=True)
    assert changes[-1].kind == "removed"
    assert companion.index.document(document.source.id) is None
    assert (
        companion.index.document(document.source.id, document.source.revision)
        is not None
    )
    await companion.close()
    memory.close()


@pytest.mark.anyio
async def test_native_history_bootstrap_is_bounded_and_never_adopts_external_work(
    tmp_path: Path,
) -> None:
    source = tmp_path / "session.jsonl"
    records = [
        {
            "timestamp": "2026-11-05T12:00:00Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Prepare the migration"}],
            },
        },
        {
            "timestamp": "2026-11-05T12:01:00Z",
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Migration is waiting on rollback ownership.",
                    }
                ],
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "name": "Bash",
                "arguments": "private tool output must stay out",
            },
        },
    ]
    source.write_text("\n".join(json.dumps(record) for record in records))
    memory = Memory(tmp_path / "memory.sqlite3", "pet", clock=lambda: 100)
    config = CoordinatorConfig(
        working_directory=tmp_path / "sdk",
        history=(
            NativeHistoryConfig(
                name="codex", path=source, format="codex", maximum_records=1
            ),
        ),
    )
    companion = SdkCompanion(config, memory, lambda: 100)
    await companion.refresh_sources()
    brief = companion.context.brief("migration rollback")
    assert len(brief.recent_work) == 1
    assert brief.recent_work[0].kind == "harness_history_report"
    assert "rollback ownership" in brief.recent_work[0].text
    assert "private tool" not in str(brief)
    assert brief.recent_work[0].source.original_at is not None
    assert memory.coordination.tasks() == ()
    assert memory.coordination.responsibilities() == ()
    assert memory.coordination.instructions() == ()
    await companion.close()
    memory.close()


@pytest.mark.anyio
async def test_scoped_corrections_apply_before_alias_expansion_and_diverse_ranking(
    tmp_path: Path,
) -> None:
    memory = Memory(tmp_path / "memory.sqlite3", "pet", clock=lambda: 100)
    user = ChatMessage(
        str(uuid.uuid4()),
        str(uuid.uuid4()),
        ChatRole.USER,
        "That ticket is unrelated to migration.",
        100,
    )
    assert memory.coordination.begin_turn(user)
    excluded = "https://linear.app/fixture/issue/PET-123"
    memory.coordination.save_note(
        Note(
            str(uuid.uuid4()),
            NoteOrigin.USER,
            NoteKind.CORRECTION,
            user.text,
            "migration",
            100,
            100,
            instruction_id="user:" + user.id,
            support=NoteSupport.CONFIRMED,
            actor="user",
            excluded_references=(excluded,),
        )
    )
    memory.coordination.save_project(
        Project(
            str(uuid.uuid4()), "Migration", "migration", aliases=("rollback", excluded)
        )
    )
    index = SourceIndex(memory.db, memory.pet)
    context = WorkContext(memory, index, RetrievalConfig(results=2), lambda: 100)
    search_context = context.search_context(context.brief("rollback", "migration"))
    assert excluded not in expanded_queries("rollback", search_context, 10)
    documents = (
        Document(
            SourceReference(
                str(uuid.uuid4()), "meetings", "first", 90, scope="migration"
            ),
            "Older decision",
            "Decision: confirm rollback ownership because deployment needs it.",
        ),
        Document(
            SourceReference(
                str(uuid.uuid4()), "meetings", "second", 100, scope="migration"
            ),
            "Status",
            "Rollback status update.",
        ),
        Document(
            SourceReference(str(uuid.uuid4()), "work", "third", 100, scope="migration"),
            "Harness report",
            "Rollback progress is reported.",
        ),
        Document(
            SourceReference(excluded, "work", "fourth", 100, scope="migration"),
            "Unrelated ticket",
            "Rollback rollback rollback.",
        ),
    )
    providers = tuple(
        WorkHistory(
            lambda provider=provider: tuple(
                document
                for document in documents
                if document.source.provider == provider
            ),
            lambda: 100,
            name=provider,
        )
        for provider in ("meetings", "work")
    )
    retrieval = ContextRetrieval(
        index,
        providers,
        RetrievalConfig(results=2),
        lambda: 100,
        semantic_score=lambda question, text: 1 if "Decision:" in text else 0,
    )
    result = await retrieval.search("rollback ownership", search_context)
    assert result.hits[0].passage.title == "Older decision"
    assert {hit.passage.source.provider for hit in result.hits} == {"meetings", "work"}
    assert all(excluded not in hit.passage.document_id for hit in result.hits)
    assert result.corrections == (user.text,)
    memory.close()


@pytest.mark.anyio
async def test_selected_mcp_operation_registers_evidence_at_the_sdk_tool_boundary(
    tmp_path: Path,
) -> None:
    selection = mcp_selection("search")
    memory = Memory(tmp_path / "memory.sqlite3", "pet", clock=lambda: 100)
    companion = SdkCompanion(
        CoordinatorConfig(working_directory=tmp_path / "sdk", mcp=(selection,)),
        memory,
        lambda: 100,
    )
    await companion.refresh_sources()
    brief = companion.context.brief("rollback")
    tools = CompanionTools(
        memory,
        companion.context,
        companion.retrieval,
        brief,
        companion.context.search_context(brief),
        lambda: 100,
        lambda: None,
        companion.gateway.tools(),
    )
    selected = companion.gateway.tools()[0]
    result = await asyncio.create_task(
        tools.dispatch(selected.name, {"query": "rollback"})
    )
    assert isinstance(result, SourceResult)
    source = result.documents[0].source
    assert tools.references((source.id,)) == (source,)
    note = await tools.dispatch(
        "record_note",
        {
            "kind": "synthesis",
            "text": "The source retains a rollback ownership decision.",
            "source_ids": [source.id],
        },
    )
    assert isinstance(note, Note)
    assert note.origin == NoteOrigin.AGENT
    assert note.sources == (source,)
    assert note.support == NoteSupport.TENTATIVE
    names = {definition.name for definition in tools.definitions()}
    assert selected.name in names
    assert "write" not in names
    await companion.close()
    memory.close()


@pytest.mark.anyio
async def test_unavailable_mcp_is_a_failed_source_and_wildcards_are_rejected() -> None:
    missing = mcp_selection("search").model_copy(
        update={"command": "/nonexistent/companion-mcp-fixture"}
    )
    gateway = McpGateway((missing,), lambda: 100, 10)
    statuses = await gateway.discover()
    assert statuses[0].availability == Availability.FAILED
    assert statuses[0].detail
    assert gateway.tools() == ()
    with pytest.raises(ValueError, match="wildcards"):
        mcp_selection("search*")


@pytest.mark.anyio
async def test_mcp_sessions_cancel_in_their_owning_task_and_eof_remains_a_source_failure() -> (
    None
):
    selection = mcp_selection("search", "slow")
    gateway = McpGateway((selection,), lambda: 100, 10)
    started = asyncio.Event()

    async def reading() -> None:
        async with gateway.session(selection) as session:
            started.set()
            await session.call_tool("slow", {"seconds": 30})

    task = asyncio.create_task(reading())
    await started.wait()
    task.cancel()
    async with asyncio.timeout(5):
        with pytest.raises(asyncio.CancelledError):
            await task
    result = await asyncio.create_task(
        gateway.fetch("fixture", "search", {"query": "rollback"})
    )
    assert result.status.availability == Availability.AVAILABLE
    eof = selection.model_copy(update={"args": (*selection.args, "eof")})
    failed = await McpGateway((eof,), lambda: 100, 10).discover()
    assert failed[0].availability == Availability.FAILED
    assert failed[0].detail


def test_refreshing_an_unrelated_document_does_not_extend_evidence_freshness(
    tmp_path: Path,
) -> None:
    memory = Memory(tmp_path / "memory.sqlite3", "pet", clock=lambda: 100)
    index = SourceIndex(memory.db, memory.pet)
    first = Document(
        SourceReference(str(uuid.uuid4()), "meetings", "first", 100),
        "Decision",
        "Retain source scope.",
    )
    second = Document(
        SourceReference(str(uuid.uuid4()), "meetings", "second", 500),
        "Update",
        "Different work changed.",
    )
    index.capture(
        SourceResult((first,), SourceStatus("meetings", Availability.AVAILABLE, 100))
    )
    index.capture(
        SourceResult((second,), SourceStatus("meetings", Availability.AVAILABLE, 500))
    )
    assert not index.current(first.source, now=500, max_age=300)
    assert index.current(second.source, now=500, max_age=300)
    index.capture(
        SourceResult(
            (replace(first, checked_at=600),),
            SourceStatus("meetings", Availability.AVAILABLE, 600),
        )
    )
    retained = index.document(first.source.id)
    assert retained is not None
    assert retained.source.observed_at == 100
    assert retained.checked_at == 600
    assert index.current(first.source, now=600, max_age=300)
    memory.close()


@pytest.mark.anyio
async def test_cached_operational_reads_keep_external_check_time_at_the_daemon_effect_guard(
    tmp_path: Path,
) -> None:
    now = 10000.0
    memory = Memory(tmp_path / "memory.sqlite3", "pet", clock=lambda: now)
    config = CoordinatorConfig(working_directory=tmp_path / "sdk")
    companion = SdkCompanion(config, memory, lambda: now)
    daemon = object.__new__(Daemon)
    daemon.companion = companion
    daemon.coordination_store = memory.coordination
    daemon.config = Config(pet_dir=tmp_path, coordinator=config)
    daemon.wall_clock = lambda: now
    observation = CheckoutObservation(worktree="/project", local=None, checked_at=100)
    memory.work.observe(observation)
    await companion.refresh_sources()
    cached = next(
        document
        for document in companion.index.documents()
        if document.kind == "observation"
    )
    assert cached.checked_at == 100
    assert not daemon.sources_current((cached.source,))
    result = await companion.retrieval.search("Repository observation")
    hit = next(hit for hit in result.hits if hit.passage.kind == "observation")
    expanded = await companion.retrieval.read_passage(hit.passage.source)
    assert expanded[0].checked_at == 100
    assert not daemon.sources_current((hit.passage.source,))
    historical = Document(
        SourceReference(str(uuid.uuid4()), "meetings", "decision", 100),
        "Older decision",
        "The accepted design retains source scope.",
        kind="meeting_export",
        checked_at=100,
    )
    companion.index.capture(
        SourceResult(
            (historical,), SourceStatus("meetings", Availability.AVAILABLE, now)
        )
    )
    assert daemon.sources_current((historical.source,))
    memory.work.observe(observation.model_copy(update={"checked_at": now}))
    await companion.refresh_sources(force=True)
    assert daemon.sources_current((cached.source,))
    fresh = companion.index.document(cached.source.id)
    assert fresh is not None
    assert fresh.checked_at == now
    assert fresh.source.observed_at == 100
    memory.work.observe(
        observation.model_copy(
            update={"checked_at": now, "error": "External refresh failed"}
        )
    )
    await companion.refresh_sources(force=True)
    failed = companion.index.document(cached.source.id)
    assert failed is not None
    assert not failed.source.available
    assert not daemon.sources_current((failed.source,))
    await companion.close()
    memory.close()


class LinearSession:
    def __init__(self) -> None:
        self.payload: dict[str, object] = {}
        self.failed = False

    async def list_tools(self, *, params: object = None) -> ListToolsResult:
        return ListToolsResult(
            tools=[
                Tool(
                    name=name,
                    input_schema={"type": "object"},
                    annotations=ToolAnnotations(
                        read_only_hint=True, destructive_hint=False
                    ),
                )
                for name in ("lookup", "search")
            ]
        )

    async def call_tool(
        self, name: str, arguments: dict[str, object] | None = None
    ) -> CallToolResult:
        return CallToolResult(
            content=[], structured_content=self.payload, is_error=self.failed
        )


@pytest.mark.anyio
async def test_linear_mcp_successful_empty_search_and_exact_missing_lookup_preserve_distinct_outcomes(
    tmp_path: Path,
) -> None:
    memory = Memory(tmp_path / "memory.sqlite3", "pet", clock=lambda: 100)
    index = SourceIndex(memory.db, memory.pet)
    session = LinearSession()

    class Gateway(McpGateway):
        @asynccontextmanager
        async def session(
            self, selection: SelectedMcp
        ) -> AsyncGenerator[ClientSession]:
            yield cast(ClientSession, session)

    selection = mcp_selection(
        "lookup",
        "search",
        provider="linear",
        lookup_tool="lookup",
        search_tool="search",
    )
    gateway = Gateway(
        (selection,), lambda: 100, 10, cached=index.document, documents=index.documents
    )
    tickets = McpLinearTickets(selection, gateway.fetch, lambda: 100)
    source = TicketSource(tickets)
    retrieval = ContextRetrieval(
        index,
        (source,),
        RetrievalConfig(),
        lambda: 100,
        semantic_score=lambda first, second: 0,
    )
    issue: dict[str, object] = {
        "issue": {
            "id": "fixture-issue",
            "identifier": "PET-123",
            "title": "Confirm rollback ownership",
            "description": "A rollback owner is required.",
            "url": "https://linear.app/fixture/issue/PET-123",
            "updatedAt": "2026-11-05T12:00:00Z",
            "state": {"name": "In Progress", "type": "started"},
        }
    }
    session.payload = issue
    found = await retrieval.search("rollback")
    reference = found.hits[0].passage.source
    document_id = found.hits[0].passage.document_id
    session.payload = {"issues": []}
    empty = await gateway.fetch("fixture", "search", {"query": "unknown"})
    assert empty.status.availability == Availability.AVAILABLE
    assert empty.documents == ()
    assert empty.missing_ids == ()
    assert (
        await tickets.search_tickets("unknown", 10)
    ).status.availability == Availability.AVAILABLE
    index.capture(empty)
    assert index.document(document_id) is not None
    session.payload = {"issue": None}
    missing = await gateway.fetch("fixture", "lookup", {"id": "PET-123"})
    assert missing.status.availability == Availability.AVAILABLE
    assert missing.documents == ()
    assert missing.missing_ids == (document_id,)
    index.capture(missing)
    assert index.document(document_id) is None
    session.payload = issue
    restored = await retrieval.search("rollback")
    reference = restored.hits[0].passage.source
    session.payload = {"issue": None}
    removed = await retrieval.read_passage(reference)
    assert not removed[0].source.available
    assert removed[0].source_status is not None
    assert removed[0].source_status.availability == Availability.AVAILABLE
    assert index.changes()[-1].kind == "removed"
    session.payload = issue
    restored = await retrieval.search("rollback")
    reference = restored.hits[0].passage.source
    session.failed = True
    failed = await retrieval.read_passage(reference)
    assert not failed[0].source.available
    assert failed[0].source_status is not None
    assert failed[0].source_status.availability == Availability.FAILED
    assert index.document(document_id) is not None
    assert not index.current(reference)
    session.failed = False
    session.payload = {"issues": [{"id": "partial"}]}
    incompatible = await tickets.lookup("fixture-issue")
    assert incompatible.status.availability == Availability.UNAVAILABLE
    assert incompatible.tickets == ()
    memory.close()
