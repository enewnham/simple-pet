import asyncio
import json
import os
import re
from collections.abc import AsyncGenerator, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, replace

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client
from mcp.shared.exceptions import MCPError
from mcp.types import CallToolResult, PaginatedRequestParams

from ..coordination.models import SourceReference
from .config import SelectedMcp, StdioMcp
from .providers import Availability, Document, SourceResult, SourceStatus, revision
from .tickets import LinearResultKind, linear_result
from .tools import ToolAuthority, WorkflowTool


@dataclass(frozen=True)
class RemoteTool:
    name: str
    description: str
    schema: dict[str, object]


def failure_detail(error: BaseException) -> str:
    if isinstance(error, BaseExceptionGroup):
        return "; ".join(
            dict.fromkeys(failure_detail(child) for child in error.exceptions)
        )
    return str(error)


class McpGateway:
    def __init__(
        self,
        selections: tuple[SelectedMcp, ...],
        clock: Callable[[], float],
        timeout: float,
        cached: Callable[[str], Document | None] | None = None,
        documents: Callable[[], tuple[Document, ...]] | None = None,
    ) -> None:
        self.selections = {selection.name: selection for selection in selections}
        self.clock = clock
        self.timeout = timeout
        self.cached = cached
        self.documents = documents
        self.catalog: dict[str, tuple[RemoteTool, ...]] = {}
        self.statuses: dict[str, SourceStatus] = {}
        self.results: dict[str, Document] = {}
        self.queries: dict[str, tuple[str, str, dict[str, object]]] = {}

    @asynccontextmanager
    async def session(self, selection: SelectedMcp) -> AsyncGenerator[ClientSession]:
        async with AsyncExitStack() as stack:
            if isinstance(selection, StdioMcp):
                environment = {
                    name: os.environ[name]
                    for name in selection.environment
                    if name in os.environ
                }
                streams = await stack.enter_async_context(
                    stdio_client(
                        StdioServerParameters(
                            command=selection.command,
                            args=list(selection.args),
                            env=environment,
                        )
                    )
                )
            else:
                headers = {}
                if selection.authorization_env:
                    authorization = os.environ.get(selection.authorization_env)
                    if authorization is None:
                        raise ValueError(
                            f"Missing MCP credential environment variable: {selection.authorization_env}"
                        )
                    headers["Authorization"] = authorization
                client = await stack.enter_async_context(
                    create_mcp_http_client(headers=headers)
                )
                streams = await stack.enter_async_context(
                    streamable_http_client(selection.url, http_client=client)
                )
            session = await stack.enter_async_context(
                ClientSession(streams[0], streams[1], read_timeout_seconds=self.timeout)
            )
            await session.initialize()
            yield session

    async def _catalog(
        self, session: ClientSession, selection: SelectedMcp
    ) -> tuple[RemoteTool, ...]:
        found = {}
        cursor = None
        seen = set()
        while True:
            page = await session.list_tools(
                params=PaginatedRequestParams(cursor=cursor)
            )
            for item in page.tools:
                if item.name not in selection.tools:
                    continue
                if (
                    item.annotations is None
                    or item.annotations.read_only_hint is not True
                ):
                    raise ValueError(
                        f"Selected MCP tool lacks a read-only declaration: {item.name}"
                    )
                if item.annotations.destructive_hint is True:
                    raise ValueError(
                        f"Selected MCP tool declares destructive effects: {item.name}"
                    )
                found[item.name] = RemoteTool(
                    item.name, item.description or item.name, item.input_schema
                )
            cursor = page.next_cursor
            if cursor is None:
                break
            if cursor in seen:
                raise ValueError("MCP tool discovery repeated a pagination cursor")
            seen.add(cursor)
        missing = set(selection.tools) - found.keys()
        if missing:
            raise ValueError(
                f"Configured MCP tools are unavailable: {', '.join(sorted(missing))}"
            )
        if selection.search_tool is not None and selection.search_tool not in found:
            raise ValueError(
                "The configured MCP search tool must be explicitly selected"
            )
        if selection.lookup_tool is not None and selection.lookup_tool not in found:
            raise ValueError(
                "The configured MCP lookup tool must be explicitly selected"
            )
        return tuple(found[name] for name in selection.tools)

    async def discover(self) -> tuple[SourceStatus, ...]:
        async def inspect(selection: SelectedMcp) -> SourceStatus:
            provider = f"mcp:{selection.name}"
            try:
                async with (
                    asyncio.timeout(self.timeout),
                    self.session(selection) as session,
                ):
                    self.catalog[selection.name] = await self._catalog(
                        session, selection
                    )
                status = SourceStatus(
                    provider,
                    Availability.AVAILABLE,
                    self.clock(),
                    collections=selection.tools,
                )
            except (
                MCPError,
                OSError,
                ValueError,
                RuntimeError,
                TimeoutError,
                ExceptionGroup,
            ) as error:
                self.catalog.pop(selection.name, None)
                status = SourceStatus(
                    provider, Availability.FAILED, self.clock(), failure_detail(error)
                )
            self.statuses[selection.name] = status
            return status

        return tuple(
            await asyncio.gather(
                *(inspect(selection) for selection in self.selections.values())
            )
        )

    async def call(
        self, server: str, name: str, arguments: dict[str, object]
    ) -> SourceResult:
        selection = self.selections.get(server)
        if selection is None or name not in selection.tools:
            raise ValueError("The MCP operation is not explicitly selected")
        async with asyncio.timeout(self.timeout), self.session(selection) as session:
            catalog = await self._catalog(session, selection)
            if name not in {item.name for item in catalog}:
                raise ValueError("The MCP operation is unavailable")
            result = await session.call_tool(name, arguments)
            if not isinstance(result, CallToolResult) or result.is_error:
                raise RuntimeError(
                    "The MCP operation failed or requires unsupported interaction"
                )
        text = "\n".join(block.text for block in result.content if block.type == "text")
        structured = result.structured_content
        if structured:
            text = json.dumps(structured, sort_keys=True, ensure_ascii=False)
        provider = "linear" if selection.provider == "linear" else f"mcp:{server}"
        query = json.dumps(
            {"server": server, "tool": name, "arguments": arguments}, sort_keys=True
        )
        if selection.provider == "linear":
            try:
                parsed = structured or json.loads(text)
            except json.JSONDecodeError:
                parsed = None
            normalized = linear_result(parsed, self.clock())
            if normalized.kind != LinearResultKind.UNRECOGNIZED:
                documents = tuple(
                    replace(document, query=query) for document in normalized.documents
                )
                for document in documents:
                    self.results[document.source.id] = document
                    self.queries[document.source.id] = (server, name, arguments)
                status = SourceStatus(
                    "linear",
                    Availability.AVAILABLE,
                    self.clock(),
                    collections=selection.tools,
                )
                self.statuses[server] = status
                missing = (
                    self._missing_linear_ids(selection, name, arguments)
                    if normalized.kind == LinearResultKind.NO_MATCH
                    else ()
                )
                return SourceResult(documents, status, missing_ids=missing)
        identifier = (
            f"{provider}:{name}:{revision(json.dumps(arguments, sort_keys=True))}"
        )
        document = Document(
            SourceReference(
                identifier,
                provider,
                revision(text),
                self.clock(),
                uri=f"mcp://{server}/{name}",
            ),
            f"{server}: {name}",
            text,
            kind="mcp_result",
            links=tuple(re.findall(r"https?://[^\s\"'<>]+", text)),
            query=query,
            checked_at=self.clock(),
        )
        self.results[identifier] = document
        self.queries[identifier] = (server, name, arguments)
        status = SourceStatus(
            provider, Availability.AVAILABLE, self.clock(), collections=selection.tools
        )
        self.statuses[server] = status
        return SourceResult((document,), status)

    def tools(self) -> tuple[WorkflowTool, ...]:
        result = []
        for server, remote_tools in self.catalog.items():
            for remote in remote_tools:
                alias = "source_" + server + "_" + revision(remote.name)[:12]

                async def invoke(
                    arguments: dict[str, object],
                    authority: ToolAuthority,
                    selected_server: str = server,
                    selected_tool: str = remote.name,
                ) -> object:
                    return await self.fetch(selected_server, selected_tool, arguments)

                result.append(
                    WorkflowTool(
                        alias,
                        f"Read-only {server}/{remote.name}: {remote.description}",
                        remote.schema,
                        invoke,
                    )
                )
        return tuple(result)

    def _missing_linear_ids(
        self, selection: SelectedMcp, name: str, arguments: dict[str, object]
    ) -> tuple[str, ...]:
        if name != selection.lookup_tool:
            return ()
        identifier = arguments.get(selection.identifier_parameter)
        if not isinstance(identifier, str):
            return ()
        cached = self.documents() if self.documents is not None else ()
        known = {
            document.source.id: document
            for document in (*cached, *self.results.values())
        }
        matches = []
        for document in known.values():
            if document.source.provider != "linear" or document.kind != "ticket":
                continue
            data = json.loads(document.text)
            if identifier in {
                document.source.id,
                document.source.id.removeprefix("linear:"),
                document.source.uri,
                data["identifier"],
            }:
                matches.append(document.source.id)
        return tuple(matches)

    async def fetch(
        self, server: str, name: str, arguments: dict[str, object]
    ) -> SourceResult:
        if server not in self.selections or name not in self.selections[server].tools:
            raise ValueError("The MCP operation is not explicitly selected")
        try:
            return await self.call(server, name, arguments)
        except (
            MCPError,
            OSError,
            ValueError,
            RuntimeError,
            TimeoutError,
            ExceptionGroup,
        ) as error:
            status = SourceStatus(
                f"mcp:{server}",
                Availability.FAILED,
                self.clock(),
                failure_detail(error),
            )
            self.statuses[server] = status
            return SourceResult((), status)


class McpSource:
    def __init__(self, gateway: McpGateway, server: str) -> None:
        self.gateway = gateway
        self.server = server
        self.name = f"mcp:{server}"

    async def search(self, query: str, limit: int) -> SourceResult:
        selection = self.gateway.selections[self.server]
        if not query and selection.bootstrap_query is None:
            return SourceResult(
                (),
                self.gateway.statuses.get(self.server)
                or SourceStatus(
                    self.name,
                    Availability.UNAVAILABLE,
                    self.gateway.clock(),
                    "MCP discovery has not completed.",
                ),
            )
        if selection.search_tool is None:
            status = self.gateway.statuses.get(self.server) or SourceStatus(
                self.name,
                Availability.UNAVAILABLE,
                self.gateway.clock(),
                "No contextual search operation selected; precise read tools remain available.",
            )
            return SourceResult((), status)
        arguments = dict(selection.search_arguments)
        arguments[selection.query_parameter] = query or selection.bootstrap_query
        return await self.gateway.fetch(self.server, selection.search_tool, arguments)

    async def read(self, source_id: str) -> SourceResult:
        query = self.gateway.queries.get(source_id)
        if query is None and self.gateway.cached is not None:
            document = self.gateway.cached(source_id)
            if document is not None and document.query:
                stored = json.loads(document.query)
                query = (stored["server"], stored["tool"], stored["arguments"])
        if query is not None:
            return await self.gateway.fetch(*query)
        document = self.gateway.results.get(source_id)
        return SourceResult(
            (document,) if document else (),
            self.gateway.statuses.get(self.server)
            or SourceStatus(self.name, Availability.UNAVAILABLE, self.gateway.clock()),
        )
