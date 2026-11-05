import asyncio
import json
import os
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Protocol
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field

from ..coordination.models import SourceReference
from .config import LinearConfig, SelectedMcp
from .providers import Availability, Document, SourceResult, SourceStatus


class TicketCapability(StrEnum):
    LOOKUP = "lookup"
    SEARCH = "search"
    ASSIGNED = "assigned"
    FOLLOWED = "followed"
    LINKED = "linked"
    STATUS = "status"
    PROJECT = "project"


@dataclass(frozen=True)
class Ticket:
    id: str
    provider: str
    identifier: str
    title: str
    body: str
    uri: str
    revision: str
    updated_at: float | None
    state: str | None = None
    state_type: str | None = None
    assignee: str | None = None
    project: str | None = None
    team: str | None = None
    links: tuple[str, ...] = ()


@dataclass(frozen=True)
class TicketResult:
    tickets: tuple[Ticket, ...]
    status: SourceStatus
    capabilities: tuple[TicketCapability, ...]
    complete: bool = False


class TicketProvider(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def capabilities(self) -> tuple[TicketCapability, ...]: ...

    async def lookup(self, identifier: str) -> TicketResult: ...
    async def search_tickets(self, query: str, limit: int) -> TicketResult: ...
    async def collect(self, limit: int) -> TicketResult: ...


class LinearIssue(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    identifier: str
    title: str
    description: str | None = None
    url: str
    updated_at: str = Field(alias="updatedAt")
    state: dict[str, str] | None = None
    assignee: dict[str, str] | None = None
    project: dict[str, str] | None = None
    team: dict[str, str] | None = None

    def ticket(self) -> Ticket:
        return Ticket(
            self.id,
            "linear",
            self.identifier,
            self.title,
            self.description or "",
            self.url,
            self.updated_at,
            datetime.fromisoformat(self.updated_at).timestamp(),
            (self.state or {}).get("name"),
            (self.state or {}).get("type"),
            (self.assignee or {}).get("id"),
            (self.project or {}).get("name"),
            (self.team or {}).get("key"),
            tuple(re.findall(r"https?://[^\s)>]+", self.description or "")),
        )


def ticket_document(ticket: Ticket, checked_at: float) -> Document:
    detail = {
        "identifier": ticket.identifier,
        "title": ticket.title,
        "body": ticket.body,
        "state": ticket.state,
        "state_type": ticket.state_type,
        "assignee": ticket.assignee,
        "project": ticket.project,
        "team": ticket.team,
    }
    return Document(
        SourceReference(
            f"{ticket.provider}:{ticket.id}",
            ticket.provider,
            ticket.revision,
            checked_at,
            uri=ticket.uri,
            scope=ticket.project or ticket.team or "",
            original_at=ticket.updated_at,
        ),
        f"{ticket.identifier}: {ticket.title}",
        json.dumps(detail, ensure_ascii=False),
        kind="ticket",
        topics=tuple(
            value for value in (ticket.identifier, ticket.project, ticket.team) if value
        ),
        links=tuple(dict.fromkeys((ticket.uri, *ticket.links))),
        checked_at=checked_at,
    )


def linear_documents(payload: object, checked_at: float) -> tuple[Document, ...]:
    found = {}

    def visit(value: object) -> None:
        if isinstance(value, Mapping):
            if {
                "id",
                "identifier",
                "title",
                "description",
                "url",
                "updatedAt",
                "state",
            } <= value.keys():
                try:
                    issue = LinearIssue.model_validate(value)
                except ValueError:
                    return
                found[issue.id] = ticket_document(issue.ticket(), checked_at)
            else:
                for child in value.values():
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(payload)
    return tuple(found.values())


class LinearResultKind(StrEnum):
    MATCHES = "matches"
    NO_MATCH = "no_match"
    UNRECOGNIZED = "unrecognized"


@dataclass(frozen=True)
class LinearResult:
    kind: LinearResultKind
    documents: tuple[Document, ...] = ()


def _empty_linear_envelope(payload: object) -> bool:
    if (
        not isinstance(payload, Mapping)
        or payload.get("errors")
        or payload.get("error")
    ):
        return False
    if "issue" in payload:
        return payload["issue"] is None
    if "issues" in payload:
        issues = payload["issues"]
        return issues == [] or isinstance(issues, Mapping) and issues.get("nodes") == []
    return any(
        _empty_linear_envelope(payload[key])
        for key in ("data", "result")
        if key in payload
    )


def linear_result(payload: object, checked_at: float) -> LinearResult:
    documents = linear_documents(payload, checked_at)
    if documents:
        return LinearResult(LinearResultKind.MATCHES, documents)
    if _empty_linear_envelope(payload):
        return LinearResult(LinearResultKind.NO_MATCH)
    return LinearResult(LinearResultKind.UNRECOGNIZED)


type GraphqlRequest = Callable[[str, dict[str, object]], Awaitable[dict[str, object]]]

FIELDS = "id identifier title description url updatedAt state { name type } assignee { id } project { name } team { key }"


class LinearTickets:
    name = "linear"
    capabilities = (
        TicketCapability.LOOKUP,
        TicketCapability.SEARCH,
        TicketCapability.ASSIGNED,
        TicketCapability.FOLLOWED,
        TicketCapability.LINKED,
        TicketCapability.STATUS,
        TicketCapability.PROJECT,
    )

    def __init__(
        self,
        config: LinearConfig,
        clock: Callable[[], float],
        request: GraphqlRequest | None = None,
    ) -> None:
        self.config = config
        self.clock = clock
        self.request = request or self._request

    def _send(self, query: str, variables: dict[str, object]) -> dict[str, object]:
        authorization = os.environ.get(self.config.authorization_env)
        if not authorization:
            raise RuntimeError(
                f"Missing Linear credential environment variable: {self.config.authorization_env}"
            )
        request = Request(
            self.config.endpoint,
            data=json.dumps({"query": query, "variables": variables}).encode(),
            headers={
                "Authorization": authorization,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urlopen(request, timeout=self.config.timeout_seconds) as response:
            result = json.load(response)
        if not isinstance(result, dict):
            raise TypeError("Linear returned an invalid response")
        return result

    async def _request(
        self, query: str, variables: dict[str, object]
    ) -> dict[str, object]:
        return await asyncio.to_thread(self._send, query, variables)

    async def _query(
        self, query: str, variables: dict[str, object]
    ) -> dict[str, object]:
        result = await self.request(query, variables)
        if result.get("errors"):
            raise RuntimeError(
                "Linear returned a GraphQL error; partial results are unavailable"
            )
        data = result.get("data")
        if not isinstance(data, dict):
            raise TypeError("Linear returned no data object")
        return data

    async def lookup(self, identifier: str) -> TicketResult:
        try:
            data = await self._query(
                f"query CompanionIssue($id: String!) {{ issue(id: $id) {{ {FIELDS} }} }}",
                {"id": identifier},
            )
            issue = data.get("issue")
            tickets = (
                (LinearIssue.model_validate(issue).ticket(),)
                if issue is not None
                else ()
            )
            return TicketResult(tickets, self._status(), self.capabilities)
        except (OSError, ValueError, TypeError, RuntimeError, TimeoutError) as error:
            return TicketResult((), self._status(str(error)), self.capabilities)

    def _status(self, error: str = "") -> SourceStatus:
        return SourceStatus(
            self.name,
            Availability.FAILED if error else Availability.AVAILABLE,
            self.clock(),
            error,
            collections=tuple(
                dict.fromkeys(
                    (
                        *self.config.team_ids,
                        *self.config.assignee_ids,
                        *self.config.followed_ids,
                        *self.config.linked_ids,
                    )
                )
            ),
        )

    async def search_tickets(self, query: str, limit: int) -> TicketResult:
        identifier = re.search(r"\b[A-Z][A-Z0-9]*-\d+\b", query)
        if identifier is not None:
            return await self.lookup(identifier.group())
        scope: dict[str, object] = {}
        if self.config.team_ids:
            scope["team"] = {"id": {"in": list(self.config.team_ids)}}
        if self.config.assignee_ids:
            scope["assignee"] = {"id": {"in": list(self.config.assignee_ids)}}
        if not scope:
            linked = tuple(
                dict.fromkeys((*self.config.followed_ids, *self.config.linked_ids))
            )[:limit]
            if linked:
                results = await asyncio.gather(
                    *(self.lookup(identifier) for identifier in linked)
                )
                tokens = set(query.casefold().split())
                tickets = tuple(
                    ticket
                    for result in results
                    for ticket in result.tickets
                    if not tokens
                    or tokens
                    & set((ticket.title + " " + ticket.body).casefold().split())
                )
                failures = [
                    result.status.detail
                    for result in results
                    if result.status.availability == Availability.FAILED
                ]
                return TicketResult(
                    tickets, self._status("; ".join(failures)), self.capabilities
                )
            return TicketResult(
                (),
                SourceStatus(
                    self.name,
                    Availability.UNAVAILABLE,
                    self.clock(),
                    "Configure assigned, followed, linked, or team ticket scope.",
                ),
                self.capabilities,
            )
        if query:
            scope["or"] = [
                {"title": {"containsIgnoreCase": query}},
                {"description": {"containsIgnoreCase": query}},
            ]
        try:
            data = await self._query(
                f"query CompanionIssues($filter: IssueFilter!, $first: Int!) {{ issues(filter: $filter, first: $first, orderBy: updatedAt) {{ nodes {{ {FIELDS} }} pageInfo {{ hasNextPage endCursor }} }} }}",
                {"filter": scope, "first": min(limit, self.config.results)},
            )
            collection = data.get("issues")
            if not isinstance(collection, dict) or not isinstance(
                collection.get("nodes"), list
            ):
                raise TypeError("Linear returned no issue collection")
            tickets = tuple(
                LinearIssue.model_validate(node).ticket()
                for node in collection["nodes"]
            )
            return TicketResult(tickets, self._status(), self.capabilities)
        except (OSError, ValueError, TypeError, RuntimeError, TimeoutError) as error:
            return TicketResult((), self._status(str(error)), self.capabilities)

    async def collect(self, limit: int) -> TicketResult:
        if not self.config.team_ids and not self.config.assignee_ids:
            return await self.search_tickets("", limit)
        linked = tuple(
            dict.fromkeys((*self.config.followed_ids, *self.config.linked_ids))
        )[:limit]
        results = await asyncio.gather(
            self.search_tickets("", limit),
            *(self.lookup(identifier) for identifier in linked),
        )
        tickets = {ticket.id: ticket for result in results for ticket in result.tickets}
        failures = [
            result.status.detail
            for result in results
            if result.status.availability == Availability.FAILED
        ]
        status = self._status("; ".join(failures)) if failures else self._status()
        if not tickets and all(
            result.status.availability == Availability.UNAVAILABLE for result in results
        ):
            status = results[0].status
        return TicketResult(tuple(tickets.values())[:limit], status, self.capabilities)


class McpLinearTickets:
    name = "linear"

    def __init__(
        self,
        selection: SelectedMcp,
        call: Callable[[str, str, dict[str, object]], Awaitable[SourceResult]],
        clock: Callable[[], float],
        statuses: Mapping[str, SourceStatus] | None = None,
    ) -> None:
        self.selection = selection
        self.call = call
        self.clock = clock
        self.statuses = statuses
        self.capabilities = (
            *((TicketCapability.LOOKUP,) if selection.lookup_tool else ()),
            *((TicketCapability.SEARCH,) if selection.search_tool else ()),
        )

    async def _call(
        self, tool: str | None, arguments: dict[str, object]
    ) -> TicketResult:
        if tool is None:
            return TicketResult(
                (),
                SourceStatus(
                    self.name,
                    Availability.UNAVAILABLE,
                    self.clock(),
                    "The configured MCP provider does not expose this optional ticket capability.",
                ),
                self.capabilities,
            )
        result = await self.call(self.selection.name, tool, arguments)
        tickets = []
        for document in result.documents:
            if document.source.provider != "linear" or document.kind != "ticket":
                continue
            data = json.loads(document.text)
            tickets.append(
                Ticket(
                    document.source.id.removeprefix("linear:"),
                    "linear",
                    data["identifier"],
                    data["title"],
                    data["body"],
                    document.source.uri,
                    document.source.revision,
                    document.source.original_at,
                    data["state"],
                    data["state_type"],
                    data["assignee"],
                    data["project"],
                    data["team"],
                    document.links,
                )
            )
        status = SourceStatus(
            self.name,
            result.status.availability,
            result.status.checked_at,
            result.status.detail,
            collections=self.selection.tools,
        )
        if result.documents and not tickets:
            status = SourceStatus(
                self.name,
                Availability.UNAVAILABLE,
                self.clock(),
                "The MCP result lacks a complete versioned Linear issue. Use the selected precise operation as source evidence instead.",
            )
        return TicketResult(tuple(tickets), status, self.capabilities)

    async def lookup(self, identifier: str) -> TicketResult:
        return await self._call(
            self.selection.lookup_tool,
            {self.selection.identifier_parameter: identifier},
        )

    async def search_tickets(self, query: str, limit: int) -> TicketResult:
        return await self._call(
            self.selection.search_tool,
            {**self.selection.search_arguments, self.selection.query_parameter: query},
        )

    async def collect(self, limit: int) -> TicketResult:
        if self.selection.bootstrap_query is None:
            discovered = (
                self.statuses.get(self.selection.name)
                if self.statuses is not None
                else None
            )
            return TicketResult(
                (),
                SourceStatus(
                    self.name,
                    discovered.availability if discovered else Availability.UNAVAILABLE,
                    self.clock(),
                    discovered.detail
                    if discovered
                    else "MCP discovery has not completed.",
                ),
                self.capabilities,
            )
        return await self.search_tickets(self.selection.bootstrap_query, limit)


class TicketSource:
    def __init__(self, tickets: TicketProvider) -> None:
        self.tickets = tickets
        self.name = tickets.name

    async def search(self, query: str, limit: int) -> SourceResult:
        result = (
            await self.tickets.search_tickets(query, limit)
            if query
            else await self.tickets.collect(limit)
        )
        return SourceResult(
            tuple(
                ticket_document(ticket, result.status.checked_at)
                for ticket in result.tickets
            ),
            self._status(result),
        )

    async def read(self, source_id: str) -> SourceResult:
        result = await self.tickets.lookup(source_id.removeprefix(f"{self.name}:"))
        return SourceResult(
            tuple(
                ticket_document(ticket, result.status.checked_at)
                for ticket in result.tickets
            ),
            self._status(result),
        )

    @staticmethod
    def _status(result: TicketResult) -> SourceStatus:
        from dataclasses import replace

        return replace(
            result.status,
            capabilities=tuple(capability.value for capability in result.capabilities),
        )
