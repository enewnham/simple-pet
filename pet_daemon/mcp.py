import argparse
import hashlib
import json
import re
import sqlite3
import time
from collections import defaultdict
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, NotRequired, TypedDict

from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.provider import TokenVerifier
from mcp.server.auth.settings import AuthSettings
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from .behavior.sprites import Manifest
from .config import Config
from .domain import PullRequestState
from .storage.migrations import SCHEMA_VERSION
from .work.history import report_prose
from .work.models import ExecutionEvidence, SessionReport
from .workstream_marks import (
    NOTE_LIMIT,
    Identifier,
    MarkField,
    MarkPatch,
    MarkPriority,
    MarkStatus,
    MarkStore,
    PresenceStatus,
    SessionRef,
    WorkstreamNote,
)

if TYPE_CHECKING:
    from .mcp_marks import WorkstreamMarks

WRITE_PERMISSION = "write"

COVERAGE = (
    "Retained coding-agent reports and latest repository observations captured by "
    "the pet. Reports describe what agents said. Repository observations describe "
    "state at their checked_at time, not live state. Removals and overwritten "
    "repository observations are not tracked."
)
OBSERVATION_TABLES = ("work_checkouts", "work_branches", "work_pull_requests")
type ResultLimit = Annotated[int, Field(ge=1, le=50)]


class Passage(TypedDict):
    id: str
    text: str
    revision: str
    changed_at: float
    provenance: dict[str, object]
    offset: int
    total_characters: int
    next_offset: int | None


class Selection(TypedDict):
    pet: str
    checked_at: float
    coverage: str
    truncated: bool
    sources: list[Passage]
    workstreams: NotRequired[list[dict[str, object]]]


@dataclass(frozen=True)
class Source:
    id: str
    text: str
    revision: str
    changed_at: float
    provenance: dict[str, object]

    def passage(self, offset: int = 0, length: int = 1200) -> Passage:
        return {
            "id": self.id,
            "text": self.text[offset : offset + length],
            "revision": self.revision,
            "changed_at": self.changed_at,
            "provenance": self.provenance,
            "offset": offset,
            "total_characters": len(self.text),
            "next_offset": offset + length
            if offset + length < len(self.text)
            else None,
        }


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def report_source(row: sqlite3.Row) -> Source:
    report = SessionReport.model_validate_json(row["data"])
    session = report.payload.session
    return Source(
        f"report:{row['id']}",
        report_prose(report),
        digest(row["data"]),
        report.ingested_at if report.ingested_at is not None else report.observed_at,
        {
            "basis": "reported",
            "source": report.payload.source.value,
            "kind": report.payload.kind.value,
            "session": session.model_dump(mode="json"),
            "observed_at": report.observed_at,
            "capture_at": report.capture_at,
            "ingested_at": report.ingested_at,
            "context": report.context.model_dump(mode="json"),
            "hook_kind": report.hook_kind,
            **(
                {
                    "execution": {
                        "event": report.payload.event,
                        "outcome": report.payload.outcome,
                    }
                }
                if isinstance(report.payload, ExecutionEvidence)
                else {}
            ),
        },
    )


def observation_source(table: str, row: sqlite3.Row) -> Source:
    value = json.loads(row["data"])
    columns = row.keys()
    identity = {key: row[key] for key in columns if key not in {"pet", "data"}}
    identifier = digest(json.dumps(identity, sort_keys=True))
    text = json.dumps(value, indent=2, ensure_ascii=False)
    return Source(
        f"{table}:{identifier}",
        text,
        digest(row["data"]),
        value["checked_at"],
        {"basis": "observed", "checked_at": value["checked_at"], **identity},
    )


CONTEXT_STALE_SECONDS = 7 * 24 * 60 * 60


def context_priority(source: Source, now: float) -> int:
    if source.id.startswith("report:"):
        return int(now - source.changed_at > CONTEXT_STALE_SECONDS)
    value = json.loads(source.text)
    pr = value.get("pr")
    if pr and pr["state"] != PullRequestState.OPEN:
        return 2
    last_success_at = value.get("last_success_at")
    observed_at = source.changed_at if last_success_at is None else last_success_at
    return int(now - observed_at > CONTEXT_STALE_SECONDS)


def context_groups(sources: list[Source]) -> dict[str, str]:
    branches: dict[tuple[str, str], list[str]] = defaultdict(list)
    for source in sources:
        if source.id.startswith("work_checkouts:"):
            value = json.loads(source.text)
            local = value.get("local")
            if local:
                key = json.dumps(local["repository"], sort_keys=True), local["branch"]
                branches[key].append(value["worktree"])
    groups = {}
    for source in sources:
        if source.id.startswith("report:"):
            context = source.provenance["context"]
            assert isinstance(context, dict)
            session = source.provenance["session"]
            groups[source.id] = context["checkout"] or json.dumps(
                session, sort_keys=True
            )
            continue
        value = json.loads(source.text)
        if "worktree" in value:
            groups[source.id] = value["worktree"]
            continue
        pr = value.get("pr") or {}
        repository = value.get("repository") or pr.get("head_repository")
        branch = value.get("branch") or pr.get("head_branch")
        key = json.dumps(repository, sort_keys=True), branch
        checkouts = branches.get(key, [])
        groups[source.id] = (
            checkouts[0] if len(checkouts) == 1 else pr.get("url") or source.id
        )
    return groups


def select_context(
    reports: list[Source], observations: list[Source], limit: int, now: float
) -> list[Source]:
    sources = reports + observations
    groups = context_groups(sources)
    sources.sort(
        key=lambda source: (
            context_priority(source, now),
            not source.id.startswith("report:"),
            -source.changed_at,
            source.id,
        )
    )
    grouped: dict[str, list[Source]] = defaultdict(list)
    for source in sources:
        grouped[groups[source.id]].append(source)
    selected = []
    for priority in range(3):
        eligible = [
            [source for source in group if context_priority(source, now) == priority]
            for group in grouped.values()
        ]
        while any(eligible):
            for group in eligible:
                if group:
                    selected.append(group.pop(0))
                    if len(selected) == limit:
                        return selected
    return selected


class WorkContext:
    def __init__(
        self, path: Path, pet: str, clock: Callable[[], float] = time.time
    ) -> None:
        self.path = path.resolve()
        self.pet = pet
        self.clock = clock

    @contextmanager
    def database(self) -> Generator[sqlite3.Connection]:
        db = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)
        db.row_factory = sqlite3.Row
        try:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version != SCHEMA_VERSION:
                raise ValueError(
                    "Pet memory schema differs from this server. Start the matching pet daemon first."
                )
            db.execute("BEGIN")
            yield db
        finally:
            db.close()

    def reports(
        self,
        db: sqlite3.Connection,
        limit: int,
        *,
        query: str = "",
        after: float | None = None,
        balanced: bool = False,
    ) -> list[Source]:
        parameters: list[str | int | float] = [self.pet]
        condition = (
            "r.pet=? AND COALESCE(json_extract(r.data,'$.hook_kind'),'') NOT GLOB 'tool_*' "
            "AND (json_extract(r.data,'$.payload.type') IN ('IntentEvidence','ProgressEvidence') "
            "OR json_extract(r.data,'$.payload.event') IN ('turn_done','turn_error','subagent_done','interrupt'))"
        )
        terms = re.findall(r"\w+", query)
        if query:
            if not terms:
                return []
            condition += " AND r.id IN (SELECT rowid FROM work_reports_fts WHERE work_reports_fts MATCH ?)"
            parameters.append(" OR ".join(f'"{term}"' for term in terms))
        if after is not None:
            condition += (
                " AND COALESCE(json_extract(r.data,'$.ingested_at'),r.observed_at)>?"
            )
            parameters.append(after)
        rows = db.execute(
            f"SELECT r.id,r.data FROM work_reports r WHERE {condition} "
            "ORDER BY COALESCE(json_extract(r.data,'$.ingested_at'),r.observed_at) DESC, "
            "json_extract(r.data,'$.payload.type')='ExecutionEvidence' DESC,r.id DESC",
            parameters,
        )
        result = []
        seen: dict[str, str] = {}
        for row in rows:
            source = report_source(row)
            if not source.text.strip():
                continue
            kind = source.provenance["kind"]
            if kind in {"progress", "execution"}:
                identity = {
                    key: value
                    for key, value in source.provenance.items()
                    if key not in {"kind", "execution"}
                }
                key = json.dumps([identity, source.text], sort_keys=True)
                if key in seen and seen[key] != kind:
                    continue
                seen[key] = str(kind)
            result.append(source)
            if not balanced and len(result) == limit:
                break
        return result

    def observations(
        self,
        db: sqlite3.Connection,
        limit: int | None,
        *,
        query: str = "",
        after: float | None = None,
    ) -> list[Source]:
        result = []
        for table in OBSERVATION_TABLES:
            condition = "pet=?"
            parameters: list[str | float | int] = [self.pet]
            if query:
                condition += " AND instr(lower(data),lower(?))>0"
                parameters.append(query)
            if after is not None:
                condition += " AND json_extract(data,'$.checked_at')>?"
                parameters.append(after)
            if limit is not None:
                parameters.append(limit)
            suffix = " LIMIT ?" if limit is not None else ""
            rows = db.execute(
                f"SELECT * FROM {table} WHERE {condition} "
                f"ORDER BY json_extract(data,'$.checked_at') DESC{suffix}",
                parameters,
            )
            result.extend(observation_source(table, row) for row in rows)
        return result

    def select(
        self,
        limit: int,
        *,
        query: str = "",
        after: float | None = None,
        reserve_reports: bool = False,
    ) -> Selection:
        checked_at = self.clock()
        with self.database() as db:
            reports = self.reports(
                db, limit + 1, query=query, after=after, balanced=reserve_reports
            )
            observations = self.observations(
                db, None if reserve_reports else limit + 1, query=query, after=after
            )
        summaries = []
        if reserve_reports:
            from .mcp_marks import selected_context
            from .mcp_work import group_threads

            marks = MarkStore(self.path, self.pet, self.clock).all_effective()
            sources, summaries, truncated = selected_context(
                group_threads(reports + observations), marks, limit, checked_at
            )
        else:
            truncated = len(reports) + len(observations) > limit
            sources = sorted(
                reports + observations,
                key=lambda source: (source.changed_at, source.id),
                reverse=True,
            )
        result: Selection = {
            "pet": self.pet,
            "checked_at": checked_at,
            "coverage": COVERAGE,
            "truncated": truncated,
            "sources": [source.passage() for source in sources[:limit]],
        }
        if reserve_reports:
            result["workstreams"] = summaries
        return result

    def read(self, source_id: str, offset: int, length: int) -> Passage:
        with self.database() as db:
            if source_id.startswith("report:"):
                row = db.execute(
                    "SELECT id,data FROM work_reports WHERE pet=? AND id=?",
                    (self.pet, source_id.removeprefix("report:")),
                ).fetchone()
                source = report_source(row) if row else None
            elif source_id.startswith("session:"):
                from .mcp_companion import session_source
                from .mcp_sessions import read_snapshot

                snapshot = read_snapshot(self.path, self.pet, self.clock())
                source = (
                    next(
                        (
                            item
                            for session in snapshot.sessions
                            if (item := session_source(session)).id == source_id
                        ),
                        None,
                    )
                    if snapshot
                    else None
                )
            else:
                table = source_id.partition(":")[0]
                if table not in OBSERVATION_TABLES:
                    raise ValueError("Unknown source ID")
                rows = db.execute(f"SELECT * FROM {table} WHERE pet=?", (self.pet,))
                source = next(
                    (
                        item
                        for row in rows
                        if (item := observation_source(table, row)).id == source_id
                    ),
                    None,
                )
        if source is None:
            raise ValueError("Source is no longer retained")
        return source.passage(offset, length)


def register_mark_tools(
    server: MCPServer, marks: WorkstreamMarks, *, remote: bool
) -> None:
    annotations = ToolAnnotations(read_only_hint=True, destructive_hint=False)

    def write_principal() -> str:
        if not remote:
            return "local-stdio"
        token = get_access_token()
        if token is None or WRITE_PERMISSION not in token.scopes:
            raise ToolError("This MCP credential cannot write")
        return f"bearer:{token.client_id}"

    writes = ToolAnnotations(
        read_only_hint=False,
        destructive_hint=False,
        idempotent_hint=False,
        open_world_hint=False,
    )

    @server.tool(annotations=writes)
    def mark_workstream(
        id: Annotated[str, Field(min_length=1, max_length=128)],
        author: Annotated[str, Field(min_length=1, max_length=128)],
        reason: Annotated[str, Field(min_length=1, max_length=2000)],
        status: MarkStatus | None = None,
        priority: MarkPriority | None = None,
        tags: list[str] | None = None,
        note: str | None = None,
        goal: str | None = None,
        title: str | None = None,
        blocked_on: str | None = None,
        snooze_until: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None = None,
        notes: Annotated[list[WorkstreamNote], Field(max_length=NOTE_LIMIT)]
        | None = None,
    ) -> dict[str, object]:
        """Append attributed information about a workstream. Captured evidence stays separate.

        Omitted fields stay unchanged. Tags replace the shared list. Typed notes
        replace the calling principal's list.
        """
        write_principal()
        try:
            patch = MarkPatch(
                status=status,
                priority=priority,
                tags=tags,
                note=note,
                goal=goal,
                title=title,
                blocked_on=blocked_on,
                snooze_until=snooze_until,
                notes=notes,
            )
            return marks.mark(id, patch, author, reason)
        except ValueError as error:
            raise ToolError(str(error)) from error

    @server.tool(annotations=writes)
    def unmark_workstream(
        id: Annotated[str, Field(min_length=1, max_length=128)],
        author: Annotated[str, Field(min_length=1, max_length=128)],
        reason: Annotated[str, Field(min_length=1, max_length=2000)],
        fields: list[MarkField] | None = None,
    ) -> dict[str, object]:
        """Clear selected fields or all fields when omitted. Clears only the caller's typed notes."""
        write_principal()
        try:
            return marks.unmark(id, fields, author, reason)
        except ValueError as error:
            raise ToolError(str(error)) from error

    @server.tool(annotations=writes)
    def record_presence(
        status: PresenceStatus,
        author: Annotated[str, Field(min_length=1, max_length=128)],
        reason: Annotated[str, Field(min_length=1, max_length=2000)],
        renderer_event_id: Identifier | None = None,
        session: SessionRef | None = None,
        turn: Identifier | None = None,
    ) -> dict[str, object]:
        """Record observed human presence. Supply the session and turn for an observed human prompt."""
        principal = write_principal()
        try:
            return marks.store.record_presence(
                status,
                author,
                reason,
                principal,
                renderer_event_id=renderer_event_id,
                session=session,
                turn=turn,
            )
        except ValueError as error:
            raise ToolError(str(error)) from error

    @server.tool(annotations=annotations)
    def workstream_mark_history(
        id: Annotated[str, Field(min_length=1, max_length=128)],
        limit: ResultLimit = 20,
        before: Annotated[int, Field(ge=1)] | None = None,
    ) -> dict[str, object]:
        """Read append-only mark history, newest first. Follow next_before to older events."""
        return marks.history(id, limit, before)


def create_server(
    context: WorkContext,
    *,
    token_verifier: TokenVerifier | None = None,
    auth: AuthSettings | None = None,
) -> MCPServer:
    from .mcp_companion import CompanionSessions
    from .mcp_marks import WorkstreamMarks
    from .mcp_work import CompanionContext

    work = CompanionContext(context)
    marks = WorkstreamMarks(context)
    live = CompanionSessions(context)

    server = MCPServer(
        "simple-pet",
        token_verifier=token_verifier,
        auth=auth,
        instructions=(
            "Read the pet's captured work evidence. Check source timestamps before "
            "describing current state. Start with workstreams, then workstream for detail. "
            "Use sessions for fresh daemon runtime and attention for follow-up candidates. "
            "Decision and question excerpts are candidates, not verified conclusions. "
            "Use your own integrations to verify live repository state."
        ),
    )
    annotations = ToolAnnotations(read_only_hint=True, destructive_hint=False)

    register_mark_tools(server, marks, remote=token_verifier is not None)

    @server.tool(annotations=annotations)
    def workstreams(limit: ResultLimit = 12) -> dict[str, object]:
        """List recent captured threads with goals, branch, PR, ticket and source evidence."""
        return work.workstreams(limit)

    @server.tool(annotations=annotations)
    def workstream(
        id: Annotated[str, Field(min_length=1, max_length=128)],
        limit: ResultLimit = 12,
    ) -> dict[str, object]:
        """Read a thread's first intent, recent turns, decision/question candidates and repo evidence."""
        return work.workstream(id, limit)

    @server.tool(annotations=annotations)
    def since(
        ts: Annotated[float, Field(ge=0, allow_inf_nan=False)],
        limit: ResultLimit = 20,
    ) -> dict[str, object]:
        """Read retained changes after a Unix timestamp grouped by captured thread.

        Latest observations overwrite older state. This is not a complete change log.
        """
        changes = marks.store.changes(ts, limit + 1)
        return {
            **work.since(ts, limit),
            "mark_changes": changes[:limit],
            "mark_changes_truncated": len(changes) > limit,
        }

    @server.tool(annotations=annotations)
    def sessions(limit: ResultLimit = 20) -> dict[str, object]:
        """Read fresh, process-verified harness sessions joined with captured work.

        An unavailable daemon snapshot cannot establish whether sessions are live.
        """
        presence = marks.store.latest_presence()
        return {
            **live.sessions(limit),
            "reported_presence": presence,
        }

    @server.tool(annotations=annotations)
    def attention(limit: ResultLimit = 20) -> dict[str, object]:
        """Find pending permissions, failed turns, unanswered questions and stalled PR candidates.

        Stalled PR candidates have no captured agent activity for at least 24 hours.
        Verify live PR state before acting on a candidate.
        """
        return dict(live.attention(limit))

    @server.tool(annotations=annotations)
    def decisions(
        query: Annotated[str, Field(max_length=1000)] = "",
        limit: ResultLimit = 12,
    ) -> dict[str, object]:
        """Find explicit decision/correction wording in retained reports with where and when.

        Excerpts are lexical candidates, including possible quotes or reversals.
        """
        return work.decisions(query, limit)

    @server.tool(annotations=annotations)
    def current_work_context(limit: ResultLimit = 12) -> Selection:
        """Read flat captured work. Prefer workstreams for thread summaries.

        Fresh evidence ranks before evidence older than seven days. Closed PR
        observations rank last. Each group supplies a representative before detail.
        """
        return context.select(limit, reserve_reports=True)

    @server.tool(annotations=annotations)
    def context_search(
        query: Annotated[str, Field(min_length=1, max_length=1000)],
        limit: ResultLimit = 12,
    ) -> Selection:
        """Find retained reports by words and repository observations by a literal phrase."""
        return context.select(limit, query=query)

    @server.tool(annotations=annotations)
    def read_passage(
        source_id: str,
        offset: Annotated[int, Field(ge=0)] = 0,
        length: Annotated[int, Field(ge=1, le=12000)] = 2400,
    ) -> Passage:
        """Read a source's text and provenance. Follow next_offset for longer sources."""
        return context.read(source_id, offset, length)

    @server.tool(annotations=annotations)
    def relevant_changes(
        after: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 0,
        limit: ResultLimit = 20,
    ) -> Selection:
        """Read flat retained changes. Prefer since for changes grouped by thread.

        This is a bounded view, not a complete change log. A truncated response omits
        older matching sources. Removals and overwritten observations are unavailable.
        """
        return context.select(limit, after=after)

    return server


def main() -> None:
    from .mcp_http import DEFAULT_TOKEN_FILE, LOOPBACK_HOSTS, run_http

    parser = argparse.ArgumentParser(
        prog="pet-mcp",
        description="Serve captured work and caller workstream marks through MCP",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "pet.config.json",
    )
    parser.add_argument("--memory-file", type=Path)
    parser.add_argument("--pet", help="Select a pet's records by its manifest name")
    parser.add_argument(
        "--http", action="store_true", help="Serve authenticated HTTP at /mcp"
    )
    parser.add_argument("--host", choices=LOOPBACK_HOSTS, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18765)
    parser.add_argument("--token-file", type=Path, default=DEFAULT_TOKEN_FILE)
    parser.add_argument(
        "--write-token-file", type=Path, help="Separate bearer for HTTP writes"
    )
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.memory_file is not None and args.pet is not None:
        path, pet = args.memory_file, args.pet
    else:
        config = Config.load(args.config)
        manifest = Manifest.model_validate_json(
            (config.pet_dir / "manifest.json").read_text()
        )
        path, pet = args.memory_file or config.memory_file, args.pet or manifest.pet
    context = WorkContext(path, pet)
    if args.http:
        try:
            run_http(
                context, args.token_file, args.host, args.port, args.write_token_file
            )
        except (OSError, ValueError) as error:
            parser.exit(1, f"pet-mcp: {error}\n")
    else:
        create_server(context).run()


if __name__ == "__main__":
    main()
