import asyncio
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult

from pet_daemon.domain import PullRequestState, SessionSource
from pet_daemon.identity import SessionIdentity
from pet_daemon.mcp import WorkContext, create_server
from pet_daemon.repository_identity import RemoteRepository
from pet_daemon.storage import Memory
from pet_daemon.work.models import (
    ExecutionEvidence,
    IntentEvidence,
    ReportContext,
    SessionReport,
)
from pet_daemon.work.observations import (
    BranchObservation,
    CheckoutObservation,
    GitObservation,
    PullRequest,
    PullRequestObservation,
)


class McpTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "memory.sqlite3"
        self.memory = Memory(self.path, "fixture", clock=lambda: 1000)
        self.context = WorkContext(self.path, "fixture", clock=lambda: 1000)
        self.report("Check parser behavior", observed_at=100)
        self.report("Other pet's secret", observed_at=900, pet="other")
        with self.memory.transaction():
            self.memory._work_repository.put_observation(
                CheckoutObservation(
                    worktree="/parser", local=None, checked_at=200, error="Unavailable"
                )
            )

    def tearDown(self) -> None:
        self.memory.close()
        self.temporary.cleanup()

    def report(
        self,
        text: str,
        *,
        observed_at: float,
        pet: str = "fixture",
        checkout: str = "/parser",
        session: str = "parser-session",
    ) -> None:
        memory = self.memory if pet == "fixture" else Memory(self.path, pet)
        try:
            report = SessionReport(
                payload=IntentEvidence(
                    session=SessionIdentity(
                        source=SessionSource.CODEX, native_id=session
                    ),
                    text=text,
                ),
                context=ReportContext(checkout=checkout),
                observed_at=observed_at,
            )
            with memory.transaction():
                memory._work_repository.append_report(report)
        finally:
            if memory is not self.memory:
                memory.close()

    async def test_external_client_reads_only_selected_pet_and_read_tools(
        self,
    ) -> None:
        parameters = StdioServerParameters(
            command=sys.executable,
            args=[
                "-m",
                "pet_daemon.mcp",
                "--memory-file",
                str(self.path),
                "--pet",
                "fixture",
            ],
            cwd=Path(__file__).resolve().parents[1],
        )
        async with (
            asyncio.timeout(20),
            stdio_client(parameters) as streams,
            ClientSession(*streams) as client,
        ):
            await client.initialize()
            catalog = await client.list_tools()
            self.assertEqual(
                {tool.name for tool in catalog.tools},
                {
                    "mark_workstream",
                    "unmark_workstream",
                    "workstream_mark_history",
                    "workstreams",
                    "workstream",
                    "since",
                    "sessions",
                    "attention",
                    "decisions",
                    "current_work_context",
                    "context_search",
                    "read_passage",
                    "relevant_changes",
                },
            )
            self.assertTrue(
                all(
                    tool.annotations is not None
                    and tool.annotations.read_only_hint
                    == (tool.name not in {"mark_workstream", "unmark_workstream"})
                    and tool.annotations.destructive_hint is False
                    for tool in catalog.tools
                )
            )
            response = await client.call_tool("current_work_context", {})
            self.assertIsInstance(response, CallToolResult)
            assert isinstance(response, CallToolResult)
            self.assertFalse(response.is_error)
            data = response.structured_content
            assert data is not None
            sources = data["sources"]
            self.assertEqual(len(sources), 2)
            self.assertNotIn("Other pet's secret", json.dumps(data))
            report = next(item for item in sources if item["id"].startswith("report:"))
            self.assertEqual(report["provenance"]["basis"], "reported")
            self.assertEqual(report["provenance"]["observed_at"], 100)
            passage = await client.call_tool(
                "read_passage", {"source_id": report["id"]}
            )
            assert isinstance(passage, CallToolResult)
            assert passage.structured_content is not None
            self.assertEqual(
                passage.structured_content["text"], "Check parser behavior"
            )
            invalid = await client.call_tool("current_work_context", {"limit": 1000})
            assert isinstance(invalid, CallToolResult)
            self.assertTrue(invalid.is_error)

    async def test_search_and_changes_follow_new_reports_without_writing(self) -> None:
        with (
            self.context.database() as db,
            self.assertRaisesRegex(sqlite3.OperationalError, "readonly"),
        ):
            db.execute("DELETE FROM work_reports")
        server = create_server(self.context)
        self.report("Parser compatibility verified", observed_at=300)
        result = await server.call_tool("context_search", {"query": "compatibility"})
        assert isinstance(result, CallToolResult)
        assert result.structured_content is not None
        self.assertEqual(
            result.structured_content["sources"][0]["text"],
            "Parser compatibility verified",
        )
        sources = self.context.select(20, after=250)["sources"]
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["text"], "Parser compatibility verified")
        self.assertEqual(self.context.select(1)["truncated"], True)

    async def test_missing_memory_does_not_create_a_database(self) -> None:
        path = self.path.with_name("missing.sqlite3")
        with self.assertRaises(sqlite3.OperationalError):
            WorkContext(path, "fixture").select(12)
        self.assertFalse(path.exists())

    async def test_tool_reports_do_not_hide_authored_work(self) -> None:
        with self.memory.transaction():
            for index in range(10):
                self.memory._work_repository.append_report(
                    SessionReport(
                        payload=ExecutionEvidence(
                            session=SessionIdentity(
                                source=SessionSource.CODEX, native_id="parser-session"
                            ),
                            event="tool_done",
                            outcome=None,
                            tool_name="Read",
                            progress=None,
                        ),
                        context=ReportContext(checkout="/parser"),
                        observed_at=400 + index,
                        hook_kind="tool_done",
                    )
                )
        selected = self.context.select(2)
        self.assertEqual(len(selected["sources"]), 2)
        report = next(
            item for item in selected["sources"] if item["id"].startswith("report:")
        )
        self.assertEqual(report["text"], "Check parser behavior")
        self.assertEqual(report["provenance"]["source"], "user")

    async def test_repository_refresh_keeps_authored_work_in_current_context(
        self,
    ) -> None:
        with self.memory.transaction():
            for index in range(20):
                self.memory._work_repository.put_observation(
                    CheckoutObservation(
                        worktree=f"/project-{index}",
                        local=None,
                        checked_at=300 + index,
                        error="Unavailable",
                    )
                )
        server = create_server(self.context)
        result = await server.call_tool("current_work_context", {"limit": 12})
        assert isinstance(result, CallToolResult)
        assert result.structured_content is not None
        sources = result.structured_content["sources"]
        self.assertEqual(len(sources), 12)
        self.assertEqual(
            [
                source["text"]
                for source in sources
                if source["id"].startswith("report:")
            ],
            ["Check parser behavior"],
        )
        observations = [
            source for source in sources if not source["id"].startswith("report:")
        ]
        self.assertEqual(len(observations), 11)
        self.assertEqual(observations[0]["provenance"]["checked_at"], 319)
        self.assertTrue(result.structured_content["truncated"])
        self.assertFalse(
            any(
                source["id"].startswith("report:")
                for source in self.context.select(12)["sources"]
            )
        )

    async def test_current_context_fills_available_space_with_reports(self) -> None:
        with self.memory.transaction():
            self.memory.db.execute("DELETE FROM work_checkouts")
        for index in range(15):
            self.report(f"Check parser case {index}", observed_at=300 + index)
        result = await create_server(self.context).call_tool(
            "current_work_context", {"limit": 12}
        )
        assert isinstance(result, CallToolResult)
        assert result.structured_content is not None
        sources = result.structured_content["sources"]
        self.assertEqual(len(sources), 12)
        self.assertTrue(all(source["id"].startswith("report:") for source in sources))
        self.assertEqual(sources[0]["text"], "Check parser case 14")

    async def test_current_context_represents_checkouts_before_repeating_details(
        self,
    ) -> None:
        self.report(
            "Workbench task", observed_at=110, checkout="/workstreams/task/backend"
        )
        self.report("Dotfiles task", observed_at=120, checkout="/home/user/.dotfiles")
        for index in range(30):
            self.report(f"Parser detail {index}", observed_at=300 + index)
        sources = self.context.select(3, reserve_reports=True)["sources"]
        self.assertEqual(
            [source["text"] for source in sources],
            ["Parser detail 29", "Dotfiles task", "Workbench task"],
        )
        self.assertTrue(self.context.select(3, reserve_reports=True)["truncated"])

    async def test_current_context_balances_sessions_without_a_checkout(self) -> None:
        for index in range(20):
            self.report(
                f"Busy turn {index}",
                observed_at=300 + index,
                checkout="",
                session="busy",
            )
        self.report("Other session", observed_at=200, checkout="", session="other")
        sources = self.context.select(3, reserve_reports=True)["sources"]
        self.assertEqual(
            [source["text"] for source in sources],
            ["Busy turn 19", "Other session", "Check parser behavior"],
        )

    async def test_current_context_ranks_closed_and_stale_prs_below_recent_work(
        self,
    ) -> None:
        with self.memory.transaction():
            for number, state, checked_at in (
                (1, PullRequestState.CLOSED, 999),
                (2, PullRequestState.OPEN, 998),
                (3, PullRequestState.OPEN, -700000),
            ):
                self.memory._work_repository.put_observation(
                    PullRequestObservation(
                        ref=f"https://github.com/example/repo/pull/{number}",
                        pr=PullRequest(
                            number=number,
                            url=f"https://github.com/example/repo/pull/{number}",
                            state=state,
                            merged_at=None,
                            is_draft=True,
                            repository=RemoteRepository(
                                value="github.com/example/repo"
                            ),
                        ),
                        checked_at=checked_at,
                    )
                )
        sources = self.context.select(3, reserve_reports=True)["sources"]
        self.assertEqual(sources[0]["text"], "Check parser behavior")
        prs = [
            json.loads(source["text"])["pr"]["number"]
            for source in sources
            if source["id"].startswith("work_pull_requests:")
        ]
        self.assertEqual(prs, [2])
        all_sources = self.context.select(10, reserve_reports=True)["sources"]
        self.assertEqual(json.loads(all_sources[-1]["text"])["pr"]["number"], 1)

    async def test_current_context_groups_branch_observations_with_their_checkout(
        self,
    ) -> None:
        repository = RemoteRepository(value="github.com/example/parser")
        with self.memory.transaction():
            self.memory._work_repository.put_observation(
                CheckoutObservation(
                    worktree="/parser",
                    local=GitObservation(
                        worktree="/parser",
                        branch="feature",
                        head="abc",
                        staged=0,
                        unstaged=1,
                        untracked=0,
                        conflicts=0,
                        upstream="",
                        ahead=0,
                        behind=0,
                        repository=repository,
                    ),
                    checked_at=998,
                )
            )
            self.memory._work_repository.put_observation(
                BranchObservation(
                    repository=repository, branch="feature", pr=None, checked_at=999
                )
            )
        self.report("Another checkout", observed_at=50, checkout="/another")
        sources = self.context.select(2, reserve_reports=True)["sources"]
        self.assertEqual(
            [source["text"] for source in sources],
            ["Check parser behavior", "Another checkout"],
        )

    async def test_failed_refresh_preserves_stale_evidence_rank(self) -> None:
        with self.memory.transaction():
            self.memory._work_repository.put_observation(
                CheckoutObservation(
                    worktree="/stale",
                    local=None,
                    checked_at=999,
                    last_success_at=-700000,
                    error="Unavailable",
                )
            )
        sources = self.context.select(2, reserve_reports=True)["sources"]
        self.assertEqual(
            [source["id"].split(":")[0] for source in sources],
            ["report", "work_checkouts"],
        )
        self.assertEqual(sources[1]["provenance"]["worktree"], "/parser")
