import tempfile
import unittest
from pathlib import Path
from typing import Any, cast

from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import CallToolResult

from pet_daemon.domain import SessionSource
from pet_daemon.identity import SessionIdentity
from pet_daemon.mcp import WorkContext, create_server
from pet_daemon.mcp_companion import CompanionSessions
from pet_daemon.mcp_marks import WorkstreamMarks
from pet_daemon.mcp_work import CompanionContext
from pet_daemon.storage import Memory
from pet_daemon.work.models import (
    IntentEvidence,
    ProgressEvidence,
    ReportContext,
    SessionReport,
)
from pet_daemon.workstream_marks import MarkPatch, MarkStatus, marks_path, thread_id


class WorkstreamMarkToolTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "memory.sqlite3"
        self.now = 1000.0
        self.memory = Memory(self.path, "fixture")
        self.context = WorkContext(self.path, "fixture", clock=lambda: self.now)
        self.work = CompanionContext(self.context)
        self.marks = WorkstreamMarks(self.context)
        for checkout, timestamp in (("/first", 100), ("/second", 200)):
            with self.memory.transaction():
                self.memory._work_repository.append_report(
                    SessionReport(
                        payload=IntentEvidence(
                            session=SessionIdentity(
                                source=SessionSource.CODEX, native_id=checkout
                            ),
                            text=f"Implement {checkout}",
                        ),
                        context=ReportContext(checkout=checkout),
                        observed_at=timestamp,
                    )
                )

    def tearDown(self) -> None:
        self.memory.close()
        self.temporary.cleanup()

    async def test_marked_title_and_goal_keep_field_authors_and_original_evidence(
        self,
    ) -> None:
        id = thread_id("/first")
        before = self.context.select(50)["sources"]
        server = create_server(self.context)
        response = await server.call_tool(
            "mark_workstream",
            {
                "id": id,
                "author": "companion-a",
                "reason": "User clarified intent",
                "title": "Editor setup",
                "goal": "Configure the editor",
                "tags": ["personal"],
            },
        )
        assert isinstance(response, CallToolResult)
        self.assertFalse(response.is_error, response.content)
        self.now += 1
        self.marks.mark(
            id, MarkPatch(status=MarkStatus.BLOCKED), "companion-b", "Needs a choice"
        )
        detail: dict[str, Any] = self.work.workstream(id)
        self.assertEqual(detail["title"], "Editor setup")
        self.assertEqual(detail["title_provenance"]["author"], "companion-a")
        self.assertEqual(detail["title_provenance"]["timestamp"], 1000)
        self.assertEqual(detail["goal"]["text"], "Configure the editor")
        self.assertEqual(detail["captured_goal"]["text"], "Implement /first")
        self.assertEqual(
            detail["mark"]["fields"]["status"]["provenance"]["author"], "companion-b"
        )
        self.assertEqual(self.context.select(50)["sources"], before)
        self.marks.unmark(
            id, ["title", "goal"], "companion-a", "Return to captured wording"
        )
        restored: dict[str, Any] = self.work.workstream(id)
        self.assertEqual(restored["title"], "Implement /first")
        self.assertEqual(restored["title_provenance"]["author"], "simple-pet")
        self.assertEqual(restored["goal"]["text"], "Implement /first")
        self.assertEqual(restored["mark"]["values"]["status"], "blocked")

    async def test_marks_order_and_suppress_workstream_context_and_attention(
        self,
    ) -> None:
        first = thread_id("/first")
        self.marks.mark(
            first,
            MarkPatch(
                priority="focus", status="blocked", blocked_on="Choose an editor"
            ),
            "companion",
            "Needs input",
        )
        streams: dict[str, Any] = self.work.workstreams()
        self.assertEqual(streams["workstreams"][0]["id"], first)
        context = self.context.select(1, reserve_reports=True)
        self.assertEqual(context["sources"][0]["text"], "Implement /first")
        summary = cast(dict[str, Any], context["workstreams"][0])
        self.assertEqual(summary["mark"]["values"]["priority"], "focus")
        attention = CompanionSessions(self.context).attention()
        self.assertEqual(attention["items"][0]["reason"], "Choose an editor")
        self.marks.mark(first, MarkPatch(status="done"), "companion", "Complete")
        self.assertNotIn(first, str(self.work.workstreams()))
        self.assertNotIn(
            "Implement /first", str(self.context.select(50, reserve_reports=True))
        )
        self.assertEqual(CompanionSessions(self.context).attention()["items"], [])
        self.assertEqual(self.work.workstream(first)["title"], "Implement /first")
        self.assertEqual(len(self.context.select(50)["sources"]), 2)

    async def test_snooze_expires_and_history_pages_over_clear_events(self) -> None:
        id = thread_id("/first")
        self.marks.mark(id, MarkPatch(snooze_until=1005), "a", "Later")
        self.assertNotIn(id, str(self.work.workstreams()))
        self.now = 1006
        self.assertIn(id, str(self.work.workstreams()))
        self.marks.unmark(id, None, "b", "Clear all")
        self.marks.mark(id, MarkPatch(title="A title"), "c", "Name it")
        first: dict[str, Any] = self.marks.history(id, limit=1)
        second: dict[str, Any] = self.marks.history(
            id, limit=1, before=first["next_before"]
        )
        third: dict[str, Any] = self.marks.history(
            id, limit=1, before=second["next_before"]
        )
        self.assertEqual(
            [page["events"][0]["author"] for page in (first, second, third)],
            ["c", "b", "a"],
        )
        self.assertEqual(second["events"][0]["operation"], "unmark")
        self.assertIsNone(third["next_before"])

    async def test_unknown_thread_and_invalid_marks_do_not_create_storage(self) -> None:
        server = create_server(self.context)
        for arguments in (
            {"id": "unknown", "status": "done"},
            {"id": thread_id("/first"), "goal": "two\nlines"},
            {"id": thread_id("/first"), "status": "invalid"},
            {"id": thread_id("/first")},
        ):
            with self.assertRaises(ToolError):
                await server.call_tool(
                    "mark_workstream",
                    {"author": "companion", "reason": "Test", **arguments},
                )
        self.assertFalse(marks_path(self.path).exists())

    async def test_focus_keeps_checkout_representatives_before_details(self) -> None:
        first = thread_id("/first")
        for index in range(10):
            with self.memory.transaction():
                self.memory._work_repository.append_report(
                    SessionReport(
                        payload=ProgressEvidence(
                            session=SessionIdentity(
                                source=SessionSource.CODEX, native_id="/first"
                            ),
                            text=f"Detail {index}",
                        ),
                        context=ReportContext(checkout="/first"),
                        observed_at=110 + index,
                    )
                )
        self.marks.mark(first, MarkPatch(priority="focus"), "companion", "Focus here")
        result = self.context.select(2, reserve_reports=True)
        self.assertEqual(
            [source["text"] for source in result["sources"]],
            ["Detail 9", "Implement /second"],
        )
        self.assertEqual(
            [summary["id"] for summary in result["workstreams"]],
            [first, thread_id("/second")],
        )

    async def test_unmark_remains_available_after_captured_evidence_is_cleared(
        self,
    ) -> None:
        id = thread_id("/first")
        self.marks.mark(id, MarkPatch(status="done"), "companion", "Complete")
        self.memory.work.clear()
        self.marks.unmark(id, None, "companion", "Reset this judgment")
        self.assertEqual(self.marks.store.effective(id), {})
        self.assertEqual(len(self.marks.store.history(id)), 2)
