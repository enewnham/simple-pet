import json
import tempfile
import unittest
from pathlib import Path
from typing import Literal

from mcp.types import CallToolResult

from pet_daemon.domain import ExecutionOutcome, PullRequestState, SessionSource
from pet_daemon.identity import SessionIdentity
from pet_daemon.mcp import WorkContext, create_server
from pet_daemon.mcp_companion import CompanionSessions
from pet_daemon.mcp_sessions import LiveSession, SessionSnapshot, snapshot_path
from pet_daemon.repository_identity import RemoteRepository
from pet_daemon.storage import Memory
from pet_daemon.work.models import (
    ExecutionEvidence,
    IntentEvidence,
    ProgressEvidence,
    ReportContext,
    SessionReport,
)
from pet_daemon.work.observations import PullRequest, PullRequestObservation


class CompanionSessionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "memory.sqlite3"
        self.now = 100000.0
        self.memory = Memory(self.path, "fixture")
        self.context = WorkContext(self.path, "fixture", clock=lambda: self.now)
        self.companion = CompanionSessions(self.context)
        self.identity = SessionIdentity(
            source=SessionSource.CLAUDE, native_id="session"
        )

    def tearDown(self) -> None:
        self.memory.close()
        self.temporary.cleanup()

    def snapshot(
        self,
        state: Literal["working", "idle", "waiting_on_permission"] = "working",
        waiting_kind: str | None = None,
        checkout: str = "/home/user/.dotfiles",
    ) -> None:
        snapshot = SessionSnapshot(
            pet="fixture",
            observed_at=self.now,
            sessions=(
                LiveSession(
                    id=self.identity,
                    checkout=checkout,
                    observed_at=self.now,
                    state=state,
                    presence_checked_at=self.now,
                    activity_uncertain=False,
                    waiting_kind=waiting_kind,
                    waiting_prompt="May I continue?" if waiting_kind else None,
                    waiting_tool=None,
                    outcome=None,
                    failure=None,
                    goal="Configure editor",
                    progress="",
                    last_activity_at=self.now - 10,
                ),
            ),
        )
        snapshot_path(self.path).write_text(snapshot.model_dump_json())

    def report(self, timestamp: float, *, failure: bool = False) -> None:
        payload = (
            ExecutionEvidence(
                session=self.identity,
                event="turn_error",
                outcome=ExecutionOutcome.ERROR,
                tool_name=None,
                progress="Build failed",
            )
            if failure
            else IntentEvidence(session=self.identity, text="Configure editor")
        )
        with self.memory.transaction():
            self.memory._work_repository.append_report(
                SessionReport(
                    payload=payload,
                    context=ReportContext(
                        checkout="/home/user/.dotfiles",
                        pull_request="https://github.com/example/repo/pull/1",
                    ),
                    observed_at=timestamp,
                )
            )

    async def test_live_sessions_join_reports_and_keep_both_provenances(self) -> None:
        self.report(self.now - 100)
        self.snapshot()
        selected = self.companion.sessions()
        self.assertTrue(selected["live_available"])
        session = selected["sessions"][0]
        self.assertEqual(session["state"], "working")
        self.assertEqual(session["checkout"], "/home/user/.dotfiles")
        self.assertEqual(
            [source["provenance"]["basis"] for source in session["sources"]],
            ["observed", "reported"],
        )
        self.assertEqual(session["sources"][0]["provenance"]["observed_at"], self.now)
        self.assertEqual(
            session["sources"][1]["provenance"]["observed_at"], self.now - 100
        )
        self.assertEqual(
            self.context.read(session["id"], 0, 12000)["id"], session["id"]
        )
        self.now += 16
        expired = self.companion.sessions()
        self.assertFalse(expired["live_available"])
        self.assertEqual(expired["sessions"], [])
        with self.assertRaisesRegex(ValueError, "no longer retained"):
            self.context.read(session["id"], 0, 100)

    async def test_attention_uses_live_permission_and_question_states(self) -> None:
        self.snapshot("waiting_on_permission", "permission")
        permission = self.companion.attention()["items"][0]
        self.assertEqual(permission["kind"], "permission_pending")
        self.assertFalse(permission["candidate"])
        self.snapshot("idle", "question")
        question = self.companion.attention()["items"][0]
        self.assertEqual(question["kind"], "unanswered_question")
        self.snapshot("working")
        self.assertEqual(self.companion.attention()["items"], [])
        self.now += 16
        self.assertFalse(self.companion.attention()["live_available"])
        self.assertEqual(self.companion.attention()["items"], [])

    async def test_failure_candidate_clears_when_the_user_starts_another_turn(
        self,
    ) -> None:
        self.report(self.now - 20, failure=True)
        failure = self.companion.attention()["items"][0]
        self.assertEqual(failure["kind"], "failed_turn")
        self.assertTrue(failure["candidate"])
        execution = failure["sources"][0]["provenance"]["execution"]
        assert isinstance(execution, dict)
        self.assertEqual(execution["outcome"], "error")
        self.report(self.now)
        self.assertEqual(self.companion.attention()["items"], [])

    async def test_stalled_pr_is_an_evidence_backed_candidate_and_closed_pr_clears_it(
        self,
    ) -> None:
        self.report(self.now - 90000)
        pr = PullRequest(
            number=1,
            url="https://github.com/example/repo/pull/1",
            state=PullRequestState.OPEN,
            merged_at=None,
            repository=RemoteRepository(value="github.com/example/repo"),
        )
        with self.memory.transaction():
            self.memory._work_repository.put_observation(
                PullRequestObservation(ref=pr.url, pr=pr, checked_at=self.now)
            )
        items = self.companion.attention()["items"]
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["kind"], "stalled_pr")
        self.assertTrue(items[0]["candidate"])
        self.assertEqual(
            [source["provenance"]["basis"] for source in items[0]["sources"]],
            ["observed", "reported"],
        )
        self.snapshot()
        self.assertEqual(self.companion.attention()["items"], [])
        snapshot_path(self.path).unlink()
        with self.memory.transaction():
            self.memory._work_repository.put_observation(
                PullRequestObservation(
                    ref=pr.url,
                    pr=pr.model_copy(update={"state": PullRequestState.CLOSED}),
                    checked_at=self.now + 1,
                )
            )
        self.assertEqual(self.companion.attention()["items"], [])

    async def test_new_tools_return_structured_read_only_results(self) -> None:
        self.report(self.now)
        self.snapshot()
        server = create_server(self.context)
        for name, arguments in (
            ("workstreams", {}),
            ("since", {"ts": 0}),
            ("sessions", {}),
            ("attention", {}),
            ("decisions", {"query": "editor"}),
        ):
            with self.subTest(tool=name):
                result = await server.call_tool(name, arguments)
                assert isinstance(result, CallToolResult)
                self.assertFalse(result.is_error, result.content)
                self.assertIsNotNone(result.structured_content)
                self.assertNotIn("Other pet", json.dumps(result.structured_content))
        result = self.companion.work.workstreams()
        threads = result["workstreams"]
        assert isinstance(threads, list)
        detail = await server.call_tool("workstream", {"id": threads[0]["id"]})
        assert isinstance(detail, CallToolResult)
        self.assertFalse(detail.is_error, detail.content)

    async def test_idle_question_candidate_clears_after_a_later_user_intent(
        self,
    ) -> None:
        with self.memory.transaction():
            self.memory._work_repository.append_report(
                SessionReport(
                    payload=ProgressEvidence(
                        session=self.identity, text="Which editor should I configure?"
                    ),
                    context=ReportContext(checkout="/home/user/.dotfiles"),
                    observed_at=self.now - 20,
                )
            )
        self.snapshot("idle")
        item = self.companion.attention()["items"][0]
        self.assertEqual(item["kind"], "unanswered_question")
        self.assertTrue(item["candidate"])
        self.assertEqual(len(item["sources"]), 2)
        self.report(self.now)
        self.assertEqual(self.companion.attention()["items"], [])

    async def test_failure_in_a_former_checkout_clears_after_session_moves(
        self,
    ) -> None:
        self.report(self.now - 20, failure=True)
        with self.memory.transaction():
            self.memory._work_repository.append_report(
                SessionReport(
                    payload=IntentEvidence(
                        session=self.identity, text="Work in another checkout"
                    ),
                    context=ReportContext(checkout="/another"),
                    observed_at=self.now,
                )
            )
        self.assertEqual(self.companion.attention()["items"], [])

    async def test_enclosing_checkout_marks_apply_to_live_permission_items(
        self,
    ) -> None:
        from pet_daemon.workstream_marks import MarkPatch, MarkStore, thread_id

        self.report(self.now - 100)
        self.snapshot(
            "waiting_on_permission",
            "permission",
            checkout="/home/user/.dotfiles/editor",
        )
        marks = MarkStore(self.path, "fixture", clock=lambda: self.now)
        id = thread_id("/home/user/.dotfiles")
        marks.append(
            id,
            MarkPatch(status="done", title="Editor setup"),
            "companion",
            "Hide this thread",
        )
        self.assertEqual(self.companion.attention()["items"], [])
        marks.unmark(id, "companion", "Show again", fields=["status"])
        item = self.companion.attention()["items"][0]
        self.assertEqual(item["workstream_id"], id)
        self.assertEqual(item["title"], "Editor setup")
