import json
import random
import tempfile
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Unpack

from pet_daemon.attention import Attention
from pet_daemon.commentary import Cognition
from pet_daemon.commentary.context import CommentaryPayload, PreparedCommentary
from pet_daemon.commentary.contracts import ModelRequest
from pet_daemon.config import Config
from pet_daemon.runtime import WorkSnapshot
from pet_daemon.storage import Memory
from pet_daemon.work import WorkSubject, build_work
from pet_daemon.work.observations import (
    GitObservation,
    PullRequest,
    RepositoryObservation,
)
from Tests.fixtures import (
    SessionOptions,
    capture_snapshot,
    session_snapshot,
    sessions_snapshot,
)


class NoRequestsModel:
    async def complete(self, request: ModelRequest, reason: str) -> dict[str, object]:
        raise AssertionError("Context preparation must not request model output")


class ContextTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = Config(pet_dir=Path(self.temp.name))
        self.memory = Memory(
            Path(self.temp.name) / "memory.sqlite3", "pet", clock=lambda: 1000
        )
        self.addCleanup(self.memory.close)
        self.cognition = Cognition(
            NoRequestsModel(), self.config, clock=lambda: 1000, memory=self.memory
        )
        self.attention = Attention(self.config, lambda: 1000, random.Random(0))
        self.snapshot = sessions_snapshot(
            session_snapshot(
                id="cli:parser",
                turn_active=True,
                generation=1,
                recent_requests=(
                    "Explore parsing options",
                    "Keep the public API stable",
                ),
                progress="Checking parser compatibility",
                session_context="Implement the parser.",
            )
        )
        self.work: tuple[WorkSubject, ...] = ()

    def update_session(self, index: int = 0, **values: Unpack[SessionOptions]) -> None:
        sessions = list(self.snapshot.sessions)
        sessions[index] = replace(sessions[index], **values)
        self.snapshot = replace(self.snapshot, sessions=tuple(sessions))

    def current_snapshot(self) -> WorkSnapshot:
        return capture_snapshot(self.snapshot, work=self.work)

    def prepare(
        self, reason: str = "routine", context: str | None = None
    ) -> PreparedCommentary:
        selection = self.attention.select(self.current_snapshot(), reason, context)
        return self.cognition.context.prepare(
            self.current_snapshot(), selection, self.cognition.request
        )

    def payload(self, prepared: PreparedCommentary) -> CommentaryPayload:
        return json.loads(prepared.request["messages"][0]["content"])

    def test_small_budget_keeps_intent_latest_request_and_progress(self) -> None:
        self.config.prompt_target_bytes = 1
        before = deepcopy(self.snapshot)
        prepared = self.prepare()
        focus = self.payload(prepared)["focus_session"]
        assert focus is not None
        self.assertEqual(focus["recent_requests"], ["Keep the public API stable"])
        self.assertEqual(focus["session_context"], "Implement the parser.")
        self.assertEqual(focus["progress"], "Checking parser compatibility")
        self.assertEqual(self.snapshot, before)

    def test_available_budget_keeps_requests_in_chronological_order(self) -> None:
        focus = self.payload(self.prepare())["focus_session"]
        assert focus is not None
        self.assertEqual(
            focus["recent_requests"], list(self.snapshot.sessions[0].recent_requests)
        )

    def test_manifest_ignores_an_omitted_session_changing(self) -> None:
        self.config.prompt_target_bytes = 1
        self.snapshot = replace(
            self.snapshot,
            sessions=(
                *self.snapshot.sessions,
                session_snapshot(
                    id="cli:renderer", turn_active=True, progress="Checking layout"
                ),
            ),
        )
        prepared = self.prepare()
        self.assertEqual(prepared.manifest.session_ids, ("cli:parser",))
        self.update_session(1, progress="Renderer completed", turn_active=False)
        self.assertTrue(prepared.manifest.valid(self.current_snapshot()))

    def test_manifest_ignores_a_request_omitted_by_the_budget(self) -> None:
        self.config.prompt_target_bytes = 1
        prepared = self.prepare()
        self.update_session(
            recent_requests=(
                "Discarded exploration",
                *self.snapshot.sessions[0].recent_requests[1:],
            )
        )
        self.assertTrue(prepared.manifest.valid(self.current_snapshot()))

    def test_manifest_rejects_changed_included_progress(self) -> None:
        prepared = self.prepare()
        self.update_session(progress="Parser verification complete")
        self.assertFalse(prepared.manifest.valid(self.current_snapshot()))

    def test_manifest_ignores_elapsed_time_and_metadata(self) -> None:
        prepared = self.prepare()
        self.update_session(duration_seconds=300, quiet_seconds=150, changed_at=1200)
        self.assertTrue(prepared.manifest.valid(self.current_snapshot()))

    def test_process_presence_metadata_does_not_supersede_ongoing_work(self) -> None:
        self.update_session(presence="unknown")
        prepared = self.prepare()
        self.update_session(presence="alive")
        self.assertTrue(prepared.manifest.valid(self.current_snapshot()))

    def test_quiet_connected_work_keeps_its_intent_without_a_live_session(self) -> None:
        self.memory.work.session_intent("cli:parser", "Preserve parser compatibility")
        self.snapshot = replace(self.snapshot, sessions=())
        self.work = build_work(self.snapshot, self.memory.work.snapshot())
        prepared = self.prepare()
        payload = self.payload(prepared)
        self.assertIsNone(payload["focus_session"])
        self.assertEqual(
            payload["focus_work"]["intent"], "Preserve parser compatibility"
        )
        self.assertTrue(prepared.manifest.valid(self.current_snapshot()))

    def test_manifest_rejects_a_superseded_runtime_transition(self) -> None:
        prepared = self.prepare()
        self.update_session(turn_active=False, outcome="success")
        self.assertFalse(prepared.manifest.valid(self.current_snapshot()))

    def test_quiet_greeting_remains_valid_without_session_context(self) -> None:
        self.snapshot = replace(self.snapshot, sessions=())
        prepared = self.prepare("click")
        self.assertTrue(prepared.manifest.valid(self.current_snapshot()))

    def test_explicit_context_stays_in_data_and_keeps_work_out_of_the_prompt(
        self,
    ) -> None:
        prepared = self.prepare(context="Inspect the parser result")
        payload = self.payload(prepared)
        self.assertEqual(payload["context"], "Inspect the parser result")
        self.assertIsNone(payload["focus_session"])
        self.assertNotIn(
            "Inspect the parser result", json.dumps(prepared.request["system"])
        )

    def output(
        self,
        *,
        review: bool = True,
        head: str = "one",
        checks: tuple[tuple[str, str], ...] = (),
    ) -> None:
        if not self.memory.work.work_for_session("cli:parser"):
            self.memory.work.session_intent(
                "cli:parser", "Make session tracking survive restarts"
            )
        self.memory.work.observe_repository(
            RepositoryObservation(
                subject="parser",
                repository="example.test/team/parser",
                worktree="/parser",
                branch="persistence",
                sessions=("cli:parser",),
                current=True,
                local_checked_at=1000,
                github_checked_at=1000,
                local=GitObservation(
                    "/parser", "persistence", head, 0, 7, 0, 0, "", 0, 0
                ),
                pr=PullRequest(
                    42,
                    "https://example.test/team/parser/pull/42",
                    "OPEN",
                    None,
                    "",
                    tuple(checks),
                    is_draft=True,
                    title="Persist coding session context",
                    head=head,
                    check_revision=head,
                    head_repository="example.test/team/parser",
                    head_branch="persistence",
                    review_requested=review,
                ),
            )
        )
        self.work = build_work(self.snapshot, self.memory.work.snapshot())

    def test_presented_work_omits_checkout_paths_and_internal_session_names(
        self,
    ) -> None:
        self.output()
        encoded = json.dumps(self.payload(self.prepare()))
        self.assertNotIn('"checkout"', encoded)
        self.assertNotIn('"/parser"', encoded)
        self.assertNotIn("cli:parser", encoded)
        self.assertIn("https://example.test/team/parser/pull/42", encoded)

    def test_review_withdrawal_invalidates_the_selected_action(self) -> None:
        self.output()
        prepared = self.prepare()
        assert prepared.selection.action is not None
        self.assertEqual(prepared.selection.action.kind, "review_output")
        self.output(review=False)
        self.assertFalse(prepared.manifest.valid(self.current_snapshot()))

    def test_resolved_checks_invalidate_the_selected_obstacle(self) -> None:
        self.output(review=False, checks=(("tests", "FAILURE"),))
        prepared = self.prepare()
        assert prepared.selection.action is not None
        self.assertEqual(prepared.selection.action.kind, "resolve_checks")
        self.output(review=False, checks=(("tests", "SUCCESS"),))
        self.assertFalse(prepared.manifest.valid(self.current_snapshot()))

    def test_superseded_head_invalidates_a_selected_action(self) -> None:
        self.output()
        prepared = self.prepare()
        self.output(head="two")
        self.assertFalse(prepared.manifest.valid(self.current_snapshot()))
