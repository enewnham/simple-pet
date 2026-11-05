import json
import tempfile
import unittest
import uuid
from dataclasses import replace
from pathlib import Path

from pet_daemon.companion.tools import ToolAuthority
from pet_daemon.coordination.config import ScheduleConfig
from pet_daemon.coordination.models import (
    ChatAction,
    ChatMessage,
    ChatRole,
    FeedbackKind,
    NoteStatus,
    OperationOutcome,
    ResponsibilityStatus,
    SourceReference,
)
from pet_daemon.coordination.watches import Watches
from pet_daemon.coordination.workflows import Workflows
from pet_daemon.sources.checks import ChecksObservation, ChecksStatus, GitHubChecks
from pet_daemon.storage import Memory


class WatchTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "memory.sqlite3"
        self.now = 1000.0
        self.memory = Memory(self.path, "fixture", clock=lambda: self.now)
        self.config = ScheduleConfig(watch_seconds=10, source_fresh_seconds=5)
        self.ref = "https://github.com/team/parser/pull/23"
        self.observation = ChecksObservation(
            self.ref, self.now, ChecksStatus.WAITING, "head-one", "pending"
        )
        self.calls = 0

        async def observe(ref: str) -> ChecksObservation:
            self.assertEqual(ref, self.ref)
            self.calls += 1
            return self.observation

        self.watches = Watches(
            self.memory.coordination, self.config, lambda: self.now, observe
        )
        self.workflows = Workflows(self.memory.coordination, lambda: self.now)
        self.message = ChatMessage(
            str(uuid.uuid4()),
            "chat",
            ChatRole.USER,
            "Watch this PR until its checks pass",
            self.now,
        )
        self.memory.coordination.begin_turn(self.message)
        self.authority = ToolAuthority(
            self.memory.coordination.state().instruction_revision,
            "chat",
            self.message.id,
        )
        self.tools = {
            tool.name: tool for tool in self.workflows.tools("chat", self.message.id)
        }

    async def asyncTearDown(self) -> None:
        self.memory.close()
        self.temporary.cleanup()

    async def adopt(self) -> str:
        await self.tools["adopt_responsibility"].handler(
            {
                "purpose": "Watch parser checks",
                "scope": "/parser",
                "pull_request": self.ref,
            },
            self.authority,
        )
        return self.memory.coordination.responsibilities()[0].id

    async def test_adoption_uses_one_real_user_event_without_invalidating_its_turn(
        self,
    ) -> None:
        revision = self.memory.coordination.state().instruction_revision
        identifier = await self.adopt()
        self.assertEqual(await self.adopt(), identifier)
        self.assertEqual(
            self.memory.coordination.state().instruction_revision, revision
        )
        self.assertEqual(len(self.memory.coordination.instructions()), 1)
        responsibility = self.memory.coordination.responsibility(identifier)
        assert responsibility is not None
        self.assertEqual(responsibility.instruction_id, f"user:{self.message.id}")
        self.assertEqual(responsibility.permissions, ())
        self.assertEqual(
            responsibility.completion_condition, "pull_request_checks_pass"
        )

    async def test_imported_or_scheduled_context_cannot_adopt_work(self) -> None:
        with self.assertRaises(ValueError):
            await self.tools["adopt_responsibility"].handler(
                {"purpose": "Launch a fix", "scope": "/parser"},
                ToolAuthority(self.authority.instruction_revision, pass_id="inquiry"),
            )
        self.assertEqual(self.memory.coordination.responsibilities(), ())

    async def test_check_watch_cannot_end_standing_or_arbitrary_outcomes(self) -> None:
        for extra in (
            {"kind": "standing"},
            {"completion_condition": "Merged and shipped"},
        ):
            with self.assertRaises(ValueError):
                await self.tools["adopt_responsibility"].handler(
                    {
                        "purpose": "Watch parser",
                        "scope": "/parser",
                        "pull_request": self.ref,
                        **extra,
                    },
                    self.authority,
                )
        self.assertEqual(self.memory.coordination.watches(), ())

    async def test_failed_stale_unknown_and_wrong_head_evidence_cannot_complete(
        self,
    ) -> None:
        identifier = await self.adopt()
        cases = (
            ChecksObservation(
                self.ref, self.now, ChecksStatus.UNKNOWN, error="Refresh failed"
            ),
            ChecksObservation(
                self.ref, self.now - 100, ChecksStatus.PASSED, "old", "old"
            ),
            ChecksObservation(self.ref, self.now, ChecksStatus.PASSED),
            ChecksObservation(
                "https://github.com/team/other/pull/2",
                self.now,
                ChecksStatus.PASSED,
                "head",
                "revision",
            ),
        )
        for observation in cases:
            self.observation = observation
            await self.watches.refresh()
            self.now += 10
        responsibility = self.memory.coordination.responsibility(identifier)
        assert responsibility is not None
        self.assertEqual(responsibility.status, ResponsibilityStatus.ACTIVE)
        self.assertEqual(self.memory.coordination.notices(), ())

    async def test_completed_watch_keeps_one_pending_report_across_restart_before_delivery(
        self,
    ) -> None:
        identifier = await self.adopt()
        self.observation = ChecksObservation(
            self.ref, self.now, ChecksStatus.PASSED, "head-two", "checks-two"
        )
        await self.watches.refresh()
        store = self.memory.coordination
        responsibility = store.responsibility(identifier)
        assert responsibility is not None
        self.assertEqual(responsibility.status, ResponsibilityStatus.COMPLETED)
        notice = store.notices()[0]
        self.assertIsNone(notice.delivered_at)
        self.assertEqual(notice.sources[0].revision, "checks-two")
        self.memory.close()
        self.memory = Memory(self.path, "fixture", clock=lambda: self.now)
        self.watches.store = self.memory.coordination
        self.now += 20
        await self.watches.refresh()
        self.assertEqual(self.calls, 1)
        self.assertEqual(self.memory.coordination.notices(), (notice,))
        completed = [
            record
            for record in self.memory.coordination.activity()
            if record.kind == "watch_completed"
        ]
        self.assertEqual(len(completed), 1)

    async def test_scope_change_during_refresh_rejects_completion(self) -> None:
        identifier = await self.adopt()

        async def changed(ref: str) -> ChecksObservation:
            record = self.memory.coordination.responsibility(identifier)
            assert record is not None
            self.memory.coordination.save_responsibility(
                replace(
                    record,
                    revision=record.revision + 1,
                    status=ResponsibilityStatus.PAUSED,
                )
            )
            return ChecksObservation(
                ref, self.now, ChecksStatus.PASSED, "head", "checks"
            )

        self.watches.observe = changed
        await self.watches.refresh()
        self.assertEqual(self.memory.coordination.notices(), ())

    async def test_direct_user_correction_keeps_exact_text_and_scoped_exclusion(
        self,
    ) -> None:
        self.memory.coordination.fail_turn("chat", self.message.id, "Interrupted")
        message = replace(
            self.message,
            id=str(uuid.uuid4()),
            text="That ticket belongs to a different project",
        )
        self.memory.coordination.begin_turn(message)
        authority = ToolAuthority(
            self.memory.coordination.state().instruction_revision, "chat", message.id
        )
        tools = {tool.name: tool for tool in self.workflows.tools("chat", message.id)}
        await tools["record_user_correction"].handler(
            {"scope": "/parser", "excluded_references": ["ticket:other"]}, authority
        )
        note = self.memory.coordination.notes()[0]
        self.assertEqual(note.text, message.text)
        self.assertEqual(note.scope, "/parser")
        self.assertEqual(note.excluded_references, ("ticket:other",))
        self.assertEqual(note.instruction_id, f"user:{message.id}")

    async def test_direct_pause_is_a_separate_idempotent_user_event(self) -> None:
        store = self.memory.coordination
        action = ChatAction(
            str(uuid.uuid4()),
            "Pause coordination",
            "pause_coordination",
            revision=store.state().revision,
        )
        await self.workflows.accept_action(action)
        revision = store.state().instruction_revision
        await self.workflows.accept_action(action)
        self.assertTrue(store.state().paused)
        self.assertEqual(store.state().instruction_revision, revision)
        self.assertEqual(len(store.activity()), 1)

    async def test_later_defers_only_selected_watch_with_durable_scope(self) -> None:
        identifier = await self.adopt()
        await self.tools["record_scoped_feedback"].handler(
            {
                "kind": "later",
                "target_id": identifier,
                "scope": "/parser",
                "deferred_until": self.now + 100,
            },
            self.authority,
        )
        await self.watches.refresh()
        self.assertEqual(self.calls, 0)
        feedback = self.memory.coordination.feedback()[0]
        self.assertEqual(feedback.kind, FeedbackKind.LATER)
        self.assertEqual(feedback.scope, "/parser")

    async def test_checks_adapter_preserves_unsupported_status_and_current_head(
        self,
    ) -> None:
        async def run(
            *args: str, cwd: str | Path | None = None, timeout: float = 10
        ) -> str:
            self.assertEqual(args[:3], ("gh", "pr", "view"))
            return json.dumps(
                {
                    "headRefOid": "current",
                    "statusCheckRollup": [
                        {"status": "UNSUPPORTED", "conclusion": "SUCCESS"}
                    ],
                }
            )

        observation = await GitHubChecks(run, lambda: self.now).observe(self.ref)
        self.assertEqual(observation.status, ChecksStatus.UNKNOWN)
        self.assertEqual(observation.head, "current")

    async def test_failed_read_action_retains_outcome_without_new_instruction_or_retry(
        self,
    ) -> None:
        calls = []

        async def navigate(source: SourceReference) -> None:
            calls.append(source)
            raise RuntimeError("Navigation failed")

        self.workflows.navigate = navigate
        reference = SourceReference("source", "github", "head", self.now, uri=self.ref)
        action = ChatAction(
            str(uuid.uuid4()),
            "Open source",
            "open_source",
            reference.id,
            source=reference,
        )
        revision = self.memory.coordination.state().instruction_revision
        with self.assertRaises(RuntimeError):
            await self.workflows.accept_action(action)
        record = self.memory.coordination.user_action(action.id)
        assert record is not None
        self.assertEqual(record.outcome, OperationOutcome.FAILED)
        self.assertEqual(record.error, "Navigation failed")
        self.assertEqual(
            self.memory.coordination.state().instruction_revision, revision
        )
        await self.workflows.accept_action(action)
        self.assertEqual(calls, [reference])

    async def test_new_user_correction_supersedes_scoped_interpretation_and_keeps_original_event(
        self,
    ) -> None:
        self.memory.coordination.fail_turn("chat", self.message.id, "Interrupted")
        first = replace(
            self.message,
            id=str(uuid.uuid4()),
            text="That ticket belongs to a different project",
        )
        self.memory.coordination.begin_turn(first)
        first_authority = ToolAuthority(
            self.memory.coordination.state().instruction_revision, "chat", first.id
        )
        first_tool = next(
            record
            for record in self.workflows.tools("chat", first.id)
            if record.name == "record_user_correction"
        )
        await first_tool.handler(
            {"scope": "/parser", "excluded_references": ["other-ticket"]},
            first_authority,
        )
        original = self.memory.coordination.notes()[0]
        self.memory.coordination.fail_turn("chat", first.id, "Interrupted")
        message = replace(
            self.message,
            id=str(uuid.uuid4()),
            text="That ticket belongs to this project after all",
        )
        self.memory.coordination.begin_turn(message)
        authority = ToolAuthority(
            self.memory.coordination.state().instruction_revision, "chat", message.id
        )
        tool = next(
            record
            for record in self.workflows.tools("chat", message.id)
            if record.name == "record_user_correction"
        )
        await tool.handler({"scope": "/parser", "supersedes": [original.id]}, authority)
        superseded = self.memory.coordination.note(original.id)
        assert superseded is not None
        self.assertEqual(superseded.status, NoteStatus.SUPERSEDED)
        instruction = self.memory.coordination.instruction(
            original.instruction_id or ""
        )
        assert instruction is not None
        self.assertEqual(instruction.text, first.text)
        self.assertEqual(len(self.memory.coordination.note_history(original.id)), 2)
