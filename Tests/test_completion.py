import asyncio
import tempfile
import unittest
import uuid
from dataclasses import replace
from pathlib import Path

from pet_daemon.companion.tools import ToolAuthority
from pet_daemon.coordination.config import ScheduleConfig
from pet_daemon.coordination.models import (
    CandidateCause,
    CandidateDecision,
    ChatMessage,
    ChatRole,
    CompletionAssessment,
    CompletionBasis,
    Disposition,
    InquiryPass,
    InquiryResult,
    Instruction,
    Responsibility,
    ResponsibilityKind,
    ResponsibilityStatus,
    SourceReference,
)
from pet_daemon.coordination.runtime import Coordinator
from pet_daemon.coordination.workflows import Workflows
from pet_daemon.storage import Memory
from pet_daemon.storage.coordination import RevisionConflict


class CompletionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "memory.sqlite3"
        self.now = 1000.0
        self.memory = Memory(self.path, "fixture", clock=lambda: self.now)
        store = self.memory.coordination
        store.save_instruction(
            Instruction("adopt", "Find the rollback owner", "migration", self.now)
        )
        self.responsibility = Responsibility(
            "owner",
            "Find the rollback owner",
            "migration",
            "adopt",
            self.now,
            completion_condition="The rollback owner is identified",
        )
        store.save_responsibility(self.responsibility)
        self.source = SourceReference(
            "decision", "meetings", "decision-one", self.now, scope="migration"
        )
        self.assessment = CompletionAssessment(
            "owner",
            1,
            self.responsibility.completion_condition,
            "The decision identifies Sam as rollback owner.",
            CompletionBasis.OBSERVED,
            (self.source,),
        )
        self.requests: list[InquiryPass] = []

        async def reason(request: InquiryPass) -> InquiryResult:
            self.requests.append(request)
            candidate = request.candidates[0]
            return InquiryResult(
                request.id,
                (
                    CandidateDecision(
                        candidate.id,
                        Disposition.COMPLETE,
                        self.assessment.claim,
                        sources=self.assessment.sources,
                        completion=self.assessment,
                    ),
                ),
            )

        self.coordinator = Coordinator(
            store,
            ScheduleConfig(),
            lambda: self.now,
            reason,
            sources_current=lambda references: references == (self.source,),
            completion_sources_valid=lambda references: references == (self.source,),
        )
        self.candidate = self.coordinator.queue(
            self.responsibility.purpose,
            "migration",
            CandidateCause("source", self.source.id, self.source.revision, self.now),
            responsibility_id="owner",
        )

    async def asyncTearDown(self) -> None:
        await self.coordinator.close()
        self.memory.close()
        self.temporary.cleanup()

    async def test_observed_finite_condition_completes_with_evidence_and_one_durable_report(
        self,
    ) -> None:
        await self.coordinator.run_pass()
        store = self.memory.coordination
        completed = store.responsibility("owner")
        assert completed is not None
        self.assertEqual(completed.status, ResponsibilityStatus.COMPLETED)
        self.assertEqual(store.completions()[0].assessment, self.assessment)
        self.assertEqual(store.considerations()[0].disposition, Disposition.COMPLETE)
        notice = store.notices()[0]
        self.assertIsNone(notice.delivered_at)
        self.assertEqual(notice.sources, (self.source,))
        self.assertIsNone(await self.coordinator.run_pass())
        self.memory.close()
        self.memory = Memory(self.path, "fixture", clock=lambda: self.now)
        self.assertEqual(self.memory.coordination.notices(), (notice,))

    async def test_reported_and_inferred_outcomes_remain_unverified_without_completion(
        self,
    ) -> None:
        for basis in (CompletionBasis.REPORTED, CompletionBasis.INFERRED):
            self.assessment = replace(self.assessment, basis=basis)
            await self.coordinator.run_pass()
            current = self.memory.coordination.responsibility("owner")
            assert current is not None
            self.assertEqual(current.status, ResponsibilityStatus.ACTIVE)
            self.now += 1000
        self.assertEqual(self.memory.coordination.completions(), ())
        self.assertEqual(self.memory.coordination.notices(), ())
        self.assertTrue(
            all(
                record.disposition == Disposition.WAIT
                for record in self.memory.coordination.considerations()
            )
        )

    async def test_stale_or_report_sources_cannot_be_promoted_to_observed_completion(
        self,
    ) -> None:
        self.coordinator.completion_sources_valid = lambda sources: False
        with self.assertRaises(ValueError):
            await self.coordinator.run_pass()
        self.assertEqual(self.memory.coordination.completions(), ())
        self.assertEqual(self.memory.coordination.notices(), ())

    async def test_superseded_condition_and_wrong_scope_are_rejected(self) -> None:
        self.assessment = replace(self.assessment, responsibility_revision=2)
        with self.assertRaises(RevisionConflict):
            await self.coordinator.run_pass()
        self.assessment = replace(
            self.assessment, responsibility_id="another", responsibility_revision=1
        )
        self.now += self.coordinator.config.failure_retry_seconds
        with self.assertRaises(RevisionConflict):
            await self.coordinator.run_pass()
        self.assertEqual(self.memory.coordination.completions(), ())

    async def test_human_confirmation_is_separate_and_can_end_a_standing_purpose(
        self,
    ) -> None:
        store = self.memory.coordination
        store.save_responsibility(
            replace(self.responsibility, revision=2, kind=ResponsibilityKind.STANDING)
        )
        message = ChatMessage(
            str(uuid.uuid4()),
            "chat",
            ChatRole.USER,
            "This standing purpose can end. I am done with it.",
            self.now,
        )
        store.begin_turn(message)
        authority = ToolAuthority(
            store.state().instruction_revision, "chat", message.id
        )
        workflow = Workflows(store, lambda: self.now)
        tool = next(
            tool
            for tool in workflow.tools("chat", message.id)
            if tool.name == "confirm_responsibility_complete"
        )
        await tool.handler({"id": "owner", "expected_revision": 2}, authority)
        record = store.completions()[0]
        self.assertEqual(record.assessment.basis, CompletionBasis.USER_CONFIRMED)
        self.assertEqual(record.user_message_id, message.id)
        self.assertEqual(record.assessment.sources[0].provider, "user")
        self.assertEqual(store.notices(), ())

    async def test_scheduled_inquiry_cannot_forge_human_confirmation_or_machine_checks(
        self,
    ) -> None:
        self.assessment = replace(self.assessment, basis=CompletionBasis.USER_CONFIRMED)
        with self.assertRaises(ValueError):
            await self.coordinator.run_pass()
        store = self.memory.coordination
        changed = replace(
            self.responsibility,
            revision=2,
            completion_condition="pull_request_checks_pass",
        )
        store.save_responsibility(changed)
        self.assessment = replace(
            self.assessment,
            basis=CompletionBasis.OBSERVED,
            responsibility_revision=2,
            condition=changed.completion_condition,
        )
        self.now += self.coordinator.config.failure_retry_seconds
        with self.assertRaises(ValueError):
            await self.coordinator.run_pass()
        self.assertEqual(store.completions(), ())

    async def test_user_revision_pauses_scope_and_invalidates_pending_assessment(
        self,
    ) -> None:
        started, release = asyncio.Event(), asyncio.Event()
        original_reason = self.coordinator.reason

        async def delayed(request: InquiryPass) -> InquiryResult:
            started.set()
            await release.wait()
            return await original_reason(request)

        self.coordinator.reason = delayed
        task = asyncio.create_task(self.coordinator.run_pass())
        await started.wait()
        store = self.memory.coordination
        message = ChatMessage(
            str(uuid.uuid4()),
            "chat",
            ChatRole.USER,
            "Pause this investigation, move it to staging, and put its priority at two",
            self.now,
        )
        store.begin_turn(message)
        authority = ToolAuthority(
            store.state().instruction_revision, "chat", message.id
        )
        revised: list[Responsibility] = []
        workflow = Workflows(store, lambda: self.now, revised=revised.append)
        tool = next(
            record
            for record in workflow.tools("chat", message.id)
            if record.name == "revise_responsibility"
        )
        await tool.handler(
            {
                "id": "owner",
                "expected_revision": 1,
                "status": "paused",
                "priority": 2,
                "scope": "staging",
            },
            authority,
        )
        current = store.responsibility("owner")
        assert current is not None
        self.assertEqual(current.status, ResponsibilityStatus.PAUSED)
        self.assertEqual(current.priority, 2)
        self.assertEqual(current.scope, "staging")
        self.assertEqual(current.instruction_id, f"user:{message.id}")
        self.assertEqual(revised, [current])
        candidate = store.candidate(self.candidate.id)
        assert candidate is not None
        self.assertEqual(candidate.first_queued_at, self.candidate.first_queued_at)
        self.assertEqual(candidate.scope, "staging")
        self.assertGreater(
            candidate.evidence_revision, self.candidate.evidence_revision
        )
        with self.assertRaises(ValueError):
            await tool.handler(
                {"id": "owner", "expected_revision": 1, "status": "active"}, authority
            )
        release.set()
        with self.assertRaises(RevisionConflict):
            await task
        self.assertEqual(store.completions(), ())
        self.assertEqual(store.notices(), ())
