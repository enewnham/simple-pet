import asyncio
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from pet_daemon.coordination.config import ScheduleConfig
from pet_daemon.coordination.models import (
    Candidate,
    CandidateCause,
    CandidateDecision,
    DelegatedTask,
    Disposition,
    InquiryPass,
    InquiryResult,
    Instruction,
    Operation,
    OperationKind,
    OperationOutcome,
    Responsibility,
    TaskAction,
)
from pet_daemon.coordination.runtime import Coordinator
from pet_daemon.coordination.scheduling import eligible_candidates, select
from pet_daemon.storage import Memory
from pet_daemon.storage.coordination import RevisionConflict


class CoordinationLoopTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.now = 1000.0
        self.memory = Memory(
            Path(self.temporary.name) / "memory.sqlite3",
            "fixture",
            clock=lambda: self.now,
        )
        self.config = ScheduleConfig(
            review_seconds=10,
            max_deferral_seconds=20,
            minimum_interval_seconds=0,
            max_passes_per_hour=100,
        )
        self.requests: list[InquiryPass] = []

        async def reason(request: InquiryPass) -> InquiryResult:
            self.requests.append(request)
            selected = request.focus_id or request.candidates[0].id
            return InquiryResult(
                request.id,
                (
                    CandidateDecision(
                        selected, Disposition.NO_FINDING, "Nothing actionable changed"
                    ),
                ),
            )

        self.coordinator = Coordinator(
            self.memory.coordination, self.config, lambda: self.now, reason
        )

    async def asyncTearDown(self) -> None:
        await self.coordinator.close()
        self.memory.close()
        self.temporary.cleanup()

    def quiet(self, project: str = "quiet") -> Candidate:
        candidate = Candidate(
            "quiet-question",
            "Does the earlier decision still apply?",
            project,
            900,
            project_id=project,
            last_serviced_at=970,
            next_review_at=980,
            serviced_evidence_revision=1,
        )
        self.memory.coordination.save_candidate(candidate)
        return candidate

    async def test_overdue_quiet_work_precedes_noisy_changes_across_and_within_projects(
        self,
    ) -> None:
        quiet = self.quiet()
        for index in range(20):
            self.coordinator.queue(
                "What changed?",
                "busy",
                CandidateCause("source", "busy-source", str(index), self.now),
                project_id="busy",
            )
        await self.coordinator.run_pass()
        self.assertEqual(self.requests[0].focus_id, quiet.id)
        current = self.memory.coordination.candidate(quiet.id)
        assert current is not None
        self.assertEqual(current.last_serviced_at, self.now)
        self.now += 11
        peer = self.coordinator.queue(
            "What changed again?",
            "quiet",
            CandidateCause("source", "same-project", "one", self.now),
            project_id="quiet",
        )
        self.assertEqual(
            select((current, peer), self.now, self.config).focus_id, quiet.id
        )

    async def test_repeated_events_keep_first_queue_age_and_checkpointed_revisions(
        self,
    ) -> None:
        cause = CandidateCause("source", "report", "one", self.now)
        queued = self.coordinator.queue("Is review needed?", "parser", cause)
        self.now += 2
        duplicate = self.coordinator.queue(
            "Is review needed?", "parser", replace(cause, observed_at=self.now)
        )
        self.assertEqual(queued, duplicate)
        changed = self.coordinator.queue(
            "Is review needed?",
            "parser",
            replace(cause, source_revision="two", observed_at=self.now),
        )
        self.assertEqual(changed.first_queued_at, queued.first_queued_at)
        self.assertEqual(len(changed.causes), 1)
        await self.coordinator.run_pass()
        self.assertIsNone(await self.coordinator.run_pass())
        self.assertEqual(len(self.requests), 1)
        latest = self.coordinator.queue(
            "Is review needed?",
            "parser",
            replace(cause, source_revision="three", observed_at=self.now),
        )
        self.assertEqual(latest.first_queued_at, queued.first_queued_at)
        await self.coordinator.run_pass()
        self.assertEqual(len(self.requests), 2)

    async def test_direct_priority_override_leaves_overdue_work_unserviced_and_visible(
        self,
    ) -> None:
        quiet = self.quiet()
        urgent = Candidate(
            "urgent", "User priority", "busy", self.now, priority=2, exclusive=True
        )
        self.memory.coordination.save_candidate(urgent)
        await self.coordinator.run_pass()
        self.assertEqual(self.requests[0].focus_id, urgent.id)
        self.assertEqual(self.memory.coordination.candidate(quiet.id), quiet)
        activities = self.memory.coordination.activity()
        self.assertTrue(
            any(record.kind == "priority_override" for record in activities)
        )

    def test_due_obligation_can_override_review_and_routine_projects_rotate(
        self,
    ) -> None:
        quiet = Candidate(
            "quiet",
            "Quiet",
            "quiet",
            900,
            last_serviced_at=970,
            serviced_evidence_revision=1,
        )
        due = Candidate("due", "Due", "busy", 1000, due_at=1000)
        self.assertEqual(select((quiet, due), self.now, self.config).focus_id, due.id)
        first = Candidate("first", "First", "a", 1000)
        second = Candidate("second", "Second", "b", 1000)
        serviced_peer = replace(first, id="served-peer", last_serviced_at=999)
        selection = select(
            (first, second), self.now, self.config, (first, second, serviced_peer)
        )
        self.assertEqual(selection.candidates[0].id, second.id)

    async def test_budget_exhaustion_leaves_other_candidates_unconsidered(self) -> None:
        self.coordinator.config = self.config.model_copy(
            update={"max_passes_per_hour": 1}
        )
        first = self.coordinator.queue(
            "First", "a", CandidateCause("source", "a", "one", self.now)
        )
        second = self.coordinator.queue(
            "Second", "b", CandidateCause("source", "b", "one", self.now)
        )
        await self.coordinator.run_pass()
        self.assertIsNone(await self.coordinator.run_pass())
        history = self.memory.coordination.considerations()
        self.assertEqual(len(history), 1)
        skipped = second if history[0].candidate_id == first.id else first
        self.assertEqual(self.memory.coordination.candidate(skipped.id), skipped)

    async def test_waiting_for_human_does_not_repeat_without_meaningful_evidence(
        self,
    ) -> None:
        candidate = self.coordinator.queue(
            "Need a decision", "a", CandidateCause("source", "a", "one", self.now)
        )

        async def ask(request: InquiryPass) -> InquiryResult:
            return InquiryResult(
                request.id,
                (CandidateDecision(candidate.id, Disposition.ASK, "Which scope?"),),
            )

        self.coordinator.reason = ask
        await self.coordinator.run_pass()
        self.now += 100
        self.assertIsNone(await self.coordinator.run_pass())
        self.coordinator.queue(
            "Need a decision", "a", CandidateCause("source", "a", "two", self.now)
        )
        self.assertIsNotNone(await self.coordinator.run_pass())

    async def test_changed_evidence_rejects_an_inflight_result_without_service(
        self,
    ) -> None:
        candidate = self.coordinator.queue(
            "What matters?", "a", CandidateCause("source", "a", "one", self.now)
        )

        async def changed(request: InquiryPass) -> InquiryResult:
            self.coordinator.queue(
                "What matters?", "a", CandidateCause("source", "a", "two", self.now)
            )
            return InquiryResult(
                request.id,
                (
                    CandidateDecision(
                        candidate.id, Disposition.NO_FINDING, "Old result"
                    ),
                ),
            )

        self.coordinator.reason = changed
        with self.assertRaises(RevisionConflict):
            await self.coordinator.run_pass()
        self.assertEqual(self.memory.coordination.considerations(), ())
        current = self.memory.coordination.candidate(candidate.id)
        assert current is not None
        self.assertIsNone(current.last_serviced_at)

    async def test_far_future_review_cannot_bypass_maximum_deferral(self) -> None:
        candidate = Candidate(
            "old",
            "Review quiet work",
            "a",
            900,
            last_serviced_at=970,
            next_review_at=5000,
            serviced_evidence_revision=1,
        )
        self.memory.coordination.save_candidate(candidate)
        await self.coordinator.run_pass()
        self.assertEqual(self.requests[0].focus_id, candidate.id)
        deferred = replace(candidate, id="explicit-later", deferred_until=5000)
        self.assertEqual(
            eligible_candidates((deferred,), (), (), (), self.now, self.config), ()
        )

    async def test_unknown_inspection_leaves_reasoning_eligible_and_unknown_launch_blocks_only_its_work(
        self,
    ) -> None:
        store = self.memory.coordination
        store.save_instruction(
            Instruction("adopt", "Follow parser work", "a", self.now)
        )
        responsibility = Responsibility("work", "Follow parser", "a", "adopt", self.now)
        store.save_responsibility(responsibility)
        task = DelegatedTask(
            "task", responsibility.id, "Review parser", "Review result", self.now, 1
        )
        store.save_task(task)
        candidate = self.coordinator.queue(
            "Next step",
            "a",
            CandidateCause("task", "task", "one", self.now),
            responsibility_id=responsibility.id,
        )
        for kind in (OperationKind.INSPECT, OperationKind.OPEN):
            store.save_operation(
                Operation(
                    kind.value,
                    task.id,
                    kind,
                    self.now,
                    1,
                    1,
                    1,
                    outcome=OperationOutcome.UNKNOWN,
                )
            )
        await self.coordinator.run_pass()
        self.assertEqual(self.requests[-1].candidates[0].id, candidate.id)
        self.coordinator.queue(
            "Next step",
            "a",
            CandidateCause("task", "task", "two", self.now),
            responsibility_id=responsibility.id,
        )
        store.save_operation(
            Operation(
                "launch",
                task.id,
                OperationKind.LAUNCH,
                self.now,
                1,
                1,
                1,
                outcome=OperationOutcome.UNKNOWN,
            )
        )
        other = self.coordinator.queue(
            "Other work", "b", CandidateCause("source", "other", "one", self.now)
        )
        await self.coordinator.run_pass()
        self.assertEqual(
            tuple(record.id for record in self.requests[-1].candidates), (other.id,)
        )

    async def test_pause_after_external_effect_prevents_recording_service(
        self,
    ) -> None:
        store = self.memory.coordination
        store.save_instruction(
            Instruction("adopt", "Review two projects", "", self.now)
        )
        for scope in ("a", "b"):
            store.save_responsibility(
                Responsibility(
                    scope, "Review", scope, "adopt", self.now, permissions=("launch",)
                )
            )
            self.coordinator.queue(
                "Review",
                scope,
                CandidateCause("source", scope, "one", self.now),
                responsibility_id=scope,
            )
        effects: list[TaskAction] = []

        async def reason(request: InquiryPass) -> InquiryResult:
            return InquiryResult(
                request.id,
                tuple(
                    CandidateDecision(
                        candidate.id,
                        Disposition.INVESTIGATE,
                        "Review requested work",
                        action=TaskAction(
                            candidate.id,
                            OperationKind.LAUNCH,
                            candidate.responsibility_id or "",
                            objective="Review",
                            expected_result="Review",
                            repository=candidate.scope,
                        ),
                    )
                    for candidate in request.candidates[:1]
                ),
            )

        async def effect(action: TaskAction) -> Operation:
            effects.append(action)
            store.set_paused(True)
            return Operation(action.id, "task", action.kind, self.now, 1, 1, 1)

        self.coordinator.reason = reason
        self.coordinator.effect = effect
        with self.assertRaises(RevisionConflict):
            await self.coordinator.run_pass()
        self.assertEqual(len(effects), 1)
        self.assertEqual(store.considerations(), ())

    async def test_sleep_cancels_a_pass_without_service_and_wake_can_schedule_again(
        self,
    ) -> None:
        self.coordinator.queue(
            "Review", "a", CandidateCause("source", "a", "one", self.now)
        )
        started = asyncio.Event()

        async def reason(request: InquiryPass) -> InquiryResult:
            started.set()
            await asyncio.Event().wait()
            return InquiryResult(request.id, ())

        self.coordinator.reason = reason
        self.coordinator.tick()
        await started.wait()
        await self.coordinator.suspend()
        self.assertEqual(self.memory.coordination.considerations(), ())
        self.assertEqual(len(self.memory.coordination.attempts()), 1)
        self.assertEqual(self.memory.coordination.attempts()[0].error, "Interrupted")
        started.clear()
        await self.coordinator.resume()
        self.coordinator.tick()
        await started.wait()
        self.assertEqual(len(self.memory.coordination.attempts()), 2)
