import asyncio
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from pet_daemon.companion.usage import SdkFailure
from pet_daemon.coordination.config import ScheduleConfig
from pet_daemon.coordination.models import (
    Candidate,
    CandidateCause,
    CandidateDecision,
    ChatMessage,
    ChatRole,
    DelegatedTask,
    Disposition,
    InquiryAttempt,
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
        self.memory.coordination.save_instruction(
            Instruction("quiet-adoption", "Review the decision when due", project, 900)
        )
        self.memory.coordination.save_responsibility(
            Responsibility(
                "quiet-obligation",
                "Review the decision",
                project,
                "quiet-adoption",
                900,
                followup_at=980,
            )
        )
        candidate = Candidate(
            "quiet-question",
            "Does the earlier decision still apply?",
            project,
            900,
            project_id=project,
            responsibility_id="quiet-obligation",
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

    async def test_source_no_finding_ignores_expired_dates_and_unchanged_polls_for_eight_hours(
        self,
    ) -> None:
        cause = CandidateCause("source", "checkout", "one", self.now)
        candidate = self.coordinator.queue(
            "Did the checkout change anything?", "parser", cause
        )

        async def reason(request: InquiryPass) -> InquiryResult:
            self.requests.append(request)
            return InquiryResult(
                request.id,
                (
                    CandidateDecision(
                        candidate.id,
                        Disposition.NO_FINDING,
                        "No new finding",
                        next_review_at=self.now - 100,
                    ),
                ),
            )

        self.coordinator.reason = reason
        self.coordinator.source_checkpoint = lambda: 42
        await self.coordinator.run_pass()
        for _ in range(960):
            self.now += 30
            self.coordinator.queue(
                candidate.question,
                candidate.scope,
                replace(cause, observed_at=self.now),
            )
            self.assertIsNone(await self.coordinator.run_pass())
        self.assertEqual(len(self.requests), 1)
        current = self.memory.coordination.candidate(candidate.id)
        assert current is not None
        self.assertIsNone(current.next_review_at)
        self.assertEqual(
            self.memory.coordination.considerations(candidate.id)[0].source_checkpoint,
            42,
        )
        self.coordinator.queue(
            candidate.question,
            candidate.scope,
            replace(cause, source_revision="two", observed_at=self.now),
        )
        await self.coordinator.run_pass()
        self.assertEqual(len(self.requests), 2)

    async def test_failure_retries_are_bounded_for_evidence_across_restart(
        self,
    ) -> None:
        cause = CandidateCause("source", "checkout", "one", self.now)
        candidate = self.coordinator.queue("Inspect checkout", "parser", cause)

        async def fail(request: InquiryPass) -> InquiryResult:
            self.requests.append(request)
            raise ValueError("A source reference must come from supplied evidence")

        self.coordinator.reason = fail
        with self.assertRaises(ValueError):
            await self.coordinator.run_pass()
        self.now += self.config.failure_retry_seconds - 1
        self.assertIsNone(await self.coordinator.run_pass())
        self.now += 1
        with self.assertRaises(ValueError):
            await self.coordinator.run_pass()
        await self.coordinator.close()
        self.memory.close()
        self.memory = Memory(
            Path(self.temporary.name) / "memory.sqlite3",
            "fixture",
            clock=lambda: self.now,
        )
        self.coordinator = Coordinator(
            self.memory.coordination, self.config, lambda: self.now, fail
        )
        for _ in range(960):
            self.now += 30
            self.assertIsNone(await self.coordinator.run_pass())
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(self.memory.coordination.considerations(), ())
        self.coordinator.queue(
            candidate.question,
            candidate.scope,
            replace(cause, source_revision="two", observed_at=self.now),
        )
        with self.assertRaises(ValueError):
            await self.coordinator.run_pass()
        self.assertEqual(len(self.requests), 3)

    async def test_billing_failure_has_no_automatic_retry_and_new_user_direction_can_retry(
        self,
    ) -> None:
        self.coordinator.queue(
            "Inspect checkout",
            "parser",
            CandidateCause("source", "checkout", "one", self.now),
        )

        async def fail(request: InquiryPass) -> InquiryResult:
            self.requests.append(request)
            raise SdkFailure("billing_error", candidate_id=request.candidates[0].id)

        self.coordinator.reason = fail
        with self.assertRaises(SdkFailure):
            await self.coordinator.run_pass()
        self.now += 8 * 3600
        self.assertIsNone(await self.coordinator.run_pass())
        self.coordinator.queue(
            "Inspect checkout",
            "parser",
            CandidateCause("source", "checkout", "two", self.now),
        )
        self.coordinator.queue(
            "Inspect another project",
            "other",
            CandidateCause("source", "other-checkout", "one", self.now),
        )
        self.assertIsNone(await self.coordinator.run_pass())
        self.memory.coordination.save_instruction(
            Instruction(
                "human-retry", "Please try the investigation again", "parser", self.now
            )
        )
        with self.assertRaises(SdkFailure):
            await self.coordinator.run_pass()
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(
            self.memory.coordination.attempts()[0].error_kind, "billing_error"
        )

    async def test_adopted_due_and_explicit_later_are_consumed_once_despite_expired_model_dates(
        self,
    ) -> None:
        store = self.memory.coordination
        store.save_instruction(
            Instruction("adopt", "Review on the agreed date", "parser", self.now)
        )
        store.save_responsibility(
            Responsibility(
                "goal",
                "Review parser",
                "parser",
                "adopt",
                self.now,
                followup_at=self.now + 10,
            )
        )
        candidate = Candidate(
            "due-goal",
            "Review parser",
            "parser",
            self.now - 100,
            responsibility_id="goal",
            last_serviced_at=self.now - 50,
            serviced_evidence_revision=1,
            next_review_at=self.now - 10,
        )
        store.save_candidate(candidate)
        self.assertIsNone(await self.coordinator.run_pass())
        self.now += 10
        await self.coordinator.run_pass()
        self.now += 8 * 3600
        self.assertIsNone(await self.coordinator.run_pass())
        self.assertEqual(len(self.requests), 1)
        await self.coordinator.close()
        self.memory.close()
        self.memory = Memory(
            Path(self.temporary.name) / "memory.sqlite3",
            "fixture",
            clock=lambda: self.now,
        )
        self.coordinator = Coordinator(
            self.memory.coordination,
            self.config,
            lambda: self.now,
            self.coordinator.reason,
        )
        store = self.memory.coordination
        self.assertIsNone(await self.coordinator.run_pass())
        current = store.candidate(candidate.id)
        assert current is not None
        store.save_candidate(
            replace(current, revision=current.revision + 1, deferred_until=self.now + 5)
        )
        self.assertIsNone(await self.coordinator.run_pass())
        self.now += 5
        await self.coordinator.run_pass()
        self.now += 3600
        self.assertIsNone(await self.coordinator.run_pass())
        self.assertEqual(len(self.requests), 2)

    async def test_legacy_unchanged_failures_stop_without_schema_reset_or_false_service(
        self,
    ) -> None:
        candidate = self.coordinator.queue(
            "Inspect checkout",
            "parser",
            CandidateCause("source", "checkout", "one", self.now - 120),
        )
        store = self.memory.coordination
        store.save_instruction(
            Instruction(
                "existing-direction", "Inspect this checkout", "parser", self.now - 60
            )
        )
        for index in range(3):
            store.save_attempt(
                InquiryAttempt(
                    f"legacy-{index}",
                    self.now - 30 + index,
                    (candidate.id,),
                    completed_at=self.now - 20 + index,
                    error="Legacy citation validation failure",
                )
            )
        self.now += 3600
        self.assertIsNone(await self.coordinator.run_pass())
        self.assertEqual(self.requests, [])
        self.assertEqual(len(store.attempts()), 3)
        self.assertEqual(store.considerations(), ())
        self.assertEqual(
            self.memory.db.execute("PRAGMA user_version").fetchone()[0], 27
        )

    async def test_legacy_higher_evidence_revisions_stay_contained_until_a_later_cause(
        self,
    ) -> None:
        for evidence_revision in (3, 26):
            with self.subTest(evidence_revision=evidence_revision):
                source = f"checkout-{evidence_revision}"
                candidate = Candidate(
                    source,
                    "Inspect checkout",
                    source,
                    self.now - 120,
                    evidence_revision=evidence_revision,
                    serviced_evidence_revision=1,
                    causes=(CandidateCause("source", source, "one", self.now - 120),),
                )
                store = self.memory.coordination
                store.save_candidate(candidate)
                for index in range(3):
                    store.save_attempt(
                        InquiryAttempt(
                            f"legacy-{source}-{index}",
                            self.now - 30 + index,
                            (candidate.id,),
                            completed_at=self.now - 20 + index,
                            error="Legacy citation validation failure",
                        )
                    )
                self.assertIsNone(await self.coordinator.run_pass())
                await self.coordinator.close()
                self.memory.close()
                self.memory = Memory(
                    Path(self.temporary.name) / "memory.sqlite3",
                    "fixture",
                    clock=lambda: self.now,
                )
                self.coordinator = Coordinator(
                    self.memory.coordination,
                    self.config,
                    lambda: self.now,
                    self.coordinator.reason,
                )
                self.now += 8 * 3600
                self.assertIsNone(await self.coordinator.run_pass())
                previous = len(self.requests)
                self.coordinator.queue(
                    candidate.question,
                    candidate.scope,
                    CandidateCause("source", source, "two", self.now),
                )
                await self.coordinator.run_pass()
                self.assertEqual(len(self.requests), previous + 1)
                self.now += 3600
                self.assertIsNone(await self.coordinator.run_pass())

    async def test_retained_sdk_billing_error_blocks_new_sources_until_accepted_user_input(
        self,
    ) -> None:
        store = self.memory.coordination
        store.save_attempt(
            InquiryAttempt(
                "legacy-billing",
                self.now - 60,
                ("old-question",),
                completed_at=self.now - 50,
                error="Companion model failed: billing_error",
            )
        )
        self.coordinator.queue(
            "Inspect new work", "new", CandidateCause("source", "new", "one", self.now)
        )
        self.assertIsNone(await self.coordinator.run_pass())
        store.begin_turn(
            ChatMessage(
                "human-retry",
                "chat",
                ChatRole.USER,
                "Billing is fixed. Please try again.",
                self.now,
            )
        )
        await self.coordinator.run_pass()
        self.assertEqual(len(self.requests), 1)

    async def test_selected_failure_does_not_exhaust_unselected_work_or_advance_its_checkpoint(
        self,
    ) -> None:
        a = self.coordinator.queue(
            "Inspect A", "a", CandidateCause("source", "a", "one", self.now)
        )
        b = self.coordinator.queue(
            "Inspect B", "b", CandidateCause("source", "b", "one", self.now)
        )
        outcomes = []
        self.coordinator.finished = lambda identifier, outcome, error: outcomes.append(
            (identifier, outcome, error)
        )
        self.coordinator.source_checkpoint = lambda: 42

        async def reason(request: InquiryPass) -> InquiryResult:
            self.requests.append(request)
            if any(candidate.id == a.id for candidate in request.candidates):
                raise SdkFailure("validation", candidate_id=a.id)
            return InquiryResult(
                request.id,
                (
                    CandidateDecision(
                        b.id, Disposition.NO_FINDING, "B has no new finding"
                    ),
                ),
            )

        self.coordinator.reason = reason
        with self.assertRaises(SdkFailure):
            await self.coordinator.run_pass()
        await self.coordinator.run_pass()
        self.assertEqual(
            tuple(candidate.id for candidate in self.requests[-1].candidates), (b.id,)
        )
        self.assertEqual(self.memory.coordination.considerations(a.id), ())
        self.assertEqual(
            self.memory.coordination.considerations(b.id)[0].source_checkpoint, 42
        )
        self.assertEqual(
            [outcome for _, outcome, _ in outcomes], ["failed", "accepted"]
        )

    async def test_selected_cancellation_preserves_unassessed_peers_retry_budget(
        self,
    ) -> None:
        self.coordinator.config = self.config.model_copy(
            update={"review_seconds": 100_000, "max_deferral_seconds": 200_000}
        )
        a = self.coordinator.queue(
            "Inspect A", "a", CandidateCause("source", "a", "one", self.now)
        )
        b = self.coordinator.queue(
            "Inspect B", "b", CandidateCause("source", "b", "one", self.now)
        )

        async def cancel(request: InquiryPass) -> InquiryResult:
            self.requests.append(request)
            if all(candidate.id != b.id for candidate in request.candidates):
                return InquiryResult(
                    request.id,
                    (
                        CandidateDecision(
                            a.id, Disposition.NO_FINDING, "A is assessed independently"
                        ),
                    ),
                )
            interrupted = asyncio.CancelledError()
            interrupted.__dict__["candidate_id"] = b.id
            raise interrupted

        self.coordinator.reason = cancel
        for _ in range(2):
            with self.assertRaises(asyncio.CancelledError):
                await self.coordinator.run_pass()
            self.now += self.config.failure_retry_seconds
        self.assertEqual(
            {
                attempt.selected_candidate_id
                for attempt in self.memory.coordination.attempts()
            },
            {b.id},
        )
        await self.coordinator.run_pass()
        self.assertEqual(
            tuple(candidate.id for candidate in self.requests[-1].candidates), (a.id,)
        )

    async def test_cancelled_effect_retains_the_returned_selection_without_servicing_peers(
        self,
    ) -> None:
        self.coordinator.config = self.config.model_copy(
            update={"review_seconds": 100_000, "max_deferral_seconds": 200_000}
        )
        store = self.memory.coordination
        store.save_instruction(Instruction("adopt", "Review B", "b", self.now))
        store.save_responsibility(
            Responsibility(
                "b-work", "Review B", "b", "adopt", self.now, permissions=("launch",)
            )
        )
        a = self.coordinator.queue(
            "Inspect A", "a", CandidateCause("source", "a", "one", self.now)
        )
        b = self.coordinator.queue(
            "Inspect B",
            "b",
            CandidateCause("source", "b", "one", self.now),
            responsibility_id="b-work",
        )
        entered = asyncio.Event()

        async def reason(request: InquiryPass) -> InquiryResult:
            selected = (
                b
                if any(candidate.id == b.id for candidate in request.candidates)
                else a
            )
            action = (
                TaskAction(
                    "launch-b",
                    OperationKind.LAUNCH,
                    "b-work",
                    objective="Review B",
                    expected_result="Review evidence",
                    repository="b",
                )
                if selected == b
                else None
            )
            return InquiryResult(
                request.id,
                (
                    CandidateDecision(
                        selected.id,
                        Disposition.INVESTIGATE,
                        "Review selected work",
                        action=action,
                    ),
                ),
            )

        async def effect(action: TaskAction) -> Operation:
            entered.set()
            await asyncio.Event().wait()
            raise AssertionError("The held effect must be cancelled")

        self.coordinator.reason = reason
        self.coordinator.effect = effect
        for _ in range(2):
            entered.clear()
            task = asyncio.create_task(self.coordinator.run_pass())
            await entered.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.now += self.config.failure_retry_seconds
        self.assertEqual(
            {attempt.selected_candidate_id for attempt in store.attempts()}, {b.id}
        )
        self.assertEqual(store.considerations(), ())
        await self.coordinator.run_pass()
        self.assertEqual(
            tuple(record.candidate_id for record in store.considerations()), (a.id,)
        )

    async def test_deferred_model_output_waits_for_evidence_without_claiming_service(
        self,
    ) -> None:
        candidate = self.coordinator.queue(
            "Inspect checkout",
            "parser",
            CandidateCause("source", "checkout", "one", self.now),
        )

        async def reason(request: InquiryPass) -> InquiryResult:
            return InquiryResult(
                request.id,
                (
                    CandidateDecision(
                        candidate.id,
                        Disposition.DEFER,
                        "No useful step",
                        next_review_at=self.now - 100,
                    ),
                ),
            )

        self.coordinator.reason = reason
        await self.coordinator.run_pass()
        self.now += 8 * 3600
        self.assertIsNone(await self.coordinator.run_pass())
        current = self.memory.coordination.candidate(candidate.id)
        assert current is not None
        self.assertIsNone(current.last_serviced_at)
        self.assertEqual(len(self.memory.coordination.considerations()), 1)

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
        self.memory.coordination.save_instruction(
            Instruction("due-adoption", "Review due work", "a", 900)
        )
        self.memory.coordination.save_responsibility(
            Responsibility(
                "due-work", "Review due work", "a", "due-adoption", 900, followup_at=980
            )
        )
        candidate = Candidate(
            "old",
            "Review quiet work",
            "a",
            900,
            last_serviced_at=970,
            next_review_at=5000,
            serviced_evidence_revision=1,
            responsibility_id="due-work",
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
        self.now += self.config.failure_retry_seconds
        self.coordinator.tick()
        await started.wait()
        self.assertEqual(len(self.memory.coordination.attempts()), 2)
