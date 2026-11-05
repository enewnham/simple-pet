import asyncio
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import replace

from ..logging import get_logger
from ..storage.coordination import RevisionConflict, SQLiteCoordinationRepository
from ..storage.transactions import transaction
from .config import ScheduleConfig
from .models import (
    ActivityRecord,
    Candidate,
    CandidateCause,
    CandidateDecision,
    CompletionAssessment,
    CompletionBasis,
    CompletionRecord,
    Consideration,
    Disposition,
    InquiryAttempt,
    InquiryPass,
    InquiryResult,
    Operation,
    Responsibility,
    SourceReference,
    TaskAction,
)
from .scheduling import eligible_candidates, select

log = get_logger("pet-daemon")


class Coordinator:
    def __init__(
        self,
        store: SQLiteCoordinationRepository,
        config: ScheduleConfig,
        clock: Callable[[], float],
        reason: Callable[[InquiryPass], Awaitable[InquiryResult]],
        *,
        effect: Callable[[TaskAction], Awaitable[Operation]] | None = None,
        reconcile: Callable[[], Awaitable[None]] | None = None,
        refresh_watches: Callable[[], Awaitable[None]] | None = None,
        sources_current: Callable[[tuple[SourceReference, ...]], bool] | None = None,
        completion_sources_valid: Callable[[tuple[SourceReference, ...]], bool]
        | None = None,
        completed: Callable[[Responsibility], None] | None = None,
    ) -> None:
        self.store = store
        self.config = config
        self.clock = clock
        self.reason = reason
        self.effect = effect
        self.reconcile = reconcile
        self.refresh_watches = refresh_watches
        self.sources_current = sources_current
        self.completion_sources_valid = completion_sources_valid
        self.completed = completed
        self.task: asyncio.Task[None] | None = None
        self.suspended = False
        self.next_tick = 0.0
        self.epoch = 0

    def reset(self) -> None:
        self.epoch += 1
        if self.task is not None:
            self.task.cancel()
        self.next_tick = 0

    def queue(
        self,
        question: str,
        scope: str,
        cause: CandidateCause,
        *,
        responsibility_id: str | None = None,
        project_id: str | None = None,
        priority: int = 0,
        due_at: float | None = None,
        exclusive: bool = False,
    ) -> Candidate:
        identifier = str(
            uuid.uuid5(uuid.NAMESPACE_URL, f"{responsibility_id or scope}:{question}")
        )
        previous = self.store.candidate(identifier)
        if previous is None:
            candidate = Candidate(
                identifier,
                question,
                scope,
                self.clock(),
                responsibility_id=responsibility_id,
                project_id=project_id,
                causes=(cause,),
                priority=priority,
                due_at=due_at,
                exclusive=exclusive,
            )
        else:
            causes = {record.source_id: record for record in previous.causes}
            known = causes.get(cause.source_id)
            if (
                known is not None
                and known.source_revision == cause.source_revision
                and known.kind == cause.kind
                and previous.scope == scope
                and previous.project_id == project_id
                and previous.priority == priority
                and previous.due_at == due_at
                and previous.exclusive == exclusive
            ):
                return previous
            causes[cause.source_id] = cause
            candidate = replace(
                previous,
                scope=scope,
                project_id=project_id,
                revision=previous.revision + 1,
                evidence_revision=previous.evidence_revision + 1,
                causes=tuple(causes[key] for key in sorted(causes)),
                priority=priority,
                due_at=due_at,
                exclusive=exclusive,
            )
        self.store.save_candidate(candidate)
        return candidate

    def tick(self) -> None:
        now = self.clock()
        if self.suspended or self.store.state().paused or now < self.next_tick:
            return
        if self.task is not None and not self.task.done():
            return
        if self.task is not None and not self.task.cancelled():
            self.task.result()
        self.next_tick = now + self.config.minimum_interval_seconds
        self.task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        try:
            if self.reconcile is not None:
                await self.reconcile()
            if self.refresh_watches is not None:
                await self.refresh_watches()
            await self.run_pass()
        except OSError, ValueError, RuntimeError:
            log.exception("coordination.pass_failed")

    async def run_pass(self) -> InquiryResult | None:
        epoch = self.epoch
        now = self.clock()
        if self.suspended or self.store.state().paused:
            return None
        attempts = self.store.attempts()
        recent = [record for record in attempts if record.created_at > now - 3600]
        if len(recent) >= self.config.max_passes_per_hour:
            return None
        history = self.store.candidates()
        eligible = eligible_candidates(
            history,
            self.store.responsibilities(),
            self.store.tasks(),
            self.store.operations(),
            now,
            self.config,
        )
        selection = select(eligible, now, self.config, history)
        if not selection.candidates:
            return None
        request = InquiryPass(
            str(uuid.uuid4()),
            selection.candidates,
            self.store.state().instruction_revision,
            selection.focus_id,
            tuple(
                record
                for record in self.store.considerations()
                if record.candidate_id
                in {candidate.id for candidate in selection.candidates}
            ),
            tuple(
                record
                for record in self.store.responsibilities()
                if record.id
                in {candidate.responsibility_id for candidate in selection.candidates}
            ),
            tuple(
                record
                for record in self.store.tasks()
                if record.responsibility_id
                in {candidate.responsibility_id for candidate in selection.candidates}
            ),
        )
        attempt = InquiryAttempt(
            request.id, now, tuple(candidate.id for candidate in request.candidates)
        )
        self.store.save_attempt(attempt)
        if selection.override is not None:
            self.store.record_activity(
                ActivityRecord(
                    f"override:{request.id}",
                    "priority_override",
                    selection.override,
                    now,
                    entity_id=selection.focus_id,
                    detail="Overdue work retains its waiting age and remains due for review.",
                )
            )
        try:
            result = await self.reason(request)
            if epoch != self.epoch:
                raise RevisionConflict("The coordination history was cleared")
            await self._apply(request, result)
        except asyncio.CancelledError:
            if self.store.attempt(attempt.id) is not None:
                self.store.save_attempt(
                    replace(
                        attempt,
                        revision=2,
                        completed_at=self.clock(),
                        error="Interrupted",
                    )
                )
            raise
        except (OSError, ValueError, RuntimeError) as error:
            if self.store.attempt(attempt.id) is not None:
                self.store.save_attempt(
                    replace(
                        attempt, revision=2, completed_at=self.clock(), error=str(error)
                    )
                )
            raise
        self.store.save_attempt(replace(attempt, revision=2, completed_at=self.clock()))
        return result

    async def _apply(self, request: InquiryPass, result: InquiryResult) -> None:
        offered = {candidate.id: candidate for candidate in request.candidates}
        decisions = {decision.candidate_id: decision for decision in result.decisions}
        if (
            result.pass_id != request.id
            or len(decisions) != len(result.decisions)
            or not decisions.keys() <= offered.keys()
            or request.focus_id is not None
            and request.focus_id not in decisions
        ):
            raise ValueError(
                "An inquiry must address its reserved focus and offered work"
            )
        if self.store.state().paused or self.suspended:
            raise RevisionConflict("Coordination paused during consideration")
        if self.store.state().instruction_revision != request.instruction_revision:
            raise RevisionConflict("Instructions changed during consideration")
        for identifier in decisions:
            current = self.store.candidate(identifier)
            if current != offered[identifier]:
                raise RevisionConflict("Evidence changed during consideration")
        for identifier, decision in decisions.items():
            candidate = offered[identifier]
            self._validate_candidate(request, candidate)
            now = self.clock()
            serviced = decision.disposition != Disposition.DEFER
            origin = (
                now
                if serviced
                else (
                    candidate.last_serviced_at
                    if candidate.last_serviced_at is not None
                    else candidate.first_queued_at
                )
            )
            next_review = min(
                decision.next_review_at
                if decision.next_review_at is not None
                else now + self.config.review_seconds,
                origin + self.config.max_deferral_seconds,
            )
            next_review = max(now, next_review)
            if decision.completion is not None:
                if (
                    decision.action is not None
                    and decision.completion.basis == CompletionBasis.OBSERVED
                ):
                    raise ValueError("A completed goal cannot also request new work")
            elif decision.disposition == Disposition.COMPLETE:
                raise ValueError("Completion requires a supported assessment")
            if decision.action is not None:
                if decision.disposition not in (
                    Disposition.INVESTIGATE,
                    Disposition.CONTINUE,
                ):
                    raise ValueError("This disposition cannot request a desktop effect")
                if self.sources_current is not None and not self.sources_current(
                    (*decision.sources, *decision.action.sources)
                ):
                    raise RevisionConflict(
                        "The action's source evidence changed or became unavailable"
                    )
                if (
                    self.effect is None
                    or candidate.responsibility_id != decision.action.responsibility_id
                ):
                    raise ValueError(
                        "The proposed effect has no applicable responsibility or adapter"
                    )
                responsibility = next(
                    (
                        record
                        for record in request.responsibilities
                        if record.id == decision.action.responsibility_id
                    ),
                    None,
                )
                if (
                    responsibility is None
                    or self.store.responsibility(responsibility.id) != responsibility
                ):
                    raise RevisionConflict(
                        "The responsibility changed during consideration"
                    )
                task = next(
                    (
                        record
                        for record in request.tasks
                        if record.id == decision.action.task_id
                    ),
                    None,
                )
                if decision.action.task_id and (
                    task is None or self.store.task(task.id) != task
                ):
                    raise RevisionConflict("The task changed during consideration")
                action = replace(
                    decision.action,
                    expected_instruction_revision=request.instruction_revision,
                    expected_responsibility_revision=responsibility.revision,
                    expected_task_revision=task.revision if task else None,
                    expected_user_revision=task.user_revision if task else None,
                )
                await self.effect(action)
                self._validate_candidate(request, candidate)
            self._record_decision(
                request,
                candidate,
                decision,
                self.clock(),
                max(self.clock(), next_review),
                serviced,
            )

    def _record_decision(
        self,
        request: InquiryPass,
        candidate: Candidate,
        decision: CandidateDecision,
        now: float,
        next_review: float,
        serviced: bool,
    ) -> None:
        with transaction(self.store.db):
            completed = (
                self._complete(request, candidate, decision.completion)
                if decision.completion is not None
                else None
            )
            disposition = (
                Disposition.COMPLETE
                if completed is not None
                else Disposition.WAIT
                if decision.disposition == Disposition.COMPLETE
                else decision.disposition
            )
            record = Consideration(
                f"{request.id}:{candidate.id}",
                candidate.id,
                candidate.revision,
                request.instruction_revision,
                now,
                disposition,
                decision.reason,
                serviced_at=now if serviced else None,
                sources=decision.sources,
                action_id=decision.action.id if decision.action else None,
                next_review_at=next_review,
                wake_condition="human"
                if decision.disposition == Disposition.ASK
                else decision.wake_condition,
            )
            self.store.record_consideration(record)
            self.store.record_activity(
                ActivityRecord(
                    record.id,
                    "consideration",
                    decision.reason,
                    now,
                    entity_id=candidate.id,
                    detail=disposition.value,
                    sources=decision.sources,
                )
            )

            if completed is not None and self.completed is not None:
                self.completed(completed)

    def _validate_candidate(self, request: InquiryPass, candidate: Candidate) -> None:
        state = self.store.state()
        if (
            self.suspended
            or state.paused
            or state.instruction_revision != request.instruction_revision
        ):
            raise RevisionConflict(
                "Coordination or instructions changed before an effect"
            )
        if self.store.candidate(candidate.id) != candidate:
            raise RevisionConflict("Candidate evidence changed before an effect")

    def _complete(
        self,
        request: InquiryPass,
        candidate: Candidate,
        assessment: CompletionAssessment,
    ) -> Responsibility | None:
        responsibility = next(
            (
                record
                for record in request.responsibilities
                if record.id == assessment.responsibility_id
            ),
            None,
        )
        if (
            responsibility is None
            or candidate.responsibility_id != responsibility.id
            or assessment.responsibility_revision != responsibility.revision
            or assessment.condition != responsibility.completion_condition
            or self.store.responsibility(responsibility.id) != responsibility
        ):
            raise RevisionConflict("The assessed goal or completion condition changed")
        if assessment.basis in (CompletionBasis.REPORTED, CompletionBasis.INFERRED):
            self.store.record_activity(
                ActivityRecord(
                    f"unverified:{request.id}:{candidate.id}",
                    "completion_unverified",
                    "The reported result remains unverified.",
                    self.clock(),
                    entity_id=responsibility.id,
                    detail=assessment.claim,
                    sources=assessment.sources,
                )
            )
            return
        if assessment.basis != CompletionBasis.OBSERVED:
            raise ValueError("Scheduled inquiry cannot assign human confirmation")
        if responsibility.completion_condition == "pull_request_checks_pass":
            raise ValueError("Exact checks completion belongs to the machine watch")
        if (
            self.completion_sources_valid is None
            or not self.completion_sources_valid(assessment.sources)
            or self.sources_current is not None
            and not self.sources_current(assessment.sources)
        ):
            raise ValueError(
                "Current independent leaf evidence is required for completion"
            )
        record = CompletionRecord(
            f"completion:{responsibility.id}:{responsibility.revision}",
            assessment,
            self.clock(),
            request.instruction_revision,
        )
        self.store.record_completion(record)
        return self.store.responsibility(responsibility.id)

    async def suspend(self) -> None:
        self.suspended = True
        if self.task is not None and not self.task.done():
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)

    async def resume(self) -> None:
        self.suspended = False
        self.next_tick = 0

    async def close(self) -> None:
        await self.suspend()
