import asyncio
import hashlib
import math
import sqlite3
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Protocol

import structlog

from .attention import Attention, AttentionSelection
from .behavior.controller import Behavior
from .commentary import Cognition, SpeechConditions
from .commentary.cognition import CannotRespond, CommentaryResult
from .commentary.usage import collect_usage
from .config import Config
from .conversation import Conversations
from .coordination.models import CandidateCause
from .coordination.runtime import Coordinator
from .delivery import (
    DeliveryConditions,
    DeliveryOutcome,
    SpeechDelivery,
    SpeechMessage,
    SpeechPacket,
)
from .desktop import Delegation
from .domain import DeliveryStatus, SpeechReason, SpeechSituation
from .identity import SessionIdentity
from .interactions import Interactions
from .logging import get_logger
from .repository_identity import repository_columns
from .runtime import (
    HookEvent,
    OverlayPacket,
    PetContext,
    RandomSource,
    ReactionTrigger,
    Trigger,
)
from .sessions.coordinator import ActivityCoordinator
from .sessions.processes import ProcessTracker
from .sessions.reactions import SessionReactions
from .sessions.tracking import Sessions
from .storage import Memory
from .work.history import (
    SessionContext,
    build_session_entry,
    report_prose,
    report_source_key,
)
from .work.models import ExecutionEvidence, ProgressEvidence
from .work.observations import (
    CheckoutObservation,
    PullRequestObservation,
    RepositoryObservation,
)

log = get_logger("pet-daemon")


class ObservationProvider(Protocol):
    def tick(self) -> None: ...
    def pause(self) -> None: ...
    def resume(self) -> None: ...
    def clear(self) -> None: ...
    async def close(self) -> None: ...


class CompanionLifecycle(Protocol):
    def reset(self) -> None: ...
    async def suspend(self) -> None: ...
    async def resume(self) -> None: ...
    async def close(self) -> None: ...


class Application:
    def __init__(
        self,
        config: Config,
        *,
        sessions: Sessions,
        reactions: SessionReactions,
        attention: Attention,
        behavior: Behavior,
        cognition: Cognition,
        providers: tuple[ObservationProvider, ...],
        processes: ProcessTracker,
        memory: Memory,
        emit: Callable[[OverlayPacket], object],
        clock: Callable[[], float],
        rng: RandomSource,
        checkout_target: Callable[[str, bool], None] | None = None,
        reconcile_work: Callable[[], None] | None = None,
        conversations: Conversations | None = None,
        companion: CompanionLifecycle | None = None,
        coordinator: Coordinator | None = None,
        desktop: Delegation | None = None,
    ) -> None:
        self.config = config
        self.sessions = sessions
        self.reactions = reactions
        self.attention = attention
        self.behavior = behavior
        self.cognition = cognition
        self.providers = tuple(providers)
        self.processes = processes
        self.memory = memory
        self.conversations = conversations
        self.companion = companion
        self.coordinator = coordinator
        self.desktop = desktop
        self.companion_tasks: set[asyncio.Task[None]] = set()
        self.companion_lock = asyncio.Lock()
        if conversations is not None:
            for conversation in memory.coordination.conversations():
                if conversation.pending_message_id is not None:
                    memory.coordination.fail_turn(
                        conversation.id,
                        conversation.pending_message_id,
                        "The reply was interrupted by a restart. You can retry it.",
                        expected_revision=conversation.revision,
                    )
        self.sessions.aliases.update(memory.session_aliases())
        self.emit = emit
        self.clock = clock
        self.rng = rng
        self.checkout_target = checkout_target
        self.reconcile_work = reconcile_work
        self.task: asyncio.Task[None] | None = None
        self.reason: SpeechReason | None = None
        self.revision = 0
        self.history_epoch = 0
        self.suspended = False
        self.manual: tuple[str, float | None] | None = None
        self.clicked = False
        self.last_speech: tuple[str, float] | None = None
        self.attention.restore(memory.load_attention_history())
        self.attention.restore_sources(memory.load_source_attention())
        self.process_task: asyncio.Task[None] | None = None
        self.next_prune = 0
        self.reconciling = False
        self.agent_activity = ActivityCoordinator(
            config,
            sessions=sessions,
            reactions=reactions,
            behavior=behavior,
            queue=self.queue,
            clock=clock,
            rng=rng,
        )
        self.delivery = SpeechDelivery(
            lambda packet: emit(
                {
                    "text": packet["text"],
                    "ttl": packet["ttl"],
                    "delivery_id": packet["delivery_id"],
                    **(
                        {"conversation": packet["conversation"]}
                        if "conversation" in packet
                        else {}
                    ),
                }
            ),
            clock=clock,
            on_outcome=self.delivery_outcome,
            valid=self.is_delivery_valid,
        )
        self.notice_delivery = SpeechDelivery(
            self.emit_notice,
            clock=clock,
            on_outcome=self.notice_outcome,
            valid=self.is_notice_valid,
        )
        self.interactions = Interactions(
            config,
            memory=memory,
            attention=attention,
            cognition=cognition,
            context=self.context,
            emit=emit,
            more=lambda: self.click(more=True),
            rounds=self.rounds,
            displayed=self.interaction_displayed,
            dismissed=self.preempt,
            clock=clock,
        )

    def interaction_displayed(self, text: str) -> None:
        self.last_speech = (text, self.clock())
        self.attention.last_attempt = self.clock()

    def clear_memory(self) -> None:
        if self.conversations is not None:
            if self.conversations.task is not None:
                self.conversations.task.cancel()
            self.conversations.dismiss()
        try:
            self.memory.clear()
        except sqlite3.Error as error:
            log.exception("memory.clear_failed", error=str(error))
            return
        if self.companion is not None:
            self.companion.reset()
        if self.coordinator is not None:
            self.coordinator.reset()
        self.preempt()
        self.notice_delivery.clear()
        self.interactions.clear()
        self.history_epoch += 1
        self.sessions = Sessions(
            self.clock,
            self.config.stale_seconds,
            self.config.completed_retention_seconds,
            self.config.activity_fresh_seconds,
            wall_clock=self.sessions.wall_clock,
        )
        self.reactions = SessionReactions()
        self.agent_activity.reset(self.sessions, self.reactions)
        self.manual = None
        self.clicked = False
        self.last_speech = None
        self.delivery.clear()
        self.attention.reset()
        for provider in self.providers:
            provider.clear()
        if self.process_task:
            self.process_task.cancel()
        self.emit({"text": "", "ttl": 0})
        log.info("memory.cleared", pet=self.memory.pet)

    def system_sleep_changed(self, sleeping: bool) -> None:
        if sleeping == self.suspended:
            return
        self.suspended = sleeping
        log.info("system.sleep" if sleeping else "system.wake")
        if sleeping:
            if self.desktop is not None:
                self.desktop.suspended = True
            if self.coordinator is not None:
                self._companion_task(self.coordinator.suspend)
            self.notice_delivery.cancel_pending()
            if self.conversations is not None:
                self._companion_task(self.conversations.suspend)
            if self.companion is not None:
                self._companion_task(self.companion.suspend)
            self.behavior.suspend("system_sleep")
            self.preempt()
            self.interactions.clear()
            for provider in self.providers:
                provider.pause()
            if self.process_task:
                self.process_task.cancel()
        else:
            if self.companion is not None:
                self._companion_task(self.companion.resume)
            if self.desktop is not None:
                self.desktop.suspended = False
            if self.coordinator is not None:
                self._companion_task(self.coordinator.resume)
            self.behavior.resume()
            self.sessions.prune()
            for provider in self.providers:
                provider.resume()
            for session in self.sessions.by_id.values():
                if session.active:
                    session.last_check_in = self.clock()
            self.attention.next_observation = self.clock() + self.rng.uniform(
                *self.config.observation_seconds
            )

    def _companion_task(self, operation: Callable[[], Awaitable[None]]) -> None:
        async def run() -> None:
            try:
                async with self.companion_lock:
                    await operation()
            except Exception:
                log.exception("companion.lifecycle_failed")

        task = asyncio.create_task(run())
        self.companion_tasks.add(task)
        task.add_done_callback(self.companion_tasks.discard)

    def speak(self, context: str, ttl: float | None = None) -> None:
        self.behavior.wake("speech_request")
        self.behavior.stop_movement()
        self.preempt()
        if self.interactions.identifier:
            self.interactions.clear()
        self.manual = (context, ttl)
        log.debug("speech.requested", context=context, ttl=ttl)

    def queue(self, reason: SpeechReason, trigger: Trigger | None = None) -> None:
        self.attention.queue(reason, trigger, self.sessions.has_current_failures)

    def context(self, trigger: Trigger | None = None) -> PetContext:
        runtime = self.sessions.activity()
        for session in runtime.sessions:
            context = self.memory.context(session.id)
            if context:
                session.goal = context.initial_request or session.goal
                session.initial_request = context.initial_request
                session.initial_request_at = context.initial_request_at
                session.summary = context.summary
                session.recent_requests = tuple(context.recent_requests)
                session.request_count = context.request_count
                session.first_summary_attempted = context.first_summary_attempted
        subjects = self.memory.connected_work(runtime)
        stories = {
            session: subject.story
            for subject in subjects
            for session in subject.sessions
        }
        for session in runtime.sessions:
            session.connected_work = stories.get(session.id)
        return PetContext(
            activity=runtime.activity,
            sessions=runtime.sessions,
            sleeping=self.behavior.is_sleeping,
            trigger=trigger,
            work=subjects,
            convergence=runtime.convergence,
        )

    def ingest(self, observation: HookEvent | RepositoryObservation) -> None:
        if isinstance(observation, HookEvent):
            self._ingest_hook(observation)
            return
        try:
            self.memory.work.observe(observation)
            if self.coordinator is not None:
                if isinstance(observation, CheckoutObservation):
                    scope = observation.worktree
                elif isinstance(observation, PullRequestObservation):
                    scope = observation.ref
                else:
                    _, repository = repository_columns(observation.repository)
                    scope = f"{repository}:{observation.branch}"
                revision = hashlib.sha256(
                    observation.model_dump_json(
                        exclude={"checked_at", "last_success_at"}
                    ).encode()
                ).hexdigest()
                self.coordinator.queue(
                    "Do these repository changes affect recent work?",
                    scope,
                    CandidateCause("repository", scope, revision, self.memory.clock()),
                )
            self.reconcile_targets()
        except (sqlite3.Error, ValueError) as error:
            log.exception("work.ingest_failed", error=str(error))

    def _ingest_hook(self, hook: HookEvent) -> None:
        if hook.repo:
            hook.repo = str(Path(hook.repo).resolve())
        result = self.agent_activity.handle(hook, self.suspended)
        if not result:
            return
        try:
            with self.memory.transaction():
                self.memory.work.record_session(result.observation)
                self.memory.observe_session_context(result.observation)
                session = self.sessions.by_id[result.session]
                self.memory.record_event(
                    build_session_entry(
                        hook,
                        SessionContext(
                            session_id=result.session,
                            working_directory=session.repo,
                            goal=session.goal,
                            last_tool=session.last_tool,
                            outcome=session.outcome,
                            activity=result.activity,
                        ),
                    )
                )
            if self.checkout_target is not None and hook.repo:
                self.checkout_target(
                    hook.repo,
                    hook.event in ("intent", "subagent_start"),
                )
            if self.desktop is not None:
                self.desktop.observe_hook(hook, result.session)
            if self.coordinator is not None:
                scope = hook.repo or self.sessions.by_id[result.session].repo
                revision = hashlib.sha256(
                    hook.model_dump_json(exclude={"capture_at", "timestamp"}).encode()
                ).hexdigest()
                self.coordinator.queue(
                    "Does this recent session need follow-up?",
                    scope,
                    CandidateCause(
                        "session", str(result.session), revision, self.memory.clock()
                    ),
                )
            self.reconcile_targets()
        except (sqlite3.Error, ValueError) as error:
            log.exception("work.evidence_failed", error=str(error))

    def reconcile_targets(self) -> None:
        if self.reconciling:
            return
        self.reconciling = True
        try:
            if self.reconcile_work is not None:
                self.reconcile_work()
        finally:
            self.reconciling = False

    def receive(self, packet: Mapping[str, object]) -> None:
        kind = packet.get("kind")
        if kind == "clear_speech":
            self.clear_speech()
        elif kind == "clear_memory":
            self.clear_memory()
        elif kind == "renderer":
            self.receive_renderer(packet)
        elif kind == "speak":
            context, ttl = packet.get("context"), packet.get("ttl")
            if isinstance(context, str) and (
                ttl is None
                or (type(ttl) is int or type(ttl) is float)
                and math.isfinite(ttl)
                and ttl >= 0
            ):
                self.speak(context, ttl)

    def receive_renderer(self, packet: Mapping[str, object]) -> None:
        if self.conversations is not None and self.conversations.receive(packet):
            return
        if self.interactions.receive(packet):
            return
        feedback = packet.get("event")
        if feedback == "speech_accepted":
            try:
                if self.notice_delivery.acknowledge(packet):
                    return
                self.delivery.acknowledge(packet)
            except sqlite3.Error as error:
                log.exception("renderer.acceptance_failed", error=str(error))
        elif feedback == "status":
            if self.behavior.update_renderer_status(packet):
                status = self.behavior.status
                assert status is not None
                self.system_sleep_changed(status.system_sleeping)
                if not self.suspended:
                    self.behavior.tick(self.agent_activity.activity(), self.sessions)
        elif feedback in ("system_sleep", "system_wake"):
            self.system_sleep_changed(feedback == "system_sleep")
        elif feedback == "click":
            self.click()
        elif feedback == "rounds":
            self.rounds()
        else:
            self.behavior.completion(packet)

    def clear_speech(self) -> None:
        try:
            self.memory.clear_speech()
        except sqlite3.Error as error:
            log.exception("speech.clear_failed", error=str(error))
            return
        self.history_epoch += 1
        self.delivery.clear()
        self.interactions.clear()
        self.last_speech = None
        self.emit({"text": "", "ttl": 0})
        log.info("speech.cleared", pet=self.memory.pet)

    def speech_duration(self, speech: str) -> float:
        return max(self.config.speech_ttl, 3 + len(speech.split()) / 2)

    def preempt(self) -> None:
        self.delivery.cancel_pending()
        self.revision += 1
        if self.task:
            self.task.cancel()
        self.attention.pending = None

    def click(self, *, more: bool = False) -> None:
        self.behavior.defer_care()
        self.behavior.wake("click")
        self.behavior.react("click", self.sessions)
        if self.conversations is not None and not more:
            self.preempt()
            self.interactions.clear()
            self.clicked = False
            self.conversations.open()
            return
        if self.task and self.reason == SpeechReason.CLICK and not self.task.done():
            return
        self.preempt()
        self.interactions.begin(more=more)
        self.clicked = True
        log.debug("speech.requested", reason="click")

    def rounds(self) -> None:
        self.behavior.wake("rounds")
        self.behavior.stop_movement()
        self.preempt()
        self.clicked = False
        self.reason = SpeechReason.CLICK
        self.task = asyncio.create_task(self.interactions.show_rounds())

    async def generate_commentary(
        self,
        snapshot: PetContext,
        selection: AttentionSelection,
        trigger: Trigger | None,
    ) -> CommentaryResult:
        reflection = selection.situation == SpeechSituation.IDLE_REFLECTION
        alternatives = (
            self.attention.recollection_alternatives(snapshot, selection)
            if reflection
            else ()
        )
        preferred: CommentaryResult | None = None
        for index, choice in enumerate((selection, *alternatives)):
            if index:
                assert choice.id is not None
                self.attention.mark_selected(choice.id)
                self.persist_attention(choice.id)
            automatic = choice.reason != SpeechReason.CLICK
            result = await self.cognition.generate_commentary(
                snapshot,
                choice,
                refresh=lambda: self.context(trigger),
                eligible=self.attention.is_selection_eligible,
                actions=self.interactions.offers(choice),
                memory_eligible=(
                    lambda source, ongoing_only=bool(index), automatic=automatic: (
                        self.attention.is_source_available(
                            source.key, source.observed_at
                        )
                        and (not ongoing_only or source.is_ongoing)
                        and (not automatic or self.attention.is_memory_eligible(source))
                    )
                )
                if reflection
                else None,
            )
            preferred = preferred or result
            if not reflection or result.request.selection.memory is not None:
                return result
        assert preferred is not None
        return preferred

    async def comment(
        self,
        reason: SpeechReason,
        context: str | None = None,
        ttl: float | None = None,
        trigger: Trigger | None = None,
        *,
        snapshot: PetContext | None = None,
        selection: AttentionSelection | None = None,
    ) -> None:
        if self.suspended:
            return
        snapshot = self.context(trigger) if snapshot is None else snapshot
        selection = (
            self.attention.select(snapshot, reason, context)
            if selection is None
            else selection
        )
        focus_snapshot = selection.focus_session
        focus_session = focus_snapshot.id if focus_snapshot else None
        with (
            collect_usage() as usage,
            structlog.contextvars.bound_contextvars(
                observation_id=uuid.uuid4().hex,
                reason=reason,
                subject=selection.id,
                session=focus_session,
            ),
        ):
            started = self.clock()
            outcome = "failed"
            epoch = self.revision
            history_epoch = self.history_epoch
            sessions = self.sessions
            log.debug("observation.started", reason=reason, subject=selection.id)
            try:
                result = await asyncio.wait_for(
                    self.generate_commentary(snapshot, selection, trigger),
                    timeout=self.config.request_timeout,
                )
                if sessions is not self.sessions:
                    outcome = "discarded"
                    return
                if (
                    epoch != self.revision
                    or history_epoch != self.history_epoch
                    or not self.is_selection_valid(result.request.conditions, trigger)
                ):
                    outcome = "discarded"
                    log.info(
                        "observation.discarded",
                        reason="subject_superseded",
                        subject=selection.id,
                    )
                    return
                selection = result.request.selection
                sources = self.selected_sources(selection)
                if self.interactions.is_suppressed(result, sources):
                    if reason == SpeechReason.CLICK:
                        self.interactions.finish_empty()
                    outcome = "suppressed"
                    return
                if result.speak is None:
                    self.mark_considered(selection)
                    outcome = "silent"
                    if reason == SpeechReason.CLICK:
                        self.interactions.finish_empty()
                else:
                    seconds = self.speech_duration(result.speak) if ttl is None else ttl
                    if not result.speak.strip() or seconds == 0:
                        self.emit({"text": result.speak, "ttl": seconds})
                        outcome = "suppressed"
                    else:
                        self.behavior.wake("speech")
                        self.behavior.stop_movement()
                        conversation = self.interactions.decorate(
                            result, sources, seconds
                        )
                        if conversation is None and self.interactions.identifier:
                            self.interactions.clear(cancel_task=False)
                        message = SpeechMessage(
                            text=result.speak,
                            ttl=seconds,
                            reason=reason,
                            selection_id=str(uuid.uuid4()),
                            subject_id=selection.id,
                            revision=selection.subject.revision
                            if selection.subject
                            else None,
                            history_epoch=history_epoch,
                            mentioned_sources=result.mentions,
                            conversation=conversation,
                            conditions=DeliveryConditions(
                                commentary=result.request.conditions,
                                trigger=trigger,
                                request_epoch=epoch,
                            ),
                        )
                        self.delivery.submit(message)
                        outcome = "submitted"
            except CannotRespond as error:
                outcome = "deferred"
                log.info(
                    "observation.deferred", reason=str(error), subject=selection.id
                )
                if reason == SpeechReason.CLICK:
                    self.interactions.finish_empty(
                        "That thread changed. Ask me for another look."
                    )
            except asyncio.CancelledError:
                outcome = "cancelled"
                raise
            except Exception as error:
                log.exception(
                    "observation.failed",
                    reason=reason,
                    session=focus_session,
                    error=str(error),
                )
                if reason == SpeechReason.CLICK:
                    self.interactions.finish_empty(
                        "I couldn't get the work context. Try again in a moment."
                    )
            finally:
                tokens = usage.get(reason)
                log.info(
                    "observation.finished",
                    reason=reason,
                    subject=selection.id,
                    session=focus_session,
                    outcome=outcome,
                    duration_seconds=round(self.clock() - started, 3),
                    **(tokens.model_dump() if tokens is not None else {}),
                )

    def is_selection_valid(
        self, conditions: SpeechConditions, trigger: Trigger | None
    ) -> bool:
        if isinstance(trigger, ReactionTrigger) and not self.reactions.is_relevant(
            trigger.reaction, self.sessions
        ):
            return False
        snapshot = self.context(trigger)
        return (
            self.attention.is_selection_eligible(conditions.selection, snapshot)
            and conditions.is_valid(snapshot)
            and self.cognition.are_source_scopes_valid(conditions)
        )

    def selected_sources(self, selection: AttentionSelection) -> tuple[str, ...]:
        if selection.memory is not None:
            return (selection.memory.key,)
        if selection.update is None or selection.situation in (
            SpeechSituation.PROMPT,
            SpeechSituation.FAILURE,
        ):
            return ()
        reports = tuple(
            report
            for report in self.memory.narrative_reports((selection.update.session,))
            if report.observed_at == selection.update.observed_at
            and (
                isinstance(report.payload, ProgressEvidence)
                or isinstance(report.payload, ExecutionEvidence)
                and report.payload.event
                in {"turn_done", "turn_error", "subagent_done", "interrupt"}
            )
        )
        latest = max(reports, key=lambda report: report.observed_at, default=None)
        if latest is None:
            return ()
        return tuple(
            report_source_key(report)
            for report in reports
            if report.observed_at == latest.observed_at
            and report_prose(report) == report_prose(latest)
        )

    def mark_considered(self, selection: AttentionSelection) -> None:
        reflection = selection.situation == SpeechSituation.IDLE_REFLECTION
        subject = None if reflection else selection.subject
        update = None if reflection else selection.update
        records = self.attention.consideration(subject, update)
        with self.memory.transaction():
            for identifier, history in records.items():
                self.memory.save_attention_history(identifier, history)
            self.memory.record_source_consideration(self.selected_sources(selection))
        self.attention.mark_considered(subject, update)
        self.attention.restore_sources(self.memory.load_source_attention())

    def persist_attention(self, subject_id: SessionIdentity | None) -> None:
        if subject_id is not None:
            self.memory.save_attention_history(
                subject_id, self.attention.history(subject_id)
            )

    def is_delivery_valid(self, message: SpeechMessage) -> bool:
        conditions = message.conditions
        assert conditions is not None
        conversation = message.conversation
        if conversation is not None and (
            conversation["id"] != self.interactions.identifier
            or conversation["revision"] != self.interactions.revision
        ):
            return False
        return (
            message.history_epoch == self.history_epoch
            and conditions.request_epoch == self.revision
            and self.is_selection_valid(conditions.commentary, conditions.trigger)
        )

    def emit_notice(self, packet: SpeechPacket) -> None:
        self.emit(
            {
                "text": packet["text"],
                "ttl": packet["ttl"],
                "delivery_id": packet["delivery_id"],
            }
        )

    def is_notice_valid(self, message: SpeechMessage) -> bool:
        notice = self.memory.coordination.notice(message.message_id)
        return (
            not self.suspended
            and not self.memory.coordination.state().paused
            and not self.interactions.is_interacting
            and not (
                self.conversations is not None and self.conversations.is_interacting
            )
            and notice is not None
            and notice.delivered_at is None
            and notice.text == message.text
        )

    def notice_outcome(self, outcome: DeliveryOutcome) -> None:
        if outcome.status != DeliveryStatus.ACCEPTED:
            return
        notice = self.memory.coordination.notice(outcome.message.message_id)
        if notice is None or notice.delivered_at is not None:
            return
        with self.memory.transaction():
            self.memory.coordination.save_notice(
                replace(
                    notice,
                    revision=notice.revision + 1,
                    delivered_at=self.memory.clock(),
                )
            )
            self.memory.record_acceptance(
                notice.id, notice.text, SpeechReason.COMPLETION
            )
        self.last_speech = (notice.text, self.clock())
        self.attention.last_attempt = self.clock()

    def deliver_notice(self) -> bool:
        if self.memory.coordination.state().paused or self.delivery.has_pending():
            return False
        if self.notice_delivery.has_pending():
            return True
        notice = next(
            (
                record
                for record in self.memory.coordination.notices()
                if record.delivered_at is None
            ),
            None,
        )
        if notice is None or not self.attention.can_speak(
            SpeechReason.COMPLETION, None
        ):
            return False
        self.behavior.stop_movement()
        self.notice_delivery.submit(
            SpeechMessage(
                text=notice.text,
                ttl=self.speech_duration(notice.text),
                reason=SpeechReason.COMPLETION,
                selection_id=notice.id,
                message_id=notice.id,
                history_epoch=self.history_epoch,
            )
        )
        return True

    def delivery_outcome(self, outcome: DeliveryOutcome) -> None:
        message = outcome.message
        if message.history_epoch != self.history_epoch:
            return
        subject = next(
            (item for item in self.context().work if item.id == message.subject_id),
            None,
        )
        current = subject is not None and subject.revision == message.revision
        selection = (
            message.conditions.commentary.selection if message.conditions else None
        )
        update = selection.update if selection else None
        update_current = bool(subject and update and update in subject.updates)
        considered_subject = (
            (selection.subject if selection else subject) if current else None
        )
        considered_update = update if update_current else None
        if selection and selection.situation == SpeechSituation.IDLE_REFLECTION:
            considered_subject, considered_update = None, None
        if outcome.status == DeliveryStatus.ACCEPTED:
            records = self.attention.consideration(
                considered_subject, considered_update
            )
            if considered_subject is not None and message.conditions is not None:
                content_revision = message.conditions.commentary.content_revision
                if content_revision is not None:
                    history = records[considered_subject.id]
                    history.spoken_content_revision = content_revision
            with self.memory.transaction():
                self.memory.record_acceptance(
                    message.message_id,
                    message.text,
                    message.reason,
                    subject_id=message.subject_id,
                    revision=message.revision,
                )
                for identifier, history in records.items():
                    self.memory.save_attention_history(identifier, history)
                self.memory.record_source_mentions(message.mentioned_sources)
                if selection:
                    self.memory.record_source_consideration(
                        self.selected_sources(selection)
                    )
            self.last_speech = (message.text, self.clock())
            self.attention.last_attempt = self.clock()
        elif outcome.status == DeliveryStatus.FAILED:
            try:
                self.memory.record_delivery_failure(
                    message.message_id,
                    message.reason,
                    text=message.text,
                    subject_id=message.subject_id,
                    revision=message.revision,
                )
            except sqlite3.Error as error:
                log.exception("speech.delivery_record_failed", error=str(error))
                return
        if outcome.status == DeliveryStatus.ACCEPTED:
            self.attention.histories.update(records)
            self.attention.restore_sources(self.memory.load_source_attention())
        log.info(
            "speech.delivery",
            message=message.message_id,
            text=message.text,
            subject=message.subject_id,
            version=message.revision,
            outcome=outcome.status,
        )

    async def probe_processes(self) -> None:
        sessions = self.sessions
        result = await self.processes.refresh(sessions.process_owners())
        if sessions is self.sessions and not self.suspended:
            sessions.apply_process_status(result)

    def tick(self) -> None:
        now = self.clock()
        if not self.suspended:
            if self.coordinator is not None:
                self.coordinator.tick()
            for provider in self.providers:
                provider.tick()
            if self.process_task is None or self.process_task.done():
                if self.process_task is not None and not self.process_task.cancelled():
                    self.process_task.result()
                self.process_task = asyncio.create_task(self.probe_processes())
            self.delivery.tick()
            self.notice_delivery.tick()
            self.interactions.delivery.tick()
        if now >= self.next_prune:
            self.sessions.prune()
            self.next_prune = now + 1
        if not self.suspended:
            self.behavior.tick(self.agent_activity.activity(), self.sessions)
        if self.task and self.task.done():
            if not self.task.cancelled():
                self.task.result()
            self.task = None
            self.reason = None
        if self.suspended or self.task is not None or not self.behavior.status:
            return
        self._request_commentary(now)

    def _request_commentary(self, now: float) -> None:
        if (
            self.memory.coordination.state().paused
            and self.manual is None
            and not self.clicked
        ):
            return
        if (
            self.conversations is not None
            and self.conversations.is_interacting
            and self.manual is None
        ):
            return
        if (
            self.interactions.is_interacting
            and not self.clicked
            and self.manual is None
        ):
            return
        if self.manual is None and not self.clicked and self.deliver_notice():
            return
        snapshot = self.context()
        check_in = tuple(
            key
            for key in self.sessions.check_in_due(self.config.idle_message_seconds)
            if not self.sessions.by_id[key].parent
            or self.sessions.by_id[key].is_waiting_for_input
        )
        request = self.attention.request(
            snapshot,
            check_in=check_in,
            manual=self.manual,
            clicked=self.clicked,
            care=self.behavior.care_remark,
            reaction_relevant=lambda trigger: self.reactions.is_relevant(
                trigger, self.sessions
            ),
            delivery_pending=self.delivery.has_pending(),
        )
        if request is None:
            return
        if request.reason == SpeechReason.EXPLICIT:
            self.manual = None
        elif request.reason == SpeechReason.CLICK:
            self.clicked = False
        elif request.reason == SpeechReason.CARE:
            self.behavior.care_remark = None
        if not self.attention.can_speak(request.reason, request.trigger):
            return
        snapshot = self.context(request.trigger)
        selection = self.attention.select(
            snapshot,
            request.reason,
            request.context,
            request.eligible,
            reflection=request.reflection,
        )
        if request.reason == SpeechReason.CLICK:
            choice = self.interactions.choose(snapshot, selection)
            if choice is None:
                self.interactions.finish_empty()
                return
            selection = choice
        if not self.attention.is_selection_eligible(selection, snapshot):
            return
        if request.reason == SpeechReason.CHECK_IN and selection.focus_session:
            self.sessions.by_id[selection.focus_session.id].last_check_in = now
        self.attention.mark_scheduled(selection.id)
        self.persist_attention(selection.id)
        self.behavior.stop_movement()
        self.reason = request.reason
        self.task = asyncio.create_task(
            self.comment(
                request.reason,
                request.context,
                request.ttl,
                request.trigger,
                snapshot=snapshot,
                selection=selection,
            )
        )

    async def close(self) -> None:
        if self.coordinator is not None:
            await self.coordinator.close()
        self.notice_delivery.close()
        if self.conversations is not None:
            await self.conversations.close()
        for task in self.companion_tasks:
            task.cancel()
        await asyncio.gather(*self.companion_tasks, return_exceptions=True)
        if self.companion is not None:
            await self.companion.close()
        self.delivery.close()
        await self.interactions.close()
        tasks = [task for task in (self.task, self.process_task) if task]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.gather(*(provider.close() for provider in self.providers))
