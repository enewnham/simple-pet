import asyncio
import math
import sqlite3
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Protocol

import structlog

from .attention import Attention, AttentionSelection
from .behavior.controller import Behavior
from .commentary import Cognition, SpeechConditions
from .commentary.cognition import StoryUnavailable
from .commentary.usage import collect_usage
from .config import Config
from .delivery import DeliveryConditions, DeliveryOutcome, SpeechDelivery, SpeechMessage
from .domain import DeliveryStatus, SpeechReason
from .identity import SessionIdentity
from .logging import get_logger
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
from .work.history import SessionContext, build_session_entry
from .work.narrative import semantic_revision
from .work.observations import RepositoryObservation

log = get_logger("pet-daemon")


class ObservationProvider(Protocol):
    def tick(self) -> None: ...
    def pause(self) -> None: ...
    def resume(self) -> None: ...
    def clear(self) -> None: ...
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
                }
            ),
            clock=clock,
            on_outcome=self.delivery_outcome,
            valid=self.is_delivery_valid,
        )

    def clear_memory(self) -> None:
        try:
            self.memory.clear()
        except sqlite3.Error as error:
            log.exception("memory.clear_failed", error=str(error))
            return
        self.preempt()
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
            self.behavior.suspend("system_sleep")
            self.preempt()
            for provider in self.providers:
                provider.pause()
            if self.process_task:
                self.process_task.cancel()
        else:
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

    def speak(self, context: str, ttl: float | None = None) -> None:
        self.behavior.wake("speech_request")
        self.behavior.stop_movement()
        self.preempt()
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
                session.summary = context.summary
                session.recent_requests = tuple(context.recent_requests)
                session.request_count = context.request_count
                session.first_summary_attempted = context.first_summary_attempted
        subjects = self.memory.connected_work(runtime)
        for subject in subjects:
            subject.revision = semantic_revision(subject.story)
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
        feedback = packet.get("event")
        if feedback == "speech_accepted":
            try:
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

    def click(self) -> None:
        self.behavior.defer_care()
        self.behavior.wake("click")
        self.behavior.react("click", self.sessions)
        if self.task and self.reason == SpeechReason.CLICK and not self.task.done():
            return
        self.preempt()
        self.clicked = True
        log.debug("speech.requested", reason="click")

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
                    self.cognition.generate_commentary(
                        snapshot,
                        selection,
                        refresh=lambda: self.context(trigger),
                        eligible=self.attention.is_selection_eligible,
                    ),
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
                if result.speak is None:
                    self.mark_considered(selection)
                    outcome = "silent"
                else:
                    seconds = self.speech_duration(result.speak) if ttl is None else ttl
                    if not result.speak.strip() or seconds == 0:
                        self.emit({"text": result.speak, "ttl": seconds})
                        outcome = "suppressed"
                    else:
                        self.behavior.wake("speech")
                        self.behavior.stop_movement()
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
                            conditions=DeliveryConditions(
                                commentary=result.request.conditions,
                                trigger=trigger,
                                request_epoch=epoch,
                            ),
                        )
                        self.delivery.submit(message)
                        outcome = "submitted"
            except StoryUnavailable as error:
                outcome = "deferred"
                log.info(
                    "observation.deferred", reason=str(error), subject=selection.id
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
        return self.attention.is_selection_eligible(
            conditions.selection, snapshot
        ) and conditions.is_valid(snapshot)

    def mark_considered(self, selection: AttentionSelection) -> None:
        records = self.attention.consideration(selection.subject, selection.update)
        with self.memory.transaction():
            for identifier, history in records.items():
                self.memory.save_attention_history(identifier, history)
        self.attention.mark_considered(selection.subject, selection.update)

    def persist_attention(self, subject_id: SessionIdentity | None) -> None:
        if subject_id is not None:
            self.memory.save_attention_history(
                subject_id, self.attention.history(subject_id)
            )

    def is_delivery_valid(self, message: SpeechMessage) -> bool:
        conditions = message.conditions
        assert conditions is not None
        return (
            message.history_epoch == self.history_epoch
            and conditions.request_epoch == self.revision
            and self.is_selection_valid(conditions.commentary, conditions.trigger)
        )

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
        considered_subject = subject if current else None
        considered_update = update if update_current else None
        if outcome.status == DeliveryStatus.ACCEPTED:
            records = self.attention.consideration(
                considered_subject, considered_update
            )
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
            self.attention.mark_considered(considered_subject, considered_update)
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
            for provider in self.providers:
                provider.tick()
            if self.process_task is None or self.process_task.done():
                if self.process_task is not None and not self.process_task.cancelled():
                    self.process_task.result()
                self.process_task = asyncio.create_task(self.probe_processes())
            self.delivery.tick()
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
            snapshot, request.reason, request.context, request.eligible
        )
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
        self.delivery.close()
        tasks = [task for task in (self.task, self.process_task) if task]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.gather(*(provider.close() for provider in self.providers))
