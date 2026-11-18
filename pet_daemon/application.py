import asyncio
import math
import sqlite3
import uuid
from collections import OrderedDict
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Protocol

import structlog

from .attention import Attention, AttentionSelection
from .behavior.controller import Behavior
from .behavior.mood import IdlePhase, Mood, MoodCue, MoodState
from .commentary import Cognition, SpeechConditions
from .commentary.cognition import CannotRespond, CommentaryResult
from .commentary.usage import UsageTokens, collect_usage
from .config import Config
from .delivery import (
    DeliveryConditions,
    DeliveryOutcome,
    SpeechDelivery,
    SpeechMessage,
)
from .domain import (
    CareKind,
    DeliveryStatus,
    ExecutionOutcome,
    SpeechReason,
    SpeechSituation,
)
from .identity import SessionIdentity
from .interactions import Interactions
from .logging import get_logger
from .occasion import (
    OCCASION_SECONDS,
    Occasion,
    Timers,
    direction_changed,
    occasion_fingerprint,
)
from .occasions import Occasions
from .prose import work_prose
from .runtime import (
    CareTrigger,
    HookEvent,
    HookEventKind,
    OverlayPacket,
    PetContext,
    RandomSource,
    ReactionKind,
    ReactionTrigger,
    Trigger,
)
from .session_notes import SessionNotes
from .sessions.coordinator import ActivityCoordinator
from .sessions.processes import ProcessTracker
from .sessions.reactions import SessionReactions
from .sessions.tracking import Sessions, SessionUpdate
from .storage import Memory
from .work.history import (
    SessionContext,
    build_session_entry,
)
from .work.observations import RepositoryObservation

log = get_logger("pet-daemon")

RECENT_CLICK_LIMIT = 512
SAID_SECONDS = 24 * 60 * 60
REACTION_CUES = {
    ReactionKind.START: MoodCue.NOVELTY,
    ReactionKind.TROUBLE: MoodCue.STRUGGLE,
    ReactionKind.FAILURE: MoodCue.STRUGGLE,
    ReactionKind.RECOVERY: MoodCue.RECOVERY,
    ReactionKind.SUCCESS: MoodCue.VICTORY,
    ReactionKind.CANCELLED: MoodCue.INTERRUPTED,
    ReactionKind.WAITING: MoodCue.WAITING,
}


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
        self.mood, self.timers = self.load_behavior()
        self.behavior.mood = self.mood
        self.occasions = Occasions(memory, sessions)
        self.recent_occasion: Occasion | None = None
        self.session_notes = SessionNotes(memory.path, memory.pet, memory.clock)
        self.next_notes = 0.0
        self.hook_reactions: list[ReactionTrigger] = []
        self.checkout_target = checkout_target
        self.reconcile_work = reconcile_work
        self.task: asyncio.Task[None] | None = None
        self.reason: SpeechReason | None = None
        self.revision = 0
        self.history_epoch = 0
        self.suspended = False
        self.manual: tuple[str, float | None] | None = None
        self.clicked = False
        self.recent_clicks: OrderedDict[str, None] = OrderedDict()
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

    def load_behavior(self) -> tuple[Mood, Timers]:
        data = self.memory.load_behavior()
        state = MoodState.model_validate_json(data[0]) if data else None
        timers = Timers.model_validate_json(data[1]) if data else Timers()
        mood = Mood(
            self.memory.clock,
            state=state,
            config=self.config.behavior.mood,
        )
        return mood, timers

    def save_mood(self) -> None:
        self.memory.save_behavior(
            self.mood.checkpoint().model_dump_json(),
            self.timers.model_dump_json(),
        )

    def engage(self, event_id: str | None = None) -> None:
        now = self.memory.clock()
        identifier = event_id or uuid.uuid4().hex
        previous = now - self.mood.engagement_age
        if not self.mood.interact(f"renderer:{identifier}", observed_at=now):
            return
        self.notice_return(previous, now)
        try:
            self.save_mood()
        except sqlite3.Error as error:
            log.exception("behavior.engagement_save_failed", error=str(error))

    def notice_return(self, previous: float, observed_at: float) -> None:
        self.timers = self.timers.model_copy(update={"unanswered_bid_at": None})
        if observed_at - previous >= self.config.behavior.mood.idle_period[0]:
            self.recent_occasion = Occasion(
                fingerprint=occasion_fingerprint(SpeechSituation.RETURNED, observed_at),
                fact="The developer engaged again after some time away.",
                situation=SpeechSituation.RETURNED,
                expires_at=observed_at + OCCASION_SECONDS,
            )

    def interaction_displayed(self, text: str) -> None:
        self.last_speech = (text, self.clock())
        self.attention.last_attempt = self.clock()

    def clear_memory(self) -> None:
        try:
            self.memory.clear()
        except sqlite3.Error as error:
            log.exception("memory.clear_failed", error=str(error))
            return
        self.preempt()
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
        self.mood, self.timers = self.load_behavior()
        self.behavior.mood = self.mood
        self.recent_occasion = None
        self.occasions = Occasions(self.memory, self.sessions)
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
            self.timers = self.timers.model_copy(update={"unanswered_bid_at": None})
            self.mood.suspend()
            self.save_mood()
            self.behavior.suspend("system_sleep")
            self.preempt()
            self.interactions.clear()
            for provider in self.providers:
                provider.pause()
            if self.process_task:
                self.process_task.cancel()
        else:
            self.mood.resume()
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
        if self.interactions.identifier:
            self.interactions.clear()
        self.manual = (context, ttl)
        log.debug("speech.requested", context=context, ttl=ttl)

    def queue(self, reason: SpeechReason, trigger: Trigger | None = None) -> None:
        if isinstance(trigger, ReactionTrigger):
            self.hook_reactions.append(trigger)
        self.attention.queue(reason, trigger, self.sessions.has_current_failures)

    def observe_hook_mood(
        self, hook: HookEvent, result: SessionUpdate, operation: str | None
    ) -> bool:
        session_id = result.session
        session = self.sessions.by_id[session_id]
        self.mood.observe_work(self.sessions.is_engaged, changed=True)
        now = self.memory.clock()
        observed_at = min(hook.capture_at, now) if hook.capture_at is not None else now
        changed = self.observe_reactions(
            occasion_fingerprint(hook, session_id, hook.turn or session.generation)
        )
        if hook.event == HookEventKind.INTENT and not self.suspended:
            if hook.goal and direction_changed(hook.goal):
                timers = self.timers.cancelled(session_id)
                changed |= timers != self.timers
                self.timers = timers
                self.recent_occasion = Occasion(
                    fingerprint=occasion_fingerprint(
                        SpeechSituation.DIRECTION_CHANGE, session_id, session.generation
                    ),
                    fact="The work changed direction.",
                    situation=SpeechSituation.DIRECTION_CHANGE,
                    expires_at=observed_at + OCCASION_SECONDS,
                )
            if session.user_sentiment is not None and self.session_notes.human_prompt(
                session_id, session.turn
            ):
                changed |= self.mood.observe(
                    MoodCue.SENTIMENT,
                    occasion_fingerprint(MoodCue.SENTIMENT, session_id, session.turn),
                )
        elif (
            hook.event in {HookEventKind.TURN_DONE, HookEventKind.SUBAGENT_DONE}
            and session.outcome == ExecutionOutcome.SUCCESS
            and (
                shared := self.occasions.shared_success(
                    session_id, hook.progress, observed_at
                )
            )
        ):
            self.recent_occasion = shared
        return self.observe_tool(hook, result, operation, observed_at) or changed

    def observe_tool(
        self, hook: HookEvent, result: SessionUpdate, operation: str | None, at: float
    ) -> bool:
        prior = self.timers
        if hook.event == HookEventKind.INTERRUPT:
            self.timers = self.timers.cancelled(result.session)
        elif hook.event in {HookEventKind.TOOL_DONE, HookEventKind.TOOL_ERROR}:
            if result.failure or hook.event == HookEventKind.TOOL_ERROR:
                self.timers = self.timers.failed(
                    result.session, result.activity, operation, at
                )
            elif hook.outcome not in {
                ExecutionOutcome.CANCELLED,
                ExecutionOutcome.ERROR,
            }:
                self.timers, occasion = self.timers.recovered(
                    result.session, result.activity, operation, at
                )
                if occasion is not None and not any(
                    item.reaction.kind == ReactionKind.RECOVERY
                    for item in self.hook_reactions
                ):
                    self.recent_occasion = occasion
                    if not self.suspended:
                        self.mood.observe(MoodCue.RECOVERY, occasion.fingerprint)
        return self.timers != prior

    def observe_reactions(self, fingerprint: str) -> bool:
        changed = False
        if self.suspended:
            return changed
        for trigger in self.hook_reactions:
            reaction = trigger.reaction
            cue = REACTION_CUES.get(reaction.kind)
            if cue is not None:
                changed |= self.mood.observe(
                    cue,
                    occasion_fingerprint(fingerprint, reaction.kind, reaction.activity),
                )
        if changed:
            self.timers = self.timers.model_copy(update={"unanswered_bid_at": None})
        return changed

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
            self.reconcile_targets()
        except (sqlite3.Error, ValueError) as error:
            log.exception("work.ingest_failed", error=str(error))

    def _ingest_hook(self, hook: HookEvent) -> None:
        if hook.repo:
            hook.repo = str(Path(hook.repo).resolve())
        previous = self.sessions.by_id.get(
            self.sessions.aliases.get(hook.subject, hook.subject)
        )
        call = previous.calls.get(hook.call) if previous and hook.call else None
        operation = work_prose(call.tool if call else hook.tool, 160) or None
        pending = self.attention.pending
        recent_occasion = self.recent_occasion
        self.hook_reactions.clear()
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
                if self.observe_hook_mood(hook, result, operation):
                    self.save_mood()
        except Exception as error:
            self.attention.pending = pending
            self.mood, self.timers = self.load_behavior()
            self.recent_occasion = recent_occasion
            if self.suspended:
                self.mood.suspend()
            self.behavior.mood = self.mood
            if not isinstance(error, (sqlite3.Error, ValueError)):
                raise
            log.exception("work.evidence_failed", error=str(error))
            return
        try:
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
        if self.interactions.receive(packet):
            if packet.get("event") in {"interaction_action", "interaction_dismissed"}:
                self.engage()
            return
        feedback = packet.get("event")
        if feedback == "speech_accepted":
            try:
                self.delivery.acknowledge(packet)
            except sqlite3.Error as error:
                log.exception("renderer.acceptance_failed", error=str(error))
        elif feedback == "status":
            previous = self.behavior.status
            if self.behavior.update_renderer_status(packet):
                status = self.behavior.status
                assert status is not None
                if previous and previous.dragging and not status.dragging:
                    self.engage()
                self.system_sleep_changed(status.system_sleeping)
                if not self.suspended:
                    self.behavior.tick(self.agent_activity.activity(), self.sessions)
        elif feedback in ("system_sleep", "system_wake"):
            self.system_sleep_changed(feedback == "system_sleep")
        elif feedback == "click":
            event_id = packet.get("event_id")
            self.click(
                event_id=event_id
                if isinstance(event_id, str) and 0 < len(event_id) <= 128
                else None
            )
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

    def click(self, *, more: bool = False, event_id: str | None = None) -> None:
        if event_id is not None:
            if event_id in self.recent_clicks:
                return
            self.recent_clicks[event_id] = None
            if len(self.recent_clicks) > RECENT_CLICK_LIMIT:
                self.recent_clicks.popitem(last=False)
        self.engage(event_id)
        self.behavior.defer_care()
        self.behavior.wake("click")
        self.behavior.react("click", self.sessions)
        if self.task and self.reason == SpeechReason.CLICK and not self.task.done():
            return
        self.preempt()
        self.interactions.begin(more=more)
        self.clicked = True
        log.debug("speech.requested", reason="click")

    def rounds(self) -> None:
        self.engage()
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
        if selection.occasion is not None:
            return await self.cognition.generate_commentary(
                snapshot,
                selection,
                refresh=lambda: self.context(trigger),
                eligible=self.attention.is_selection_eligible,
                occasion=selection.occasion,
                feeling=self.mood.emotion,
            )
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
                sources = self.occasions.selected_sources(selection)
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
                tokens = UsageTokens()
                for item in usage.values():
                    tokens += item
                log.info(
                    "observation.finished",
                    reason=reason,
                    subject=selection.id,
                    session=focus_session,
                    outcome=outcome,
                    duration_seconds=round(self.clock() - started, 3),
                    **tokens.model_dump(),
                )

    def is_selection_valid(
        self, conditions: SpeechConditions, trigger: Trigger | None
    ) -> bool:
        if conditions.selection.reason == SpeechReason.CARE and (
            self.suspended or self.interactions.is_interacting
        ):
            return False
        occasion = conditions.selection.occasion
        if occasion is not None and (
            self.suspended
            or occasion.expires_at <= self.memory.clock()
            or self.memory.has_said(occasion.fingerprint)
        ):
            return False
        if isinstance(trigger, ReactionTrigger) and not self.reactions.is_relevant(
            trigger.reaction, self.sessions
        ):
            return False
        snapshot = self.context(trigger)
        if occasion and not self.is_occasion_current(
            occasion, conditions.selection, snapshot
        ):
            return False
        if occasion and self.is_presence_occasion(occasion, conditions.selection):
            return not snapshot.sleeping and self.cognition.are_source_scopes_valid(
                conditions
            )
        return (
            self.attention.is_selection_eligible(conditions.selection, snapshot)
            and conditions.is_valid(snapshot)
            and self.cognition.are_source_scopes_valid(conditions)
        )

    def is_occasion_current(
        self, occasion: Occasion, selection: AttentionSelection, snapshot: PetContext
    ) -> bool:
        if occasion.situation == SpeechSituation.COMPANY:
            return (
                not self.sessions.is_engaged
                and self.mood.idle_phase == IdlePhase.PRESENCE
            )
        if self.is_presence_occasion(occasion, selection):
            return occasion in self.presence_occasions(snapshot)
        if occasion.situation in {
            SpeechSituation.CHECK_IN_RUNNING,
            SpeechSituation.CHECK_IN_WAITING,
        }:
            current = self.occasions.select(selection.refresh(snapshot), None)
            return current is not None and current.fingerprint == occasion.fingerprint
        return True

    @staticmethod
    def is_presence_occasion(occasion: Occasion, selection: AttentionSelection) -> bool:
        return (
            selection.reason == SpeechReason.CARE
            and occasion.situation != selection.situation
        )

    def mark_considered(self, selection: AttentionSelection) -> None:
        reflection = selection.situation == SpeechSituation.IDLE_REFLECTION
        subject = None if reflection else selection.subject
        update = None if reflection else selection.update
        records = self.attention.consideration(subject, update)
        with self.memory.transaction():
            for identifier, history in records.items():
                self.memory.save_attention_history(identifier, history)
            self.memory.record_source_consideration(
                self.occasions.selected_sources(selection)
            )
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
                        self.occasions.selected_sources(selection)
                    )
                    if (occasion := selection.occasion) is not None:
                        self.memory.mark_said(
                            occasion.fingerprint, self.memory.clock() + SAID_SECONDS
                        )
                        if occasion.situation == SpeechSituation.COMPANY:
                            self.timers = self.timers.model_copy(
                                update={"unanswered_bid_at": self.memory.clock()}
                            )
                            self.save_mood()
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

    def refresh_notes(self) -> None:
        if self.clock() < self.next_notes:
            return
        self.next_notes = self.clock() + 1
        try:
            self.session_notes.refresh()
        except sqlite3.Error as error:
            log.warning("session_notes.unavailable", error=str(error))
            return
        now = self.memory.clock()
        previous = now - self.mood.engagement_age
        presence = self.session_notes.presence
        if presence is None or presence.renderer_event_id:
            return
        if presence.status == "away":
            if (
                self.recent_occasion
                and self.recent_occasion.situation == SpeechSituation.RETURNED
            ):
                self.recent_occasion = None
            return
        if presence.observed_at <= previous or not self.mood.set_external_engagement(
            presence.observed_at
        ):
            return
        if now - presence.observed_at < OCCASION_SECONDS:
            self.notice_return(previous, presence.observed_at)
        self.save_mood()

    def presence_occasions(self, snapshot: PetContext) -> tuple[Occasion, ...]:
        now = self.memory.clock()
        workstreams = frozenset(
            self.occasions.session_thread(session.id, session.repo)
            for session in snapshot.sessions
            if not session.retired
        )
        recent = self.recent_occasion
        ignored = self.timers.ignored(now, self.config.behavior.mood.idle_period[0])
        return (
            *((recent,) if recent and recent.expires_at > now else ()),
            *self.session_notes.occasions(workstreams, now),
            *((ignored,) if ignored else ()),
        )

    def tick(self) -> None:
        now = self.clock()
        if not self.suspended:
            self.refresh_notes()
            for provider in self.providers:
                provider.tick()
            if self.process_task is None or self.process_task.done():
                if self.process_task is not None and not self.process_task.cancelled():
                    self.process_task.result()
                self.process_task = asyncio.create_task(self.probe_processes())
            self.delivery.tick()
            self.interactions.delivery.tick()
        if now >= self.next_prune:
            self.sessions.prune()
            self.next_prune = now + 1
        if not self.suspended:
            self.mood.observe_work(self.sessions.is_engaged)
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
            self.interactions.is_interacting
            and not self.clicked
            and self.manual is None
        ):
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
            self.schedule_presence(snapshot)
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
        character = self.occasions.character_for(selection, self.session_notes)
        if character is not None:
            selection = selection.model_copy(update={"character": character})
        if request.reason not in {SpeechReason.CLICK, SpeechReason.EXPLICIT}:
            occasion = self.occasions.select(selection, request.trigger)
            if (
                occasion is None
                or self.memory.has_said(occasion.fingerprint)
                or occasion.expires_at <= self.memory.clock()
            ):
                if request.reason != SpeechReason.CHECK_IN:
                    self.mark_considered(selection)
                self.schedule_presence(snapshot)
                return
            if (
                (recent := self.recent_occasion)
                and recent.fingerprint == occasion.fingerprint
                and recent.expires_at > self.memory.clock()
            ):
                occasion = recent
            selection = selection.model_copy(update={"occasion": occasion})
            if (
                occasion.situation == SpeechSituation.CHECK_IN_RUNNING
                and self.mood.observe(MoodCue.LONG_HAUL, occasion.fingerprint)
            ):
                self.save_mood()
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

    def schedule_presence(self, snapshot: PetContext) -> bool:
        if snapshot.sleeping or self.delivery.has_pending():
            return False
        if not self.attention.can_speak(SpeechReason.CARE):
            return False
        occasion = next(
            (
                item
                for item in self.presence_occasions(snapshot)
                if not self.memory.has_said(item.fingerprint)
            ),
            None,
        )
        if occasion is None:
            if (
                self.sessions.is_engaged
                or any(
                    session.turn_active and session.active_tools
                    for session in snapshot.sessions
                )
                or not self.config.behavior.care
                or not self.mood.claim_presence()
            ):
                return False
            self.save_mood()
            now = self.memory.clock()
            occasion = Occasion(
                fingerprint=occasion_fingerprint(SpeechSituation.COMPANY, now),
                fact="The developer has been quiet for a while.",
                situation=SpeechSituation.COMPANY,
                expires_at=now + OCCASION_SECONDS,
            )
        elif occasion.situation == SpeechSituation.UNANSWERED and self.mood.observe(
            MoodCue.UNANSWERED, occasion.fingerprint
        ):
            self.save_mood()
        care = CareTrigger(care=CareKind.ATTENTION)
        selection = self.attention.select(
            snapshot.model_copy(update={"trigger": care}), SpeechReason.CARE
        )
        selection = selection.model_copy(update={"occasion": occasion})
        self.attention.mark_scheduled()
        self.reason = SpeechReason.CARE
        self.task = asyncio.create_task(
            self.comment(SpeechReason.CARE, snapshot=snapshot, selection=selection)
        )
        return True

    async def close(self) -> None:
        self.delivery.close()
        await self.interactions.close()
        tasks = [task for task in (self.task, self.process_task) if task]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.gather(*(provider.close() for provider in self.providers))
