import asyncio
import logging
import math
from dataclasses import asdict, dataclass
import uuid
import sqlite3
import time
from typing import Protocol

from .delivery import SpeechDelivery, SpeechMessage
from .episodes import Episodes
from .events import AgentEvents
from .sessions import Sessions
from .runtime import SessionSnapshot, WorkSnapshot
from .work.observations import RepositoryObservation, RepositoryTarget
from .commentary import EvidenceManifest
from .telemetry import event, observation, trace
from .work import build_work


log = logging.getLogger("pet-daemon")


class ObservationProvider(Protocol):
    def associate(self, session: str, path: str, *, refresh: bool = False) -> None: ...
    def reconcile_targets(self, targets: tuple[RepositoryTarget, ...]) -> None: ...
    def tick(self) -> None: ...
    def pause(self) -> None: ...
    def resume(self) -> None: ...
    def clear(self) -> None: ...
    async def close(self) -> None: ...


@dataclass(frozen=True)
class MessageSupport:
    manifest: EvidenceManifest
    trigger: dict | None
    request_epoch: int


class Application:
    def __init__(
        self,
        config,
        *,
        sessions,
        episodes,
        attention,
        behavior,
        cognition,
        providers: tuple[ObservationProvider, ...],
        processes,
        memory,
        emit,
        clock,
        rng,
    ):
        self.config = config
        self.sessions = sessions
        self.episodes = episodes
        self.attention = attention
        self.behavior = behavior
        self.cognition = cognition
        self.providers = tuple(providers)
        self.processes = processes
        self.memory = memory
        self.emit = emit
        self.clock = clock
        self.rng = rng
        self.task = None
        self.reason = None
        self.revision = 0
        self.history_epoch = 0
        self.suspended = False
        self.manual = None
        self.clicked = False
        self.last_speech = None
        self.attention.restore(memory.load_attention_history())
        self.process_task = None
        self.next_prune = 0
        self.reconciling = False
        self.events = AgentEvents(
            config,
            sessions=sessions,
            episodes=episodes,
            memory=memory,
            behavior=behavior,
            queue=self.queue,
            clock=clock,
            rng=rng,
        )
        self.delivery = SpeechDelivery(
            emit,
            clock=clock,
            on_outcome=self.delivery_outcome,
            valid=self.delivery_valid,
        )

    def clear_memory(self):
        try:
            self.memory.clear()
        except sqlite3.Error as error:
            log.warning("Could not clear pet memory: %s", error)
            return
        self.preempt()
        self.history_epoch += 1
        self.sessions = Sessions(
            self.clock,
            self.config.stale_seconds,
            self.config.completed_retention_seconds,
            self.config.activity_fresh_seconds,
        )
        self.episodes = Episodes()
        self.events.reset(self.sessions, self.episodes)
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
        event("memory.cleared", pet=self.memory.pet)

    def system_sleep_changed(self, sleeping):
        if sleeping == self.suspended:
            return
        self.suspended = sleeping
        event("system.sleep" if sleeping else "system.wake")
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
            for lane in self.sessions.lanes.values():
                if lane.active:
                    lane.last_check_in = self.clock()
            self.attention.next_observation = self.clock() + self.rng.uniform(
                *self.config.observation_seconds
            )

    def speak(self, context, ttl=None):
        self.behavior.wake("speech_request")
        self.behavior.stop_movement()
        self.preempt()
        self.manual = (context, ttl)
        trace("speech.requested", context=context, ttl=ttl)

    def queue(self, reason, trigger=None):
        self.attention.queue(reason, trigger, self.sessions.failed)

    def snapshot(self, trigger=None):
        runtime = self.sessions.snapshot()
        state = self.memory.work.snapshot()
        subjects = build_work(runtime, state, now=time.time())
        views = {subject.id: subject.view for subject in subjects}
        members = []
        for session in runtime["sessions"]:
            work = state.current(session["id"])
            view = views.get(work.id) if work else None
            members.append(SessionSnapshot.capture(session, view))
        return WorkSnapshot(
            runtime["activity"],
            tuple(members),
            self.behavior.sleeping,
            trigger=trigger,
            work=subjects,
            convergence=runtime["convergence"],
        )

    def ingest(self, observation: RepositoryObservation):
        try:
            with self.memory.transaction():
                self.memory.save_repository(observation.subject, asdict(observation))
                self.memory.work.observe_repository(observation)
            self.reconcile_targets()
        except (sqlite3.Error, ValueError) as error:
            log.warning("Could not ingest work observation: %s", error)

    def ingest_hook(self, hook):
        packet = hook.packet()
        result = self.events.handle(packet, self.suspended)
        if not result:
            return
        key = result["session"]
        lane = self.sessions.lanes[key]
        try:
            with self.memory.transaction():
                if packet["event"] in ("intent", "subagent_start") and packet.get(
                    "goal"
                ):
                    aliases = tuple(
                        f"{packet['harness']}:{alias}"
                        for alias in packet.get("aliases", ())
                    )
                    self.memory.work.session_intent(
                        key, packet["goal"], aliases=aliases
                    )
                self.memory.work.session_event(key, packet)
                if lane.repo:
                    self.memory.associate_repository(key, lane.repo)
            for provider in self.providers:
                provider.associate(
                    key,
                    lane.repo,
                    refresh=packet["event"] in ("intent", "subagent_start"),
                )
            self.reconcile_targets()
        except (sqlite3.Error, ValueError) as error:
            log.warning("Could not record work evidence: %s", error)

    def reconcile_targets(self):
        if self.reconciling:
            return
        self.reconciling = True
        try:
            targets = self.memory.work.collection_targets()
            for provider in self.providers:
                provider.reconcile_targets(targets)
        finally:
            self.reconciling = False

    def receive(self, packet):
        kind = packet.get("kind")
        if kind == "clear_speech":
            self.clear_speech()
        elif kind == "clear_memory":
            self.clear_memory()
        elif kind == "renderer":
            self.receive_renderer(packet)
        elif kind == "speak" and isinstance(packet.get("context"), str):
            ttl = packet.get("ttl")
            if ttl is None or (
                type(ttl) in (int, float) and math.isfinite(ttl) and ttl >= 0
            ):
                self.speak(packet["context"], ttl)

    def receive_renderer(self, packet):
        feedback = packet.get("event")
        if feedback == "speech_accepted":
            try:
                self.delivery.acknowledge(packet)
            except sqlite3.Error as error:
                log.warning("Could not record renderer acceptance: %s", error)
        elif feedback == "status":
            if self.behavior.renderer_status(packet):
                sleeping = packet.get("system_sleeping")
                if type(sleeping) is bool:
                    self.system_sleep_changed(sleeping)
                if not self.suspended:
                    self.behavior.tick(self.events.activity(), self.sessions)
        elif feedback in ("system_sleep", "system_wake"):
            self.system_sleep_changed(feedback == "system_sleep")
        elif feedback == "click":
            self.click()
        else:
            self.behavior.completion(packet)

    def clear_speech(self):
        try:
            self.memory.clear_speech()
        except sqlite3.Error as error:
            log.warning("Could not clear pet speech: %s", error)
            return
        self.history_epoch += 1
        self.delivery.clear()
        self.last_speech = None
        self.emit({"text": "", "ttl": 0})
        event("speech.cleared", pet=self.memory.pet)

    def speech_duration(self, speech):
        return max(self.config.speech_ttl, 3 + len(speech.split()) / 2)

    def preempt(self):
        self.delivery.cancel_pending()
        self.revision += 1
        if self.task:
            self.task.cancel()
        self.attention.pending = None

    def click(self):
        self.behavior.satisfy()
        self.behavior.wake("click")
        self.behavior.react("click", self.sessions)
        if self.task and self.reason == "click" and not self.task.done():
            return
        self.preempt()
        self.clicked = True
        trace("speech.requested", reason="click")

    async def observe(
        self,
        reason,
        context=None,
        ttl=None,
        trigger=None,
        *,
        snapshot=None,
        selection=None,
    ):
        if self.suspended:
            return
        snapshot = self.snapshot(trigger) if snapshot is None else snapshot
        selection = (
            self.attention.select(snapshot, reason, context)
            if selection is None
            else selection
        )
        with observation() as usage:
            started = self.clock()
            outcome = "failed"
            epoch = self.revision
            history_epoch = self.history_epoch
            sessions = self.sessions
            focus_snapshot = selection.focus_session
            focus_session = focus_snapshot["id"] if focus_snapshot else None
            trace("observation.started", reason=reason, subject=selection.id)
            try:
                if (
                    self.config.summarize_first_request_only
                    and self.cognition.should_summarize(focus_snapshot, reason)
                ):
                    focus = self.sessions.lanes[focus_session]
                    focus.first_summary_attempted = True
                    self.memory.save_context(
                        focus_session,
                        focus.first_prompt,
                        focus.session_context,
                        focus.recent_requests,
                        focus.request_count,
                        True,
                    )
                result = await asyncio.wait_for(
                    self.cognition.observe(
                        snapshot, selection, refresh=lambda: self.snapshot(trigger)
                    ),
                    timeout=self.config.request_timeout,
                )
                if sessions is not self.sessions:
                    outcome = "discarded"
                    return
                self.save_summary(result)
                if (
                    epoch != self.revision
                    or history_epoch != self.history_epoch
                    or not self.selection_valid(result.prepared.manifest, trigger)
                ):
                    outcome = "discarded"
                    event(
                        "observation.discarded",
                        reason="subject_superseded",
                        subject=selection.id,
                    )
                    return
                selection = result.prepared.selection
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
                        action = selection.action
                        opportunity = (action.id, action.version) if action else None
                        message = SpeechMessage(
                            result.speak,
                            seconds,
                            reason,
                            str(uuid.uuid4()),
                            selection.id,
                            selection.subject.revision if selection.subject else None,
                            opportunity,
                            history_epoch,
                            support=MessageSupport(
                                result.prepared.manifest, trigger, epoch
                            ),
                        )
                        self.delivery.submit(message)
                        self.mark_considered(selection)
                        outcome = "submitted"
            except asyncio.CancelledError:
                outcome = "cancelled"
                raise
            except Exception as error:
                event(
                    "observation.failed",
                    level=logging.WARNING,
                    reason=reason,
                    session=focus_session,
                    error=str(error),
                    exc_info=True,
                )
            finally:
                event(
                    "observation.finished",
                    reason=reason,
                    subject=selection.id,
                    session=focus_session,
                    outcome=outcome,
                    duration_seconds=round(self.clock() - started, 3),
                    **usage.get(reason, {}),
                )

    def selection_valid(self, manifest, trigger):
        if (
            trigger
            and "episode" in trigger
            and not self.episodes.relevant(trigger, self.sessions)
        ):
            return False
        return manifest.valid(self.snapshot(trigger))

    def mark_considered(self, selection):
        if selection.subject is not None:
            self.attention.mark_considered(selection.subject)
            self.persist_attention(selection.subject.id)

    def persist_attention(self, subject_id):
        if subject_id is not None:
            last_selected, considered = self.attention.history(subject_id)
            self.memory.save_attention_history(subject_id, last_selected, considered)

    def delivery_valid(self, message):
        support = message.support
        return (
            message.history_epoch == self.history_epoch
            and support.request_epoch == self.revision
            and self.selection_valid(support.manifest, support.trigger)
        )

    def save_summary(self, result):
        if result.session_context is None:
            return
        summary = " ".join(result.session_context.split())
        key = result.context_session
        focus = self.sessions.lanes.get(key)
        if not summary or focus is None:
            return
        current = self.sessions.snapshot(keys=[key])["sessions"][0]
        if self.cognition.summary_version(current) != result.context_version:
            trace("memory.context_discarded", reason="session_changed", session=key)
            return
        focus.session_context = summary
        try:
            self.memory.save_context(
                key,
                focus.first_prompt,
                summary,
                focus.recent_requests,
                focus.request_count,
                focus.first_summary_attempted,
            )
        except sqlite3.Error as error:
            log.warning("Could not save pet session context: %s", error)
        trace("memory.context_updated", session=key)

    def delivery_outcome(self, outcome):
        message = outcome.message
        if message.history_epoch != self.history_epoch:
            return
        opportunity_id, version = message.opportunity or (None, None)
        if outcome.status == "accepted":
            self.memory.record_acceptance(
                message.message_id,
                message.text,
                message.reason,
                opportunity_id=opportunity_id,
                version=version,
            )
            self.last_speech = (message.text, self.clock())
            self.attention.last_attempt = self.clock()
        elif outcome.status == "failed":
            try:
                self.memory.record_delivery_failure(
                    message.message_id,
                    message.reason,
                    text=message.text,
                    opportunity_id=opportunity_id,
                    version=version,
                )
            except sqlite3.Error as error:
                log.warning("Could not record failed speech delivery: %s", error)
        event(
            "speech.delivery",
            message=message.message_id,
            subject=message.subject_id,
            version=message.revision,
            outcome=outcome.status,
        )

    async def probe_processes(self):
        sessions = self.sessions
        result = await self.processes.refresh(sessions.process_owners())
        if sessions is self.sessions and not self.suspended:
            sessions.apply_process_status(result)

    def tick(self):
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
            self.behavior.tick(self.events.activity(), self.sessions)
        if self.task and self.task.done():
            if not self.task.cancelled():
                self.task.result()
            self.task = None
            self.reason = None
        if self.suspended or self.task is not None or not self.behavior.status:
            return
        snapshot = self.snapshot()
        check_in = tuple(
            key
            for key in self.sessions.check_in_due(self.config.idle_message_seconds)
            if not self.sessions.lanes[key].waiting
        )
        request = self.attention.request(
            snapshot,
            check_in=check_in,
            manual=self.manual,
            clicked=self.clicked,
            care=self.behavior.care_remark,
            episode_relevant=lambda trigger: self.episodes.relevant(
                trigger, self.sessions
            ),
        )
        if request is None:
            return
        if request.reason == "explicit":
            self.manual = None
        elif request.reason == "click":
            self.clicked = False
        elif request.reason == "care":
            self.behavior.care_remark = None
        if not self.attention.allowed(request.reason, request.trigger):
            return
        snapshot = self.snapshot(request.trigger)
        selection = self.attention.select(
            snapshot, request.reason, request.context, request.eligible
        )
        if request.reason == "check_in" and selection.focus_session:
            self.sessions.lanes[selection.focus_session["id"]].last_check_in = now
        self.attention.scheduled(selection.id)
        self.persist_attention(selection.id)
        self.behavior.stop_movement()
        self.reason = request.reason
        self.task = asyncio.create_task(
            self.observe(
                request.reason,
                request.context,
                request.ttl,
                request.trigger,
                snapshot=snapshot,
                selection=selection,
            )
        )

    async def close(self):
        self.delivery.close()
        tasks = [task for task in (self.task, self.process_task) if task]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.gather(*(provider.close() for provider in self.providers))
