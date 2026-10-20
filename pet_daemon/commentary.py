import asyncio
import logging
import sqlite3
import time

from .delivery import SpeechDelivery
from .telemetry import event, observation, trace
from .work import build_work


log = logging.getLogger("pet-daemon")


class Commentary:
    def __init__(self, config, *, sessions, episodes, attention, behavior, cognition, repositories, memory, emit, clock, rng):
        self.config = config
        self.sessions = sessions
        self.episodes = episodes
        self.attention = attention
        self.behavior = behavior
        self.cognition = cognition
        self.repositories = repositories
        self.memory = memory
        self.emit = emit
        self.clock = clock
        self.rng = rng
        self.task = None
        self.reason = None
        self.revision = 0
        self.speech_revision = 0
        self.suspended = False
        self.manual = None
        self.clicked = False
        self.last_speech = None
        self.last_work = {}
        self.delivery = SpeechDelivery(emit, clock=clock, on_outcome=self.surfacing_outcome,
                                       valid=memory.work.is_current)

    def reset(self, sessions, episodes):
        self.preempt()
        self.sessions = sessions
        self.episodes = episodes
        self.manual = None
        self.clicked = False
        self.last_speech = None
        self.last_work.clear()
        self.delivery.clear()
        self.attention.reset()

    def sleep_changed(self, sleeping):
        self.suspended = sleeping
        if sleeping:
            self.preempt()
            self.last_speech = None
        else:
            self.attention.pending = None
            for lane in self.sessions.lanes.values():
                if lane.active:
                    lane.last_check_in = self.clock()
            self.last_work = {subject.id: subject.revision for subject in self.snapshot()["work"]}
            self.attention.next_observation = self.clock() + self.rng.uniform(*self.config.observation_seconds)

    def speak(self, context, ttl=None):
        self.behavior.wake("speech_request")
        self.behavior.stop_movement()
        self.preempt()
        self.manual = (context, ttl)
        trace("speech.requested", context=context, ttl=ttl)

    def queue(self, reason, trigger=None):
        self.attention.queue(reason, trigger, self.sessions.failed)

    def snapshot(self, trigger=None):
        snapshot = {**self.sessions.snapshot(), "trigger": trigger, "sleeping": self.behavior.sleeping}
        for session in snapshot["sessions"]:
            session["connected_work"] = self.memory.work.session_context(session["id"])
        snapshot["work"] = tuple(subject for subject in build_work(snapshot, self.memory.work, now=time.time())
                                 if subject.kind != "opportunity" or self.delivery.can_submit(subject.id, subject.revision))
        return snapshot

    def clear_speech(self):
        try:
            self.memory.clear_speech()
        except sqlite3.Error as error:
            log.warning("Could not clear pet speech: %s", error)
            return
        self.delivery.clear()
        self.speech_revision += 1
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

    async def observe(self, reason, context=None, ttl=None, trigger=None):
        if self.suspended:
            return
        with observation() as usage:
            started = self.clock()
            outcome = "failed"
            focus_session = None
            revision = self.revision
            speech_revision = self.speech_revision
            sessions = self.sessions
            include_activity = reason == "check_in"
            keys = trigger.get("sessions") if include_activity and trigger else None
            episode_key = trigger.get("target_session") if trigger and "episode" in trigger else None
            if episode_key:
                keys = [self.sessions.aliases.get(episode_key, episode_key)]
            observation_context = self.sessions.observation_context(include_activity, keys)
            trace("observation.started", reason=reason)
            timeout = self.config.request_timeout
            try:
                snapshot = self.snapshot(trigger)
                selected = self.cognition.selected_sessions(snapshot, reason)
                focus_snapshot = selected[0] if selected else None
                focus_session = focus_snapshot["id"] if focus_snapshot else None
                if (self.config.summarize_first_request_only
                        and self.cognition.should_summarize(focus_snapshot, reason)):
                    focus = self.sessions.lanes[focus_snapshot["id"]]
                    focus.first_summary_attempted = True
                    try:
                        self.memory.save_context(focus_snapshot["id"], focus.first_prompt, focus.session_context,
                                                           focus.recent_requests, focus.request_count, True)
                    except sqlite3.Error as error:
                        log.warning("Could not save pet session context: %s", error)
                        focus_snapshot["first_summary_attempted"] = True
                result = await asyncio.wait_for(
                    self.cognition.observe(snapshot, reason, context, refresh=lambda: self.snapshot(trigger)),
                    timeout=timeout,
                )
                if sessions is not self.sessions:
                    outcome = "discarded"
                    event("observation.discarded", reason="memory_cleared", trigger=reason, session=focus_session)
                    return
                if keys is None:
                    keys = result["observed_sessions"]
                    observation_context = {key: value for key, value in observation_context.items() if key in keys}
                relevant = observation_context == self.sessions.observation_context(include_activity, keys)
                if reason == "care":
                    relevant = relevant and snapshot["activity"] == self.sessions.snapshot()["activity"]
                opportunity = result["observed_opportunity"]
                opportunity_relevant = opportunity is None or self.memory.work.is_current(*opportunity)
                connected_relevant = all(self.memory.work.session_context(key).get("revision") == version
                                         for key, version in result["observed_work"].items())
                if episode_key:
                    canonical = self.sessions.aliases.get(episode_key, episode_key)
                    before = next(iter(observation_context.values()), None)
                    current = self.sessions.observation_context(include_activity, [canonical]).get(canonical)
                    if trigger["episode"] == "waiting" and before is not None and current is not None:
                        before, current = before[:5], current[:5]
                    relevant = self.episodes.relevant(trigger, self.sessions) and before == current
                if revision != self.revision or (context is None and (not relevant or not opportunity_relevant or not connected_relevant)):
                    outcome = "discarded"
                    event("observation.discarded", reason="preempted" if revision != self.revision else "subject_changed",
                          trigger=reason, session=focus_session, decision=result)
                    return
                covered = set(result["observed_sessions"])
                if opportunity:
                    selected_work = next(subject for subject in snapshot["work"] if subject.id == opportunity[0])
                    covered.update(selected_work.payload["sessions"])
                    covered.add(opportunity[0])
                for subject in snapshot["work"]:
                    if subject.id in covered:
                        self.last_work[subject.id] = subject.revision
                self.save_summary(result)
                if speech_revision != self.speech_revision:
                    outcome = "discarded"
                    event("observation.discarded", reason="speech_cleared", trigger=reason, session=focus_session)
                    return
                if result["speak"] is None:
                    outcome = "silent"
                elif opportunity is not None:
                    speech = result["speak"]
                    self.behavior.wake("speech")
                    self.behavior.stop_movement()
                    self.delivery.submit(speech, self.speech_duration(speech), *opportunity)
                    outcome = "submitted"
                else:
                    outcome = self.deliver_speech(result["speak"], reason, ttl, focus_session)
            except asyncio.CancelledError:
                outcome = "cancelled"
                trace("observation.cancelled")
                raise
            except Exception as error:
                event("observation.failed", level=logging.WARNING, reason=reason, session=focus_session,
                      error=str(error), exc_info=True)
            finally:
                event("observation.finished", reason=reason, session=focus_session, outcome=outcome,
                      duration_seconds=round(self.clock() - started, 3), **usage.get(reason, {}))

    def save_summary(self, result):
        if result["session_context"] is None:
            return
        summary = " ".join(result["session_context"].split())
        key = result["context_session"]
        focus = self.sessions.lanes.get(key)
        if not summary or focus is None:
            return
        current = self.sessions.snapshot(keys=[key])["sessions"][0]
        if self.cognition.summary_version(current) != result["context_version"]:
            trace("memory.context_discarded", reason="session_changed", session=key)
            return
        focus.session_context = summary
        focus.goal = summary
        try:
            self.memory.save_context(key, focus.first_prompt, summary, focus.recent_requests,
                                     focus.request_count, focus.first_summary_attempted)
        except sqlite3.Error as error:
            log.warning("Could not save pet session context: %s", error)
        trace("memory.context_updated", session=key)

    def surfacing_outcome(self, identifier, version, outcome):
        self.memory.work.record_surfacing(identifier, version, outcome)
        if outcome == "accepted":
            speech = self.delivery.speech(identifier, version)
            self.last_speech = (speech, self.clock())
            self.attention.last_attempt = self.clock()
            self.memory.record_speech(speech, "opportunity")
        event("opportunity.delivery", opportunity=identifier, version=version, outcome=outcome)

    def deliver_speech(self, speech, reason, ttl, session):
        visible = bool(speech.strip()) and ttl != 0
        if visible:
            self.behavior.wake("speech")
            self.behavior.stop_movement()
        seconds = self.speech_duration(speech) if ttl is None else ttl
        delivered = self.emit({"text": speech, "ttl": seconds})
        event("speech.delivered" if delivered else "speech.delivery_failed",
              level=logging.INFO if delivered else logging.WARNING,
              session=session, text=speech, ttl=seconds)
        if not delivered:
            return "delivery_failed"
        if not visible:
            return "suppressed"
        self.last_speech = (speech, self.clock())
        self.attention.last_attempt = self.clock()
        self.clicked = False
        self.memory.record_speech(speech, reason)
        return "spoken"

    def next_request(self, now):
        if not self.behavior.status:
            return None
        if self.manual is not None:
            context, ttl = self.manual
            self.manual = None
            return ("explicit", context, ttl, None)
        if self.clicked:
            self.clicked = False
            return ("click", None, None, None)
        pending = self.attention.pending
        if pending and now >= pending[1]:
            reason, _, trigger = pending
            self.attention.pending = None
            if not trigger or "episode" not in trigger or self.episodes.relevant(trigger, self.sessions):
                return (reason, None, None, trigger)
            trace("observation.skipped", reason=reason, snapshot=self.sessions.snapshot())
            return None
        check_in = [key for key in self.sessions.check_in_due(self.config.idle_message_seconds)
                    if not self.sessions.lanes[key].waiting]
        if check_in:
            selected = self.attention.choose(self.sessions.snapshot(), "check_in", set(check_in))
            key = selected["id"]
            return ("check_in", None, None, {"sessions": [key], "target_session": key})
        if now >= self.attention.next_observation:
            self.attention.next_observation = now + self.rng.uniform(*self.config.observation_seconds)
            snapshot = self.snapshot()
            snapshot["work"] = tuple(subject for subject in snapshot["work"]
                                     if subject.kind == "opportunity" or
                                     (subject.priority < 4 or subject.payload) and self.last_work.get(subject.id) != subject.revision)
            work = self.attention.choose_work(snapshot)
            if work and work.kind == "opportunity":
                return ("routine", None, None, {"target_opportunity": work.id})
            if work:
                return ("routine", None, None, {"target_session": work.id})
            trace("observation.skipped", reason="unchanged_activity")
            return None
        if self.behavior.care_remark is not None:
            care = self.behavior.care_remark
            self.behavior.care_remark = None
            if not self.sessions.engaged:
                return ("care", None, None, {"care": care})
        return None

    def tick(self):
        if not self.suspended:
            self.delivery.tick()
        if self.task and self.task.done():
            if not self.task.cancelled():
                self.task.result()
            self.task = None
            self.reason = None
        if self.suspended or self.task is not None:
            return
        now = self.clock()
        request = self.next_request(now)
        if request is None:
            return
        reason, context, ttl, trigger = request
        if not self.attention.allowed(reason, trigger):
            trace("observation.skipped", reason="attention_cooldown")
            return
        snapshot = self.snapshot(trigger)
        selected = self.attention.choose(snapshot, reason)
        work = self.attention.choose_work(snapshot)
        key = work.id if work and work.kind == "opportunity" else selected["id"] if selected else None
        if reason == "check_in" and key in self.sessions.lanes:
            self.sessions.lanes[key].last_check_in = now
        self.attention.scheduled(key)
        self.behavior.stop_movement()
        self.reason = reason
        trace("observation.scheduled", reason=reason, next_observation=self.attention.next_observation)
        self.task = asyncio.create_task(self.observe(reason, context, ttl, trigger))

    async def close(self):
        self.delivery.close()
        if self.task:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
