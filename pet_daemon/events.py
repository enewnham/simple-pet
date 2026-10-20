import logging
import sqlite3

from .sentiment import (
    analyzer,
    sentiment_score,
    NEGATIVE_THRESHOLD,
    REACTION_CHANCE,
    COOLDOWN_SECONDS,
    DUPLICATE_SECONDS,
)
from .telemetry import trace, verbose
from .work.history import project_event


log = logging.getLogger("pet-daemon")


class AgentEvents:
    def __init__(
        self, config, *, sessions, episodes, memory, behavior, queue, clock, rng
    ):
        self.config = config
        self.sessions = sessions
        self.episodes = episodes
        self.memory = memory
        self.behavior = behavior
        self.queue = queue
        self.clock = clock
        self.rng = rng
        self.error_until = 0
        self.sentiment_after = float("-inf")
        analyzer()

    def reset(self, sessions, episodes):
        self.sessions = sessions
        self.episodes = episodes
        self.error_until = 0
        self.sentiment_after = float("-inf")

    def activity(self):
        return (
            "error"
            if self.clock() < self.error_until
            else "working"
            if self.sessions.working
            else "idle"
        )

    def handle(self, packet, sleeping=False):
        result = self.sessions.handle(packet)
        if result:
            key = result["session"]
            lane = self.sessions.lanes[key]
            self.restore_context(packet, key, lane)
            self.record(packet, key, lane, result["activity"])
            trigger = {
                **packet,
                "target_session": key,
                "repo": lane.repo,
                "activity": result["activity"],
            }
            if packet["event"] in (
                "turn_done",
                "turn_error",
                "subagent_done",
                "interrupt",
                "session_end",
            ):
                trigger["outcome"] = lane.outcome
            for episode in self.episodes.handle(packet, result, self.sessions):
                if not sleeping:
                    self.behavior.react(episode.kind, self.sessions, key)
                self.queue_episode(episode, trigger)
            if packet["event"] == "intent" and not lane.parent:
                self.react_to_sentiment(packet, key, lane, sleeping)
            if not sleeping:
                self.behavior.tick(self.activity(), self.sessions)
        verbose(
            "hook.received",
            packet=packet,
            result=result,
            snapshot=self.sessions.snapshot(),
        )
        return result

    def restore_context(self, packet, key, lane):
        if lane.context_loaded:
            return
        lane.context_loaded = True
        try:
            stored = self.memory.context(key)
            if stored is None:
                for alias in packet.get("aliases", []):
                    stored = self.memory.context(f"{packet['harness']}:{alias}")
                    if stored is not None:
                        break
            if stored:
                if stored["request_count"] or not lane.request_count:
                    lane.first_prompt = stored["first_prompt"] or lane.first_prompt
                lane.session_context = stored["session_context"]
                lane.recent_requests = [
                    *stored["recent_requests"],
                    *lane.recent_requests,
                ][-4:]
                lane.request_count += stored["request_count"]
                lane.first_summary_attempted = stored["first_summary_attempted"]
                lane.goal = lane.first_prompt
        except sqlite3.Error as error:
            log.warning("Could not restore pet session context: %s", error)

    def record(self, packet, key, lane, activity):
        if packet["event"] in ("intent", "subagent_start") and lane.first_prompt:
            try:
                self.memory.save_context(
                    key,
                    lane.first_prompt,
                    lane.session_context,
                    lane.recent_requests,
                    lane.request_count,
                    lane.first_summary_attempted,
                )
            except sqlite3.Error as error:
                log.warning("Could not save pet session context: %s", error)
        state = {
            "id": key,
            "repo": lane.repo,
            "goal": lane.goal,
            "last_tool": lane.last_tool,
            "outcome": lane.outcome,
            "activity": activity,
        }
        try:
            self.memory.record_event(project_event(packet, state))
        except sqlite3.Error as error:
            log.warning("Could not record pet event: %s", error)

    def queue_episode(self, episode, trigger):
        trigger = {
            **trigger,
            "episode": episode.kind,
            "generation": episode.generation,
            "serial": episode.serial,
            "activity": episode.activity,
        }
        if episode.kind in ("trouble", "failure"):
            self.error_until = self.clock() + self.config.behavior.error_seconds
            self.queue("failure", trigger)
        elif episode.kind == "start" and trigger["event"] in (
            "intent",
            "subagent_start",
        ):
            self.queue("prompt", trigger)
        elif episode.kind == "waiting":
            trigger["sessions"] = [trigger["target_session"]]
            self.queue("check_in", trigger)
        elif episode.kind in ("success", "cancelled", "recovery"):
            self.queue("completion", trigger)

    def react_to_sentiment(self, packet, key, lane, sleeping):
        if not self.config.behavior.sentiment_reactions:
            return
        now = self.clock()
        text = packet.get("sentiment_text", packet.get("goal", ""))
        identity = (packet.get("turn"), text)
        if lane.sentiment_identity == identity and (
            packet.get("turn") or now - lane.sentiment_seen < DUPLICATE_SECONDS
        ):
            trace("sentiment.evaluated", session=key, decision="duplicate")
            return
        lane.sentiment_identity = identity
        lane.sentiment_seen = now
        score = sentiment_score(text)
        lane.user_sentiment = score if score <= NEGATIVE_THRESHOLD else None
        if score > NEGATIVE_THRESHOLD:
            decision = "neutral"
        elif now < self.sentiment_after:
            decision = "cooldown"
        elif sleeping:
            decision = "sleeping"
        elif self.rng.random() >= REACTION_CHANCE:
            decision = "chance"
        elif self.behavior.react("frustration", self.sessions, key):
            self.sentiment_after = now + COOLDOWN_SECONDS
            decision = "played"
        else:
            decision = "blocked"
        trace("sentiment.evaluated", session=key, compound=score, decision=decision)
