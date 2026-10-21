from collections.abc import Callable
from dataclasses import asdict

from ..behavior.controller import Behavior
from ..config import Config
from ..runtime import EpisodeTrigger, HookEvent, RandomSource, Trigger
from ..sentiment import (
    COOLDOWN_SECONDS,
    DUPLICATE_SECONDS,
    NEGATIVE_THRESHOLD,
    REACTION_CHANCE,
    analyzer,
    sentiment_score,
)
from ..telemetry import trace, verbose
from .episodes import Episode, Episodes
from .tracking import AgentSession, Sessions, SessionUpdate


class ActivityCoordinator:
    def __init__(
        self,
        config: Config,
        *,
        sessions: Sessions,
        episodes: Episodes,
        behavior: Behavior,
        queue: Callable[[str, Trigger | None], None],
        clock: Callable[[], float],
        rng: RandomSource,
    ) -> None:
        self.config = config
        self.sessions = sessions
        self.episodes = episodes
        self.behavior = behavior
        self.queue = queue
        self.clock = clock
        self.rng = rng
        self.error_until = 0
        self.sentiment_after = float("-inf")
        analyzer()

    def reset(self, sessions: Sessions, episodes: Episodes) -> None:
        self.sessions = sessions
        self.episodes = episodes
        self.error_until = 0
        self.sentiment_after = float("-inf")

    def activity(self) -> str:
        return (
            "error"
            if self.clock() < self.error_until
            else "working"
            if self.sessions.working
            else "idle"
        )

    def handle(self, packet: HookEvent, sleeping: bool = False) -> SessionUpdate | None:
        result = self.sessions.apply(packet)
        if result:
            key = result.session
            session = self.sessions.by_id[key]
            for episode in self.episodes.handle(packet, result, self.sessions):
                if not sleeping:
                    self.behavior.react(episode.kind, self.sessions, key)
                self.queue_episode(episode, packet)
            if packet.event == "intent" and not session.parent:
                self.react_to_sentiment(packet, key, session, sleeping)
            if not sleeping:
                self.behavior.tick(self.activity(), self.sessions)
        verbose(
            "hook.received",
            packet=packet.packet(),
            result=asdict(result) if result is not None else None,
            snapshot=asdict(self.sessions.activity_view()),
        )
        return result

    def queue_episode(self, episode: Episode, hook: HookEvent) -> None:
        trigger = EpisodeTrigger(episode)
        if episode.kind in ("trouble", "failure"):
            self.error_until = self.clock() + self.config.behavior.error_seconds
            self.queue("failure", trigger)
        elif episode.kind == "start" and hook.event in ("intent", "subagent_start"):
            self.queue("prompt", trigger)
        elif episode.kind == "waiting":
            self.queue("check_in", trigger)
        elif episode.kind in ("success", "cancelled", "recovery"):
            self.queue("completion", trigger)

    def react_to_sentiment(
        self, packet: HookEvent, key: str, session: AgentSession, sleeping: bool
    ) -> None:
        if not self.config.behavior.sentiment_reactions:
            return
        now = self.clock()
        text = (
            packet.sentiment_text
            if packet.sentiment_text is not None
            else packet.goal or ""
        )
        identity = (packet.turn, text)
        if session.sentiment_identity == identity and (
            packet.turn or now - session.sentiment_seen < DUPLICATE_SECONDS
        ):
            trace("sentiment.evaluated", session=key, decision="duplicate")
            return
        session.sentiment_identity = identity
        session.sentiment_seen = now
        score = sentiment_score(text)
        session.user_sentiment = score if score <= NEGATIVE_THRESHOLD else None
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
