from collections.abc import Callable
from dataclasses import asdict

from ..behavior.controller import Behavior
from ..config import Config
from ..domain import PetState, SpeechReason
from ..identity import SessionIdentity
from ..logging import get_logger
from ..runtime import HookEvent, HookEventKind, RandomSource, ReactionTrigger, Trigger
from ..sentiment import (
    COOLDOWN_SECONDS,
    DUPLICATE_SECONDS,
    NEGATIVE_THRESHOLD,
    REACTION_CHANCE,
    analyzer,
    sentiment_score,
)
from .reactions import ReactionKind, SessionReaction, SessionReactions
from .tracking import AgentSession, Sessions, SessionUpdate

log = get_logger("pet-daemon")


class ActivityCoordinator:
    def __init__(
        self,
        config: Config,
        *,
        sessions: Sessions,
        reactions: SessionReactions,
        behavior: Behavior,
        queue: Callable[[SpeechReason, Trigger | None], None],
        clock: Callable[[], float],
        rng: RandomSource,
    ) -> None:
        self.config = config
        self.sessions = sessions
        self.reactions = reactions
        self.behavior = behavior
        self.queue = queue
        self.clock = clock
        self.rng = rng
        self.error_until = 0
        self.sentiment_after = float("-inf")
        analyzer()

    def reset(self, sessions: Sessions, reactions: SessionReactions) -> None:
        self.sessions = sessions
        self.reactions = reactions
        self.error_until = 0
        self.sentiment_after = float("-inf")

    def activity(self) -> PetState:
        return (
            PetState.ERROR
            if self.clock() < self.error_until
            else PetState.WORKING
            if self.sessions.working
            else PetState.IDLE
        )

    def handle(self, packet: HookEvent, sleeping: bool = False) -> SessionUpdate | None:
        result = self.sessions.apply(packet)
        if result:
            key = result.session
            session = self.sessions.by_id[key]
            for reaction in self.reactions.handle(packet, result, self.sessions):
                if session.parent and reaction.kind != ReactionKind.WAITING:
                    continue
                if not sleeping:
                    self.behavior.react(reaction.kind, self.sessions, key)
                self.queue_reaction(reaction, packet)
            if packet.event == HookEventKind.INTENT and not session.parent:
                self.react_to_sentiment(packet, key, session, sleeping)
            if not sleeping:
                self.behavior.tick(self.activity(), self.sessions)
        log.debug(
            "hook.received",
            detail="payload",
            packet=packet.packet(),
            result=asdict(result) if result is not None else None,
            snapshot=asdict(self.sessions.activity_view()),
        )
        return result

    def queue_reaction(self, reaction: SessionReaction, hook: HookEvent) -> None:
        trigger = ReactionTrigger(reaction)
        if reaction.kind in (ReactionKind.TROUBLE, ReactionKind.FAILURE):
            self.error_until = self.clock() + self.config.behavior.error_seconds
            self.queue(SpeechReason.FAILURE, trigger)
        elif reaction.kind == ReactionKind.START and hook.event in (
            HookEventKind.INTENT,
            HookEventKind.SUBAGENT_START,
        ):
            self.queue(SpeechReason.PROMPT, trigger)
        elif reaction.kind == ReactionKind.WAITING:
            session = self.sessions.by_id[reaction.session]
            if session.waiting_for_input:
                self.queue(SpeechReason.CHECK_IN, trigger)
        elif reaction.kind in (
            ReactionKind.SUCCESS,
            ReactionKind.CANCELLED,
            ReactionKind.RECOVERY,
        ):
            self.queue(SpeechReason.COMPLETION, trigger)

    def react_to_sentiment(
        self,
        packet: HookEvent,
        key: SessionIdentity,
        session: AgentSession,
        sleeping: bool,
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
            log.debug("sentiment.evaluated", session=key, decision="duplicate")
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
        log.debug("sentiment.evaluated", session=key, compound=score, decision=decision)
