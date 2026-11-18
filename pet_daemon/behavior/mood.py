import math
from collections import OrderedDict
from collections.abc import Callable
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from ..config import MoodConfig, Nonnegative, Unit


class MoodCue(StrEnum):
    NOVELTY = "novelty"
    STRUGGLE = "struggle"
    RECOVERY = "recovery"
    VICTORY = "victory"
    INTERRUPTED = "interrupted"
    WAITING = "waiting"
    DISCOVERY = "discovery"
    LONG_HAUL = "long_haul"
    HANDOFF = "handoff"
    SENTIMENT = "sentiment"
    UNANSWERED = "unanswered"


class IdlePhase(StrEnum):
    FRESH = "fresh"
    PRESENCE = "presence"
    SETTLED = "settled"


class MoodState(BaseModel):
    model_config = ConfigDict(frozen=True)
    energy: Unit
    curiosity: Unit
    social_appetite: Unit
    agitation: Unit
    updated_at: Nonnegative
    last_work_at: Nonnegative
    last_novelty_at: Nonnegative
    last_engagement_at: Nonnegative
    external_engagement_at: Nonnegative | None = None
    presence_claimed_through: Nonnegative | None = None
    applied_events: tuple[tuple[str, float], ...] = ()


_EFFECTS = {
    MoodCue.NOVELTY: (0, 0.18, 0, 0),
    MoodCue.STRUGGLE: (-0.08, 0, 0, 0.44),
    MoodCue.RECOVERY: (0.15, 0.05, 0, -0.50),
    MoodCue.VICTORY: (0.20, 0.05, 0, -0.22),
    MoodCue.INTERRUPTED: (-0.07, -0.12, 0, 0.28),
    MoodCue.WAITING: (0, 0.07, 0, 0.07),
    MoodCue.DISCOVERY: (0.10, 0.35, 0, 0),
    MoodCue.LONG_HAUL: (-0.24, -0.20, 0, 0),
    MoodCue.HANDOFF: (0.04, 0.16, 0, 0),
    MoodCue.SENTIMENT: (0, 0, 0, 0.08),
    MoodCue.UNANSWERED: (0, 0, 0, 0.40),
}
_IDENTITY_SECONDS = 24 * 60 * 60
EVENT_HISTORY_LIMIT = 512
_MOOD_FIELDS = ("energy", "curiosity", "social_appetite", "agitation")
_DEFAULT_MOOD = MoodConfig()


def _decay(value: float, target: float, seconds: float, half_life: float) -> float:
    return target + (value - target) * math.exp2(-seconds / half_life)


class Mood:
    def __init__(
        self,
        wall_clock: Callable[[], float],
        *,
        state: MoodState | None = None,
        config: MoodConfig = _DEFAULT_MOOD,
    ) -> None:
        self.wall_clock = wall_clock
        self.config = config
        self._sleep_at: float | None = None
        now = self._now()
        self._working = False
        if state is None:
            self._state = MoodState(
                energy=config.energy,
                curiosity=config.curiosity,
                social_appetite=config.social_appetite,
                agitation=config.agitation,
                updated_at=now,
                last_work_at=now,
                last_novelty_at=now,
                last_engagement_at=now,
            )
        else:
            self._state = self._resume(state, now)
        self._events = OrderedDict(
            (identifier, min(now, timestamp))
            for identifier, timestamp in self._state.applied_events[
                -EVENT_HISTORY_LIMIT:
            ]
            if timestamp > now - _IDENTITY_SECONDS
        )
        self._state = self._state.model_copy(update={"applied_events": ()})

    def _now(self) -> float:
        if self._sleep_at is not None:
            return self._sleep_at
        return self.wall_clock()

    def _resume(self, state: MoodState, now: float) -> MoodState:
        elapsed = max(0, now - state.updated_at)
        return state.model_copy(
            update={
                "energy": _decay(state.energy, self.config.energy, elapsed, 300),
                "curiosity": _decay(
                    state.curiosity, self.config.curiosity, elapsed, 240
                ),
                "social_appetite": _decay(
                    state.social_appetite, self.config.social_appetite, elapsed, 480
                ),
                "agitation": _decay(
                    state.agitation, self.config.agitation, elapsed, 60
                ),
                "updated_at": now,
                "last_work_at": min(now, state.last_work_at),
                "last_novelty_at": min(now, state.last_novelty_at),
                "last_engagement_at": min(now, state.last_engagement_at),
                "presence_claimed_through": min(
                    now,
                    max(
                        state.presence_claimed_through or 0,
                        self._presence_window(state),
                    ),
                ),
            }
        )

    def _advance(self) -> MoodState:
        state = self._state
        now = self._now()
        elapsed = max(0, now - state.updated_at)
        energy_target = self.config.energy * (0.45 if self._working else 1)
        curiosity_target = self.config.curiosity * (0.45 if self._working else 1)
        return state.model_copy(
            update={
                "energy": _decay(state.energy, energy_target, elapsed, 240),
                "curiosity": _decay(state.curiosity, curiosity_target, elapsed, 180),
                "social_appetite": _decay(
                    state.social_appetite, self._social_target, elapsed, 600
                ),
                "agitation": _decay(
                    state.agitation, self.config.agitation, elapsed, 60
                ),
                "last_work_at": now if self._working else state.last_work_at,
                "updated_at": now,
            }
        )

    @property
    def _social_target(self) -> float:
        baseline = self.config.social_appetite
        return baseline + (1 - baseline) * 0.75

    def snapshot(self) -> MoodState:
        state = self._advance()
        external = state.external_engagement_at
        if external is None or external <= state.last_engagement_at:
            return state
        projected = _decay(
            self.config.social_appetite,
            self._social_target,
            max(0, state.updated_at - external),
            600,
        )
        return state.model_copy(
            update={"social_appetite": min(state.social_appetite, projected)}
        )

    def checkpoint(self) -> MoodState:
        state = self._advance()
        self._prune_events(state.updated_at)
        return state.model_copy(update={"applied_events": tuple(self._events.items())})

    def suspend(self) -> None:
        self._state = self._advance()
        self._sleep_at = self._state.updated_at
        self._working = False

    def resume(self) -> None:
        if self._sleep_at is None:
            return
        now = max(self._sleep_at, self.wall_clock())
        self._sleep_at = None
        self._state = self._resume(self._state, now)

    def observe_work(self, active: bool, *, changed: bool = False) -> None:
        if self._working != active or changed:
            state = self._advance()
            self._state = state.model_copy(update={"last_work_at": state.updated_at})
        self._working = active

    def _prune_events(self, now: float) -> None:
        cutoff = now - _IDENTITY_SECONDS
        while self._events and next(iter(self._events.values())) <= cutoff:
            self._events.popitem(last=False)

    @staticmethod
    def _event_time(observed_at: float | None, now: float) -> float | None:
        if observed_at is None:
            return now
        if not 0 <= observed_at <= now + 1:
            return None
        return min(observed_at, now)

    def _remember(self, event_id: str, now: float) -> bool:
        self._prune_events(now)
        if event_id in self._events:
            return False
        if len(self._events) == EVENT_HISTORY_LIMIT:
            self._events.popitem(last=False)
        self._events[event_id] = now
        return True

    def observe(
        self, cue: MoodCue, event_id: str, *, observed_at: float | None = None
    ) -> bool:
        now = self._now()
        timestamp = self._event_time(observed_at, now)
        if timestamp is None or timestamp <= now - _IDENTITY_SECONDS:
            return False
        if not self._remember(event_id, now):
            return False
        state = self._advance()
        effects = {
            key: min(1, max(0, getattr(state, key) + amount * self.config.sensitivity))
            for key, amount in zip(_MOOD_FIELDS, _EFFECTS[cue], strict=True)
        }
        if cue not in {MoodCue.SENTIMENT, MoodCue.UNANSWERED}:
            effects |= {
                "last_novelty_at": state.updated_at,
            }
        self._state = state.model_copy(update=effects)
        return True

    def interact(self, event_id: str, *, observed_at: float | None = None) -> bool:
        now = self._now()
        timestamp = self._event_time(observed_at, now)
        if timestamp is None or timestamp < self._state.last_engagement_at:
            return False
        if not self._remember(event_id, now):
            return False
        state = self._advance()
        self._state = state.model_copy(
            update={
                "last_engagement_at": timestamp,
                "social_appetite": _decay(
                    self.config.social_appetite,
                    self._social_target,
                    state.updated_at - timestamp,
                    600,
                ),
                "agitation": state.agitation * 0.6,
            }
        )
        return True

    def set_external_engagement(self, observed_at: float | None) -> bool:
        timestamp = (
            None if observed_at is None else self._event_time(observed_at, self._now())
        )
        if observed_at is not None and timestamp is None:
            return False
        if timestamp == self._state.external_engagement_at:
            return False
        self._state = self._state.model_copy(
            update={"external_engagement_at": timestamp}
        )
        return True

    @property
    def last_local_engagement_at(self) -> float:
        return self._state.last_engagement_at

    @staticmethod
    def _last_engagement(state: MoodState) -> float:
        return max(state.last_engagement_at, state.external_engagement_at or 0)

    @classmethod
    def _presence_window(cls, state: MoodState) -> float:
        return max(state.last_novelty_at, cls._last_engagement(state))

    @property
    def engagement_age(self) -> float:
        return self._now() - self._last_engagement(self._state)

    @property
    def novelty_age(self) -> float:
        return self._now() - self._state.last_novelty_at

    @property
    def work_quiet_age(self) -> float:
        return 0 if self._working else self._now() - self._state.last_work_at

    @property
    def idle_phase(self) -> IdlePhase:
        age = min(self.engagement_age, self.novelty_age)
        fresh, settle = self.config.idle_period
        if age < fresh:
            return IdlePhase.FRESH
        return IdlePhase.PRESENCE if age < settle else IdlePhase.SETTLED

    @property
    def presence_due(self) -> bool:
        claimed = self._state.presence_claimed_through
        is_unclaimed = claimed is None or self._presence_window(self._state) > claimed
        return self.idle_phase == IdlePhase.PRESENCE and is_unclaimed

    def claim_presence(self) -> bool:
        if not self.presence_due:
            return False
        self._state = self._state.model_copy(
            update={"presence_claimed_through": self._presence_window(self._state)}
        )
        return True

    @property
    def emotion(self) -> str:
        state = self.snapshot()
        if state.agitation > 0.38:
            return "a little vexed"
        if state.curiosity > 0.68:
            return "curious"
        if state.social_appetite > 0.58:
            return "eager for company"
        if state.energy < 0.36:
            return "mellow"
        return "at ease"
