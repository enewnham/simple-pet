from collections.abc import Callable
from dataclasses import dataclass

from ..domain import ActionKind
from ..runtime import RandomSource
from .mood import MoodState


@dataclass(frozen=True)
class Observation:
    active: bool
    working: bool
    investigating: bool
    waiting: bool
    interacting: bool
    pointer_recent: bool
    can_roam: bool
    can_nap: bool
    restlessness: float
    rest_pressure: float
    mood: MoodState | None = None
    settled: bool = False


@dataclass(frozen=True)
class Candidate:
    name: str
    score: float
    role: str


@dataclass(frozen=True)
class BehaviorSpec:
    name: str
    role: str
    eligible: Callable[[Observation], bool]
    utility: Callable[[Observation], float]


@dataclass(frozen=True)
class ReactionSpec:
    role: str
    preferred: str
    priority: int
    kind: ActionKind = ActionKind.REACTION
    fallback: str = "idle"


ANIMATION_ROLES = {
    "review": "attention",
    "waiting": "attention",
    "running": "work",
    "failed": "error",
    "jumping": "celebrate",
    "waving": "greeting",
    "idle": "rest",
}

REACTIONS = {
    "start": ReactionSpec("attention", "review", 1, fallback="running"),
    "investigation": ReactionSpec("attention", "review", 1, fallback="running"),
    "waiting": ReactionSpec("attention", "waiting", 3, fallback="review"),
    "trouble": ReactionSpec("error", "failed", 4, ActionKind.ERROR, "review"),
    "failure": ReactionSpec("error", "failed", 4, ActionKind.ERROR, "review"),
    "frustration": ReactionSpec("error", "failed", 2),
    "recovery": ReactionSpec("celebrate", "jumping", 4, ActionKind.SUCCESS, "waving"),
    "success": ReactionSpec("celebrate", "jumping", 4, ActionKind.SUCCESS, "waving"),
    "cancelled": ReactionSpec("attention", "review", 3, fallback="idle"),
    "click": ReactionSpec("greeting", "waving", 5, fallback="review"),
}

CATALOGUE = (
    BehaviorSpec("rest", "rest", lambda o: True, lambda o: 0.35),
    BehaviorSpec("focus", "work", lambda o: o.working, lambda o: 0.85),
    BehaviorSpec(
        "attentive", "rest", lambda o: o.waiting and not o.working, lambda o: 0.85
    ),
    BehaviorSpec(
        "observe",
        "gaze",
        lambda o: o.pointer_recent and not o.interacting,
        lambda o: 0.45,
    ),
    BehaviorSpec(
        "roam",
        "walk",
        lambda o: o.can_roam,
        lambda o: (
            0.2 + 0.6 * o.restlessness
            if o.working
            else (0.2 + 0.5 * o.restlessness) * (0.5 if o.active else 1)
        ),
    ),
    BehaviorSpec(
        "nap", "sleep", lambda o: o.can_nap, lambda o: 0.9 * o.rest_pressure**2
    ),
)


class Policy:
    def candidates(self, observation: Observation) -> list[Candidate]:
        return [
            Candidate(
                spec.name,
                spec.utility(observation) * _mood_weight(spec.name, observation),
                "attention"
                if spec.name == "focus" and observation.investigating
                else spec.role,
            )
            for spec in CATALOGUE
            if spec.eligible(observation)
        ]

    def select(
        self,
        observation: Observation,
        last_completed: str | None,
        rng: RandomSource,
        available: Callable[[Candidate], bool],
    ) -> Candidate:
        choices = [
            Candidate(
                c.name, c.score * (0.9 if c.name == last_completed else 1), c.role
            )
            for c in self.candidates(observation)
            if available(c)
        ]
        best = max(c.score for c in choices)
        choices = [c for c in choices if c.score >= best * 0.9]
        threshold = rng.random() * sum(c.score for c in choices)
        for candidate in choices:
            threshold -= candidate.score
            if threshold <= 0:
                return candidate
        return choices[-1]


def _mood_weight(name: str, observation: Observation) -> float:
    state = observation.mood
    if state is None:
        return 1
    if name == "rest" or name == "nap":
        return 1.3 - 0.5 * state.energy
    if name == "focus" or name == "observe":
        return 0.8 + 0.4 * state.curiosity
    if name == "roam":
        engagement = 0.4 * state.energy + 0.2 * state.social_appetite
        return (0.75 + engagement) * (0.7 if observation.settled else 1)
    return 1
