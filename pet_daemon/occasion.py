import hashlib
import json
import re
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from .domain import Activity, SpeechSituation
from .identity import SessionIdentity

OCCASION_SECONDS = 300
STRUGGLE_SECONDS = 1800

_DIRECTION_CHANGE = re.compile(
    r"^(?:actually[, ]+)?(?:change (?:of )?(?:plan|direction)\b|"
    r"(?:let['’]s |we should )?(?:switch|pivot) (?:to|from)\b|"
    r"(?:abandon|discard|drop|cancel|stop) (?:that|this|the|our) (?:plan|approach|work)\b)",
    re.IGNORECASE,
)


def occasion_fingerprint(*parts: object) -> str:
    payload = json.dumps(
        parts,
        ensure_ascii=False,
        sort_keys=True,
        default=lambda value: (
            value.model_dump(mode="json")
            if isinstance(value, BaseModel)
            else str(value)
        ),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class Occasion(BaseModel):
    model_config = ConfigDict(frozen=True)
    fingerprint: str
    fact: str | None
    situation: SpeechSituation
    expires_at: float
    sources: tuple[str, ...] = ()


@dataclass(frozen=True)
class OccasionTone:
    instruction: str
    feeling: str | None = None
    more_effort: bool = False


OCCASION_TONE = {
    SpeechSituation.ROUTINE: OccasionTone("react to the new development"),
    SpeechSituation.PROMPT: OccasionTone("show interest in the work"),
    SpeechSituation.COMPLETION: OccasionTone("mark the result"),
    SpeechSituation.FAILURE: OccasionTone("react to the setback"),
    SpeechSituation.STRUGGLE: OccasionTone("react to the struggle"),
    SpeechSituation.RECOVERY: OccasionTone("celebrate the recovery", "relieved", True),
    SpeechSituation.INTERRUPTED: OccasionTone(
        "acknowledge the interruption", "a little disappointed"
    ),
    SpeechSituation.CHECK_IN_WAITING: OccasionTone("keep the developer company"),
    SpeechSituation.CHECK_IN_RUNNING: OccasionTone(
        "keep the developer company during the wait", "a little bored"
    ),
    SpeechSituation.CHECK_IN_QUIET: OccasionTone(
        "keep the developer company in the quiet"
    ),
    SpeechSituation.CARE_EXERCISE: OccasionTone(
        "invite the developer to take a brief break"
    ),
    SpeechSituation.CARE_ATTENTION: OccasionTone("give a casual greeting"),
    SpeechSituation.CARE_CONSTERNATION: OccasionTone("react playfully to the quiet"),
    SpeechSituation.COMPANY: OccasionTone("offer a brief moment of company"),
    SpeechSituation.RETURNED: OccasionTone(
        "greet the developer on their return", "pleased to have company", True
    ),
    SpeechSituation.HANDOFF: OccasionTone("react to the handoff", more_effort=True),
    SpeechSituation.CONTRIBUTOR: OccasionTone("react to the new contributor"),
    SpeechSituation.EXTERNAL_WAIT: OccasionTone("react to who is waiting for what"),
    SpeechSituation.UNANSWERED: OccasionTone(
        "express the pet's wish for company", "a little neglected"
    ),
    SpeechSituation.SHARED_SUCCESS: OccasionTone(
        "celebrate the shared result", "pleased with the shared result", True
    ),
    SpeechSituation.DIRECTION_CHANGE: OccasionTone(
        "acknowledge the change of direction", "curious about the new direction"
    ),
}


def direction_changed(text: str) -> bool:
    return _DIRECTION_CHANGE.match(text.strip()) is not None


class Struggle(BaseModel):
    model_config = ConfigDict(frozen=True)
    session: SessionIdentity
    activity: Activity
    operation: str | None
    started_at: float
    failures: int = 1

    def matches(
        self,
        session: SessionIdentity,
        activity: Activity,
        operation: str | None,
        now: float,
    ) -> bool:
        return (
            self.session == session
            and self.activity == activity
            and self.operation == operation
            and self.started_at <= now < self.started_at + STRUGGLE_SECONDS
        )


class Timers(BaseModel):
    model_config = ConfigDict(frozen=True)
    struggle: Struggle | None = None
    unanswered_bid_at: float | None = None

    def failed(
        self,
        session: SessionIdentity,
        activity: Activity,
        operation: str | None,
        now: float,
    ) -> Timers:
        previous = self.struggle
        if previous is not None and now < previous.started_at:
            return self
        if previous and previous.matches(session, activity, operation, now):
            struggle = previous.model_copy(
                update={"failures": min(2, previous.failures + 1)}
            )
        else:
            struggle = Struggle(
                session=session, activity=activity, operation=operation, started_at=now
            )
        return self.model_copy(update={"struggle": struggle})

    def recovered(
        self,
        session: SessionIdentity,
        activity: Activity,
        operation: str | None,
        now: float,
    ) -> tuple[Timers, Occasion | None]:
        struggle = self.struggle
        if struggle is None or not struggle.matches(session, activity, operation, now):
            return self, None
        timers = self.model_copy(update={"struggle": None})
        if struggle.failures < 2:
            return timers, None
        return timers, Occasion(
            fingerprint=occasion_fingerprint(
                SpeechSituation.RECOVERY, session, operation, struggle.started_at
            ),
            fact="An operation recovered after earlier failures.",
            situation=SpeechSituation.RECOVERY,
            expires_at=now + OCCASION_SECONDS,
        )

    def cancelled(self, session: SessionIdentity) -> Timers:
        if self.struggle is None or self.struggle.session != session:
            return self
        return self.model_copy(update={"struggle": None})

    def ignored(self, now: float, delay: float) -> Occasion | None:
        if self.unanswered_bid_at is None:
            return None
        due_at = self.unanswered_bid_at + delay
        expires_at = due_at + OCCASION_SECONDS
        if not due_at <= now < expires_at:
            return None
        return Occasion(
            fingerprint=occasion_fingerprint(
                SpeechSituation.UNANSWERED, self.unanswered_bid_at
            ),
            fact="The pet offered company and has not received a reply.",
            situation=SpeechSituation.UNANSWERED,
            expires_at=expires_at,
        )
