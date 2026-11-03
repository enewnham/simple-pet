from ..domain import ExecutionOutcome
from .templates import render


def event_text(event: str) -> str:
    return render("observation.jinja", kind="event", value=event)


def outcome_text(outcome: ExecutionOutcome) -> str:
    return render("observation.jinja", kind="outcome", value=outcome)
