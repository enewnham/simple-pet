from ..domain import Activity, ExecutionOutcome
from .templates import render


def event_text(event: str) -> str:
    return render("observation.jinja", kind="event", value=event)


def outcome_text(outcome: ExecutionOutcome) -> str:
    return render("observation.jinja", kind="outcome", value=outcome)


def activity_text(activity: Activity) -> str:
    return render("observation.jinja", kind="activity", value=activity)


def state_text(state: str) -> str:
    return render("observation.jinja", kind="state", value=state)
