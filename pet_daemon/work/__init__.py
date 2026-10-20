from dataclasses import asdict, dataclass, field

from .store import Opportunity, WorkStore


@dataclass(frozen=True)
class WorkSubject:
    id: str
    kind: str
    priority: int
    changed_at: float
    revision: object
    evidence: tuple
    payload: dict = field(default_factory=dict)


def build_work(sessions, store, now=0):
    subjects = []
    for session in sessions.get("sessions", ()):
        failure = session.get("failure")
        quiet = session.get("quiet_seconds", 0)
        waiting = session.get("waiting_for_input")
        failed = bool(failure and failure.get("current_turn") and not failure.get("resolved")
                      and failure.get("seconds_ago", 0) < 300)
        active = session.get("turn_active") and not session.get("activity_uncertain")
        priority = 0 if waiting or failed else 1 if active else 3 if session.get("outcome") and quiet < 300 else 4
        account = store.session_context(session["id"])
        runtime = tuple(session.get(key) for key in ("generation", "turn_active", "outcome", "waiting_for_input", "progress", "activity_uncertain", "phase", "last_tool"))
        tools = tuple(sorted({(tool.get("tool"), tool.get("activity")) for tool in session.get("active_tools", ())}))
        failure_state = tuple((key, failure.get(key)) for key in ("current_turn", "resolved", "activity", "tool")) if failure else ()
        subjects.append(WorkSubject(session["id"], "session", priority,
                                    session.get("changed_at", now - quiet), (runtime, tools, failure_state, account.get("revision")),
                                    tuple((key, session.get(key)) for key in ("repo", "goal", "outcome")), account))
    for opportunity in store.opportunities():
        priority = 0 if opportunity.kind in {"resolve_checks", "review_output"} else 2
        payload = asdict(opportunity)
        if opportunity.sessions:
            payload["connected_work"] = store.session_context(opportunity.sessions[0])
        subjects.append(WorkSubject(opportunity.id, "opportunity", priority, opportunity.changed_at,
                                    opportunity.version, opportunity.evidence, payload))
    return tuple(sorted(subjects, key=lambda item: (item.priority, -item.changed_at, item.id)))


__all__ = ["Opportunity", "WorkStore", "WorkSubject", "build_work"]
