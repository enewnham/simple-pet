from dataclasses import dataclass


@dataclass(frozen=True)
class WorkSubject:
    id: str
    kind: str
    priority: int
    changed_at: float
    revision: object
    evidence: tuple


def build_work(sessions, repositories, now=0):
    subjects = []
    for session in sessions.get("sessions", ()):
        failure = session.get("failure")
        quiet = session.get("quiet_seconds", 0)
        waiting = session.get("waiting_for_input")
        failed = bool(failure and failure.get("current_turn") and not failure.get("resolved") and failure.get("seconds_ago", 0) < 300)
        active = session.get("turn_active") and not session.get("activity_uncertain")
        priority = 0 if waiting or failed else 1 if active else 3 if session.get("outcome") and quiet < 300 else 4
        subjects.append(WorkSubject(session["id"], "session", priority,
                                    session.get("changed_at", now - session.get("quiet_seconds", 0)), session.get("generation", 0),
                                    tuple((key, session.get(key)) for key in ("repo", "goal", "outcome"))))
    for repository in repositories:
        if repository.pending:
            failed = repository.pr and any(check[1] in {"FAILURE", "ERROR", "CANCELLED", "TIMED_OUT", "ACTION_REQUIRED"}
                                           for check in repository.pr.checks)
            age = now - repository.changed_at
            priority = 0 if failed and age < 300 else 2 if age < 86400 else 4
            subjects.append(WorkSubject(repository.subject, "repository", priority, repository.changed_at,
                                        repository.revision, (("repository", repository),)))
        elif repository.pr and repository.pr.state == "MERGED":
            subjects.append(WorkSubject(repository.subject, "repository", 3 if now - repository.changed_at < 300 else 4, repository.changed_at,
                                        repository.revision, (("repository", repository),)))
    return tuple(sorted(subjects, key=lambda item: (item.priority, -item.changed_at, item.id)))
