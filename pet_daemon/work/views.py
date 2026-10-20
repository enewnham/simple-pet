from dataclasses import dataclass
import hashlib

from .models import Opportunity, WorkState
from .association import work_context
from .opportunities import eligible, project
from .validation import canonical


@dataclass(frozen=True)
class EvidenceView:
    source: str
    kind: str
    details: dict
    observed_at: float | None = None


@dataclass(frozen=True)
class ArtifactView:
    kind: str
    ref: str
    title: str = ""
    repository: str = ""
    revision: str = ""
    evidence: tuple[EvidenceView, ...] = ()


@dataclass(frozen=True)
class WorkView:
    id: str
    intent: str
    revision: str
    requests: tuple[EvidenceView, ...] = ()
    activity: tuple[EvidenceView, ...] = ()
    artifacts: tuple[ArtifactView, ...] = ()
    opportunities: tuple[Opportunity, ...] = ()
    current_obstacle: bool = False


@dataclass(frozen=True)
class WorkSubject:
    id: str
    kind: str
    changed_at: float
    revision: str
    view: WorkView
    sessions: tuple[str, ...] = ()


def project_evidence(record):
    source, kind, data = record["source"], record["kind"], record["data"]
    details = {}
    if source == "git" and kind == "observation":
        repository = data.get("repository", "")
        details.update(
            repository=repository if not repository.startswith("local:") else None,
            branch=data.get("branch"),
            output_present=any(
                data.get(key) for key in ("staged", "unstaged", "untracked", "ahead")
            ),
            conflicts=bool(data.get("conflicts")),
            behind=bool(data.get("behind")),
        )
    elif source == "github" and kind == "observation":
        pr = data.get("pr")
        if pr:
            current = bool(
                pr.get("head") and pr.get("head") == pr.get("check_revision")
            )
            checks = tuple(
                (check[0], check[1])
                for check in pr.get("checks", ())
                if len(check) >= 2
                and current
                and (len(check) < 4 or not check[3] or check[3] == pr.get("head"))
            )
            details.update(
                output={
                    key: pr.get(key)
                    for key in (
                        "url",
                        "title",
                        "state",
                        "is_draft",
                        "review",
                        "review_requested",
                    )
                },
                checks=checks,
                checks_current=current,
            )
        else:
            details["pull_request_found"] = False
    elif source == "harness" and kind == "execution":
        event = data.get("event")
        details["execution"] = (
            "input requested"
            if event == "waiting"
            else "turn ended"
            if event
            in {"turn_done", "turn_error", "subagent_done", "interrupt", "session_end"}
            else "in progress"
        )
        if data.get("progress"):
            details["reported_progress"] = data["progress"]
        if details["execution"] == "turn ended" and data.get("outcome"):
            details["reported_execution_outcome"] = data["outcome"]
    elif source == "observer" and kind == "availability":
        unavailable = tuple(
            name for name in ("local", "github") if data.get(name + "_error")
        )
        if not unavailable:
            return None
        details["uncertainty"] = tuple(
            f"{name} observation unavailable" for name in unavailable
        )
    elif source == "user" and kind == "intent":
        details["text"] = data.get("text", "")
    elif source == "harness" and kind == "progress":
        details["reported_progress"] = data.get("text", "")
    elif kind == "relationship" and source in {"harness", "github", "github_stack"}:
        relationship = data.get("relationship")
        associations = {
            "session_checkout": "The agent session used this checkout.",
            "checkout_pull_request": "The pull request matches the observed branch.",
            "stack_entry": "The pull requests share an observed stack.",
        }
        if relationship not in associations:
            return None
        details = {
            "association": associations[relationship],
            "authorship": "Association does not establish authorship.",
        }
    else:
        return None
    for key in ("text", "reported_progress", "title"):
        if isinstance(details.get(key), str):
            limit = 1000 if source == "user" else 800
            details[key] = " ".join(details[key].split())[:limit]
    return EvidenceView(source, kind, details, record.get("observed_at"))


def project_work(identifier, account, opportunities=()):
    def evidence(records):
        return tuple(
            value
            for record in records
            if (value := project_evidence(record)) is not None
        )

    artifacts = tuple(
        ArtifactView(
            item["kind"],
            item["ref"],
            " ".join(item.get("title", "").split())[:240],
            item.get("repository", ""),
            item.get("revision", ""),
            evidence(item.get("evidence", ())),
        )
        for item in account.get("artifacts", ())
    )
    obstacle = any(action.kind == "resolve_checks" for action in opportunities) or any(
        record.details.get("conflicts")
        for item in artifacts
        for record in item.evidence
    )
    return WorkView(
        identifier,
        account.get("intent", ""),
        account.get("revision", ""),
        evidence(account.get("requests", ())),
        evidence(account.get("activity", ())),
        artifacts,
        tuple(opportunities),
        obstacle,
    )


def runtime_revision(session):
    state = tuple(
        session.get(key)
        for key in (
            "generation",
            "turn_active",
            "outcome",
            "waiting_for_input",
            "progress",
            "activity_uncertain",
            "phase",
            "last_tool",
        )
    )
    tools = tuple(
        sorted(
            {
                (tool.get("tool"), tool.get("activity"))
                for tool in session.get("active_tools", ())
            }
        )
    )
    return state, tools


def revision_key(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def build_work(sessions, state: WorkState, now=0) -> tuple[WorkSubject, ...]:
    runtime = {session["id"]: session for session in sessions.get("sessions", ())}
    subjects, bound = [], set()
    opportunities = tuple(
        project(record) for record in state.opportunities.values() if eligible(record)
    )
    for work in state.items.values():
        names = tuple(
            sorted(
                session for work_id, session in state.memberships if work_id == work.id
            )
        )
        members = tuple(runtime[name] for name in names if name in runtime)
        bound.update(names)
        account = work_context(state, work.id)
        actions = tuple(action for action in opportunities if action.work_id == work.id)
        view = project_work(work.id, account, actions)
        changed_at = max(
            (
                member.get("changed_at", now - member.get("quiet_seconds", 0))
                for member in members
            ),
            default=work.changed_at,
        )
        revision = revision_key(
            (
                account["revision"],
                tuple((member["id"], runtime_revision(member)) for member in members),
            )
        )
        subjects.append(WorkSubject(work.id, "work", changed_at, revision, view, names))
    for identifier, session in runtime.items():
        if identifier not in bound:
            revision = revision_key(runtime_revision(session))
            view = WorkView(identifier, session.get("goal", ""), revision)
            subjects.append(
                WorkSubject(
                    identifier,
                    "session",
                    session.get("changed_at", now - session.get("quiet_seconds", 0)),
                    revision,
                    view,
                    (identifier,),
                )
            )
    return tuple(sorted(subjects, key=lambda item: (-item.changed_at, item.id)))
