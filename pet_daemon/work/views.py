import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

from ..runtime import SessionSnapshot, SessionsSnapshot
from .association import work_context
from .models import (
    AvailabilityEvidence,
    CheckoutAssociation,
    EvidenceRecord,
    ExecutionEvidence,
    GitHubEvidence,
    IntentEvidence,
    Opportunity,
    ProgressEvidence,
    PullRequestAssociation,
    StackAssociation,
    WorkAccount,
    WorkState,
)
from .observations import GitObservation
from .opportunities import eligible, project
from .validation import canonical


@dataclass(frozen=True)
class PresentedPullRequest:
    url: str
    title: str
    state: str
    is_draft: bool | None
    review: str
    review_requested: bool | None


@dataclass(frozen=True)
class GitEvidenceView:
    repository: str | None
    branch: str
    output_present: bool
    conflicts: bool
    behind: bool


@dataclass(frozen=True)
class PullRequestEvidenceView:
    output: PresentedPullRequest
    checks: tuple[tuple[str, str], ...]
    checks_current: bool


@dataclass(frozen=True)
class MissingPullRequestView:
    pull_request_found: bool = False


@dataclass(frozen=True)
class ExecutionView:
    execution: str
    reported_progress: str | None
    reported_execution_outcome: str | None


@dataclass(frozen=True)
class AvailabilityView:
    uncertainty: tuple[str, ...]


@dataclass(frozen=True)
class IntentView:
    text: str


@dataclass(frozen=True)
class ProgressView:
    reported_progress: str


@dataclass(frozen=True)
class AssociationView:
    association: str
    authorship: str = "Association does not establish authorship."


type EvidenceDetails = (
    GitEvidenceView
    | PullRequestEvidenceView
    | MissingPullRequestView
    | ExecutionView
    | AvailabilityView
    | IntentView
    | ProgressView
    | AssociationView
)


@dataclass(frozen=True)
class EvidenceView:
    source: str
    kind: str
    details: EvidenceDetails
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


def project_evidence(record: EvidenceRecord) -> EvidenceView | None:
    source, kind, payload = record.source, record.kind, record.payload
    details: EvidenceDetails
    match payload:
        case GitObservation() if source == "git" and kind == "observation":
            details = GitEvidenceView(
                payload.repository
                if not payload.repository.startswith("local:")
                else None,
                payload.branch,
                bool(
                    payload.staged
                    or payload.unstaged
                    or payload.untracked
                    or payload.ahead
                ),
                bool(payload.conflicts),
                bool(payload.behind),
            )
        case GitHubEvidence() if source == "github" and kind == "observation":
            pr = payload.pr
            if pr:
                current = bool(pr.head and pr.head == pr.check_revision)
                checks = tuple(
                    (check[0], check[1])
                    for check in pr.checks
                    if len(check) >= 2
                    and current
                    and (len(check) < 4 or not check[3] or check[3] == pr.head)
                )
                details = PullRequestEvidenceView(
                    PresentedPullRequest(
                        pr.url,
                        pr.title,
                        pr.state,
                        pr.is_draft,
                        pr.review,
                        pr.review_requested,
                    ),
                    checks,
                    current,
                )
            else:
                details = MissingPullRequestView()
        case ExecutionEvidence() if source == "harness" and kind == "execution":
            ended = payload.event in {
                "turn_done",
                "turn_error",
                "subagent_done",
                "interrupt",
                "session_end",
            }
            execution = (
                "input requested"
                if payload.event == "waiting"
                else "turn ended"
                if ended
                else "in progress"
            )
            details = ExecutionView(
                execution,
                " ".join(payload.progress.split())[:800] if payload.progress else None,
                payload.outcome if ended and payload.outcome else None,
            )
        case AvailabilityEvidence() if source == "observer" and kind == "availability":
            unavailable = tuple(
                name
                for name, error in (
                    ("local", payload.local_error),
                    ("github", payload.github_error),
                )
                if error
            )
            if not unavailable:
                return None
            details = AvailabilityView(
                tuple(f"{name} observation unavailable" for name in unavailable)
            )
        case IntentEvidence() if source == "user" and kind == "intent":
            details = IntentView(" ".join(payload.text.split())[:1000])
        case ProgressEvidence() if source == "harness" and kind == "progress":
            details = ProgressView(" ".join(payload.text.split())[:800])
        case CheckoutAssociation() if kind == "relationship" and source == "harness":
            details = AssociationView("The agent session used this checkout.")
        case PullRequestAssociation() if kind == "relationship" and source == "github":
            details = AssociationView("The pull request matches the observed branch.")
        case StackAssociation() if kind == "relationship" and source == "github_stack":
            details = AssociationView("The pull requests share an observed stack.")
        case _:
            return None
    return EvidenceView(source, kind, details, record.observed_at)


def project_work(
    identifier: str, account: WorkAccount, opportunities: Sequence[Opportunity] = ()
) -> WorkView:
    def evidence(records: tuple[EvidenceRecord, ...]) -> tuple[EvidenceView, ...]:
        return tuple(
            value
            for record in records
            if (value := project_evidence(record)) is not None
        )

    artifacts = tuple(
        ArtifactView(
            item.artifact.kind,
            item.artifact.ref,
            " ".join(item.artifact.title.split())[:240] if item.artifact.title else "",
            item.artifact.repository,
            item.artifact.revision or "",
            evidence(item.evidence),
        )
        for item in account.artifacts
    )
    obstacle = any(action.kind == "resolve_checks" for action in opportunities) or any(
        isinstance(record.details, GitEvidenceView) and record.details.conflicts
        for item in artifacts
        for record in item.evidence
    )
    return WorkView(
        identifier,
        account.intent,
        account.revision,
        evidence(account.requests),
        evidence(account.activity),
        artifacts,
        tuple(opportunities),
        obstacle,
    )


def runtime_revision(
    session: SessionSnapshot,
) -> tuple[
    tuple[int, bool, str | None, bool, str, bool, str, str], tuple[tuple[str, str], ...]
]:
    state = (
        session.generation,
        session.turn_active,
        session.outcome,
        session.waiting_for_input,
        session.progress,
        session.activity_uncertain,
        session.phase,
        session.last_tool,
    )
    tools = tuple(sorted({(tool.tool, tool.activity) for tool in session.active_tools}))
    return state, tools


def revision_key(value: object) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def build_work(sessions: SessionsSnapshot, state: WorkState) -> tuple[WorkSubject, ...]:
    runtime = {session.id: session for session in sessions.sessions}
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
        assert account is not None
        actions = tuple(action for action in opportunities if action.work_id == work.id)
        view = project_work(work.id, account, actions)
        changed_at = max(
            (member.changed_at for member in members), default=work.changed_at
        )
        revision = revision_key(
            (
                account.revision,
                tuple((member.id, runtime_revision(member)) for member in members),
            )
        )
        subjects.append(WorkSubject(work.id, "work", changed_at, revision, view, names))
    for identifier, session in runtime.items():
        if identifier not in bound:
            revision = revision_key(runtime_revision(session))
            subjects.append(
                WorkSubject(
                    identifier,
                    "session",
                    session.changed_at,
                    revision,
                    WorkView(identifier, session.goal, revision),
                    (identifier,),
                )
            )
    return tuple(sorted(subjects, key=lambda item: (-item.changed_at, item.id)))
