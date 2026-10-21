from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, replace

from ..runtime import SessionContext, SwarmActivity
from .association import TERMINAL_EVENTS, enclosing_checkout, references, report_context
from .models import (
    ArtifactReference,
    AvailabilityEvidence,
    CheckoutAssociation,
    EvidenceRecord,
    ExecutionEvidence,
    GitHubEvidence,
    IntentEvidence,
    ProgressEvidence,
    PullRequestAssociation,
    ReferenceAssociation,
    ReportContext,
    SessionRecord,
    SessionReport,
    StackAssociation,
    WorkFacts,
)
from .observations import GitObservation, RepositoryTarget
from .serialization import payload_record
from .validation import canonical, normalize_reference, repository_identity


@dataclass(frozen=True)
class PresentedPullRequest:
    url: str
    title: str
    state: str
    is_draft: bool | None


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
    context: ReportContext | None = None


@dataclass(frozen=True)
class ArtifactContext:
    kind: str
    ref: str
    title: str = ""
    repository: str = ""
    revision: str = ""
    evidence: tuple[EvidenceView, ...] = ()


@dataclass(frozen=True)
class SessionContribution:
    scope: str
    requests: tuple[EvidenceView, ...] = ()
    activity: tuple[EvidenceView, ...] = ()
    context: ReportContext | None = None


@dataclass(frozen=True)
class WorkStory:
    id: str
    intent: str
    revision: str
    requests: tuple[EvidenceView, ...] = ()
    activity: tuple[EvidenceView, ...] = ()
    artifacts: tuple[ArtifactContext, ...] = ()
    sessions: tuple[SessionContribution, ...] = ()


@dataclass(frozen=True)
class WorkSubject:
    id: str
    kind: str
    changed_at: float
    revision: str
    story: WorkStory
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
                details = PullRequestEvidenceView(
                    PresentedPullRequest(pr.url, pr.title, pr.state, pr.is_draft)
                )
            else:
                details = MissingPullRequestView()
        case ExecutionEvidence() if (
            source == "harness"
            and kind == "execution"
            and not payload.event.startswith("tool_")
        ):
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
        case ReferenceAssociation() if kind == "relationship":
            details = AssociationView(
                "The reported text explicitly references this URL."
            )
        case StackAssociation() if kind == "relationship" and source == "github_stack":
            details = AssociationView("The pull requests share an observed stack.")
        case _:
            return None
    return EvidenceView(source, kind, details, record.observed_at, record.context)


def current_context(state: WorkFacts, agent: SessionRecord) -> ReportContext:
    if (
        agent.execution
        and isinstance(agent.execution.payload, ExecutionEvidence)
        and agent.execution.payload.event in TERMINAL_EVENTS
    ):
        return resolve_context(state, agent.execution.context)
    checkout = enclosing_checkout(state.checkouts, agent.location)
    local = checkout.local if checkout else None
    branch = (
        state.branches.get((repository_identity(local.repository), local.branch))
        if local
        else None
    )
    return report_context(agent.location, checkout, branch)


def resolve_context(state: WorkFacts, context: ReportContext) -> ReportContext:
    if context.pull_request or not context.repository or not context.branch:
        return context
    branch = state.branches.get((context.repository, context.branch))
    return (
        replace(context, pull_request=branch.resolutions[context.branch_resolution])
        if branch and context.branch_resolution < len(branch.resolutions)
        else context
    )


def report_record(state: WorkFacts, report: SessionReport) -> EvidenceRecord:
    return replace(report.record(), context=resolve_context(state, report.context))


def session_work(state: WorkFacts, agent: SessionRecord) -> WorkStory:
    request_reports = tuple(
        report for report in agent.reports if isinstance(report.payload, IntentEvidence)
    )[-4:]
    requests = tuple(report_record(state, report) for report in request_reports)
    activity = []
    activity_reports = []
    contexts = []
    latest_progress = next(
        (
            report
            for report in reversed(agent.reports)
            if isinstance(report.payload, ProgressEvidence)
        ),
        None,
    )
    if latest_progress:
        activity.append(report_record(state, latest_progress))
        activity_reports.append(latest_progress)
    if agent.execution:
        activity.append(report_record(state, agent.execution))
        activity_reports.append(agent.execution)
    ended = (
        agent.execution
        and isinstance(agent.execution.payload, ExecutionEvidence)
        and agent.execution.payload.event in TERMINAL_EVENTS
    )
    if not ended:
        contexts.append((current_context(state, agent), True))
    contexts.extend((report.context, False) for report in agent.reports)
    if agent.execution:
        contexts.append((agent.execution.context, False))
    artifacts: dict[tuple[str, str, str], ArtifactReference] = {}
    artifact_records: dict[tuple[str, str, str], tuple[EvidenceRecord, ...]] = {}

    def evidence(records: tuple[EvidenceRecord, ...]) -> tuple[EvidenceView, ...]:
        return tuple(
            value
            for record in records
            if (value := project_evidence(record)) is not None
        )

    def add(value: ArtifactReference, records: tuple[EvidenceRecord, ...]) -> None:
        key = value.kind, value.ref, value.repository
        previous = artifacts.get(key)
        if previous:
            value = replace(
                value,
                revision=value.revision or previous.revision,
                checkout=value.checkout or previous.checkout,
                title=value.title or previous.title,
            )
        artifacts[key] = value
        artifact_records[key] = tuple(
            dict.fromkeys((*artifact_records.get(key, ()), *records))
        )

    def add_pr(
        ref: str, relationship: EvidenceRecord | None = None, *, stack: bool = True
    ) -> None:
        observed = state.pull_requests.get(ref)
        pr = observed.pr if observed else None
        value = normalize_reference(
            ArtifactReference(
                "pull_request",
                ref,
                revision=pr.head or None if pr else None,
                title=pr.title or None if pr else None,
            )
        )
        records = (relationship,) if relationship else ()
        if observed:
            if observed.last_success_at is not None:
                records += (
                    EvidenceRecord(
                        "github",
                        "observation",
                        GitHubEvidence(pr, value.repository),
                        observed.last_success_at,
                    ),
                )
            if observed.error or pr and pr.related_links_error:
                records += (
                    EvidenceRecord(
                        "observer",
                        "availability",
                        AvailabilityEvidence(
                            None,
                            observed.error or (pr.related_links_error if pr else None),
                        ),
                        observed.checked_at,
                    ),
                )
        add(value, records)
        if not pr:
            return
        for explicit in references(pr.title + "\n" + pr.body):
            if explicit.kind == "ticket":
                add(
                    explicit,
                    (
                        EvidenceRecord(
                            "pr_reference",
                            "relationship",
                            ReferenceAssociation(explicit.ref),
                        ),
                    ),
                )
        if stack:
            for link in pr.related_links:
                if link.kind == "pull_request" and link.relationship == "stack_entry":
                    add_pr(
                        link.ref,
                        EvidenceRecord(
                            "github_stack",
                            "relationship",
                            StackAssociation(link, ref),
                            observed.last_success_at if observed else None,
                        ),
                        stack=False,
                    )

    for context, current in contexts:
        context = resolve_context(state, context)
        if context.repository and context.branch:
            records = (
                EvidenceRecord(
                    "harness",
                    "relationship",
                    CheckoutAssociation(context.checkout, context.branch),
                ),
            )
            checkout = state.checkouts.get(context.checkout) if current else None
            if checkout and checkout.local:
                records += (
                    EvidenceRecord(
                        "git", "observation", checkout.local, checkout.last_success_at
                    ),
                )
            branch = state.branches.get((context.repository, context.branch))
            if (
                branch
                and branch.pr is None
                and not branch.error
                and (current or not context.pull_request)
            ):
                records += (
                    EvidenceRecord(
                        "github",
                        "observation",
                        GitHubEvidence(None, context.repository),
                        branch.last_success_at,
                    ),
                )
            if checkout and (checkout.error or not checkout.available):
                records += (
                    EvidenceRecord(
                        "observer",
                        "availability",
                        AvailabilityEvidence(
                            checkout.error or "Checkout unavailable", None
                        ),
                        checkout.checked_at,
                    ),
                )
            if branch and branch.error:
                records += (
                    EvidenceRecord(
                        "observer",
                        "availability",
                        AvailabilityEvidence(None, branch.error),
                        branch.checked_at,
                    ),
                )
            add(
                ArtifactReference(
                    "branch",
                    context.branch,
                    context.repository,
                    revision=checkout.local.head or None
                    if checkout and checkout.local
                    else None,
                    checkout=context.checkout or None,
                ),
                records,
            )
            ref = context.pull_request
            if ref:
                add_pr(
                    ref,
                    EvidenceRecord(
                        "github",
                        "relationship",
                        PullRequestAssociation(context.checkout, context.branch),
                    ),
                )
        elif context.pull_request:
            add_pr(context.pull_request)
        for explicit in context.references:
            relationship = EvidenceRecord(
                "agent_reference", "relationship", ReferenceAssociation(explicit.ref)
            )
            if explicit.kind == "pull_request":
                add_pr(explicit.ref, relationship)
            else:
                add(explicit, (relationship,))

    def report_facts(report: SessionReport) -> tuple[object, ...]:
        return (
            payload_record(report.payload),
            asdict(report.context),
        )

    grounds = {
        "intent": agent.goal,
        "requests": [report_facts(report) for report in request_reports],
        "activity": [report_facts(report) for report in activity_reports],
    }
    revision = hashlib.sha256(canonical(grounds).encode()).hexdigest()
    return WorkStory(
        agent.id,
        agent.goal,
        revision,
        evidence(requests),
        evidence(tuple(activity)),
        tuple(
            ArtifactContext(
                value.kind,
                value.ref,
                " ".join(value.title.split())[:240] if value.title else "",
                value.repository,
                value.revision or "",
                evidence(artifact_records[key]),
            )
            for key, value in sorted(artifacts.items())
        ),
    )


def collection_targets(state: WorkFacts) -> tuple[RepositoryTarget, ...]:
    targets = {
        ("checkout", agent.location, ""): RepositoryTarget("checkout", agent.location)
        for agent in state.sessions.values()
        if agent.location
    }
    for agent in state.sessions.values():
        reports = (
            (*agent.reports, agent.execution) if agent.execution else agent.reports
        )
        for report in reports:
            context = resolve_context(state, report.context)
            if context.repository and context.branch and not context.pull_request:
                targets["branch", context.branch, context.repository] = (
                    RepositoryTarget("branch", context.branch, context.repository)
                )
    for agent in state.sessions.values():
        for artifact in session_work(state, agent).artifacts:
            if artifact.kind == "pull_request":
                targets[artifact.kind, artifact.ref, artifact.repository] = (
                    RepositoryTarget(artifact.kind, artifact.ref, artifact.repository)
                )
    return tuple(targets[key] for key in sorted(targets))


def runtime_revision(
    session: SessionContext,
) -> tuple[int, bool, str | None, bool, bool]:
    return (
        session.generation,
        session.turn_active,
        session.outcome,
        session.waiting_for_input,
        session.activity_uncertain,
    )


def revision_key(value: object) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def connected_work(
    state: WorkFacts, activity: SwarmActivity
) -> tuple[WorkSubject, ...]:
    runtime = {session.id: session for session in activity.sessions}
    groups: dict[str, set[str]] = {}
    for identifier in (*state.sessions, *runtime):
        root = state.aliases.get(identifier, identifier)
        seen: set[str] = set()
        while root not in seen:
            seen.add(root)
            agent = state.sessions.get(root)
            session = runtime.get(root)
            parent = agent.parent if agent else session.parent if session else None
            if not parent:
                break
            root = state.aliases.get(parent, parent)
        else:
            root = min(seen)
        groups.setdefault(root, set()).add(identifier)

    subjects = []
    for root, identifiers in sorted(groups.items()):
        stories = {
            identifier: session_work(state, agent)
            for identifier in sorted(identifiers)
            if (agent := state.sessions.get(identifier))
        }
        root_session = runtime.get(root)
        root_story = stories.get(root) or WorkStory(
            root, root_session.goal if root_session else "", ""
        )
        children = tuple(
            SessionContribution(
                story.intent,
                story.requests,
                story.activity,
                current_context(state, state.sessions[identifier]),
            )
            for identifier, story in stories.items()
            if identifier != root
        )
        artifacts: dict[tuple[str, str, str], ArtifactContext] = {}
        for story in stories.values():
            for artifact in story.artifacts:
                key = artifact.kind, artifact.ref, artifact.repository
                previous = artifacts.get(key)
                artifacts[key] = (
                    replace(
                        artifact,
                        evidence=tuple(
                            dict.fromkeys((*previous.evidence, *artifact.evidence))
                        ),
                    )
                    if previous
                    else artifact
                )
        if len(stories) == 1 and root in stories:
            revision = root_story.revision
        elif stories:
            revision = revision_key(
                tuple((name, story.revision) for name, story in stories.items())
            )
        else:
            revision = revision_key(
                tuple(
                    (name, runtime_revision(runtime[name]), runtime[name].progress)
                    for name in sorted(identifiers)
                    if name in runtime
                )
            )
        story = replace(
            root_story,
            revision=revision,
            sessions=children,
            artifacts=tuple(artifact for _, artifact in sorted(artifacts.items())),
        )
        names = identifiers | {
            alias for alias, target in state.aliases.items() if target in identifiers
        }
        changed_at = max(
            (
                *(runtime[name].changed_at for name in identifiers if name in runtime),
                *(
                    state.sessions[name].changed_at
                    for name in identifiers
                    if name in state.sessions
                ),
                0,
            )
        )
        subjects.append(
            WorkSubject(
                root,
                "work" if stories else "session",
                changed_at,
                revision,
                story,
                tuple(sorted(names)),
            )
        )
    return tuple(sorted(subjects, key=lambda item: (-item.changed_at, item.id)))
