from __future__ import annotations

import hashlib
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from ..domain import (
    ArtifactKind,
    CollectionKind,
    EvidenceKind,
    EvidenceSource,
    ExecutionOutcome,
    LinkRelationship,
    PullRequestState,
    WorkSubjectKind,
)
from ..identity import SessionIdentity
from ..repository_identity import (
    RemoteRepository,
    RepositoryIdentity,
    repository_columns,
)
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
from .validation import canonical, normalize_reference


class PresentedPullRequest(BaseModel):
    url: str
    title: str
    state: PullRequestState
    is_draft: bool | None


class GitEvidenceView(BaseModel):
    type: Literal["GitEvidenceView"] = "GitEvidenceView"
    repository: str | None
    branch: str
    output_present: bool
    conflicts: bool
    behind: bool


class PullRequestEvidenceView(BaseModel):
    type: Literal["PullRequestEvidenceView"] = "PullRequestEvidenceView"
    output: PresentedPullRequest


class MissingPullRequestView(BaseModel):
    type: Literal["MissingPullRequestView"] = "MissingPullRequestView"
    pull_request_found: bool = False


class ExecutionView(BaseModel):
    type: Literal["ExecutionView"] = "ExecutionView"
    execution: str
    reported_progress: str | None
    reported_execution_outcome: ExecutionOutcome | None


class AvailabilityView(BaseModel):
    type: Literal["AvailabilityView"] = "AvailabilityView"
    uncertainty: tuple[str, ...]


class IntentView(BaseModel):
    type: Literal["IntentView"] = "IntentView"
    text: str


class ProgressView(BaseModel):
    type: Literal["ProgressView"] = "ProgressView"
    reported_progress: str


class AssociationView(BaseModel):
    type: Literal["AssociationView"] = "AssociationView"
    association: str
    authorship: str = "Association does not establish authorship."


type EvidenceDetails = Annotated[
    (
        GitEvidenceView
        | PullRequestEvidenceView
        | MissingPullRequestView
        | ExecutionView
        | AvailabilityView
        | IntentView
        | ProgressView
        | AssociationView
    ),
    Field(discriminator="type"),
]


class EvidenceView(BaseModel):
    source: EvidenceSource
    kind: EvidenceKind
    details: EvidenceDetails
    observed_at: float | None = None
    context: ReportContext | None = None


class ArtifactContext(BaseModel):
    kind: ArtifactKind
    ref: str
    title: str = ""
    repository: RepositoryIdentity | None = None
    revision: str = ""
    evidence: tuple[EvidenceView, ...] = ()


class SessionContribution(BaseModel):
    scope: str
    requests: tuple[EvidenceView, ...] = ()
    activity: tuple[EvidenceView, ...] = ()
    context: ReportContext | None = None
    session: SessionIdentity = Field()


class WorkStory(BaseModel):
    id: SessionIdentity
    intent: str
    revision: str
    requests: tuple[EvidenceView, ...] = ()
    activity: tuple[EvidenceView, ...] = ()
    artifacts: tuple[ArtifactContext, ...] = ()
    sessions: tuple[SessionContribution, ...] = ()


class WorkUpdate(BaseModel):
    session: SessionIdentity
    revision: str
    observed_at: float
    ended_at: float | None = None
    aliases: tuple[SessionIdentity, ...] = ()


class WorkSubject(BaseModel):
    id: SessionIdentity
    kind: WorkSubjectKind
    changed_at: float
    revision: str
    story: WorkStory
    sessions: tuple[SessionIdentity, ...] = ()
    updates: tuple[WorkUpdate, ...] = ()


def project_evidence(record: EvidenceRecord) -> EvidenceView | None:
    source, kind, payload = record.source, record.kind, record.payload
    details: EvidenceDetails
    match payload:
        case GitObservation():
            details = GitEvidenceView(
                repository=payload.repository.value
                if isinstance(payload.repository, RemoteRepository)
                else None,
                branch=payload.branch,
                output_present=bool(
                    payload.staged
                    or payload.unstaged
                    or payload.untracked
                    or payload.ahead
                ),
                conflicts=bool(payload.conflicts),
                behind=bool(payload.behind),
            )
        case GitHubEvidence():
            pr = payload.pr
            if pr:
                details = PullRequestEvidenceView(
                    output=PresentedPullRequest(
                        url=pr.url, title=pr.title, state=pr.state, is_draft=pr.is_draft
                    )
                )
            else:
                details = MissingPullRequestView()
        case ExecutionEvidence() if not payload.event.startswith("tool_"):
            ended = payload.event in {
                "turn_done",
                "turn_error",
                "subagent_done",
                "interrupt",
                "session_end",
            }
            execution = (
                "waiting"
                if payload.event == "waiting"
                else "turn ended"
                if ended
                else "in progress"
            )
            details = ExecutionView(
                execution=execution,
                reported_progress=" ".join(payload.progress.split())[:800]
                if payload.progress
                else None,
                reported_execution_outcome=payload.outcome
                if ended and payload.outcome
                else None,
            )
        case AvailabilityEvidence():
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
                uncertainty=tuple(
                    f"{name} observation unavailable" for name in unavailable
                )
            )
        case IntentEvidence():
            details = IntentView(text=" ".join(payload.text.split())[:1000])
        case ProgressEvidence():
            details = ProgressView(
                reported_progress=" ".join(payload.text.split())[:800]
            )
        case CheckoutAssociation():
            details = AssociationView(
                association="The agent session used this checkout."
            )
        case PullRequestAssociation():
            details = AssociationView(
                association="The pull request matches the observed branch."
            )
        case ReferenceAssociation():
            details = AssociationView(
                association="The reported text explicitly references this URL."
            )
        case StackAssociation():
            details = AssociationView(
                association="The pull requests share an observed stack."
            )
        case _:
            return None
    return EvidenceView(
        source=source,
        kind=kind,
        details=details,
        observed_at=record.observed_at,
        context=record.context,
    )


def current_context(state: WorkFacts, agent: SessionRecord) -> ReportContext:
    if (
        agent.execution
        and isinstance(agent.execution.payload, ExecutionEvidence)
        and agent.execution.payload.event in TERMINAL_EVENTS
    ):
        return resolve_context(state, agent.execution.context)
    checkout = enclosing_checkout(state.checkouts, agent.location)
    local = checkout.local if checkout else None
    branch = state.branches.get((local.repository, local.branch)) if local else None
    return report_context(agent.location, checkout, branch)


def resolve_context(state: WorkFacts, context: ReportContext) -> ReportContext:
    if context.pull_request or not context.repository or not context.branch:
        return context
    branch = state.branches.get((context.repository, context.branch))
    return (
        context.model_copy(
            update={"pull_request": branch.resolutions[context.branch_resolution]}
        )
        if branch and context.branch_resolution < len(branch.resolutions)
        else context
    )


def report_record(state: WorkFacts, report: SessionReport) -> EvidenceRecord:
    record = report.record()
    record.context = resolve_context(state, report.context)
    return record


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
    closed = agent.closure and (
        not agent.execution or agent.closure.observed_at >= agent.execution.observed_at
    )
    if not ended and not closed:
        contexts.append((current_context(state, agent), True))
    contexts.extend((report.context, False) for report in agent.reports)
    if agent.execution:
        contexts.append((agent.execution.context, False))
    artifacts: dict[tuple[str, str, RepositoryIdentity | None], ArtifactReference] = {}
    artifact_records: dict[
        tuple[str, str, RepositoryIdentity | None], tuple[EvidenceRecord, ...]
    ] = {}

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
            value = value.model_copy()
            value.revision = value.revision or previous.revision
            value.checkout = value.checkout or previous.checkout
            value.title = value.title or previous.title
        artifacts[key] = value
        combined = [*artifact_records.get(key, ())]
        for record in records:
            if record not in combined:
                combined.append(record)
        artifact_records[key] = tuple(combined)

    def add_pr(
        ref: str, relationship: EvidenceRecord | None = None, *, stack: bool = True
    ) -> None:
        observed = state.pull_requests.get(ref)
        pr = observed.pr if observed else None
        value = normalize_reference(
            ArtifactReference(
                kind=ArtifactKind.PULL_REQUEST,
                ref=ref,
                revision=pr.head or None if pr else None,
                title=pr.title or None if pr else None,
            )
        )
        records = (relationship,) if relationship else ()
        if observed:
            if observed.last_success_at is not None:
                records += (
                    EvidenceRecord(
                        payload=GitHubEvidence(pr=pr, repository=value.repository),
                        observed_at=observed.last_success_at,
                    ),
                )
            if observed.error or pr and pr.related_links_error:
                records += (
                    EvidenceRecord(
                        payload=AvailabilityEvidence(
                            local_error=None,
                            github_error=observed.error
                            or (pr.related_links_error if pr else None),
                        ),
                        observed_at=observed.checked_at,
                    ),
                )
        add(value, records)
        if not pr:
            return
        for explicit in references(pr.title + "\n" + pr.body):
            if explicit.kind == ArtifactKind.TICKET:
                add(
                    explicit,
                    (
                        EvidenceRecord(
                            payload=ReferenceAssociation(
                                ref=explicit.ref, source=EvidenceSource.PR_REFERENCE
                            )
                        ),
                    ),
                )
        if stack:
            for link in pr.related_links:
                if (
                    link.kind == ArtifactKind.PULL_REQUEST
                    and link.relationship == LinkRelationship.STACK_ENTRY
                ):
                    add_pr(
                        link.ref,
                        EvidenceRecord(
                            payload=StackAssociation(link=link, from_ref=ref),
                            observed_at=observed.last_success_at if observed else None,
                        ),
                        stack=False,
                    )

    for context, current in contexts:
        context = resolve_context(state, context)
        if context.repository and context.branch:
            records = (
                EvidenceRecord(
                    payload=CheckoutAssociation(
                        checkout=context.checkout, branch=context.branch
                    )
                ),
            )
            checkout = state.checkouts.get(context.checkout) if current else None
            if checkout and checkout.local:
                records += (
                    EvidenceRecord(
                        payload=checkout.local, observed_at=checkout.last_success_at
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
                        payload=GitHubEvidence(pr=None, repository=context.repository),
                        observed_at=branch.last_success_at,
                    ),
                )
            if checkout and (checkout.error or not checkout.available):
                records += (
                    EvidenceRecord(
                        payload=AvailabilityEvidence(
                            local_error=checkout.error or "Checkout unavailable",
                            github_error=None,
                        ),
                        observed_at=checkout.checked_at,
                    ),
                )
            if branch and branch.error:
                records += (
                    EvidenceRecord(
                        payload=AvailabilityEvidence(
                            local_error=None, github_error=branch.error
                        ),
                        observed_at=branch.checked_at,
                    ),
                )
            add(
                ArtifactReference(
                    kind=ArtifactKind.BRANCH,
                    ref=context.branch,
                    repository=context.repository,
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
                        payload=PullRequestAssociation(
                            checkout=context.checkout, branch=context.branch
                        )
                    ),
                )
        elif context.pull_request:
            add_pr(context.pull_request)
        for explicit in context.references:
            relationship = EvidenceRecord(
                payload=ReferenceAssociation(
                    ref=explicit.ref, source=EvidenceSource.AGENT_REFERENCE
                )
            )
            if explicit.kind == ArtifactKind.PULL_REQUEST:
                add_pr(explicit.ref, relationship)
            else:
                add(explicit, (relationship,))

    def report_facts(report: SessionReport) -> tuple[object, ...]:
        return (
            payload_record(report.payload),
            report.context.model_dump(mode="json"),
        )

    grounds = {
        "intent": agent.goal,
        "requests": [report_facts(report) for report in request_reports],
        "activity": [report_facts(report) for report in activity_reports],
    }
    revision = hashlib.sha256(canonical(grounds).encode()).hexdigest()
    return WorkStory(
        id=agent.id,
        intent=agent.goal,
        revision=revision,
        requests=evidence(requests),
        activity=evidence(tuple(activity)),
        artifacts=tuple(
            ArtifactContext(
                kind=value.kind,
                ref=value.ref,
                title=" ".join(value.title.split())[:240] if value.title else "",
                repository=value.repository,
                revision=value.revision or "",
                evidence=evidence(artifact_records[key]),
            )
            for key, value in sorted(
                artifacts.items(),
                key=lambda item: (
                    item[0][0],
                    item[0][1],
                    repository_columns(item[0][2]),
                ),
            )
        ),
    )


def collection_targets(state: WorkFacts) -> tuple[RepositoryTarget, ...]:
    targets = {
        ("checkout", agent.location, None): RepositoryTarget(
            kind=CollectionKind.CHECKOUT, ref=agent.location
        )
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
                    RepositoryTarget(
                        kind=CollectionKind.BRANCH,
                        ref=context.branch,
                        repository=context.repository,
                    )
                )
    for agent in state.sessions.values():
        for artifact in session_work(state, agent).artifacts:
            if artifact.kind == ArtifactKind.PULL_REQUEST:
                targets[artifact.kind, artifact.ref, artifact.repository] = (
                    RepositoryTarget(
                        kind=CollectionKind.PULL_REQUEST,
                        ref=artifact.ref,
                        repository=artifact.repository,
                    )
                )
    return tuple(
        targets[key]
        for key in sorted(
            targets, key=lambda key: (key[0], key[1], repository_columns(key[2]))
        )
    )


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


def work_update(agent: SessionRecord, story: WorkStory) -> WorkUpdate:
    from .narrative import semantic_revision

    execution = agent.execution
    ended_at = (
        execution.observed_at
        if execution
        and isinstance(execution.payload, ExecutionEvidence)
        and execution.payload.event in TERMINAL_EVENTS
        else None
    )
    return WorkUpdate(
        session=agent.id,
        revision=revision_key(
            (
                semantic_revision(story),
                execution.observed_at if execution else None,
                tuple(record.observed_at for record in story.requests),
            )
        ),
        observed_at=max(
            (record.observed_at or 0 for record in (*story.requests, *story.activity)),
            default=0,
        ),
        ended_at=ended_at,
        aliases=agent.aliases,
    )


def connected_work(
    state: WorkFacts, activity: SwarmActivity
) -> tuple[WorkSubject, ...]:
    runtime = {session.id: session for session in activity.sessions}
    groups: dict[SessionIdentity, set[SessionIdentity]] = {}
    for identifier in (*state.sessions, *runtime):
        root = state.aliases.get(identifier, identifier)
        seen: set[SessionIdentity] = set()
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
            id=root, intent=root_session.goal if root_session else "", revision=""
        )
        children = tuple(
            SessionContribution(
                scope=story.intent,
                requests=story.requests,
                activity=story.activity,
                context=current_context(state, state.sessions[identifier]),
                session=identifier,
            )
            for identifier, story in stories.items()
            if identifier != root
        )
        artifacts: dict[
            tuple[str, str, RepositoryIdentity | None], ArtifactContext
        ] = {}
        for story in stories.values():
            for artifact in story.artifacts:
                key = artifact.kind, artifact.ref, artifact.repository
                previous = artifacts.get(key)
                if previous:
                    combined = [*previous.evidence]
                    for record in artifact.evidence:
                        if record not in combined:
                            combined.append(record)
                    artifact.evidence = tuple(combined)
                artifacts[key] = artifact
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
        story = root_story.model_copy()
        story.revision = revision
        story.sessions = children
        story.artifacts = tuple(
            artifact
            for _, artifact in sorted(
                artifacts.items(),
                key=lambda item: (
                    item[0][0],
                    item[0][1],
                    repository_columns(item[0][2]),
                ),
            )
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
                id=root,
                kind=WorkSubjectKind.WORK if stories else WorkSubjectKind.SESSION,
                changed_at=changed_at,
                revision=revision,
                story=story,
                sessions=tuple(sorted(names)),
                updates=tuple(
                    work_update(state.sessions[identifier], member)
                    for identifier, member in stories.items()
                ),
            )
        )
    return tuple(sorted(subjects, key=lambda item: (-item.changed_at, item.id)))


from ..runtime import PetContext, SessionReaction

_runtime_types = {
    "SessionReaction": SessionReaction,
    "WorkStory": WorkStory,
    "WorkSubject": WorkSubject,
}
for _runtime_model in (SessionContext, SwarmActivity, PetContext):
    _runtime_model.model_rebuild(_types_namespace=_runtime_types)
