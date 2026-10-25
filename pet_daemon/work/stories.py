from __future__ import annotations

import hashlib

from pydantic import BaseModel, Field

from ..domain import (
    ArtifactKind,
    CollectionKind,
    EvidenceSource,
    LinkRelationship,
)
from ..identity import SessionIdentity
from ..repository_identity import (
    RepositoryIdentity,
    repository_columns,
)
from .association import TERMINAL_EVENTS, enclosing_checkout, references, report_context
from .evidence import CommentaryEvidence, commentary_evidence
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
from .observations import RepositoryTarget
from .serialization import payload_record
from .validation import canonical, normalize_reference


class ArtifactContext(BaseModel):
    kind: ArtifactKind
    ref: str
    title: str = ""
    repository: RepositoryIdentity | None = None
    revision: str = ""
    evidence: tuple[CommentaryEvidence, ...] = ()


class SessionContribution(BaseModel):
    scope: str
    requests: tuple[CommentaryEvidence, ...] = ()
    activity: tuple[CommentaryEvidence, ...] = ()
    context: ReportContext | None = None
    session: SessionIdentity = Field()


class WorkStory(BaseModel):
    id: SessionIdentity
    intent: str
    revision: str
    requests: tuple[CommentaryEvidence, ...] = ()
    activity: tuple[CommentaryEvidence, ...] = ()
    artifacts: tuple[ArtifactContext, ...] = ()
    sessions: tuple[SessionContribution, ...] = ()


def current_context(state: WorkFacts, agent: SessionRecord) -> ReportContext:
    if (
        agent.execution
        and isinstance(agent.execution.payload, ExecutionEvidence)
        and agent.execution.payload.event in TERMINAL_EVENTS
    ):
        return resolve_report_pull_request(state, agent.execution.context)
    checkout = enclosing_checkout(state.checkouts, agent.location)
    local = checkout.local if checkout else None
    branch = state.branches.get((local.repository, local.branch)) if local else None
    return report_context(agent.location, checkout, branch)


def resolve_report_pull_request(
    state: WorkFacts, context: ReportContext
) -> ReportContext:
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
    record.context = resolve_report_pull_request(state, report.context)
    return record


def _commentary_evidence(
    records: tuple[EvidenceRecord, ...],
) -> tuple[CommentaryEvidence, ...]:
    return tuple(
        value
        for record in records
        if (value := commentary_evidence(record)) is not None
    )


class _StoryArtifacts:
    def __init__(self, state: WorkFacts) -> None:
        self.state = state
        self.artifacts: dict[
            tuple[str, str, RepositoryIdentity | None], ArtifactReference
        ] = {}
        self.records: dict[
            tuple[str, str, RepositoryIdentity | None], tuple[EvidenceRecord, ...]
        ] = {}

    def add(
        self, value: ArtifactReference, records: tuple[EvidenceRecord, ...]
    ) -> None:
        key = value.kind, value.ref, value.repository
        previous = self.artifacts.get(key)
        if previous:
            value = value.model_copy()
            value.revision = value.revision or previous.revision
            value.checkout = value.checkout or previous.checkout
            value.title = value.title or previous.title
        self.artifacts[key] = value
        combined = [*self.records.get(key, ())]
        for record in records:
            if record not in combined:
                combined.append(record)
        self.records[key] = tuple(combined)

    def add_pr(
        self,
        ref: str,
        relationship: EvidenceRecord | None = None,
        *,
        stack: bool = True,
    ) -> None:
        observed = self.state.pull_requests.get(ref)
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
        self.add(value, records)
        if not pr:
            return
        for explicit in references(pr.title + "\n" + pr.body):
            if explicit.kind == ArtifactKind.TICKET:
                self.add(
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
                    self.add_pr(
                        link.ref,
                        EvidenceRecord(
                            payload=StackAssociation(link=link, from_ref=ref),
                            observed_at=observed.last_success_at if observed else None,
                        ),
                        stack=False,
                    )

    def add_context(self, context: ReportContext, current: bool) -> None:
        context = resolve_report_pull_request(self.state, context)
        if context.repository and context.branch:
            records = (
                EvidenceRecord(
                    payload=CheckoutAssociation(
                        checkout=context.checkout, branch=context.branch
                    )
                ),
            )
            checkout = self.state.checkouts.get(context.checkout) if current else None
            if checkout and checkout.local:
                records += (
                    EvidenceRecord(
                        payload=checkout.local, observed_at=checkout.last_success_at
                    ),
                )
            branch = self.state.branches.get((context.repository, context.branch))
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
            self.add(
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
                self.add_pr(
                    ref,
                    EvidenceRecord(
                        payload=PullRequestAssociation(
                            checkout=context.checkout, branch=context.branch
                        )
                    ),
                )
        elif context.pull_request:
            self.add_pr(context.pull_request)
        for explicit in context.references:
            relationship = EvidenceRecord(
                payload=ReferenceAssociation(
                    ref=explicit.ref, source=EvidenceSource.AGENT_REFERENCE
                )
            )
            if explicit.kind == ArtifactKind.PULL_REQUEST:
                self.add_pr(explicit.ref, relationship)
            else:
                self.add(explicit, (relationship,))

    def contexts(self) -> tuple[ArtifactContext, ...]:
        return tuple(
            ArtifactContext(
                kind=value.kind,
                ref=value.ref,
                title=" ".join(value.title.split())[:240] if value.title else "",
                repository=value.repository,
                revision=value.revision or "",
                evidence=_commentary_evidence(self.records[key]),
            )
            for key, value in sorted(
                self.artifacts.items(),
                key=lambda item: (
                    item[0][0],
                    item[0][1],
                    repository_columns(item[0][2]),
                ),
            )
        )


def context_artifacts(
    state: WorkFacts, context: ReportContext
) -> tuple[ArtifactContext, ...]:
    artifacts = _StoryArtifacts(state)
    artifacts.add_context(context, False)
    return artifacts.contexts()


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
    artifacts = _StoryArtifacts(state)
    for context, current in contexts:
        artifacts.add_context(context, current)

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
        requests=_commentary_evidence(requests),
        activity=_commentary_evidence(tuple(activity)),
        artifacts=artifacts.contexts(),
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
            context = resolve_report_pull_request(state, report.context)
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


def revision_key(value: object) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()
