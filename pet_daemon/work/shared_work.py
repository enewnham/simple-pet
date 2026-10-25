from __future__ import annotations

from pydantic import BaseModel

from ..domain import (
    ArtifactKind,
    WorkSubjectKind,
)
from ..identity import SessionIdentity
from ..runtime import AgentSessionStatus, SwarmActivity
from .association import TERMINAL_EVENTS
from .evidence import ExecutionStatus, ReportedProgress
from .models import (
    ExecutionEvidence,
    IntentEvidence,
    ProgressEvidence,
    ReportContext,
    SessionRecord,
    SessionReport,
    WorkFacts,
)
from .stories import (
    SessionContribution,
    WorkStory,
    context_artifacts,
    current_context,
    resolve_report_pull_request,
    revision_key,
    session_work,
)


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
    parent: SessionIdentity | None = None


def runtime_revision(
    session: AgentSessionStatus,
) -> tuple[int, bool, str | None, bool, bool]:
    return (
        session.generation,
        session.turn_active,
        session.outcome,
        session.waiting_for_input,
        session.activity_uncertain,
    )


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
                ended_at,
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


MAX_RELATED_SESSIONS = 3
RELATED_REPORT_SECONDS = 3600


def _connections(
    state: WorkFacts, context: ReportContext
) -> set[tuple[ArtifactKind, str]]:
    return {
        (artifact.kind, artifact.ref)
        for artifact in context_artifacts(state, context)
        if artifact.kind in (ArtifactKind.PULL_REQUEST, ArtifactKind.TICKET)
    }


def related_reports(
    state: WorkFacts,
    selected: SessionIdentity,
    reports: tuple[SessionReport, ...],
    observed_at: float,
) -> tuple[SessionReport, ...]:
    own = tuple(
        report
        for report in reports
        if state.aliases.get(report.payload.session, report.payload.session) == selected
        and report.observed_at <= observed_at
        and not (
            isinstance(report.payload, ExecutionEvidence)
            and (
                report.payload.event.startswith("tool_")
                or report.payload.event == "session_end"
            )
        )
    )
    if not own:
        return ()
    latest = max(
        own,
        key=lambda report: (
            report.observed_at,
            isinstance(report.payload, ProgressEvidence),
        ),
    )
    assignments: dict[SessionIdentity, SessionReport] = {}
    connections = _connections(
        state, resolve_report_pull_request(state, latest.context)
    )
    intent = next(
        (
            report
            for report in reversed(own)
            if isinstance(report.payload, IntentEvidence)
        ),
        None,
    )
    if intent:
        connections |= _connections(
            state, resolve_report_pull_request(state, intent.context)
        )
    agent = state.sessions[selected]
    parent = state.aliases.get(agent.parent, agent.parent) if agent.parent else None
    result: list[SessionReport] = []
    for report in reports:
        if report.observed_at > observed_at:
            continue
        identity = state.aliases.get(report.payload.session, report.payload.session)
        if identity == selected:
            continue
        if isinstance(report.payload, IntentEvidence):
            assignments[identity] = report
        peer = state.sessions[identity]
        peer_parent = (
            state.aliases.get(peer.parent, peer.parent) if peer.parent else None
        )
        assignment = assignments.get(identity)
        contexts = (
            (report.context, assignment.context) if assignment else (report.context,)
        )
        links = {
            link
            for context in contexts
            for link in _connections(state, resolve_report_pull_request(state, context))
        }
        if parent != identity and peer_parent != selected and not connections & links:
            continue
        if assignment and assignment not in result:
            result.append(assignment)
        if report not in result:
            result.append(report)
    return tuple(sorted(result, key=lambda report: report.observed_at))


def _related_story(
    state: WorkFacts,
    agent: SessionRecord,
    story: WorkStory,
    stories: dict[SessionIdentity, WorkStory],
) -> WorkStory:
    moment = max(
        (record.observed_at or 0 for record in (*story.requests, *story.activity)),
        default=agent.changed_at,
    )
    latest = max(
        tuple(reversed((*story.requests, *story.activity))),
        key=lambda record: (
            record.observed_at or 0,
            isinstance(record.details, ReportedProgress),
        ),
        default=None,
    )
    context = (
        latest.context if latest and latest.context else current_context(state, agent)
    )
    connections = _connections(state, context)
    contributions = []
    for identifier, other in stories.items():
        if identifier == agent.id:
            continue
        peer = state.sessions[identifier]
        parent = state.aliases.get(agent.parent, agent.parent) if agent.parent else None
        peer_parent = (
            state.aliases.get(peer.parent, peer.parent) if peer.parent else None
        )
        linked_parent = parent == identifier or peer_parent == agent.id
        latest_request = max(
            (record.observed_at or 0 for record in other.requests), default=0
        )
        activity = tuple(
            record
            for record in other.activity
            if record.observed_at is not None
            and moment - RELATED_REPORT_SECONDS <= record.observed_at <= moment
            and record.observed_at >= latest_request
            and (
                isinstance(record.details, ReportedProgress)
                or isinstance(record.details, ExecutionStatus)
                and record.details.execution == "turn ended"
            )
            and (
                linked_parent
                or record.context is not None
                and bool(connections & _connections(state, record.context))
            )
        )
        if not activity:
            continue
        requests = tuple(
            record
            for record in other.requests
            if record.observed_at is not None and record.observed_at <= moment
        )
        contributions.append(
            SessionContribution(
                scope=other.intent,
                requests=requests,
                activity=activity,
                context=activity[-1].context,
                session=identifier,
            )
        )
    contributions.sort(
        key=lambda item: (
            -max(record.observed_at or 0 for record in item.activity),
            item.session,
        )
    )
    selected = tuple(contributions[:MAX_RELATED_SESSIONS])
    return story.model_copy(update={"sessions": selected})


def shared_work(state: WorkFacts, activity: SwarmActivity) -> tuple[WorkSubject, ...]:
    runtime = {session.id: session for session in activity.sessions}
    stories = {
        identifier: session_work(state, agent)
        for identifier, agent in state.sessions.items()
    }
    subjects = []
    for identifier in sorted(set(state.sessions) | set(runtime)):
        if state.aliases.get(identifier, identifier) != identifier:
            continue
        agent = state.sessions.get(identifier)
        session = runtime.get(identifier)
        own_story = stories.get(identifier)
        if agent and own_story:
            story = _related_story(state, agent, own_story, stories)
            updates = (work_update(agent, own_story),)
        else:
            story = WorkStory(
                id=identifier, intent=session.goal if session else "", revision=""
            )
            updates = ()
        revision = (
            own_story.revision
            if own_story
            else revision_key(
                (runtime_revision(session), session.progress) if session else ()
            )
        )
        names = {identifier} | {
            alias for alias, target in state.aliases.items() if target == identifier
        }
        subjects.append(
            WorkSubject(
                id=identifier,
                kind=WorkSubjectKind.WORK if own_story else WorkSubjectKind.SESSION,
                changed_at=max(
                    agent.changed_at if agent else 0,
                    session.changed_at if session else 0,
                ),
                revision=revision,
                story=story,
                sessions=tuple(sorted(names)),
                updates=updates,
                parent=(
                    state.aliases.get(agent.parent, agent.parent)
                    if agent and agent.parent
                    else session.parent
                    if session
                    else None
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
for _runtime_model in (AgentSessionStatus, SwarmActivity, PetContext):
    _runtime_model.model_rebuild(_types_namespace=_runtime_types)
