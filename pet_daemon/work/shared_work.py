from __future__ import annotations

from pydantic import BaseModel

from ..domain import (
    WorkSubjectKind,
)
from ..identity import SessionIdentity
from ..repository_identity import (
    RepositoryIdentity,
    repository_columns,
)
from ..runtime import AgentSessionStatus, SwarmActivity
from .association import TERMINAL_EVENTS
from .models import (
    ExecutionEvidence,
    SessionRecord,
    WorkFacts,
)
from .stories import (
    ArtifactContext,
    SessionContribution,
    WorkStory,
    current_context,
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


def shared_work(state: WorkFacts, activity: SwarmActivity) -> tuple[WorkSubject, ...]:
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
for _runtime_model in (AgentSessionStatus, SwarmActivity, PetContext):
    _runtime_model.model_rebuild(_types_namespace=_runtime_types)
