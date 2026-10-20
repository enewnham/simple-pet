from __future__ import annotations

import hashlib
import uuid
from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..runtime import HookEvent

from .models import (
    Artifact,
    ArtifactAccount,
    ArtifactDescriptor,
    AvailabilityEvidence,
    CheckoutAssociation,
    Evidence,
    EvidencePayload,
    ExecutionEvidence,
    GitHubEvidence,
    IntentEvidence,
    ProgressEvidence,
    PullRequestAssociation,
    StackAssociation,
    WorkAccount,
    WorkItem,
    WorkState,
)
from .observations import RepositoryObservation, RepositoryTarget
from .opportunities import evidence_grounds
from .serialization import artifact_record
from .validation import canonical, descriptor, repository_identity, text


def bind_sessions(
    state: WorkState, work_id: str, session: str, aliases: Sequence[str]
) -> None:
    root = state.aliases.get(session, session)
    for alias in aliases:
        old_root = state.aliases.get(alias, alias)
        for name, target in tuple(state.aliases.items()):
            if target == old_root:
                state.aliases[name] = root
        state.aliases[alias] = root
    for name in (session, *aliases):
        target = state.aliases.get(name, name)
        names = {
            name,
            target,
            *(alias for alias, value in state.aliases.items() if value == target),
        }
        for identifier in names:
            state.sessions[identifier] = work_id
            state.memberships.add((work_id, identifier))


def record_evidence(
    state: WorkState,
    work_id: str,
    artifact_id: str,
    source: str,
    kind: str,
    payload: EvidencePayload,
    now: float,
) -> None:
    previous = state.latest(work_id, artifact_id).get((source, kind))
    if kind == "observation" and previous and now < previous.timestamp:
        return
    if previous and previous.payload == payload:
        if kind == "observation":
            previous.timestamp = now
        return
    identifier = state.next_evidence_id
    state.next_evidence_id += 1
    state.evidence.append(
        Evidence(identifier, work_id, artifact_id, source, kind, payload, now)
    )


def record_intent(
    state: WorkState, session: str, intent: str, aliases: tuple[str, ...], now: float
) -> str:
    session, intent = text(session, "session"), text(intent, "intent")
    aliases = tuple(text(alias, "alias") for alias in aliases)
    candidates = {
        item.id for name in (session, *aliases) if (item := state.current(name))
    }
    if len(candidates) > 1:
        raise ValueError("session aliases refer to different work")
    work = state.items[next(iter(candidates))] if candidates else None
    if work is None or work.status == "concluded":
        work = WorkItem(uuid.uuid4().hex, intent, "active", now)
        state.items[work.id] = work
    elif work.intent != intent:
        work.intent, work.changed_at = intent, now
    bind_sessions(state, work.id, session, aliases)
    record_evidence(
        state, work.id, "", "user", "intent", IntentEvidence(session, intent), now
    )
    return work.id


def record_session_event(
    state: WorkState, session: str, event: HookEvent, now: float
) -> None:
    aliases = tuple(text(alias, "alias") for alias in event.aliases)
    harness = event.harness
    aliases = tuple(
        alias if alias.startswith(harness + ":") else f"{harness}:{alias}"
        for alias in aliases
    )
    candidates = {
        item.id for name in (session, *aliases) if (item := state.current(name))
    }
    if len(candidates) != 1:
        return
    work_id = next(iter(candidates))
    bind_sessions(state, work_id, session, aliases)
    record_evidence(
        state,
        work_id,
        "",
        "harness",
        "execution",
        ExecutionEvidence(
            session, event.event, event.outcome, event.tool_name, event.progress
        ),
        now,
    )
    if event.progress:
        record_evidence(
            state,
            work_id,
            "",
            "harness",
            "progress",
            ProgressEvidence(session, event.progress),
            now,
        )


def artifact(
    state: WorkState, value: ArtifactDescriptor, source: str, observed_at: float
) -> str:
    value = descriptor(value)
    previous = next(
        (
            record
            for record in state.artifacts.values()
            if (record.kind, record.ref, record.repository)
            == (value.kind, value.ref, value.repository)
        ),
        None,
    )
    if previous:
        latest = max(
            (
                record.timestamp
                for record in state.evidence
                if record.artifact_id == previous.id
                and record.source == source
                and record.kind == "observation"
            ),
            default=None,
        )
        if latest is not None and observed_at < latest:
            return previous.id
        updated = replace(
            previous,
            revision=value.revision
            if value.revision is not None
            else previous.revision,
            checkout=value.checkout
            if value.checkout is not None
            else previous.checkout,
            title=value.title if value.title is not None else previous.title,
        )
        state.artifacts[previous.id] = replace(
            updated, version=previous.version + int(updated != previous)
        )
        return previous.id
    identifier = uuid.uuid4().hex
    state.artifacts[identifier] = Artifact(
        kind=value.kind,
        ref=value.ref,
        repository=value.repository,
        revision=value.revision,
        checkout=value.checkout,
        title=value.title,
        id=identifier,
    )
    return identifier


def link(
    state: WorkState,
    work_id: str,
    artifact_id: str,
    source: str,
    payload: EvidencePayload,
    now: float,
) -> None:
    state.links.add((work_id, artifact_id))
    record_evidence(state, work_id, artifact_id, source, "relationship", payload, now)


def observe_repository(
    state: WorkState, observation: RepositoryObservation, now: float
) -> None:
    local, pr = observation.local, observation.pr
    repository = repository_identity(
        observation.repository or (local.repository if local else "")
    )
    branch_name, checkout = observation.branch, observation.worktree
    work_ids = set()
    if checkout and observation.current:
        work_ids = {
            work.id
            for session in observation.sessions
            if (work := state.current(session))
        }
    branch = None
    if repository and branch_name:
        value = ArtifactDescriptor(
            "branch",
            branch_name,
            repository,
            revision=local.head if local and local.head else None,
            checkout=checkout if checkout and observation.current else None,
        )
        branch = artifact(state, value, "git", observation.local_checked_at)
        current = state.artifacts[branch]
        if checkout and not observation.current and current.checkout == checkout:
            state.artifacts[branch] = replace(
                current, checkout=None, version=current.version + 1
            )
        for work_id in work_ids:
            link(
                state,
                work_id,
                branch,
                "harness",
                CheckoutAssociation(checkout, branch_name),
                now,
            )
        work_ids.update(state.linked_work(branch))
        if local and not observation.local_error:
            for work_id in state.linked_work(branch):
                record_evidence(
                    state,
                    work_id,
                    branch,
                    "git",
                    "observation",
                    local,
                    observation.local_checked_at,
                )
    pull_request = None
    matches_branch = bool(
        branch
        and pr
        and pr.head_branch == branch_name
        and repository_identity(pr.head_repository) == repository
    )
    if pr and pr.url:
        value = ArtifactDescriptor(
            "pull_request", pr.url, revision=pr.head or None, title=pr.title or None
        )
        pull_request = artifact(state, value, "github", observation.github_checked_at)
        for work_id in work_ids if branch is None or matches_branch else ():
            link(
                state,
                work_id,
                pull_request,
                "github",
                PullRequestAssociation(checkout, branch_name),
                observation.github_checked_at,
            )
        work_ids.update(state.linked_work(pull_request))
        for related in pr.related_links:
            if (
                related.kind != "pull_request"
                or related.relationship != "stack_entry"
                or related.source != "github_stack"
            ):
                continue
            sibling = ArtifactDescriptor(
                "pull_request",
                related.ref,
                revision=related.head or None,
                title=related.title or None,
            )
            sibling_id = artifact(
                state, sibling, "github", observation.github_checked_at
            )
            for work_id in state.linked_work(pull_request):
                link(
                    state,
                    work_id,
                    sibling_id,
                    "github_stack",
                    StackAssociation(related, pr.url),
                    observation.github_checked_at,
                )
    if observation.github_checked_at and not observation.github_error:
        for artifact_id in (branch, pull_request):
            if artifact_id:
                if artifact_id == branch and pr and not matches_branch:
                    continue
                observed_repository = repository
                if artifact_id == pull_request:
                    assert pr is not None
                    observed_repository = descriptor(
                        ArtifactDescriptor("pull_request", pr.url)
                    ).repository
                for work_id in state.linked_work(artifact_id):
                    record_evidence(
                        state,
                        work_id,
                        artifact_id,
                        "github",
                        "observation",
                        GitHubEvidence(pr, observed_repository),
                        observation.github_checked_at,
                    )
    for artifact_id in (branch, pull_request):
        if artifact_id:
            for work_id in state.linked_work(artifact_id):
                github_error = observation.github_error
                if artifact_id == branch and pr and not matches_branch:
                    github_error = (
                        github_error
                        or "Pull request head identity does not match the observed branch."
                    )
                record_evidence(
                    state,
                    work_id,
                    artifact_id,
                    "observer",
                    "availability",
                    AvailabilityEvidence(observation.local_error, github_error),
                    now,
                )


def work_context(state: WorkState, work_id: str) -> WorkAccount | None:
    work = state.items.get(work_id)
    if work is None:
        return None
    requests = tuple(
        record.record()
        for record in state.evidence
        if record.work_id == work_id
        and record.source == "user"
        and record.kind == "intent"
    )[-4:]
    activity = tuple(
        record.record()
        for (source, kind), record in sorted(state.latest(work_id, "").items())
        if source == "harness" and kind in {"execution", "progress"}
    )
    linked = [
        state.artifacts[artifact_id]
        for linked_work, artifact_id in state.links
        if linked_work == work_id
    ]
    artifacts = tuple(
        ArtifactAccount(
            record.descriptor(),
            tuple(
                item.record()
                for _, item in sorted(state.latest(work_id, record.id).items())
            ),
        )
        for record in sorted(linked, key=lambda record: (record.kind, record.ref))
    )
    grounds = {
        "intent": work.intent,
        "requests": evidence_grounds(requests),
        "activity": evidence_grounds(activity),
        "artifacts": [
            {
                **artifact_record(item.artifact),
                "evidence": evidence_grounds(item.evidence),
            }
            for item in artifacts
        ],
    }
    revision = hashlib.sha256(canonical(grounds).encode()).hexdigest()
    return WorkAccount(work.intent, revision, requests, activity, artifacts)


def collection_targets(state: WorkState) -> tuple[RepositoryTarget, ...]:
    targets = []
    for record in sorted(
        state.artifacts.values(), key=lambda item: (item.kind, item.ref)
    ):
        if not state.linked_work(record.id) or record.kind not in {
            "branch",
            "pull_request",
        }:
            continue
        relationships = [
            item
            for item in state.evidence
            if item.artifact_id == record.id and item.kind == "relationship"
        ]
        direct = any(item.source != "github_stack" for item in relationships)
        targets.append(
            RepositoryTarget(
                record.kind,
                record.ref,
                record.repository,
                record.checkout or "",
                direct,
            )
        )
    return tuple(targets)
