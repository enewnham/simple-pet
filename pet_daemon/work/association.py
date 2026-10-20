from dataclasses import asdict
import hashlib
import uuid

from .models import Artifact, Evidence, WorkItem, WorkState
from .observations import RepositoryObservation, RepositoryTarget
from .opportunities import evidence_grounds
from .validation import canonical, descriptor, repository_identity, text


def bind_sessions(state: WorkState, work_id, session, aliases):
    root = state.aliases.get(session, session)
    for alias in aliases:
        old_root = state.aliases.get(alias, alias)
        for name, target in tuple(state.aliases.items()):
            if target == old_root:
                state.aliases[name] = root
        state.aliases[alias] = root
    for name in (session, *aliases):
        target = state.aliases.get(name, name)
        names = {name, target, *(alias for alias, value in state.aliases.items() if value == target)}
        for identifier in names:
            state.sessions[identifier] = work_id
            state.memberships.add((work_id, identifier))


def record_evidence(state: WorkState, work_id, artifact_id, source, kind, data, now):
    previous = state.latest(work_id, artifact_id).get((source, kind))
    if kind == "observation" and previous and now < previous.timestamp:
        return
    if previous and canonical(previous.data) == canonical(data):
        if kind == "observation":
            previous.timestamp = now
        return
    identifier = state.next_evidence_id
    state.next_evidence_id += 1
    state.evidence.append(Evidence(identifier, work_id, artifact_id, source, kind, data, now))


def record_intent(state: WorkState, session, intent, aliases, now):
    session, intent = text(session, "session"), text(intent, "intent")
    if not isinstance(aliases, (list, tuple)):
        raise ValueError("aliases must be a list")
    aliases = tuple(text(alias, "alias") for alias in aliases)
    candidates = {item.id for name in (session, *aliases) if (item := state.current(name))}
    if len(candidates) > 1:
        raise ValueError("session aliases refer to different work")
    work = state.items[next(iter(candidates))] if candidates else None
    if work is None or work.status == "concluded":
        work = WorkItem(uuid.uuid4().hex, intent, "active", now)
        state.items[work.id] = work
    elif work.intent != intent:
        work.intent, work.changed_at = intent, now
    bind_sessions(state, work.id, session, aliases)
    record_evidence(state, work.id, "", "user", "intent", {"session": session, "text": intent}, now)
    return work.id


def record_session_event(state: WorkState, session, event, now):
    if not isinstance(event, dict):
        raise ValueError("session event must be an object")
    aliases = event.get("aliases", ())
    if not isinstance(aliases, (list, tuple)):
        raise ValueError("aliases must be a list")
    aliases = tuple(text(alias, "alias") for alias in aliases)
    harness = event.get("harness")
    if harness:
        aliases = tuple(alias if alias.startswith(harness + ":") else f"{harness}:{alias}" for alias in aliases)
    candidates = {item.id for name in (session, *aliases) if (item := state.current(name))}
    if len(candidates) != 1:
        return
    work_id = next(iter(candidates))
    bind_sessions(state, work_id, session, aliases)
    observation = {key: event[key] for key in ("event", "outcome", "tool_name", "progress") if key in event}
    record_evidence(state, work_id, "", "harness", "execution", {"session": session, **observation}, now)
    if event.get("progress"):
        record_evidence(state, work_id, "", "harness", "progress", {"session": session, "text": event["progress"]}, now)


def artifact(state: WorkState, value, source, observed_at):
    value = descriptor(value)
    previous = next((record for record in state.artifacts.values() if all(record.data[key] == value[key]
                     for key in ("kind", "ref", "repository"))), None)
    if previous:
        latest = max((record.timestamp for record in state.evidence if record.artifact_id == previous.id
                      and record.source == source and record.kind == "observation"), default=None)
        if latest is not None and observed_at < latest:
            return previous.id
        data = {**previous.data, **value}
        previous.version += int(canonical(data) != canonical(previous.data))
        previous.data = data
        return previous.id
    identifier = uuid.uuid4().hex
    state.artifacts[identifier] = Artifact(identifier, value)
    return identifier


def link(state: WorkState, work_id, artifact_id, source, data, now):
    state.links.add((work_id, artifact_id))
    record_evidence(state, work_id, artifact_id, source, "relationship", data, now)


def observe_repository(state: WorkState, observation: RepositoryObservation, now: float):
    local = asdict(observation.local) if observation.local else None
    pr = asdict(observation.pr) if observation.pr else None
    repository = repository_identity(observation.repository or (local or {}).get("repository", ""))
    branch_name, checkout = observation.branch, observation.worktree
    work_ids = set()
    if checkout and observation.current:
        work_ids = {work.id for session in observation.sessions if (work := state.current(session))}
    branch = None
    if repository and branch_name:
        value = {"kind": "branch", "ref": branch_name, "repository": repository}
        if checkout and observation.current:
            value["checkout"] = checkout
        if local and local.get("head"):
            value["revision"] = local["head"]
        branch = artifact(state, value, "git", observation.local_checked_at)
        if checkout and not observation.current and state.artifacts[branch].data.get("checkout") == checkout:
            state.artifacts[branch].data.pop("checkout")
            state.artifacts[branch].version += 1
        for work_id in work_ids:
            link(state, work_id, branch, "harness", {"relationship": "session_checkout", "checkout": checkout,
                 "branch": branch_name, "association_only": True}, now)
        work_ids.update(state.linked_work(branch))
        if local and not observation.local_error:
            for work_id in state.linked_work(branch):
                record_evidence(state, work_id, branch, "git", "observation", local, observation.local_checked_at)
    pull_request = None
    matches_branch = bool(branch and pr and pr.get("head_branch") == branch_name
                          and repository_identity(pr.get("head_repository", "")) == repository)
    if pr and pr.get("url"):
        value = {"kind": "pull_request", "ref": pr["url"]}
        if pr.get("head"):
            value["revision"] = pr["head"]
        if pr.get("title"):
            value["title"] = pr["title"]
        pull_request = artifact(state, value, "github", observation.github_checked_at)
        for work_id in work_ids if branch is None or matches_branch else ():
            link(state, work_id, pull_request, "github", {"relationship": "checkout_pull_request", "checkout": checkout or "",
                 "branch": branch_name or "", "association_only": True}, observation.github_checked_at)
        work_ids.update(state.linked_work(pull_request))
        for related in pr.get("related_links", ()):
            if related.get("kind") != "pull_request" or related.get("relationship") != "stack_entry" or related.get("source") != "github_stack":
                continue
            sibling = {"kind": "pull_request", "ref": related["ref"]}
            if related.get("head"):
                sibling["revision"] = related["head"]
            if related.get("title"):
                sibling["title"] = related["title"]
            sibling_id = artifact(state, sibling, "github", observation.github_checked_at)
            for work_id in state.linked_work(pull_request):
                link(state, work_id, sibling_id, "github_stack", {**related, "from": pr["url"], "association_only": True}, observation.github_checked_at)
    if observation.github_checked_at and not observation.github_error:
        for artifact_id in (branch, pull_request):
            if artifact_id:
                if artifact_id == branch and pr and not matches_branch:
                    continue
                observed_repository = descriptor({"kind": "pull_request", "ref": pr["url"]})["repository"] if artifact_id == pull_request else repository
                for work_id in state.linked_work(artifact_id):
                    record_evidence(state, work_id, artifact_id, "github", "observation", {"pr": pr, "repository": observed_repository}, observation.github_checked_at)
    for artifact_id in (branch, pull_request):
        if artifact_id:
            for work_id in state.linked_work(artifact_id):
                github_error = observation.github_error
                if artifact_id == branch and pr and not matches_branch:
                    github_error = github_error or "Pull request head identity does not match the observed branch."
                record_evidence(state, work_id, artifact_id, "observer", "availability",
                                {"local_error": observation.local_error, "github_error": github_error}, now)


def work_context(state: WorkState, work_id):
    work = state.items.get(work_id)
    if not work:
        return {}
    requests = [record for record in state.evidence if record.work_id == work_id and record.source == "user" and record.kind == "intent"][-4:]
    activity = [record for (source, kind), record in sorted(state.latest(work_id, "").items())
                if source == "harness" and kind in {"execution", "progress"}]
    artifacts = []
    linked = [state.artifacts[artifact_id] for linked_work, artifact_id in state.links if linked_work == work_id]
    for record in sorted(linked, key=lambda record: (record.data["kind"], record.data["ref"])):
        evidence = tuple(item.record() for _, item in sorted(state.latest(work_id, record.id).items()))
        artifacts.append({**record.data, "evidence": evidence})
    context = {"intent": work.intent, "requests": tuple(record.record() for record in requests),
               "activity": tuple(record.record() for record in activity), "artifacts": tuple(artifacts)}
    grounds = {"intent": context["intent"], "requests": evidence_grounds(context["requests"]),
               "activity": evidence_grounds(context["activity"]),
               "artifacts": [{**{key: value for key, value in item.items() if key != "evidence"},
                              "evidence": evidence_grounds(item["evidence"])} for item in artifacts]}
    context["revision"] = hashlib.sha256(canonical(grounds).encode()).hexdigest()
    return context


def collection_targets(state: WorkState) -> tuple[RepositoryTarget, ...]:
    targets = []
    for record in sorted(state.artifacts.values(), key=lambda item: (item.data["kind"], item.data["ref"])):
        if not state.linked_work(record.id):
            continue
        value = record.data
        if value["kind"] not in {"branch", "pull_request"}:
            continue
        relationships = [item for item in state.evidence if item.artifact_id == record.id and item.kind == "relationship"]
        direct = any(item.source != "github_stack" for item in relationships)
        targets.append(RepositoryTarget(value["kind"], value["ref"], value.get("repository", ""),
                                        value.get("checkout", ""), direct))
    return tuple(targets)
