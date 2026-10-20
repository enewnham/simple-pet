import hashlib

from .models import Opportunity, OpportunityState, WorkState
from .validation import FAILURES, canonical


def evidence_grounds(records):
    result = []
    for record in records:
        data = {key: value for key, value in record["data"].items() if key not in {"update", "session"}}
        if record["source"] == "harness" and record["kind"] == "execution":
            event = data.get("event")
            ended = event in {"turn_done", "turn_error", "subagent_done", "interrupt", "session_end"}
            execution = {"execution": "turn ended" if ended else "input requested" if event == "waiting" else "in progress"}
            if data.get("progress"):
                execution["reported_progress"] = data["progress"]
            if ended and data.get("outcome"):
                execution["reported_execution_outcome"] = data["outcome"]
            data = execution
        if record["source"] == "git" and record["kind"] == "observation":
            data = {key: data.get(key) for key in ("repository", "branch", "head", "upstream")}
            local = record["data"]
            data.update(dirty=any(local.get(key) for key in ("staged", "unstaged", "untracked")),
                        conflicts=bool(local.get("conflicts")), ahead=bool(local.get("ahead")), behind=bool(local.get("behind")))
        if "provenance" in data:
            data["provenance"] = {key: value for key, value in data["provenance"].items() if key != "session"}
        if record["source"] == "observer" and record["kind"] == "availability":
            data = {key: bool(value) for key, value in data.items()}
        if record["source"] == "github" and data.get("pr"):
            data["pr"] = {key: value for key, value in data["pr"].items() if key != "related_links_error"}
        result.append({"source": record["source"], "kind": record["kind"], "data": data})
    return sorted(result, key=lambda item: (item["source"], item["kind"]))



def derive_opportunities(state: WorkState, now: float, context_for_work):
    candidates = set()
    for work in state.items.values():
        if work.status != "active":
            continue
        progress = state.latest(work.id, "").get(("harness", "progress"))
        sessions = tuple(sorted(session for work_id, session in state.memberships if work_id == work.id))
        account_revision = context_for_work(work.id).get("revision")
        artifacts = (state.artifacts[artifact_id] for work_id, artifact_id in sorted(state.links) if work_id == work.id)
        for artifact_record in artifacts:
            artifact = artifact_record.data
            latest = state.latest(work.id, artifact_record.id)
            github_record = latest.get(("github", "observation"))
            github = github_record.data if github_record else None
            pr = (github or {}).get("pr")
            kinds = []
            uncertainty = ["Checkout association does not establish who authored the output."]
            availability_record = latest.get(("observer", "availability"))
            availability = availability_record.data if availability_record else {}
            if availability.get("github_error"):
                uncertainty.append("GitHub refresh failed. Status comes from the last successful observation.")
            if availability.get("local_error"):
                uncertainty.append("Local repository refresh failed. Output information comes from the last successful observation.")
            if artifact["kind"] == "pull_request" and pr and pr.get("state") == "OPEN":
                if pr.get("review_requested") is True:
                    kinds.append(("review_output", "Review the pull request that requests your review."))
                current_checks = pr.get("head") and pr.get("check_revision") == pr["head"]
                failing = any(len(check) >= 2 and check[1] in FAILURES and
                              (len(check) < 4 or not check[3] or check[3] == pr.get("head")) for check in pr.get("checks", ()))
                if current_checks and failing:
                    kinds.append(("resolve_checks", "Inspect the failing checks on this pull request."))
            if (artifact["kind"] == "branch" and artifact.get("checkout") and github is not None and pr is None and progress
                    and not availability.get("github_error") and not availability.get("local_error")):
                git_record = latest.get(("git", "observation"))
                local = git_record.data if git_record else {}
                if any(local.get(key) for key in ("staged", "unstaged", "untracked", "ahead")):
                    kinds.append(("prepare_pr", "The branch output may benefit from review and pull request preparation."))
                    uncertainty.append("Agent progress does not establish that the work is complete.")
            evidence_records = [*latest.values()]
            if progress:
                evidence_records.append(progress)
            evidence = tuple(record.record() for record in evidence_records)
            for kind, next_step in kinds:
                identifier = hashlib.sha256(f"{work.id}:{artifact_record.id}:{kind}".encode()).hexdigest()[:24]
                data = {"intent": work.intent, "artifact": artifact, "evidence": evidence, "uncertainty": uncertainty,
                        "next_step": next_step, "sessions": sessions}
                fingerprint = canonical({"intent": work.intent, "artifact": artifact, "evidence": evidence_grounds(evidence),
                                         "kind": kind, "next_step": next_step, "connected_work": account_revision})
                previous = state.opportunities.get(identifier)
                if previous:
                    changed = previous.fingerprint != fingerprint or not previous.active
                    previous.version += int(changed)
                    previous.fingerprint = fingerprint
                    previous.data = data
                    previous.active = True
                    if changed:
                        previous.changed_at = now
                else:
                    state.opportunities[identifier] = OpportunityState(identifier, work.id, kind, artifact_record.id, 1,
                                                                      fingerprint, data, True, 0, now)
                candidates.add(identifier)
    for record in state.opportunities.values():
        if record.active and record.id not in candidates:
            record.active = False
            record.version += 1
            record.changed_at = now


def eligible(record):
    return record.active and record.surfaced_version != record.version


def project(record):
    data = record.data
    return Opportunity(record.id, record.version, record.kind, record.work_id, data["intent"], data["artifact"],
                       tuple(data["evidence"]), tuple(data["uncertainty"]), data["next_step"], record.changed_at, tuple(data["sessions"]))
