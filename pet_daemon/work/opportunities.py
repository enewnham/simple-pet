from dataclasses import dataclass
import hashlib
import json

from .validation import FAILURES, canonical


@dataclass(frozen=True)
class Opportunity:
    id: str
    version: int
    kind: str
    work_id: str
    intent: str
    artifact: dict
    evidence: tuple
    uncertainty: tuple
    next_step: str
    changed_at: float
    sessions: tuple


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


def refresh(store):
    candidates = set()
    for work in store._rows("SELECT * FROM work_items WHERE pet=? AND status='active'"):
        activity = store._latest(work["id"], "")
        progress = activity.get(("harness", "progress"))
        sessions = tuple(row["session"] for row in store._rows("SELECT session FROM work_memberships WHERE pet=? AND work_id=? ORDER BY session", (work["id"],)))
        account_revision = store.session_context(sessions[0]).get("revision") if sessions else None
        links = store._rows("SELECT a.* FROM work_artifacts a JOIN work_links l ON a.pet=l.pet AND a.id=l.artifact_id WHERE a.pet=? AND l.work_id=?", (work["id"],))
        for artifact_row in links:
            artifact = json.loads(artifact_row["data"])
            latest = store._latest(work["id"], artifact_row["id"])
            github_row = latest.get(("github", "observation"))
            github = json.loads(github_row["data"]) if github_row else None
            pr = (github or {}).get("pr")
            kinds = []
            uncertainty = ["Checkout association does not establish who authored the output."]
            availability_row = latest.get(("observer", "availability"))
            availability = json.loads(availability_row["data"]) if availability_row else {}
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
            if (artifact["kind"] == "branch" and github is not None and pr is None and progress
                    and not availability.get("github_error") and not availability.get("local_error")):
                git_row = latest.get(("git", "observation"))
                local = json.loads(git_row["data"]) if git_row else {}
                output = any(local.get(key) for key in ("staged", "unstaged", "untracked", "ahead"))
                if output:
                    kinds.append(("prepare_pr", "The branch output may benefit from review and pull request preparation."))
                    uncertainty.append("Agent progress does not establish that the work is complete.")
            evidence_rows = [*latest.values()]
            if progress:
                evidence_rows.append(progress)
            evidence = store._records(evidence_rows)
            for kind, next_step in kinds:
                identifier = hashlib.sha256(f"{work['id']}:{artifact_row['id']}:{kind}".encode()).hexdigest()[:24]
                data = {"intent": work["intent"], "artifact": artifact, "evidence": evidence, "uncertainty": uncertainty,
                        "next_step": next_step, "sessions": sessions}
                fingerprint = canonical({"intent": work["intent"], "artifact": artifact, "evidence": evidence_grounds(evidence),
                                         "kind": kind, "next_step": next_step, "connected_work": account_revision})
                previous = store._rows("SELECT * FROM work_opportunities WHERE pet=? AND id=?", (identifier,))
                if previous:
                    row = previous[0]
                    changed = row["fingerprint"] != fingerprint or not row["active"]
                    store.db.execute("UPDATE work_opportunities SET version=version+?,fingerprint=?,data=?,active=1,changed_at=? WHERE pet=? AND id=?",
                                     (int(changed), fingerprint, canonical(data), store.clock() if changed else row["changed_at"], store.pet, identifier))
                else:
                    store.db.execute("INSERT INTO work_opportunities VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                                     (store.pet, identifier, work["id"], kind, artifact_row["id"], 1, fingerprint, canonical(data), 1, 0, store.clock()))
                candidates.add(identifier)
    for row in store._rows("SELECT id FROM work_opportunities WHERE pet=? AND active=1"):
        if row["id"] not in candidates:
            store.db.execute("UPDATE work_opportunities SET active=0,version=version+1,changed_at=? WHERE pet=? AND id=?",
                             (store.clock(), store.pet, row["id"]))


def eligible(row):
    return bool(row["active"] and row["surfaced_version"] != row["version"])


def project(row):
    data = json.loads(row["data"])
    return Opportunity(row["id"], row["version"], row["kind"], row["work_id"], data["intent"], data["artifact"],
                       tuple(data["evidence"]), tuple(data["uncertainty"]), data["next_step"], row["changed_at"], tuple(data["sessions"]))
