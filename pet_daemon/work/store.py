import hashlib
import json
import time
import uuid

from .opportunities import Opportunity, eligible, evidence_grounds, project, refresh
from .validation import TABLES, canonical, descriptor, repository_identity, text


class WorkStore:
    def __init__(self, db, pet, clock=time.time, retention_days=30):
        self.db = db
        self.pet = pet
        self.clock = clock
        self.retention_days = retention_days
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS work_items (
                pet TEXT NOT NULL, id TEXT NOT NULL, intent TEXT NOT NULL, status TEXT NOT NULL,
                changed_at REAL NOT NULL, PRIMARY KEY(pet,id));
            CREATE TABLE IF NOT EXISTS work_sessions (
                pet TEXT NOT NULL, session TEXT NOT NULL, work_id TEXT NOT NULL,
                PRIMARY KEY(pet,session));
            CREATE TABLE IF NOT EXISTS work_memberships (
                pet TEXT NOT NULL, work_id TEXT NOT NULL, session TEXT NOT NULL,
                PRIMARY KEY(pet,work_id,session));
            CREATE TABLE IF NOT EXISTS work_aliases (
                pet TEXT NOT NULL, alias TEXT NOT NULL, session TEXT NOT NULL,
                PRIMARY KEY(pet,alias));
            CREATE TABLE IF NOT EXISTS work_artifacts (
                pet TEXT NOT NULL, id TEXT NOT NULL, kind TEXT NOT NULL, ref TEXT NOT NULL,
                repository TEXT NOT NULL, data TEXT NOT NULL, version INTEGER NOT NULL,
                PRIMARY KEY(pet,id));
            CREATE TABLE IF NOT EXISTS work_links (
                pet TEXT NOT NULL, work_id TEXT NOT NULL, artifact_id TEXT NOT NULL,
                PRIMARY KEY(pet,work_id,artifact_id));
            CREATE TABLE IF NOT EXISTS work_evidence (
                id INTEGER PRIMARY KEY, pet TEXT NOT NULL, work_id TEXT NOT NULL,
                artifact_id TEXT NOT NULL, source TEXT NOT NULL, kind TEXT NOT NULL,
                data TEXT NOT NULL, timestamp REAL NOT NULL);
            CREATE INDEX IF NOT EXISTS work_evidence_scope ON work_evidence(pet,work_id,artifact_id,source,kind,id);
            CREATE TABLE IF NOT EXISTS work_opportunities (
                pet TEXT NOT NULL, id TEXT NOT NULL, work_id TEXT NOT NULL, kind TEXT NOT NULL,
                artifact_id TEXT NOT NULL, version INTEGER NOT NULL, fingerprint TEXT NOT NULL,
                data TEXT NOT NULL, active INTEGER NOT NULL, surfaced_version INTEGER NOT NULL,
                changed_at REAL NOT NULL, PRIMARY KEY(pet,id));
            CREATE TABLE IF NOT EXISTS work_surfacings (
                id INTEGER PRIMARY KEY, pet TEXT NOT NULL, opportunity_id TEXT NOT NULL,
                version INTEGER NOT NULL, outcome TEXT NOT NULL, timestamp REAL NOT NULL);
        """)
        with self.db:
            refresh(self)

    def _rows(self, query, values=()):
        return self.db.execute(query, (self.pet, *values)).fetchall()

    def _work(self, identifier):
        rows = self._rows("SELECT * FROM work_items WHERE pet=? AND id=?", (identifier,))
        return rows[0] if rows else None

    def _current(self, session):
        rows = self._rows("SELECT w.* FROM work_items w JOIN work_sessions s ON w.pet=s.pet AND w.id=s.work_id WHERE w.pet=? AND s.session=?", (session,))
        return rows[0] if rows else None

    def _bind(self, work_id, sessions):
        for session in sessions:
            roots = self._rows("SELECT session FROM work_aliases WHERE pet=? AND alias=?", (session,))
            root = roots[0]["session"] if roots else session
            names = {session, root, *(row["alias"] for row in self._rows("SELECT alias FROM work_aliases WHERE pet=? AND session=?", (root,)))}
            for name in names:
                self.db.execute("INSERT INTO work_sessions VALUES (?,?,?) ON CONFLICT(pet,session) DO UPDATE SET work_id=excluded.work_id", (self.pet, name, work_id))
                self.db.execute("INSERT OR IGNORE INTO work_memberships VALUES (?,?,?)", (self.pet, work_id, name))

    def _aliases(self, session, aliases):
        roots = self._rows("SELECT session FROM work_aliases WHERE pet=? AND alias=?", (session,))
        root = roots[0]["session"] if roots else session
        for alias in aliases:
            old = self._rows("SELECT session FROM work_aliases WHERE pet=? AND alias=?", (alias,))
            old_root = old[0]["session"] if old else alias
            self.db.execute("UPDATE work_aliases SET session=? WHERE pet=? AND session=?", (root, self.pet, old_root))
            self.db.execute("INSERT INTO work_aliases VALUES (?,?,?) ON CONFLICT(pet,alias) DO UPDATE SET session=excluded.session", (self.pet, alias, root))

    def _new(self, intent):
        identifier = uuid.uuid4().hex
        self.db.execute("INSERT INTO work_items VALUES (?,?,?,?,?)", (self.pet, identifier, intent, "active", self.clock()))
        return self._work(identifier)

    def _evidence(self, work_id, artifact_id, source, kind, data, observed_at=None):
        content = canonical(data)
        timestamp = self.clock() if observed_at is None else observed_at
        previous = self._rows("SELECT id,data,timestamp FROM work_evidence WHERE pet=? AND work_id=? AND artifact_id=? AND source=? AND kind=? ORDER BY id DESC LIMIT 1", (work_id, artifact_id, source, kind))
        if kind == "observation" and previous and timestamp < previous[0]["timestamp"]:
            return
        if previous and previous[0]["data"] == content:
            if kind == "observation":
                self.db.execute("UPDATE work_evidence SET timestamp=? WHERE pet=? AND id=?", (timestamp, self.pet, previous[0]["id"]))
            return
        self.db.execute("INSERT INTO work_evidence(pet,work_id,artifact_id,source,kind,data,timestamp) VALUES (?,?,?,?,?,?,?)", (self.pet, work_id, artifact_id, source, kind, content, timestamp))

    def session_intent(self, session, intent, aliases=()):
        session = text(session, "session")
        intent = text(intent, "intent")
        if not isinstance(aliases, (tuple, list)):
            raise ValueError("aliases must be a list")
        aliases = tuple(text(alias, "alias") for alias in aliases)
        with self.db:
            candidates = {item["id"] for name in (session, *aliases) if (item := self._current(name))}
            if len(candidates) > 1:
                raise ValueError("session aliases refer to different work")
            work = self._work(next(iter(candidates))) if candidates else None
            if work is None or work["status"] == "concluded":
                work = self._new(intent)
            elif work["intent"] != intent:
                self.db.execute("UPDATE work_items SET intent=?,changed_at=? WHERE pet=? AND id=?", (intent, self.clock(), self.pet, work["id"]))
            self._aliases(session, aliases)
            self._bind(work["id"], (session, *aliases))
            self._evidence(work["id"], "", "user", "intent", {"session": session, "text": intent})
            refresh(self)
        return work["id"]

    def session_event(self, session, event):
        if not isinstance(event, dict):
            raise ValueError("session event must be an object")
        aliases = event.get("aliases", ())
        if not isinstance(aliases, (list, tuple)):
            raise ValueError("aliases must be a list")
        aliases = tuple(text(alias, "alias") for alias in aliases)
        harness = event.get("harness")
        if harness:
            aliases = tuple(alias if alias.startswith(harness + ":") else f"{harness}:{alias}" for alias in aliases)
        with self.db:
            candidates = {item["id"] for name in (session, *aliases) if (item := self._current(name))}
            if len(candidates) > 1:
                return
            if not candidates:
                return
            work_id = next(iter(candidates))
            self._aliases(session, aliases)
            self._bind(work_id, (session, *aliases))
            observation = {key: event[key] for key in ("event", "outcome", "tool_name", "progress") if key in event}
            self._evidence(work_id, "", "harness", "execution", {"session": session, **observation})
            if event.get("progress"):
                self._evidence(work_id, "", "harness", "progress", {"session": session, "text": event["progress"]})
            refresh(self)

    def _artifact(self, value, source=None, observed_at=None):
        value = descriptor(value)
        rows = self._rows("SELECT * FROM work_artifacts WHERE pet=? AND kind=? AND ref=? AND repository=?",
                          (value["kind"], value["ref"], value["repository"]))
        if rows:
            previous = rows[0]
            if source is not None and observed_at is not None:
                latest = self._rows("SELECT MAX(timestamp) AS timestamp FROM work_evidence WHERE pet=? AND artifact_id=? AND source=? AND kind='observation'",
                                    (previous["id"], source))[0]["timestamp"]
                if latest is not None and observed_at < latest:
                    return previous["id"]
            data = canonical({**json.loads(previous["data"]), **value})
            self.db.execute("UPDATE work_artifacts SET data=?,version=version+? WHERE pet=? AND id=?",
                            (data, int(data != previous["data"]), self.pet, previous["id"]))
            return previous["id"]
        identifier = uuid.uuid4().hex
        self.db.execute("INSERT INTO work_artifacts VALUES (?,?,?,?,?,?,?)",
                        (self.pet, identifier, value["kind"], value["ref"], value["repository"], canonical(value), 1))
        return identifier

    def _link(self, work_id, artifact_id, source, data, observed_at=None):
        self.db.execute("INSERT OR IGNORE INTO work_links VALUES (?,?,?)", (self.pet, work_id, artifact_id))
        self._evidence(work_id, artifact_id, source, "relationship", data, observed_at)

    def _linked_work(self, artifact_id):
        return {row["work_id"] for row in self._rows("SELECT work_id FROM work_links WHERE pet=? AND artifact_id=?", (artifact_id,))}

    def _latest(self, work_id, artifact_id):
        rows = self._rows("""SELECT * FROM work_evidence WHERE pet=? AND id IN (
            SELECT MAX(id) FROM work_evidence WHERE pet=? AND work_id=? AND artifact_id=?
            GROUP BY source,kind) ORDER BY source,kind""", (self.pet, work_id, artifact_id))
        return {(row["source"], row["kind"]): row for row in rows}

    def _records(self, rows):
        return tuple({"source": row["source"], "kind": row["kind"], "data": json.loads(row["data"]),
                      "observed_at": row["timestamp"]} for row in rows)

    def tracked_artifacts(self):
        return tuple(json.loads(row["data"]) for row in self._rows("SELECT data FROM work_artifacts WHERE pet=? ORDER BY kind,ref"))

    def observe_repository(self, observation):
        if not isinstance(observation, dict):
            raise ValueError("repository observation must be an object")
        local = observation.get("local")
        pr = observation.get("pr")
        repository = repository_identity(observation.get("repository") or (local or {}).get("repository", ""))
        branch_name = observation.get("branch")
        checkout = observation.get("worktree")
        with self.db:
            work_ids = set()
            if checkout and observation.get("current", True):
                work_ids = {work["id"] for session in observation.get("sessions", ()) if (work := self._current(session))}
            branch = None
            if repository and branch_name:
                value = {"kind": "branch", "ref": branch_name, "repository": repository}
                if checkout:
                    value["checkout"] = checkout
                if local and local.get("head"):
                    value["revision"] = local["head"]
                branch = self._artifact(value, "git", observation.get("local_checked_at", 0))
                for work_id in work_ids:
                    self._link(work_id, branch, "harness", {"relationship": "session_checkout", "checkout": checkout,
                                                          "branch": branch_name, "association_only": True})
                work_ids.update(self._linked_work(branch))
                if local and not observation.get("local_error"):
                    for work_id in self._linked_work(branch):
                        self._evidence(work_id, branch, "git", "observation", local, observation.get("local_checked_at", 0))
            pull_request = None
            matches_branch = bool(branch and pr and pr.get("head_branch") == branch_name
                                  and repository_identity(pr.get("head_repository", "")) == repository)
            if pr and pr.get("url"):
                value = {"kind": "pull_request", "ref": pr["url"]}
                if pr.get("head"):
                    value["revision"] = pr["head"]
                if pr.get("title"):
                    value["title"] = pr["title"]
                pull_request = self._artifact(value, "github", observation.get("github_checked_at", 0))
                for work_id in work_ids if branch is None or matches_branch else ():
                    self._link(work_id, pull_request, "github", {"relationship": "checkout_pull_request", "checkout": checkout or "",
                                                                 "branch": branch_name or "", "association_only": True},
                               observation.get("github_checked_at", 0))
                work_ids.update(self._linked_work(pull_request))
                for link in pr.get("related_links", ()):
                    if link.get("kind") != "pull_request" or link.get("relationship") != "stack_entry" or link.get("source") != "github_stack":
                        continue
                    sibling = {"kind": "pull_request", "ref": link["ref"]}
                    if link.get("head"):
                        sibling["revision"] = link["head"]
                    if link.get("title"):
                        sibling["title"] = link["title"]
                    sibling_id = self._artifact(sibling, "github", observation.get("github_checked_at", 0))
                    for work_id in self._linked_work(pull_request):
                        self._link(work_id, sibling_id, "github_stack", {**link, "from": pr["url"], "association_only": True},
                                   observation.get("github_checked_at", 0))
            checked = observation.get("github_checked_at") and not observation.get("github_error")
            if checked:
                for artifact_id in (branch, pull_request):
                    if artifact_id:
                        if artifact_id == branch and pr and not matches_branch:
                            continue
                        observed_repository = descriptor({"kind": "pull_request", "ref": pr["url"]})["repository"] if artifact_id == pull_request else repository
                        data = {"pr": pr, "repository": observed_repository}
                        for work_id in self._linked_work(artifact_id):
                            self._evidence(work_id, artifact_id, "github", "observation", data, observation["github_checked_at"])
            for artifact_id in (branch, pull_request):
                if artifact_id:
                    for work_id in self._linked_work(artifact_id):
                        github_error = observation.get("github_error")
                        if artifact_id == branch and pr and not matches_branch:
                            github_error = github_error or "Pull request head identity does not match the observed branch."
                        self._evidence(work_id, artifact_id, "observer", "availability",
                                       {"local_error": observation.get("local_error"), "github_error": github_error})
            refresh(self)

    def session_context(self, session):
        work = self._current(session)
        if not work:
            return {}
        requests = self._rows("SELECT * FROM work_evidence WHERE pet=? AND work_id=? AND source='user' AND kind='intent' ORDER BY id DESC LIMIT 4", (work["id"],))
        activity = [row for (source, kind), row in self._latest(work["id"], "").items()
                    if source == "harness" and kind in {"execution", "progress"}]
        artifacts = []
        rows = self._rows("SELECT a.* FROM work_artifacts a JOIN work_links l ON a.pet=l.pet AND a.id=l.artifact_id WHERE a.pet=? AND l.work_id=? ORDER BY a.kind,a.ref", (work["id"],))
        for row in rows:
            evidence = self._records(self._latest(work["id"], row["id"]).values())
            artifacts.append({**json.loads(row["data"]), "evidence": evidence})
        context = {"intent": work["intent"], "requests": self._records(reversed(requests)),
                   "activity": self._records(activity), "artifacts": tuple(artifacts)}
        grounds = {"intent": context["intent"], "requests": evidence_grounds(context["requests"]),
                   "activity": evidence_grounds(context["activity"]),
                   "artifacts": [{**{key: value for key, value in artifact.items() if key != "evidence"},
                                  "evidence": evidence_grounds(artifact["evidence"])} for artifact in artifacts]}
        context["revision"] = hashlib.sha256(canonical(grounds).encode()).hexdigest()
        return context

    def opportunities(self):
        return tuple(project(row) for row in self._rows("SELECT * FROM work_opportunities WHERE pet=? ORDER BY changed_at DESC,id") if eligible(row))

    def opportunity(self, identifier):
        rows = self._rows("SELECT * FROM work_opportunities WHERE pet=? AND id=?", (identifier,))
        return project(rows[0]) if rows and eligible(rows[0]) else None

    def is_current(self, identifier, version):
        opportunity = self.opportunity(identifier)
        return bool(opportunity and opportunity.version == version)

    def record_surfacing(self, identifier, version, outcome):
        with self.db:
            self.db.execute("INSERT INTO work_surfacings(pet,opportunity_id,version,outcome,timestamp) VALUES (?,?,?,?,?)",
                            (self.pet, identifier, version, outcome, self.clock()))
            if outcome == "accepted" and self.is_current(identifier, version):
                self.db.execute("UPDATE work_opportunities SET surfaced_version=? WHERE pet=? AND id=? AND version=?",
                                (version, self.pet, identifier, version))

    def prune(self):
        cutoff = self.clock() - self.retention_days * 86400
        with self.db:
            self.db.execute("""DELETE FROM work_evidence WHERE pet=? AND timestamp < ? AND id NOT IN (
                SELECT MAX(id) FROM work_evidence WHERE pet=? GROUP BY work_id,artifact_id,source,kind)""", (self.pet, cutoff, self.pet))
            self.db.execute("""DELETE FROM work_surfacings WHERE pet=? AND timestamp < ? AND id NOT IN (
                SELECT MAX(id) FROM work_surfacings WHERE pet=? GROUP BY opportunity_id,version,outcome)""", (self.pet, cutoff, self.pet))

    def clear(self):
        with self.db:
            for table in TABLES:
                self.db.execute(f"DELETE FROM {table} WHERE pet=?", (self.pet,))
            for legacy in ("work_updates", "work_decisions"):
                if self.db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (legacy,)).fetchone():
                    self.db.execute(f"DELETE FROM {legacy} WHERE pet=?", (self.pet,))
