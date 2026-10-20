import json

from ..work.models import Artifact, Evidence, OpportunityState, Surfacing, WorkItem, WorkState
from ..work.validation import canonical
from .transactions import transaction

TABLES = ("work_items", "work_sessions", "work_memberships", "work_aliases", "work_artifacts", "work_links",
          "work_evidence", "work_opportunities", "work_surfacings")


class SQLiteWorkRepository:
    def __init__(self, db, pet):
        self.db = db
        self.pet = pet

    def transaction(self):
        return transaction(self.db)

    def rows(self, table):
        return self.db.execute(f"SELECT * FROM {table} WHERE pet=?", (self.pet,)).fetchall()

    def load(self):
        state = WorkState()
        state.next_evidence_id = self.db.execute("SELECT COALESCE(MAX(id),0)+1 FROM work_evidence").fetchone()[0]
        state.next_surfacing_id = self.db.execute("SELECT COALESCE(MAX(id),0)+1 FROM work_surfacings").fetchone()[0]
        state.items = {row["id"]: WorkItem(row["id"], row["intent"], row["status"], row["changed_at"]) for row in self.rows("work_items")}
        state.sessions = {row["session"]: row["work_id"] for row in self.rows("work_sessions")}
        state.memberships = {(row["work_id"], row["session"]) for row in self.rows("work_memberships")}
        state.aliases = {row["alias"]: row["session"] for row in self.rows("work_aliases")}
        state.artifacts = {row["id"]: Artifact(row["id"], json.loads(row["data"]), row["version"]) for row in self.rows("work_artifacts")}
        state.links = {(row["work_id"], row["artifact_id"]) for row in self.rows("work_links")}
        state.evidence = [Evidence(row["id"], row["work_id"], row["artifact_id"], row["source"], row["kind"], json.loads(row["data"]), row["timestamp"]) for row in sorted(self.rows("work_evidence"), key=lambda row: row["id"])]
        state.opportunities = {row["id"]: OpportunityState(row["id"], row["work_id"], row["kind"], row["artifact_id"], row["version"], row["fingerprint"], json.loads(row["data"]), bool(row["active"]), row["surfaced_version"], row["changed_at"]) for row in self.rows("work_opportunities")}
        state.surfacings = [Surfacing(row["id"], row["opportunity_id"], row["version"], row["outcome"], row["timestamp"]) for row in sorted(self.rows("work_surfacings"), key=lambda row: row["id"])]
        return state

    def save(self, state):
        pet = self.pet
        self._sync("work_items", ("pet", "id", "intent", "status", "changed_at"), ("id",),
                   ((pet, item.id, item.intent, item.status, item.changed_at) for item in state.items.values()))
        self._sync("work_sessions", ("pet", "session", "work_id"), ("session",),
                   ((pet, session, work_id) for session, work_id in state.sessions.items()))
        self._sync("work_memberships", ("pet", "work_id", "session"), ("work_id", "session"),
                   ((pet, work_id, session) for work_id, session in state.memberships))
        self._sync("work_aliases", ("pet", "alias", "session"), ("alias",),
                   ((pet, alias, session) for alias, session in state.aliases.items()))
        self._sync("work_artifacts", ("pet", "id", "kind", "ref", "repository", "data", "version"), ("id",),
                   ((pet, item.id, item.data["kind"], item.data["ref"], item.data["repository"], canonical(item.data), item.version)
                    for item in state.artifacts.values()))
        self._sync("work_links", ("pet", "work_id", "artifact_id"), ("work_id", "artifact_id"),
                   ((pet, work_id, artifact_id) for work_id, artifact_id in state.links))
        self._sync("work_evidence", ("id", "pet", "work_id", "artifact_id", "source", "kind", "data", "timestamp"), ("id",),
                   ((item.id, pet, item.work_id, item.artifact_id, item.source, item.kind, canonical(item.data), item.timestamp)
                    for item in state.evidence))
        self._sync("work_opportunities", ("pet", "id", "work_id", "kind", "artifact_id", "version", "fingerprint", "data", "active", "surfaced_version", "changed_at"), ("id",),
                   ((pet, item.id, item.work_id, item.kind, item.artifact_id, item.version, item.fingerprint, canonical(item.data), int(item.active), item.surfaced_version, item.changed_at)
                    for item in state.opportunities.values()))
        self._sync("work_surfacings", ("id", "pet", "opportunity_id", "version", "outcome", "timestamp"), ("id",),
                   ((item.id, pet, item.opportunity_id, item.version, item.outcome, item.timestamp) for item in state.surfacings))

    def _sync(self, table, columns, keys, records):
        key_positions = tuple(columns.index(key) for key in keys)
        existing = {tuple(row[key] for key in keys): tuple(row[column] for column in columns) for row in self.rows(table)}
        desired = {tuple(record[index] for index in key_positions): record for record in records}
        where = " AND ".join(("pet=?", *(f"{key}=?" for key in keys)))
        for key in existing.keys() - desired.keys():
            self.db.execute(f"DELETE FROM {table} WHERE {where}", (self.pet, *key))
        names = ",".join(columns)
        placeholders = ",".join("?" for _ in columns)
        for key, record in desired.items():
            previous = existing.get(key)
            if previous is None:
                self.db.execute(f"INSERT INTO {table} ({names}) VALUES ({placeholders})", record)
            elif previous != record:
                changed = tuple(index for index, value in enumerate(record) if previous[index] != value)
                assignments = ",".join(f"{columns[index]}=?" for index in changed)
                values = tuple(record[index] for index in changed)
                self.db.execute(f"UPDATE {table} SET {assignments} WHERE {where}", (*values, self.pet, *key))

    def clear(self):
        for table in TABLES:
            self.db.execute(f"DELETE FROM {table} WHERE pet=?", (self.pet,))

    def clear_legacy(self):
        for table in ("work_updates", "work_decisions"):
            if self.db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone():
                self.db.execute(f"DELETE FROM {table} WHERE pet=?", (self.pet,))
