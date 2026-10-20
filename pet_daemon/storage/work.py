import sqlite3
from collections.abc import Iterable
from contextlib import AbstractContextManager

type SQLValue = str | int | float | bytes | None


from ..work.models import (
    Artifact,
    Evidence,
    OpportunityState,
    Surfacing,
    WorkItem,
    WorkState,
)
from ..work.serialization import (
    decode_artifact,
    decode_evidence,
    decode_opportunity,
    encode_artifact,
    encode_evidence,
    encode_opportunity,
)
from .transactions import transaction

TABLES = (
    "work_items",
    "work_sessions",
    "work_memberships",
    "work_aliases",
    "work_artifacts",
    "work_links",
    "work_evidence",
    "work_opportunities",
    "work_surfacings",
)


class SQLiteWorkRepository:
    def __init__(self, db: sqlite3.Connection, pet: str) -> None:
        self.db = db
        self.pet = pet

    def transaction(self) -> AbstractContextManager[None]:
        return transaction(self.db)

    def rows(self, table: str) -> list[sqlite3.Row]:
        return self.db.execute(
            f"SELECT * FROM {table} WHERE pet=?", (self.pet,)
        ).fetchall()

    def load(self) -> WorkState:
        state = WorkState()
        state.next_evidence_id = self.db.execute(
            "SELECT COALESCE(MAX(id),0)+1 FROM work_evidence"
        ).fetchone()[0]
        state.next_surfacing_id = self.db.execute(
            "SELECT COALESCE(MAX(id),0)+1 FROM work_surfacings"
        ).fetchone()[0]
        state.items = {
            row["id"]: WorkItem(
                row["id"], row["intent"], row["status"], row["changed_at"]
            )
            for row in self.rows("work_items")
        }
        state.sessions = {
            row["session"]: row["work_id"] for row in self.rows("work_sessions")
        }
        state.memberships = {
            (row["work_id"], row["session"]) for row in self.rows("work_memberships")
        }
        state.aliases = {
            row["alias"]: row["session"] for row in self.rows("work_aliases")
        }
        for row in self.rows("work_artifacts"):
            descriptor = decode_artifact(row["data"])
            state.artifacts[row["id"]] = Artifact(
                kind=descriptor.kind,
                ref=descriptor.ref,
                repository=descriptor.repository,
                revision=descriptor.revision,
                checkout=descriptor.checkout,
                title=descriptor.title,
                id=row["id"],
                version=row["version"],
            )
        state.links = {
            (row["work_id"], row["artifact_id"]) for row in self.rows("work_links")
        }
        state.evidence = [
            Evidence(
                row["id"],
                row["work_id"],
                row["artifact_id"],
                row["source"],
                row["kind"],
                decode_evidence(row["source"], row["kind"], row["data"]),
                row["timestamp"],
            )
            for row in sorted(self.rows("work_evidence"), key=lambda row: row["id"])
        ]
        for row in self.rows("work_opportunities"):
            content = decode_opportunity(row["data"])
            state.opportunities[row["id"]] = OpportunityState(
                id=row["id"],
                work_id=row["work_id"],
                kind=row["kind"],
                artifact_id=row["artifact_id"],
                version=row["version"],
                fingerprint=row["fingerprint"],
                intent=content.intent,
                artifact=content.artifact,
                evidence=content.evidence,
                uncertainty=content.uncertainty,
                next_step=content.next_step,
                sessions=content.sessions,
                active=bool(row["active"]),
                surfaced_version=row["surfaced_version"],
                changed_at=row["changed_at"],
            )
        state.surfacings = [
            Surfacing(
                row["id"],
                row["opportunity_id"],
                row["version"],
                row["outcome"],
                row["timestamp"],
            )
            for row in sorted(self.rows("work_surfacings"), key=lambda row: row["id"])
        ]
        return state

    def save(self, state: WorkState) -> None:
        pet = self.pet
        self._sync(
            "work_items",
            ("pet", "id", "intent", "status", "changed_at"),
            ("id",),
            (
                (pet, item.id, item.intent, item.status, item.changed_at)
                for item in state.items.values()
            ),
        )
        self._sync(
            "work_sessions",
            ("pet", "session", "work_id"),
            ("session",),
            ((pet, session, work_id) for session, work_id in state.sessions.items()),
        )
        self._sync(
            "work_memberships",
            ("pet", "work_id", "session"),
            ("work_id", "session"),
            ((pet, work_id, session) for work_id, session in state.memberships),
        )
        self._sync(
            "work_aliases",
            ("pet", "alias", "session"),
            ("alias",),
            ((pet, alias, session) for alias, session in state.aliases.items()),
        )
        self._sync(
            "work_artifacts",
            ("pet", "id", "kind", "ref", "repository", "data", "version"),
            ("id",),
            (
                (
                    pet,
                    item.id,
                    item.kind,
                    item.ref,
                    item.repository,
                    encode_artifact(item),
                    item.version,
                )
                for item in state.artifacts.values()
            ),
        )
        self._sync(
            "work_links",
            ("pet", "work_id", "artifact_id"),
            ("work_id", "artifact_id"),
            ((pet, work_id, artifact_id) for work_id, artifact_id in state.links),
        )
        self._sync(
            "work_evidence",
            (
                "id",
                "pet",
                "work_id",
                "artifact_id",
                "source",
                "kind",
                "data",
                "timestamp",
            ),
            ("id",),
            (
                (
                    item.id,
                    pet,
                    item.work_id,
                    item.artifact_id,
                    item.source,
                    item.kind,
                    encode_evidence(item.payload),
                    item.timestamp,
                )
                for item in state.evidence
            ),
        )
        self._sync(
            "work_opportunities",
            (
                "pet",
                "id",
                "work_id",
                "kind",
                "artifact_id",
                "version",
                "fingerprint",
                "data",
                "active",
                "surfaced_version",
                "changed_at",
            ),
            ("id",),
            (
                (
                    pet,
                    item.id,
                    item.work_id,
                    item.kind,
                    item.artifact_id,
                    item.version,
                    item.fingerprint,
                    encode_opportunity(item),
                    int(item.active),
                    item.surfaced_version,
                    item.changed_at,
                )
                for item in state.opportunities.values()
            ),
        )
        self._sync(
            "work_surfacings",
            ("id", "pet", "opportunity_id", "version", "outcome", "timestamp"),
            ("id",),
            (
                (
                    item.id,
                    pet,
                    item.opportunity_id,
                    item.version,
                    item.outcome,
                    item.timestamp,
                )
                for item in state.surfacings
            ),
        )

    def _sync(
        self,
        table: str,
        columns: tuple[str, ...],
        keys: tuple[str, ...],
        records: Iterable[tuple[SQLValue, ...]],
    ) -> None:
        key_positions = tuple(columns.index(key) for key in keys)
        existing = {
            tuple(row[key] for key in keys): tuple(row[column] for column in columns)
            for row in self.rows(table)
        }
        desired = {
            tuple(record[index] for index in key_positions): record
            for record in records
        }
        where = " AND ".join(("pet=?", *(f"{key}=?" for key in keys)))
        for key in existing.keys() - desired.keys():
            self.db.execute(f"DELETE FROM {table} WHERE {where}", (self.pet, *key))
        names = ",".join(columns)
        placeholders = ",".join("?" for _ in columns)
        for key, record in desired.items():
            previous = existing.get(key)
            if previous is None:
                self.db.execute(
                    f"INSERT INTO {table} ({names}) VALUES ({placeholders})", record
                )
            elif previous != record:
                changed = tuple(
                    index
                    for index, value in enumerate(record)
                    if previous[index] != value
                )
                assignments = ",".join(f"{columns[index]}=?" for index in changed)
                values = tuple(record[index] for index in changed)
                self.db.execute(
                    f"UPDATE {table} SET {assignments} WHERE {where}",
                    (*values, self.pet, *key),
                )

    def clear(self) -> None:
        for table in TABLES:
            self.db.execute(f"DELETE FROM {table} WHERE pet=?", (self.pet,))
