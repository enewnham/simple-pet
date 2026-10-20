import json
import sqlite3
import time
from collections.abc import Callable, Iterable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path

from ..prompts import select_prompt
from ..telemetry import verbose
from ..work import WorkService
from ..work.history import RecallQuery, SessionFact
from ..work.observations import (
    RepositoryObservation,
    repository_observation_from_mapping,
)
from .migrations import migrate
from .transactions import transaction
from .work import SQLiteWorkRepository


@dataclass(frozen=True)
class HistoryRecord:
    last_selected: float
    considered_revision: str | None


@dataclass(frozen=True)
class EventRecord:
    id: int
    session: str
    repo: str
    event_type: str
    summary: str
    goal: str
    timestamp: float


@dataclass(frozen=True)
class ProjectRecord:
    repo: str
    goal: str
    last_result: str
    timestamp: float


@dataclass(frozen=True)
class ContextRecord:
    first_prompt: str
    session_context: str
    recent_requests: list[str]
    request_count: int
    first_summary_attempted: bool


@dataclass(frozen=True)
class MessageOutcome:
    message_id: str
    outcome: str
    text: str
    reason: str
    opportunity_id: str | None
    version: int | None
    timestamp: float


@dataclass(frozen=True)
class SpeechRecord:
    text: str
    reason: str
    timestamp: float


@dataclass(frozen=True)
class RepositoryAssociation:
    session: str
    worktree: str


class Memory:
    def __init__(
        self,
        path: str | Path,
        pet: str,
        clock: Callable[[], float] = time.time,
        retention_days: int = 30,
        max_events: int = 10000,
        speech_history_seconds: float = 1800,
    ) -> None:
        self.pet = pet
        self.clock = clock
        self.retention_days = retention_days
        self.max_events = max_events
        self.speech_history_seconds = speech_history_seconds
        path = Path(path)
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        path.chmod(0o600)
        self.db.row_factory = sqlite3.Row
        try:
            migrate(self.db)
        except BaseException:
            self.db.close()
            raise
        self.work = WorkService(
            SQLiteWorkRepository(self.db, pet),
            clock,
            retention_days=self.retention_days,
        )
        self.prune()

    def transaction(self) -> AbstractContextManager[None]:
        return transaction(self.db)

    def load_attention_history(self) -> dict[str, HistoryRecord]:
        rows = self.db.execute(
            "SELECT subject,last_selected,considered_revision FROM attention_history WHERE pet=?",
            (self.pet,),
        ).fetchall()
        return {
            row["subject"]: HistoryRecord(
                last_selected=row["last_selected"],
                considered_revision=json.loads(row["considered_revision"]),
            )
            for row in rows
        }

    def save_attention_history(
        self, subject_id: str, last_selected: float, considered_revision: str | None
    ) -> None:
        with self.transaction():
            self.db.execute(
                """INSERT INTO attention_history VALUES (?,?,?,?,?) ON CONFLICT(pet,subject) DO UPDATE SET
                last_selected=excluded.last_selected,considered_revision=excluded.considered_revision,updated_at=excluded.updated_at""",
                (
                    self.pet,
                    subject_id,
                    last_selected,
                    json.dumps(considered_revision),
                    self.clock(),
                ),
            )

    def record_acceptance(
        self,
        message_id: str,
        text: str,
        reason: str,
        opportunity_id: str | None = None,
        version: int | None = None,
    ) -> bool:
        with self.transaction():
            previous = self.db.execute(
                "SELECT outcome FROM message_outcomes WHERE pet=? AND message_id=?",
                (self.pet, message_id),
            ).fetchone()
            if previous and previous["outcome"] == "accepted":
                return False
            timestamp = self.clock()
            self.db.execute(
                """INSERT INTO message_outcomes VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(pet,message_id) DO UPDATE SET
                outcome=excluded.outcome,text=excluded.text,reason=excluded.reason,opportunity_id=excluded.opportunity_id,
                version=excluded.version,timestamp=excluded.timestamp""",
                (
                    self.pet,
                    message_id,
                    "accepted",
                    text,
                    reason,
                    opportunity_id,
                    version,
                    timestamp,
                ),
            )
            self.db.execute(
                "INSERT INTO speech(pet,text,reason,timestamp) VALUES (?,?,?,?)",
                (self.pet, text, reason, timestamp),
            )
            if opportunity_id is not None and version is not None:
                self.db.execute(
                    "INSERT INTO work_surfacings(pet,opportunity_id,version,outcome,timestamp) VALUES (?,?,?,?,?)",
                    (self.pet, opportunity_id, version, "accepted", timestamp),
                )
                self.db.execute(
                    "UPDATE work_opportunities SET surfaced_version=? WHERE pet=? AND id=? AND version=?",
                    (version, self.pet, opportunity_id, version),
                )
        self.prune()
        return True

    def record_delivery_failure(
        self,
        message_id: str,
        reason: str,
        text: str = "",
        opportunity_id: str | None = None,
        version: int | None = None,
    ) -> bool:
        with self.transaction():
            result = self.db.execute(
                "INSERT OR IGNORE INTO message_outcomes VALUES (?,?,?,?,?,?,?,?)",
                (
                    self.pet,
                    message_id,
                    "failed",
                    text,
                    reason,
                    opportunity_id,
                    version,
                    self.clock(),
                ),
            )
        return bool(result.rowcount)

    def message_outcome(self, message_id: str) -> MessageOutcome | None:
        row = self.db.execute(
            "SELECT message_id,outcome,text,reason,opportunity_id,version,timestamp FROM message_outcomes WHERE pet=? AND message_id=?",
            (self.pet, message_id),
        ).fetchone()
        return (
            MessageOutcome(
                message_id=row["message_id"],
                outcome=row["outcome"],
                text=row["text"],
                reason=row["reason"],
                opportunity_id=row["opportunity_id"],
                version=row["version"],
                timestamp=row["timestamp"],
            )
            if row
            else None
        )

    def prune(self) -> None:
        now = self.clock()
        cutoff = now - self.retention_days * 86400
        speech_cutoff = now - self.speech_history_seconds
        with self.transaction():
            self.db.execute(
                "DELETE FROM events WHERE pet=? AND timestamp < ?", (self.pet, cutoff)
            )
            self.db.execute(
                "DELETE FROM events WHERE pet=? AND id IN (SELECT id FROM events WHERE pet=? ORDER BY id DESC LIMIT -1 OFFSET ?)",
                (self.pet, self.pet, self.max_events),
            )
            self.db.execute(
                "DELETE FROM speech WHERE pet=? AND (timestamp <= ? OR id NOT IN (SELECT id FROM speech WHERE pet=? ORDER BY id DESC LIMIT 256))",
                (self.pet, speech_cutoff, self.pet),
            )
            self.db.execute(
                "DELETE FROM projects WHERE pet=? AND (timestamp < ? OR rowid NOT IN (SELECT rowid FROM projects WHERE pet=? ORDER BY timestamp DESC LIMIT 128))",
                (self.pet, cutoff, self.pet),
            )
            self.db.execute(
                "DELETE FROM contexts WHERE pet=? AND (timestamp < ? OR rowid NOT IN (SELECT rowid FROM contexts WHERE pet=? ORDER BY timestamp DESC LIMIT 128))",
                (self.pet, cutoff, self.pet),
            )
            self.db.execute(
                "DELETE FROM attention_history WHERE pet=? AND updated_at < ?",
                (self.pet, cutoff),
            )
            self.db.execute(
                "DELETE FROM message_outcomes WHERE pet=? AND timestamp < ?",
                (self.pet, cutoff),
            )
        self.db.execute("PRAGMA incremental_vacuum(64)")
        self.work.prune()

    def record_event(self, record: SessionFact | None) -> int | None:
        if record is None:
            return None
        if record.coalesce:
            previous = self.db.execute(
                "SELECT event_type,summary FROM events WHERE pet=? AND session=? AND repo=? ORDER BY id DESC LIMIT 1",
                (self.pet, record.session, record.repo),
            ).fetchone()
            if (
                previous
                and previous["event_type"] == record.event_type
                and previous["summary"] == record.summary
            ):
                return None
        timestamp = self.clock()
        with self.transaction():
            cursor = self.db.execute(
                "INSERT INTO events(pet,session,repo,event_type,summary,goal,timestamp) VALUES (?,?,?,?,?,?,?)",
                (
                    self.pet,
                    record.session,
                    record.repo,
                    record.event_type,
                    record.summary,
                    record.goal,
                    timestamp,
                ),
            )
            if record.repo and record.requested_goal:
                self.db.execute(
                    """INSERT INTO projects VALUES (?,?,?,?,?) ON CONFLICT(pet,repo) DO UPDATE SET
                    goal=excluded.goal, last_result=excluded.last_result, timestamp=excluded.timestamp""",
                    (
                        self.pet,
                        record.repo,
                        record.requested_goal,
                        record.summary,
                        timestamp,
                    ),
                )
        self.prune()
        verbose("memory.event", id=cursor.lastrowid, event_type=record.event_type)
        return cursor.lastrowid

    def recent(
        self,
        limit: int = 5,
        *,
        session: str | None = None,
        repo: str | None = None,
        event_types: Iterable[str] | None = None,
    ) -> list[EventRecord]:
        filters = ["pet=?"]
        values = [self.pet]
        if session is not None:
            filters.append("session=?")
            values.append(session)
        if repo is not None:
            filters.append("repo=?")
            values.append(repo)
        if event_types is not None:
            kinds = tuple(event_types)
            if not kinds:
                return []
            placeholders = ",".join("?" for _ in kinds)
            filters.append(f"event_type IN ({placeholders})")
            values.extend(kinds)
        where = " AND ".join(filters)
        rows = self.db.execute(
            f"""WITH ranked AS (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY session ORDER BY id DESC) AS rank FROM events WHERE {where}
        ) SELECT id, session, repo, event_type, summary, goal, timestamp
          FROM ranked ORDER BY rank=1 DESC, id DESC LIMIT ?""",
            (*values, limit),
        ).fetchall()
        return [
            EventRecord(
                id=row["id"],
                session=row["session"],
                repo=row["repo"],
                event_type=row["event_type"],
                summary=row["summary"],
                goal=row["goal"],
                timestamp=row["timestamp"],
            )
            for row in sorted(rows, key=lambda row: row["id"])
        ]

    def recall(self, request: RecallQuery | None, limit: int = 2) -> list[EventRecord]:
        if request is None:
            return []
        query = " AND ".join('"' + term + '"' for term in request.terms)
        excluded = [row.id for row in self.recent()]
        placeholders = ",".join("?" for _ in excluded) or "NULL"
        rows = self.db.execute(
            f"""SELECT e.* FROM events_fts JOIN events e ON e.id=events_fts.rowid
            WHERE events_fts MATCH ? AND e.pet=? AND e.repo=? AND e.id NOT IN ({placeholders})
            ORDER BY e.session=? DESC, bm25(events_fts), e.id DESC LIMIT ?""",
            (query, self.pet, request.repository, *excluded, request.session, limit),
        ).fetchall()
        return [
            EventRecord(
                id=row["id"],
                session=row["session"],
                repo=row["repo"],
                event_type=row["event_type"],
                summary=row["summary"],
                goal=row["goal"],
                timestamp=row["timestamp"],
            )
            for row in rows
        ]

    def projects(self, repos: Iterable[str]) -> list[ProjectRecord]:
        result = []
        for repo in list(dict.fromkeys(repos))[:3]:
            row = self.db.execute(
                "SELECT repo,goal,last_result,timestamp FROM projects WHERE pet=? AND repo=?",
                (self.pet, repo),
            ).fetchone()
            if row:
                result.append(
                    ProjectRecord(
                        repo=row["repo"],
                        goal=row["goal"],
                        last_result=row["last_result"],
                        timestamp=row["timestamp"],
                    )
                )
        return result

    def context(self, session: str) -> ContextRecord | None:
        row = self.db.execute(
            "SELECT first_prompt,session_context,recent_requests,request_count,first_summary_attempted FROM contexts WHERE pet=? AND session=?",
            (self.pet, session),
        ).fetchone()
        if row is None:
            return None
        return ContextRecord(
            first_prompt=row["first_prompt"],
            session_context=row["session_context"],
            recent_requests=json.loads(row["recent_requests"]),
            request_count=row["request_count"],
            first_summary_attempted=bool(row["first_summary_attempted"]),
        )

    def save_context(
        self,
        session: str,
        first_prompt: str,
        summary: str | None,
        recent_requests: Sequence[str],
        request_count: int | None = None,
        first_summary_attempted: bool = False,
    ) -> None:
        requests = [select_prompt(request) for request in recent_requests[-4:]]
        count = len(requests) if request_count is None else request_count
        with self.transaction():
            self.db.execute(
                """INSERT INTO contexts VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(pet,session) DO UPDATE SET
                first_prompt=excluded.first_prompt, session_context=excluded.session_context,
                recent_requests=excluded.recent_requests, timestamp=excluded.timestamp,
                request_count=excluded.request_count, first_summary_attempted=excluded.first_summary_attempted""",
                (
                    self.pet,
                    session,
                    select_prompt(first_prompt),
                    summary or "",
                    json.dumps(requests),
                    self.clock(),
                    count,
                    first_summary_attempted,
                ),
            )
        self.prune()

    def record_speech(self, text: str, reason: str) -> None:
        with self.transaction():
            self.db.execute(
                "INSERT INTO speech(pet,text,reason,timestamp) VALUES (?,?,?,?)",
                (self.pet, text, reason, self.clock()),
            )
        self.prune()

    def associate_repository(self, session: str, worktree: str) -> None:
        with self.transaction():
            self.db.execute(
                "INSERT OR IGNORE INTO repository_associations VALUES (?,?,?)",
                (self.pet, session, worktree),
            )

    def repository_associations(self) -> list[RepositoryAssociation]:
        rows = self.db.execute(
            "SELECT session,worktree FROM repository_associations WHERE pet=?",
            (self.pet,),
        ).fetchall()
        return [
            RepositoryAssociation(session=row["session"], worktree=row["worktree"])
            for row in rows
        ]

    def forget_repository(self, worktree: str) -> None:
        with self.transaction():
            self.db.execute(
                "DELETE FROM repository_associations WHERE pet=? AND worktree=?",
                (self.pet, worktree),
            )
            rows = self.db.execute(
                "SELECT worktree,observation FROM repositories WHERE pet=?", (self.pet,)
            ).fetchall()
            subjects = [
                row["worktree"]
                for row in rows
                if json.loads(row["observation"])["worktree"] == worktree
            ]
            self.db.executemany(
                "DELETE FROM repositories WHERE pet=? AND worktree=?",
                ((self.pet, subject) for subject in subjects),
            )

    def save_repository(self, observation: RepositoryObservation) -> None:
        with self.transaction():
            self.db.execute(
                "INSERT INTO repositories VALUES (?,?,?) ON CONFLICT(pet,worktree) DO UPDATE SET observation=excluded.observation",
                (self.pet, observation.subject, json.dumps(observation.packet())),
            )

    def load_repositories(self) -> tuple[RepositoryObservation, ...]:
        rows = self.db.execute(
            "SELECT observation FROM repositories WHERE pet=?", (self.pet,)
        ).fetchall()
        return tuple(
            repository_observation_from_mapping(json.loads(row["observation"]))
            for row in rows
        )

    def speech(self, limit: int = 3) -> list[SpeechRecord]:
        rows = self.db.execute(
            "SELECT text,reason,timestamp FROM speech WHERE pet=? AND timestamp > ? ORDER BY id DESC LIMIT ?",
            (self.pet, self.clock() - self.speech_history_seconds, limit),
        ).fetchall()
        return [
            SpeechRecord(
                text=row["text"], reason=row["reason"], timestamp=row["timestamp"]
            )
            for row in reversed(rows)
        ]

    def clear_speech(self) -> None:
        with self.transaction():
            self.db.execute("DELETE FROM speech WHERE pet=?", (self.pet,))

    def clear(self) -> None:
        with self.transaction():
            self.work.clear()
            for table in (
                "events",
                "projects",
                "speech",
                "contexts",
                "repository_associations",
                "repositories",
                "attention_history",
                "message_outcomes",
            ):
                self.db.execute(f"DELETE FROM {table} WHERE pet=?", (self.pet,))

    def close(self) -> None:
        self.db.close()
