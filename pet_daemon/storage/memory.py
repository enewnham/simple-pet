import json
import sqlite3
import time
from collections.abc import Callable, Iterable, Sequence
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypedDict

from ..domain import DeliveryStatus, SpeechReason
from ..logging import get_logger
from ..prompts import select_prompt
from ..runtime import SwarmActivity
from ..work import WorkIngestion
from ..work.history import HistoryDetails, RecallQuery, SessionEntry
from ..work.models import IntentEvidence, SessionReport
from ..work.narrative import NarrativeStatement, WorkNarrative
from ..work.observations import (
    RepositoryObservation,
    RepositoryTarget,
    SessionObservation,
)
from ..work.serialization import report_from_data
from ..work.validation import repository_identity
from ..work.views import WorkSubject, collection_targets, connected_work
from .migrations import migrate
from .transactions import transaction
from .work import SQLiteWorkRepository

log = get_logger("pet-daemon")


@dataclass(frozen=True)
class HistoryRecord:
    last_selected: float
    considered_revision: str | None


@dataclass(frozen=True)
class EventRecord:
    id: int
    session: str
    working_directory: str
    repository: str
    event_type: str
    details: HistoryDetails
    goal: str
    timestamp: float


@dataclass(frozen=True)
class ProjectRecord:
    repo: str
    goal: str
    last_request: str
    timestamp: float


@dataclass(frozen=True)
class ContextRecord:
    initial_request: str
    summary: str
    recent_requests: list[str]
    request_count: int
    first_summary_attempted: bool


@dataclass(frozen=True)
class MessageOutcome:
    message_id: str
    outcome: DeliveryStatus
    text: str
    reason: SpeechReason
    subject_id: str | None
    revision: str | None
    timestamp: float


@dataclass(frozen=True)
class SpeechRecord:
    text: str
    reason: SpeechReason
    timestamp: float


class StoredStatement(TypedDict):
    text: str
    sources: list[str]


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
        self.db.row_factory = sqlite3.Row
        try:
            migrate(self.db)
        except BaseException:
            self.db.close()
            raise
        path.chmod(0o600)
        self._work_repository = SQLiteWorkRepository(self.db, pet)
        self.work = WorkIngestion(
            self._work_repository,
            clock,
        )
        self.prune()

    def session_aliases(self) -> dict[str, str]:
        return self._work_repository.aliases()

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

    def load_narrative(self, subject_id: str) -> WorkNarrative | None:
        row = self.db.execute(
            "SELECT data FROM work_narratives WHERE pet=? AND subject=?",
            (self.pet, subject_id),
        ).fetchone()
        if row is None:
            return None
        data = json.loads(row["data"])

        def statement(value: StoredStatement) -> NarrativeStatement:
            return NarrativeStatement(value["text"], tuple(value["sources"]))

        return WorkNarrative(
            subject_id,
            data["revision"],
            statement(data["intent"]),
            tuple(statement(value) for value in data["decisions"]),
            tuple(statement(value) for value in data["progress"]),
            data["sources"],
        )

    def save_narrative(self, narrative: WorkNarrative) -> None:
        with self.transaction():
            self.db.execute(
                """INSERT INTO work_narratives VALUES (?,?,?,?)
                ON CONFLICT(pet,subject) DO UPDATE SET
                data=excluded.data,updated_at=excluded.updated_at""",
                (
                    self.pet,
                    narrative.subject_id,
                    json.dumps(asdict(narrative), ensure_ascii=False),
                    self.clock(),
                ),
            )

    def narrative_reports(self, sessions: Sequence[str]) -> tuple[SessionReport, ...]:
        aliases = self.session_aliases()
        identities = {aliases.get(identifier, identifier) for identifier in sessions}
        identities.update(
            alias for alias, target in aliases.items() if target in identities
        )
        if not identities:
            return ()
        placeholders = ",".join("?" for _ in identities)
        rows = self.db.execute(
            f"SELECT data FROM work_reports WHERE pet=? AND session IN ({placeholders}) ORDER BY observed_at,id",
            (self.pet, *sorted(identities)),
        )
        return tuple(report_from_data(json.loads(row["data"])) for row in rows)

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
        reason: SpeechReason,
        subject_id: str | None = None,
        revision: str | None = None,
        last_selected: float = 0,
        considered_revision: str | None = None,
    ) -> bool:
        with self.transaction():
            previous = self.db.execute(
                "SELECT outcome FROM message_outcomes WHERE pet=? AND message_id=?",
                (self.pet, message_id),
            ).fetchone()
            if previous and previous["outcome"] == DeliveryStatus.ACCEPTED:
                return False
            timestamp = self.clock()
            self.db.execute(
                """INSERT INTO message_outcomes
                (pet,message_id,outcome,text,reason,subject_id,revision,timestamp)
                VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(pet,message_id) DO UPDATE SET
                outcome=excluded.outcome,text=excluded.text,reason=excluded.reason,
                subject_id=excluded.subject_id,revision=excluded.revision,timestamp=excluded.timestamp""",
                (
                    self.pet,
                    message_id,
                    DeliveryStatus.ACCEPTED,
                    text,
                    reason,
                    subject_id,
                    revision,
                    timestamp,
                ),
            )
            self.db.execute(
                "INSERT INTO speech(pet,text,reason,timestamp) VALUES (?,?,?,?)",
                (self.pet, text, reason, timestamp),
            )
            if subject_id is not None and considered_revision is not None:
                self.save_attention_history(
                    subject_id, last_selected, considered_revision
                )
            self.prune()
        return True

    def record_delivery_failure(
        self,
        message_id: str,
        reason: SpeechReason,
        text: str = "",
        subject_id: str | None = None,
        revision: str | None = None,
        last_selected: float = 0,
        considered_revision: str | None = None,
    ) -> bool:
        with self.transaction():
            result = self.db.execute(
                """INSERT OR IGNORE INTO message_outcomes
                (pet,message_id,outcome,text,reason,subject_id,revision,timestamp)
                VALUES (?,?,?,?,?,?,?,?)""",
                (
                    self.pet,
                    message_id,
                    DeliveryStatus.FAILED,
                    text,
                    reason,
                    subject_id,
                    revision,
                    self.clock(),
                ),
            )
            if (
                result.rowcount
                and subject_id is not None
                and considered_revision is not None
            ):
                self.save_attention_history(
                    subject_id, last_selected, considered_revision
                )
        return bool(result.rowcount)

    def message_outcome(self, message_id: str) -> MessageOutcome | None:
        row = self.db.execute(
            "SELECT message_id,outcome,text,reason,subject_id,revision,timestamp FROM message_outcomes WHERE pet=? AND message_id=?",
            (self.pet, message_id),
        ).fetchone()
        return (
            MessageOutcome(
                message_id=row["message_id"],
                outcome=DeliveryStatus(row["outcome"]),
                text=row["text"],
                reason=SpeechReason(row["reason"]),
                subject_id=row["subject_id"],
                revision=row["revision"],
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
                "DELETE FROM events WHERE pet=? AND timestamp < ?",
                (self.pet, cutoff),
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
            self.db.execute(
                "DELETE FROM work_narratives WHERE pet=? AND updated_at < ?",
                (self.pet, cutoff),
            )
        self.db.execute("PRAGMA incremental_vacuum(64)")

    def record_event(self, record: SessionEntry | None) -> int | None:
        if record is None:
            return None
        details = json.dumps(asdict(record.details), ensure_ascii=False)
        if record.event_type in {"progress", "waiting", "session_start"}:
            previous = self.db.execute(
                "SELECT event_type,details FROM events WHERE pet=? AND session=? AND working_directory=? ORDER BY id DESC LIMIT 1",
                (self.pet, record.session, record.working_directory),
            ).fetchone()
            if (
                previous
                and previous["event_type"] == record.event_type
                and previous["details"] == details
            ):
                return None
        checkout = (
            self._work_repository.get_checkout(record.working_directory, enclosing=True)
            if record.working_directory
            else None
        )
        repository = repository_identity(
            record.repository
            or (checkout.local.repository if checkout and checkout.local else "")
        )
        timestamp = self.clock()
        with self.transaction():
            cursor = self.db.execute(
                "INSERT INTO events(pet,session,working_directory,repository,event_type,details,goal,timestamp) VALUES (?,?,?,?,?,?,?,?)",
                (
                    self.pet,
                    record.session,
                    record.working_directory,
                    repository,
                    record.event_type,
                    details,
                    record.goal,
                    timestamp,
                ),
            )
            if record.working_directory and record.explicit_goal:
                self.db.execute(
                    """INSERT INTO projects VALUES (?,?,?,?,?) ON CONFLICT(pet,repo) DO UPDATE SET
                    goal=excluded.goal, last_request=excluded.last_request, timestamp=excluded.timestamp""",
                    (
                        self.pet,
                        record.working_directory,
                        record.explicit_goal,
                        record.explicit_goal,
                        timestamp,
                    ),
                )
        self.prune()
        log.debug(
            "memory.event",
            detail="payload",
            id=cursor.lastrowid,
            event_type=record.event_type,
        )
        return cursor.lastrowid

    def recent(
        self,
        limit: int = 5,
        *,
        session: str | None = None,
        working_directory: str | None = None,
        event_types: Iterable[str] | None = None,
    ) -> list[EventRecord]:
        filters = ["pet=?"]
        values = [self.pet]
        if session is not None:
            filters.append("session=?")
            values.append(session)
        if working_directory is not None:
            filters.append("working_directory=?")
            values.append(working_directory)
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
        ) SELECT id, session, working_directory, repository, event_type, details, goal, timestamp
          FROM ranked ORDER BY rank=1 DESC, id DESC LIMIT ?""",
            (*values, limit),
        ).fetchall()
        return [
            EventRecord(
                id=row["id"],
                session=row["session"],
                working_directory=row["working_directory"],
                repository=row["repository"],
                event_type=row["event_type"],
                details=HistoryDetails.from_data(json.loads(row["details"])),
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
        placeholders = ",".join("?" for _ in excluded)
        rows = self.db.execute(
            f"""SELECT e.* FROM events_fts JOIN events e ON e.id=events_fts.rowid
            WHERE events_fts MATCH ? AND e.pet=? AND e.repository=? AND e.id NOT IN ({placeholders})
            ORDER BY e.session=? DESC, bm25(events_fts), e.id DESC LIMIT ?""",
            (query, self.pet, request.repository, *excluded, request.session, limit),
        ).fetchall()
        return [
            EventRecord(
                id=row["id"],
                session=row["session"],
                working_directory=row["working_directory"],
                repository=row["repository"],
                event_type=row["event_type"],
                details=HistoryDetails.from_data(json.loads(row["details"])),
                goal=row["goal"],
                timestamp=row["timestamp"],
            )
            for row in rows
        ]

    def projects(self, repos: Iterable[str]) -> list[ProjectRecord]:
        result = []
        for repo in list(dict.fromkeys(repos))[:3]:
            row = self.db.execute(
                "SELECT repo,goal,last_request,timestamp FROM projects WHERE pet=? AND repo=?",
                (self.pet, repo),
            ).fetchone()
            if row:
                result.append(
                    ProjectRecord(
                        repo=row["repo"],
                        goal=row["goal"],
                        last_request=row["last_request"],
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
            initial_request=row["first_prompt"],
            summary=row["session_context"],
            recent_requests=json.loads(row["recent_requests"]),
            request_count=row["request_count"],
            first_summary_attempted=bool(row["first_summary_attempted"]),
        )

    def observe_session_context(self, observation: SessionObservation) -> None:
        stored = self.context(observation.session)
        first = stored.initial_request if stored else ""
        summary = stored.summary if stored else ""
        requests = list(stored.recent_requests) if stored else []
        count = stored.request_count if stored else 0
        attempted = stored.first_summary_attempted if stored else False
        for alias in observation.aliases:
            other = self.context(alias)
            if other is None:
                continue
            first = first if count else other.initial_request or first
            summary = summary or other.summary
            requests = [*requests, *other.recent_requests][-4:]
            count += other.request_count
            attempted = attempted or other.first_summary_attempted
            self.db.execute(
                "DELETE FROM contexts WHERE pet=? AND session=?", (self.pet, alias)
            )
        for report in observation.reports:
            if isinstance(report, IntentEvidence):
                request = select_prompt(report.text)
                first = first if count else request
                requests = [*requests, request][-4:]
                count += 1
        if not first and observation.parent:
            parent = self.context(observation.parent)
            if parent:
                first = parent.initial_request
                summary = parent.summary
        if first:
            self.save_context(
                observation.session, first, summary, requests, count, attempted
            )

    def save_context(
        self,
        session: str,
        initial_request: str,
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
                    select_prompt(initial_request),
                    summary or "",
                    json.dumps(requests),
                    self.clock(),
                    count,
                    first_summary_attempted,
                ),
            )
        self.prune()

    def record_speech(self, text: str, reason: SpeechReason) -> None:
        with self.transaction():
            self.db.execute(
                "INSERT INTO speech(pet,text,reason,timestamp) VALUES (?,?,?,?)",
                (self.pet, text, reason, self.clock()),
            )
        self.prune()

    def connected_work(self, activity: SwarmActivity) -> tuple[WorkSubject, ...]:
        return connected_work(self._work_repository.load(), activity)

    def collection_targets(self) -> tuple[RepositoryTarget, ...]:
        return collection_targets(self._work_repository.load())

    def load_repositories(self) -> tuple[RepositoryObservation, ...]:
        state = self._work_repository.load()
        return (
            *state.checkouts.values(),
            *state.branches.values(),
            *state.pull_requests.values(),
        )

    def speech(self, limit: int = 3) -> list[SpeechRecord]:
        rows = self.db.execute(
            "SELECT text,reason,timestamp FROM speech WHERE pet=? AND timestamp > ? ORDER BY id DESC LIMIT ?",
            (self.pet, self.clock() - self.speech_history_seconds, limit),
        ).fetchall()
        return [
            SpeechRecord(
                text=row["text"],
                reason=SpeechReason(row["reason"]),
                timestamp=row["timestamp"],
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
                "attention_history",
                "message_outcomes",
                "work_narratives",
            ):
                self.db.execute(f"DELETE FROM {table} WHERE pet=?", (self.pet,))

    def close(self) -> None:
        self.db.close()
