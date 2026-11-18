import hashlib
import json
import os
import sqlite3
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from .domain import SessionSource
from .identity import SessionIdentity

SCHEMA_VERSION = 1
NOTE_LIMIT = 16
SESSION_NOTE_LIMIT = 256

type Identifier = Annotated[
    str, StringConstraints(min_length=1, max_length=160, pattern=r"^\S+$")
]
type Line = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True, min_length=1, max_length=240, pattern=r"^[^\r\n]+$"
    ),
]


class SessionRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    source: SessionSource
    session_id: Identifier

    def identity(self) -> SessionIdentity:
        return SessionIdentity(source=self.source, native_id=self.session_id)


class Actor(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: Line
    description: Line | None = None


class Assignment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["assignment"]
    session: SessionRef
    actor: Actor
    acting_for: Actor | None = None


class Launch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["launch"]
    session: SessionRef
    by: Actor


class Wait(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["wait"]
    actor: Actor
    on: Actor
    detail: Line


class Handoff(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["handoff"]
    sender: Actor
    to: Actor
    work: Line


type WorkstreamNote = Annotated[
    Assignment | Launch | Wait | Handoff, Field(discriminator="kind")
]
type PresenceStatus = Literal["available", "away"]


class Presence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    status: PresenceStatus
    observed_at: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    renderer_event_id: Identifier | None = None
    session: SessionRef | None = None
    turn: Identifier | None = None


class MarkStatus(StrEnum):
    ACTIVE = "active"
    PAUSED = "paused"
    BLOCKED = "blocked"
    WAITING_REVIEW = "waiting-review"
    DONE = "done"
    ABANDONED = "abandoned"


class MarkPriority(StrEnum):
    FOCUS = "focus"
    NORMAL = "normal"
    BACKGROUND = "background"


type MarkField = Literal[
    "status",
    "priority",
    "tags",
    "note",
    "goal",
    "title",
    "blocked_on",
    "snooze_until",
    "notes",
]
MARK_FIELDS: tuple[MarkField, ...] = (
    "status",
    "priority",
    "tags",
    "note",
    "goal",
    "title",
    "blocked_on",
    "snooze_until",
    "notes",
)


class MarkPatch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    status: MarkStatus | None = None
    priority: MarkPriority | None = None
    tags: tuple[Annotated[str, Field(min_length=1, max_length=64)], ...] | None = Field(
        default=None, max_length=64
    )
    note: str | None = Field(default=None, max_length=2048)
    notes: tuple[WorkstreamNote, ...] | None = Field(
        default=None, max_length=NOTE_LIMIT
    )
    goal: str | None = Field(default=None, min_length=1, max_length=240)
    title: str | None = Field(default=None, min_length=1, max_length=160)
    blocked_on: str | None = Field(default=None, max_length=512)
    snooze_until: float | None = Field(default=None, ge=0, allow_inf_nan=False)

    @field_validator("goal", "title")
    @classmethod
    def single_line(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value.strip() or "\n" in value or "\r" in value:
            raise ValueError("Mark goal and title require one nonblank line")
        return value.strip()

    @field_validator("tags")
    @classmethod
    def distinct_tags(cls, value: tuple[str, ...] | None) -> tuple[str, ...] | None:
        return tuple(dict.fromkeys(value)) if value is not None else None


class Attribution(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    author: str = Field(min_length=1, max_length=128)
    principal: str = Field(min_length=1, max_length=256)
    reason: str = Field(min_length=1, max_length=2048)

    @field_validator("author", "principal", "reason")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Mark attribution requires nonblank values")
        return value.strip()


def thread_id(key: str) -> str:
    return "thread:" + hashlib.sha256(key.encode()).hexdigest()


def marks_path(memory_file: Path) -> Path:
    return memory_file.with_name(memory_file.name + ".marks.sqlite3")


def event_provenance(event: dict[str, object]) -> dict[str, object]:
    return {
        "basis": "marked",
        "event_id": event["id"],
        "author": event["author"],
        "principal": event["principal"],
        "timestamp": event["timestamp"],
        "reason": event["reason"],
    }


def event_record(row: sqlite3.Row) -> dict[str, object]:
    event = dict(row)
    event.pop("pet")
    event["values"] = json.loads(cast(str, event["values"]))
    event["fields"] = json.loads(cast(str, event["fields"]))
    event["provenance"] = event_provenance(event)
    return event


def effective_marks(events: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    notes: dict[str, dict[str, list[dict[str, object]]]] = {}
    for event in events:
        identifier = cast(str, event["workstream_id"])
        principal = cast(str, event["principal"])
        mark = result.setdefault(
            identifier, {"workstream_id": identifier, "values": {}, "fields": {}}
        )
        values = cast(dict[str, object], mark["values"])
        fields = cast(dict[str, object], mark["fields"])
        provenance = event["provenance"]
        if event["operation"] == "mark":
            for name, value in cast(dict[str, object], event["values"]).items():
                if name == "notes":
                    notes.setdefault(identifier, {})[principal] = [
                        {"value": note, "provenance": provenance}
                        for note in cast(list[object], value)
                    ]
                    continue
                values[name] = value
                fields[name] = {"value": value, "provenance": provenance}
        else:
            for name in cast(list[str], event["fields"]):
                if name == "notes":
                    notes.get(identifier, {}).pop(principal, None)
                    continue
                values.pop(name, None)
                fields.pop(name, None)
        mark["updated_at"] = event["timestamp"]
        mark["provenance"] = provenance
    for identifier, writers in notes.items():
        entries = [entry for batch in writers.values() for entry in batch]
        if entries:
            mark = result[identifier]
            values = cast(dict[str, object], mark["values"])
            fields = cast(dict[str, object], mark["fields"])
            values["notes"] = [entry["value"] for entry in entries]
            fields["notes"] = {"value": values["notes"], "entries": entries}
    return {identifier: mark for identifier, mark in result.items() if mark["values"]}


def is_suppressed(mark: dict[str, object], now: float) -> bool:
    values = cast(dict[str, object], mark.get("values", {}))
    if values.get("status") in (MarkStatus.DONE, MarkStatus.ABANDONED):
        return True
    snooze_until = cast(float | None, values.get("snooze_until"))
    return snooze_until is not None and snooze_until > now


def mark_rank(mark: dict[str, object]) -> int:
    values = cast(dict[str, object], mark.get("values", {}))
    if (
        values.get("priority") == MarkPriority.FOCUS
        or values.get("status") == MarkStatus.BLOCKED
    ):
        return -1
    if (
        values.get("priority") == MarkPriority.BACKGROUND
        or values.get("status") == MarkStatus.PAUSED
    ):
        return 1
    return 0


class MarkStore:
    def __init__(
        self, memory_file: Path, pet: str, clock: Callable[[], float] = time.time
    ) -> None:
        self.path = marks_path(memory_file).resolve()
        self.pet = pet
        self.clock = clock

    @contextmanager
    def database(self, *, write: bool = False) -> Generator[sqlite3.Connection]:
        if write:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(
                self.path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600
            )
            os.close(descriptor)
        db = sqlite3.connect(
            self.path.as_uri() + ("?mode=rw" if write else "?mode=ro"), uri=True
        )
        db.row_factory = sqlite3.Row
        try:
            if write:
                db.execute("BEGIN IMMEDIATE")
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version > SCHEMA_VERSION:
                raise ValueError(
                    "Workstream mark database schema is newer than supported"
                )
            if version == 0 and write:
                self.initialize(db)
            elif version != SCHEMA_VERSION:
                raise ValueError("Workstream mark database schema is unsupported")
            if write:
                self.ensure_indexes(db)
                os.chmod(self.path, 0o600)
            yield db
            if write:
                db.commit()
        except BaseException:
            if write:
                db.rollback()
            raise
        finally:
            db.close()

    def initialize(self, db: sqlite3.Connection) -> None:
        db.execute("""CREATE TABLE mark_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pet TEXT NOT NULL, workstream_id TEXT NOT NULL,
            operation TEXT NOT NULL, author TEXT NOT NULL, principal TEXT NOT NULL,
            timestamp REAL NOT NULL, reason TEXT NOT NULL,
            \"values\" TEXT NOT NULL, fields TEXT NOT NULL
        )""")
        db.execute("CREATE INDEX marks_thread ON mark_events(pet,workstream_id,id)")
        db.execute("""CREATE TRIGGER marks_no_update BEFORE UPDATE ON mark_events
            BEGIN SELECT RAISE(ABORT, 'Workstream marks are append-only'); END""")
        db.execute("""CREATE TRIGGER marks_no_delete BEFORE DELETE ON mark_events
            BEGIN SELECT RAISE(ABORT, 'Workstream marks are append-only'); END""")
        db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    @staticmethod
    def ensure_indexes(db: sqlite3.Connection) -> None:
        db.execute("""CREATE INDEX IF NOT EXISTS marks_notes
            ON mark_events(pet,id) WHERE json_type("values",'$.notes') IS NOT NULL
               OR (operation='unmark' AND instr(fields,'"notes"')>0)""")
        db.execute(
            "CREATE INDEX IF NOT EXISTS marks_changed ON mark_events(pet,timestamp,id)"
        )
        db.execute("""CREATE INDEX IF NOT EXISTS marks_presence
            ON mark_events(pet,id) WHERE operation='presence'""")

    def record_presence(
        self,
        status: PresenceStatus,
        author: str,
        reason: str,
        principal: str = "stdio",
        *,
        session: SessionRef | None = None,
        turn: str | None = None,
        renderer_event_id: str | None = None,
    ) -> dict[str, object]:
        if turn is not None and session is None:
            raise ValueError("A prompt turn requires its session")
        if status == "away" and (renderer_event_id is not None or session is not None):
            raise ValueError("An away observation has no renderer event or session")
        values = Presence(
            status=status,
            observed_at=self.clock(),
            session=session,
            turn=turn,
            renderer_event_id=renderer_event_id,
        ).model_dump(mode="json", exclude_none=True)
        return self._append(
            "",
            "presence",
            values,
            [],
            Attribution(author=author, reason=reason, principal=principal),
        )

    def latest_presence(self) -> dict[str, object] | None:
        if not self.path.exists():
            return None
        with self.database() as db:
            row = db.execute(
                """SELECT * FROM mark_events WHERE pet=? AND operation='presence'
                ORDER BY id DESC LIMIT 1""",
                (self.pet,),
            ).fetchone()
        return event_record(row) if row else None

    def changes(self, after: float, limit: int) -> list[dict[str, object]]:
        if not self.path.exists():
            return []
        with self.database() as db:
            rows = db.execute(
                """SELECT * FROM mark_events WHERE pet=? AND timestamp>?
                ORDER BY timestamp DESC,id DESC LIMIT ?""",
                (self.pet, after, limit),
            )
            return [event_record(row) for row in rows]

    def note_updates(self, after: int | None) -> list[dict[str, object]]:
        if not self.path.exists():
            return []
        with self.database() as db:
            if after is None:
                rows = db.execute(
                    """WITH note_events AS (
                        SELECT id,workstream_id,principal FROM mark_events
                        WHERE pet=? AND (json_type("values",'$.notes') IS NOT NULL
                           OR (operation='unmark' AND instr(fields,'"notes"')>0))
                    ), recent_workstreams AS (
                        SELECT workstream_id FROM note_events
                        GROUP BY workstream_id ORDER BY MAX(id) DESC LIMIT ?
                    )
                    SELECT * FROM mark_events WHERE id IN (
                        SELECT MAX(id) FROM note_events
                        WHERE workstream_id IN (SELECT workstream_id FROM recent_workstreams)
                        GROUP BY workstream_id,principal
                    ) ORDER BY id""",
                    (self.pet, SESSION_NOTE_LIMIT),
                )
            else:
                rows = db.execute(
                    """SELECT * FROM mark_events WHERE pet=? AND id>?
                        AND (json_type("values",'$.notes') IS NOT NULL
                           OR (operation='unmark' AND instr(fields,'"notes"')>0))
                    ORDER BY id LIMIT ?""",
                    (self.pet, after, SESSION_NOTE_LIMIT),
                )
            return [event_record(row) for row in rows]

    def append(
        self,
        id: str,
        patch: MarkPatch,
        author: str,
        reason: str,
        principal: str = "stdio",
    ) -> dict[str, object]:
        values = patch.model_dump(mode="json", exclude_none=True)
        if not values:
            raise ValueError("A workstream mark requires at least one field")
        return self._append(
            id,
            "mark",
            values,
            [],
            Attribution(author=author, reason=reason, principal=principal),
        )

    def unmark(
        self,
        id: str,
        author: str,
        reason: str,
        fields: tuple[MarkField, ...] | list[MarkField] | None = None,
        principal: str = "stdio",
    ) -> dict[str, object]:
        selected = list(MARK_FIELDS if fields is None else fields)
        if not selected or any(field not in MARK_FIELDS for field in selected):
            raise ValueError("Unmark requires recognized fields")
        return self._append(
            id,
            "unmark",
            {},
            list(dict.fromkeys(selected)),
            Attribution(author=author, reason=reason, principal=principal),
        )

    def _append(
        self,
        id: str,
        operation: str,
        values: dict[str, object],
        fields: list[str],
        attribution: Attribution,
    ) -> dict[str, object]:
        with self.database(write=True) as db:
            return self.insert_mark(db, id, operation, values, fields, attribution)

    def insert_mark(
        self,
        db: sqlite3.Connection,
        id: str,
        operation: str,
        values: dict[str, object],
        fields: list[str],
        attribution: Attribution,
    ) -> dict[str, object]:
        cursor = db.execute(
            'INSERT INTO mark_events(pet,workstream_id,operation,author,principal,timestamp,reason,"values",fields) VALUES (?,?,?,?,?,?,?,?,?)',
            (
                self.pet,
                id,
                operation,
                attribution.author,
                attribution.principal,
                self.clock(),
                attribution.reason,
                json.dumps(values),
                json.dumps(fields),
            ),
        )
        row = db.execute(
            "SELECT * FROM mark_events WHERE id=?", (cursor.lastrowid,)
        ).fetchone()
        assert row is not None
        return event_record(row)

    def effective(self, id: str) -> dict[str, object]:
        if not self.path.exists():
            return {}
        with self.database() as db:
            rows = db.execute(
                "SELECT * FROM mark_events WHERE pet=? AND workstream_id=? ORDER BY id",
                (self.pet, id),
            )
            marks = effective_marks([event_record(row) for row in rows])
        return marks.get(id, {})

    def all_effective(self) -> dict[str, dict[str, object]]:
        if not self.path.exists():
            return {}
        with self.database() as db:
            rows = db.execute(
                "SELECT * FROM mark_events WHERE pet=? AND operation!='presence' ORDER BY id",
                (self.pet,),
            )
            return effective_marks([event_record(row) for row in rows])

    def history(
        self, id: str, limit: int = 20, before: int | None = None
    ) -> list[dict[str, object]]:
        if not self.path.exists():
            return []
        with self.database() as db:
            rows = db.execute(
                "SELECT * FROM mark_events WHERE pet=? AND workstream_id=? AND (? IS NULL OR id<?) ORDER BY id DESC LIMIT ?",
                (self.pet, id, before, before, limit),
            )
            return [event_record(row) for row in rows]
