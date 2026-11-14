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

from pydantic import BaseModel, ConfigDict, Field, field_validator

SCHEMA_VERSION = 1


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
    "status", "priority", "tags", "note", "goal", "title", "blocked_on", "snooze_until"
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
)


class MarkPatch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    status: MarkStatus | None = None
    priority: MarkPriority | None = None
    tags: tuple[Annotated[str, Field(min_length=1, max_length=64)], ...] | None = Field(
        default=None, max_length=64
    )
    note: str | None = Field(default=None, max_length=2048)
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
    for event in events:
        identifier = cast(str, event["workstream_id"])
        mark = result.setdefault(
            identifier, {"workstream_id": identifier, "values": {}, "fields": {}}
        )
        values = cast(dict[str, object], mark["values"])
        fields = cast(dict[str, object], mark["fields"])
        provenance = event["provenance"]
        if event["operation"] == "mark":
            for name, value in cast(dict[str, object], event["values"]).items():
                values[name] = value
                fields[name] = {"value": value, "provenance": provenance}
        else:
            for name in cast(list[str], event["fields"]):
                values.pop(name, None)
                fields.pop(name, None)
        mark["updated_at"] = event["timestamp"]
        mark["provenance"] = provenance
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
        now = self.clock()
        with self.database(write=True) as db:
            cursor = db.execute(
                'INSERT INTO mark_events(pet,workstream_id,operation,author,principal,timestamp,reason,"values",fields) VALUES (?,?,?,?,?,?,?,?,?)',
                (
                    self.pet,
                    id,
                    operation,
                    attribution.author,
                    attribution.principal,
                    now,
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
                "SELECT * FROM mark_events WHERE pet=? ORDER BY id", (self.pet,)
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
