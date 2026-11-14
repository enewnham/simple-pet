import asyncio
import contextlib
import hashlib
import json
import os
import re
import signal
import sqlite3
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .logging import get_logger
from .runtime import HookEvent, WaitingKind
from .workstream_marks import MarkPriority, MarkStatus

log = get_logger("pet-daemon")
type NotificationKind = Literal[
    "turn_done", "turn_error", "session_end", "permission_pending"
]


class MarkFilter(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    status: MarkStatus | None = None
    priority: MarkPriority | None = None
    tags: tuple[str, ...] = ()


class NotificationFilter(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    cwd: str | None = None
    zmx_session: str | None = None
    labels: dict[str, str] = Field(default_factory=dict)
    needs_attention: bool = False
    marks: MarkFilter = Field(default_factory=MarkFilter)


class NotifyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    command: tuple[str, ...]
    events: tuple[NotificationKind, ...] = (
        "turn_done",
        "session_end",
        "permission_pending",
    )
    filter: NotificationFilter = Field(default_factory=NotificationFilter)
    rules: tuple[NotificationFilter, ...] | None = None
    debounce_seconds: float = Field(default=10, ge=0, allow_inf_nan=False)
    timeout_seconds: float = Field(default=10, gt=0, allow_inf_nan=False)
    permission_grace_seconds: float = Field(default=5, ge=0, allow_inf_nan=False)
    max_events: int = Field(default=100, gt=0, le=1000)
    max_attempts: int = Field(default=8, gt=0, le=100)

    @field_validator("command")
    @classmethod
    def valid_command(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or any(not item or "\0" in item for item in value):
            raise ValueError("notify.command requires nonempty argv without NUL")
        return value


SECRET_PATTERNS = (
    r"-----BEGIN [^-]*PRIVATE KEY-----[\s\S]*?(?:-----END [^-]*PRIVATE KEY-----|$)",
    r"(?i)\b(?:authorization\s*:\s*(?:bearer|basic)\s+)[^\s,;]+",
    r"(?i)\b(?:[\w-]*(?:token|password|passwd|secret|api[_-]?key))[\"']?\s*[=:]\s*[\"']?[^\s\"',;}]+",
    r"\b(?:sk-[A-Za-z0-9_-]{12,}|gh[pousr]_[A-Za-z0-9_]{16,}|github_pat_[A-Za-z0-9_]+|AKIA[A-Z0-9]{16})\b",
    r"https?://[^\s/@]+:[^\s/@]+@",
)


def redact(text: str) -> str:
    for pattern in SECRET_PATTERNS:
        text = re.sub(pattern, "[REDACTED]", text)
    return text


def bounded(text: str | None) -> str:
    return redact(text or "").encode("utf-8")[:2048].decode("utf-8", errors="ignore")


def notification_kind(hook: HookEvent) -> NotificationKind | None:
    if hook.event == "waiting" and hook.waiting_kind in (
        WaitingKind.PERMISSION,
        WaitingKind.PERMISSION_CHECK,
    ):
        return "permission_pending"
    if hook.event in ("turn_done", "turn_error", "session_end"):
        return cast(NotificationKind, hook.event.value)
    return None


def matches(
    config: NotifyConfig,
    hook: HookEvent,
    kind: NotificationKind,
    marks: dict[str, object] | None = None,
) -> bool:
    if kind not in config.events:
        return False
    policies = config.rules if config.rules is not None else (config.filter,)
    return any(matches_filter(policy, hook, kind, marks) for policy in policies)


def matches_filter(
    policy: NotificationFilter,
    hook: HookEvent,
    kind: NotificationKind,
    marks: dict[str, object] | None = None,
) -> bool:
    if not matches_marks(policy.marks, marks or {}):
        return False
    if policy.cwd is not None and not fnmatchcase(hook.repo or "", policy.cwd):
        return False
    if policy.zmx_session is not None and not fnmatchcase(
        hook.zmx_session or "", policy.zmx_session
    ):
        return False
    if any(
        not fnmatchcase(hook.labels.get(key, ""), value)
        for key, value in policy.labels.items()
    ):
        return False
    attention = kind in ("permission_pending", "turn_error") or (
        kind == "turn_done"
        and re.search(
            r"\b(error|failed|failure|exception|unable|blocked)\b",
            hook.progress or "",
            re.IGNORECASE,
        )
        is not None
    )
    return not policy.needs_attention or attention


def matches_marks(policy: MarkFilter, marks: dict[str, object]) -> bool:
    if policy.status is not None and marks.get("status") != policy.status:
        return False
    if policy.priority is not None and marks.get("priority") != policy.priority:
        return False
    tags = cast(list[str], marks.get("tags", []))
    return all(
        any(fnmatchcase(tag, pattern) for tag in tags) for pattern in policy.tags
    )


def notification_marks(marks: dict[str, object]) -> dict[str, object]:
    values = {name: marks[name] for name in ("status", "priority") if name in marks}
    if "tags" in marks:
        tags = cast(list[str], marks["tags"])
        values["tags"] = [bounded(tag) for tag in tags[:32]]
    return values


async def run_command(command: tuple[str, ...], payload: bytes, timeout: float) -> bool:
    process = await asyncio.create_subprocess_exec(
        *command,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
        env={"PATH": "/usr/bin:/bin:/usr/sbin:/sbin"},
        start_new_session=True,
    )
    try:
        async with asyncio.timeout(timeout):
            await process.communicate(payload)
        return process.returncode == 0
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        await process.wait()


RETRY_BASE_SECONDS = 1
RETRY_MAX_SECONDS = 15


@dataclass
class PermissionRequest:
    hook: HookEvent
    token: str
    queued: bool = False


def same_request(left: HookEvent, right: HookEvent) -> bool:
    if left.turn is not None and right.turn is not None and left.turn != right.turn:
        return False
    return left.call == right.call


def resumes_permission(request: HookEvent, hook: HookEvent) -> bool:
    if hook.event in (
        "intent",
        "session_start",
        "session_end",
        "turn_done",
        "turn_error",
        "interrupt",
        "subagent_done",
    ):
        return True
    if hook.event == "waiting" and hook.waiting_kind == WaitingKind.QUESTION:
        return True
    if hook.event not in ("tool_start", "tool_done", "tool_error"):
        return False
    if request.call is None:
        return request.turn is None or hook.turn is None or request.turn == hook.turn
    return same_request(request, hook)


def request_hook(hook: HookEvent) -> HookEvent:
    return HookEvent(
        source=hook.source,
        session=hook.session,
        event="waiting",
        call=hook.call,
        turn=hook.turn,
        waiting_kind=hook.waiting_kind,
    )


class Notifications:
    def __init__(
        self,
        config: NotifyConfig,
        path: Path,
        clock: Callable[[], float] = time.time,
        *,
        mark_lookup: Callable[[HookEvent], dict[str, object]] | None = None,
        permission_pending: Callable[[HookEvent], bool] | None = None,
    ) -> None:
        self.config = config
        self.clock = clock
        self.mark_lookup = mark_lookup
        self.permission_pending = permission_pending
        self.permissions: dict[str, PermissionRequest] = {}
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        version = self.db.execute("PRAGMA user_version").fetchone()[0]
        if version > 1:
            self.db.close()
            raise ValueError("Notification database schema is newer than supported")
        os.chmod(path, 0o600)
        self.db.executescript("""
            PRAGMA user_version = 1;
            CREATE TABLE IF NOT EXISTS outbox (
                id TEXT PRIMARY KEY, session TEXT, payload TEXT, attempts INTEGER,
                due REAL, created REAL
            );
            CREATE TABLE IF NOT EXISTS context (session TEXT PRIMARY KEY, payload TEXT, seen REAL);
            CREATE TABLE IF NOT EXISTS debounce (session TEXT PRIMARY KEY, accepted REAL);
        """)
        self.task: asyncio.Task[None] | None = None

    def ingest(self, hook: HookEvent) -> None:
        session = hashlib.sha256(
            json.dumps([hook.source, hook.session]).encode()
        ).hexdigest()
        hook = self.remember(session, hook)
        kind = notification_kind(hook)
        if (
            kind == "permission_pending"
            and self.permission_pending is not None
            and not self.permission_pending(request_hook(hook))
        ):
            return
        self.update_permission(session, hook, kind)
        if kind is None:
            return
        marks = self.mark_lookup(hook) if self.mark_lookup is not None else {}
        if not matches(self.config, hook, kind, marks):
            return
        request = (
            self.permissions.get(session) if kind == "permission_pending" else None
        )
        if request is not None and request.queued:
            return
        now = self.clock()
        debounce_key = f"{session}:{kind}"
        if request is None and self.is_debounced(debounce_key, now):
            return
        payload: dict[str, object] = {
            "id": uuid.uuid4().hex,
            "event": kind,
            "harness": hook.source,
            "session": bounded(hook.session),
            "zmx_session": bounded(hook.zmx_session),
            "cwd": bounded(hook.repo),
            "tab_title": bounded(hook.tab_title),
            "timestamp": hook.capture_at if hook.capture_at is not None else now,
            "received_at": now,
            "last_assistant_message": bounded(hook.progress),
            "workstream_marks": notification_marks(marks),
        }
        due = now
        if request is not None:
            payload["_permission_request"] = request.hook.model_dump(
                mode="json", exclude_none=True
            )
            payload["_permission_token"] = request.token
            due += self.config.permission_grace_seconds
            request.queued = True
        with self.db:
            self.db.execute(
                "DELETE FROM debounce WHERE accepted < ?",
                (now - self.config.debounce_seconds,),
            )
            if request is None:
                self.db.execute(
                    "INSERT OR REPLACE INTO debounce VALUES (?, ?)", (debounce_key, now)
                )
            self.db.execute(
                "INSERT INTO outbox VALUES (?, ?, ?, 0, ?, ?)",
                (payload["id"], session, json.dumps(payload), due, now),
            )
            self.db.execute(
                "DELETE FROM outbox WHERE id IN (SELECT id FROM outbox ORDER BY created DESC, rowid DESC LIMIT -1 OFFSET ?)",
                (self.config.max_events,),
            )

    def is_debounced(self, key: str, now: float) -> bool:
        previous = self.db.execute(
            "SELECT accepted FROM debounce WHERE session=?", (key,)
        ).fetchone()
        return previous is not None and now - previous[0] < self.config.debounce_seconds

    def update_permission(
        self, session: str, hook: HookEvent, kind: NotificationKind | None
    ) -> None:
        previous = self.permissions.get(session)
        if kind == "permission_pending":
            if previous is not None and same_request(previous.hook, hook):
                if (
                    previous.hook.waiting_kind == WaitingKind.PERMISSION_CHECK
                    and hook.waiting_kind == WaitingKind.PERMISSION
                ):
                    previous.hook = previous.hook.model_copy(
                        update={"waiting_kind": WaitingKind.PERMISSION}
                    )
                    with self.db:
                        self.db.execute(
                            "UPDATE outbox SET payload=json_set(payload,'$._permission_request.waiting_kind',?) "
                            "WHERE session=? AND json_extract(payload,'$._permission_token')=?",
                            (WaitingKind.PERMISSION, session, previous.token),
                        )
                return
            self.clear_permission(session)
            self.permissions[session] = PermissionRequest(
                request_hook(hook), uuid.uuid4().hex
            )
            if len(self.permissions) > 1000:
                self.clear_permission(next(iter(self.permissions)))
        elif previous is not None and resumes_permission(previous.hook, hook):
            if self.permission_pending is None or not self.permission_pending(
                previous.hook
            ):
                self.clear_permission(session)

    def clear_permission(self, session: str) -> None:
        self.permissions.pop(session, None)
        with self.db:
            self.db.execute(
                "DELETE FROM outbox WHERE session=? AND json_extract(payload,'$.event')='permission_pending'",
                (session,),
            )

    def is_permission_pending(self, session: str, payload: dict[str, object]) -> bool:
        raw = payload.get("_permission_request")
        if not isinstance(raw, dict):
            return False
        hook = HookEvent.model_validate(raw)
        current = self.permissions.get(session)
        if current is not None and current.token != payload.get("_permission_token"):
            return False
        if self.permission_pending is not None:
            return self.permission_pending(hook)
        return current is not None and same_request(current.hook, hook)

    def delivery_payload(self, payload: dict[str, object], attempt: int) -> bytes:
        data = {
            name: value for name, value in payload.items() if not name.startswith("_")
        }
        now = self.clock()
        data["elapsed_seconds"] = max(0, now - cast(float, payload["timestamp"]))
        data["delivery_attempt_at"] = now
        data["delivery_attempt"] = attempt
        if data["event"] == "permission_pending":
            request = cast(dict[str, object], payload["_permission_request"])
            data["needs_human"] = True
            data["permission_confirmation"] = (
                "explicit_permission_prompt"
                if request["waiting_kind"] == WaitingKind.PERMISSION
                else "still_pending_after_grace"
            )
        return json.dumps(data).encode()

    def remember(self, session: str, hook: HookEvent) -> HookEvent:
        row = self.db.execute(
            "SELECT payload FROM context WHERE session=?", (session,)
        ).fetchone()
        values = json.loads(row[0]) if row else {}
        for name in ("repo", "zmx_session", "tab_title"):
            value = getattr(hook, name)
            if value is not None:
                values[name] = bounded(value)
        if hook.labels:
            values["labels"] = {
                bounded(k): bounded(v) for k, v in list(hook.labels.items())[:32]
            }
        if hook.event in ("intent", "session_start"):
            values.pop("progress", None)
        if hook.progress is not None and hook.event in (
            "progress",
            "turn_done",
            "turn_error",
        ):
            values["progress"] = bounded(hook.progress)
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO context VALUES (?, ?, ?)",
                (session, json.dumps(values), self.clock()),
            )
            self.db.execute(
                "DELETE FROM context WHERE session IN (SELECT session FROM context ORDER BY seen DESC LIMIT -1 OFFSET 1000)"
            )
        return hook.model_copy(update={**values, "progress": values.get("progress")})

    def tick(self) -> None:
        if self.task is None or self.task.done():
            self.task = asyncio.create_task(self.deliver())

    async def deliver(self) -> None:
        row = self.db.execute(
            "SELECT id, session, payload, attempts FROM outbox WHERE due<=? "
            "ORDER BY (attempts>0), due, created, rowid LIMIT 1",
            (self.clock(),),
        ).fetchone()
        if row is None:
            return
        identity, session, stored, attempts = row
        payload = json.loads(stored)
        if payload["event"] == "permission_pending" and not self.is_permission_pending(
            session, payload
        ):
            with self.db:
                self.db.execute("DELETE FROM outbox WHERE id=?", (identity,))
            return
        try:
            success = await run_command(
                self.config.command,
                self.delivery_payload(payload, attempts + 1),
                self.config.timeout_seconds,
            )
        except OSError, TimeoutError:
            success = False
        attempts += 1
        exhausted = attempts >= self.config.max_attempts
        with self.db:
            if success or exhausted:
                self.db.execute("DELETE FROM outbox WHERE id=?", (identity,))
            else:
                self.db.execute(
                    "UPDATE outbox SET attempts=?, due=? WHERE id=?",
                    (
                        attempts,
                        self.clock()
                        + min(
                            RETRY_MAX_SECONDS, RETRY_BASE_SECONDS * 2 ** (attempts - 1)
                        ),
                        identity,
                    ),
                )
        log.info(
            "notify.outcome",
            event_id=identity,
            success=success,
            attempts=attempts,
            exhausted=exhausted,
        )

    async def close(self) -> None:
        if self.task is not None:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
        self.db.close()
