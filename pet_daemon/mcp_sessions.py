import os
import tempfile
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from .domain import ExecutionOutcome, ProcessStatus
from .identity import SessionIdentity
from .runtime import AgentSessionStatus, FailureSnapshot, WaitingKind

PUBLISH_SECONDS = 5
FRESH_SECONDS = 15


class LiveSession(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: SessionIdentity
    checkout: str
    observed_at: float
    state: Literal["working", "idle", "waiting_on_permission"]
    presence: Literal[ProcessStatus.ALIVE] = ProcessStatus.ALIVE
    presence_checked_at: float
    activity_uncertain: bool
    waiting_kind: WaitingKind | None
    waiting_prompt: str | None
    waiting_tool: str | None
    outcome: ExecutionOutcome | None
    failure: FailureSnapshot | None
    goal: str
    progress: str
    last_activity_at: float


class SessionSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)
    version: Literal[1] = 1
    pet: str
    observed_at: float
    suspended: bool = False
    sessions: tuple[LiveSession, ...] = ()


def snapshot_path(memory_file: Path) -> Path:
    return memory_file.with_name(memory_file.name + ".sessions.json")


def live_session(status: AgentSessionStatus, observed_at: float) -> LiveSession:
    waiting_permission = status.waiting_for_input and status.waiting_kind in (
        WaitingKind.PERMISSION,
        WaitingKind.PERMISSION_CHECK,
    )
    if waiting_permission:
        state = "waiting_on_permission"
    elif status.turn_active and not status.waiting_for_input:
        state = "working"
    else:
        state = "idle"
    assert status.owner_last_alive_seconds is not None
    return LiveSession(
        id=status.id,
        checkout=status.repo,
        observed_at=observed_at,
        state=state,
        presence_checked_at=observed_at - status.owner_last_alive_seconds,
        activity_uncertain=status.activity_uncertain,
        waiting_kind=status.waiting_kind,
        waiting_prompt=status.waiting_prompt,
        waiting_tool=status.waiting_tool,
        outcome=status.outcome,
        failure=status.failure,
        goal=status.goal,
        progress=status.progress,
        last_activity_at=observed_at - status.quiet_seconds,
    )


def read_snapshot(memory_file: Path, pet: str, now: float) -> SessionSnapshot | None:
    try:
        snapshot = SessionSnapshot.model_validate_json(
            snapshot_path(memory_file).read_text()
        )
    except OSError, ValueError, ValidationError:
        return None
    if snapshot.pet != pet or not 0 <= now - snapshot.observed_at <= FRESH_SECONDS:
        return None
    return snapshot.model_copy(
        update={
            "sessions": tuple(
                session
                for session in snapshot.sessions
                if 0 <= now - session.presence_checked_at <= FRESH_SECONDS
            )
        }
    )


class SessionPublisher:
    def __init__(
        self,
        memory_file: Path,
        pet: str,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self.path = snapshot_path(memory_file)
        self.pet = pet
        self.clock = clock
        self.wall_clock = wall_clock
        self.next_publish_at = 0.0
        self.identity: tuple[int, int] | None = None

    def is_due(self) -> bool:
        return self.clock() >= self.next_publish_at

    def publish(
        self,
        statuses: Iterable[AgentSessionStatus],
        *,
        suspended: bool = False,
        force: bool = False,
    ) -> None:
        now = self.clock()
        if not force and now < self.next_publish_at:
            return
        observed_at = self.wall_clock()
        sessions = tuple(
            live_session(status, observed_at)
            for status in statuses
            if not suspended
            and not status.retired
            and status.presence == ProcessStatus.ALIVE
            and status.owner_last_alive_seconds is not None
            and status.owner_last_alive_seconds <= FRESH_SECONDS
        )
        snapshot = SessionSnapshot(
            pet=self.pet,
            observed_at=observed_at,
            suspended=suspended,
            sessions=sessions,
        )
        self.next_publish_at = now + PUBLISH_SECONDS
        self._write(snapshot)

    def _write(self, snapshot: SessionSnapshot) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=self.path.name + ".", dir=self.path.parent
        )
        try:
            with os.fdopen(descriptor, "w") as stream:
                stream.write(snapshot.model_dump_json())
                metadata = os.fstat(stream.fileno())
            os.replace(temporary, self.path)
            self.identity = metadata.st_dev, metadata.st_ino
        finally:
            Path(temporary).unlink(missing_ok=True)

    def close(self) -> None:
        if self.identity is None:
            return
        try:
            current = self.path.lstat()
        except FileNotFoundError:
            return
        if (current.st_dev, current.st_ino) == self.identity:
            self.path.unlink()
        self.identity = None
