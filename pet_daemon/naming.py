import asyncio
import json
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .logging import get_logger
from .notifications import bounded, run_command
from .runtime import HookEvent
from .user_text import user_prose

log = get_logger("pet-daemon")


class NamingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    command: tuple[str, ...]
    debounce_seconds: float = Field(default=10, ge=0, allow_inf_nan=False)
    timeout_seconds: float = Field(default=10, gt=0, allow_inf_nan=False)

    @field_validator("command")
    @classmethod
    def valid_command(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or any(not item or "\0" in item for item in value):
            raise ValueError("naming.command requires nonempty argv without NUL")
        return value


def proposed_name(text: str) -> str:
    lines = bounded(user_prose(text)).splitlines()
    for line in lines:
        line = re.sub(r"^[\s#>*`-]+", "", line).strip()
        if not line or line.startswith("<"):
            continue
        line = re.sub(
            r"^(?:please\s+)?(?:can|could|would) you\s+", "", line, flags=re.IGNORECASE
        )
        line = re.sub(r"^please\s+", "", line, flags=re.IGNORECASE)
        words = line.split()[:8]
        name = " ".join(words)[:64].rstrip()
        return name.rstrip(".,:;!?")
    return ""


@dataclass
class NamingSession:
    hook: HookEvent
    first_prompt: str = ""
    candidate: str = ""
    attempted: str = ""
    due: float = 0


class SessionNaming:
    def __init__(
        self,
        config: NamingConfig,
        mark_lookup: Callable[[HookEvent], dict[str, object]],
        clock: Callable[[], float] = time.monotonic,
        runner: Callable[
            [tuple[str, ...], bytes, float], Awaitable[bool]
        ] = run_command,
    ) -> None:
        self.config = config
        self.mark_lookup = mark_lookup
        self.clock = clock
        self.runner = runner
        self.sessions: dict[tuple[str, str], NamingSession] = {}
        self.task: asyncio.Task[None] | None = None
        self.next_refresh = 0.0

    def ingest(self, hook: HookEvent) -> None:
        key = (hook.source, hook.session)
        state = self.sessions.get(key)
        if state is None:
            state = NamingSession(hook)
            self.sessions[key] = state
            if len(self.sessions) > 1000:
                self.sessions.pop(next(iter(self.sessions)))
        state.hook = hook.model_copy(
            update={
                "repo": hook.repo if hook.repo is not None else state.hook.repo,
                "zmx_session": hook.zmx_session
                if hook.zmx_session is not None
                else state.hook.zmx_session,
            }
        )
        if hook.event == "intent" and hook.goal and not state.first_prompt:
            state.first_prompt = hook.goal

    def refresh(self, state: NamingSession) -> None:
        marks = self.mark_lookup(state.hook)
        text = marks.get("title") or marks.get("goal") or state.first_prompt
        candidate = proposed_name(text) if isinstance(text, str) else ""
        if candidate != state.candidate:
            state.candidate = candidate
            state.due = self.clock() + self.config.debounce_seconds

    def tick(self) -> None:
        if self.task is not None and not self.task.done():
            return
        if self.task is not None:
            self.task.result()
        if self.clock() >= self.next_refresh:
            for state in self.sessions.values():
                self.refresh(state)
            self.next_refresh = self.clock() + 1
        self.task = asyncio.create_task(self.deliver())

    async def deliver(self) -> None:
        for state in tuple(self.sessions.values()):
            name = state.candidate
            if not name or name == state.attempted or self.clock() < state.due:
                continue
            state.attempted = name
            hook = state.hook
            payload = {
                "harness": hook.source,
                "session": hook.session,
                "thread_id": hook.session,
                "zmx_session": hook.zmx_session,
                "cwd": hook.repo,
                "name": name,
            }
            try:
                success = await self.runner(
                    self.config.command,
                    json.dumps(payload).encode(),
                    self.config.timeout_seconds,
                )
            except OSError, TimeoutError:
                success = False
            log.info("naming.outcome", harness=hook.source, success=success)
            return

    async def close(self) -> None:
        if self.task is not None:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
