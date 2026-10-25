import asyncio
import json
from dataclasses import asdict, dataclass
from datetime import datetime, time, timedelta, tzinfo
from enum import StrEnum
from urllib.parse import quote

from .config import NavigationHandlers
from .domain import ArtifactKind
from .work.models import ReportContext
from .work.shared_work import WorkSubject


class TargetKind(StrEnum):
    WORKTREE = "worktree"
    PULL_REQUEST = "pull_request"
    SESSION = "session"


@dataclass(frozen=True)
class Target:
    kind: TargetKind
    value: str
    source: str = ""


@dataclass(frozen=True)
class OpenResult:
    success: bool
    error: str | None = None


def next_morning(timestamp: float, timezone: tzinfo | None = None) -> float:
    moment = datetime.fromtimestamp(timestamp, timezone)
    tomorrow = moment.date() + timedelta(days=1)
    return datetime.combine(tomorrow, time(9), timezone).timestamp()


class Navigator:
    def __init__(self, handlers: NavigationHandlers) -> None:
        self.handlers = handlers

    def targets(
        self, subject: WorkSubject, context: ReportContext | None = None
    ) -> tuple[Target, ...]:
        if context is None:
            records = (*subject.story.requests, *subject.story.activity)
            latest = max(
                (record for record in records if record.context is not None),
                key=lambda record: record.observed_at or 0,
                default=None,
            )
            context = latest.context if latest else None
        candidates = []
        if context is not None:
            if context.checkout:
                candidates.append(Target(TargetKind.WORKTREE, context.checkout))
            if context.pull_request:
                candidates.append(Target(TargetKind.PULL_REQUEST, context.pull_request))
            candidates.extend(
                Target(TargetKind.PULL_REQUEST, reference.ref)
                for reference in context.references
                if reference.kind == ArtifactKind.PULL_REQUEST
            )
        session = subject.id
        if session.native_id:
            candidates.append(
                Target(TargetKind.SESSION, session.native_id, session.source)
            )
        return tuple(
            dict.fromkeys(
                target
                for target in candidates
                if getattr(self.handlers, target.kind) is not None
            )
        )

    def label(self, target: Target) -> str:
        return {
            TargetKind.WORKTREE: "Show worktree",
            TargetKind.PULL_REQUEST: "See PR",
            TargetKind.SESSION: "View session",
        }[target.kind]

    async def open(self, target: Target) -> OpenResult:
        handler = getattr(self.handlers, target.kind)
        if handler is None:
            return OpenResult(False, "No handler configured")
        values = {
            "target": target.value,
            "path": target.value if target.kind == TargetKind.WORKTREE else "",
            "session_id": target.value if target.kind == TargetKind.SESSION else "",
            "source": target.source,
        }
        if handler.uri_template is not None:
            encoded = {
                name: quote(value, safe="/" if name == "path" else "")
                for name, value in values.items()
            }
            argv = ("/usr/bin/open", handler.uri_template.format_map(encoded))
        else:
            assert handler.argv is not None
            argv = tuple(argument.format_map(values) for argument in handler.argv)
        process = None
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            async with asyncio.timeout(self.handlers.timeout_seconds):
                await process.communicate(json.dumps(asdict(target)).encode())
            if process.returncode:
                return OpenResult(
                    False, f"Handler exited with status {process.returncode}"
                )
            return OpenResult(True)
        except TimeoutError:
            assert process is not None
            process.kill()
            await process.wait()
            return OpenResult(False, "Handler timed out")
        except OSError as error:
            return OpenResult(False, str(error))
        except asyncio.CancelledError:
            if process is not None:
                process.kill()
                await process.wait()
            raise
