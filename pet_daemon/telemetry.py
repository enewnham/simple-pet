import contextlib
import contextvars
import json
import logging
import os
import sys
import time
import uuid
from collections.abc import Generator
from dataclasses import dataclass
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from types import TracebackType

type ExcInfo = (
    bool
    | BaseException
    | tuple[type[BaseException], BaseException, TracebackType | None]
    | tuple[None, None, None]
    | None
)

from typing import ClassVar


@dataclass(frozen=True)
class UsageTokens:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def __add__(self, other: "UsageTokens") -> "UsageTokens":
        return UsageTokens(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
        )


log = logging.getLogger("pet-daemon")
VERBOSE = 5
logging.addLevelName(VERBOSE, "VERBOSE")
observation_id = contextvars.ContextVar[str | None]("observation_id", default=None)
request_usage = contextvars.ContextVar[dict[str, UsageTokens] | None](
    "request_usage", default=None
)


class JSONFormatter(logging.Formatter):
    def __init__(self) -> None:
        super().__init__()
        self.run_id = uuid.uuid4().hex

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "run_id": self.run_id,
            "event": getattr(record, "event", "log"),
            **getattr(record, "fields", {}),
        }
        if observation_id.get() is not None:
            entry["observation_id"] = observation_id.get()
        if not hasattr(record, "event"):
            entry["message"] = record.getMessage()
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False, default=str)


class ConsoleFormatter(logging.Formatter):
    COLORS: ClassVar[dict[int, str]] = {
        VERBOSE: "90",
        logging.DEBUG: "90",
        logging.INFO: "36",
        logging.WARNING: "33",
        logging.ERROR: "31",
        logging.CRITICAL: "31",
    }
    PAYLOADS: ClassVar[set[str]] = {
        "request",
        "response",
        "snapshot",
        "config",
        "clips",
        "packet",
        "decision",
        "context",
        "scheduled",
        "monotonic_seconds",
    }

    def __init__(self, color: bool = False, verbose: bool = False) -> None:
        super().__init__()
        self.color = color
        self.verbose = verbose

    def format(self, record: logging.LogRecord) -> str:
        timestamp = self.formatTime(record, "%H:%M:%S")
        level = f"{record.levelname:<7}"
        if self.color:
            level = f"\033[{self.COLORS.get(record.levelno, '0')}m{level}\033[0m"
        message = getattr(record, "event", record.getMessage())
        fields = getattr(record, "fields", {})
        details = []
        for key, value in fields.items():
            payload = key in self.PAYLOADS
            if payload and not self.verbose:
                continue
            rendered = json.dumps(
                value, ensure_ascii=False, default=str, separators=(",", ":")
            )
            if not payload and len(rendered) > 300:
                rendered = rendered[:297] + "..."
            details.append(f"{key}={rendered}")
        identifier = observation_id.get()
        if identifier is not None:
            details.append(f"obs={identifier[:8]}")
        line = f"{timestamp} {level} {message}"
        if details:
            line += " " + " ".join(details)
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


def configure(path: str | Path, verbose: bool = False, debug: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file = RotatingFileHandler(
        path, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file.setFormatter(JSONFormatter())
    console = logging.StreamHandler()
    console.setLevel(VERBOSE if verbose else logging.DEBUG if debug else logging.INFO)
    console.setFormatter(
        ConsoleFormatter(
            color=sys.stderr.isatty() and "NO_COLOR" not in os.environ, verbose=verbose
        )
    )
    log.setLevel(VERBOSE)
    log.propagate = False
    log.addHandler(file)
    log.addHandler(console)
    event(
        "logging.started",
        path=str(path),
        console_level=logging.getLevelName(console.level),
    )


def event(
    name: str, level: int = logging.INFO, *, exc_info: ExcInfo = None, **fields: object
) -> None:
    log.log(
        level,
        name,
        exc_info=exc_info,
        extra={
            "event": name,
            "fields": {"monotonic_seconds": time.monotonic(), **fields},
        },
    )


def trace(name: str, *, exc_info: ExcInfo = None, **fields: object) -> None:
    event(name, logging.DEBUG, exc_info=exc_info, **fields)


def verbose(name: str, *, exc_info: ExcInfo = None, **fields: object) -> None:
    event(name, VERBOSE, exc_info=exc_info, **fields)


@contextlib.contextmanager
def observation() -> Generator[dict[str, UsageTokens], None, None]:
    token = observation_id.set(uuid.uuid4().hex)
    usage = {}
    usage_token = request_usage.set(usage)
    try:
        yield usage
    finally:
        request_usage.reset(usage_token)
        observation_id.reset(token)
