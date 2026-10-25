import json
import logging
import os
import sys
import time
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import cast

import structlog
from pydantic import BaseModel
from structlog.stdlib import get_logger
from structlog.typing import EventDict, Processor, WrappedLogger

__all__ = ["configure", "get_logger"]

RUN_ID = uuid.uuid4().hex
PAYLOAD_FIELDS = {
    "request",
    "response",
    "snapshot",
    "config",
    "clips",
    "packet",
    "decision",
    "context",
    "scheduled",
}


def add_metadata(logger: WrappedLogger, method: str, event: EventDict) -> EventDict:
    event["run_id"] = RUN_ID
    event["monotonic_seconds"] = time.monotonic()
    return event


SHARED_PROCESSORS: list[Processor] = [
    structlog.contextvars.merge_contextvars,
    structlog.stdlib.add_log_level,
    structlog.stdlib.add_logger_name,
    structlog.processors.TimeStamper(fmt="iso", utc=True),
    add_metadata,
    structlog.processors.StackInfoRenderer(),
]
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        *SHARED_PROCESSORS,
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)
log = get_logger("pet-daemon")


def json_default(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (set, frozenset)):
        return list(value)
    return str(value)


class ConsoleFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not (
            isinstance(record.msg, dict) and record.msg.get("detail") == "payload"
        )


class ConsolePresentation:
    def __init__(self, verbose: bool) -> None:
        self.verbose = verbose

    def __call__(
        self, logger: WrappedLogger, method: str, event: EventDict
    ) -> EventDict:
        event.pop("run_id", None)
        event.pop("monotonic_seconds", None)
        event.pop("detail", None)
        if event.get("logger") == "pet-daemon":
            event.pop("logger")
        timestamp = event.get("timestamp")
        if isinstance(timestamp, str):
            event["timestamp"] = (
                datetime.fromisoformat(timestamp).astimezone().strftime("%H:%M:%S")
            )
        identifier = event.get("observation_id")
        if isinstance(identifier, str):
            event["observation_id"] = identifier[:8]
        if not self.verbose:
            for key in PAYLOAD_FIELDS:
                event.pop(key, None)
        return event


def configure(
    path: str | Path | None = None, verbose: bool = False, debug: bool = False
) -> None:
    foreign_processors = [structlog.stdlib.ExtraAdder(), *SHARED_PROCESSORS]
    colors = sys.stderr.isatty() and "NO_COLOR" not in os.environ
    styles = structlog.dev.ConsoleRenderer.get_default_column_styles(colors)
    level_styles = structlog.dev.ConsoleRenderer.get_default_level_styles(colors)
    level_styles["debug"] = styles.logger_name

    def render_level(key: str, value: object) -> str:
        level = str(value)
        return f"{styles.bright}{level_styles[level]}{level[0].upper()}{styles.reset}"

    renderer = structlog.dev.ConsoleRenderer(
        colors=colors,
        pad_event_to=28,
        exception_formatter=structlog.dev.RichTracebackFormatter(
            show_locals=False,
            color_system="truecolor" if colors else None,
        ),
    )
    default_formatter = cast(
        structlog.dev.KeyValueColumnFormatter, renderer.columns[0].formatter
    )
    scalar_repr = default_formatter.value_repr

    def render_value(value: object) -> str:
        if isinstance(value, (BaseModel, dict, list, tuple, set, frozenset, Enum)) or (
            is_dataclass(value) and not isinstance(value, type)
        ):
            rendered = json.dumps(value, default=json_default, ensure_ascii=False)
        else:
            rendered = scalar_repr(value)
        return (
            rendered[:297] + "..." if not verbose and len(rendered) > 300 else rendered
        )

    default_formatter.value_repr = render_value
    for column in renderer.columns:
        if column.key == "level":
            column.formatter = render_level
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if debug or verbose else logging.INFO)
    if not verbose:
        console.addFilter(ConsoleFilter())
    console.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=foreign_processors,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                ConsolePresentation(verbose),
                renderer,
            ],
        )
    )
    handlers: list[logging.Handler] = [console]
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        file = RotatingFileHandler(
            path, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        file.setLevel(logging.DEBUG)
        file.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                foreign_pre_chain=foreign_processors,
                processors=[
                    structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                    structlog.processors.format_exc_info,
                    structlog.processors.JSONRenderer(
                        default=json_default, ensure_ascii=False
                    ),
                ],
            )
        )
        handlers.append(file)
    logging.basicConfig(level=logging.DEBUG, handlers=handlers, force=True)
    for name in ("httpx", "httpcore", "httpx2", "httpcore2", "anthropic", "asyncio"):
        logging.getLogger(name).setLevel(logging.WARNING)
    if path is not None:
        log.info(
            "logging.started",
            path=path,
            console_level="debug" if debug or verbose else "info",
            payloads=verbose,
        )
