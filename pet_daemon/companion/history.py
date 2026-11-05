import asyncio
import json
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime

from ..coordination.models import SourceReference
from ..domain import SessionSource
from ..harness_text import clean_text
from .config import NativeHistoryConfig
from .providers import (
    Availability,
    Document,
    SourceResult,
    SourceStatus,
    lexical_tokens,
    revision,
)


def _text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(block["text"])
            for block in content
            if isinstance(block, dict)
            and block.get("type") in ("text", "input_text", "output_text")
            and isinstance(block.get("text"), str)
        )
    return ""


def _message(record: dict[str, object], format_name: str) -> tuple[str, str]:
    message = (
        record.get("message")
        if format_name == "claude"
        else record.get("payload")
        if format_name == "codex"
        else record
    )
    if not isinstance(message, dict):
        return "", ""
    role = message.get("role")
    if role not in ("user", "assistant"):
        return "", ""
    content = _text(message.get("content"))
    if format_name in ("claude", "codex"):
        content = clean_text(SessionSource(format_name), content)
    return str(role), content


class NativeHistory:
    def __init__(self, config: NativeHistoryConfig, clock: Callable[[], float]) -> None:
        self.config = config
        self.clock = clock
        self.name = f"history:{config.name}"

    def _scan(self) -> SourceResult:
        root = self.config.path.resolve()
        status = SourceStatus(
            self.name,
            Availability.AVAILABLE,
            self.clock(),
            collections=(str(root),),
            aliases=(self.config.format,),
        )
        if not root.exists():
            return SourceResult(
                (),
                replace(
                    status,
                    availability=Availability.UNAVAILABLE,
                    detail="Configured harness history is unavailable.",
                ),
            )
        files = (
            [root]
            if root.is_file()
            else sorted(
                (
                    path
                    for path in root.rglob("*.jsonl")
                    if path.is_file() and path.resolve().is_relative_to(root)
                ),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )[: self.config.maximum_files]
        )
        documents = []
        errors = []
        for path in files:
            try:
                with path.open("rb") as source:
                    size = path.stat().st_size
                    start = max(0, size - self.config.maximum_file_bytes)
                    source.seek(start)
                    if start:
                        source.readline()
                    lines = source.read(self.config.maximum_file_bytes).splitlines()
                captured = []
                for line in reversed(lines):
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError, UnicodeDecodeError:
                        continue
                    if not isinstance(record, dict):
                        continue
                    role, text = _message(record, self.config.format)
                    if not text:
                        continue
                    original = None
                    timestamp = record.get("timestamp")
                    if isinstance(timestamp, str):
                        try:
                            date = datetime.fromisoformat(timestamp)
                            original = (
                                date.timestamp() if date.tzinfo is not None else None
                            )
                        except ValueError:
                            pass
                    key = revision(line.decode("utf-8"))
                    captured.append(
                        Document(
                            SourceReference(
                                f"{self.name}:{path}:{key}",
                                self.name,
                                key,
                                self.clock(),
                                uri=path.as_uri(),
                                scope=str(root),
                                original_at=original,
                            ),
                            f"{self.config.format} historical {role} message",
                            text,
                            kind="harness_history_user"
                            if role == "user"
                            else "harness_history_report",
                        )
                    )
                    if len(captured) >= self.config.maximum_records:
                        break
                documents.extend(captured)
            except (OSError, UnicodeError) as error:
                errors.append(str(error))
        if errors:
            status = replace(
                status, availability=Availability.FAILED, detail="; ".join(errors)
            )
        return SourceResult(tuple(documents[: self.config.maximum_records]), status)

    async def search(self, query: str, limit: int) -> SourceResult:
        result = await asyncio.to_thread(self._scan)
        tokens = lexical_tokens(query)
        ordered = sorted(
            result.documents,
            key=lambda document: len(tokens & lexical_tokens(document.text)),
            reverse=True,
        )
        return replace(result, documents=tuple(ordered[:limit]))

    async def read(self, source_id: str) -> SourceResult:
        result = await asyncio.to_thread(self._scan)
        if not any(document.source.id == source_id for document in result.documents):
            return replace(
                result,
                documents=(),
                status=replace(
                    result.status,
                    availability=Availability.UNAVAILABLE,
                    detail="The record is outside the configured bounded history window or unavailable.",
                ),
            )
        return replace(
            result,
            documents=tuple(
                document
                for document in result.documents
                if document.source.id == source_id
            ),
        )
