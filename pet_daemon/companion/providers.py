import asyncio
import hashlib
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from enum import StrEnum
from typing import Protocol

from ..coordination.models import SourceReference
from .config import MeetingConfig


class Availability(StrEnum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


@dataclass(frozen=True)
class SourceStatus:
    provider: str
    availability: Availability
    checked_at: float
    detail: str = ""
    collections: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()


@dataclass(frozen=True)
class Document:
    source: SourceReference
    title: str
    text: str
    kind: str = "source"
    topics: tuple[str, ...] = ()
    links: tuple[str, ...] = ()
    superseded: bool = False
    leaf_sources: tuple[SourceReference, ...] = ()
    checked_at: float | None = None
    query: str = ""


@dataclass(frozen=True)
class Passage:
    source: SourceReference
    document_id: str
    title: str
    text: str
    start_line: int
    end_line: int
    kind: str = "source"
    topics: tuple[str, ...] = ()
    links: tuple[str, ...] = ()
    superseded: bool = False
    source_status: SourceStatus | None = None
    leaf_sources: tuple[SourceReference, ...] = ()
    checked_at: float | None = None


@dataclass(frozen=True)
class SourceResult:
    documents: tuple[Document, ...]
    status: SourceStatus
    complete: bool = False
    missing_ids: tuple[str, ...] = ()


class SourceProvider(Protocol):
    name: str

    async def search(self, query: str, limit: int) -> SourceResult: ...

    async def read(self, source_id: str) -> SourceResult: ...


def revision(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def passages(document: Document, characters: int) -> tuple[Passage, ...]:
    lines = document.text.splitlines()
    result = []
    start = 0
    size = 0
    for index, line in enumerate(lines):
        size += len(line) + 1
        is_boundary = not line.strip() or size >= characters or index == len(lines) - 1
        if not is_boundary:
            continue
        text = "\n".join(lines[start : index + 1]).strip()
        if text:
            source = replace(
                document.source,
                id=f"{document.source.id}#L{start + 1}-L{index + 1}",
                uri=f"{document.source.uri}#L{start + 1}-L{index + 1}",
            )
            result.append(
                Passage(
                    source,
                    document.source.id,
                    document.title,
                    text,
                    start + 1,
                    index + 1,
                    document.kind,
                    document.topics,
                    document.links,
                    document.superseded,
                    leaf_sources=document.leaf_sources,
                    checked_at=document.checked_at,
                )
            )
        start = index + 1
        size = 0
    return tuple(result)


def lexical_tokens(text: str) -> set[str]:
    return set(re.findall(r"[\w-]+", text.casefold()))


def _original_date(text: str) -> float | None:
    if not text.startswith("---\n"):
        return None
    _, _, metadata = text.partition("---\n")
    metadata, separator, _ = metadata.partition("\n---")
    if not separator:
        return None
    match = re.search(r"(?m)^(?:date|meeting_date):\s*[\"']?([^\n\"']+)", metadata)
    if match is None:
        return None
    value = match.group(1).strip()
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed.timestamp() if parsed.tzinfo is not None else None


class LocalMeetings:
    name = "meetings"

    def __init__(self, config: MeetingConfig, clock: Callable[[], float]) -> None:
        self.config = config
        self.clock = clock

    def _scan(self) -> SourceResult:
        checked_at = self.clock()
        documents = []
        unavailable = []
        for configured in self.config.folders:
            folder = configured.resolve()
            if not folder.is_dir():
                unavailable.append(str(configured))
                continue
            for path in sorted(folder.rglob("*")):
                if (
                    path.suffix.lower() not in self.config.extensions
                    or not path.is_file()
                ):
                    continue
                resolved = path.resolve()
                if not resolved.is_relative_to(folder):
                    continue
                try:
                    if resolved.stat().st_size > self.config.maximum_file_bytes:
                        unavailable.append(f"{path}: exceeds configured file limit")
                        continue
                    text = resolved.read_text(encoding="utf-8")
                except (OSError, UnicodeError) as error:
                    unavailable.append(f"{path}: {error}")
                    continue
                title = next(
                    (
                        line.lstrip("# ")
                        for line in text.splitlines()
                        if line.startswith("# ")
                    ),
                    path.stem,
                )
                documents.append(
                    Document(
                        source=SourceReference(
                            id=f"meetings:{resolved}",
                            provider=self.name,
                            revision=revision(text),
                            observed_at=checked_at,
                            uri=resolved.as_uri(),
                            scope=str(folder),
                            original_at=_original_date(text),
                        ),
                        title=title,
                        text=text,
                        kind="meeting_export",
                        links=tuple(re.findall(r"https?://[^\s)>]+", text)),
                        checked_at=checked_at,
                    )
                )
        status = Availability.AVAILABLE
        if not self.config.folders:
            status = Availability.UNAVAILABLE
        elif unavailable:
            status = Availability.FAILED if documents else Availability.UNAVAILABLE
        return SourceResult(
            tuple(documents),
            SourceStatus(
                self.name,
                status,
                checked_at,
                "; ".join(unavailable),
                tuple(str(folder) for folder in self.config.folders),
            ),
        )

    async def snapshot(self) -> SourceResult:
        result = await asyncio.to_thread(self._scan)
        return replace(
            result, complete=result.status.availability == Availability.AVAILABLE
        )

    async def search(self, query: str, limit: int) -> SourceResult:
        result = await self.snapshot()
        tokens = lexical_tokens(query)
        ordered = sorted(
            result.documents,
            key=lambda item: len(tokens & lexical_tokens(item.title + " " + item.text)),
            reverse=True,
        )
        return replace(result, documents=tuple(ordered[:limit]), complete=False)

    async def read(self, source_id: str) -> SourceResult:
        result = await asyncio.to_thread(self._scan)
        return replace(
            result,
            documents=tuple(
                item for item in result.documents if item.source.id == source_id
            ),
        )


class WorkHistory:
    name = "work"

    def __init__(
        self,
        documents: Callable[[], Sequence[Document]],
        clock: Callable[[], float],
        name: str = "work",
    ) -> None:
        self.name = name
        self.documents = documents
        self.clock = clock

    async def search(self, query: str, limit: int) -> SourceResult:
        tokens = lexical_tokens(query)
        documents = sorted(
            self.documents(),
            key=lambda item: (
                len(tokens & lexical_tokens(item.title + " " + item.text)),
                item.source.observed_at,
            ),
            reverse=True,
        )
        return SourceResult(
            tuple(documents[:limit]),
            SourceStatus(self.name, Availability.AVAILABLE, self.clock()),
        )

    async def read(self, source_id: str) -> SourceResult:
        return SourceResult(
            tuple(item for item in self.documents() if item.source.id == source_id),
            SourceStatus(self.name, Availability.AVAILABLE, self.clock()),
        )
