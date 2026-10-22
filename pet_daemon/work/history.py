from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypedDict

from ..prompts import bounded
from ..sessions.activity import activity
from .validation import repository_identity

if TYPE_CHECKING:
    from ..runtime import HookEvent


class SessionDetail(TypedDict):
    kind: Literal["reported_outcome", "activity", "progress"]
    text: str


HISTORY_SEARCH_STOP_WORDS = {
    "please",
    "continue",
    "failed",
    "error",
    "build",
    "tests",
    "running",
}


@dataclass(frozen=True)
class SessionEntry:
    session: str
    working_directory: str
    repository: str
    event_type: str
    details: list[SessionDetail]
    goal: str
    explicit_goal: str


@dataclass(frozen=True)
class RecallQuery:
    repository: str
    session: str
    terms: tuple[str, ...]


@dataclass(frozen=True)
class SessionContext:
    session_id: str
    working_directory: str
    goal: str
    last_tool: str
    outcome: str | None
    activity: str


def build_session_entry(
    event: HookEvent, context: SessionContext | None = None
) -> SessionEntry | None:
    kind = event.event
    if kind in {"tool_start", "tool_done", "tool_error"}:
        return None
    progress = bounded(event.progress)
    if kind == "progress" and not progress:
        return None
    working_directory = bounded(
        event.repo or (context.working_directory if context else None), 1024
    )
    session_id = bounded(
        context.session_id if context else f"{event.harness}:{event.session}", 256
    )
    explicit_goal = bounded(event.goal)
    goal = explicit_goal or bounded(context.goal if context else None)
    terminal = kind in {
        "turn_done",
        "turn_error",
        "subagent_done",
        "interrupt",
        "session_end",
    }
    outcome = (
        context.outcome if terminal and context and context.outcome else event.outcome
    )
    if kind == "turn_error":
        outcome = "error"
    details: list[SessionDetail] = []
    if outcome:
        details.append({"kind": "reported_outcome", "text": outcome})
        if not terminal:
            tool = bounded(event.tool or (context.last_tool if context else None), 160)
            category = (
                context.activity
                if context and context.activity
                else activity(tool, event.tool_name or "")
            )
            details.append({"kind": "activity", "text": category})
    if progress:
        details.append({"kind": "progress", "text": progress})
    return SessionEntry(
        session=session_id,
        working_directory=working_directory,
        repository=repository_identity(bounded(event.repository, 1024)),
        event_type=kind,
        details=details,
        goal=goal,
        explicit_goal=explicit_goal,
    )


def build_recall_query(
    event: HookEvent | None, *, repository: str
) -> RecallQuery | None:
    if not event or event.event != "intent" or not repository or not event.goal:
        return None
    tokens = re.findall(r"[\w]+", event.goal)
    search_terms = tuple(
        dict.fromkeys(
            token.lower()
            for token in tokens
            if len(token) >= 5 and token.lower() not in HISTORY_SEARCH_STOP_WORDS
        )
    )[:3]
    if not search_terms:
        return None
    return RecallQuery(
        repository=repository_identity(bounded(repository, 1024)),
        session=bounded(f"{event.harness}:{event.session}", 256),
        terms=search_terms,
    )
