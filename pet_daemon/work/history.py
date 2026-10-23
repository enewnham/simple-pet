from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..domain import Activity, ExecutionOutcome
from ..prompts import bounded
from ..prose import work_prose
from ..sessions.activity import activity
from .validation import repository_identity

if TYPE_CHECKING:
    from ..runtime import HookEvent


@dataclass(frozen=True)
class HistoryDetails:
    reported_outcome: ExecutionOutcome | None = None
    activity: Activity | None = None
    progress: str | None = None

    @classmethod
    def from_data(cls, data: dict[str, str | None]) -> HistoryDetails:
        outcome, activity = data["reported_outcome"], data["activity"]
        return cls(
            ExecutionOutcome(outcome) if outcome is not None else None,
            Activity(activity) if activity is not None else None,
            data["progress"],
        )


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
    details: HistoryDetails
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
    outcome: ExecutionOutcome | None
    activity: Activity | None


def build_session_entry(
    event: HookEvent, context: SessionContext | None = None
) -> SessionEntry | None:
    kind = event.event
    if kind in {"tool_start", "tool_done", "tool_error"}:
        return None
    progress = work_prose(event.progress)
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
        outcome = ExecutionOutcome.ERROR
    category = None
    if outcome and not terminal:
        tool = bounded(event.tool or (context.last_tool if context else None), 160)
        category = (
            context.activity
            if context and context.activity
            else activity(tool, event.tool_name or "")
        )
    details = HistoryDetails(outcome, category, progress or None)
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
