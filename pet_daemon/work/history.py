from __future__ import annotations

import re
from typing import TYPE_CHECKING

from pydantic import BaseModel

from ..domain import Activity, ExecutionOutcome
from ..identity import SessionIdentity
from ..prompts import bounded
from ..prose import work_prose
from ..repository_identity import RepositoryIdentity, parse_remote_repository
from ..runtime import WaitingKind
from ..sessions.activity import activity

if TYPE_CHECKING:
    from ..runtime import HookEvent


class HistoryDetails(BaseModel):
    reported_outcome: ExecutionOutcome | None = None
    activity: Activity | None = None
    progress: str | None = None


HISTORY_SEARCH_STOP_WORDS = {
    "please",
    "continue",
    "failed",
    "error",
    "build",
    "tests",
    "running",
}


class SessionEntry(BaseModel):
    session: SessionIdentity
    working_directory: str
    repository: RepositoryIdentity | None
    event_type: str
    details: HistoryDetails
    goal: str
    explicit_goal: str


class RecallQuery(BaseModel):
    repository: RepositoryIdentity | None
    session: SessionIdentity
    terms: tuple[str, ...]


class SessionContext(BaseModel):
    session_id: SessionIdentity
    working_directory: str
    goal: str
    last_tool: str
    outcome: ExecutionOutcome | None
    activity: Activity | None


def build_session_entry(
    event: HookEvent, context: SessionContext | None = None
) -> SessionEntry | None:
    kind = event.event
    if (
        kind in {"tool_start", "tool_done", "tool_error"}
        or event.waiting_kind == WaitingKind.PERMISSION_CHECK
    ):
        return None
    progress = work_prose(event.progress)
    if kind == "progress" and not progress:
        return None
    working_directory = bounded(
        event.repo or (context.working_directory if context else None), 1024
    )
    session_id = context.session_id if context else event.subject
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
    details = HistoryDetails(
        reported_outcome=outcome, activity=category, progress=progress or None
    )
    return SessionEntry(
        session=session_id,
        working_directory=working_directory,
        repository=parse_remote_repository(event.repository)
        if event.repository
        else None,
        event_type=kind,
        details=details,
        goal=goal,
        explicit_goal=explicit_goal,
    )


def build_recall_query(
    event: HookEvent | None, *, repository: RepositoryIdentity | None
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
        repository=repository,
        session=event.subject,
        terms=search_terms,
    )
