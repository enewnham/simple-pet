from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..prompts import bounded
from ..sessions.activity import activity, result_summary

if TYPE_CHECKING:
    from ..runtime import HookEvent


@dataclass(frozen=True)
class SessionFact:
    session: str
    repo: str
    event_type: str
    summary: str
    goal: str
    requested_goal: str
    coalesce: bool = False


@dataclass(frozen=True)
class RecallQuery:
    repository: str
    session: str
    terms: tuple[str, ...]


@dataclass(frozen=True)
class EventContext:
    id: str
    repo: str
    goal: str
    last_tool: str
    outcome: str | None
    activity: str


def project_event(
    event: HookEvent, context: EventContext | None = None
) -> SessionFact | None:
    kind = event.event
    repo = bounded(event.repo or (context.repo if context else None), 1024)
    identifier = bounded(
        context.id if context else f"{event.harness}:{event.session}", 256
    )
    progress = bounded(event.progress)
    requested_goal = bounded(event.goal)
    goal = requested_goal or bounded(context.goal if context else None)
    tool = bounded(event.tool or (context.last_tool if context else None), 160)
    if kind in {"tool_start", "tool_done", "tool_error"}:
        return None
    if kind == "progress" and not progress:
        return None
    terminal = kind in (
        "turn_done",
        "turn_error",
        "subagent_done",
        "interrupt",
        "session_end",
    )
    outcome = (
        context.outcome if terminal and context and context.outcome else event.outcome
    )
    if kind == "turn_error":
        outcome = "error"
    category = (
        context.activity
        if context and context.activity
        else activity(tool, event.tool_name or "")
    )
    if terminal:
        category = "harness"
    result = result_summary(category, outcome) if outcome else ""
    if terminal and outcome == "success":
        result = "Turn completed (agent reported success)"
    elif terminal and outcome:
        result = f"{result} (agent reported {outcome})"
    facts = [
        kind,
        goal if kind in ("intent", "subagent_start") else "",
        result,
        progress,
    ]
    summary = bounded(" | ".join(part for part in facts if part), 320)
    return SessionFact(
        identifier,
        repo,
        kind,
        summary,
        goal,
        requested_goal,
        kind in {"progress", "waiting", "session_start"},
    )


def recall_query(event: HookEvent | None) -> RecallQuery | None:
    if not event or event.event != "intent" or not event.repo or not event.goal:
        return None
    repo = bounded(event.repo, 1024)
    tokens = re.findall(r"[\w]+", event.goal)
    distinctive = list(
        dict.fromkeys(
            token
            for token in tokens
            if len(token) >= 5
            and token.lower()
            not in {
                "please",
                "continue",
                "failed",
                "error",
                "build",
                "tests",
                "running",
            }
        )
    )[:3]
    if not distinctive:
        return None

    session = bounded(f"{event.harness}:{event.session}", 256)
    return RecallQuery(repo, session, tuple(distinctive))
