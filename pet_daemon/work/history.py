from __future__ import annotations

import hashlib
import json
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
from .models import ExecutionEvidence, IntentEvidence, ProgressEvidence, SessionReport

if TYPE_CHECKING:
    from ..runtime import HookEvent


class HistoryDetails(BaseModel):
    reported_outcome: ExecutionOutcome | None = None
    activity: Activity | None = None
    progress: str | None = None


REPORT_SEARCH_STOP_WORDS = {
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


def report_prose(report: SessionReport) -> str:
    payload = report.payload
    if isinstance(payload, IntentEvidence | ProgressEvidence):
        return payload.text
    if isinstance(payload, ExecutionEvidence) and payload.event in {
        "turn_done",
        "turn_error",
        "subagent_done",
        "interrupt",
    }:
        return payload.progress or " ".join(
            part
            for part in (
                payload.event.replace("_", " "),
                payload.outcome.value if payload.outcome else "",
            )
            if part
        )
    return ""


def report_search_terms(text: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            token.lower()
            for token in re.findall(r"[\w]+", text)
            if len(token) >= 5 and token.lower() not in REPORT_SEARCH_STOP_WORDS
        )
    )


def report_source_key(report: SessionReport) -> str:
    source = {
        "payload": report.payload.model_dump(mode="json"),
        "observed_at": report.observed_at,
    }
    encoded = json.dumps(source, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()
