from dataclasses import dataclass
import re

from ..activity import activity, result_summary
from ..prompts import bounded


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


def project_event(event, session=None) -> SessionFact | None:
    kind = event["event"]
    state = session or {}
    repo = bounded(event.get("repo") or state.get("repo"), 1024)
    identifier = bounded(
        state.get("id") or f"{event['harness']}:{event['session']}", 256
    )
    progress = bounded(event.get("progress"))
    requested_goal = bounded(event.get("goal"))
    goal = requested_goal or bounded(state.get("goal"))
    tool = bounded(event.get("tool") or state.get("last_tool"), 160)
    if kind == "tool_start":
        return None
    name = event.get("tool_name", "").lower().rsplit("__", 1)[-1].rsplit(".", 1)[-1]
    read_only = name in (
        "read",
        "read_file",
        "readfile",
        "grep",
        "find",
        "search",
        "glob",
    )
    if kind == "tool_done" and read_only:
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
        state.get("outcome")
        if terminal and state.get("outcome")
        else event.get("outcome")
    )
    if kind in ("tool_error", "turn_error"):
        outcome = "error"
    category = state.get("activity") or activity(tool, event.get("tool_name", ""))
    if terminal:
        category = "harness"
    if outcome == "error":
        progress = ""
    result = result_summary(category, outcome) if outcome else ""
    if terminal and outcome == "success":
        result = "Turn completed (agent reported success)"
    facts = [
        kind,
        goal if kind in ("intent", "subagent_start") else "",
        result,
        tool if kind.startswith("tool_") and outcome != "error" else "",
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


def recall_query(event) -> RecallQuery | None:
    if (
        not event
        or event.get("event") != "intent"
        or not event.get("repo")
        or not event.get("goal")
    ):
        return None
    repo = bounded(event["repo"], 1024)
    tokens = re.findall(r"[\w]+", event["goal"])
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

    session = bounded(
        event.get("target_session")
        or f"{event.get('harness', '')}:{event.get('session', '')}",
        256,
    )
    return RecallQuery(repo, session, tuple(distinctive))
