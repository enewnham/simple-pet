from collections.abc import Mapping
from typing import cast

from ..domain import ExecutionOutcome
from ..runtime import HookEvent, HookTextFields, ProcessIdentity

EVENTS = frozenset(
    {
        "session_start",
        "session_end",
        "intent",
        "tool_start",
        "tool_done",
        "tool_error",
        "turn_done",
        "turn_error",
        "interrupt",
        "subagent_start",
        "subagent_done",
        "progress",
        "waiting",
    }
)
HARNESS = frozenset({"cli", "claude", "codex", "cursor"})
TEXT_FIELDS = (
    "parent",
    "parent_call",
    "call",
    "turn",
    "tool",
    "goal",
    "progress",
    "repo",
    "repository",
    "branch",
    "pull_request",
    "tool_name",
    "sentiment_text",
)


def normalize_hook(packet: object) -> HookEvent | None:
    if not isinstance(packet, Mapping):
        return None
    harness, event, session = (
        packet.get("harness"),
        packet.get("event"),
        packet.get("session"),
    )
    if (
        not isinstance(harness, str)
        or harness not in HARNESS
        or not isinstance(event, str)
        or event not in EVENTS
        or not isinstance(session, str)
        or not session
    ):
        return None
    if any(
        name in packet and not isinstance(packet[name], str) for name in TEXT_FIELDS
    ):
        return None
    if "waiting_kind" in packet and packet["waiting_kind"] not in (
        "permission",
        "question",
    ):
        return None
    aliases = packet.get("aliases", [])
    if not isinstance(aliases, list) or not all(
        isinstance(alias, str) and alias for alias in aliases
    ):
        return None
    outcome = None
    if "outcome" in packet:
        try:
            outcome = ExecutionOutcome(packet["outcome"])
        except ValueError, TypeError:
            return None
        if outcome == ExecutionOutcome.STALE:
            return None
    if "\0" in packet.get("repo", ""):
        return None
    validated = cast(HookTextFields, packet)
    code = packet.get("exit_code")
    return HookEvent(
        harness,
        session,
        event,
        aliases=tuple(aliases),
        parent=validated.get("parent"),
        parent_call=validated.get("parent_call"),
        call=validated.get("call"),
        turn=validated.get("turn"),
        tool=validated.get("tool"),
        goal=validated.get("goal"),
        progress=validated.get("progress"),
        repo=validated.get("repo"),
        repository=validated.get("repository"),
        branch=validated.get("branch"),
        pull_request=validated.get("pull_request"),
        tool_name=validated.get("tool_name"),
        sentiment_text=validated.get("sentiment_text"),
        outcome=outcome,
        manual=bool(packet["manual"]) if "manual" in packet else None,
        owner=ProcessIdentity.parse(packet.get("owner")),
        exit_code=code if type(code) is int else None,
        waiting_kind=validated.get("waiting_kind"),
    )
