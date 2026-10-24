from collections.abc import Mapping
from typing import cast

from ..domain import ExecutionOutcome, SessionSource
from ..runtime import (
    HookEvent,
    HookEventKind,
    HookTextFields,
    ProcessIdentity,
    WaitingKind,
)

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
    source, event, session = (
        packet.get("source"),
        packet.get("event"),
        packet.get("session"),
    )
    if (
        not isinstance(source, str)
        or not isinstance(event, str)
        or not isinstance(session, str)
        or not session
    ):
        return None
    try:
        session_source = SessionSource(source)
        event_kind = HookEventKind(event)
    except ValueError:
        return None
    if any(
        name in packet and not isinstance(packet[name], str) for name in TEXT_FIELDS
    ):
        return None
    waiting_kind = None
    if "waiting_kind" in packet:
        try:
            waiting_kind = WaitingKind(packet["waiting_kind"])
        except ValueError, TypeError:
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
        session_source,
        session,
        event_kind,
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
        waiting_kind=waiting_kind,
    )
