"""Forward native agent hook metadata to the pet daemon."""

import argparse
import json
import os
import sys
from collections.abc import Mapping

from ..domain import ExecutionOutcome, SessionSource
from ..prose import work_prose
from ..runtime import HookEvent, HookEventKind, WaitingKind
from ..sessions.processes import discover_owner
from ..user_text import sentiment_excerpt
from .notify import send

COMMON_EVENTS = {
    "SessionStart": HookEventKind.SESSION_START,
    "SessionEnd": HookEventKind.SESSION_END,
    "UserPromptSubmit": HookEventKind.INTENT,
    "PreToolUse": HookEventKind.TOOL_START,
    "PostToolUse": HookEventKind.TOOL_DONE,
    "PostToolUseFailure": HookEventKind.TOOL_ERROR,
    "Stop": HookEventKind.TURN_DONE,
    "StopFailure": HookEventKind.TURN_ERROR,
    "Interrupt": HookEventKind.INTERRUPT,
    "SubagentStart": HookEventKind.SUBAGENT_START,
    "SubagentStop": HookEventKind.SUBAGENT_DONE,
    "PermissionRequest": HookEventKind.WAITING,
    "Elicitation": HookEventKind.WAITING,
    "Notification": HookEventKind.WAITING,
}
EVENTS = {
    "claude": {
        key: value for key, value in COMMON_EVENTS.items() if key != "Interrupt"
    },
    "codex": {
        key: value
        for key, value in COMMON_EVENTS.items()
        if key
        not in ("PostToolUseFailure", "StopFailure", "Elicitation", "Notification")
    },
    "cursor": {
        "sessionStart": HookEventKind.SESSION_START,
        "sessionEnd": HookEventKind.SESSION_END,
        "beforeSubmitPrompt": HookEventKind.INTENT,
        "preToolUse": HookEventKind.TOOL_START,
        "postToolUse": HookEventKind.TOOL_DONE,
        "postToolUseFailure": HookEventKind.TOOL_ERROR,
        "stop": HookEventKind.TURN_DONE,
        "subagentStart": HookEventKind.SUBAGENT_START,
        "subagentStop": HookEventKind.SUBAGENT_DONE,
        "afterAgentResponse": HookEventKind.PROGRESS,
        "afterAgentThought": HookEventKind.PROGRESS,
    },
}
INPUT_TOOLS = ("AskUserQuestion", "request_user_input", "RequestUserInput")


def string(*values: object) -> str:
    return next((value for value in values if isinstance(value, str) and value), "")


def excerpt(*values: object, limit: int = 240) -> str:
    return " ".join(string(*values).split())[:limit]


def object_value(value: object) -> dict[str, object]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return {}
    return value if isinstance(value, dict) else {}


def failed(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    code = value.get("exit_code", value.get("exitCode"))
    return (
        value.get("isError") is True
        or value.get("is_error") is True
        or (type(code) is int and code != 0)
    )


def update(source: str, event: str, payload: Mapping[str, object]) -> HookEvent | None:
    if source == "claude" and payload.get("cursor_version"):
        return None
    if event == "Notification" and payload.get("notification_type") not in (
        "permission_prompt",
        "elicitation_dialog",
    ):
        return None
    normalized = EVENTS[source][event]
    parent = ""
    aliases = []
    if source == "cursor":
        session = string(payload.get("conversation_id"))
        if normalized in (HookEventKind.SUBAGENT_START, HookEventKind.SUBAGENT_DONE):
            parent = string(payload.get("parent_conversation_id"), session)
            aliases = list(
                dict.fromkeys(
                    value
                    for value in (
                        payload.get("subagent_id"),
                        payload.get("child_conversation_id"),
                    )
                    if isinstance(value, str) and value
                )
            )
            session = string(
                payload.get("child_conversation_id"), payload.get("subagent_id")
            )
    else:
        session = string(payload.get("session_id"))
        agent = string(payload.get("agent_id"))
        if agent:
            parent, session = session, agent
    if not session:
        return None
    cwd = string(payload.get("cwd"))
    repo = os.path.normpath(cwd) if cwd else None
    turn = string(payload.get("generation_id"), payload.get("turn_id"))
    if normalized in (HookEventKind.SUBAGENT_START, HookEventKind.SUBAGENT_DONE):
        turn = ""
    call = string(payload.get("tool_use_id"), payload.get("tool_call_id"))
    if normalized not in (
        HookEventKind.TOOL_START,
        HookEventKind.TOOL_DONE,
        HookEventKind.TOOL_ERROR,
        HookEventKind.WAITING,
    ):
        call = ""
    parent_call = string(payload.get("parent_tool_call_id"))
    if normalized in (HookEventKind.SUBAGENT_START, HookEventKind.SUBAGENT_DONE):
        parent_call = string(parent_call, payload.get("tool_call_id"))
    tool_name = string(payload.get("tool_name"))
    tool_input = object_value(payload.get("tool_input"))
    command = string(
        tool_input.get("command"),
        tool_input.get("cmd"),
        tool_input.get("file_path"),
        tool_input.get("path"),
        tool_input.get("description"),
    )
    tool = excerpt(command, tool_name, limit=160)
    goal = None
    sentiment_text = None
    progress = None
    waiting_kind = None
    if normalized == HookEventKind.INTENT:
        goal = string(payload.get("prompt"))
        if not parent:
            sentiment_text = sentiment_excerpt(goal)
    elif normalized == HookEventKind.SUBAGENT_START and payload.get("task"):
        goal = string(payload.get("task"))
    normalized_event = normalized
    if normalized == HookEventKind.TOOL_START and tool_name in INPUT_TOOLS:
        normalized_event = HookEventKind.WAITING
        waiting_kind = WaitingKind.QUESTION
        questions = tool_input.get("questions")
        if isinstance(questions, list):
            progress = (
                excerpt(
                    " ".join(
                        string(question.get("question"), question.get("title"))
                        for question in questions
                        if isinstance(question, dict)
                    )
                )
                or None
            )
    elif event == "PermissionRequest":
        waiting_kind = WaitingKind.PERMISSION_CHECK
    elif (
        event == "Notification"
        and payload.get("notification_type") == "permission_prompt"
    ):
        waiting_kind = WaitingKind.PERMISSION
    elif normalized == HookEventKind.WAITING:
        waiting_kind = WaitingKind.QUESTION
    response = payload.get("tool_response")
    if source == "cursor":
        output = payload.get("tool_output")
        response = object_value(output) or output
    if normalized == HookEventKind.TOOL_DONE and failed(response):
        normalized_event = HookEventKind.TOOL_ERROR
    status = payload.get("status")
    if normalized == HookEventKind.TURN_DONE and source == "cursor":
        if status == "error":
            normalized_event = HookEventKind.TURN_ERROR
        elif status == "aborted":
            normalized_event = HookEventKind.INTERRUPT
    interrupted = (
        normalized == HookEventKind.TOOL_ERROR and payload.get("is_interrupt") is True
    )
    outcome = None
    if interrupted:
        normalized_event = HookEventKind.TOOL_DONE
        outcome = ExecutionOutcome.CANCELLED
    if normalized_event in (HookEventKind.TOOL_ERROR, HookEventKind.TURN_ERROR):
        outcome = ExecutionOutcome.ERROR
    elif normalized_event == HookEventKind.INTERRUPT:
        outcome = ExecutionOutcome.CANCELLED
    elif normalized == HookEventKind.SUBAGENT_DONE and source == "cursor":
        outcome = {
            "error": ExecutionOutcome.ERROR,
            "aborted": ExecutionOutcome.CANCELLED,
        }.get(string(status), ExecutionOutcome.SUCCESS)
    elif not interrupted and normalized in (
        HookEventKind.TOOL_DONE,
        HookEventKind.TURN_DONE,
        HookEventKind.SUBAGENT_DONE,
    ):
        outcome = ExecutionOutcome.SUCCESS
    if outcome != ExecutionOutcome.ERROR and normalized in (
        HookEventKind.TOOL_DONE,
        HookEventKind.TURN_DONE,
        HookEventKind.SUBAGENT_DONE,
        HookEventKind.PROGRESS,
    ):
        reported = work_prose(string(payload.get("last_assistant_message")))
        if normalized != HookEventKind.TOOL_DONE:
            reported = work_prose(
                string(
                    reported,
                    payload.get("text"),
                    payload.get("response"),
                    payload.get("thought"),
                    payload.get("summary"),
                )
            )
        progress = reported or None
    return HookEvent(
        source=SessionSource(source),
        session=session,
        event=normalized_event,
        repo=repo,
        repository=string(payload.get("repository")) or None,
        branch=string(payload.get("branch")) or None,
        pull_request=string(payload.get("pull_request")) or None,
        parent=parent if parent and parent != session else None,
        aliases=tuple(aliases),
        turn=turn or None,
        call=call or None,
        parent_call=parent_call or None,
        tool_name=tool_name or None,
        tool=tool or None,
        goal=goal,
        sentiment_text=sentiment_text,
        progress=progress,
        waiting_kind=waiting_kind,
        outcome=outcome,
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="pet-hook", description=__doc__)
    parser.add_argument("source", choices=EVENTS)
    parser.add_argument("event")
    args = parser.parse_args()
    if args.event not in EVENTS[args.source]:
        parser.error("unsupported hook event")
    try:
        payload = json.load(sys.stdin)
    except ValueError, OSError:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    hook = update(args.source, args.event, payload)
    if hook is not None:
        owner = discover_owner(hook.source)
        hook.owner = owner
        send(hook.packet())
    response = (
        {"continue": True}
        if args.source == "cursor" and args.event == "beforeSubmitPrompt"
        else {}
    )
    print(json.dumps(response))


if __name__ == "__main__":
    main()
