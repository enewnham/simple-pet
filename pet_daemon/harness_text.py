import re
from textwrap import dedent

from .domain import SessionSource

_CLAUDE_MESSAGE = re.compile(
    r'Another Claude session sent a message:\n<agent-message from="[^"\n]+">\n'
    r"(?P<body>.*?)\n</agent-message>"
    r'(?:\n\nThat "other Claude session" is an agent working inside this same session [^\n]*)?',
    re.DOTALL,
)
_CLAUDE_HANDBACK = re.compile(
    r"\[Subagent hand-back\] The text below is the final report of a subagent "
    r"this session delegated to\.[^\n]* The report follows:\n(?P<report>.*)",
    re.DOTALL,
)
_CLAUDE_NOTIFICATION = re.compile(
    r"<task-notification>\n.*?\n<result>"
    r'This agent\'s report was delivered to you as a message from "[^"\n]+" '
    r"\(its SubagentHandback call\)\. Read it there; it is not repeated here\."
    r"\s*</result>.*?</task-notification>",
    re.DOTALL,
)
_CLAUDE_COMMAND = re.compile(
    r"<(command-name|command-message|command-args)>(.*?)</\1>\s*",
    re.DOTALL,
)
_CODEX_MESSAGE = re.compile(
    r"(?:Agent message from /[^\n]+:\n)?"
    r"Message Type: (?:MESSAGE|FINAL_ANSWER)\n"
    r"Task name: /[^\n]+\nSender: /[^\n]+\nPayload:\n(?P<body>.*)",
    re.DOTALL,
)


def clean_text(source: SessionSource, text: str) -> str:
    match source:
        case SessionSource.CLAUDE:
            return _claude_text(text)
        case SessionSource.CODEX:
            message = _CODEX_MESSAGE.fullmatch(text)
            return message["body"] if message else text
        case _:
            return text


def _claude_text(text: str) -> str:
    message = _CLAUDE_MESSAGE.fullmatch(text)
    if message:
        body = message["body"]
        handback = _CLAUDE_HANDBACK.fullmatch(body)
        return dedent(handback["report"]) if handback else body
    if _CLAUDE_NOTIFICATION.fullmatch(text):
        return ""
    fields = list(_CLAUDE_COMMAND.finditer(text))
    if (
        len(fields) == 3
        and "".join(field[0] for field in fields) == text
        and {field[1] for field in fields}
        == {"command-name", "command-message", "command-args"}
    ):
        return next(field[2] for field in fields if field[1] == "command-args")
    return text
