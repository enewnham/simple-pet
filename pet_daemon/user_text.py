import re

from .prompts import select_prompt

EMBEDDED = re.compile(
    r"<(?P<tag>agent-message|task-notification|pasted_content|skill|environment_context|instructions)\b[^>]*>.*?</(?P=tag)\b[^>]*>",
    re.IGNORECASE | re.DOTALL,
)


def user_prose(text: str) -> str:
    text = EMBEDDED.sub("", text)
    lines = []
    fence = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(("```", "~~~")):
            marker = stripped[:3]
            if fence is None:
                fence = marker
            elif fence == marker:
                fence = None
            continue
        if stripped and fence is None and not stripped.startswith(">"):
            lines.append(re.sub(r"`[^`]*`", "", stripped))
    return "\n".join(lines).strip()


def sentiment_excerpt(text: str) -> str:
    return select_prompt(user_prose(text), limit=600, max_bytes=602)
