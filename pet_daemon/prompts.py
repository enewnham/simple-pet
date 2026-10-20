"""Keep the actionable prose in bounded agent prompts."""

import json
import re


ACTION = re.compile(r"^(?:[-*+]\s+|\d+[.)]\s+)?(?:fix|add|update|refactor|implement|create|build|remove|debug|test|explain|check|why|how)\b", re.I)
ACTION_ANYWHERE = re.compile(r"\b(?:fix|add|update|refactor|implement|create|build|remove|debug|test|explain|check|why|how)\b", re.I)
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
PROMPT_MAX_CHARS = 1000


def bounded(value, limit=240):
    return " ".join(str(value or "").split())[:limit]


def _byte_cost(text):
    return len(json.dumps(text, ensure_ascii=False).encode("utf-8"))


def _units(text):
    units = []
    fenced = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            fenced = not fenced
            continue
        if not stripped:
            continue
        for sentence in SENTENCE_END.split(stripped):
            words = sentence.split()
            chunk = ""
            for word in words:
                if len(chunk) + len(word) + bool(chunk) > 360 and chunk:
                    units.append((chunk, not fenced))
                    chunk = ""
                chunk = f"{chunk} {word}" if chunk else word
            if chunk:
                units.append((chunk, not fenced))
    return units


def _rank(unit):
    text, prose = unit
    if not prose:
        return 0
    if ACTION.search(text):
        return 3
    if "?" in text:
        return 2
    return 1 if ACTION_ANYWHERE.search(text) else 0


def _fits(text, limit, max_bytes):
    return len(text) <= limit and (max_bytes is None or _byte_cost(text) <= max_bytes)


def _shorten(text, limit, max_bytes, preceding=False, following=False):
    prefix = "[omitted] " if preceding else ""
    complete = prefix + text + (" [omitted]" if following else "")
    if _fits(complete, limit, max_bytes):
        return complete
    low, high = 0, min(len(text), limit)
    while low < high:
        middle = (low + high + 1) // 2
        suffix = " [omitted]" if following or middle < len(text) else ""
        candidate = prefix + text[:middle] + suffix
        if _fits(candidate, limit, max_bytes):
            low = middle
        else:
            high = middle - 1
    if low:
        suffix = " [omitted]" if following or low < len(text) else ""
        return prefix + text[:low] + suffix
    return "[omitted]" if _fits("[omitted]", limit, max_bytes) else ""


def select_prompt(text, limit=PROMPT_MAX_CHARS, max_bytes=None):
    """Select prose within character and JSON string byte budgets."""
    if not isinstance(text, str) or not text or limit <= 0 or (max_bytes is not None and max_bytes < 2):
        return ""
    if _fits(text, limit, max_bytes):
        return text

    units = _units(text)
    if not units:
        return _shorten(text, limit, max_bytes)
    prose = [index for index, (_, is_prose) in enumerate(units) if is_prose]
    anchors = list(dict.fromkeys((prose[-1], prose[0]))) if prose else []
    preferred = max(prose, key=lambda index: (_rank(units[index]), index)) if prose else len(units) - 1
    ranked = sorted((index for index in prose if index not in anchors),
                    key=lambda index: (_rank(units[index]), index), reverse=True)
    selected = set()

    def render(indices):
        parts = []
        previous = -1
        for index in sorted(indices):
            if index > previous + 1:
                parts.append("[omitted]")
            parts.append(units[index][0])
            previous = index
        if previous < len(units) - 1:
            parts.append("[omitted]")
        return " ".join(parts)

    if anchors and not _fits(render({anchors[0]}), limit, max_bytes):
        index = preferred
        return _shorten(units[index][0], limit, max_bytes,
                        preceding=index > 0, following=index < len(units) - 1)

    for index in anchors + ranked:
        candidate = render(selected | {index})
        if _fits(candidate, limit, max_bytes):
            selected.add(index)
    if selected:
        return render(selected)

    index = preferred
    return _shorten(units[index][0], limit, max_bytes,
                    preceding=index > 0, following=index < len(units) - 1)
