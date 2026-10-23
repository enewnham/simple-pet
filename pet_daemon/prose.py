"""Extract bounded work prose without quoted or executable material."""

import re
from collections.abc import Sequence

_SENTENCES = re.compile(r'(?<=[.!?])["”]?\s+')
_MARKER = "[omitted]"


def exclude_speech(text: str, excluded: Sequence[str] = ()) -> str:
    prose = text
    for utterance in excluded:
        words = utterance.split()
        if words:
            prose = re.sub(r"\s+".join(re.escape(word) for word in words), "", prose)
    prose = re.sub(r'(?:The pet said:\s*)?["“]\s*["”]', "", prose)
    return "\n".join(" ".join(line.split()) for line in prose.splitlines()).strip(
        ' \n"“”'
    )


def prose_text(text: str | None, excluded: Sequence[str] = ()) -> str:
    if not text:
        return ""
    paragraphs = []
    fence = ""
    table = False
    lines = text.splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(("```", "~~~")):
            delimiter = stripped[:3]
            if not fence:
                fence = delimiter
            elif delimiter == fence:
                fence = ""
            continue
        if fence or stripped.startswith(">") or line.startswith(("    ", "\t")):
            continue
        separator = lines[index + 1].strip() if index + 1 < len(lines) else ""
        if "|" in stripped and re.fullmatch(
            r"\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*", separator
        ):
            table = True
        if table:
            if "|" in stripped:
                continue
            table = False
        if "|" in stripped and (
            stripped.startswith("|")
            or stripped.endswith("|")
            or re.fullmatch(r"[\s|:\-]+", stripped)
        ):
            continue
        stripped = re.sub(r"^\s*(?:#{1,6}\s+|[-*+]\s+|\d+[.)]\s+)", "", stripped)
        paragraphs.append(stripped)
    return exclude_speech("\n".join(paragraphs), excluded)


def work_prose(text: str | None, limit: int = 800, excluded: Sequence[str] = ()) -> str:
    if limit <= 0:
        return ""
    prose = " ".join(prose_text(text, excluded).split())
    if len(prose) <= limit:
        return prose
    if limit <= len(_MARKER):
        return ""
    available = limit - len(_MARKER) - 1
    complete = []
    for sentence in _SENTENCES.split(prose):
        if len(" ".join([*complete, sentence])) > available:
            break
        if sentence.endswith((".", "!", "?")):
            complete.append(sentence)
        else:
            break
    excerpt = " ".join(complete)
    if not excerpt:
        excerpt = prose[:available].rsplit(" ", 1)[0]
    return f"{excerpt} {_MARKER}".strip()[:limit]
