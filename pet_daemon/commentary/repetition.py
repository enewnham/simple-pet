import asyncio
import json
import math
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from ..telemetry import event

SIMILARITY_BINARY = (
    Path(__file__).resolve().parents[2] / ".build" / "pet-speech-similarity"
)
SEMANTIC_THRESHOLD = 0.9
NEGATIONS = {
    "no",
    "not",
    "never",
    "without",
    "nothing",
    "nobody",
    "cannot",
    "cant",
    "dont",
    "isnt",
    "arent",
    "wasnt",
    "werent",
    "hasnt",
    "havent",
    "wont",
    "didnt",
}
OUTCOMES = (
    {"pass", "passed", "passes", "passing", "success", "successful", "succeeded"},
    {"fail", "failed", "fails", "failing", "failure", "error", "errors"},
    {"merged", "merge"},
    {"open", "opened"},
    {"closed", "close"},
    {"cancelled", "canceled", "cancel"},
    {"pending"},
    {"staged"},
    {"unstaged"},
    {"untracked"},
    {"pushed"},
    {"unpushed"},
)
GRAMMAR = {
    "a",
    "an",
    "the",
    "i",
    "me",
    "my",
    "we",
    "us",
    "our",
    "you",
    "your",
    "he",
    "him",
    "his",
    "she",
    "her",
    "they",
    "them",
    "their",
    "it",
    "its",
    "this",
    "that",
    "these",
    "those",
    "am",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "shall",
    "should",
    "can",
    "could",
    "may",
    "might",
    "must",
    "to",
    "of",
    "in",
    "on",
    "at",
    "for",
    "with",
    "by",
    "from",
    "and",
    "or",
    "but",
    "as",
    "than",
    "im",
    "ive",
    "weve",
    "youre",
    "youve",
    "theyre",
    "theyve",
    "again",
    "still",
    "already",
    "just",
    "yet",
}


@dataclass(frozen=True)
class RepeatMatch:
    index: int
    similarity: float
    method: str


def normalized(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold().replace("’", "'")
    text = re.sub(r"(?<=\w)'(?=\w)", "", text)
    return " ".join(re.findall(r"[+-]?\d+(?:\.\d+)?|\w+", text))


def anchors(text: str) -> tuple[tuple[str, ...], bool, tuple[bool, ...]]:
    words = set(normalized(text).split())
    numbers = tuple(re.findall(r"[+-]?\d+(?:\.\d+)?", text))
    return (
        numbers,
        bool(words & NEGATIONS),
        tuple(bool(words & group) for group in OUTCOMES),
    )


def content_tokens(text: str) -> set[str]:
    words = set(normalized(text).split())
    outcomes = set().union(*OUTCOMES)
    return {
        word
        for word in words - GRAMMAR - NEGATIONS - outcomes
        if not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", word)
    }


@lru_cache(maxsize=16)
def unavailable(reason: str) -> None:
    event("speech.similarity_unavailable", reason=reason)


async def repeat_match(
    text: str,
    history: Sequence[str],
    *,
    executable: Path = SIMILARITY_BINARY,
    threshold: float = SEMANTIC_THRESHOLD,
    timeout: float = 2,
) -> RepeatMatch | None:
    value = normalized(text)
    candidates = [
        (index, previous)
        for index, previous in enumerate(history)
        if anchors(text) == anchors(previous)
    ]
    for index, previous in candidates:
        if value == normalized(previous):
            return RepeatMatch(index, 1.0, "exact")
    content = content_tokens(text)
    eligible = []
    for index, previous in candidates:
        previous_content = content_tokens(previous)
        if content != previous_content:
            continue
        eligible.append((index, previous))
    candidates = eligible
    if not candidates:
        return None
    process = None
    try:
        process = await asyncio.create_subprocess_exec(
            str(executable),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        request = json.dumps(
            {"text": text, "previous": [previous for _, previous in candidates]}
        ).encode()
        output, _ = await asyncio.wait_for(process.communicate(request), timeout)
        if process.returncode:
            unavailable("helper_failed")
            return None
        response = json.loads(output)
        if not response["available"]:
            unavailable("embedding_unavailable")
            return None
        values = response["similarities"]
        if len(values) != len(candidates):
            raise ValueError("Similarity response has the wrong size")
        scored = [
            (score, candidates[index][0])
            for index, score in enumerate(values)
            if type(score) in (int, float)
            and math.isfinite(score)
            and threshold <= score <= 1
        ]
        if scored:
            score, index = max(scored)
            return RepeatMatch(index, score, "semantic")
    except (TimeoutError, OSError, ValueError, KeyError, TypeError):
        unavailable("query_failed")
    finally:
        if process is not None and process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            await process.wait()
    return None
