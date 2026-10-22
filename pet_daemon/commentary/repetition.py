import asyncio
import math
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

import objc
from NaturalLanguage import NLEmbedding, NLLanguageEnglish  # ty: ignore[unresolved-import]

from ..telemetry import event

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


def similarity(first: Sequence[float], second: Sequence[float]) -> float | None:
    if len(first) != len(second):
        return None
    dot = 0.0
    first_norm = 0.0
    second_norm = 0.0
    for left, right in zip(first, second):
        dot += left * right
        first_norm += left * left
        second_norm += right * right
    if first_norm <= 0 or second_norm <= 0:
        return None
    value = dot / math.sqrt(first_norm * second_norm)
    return min(1.0, max(-1.0, value)) if math.isfinite(value) else None


def semantic_match(
    text: str, candidates: Sequence[tuple[int, str]], threshold: float
) -> RepeatMatch | None:
    with objc.autorelease_pool():
        try:
            embedding = NLEmbedding.sentenceEmbeddingForLanguage_(NLLanguageEnglish)
            vector = embedding.vectorForString_(text) if embedding is not None else None
            if vector is None:
                unavailable("embedding_unavailable")
                return None
            scored = []
            for index, previous in candidates:
                previous_vector = embedding.vectorForString_(previous)
                if previous_vector is None:
                    continue
                score = similarity(vector, previous_vector)
                if score is not None and threshold <= score <= 1:
                    scored.append((score, index))
            if scored:
                score, index = max(scored)
                return RepeatMatch(index, score, "semantic")
        except objc.error:  # ty: ignore[unresolved-attribute]
            unavailable("query_failed")
    return None


async def repeat_match(
    text: str,
    history: Sequence[str],
    *,
    threshold: float = SEMANTIC_THRESHOLD,
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
    return await asyncio.to_thread(semantic_match, text, candidates, threshold)
