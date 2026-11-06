import asyncio
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from functools import lru_cache

import objc
from NaturalLanguage import NLEmbedding, NLLanguageEnglish  # ty: ignore[unresolved-import]

from ..coordination.models import SourceReference
from .config import RetrievalConfig
from .index import SourceChange, SourceIndex
from .providers import (
    Availability,
    Passage,
    SourceProvider,
    SourceResult,
    SourceStatus,
    lexical_tokens,
    passages,
)


@dataclass(frozen=True)
class SearchContext:
    scope: str = ""
    references: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    confirmed_links: tuple[str, ...] = ()
    excluded_references: tuple[str, ...] = ()
    corrections: tuple[str, ...] = ()


@dataclass(frozen=True)
class SearchHit:
    passage: Passage
    reason: tuple[str, ...]
    semantic_relevance: float


@dataclass(frozen=True)
class SearchResult:
    hits: tuple[SearchHit, ...]
    sources: tuple[SourceStatus, ...]
    queries: tuple[str, ...]
    corrections: tuple[str, ...] = ()


@lru_cache(maxsize=1024)
def _embedding(text: str) -> tuple[float, ...]:
    with objc.autorelease_pool():
        model = NLEmbedding.sentenceEmbeddingForLanguage_(NLLanguageEnglish)
        vector = model.vectorForString_(text) if model is not None else None
        return tuple(vector) if vector is not None else ()


def semantic_relevance(question: str, text: str) -> float:
    first = _embedding(question)
    second = _embedding(text)
    if not first or not second:
        return 0
    numerator = sum(left * right for left, right in zip(first, second))
    denominator = math.sqrt(sum(value * value for value in first)) * math.sqrt(
        sum(value * value for value in second)
    )
    return max(0, min(1, numerator / denominator)) if denominator else 0


def expanded_queries(
    question: str, context: SearchContext, limit: int
) -> tuple[str, ...]:
    excluded = set(context.excluded_references)
    identifiers = tuple(
        value
        for value in (*context.references, *context.confirmed_links, *context.aliases)
        if value not in excluded
    )
    return tuple(dict.fromkeys((question, *identifiers)))[:limit]


class ContextRetrieval:
    def __init__(
        self,
        index: SourceIndex,
        providers: Sequence[SourceProvider],
        config: RetrievalConfig,
        clock: Callable[[], float],
        semantic_score: Callable[[str, str], float] = semantic_relevance,
    ) -> None:
        self.index = index
        self.providers = {provider.name: provider for provider in providers}
        self.config = config
        self.clock = clock
        self.semantic_score = semantic_score

    async def _query(self, provider: SourceProvider, query: str) -> SourceResult:
        try:
            return await provider.search(query, self.config.results)
        except (OSError, RuntimeError, ValueError, TimeoutError) as error:
            return SourceResult(
                (),
                SourceStatus(
                    provider.name, Availability.FAILED, self.clock(), str(error)
                ),
            )

    async def search(
        self, question: str, context: SearchContext | None = None
    ) -> SearchResult:
        context = context or SearchContext()
        queries = expanded_queries(question, context, self.config.queries)
        results = await asyncio.gather(
            *(
                self._query(provider, query)
                for provider in self.providers.values()
                for query in queries
            )
        )
        for result in results:
            self.index.capture(result)
        statuses = {status.provider: status for status in self.index.statuses()}
        for result in results:
            previous = statuses.get(result.status.provider)
            if previous is None or result.status.availability != Availability.AVAILABLE:
                statuses[result.status.provider] = result.status
        documents = {}
        for result in results:
            for document in result.documents:
                stored = self.index.document(
                    document.source.id, document.source.revision
                )
                assert stored is not None
                documents[(stored.source.id, stored.source.revision)] = stored
        candidates = tuple(
            passage
            for document in documents.values()
            for passage in passages(document, self.config.passage_characters)
        )
        ranked = await asyncio.to_thread(self._rank, question, context, candidates)
        selected = self._diverse(ranked)
        return SearchResult(
            selected, tuple(statuses.values()), queries, context.corrections
        )

    def _rank(
        self, question: str, context: SearchContext, candidates: Sequence[Passage]
    ) -> tuple[SearchHit, ...]:
        tokens = lexical_tokens(question)
        ranked = []
        for passage in candidates:
            location = f"{passage.document_id} {passage.source.uri} {passage.title} {passage.text}"
            if any(reference in location for reference in context.excluded_references):
                continue
            reasons = []
            exact = any(reference in location for reference in context.references)
            linked = any(reference in location for reference in context.confirmed_links)
            if exact:
                reasons.append("exact reference")
            if linked:
                reasons.append("confirmed relationship")
            overlap = len(tokens & lexical_tokens(location)) / max(1, len(tokens))
            semantic = self.semantic_score(question, passage.text)
            decision = bool(
                lexical_tokens(passage.text)
                & {
                    "decided",
                    "decision",
                    "because",
                    "agreed",
                    "requirement",
                    "correction",
                }
            )
            if overlap:
                reasons.append("matching terms")
            if semantic:
                reasons.append("semantic relevance")
            if decision:
                reasons.append("decision or rationale")
            if passage.superseded:
                reasons.append("superseded historical evidence")
            if not reasons:
                continue
            score = (
                not passage.superseded,
                exact,
                linked,
                overlap + semantic,
                decision,
                passage.source.original_at or passage.source.observed_at,
            )
            ranked.append((score, SearchHit(passage, tuple(reasons), semantic)))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return tuple(hit for _, hit in ranked)

    def _diverse(self, ranked: Sequence[SearchHit]) -> tuple[SearchHit, ...]:
        selected = []
        remaining = []
        seen = set()
        for hit in ranked:
            provider = hit.passage.source.provider
            if provider not in seen:
                selected.append(hit)
                seen.add(provider)
            else:
                remaining.append(hit)
        return tuple((*selected, *remaining)[: self.config.results])

    async def read_passage(
        self, reference: SourceReference, neighbors: int | None = None
    ) -> tuple[Passage, ...]:
        document_id = reference.id.split("#L", 1)[0]
        document = self.index.document(document_id, reference.revision)
        provider = self.providers.get(reference.provider)
        status = None
        if provider is not None:
            try:
                result = await provider.read(document_id)
                self.index.capture(result)
                status = result.status
                if (
                    not result.documents
                    and result.status.availability == Availability.AVAILABLE
                    and document is not None
                ):
                    self.index.mark_removed(document.source, result.status.checked_at)
            except (OSError, RuntimeError, ValueError, TimeoutError) as error:
                status = SourceStatus(
                    provider.name, Availability.FAILED, self.clock(), str(error)
                )
                self.index.capture(SourceResult((), status))
        document = self.index.document(document_id, reference.revision) or document
        if document is None:
            return ()
        current = self.index.document(document_id)
        is_current = (
            current is not None and current.source.revision == reference.revision
        )
        available = (
            document.source.available
            and is_current
            and (status is None or status.availability == Availability.AVAILABLE)
        )
        document = replace(
            document,
            source=replace(document.source, available=available),
            superseded=current is not None and not is_current,
        )
        candidates = tuple(
            replace(passage, source_status=status)
            for passage in passages(document, self.config.passage_characters)
        )
        position = next(
            (
                index
                for index, passage in enumerate(candidates)
                if passage.source.id == reference.id
            ),
            None,
        )
        if position is None:
            return candidates[:1] if document.source.id == reference.id else ()
        surrounding = (
            self.config.neighboring_passages if neighbors is None else neighbors
        )
        if surrounding < 0 or surrounding > self.config.neighboring_passages:
            raise ValueError("Passage expansion exceeds configured neighboring scope")
        return candidates[max(0, position - surrounding) : position + surrounding + 1]

    def changes(self, checkpoint: int) -> tuple[SourceChange, ...]:
        return self.index.changes(checkpoint)
