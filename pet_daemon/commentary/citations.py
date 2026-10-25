from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Citations:
    sources: tuple[str, ...]

    @classmethod
    def from_sources(cls, sources: Iterable[str]) -> Citations:
        return cls(tuple(dict.fromkeys(sources)))

    def label(self, source: str) -> str:
        return f"s{self.sources.index(source) + 1}"

    def labels(self, sources: Sequence[str]) -> list[str]:
        return [self.label(source) for source in sources]

    def resolve(self, label: str) -> str:
        for index, source in enumerate(self.sources, 1):
            if label == f"s{index}":
                return source
        raise ValueError("Model cited unavailable evidence")

    def resolve_many(self, labels: Sequence[str]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self.resolve(label) for label in labels))
