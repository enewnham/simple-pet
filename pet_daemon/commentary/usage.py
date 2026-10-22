import contextlib
import contextvars
from collections.abc import Generator
from dataclasses import dataclass


@dataclass(frozen=True)
class UsageTokens:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def __add__(self, other: UsageTokens) -> UsageTokens:
        return UsageTokens(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
        )


request_usage = contextvars.ContextVar[dict[str, UsageTokens] | None](
    "request_usage", default=None
)


@contextlib.contextmanager
def collect_usage() -> Generator[dict[str, UsageTokens]]:
    usage: dict[str, UsageTokens] = {}
    usage_token = request_usage.set(usage)
    try:
        yield usage
    finally:
        request_usage.reset(usage_token)
