import contextlib
import contextvars
from collections.abc import Generator

from pydantic import BaseModel, computed_field


class UsageTokens(BaseModel):
    input_tokens: int | None = 0
    output_tokens: int | None = 0
    cache_read_tokens: int | None = 0
    cache_write_tokens: int | None = 0
    model_calls: int = 0
    retries: int = 0
    request_bytes: int = 0
    system_bytes: int = 0
    user_bytes: int = 0

    @computed_field
    @property
    def total_context_tokens(self) -> int | None:
        values = (self.input_tokens, self.cache_read_tokens, self.cache_write_tokens)
        return (
            None
            if None in values
            else sum(value for value in values if value is not None)
        )

    @computed_field
    @property
    def total_tokens(self) -> int | None:
        return add_tokens(self.total_context_tokens, self.output_tokens)

    def __add__(self, other: UsageTokens) -> UsageTokens:
        return UsageTokens(
            input_tokens=add_tokens(self.input_tokens, other.input_tokens),
            output_tokens=add_tokens(self.output_tokens, other.output_tokens),
            cache_read_tokens=add_tokens(
                self.cache_read_tokens, other.cache_read_tokens
            ),
            cache_write_tokens=add_tokens(
                self.cache_write_tokens, other.cache_write_tokens
            ),
            model_calls=self.model_calls + other.model_calls,
            retries=self.retries + other.retries,
            request_bytes=self.request_bytes + other.request_bytes,
            system_bytes=self.system_bytes + other.system_bytes,
            user_bytes=self.user_bytes + other.user_bytes,
        )


def add_tokens(left: int | None, right: int | None) -> int | None:
    return None if left is None or right is None else left + right


request_usage = contextvars.ContextVar[dict[str, UsageTokens] | None](
    "request_usage", default=None
)
correction_attempt = contextvars.ContextVar("correction_attempt", default=0)


@contextlib.contextmanager
def collect_usage() -> Generator[dict[str, UsageTokens]]:
    usage: dict[str, UsageTokens] = {}
    usage_token = request_usage.set(usage)
    try:
        yield usage
    finally:
        request_usage.reset(usage_token)
