import json
import logging
import time
from typing import Protocol

from .context import encoded_size
from ..telemetry import event, request_usage, verbose


class ModelClient(Protocol):
    async def complete(self, request: dict, reason: str) -> dict: ...


class AnthropicModelClient:
    def __init__(self, client):
        self.client = client

    async def complete(self, request, reason):
        verbose(
            "observation.request",
            reason=reason,
            request=request,
            request_bytes=encoded_size(request),
        )
        started = time.monotonic()
        response = await self.client.messages.create(**request)
        verbose(
            "observation.response",
            reason=reason,
            response=response.model_dump(mode="json"),
        )
        usage = response.usage
        fields = dict(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
        )
        collected = request_usage.get()
        if collected is not None:
            previous = collected.get(reason, {})
            collected[reason] = {
                key: previous.get(key, 0) + value for key, value in fields.items()
            }
        event(
            "observation.usage",
            level=logging.DEBUG,
            reason=reason,
            duration_seconds=round(time.monotonic() - started, 3),
            **fields,
        )
        if response.stop_reason != "end_turn":
            raise ValueError(f"Observation ended with {response.stop_reason}")
        content = "".join(
            block.text for block in response.content if block.type == "text"
        )
        result = json.loads(content)
        if not isinstance(result, dict):
            raise ValueError("Observation requires an object")
        if reason == "summary":
            event(
                "observation.summary",
                duration_seconds=round(time.monotonic() - started, 3),
                **fields,
            )
        return result
