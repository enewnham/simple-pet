import json
import time
from typing import Protocol

from anthropic import AsyncAnthropic

from ..domain import ModelPurpose, SpeechReason
from ..logging import get_logger
from .context import encoded_size
from .contracts import ModelRequest
from .usage import UsageTokens, request_usage

log = get_logger("pet-daemon")


class ModelClient(Protocol):
    async def complete(
        self, request: ModelRequest, reason: SpeechReason | ModelPurpose
    ) -> dict[str, object]: ...


class AnthropicModelClient:
    def __init__(self, client: AsyncAnthropic) -> None:
        self.client = client

    async def complete(
        self, request: ModelRequest, reason: SpeechReason | ModelPurpose
    ) -> dict[str, object]:
        log.debug(
            "observation.request",
            detail="payload",
            reason=reason,
            request=request,
            request_bytes=encoded_size(request),
        )
        started = time.monotonic()
        response = await self.client.messages.create(
            model=request["model"],
            max_tokens=request["max_tokens"],
            timeout=request["timeout"],
            thinking=request["thinking"],
            system=request["system"],
            messages=[
                {"role": "user", "content": message["content"]}
                for message in request["messages"]
            ],
            output_config={
                "effort": request["output_config"]["effort"],
                "format": {
                    "type": "json_schema",
                    "schema": dict(request["output_config"]["format"]["schema"]),
                },
            },
        )
        log.debug(
            "observation.response",
            detail="payload",
            reason=reason,
            response=response.model_dump(mode="json"),
        )
        usage = response.usage
        tokens = UsageTokens(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=usage.cache_read_input_tokens or 0,
            cache_write_tokens=usage.cache_creation_input_tokens or 0,
        )
        collected = request_usage.get()
        if collected is not None:
            previous = collected.get(reason, UsageTokens())
            collected[reason] = previous + tokens
        log.debug(
            "observation.usage",
            reason=reason,
            duration_seconds=round(time.monotonic() - started, 3),
            **tokens.model_dump(),
        )
        if response.stop_reason != "end_turn":
            raise ValueError(f"Observation ended with {response.stop_reason}")
        content = "".join(
            block.text for block in response.content if block.type == "text"
        )
        result = json.loads(content)
        if not isinstance(result, dict):
            raise ValueError("Observation requires an object")  # noqa: TRY004 - Report malformed JSON values through the validation contract.
        if reason == ModelPurpose.STORY:
            log.info(
                "observation.story",
                duration_seconds=round(time.monotonic() - started, 3),
                **tokens.model_dump(),
            )
        return result
