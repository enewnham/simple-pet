import asyncio
import time
from typing import Protocol

from anthropic import APIStatusError, AsyncAnthropic
from pydantic import BaseModel

from ..domain import ModelPurpose, SpeechReason
from ..logging import get_logger
from .contracts import ModelRequest
from .request import encoded_size
from .usage import UsageTokens, request_usage

log = get_logger("pet-daemon")


class ModelClient(Protocol):
    async def complete(
        self,
        request: ModelRequest,
        reason: SpeechReason | ModelPurpose,
        *,
        response_type: type[BaseModel],
    ) -> dict[str, object]: ...


class AnthropicModelClient:
    def __init__(self, client: AsyncAnthropic) -> None:
        self.client = client

    async def complete(
        self,
        request: ModelRequest,
        reason: SpeechReason | ModelPurpose,
        *,
        response_type: type[BaseModel],
    ) -> dict[str, object]:
        log.debug(
            "observation.request",
            detail="payload",
            reason=reason,
            request=request,
            request_bytes=encoded_size(request),
        )
        started = time.monotonic()
        client = self.client.with_options(
            max_retries=1 if reason == SpeechReason.CLICK else 0
        )
        try:
            async with asyncio.timeout(request["timeout"]):
                response = await client.messages.parse(
                    model=request["model"],
                    max_tokens=request["max_tokens"],
                    timeout=request["timeout"],
                    thinking=request["thinking"],
                    system=request["system"],
                    messages=[
                        {"role": "user", "content": message["content"]}
                        for message in request["messages"]
                    ],
                    output_format=response_type,
                    output_config={"effort": request["output_config"]["effort"]},
                )
        except APIStatusError as error:
            log.warning(
                "observation.api_failed",
                reason=reason,
                request_id=error.request_id,
                status_code=error.status_code,
            )
            raise
        request_id = response._request_id
        log.debug(
            "observation.response",
            detail="payload",
            reason=reason,
            request_id=request_id,
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
            request_id=request_id,
            stop_reason=response.stop_reason,
            duration_seconds=round(time.monotonic() - started, 3),
            **tokens.model_dump(),
        )
        if response.stop_reason != "end_turn":
            log.warning(
                "observation.incomplete",
                reason=reason,
                request_id=request_id,
                stop_reason=response.stop_reason,
            )
            if response.stop_reason == "refusal":
                raise ValueError("Observation refused by model")
            if response.stop_reason == "max_tokens":
                raise ValueError("Observation reached output token limit")
            raise ValueError(f"Observation ended with {response.stop_reason}")
        result = response.parsed_output
        if result is None:
            raise ValueError("Observation requires a structured response")
        if reason == ModelPurpose.STORY:
            log.info(
                "observation.story",
                request_id=request_id,
                duration_seconds=round(time.monotonic() - started, 3),
                **tokens.model_dump(),
            )
        return result.model_dump(mode="json", exclude_unset=True)
