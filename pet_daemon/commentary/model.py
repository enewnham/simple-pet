import asyncio
import time
import uuid
from typing import Protocol, cast

from anthropic import APIStatusError, AsyncAnthropic
from anthropic.types import OutputConfigParam
from pydantic import BaseModel

from ..domain import ModelPurpose, SpeechReason
from ..logging import get_logger
from .contracts import ModelRequest
from .request import encoded_size
from .usage import UsageTokens, correction_attempt, request_usage

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
        request_id = uuid.uuid4().hex
        outcome = "failed"
        stop_reason = None
        tokens = UsageTokens(
            input_tokens=None,
            output_tokens=None,
            cache_read_tokens=None,
            cache_write_tokens=None,
        )
        retry = correction_attempt.get()
        sizes = {
            "request_bytes": encoded_size(
                {key: value for key, value in request.items() if key != "timeout"}
            ),
            "system_bytes": encoded_size(request["system"]),
            "user_bytes": encoded_size(request["messages"]),
        }
        transport_retry_limit = 1 if reason == SpeechReason.CLICK else 0
        client = self.client.with_options(max_retries=transport_retry_limit)
        try:
            async with asyncio.timeout(request["timeout"]):
                response = await client.messages.create(
                    model=request["model"],
                    max_tokens=request["max_tokens"],
                    timeout=request["timeout"],
                    thinking=request["thinking"],
                    system=request["system"],
                    messages=[
                        {"role": "user", "content": message["content"]}
                        for message in request["messages"]
                    ],
                    output_config=cast(OutputConfigParam, request["output_config"]),
                )
            request_id = response._request_id or request_id
            stop_reason = response.stop_reason
            usage = response.usage
            tokens = UsageTokens(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cache_read_tokens=usage.cache_read_input_tokens or 0,
                cache_write_tokens=usage.cache_creation_input_tokens or 0,
            )
            log.debug(
                "observation.response",
                detail="payload",
                reason=reason,
                request_id=request_id,
                response=response.model_dump(mode="json"),
            )
            if stop_reason != "end_turn":
                outcome = "refused" if stop_reason == "refusal" else "incomplete"
                log.warning(
                    "observation.incomplete",
                    reason=reason,
                    request_id=request_id,
                    stop_reason=stop_reason,
                )
                raise ValueError(f"Observation ended with {stop_reason}")
            text = "".join(
                block.text for block in response.content if block.type == "text"
            )
            result = response_type.model_validate_json(text)
            outcome = "completed"
            if reason == ModelPurpose.STORY:
                log.info(
                    "observation.story",
                    request_id=request_id,
                    duration_seconds=round(time.monotonic() - started, 3),
                    **tokens.model_dump(),
                )
            return result.model_dump(mode="json", exclude_unset=True)
        except APIStatusError as error:
            request_id = error.request_id or request_id
            log.warning(
                "observation.api_failed",
                reason=reason,
                request_id=request_id,
                status_code=error.status_code,
            )
            raise
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        except TimeoutError:
            outcome = "timeout"
            raise
        except ValueError:
            if outcome == "failed":
                outcome = "invalid_response"
            raise
        finally:
            tokens = tokens.model_copy(
                update={"model_calls": 1, "retries": retry, **sizes}
            )
            collected = request_usage.get()
            if collected is not None:
                collected[reason] = collected.get(reason, UsageTokens()) + tokens
            log.info(
                "observation.usage",
                reason=reason,
                model=request["model"],
                request_id=request_id,
                stop_reason=stop_reason,
                outcome=outcome,
                transport_retry_limit=transport_retry_limit,
                duration_seconds=round(time.monotonic() - started, 3),
                **tokens.model_dump(),
            )
