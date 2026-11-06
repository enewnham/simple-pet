import asyncio
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path

import pytest
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    Message,
    ResultMessage,
    StreamEvent,
    SystemMessage,
    ToolUseBlock,
)
from structlog.testing import capture_logs

from pet_daemon.companion.config import CoordinatorConfig
from pet_daemon.companion.sdk import SdkCompanion
from pet_daemon.companion.usage import SdkUsage
from pet_daemon.coordination.models import Candidate, InquiryPass
from pet_daemon.storage import Memory
from Tests.test_companion import FakeSdk, history, user_turn


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class MeteredSdk(FakeSdk):
    def __init__(self, mode: str) -> None:
        super().__init__()
        self.mode = mode
        self.invented_source = mode == "invalid_citation"

    async def __call__(
        self,
        *,
        prompt: str | AsyncIterator[dict[str, object]],
        options: ClaudeAgentOptions,
    ) -> AsyncIterator[Message]:
        if self.mode == "billing":
            yield AssistantMessage(
                [], "fixture", error="billing_error", message_id="billing-response"
            )
            return
        first = {
            "input_tokens": 10,
            "output_tokens": 1,
            "cache_creation_input_tokens": 100,
            "cache_read_input_tokens": 0,
        }
        block = ToolUseBlock(
            "tool-id", "context_search", {"query": "PRIVATE_SOURCE_TEXT"}
        )
        yield AssistantMessage([block], "fixture", usage=first, message_id="first")
        yield AssistantMessage(
            [block], "fixture", usage={**first, "output_tokens": 5}, message_id="first"
        )
        if self.mode == "transport":
            raise OSError("Offline provider transport failure")
        yield SystemMessage("api_retry", {"attempt": 1, "error_status": 429})
        second = {
            "input_tokens": 2,
            "output_tokens": 7,
            "cache_read_input_tokens": 150,
            "cache_creation_input_tokens": 0,
        }
        yield AssistantMessage([], "fixture", usage=second, message_id="second")
        async for result in super().__call__(prompt=prompt, options=options):
            assert isinstance(result, ResultMessage)
            if (
                self.mode == "invalid_citation"
                and "decisions" in result.structured_output
            ):
                result.structured_output["decisions"][0]["source_ids"] = ["invented"]
            yield replace(
                result,
                usage={
                    "input_tokens": 12,
                    "output_tokens": 12,
                    "cache_read_input_tokens": 150,
                    "cache_creation_input_tokens": 100,
                },
                total_cost_usd=0.001,
            )


@pytest.mark.anyio
@pytest.mark.parametrize("purpose", ["conversation", "inquiry"])
@pytest.mark.parametrize(
    "mode", ["accepted", "invalid_citation", "billing", "transport"]
)
async def test_sdk_usage_survives_acceptance_validation_and_provider_failures_without_double_counting(
    tmp_path: Path, purpose: str, mode: str
) -> None:
    memory = Memory(tmp_path / "memory.sqlite3", "fixture", clock=lambda: 100)
    history(memory)
    model = MeteredSdk(mode)
    companion = SdkCompanion(
        CoordinatorConfig(working_directory=tmp_path / "sdk"),
        memory,
        lambda: 100,
        query=model,
    )
    message = user_turn(memory)
    candidate = Candidate("usage-candidate", "Review retrieval", "/project", 100)
    memory.coordination.save_candidate(candidate)
    with capture_logs() as logs:
        call = (
            companion.reply(
                message.conversation_id,
                memory.coordination.messages(message.conversation_id),
                message.id,
            )
            if purpose == "conversation"
            else companion.investigate(
                InquiryPass(
                    "usage-pass",
                    (candidate,),
                    memory.coordination.state().instruction_revision,
                    candidate.id,
                )
            )
        )
        if mode == "accepted":
            await call
        else:
            with pytest.raises((ValueError, RuntimeError, OSError)):
                await call
        if purpose == "inquiry":
            companion.record_inquiry_outcome(
                "usage-pass",
                "accepted" if mode == "accepted" else "rejected",
                None if mode == "accepted" else mode,
            )
    summaries = [item for item in logs if item["event"] == "companion.usage"]
    assert len(summaries) == 1
    usage = summaries[0]
    assert usage["purpose"] == purpose
    assert usage["outcome"] == ("sdk_validated" if mode == "accepted" else "rejected")
    assert usage["request_bytes"] > usage["user_bytes"] > 0
    assert (
        usage["system_bytes"] and usage["tool_schema_bytes"] and usage["schema_bytes"]
    )
    assert usage["estimate_method"] == "utf8_bytes_div_4_excludes_sdk_overhead"
    if mode == "billing":
        assert usage["error_kind"] == "billing_error"
        assert usage["usage_known"] is False and usage["total_tokens"] is None
    elif mode == "transport":
        assert usage["usage_known"] is True and usage["usage_complete"] is False
        assert usage["output_tokens"] == 5 and usage["cache_write_tokens"] == 100
    else:
        assert usage["usage_source"] == "provider_result"
        assert (
            usage["input_tokens"],
            usage["output_tokens"],
            usage["cache_read_tokens"],
            usage["cache_write_tokens"],
        ) == (12, 12, 150, 100)
        assert usage["total_context_tokens"] == 262 and usage["total_tokens"] == 274
        assert usage["model_calls"] == 2 and usage["retries"] == 1
        assert usage["tool_calls"] == 1
        calls = [item for item in logs if item["event"] == "companion.model_call"]
        assert len(calls) == 3 and calls[1]["update"] is True
    assert "PRIVATE_SOURCE_TEXT" not in str(logs)
    if purpose == "inquiry":
        outcome = next(item for item in logs if item["event"] == "companion.outcome")
        assert outcome["request_id"] == usage["request_id"] == "usage-pass"
    await companion.close()
    memory.close()


@pytest.mark.anyio
async def test_cancelled_sdk_work_retains_observed_call_usage_and_candidate_attribution(
    tmp_path: Path,
) -> None:
    memory = Memory(tmp_path / "memory.sqlite3", "fixture", clock=lambda: 100)
    history(memory)
    model = MeteredSdk("accepted")
    model.release.clear()
    companion = SdkCompanion(
        CoordinatorConfig(working_directory=tmp_path / "sdk"),
        memory,
        lambda: 100,
        query=model,
    )
    candidate = Candidate("cancelled-candidate", "Review retrieval", "/project", 100)
    memory.coordination.save_candidate(candidate)
    with capture_logs() as logs:
        task = asyncio.create_task(
            companion.investigate(
                InquiryPass("cancelled-pass", (candidate,), 0, candidate.id)
            )
        )
        await model.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    usage = next(item for item in logs if item["event"] == "companion.usage")
    assert usage["outcome"] == "cancelled" and usage["focus_id"] == candidate.id
    assert usage["input_tokens"] == 12 and usage["output_tokens"] == 12
    assert usage["usage_complete"] is False
    await companion.close()
    memory.close()


@pytest.mark.anyio
@pytest.mark.parametrize("purpose", ["conversation", "inquiry"])
async def test_streamed_usage_is_retained_when_cancelled_before_an_assistant_message(
    tmp_path: Path, purpose: str
) -> None:
    memory = Memory(tmp_path / "memory.sqlite3", "fixture", clock=lambda: 100)
    started = asyncio.Event()

    async def stream(
        *, prompt: str | AsyncIterator[dict[str, object]], options: ClaudeAgentOptions
    ) -> AsyncIterator[Message]:
        assert options.include_partial_messages
        yield StreamEvent(
            "start",
            "stream-session",
            {
                "type": "message_start",
                "message": {
                    "id": "actual-api-call",
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 1,
                        "cache_read_input_tokens": 20,
                        "cache_creation_input_tokens": 100,
                    },
                },
            },
        )
        yield StreamEvent(
            "delta",
            "stream-session",
            {"type": "message_delta", "usage": {"output_tokens": 9}},
        )
        started.set()
        await asyncio.Event().wait()

    companion = SdkCompanion(
        CoordinatorConfig(working_directory=tmp_path / "sdk"),
        memory,
        lambda: 100,
        query=stream,
    )
    message = user_turn(memory)
    candidate = Candidate("stream-candidate", "Review", "/project", 100)
    memory.coordination.save_candidate(candidate)
    with capture_logs() as logs:
        call = (
            companion.reply(
                message.conversation_id,
                memory.coordination.messages(message.conversation_id),
                message.id,
            )
            if purpose == "conversation"
            else companion.investigate(
                InquiryPass(
                    "stream-pass",
                    (candidate,),
                    memory.coordination.state().instruction_revision,
                    candidate.id,
                )
            )
        )
        task = asyncio.create_task(call)
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    usage = next(item for item in logs if item["event"] == "companion.usage")
    assert usage["outcome"] == "cancelled" and usage["model_calls"] == 1
    assert usage["usage_complete"] is False
    assert (
        usage["input_tokens"],
        usage["output_tokens"],
        usage["cache_read_tokens"],
        usage["cache_write_tokens"],
    ) == (10, 9, 20, 100)
    calls = [item for item in logs if item["event"] == "companion.model_call"]
    assert len(calls) == 2 and {item["call_id"] for item in calls} == {
        "actual-api-call"
    }
    await companion.close()
    memory.close()


def test_per_model_provider_totals_supply_actual_usage_without_a_result_usage_field() -> (
    None
):
    usage = SdkUsage("model-usage", "inquiry", "fixture")
    usage.receive(
        ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id="fixture",
            model_usage={
                "fixture": {
                    "inputTokens": 10,
                    "outputTokens": 5,
                    "cacheReadInputTokens": 20,
                    "cacheCreationInputTokens": 100,
                    "webSearchRequests": 0,
                    "costUSD": 0.001,
                    "contextWindow": 1000,
                    "maxOutputTokens": 100,
                }
            },
        )
    )
    with capture_logs() as logs:
        usage.finish("sdk_validated")
    summary = logs[0]
    assert summary["usage_source"] == "provider_model_usage"
    assert summary["total_context_tokens"] == 130 and summary["total_tokens"] == 135
