import asyncio
import json
import logging
from pathlib import Path

import anthropic
import httpx2 as httpx
import pytest

from pet_daemon.commentary import care
from pet_daemon.commentary.cognition import Cognition
from pet_daemon.commentary.model import AnthropicModelClient
from pet_daemon.commentary.usage import collect_usage
from pet_daemon.config import Config
from pet_daemon.domain import SpeechReason
from pet_daemon.storage import Memory


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
@pytest.mark.parametrize(
    "outcome",
    [
        "completed",
        "invalid_response",
        "incomplete",
        "refused",
        "failed",
        "timeout",
        "cancelled",
    ],
)
async def test_usage_is_retained_for_every_model_completion_path(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, outcome: str
) -> None:
    entered = asyncio.Event()
    requests = []

    async def transport(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        entered.set()
        if outcome in ("timeout", "cancelled"):
            await asyncio.Event().wait()
        if outcome == "failed":
            return httpx.Response(
                402,
                json={
                    "type": "error",
                    "error": {"type": "billing_error", "message": "Test billing error"},
                },
            )
        return httpx.Response(
            200,
            headers={"request-id": "request-fixture"},
            json={
                "id": "message-fixture",
                "type": "message",
                "role": "assistant",
                "model": "fixture",
                "content": [
                    {
                        "type": "text",
                        "text": "invalid json"
                        if outcome == "invalid_response"
                        else '{"speak":null}',
                    }
                ],
                "stop_reason": "max_tokens"
                if outcome == "incomplete"
                else "refusal"
                if outcome == "refused"
                else "end_turn",
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "cache_read_input_tokens": 50,
                    "cache_creation_input_tokens": 10,
                },
            },
        )

    memory = Memory(tmp_path / "memory.sqlite3", "usage-pet")
    client = anthropic.AsyncAnthropic(
        api_key="fixture",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(transport)),
    )
    try:
        model = AnthropicModelClient(client)
        cognition = Cognition(model, Config(pet_dir=tmp_path), memory=memory)
        request = cognition.request(
            "Optional quiet remark",
            [{"type": "text", "text": "Keep it short."}],
            response=care.Response,
        )
        if outcome == "timeout":
            request["timeout"] = 0.01
        caplog.set_level(logging.INFO, logger="pet-daemon")
        with collect_usage() as usage:
            if outcome == "cancelled":
                task = asyncio.create_task(
                    model.complete(
                        request, SpeechReason.CARE, response_type=care.Response
                    )
                )
                await entered.wait()
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            elif outcome == "completed":
                assert await model.complete(
                    request, SpeechReason.CARE, response_type=care.Response
                ) == {"speak": None}
            else:
                with pytest.raises(
                    (ValueError, anthropic.APIStatusError, TimeoutError)
                ):
                    await model.complete(
                        request, SpeechReason.CARE, response_type=care.Response
                    )
        summary = usage[SpeechReason.CARE]
        assert len(requests) == summary.model_calls == 1
        assert summary.retries == 0
        assert summary.request_bytes > summary.system_bytes > 0
        assert summary.user_bytes > 0
        logs = [
            record.msg
            for record in caplog.records
            if isinstance(record.msg, dict)
            and record.msg.get("event") == "observation.usage"
        ]
        assert len(logs) == 1 and logs[0]["outcome"] == outcome
        assert logs[0]["request_id"]
        if outcome in ("completed", "invalid_response", "incomplete", "refused"):
            assert summary.input_tokens == 100
            assert summary.cache_read_tokens == 50
            assert summary.cache_write_tokens == 10
            assert summary.total_context_tokens == 160
            assert summary.total_tokens == 180
        else:
            assert summary.input_tokens is None
            assert summary.total_context_tokens is None
            assert summary.total_tokens is None
    finally:
        memory.close()
        await client.close()
