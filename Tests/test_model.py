import asyncio
import json
import unittest
from typing import cast

import anthropic
import httpx2 as httpx
from anthropic import transform_schema

from pet_daemon.commentary import reaction
from pet_daemon.commentary.contracts import ModelRequest
from pet_daemon.commentary.model import AnthropicModelClient
from pet_daemon.domain import SpeechReason
from pet_daemon.types import JSONObject


class ModelTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.requests: list[httpx.Request] = []
        self.responses: list[httpx.Response | Exception] = []
        self.client = anthropic.AsyncAnthropic(
            api_key="test-key",
            max_retries=0,
            http_client=httpx.AsyncClient(
                transport=httpx.MockTransport(self.transport)
            ),
        )
        self.model = AnthropicModelClient(self.client)

    async def asyncTearDown(self) -> None:
        await self.client.close()

    def transport(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def request(self, timeout: float = 2) -> ModelRequest:
        return {
            "model": "claude-haiku-5-5",
            "max_tokens": 512,
            "timeout": timeout,
            "thinking": {"type": "disabled"},
            "system": [{"type": "text", "text": "React briefly."}],
            "messages": [{"role": "user", "content": "A check passed."}],
            "output_config": {
                "effort": "low",
                "format": {
                    "type": "json_schema",
                    "schema": cast(JSONObject, transform_schema(reaction.Response)),
                },
            },
        }

    def message(
        self,
        text: str = '{"speak":"That check passed.","mentions":[]}',
        *,
        stop: str = "end_turn",
    ) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"request-id": "req_success"},
            json={
                "id": "msg_fixture",
                "type": "message",
                "role": "assistant",
                "model": "claude-haiku-5-5",
                "content": [{"type": "text", "text": text}],
                "stop_reason": stop,
                "stop_sequence": None,
                "usage": {"input_tokens": 50, "output_tokens": 10},
            },
        )

    def unavailable(self, status: int = 503, delay: str = "1") -> httpx.Response:
        return httpx.Response(
            status,
            headers={"retry-after-ms": delay, "request-id": "req_failure"},
            json={
                "type": "error",
                "error": {"type": "api_error", "message": "Unavailable"},
            },
        )

    async def test_click_recovers_from_one_transient_failure(self) -> None:
        for failure in (
            self.unavailable(),
            self.unavailable(429),
            httpx.ConnectError("Connection interrupted"),
        ):
            with self.subTest(failure=type(failure).__name__):
                self.requests.clear()
                self.responses = [failure, self.message()]
                result = await self.model.complete(
                    self.request(), SpeechReason.CLICK, response_type=reaction.Response
                )
                self.assertEqual(
                    result, {"speak": "That check passed.", "mentions": []}
                )
                self.assertEqual(len(self.requests), 2)
                self.assertEqual(self.requests[0].content, self.requests[1].content)

    async def test_automatic_reaction_drops_transient_failure(self) -> None:
        self.responses = [self.unavailable(), self.message()]
        with self.assertRaises(anthropic.APIStatusError):
            await self.model.complete(
                self.request(), SpeechReason.ROUTINE, response_type=reaction.Response
            )
        self.assertEqual(len(self.requests), 1)

    async def test_click_stops_after_one_retry(self) -> None:
        self.responses = [self.unavailable(), self.unavailable(), self.message()]
        with self.assertRaises(anthropic.APIStatusError):
            await self.model.complete(
                self.request(), SpeechReason.CLICK, response_type=reaction.Response
            )
        self.assertEqual(len(self.requests), 2)

    async def test_request_deadline_includes_retry_backoff(self) -> None:
        self.responses = [self.unavailable(delay="1000"), self.message()]
        async with asyncio.timeout(0.5):
            with self.assertRaises(TimeoutError):
                await self.model.complete(
                    self.request(timeout=0.03),
                    SpeechReason.CLICK,
                    response_type=reaction.Response,
                )
        self.assertEqual(len(self.requests), 1)

    async def test_refused_and_truncated_responses_have_distinct_failures(self) -> None:
        for stop, diagnostic in (("refusal", "refused"), ("max_tokens", "token limit")):
            with self.subTest(stop=stop):
                self.requests.clear()
                self.responses = [
                    self.message(stop=stop),
                    self.message(),
                ]
                with self.assertRaisesRegex(ValueError, diagnostic):
                    await self.model.complete(
                        self.request(),
                        SpeechReason.CLICK,
                        response_type=reaction.Response,
                    )
                self.assertEqual(len(self.requests), 1)

    async def test_invalid_json_is_rejected_without_retry(self) -> None:
        for text, stop in (
            ("Not JSON", "end_turn"),
            ("[]", "end_turn"),
            ("Refused", "refusal"),
            ('{"speak":', "max_tokens"),
        ):
            with self.subTest(text=text, stop=stop):
                self.requests.clear()
                self.responses = [self.message(text, stop=stop), self.message()]
                with self.assertRaises(ValueError):
                    await self.model.complete(
                        self.request(),
                        SpeechReason.CLICK,
                        response_type=reaction.Response,
                    )
                self.assertEqual(len(self.requests), 1)

    async def test_sends_schema_and_logs_success_request_id(self) -> None:
        request = self.request()
        self.responses = [self.message()]
        with self.assertLogs("pet-daemon", level="DEBUG") as captured:
            result = await self.model.complete(
                request, SpeechReason.ROUTINE, response_type=reaction.Response
            )
        self.assertEqual(result["speak"], "That check passed.")
        body = json.loads(self.requests[0].content)
        self.assertEqual(body["output_config"], request["output_config"])
        self.assertTrue(
            any(
                isinstance(record.msg, dict)
                and record.msg.get("request_id") == "req_success"
                for record in captured.records
            )
        )

    async def test_schema_violations_are_rejected_without_retry(self) -> None:
        for result in ({"speak": 123}, {"speak": "Hello", "command": "open"}):
            with self.subTest(result=result):
                self.requests.clear()
                self.responses = [self.message(json.dumps(result)), self.message()]
                with self.assertRaises(ValueError):
                    await self.model.complete(
                        self.request(),
                        SpeechReason.CLICK,
                        response_type=reaction.Response,
                    )
                self.assertEqual(len(self.requests), 1)

    async def test_logs_request_id_for_http_failure(self) -> None:
        self.responses = [self.unavailable()]
        with (
            self.assertLogs("pet-daemon", level="WARNING") as captured,
            self.assertRaises(anthropic.APIStatusError),
        ):
            await self.model.complete(
                self.request(), SpeechReason.ROUTINE, response_type=reaction.Response
            )
        self.assertTrue(
            any(
                isinstance(record.msg, dict)
                and record.msg.get("request_id") == "req_failure"
                for record in captured.records
            )
        )
