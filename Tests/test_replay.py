import argparse
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

from pydantic import BaseModel

from pet_daemon.commentary.contracts import ModelRequest
from pet_daemon.commentary.usage import UsageTokens, request_usage
from pet_daemon.domain import ModelPurpose, SpeechReason
from pet_daemon.replay import FixtureModel, ReplayHarness, replay_sequence, run

FIXTURES = Path(__file__).parent / "fixtures" / "replay"


class FailingModel(FixtureModel):
    def __init__(self, failed_call: int, error: BaseException | None = None) -> None:
        super().__init__()
        self.speech = "Checked the explorer interactions."
        self.failed_call = failed_call
        self.error = error
        self.calls = 0

    async def complete(
        self,
        request: ModelRequest,
        reason: SpeechReason | ModelPurpose,
        *,
        response_type: type[BaseModel],
    ) -> dict[str, object]:
        self.calls += 1
        usage = request_usage.get()
        assert usage is not None
        usage[str(reason)] = UsageTokens(input_tokens=15, output_tokens=4)
        if self.calls == self.failed_call:
            if self.error is not None:
                raise self.error
            return {"intent": None, "decisions": [], "progress": []}
        return await super().complete(request, reason, response_type=response_type)


class ReplayTests(unittest.IsolatedAsyncioTestCase):
    async def test_model_recording_preserves_application_token_usage(self) -> None:
        async with ReplayHarness(client=FailingModel(0)) as replay:
            await replay.comment(SpeechReason.EXPLICIT, "Check the parser.")
            finished = next(
                event
                for event in replay.events
                if event["event"] == "observation.finished"
            )
            self.assertEqual(finished["input_tokens"], 15)
            self.assertEqual(finished["output_tokens"], 4)
            self.assertEqual(
                replay.model.calls[0]["usage"]["explicit"]["input_tokens"], 15
            )

    async def replay(self, name: str) -> dict[str, Any]:
        return await replay_sequence(
            json.loads((FIXTURES / f"{name}.json").read_text())
        )

    async def test_dry_run_is_deterministic_without_network_calls(self) -> None:
        with patch("anthropic.AsyncAnthropic", side_effect=AssertionError("network")):
            for path in sorted(FIXTURES.glob("*.json")):
                fixture = json.loads(path.read_text())
                first = await replay_sequence(fixture)
                second = await replay_sequence(fixture)
                self.assertEqual(first, second, path.name)
                self.assertTrue(first["steps"])

    async def test_accepted_completed_work_stays_quiet_on_repeat(self) -> None:
        result = await self.replay("repeated_completed")
        self.assertTrue(result["steps"][0]["accepted"])
        repeated = result["steps"][1]
        self.assertEqual(repeated["decision"], "unchanged_work")
        self.assertFalse(repeated["calls"])

    async def test_silence_considers_unchanged_work_without_acceptance(self) -> None:
        fixture = json.loads((FIXTURES / "repeated_completed.json").read_text())
        fixture["events"][0]["candidate_speech"] = None
        result = await replay_sequence(fixture)
        first, repeated = result["steps"]
        self.assertEqual(first["decision"], "silence")
        self.assertFalse(first["accepted"])
        self.assertEqual(repeated["decision"], "unchanged_work")
        self.assertFalse(repeated["calls"])

    async def test_completion_stays_consumed_across_restart_and_closure(self) -> None:
        result = await self.replay("spaced_json_restart")
        first, *later = result["steps"]
        self.assertTrue(first["accepted"])
        for step in later:
            self.assertEqual(step["decision"], "unchanged_work")
            self.assertIsNone(step["speech"])
            self.assertFalse(step["calls"])

    async def test_silent_completion_stays_consumed_after_restart(self) -> None:
        fixture = json.loads((FIXTURES / "spaced_json_restart.json").read_text())
        fixture["events"][0]["candidate_speech"] = None
        result = await replay_sequence(fixture)
        first, *later = result["steps"]
        self.assertEqual(first["decision"], "silence")
        for step in later:
            self.assertFalse(step["calls"])

    async def test_unspoken_completion_expires_at_original_deadline_after_restart(
        self,
    ) -> None:
        result = await self.replay("unspoken_completion_expiry")
        before, at_deadline, after_sleep = result["steps"]
        self.assertEqual(before["decision"], "speech")
        self.assertFalse(before["accepted"])
        for step in (at_deadline, after_sleep):
            self.assertEqual(step["decision"], "unchanged_work")
            self.assertFalse(step["calls"])

    async def test_failed_delivery_leaves_unchanged_work_eligible(self) -> None:
        fixture = json.loads((FIXTURES / "repeated_completed.json").read_text())
        fixture["events"][0]["accepted"] = False
        result = await replay_sequence(fixture)
        first, repeated = result["steps"]
        self.assertFalse(first["accepted"])
        self.assertTrue(repeated["accepted"])
        self.assertEqual(repeated["decision"], "speech")
        self.assertEqual(len(repeated["calls"]), 1)

    async def test_failed_story_retries_unchanged_work_on_the_next_step(self) -> None:
        fixture = json.loads((FIXTURES / "repeated_completed.json").read_text())
        result = await replay_sequence(fixture, client=FailingModel(1))
        failed, retried = result["steps"]
        self.assertEqual(failed["decision"], "deferred")
        self.assertFalse(failed["accepted"])
        self.assertEqual(failed["error"]["type"], "StoryUnavailable")
        self.assertEqual(failed["calls"][0]["response"]["intent"], None)
        self.assertTrue(retried["accepted"])
        self.assertEqual(len(retried["calls"]), 2)

    async def test_failed_speech_keeps_story_and_request_diagnostics(self) -> None:
        fixture = json.loads((FIXTURES / "repeated_completed.json").read_text())
        result = await replay_sequence(
            fixture, client=FailingModel(2, TimeoutError("Request timed out"))
        )
        failed, retried = result["steps"]
        self.assertEqual(failed["decision"], "error")
        self.assertFalse(failed["accepted"])
        story, speech = failed["calls"]
        self.assertIn("response", story)
        self.assertEqual(speech["error"]["type"], "TimeoutError")
        self.assertEqual(speech["usage"]["routine"]["input_tokens"], 15)
        self.assertGreaterEqual(speech["latency_seconds"], 0)
        self.assertTrue(retried["accepted"])
        self.assertEqual(len(retried["calls"]), 1)

    async def test_cancellation_stops_the_replay(self) -> None:
        fixture = json.loads((FIXTURES / "repeated_completed.json").read_text())
        with self.assertRaises(asyncio.CancelledError):
            await replay_sequence(
                fixture, client=FailingModel(1, asyncio.CancelledError())
            )

    async def test_model_run_saves_comparisons_after_an_invalid_story_response(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            args = argparse.Namespace(
                fixtures=FIXTURES / "repeated_completed.json",
                samples=2,
                model_run=True,
                config=None,
                model=None,
                output=destination,
            )
            anthropic = AsyncMock()
            with (
                patch("anthropic.AsyncAnthropic", return_value=anthropic),
                patch(
                    "pet_daemon.replay.AnthropicModelClient",
                    return_value=FailingModel(1),
                ),
                patch("builtins.print"),
            ):
                await run(args)
            results = json.loads((destination / "comparison.json").read_text())
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0]["steps"][0]["decision"], "deferred")
            self.assertTrue(results[0]["steps"][1]["accepted"])
            self.assertTrue(results[1]["steps"][0]["accepted"])
            self.assertIn(
                "StoryUnavailable", (destination / "comparison.md").read_text()
            )
            anthropic.close.assert_awaited_once()

    async def test_accepted_speech_does_not_enter_later_model_inputs(self) -> None:
        result = await self.replay("quoted_speech")
        requests = [call["request"]["messages"] for call in result["steps"][1]["calls"]]
        self.assertNotIn("My mail is haunted", json.dumps(requests))
        self.assertEqual(len(result["steps"][1]["calls"]), 2)

    async def test_reports_exclude_tables_and_preserve_following_prose(self) -> None:
        result = await self.replay("truncated_tables")
        story = result["steps"][0]["calls"][0]
        evidence = json.loads(story["request"]["messages"][0]["content"])
        sources = json.dumps(evidence["current_moment"]["evidence"])
        self.assertNotIn("| Request", sources)
        self.assertIn("Checked both request builders.", sources)
