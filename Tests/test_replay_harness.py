import asyncio
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from pet_daemon.domain import DeliveryStatus, ProcessStatus, SpeechReason
from pet_daemon.replay import FixtureModel, ReplayClock, ReplayHarness
from pet_daemon.runtime import ProcessIdentity


def test_scheduler_persists_only_renderer_accepted_speech() -> None:
    async def scenario() -> None:
        async with ReplayHarness() as replay:
            assert replay.fixture_model is not None
            replay.fixture_model.speech = (
                "I am checking how the parser handles escapes."
            )
            replay.ingest(
                {
                    "harness": "codex",
                    "session": "parser",
                    "event": "intent",
                    "goal": "Check how the parser handles escaped delimiters.",
                }
            )
            await replay.advance(0)
            packet = next(item for item in replay.packets if "delivery_id" in item)
            identifier = packet["delivery_id"]
            assert replay.memory.message_outcome(identifier) is None
            assert replay.app.delivery.has_pending()
            assert [call["purpose"] for call in replay.model.calls] == [
                "story",
                "prompt",
            ]
            payload = json.loads(
                replay.model.calls[-1]["request"]["messages"][0]["content"]
            )
            assert payload["current_moment"]["kind"] == "request_received"
            assert any(
                item.get("animation", {}).get("id") == "review"
                for item in replay.packets
            )

            replay.receive(
                {
                    "kind": "renderer",
                    "event": "speech_accepted",
                    "delivery_id": identifier,
                }
            )
            outcome = replay.memory.message_outcome(identifier)
            assert outcome is not None
            assert outcome.outcome == DeliveryStatus.ACCEPTED
            assert outcome.text == replay.fixture_model.speech
            assert not replay.app.delivery.has_pending()
            assert replay.memory.speech()[0].text == outcome.text
            await replay.restart()
            assert replay.memory.message_outcome(identifier) == outcome
            assert replay.memory.load_attention_history()

    asyncio.run(scenario())


def test_replay_nap_configuration_changes_actual_renderer_actions() -> None:
    async def scenario(naps: bool) -> list[dict[str, object]]:
        async with ReplayHarness(seed=12) as replay:
            behavior = replay.config.behavior
            behavior.naps = naps
            behavior.roaming = False
            behavior.care = False
            behavior.nap_interval = (1, 1)
            behavior.nap_seconds = (20, 20)
            await replay.restart()
            await replay.advance(240, tick_seconds=1)
            result = replay.result()
            assert result["model_mode"] == "fixture"
            assert result["calls"] == []
            assert any(
                event["event"] == "behavior.state_changed" for event in result["events"]
            )
            return result["packets"]

    baseline = asyncio.run(scenario(False))
    candidate = asyncio.run(scenario(True))
    assert not any(packet.get("sleeping") is True for packet in baseline)
    assert any(packet.get("sleeping") is True for packet in candidate)
    assert candidate == asyncio.run(scenario(True))


def test_process_evidence_is_injected_without_host_probes() -> None:
    async def scenario() -> None:
        async with ReplayHarness() as replay:
            identity = ProcessIdentity(43210, "replayed-process")
            replay.processes.statuses[identity] = ProcessStatus.ALIVE
            replay.ingest(
                {
                    "harness": "codex",
                    "session": "parser",
                    "event": "intent",
                    "goal": "Check the parser",
                    "owner": {"pid": identity.pid, "started": identity.started},
                }
            )
            with patch(
                "pet_daemon.sessions.processes.probe",
                side_effect=AssertionError("host probe"),
            ):
                await replay.advance(0)
                assert (
                    replay.app.sessions.by_id["codex:parser"].presence
                    == ProcessStatus.ALIVE
                )
                replay.processes.statuses[identity] = ProcessStatus.EXITED
                await replay.advance(1)
                assert (
                    replay.app.sessions.by_id["codex:parser"].presence
                    == ProcessStatus.ALIVE
                )
                await replay.advance(replay.config.process_probe_seconds - 1)
                assert (
                    replay.app.sessions.by_id["codex:parser"].presence
                    == ProcessStatus.EXITED
                )

    asyncio.run(scenario())


def test_failed_delivery_uses_retry_and_failure_storage() -> None:
    async def scenario() -> None:
        async with ReplayHarness() as replay:
            assert replay.fixture_model is not None
            replay.fixture_model.speech = "Hello from a replay."
            await replay.comment(SpeechReason.EXPLICIT, "Say hello")
            packet = next(item for item in replay.packets if "delivery_id" in item)
            identifier = packet["delivery_id"]
            await replay.advance(5)
            outcome = replay.memory.message_outcome(identifier)
            assert outcome is not None
            assert outcome.outcome == DeliveryStatus.FAILED
            attempts = [
                item for item in replay.packets if item.get("delivery_id") == identifier
            ]
            assert len(attempts) == 2
            assert replay.memory.speech() == []
            assert any(
                event["event"] == "speech.delivery" and event["outcome"] == "failed"
                for event in replay.events
            )

    asyncio.run(scenario())


def test_renderer_sleep_and_wake_drive_application_suspension() -> None:
    async def scenario() -> None:
        async with ReplayHarness() as replay:
            replay.receive({"kind": "renderer", "event": "system_sleep"})
            assert replay.app.suspended
            await replay.advance(2, wall_seconds=3600)
            assert replay.clock.now == 1002
            assert replay.clock.wall_now == 4600
            assert not replay.model.calls
            replay.receive({"kind": "renderer", "event": "system_wake"})
            assert not replay.app.suspended
            assert [
                event["event"]
                for event in replay.events
                if event["event"] in {"system.sleep", "system.wake"}
            ] == ["system.sleep", "system.wake"]

    asyncio.run(scenario())


def test_injected_client_uses_current_prompt_and_configured_pack(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        async with ReplayHarness() as source:
            client = FixtureModel()
            client.speech = "I am trying the configured persona."
            config = replace(source.config, persona="Use a quiet botanical voice.")
            async with ReplayHarness(
                config, client, memory_file=tmp_path / "experiment.sqlite3"
            ) as replay:
                await replay.comment(SpeechReason.EXPLICIT, "A parser check passed.")
                request = replay.model.calls[-1]["request"]
                assert "Use a quiet botanical voice." in request["system"][0]["text"]
                assert (
                    json.loads(request["messages"][0]["content"])["current_moment"][
                        "text"
                    ]
                    == "A parser check passed."
                )
                assert replay.result()["model_mode"] == "provided_client"
                assert replay.pack.directory == source.pack.directory

    asyncio.run(scenario())


def test_invalid_clock_steps_fail_before_mutating_time() -> None:
    clock = ReplayClock()
    for seconds in (-1, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            clock.advance(seconds)
        assert clock() == 1000
