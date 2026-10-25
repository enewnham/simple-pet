import asyncio
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from unittest.mock import patch

import pytest

from pet_daemon.cli.hook import update as hook_update
from pet_daemon.domain import (
    DeliveryStatus,
    ExecutionOutcome,
    ProcessStatus,
    SessionSource,
    SpeechReason,
)
from pet_daemon.identity import SessionIdentity
from pet_daemon.replay import FixtureModel, ReplayClock, ReplayHarness
from pet_daemon.runtime import HookEvent, ProcessIdentity, WaitingKind
from pet_daemon.work.evidence import ExecutionStatus
from pet_daemon.work.models import ExecutionEvidence, ProgressEvidence


def test_scheduler_persists_only_renderer_accepted_speech() -> None:
    async def scenario() -> None:
        async with ReplayHarness() as replay:
            assert replay.fixture_model is not None
            replay.fixture_model.speech = (
                "I am checking how the parser handles escapes."
            )
            replay.ingest(
                {
                    "source": "codex",
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
            assert (
                payload["current_moment"]["event"]
                == "The developer sent a message about this work."
            )
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
            identity = ProcessIdentity(pid=43210, started="replayed-process")
            replay.processes.statuses[identity] = ProcessStatus.ALIVE
            replay.ingest(
                {
                    "source": "codex",
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
                    replay.app.sessions.by_id[
                        SessionIdentity(source=SessionSource.CODEX, native_id="parser")
                    ].presence
                    == ProcessStatus.ALIVE
                )
                replay.processes.statuses[identity] = ProcessStatus.EXITED
                await replay.advance(1)
                assert (
                    replay.app.sessions.by_id[
                        SessionIdentity(source=SessionSource.CODEX, native_id="parser")
                    ].presence
                    == ProcessStatus.ALIVE
                )
                await replay.advance(replay.config.process_probe_seconds - 1)
                assert (
                    replay.app.sessions.by_id[
                        SessionIdentity(source=SessionSource.CODEX, native_id="parser")
                    ].presence
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
            config = source.config.model_copy(
                update={"persona": "Use a quiet botanical voice."}
            )
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


@pytest.fixture
def claude_hook() -> Callable[[str, Mapping[str, object]], HookEvent]:
    def update(event: str, payload: Mapping[str, object]) -> HookEvent:
        hook = hook_update("claude", event, payload)
        assert isinstance(hook, HookEvent)
        return hook

    return update


@pytest.mark.parametrize(
    ("event", "extra", "waiting_kind", "moment_event"),
    [
        (
            "PermissionRequest",
            {},
            WaitingKind.PERMISSION_CHECK,
            "An observation of the selected work is available.",
        ),
        (
            "Notification",
            {"notification_type": "permission_prompt"},
            WaitingKind.PERMISSION,
            "The work is waiting for the developer to approve an operation.",
        ),
        (
            "PreToolUse",
            {
                "tool_name": "AskUserQuestion",
                "tool_input": {
                    "questions": [{"question": "Which parser should I check?"}]
                },
            },
            WaitingKind.QUESTION,
            "The work is waiting for an answer to a question.",
        ),
    ],
)
def test_wait_kind_controls_scheduled_and_clicked_commentary(
    claude_hook: Callable[[str, Mapping[str, object]], HookEvent],
    event: str,
    extra: Mapping[str, object],
    waiting_kind: WaitingKind,
    moment_event: str,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        async with ReplayHarness() as replay:
            replay.config.idle_message_seconds = 10
            replay.config.observation_seconds = (1000, 1000)
            replay.config.behavior.automatic_speech_cooldown_seconds = 0
            replay.config.behavior.care = False
            replay.config.behavior.naps = False
            replay.config.behavior.roaming = False
            await replay.restart()
            replay.ingest(
                claude_hook(
                    "UserPromptSubmit",
                    {"session_id": "parser", "prompt": "Check parser behavior"},
                ).packet()
            )
            replay.ingest(
                claude_hook(
                    "PreToolUse",
                    {
                        "session_id": "parser",
                        "tool_use_id": "check",
                        "tool_name": "Bash",
                        "tool_input": {"command": "pytest Tests/test_parser.py"},
                    },
                ).packet()
            )
            await replay.advance(0)
            calls_before_wait = len(replay.model.calls)
            replay.ingest(
                claude_hook(
                    event,
                    {
                        "session_id": "parser",
                        "tool_use_id": "check",
                        "tool_name": "Bash",
                        "tool_input": {"command": "pytest Tests/test_parser.py"},
                        **extra,
                    },
                ).packet()
            )
            await replay.advance(0)
            session = replay.app.sessions.by_id[
                SessionIdentity(source=SessionSource.CLAUDE, native_id="parser")
            ]
            assert session.waiting
            assert not session.is_working
            assert session.waiting_kind == waiting_kind
            assert session.is_waiting_for_input == (
                waiting_kind != WaitingKind.PERMISSION_CHECK
            )
            assert any(
                packet.get("animation", {}).get("id") == "waiting"
                for packet in replay.packets
            )
            if waiting_kind == WaitingKind.PERMISSION_CHECK:
                assert len(replay.model.calls) == calls_before_wait
            else:
                assert replay.model.calls[-1]["purpose"] == "check_in"
                payload = json.loads(
                    replay.model.calls[-1]["request"]["messages"][0]["content"]
                )
                assert payload["current_moment"]["event"] == moment_event
            replay.receive({"kind": "renderer", "event": "click"})
            await replay.advance(0)
            assert replay.model.calls[-1]["purpose"] == "click"
            payload = json.loads(
                replay.model.calls[-1]["request"]["messages"][0]["content"]
            )
            assert payload["current_moment"]["event"] == moment_event
            conversation = next(
                packet["conversation"]
                for packet in reversed(replay.packets)
                if "conversation" in packet
            )
            assert isinstance(conversation, dict)
            replay.receive(
                {
                    "kind": "renderer",
                    "event": "interaction_dismissed",
                    "interaction_id": conversation["id"],
                    "revision": conversation["revision"],
                }
            )
            await replay.advance(10, tick_seconds=1)
            assert replay.model.calls[-1]["purpose"] == "check_in"
            payload = json.loads(
                replay.model.calls[-1]["request"]["messages"][0]["content"]
            )
            assert payload["current_moment"]["event"] == moment_event
            if waiting_kind == WaitingKind.QUESTION:
                assert (
                    payload["current_moment"]["reported_progress"]
                    == "Which parser should I check?"
                )
            activity = replay.app.context().work[0].story.activity
            assert any(
                isinstance(item.details, ExecutionStatus)
                and item.details.execution == "waiting"
                for item in activity
            ) == (waiting_kind != WaitingKind.PERMISSION_CHECK)
            replay.ingest(
                claude_hook(
                    "PostToolUse",
                    {
                        "session_id": "parser",
                        "tool_use_id": "check",
                        "tool_name": "Bash",
                    },
                ).packet()
            )
            assert not session.waiting
            reports = replay.memory.narrative_reports(
                (SessionIdentity(source=SessionSource.CLAUDE, native_id="parser"),)
            )
            transitions = [
                report.payload.event
                for report in reports
                if isinstance(report.payload, ExecutionEvidence)
            ]
            assert ("waiting" in transitions) == (
                waiting_kind != WaitingKind.PERMISSION_CHECK
            )
            assert ("resumed" in transitions) == (
                waiting_kind != WaitingKind.PERMISSION_CHECK
            )
            replay.ingest(
                claude_hook(
                    "Stop",
                    {
                        "session_id": "parser",
                        "last_assistant_message": "Checked parser behavior and confirmed escaped delimiters.",
                    },
                ).packet()
            )
            await replay.advance(1)
            assert session.outcome == ExecutionOutcome.SUCCESS
            assert replay.model.calls[-1]["purpose"] == "completion"
            payload = json.loads(
                replay.model.calls[-1]["request"]["messages"][0]["content"]
            )
            assert (
                payload["current_moment"]["event"] == "The current work segment ended."
            )
            assert (
                payload["current_moment"]["execution"]
                == "The work segment ended without a reported execution error. This alone does not establish task completion."
            )
            reports = replay.memory.narrative_reports(
                (SessionIdentity(source=SessionSource.CLAUDE, native_id="parser"),)
            )
            assert any(
                isinstance(report.payload, ProgressEvidence)
                and report.payload.text
                == "Checked parser behavior and confirmed escaped delimiters."
                for report in reports
            )
            (tmp_path / "replay.json").write_text(json.dumps(replay.result(), indent=2))

    asyncio.run(scenario())


@pytest.mark.parametrize("internal_check", [True, False])
def test_child_check_ins_follow_requests_for_input(
    claude_hook: Callable[[str, Mapping[str, object]], HookEvent],
    internal_check: bool,
) -> None:
    async def scenario() -> None:
        async with ReplayHarness() as replay:
            replay.config.idle_message_seconds = 10
            replay.config.observation_seconds = (1000, 1000)
            replay.config.behavior.automatic_speech_cooldown_seconds = 0
            replay.config.behavior.care = False
            replay.config.behavior.naps = False
            await replay.restart()
            replay.ingest(
                claude_hook(
                    "UserPromptSubmit",
                    {"session_id": "parser", "prompt": "Check parser behavior"},
                ).packet()
            )
            await replay.advance(0)
            replay.ingest(
                claude_hook(
                    "SubagentStart",
                    {
                        "session_id": "parser",
                        "agent_id": "examples",
                        "task": "Check escaped delimiters",
                    },
                ).packet()
            )
            replay.ingest(
                claude_hook(
                    "PermissionRequest" if internal_check else "Notification",
                    {
                        "session_id": "parser",
                        "agent_id": "examples",
                        "tool_use_id": "check",
                        "tool_name": "Bash",
                        "notification_type": "permission_prompt",
                    },
                ).packet()
            )
            await replay.advance(0)
            calls_after_wait = len(replay.model.calls)
            await replay.advance(9, tick_seconds=1)
            replay.ingest(
                claude_hook(
                    "PreToolUse",
                    {
                        "session_id": "parser",
                        "tool_use_id": "parent-check",
                        "tool_name": "Bash",
                    },
                ).packet()
            )
            await replay.advance(1)
            child = replay.app.sessions.by_id[
                SessionIdentity(source=SessionSource.CLAUDE, native_id="examples")
            ]
            if internal_check:
                assert len(replay.model.calls) == calls_after_wait
                assert child.last_check_in is None
            else:
                assert replay.model.calls[-1]["purpose"] == "check_in"
                assert child.last_check_in == replay.clock()
                payload = json.loads(
                    replay.model.calls[-1]["request"]["messages"][0]["content"]
                )
                assert (
                    payload["current_moment"]["event"]
                    == "The work is waiting for the developer to approve an operation."
                )
                assert payload["current_work"]["request"] == "Check escaped delimiters"

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", [False, True])
def test_internal_checks_retain_terminal_results_and_runtime_evidence(
    claude_hook: Callable[[str, Mapping[str, object]], HookEvent],
    failure: bool,
) -> None:
    async def scenario() -> None:
        async with ReplayHarness() as replay:
            replay.config.behavior.automatic_speech_cooldown_seconds = 0
            replay.ingest(
                claude_hook(
                    "UserPromptSubmit",
                    {"session_id": "parser", "prompt": "Check parser behavior"},
                ).packet()
            )
            await replay.advance(0)
            replay.ingest(
                claude_hook(
                    "PermissionRequest",
                    {
                        "session_id": "parser",
                        "tool_use_id": "check",
                        "tool_name": "Bash",
                    },
                ).packet()
            )
            await replay.advance(3)
            replay.ingest(
                claude_hook(
                    "StopFailure" if failure else "Stop",
                    {
                        "session_id": "parser",
                        "last_assistant_message": "Checked parser behavior.",
                    },
                ).packet()
            )
            await replay.advance(1)
            session = replay.app.sessions.by_id[
                SessionIdentity(source=SessionSource.CLAUDE, native_id="parser")
            ]
            assert not session.waiting
            assert session.outcome == (
                ExecutionOutcome.ERROR if failure else ExecutionOutcome.SUCCESS
            )
            assert replay.model.calls[-1]["purpose"] == (
                "failure" if failure else "completion"
            )
            reports = replay.memory.narrative_reports(
                (SessionIdentity(source=SessionSource.CLAUDE, native_id="parser"),)
            )
            transitions = [
                report.payload
                for report in reports
                if isinstance(report.payload, ExecutionEvidence)
            ]
            assert all(
                report.event not in ("waiting", "resumed") for report in transitions
            )
            assert transitions[-1].outcome == session.outcome
            assert any(event["event"] == "session.waiting" for event in replay.events)

    asyncio.run(scenario())
