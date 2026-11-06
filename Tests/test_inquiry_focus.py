import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
from claude_agent_sdk import ClaudeAgentOptions, Message, ResultMessage
from mcp.server import Server
from mcp.types import CallToolResult, TextContent

from pet_daemon.companion.config import CoordinatorConfig
from pet_daemon.companion.providers import (
    Availability,
    Document,
    SourceResult,
    SourceStatus,
)
from pet_daemon.companion.sdk import SdkCompanion
from pet_daemon.coordination.config import ScheduleConfig
from pet_daemon.coordination.models import (
    Candidate,
    CandidateCause,
    DelegatedTask,
    Instruction,
    Operation,
    OperationKind,
    Responsibility,
    SourceReference,
    TaskAction,
    TaskReadiness,
)
from pet_daemon.coordination.runtime import Coordinator
from pet_daemon.storage import Memory


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class FocusSdk:
    def __init__(self, scope: str = "b", action: bool = False) -> None:
        self.scope = scope
        self.action = action
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.inputs: list[dict[str, Any]] = []
        self.focused: dict[str, Any] = {}
        self.gated = False
        self.switch_rejected = False
        self.notes_gated = False
        self.unrelated_note_rejected = False
        self.source_ids: tuple[str, ...] = ()
        self.disposition = "no_finding"
        self.schema: dict[str, Any] = {}

    async def __call__(
        self,
        *,
        prompt: str | AsyncIterator[dict[str, object]],
        options: ClaudeAgentOptions,
    ) -> AsyncIterator[Message]:
        assert not isinstance(prompt, str)
        packet = await anext(prompt)
        message = packet["message"]
        assert isinstance(message, dict)
        payload = json.loads(message["content"])
        self.inputs.append(payload)
        assert options.output_format is not None
        self.schema = cast(dict[str, Any], options.output_format["schema"])
        candidate = next(
            item for item in payload["candidates"] if item["scope"] == self.scope
        )
        assert isinstance(options.mcp_servers, dict)
        config = options.mcp_servers["pet"]
        assert isinstance(config, dict) and config["type"] == "sdk"
        server = config["instance"]
        assert isinstance(server, Server)
        handler = server.get_request_handler("tools/call")
        assert handler is not None

        async def call(name: str, arguments: dict[str, object]) -> CallToolResult:
            return CallToolResult.model_validate(
                await cast(Any, handler.handler)(
                    None,
                    handler.params_type.model_validate(
                        {"name": name, "arguments": arguments}
                    ),
                )
            )

        result = await call("current_work_context", {})
        self.gated = result.is_error
        result = await call(
            "record_note",
            {
                "kind": "hypothesis",
                "text": "A tentative possibility",
                "scope": self.scope,
            },
        )
        self.notes_gated = result.is_error
        result = await call("focus_work", {"candidate_id": candidate["id"]})
        if not result.is_error:
            content = result.content[0]
            assert isinstance(content, TextContent)
            self.focused = json.loads(content.text)
            other = next(
                (
                    item
                    for item in payload["candidates"]
                    if item["id"] != candidate["id"]
                ),
                None,
            )
            if other is not None:
                switched = await call("focus_work", {"candidate_id": other["id"]})
                self.switch_rejected = switched.is_error
                note = await call(
                    "record_note",
                    {
                        "kind": "hypothesis",
                        "text": "Unrelated possibility",
                        "scope": other["scope"],
                    },
                )
                self.unrelated_note_rejected = note.is_error
        self.started.set()
        await self.release.wait()
        decision = {
            "candidate_id": candidate["id"],
            "disposition": self.disposition,
            "reason": "No further action is needed for the selected thread.",
            "source_ids": self.source_ids,
        }
        if self.action:
            decision.update(
                disposition="continue",
                action={
                    "id": "continue-b",
                    "kind": "continue",
                    "responsibility_id": "b",
                    "task_id": "task-b",
                    "instruction": "Finish the adopted review",
                },
            )
        yield ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id="focus-fixture",
            structured_output={"decisions": [decision]},
        )


def setup(
    tmp_path: Path, model: FocusSdk
) -> tuple[Memory, SdkCompanion, Coordinator, tuple[Candidate, ...]]:
    memory = Memory(tmp_path / "memory.sqlite3", "fixture", clock=lambda: 100)
    store = memory.coordination
    store.save_instruction(Instruction("adopt", "Review both projects", "", 100))
    dependency = Responsibility("dependency", "Keep the agreed API", "b", "adopt", 100)
    store.save_responsibility(dependency)
    for scope in ("a", "b"):
        store.save_responsibility(
            Responsibility(
                scope,
                "Review",
                scope,
                "adopt",
                100,
                permissions=("continue",),
                dependencies=("dependency",) if scope == "b" else (),
            )
        )
        store.save_task(
            DelegatedTask(
                "task-" + scope,
                scope,
                "Review",
                "Reviewed",
                100,
                1,
                readiness=TaskReadiness.READY,
                adapter_revision="token",
            )
        )
    companion = SdkCompanion(
        CoordinatorConfig(working_directory=tmp_path / "sdk"),
        memory,
        lambda: 100,
        query=model,
    )
    for scope in ("a", "b"):
        companion.index.capture(
            SourceResult(
                (
                    Document(
                        SourceReference(scope, "fixture", "one", 100, scope=scope),
                        "Current source",
                        "Direct source observation",
                        checked_at=100,
                    ),
                ),
                SourceStatus("fixture", Availability.AVAILABLE, 100),
            )
        )
    coordinator = Coordinator(
        store, ScheduleConfig(max_passes_per_hour=1), lambda: 100, companion.investigate
    )
    candidates = tuple(
        coordinator.queue(
            "Review " + scope,
            scope,
            CandidateCause("source", scope, "one", 100),
            responsibility_id=scope,
        )
        for scope in ("a", "b")
    )
    return memory, companion, coordinator, candidates


@pytest.mark.anyio
async def test_model_selected_quiet_peer_survives_busy_work_in_one_charged_attempt(
    tmp_path: Path,
) -> None:
    model = FocusSdk()
    memory, companion, coordinator, (busy, quiet) = setup(tmp_path, model)
    pending = asyncio.create_task(coordinator.run_pass())
    await model.started.wait()
    assert model.gated
    assert model.notes_gated and model.unrelated_note_rejected
    assert memory.coordination.notes() == ()
    assert model.switch_rejected
    assert [item["id"] for item in model.inputs[0]["candidates"]] == [busy.id, quiet.id]
    assert {item["id"] for item in model.focused["responsibilities"]} == {
        "b",
        "dependency",
    }
    assert [item["id"] for item in model.focused["tasks"]] == ["task-b"]
    coordinator.queue(
        "Review a",
        "a",
        CandidateCause("source", "a", "two", 100),
        responsibility_id="a",
    )
    store = memory.coordination
    task = store.task("task-a")
    responsibility = store.responsibility("a")
    assert task is not None and responsibility is not None
    store.save_task(replace(task, revision=2, user_revision=1))
    store.save_task(replace(task, id="new-peer-task"))
    store.save_responsibility(
        replace(responsibility, revision=2, purpose="Changed peer")
    )
    model.release.set()
    result = await pending
    assert result is not None
    assert result.decisions[0].candidate_id == quiet.id
    serviced = store.candidate(quiet.id)
    unserviced = store.candidate(busy.id)
    assert serviced is not None and unserviced is not None
    assert serviced.last_serviced_at == 100
    assert unserviced.last_serviced_at is None
    assert len(store.attempts()) == len(model.inputs) == 1
    assert await coordinator.run_pass() is None
    await companion.close()
    memory.close()


@pytest.mark.anyio
async def test_cancelled_selected_work_does_not_exhaust_unselected_peers(
    tmp_path: Path,
) -> None:
    model = FocusSdk()
    memory, companion, coordinator, (a, b) = setup(tmp_path, model)
    now = [100.0]
    coordinator.clock = lambda: now[0]
    companion.clock = lambda: now[0]
    coordinator.config = ScheduleConfig(
        max_passes_per_hour=100, failure_retry_seconds=1
    )
    outcomes = []
    coordinator.finished = lambda identifier, outcome, error: outcomes.append(outcome)
    for _ in range(2):
        model.started.clear()
        model.release.clear()
        task = asyncio.create_task(coordinator.run_pass())
        await asyncio.wait_for(model.started.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        now[0] += 2
    assert [item.selected_candidate_id for item in memory.coordination.attempts()] == [
        b.id,
        b.id,
    ]
    assert all(
        item.error_kind == "interrupted" for item in memory.coordination.attempts()
    )
    model.scope = "a"
    model.release.set()
    result = await coordinator.run_pass()
    assert result is not None and result.decisions[0].candidate_id == a.id
    assert [item["id"] for item in model.inputs[-1]["candidates"]] == [a.id]
    assert len(model.inputs) == len(memory.coordination.attempts()) == 3
    assert outcomes == ["cancelled", "cancelled", "accepted"]
    current_a = memory.coordination.candidate(a.id)
    current_b = memory.coordination.candidate(b.id)
    assert current_a is not None and current_a.last_serviced_at == now[0]
    assert current_b is not None and current_b.last_serviced_at is None
    await companion.close()
    memory.close()


@pytest.mark.anyio
@pytest.mark.parametrize("chosen", ["a", "b"])
async def test_reserved_focus_is_locked_before_tools_and_cannot_be_bypassed(
    tmp_path: Path, chosen: str
) -> None:
    model = FocusSdk(scope=chosen)
    memory, companion, coordinator, (_, quiet) = setup(tmp_path, model)
    store = memory.coordination
    store.save_candidate(replace(quiet, revision=2, exclusive=True, priority=1))
    model.release.set()
    if chosen == "a":
        with pytest.raises(ValueError, match="fair focus"):
            await coordinator.run_pass()
        assert store.considerations() == ()
    else:
        result = await coordinator.run_pass()
        assert result is not None and result.decisions[0].candidate_id == quiet.id
        assert model.switch_rejected
    assert model.inputs[0]["reserved_focus"] == quiet.id
    assert not model.gated
    await companion.close()
    memory.close()


@pytest.mark.anyio
@pytest.mark.parametrize("goal", ["none", "dependency", "offered", "checks_watch"])
async def test_schema_and_selected_guidance_offer_completion_only_for_an_offered_finite_goal(
    tmp_path: Path, goal: str
) -> None:
    model = FocusSdk()
    memory, companion, coordinator, _ = setup(tmp_path, model)
    if goal != "none":
        store = memory.coordination
        responsibility = store.responsibility(
            "dependency" if goal == "dependency" else "b"
        )
        assert responsibility is not None
        store.save_responsibility(
            replace(
                responsibility,
                revision=2,
                completion_condition="pull_request_checks_pass"
                if goal == "checks_watch"
                else "Find the recorded decision date",
            )
        )
    model.release.set()
    await coordinator.run_pass()
    expected = goal == "offered"
    assert ("complete" in model.schema["$defs"]["Disposition"]["enum"]) == expected
    assert ("complete" in model.focused["allowed_dispositions"]) == expected
    assert bool(model.inputs[0]["completion_targets"]) == expected
    assert "read-only finding" in model.focused["disposition_guidance"]
    assert (
        "read-only finding"
        in model.schema["$defs"]["DecisionOutput"]["properties"]["disposition"][
            "description"
        ]
    )
    await companion.close()
    memory.close()


@pytest.mark.anyio
async def test_host_rejects_fake_completion_without_assessment_even_when_schema_forbids_it(
    tmp_path: Path,
) -> None:
    model = FocusSdk()
    model.disposition = "complete"
    memory, companion, coordinator, _ = setup(tmp_path, model)
    model.release.set()
    with pytest.raises(ValueError, match="Completion requires a supported assessment"):
        await coordinator.run_pass()
    assert memory.coordination.considerations() == ()
    assert len(memory.coordination.attempts()) == 1
    assert memory.coordination.operations() == ()
    await companion.close()
    memory.close()


@pytest.mark.anyio
@pytest.mark.parametrize(
    "change",
    [
        "candidate",
        "instructions",
        "pause",
        "reset",
        "responsibility",
        "dependency",
        "user_input",
        "readiness",
        "token",
        "evidence",
        "source",
        "new_task",
        "dependency_task",
    ],
)
async def test_selected_work_and_authority_changes_reject_without_service_or_dispatch(
    tmp_path: Path, change: str
) -> None:
    model = FocusSdk(action=True)
    memory, companion, coordinator, (_, quiet) = setup(tmp_path, model)
    effects = []

    async def effect(action: TaskAction) -> Operation:
        effects.append(action)
        return Operation(action.id, action.task_id, action.kind, 100, 1, 1, 1)

    coordinator.effect = effect
    pending = asyncio.create_task(coordinator.run_pass())
    await model.started.wait()
    store = memory.coordination
    task = store.task("task-b")
    assert task is not None
    if change == "candidate":
        coordinator.queue(
            "Review b",
            "b",
            CandidateCause("source", "b", "two", 100),
            responsibility_id="b",
        )
    elif change == "instructions":
        store.save_instruction(Instruction("new", "Do not continue", "b", 100))
    elif change == "pause":
        store.set_paused(True)
    elif change == "reset":
        companion.reset()
    elif change in ("responsibility", "dependency"):
        record = store.responsibility(
            "b" if change == "responsibility" else "dependency"
        )
        assert record is not None
        store.save_responsibility(replace(record, revision=2, purpose="Changed scope"))
    elif change == "source":
        document = companion.index.document("b")
        assert document is not None
        companion.index.capture(
            SourceResult(
                (replace(document, source=replace(document.source, revision="two")),),
                SourceStatus("fixture", Availability.AVAILABLE, 100),
            )
        )
    elif change in ("new_task", "dependency_task"):
        store.save_task(
            replace(
                task,
                id="new-owned-task",
                responsibility_id="dependency" if change == "dependency_task" else "b",
            )
        )
    else:
        changes = {
            "user_input": {"user_revision": 1},
            "readiness": {"readiness": TaskReadiness.BUSY},
            "token": {"adapter_revision": "changed"},
            "evidence": {"constraints": ("Do not continue",)},
        }
        store.save_task(replace(task, revision=2, **changes[change]))
    model.release.set()
    with pytest.raises((ValueError, asyncio.CancelledError)):
        await pending
    assert effects == []
    current = store.candidate(quiet.id)
    assert current is not None and current.last_serviced_at is None
    assert store.considerations() == ()
    assert len(store.attempts()) == len(model.inputs) == 1
    assert store.attempts()[0].error
    await companion.close()
    memory.close()


@pytest.mark.anyio
@pytest.mark.parametrize("change", ["none", "version", "provider_state"])
async def test_wait_can_cite_stable_unavailable_history_but_rejects_changed_binding(
    tmp_path: Path, change: str
) -> None:
    model = FocusSdk()
    model.disposition = "wait"
    model.source_ids = ("history-ref",)
    memory, companion, coordinator, (_, quiet) = setup(tmp_path, model)
    document = Document(
        SourceReference("history-ref", "history:fixture", "one", 10, scope="b"),
        "Retained harness report",
        "Reported work remains unverified while the provider is unavailable.",
        kind="harness_report",
    )
    unavailable = SourceStatus(
        "history:fixture", Availability.FAILED, 100, "Provider unavailable"
    )
    companion.index.capture(SourceResult((document,), unavailable))
    assert not companion.index.current(document.source)
    pending = asyncio.create_task(coordinator.run_pass())
    await model.started.wait()
    assert any(
        item["source"]["id"] == "history-ref" for item in model.focused["recent_work"]
    )
    assert any(
        item["provider"] == "history:fixture" and item["availability"] == "failed"
        for item in model.focused["sources"]
    )
    if change == "version":
        companion.index.capture(
            SourceResult(
                (replace(document, source=replace(document.source, revision="two")),),
                unavailable,
            )
        )
    elif change == "provider_state":
        companion.index.capture(
            SourceResult((), replace(unavailable, availability=Availability.AVAILABLE))
        )
    model.release.set()
    store = memory.coordination
    if change == "none":
        result = await pending
        assert result is not None and result.decisions[0].sources == (document.source,)
        current = store.candidate(quiet.id)
        assert current is not None and current.last_serviced_at == 100
        assert len(store.considerations()) == 1
        assert store.attempts()[0].error is None
    else:
        with pytest.raises(ValueError, match="supporting evidence changed"):
            await pending
        assert store.considerations() == ()
        assert store.attempts()[0].error
    assert len(store.attempts()) == len(model.inputs) == 1
    await companion.close()
    memory.close()


@pytest.mark.anyio
@pytest.mark.parametrize(
    "action,token_refresh", [(False, False), (False, True), (True, False)]
)
async def test_observation_refresh_preserves_reasoning_and_binds_current_exact_action_revision(
    tmp_path: Path, action: bool, token_refresh: bool
) -> None:
    model = FocusSdk(action=action)
    memory, companion, coordinator, (_, quiet) = setup(tmp_path, model)
    effects = []

    async def effect(proposal: TaskAction) -> Operation:
        effects.append(proposal)
        current = memory.coordination.task("task-b")
        assert current is not None
        assert proposal.expected_task_revision == current.revision == 2
        assert proposal.expected_user_revision == 0
        return Operation(
            proposal.id, proposal.task_id, OperationKind.CONTINUE, 100, 1, 1, 2
        )

    coordinator.effect = effect
    pending = asyncio.create_task(coordinator.run_pass())
    await model.started.wait()
    store = memory.coordination
    task = store.task("task-b")
    assert task is not None
    store.save_task(
        replace(
            task,
            revision=2,
            observed_at=101,
            adapter_revision=None if token_refresh else task.adapter_revision,
        )
    )
    model.release.set()
    await pending
    assert len(effects) == int(action)
    current = store.candidate(quiet.id)
    assert current is not None and current.last_serviced_at == 100
    assert len(store.attempts()) == 1
    await companion.close()
    memory.close()
