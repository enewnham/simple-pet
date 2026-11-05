import asyncio
import json
import os
import socket
import sys
import uuid
from dataclasses import replace
from pathlib import Path

import pytest

from pet_daemon.cli.hook import update
from pet_daemon.coordination.models import (
    Instruction,
    OperationKind,
    OperationOutcome,
    Responsibility,
    TaskAction,
    TaskReadiness,
    TaskStatus,
)
from pet_daemon.desktop import AdapterClient, AuthorityError, Delegation, DesktopConfig
from pet_daemon.runtime import HookEventKind
from pet_daemon.sessions.tracking import Sessions
from pet_daemon.sources.hooks import normalize_hook
from pet_daemon.storage import Memory
from Tests.fixtures import require

FIXTURE = Path(__file__).parent / "fixtures/desktop/adapter.py"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class Desktop:
    def __init__(self, root: Path, *, timeout: float = 2) -> None:
        self.root = root
        self.memory = Memory(root / "memory.sqlite3", "pet", clock=lambda: 1000)
        self.store = self.memory.coordination
        if not self.store.responsibility("watch"):
            self.store.save_instruction(
                Instruction("adopt", "Validate this repository", str(root), 1000)
            )
            self.store.save_responsibility(
                Responsibility(
                    "watch",
                    "Validation passes",
                    str(root),
                    "adopt",
                    1000,
                    permissions=("launch", "continue"),
                )
            )
        self.config = DesktopConfig(
            name="fixture",
            argv=(sys.executable, str(FIXTURE), str(root)),
            capabilities=tuple(OperationKind),
            timeout_seconds=timeout,
        )
        self.service = Delegation(
            self.store, AdapterClient(self.config), clock=lambda: 1000
        )

    def action(
        self, kind: OperationKind = OperationKind.LAUNCH, task_id: str = ""
    ) -> TaskAction:
        return TaskAction(
            str(uuid.uuid4()),
            kind,
            "watch",
            task_id,
            instruction="Check rollback validation",
            objective="Validate migration",
            expected_result="A validation report with check evidence",
            repository=str(self.root),
        )

    def mode(self, mode: str) -> None:
        (self.root / "mode").write_text(mode)

    def calls(self, kind: str) -> list[dict[str, object]]:
        return [
            entry
            for line in (self.root / "calls.jsonl").read_text().splitlines()
            if (entry := json.loads(line))["operation"] == kind
        ]

    async def entered(self) -> None:
        async with asyncio.timeout(2):
            while not (self.root / "entered").exists():
                await asyncio.sleep(0.005)


@pytest.mark.anyio
async def test_launch_intent_is_persisted_and_duplicate_command_does_not_launch_twice(
    tmp_path: Path,
) -> None:
    desktop = Desktop(tmp_path)
    desktop.mode("wait")
    action = desktop.action()
    pending = asyncio.create_task(desktop.service.dispatch(action))
    await desktop.entered()
    recorded = desktop.store.operation(action.id)
    assert recorded is not None and recorded.outcome == OperationOutcome.PENDING
    task = desktop.store.task(recorded.task_id)
    assert task is not None and task.session is None
    request = desktop.calls("launch")[0]
    assert request["command_id"] == action.id
    assert request["environment"] == {
        "SIMPLE_PET_TASK_ID": task.id,
        "SIMPLE_PET_COMMAND_ID": action.id,
    }
    assert request["observation"] == {
        "task_revision": task.revision,
        "user_revision": 0,
        "status": "preparing",
        "readiness": "unknown",
        "observed_at": None,
        "input": None,
        "input_observed_at": None,
        "native_session": None,
    }
    (tmp_path / "release").touch()
    result = await pending
    assert result.outcome == OperationOutcome.ACCEPTED
    assert await desktop.service.dispatch(action) == result
    assert len(desktop.calls("launch")) == 1
    assert require(desktop.store.task(task.id)).status == TaskStatus.PREPARING
    desktop.memory.close()


@pytest.mark.anyio
@pytest.mark.parametrize("mode", ["timeout", "malformed"])
async def test_lost_receipts_reconcile_the_original_command_after_restart(
    tmp_path: Path, mode: str
) -> None:
    desktop = Desktop(tmp_path, timeout=0.15)
    desktop.mode(mode)
    action = desktop.action()
    result = await desktop.service.dispatch(action)
    assert result.outcome == OperationOutcome.UNKNOWN
    assert (
        require(desktop.store.task(result.task_id)).readiness == TaskReadiness.UNKNOWN
    )
    desktop.memory.close()
    desktop = Desktop(tmp_path)
    desktop.mode("ready")
    await desktop.service.reconcile()
    established = desktop.store.operation(action.id)
    assert established is not None and established.outcome == OperationOutcome.ACCEPTED
    assert require(desktop.store.task(result.task_id)).session is not None
    assert desktop.calls("inspect")[0]["command_id"] == action.id
    assert len(desktop.calls("launch")) == 1
    desktop.memory.close()


@pytest.mark.anyio
@pytest.mark.parametrize("hook_first", [True, False])
async def test_hook_and_receipt_orderings_join_one_explicitly_owned_session(
    tmp_path: Path, hook_first: bool
) -> None:
    desktop = Desktop(tmp_path)
    desktop.mode("wait" if hook_first else "ready")
    action = desktop.action()
    pending = asyncio.create_task(desktop.service.dispatch(action))
    if hook_first:
        await desktop.entered()
        operation = desktop.store.operation(action.id)
    else:
        operation = await pending
    assert operation is not None
    task_id = operation.task_id
    hook = update(
        "codex",
        "SessionStart",
        {"session_id": "native-" + task_id, "cwd": str(tmp_path / "worktree")},
        delegation_id=task_id,
        delegation_command_id=action.id,
    )
    assert hook is not None
    assert desktop.service.observe_hook(hook)
    if hook_first:
        assert require(desktop.store.task(task_id)).session is None
        (tmp_path / "release").touch()
        await pending
    owned = desktop.store.task(task_id)
    assert (
        owned is not None
        and owned.session is not None
        and owned.native_session == hook.subject
    )
    unrelated = hook.model_copy(update={"session": "unrelated"})
    assert not desktop.service.observe_hook(unrelated)
    assert len(desktop.store.tasks()) == 1
    desktop.memory.close()


@pytest.mark.anyio
async def test_pause_preserves_a_late_launch_receipt_and_stop_has_an_independent_effect(
    tmp_path: Path,
) -> None:
    desktop = Desktop(tmp_path)
    desktop.mode("wait")
    pending = asyncio.create_task(desktop.service.dispatch(desktop.action()))
    await desktop.entered()
    desktop.store.set_paused(True)
    (tmp_path / "release").touch()
    result = await pending
    task = desktop.store.task(result.task_id)
    assert task is not None and task.session is not None
    desktop.mode("ready")
    await desktop.service.inspect(task.id)
    with pytest.raises(AuthorityError):
        await desktop.service.dispatch(desktop.action(OperationKind.CONTINUE, task.id))
    stopped = await desktop.service.stop_task(task.id)
    assert stopped.outcome == OperationOutcome.ACCEPTED
    assert require(desktop.store.task(task.id)).status == TaskStatus.STOPPED
    assert desktop.store.state().paused
    assert not desktop.calls("continue")
    desktop.memory.close()


@pytest.mark.anyio
async def test_unknown_readiness_and_changed_user_revision_block_continuation(
    tmp_path: Path,
) -> None:
    desktop = Desktop(tmp_path)
    result = await desktop.service.dispatch(desktop.action())
    desktop.mode("no_token")
    await desktop.service.inspect(result.task_id)
    with pytest.raises(AuthorityError):
        await desktop.service.dispatch(
            desktop.action(OperationKind.CONTINUE, result.task_id)
        )
    desktop.mode("unknown")
    await desktop.service.inspect(result.task_id)
    with pytest.raises(AuthorityError):
        await desktop.service.dispatch(
            desktop.action(OperationKind.CONTINUE, result.task_id)
        )
    desktop.mode("ready")
    task = await desktop.service.inspect(result.task_id)
    captured = replace(
        desktop.action(OperationKind.CONTINUE, task.id),
        expected_user_revision=task.user_revision,
    )
    hook = update(
        "codex",
        "UserPromptSubmit",
        {"session_id": require(task.session).session, "prompt": "Change direction"},
        delegation_id=task.id,
        delegation_command_id=result.id,
    )
    assert hook is not None and desktop.service.observe_hook(hook)
    observed = require(desktop.store.task(task.id))
    assert (
        observed.adapter_revision is None
        and observed.observed_input == "Change direction"
    )
    await desktop.service.inspect(task.id)
    with pytest.raises(AuthorityError):
        await desktop.service.dispatch(captured)
    assert not desktop.calls("continue")
    desktop.memory.close()


@pytest.mark.anyio
async def test_native_user_input_interrupts_an_inflight_continuation_without_claiming_stop(
    tmp_path: Path,
) -> None:
    desktop = Desktop(tmp_path)
    result = await desktop.service.dispatch(desktop.action())
    task = await desktop.service.inspect(result.task_id)
    assert task.session is not None
    desktop.mode("wait")
    action = desktop.action(OperationKind.CONTINUE, task.id)
    pending = asyncio.create_task(desktop.service.dispatch(action))
    await desktop.entered()
    hook = update(
        "codex",
        "UserPromptSubmit",
        {"session_id": task.session.session, "prompt": "I'll take over"},
        delegation_id=task.id,
        delegation_command_id=result.id,
    )
    assert hook is not None and desktop.service.observe_hook(hook)
    with pytest.raises(asyncio.CancelledError):
        await pending
    operation = desktop.store.operation(action.id)
    assert operation is not None and operation.outcome == OperationOutcome.UNKNOWN
    assert require(desktop.store.task(task.id)).status != TaskStatus.STOPPED
    desktop.mode("ready")
    await desktop.service.inspect(task.id)
    assert (
        require(desktop.store.operation(action.id)).outcome == OperationOutcome.ACCEPTED
    )
    assert len(desktop.calls("continue")) == 1
    desktop.memory.close()


@pytest.mark.anyio
async def test_native_hook_process_preserves_explicit_delegation_identity(
    tmp_path: Path,
) -> None:
    desktop = Desktop(tmp_path)
    result = await desktop.service.dispatch(desktop.action())
    task = require(desktop.store.task(result.task_id))
    session = require(task.session)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    socket_path = str(tmp_path / "hooks.sock")
    listener.bind(socket_path)
    listener.setblocking(False)
    try:
        process = await asyncio.create_subprocess_exec(
            str(Path(__file__).parents[1] / "scripts/pet-hook"),
            "codex",
            "SessionStart",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            env=dict(
                os.environ,
                PET_DAEMON_SOCKET=socket_path,
                SIMPLE_PET_TASK_ID=task.id,
                SIMPLE_PET_COMMAND_ID=result.id,
            ),
        )
        output, _ = await process.communicate(
            json.dumps(
                {"session_id": session.session, "cwd": session.workspace}
            ).encode()
        )
        assert process.returncode == 0 and json.loads(output) == {}
        packet = await asyncio.wait_for(
            asyncio.get_running_loop().sock_recv(listener, 65536), 2
        )
        hook = require(normalize_hook(json.loads(packet)))
        assert hook.delegation_id == task.id and hook.delegation_command_id == result.id
        sessions = Sessions(lambda: 1000, wall_clock=lambda: 1000)
        assert sessions.apply(hook) is not None
        assert desktop.service.observe_hook(hook)
        assert require(desktop.store.task(task.id)).native_session == hook.subject
    finally:
        listener.close()
        desktop.memory.close()


@pytest.mark.anyio
async def test_failed_preparation_and_unsupported_stop_keep_their_distinct_outcomes(
    tmp_path: Path,
) -> None:
    desktop = Desktop(tmp_path)
    desktop.mode("preparation_failed")
    result = await desktop.service.dispatch(desktop.action())
    assert result.outcome == OperationOutcome.FAILED
    failed = require(desktop.store.task(result.task_id))
    assert failed.session is None and failed.status == TaskStatus.UNKNOWN
    desktop.mode("ready")
    launched = await desktop.service.dispatch(desktop.action())
    desktop.service.client = AdapterClient(
        desktop.config.model_copy(
            update={"capabilities": (OperationKind.LAUNCH, OperationKind.INSPECT)}
        )
    )
    stopped = await desktop.service.stop_task(launched.task_id)
    assert stopped.outcome == OperationOutcome.FAILED
    assert require(desktop.store.task(launched.task_id)).status != TaskStatus.STOPPED
    assert not desktop.calls("stop")
    with pytest.raises(AuthorityError):
        await desktop.service.stop_task("unowned")
    desktop.memory.close()


@pytest.mark.anyio
async def test_adapter_rechecks_the_native_interaction_token_before_delivering(
    tmp_path: Path,
) -> None:
    desktop = Desktop(tmp_path)
    result = await desktop.service.dispatch(desktop.action())
    task = await desktop.service.inspect(result.task_id)
    assert task.adapter_revision == "native-input-1"
    desktop.mode("token_changed")
    action = desktop.action(OperationKind.CONTINUE, task.id)
    declined = await desktop.service.dispatch(action)
    assert declined.outcome == OperationOutcome.FAILED
    request = desktop.calls("continue")[0]
    assert request["expected_interaction_revision"] == "native-input-1"
    effects = [
        json.loads(line)
        for line in (tmp_path / "effects.jsonl").read_text().splitlines()
    ]
    assert not any(effect["command_id"] == action.id for effect in effects)
    desktop.memory.close()


@pytest.mark.anyio
async def test_lost_continuation_receipt_and_its_native_prompt_reconcile_without_new_authority(
    tmp_path: Path,
) -> None:
    desktop = Desktop(tmp_path, timeout=0.15)
    launched = await desktop.service.dispatch(desktop.action())
    task = await desktop.service.inspect(launched.task_id)
    action = desktop.action(OperationKind.CONTINUE, task.id)
    desktop.mode("timeout")
    uncertain = await desktop.service.dispatch(action)
    assert uncertain.outcome == OperationOutcome.UNKNOWN
    hook = update(
        "codex",
        "UserPromptSubmit",
        {"session_id": require(task.session).session, "prompt": action.instruction},
        delegation_id=task.id,
        delegation_command_id=launched.id,
    )
    assert hook is not None and desktop.service.observe_hook(hook)
    assert require(desktop.store.task(task.id)).observed_input == action.instruction
    desktop.mode("ready")
    await desktop.service.inspect(task.id)
    assert (
        require(desktop.store.operation(action.id)).outcome == OperationOutcome.ACCEPTED
    )
    assert len(desktop.calls("continue")) == 1
    assert require(desktop.store.task(task.id)).readiness == TaskReadiness.UNKNOWN
    task = await desktop.service.inspect(task.id)
    assert task.readiness == TaskReadiness.READY
    assert len(desktop.store.instructions()) == 1
    assert desktop.calls("inspect")[-2]["command_id"] == action.id
    desktop.memory.close()


@pytest.mark.anyio
@pytest.mark.parametrize(
    "state",
    [
        TaskStatus.PREPARING,
        TaskStatus.RUNNING,
        TaskStatus.WAITING,
        TaskStatus.UNKNOWN,
    ],
)
async def test_owned_or_uncertain_execution_keeps_the_single_coding_slot(
    tmp_path: Path, state: TaskStatus
) -> None:
    desktop = Desktop(tmp_path)
    first = await desktop.service.dispatch(desktop.action())
    task = require(desktop.store.task(first.task_id))
    desktop.store.save_task(replace(task, revision=task.revision + 1, status=state))
    with pytest.raises(AuthorityError, match="occupies the slot"):
        await desktop.service.dispatch(desktop.action())
    assert len(desktop.calls("launch")) == 1
    assert len(desktop.store.tasks()) == 1
    desktop.memory.close()


@pytest.mark.anyio
@pytest.mark.parametrize("terminal", [TaskStatus.ENDED, TaskStatus.STOPPED])
async def test_observed_terminal_execution_releases_the_slot_without_claiming_completion(
    tmp_path: Path, terminal: TaskStatus
) -> None:
    desktop = Desktop(tmp_path)
    first = await desktop.service.dispatch(desktop.action())
    task = require(desktop.store.task(first.task_id))
    desktop.store.save_task(
        replace(task, revision=task.revision + 1, status=terminal, observed_at=1001)
    )
    second = await desktop.service.dispatch(desktop.action())
    assert second.outcome == OperationOutcome.ACCEPTED
    assert second.task_id != first.task_id
    assert require(desktop.store.responsibility("watch")).status.value == "active"
    desktop.memory.close()


@pytest.mark.anyio
async def test_parallel_launches_share_the_configured_effect_boundary_capacity(
    tmp_path: Path,
) -> None:
    desktop = Desktop(tmp_path)
    desktop.mode("wait")
    first = asyncio.create_task(desktop.service.dispatch(desktop.action()))
    await desktop.entered()
    second = asyncio.create_task(desktop.service.dispatch(desktop.action()))
    (tmp_path / "release").touch()
    await first
    with pytest.raises(AuthorityError, match="occupies the slot"):
        await second
    assert len(desktop.calls("launch")) == 1
    desktop.service.coding_slots = 2
    await desktop.service.dispatch(desktop.action())
    assert len(desktop.calls("launch")) == 2
    with pytest.raises(AuthorityError, match="occupies the slot"):
        await desktop.service.dispatch(desktop.action())
    desktop.memory.close()


@pytest.mark.anyio
async def test_revised_responsibility_invalidates_then_rebinds_only_a_fresh_in_scope_continuation(
    tmp_path: Path,
) -> None:
    desktop = Desktop(tmp_path)
    launched = await desktop.service.dispatch(desktop.action())
    task = await desktop.service.inspect(launched.task_id)
    stale = replace(
        desktop.action(OperationKind.CONTINUE, task.id),
        expected_task_revision=task.revision,
        expected_responsibility_revision=task.responsibility_revision,
        expected_instruction_revision=desktop.store.state().instruction_revision,
        expected_user_revision=task.user_revision,
    )
    responsibility = require(desktop.store.responsibility("watch"))
    revised = replace(
        responsibility,
        revision=responsibility.revision + 1,
        purpose="Validate rollback and migration",
    )
    desktop.store.save_responsibility(revised)
    desktop.service.responsibility_changed(revised)
    assert require(desktop.store.task(task.id)).adapter_revision is None
    with pytest.raises(AuthorityError):
        await desktop.service.dispatch(stale)
    refreshed = await desktop.service.inspect(task.id)
    fresh = replace(
        desktop.action(OperationKind.CONTINUE, task.id),
        expected_task_revision=refreshed.revision,
        expected_responsibility_revision=revised.revision,
        expected_instruction_revision=desktop.store.state().instruction_revision,
        expected_user_revision=refreshed.user_revision,
    )
    operation = await desktop.service.dispatch(fresh)
    assert operation.outcome == OperationOutcome.ACCEPTED
    assert (
        require(desktop.store.task(task.id)).responsibility_revision == revised.revision
    )
    desktop.memory.close()


@pytest.mark.anyio
@pytest.mark.parametrize("hook_first", [True, False])
async def test_hooks_without_environment_bind_only_an_explicit_owned_native_receipt(
    tmp_path: Path, hook_first: bool
) -> None:
    desktop = Desktop(tmp_path)
    desktop.mode("wait" if hook_first else "ready")
    action = desktop.action()
    pending = asyncio.create_task(desktop.service.dispatch(action))
    if hook_first:
        await desktop.entered()
        operation = require(desktop.store.operation(action.id))
    else:
        operation = await pending
    hook = require(
        update(
            "codex",
            "UserPromptSubmit",
            {
                "session_id": "native-" + operation.task_id,
                "prompt": "Keep rollback read-only",
            },
        )
    )
    assert desktop.service.observe_hook(hook) is not hook_first
    if hook_first:
        assert require(desktop.store.task(operation.task_id)).native_session is None
        (tmp_path / "release").touch()
        await pending
    task = require(desktop.store.task(operation.task_id))
    assert (
        task.native_session == hook.subject
        and task.observed_input == "Keep rollback read-only"
    )
    assert task.user_revision == 1
    assert not desktop.service.observe_hook(
        hook.model_copy(update={"session": "unrelated"})
    )
    assert not desktop.service.observe_hook(
        hook.model_copy(
            update={"delegation_id": "invalid", "delegation_command_id": action.id}
        )
    )
    assert not desktop.service.observe_hook(
        hook.model_copy(update={"parent": "parent"})
    )
    assert not desktop.service.observe_hook(
        hook,
        accepted_identity=hook.subject.model_copy(
            update={"native_id": "unrelated-child"}
        ),
    )
    ended = hook.model_copy(update={"event": HookEventKind.SESSION_END})
    assert desktop.service.observe_hook(ended)
    assert require(desktop.store.task(task.id)).status == TaskStatus.UNKNOWN
    desktop.memory.close()


@pytest.mark.anyio
async def test_idle_owned_sessions_release_capacity_but_cannot_continue_while_another_task_runs(
    tmp_path: Path,
) -> None:
    desktop = Desktop(tmp_path)
    first = await desktop.service.dispatch(desktop.action())
    ready = await desktop.service.inspect(first.task_id)
    assert ready.status == TaskStatus.READY
    assert desktop.service.available_coding_slots() == 1
    await desktop.service.dispatch(desktop.action())
    assert desktop.service.available_coding_slots() == 0
    with pytest.raises(AuthorityError, match="occupies the slot"):
        await desktop.service.dispatch(desktop.action(OperationKind.CONTINUE, ready.id))
    assert not desktop.calls("continue")
    desktop.memory.close()


@pytest.mark.anyio
async def test_unknown_continuation_reserves_execution_capacity_until_inspected(
    tmp_path: Path,
) -> None:
    desktop = Desktop(tmp_path, timeout=0.15)
    first = await desktop.service.dispatch(desktop.action())
    ready = await desktop.service.inspect(first.task_id)
    desktop.mode("timeout")
    uncertain = await desktop.service.dispatch(
        desktop.action(OperationKind.CONTINUE, ready.id)
    )
    assert uncertain.outcome == OperationOutcome.UNKNOWN
    assert desktop.service.available_coding_slots() == 0
    with pytest.raises(AuthorityError, match="occupies the slot"):
        await desktop.service.dispatch(desktop.action())
    desktop.mode("ready")
    await desktop.service.inspect(ready.id)
    assert desktop.service.available_coding_slots() == 1
    await desktop.service.dispatch(desktop.action())
    desktop.memory.close()
