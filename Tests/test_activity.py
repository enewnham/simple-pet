import asyncio
import json
import uuid
from pathlib import Path

import pytest

from pet_daemon.conversation import Conversations
from pet_daemon.coordination.models import (
    ActivityRecord,
    Candidate,
    Consideration,
    DelegatedTask,
    Disposition,
    HarnessSession,
    Instruction,
    Note,
    NoteKind,
    NoteOrigin,
    Operation,
    OperationKind,
    OperationOutcome,
    Responsibility,
    SourceReference,
    TaskStatus,
)
from pet_daemon.coordination.workflows import Workflows
from pet_daemon.storage import Memory
from pet_daemon.types import JSONObject
from Tests.fixtures import require
from Tests.test_conversation import Model, objects


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class Surface:
    def __init__(
        self,
        root: Path,
        *,
        capabilities: tuple[OperationKind, ...] = tuple(OperationKind),
    ) -> None:
        self.memory = Memory(root / "memory.sqlite3", "pet", clock=lambda: 1000)
        self.store = self.memory.coordination
        self.store.save_instruction(
            Instruction("adopt", "Watch this migration", str(root), 1000)
        )
        self.store.save_responsibility(
            Responsibility("watch", "Migration validation", str(root), "adopt", 1000)
        )
        self.packets: list[JSONObject] = []
        self.effects: list[OperationKind] = []
        self.sources: list[SourceReference] = []
        self.model = Model()
        self.workflows = Workflows(
            self.store, lambda: 1000, effect=self.effect, navigate=self.navigate
        )
        self.controller = Conversations(
            self.store,
            self.model.reply,
            self.packets.append,
            clock=lambda: 1000,
            act=self.workflows.accept_action,
            desktop_capabilities=capabilities,
        )
        self.controller.open()

    async def effect(self, action: object) -> Operation:
        from pet_daemon.coordination.models import TaskAction

        assert isinstance(action, TaskAction)
        self.effects.append(action.kind)
        task = self.store.task(action.task_id)
        assert task is not None
        operation = Operation(
            action.id,
            task.id,
            action.kind,
            1000,
            task.revision,
            task.responsibility_revision,
            self.store.state().instruction_revision,
            outcome=OperationOutcome.ACCEPTED,
        )
        self.store.save_operation(operation)
        return operation

    async def navigate(self, source: SourceReference) -> None:
        self.sources.append(source)

    def snapshot(self) -> JSONObject:
        snapshot = self.packets[-1]["conversation"]
        assert isinstance(snapshot, dict)
        return snapshot

    async def choose(self, label: str, target: str | None = None) -> None:
        snapshot = self.snapshot()
        actions = list(objects(snapshot["actions"]))
        for entry in objects(snapshot["activity"]):
            if target is None or entry["id"] == target:
                actions.extend(objects(entry["actions"]))
        action = next(action for action in actions if action["label"] == label)
        packet = {
            "event": "interaction_action",
            "interaction_id": self.controller.identifier,
            "revision": self.controller.revision,
            "action_id": action["id"],
        }
        self.controller.receive(packet)
        pending = tuple(self.controller.action_tasks.values())
        if pending:
            await asyncio.gather(*pending)


@pytest.mark.anyio
async def test_activity_explains_deferred_work_and_preserves_note_origins_and_source_dates(
    tmp_path: Path,
) -> None:
    surface = Surface(tmp_path)
    source = SourceReference(
        "meeting",
        "meeting-export",
        "r1",
        900,
        "file:///work/meeting.md",
        str(tmp_path),
        original_at=None,
    )
    surface.store.save_candidate(
        Candidate("candidate", "Check rollback requirement", str(tmp_path), 500)
    )
    surface.store.record_consideration(
        Consideration(
            "consider",
            "candidate",
            1,
            surface.store.state().instruction_revision,
            800,
            Disposition.DEFER,
            "Another investigation occupies the review slot",
            next_review_at=1500,
            wake_condition="The active investigation returns",
        )
    )
    surface.store.save_note(
        Note(
            "hypothesis",
            NoteOrigin.AGENT,
            NoteKind.HYPOTHESIS,
            "Rollback might need another check",
            str(tmp_path),
            850,
            900,
            sources=(source,),
        )
    )
    surface.store.record_activity(
        ActivityRecord(
            "report",
            "task_result",
            "Harness reported validation",
            950,
            detail="The harness reports a passed check. Rollback scope remains uncertain.",
            sources=(source,),
        )
    )
    await surface.choose("Activity")
    entries = objects(surface.snapshot()["activity"])
    candidate = next(entry for entry in entries if entry["id"] == "candidate:candidate")
    assert candidate["category"] == "deferred"
    assert "occupies the review slot" in str(candidate["reason"])
    assert candidate["revisit_at"]
    note = next(entry for entry in entries if entry["id"] == "note:hypothesis")
    assert "Origin: agent" in str(note["reason"]) and "tentative" in str(note["reason"])
    source_packet = objects(note["sources"])[0]
    assert "Source date: unknown" in str(source_packet["detail"])
    await surface.choose("Open meeting-export source", "note:hypothesis")
    assert surface.sources == [source]
    await surface.controller.close()
    surface.memory.close()


@pytest.mark.anyio
async def test_activity_feedback_changes_only_the_selected_candidate_and_survives_reopen(
    tmp_path: Path,
) -> None:
    surface = Surface(tmp_path)
    for identifier in ("selected", "other"):
        surface.store.save_candidate(
            Candidate(identifier, "Check " + identifier, str(tmp_path), 500)
        )
    await surface.choose("Activity")
    await surface.choose("Later (1 hour)", "candidate:selected")
    selected, other = (
        surface.store.candidate("selected"),
        surface.store.candidate("other"),
    )
    assert selected is not None and selected.deferred_until == 4600
    assert other is not None and other.deferred_until is None
    assert len(surface.store.feedback()) == 1
    surface.controller.dismiss()
    surface.controller.open()
    await surface.choose("Activity")
    assert any(
        entry["id"] == "feedback:" + surface.store.feedback()[0].id
        for entry in objects(surface.snapshot()["activity"])
    )
    await surface.controller.close()
    surface.memory.close()


@pytest.mark.anyio
async def test_pause_and_stop_remain_independent_during_a_pending_chat_reply(
    tmp_path: Path,
) -> None:
    surface = Surface(tmp_path)
    surface.store.save_task(
        DelegatedTask(
            "owned",
            "watch",
            "Validate migration",
            "Report checks",
            800,
            1,
            status=TaskStatus.RUNNING,
            session=HarnessSession(
                "fixture", "codex", "native", str(tmp_path), "external"
            ),
        )
    )
    surface.controller.receive(
        {
            "event": "conversation_message",
            "interaction_id": surface.controller.identifier,
            "revision": surface.controller.revision,
            "conversation_id": surface.store.conversations()[0].id,
            "message_id": str(uuid.uuid4()),
            "text": "Explain the check",
        }
    )
    await surface.model.started.wait()
    await surface.choose("Activity")
    await surface.choose("Pause coordination")
    assert surface.store.state().paused and not surface.effects
    await surface.choose("Stop this task", "task:owned")
    assert surface.effects == [OperationKind.STOP]
    assert surface.store.state().paused
    assert require(surface.store.task("owned")).status == TaskStatus.RUNNING
    surface.model.release.set()
    assert surface.controller.task is not None
    await surface.controller.task
    await surface.controller.close()
    surface.memory.close()


@pytest.mark.anyio
async def test_passive_activity_refresh_stays_bounded_and_hides_unsupported_task_actions(
    tmp_path: Path,
) -> None:
    surface = Surface(tmp_path, capabilities=(OperationKind.INSPECT,))
    surface.store.save_task(
        DelegatedTask(
            "owned",
            "watch",
            "Validate migration",
            "Report checks",
            800,
            1,
            status=TaskStatus.RUNNING,
            session=HarnessSession(
                "fixture", "codex", "native", str(tmp_path), "external"
            ),
        )
    )
    await surface.choose("Activity")
    entry = next(
        entry
        for entry in objects(surface.snapshot()["activity"])
        if entry["id"] == "task:owned"
    )
    assert {action["label"] for action in objects(entry["actions"])} == {
        "Refresh task",
        "Ask about this",
    }
    prior = surface.controller.revision
    for index in range(30):
        surface.store.record_activity(
            ActivityRecord(
                str(index), "finding", "A finding", 1200 + index, detail="🐱" * 6000
            )
        )
    surface.controller.receive(
        {"event": "conversation_sync", "interaction_id": surface.controller.identifier}
    )
    assert surface.controller.revision > prior
    assert len(json.dumps(surface.packets[-1]).encode()) <= 60 * 1024
    assert any(
        action["label"] == "Older activity"
        for action in objects(surface.snapshot()["actions"])
    )
    await surface.controller.close()
    surface.memory.close()
