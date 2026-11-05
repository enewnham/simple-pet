import asyncio
import json
import uuid
from dataclasses import dataclass
from pathlib import Path

import pytest

from pet_daemon.conversation import DEFAULT_CONVERSATION_ID, Conversations
from pet_daemon.coordination.models import (
    ChatAction,
    ChatMessage,
    ChatRole,
    SourceReference,
)
from pet_daemon.storage import Memory
from pet_daemon.types import JSONObject, JSONValue


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@dataclass(frozen=True)
class Reply:
    text: str = "The report supports a next step."
    sources: tuple[SourceReference, ...] = ()
    suggestions: tuple[str, ...] = ("Show the evidence",)
    actions: tuple[ChatAction, ...] = ()


class Model:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.failure = False

    async def reply(
        self,
        conversation_id: str,
        messages: tuple[ChatMessage, ...],
        user_message_id: str,
    ) -> Reply:
        assert conversation_id == DEFAULT_CONVERSATION_ID
        assert messages[-1].id == user_message_id
        self.calls.append(user_message_id)
        self.started.set()
        await self.release.wait()
        if self.failure:
            raise RuntimeError("model unavailable")
        return Reply()


def submit(controller: Conversations, text: str = "Watch these checks") -> JSONObject:
    packet: JSONObject = {
        "event": "conversation_message",
        "interaction_id": controller.identifier,
        "revision": controller.revision,
        "conversation_id": DEFAULT_CONVERSATION_ID,
        "message_id": str(uuid.uuid4()),
        "text": text,
    }
    assert controller.receive(packet)
    return packet


@pytest.mark.anyio
async def test_message_retries_accept_one_turn_and_reopen_restores_history(
    tmp_path: Path,
) -> None:
    path = tmp_path / "memory.sqlite3"
    packets: list[JSONObject] = []
    model = Model()
    memory = Memory(path, "pet")
    controller = Conversations(memory.coordination, model.reply, packets.append)
    controller.open()
    initial = controller.identifier
    packet = submit(controller)
    await model.started.wait()
    assert controller.receive(packet)
    assert model.calls == [packet["message_id"]]
    assert len(memory.coordination.messages(DEFAULT_CONVERSATION_ID)) == 1
    controller.dismiss()
    model.release.set()
    assert controller.task is not None
    await controller.task
    assert packets[-1] == {"dismiss_conversation": True}
    memory.close()
    memory = Memory(path, "pet")
    restored = Conversations(memory.coordination, model.reply, packets.append)
    restored.open()
    snapshot = packets[-1]["conversation"]
    assert isinstance(snapshot, dict)
    assert snapshot["id"] != initial
    assert snapshot["conversation_id"] == DEFAULT_CONVERSATION_ID
    assert snapshot["pending_message_id"] is None
    assert len(objects(snapshot["messages"])) == 2
    assert model.calls == [packet["message_id"]]
    await restored.close()
    memory.close()


@pytest.mark.anyio
async def test_failure_retry_preserves_the_user_message_and_rejects_stale_actions(
    tmp_path: Path,
) -> None:
    packets: list[JSONObject] = []
    model = Model()
    model.failure = True
    model.release.set()
    memory = Memory(tmp_path / "memory.sqlite3", "pet")
    controller = Conversations(memory.coordination, model.reply, packets.append)
    controller.open()
    packet = submit(controller)
    assert controller.task is not None
    await controller.task
    failed = packets[-1]["conversation"]
    assert isinstance(failed, dict)
    assert failed["failed_message_id"] == packet["message_id"]
    assert failed["error"]
    revision = controller.revision
    model.failure = False
    retry = {
        "event": "conversation_retry",
        "interaction_id": controller.identifier,
        "revision": revision - 1,
        "conversation_id": DEFAULT_CONVERSATION_ID,
        "message_id": packet["message_id"],
        "command_id": str(uuid.uuid4()),
    }
    controller.receive(retry)
    assert len(model.calls) == 1
    retry["revision"] = controller.revision
    controller.receive(retry)
    assert controller.task is not None
    await controller.task
    assert model.calls == [packet["message_id"], packet["message_id"]]
    messages = memory.coordination.messages(DEFAULT_CONVERSATION_ID)
    assert len(messages) == 2
    assert messages[0].id == packet["message_id"]
    assert messages[1].reply_to == packet["message_id"]
    await controller.close()
    memory.close()


@pytest.mark.anyio
async def test_sleep_preserves_interrupted_input_for_explicit_retry(
    tmp_path: Path,
) -> None:
    model = Model()
    packets: list[JSONObject] = []
    memory = Memory(tmp_path / "memory.sqlite3", "pet")
    controller = Conversations(memory.coordination, model.reply, packets.append)
    controller.open()
    packet = submit(controller)
    await model.started.wait()
    await controller.suspend()
    controller.open()
    snapshot = packets[-1]["conversation"]
    assert isinstance(snapshot, dict)
    assert snapshot["pending_message_id"] is None
    assert snapshot["failed_message_id"] == packet["message_id"]
    assert snapshot["error"]
    assert len(memory.coordination.messages(DEFAULT_CONVERSATION_ID)) == 1
    await controller.close()
    memory.close()


@pytest.mark.anyio
async def test_snapshot_resync_recovers_a_dropped_final_reply_without_another_turn(
    tmp_path: Path,
) -> None:
    packets: list[JSONObject] = []
    model = Model()
    memory = Memory(tmp_path / "memory.sqlite3", "pet")
    controller = Conversations(memory.coordination, model.reply, packets.append)
    controller.open()
    packet = submit(controller)
    await model.started.wait()
    pending = packets[-1]
    model.release.set()
    assert controller.task is not None
    await controller.task
    packets.clear()
    controller.receive(
        {"event": "conversation_sync", "interaction_id": controller.identifier}
    )
    recovered = packets[-1]["conversation"]
    assert isinstance(recovered, dict)
    assert recovered["pending_message_id"] is None
    assert recovered != pending["conversation"]
    assert len(objects(recovered["messages"])) == 2
    assert model.calls == [packet["message_id"]]
    await controller.close()
    memory.close()


@pytest.mark.anyio
async def test_permanent_rejection_is_explicit_and_does_not_accept_a_turn(
    tmp_path: Path,
) -> None:
    packets: list[JSONObject] = []
    model = Model()
    memory = Memory(tmp_path / "memory.sqlite3", "pet")
    controller = Conversations(memory.coordination, model.reply, packets.append)
    controller.open()
    packet = submit(controller, "x" * 16001)
    rejected = packets[-1]["conversation"]
    assert isinstance(rejected, dict)
    assert rejected["rejected_message_id"] == packet["message_id"]
    assert rejected["error"]
    assert not model.calls
    assert not memory.coordination.messages(DEFAULT_CONVERSATION_ID)
    await controller.close()
    memory.close()


@pytest.mark.anyio
async def test_retry_delivery_deduplicates_a_failed_attempt_with_the_same_command(
    tmp_path: Path,
) -> None:
    packets: list[JSONObject] = []
    model = Model()
    model.failure = True
    model.release.set()
    memory = Memory(tmp_path / "memory.sqlite3", "pet")
    controller = Conversations(memory.coordination, model.reply, packets.append)
    controller.open()
    packet = submit(controller)
    assert controller.task is not None
    await controller.task
    command = str(uuid.uuid4())
    retry: JSONObject = {
        "event": "conversation_retry",
        "interaction_id": controller.identifier,
        "revision": controller.revision,
        "conversation_id": DEFAULT_CONVERSATION_ID,
        "message_id": packet["message_id"],
        "command_id": command,
    }
    controller.receive(retry)
    assert controller.task is not None
    await controller.task
    retry["revision"] = controller.revision
    controller.receive(retry)
    assert controller.task is not None
    await controller.task
    assert len(model.calls) == 2
    snapshot = packets[-1]["conversation"]
    assert isinstance(snapshot, dict)
    assert snapshot["acknowledged_command_id"] == command
    assert snapshot["failed_message_id"] == packet["message_id"]
    await controller.close()
    memory.close()


@pytest.mark.anyio
async def test_history_pages_fit_the_socket_and_preserve_every_message_fragment(
    tmp_path: Path,
) -> None:
    packets: list[JSONObject] = []
    model = Model()
    model.release.set()
    memory = Memory(tmp_path / "memory.sqlite3", "pet")
    controller = Conversations(memory.coordination, model.reply, packets.append)
    controller.open()
    for _ in range(5):
        submit(controller, "🐱" * 16000)
        assert controller.task is not None
        await controller.task
    stored = memory.coordination.messages(DEFAULT_CONVERSATION_ID)
    fragments: dict[str, dict[int, str]] = {}
    pages = 0
    while True:
        packet = packets[-1]
        assert len(json.dumps(packet).encode()) <= 60 * 1024
        snapshot = packet["conversation"]
        assert isinstance(snapshot, dict)
        for message in objects(snapshot["messages"]):
            identifier, offset, text = (
                message["id"],
                message.get("text_start", 0),
                message["text"],
            )
            assert (
                isinstance(identifier, str)
                and isinstance(offset, int)
                and isinstance(text, str)
            )
            fragments.setdefault(identifier, {})[offset] = text
        older = next(
            (
                action
                for action in objects(snapshot["actions"])
                if action["label"] == "Older messages"
            ),
            None,
        )
        pages += 1
        if older is None:
            break
        controller.receive(
            {
                "event": "interaction_action",
                "interaction_id": controller.identifier,
                "revision": controller.revision,
                "action_id": older["id"],
            }
        )
        assert pages < 50
    assert pages > 5
    for message in stored:
        assert (
            "".join(text for _, text in sorted(fragments[message.id].items()))
            == message.text
        )
    assert len(memory.coordination.messages(DEFAULT_CONVERSATION_ID)) == 10
    await controller.close()
    memory.close()


def objects(value: JSONValue) -> tuple[JSONObject, ...]:
    assert isinstance(value, (tuple, list))
    result = []
    for item in value:
        assert isinstance(item, dict)
        result.append(item)
    return tuple(result)


def test_canonical_source_navigation_selects_the_exact_durable_message(
    tmp_path: Path,
) -> None:
    memory = Memory(tmp_path / "memory.sqlite3", "pet")
    store = memory.coordination
    store.ensure_conversation(DEFAULT_CONVERSATION_ID)
    first = ChatMessage(
        str(uuid.uuid4()),
        DEFAULT_CONVERSATION_ID,
        ChatRole.USER,
        "First source correction",
        1000,
    )
    assert store.begin_turn(first)
    conversation = store.conversation(DEFAULT_CONVERSATION_ID)
    assert conversation is not None
    assert store.finish_turn(
        ChatMessage(
            str(uuid.uuid4()),
            DEFAULT_CONVERSATION_ID,
            ChatRole.ASSISTANT,
            "I will preserve that correction",
            1001,
            reply_to=first.id,
        ),
        conversation.revision,
    )
    packets: list[JSONObject] = []
    controller = Conversations(store, Model().reply, packets.append)
    controller.show_message(first.id)
    snapshot = packets[-1]["conversation"]
    assert isinstance(snapshot, dict)
    assert objects(snapshot["messages"])[-1]["id"] == first.id
    controller.show_activity()
    snapshot = packets[-1]["conversation"]
    assert isinstance(snapshot, dict) and snapshot["mode"] == "activity"
    memory.close()
