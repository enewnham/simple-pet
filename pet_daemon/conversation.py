import asyncio
import json
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

from .activity import (
    Activity,
    ActivityEntry,
    source_detail,
    stamp,
)
from .activity import action as activity_action
from .coordination.models import (
    ChatAction,
    ChatMessage,
    ChatRole,
    OperationKind,
    SourceReference,
)
from .logging import get_logger
from .storage.coordination import SQLiteCoordinationRepository
from .types import JSONObject

log = get_logger(__name__)
DEFAULT_CONVERSATION_ID = str(uuid.uuid5(uuid.NAMESPACE_URL, "simple-pet:conversation"))
MAX_MESSAGE_LENGTH = 16000
MAX_SNAPSHOT_BYTES = 60 * 1024
TRANSCRIPT_BYTES = 48 * 1024


class Reply(Protocol):
    @property
    def text(self) -> str: ...

    @property
    def sources(self) -> tuple[SourceReference, ...]: ...

    @property
    def suggestions(self) -> tuple[str, ...]: ...

    @property
    def actions(self) -> tuple[ChatAction, ...]: ...


class Responder(Protocol):
    def __call__(
        self,
        conversation_id: str,
        messages: tuple[ChatMessage, ...],
        user_message_id: str,
    ) -> Awaitable[Reply]: ...


@dataclass(frozen=True)
class Choice:
    kind: str
    target: str = ""
    action: ChatAction | None = None


class Conversations:
    def __init__(
        self,
        store: SQLiteCoordinationRepository,
        reply: Responder,
        emit: Callable[[JSONObject], None],
        *,
        clock: Callable[[], float] = time.time,
        act: Callable[[ChatAction], Awaitable[None]] | None = None,
        desktop_capabilities: tuple[OperationKind, ...] = (),
    ) -> None:
        self.store = store
        self.reply = reply
        self.emit = emit
        self.clock = clock
        self.act = act
        self.activity = Activity(store, desktop_capabilities)
        self.activity_offset = 0
        self.activity_offsets: list[int] = []
        self.activity_checkpoint: object = None
        self.identifier = ""
        self.revision = 0
        self.mode = "chat"
        self.engaged = False
        self.choices: dict[str, Choice] = {}
        self.task: asyncio.Task[None] | None = None
        self.action_tasks: dict[str, asyncio.Task[None]] = {}
        self.snapshot: JSONObject | None = None
        self.history_end: tuple[int, int] | None = None
        self.history_stack: list[tuple[int, int] | None] = []
        self.acknowledged_command_id: str | None = None

    @property
    def is_interacting(self) -> bool:
        return bool(self.identifier)

    def open(self) -> None:
        self.store.ensure_conversation(DEFAULT_CONVERSATION_ID)
        self.identifier = str(uuid.uuid4())
        self.revision = 0
        self.mode = "chat"
        self.history_end = None
        self.history_stack.clear()
        self._publish()

    def show_message(self, message_id: str) -> None:
        messages = self.store.messages(DEFAULT_CONVERSATION_ID)
        selected = next(
            (
                (index, message)
                for index, message in enumerate(messages)
                if message.id == message_id
            ),
            None,
        )
        if selected is None:
            raise ValueError("That message is unavailable in this conversation")
        if not self.identifier:
            self.open()
        index, message = selected
        self.mode = "chat"
        self.history_end = (index, len(message.text))
        self.history_stack.clear()
        self._publish()

    def show_activity(self) -> None:
        if not self.identifier:
            self.open()
        self.mode = "activity"
        self.activity_offset = 0
        self.activity_offsets.clear()
        self._publish()

    def receive(self, packet: Mapping[str, object]) -> bool:
        event = packet.get("event")
        if event not in {
            "conversation_message",
            "conversation_retry",
            "interaction_action",
            "interaction_dismissed",
            "interaction_state",
            "conversation_sync",
        }:
            return False
        if packet.get("interaction_id") != self.identifier or not self.identifier:
            return event in {"conversation_message", "conversation_retry"}
        if event == "conversation_sync":
            if (
                self.mode == "activity"
                and self.activity_checkpoint != self._activity_checkpoint()
            ):
                self._publish()
                return True
            if self.snapshot is not None:
                self.emit(self.snapshot)
            return True
        if event == "interaction_state":
            if isinstance(engaged := packet.get("engaged"), bool):
                self.engaged = engaged
            return True
        if event == "conversation_message" and self._is_duplicate(packet):
            self._publish()
            return True
        if (
            type(packet.get("revision")) is not int
            or packet["revision"] != self.revision
        ):
            self._publish(error="That conversation changed. Please try again.")
            return True
        if event == "interaction_dismissed":
            self.dismiss()
        elif event == "conversation_message":
            self._accept(packet)
        elif event == "conversation_retry":
            self._retry(packet)
        else:
            action_id = packet.get("action_id")
            choice = self.choices.get(action_id) if isinstance(action_id, str) else None
            if choice is None:
                self._publish(error="That choice changed. Please choose again.")
            else:
                self._choose(choice)
        return True

    def _is_duplicate(self, packet: Mapping[str, object]) -> bool:
        if packet.get("conversation_id") != DEFAULT_CONVERSATION_ID:
            return False
        return any(
            message.id == packet.get("message_id")
            for message in self.store.messages(DEFAULT_CONVERSATION_ID)
        )

    def _accept(self, packet: Mapping[str, object]) -> None:
        text = packet.get("text")
        identifier = packet.get("message_id")
        if packet.get("conversation_id") != DEFAULT_CONVERSATION_ID or not isinstance(
            identifier, str
        ):
            self._publish(
                error="That message has an invalid identity.",
                rejected=identifier if isinstance(identifier, str) else None,
            )
            return
        try:
            uuid.UUID(identifier)
        except ValueError:
            self._publish(
                error="That message has an invalid identity.",
                rejected=identifier if isinstance(identifier, str) else None,
            )
            return
        if (
            not isinstance(text, str)
            or not text.strip()
            or len(text) > MAX_MESSAGE_LENGTH
        ):
            self._publish(
                error="Write a message of up to 16,000 characters.", rejected=identifier
            )
            return
        message = ChatMessage(
            identifier,
            DEFAULT_CONVERSATION_ID,
            ChatRole.USER,
            text.strip(),
            self.clock(),
        )
        if self.store.begin_turn(message):
            self._start(identifier)
        else:
            self._publish(
                error="Wait for the current reply before sending another message.",
                rejected=identifier,
            )
            return
        self._publish()

    def _retry(self, packet: Mapping[str, object]) -> None:
        identifier = packet.get("message_id")
        if packet.get("conversation_id") != DEFAULT_CONVERSATION_ID or not isinstance(
            identifier, str
        ):
            self._publish(error="That retry has an invalid identity.")
            return
        command_id = packet.get("command_id")
        if not isinstance(command_id, str) or not command_id:
            self._publish(error="That retry has an invalid command identity.")
            return
        self.acknowledged_command_id = command_id
        if self.store.retry_turn(
            DEFAULT_CONVERSATION_ID, identifier, command_id=command_id
        ):
            self._start(identifier)
        self._publish()

    def _start(self, identifier: str) -> None:
        conversation = self.store.conversation(DEFAULT_CONVERSATION_ID)
        assert conversation is not None
        self.task = asyncio.create_task(
            self._respond(identifier, conversation.revision)
        )

    async def _respond(self, identifier: str, revision: int) -> None:
        try:
            reply = await self.reply(
                DEFAULT_CONVERSATION_ID,
                self.store.messages(DEFAULT_CONVERSATION_ID),
                identifier,
            )
            message = ChatMessage(
                str(uuid.uuid4()),
                DEFAULT_CONVERSATION_ID,
                ChatRole.ASSISTANT,
                reply.text,
                self.clock(),
                reply_to=identifier,
                sources=reply.sources,
                continuations=reply.suggestions,
                actions=reply.actions,
            )
            self.store.finish_turn(message, revision)
        except asyncio.CancelledError:
            self.store.fail_turn(
                DEFAULT_CONVERSATION_ID,
                identifier,
                "The reply was interrupted. You can retry it.",
                expected_revision=revision,
            )
            raise
        except Exception:
            log.exception("conversation.reply_failed", message_id=identifier)
            self.store.fail_turn(
                DEFAULT_CONVERSATION_ID,
                identifier,
                "I couldn't reply. Your message is saved. Try again.",
                expected_revision=revision,
            )
        finally:
            if self.identifier:
                self._publish()

    def _choice(
        self, label: str, kind: str, target: str = "", action: ChatAction | None = None
    ) -> JSONObject:
        identifier = str(uuid.uuid4())
        self.choices[identifier] = Choice(kind, target, action)
        return {"id": identifier, "label": label, "kind": kind}

    def _choose(self, choice: Choice) -> None:
        if choice.kind == "dismiss":
            self.dismiss()
        elif choice.kind in {"chat", "activity"}:
            self.mode = choice.kind
            self._publish()
        elif choice.kind == "activity_page":
            if choice.target == "newer":
                self.activity_offset = self.activity_offsets.pop()
            else:
                self.activity_offsets.append(self.activity_offset)
                self.activity_offset += int(choice.target)
            self._publish()
        elif choice.kind == "history":
            if choice.target == "newer":
                self.history_end = self.history_stack.pop()
            else:
                self.history_stack.append(self.history_end)
                index, offset = choice.target.split(":")
                self.history_end = (int(index), int(offset))
            self._publish()
        elif choice.kind == "continuation":
            self.mode = "chat"
            self.history_end = None
            self.history_stack.clear()
            self._accept(
                {
                    "conversation_id": DEFAULT_CONVERSATION_ID,
                    "message_id": str(uuid.uuid4()),
                    "text": choice.target,
                }
            )
        elif choice.kind == "retry":
            self._retry(
                {
                    "conversation_id": DEFAULT_CONVERSATION_ID,
                    "message_id": choice.target,
                    "command_id": str(uuid.uuid4()),
                }
            )
        elif (
            choice.action is not None
            and self.act is not None
            and choice.action.id not in self.action_tasks
        ):
            self.action_tasks[choice.action.id] = asyncio.create_task(
                self._act(choice.action)
            )

    async def _act(self, action: ChatAction) -> None:
        assert self.act is not None
        try:
            await self.act(action)
        except Exception:
            log.exception("conversation.action_failed", action_id=action.id)
            self._publish(error="I couldn't complete that action. You can try again.")
        else:
            if self.identifier:
                self._publish()
        finally:
            self.action_tasks.pop(action.id, None)

    def _publish(
        self, *, error: str | None = None, rejected: str | None = None
    ) -> None:
        if not self.identifier:
            return
        conversation = self.store.conversation(DEFAULT_CONVERSATION_ID)
        assert conversation is not None
        messages = self.store.messages(DEFAULT_CONVERSATION_ID)
        self.revision += 1
        self.choices.clear()
        actions = []
        if self.mode == "chat" and conversation.pending_message_id is None and messages:
            latest = messages[-1]
            if latest.role == ChatRole.ASSISTANT:
                actions.extend(
                    self._choice(text, "continuation", text)
                    for text in latest.continuations[:4]
                    if len(text) <= 160
                )

        failed = (
            messages[-1].id
            if conversation.error and messages and messages[-1].role == ChatRole.USER
            else None
        )
        if failed:
            actions.append(self._choice("Retry reply", "retry", failed))
        actions.append(
            self._choice(
                "Activity" if self.mode == "chat" else "Conversation",
                "activity" if self.mode == "chat" else "chat",
            )
        )
        actions.append(self._choice("Close", "dismiss"))
        visible, older = self._window(messages) if self.mode == "chat" else ((), None)
        activity = self._activity_page() if self.mode == "activity" else ()
        if self.mode == "activity":
            self.activity_checkpoint = self._activity_checkpoint()
            if self.act is not None:
                state = self.store.state()
                control = activity_action(
                    "resume_coordination" if state.paused else "pause_coordination",
                    "",
                    "Resume coordination" if state.paused else "Pause coordination",
                    state.revision,
                )
                actions.append(
                    self._choice(
                        control.label,
                        "resume" if state.paused else "pause",
                        action=control,
                    )
                )
            if self.activity_offsets:
                actions.append(self._choice("Newer activity", "activity_page", "newer"))
            if self.activity_offset + len(activity) < len(self.activity.entries()):
                actions.append(
                    self._choice(
                        "Older activity", "activity_page", str(max(1, len(activity)))
                    )
                )
        if older is not None:
            actions.append(
                self._choice("Older messages", "history", f"{older[0]}:{older[1]}")
            )
        if self.history_stack:
            actions.append(self._choice("Newer messages", "history", "newer"))
        last_user = next(
            (
                message.id
                for message in reversed(messages)
                if message.role == ChatRole.USER
            ),
            None,
        )
        snapshot: JSONObject = {
            "conversation": {
                "id": self.identifier,
                "revision": self.revision,
                "mode": self.mode,
                "persistent": True,
                "ttl": 0,
                "text": (
                    "Coordination is paused. Running tasks continue independently."
                    if self.store.state().paused
                    else "Coordination is active."
                )
                if self.mode == "activity"
                else "",
                "headline": None,
                "items": (),
                "actions": tuple(actions),
                "error": (error or conversation.error or "")[:1000] or None,
                "conversation_id": conversation.id,
                "pending_message_id": conversation.pending_message_id,
                "failed_message_id": failed,
                "messages": visible,
                "last_user_message_id": last_user,
                "rejected_message_id": rejected,
                "acknowledged_command_id": self.acknowledged_command_id,
                "activity": activity,
            }
        }
        assert len(json.dumps(snapshot).encode()) <= MAX_SNAPSHOT_BYTES
        self.snapshot = snapshot
        self.emit(snapshot)

    def _activity_checkpoint(self) -> object:
        return self.store.state(), self.activity.entries()

    def _activity_page(self) -> tuple[JSONObject, ...]:
        entries = self.activity.entries()
        self.activity_offset = min(self.activity_offset, max(0, len(entries) - 1))
        result: list[JSONObject] = []
        budget = TRANSCRIPT_BYTES
        for entry in entries[self.activity_offset :]:
            packet = self._activity_entry(entry)
            size = len(json.dumps(packet).encode())
            if size > budget:
                break
            result.append(packet)
            budget -= size
        return tuple(result)

    def _activity_entry(self, entry: ActivityEntry) -> JSONObject:
        kinds = {
            "pause_coordination": "pause",
            "resume_coordination": "resume",
            "stop_task": "stop",
            "open_task": "navigate",
            "inspect_task": "evidence",
            "open_source": "navigate",
            "selected_yes": "evidence",
            "selected_no": "evidence",
            "selected_unsure": "evidence",
        }
        actions = []
        for item in entry.actions:
            if item.kind in ("selected_yes", "selected_no", "selected_unsure"):
                actions.append(
                    self._choice(
                        item.label,
                        "continuation",
                        f"Regarding {entry.id} ({entry.text[:300]}): {item.label}. Apply this answer only to that connection.",
                    )
                )
            elif self.act is not None:
                actions.append(
                    self._choice(
                        item.label, kinds.get(item.kind, item.kind), action=item
                    )
                )
        for source in entry.sources[:8]:
            if self.act is not None and source.available and source.uri:
                selected = activity_action(
                    "open_source",
                    source.id,
                    f"Open {source.provider} source",
                    source=source,
                )
                actions.append(
                    self._choice(selected.label, "navigate", action=selected)
                )
        actions.append(
            self._choice(
                "Ask about this",
                "continuation",
                f"Explain Activity item {entry.id}: {entry.title[:200]}. What evidence supports it and what remains uncertain?",
            )
        )
        return {
            "id": entry.id,
            "category": entry.category,
            "title": self._bounded(entry.title, 600),
            "text": self._bounded(entry.text, 6000),
            "state": entry.state,
            "reason": self._bounded(entry.reason, 3000),
            "revisit_at": stamp(entry.revisit_at)
            if entry.revisit_at is not None
            else None,
            "sources": tuple(
                {
                    "id": source.id[:240],
                    "title": self._bounded(source.uri or source.id, 600),
                    "detail": self._bounded(source_detail(source), 1500),
                }
                for source in entry.sources[:8]
            ),
            "actions": tuple(actions),
        }

    @staticmethod
    def _bounded(text: str, size: int) -> str:
        if len(json.dumps(text).encode()) <= size:
            return text
        low, high = 0, len(text)
        while low < high:
            middle = (low + high + 1) // 2
            if len(json.dumps(text[:middle]).encode()) <= size - 100:
                low = middle
            else:
                high = middle - 1
        return text[:low] + "… Ask about this item for the complete context."

    def _window(
        self, messages: tuple[ChatMessage, ...]
    ) -> tuple[tuple[JSONObject, ...], tuple[int, int] | None]:
        if not messages:
            return (), None
        index, end = self.history_end or (len(messages) - 1, len(messages[-1].text))
        visible: list[JSONObject] = []
        budget = TRANSCRIPT_BYTES
        while index >= 0:
            message = messages[index]
            packet = self._message(message)
            packet["text"] = message.text[:end]
            size = len(json.dumps(packet).encode())
            if size > budget:
                if visible:
                    return tuple(reversed(visible)), (index, end)
                low, high = 0, end
                while low < high:
                    middle = (low + high) // 2
                    packet["text"] = message.text[middle:end]
                    if len(json.dumps(packet).encode()) <= budget - 150:
                        high = middle
                    else:
                        low = middle + 1
                packet["text"] = message.text[low:end]
                packet["text_start"] = low
                packet["excerpt"] = (
                    f"Message excerpt {low + 1}–{end} of {len(message.text)} characters"
                )
                visible.append(packet)
                older = (
                    (index, low)
                    if low
                    else ((index - 1, len(messages[index - 1].text)) if index else None)
                )
                return tuple(reversed(visible)), older
            if end < len(message.text):
                packet["excerpt"] = (
                    f"Message excerpt 1–{end} of {len(message.text)} characters"
                )
            visible.append(packet)
            budget -= size
            index -= 1
            if index >= 0:
                end = len(messages[index].text)
        return tuple(reversed(visible)), None

    @staticmethod
    def _message(message: ChatMessage) -> JSONObject:
        return {
            "id": message.id,
            "role": message.role.value,
            "text": message.text,
            "sources": tuple(
                {
                    "id": source.id[:240],
                    "title": (source.uri or source.id)[:240],
                    "detail": f"{source.provider} · revision {source.revision}"[:240],
                }
                for source in message.sources[:8]
            ),
        }

    def dismiss(self) -> None:
        self.identifier = ""
        self.choices.clear()
        self.engaged = False
        self.emit({"dismiss_conversation": True})

    async def suspend(self) -> None:
        task = self.task
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        tasks = tuple(self.action_tasks.values())
        for action_task in tasks:
            action_task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self.dismiss()

    async def close(self) -> None:
        await self.suspend()
