import asyncio
import base64
import json
import os
import random
import socket
import tempfile
import time
import unittest
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Unpack

import httpx
from anthropic import AsyncAnthropic

from pet_daemon.behavior.sprites import Pack
from pet_daemon.cli.hook import update
from pet_daemon.cli.notify import encode
from pet_daemon.commentary.context import SITUATIONS, CommentaryPayload
from pet_daemon.commentary.contracts import ModelRequest
from pet_daemon.config import Config
from pet_daemon.daemon import Daemon, bind_socket
from pet_daemon.domain import CareKind, SessionSource, SpeechReason, SpeechSituation
from pet_daemon.identity import SessionIdentity
from pet_daemon.runtime import (
    CareTrigger,
    OverlayPacket,
    ProcessIdentity,
    ReactionTrigger,
    SessionTrigger,
    Trigger,
)
from pet_daemon.sessions.processes import process_identity
from pet_daemon.sources import RepositoryTracker
from pet_daemon.sources.git import CommandOptions
from pet_daemon.work.observations import BranchObservation, PullRequestObservation
from Tests.fixtures import require, story_response

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII="
)


def request_reason(body: ModelRequest) -> str:
    instruction = body["system"][-1]["text"]
    for situation, direction in SITUATIONS.items():
        if direction in instruction:
            return (
                "check_in"
                if situation.startswith("check_in")
                else situation.split("_", 1)[0]
            )
    raise AssertionError(f"Unrecognized instruction: {instruction}")


def reply(
    speak: str | None = None, *, result: dict[str, object] | None = None
) -> httpx.Response:
    result = result if result is not None else {"speak": speak}
    return httpx.Response(
        200,
        json={
            "id": "msg_fixture",
            "type": "message",
            "role": "assistant",
            "model": "claude-haiku-5-5",
            "content": [{"type": "text", "text": json.dumps(result)}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 100, "output_tokens": 20},
        },
    )


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    @property
    def repositories(self) -> RepositoryTracker:
        provider = self.daemon.application.providers[0]
        assert isinstance(provider, RepositoryTracker)
        return provider

    async def asyncSetUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(dir="/tmp", prefix="pet-runtime-")
        root = Path(self.temp.name)
        (root / "pixel.png").write_bytes(PNG)
        (root / "manifest.json").write_text(
            json.dumps(
                {
                    "pet": "quiet",
                    "rest": "still",
                    "clips": {
                        "still": {"frames": ["pixel.png"], "fps": 1, "role": "rest"},
                    },
                }
            )
        )
        self.config = Config(
            pet_dir=root,
            daemon_socket=str(root / "daemon.sock"),
            overlay_socket=str(root / "overlay.sock"),
            observation_seconds=(3600, 3600),
            request_timeout=10,
            memory_file=root / "memory.sqlite3",
        )
        self.config.behavior.automatic_speech_cooldown_seconds = 0
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.requests: list[ModelRequest] = []
        self.story_requests: list[ModelRequest] = []
        self.http_status = 200
        self.story_status = 200
        self.speech: str | None = "The tests have reconsidered."
        self.fixture_speech: str = self.speech
        self.fixture_revision = 0
        self.speech_by_reason = {}
        self.slow = False
        self.delay = 0
        self.live_owner: ProcessIdentity | None = None

        async def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            story = body["output_config"]["format"]["schema"]["required"] == [
                "intent",
                "decisions",
                "progress",
            ]
            requests = self.story_requests if story else self.requests
            requests.append(body)
            self.entered.set()
            if self.delay:
                await asyncio.sleep(self.delay)
            if self.slow:
                await self.release.wait()
            status = self.story_status if story else self.http_status
            if status != 200:
                return httpx.Response(
                    status,
                    json={
                        "type": "error",
                        "error": {"type": "api_error", "message": "fixture failure"},
                    },
                )
            if story:
                return reply(result=story_response(body))
            reason = request_reason(body)
            if (
                self.speech == self.fixture_speech
                and reason not in self.speech_by_reason
            ):
                self.fixture_revision += 1
                self.fixture_speech = (
                    f"The fixture checks reported revision {self.fixture_revision}."
                )
                self.speech = self.fixture_speech
            speech = self.speech_by_reason.get(reason, self.speech)
            return reply(speech)

        self.client = AsyncAnthropic(
            api_key="test-fixture",
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        self.overlay = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.overlay.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 131072)
        self.overlay.bind(self.config.overlay_socket)
        self.overlay.setblocking(False)
        self.hook = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.renderer_packets: asyncio.Queue[OverlayPacket] = asyncio.Queue()
        self.accepted_deliveries = set()
        self.renderer_task = asyncio.create_task(self.render_packets())
        self.now = 0.0
        self.wall_offset = 0.0
        self.clock_frozen = False
        self.daemon = Daemon(
            self.config,
            self.client,
            clock=self.clock,
            wall_clock=self.wall_clock,
            rng=random.Random(7),
        )
        await self.daemon.start()
        self.send_status()
        await self.packet(lambda packet: packet.get("state") == "idle")

    async def asyncTearDown(self) -> None:
        await self.daemon.close()
        await self.client.close()
        self.renderer_task.cancel()
        await asyncio.gather(self.renderer_task, return_exceptions=True)
        self.overlay.close()
        self.hook.close()
        self.temp.cleanup()

    def send(self, packet: Mapping[str, object]) -> None:
        self.hook.sendto(json.dumps(packet).encode(), self.config.daemon_socket)

    def event(
        self, event: str, session: str = "one", source: str = "cli", **extra: object
    ) -> None:
        if self.live_owner is not None:
            extra.setdefault("owner", self.live_owner.packet())
        self.send(
            {
                "kind": "event",
                "event": event,
                "source": source,
                "session": session,
                **extra,
            }
        )

    def send_status(self, **extra: object) -> None:
        self.send(
            {
                "kind": "renderer",
                "event": "status",
                "x": 100,
                "y": 100,
                "width": 200,
                "height": 160,
                "screen": {"x": 0, "y": 0, "width": 1000, "height": 800},
                "pointer": {"x": 2000, "y": 2000},
                "dragging": False,
                "menu": False,
                "speech": False,
                "sleeping": self.daemon.application.behavior.sleeping,
                "system_sleeping": self.daemon.application.suspended,
                **extra,
            }
        )

    async def comment(
        self,
        reason: SpeechReason,
        context: str | None = None,
        ttl: float | None = None,
        trigger: Trigger | None = None,
    ) -> None:
        await self.daemon.application.comment(reason, context, ttl, trigger)
        if not self.overlay._closed:
            pending = tuple(
                item
                for item in self.daemon.application.delivery.deliveries.values()
                if item.status == "pending"
            )
            if pending:
                await self.wait_for(
                    lambda: all(item.status != "pending" for item in pending)
                )

    async def render_packets(self) -> None:
        while True:
            try:
                data = await asyncio.get_running_loop().sock_recv(self.overlay, 65536)
            except OSError:
                return
            packet = json.loads(data)
            if (
                packet.get("delivery_id")
                and packet.get("text", "").strip()
                and packet.get("ttl", 0) > 0
            ):
                self.send(
                    {
                        "kind": "renderer",
                        "event": "speech_accepted",
                        "delivery_id": packet["delivery_id"],
                    }
                )
                if packet["delivery_id"] in self.accepted_deliveries:
                    continue
                self.accepted_deliveries.add(packet["delivery_id"])
            await self.renderer_packets.put(packet)

    async def packet(
        self, predicate: Callable[[OverlayPacket], bool], timeout: float = 10
    ) -> OverlayPacket:
        async def receive() -> OverlayPacket:
            while True:
                packet = await self.renderer_packets.get()
                if predicate(packet):
                    delivery = self.daemon.application.delivery.deliveries.get(
                        packet.get("delivery_id")
                    )
                    if delivery is not None and delivery.status == "pending":
                        await self.wait_for(
                            lambda delivery=delivery: delivery.status != "pending"
                        )
                    return packet

        return await asyncio.wait_for(receive(), timeout)

    async def wait_for(
        self, condition: Callable[[], bool], timeout: float = 10
    ) -> None:
        async def wait() -> None:
            while not condition():
                await asyncio.sleep(0.005)

        await asyncio.wait_for(wait(), timeout)

    async def drain(self) -> list[OverlayPacket]:
        result = []
        await asyncio.sleep(0.01)
        while not self.renderer_packets.empty():
            result.append(self.renderer_packets.get_nowait())
        return result

    def install_sentiment_clips(self) -> None:
        path = self.config.pet_dir / "manifest.json"
        manifest = json.loads(path.read_text())
        for name, role in [
            ("failed", "error"),
            ("review", "attention"),
            ("jumping", "celebrate"),
        ]:
            manifest["clips"][name] = {"frames": ["pixel.png"], "fps": 1, "role": role}
        path.write_text(json.dumps(manifest))
        self.daemon.pack = self.daemon.application.behavior.pack = Pack(
            self.config.pet_dir
        )

    async def test_hook_frustration_plays_once_without_marking_work_failed(
        self,
    ) -> None:
        self.install_sentiment_clips()
        self.slow = True
        self.daemon.application.agent_activity.rng = random.Random(1)
        packet = require(
            update(
                "claude",
                "UserPromptSubmit",
                {
                    "session_id": "frustrated",
                    "prompt": "This is terrible. Stop making this worse.",
                },
            )
        )
        encoded = require(encode(packet.packet()))
        self.daemon.receive(encoded)
        reaction = await self.packet(
            lambda item: item.get("animation", {}).get("id") == "failed"
        )
        self.assertEqual(reaction["animation"]["repeats"], 3)
        self.assertEqual(
            require(self.daemon.application.behavior.execution.action).name,
            "frustration",
        )
        self.assertFalse(self.daemon.application.sessions.failed)
        self.assertEqual(self.daemon.application.agent_activity.error_until, 0)
        snapshot = self.daemon.application.context()
        selection = self.daemon.application.attention.select(
            snapshot, SpeechReason.PROMPT
        )
        payload = self.daemon.application.cognition.prompts.prompt(snapshot, selection)
        self.assertEqual(
            payload["current_work"]["request"],
            "This is terrible. Stop making this worse.",
        )

        self.daemon.receive(encoded)
        self.daemon.application.agent_activity.rng = random.Random(1)
        packet = packet.model_copy(update={"session": "another"})
        self.daemon.receive(require(encode(packet.packet())))
        self.assertFalse(
            any(
                item.get("animation", {}).get("id") == "failed"
                for item in await self.drain()
            )
        )

    async def test_skipped_frustration_is_not_retried_by_duplicate_hooks(self) -> None:
        self.install_sentiment_clips()
        packet = {
            "kind": "event",
            "source": "claude",
            "session": "frustrated",
            "event": "intent",
            "goal": "This is terrible. Stop making this worse.",
        }
        self.daemon.application.agent_activity.rng = random.Random(2)
        self.daemon.receive(json.dumps(packet).encode())
        self.daemon.application.agent_activity.rng = random.Random(1)
        self.daemon.receive(json.dumps(packet).encode())
        self.assertFalse(
            any(
                item.get("animation", {}).get("id") == "failed"
                for item in await self.drain()
            )
        )

    async def test_sentiment_cooldown_expires_and_a_neutral_prompt_clears_the_signal(
        self,
    ) -> None:
        self.install_sentiment_clips()
        self.slow = True
        now = [self.daemon.clock()]

        def clock() -> float:
            return now[0]

        self.daemon.clock = self.daemon.application.agent_activity.clock = (
            self.daemon.application.clock
        ) = self.daemon.application.sessions.clock = (
            self.daemon.application.behavior.clock
        ) = clock
        packet = {
            "kind": "event",
            "source": "claude",
            "session": "frustrated",
            "event": "intent",
            "turn": "first",
            "goal": "This is terrible. Stop making this worse.",
        }
        self.daemon.application.agent_activity.rng = random.Random(1)
        self.daemon.receive(json.dumps(packet).encode())
        self.assertEqual(
            self.daemon.application.agent_activity.sentiment_after, now[0] + 90
        )
        self.assertTrue(
            any(
                item.get("animation", {}).get("id") == "failed"
                for item in await self.drain()
            )
        )

        now[0] += 89
        self.daemon.application.behavior.tick(
            self.daemon.application.agent_activity.activity(),
            self.daemon.application.sessions,
        )
        packet["turn"] = "second"
        self.daemon.application.agent_activity.rng = random.Random(1)
        self.daemon.receive(json.dumps(packet).encode())
        self.assertFalse(
            any(
                item.get("animation", {}).get("id") == "failed"
                for item in await self.drain()
            )
        )

        now[0] += 2
        packet["turn"] = "third"
        self.daemon.application.agent_activity.rng = random.Random(1)
        self.daemon.receive(json.dumps(packet).encode())
        self.assertTrue(
            any(
                item.get("animation", {}).get("id") == "failed"
                for item in await self.drain()
            )
        )
        now[0] += 100
        self.assertIsNotNone(
            self.daemon.application.sessions.activity_view().sessions[0].user_sentiment
        )
        self.daemon.application.behavior.tick(
            self.daemon.application.agent_activity.activity(),
            self.daemon.application.sessions,
        )
        self.daemon.application.agent_activity.rng = random.Random(1)
        self.daemon.receive(json.dumps(packet).encode())
        self.assertFalse(
            any(
                item.get("animation", {}).get("id") == "failed"
                for item in await self.drain()
            )
        )

        packet.update(turn="fourth", goal="Thank you, this looks great.")
        self.daemon.receive(json.dumps(packet).encode())
        self.assertIsNone(
            self.daemon.application.sessions.activity_view().sessions[0].user_sentiment
        )

    async def test_subagent_prompts_and_disabled_sentiment_do_not_react(self) -> None:
        self.install_sentiment_clips()
        packet = {
            "kind": "event",
            "source": "claude",
            "session": "child",
            "parent": "parent",
            "event": "intent",
            "goal": "This is terrible. Stop making this worse.",
        }
        self.daemon.application.agent_activity.rng = random.Random(1)
        self.daemon.receive(json.dumps(packet).encode())
        self.assertFalse(
            any(
                item.get("animation", {}).get("id") == "failed"
                for item in await self.drain()
            )
        )
        self.assertIsNone(
            self.daemon.application.sessions.activity_view().sessions[0].user_sentiment
        )
        self.config.behavior.sentiment_reactions = False
        packet.pop("parent")
        packet["session"] = "disabled"
        self.daemon.receive(json.dumps(packet).encode())
        self.assertFalse(
            any(
                item.get("animation", {}).get("id") == "failed"
                for item in await self.drain()
            )
        )

    async def test_blocked_frustration_is_dropped_and_respects_reaction_priority(
        self,
    ) -> None:
        self.install_sentiment_clips()
        packet = {
            "kind": "event",
            "source": "claude",
            "session": "frustrated",
            "event": "intent",
            "goal": "This is terrible. Stop making this worse.",
        }
        self.daemon.application.behavior.status = require(
            self.daemon.application.behavior.status
        ).model_copy(update={"menu": True})
        self.daemon.application.agent_activity.rng = random.Random(1)
        self.daemon.receive(json.dumps(packet).encode())
        self.daemon.application.behavior.status = require(
            self.daemon.application.behavior.status
        ).model_copy(update={"menu": False})
        self.daemon.application.behavior.tick(
            self.daemon.application.agent_activity.activity(),
            self.daemon.application.sessions,
        )
        self.assertFalse(
            any(
                item.get("animation", {}).get("id") == "failed"
                for item in await self.drain()
            )
        )
        self.daemon.application.behavior.react("success")
        packet["session"] = "another"
        self.daemon.application.agent_activity.rng = random.Random(1)
        self.daemon.receive(json.dumps(packet).encode())
        self.assertEqual(
            require(self.daemon.application.behavior.execution.action).name, "success"
        )
        self.assertFalse(
            any(
                item.get("animation", {}).get("id") == "failed"
                for item in await self.drain()
            )
        )

    async def test_clear_memory_cancels_commentary_and_starts_fresh_sessions(
        self,
    ) -> None:
        self.slow = True
        self.event("intent", goal="Improve parser", repo="/repo/parser")
        await asyncio.wait_for(self.entered.wait(), 10)
        memory = self.daemon.application.cognition.memory
        memory.record_speech("Remember the parser", SpeechReason.PROMPT)
        self.send({"kind": "clear_memory"})
        await self.packet(
            lambda packet: packet.get("text") == "" and packet.get("ttl") == 0
        )
        self.release.set()
        await self.wait_for(lambda: self.daemon.application.task is None)
        self.assertEqual(self.daemon.application.sessions.activity_view().sessions, ())
        self.assertEqual(memory.recent(), [])
        self.assertEqual(memory.speech(), [])
        self.assertEqual(memory.projects(["/repo/parser"]), [])
        self.assertIsNone(
            memory.context(
                SessionIdentity(source=SessionSource.COMMAND, native_id="one")
            )
        )
        self.assertIsNone(self.daemon.application.last_speech)
        self.assertFalse(any(packet.get("text") for packet in await self.drain()))
        self.slow = False
        self.event("intent", goal="Write documentation", repo="/repo/parser")
        await self.packet(lambda packet: packet.get("text") == self.speech)
        self.daemon.application.sessions.by_id[
            SessionIdentity(source=SessionSource.COMMAND, native_id="one")
        ]
        self.assertEqual(
            require(
                self.daemon.application.memory.context(
                    SessionIdentity(source=SessionSource.COMMAND, native_id="one")
                )
            ).initial_request,
            "Write documentation",
        )
        self.assertNotIn(
            "Improve parser",
            json.dumps([record.model_dump() for record in memory.recent()]),
        )

    async def test_clear_speech_preserves_work_and_forgets_previous_remarks(
        self,
    ) -> None:
        await self.repository_fixture()
        self.event("intent", goal="Improve parser", repo=str(self.repository_path))
        await self.packet(lambda packet: packet.get("text") == self.speech)
        await self.wait_for_repository()
        await self.wait_for(lambda: self.daemon.application.task is None)
        self.event("tool_start", call="checks", tool="pytest")
        await self.wait_for(
            lambda: bool(
                self.daemon.application.sessions.by_id[
                    SessionIdentity(source=SessionSource.COMMAND, native_id="one")
                ].calls
            )
        )
        memory = self.daemon.application.memory
        sessions = self.daemon.application.sessions
        context = require(
            memory.context(
                SessionIdentity(source=SessionSource.COMMAND, native_id="one")
            )
        )
        events = memory.recent()
        targets = memory.collection_targets()
        last_attempt = self.daemon.application.attention.last_attempt
        self.send({"kind": "clear_speech"})
        await self.packet(
            lambda packet: packet.get("text") == "" and packet.get("ttl") == 0
        )
        self.assertEqual(memory.speech(), [])
        self.assertIsNone(self.daemon.application.last_speech)
        self.assertIs(self.daemon.application.sessions, sessions)
        self.assertIn(
            "checks",
            sessions.by_id[
                SessionIdentity(source=SessionSource.COMMAND, native_id="one")
            ].calls,
        )
        self.assertEqual(
            memory.context(
                SessionIdentity(source=SessionSource.COMMAND, native_id="one")
            ),
            context,
        )
        self.assertEqual(memory.recent(), events)
        self.assertEqual(memory.collection_targets(), targets)
        self.assertEqual(self.daemon.application.attention.last_attempt, last_attempt)
        await self.comment(SpeechReason.EXPLICIT, "Look at the parser")
        payload = json.loads(self.requests[-1]["messages"][-1]["content"])
        self.assertNotIn("repetition", payload)
        self.assertEqual(
            payload["current_moment"],
            {"kind": "explicit_context", "text": "Look at the parser"},
        )

    async def test_clear_speech_discards_pending_remark_and_preserves_session_context(
        self,
    ) -> None:
        self.slow = True
        self.daemon.application.memory.record_speech(
            "Earlier remark", SpeechReason.ROUTINE
        )
        self.event("intent", goal="Improve parser")
        await self.wait_for(lambda: len(self.story_requests) == 1)
        self.send({"kind": "clear_speech"})
        await self.packet(
            lambda packet: packet.get("text") == "" and packet.get("ttl") == 0
        )
        self.release.set()
        await self.wait_for(lambda: self.daemon.application.task is None)
        self.assertEqual(self.daemon.application.memory.speech(), [])
        self.assertTrue(self.daemon.application.sessions.active)
        self.assertEqual(
            require(
                self.daemon.application.memory.context(
                    SessionIdentity(source=SessionSource.COMMAND, native_id="one")
                )
            ).initial_request,
            "Improve parser",
        )
        self.assertFalse(any(packet.get("text") for packet in await self.drain()))
        self.slow = False
        await self.comment(SpeechReason.EXPLICIT, "Check again")
        packet = await self.packet(lambda packet: bool(packet.get("text")))
        self.assertEqual(packet["text"], self.speech)

    async def test_configured_speech_history_expires_in_model_requests(self) -> None:
        await self.daemon.close()
        self.config.speech_history_seconds = 10
        now = [1000]
        self.daemon = Daemon(
            self.config,
            self.client,
            wall_clock=lambda: now[0],
            rng=random.Random(7),
        )
        await self.daemon.start()
        self.send_status()
        await self.packet(lambda packet: packet.get("state") == "idle")
        self.daemon.application.memory.record_speech(
            "Earlier remark", SpeechReason.ROUTINE
        )
        now[0] += 10
        await self.comment(SpeechReason.EXPLICIT, "Say hello")
        payload = json.loads(self.requests[-1]["messages"][-1]["content"])
        self.assertNotIn("repetition", payload)
        self.assertNotIn("Earlier remark", json.dumps(payload))

    async def repository_fixture(self) -> None:
        from pet_daemon.sources.git import command

        self.repository_path = Path(self.temp.name) / "repository"
        self.repository_path.mkdir()
        for args in (
            ("init", "-b", "feature"),
            ("config", "user.email", "pet@example.test"),
            ("config", "user.name", "Pet"),
        ):
            await command("git", *args, cwd=self.repository_path)
        (self.repository_path / "tracked").write_text("initial\n")
        await command("git", "add", ".", cwd=self.repository_path)
        await command("git", "commit", "-m", "Initial", cwd=self.repository_path)
        await command(
            "git",
            "remote",
            "add",
            "origin",
            "https://example.test/owner/repository.git",
            cwd=self.repository_path,
        )
        head = (
            await command("git", "rev-parse", "HEAD", cwd=self.repository_path)
        ).strip()
        (self.repository_path / "tracked").write_text("pending edits\n")
        self.repository_pr = {
            "number": 1,
            "url": "https://example.test/owner/repository/pull/1",
            "state": "OPEN",
            "mergedAt": None,
            "isDraft": True,
            "title": "Persist parser sessions",
            "body": "Preserve session context",
            "headRefOid": head,
            "headRefName": "feature",
            "headRepositoryOwner": {"login": "owner"},
            "headRepository": {"name": "repository"},
        }

        async def boundary(*args: str, **kwargs: Unpack[CommandOptions]) -> str:
            if args[:3] == ("gh", "api", "user"):
                return "elliot"
            if args[:3] == ("gh", "repo", "view"):
                return json.dumps({"isFork": False})
            if args[0] == "gh":
                return json.dumps(
                    [self.repository_pr] if args[2] == "list" else self.repository_pr
                )
            return await command(*args, **kwargs)

        self.repository_boundary = boundary
        self.repositories.run = boundary

    async def wait_for_repository(self) -> BranchObservation:
        await self.wait_for(
            lambda: any(
                isinstance(item, BranchObservation) and item.pr
                for item in self.repositories.snapshot()
            ),
            timeout=3,
        )
        return next(
            item
            for item in self.repositories.snapshot()
            if isinstance(item, BranchObservation) and item.pr
        )

    async def connect_repository_work(self) -> None:
        self.repositories.github_interval = 0
        self.config.behavior.automatic_speech_cooldown_seconds = 90
        self.daemon.application.attention.last_attempt = self.daemon.clock()
        self.event(
            "intent", repo=str(self.repository_path), goal="Improve parser persistence"
        )
        await self.wait_for(
            lambda: any(
                item.kind == "pull_request"
                for subject in self.daemon.application.context().work
                for item in subject.story.artifacts
            ),
            timeout=3,
        )
        self.daemon.application.attention.pending = None
        self.config.behavior.automatic_speech_cooldown_seconds = 0

    async def test_exited_owner_leaves_the_reported_work_and_pr_context_available(
        self,
    ) -> None:
        from pet_daemon.sessions.processes import process_identity

        await self.repository_fixture()
        await self.connect_repository_work()
        owner = await asyncio.create_subprocess_exec("sleep", "30")
        try:
            identity = await asyncio.to_thread(process_identity, owner.pid)
            self.assertIsNotNone(identity)
            self.event(
                "tool_start",
                call="edit",
                tool="editing",
                repo=str(self.repository_path),
                owner=require(identity).packet(),
            )
            await self.wait_for_repository()
        finally:
            owner.terminate()
            await owner.wait()
        self.daemon.application.processes.checked.clear()
        await self.wait_for(
            lambda: (
                self.daemon.application.sessions.by_id[
                    SessionIdentity(source=SessionSource.COMMAND, native_id="one")
                ].presence
                == "exited"
            ),
            timeout=3,
        )
        self.assertEqual(self.daemon.application.agent_activity.activity(), "idle")
        await self.comment(SpeechReason.ROUTINE)
        self.assertEqual(self.requests, [])
        session = self.daemon.application.sessions.by_id[
            SessionIdentity(source=SessionSource.COMMAND, native_id="one")
        ]
        self.assertEqual(session.presence, "exited")
        self.assertFalse(session.active)
        subject = self.daemon.application.context().work[0]
        self.assertEqual(subject.story.intent, "Improve parser persistence")
        self.assertTrue(subject.story.artifacts)

    async def test_omitted_pull_request_changes_preserve_work_commentary(
        self,
    ) -> None:
        await self.repository_fixture()
        self.event("session_start", repo=str(self.repository_path))
        await self.wait_for_repository()
        await self.connect_repository_work()
        self.entered.clear()
        self.slow = True
        task = asyncio.create_task(self.comment(SpeechReason.ROUTINE))
        self.addCleanup(task.cancel)
        await asyncio.wait_for(self.entered.wait(), 10)
        self.repository_pr.update(state="MERGED", mergedAt="2026-10-17T00:00:00Z")
        self.repositories.github_interval = 0
        await self.wait_for(
            lambda: any(
                isinstance(item, (BranchObservation, PullRequestObservation))
                and item.pr
                and item.pr.state == "MERGED"
                for item in self.repositories.snapshot()
            ),
            timeout=3,
        )
        self.release.set()
        await task
        self.assertTrue(
            any(
                isinstance(item, (BranchObservation, PullRequestObservation))
                and item.pr
                and item.pr.state == "MERGED"
                for item in self.repositories.snapshot()
            )
        )
        packet = await self.packet(lambda packet: "text" in packet)
        self.assertEqual(
            [record.text for record in self.daemon.application.memory.speech()],
            [packet["text"]],
        )

    async def finish_observation_after_other_session_changes(self) -> CommentaryPayload:
        self.config.behavior.automatic_speech_cooldown_seconds = 90
        self.daemon.application.attention.last_attempt = self.daemon.clock()
        self.event("intent", session="focus", goal="Review parser behavior")
        self.event("intent", session="other", goal="Review the renderer")
        await self.wait_for(lambda: len(self.daemon.application.sessions.by_id) == 2)
        self.slow = True
        task = asyncio.create_task(
            self.comment(
                SpeechReason.ROUTINE,
                trigger=SessionTrigger(
                    session_id=SessionIdentity(
                        source=SessionSource.COMMAND, native_id="focus"
                    )
                ),
            )
        )
        self.addCleanup(task.cancel)
        await asyncio.wait_for(self.entered.wait(), 10)
        self.event("intent", session="other", goal="Document the renderer")
        await self.wait_for(
            lambda: (
                require(
                    self.daemon.application.memory.context(
                        SessionIdentity(source=SessionSource.COMMAND, native_id="other")
                    )
                ).request_count
                == 2
            )
        )
        self.release.set()
        await task
        return json.loads(self.requests[-1]["messages"][0]["content"])

    async def test_omitted_session_changes_preserve_focused_commentary(self) -> None:
        self.config.prompt_target_bytes = 1
        payload = await self.finish_observation_after_other_session_changes()
        self.assertNotIn("other_sessions", payload)
        self.assertEqual(
            [row.text for row in self.daemon.application.memory.speech()],
            [self.speech],
        )
        self.assertEqual(
            (await self.packet(lambda packet: "text" in packet))["text"], self.speech
        )

    async def test_unrelated_work_stays_outside_focused_commentary(self) -> None:
        payload = await self.finish_observation_after_other_session_changes()
        self.assertNotIn("other_sessions", payload)
        self.assertNotIn("Review the renderer", json.dumps(payload))
        self.assertEqual(
            [row.text for row in self.daemon.application.memory.speech()],
            [self.speech],
        )

    async def test_unrelated_repository_changes_preserve_session_commentary(
        self,
    ) -> None:
        await self.repository_fixture()
        self.config.behavior.automatic_speech_cooldown_seconds = 90
        self.daemon.application.attention.last_attempt = self.daemon.clock()
        self.config.prompt_target_bytes = 1
        self.event(
            "session_start", session="repository", repo=str(self.repository_path)
        )
        await self.wait_for_repository()
        self.event("intent", session="focus", goal="Review parser behavior")
        await self.wait_for(
            lambda: (
                SessionIdentity(source=SessionSource.COMMAND, native_id="focus")
                in self.daemon.application.sessions.by_id
            )
        )
        self.slow = True
        task = asyncio.create_task(
            self.comment(
                SpeechReason.ROUTINE,
                trigger=SessionTrigger(
                    session_id=SessionIdentity(
                        source=SessionSource.COMMAND, native_id="focus"
                    )
                ),
            )
        )
        self.addCleanup(task.cancel)
        await asyncio.wait_for(self.entered.wait(), 10)
        self.repository_pr.update(state="MERGED", mergedAt="2026-10-17T00:00:00Z")
        self.repositories.github_interval = 0
        await self.wait_for(
            lambda: any(
                isinstance(item, BranchObservation)
                and item.pr
                and item.pr.state == "MERGED"
                for item in self.repositories.snapshot()
            ),
            timeout=3,
        )
        self.release.set()
        await task
        self.assertEqual(
            [row.text for row in self.daemon.application.memory.speech()],
            [self.speech],
        )
        self.assertEqual(
            (await self.packet(lambda packet: "text" in packet))["text"], self.speech
        )

    async def test_reported_work_and_source_facts_survive_daemon_restart(self) -> None:
        await self.repository_fixture()
        self.event("session_start", repo=str(self.repository_path))
        original = await self.wait_for_repository()
        await self.connect_repository_work()
        await self.daemon.close()
        self.daemon = Daemon(self.config, self.client, rng=random.Random(7))
        self.repositories.run = self.repository_boundary
        restored = next(
            item
            for item in self.repositories.snapshot()
            if isinstance(item, BranchObservation)
        )
        self.assertEqual(
            (restored.repository, restored.branch),
            (original.repository, original.branch),
        )
        self.assertEqual(restored.pr, original.pr)
        self.assertTrue(require(restored.pr).pending)
        self.assertFalse(self.daemon.application.sessions.active)
        await self.daemon.start()
        self.send_status()
        await self.packet(lambda packet: packet.get("state") == "idle")
        await self.comment(SpeechReason.ROUTINE)
        self.assertEqual(self.requests, [])
        self.assertEqual(
            self.daemon.application.context().work[0].story.intent,
            "Improve parser persistence",
        )

    async def test_repository_changes_without_agent_intent_do_not_create_commentary(
        self,
    ) -> None:
        await self.repository_fixture()
        self.event("session_start", repo=str(self.repository_path))
        await self.wait_for_repository()
        self.repository_pr.update(state="MERGED", mergedAt="2026-10-17T00:00:00Z")
        self.repositories.github_interval = 0
        await self.wait_for(
            lambda: any(
                isinstance(item, (BranchObservation, PullRequestObservation))
                and item.pr
                and item.pr.state == "MERGED"
                for item in self.repositories.snapshot()
            ),
            timeout=3,
        )
        self.daemon.application.attention.next_observation = self.daemon.clock()
        await self.wait_for(
            lambda: (
                self.daemon.application.attention.next_observation > self.daemon.clock()
            )
        )
        self.assertEqual(self.requests, [])

    async def test_due_check_ins_mark_only_the_selected_session(self) -> None:
        self.use_clock()
        self.use_live_owner()
        self.config.idle_message_seconds = 5
        self.slow = True
        for session in ("one", "two"):
            self.event("tool_start", session=session, call="run", tool="pytest")
        await self.wait_for(lambda: len(self.daemon.application.sessions.by_id) == 2)
        self.now += 6
        await asyncio.wait_for(self.entered.wait(), 10)
        checked = [
            key
            for key, lane in self.daemon.application.sessions.by_id.items()
            if lane.last_check_in is not None
        ]
        self.assertEqual(len(checked), 1)
        unchecked = next(
            key for key in self.daemon.application.sessions.by_id if key not in checked
        )
        payload = json.loads(self.requests[-1]["messages"][-1]["content"])
        self.assertEqual(payload["current_moment"]["tool"], "pytest")
        self.release.set()
        await self.wait_for(lambda: len(self.requests) == 2)
        payload = json.loads(self.requests[-1]["messages"][-1]["content"])
        self.assertIsNotNone(
            self.daemon.application.sessions.by_id[unchecked].last_check_in
        )

    async def test_failed_story_commentary_can_retry_without_new_evidence(
        self,
    ) -> None:
        await self.repository_fixture()
        self.event("session_start", repo=str(self.repository_path))
        await self.wait_for_repository()
        await self.connect_repository_work()
        self.http_status = 500
        with self.assertLogs("pet-daemon", level="WARNING"):
            self.daemon.application.attention.next_observation = self.daemon.clock()
            await self.wait_for(
                lambda: len(self.requests) == 1 and self.daemon.application.task is None
            )
        self.assertTrue(self.daemon.application.context().work)
        self.http_status = 200
        self.daemon.application.attention.next_observation = self.daemon.clock()
        packet = await self.packet(lambda packet: packet.get("text") == self.speech)
        self.assertEqual(len(self.requests), 2)
        self.assertTrue(self.daemon.application.context().work)
        accepted = require(
            self.daemon.application.memory.message_outcome(packet["delivery_id"])
        )
        self.assertEqual(accepted.outcome, "accepted")

    def use_live_owner(self) -> None:
        self.live_owner = require(process_identity(os.getpid()))

    def clock(self) -> float:
        return self.now if self.clock_frozen else time.monotonic()

    def wall_clock(self) -> float:
        return self.now + self.wall_offset if self.clock_frozen else time.time()

    def use_clock(self) -> None:
        self.now = time.monotonic()
        self.wall_offset = time.time() - self.now
        self.clock_frozen = True
        self.daemon.application.next_prune = 0
        self.daemon.application.attention.next_observation = self.now + 3600

    async def test_idle_exercise_speaks_and_click_resets_care(self) -> None:
        manifest_path = self.config.pet_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["clips"]["jumping"] = {
            "frames": ["pixel.png"],
            "fps": 1,
            "role": "celebrate",
        }
        manifest_path.write_text(json.dumps(manifest))
        self.daemon.application.behavior.pack = Pack(self.config.pet_dir)
        self.use_clock()
        behavior = self.daemon.application.behavior
        behavior.config.naps = False
        behavior.config.roaming = False
        behavior.next_care = self.now
        behavior.next_care_speech = self.now
        behavior.execution.interrupt()
        animation = await self.packet(
            lambda packet: packet.get("animation", {}).get("id") == "jumping"
        )
        self.assertEqual(animation["animation"]["repeats"], 3)
        await self.packet(lambda packet: packet.get("text") == self.speech)
        self.assertEqual(request_reason(self.requests[-1]), "care")
        self.assertIn(
            SITUATIONS[SpeechSituation.CARE_EXERCISE],
            self.requests[-1]["system"][-1]["text"],
        )
        self.send({"kind": "renderer", "event": "click"})
        await self.wait_for(lambda: len(self.requests) == 2)
        self.assertEqual(request_reason(self.requests[-1]), "click")
        self.assertGreaterEqual(behavior.next_care, self.now + 300)
        self.assertIsNone(behavior.care_remark)

    async def test_new_work_discards_pending_care_speech(self) -> None:
        care_speech = "A stretch might help with those parser checks."
        self.speech_by_reason["care"] = care_speech
        self.slow = True
        task = asyncio.create_task(
            self.comment(
                SpeechReason.CARE, trigger=CareTrigger(care=CareKind.CONSTERNATION)
            )
        )
        self.addCleanup(task.cancel)
        await asyncio.wait_for(self.entered.wait(), 10)
        self.event("intent", goal="Review the parser")
        await self.wait_for(lambda: self.daemon.application.sessions.active)
        self.release.set()
        await task
        self.assertFalse(
            any(packet.get("text") == care_speech for packet in await self.drain())
        )

    async def test_real_socket_tracks_parallel_calls_without_model_latency(
        self,
    ) -> None:
        self.slow = True
        self.send({"kind": "speak", "context": "Watch these builds"})
        await asyncio.wait_for(self.entered.wait(), 10)
        self.event("tool_start", call="build", tool="cargo build", manual=True)
        await self.packet(lambda packet: packet.get("state") == "working")
        self.event(
            "tool_start", session="two", source="cursor", call="lint", tool="lint"
        )
        await self.wait_for(lambda: len(self.daemon.application.sessions.by_id) == 2)
        self.event("tool_done", call="build", manual=True)
        await self.wait_for(
            lambda: (
                not self.daemon.application.sessions.by_id[
                    SessionIdentity(source=SessionSource.COMMAND, native_id="one")
                ].calls
            )
        )
        self.assertTrue(self.daemon.application.sessions.working)
        self.event("tool_done", session="two", source="cursor", call="lint")
        await self.wait_for(
            lambda: (
                not self.daemon.application.sessions.by_id[
                    SessionIdentity(source=SessionSource.CURSOR, native_id="two")
                ].calls
            )
        )
        self.assertEqual(self.daemon.application.agent_activity.activity(), "working")
        self.event("turn_done", session="two", source="cursor")
        await self.packet(lambda packet: packet.get("state") == "idle")
        self.release.set()
        speech = await self.packet(lambda packet: "text" in packet)
        self.assertEqual(speech["text"], self.speech)
        self.assertEqual(speech["ttl"], 15)
        self.assertEqual(len(self.requests), 1)

    async def test_investigation_and_input_wait_animate_while_commentary_is_pending(
        self,
    ) -> None:
        manifest_path = self.config.pet_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for clip, role in (
            ("review", "attention"),
            ("waiting", "attention"),
            ("running", "work"),
        ):
            manifest["clips"][clip] = {"frames": ["pixel.png"], "fps": 1, "role": role}
        manifest_path.write_text(json.dumps(manifest))
        self.daemon.application.behavior.pack = Pack(self.config.pet_dir)
        self.use_clock()
        self.slow = True
        self.event("intent", goal="Inspect the parser")
        await asyncio.wait_for(self.entered.wait(), 10)
        await self.packet(
            lambda packet: packet.get("animation", {}).get("id") == "review"
        )
        self.now += 8
        self.event("tool_start", call="read", tool="parser.py", tool_name="Read")
        reaction = await self.packet(
            lambda packet: (
                packet.get("animation", {}).get("id") == "review"
                and packet["animation"].get("repeats") is not None
            )
        )
        self.assertGreater(require(reaction["animation"]["repeats"]), 0)
        self.now += 60
        self.send(
            {
                "kind": "renderer",
                "event": "animation_done",
                "command_id": reaction["animation"]["command_id"],
            }
        )
        self.event("waiting", call="input", tool="AskUserQuestion")
        waiting = await self.packet(
            lambda packet: packet.get("animation", {}).get("id") == "waiting"
        )
        self.assertGreater(require(waiting["animation"]["repeats"]), 0)
        self.assertIsNotNone(self.daemon.application.task)

    async def test_recovery_celebrates_without_waiting_for_commentary(self) -> None:
        manifest_path = self.config.pet_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["clips"]["hop"] = {
            "frames": ["pixel.png"],
            "fps": 1,
            "role": "celebrate",
        }
        manifest_path.write_text(json.dumps(manifest))
        self.daemon.application.behavior.pack = Pack(self.config.pet_dir)
        self.slow = True
        self.event("intent", goal="Run checks")
        await asyncio.wait_for(self.entered.wait(), 10)
        self.event("tool_start", call="tests", tool="pytest")
        await self.wait_for(
            lambda: (
                "tests"
                in self.daemon.application.sessions.by_id[
                    SessionIdentity(source=SessionSource.COMMAND, native_id="one")
                ].calls
            )
        )
        self.event("tool_error", call="tests")
        await self.wait_for(lambda: self.daemon.application.sessions.failed)
        self.event("tool_start", call="retry", tool="pytest")
        await self.wait_for(
            lambda: (
                "retry"
                in self.daemon.application.sessions.by_id[
                    SessionIdentity(source=SessionSource.COMMAND, native_id="one")
                ].calls
            )
        )
        self.event("tool_done", call="retry", outcome="success")
        await self.packet(lambda packet: packet.get("animation", {}).get("id") == "hop")
        self.assertTrue(self.daemon.application.sessions.active)
        await self.drain()
        self.event("tool_start", call="cancelled", tool="pytest")
        await self.wait_for(
            lambda: (
                "cancelled"
                in self.daemon.application.sessions.by_id[
                    SessionIdentity(source=SessionSource.COMMAND, native_id="one")
                ].calls
            )
        )
        self.event("tool_done", call="cancelled", outcome="cancelled")
        await self.wait_for(
            lambda: (
                not self.daemon.application.sessions.by_id[
                    SessionIdentity(source=SessionSource.COMMAND, native_id="one")
                ].calls
            )
        )
        packets = await self.drain()
        self.assertFalse(
            any(packet.get("animation", {}).get("id") == "hop" for packet in packets)
        )

    async def test_prompt_triggers_commentary_without_waiting_for_routine_timer(
        self,
    ) -> None:
        self.event("intent", source="codex", goal="Run local checks")
        speech = await self.packet(lambda packet: "text" in packet)
        self.assertEqual(speech["text"], self.speech)
        snapshot = json.loads(self.requests[0]["messages"][-1]["content"])
        self.assertEqual(request_reason(self.requests[-1]), "prompt")
        self.assertEqual(snapshot["current_work"]["request"], "Run local checks")

    async def test_click_requests_fresh_speech_without_waiting_for_routine_timer(
        self,
    ) -> None:
        self.send({"kind": "renderer", "event": "click"})
        speech = await self.packet(lambda packet: "text" in packet)
        self.assertEqual(speech["text"], self.speech)
        self.assertEqual(speech["ttl"], 15)
        self.assertEqual(request_reason(self.requests[-1]), "click")
        schema = self.requests[0]["output_config"]["format"]["schema"]
        self.assertEqual(set(schema["properties"]), {"speak"})
        self.assertEqual(schema["required"], ["speak"])
        self.assertEqual(schema["properties"]["speak"]["type"], "string")

    async def test_napping_pauses_routine_commentary_and_click_wakes_for_speech(
        self,
    ) -> None:
        self.use_clock()
        self.config.behavior.nap_interval = (1, 1)
        self.config.behavior.nap_seconds = (10, 10)
        self.daemon.application.behavior.inactive_since = self.now
        self.daemon.application.behavior.execution.interrupt()
        self.daemon.application.behavior.next_nap = self.now + 1
        self.now += 240
        await self.packet(lambda packet: packet.get("sleeping") is True)
        self.daemon.application.attention.next_observation = self.now
        self.now += 2
        self.send_status(x=101)
        await self.wait_for(
            lambda: require(self.daemon.application.behavior.status).x == 101
        )
        self.assertTrue(self.daemon.application.behavior.sleeping)
        self.assertEqual(self.requests, [])
        self.send({"kind": "renderer", "event": "click"})
        await self.packet(lambda packet: packet.get("sleeping") is False)
        speech = await self.packet(lambda packet: "text" in packet)
        self.assertEqual(speech["text"], self.speech)
        snapshot = json.loads(self.requests[-1]["messages"][-1]["content"])
        self.assertEqual(request_reason(self.requests[-1]), "click")
        self.assertNotIn("sleeping", snapshot["current_moment"])

    async def test_prompt_wakes_a_nap_before_tool_work_starts(self) -> None:
        self.use_clock()
        self.config.behavior.nap_interval = (1, 1)
        self.daemon.application.behavior.inactive_since = self.now
        self.daemon.application.behavior.execution.interrupt()
        self.daemon.application.behavior.next_nap = self.now + 1
        self.now += 240
        await self.packet(lambda packet: packet.get("sleeping") is True)
        self.event("intent", source="codex", goal="Run checks")
        await self.packet(lambda packet: packet.get("sleeping") is False)
        speech = await self.packet(lambda packet: "text" in packet)
        self.assertEqual(speech["text"], self.speech)
        self.assertTrue(self.daemon.application.sessions.working)
        self.assertFalse(self.daemon.application.behavior.sleeping)

    async def test_click_requests_fresh_speech_even_after_recent_speech(self) -> None:
        self.send({"kind": "speak", "context": "Inspect the build", "ttl": 1})
        original = await self.packet(lambda packet: "text" in packet)
        await self.wait_for(lambda: self.daemon.application.task is None)
        self.speech = "The build has something fresh to say."
        self.send({"kind": "renderer", "event": "click"})
        fresh = await self.packet(lambda packet: "text" in packet)
        self.assertNotEqual(fresh["text"], original["text"])
        self.assertEqual(fresh["text"], self.speech)
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(request_reason(self.requests[-1]), "click")

    async def test_clicks_during_a_request_use_its_response_without_another_model_pass(
        self,
    ) -> None:
        self.slow = True
        self.send({"kind": "renderer", "event": "click"})
        await asyncio.wait_for(self.entered.wait(), 10)
        for _ in range(5):
            self.send({"kind": "renderer", "event": "click"})
        self.send_status(x=101)
        await self.wait_for(
            lambda: require(self.daemon.application.behavior.status).x == 101
        )
        self.release.set()
        speech = await self.packet(lambda packet: "text" in packet)
        self.assertEqual(speech["text"], self.speech)
        await self.wait_for(lambda: self.daemon.application.task is None)
        self.assertFalse(self.daemon.application.clicked)
        self.assertEqual(len(self.requests), 1)

    async def test_click_during_a_silent_observation_gets_a_fresh_comment(self) -> None:
        self.slow = True
        self.speech_by_reason["routine"] = None
        self.event("tool_start", call="checks", tool="pytest")
        await self.wait_for(lambda: self.daemon.application.sessions.active)
        self.daemon.application.attention.next_observation = self.daemon.clock()
        await asyncio.wait_for(self.entered.wait(), 10)
        self.send({"kind": "renderer", "event": "click"})
        self.send_status(x=101)
        await self.wait_for(
            lambda: require(self.daemon.application.behavior.status).x == 101
        )
        self.release.set()
        speech = await self.packet(lambda packet: "text" in packet)
        self.assertEqual(speech["text"], self.speech)
        reasons = [request_reason(request) for request in self.requests]
        self.assertEqual(reasons, ["routine", "click"])

    async def test_automatic_cooldown_allows_one_waiting_reminder_and_manual_speech(
        self,
    ) -> None:
        self.use_clock()
        self.config.behavior.automatic_speech_cooldown_seconds = 90
        self.event("intent", goal="Inspect checks")
        await self.packet(lambda packet: "text" in packet)
        await self.wait_for(lambda: self.daemon.application.task is None)
        self.event("turn_done")
        await self.wait_for(
            lambda: self.daemon.application.attention.pending is not None
        )
        self.now += 2
        await self.wait_for(lambda: self.daemon.application.attention.pending is None)
        self.assertEqual(len(self.requests), 1)
        self.event("intent", goal="Choose the next step")
        await self.wait_for(
            lambda: (
                self.daemon.application.attention.pending is None
                and self.daemon.application.sessions.active
            )
        )
        self.event("waiting", call="question")
        await self.packet(lambda packet: "text" in packet)
        await self.wait_for(lambda: self.daemon.application.task is None)
        self.event("waiting", call="question")
        await self.wait_for(
            lambda: (
                self.daemon.application.sessions.by_id[
                    SessionIdentity(source=SessionSource.COMMAND, native_id="one")
                ].waiting
            )
        )
        self.assertEqual(len(self.requests), 2)
        self.send({"kind": "renderer", "event": "click"})
        await self.packet(lambda packet: "text" in packet)
        self.assertEqual(len(self.requests), 3)

    async def test_each_new_question_can_bypass_the_automatic_cooldown_once(
        self,
    ) -> None:
        self.config.behavior.automatic_speech_cooldown_seconds = 90
        self.event("waiting", call="first", progress="Choose a strategy")
        await self.packet(lambda packet: "text" in packet)
        await self.wait_for(lambda: self.daemon.application.task is None)
        self.event("waiting", call="second", progress="Choose a target")
        await self.packet(lambda packet: "text" in packet)
        await self.wait_for(lambda: self.daemon.application.task is None)
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(
            [
                json.loads(body["messages"][-1]["content"])["current_moment"][
                    "reported_progress"
                ]
                for body in self.requests
            ],
            ["Choose a strategy", "Choose a target"],
        )
        self.event("waiting", call="second", progress="Still choosing a target")
        await self.wait_for(
            lambda: (
                self.daemon.application.sessions.by_id[
                    SessionIdentity(source=SessionSource.COMMAND, native_id="one")
                ].progress
                == "Still choosing a target"
            )
        )
        self.assertEqual(len(self.requests), 2)

    async def test_new_question_discards_the_pending_reminder_for_the_first(
        self,
    ) -> None:
        self.slow = True
        self.event("waiting", call="first", progress="Choose a strategy")
        await asyncio.wait_for(self.entered.wait(), 10)
        self.event("waiting", call="second", progress="Choose a target")
        await self.wait_for(
            lambda: (
                self.daemon.application.sessions.by_id[
                    SessionIdentity(source=SessionSource.COMMAND, native_id="one")
                ].waiting_call
                == "second"
            )
        )
        self.release.set()
        await self.packet(lambda packet: "text" in packet)
        await self.wait_for(lambda: self.daemon.application.task is None)
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(len(self.daemon.application.cognition.memory.speech()), 1)
        self.assertEqual(
            json.loads(self.requests[-1]["messages"][-1]["content"])["current_moment"][
                "reported_progress"
            ],
            "Choose a target",
        )

    async def test_waiting_reminder_is_discarded_when_background_call_finishes(
        self,
    ) -> None:
        self.slow = True
        self.event("tool_start", call="background", tool="Read")
        self.event("waiting", call="question", progress="Choose a strategy")
        await asyncio.wait_for(self.entered.wait(), 10)
        self.event("tool_done", call="background")
        await self.wait_for(
            lambda: (
                not self.daemon.application.sessions.by_id[
                    SessionIdentity(source=SessionSource.COMMAND, native_id="one")
                ].calls
            )
        )
        self.assertFalse(
            self.daemon.application.sessions.by_id[
                SessionIdentity(source=SessionSource.COMMAND, native_id="one")
            ].waiting
        )
        self.release.set()
        await self.wait_for(lambda: self.daemon.application.task is None)
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(self.daemon.application.cognition.memory.speech(), [])
        self.assertFalse(any("text" in packet for packet in await self.drain()))

    async def test_native_permission_wait_without_call_id_ends_when_the_tool_completes(
        self,
    ) -> None:
        base = {
            "session_id": "native",
            "tool_name": "Bash",
            "tool_input": {"command": "pytest"},
        }
        for name, fields in (
            ("PreToolUse", {"tool_use_id": "tests"}),
            ("PermissionRequest", {}),
        ):
            packet = require(update("claude", name, {**base, **fields}))
            self.daemon.receive(require(encode(packet.packet())))
        self.assertIsNone(packet.call)
        self.assertTrue(
            self.daemon.application.sessions.by_id[
                SessionIdentity(source=SessionSource.CLAUDE, native_id="native")
            ].waiting
        )
        snapshot = self.daemon.application.sessions.activity_view().sessions[0]
        self.assertFalse(snapshot.waiting_for_input)
        self.assertIsNotNone(snapshot.waiting_seconds)
        self.assertEqual(self.requests, [])
        packet = require(
            update(
                "claude",
                "PostToolUse",
                {**base, "tool_use_id": "tests", "tool_response": {"exit_code": 0}},
            )
        )
        self.daemon.receive(require(encode(packet.packet())))
        snapshot = self.daemon.application.sessions.activity_view().sessions[0]
        self.assertFalse(snapshot.waiting_for_input)
        self.assertIsNone(snapshot.waiting_seconds)
        self.assertEqual(snapshot.active_tools, ())
        await self.wait_for(lambda: self.daemon.application.task is None)
        self.assertEqual(self.daemon.application.cognition.memory.speech(), [])
        self.assertFalse(any("text" in packet for packet in await self.drain()))

    async def test_manual_speech_delays_automatic_outcome_commentary(self) -> None:
        self.use_clock()
        self.config.behavior.automatic_speech_cooldown_seconds = 90
        self.event("tool_start", call="checks", tool="pytest")
        await self.wait_for(lambda: self.daemon.application.sessions.active)
        self.send({"kind": "renderer", "event": "click"})
        await self.packet(lambda packet: "text" in packet)
        await self.wait_for(lambda: self.daemon.application.task is None)
        self.event("turn_done")
        await self.wait_for(
            lambda: self.daemon.application.attention.pending is not None
        )
        self.now += 2
        await self.wait_for(lambda: self.daemon.application.attention.pending is None)
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(request_reason(self.requests[0]), "click")

    async def test_click_preempts_pending_automatic_commentary(self) -> None:
        self.slow = True
        self.speech_by_reason["prompt"] = "Superseded acknowledgment."
        self.event("intent", goal="Inspect checks")
        await asyncio.wait_for(self.entered.wait(), 10)
        self.send({"kind": "renderer", "event": "click"})
        await self.wait_for(lambda: len(self.story_requests) == 2)
        self.release.set()
        speech = await self.packet(lambda packet: "text" in packet)
        self.assertEqual(speech["text"], self.speech)
        self.assertEqual([request_reason(body) for body in self.requests], ["click"])
        self.assertNotIn(
            "Superseded acknowledgment.",
            json.dumps(
                [
                    record.model_dump()
                    for record in self.daemon.application.cognition.memory.speech()
                ]
            ),
        )

    async def test_hooks_persist_without_requesting_commentary(self) -> None:
        self.event("intent", goal="Repair the parser", cwd="/repo/parser")
        await self.wait_for(
            lambda: bool(self.daemon.application.cognition.memory.recent())
        )
        self.assertIn(
            "Repair the parser",
            json.dumps(
                [
                    record.model_dump()
                    for record in self.daemon.application.cognition.memory.recent()
                ]
            ),
        )
        self.assertEqual(self.requests, [])

    async def test_visible_speech_persists_after_renderer_delivery(self) -> None:
        await self.comment(SpeechReason.EXPLICIT, "Inspect the build")
        await self.packet(lambda packet: "text" in packet)
        self.assertIn(
            self.speech,
            json.dumps(
                [
                    record.model_dump()
                    for record in self.daemon.application.cognition.memory.speech()
                ]
            ),
        )

    async def test_zero_duration_speech_is_absent_from_memory(self) -> None:
        await self.comment(SpeechReason.EXPLICIT, "Inspect the build", ttl=0)
        await self.packet(lambda packet: "text" in packet)
        self.assertEqual(self.daemon.application.cognition.memory.speech(), [])

    async def test_computer_sleep_cancels_commentary_and_click_recovers_after_wake(
        self,
    ) -> None:
        self.slow = True
        self.send({"kind": "speak", "context": "Inspect the build"})
        await asyncio.wait_for(self.entered.wait(), 10)
        self.send({"kind": "renderer", "event": "system_sleep"})
        await self.wait_for(
            lambda: (
                self.daemon.application.suspended
                and self.daemon.application.task is None
            )
        )
        self.release.set()
        self.assertEqual(self.daemon.application.cognition.memory.speech(), [])
        self.assertFalse(any("text" in packet for packet in await self.drain()))
        self.send_status(system_sleeping=False, x=102)
        await self.wait_for(lambda: not self.daemon.application.suspended)
        self.send({"kind": "renderer", "event": "click"})
        packet = await self.packet(lambda packet: "text" in packet)
        self.assertEqual(packet["text"], self.speech)

    async def test_repeated_automatic_speech_is_suppressed_without_another_model_call(
        self,
    ) -> None:
        history = "The parser still needs its edge cases checked."
        self.speech = history
        self.daemon.application.memory.record_speech(history, SpeechReason.ROUTINE)
        self.event("intent", goal="Review parser edge cases")
        await self.wait_for(
            lambda: len(self.requests) == 1 and self.daemon.application.task is None
        )
        self.assertEqual(
            [row.text for row in self.daemon.application.memory.speech()], [history]
        )
        self.assertFalse(any(packet.get("text") for packet in await self.drain()))

    async def test_story_updates_before_commentary_and_reuses_unchanged_evidence(
        self,
    ) -> None:
        self.event("intent", goal="Refactor the parser")
        await self.packet(lambda packet: "text" in packet)
        await self.wait_for(lambda: self.daemon.application.task is None)
        self.assertEqual(len(self.story_requests), 1)
        narrative = require(
            self.daemon.application.memory.load_narrative(
                SessionIdentity(source=SessionSource.COMMAND, native_id="one")
            )
        )
        self.assertEqual(narrative.intent.text, "Refactor the parser")
        payload = json.loads(self.requests[0]["messages"][0]["content"])
        self.assertEqual(payload["current_work"]["story"], "Refactor the parser")
        await self.comment(SpeechReason.CLICK)
        self.assertEqual(len(self.story_requests), 1)
        self.assertEqual(len(self.requests), 2)
        await self.drain()
        self.event("intent", goal="Investigate the parser timeout")
        await self.packet(lambda packet: "text" in packet)
        self.assertEqual(len(self.story_requests), 2)
        self.assertEqual(
            require(
                self.daemon.application.memory.load_narrative(
                    SessionIdentity(source=SessionSource.COMMAND, native_id="one")
                )
            ).intent.text,
            "Investigate the parser timeout",
        )

    async def test_story_failure_defers_automatic_speech_and_retains_prior_story(
        self,
    ) -> None:
        self.event("intent", goal="Refactor the parser")
        await self.packet(lambda packet: "text" in packet)
        await self.wait_for(lambda: self.daemon.application.task is None)
        previous = self.daemon.application.memory.load_narrative(
            SessionIdentity(source=SessionSource.COMMAND, native_id="one")
        )
        self.story_status = 500
        self.event("intent", goal="Validate parser behavior")
        await self.wait_for(
            lambda: (
                len(self.story_requests) == 2 and self.daemon.application.task is None
            )
        )
        self.assertEqual(
            self.daemon.application.memory.load_narrative(
                SessionIdentity(source=SessionSource.COMMAND, native_id="one")
            ),
            previous,
        )
        self.assertEqual(len(self.requests), 1)
        self.assertFalse(any(packet.get("text") for packet in await self.drain()))
        self.send({"kind": "renderer", "event": "click"})
        await self.packet(lambda packet: "text" in packet)
        payload = json.loads(self.requests[-1]["messages"][0]["content"])
        self.assertNotIn("Refactor the parser", json.dumps(payload))

    async def test_silent_commentary_keeps_story_without_acknowledging_speech(
        self,
    ) -> None:
        self.speech = None
        self.event("intent", goal="Refactor the parser")
        await self.wait_for(
            lambda: len(self.requests) == 1 and self.daemon.application.task is None
        )
        self.assertEqual(
            require(
                self.daemon.application.memory.load_narrative(
                    SessionIdentity(source=SessionSource.COMMAND, native_id="one")
                )
            ).intent.text,
            "Refactor the parser",
        )
        self.assertEqual(self.daemon.application.memory.speech(), [])
        self.assertFalse(any(packet.get("text") for packet in await self.drain()))

    async def test_followup_during_story_generation_discards_the_stale_story(
        self,
    ) -> None:
        self.slow = True
        self.event("intent", goal="Refactor the parser", turn="first")
        await asyncio.wait_for(self.entered.wait(), 10)
        self.assertEqual(self.requests, [])
        self.event(
            "intent", goal="Keep the parser API unchanged", turn="first", manual=True
        )
        await self.wait_for(
            lambda: (
                "Keep the parser API unchanged"
                in require(
                    self.daemon.application.memory.context(
                        SessionIdentity(source=SessionSource.COMMAND, native_id="one")
                    )
                ).recent_requests
            )
        )
        self.release.set()
        await self.packet(lambda packet: "text" in packet)
        await self.wait_for(lambda: self.daemon.application.task is None)
        narrative = require(
            self.daemon.application.memory.load_narrative(
                SessionIdentity(source=SessionSource.COMMAND, native_id="one")
            )
        )
        self.assertEqual(narrative.intent.text, "Keep the parser API unchanged")
        self.assertEqual(len(self.requests), 1)

    async def test_interleaved_work_keeps_separate_stories(self) -> None:
        for session, goal in (
            ("parser", "Refactor the parser"),
            ("hooks", "Review Python hooks"),
        ):
            self.event("intent", session=session, goal=goal)
            await self.packet(lambda packet: "text" in packet)
            await self.wait_for(lambda: self.daemon.application.task is None)
        memory = self.daemon.application.memory
        self.assertEqual(
            require(
                memory.load_narrative(
                    SessionIdentity(source=SessionSource.COMMAND, native_id="parser")
                )
            ).intent.text,
            "Refactor the parser",
        )
        self.assertEqual(
            require(
                memory.load_narrative(
                    SessionIdentity(source=SessionSource.COMMAND, native_id="hooks")
                )
            ).intent.text,
            "Review Python hooks",
        )
        payload = json.loads(self.requests[-1]["messages"][0]["content"])
        self.assertNotIn("Refactor the parser", json.dumps(payload))
        for request, excluded in zip(self.story_requests, ("Python hooks", "parser")):
            self.assertNotIn(excluded, request["messages"][0]["content"])

    async def test_followup_with_the_same_native_turn_reaches_the_next_prompt(
        self,
    ) -> None:
        self.event(
            "intent", source="codex", goal="Compress the workflow", turn="native-turn"
        )
        await self.packet(lambda packet: "text" in packet)
        await self.wait_for(lambda: self.daemon.application.task is None)
        self.event(
            "intent",
            source="codex",
            goal="Explain the counter limits",
            turn="native-turn",
        )
        await self.packet(lambda packet: "text" in packet)
        self.assertEqual(len(self.requests), 2)
        payload = json.loads(self.requests[-1]["messages"][0]["content"])
        self.assertEqual(
            payload["current_work"]["request"], "Explain the counter limits"
        )

    async def test_restart_keeps_story_without_restoring_live_calls(self) -> None:
        self.event("intent", goal="Refactor the parser", turn="first")
        await self.packet(lambda packet: "text" in packet)
        self.event("tool_start", call="checks", tool="pytest")
        await self.wait_for(lambda: self.daemon.application.sessions.working)
        saved = self.daemon.application.memory.load_narrative(
            SessionIdentity(source=SessionSource.COMMAND, native_id="one")
        )
        await self.daemon.close()
        self.daemon = Daemon(self.config, self.client, rng=random.Random(7))
        self.assertFalse(self.daemon.application.sessions.active)
        self.assertEqual(
            self.daemon.application.memory.load_narrative(
                SessionIdentity(source=SessionSource.COMMAND, native_id="one")
            ),
            saved,
        )
        await self.daemon.start()
        self.send_status()
        await self.packet(lambda packet: packet.get("state") == "idle")
        self.event("intent", goal="Continue parser validation", turn="second")
        await self.packet(lambda packet: "text" in packet)
        self.assertFalse(
            self.daemon.application.sessions.by_id[
                SessionIdentity(source=SessionSource.COMMAND, native_id="one")
            ].calls
        )
        self.assertEqual(
            require(
                self.daemon.application.memory.load_narrative(
                    SessionIdentity(source=SessionSource.COMMAND, native_id="one")
                )
            ).intent.text,
            "Continue parser validation",
        )
        evidence = json.loads(self.story_requests[-1]["messages"][0]["content"])
        self.assertIn("Refactor the parser", json.dumps(evidence))

    async def test_routine_observations_require_a_change_in_work(self) -> None:
        self.use_clock()
        self.event("tool_start", call="checks", tool="pytest")
        await self.wait_for(lambda: self.daemon.application.sessions.working)
        self.daemon.application.attention.next_observation = self.now
        await self.packet(lambda packet: "text" in packet)
        await self.wait_for(lambda: self.daemon.application.task is None)
        self.now += 20
        self.daemon.application.attention.next_observation = self.now
        self.send_status(x=101)
        await self.wait_for(
            lambda: require(self.daemon.application.behavior.status).x == 101
        )
        self.assertEqual(len(self.requests), 1)
        self.event("progress", progress="Parser checks passed")
        await self.wait_for(
            lambda: (
                self.daemon.application.sessions.by_id[
                    SessionIdentity(source=SessionSource.COMMAND, native_id="one")
                ].progress
                == "Parser checks passed"
            )
        )
        self.daemon.application.attention.next_observation = self.now
        await self.packet(lambda packet: "text" in packet)
        self.assertEqual(len(self.requests), 2)

    async def test_repeated_failures_request_speech_at_the_third_failure_without_command_history(
        self,
    ) -> None:
        self.use_clock()
        for index in range(5):
            self.event("tool_start", call=f"check-{index}", tool="pytest")
            await self.wait_for(
                lambda index=index: (
                    SessionIdentity(source=SessionSource.COMMAND, native_id="one")
                    in self.daemon.application.sessions.by_id
                    and f"check-{index}"
                    in self.daemon.application.sessions.by_id[
                        SessionIdentity(source=SessionSource.COMMAND, native_id="one")
                    ].calls
                )
            )
            self.event(
                "tool_error",
                call=f"check-{index}",
                tool="pytest",
                detail="Parser assertion failed",
                error_signature="parser:error",
            )
            await self.wait_for(
                lambda index=index: (
                    self.daemon.application.sessions.by_id[
                        SessionIdentity(source=SessionSource.COMMAND, native_id="one")
                    ].failure
                    is not None
                    and require(
                        self.daemon.application.sessions.by_id[
                            SessionIdentity(
                                source=SessionSource.COMMAND, native_id="one"
                            )
                        ].failure
                    ).count
                    == index + 1
                )
            )
            if index == 2:
                self.now += 2
                await self.packet(lambda packet: "text" in packet)
                await self.wait_for(lambda: self.daemon.application.task is None)
        self.assertEqual(len(self.requests), 1)
        events = self.daemon.application.cognition.memory.recent(limit=10)
        failures = [event for event in events if event.event_type == "tool_error"]
        self.assertEqual(failures, [])
        self.assertEqual(
            require(
                self.daemon.application.sessions.by_id[
                    SessionIdentity(source=SessionSource.COMMAND, native_id="one")
                ].failure
            ).count,
            5,
        )

    async def test_recovered_session_skips_queued_failure_when_another_session_is_still_failing(
        self,
    ) -> None:
        self.use_clock()
        for session in ("one", "two"):
            self.event("tool_start", session=session, call="checks", tool="pytest")
            await self.wait_for(
                lambda session=session: (
                    SessionIdentity(source=SessionSource.COMMAND, native_id=session)
                    in self.daemon.application.sessions.by_id
                    and "checks"
                    in self.daemon.application.sessions.by_id[
                        SessionIdentity(source=SessionSource.COMMAND, native_id=session)
                    ].calls
                )
            )
            self.event("tool_error", session=session, call="checks")
            for index in range(2):
                call = f"failed-{index}"
                self.event("tool_start", session=session, call=call, tool="pytest")
                await self.wait_for(
                    lambda call=call, session=session: (
                        call
                        in self.daemon.application.sessions.by_id[
                            SessionIdentity(
                                source=SessionSource.COMMAND, native_id=session
                            )
                        ].calls
                    )
                )
                self.event("tool_error", session=session, call=call)
                await self.wait_for(
                    lambda index=index, session=session: (
                        require(
                            self.daemon.application.sessions.by_id[
                                SessionIdentity(
                                    source=SessionSource.COMMAND, native_id=session
                                )
                            ].failure
                        ).count
                        == index + 2
                    )
                )
            await self.wait_for(
                lambda session=session: bool(
                    self.daemon.application.attention.pending
                    and isinstance(
                        self.daemon.application.attention.pending[2], ReactionTrigger
                    )
                    and self.daemon.application.attention.pending[2].reaction.session
                    == SessionIdentity(source=SessionSource.COMMAND, native_id=session)
                )
            )
        self.event("tool_start", session="two", call="retry", tool="pytest")
        await self.wait_for(
            lambda: (
                "retry"
                in self.daemon.application.sessions.by_id[
                    SessionIdentity(source=SessionSource.COMMAND, native_id="two")
                ].calls
            )
        )
        self.event("tool_done", session="two", call="retry")
        await self.wait_for(
            lambda: (
                require(
                    self.daemon.application.sessions.by_id[
                        SessionIdentity(source=SessionSource.COMMAND, native_id="two")
                    ].failure
                ).resolved
            )
        )
        self.now += 2
        await self.wait_for(lambda: self.daemon.application.attention.pending is None)
        self.assertTrue(self.daemon.application.sessions.failed)
        self.assertFalse(
            any(request_reason(body) == "failure" for body in self.requests)
        )

    async def test_parent_routed_child_failure_retains_the_child_turn_report(
        self,
    ) -> None:
        self.use_clock()
        self.event(
            "subagent_start",
            session="child",
            source="claude",
            parent="main",
            parent_call="delegate",
            goal="Inspect parser",
            repo="/repo/parser",
        )
        await self.wait_for(
            lambda: (
                SessionIdentity(source=SessionSource.CLAUDE, native_id="child")
                in self.daemon.application.sessions.by_id
            )
        )
        self.event(
            "tool_start",
            session="main",
            source="claude",
            parent_call="delegate",
            call="check",
            tool="pytest",
        )
        await self.wait_for(
            lambda: (
                "check"
                in self.daemon.application.sessions.by_id[
                    SessionIdentity(source=SessionSource.CLAUDE, native_id="child")
                ].calls
            )
        )
        self.event(
            "tool_error",
            session="main",
            source="claude",
            parent_call="delegate",
            call="check",
            detail="Parser assertion failed",
            error_signature="parser:error",
        )
        await self.wait_for(
            lambda: (
                self.daemon.application.sessions.by_id[
                    SessionIdentity(source=SessionSource.CLAUDE, native_id="child")
                ].failure
                is not None
            )
        )
        self.event("turn_error", session="child", source="claude")
        await self.wait_for(
            lambda: (
                self.daemon.application.sessions.by_id[
                    SessionIdentity(source=SessionSource.CLAUDE, native_id="child")
                ].finished
            )
        )
        self.assertEqual(self.requests, [])
        self.assertIsNone(self.daemon.application.attention.pending)
        failures = [
            row
            for row in self.daemon.application.cognition.memory.recent(limit=10)
            if row.event_type == "turn_error"
        ]
        self.assertEqual(len(failures), 1)
        self.assertEqual(
            failures[0].session,
            SessionIdentity(source=SessionSource.CLAUDE, native_id="child"),
        )
        self.assertEqual(failures[0].working_directory, "/repo/parser")

    async def test_failure_request_and_memory_exclude_injected_diagnostics(
        self,
    ) -> None:
        self.use_clock()
        self.event("tool_start", call="check", tool="pytest")
        await self.wait_for(lambda: self.daemon.application.sessions.working)
        self.event(
            "tool_error",
            call="check",
            detail="SECRET PRIVATE STACK",
            error_signature="SECRET PRIVATE HASH",
            progress="SECRET FAILED COMMAND OUTPUT",
        )
        await self.wait_for(lambda: self.daemon.application.sessions.failed)
        await self.comment(SpeechReason.FAILURE)
        await self.packet(lambda packet: "text" in packet)
        body = self.requests[-1]
        self.assertNotIn("SECRET", json.dumps(body))
        self.assertNotIn(
            "SECRET",
            json.dumps(self.daemon.application.sessions.activity_view().model_dump()),
        )
        self.assertNotIn(
            "SECRET",
            json.dumps(
                [
                    record.model_dump()
                    for record in self.daemon.application.cognition.memory.recent()
                ]
            ),
        )
        moment = json.loads(body["messages"][0]["content"])["current_moment"]
        self.assertEqual(moment, {"kind": "failed_call", "operation": "testing"})

    async def test_failed_renderer_delivery_is_absent_from_memory(self) -> None:
        Path(self.config.overlay_socket).unlink()
        await self.daemon.application.comment(
            SpeechReason.EXPLICIT, "Inspect the build"
        )
        self.assertEqual(self.daemon.application.cognition.memory.speech(), [])
        self.assertIsNone(self.daemon.application.last_speech)

    async def test_completion_reacts_to_one_session_while_other_tools_keep_working(
        self,
    ) -> None:
        self.event("tool_start", source="codex", call="checks", tool="pytest")
        self.event(
            "tool_start",
            session="two",
            source="cursor",
            call="build",
            tool="cargo build",
        )
        await self.wait_for(lambda: len(self.daemon.application.sessions.by_id) == 2)
        self.event("turn_done", source="codex", progress="Checks passed")
        speech = await self.packet(lambda packet: "text" in packet, timeout=2)
        snapshot = json.loads(self.requests[0]["messages"][-1]["content"])
        self.assertEqual(request_reason(self.requests[-1]), "completion")
        self.assertEqual(snapshot["current_moment"]["outcome"], "success")
        self.assertEqual(
            snapshot["current_moment"]["reported_progress"], "Checks passed"
        )
        self.assertNotIn("activity", snapshot)
        self.assertEqual(speech["text"], self.speech)
        self.assertEqual(self.daemon.application.agent_activity.activity(), "working")

    async def test_unmentioned_tool_changes_preserve_work_commentary(self) -> None:
        self.slow = True
        self.event("tool_start", source="codex", call="checks", tool="pytest")
        await self.packet(lambda packet: packet.get("state") == "working")
        task = asyncio.create_task(self.comment(SpeechReason.ROUTINE))
        await asyncio.wait_for(self.entered.wait(), 10)
        self.event("tool_done", source="codex", call="checks", progress="Tests passed")
        self.event("tool_start", source="codex", call="lint", tool="lint")
        await self.wait_for(
            lambda: (
                "lint"
                in self.daemon.application.sessions.by_id[
                    SessionIdentity(source=SessionSource.CODEX, native_id="one")
                ].calls
            )
        )
        self.release.set()
        await task
        self.assertEqual(
            [record.text for record in self.daemon.application.memory.speech()],
            [self.speech],
        )
        self.assertTrue(any(packet.get("text") for packet in await self.drain()))

    async def test_check_in_uses_each_tasks_quiet_time_and_repeats_after_the_interval(
        self,
    ) -> None:
        self.use_clock()
        self.use_live_owner()
        self.event("tool_start", source="codex", call="checks", tool="pytest")
        self.event(
            "tool_start", session="two", source="cursor", call="build", tool="build"
        )
        await self.wait_for(lambda: len(self.daemon.application.sessions.by_id) == 2)
        self.now += 299
        self.event(
            "progress", session="two", source="cursor", progress="Build progressing"
        )
        await self.wait_for(
            lambda: (
                self.daemon.application.sessions.by_id[
                    SessionIdentity(source=SessionSource.CURSOR, native_id="two")
                ].progress
                == "Build progressing"
            )
        )
        await self.wait_for(lambda: self.daemon.application.next_prune > self.now)
        self.assertEqual(self.requests, [])
        self.now += 1
        speech = await self.packet(lambda packet: "text" in packet)
        self.assertEqual(speech["text"], self.speech)
        snapshot = json.loads(self.requests[0]["messages"][-1]["content"])
        self.assertEqual(request_reason(self.requests[-1]), "check_in")
        self.assertEqual(snapshot["current_moment"]["tool"], "pytest")
        self.assertIsNotNone(
            self.daemon.application.sessions.by_id[
                SessionIdentity(source=SessionSource.CODEX, native_id="one")
            ].last_check_in
        )
        self.assertNotIn("other_sessions", snapshot)
        self.assertTrue(
            self.daemon.application.sessions.by_id[
                SessionIdentity(source=SessionSource.CURSOR, native_id="two")
            ].active
        )
        self.assertEqual(snapshot["current_moment"]["elapsed"], "5 minutes")
        self.now += 299
        self.event(
            "progress", session="two", source="cursor", progress="Build continuing"
        )
        await self.wait_for(lambda: self.daemon.application.next_prune > self.now)
        self.assertEqual(len(self.requests), 1)
        self.now += 1
        await self.packet(lambda packet: "text" in packet)
        snapshot = json.loads(self.requests[1]["messages"][-1]["content"])
        self.assertEqual(snapshot["current_moment"]["tool"], "pytest")

    async def test_failed_check_in_waits_the_interval_before_retrying(self) -> None:
        self.use_clock()
        self.use_live_owner()
        self.http_status = 500
        self.event("tool_start", source="codex", call="checks", tool="pytest")
        await self.wait_for(lambda: self.daemon.application.sessions.working)
        with self.assertLogs("pet-daemon", level="WARNING"):
            self.now += 300
            await asyncio.wait_for(self.entered.wait(), 10)
            await self.wait_for(lambda: self.daemon.application.task is None)
        self.assertEqual(len(self.requests), 1)
        self.now += 299
        await self.wait_for(lambda: self.daemon.application.next_prune > self.now)
        self.assertEqual(len(self.requests), 1)
        self.http_status = 200
        self.now += 1
        await self.packet(lambda packet: "text" in packet)
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(request_reason(self.requests[-1]), "check_in")

    async def test_waiting_check_in_is_discarded_after_progress_resumes_the_session(
        self,
    ) -> None:
        self.use_clock()
        self.use_live_owner()
        self.slow = True
        self.event(
            "waiting",
            source="claude",
            call="question",
            tool="AskUserQuestion",
            progress="Choose a strategy",
        )
        await asyncio.wait_for(self.entered.wait(), 10)
        snapshot = json.loads(self.requests[0]["messages"][-1]["content"])
        self.assertEqual(request_reason(self.requests[0]), "check_in")
        self.assertEqual(snapshot["current_moment"]["kind"], "waiting")
        self.event("progress", source="claude", progress="Investigating the strategy")
        await self.wait_for(
            lambda: (
                not self.daemon.application.sessions.by_id[
                    SessionIdentity(source=SessionSource.CLAUDE, native_id="one")
                ].waiting
            )
        )
        self.release.set()
        await self.wait_for(lambda: self.daemon.application.task is None)
        self.assertEqual(self.daemon.application.cognition.memory.speech(), [])
        self.assertFalse(any("text" in packet for packet in await self.drain()))

    async def test_waiting_check_in_keeps_speech_while_another_task_progresses(
        self,
    ) -> None:
        self.use_clock()
        self.use_live_owner()
        self.slow = True
        self.event(
            "waiting",
            source="claude",
            call="question",
            tool="AskUserQuestion",
            progress="Choose a strategy",
        )
        self.event(
            "tool_start", session="two", source="cursor", call="build", tool="build"
        )
        await self.wait_for(lambda: len(self.daemon.application.sessions.by_id) == 2)
        await asyncio.wait_for(self.entered.wait(), 10)

        self.now += 300
        self.event(
            "progress", session="two", source="cursor", progress="Building modules"
        )
        await self.wait_for(
            lambda: (
                self.daemon.application.sessions.by_id[
                    SessionIdentity(source=SessionSource.CURSOR, native_id="two")
                ].progress
                == "Building modules"
            )
        )
        self.release.set()
        speech = await self.packet(lambda packet: "text" in packet)

        snapshot = json.loads(self.requests[0]["messages"][-1]["content"])
        self.assertEqual(request_reason(self.requests[-1]), "check_in")
        self.assertEqual(snapshot["current_moment"]["kind"], "waiting")
        self.assertNotIn("activity", snapshot)
        self.assertEqual(snapshot["current_moment"]["elapsed"], "less than a minute")
        self.assertEqual(speech["text"], self.speech)
        self.assertEqual(self.daemon.application.agent_activity.activity(), "working")

    async def test_explicit_waiting_after_progress_replaces_the_pending_reminder(
        self,
    ) -> None:
        self.use_clock()
        self.slow = True
        self.event(
            "waiting",
            source="claude",
            call="question",
            tool="AskUserQuestion",
            progress="Choose a strategy",
        )
        await asyncio.wait_for(self.entered.wait(), 10)
        self.now += 1
        self.event("progress", source="claude", progress="Investigating the strategy")
        await self.wait_for(
            lambda: (
                not self.daemon.application.sessions.by_id[
                    SessionIdentity(source=SessionSource.CLAUDE, native_id="one")
                ].waiting
            )
        )
        self.now += 1
        self.event(
            "waiting",
            source="claude",
            call="question",
            tool="AskUserQuestion",
            progress="Choose a target",
        )
        await self.wait_for(
            lambda: (
                self.daemon.application.sessions.by_id[
                    SessionIdentity(source=SessionSource.CLAUDE, native_id="one")
                ].waiting
            )
        )
        self.release.set()
        speech = await self.packet(lambda packet: "text" in packet)
        self.assertEqual(speech["text"], self.speech)
        await self.wait_for(lambda: self.daemon.application.task is None)
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(len(self.daemon.application.cognition.memory.speech()), 1)
        snapshot = self.daemon.application.sessions.activity_view().sessions[0]
        self.assertTrue(snapshot.waiting_for_input)
        self.assertEqual(snapshot.waiting_seconds, 0)
        self.assertEqual(
            json.loads(self.requests[-1]["messages"][-1]["content"])["current_moment"][
                "reported_progress"
            ],
            "Choose a target",
        )

    async def test_check_in_discards_speech_when_the_active_call_finishes(self) -> None:
        self.use_clock()
        self.use_live_owner()
        self.slow = True
        self.event("tool_start", source="codex", call="checks", tool="pytest")
        await self.wait_for(lambda: self.daemon.application.sessions.working)
        self.now += 300
        await asyncio.wait_for(self.entered.wait(), 10)
        self.event("tool_done", source="codex", call="checks", progress="Tests passed")
        await self.wait_for(
            lambda: (
                not self.daemon.application.sessions.by_id[
                    SessionIdentity(source=SessionSource.CODEX, native_id="one")
                ].calls
            )
        )
        self.assertEqual(self.daemon.application.agent_activity.activity(), "working")
        self.release.set()
        await self.wait_for(lambda: self.daemon.application.task is None)
        self.assertEqual(self.daemon.application.cognition.memory.speech(), [])
        self.assertFalse(any("text" in packet for packet in await self.drain()))

    async def test_check_in_discards_speech_when_a_different_call_starts(self) -> None:
        self.use_clock()
        self.use_live_owner()
        self.slow = True
        self.event("tool_start", source="codex", call="checks", tool="pytest")
        await self.wait_for(lambda: self.daemon.application.sessions.working)
        self.now += 300
        await asyncio.wait_for(self.entered.wait(), 10)
        self.event("tool_done", source="codex", call="checks", progress="Tests passed")
        self.event("tool_start", source="codex", call="lint", tool="lint")
        await self.wait_for(
            lambda: (
                "lint"
                in self.daemon.application.sessions.by_id[
                    SessionIdentity(source=SessionSource.CODEX, native_id="one")
                ].calls
            )
        )
        self.release.set()
        await self.wait_for(lambda: self.daemon.application.task is None)
        self.assertEqual(self.daemon.application.cognition.memory.speech(), [])
        self.assertFalse(any("text" in packet for packet in await self.drain()))

    async def test_recovery_discards_stale_automatic_speech_without_recording_it(
        self,
    ) -> None:
        self.config.behavior.automatic_speech_cooldown_seconds = 90
        self.daemon.application.attention.last_attempt = self.daemon.clock()
        self.slow = True
        self.event("intent", source="claude", goal="Repair tests")
        self.event("tool_start", source="claude", call="bad", tool="pytest")
        await self.packet(lambda packet: packet.get("state") == "working")
        self.event("tool_error", source="claude", call="bad", detail="AssertionError")
        await self.wait_for(lambda: self.daemon.application.sessions.failed)
        self.daemon.application.attention.pending = None
        task = asyncio.create_task(self.comment(SpeechReason.FAILURE))
        await asyncio.wait_for(self.entered.wait(), 10)
        self.event("tool_start", source="claude", call="retry", tool="pytest")
        self.event("tool_done", source="claude", call="retry")
        await self.wait_for(lambda: not self.daemon.application.sessions.failed)
        self.release.set()
        await task
        self.assertEqual(self.daemon.application.cognition.memory.speech(), [])
        self.assertFalse(any("text" in packet for packet in await self.drain()))

    async def test_trouble_speech_stays_discarded_after_recovery_and_another_failure(
        self,
    ) -> None:
        self.use_clock()
        self.slow = True
        for index in range(3):
            self.event("tool_start", call=str(index), tool="pytest")
            self.event("tool_error", call=str(index))

        def failed_three_times() -> bool:
            session = self.daemon.application.sessions.by_id.get(
                SessionIdentity(source=SessionSource.COMMAND, native_id="one")
            )
            return (
                session is not None
                and session.failure is not None
                and session.failure.count == 3
            )

        await self.wait_for(failed_three_times)
        self.now += 2
        await asyncio.wait_for(self.entered.wait(), 10)
        self.config.behavior.automatic_speech_cooldown_seconds = 90
        self.event("tool_start", call="retry", tool="pytest")
        self.event("tool_done", call="retry")
        self.event("tool_start", call="another", tool="pytest")
        self.event("tool_error", call="another")
        await self.wait_for(
            lambda: (
                require(
                    self.daemon.application.sessions.by_id[
                        SessionIdentity(source=SessionSource.COMMAND, native_id="one")
                    ].failure
                ).count
                == 1
            )
        )
        self.release.set()
        await self.wait_for(lambda: self.daemon.application.task is None)
        self.assertEqual(self.daemon.application.cognition.memory.speech(), [])
        self.assertFalse(any("text" in packet for packet in await self.drain()))

    async def test_recovery_speech_stays_discarded_after_another_failure_and_recovery(
        self,
    ) -> None:
        self.use_clock()
        self.slow = True
        self.event("tool_start", call="bad", tool="pytest")
        self.event("tool_error", call="bad")
        self.event("tool_start", call="retry", tool="pytest")
        self.event("tool_done", call="retry")

        def recovered() -> bool:
            session = self.daemon.application.sessions.by_id.get(
                SessionIdentity(source=SessionSource.COMMAND, native_id="one")
            )
            return (
                session is not None
                and session.failure is not None
                and session.failure.resolved
            )

        await self.wait_for(recovered)
        self.now += 2
        await asyncio.wait_for(self.entered.wait(), 10)
        self.config.behavior.automatic_speech_cooldown_seconds = 90
        self.event("tool_start", call="another", tool="pytest")
        self.event("tool_error", call="another")
        self.event("tool_start", call="retry-again", tool="pytest")
        self.event("tool_done", call="retry-again")
        await self.wait_for(
            lambda: (
                self.daemon.application.sessions.by_id[
                    SessionIdentity(source=SessionSource.COMMAND, native_id="one")
                ].last_tool
                == "pytest"
                and not self.daemon.application.sessions.by_id[
                    SessionIdentity(source=SessionSource.COMMAND, native_id="one")
                ].calls
            )
        )
        await self.wait_for(lambda: self.daemon.application.reactions.serial >= 5)
        self.release.set()
        await self.wait_for(lambda: self.daemon.application.task is None)
        self.assertEqual(self.daemon.application.cognition.memory.speech(), [])
        self.assertFalse(any("text" in packet for packet in await self.drain()))

    async def test_failed_request_leaves_fast_state_delivery_running(self) -> None:
        self.http_status = 500
        with self.assertLogs("pet-daemon", level="WARNING"):
            await self.comment(SpeechReason.EXPLICIT, "Check the build")
        self.assertEqual(len(self.requests), 1)
        self.event("tool_start", call="work", tool="make", manual=True)
        await self.packet(lambda packet: packet.get("state") == "working")
        self.event("tool_done", call="work", manual=True)
        await self.packet(lambda packet: packet.get("state") == "idle")
        self.http_status = 200
        await self.comment(SpeechReason.EXPLICIT, "Check the build")
        self.assertEqual(
            (await self.packet(lambda packet: "text" in packet))["text"], self.speech
        )

    async def test_timeout_cancels_request_and_accepts_next_commentary(self) -> None:
        self.slow = True
        self.config.request_timeout = 0.03
        with self.assertLogs("pet-daemon", level="WARNING"):
            await self.comment(SpeechReason.EXPLICIT, "Check the build")
        self.assertEqual(len(self.requests), 1)
        self.slow = False
        self.config.request_timeout = 2
        await self.comment(SpeechReason.EXPLICIT, "Try again", ttl=7)
        packet = await self.packet(lambda packet: "text" in packet)
        self.assertEqual(packet["ttl"], 7)

    async def test_forwards_long_unicode_speech_without_trimming(self) -> None:
        self.speech = "One sufficiently long observation. 猫\n" * 200
        await self.comment(SpeechReason.EXPLICIT, "Observe the parallel work")
        packet = await self.packet(lambda packet: "text" in packet)
        self.assertGreater(len(json.dumps(packet).encode()), 2048)
        self.assertEqual(packet["text"], self.speech)
        self.assertAlmostEqual(packet["ttl"], 3 + len(self.speech.split()) / 2)

    async def test_explicit_speech_duration_overrides_long_speech_reading_time(
        self,
    ) -> None:
        self.speech = "A detailed observation about the ongoing build. " * 100
        self.send({"kind": "speak", "context": "Inspect the build", "ttl": 2.5})
        packet = await self.packet(lambda packet: "text" in packet)
        self.assertEqual(packet["text"], self.speech)
        self.assertEqual(packet["ttl"], 2.5)

    async def test_wrapper_failure_records_command_activity_without_diagnostics(
        self,
    ) -> None:
        root = Path(__file__).resolve().parents[1]
        env = dict(os.environ, PET_DAEMON_SOCKET=self.config.daemon_socket)
        child = await asyncio.create_subprocess_exec(
            str(root / "examples/pet-run"), "/bin/sh", "-c", "exit 7", env=env
        )
        self.assertEqual(await child.wait(), 7)
        await self.wait_for(
            lambda: (
                bool(self.daemon.application.sessions.by_id)
                and all(
                    lane.finished
                    for lane in self.daemon.application.sessions.by_id.values()
                )
            )
        )
        lane = next(iter(self.daemon.application.sessions.by_id.values()))
        self.assertEqual(lane.outcome, "error")
        self.assertEqual(require(lane.failure).activity, "command")
        self.assertFalse(hasattr(require(lane.failure), "detail"))
        await self.comment(SpeechReason.FAILURE)
        snapshot = json.loads(self.requests[-1]["messages"][-1]["content"])
        self.assertEqual(
            snapshot["current_moment"], {"kind": "failed_call", "operation": "command"}
        )
        self.assertNotIn("Exit code", json.dumps(snapshot))

    async def test_live_listener_is_preserved_and_owned_socket_cleans_up(self) -> None:
        path = self.config.daemon_socket
        with self.assertRaisesRegex(ValueError, "already listening"):
            bind_socket(path)
        self.assertEqual(os.stat(path).st_mode & 0o777, 0o600)
        self.event("tool_start", call="test", manual=True)
        await self.packet(lambda packet: packet.get("state") == "working")
        await self.daemon.close()
        self.assertFalse(Path(path).exists())

    async def test_startup_failure_cleans_only_its_own_socket_and_renderer(
        self,
    ) -> None:
        await self.daemon.close()
        self.config.overlay_binary = Path(self.temp.name) / "bad-renderer"
        self.config.overlay_binary.write_text("#!/bin/sh\nexit 4\n")
        self.config.overlay_binary.chmod(0o755)
        daemon = Daemon(self.config, self.client)
        with self.assertRaisesRegex(RuntimeError, "Renderer exited"):
            await daemon.run()
        self.assertFalse(Path(self.config.daemon_socket).exists())
        self.assertTrue(Path(self.config.overlay_socket).exists())
        assert daemon.child is not None
        self.assertEqual(daemon.child.returncode, 4)

    async def test_malformed_packets_do_not_change_valid_session_state(self) -> None:
        self.event("tool_start", call="real", manual=True)
        await self.packet(lambda packet: packet.get("state") == "working")
        for data in (b"{", b"\xff", b'{"kind":"event","source":[]}', b"x" * 2001):
            self.hook.sendto(data, self.config.daemon_socket)
        self.event("tool_done", call="real", manual=True)
        await self.packet(lambda packet: packet.get("state") == "idle")
        self.assertFalse(self.daemon.application.sessions.working)


class ConfigurationTests(unittest.TestCase):
    def test_behavior_fields_reject_values_with_the_wrong_shape(self) -> None:
        for name, value in (
            ("care_interval", 5),
            ("nap_seconds", 20),
            ("idle_pause", 30),
            ("work_burst_seconds", 12),
            ("walk_speed", [20, 40]),
            ("pointer_radius", [50, 100]),
        ):
            with (
                self.subTest(name=name, value=value),
                self.assertRaisesRegex(ValueError, f"behavior.{name}"),
            ):
                Config.model_validate(
                    {"pet_dir": Path("fixture"), "behavior": {name: value}}
                )

    def test_speech_history_window_rejects_nonpositive_or_nonfinite_values(
        self,
    ) -> None:
        for value in (0, -1, True, "30", float("inf")):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "speech_history_seconds"),
            ):
                Config.model_validate(
                    {"pet_dir": Path("fixture"), "speech_history_seconds": value}
                )

    def test_loads_startup_paths_relative_to_config_and_configurable_timing(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pet.json"
            path.write_text(
                json.dumps(
                    {
                        "pet_dir": "pets/other",
                        "observation_seconds": [15, 25],
                        "idle_message_seconds": 450,
                        "prompt_target_bytes": 12000,
                        "memory_retention_days": 14,
                        "memory_max_events": 500,
                        "persona": "A calm garden companion.",
                        "log_file": "analysis/pet.jsonl",
                        "memory_file": "state/companion.sqlite3",
                        "behavior": {"walk_seconds": [2, 4], "roaming": False},
                    }
                )
            )
            config = Config.load(path)
            self.assertEqual(config.pet_dir, Path(directory).resolve() / "pets/other")
            self.assertEqual(
                config.log_file, Path(directory).resolve() / "analysis/pet.jsonl"
            )
            self.assertEqual(
                config.memory_file,
                Path(directory).resolve() / "state/companion.sqlite3",
            )
            self.assertEqual(config.persona, "A calm garden companion.")
            self.assertEqual(config.observation_seconds, (15, 25))
            self.assertEqual(config.idle_message_seconds, 450)
            self.assertEqual(config.prompt_target_bytes, 12000)
            self.assertEqual(config.memory_retention_days, 14)
            self.assertEqual(config.memory_max_events, 500)
            self.assertFalse(config.behavior.roaming)
            self.assertEqual(config.behavior.walk_seconds, (2, 4))

    def test_rejects_nonpositive_or_nonfinite_idle_message_intervals(self) -> None:
        for seconds in (0, -1, True, "300", None, float("nan"), float("inf")):
            with (
                self.subTest(seconds=seconds),
                self.assertRaisesRegex(ValueError, "idle_message_seconds"),
            ):
                Config.model_validate(
                    {"pet_dir": Path("fixture"), "idle_message_seconds": seconds}
                )

    def test_rejects_empty_or_nontext_persona(self) -> None:
        for persona in ("", "   ", None, 3):
            with self.subTest(persona=persona), self.assertRaises(ValueError):
                Config.model_validate({"pet_dir": Path("fixture"), "persona": persona})

    def test_rejects_invalid_memory_limits(self) -> None:
        for field in (
            "prompt_target_bytes",
            "memory_retention_days",
            "memory_max_events",
        ):
            for value in (0, -1, True):
                with (
                    self.subTest(field=field, value=value),
                    self.assertRaises(ValueError),
                ):
                    Config.model_validate({"pet_dir": Path("fixture"), field: value})

    def test_rejects_invalid_intervals_and_shared_socket_before_startup(self) -> None:
        for fields in (
            {"observation_seconds": (90, 60)},
            {"daemon_socket": "/tmp/same", "overlay_socket": "/tmp/same"},
        ):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                Config.model_validate({"pet_dir": Path("fixture"), **fields})
