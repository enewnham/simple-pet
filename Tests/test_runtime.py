import asyncio
import base64
import json
import os
import random
import runpy
import socket
import tempfile
import unittest
from collections.abc import Callable, Mapping
from dataclasses import asdict, replace
from pathlib import Path
from typing import Unpack

import httpx
from anthropic import AsyncAnthropic

from pet_daemon.commentary.context import SITUATIONS, CommentaryPayload
from pet_daemon.commentary.contracts import ModelRequest
from pet_daemon.config import Config
from pet_daemon.daemon import Daemon, bind_socket
from pet_daemon.runtime import (
    CareTrigger,
    EpisodeTrigger,
    OverlayPacket,
    SessionTrigger,
    Trigger,
)
from pet_daemon.sources import RepositoryTracker
from pet_daemon.sources.git import CommandOptions
from pet_daemon.sprites import Pack
from pet_daemon.work.observations import RepositoryObservation
from Tests.fixtures import require

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII="
)


def set_config_value(config: Config, name: str, value: object) -> None:
    setattr(config, name, value)


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
    speak: str | None = None, session_context: str | None = None, summary: bool = False
) -> httpx.Response:
    result = {"session_context": session_context} if summary else {"speak": speak}
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
            request_timeout=2,
            memory_file=root / "memory.sqlite3",
        )
        self.config.behavior.automatic_speech_cooldown_seconds = 0
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.retry_entered = asyncio.Event()
        self.retry_slow = False
        self.retry_delay = 0
        self.requests: list[ModelRequest] = []
        self.summary_requests: list[ModelRequest] = []
        self.http_status = 200
        self.summary_status = 200
        self.speech: str | None = "The tests have reconsidered."
        self.fixture_speech: str = self.speech
        self.fixture_revision = 0
        self.speech_by_reason = {}
        self.session_context = None
        self.slow = False
        self.delay = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            response_context = self.session_context
            summary = body["output_config"]["format"]["schema"]["required"] == [
                "session_context"
            ]
            requests = self.summary_requests if summary else self.requests
            requests.append(body)
            self.entered.set()
            if not summary and "<reconsideration>" in body["system"][-1]["text"]:
                self.retry_entered.set()
                if self.retry_slow:
                    await self.release.wait()
                if self.retry_delay:
                    await asyncio.sleep(self.retry_delay)
            if self.delay:
                await asyncio.sleep(self.delay)
            if self.slow:
                await self.release.wait()
            status = self.summary_status if summary else self.http_status
            if status != 200:
                return httpx.Response(
                    status,
                    json={
                        "type": "error",
                        "error": {"type": "api_error", "message": "fixture failure"},
                    },
                )
            if summary:
                return reply(session_context=response_context, summary=True)
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
            return reply(speech, session_context=response_context)

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
        self.daemon = Daemon(self.config, self.client, rng=random.Random(7))
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
        self, event: str, session: str = "one", harness: str = "cli", **extra: object
    ) -> None:
        self.send(
            {
                "kind": "event",
                "event": event,
                "harness": harness,
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

    async def observe(
        self,
        reason: str,
        context: str | None = None,
        ttl: float | None = None,
        trigger: Trigger | None = None,
    ) -> None:
        await self.daemon.application.observe(reason, context, ttl, trigger)
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
        self, predicate: Callable[[OverlayPacket], bool], timeout: float = 1
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

    async def wait_for(self, condition: Callable[[], bool], timeout: float = 1) -> None:
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
        self.daemon.application.events.rng = random.Random(1)
        root = Path(__file__).resolve().parents[1]
        hook = runpy.run_path(str(root / "scripts" / "pet-hook"))
        packet = hook["update"](
            "claude",
            "UserPromptSubmit",
            {
                "session_id": "frustrated",
                "prompt": "This is terrible. Stop making this worse.",
            },
        )
        encoded = hook["CLIENT"]["encode"](packet)
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
        self.assertEqual(self.daemon.application.events.error_until, 0)
        snapshot = self.daemon.application.snapshot()
        selection = self.daemon.application.attention.select(snapshot, "prompt")
        payload = self.daemon.application.cognition.context.prompt(snapshot, selection)
        self.assertLessEqual(
            require(require(payload["focus_session"])["user_sentiment"])["compound"],
            -0.45,
        )

        self.daemon.receive(encoded)
        self.daemon.application.events.rng = random.Random(1)
        packet["session"] = "another"
        self.daemon.receive(hook["CLIENT"]["encode"](packet))
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
            "harness": "claude",
            "session": "frustrated",
            "event": "intent",
            "goal": "This is terrible. Stop making this worse.",
        }
        self.daemon.application.events.rng = random.Random(2)
        self.daemon.receive(json.dumps(packet).encode())
        self.daemon.application.events.rng = random.Random(1)
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

        self.daemon.clock = self.daemon.application.events.clock = (
            self.daemon.application.clock
        ) = self.daemon.application.sessions.clock = (
            self.daemon.application.behavior.clock
        ) = clock
        packet = {
            "kind": "event",
            "harness": "claude",
            "session": "frustrated",
            "event": "intent",
            "turn": "first",
            "goal": "This is terrible. Stop making this worse.",
        }
        self.daemon.application.events.rng = random.Random(1)
        self.daemon.receive(json.dumps(packet).encode())
        self.assertEqual(self.daemon.application.events.sentiment_after, now[0] + 90)
        self.assertTrue(
            any(
                item.get("animation", {}).get("id") == "failed"
                for item in await self.drain()
            )
        )

        now[0] += 89
        self.daemon.application.behavior.tick(
            self.daemon.application.events.activity(), self.daemon.application.sessions
        )
        packet["turn"] = "second"
        self.daemon.application.events.rng = random.Random(1)
        self.daemon.receive(json.dumps(packet).encode())
        self.assertFalse(
            any(
                item.get("animation", {}).get("id") == "failed"
                for item in await self.drain()
            )
        )

        now[0] += 2
        packet["turn"] = "third"
        self.daemon.application.events.rng = random.Random(1)
        self.daemon.receive(json.dumps(packet).encode())
        self.assertTrue(
            any(
                item.get("animation", {}).get("id") == "failed"
                for item in await self.drain()
            )
        )
        now[0] += 100
        self.assertIsNotNone(
            self.daemon.application.sessions.snapshot().sessions[0].user_sentiment
        )
        self.daemon.application.behavior.tick(
            self.daemon.application.events.activity(), self.daemon.application.sessions
        )
        self.daemon.application.events.rng = random.Random(1)
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
            self.daemon.application.sessions.snapshot().sessions[0].user_sentiment
        )

    async def test_subagent_prompts_and_disabled_sentiment_do_not_react(self) -> None:
        self.install_sentiment_clips()
        packet = {
            "kind": "event",
            "harness": "claude",
            "session": "child",
            "parent": "parent",
            "event": "intent",
            "goal": "This is terrible. Stop making this worse.",
        }
        self.daemon.application.events.rng = random.Random(1)
        self.daemon.receive(json.dumps(packet).encode())
        self.assertFalse(
            any(
                item.get("animation", {}).get("id") == "failed"
                for item in await self.drain()
            )
        )
        self.assertIsNone(
            self.daemon.application.sessions.snapshot().sessions[0].user_sentiment
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
            "harness": "claude",
            "session": "frustrated",
            "event": "intent",
            "goal": "This is terrible. Stop making this worse.",
        }
        self.daemon.application.behavior.status = replace(
            require(self.daemon.application.behavior.status), menu=True
        )
        self.daemon.application.events.rng = random.Random(1)
        self.daemon.receive(json.dumps(packet).encode())
        self.daemon.application.behavior.status = replace(
            require(self.daemon.application.behavior.status), menu=False
        )
        self.daemon.application.behavior.tick(
            self.daemon.application.events.activity(), self.daemon.application.sessions
        )
        self.assertFalse(
            any(
                item.get("animation", {}).get("id") == "failed"
                for item in await self.drain()
            )
        )
        self.daemon.application.behavior.react("success")
        packet["session"] = "another"
        self.daemon.application.events.rng = random.Random(1)
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
        await asyncio.wait_for(self.entered.wait(), 1)
        memory = self.daemon.application.cognition.memory
        memory.record_speech("Remember the parser", "prompt")
        self.send({"kind": "clear_memory"})
        await self.packet(
            lambda packet: packet.get("text") == "" and packet.get("ttl") == 0
        )
        self.release.set()
        await self.wait_for(lambda: self.daemon.application.task is None)
        self.assertEqual(self.daemon.application.sessions.snapshot().sessions, ())
        self.assertEqual(memory.recent(), [])
        self.assertEqual(memory.speech(), [])
        self.assertEqual(memory.projects(["/repo/parser"]), [])
        self.assertIsNone(memory.context("cli:one"))
        self.assertIsNone(self.daemon.application.last_speech)
        self.assertFalse(any(packet.get("text") for packet in await self.drain()))
        self.slow = False
        self.event("intent", goal="Write documentation", repo="/repo/parser")
        await self.packet(lambda packet: packet.get("text") == self.speech)
        lane = self.daemon.application.sessions.lanes["cli:one"]
        self.assertEqual(lane.first_prompt, "Write documentation")
        self.assertNotIn(
            "Improve parser", json.dumps([asdict(record) for record in memory.recent()])
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
            lambda: bool(self.daemon.application.sessions.lanes["cli:one"].calls)
        )
        memory = self.daemon.application.memory
        sessions = self.daemon.application.sessions
        context = require(memory.context("cli:one"))
        events = memory.recent()
        associations = memory.repository_associations()
        repositories = self.repositories.signature()
        last_attempt = self.daemon.application.attention.last_attempt
        self.send({"kind": "clear_speech"})
        await self.packet(
            lambda packet: packet.get("text") == "" and packet.get("ttl") == 0
        )
        self.assertEqual(memory.speech(), [])
        self.assertIsNone(self.daemon.application.last_speech)
        self.assertIs(self.daemon.application.sessions, sessions)
        self.assertIn("checks", sessions.lanes["cli:one"].calls)
        self.assertEqual(memory.context("cli:one"), context)
        self.assertEqual(memory.recent(), events)
        self.assertEqual(memory.repository_associations(), associations)
        self.assertEqual(self.repositories.signature(), repositories)
        self.assertEqual(self.daemon.application.attention.last_attempt, last_attempt)
        await self.observe("explicit", "Look at the parser")
        payload = json.loads(self.requests[-1]["messages"][-1]["content"])
        self.assertNotIn("repetition", payload)
        self.assertIsNone(payload["focus_session"])
        self.assertEqual(payload["context"], "Look at the parser")

    async def test_clear_speech_discards_pending_remark_and_preserves_session_summary(
        self,
    ) -> None:
        self.slow = True
        self.session_context = "Keep the parser API stable."
        self.daemon.application.memory.record_speech("Earlier remark", "routine")
        self.event("intent", goal="Improve parser")
        await self.wait_for(
            lambda: len(self.requests) == 1 and len(self.summary_requests) == 1
        )
        self.send({"kind": "clear_speech"})
        await self.packet(
            lambda packet: packet.get("text") == "" and packet.get("ttl") == 0
        )
        self.release.set()
        await self.wait_for(lambda: self.daemon.application.task is None)
        self.assertEqual(self.daemon.application.memory.speech(), [])
        self.assertTrue(self.daemon.application.sessions.active)
        self.assertEqual(
            require(self.daemon.application.memory.context("cli:one")).session_context,
            self.session_context,
        )
        self.assertFalse(any(packet.get("text") for packet in await self.drain()))
        self.slow = False
        await self.observe("explicit", "Check again")
        packet = await self.packet(lambda packet: bool(packet.get("text")))
        self.assertEqual(packet["text"], self.speech)

    async def test_configured_speech_history_expires_in_model_requests(self) -> None:
        await self.daemon.close()
        self.config.speech_history_seconds = 10
        self.daemon = Daemon(self.config, self.client, rng=random.Random(7))
        await self.daemon.start()
        self.send_status()
        await self.packet(lambda packet: packet.get("state") == "idle")
        now = [1000]
        self.daemon.application.memory.clock = lambda: now[0]
        self.daemon.application.cognition.clock = lambda: now[0]
        self.daemon.application.memory.record_speech("Earlier remark", "routine")
        now[0] += 10
        await self.observe("explicit", "Say hello")
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

    async def wait_for_repository(self) -> RepositoryObservation:
        await self.wait_for(
            lambda: any(item.pr for item in self.repositories.snapshot()),
            timeout=3,
        )
        return next(item for item in self.repositories.snapshot() if item.current)

    async def connect_repository_work(self) -> None:
        self.repositories.github_interval = 0
        self.config.behavior.automatic_speech_cooldown_seconds = 90
        self.daemon.application.attention.last_attempt = self.daemon.clock()
        self.event(
            "intent", repo=str(self.repository_path), goal="Improve parser persistence"
        )
        await self.wait_for(
            lambda: bool(
                (
                    context := self.daemon.application.memory.work.session_context(
                        "cli:one"
                    )
                )
                is not None
                and any(
                    item.artifact.kind == "pull_request" for item in context.artifacts
                )
            ),
            timeout=3,
        )
        self.daemon.application.attention.pending = None
        self.config.behavior.automatic_speech_cooldown_seconds = 0

    async def test_exited_owner_leaves_the_reported_work_and_pr_context_available(
        self,
    ) -> None:
        from pet_daemon.processes import process_identity

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
                self.daemon.application.sessions.lanes["cli:one"].presence == "exited"
            ),
            timeout=3,
        )
        self.assertEqual(self.daemon.application.events.activity(), "idle")
        await self.observe("routine")
        payload = json.loads(self.requests[-1]["messages"][-1]["content"])
        self.assertNotIn("activity", payload)
        self.assertEqual(
            self.daemon.application.sessions.lanes["cli:one"].presence, "exited"
        )
        self.assertFalse(self.daemon.application.sessions.lanes["cli:one"].active)
        self.assertFalse(payload["focus_session"]["turn_active"])
        self.assertEqual(payload["focus_session"]["presence"], "exited")
        self.assertNotIn("connected_work", payload["focus_session"])
        self.assertEqual(payload["other_sessions"], [])
        work = payload["focus_work"]
        self.assertEqual(work["intent"], "Improve parser persistence")
        self.assertEqual(
            next(item for item in work["artifacts"] if item["kind"] == "pull_request")[
                "title"
            ],
            "Persist parser sessions",
        )

    async def test_pull_request_merge_discards_commentary_generated_for_pending_work(
        self,
    ) -> None:
        await self.repository_fixture()
        self.event("session_start", repo=str(self.repository_path))
        await self.wait_for_repository()
        await self.connect_repository_work()
        self.entered.clear()
        self.slow = True
        task = asyncio.create_task(self.observe("routine"))
        self.addCleanup(task.cancel)
        await asyncio.wait_for(self.entered.wait(), 1)
        self.repository_pr.update(state="MERGED", mergedAt="2026-10-17T00:00:00Z")
        self.repositories.github_interval = 0
        await self.wait_for(
            lambda: any(
                item.pr and item.pr.state == "MERGED"
                for item in self.repositories.snapshot()
            ),
            timeout=3,
        )
        self.release.set()
        await task
        self.assertTrue(
            any(
                item.pr and item.pr.state == "MERGED"
                for item in self.repositories.snapshot()
            )
        )
        self.assertEqual(self.daemon.application.memory.speech(), [])
        self.assertFalse(
            any(packet.get("text") == self.speech for packet in await self.drain())
        )

    async def finish_observation_after_other_session_changes(self) -> CommentaryPayload:
        self.config.behavior.automatic_speech_cooldown_seconds = 90
        self.daemon.application.attention.last_attempt = self.daemon.clock()
        self.event("intent", session="focus", goal="Review parser behavior")
        self.event("intent", session="other", goal="Review the renderer")
        await self.wait_for(lambda: len(self.daemon.application.sessions.lanes) == 2)
        self.slow = True
        task = asyncio.create_task(
            self.observe("routine", trigger=SessionTrigger("cli:focus"))
        )
        self.addCleanup(task.cancel)
        await asyncio.wait_for(self.entered.wait(), 1)
        payload = json.loads(self.requests[-1]["messages"][0]["content"])
        self.event("intent", session="other", goal="Document the renderer")
        await self.wait_for(
            lambda: (
                self.daemon.application.sessions.lanes["cli:other"].request_count == 2
            )
        )
        self.release.set()
        await task
        return payload

    async def test_omitted_session_changes_preserve_focused_commentary(self) -> None:
        self.config.prompt_target_bytes = 1
        payload = await self.finish_observation_after_other_session_changes()
        self.assertEqual(payload["other_sessions"], [])
        self.assertEqual(
            [row.text for row in self.daemon.application.memory.speech()],
            [self.speech],
        )
        self.assertEqual(
            (await self.packet(lambda packet: "text" in packet))["text"], self.speech
        )

    async def test_unrelated_work_stays_outside_focused_commentary(self) -> None:
        payload = await self.finish_observation_after_other_session_changes()
        self.assertEqual(payload["other_sessions"], [])
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
        original = await self.wait_for_repository()
        self.event("intent", session="focus", goal="Review parser behavior")
        await self.wait_for(
            lambda: "cli:focus" in self.daemon.application.sessions.lanes
        )
        self.slow = True
        task = asyncio.create_task(
            self.observe("routine", trigger=SessionTrigger("cli:focus"))
        )
        self.addCleanup(task.cancel)
        await asyncio.wait_for(self.entered.wait(), 1)
        payload = json.loads(self.requests[-1]["messages"][0]["content"])
        self.assertNotIn("Persist parser sessions", json.dumps(payload))
        self.repository_pr.update(state="MERGED", mergedAt="2026-10-17T00:00:00Z")
        self.repositories.github_interval = 0
        await self.wait_for(
            lambda: self.repositories.snapshot()[0].revision != original.revision,
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

    async def test_pending_repository_associations_survive_daemon_restart(self) -> None:
        await self.repository_fixture()
        self.event("session_start", repo=str(self.repository_path))
        original = await self.wait_for_repository()
        await self.connect_repository_work()
        await self.daemon.close()
        self.daemon = Daemon(self.config, self.client, rng=random.Random(7))
        self.repositories.run = self.repository_boundary
        restored = self.repositories.snapshot()[0]
        self.assertEqual(restored.subject, original.subject)
        self.assertEqual(restored.sessions, ("cli:one",))
        self.assertTrue(restored.pending)
        self.assertFalse(self.daemon.application.sessions.active)
        await self.daemon.start()
        self.send_status()
        await self.packet(lambda packet: packet.get("state") == "idle")
        await self.observe("routine")
        payload = json.loads(self.requests[-1]["messages"][-1]["content"])
        self.assertIsNone(payload["focus_session"])
        self.assertEqual(payload["focus_work"]["intent"], "Improve parser persistence")

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
                item.pr and item.pr.state == "MERGED"
                for item in self.repositories.snapshot()
            ),
            timeout=3,
        )
        self.assertEqual(self.daemon.application.memory.work.work_items(), ())
        self.daemon.application.attention.next_observation = self.daemon.clock()
        await self.wait_for(
            lambda: (
                self.daemon.application.attention.next_observation > self.daemon.clock()
            )
        )
        self.assertEqual(self.requests, [])

    async def test_due_check_ins_mark_only_the_selected_session(self) -> None:
        self.use_clock()
        self.config.idle_message_seconds = 5
        self.slow = True
        for session in ("one", "two"):
            self.event("tool_start", session=session, call="run", tool="pytest")
        await self.wait_for(lambda: len(self.daemon.application.sessions.lanes) == 2)
        self.now += 6
        await asyncio.wait_for(self.entered.wait(), 1)
        checked = [
            key
            for key, lane in self.daemon.application.sessions.lanes.items()
            if lane.last_check_in is not None
        ]
        self.assertEqual(len(checked), 1)
        unchecked = next(
            key for key in self.daemon.application.sessions.lanes if key not in checked
        )
        payload = json.loads(self.requests[-1]["messages"][-1]["content"])
        self.assertEqual(payload["focus_session"]["active_tools"][0]["tool"], "pytest")
        self.release.set()
        await self.wait_for(lambda: len(self.requests) == 2)
        payload = json.loads(self.requests[-1]["messages"][-1]["content"])
        self.assertIsNotNone(
            self.daemon.application.sessions.lanes[unchecked].last_check_in
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
        self.assertTrue(self.daemon.application.memory.work.work_items())
        self.http_status = 200
        self.daemon.application.attention.next_observation = self.daemon.clock()
        packet = await self.packet(lambda packet: packet.get("text") == self.speech)
        self.assertEqual(len(self.requests), 2)
        self.assertTrue(self.daemon.application.memory.work.work_items())
        accepted = require(
            self.daemon.application.memory.message_outcome(packet["delivery_id"])
        )
        self.assertEqual(accepted.outcome, "accepted")

    def use_clock(self) -> None:
        self.now = 1000.0

        def clock() -> float:
            return self.now

        self.daemon.clock = self.daemon.application.events.clock = (
            self.daemon.application.clock
        ) = clock
        self.daemon.application.attention.clock = clock
        self.daemon.application.processes.clock = clock
        self.daemon.application.sessions.clock = clock
        self.daemon.application.cognition.clock = clock
        self.daemon.application.delivery.clock = clock
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
        behavior.clock = lambda: self.now
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
            SITUATIONS["care_exercise"], self.requests[-1]["system"][-1]["text"]
        )
        self.send({"kind": "renderer", "event": "click"})
        await self.wait_for(lambda: len(self.requests) == 2)
        self.assertEqual(request_reason(self.requests[-1]), "click")
        self.assertGreaterEqual(behavior.next_care, self.now + 300)
        self.assertIsNone(behavior.care_remark)

    async def test_new_work_discards_pending_care_speech(self) -> None:
        self.slow = True
        task = asyncio.create_task(
            self.observe("care", trigger=CareTrigger("consternation"))
        )
        self.addCleanup(task.cancel)
        await asyncio.wait_for(self.entered.wait(), 1)
        self.event("intent", goal="Review the parser")
        await self.wait_for(lambda: self.daemon.application.sessions.active)
        self.release.set()
        await task
        self.assertFalse(
            any(packet.get("text") == self.speech for packet in await self.drain())
        )

    async def test_real_socket_tracks_parallel_calls_without_model_latency(
        self,
    ) -> None:
        self.slow = True
        self.send({"kind": "speak", "context": "Watch these builds"})
        await asyncio.wait_for(self.entered.wait(), 1)
        self.event("tool_start", call="build", tool="cargo build", manual=True)
        await self.packet(lambda packet: packet.get("state") == "working")
        self.event(
            "tool_start", session="two", harness="cursor", call="lint", tool="lint"
        )
        await self.wait_for(lambda: len(self.daemon.application.sessions.lanes) == 2)
        self.event("tool_done", call="build", manual=True)
        await self.wait_for(
            lambda: not self.daemon.application.sessions.lanes["cli:one"].calls
        )
        self.assertTrue(self.daemon.application.sessions.working)
        self.assertFalse(self.release.is_set())
        self.event("tool_done", session="two", harness="cursor", call="lint")
        await self.wait_for(
            lambda: not self.daemon.application.sessions.lanes["cursor:two"].calls
        )
        self.assertEqual(self.daemon.application.events.activity(), "working")
        self.event("turn_done", session="two", harness="cursor")
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
        self.daemon.application.behavior.clock = lambda: self.now
        self.slow = True
        self.event("intent", goal="Inspect the parser")
        await asyncio.wait_for(self.entered.wait(), 1)
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
        self.assertFalse(self.release.is_set())
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
        await asyncio.wait_for(self.entered.wait(), 1)
        self.event("tool_start", call="tests", tool="pytest")
        await self.wait_for(
            lambda: "tests" in self.daemon.application.sessions.lanes["cli:one"].calls
        )
        self.event("tool_error", call="tests")
        await self.wait_for(lambda: self.daemon.application.sessions.failed)
        self.event("tool_start", call="retry", tool="pytest")
        await self.wait_for(
            lambda: "retry" in self.daemon.application.sessions.lanes["cli:one"].calls
        )
        self.event("tool_done", call="retry", outcome="success")
        await self.packet(lambda packet: packet.get("animation", {}).get("id") == "hop")
        self.assertTrue(self.daemon.application.sessions.active)
        self.assertFalse(self.release.is_set())
        await self.drain()
        self.event("tool_start", call="cancelled", tool="pytest")
        await self.wait_for(
            lambda: (
                "cancelled" in self.daemon.application.sessions.lanes["cli:one"].calls
            )
        )
        self.event("tool_done", call="cancelled", outcome="cancelled")
        await self.wait_for(
            lambda: not self.daemon.application.sessions.lanes["cli:one"].calls
        )
        packets = await self.drain()
        self.assertFalse(
            any(packet.get("animation", {}).get("id") == "hop" for packet in packets)
        )

    async def test_prompt_triggers_commentary_without_waiting_for_routine_timer(
        self,
    ) -> None:
        self.event("intent", harness="codex", goal="Run local checks")
        speech = await self.packet(lambda packet: "text" in packet)
        self.assertEqual(speech["text"], self.speech)
        snapshot = json.loads(self.requests[0]["messages"][-1]["content"])
        self.assertEqual(request_reason(self.requests[-1]), "prompt")
        self.assertEqual(
            snapshot["focus_session"]["recent_requests"][-1], "Run local checks"
        )

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
        self.daemon.application.behavior.clock = lambda: self.now
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
        self.assertFalse(snapshot["sleeping"])

    async def test_prompt_wakes_a_nap_before_tool_work_starts(self) -> None:
        self.use_clock()
        self.config.behavior.nap_interval = (1, 1)
        self.daemon.application.behavior.clock = lambda: self.now
        self.daemon.application.behavior.inactive_since = self.now
        self.daemon.application.behavior.execution.interrupt()
        self.daemon.application.behavior.next_nap = self.now + 1
        self.now += 240
        await self.packet(lambda packet: packet.get("sleeping") is True)
        self.event("intent", harness="codex", goal="Run checks")
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
        await asyncio.wait_for(self.entered.wait(), 1)
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
        await asyncio.wait_for(self.entered.wait(), 1)
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
            lambda: self.daemon.application.sessions.lanes["cli:one"].waiting
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
                json.loads(body["messages"][-1]["content"])["focus_session"]["progress"]
                for body in self.requests
            ],
            ["Choose a strategy", "Choose a target"],
        )
        self.event("waiting", call="second", progress="Still choosing a target")
        await self.wait_for(
            lambda: (
                self.daemon.application.sessions.lanes["cli:one"].progress
                == "Still choosing a target"
            )
        )
        self.assertEqual(len(self.requests), 2)

    async def test_new_question_discards_the_pending_reminder_for_the_first(
        self,
    ) -> None:
        self.slow = True
        self.event("waiting", call="first", progress="Choose a strategy")
        await asyncio.wait_for(self.entered.wait(), 1)
        self.event("waiting", call="second", progress="Choose a target")
        await self.wait_for(
            lambda: (
                self.daemon.application.sessions.lanes["cli:one"].waiting_call
                == "second"
            )
        )
        self.release.set()
        await self.packet(lambda packet: "text" in packet)
        await self.wait_for(lambda: self.daemon.application.task is None)
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(len(self.daemon.application.cognition.memory.speech()), 1)
        self.assertEqual(
            json.loads(self.requests[-1]["messages"][-1]["content"])["focus_session"][
                "progress"
            ],
            "Choose a target",
        )

    async def test_waiting_reminder_is_discarded_when_background_call_finishes(
        self,
    ) -> None:
        self.slow = True
        self.event("tool_start", call="background", tool="Read")
        self.event("waiting", call="question", progress="Choose a strategy")
        await asyncio.wait_for(self.entered.wait(), 1)
        self.event("tool_done", call="background")
        await self.wait_for(
            lambda: not self.daemon.application.sessions.lanes["cli:one"].calls
        )
        self.assertFalse(self.daemon.application.sessions.lanes["cli:one"].waiting)
        self.release.set()
        await self.wait_for(lambda: self.daemon.application.task is None)
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(self.daemon.application.cognition.memory.speech(), [])
        self.assertFalse(any("text" in packet for packet in await self.drain()))

    async def test_native_permission_wait_without_call_id_ends_when_the_tool_completes(
        self,
    ) -> None:
        self.slow = True
        root = Path(__file__).resolve().parents[1]
        hook = runpy.run_path(str(root / "scripts" / "pet-hook"))
        base = {
            "session_id": "native",
            "tool_name": "Bash",
            "tool_input": {"command": "pytest"},
        }
        for name, fields in (
            ("PreToolUse", {"tool_use_id": "tests"}),
            ("PermissionRequest", {}),
        ):
            packet = hook["update"]("claude", name, {**base, **fields})
            self.daemon.receive(hook["CLIENT"]["encode"](packet))
        self.assertNotIn("call", packet)
        await asyncio.wait_for(self.entered.wait(), 1)
        self.assertTrue(self.daemon.application.sessions.lanes["claude:native"].waiting)
        packet = hook["update"](
            "claude",
            "PostToolUse",
            {**base, "tool_use_id": "tests", "tool_response": {"exit_code": 0}},
        )
        self.daemon.receive(hook["CLIENT"]["encode"](packet))
        snapshot = self.daemon.application.sessions.snapshot().sessions[0]
        self.assertFalse(snapshot.waiting_for_input)
        self.assertIsNone(snapshot.waiting_seconds)
        self.assertEqual(snapshot.active_tools, ())
        self.release.set()
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
        await asyncio.wait_for(self.entered.wait(), 1)
        self.send({"kind": "renderer", "event": "click"})
        await self.wait_for(lambda: len(self.requests) == 2)
        self.release.set()
        speech = await self.packet(lambda packet: "text" in packet)
        self.assertEqual(speech["text"], self.speech)
        self.assertEqual(
            [request_reason(body) for body in self.requests], ["prompt", "click"]
        )
        self.assertNotIn(
            "Superseded acknowledgment.",
            json.dumps(
                [
                    asdict(record)
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
                    asdict(record)
                    for record in self.daemon.application.cognition.memory.recent()
                ]
            ),
        )
        self.assertEqual(self.requests, [])

    async def test_visible_speech_persists_after_renderer_delivery(self) -> None:
        await self.observe("explicit", "Inspect the build")
        await self.packet(lambda packet: "text" in packet)
        self.assertIn(
            self.speech,
            json.dumps(
                [
                    asdict(record)
                    for record in self.daemon.application.cognition.memory.speech()
                ]
            ),
        )

    async def test_zero_duration_speech_is_absent_from_memory(self) -> None:
        await self.observe("explicit", "Inspect the build", ttl=0)
        await self.packet(lambda packet: "text" in packet)
        self.assertEqual(self.daemon.application.cognition.memory.speech(), [])

    async def test_computer_sleep_cancels_commentary_and_click_recovers_after_wake(
        self,
    ) -> None:
        self.slow = True
        self.send({"kind": "speak", "context": "Inspect the build"})
        await asyncio.wait_for(self.entered.wait(), 1)
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

    async def test_computer_sleep_cancels_a_repeated_speech_retry_without_delivery(
        self,
    ) -> None:
        history = "The parser still needs its edge cases checked."
        self.speech = history
        self.daemon.application.memory.record_speech(history, "routine")
        self.retry_slow = True
        self.event("intent", goal="Review parser edge cases")
        await asyncio.wait_for(self.retry_entered.wait(), 1)
        self.send({"kind": "renderer", "event": "system_sleep"})
        await self.wait_for(
            lambda: (
                self.daemon.application.suspended
                and self.daemon.application.task is None
            )
        )
        self.release.set()
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(
            [row.text for row in self.daemon.application.memory.speech()], [history]
        )
        self.assertFalse(any(packet.get("text") for packet in await self.drain()))

    async def test_repeated_speech_retry_shares_the_observation_timeout(self) -> None:
        history = "The parser still needs its edge cases checked."
        self.speech = history
        self.daemon.application.memory.record_speech(history, "routine")
        self.config.request_timeout = 0.15
        self.retry_delay = 1
        with self.assertLogs("pet-daemon", level="WARNING"):
            self.event("intent", goal="Review parser edge cases")
            await asyncio.wait_for(self.retry_entered.wait(), 1)
            await self.wait_for(lambda: self.daemon.application.task is None)
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(
            [row.text for row in self.daemon.application.memory.speech()], [history]
        )
        self.assertFalse(any(packet.get("text") for packet in await self.drain()))

    async def test_summary_updates_on_each_prompt_and_skips_completion_by_default(
        self,
    ) -> None:
        self.session_context = "Keep the parser API compatible."
        self.event("intent", goal="Refactor the parser")
        await self.packet(lambda packet: packet.get("text") == self.speech)
        await self.wait_for(lambda: self.daemon.application.task is None)
        self.event("intent", goal="Investigate the parser timeout")
        await self.packet(lambda packet: packet.get("text") == self.speech)
        await self.wait_for(lambda: self.daemon.application.task is None)
        self.assertEqual(len(self.summary_requests), 2)
        payload = json.loads(self.summary_requests[-1]["messages"][0]["content"])
        self.assertEqual(
            payload["recent_requests"][-1], "Investigate the parser timeout"
        )
        self.assertEqual(payload["historical_session_context"], self.session_context)
        await self.observe("completion")
        self.assertEqual(len(self.summary_requests), 2)
        self.assertEqual(len(self.requests), 3)

    async def test_new_summaries_replace_prior_context_without_a_character_limit(
        self,
    ) -> None:
        self.session_context = "Keep the parser API compatible."
        self.event("intent", goal="Refactor the parser")
        await self.packet(lambda packet: packet.get("text") == self.speech)
        await self.wait_for(lambda: self.daemon.application.task is None)
        lane = self.daemon.application.sessions.lanes["cli:one"]
        self.session_context = "The parser " + "compatibility " * 40 + "remains intact."
        self.event("intent", goal="Review parser compatibility")
        await self.packet(lambda packet: packet.get("text") == self.speech)
        await self.wait_for(lambda: self.daemon.application.task is None)
        self.assertEqual(lane.session_context, self.session_context)
        self.assertEqual(
            require(self.daemon.application.memory.context("cli:one")).session_context,
            self.session_context,
        )
        self.session_context += " Existing callers retain their signatures."
        self.event("intent", goal="Confirm parser compatibility")
        await self.packet(lambda packet: packet.get("text") == self.speech)
        await self.wait_for(lambda: self.daemon.application.task is None)
        self.assertEqual(lane.session_context, self.session_context)
        self.assertEqual(
            require(self.daemon.application.memory.context("cli:one")).session_context,
            self.session_context,
        )
        self.assertEqual(len(self.requests), 3)

    async def test_first_request_summary_overrides_triggers_and_survives_a_failed_attempt_and_restart(
        self,
    ) -> None:
        self.config.summarize_first_request_only = True
        self.config.summarize_on = []
        self.summary_status = 500
        initial = (
            "Refactor the parser while keeping the public interface unchanged. " * 8
        )
        self.event("intent", goal=initial)
        await self.packet(lambda packet: packet.get("text") == self.speech)
        await self.wait_for(lambda: self.daemon.application.task is None)
        self.assertEqual(len(self.summary_requests), 1)
        payload = json.loads(self.summary_requests[0]["messages"][0]["content"])
        self.assertEqual(payload["initial_intent"], initial)
        self.assertEqual(set(payload), {"initial_intent"})
        self.assertNotIn("progress", payload)
        self.assertNotIn("historical_session_context", payload)
        saved = require(self.daemon.application.cognition.memory.context("cli:one"))
        self.assertTrue(saved.first_summary_attempted)
        self.assertEqual(saved.first_prompt, initial)
        await self.observe("prompt")
        self.assertEqual(len(self.summary_requests), 1)
        await self.daemon.close()
        self.daemon = Daemon(self.config, self.client, rng=random.Random(7))
        await self.daemon.start()
        self.send_status()
        await self.packet(lambda packet: packet.get("state") == "idle")
        self.config.summarize_on = ["prompt", "completion", "explicit"]
        self.summary_status = 200
        self.event("intent", goal="Debug the timeout")
        await self.packet(lambda packet: packet.get("text") == self.speech)
        await self.wait_for(lambda: self.daemon.application.task is None)
        lane = self.daemon.application.sessions.lanes["cli:one"]
        self.assertEqual(lane.first_prompt, initial)
        self.assertEqual(lane.request_count, 2)
        for reason in ("completion", "explicit"):
            await self.observe(reason, "Parser work")
        self.assertEqual(len(self.summary_requests), 1)
        await self.drain()
        self.event("intent", session="another", goal="Review hooks")
        await self.packet(lambda packet: packet.get("text") == self.speech)
        await self.wait_for(lambda: self.daemon.application.task is None)
        self.assertEqual(len(self.summary_requests), 2)
        payload = json.loads(self.summary_requests[-1]["messages"][0]["content"])
        self.assertEqual(payload["initial_intent"], "Review hooks")

    async def test_silent_summary_informs_later_session_commentary(self) -> None:
        self.use_clock()
        self.speech = None
        self.session_context = "Parser refactor awaits validation."
        self.event("intent", goal="Refactor the parser", turn="first")
        await self.wait_for(
            lambda: (
                self.daemon.application.cognition.memory.context("cli:one") is not None
                and require(
                    self.daemon.application.cognition.memory.context("cli:one")
                ).session_context
                == self.session_context
            )
        )
        await self.wait_for(lambda: self.daemon.application.task is None)
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(self.daemon.application.cognition.memory.speech(), [])
        self.now += 20
        self.daemon.application.attention.next_observation = self.now
        await self.wait_for(
            lambda: self.daemon.application.attention.next_observation > self.now
        )
        self.assertEqual(len(self.requests), 1)
        self.session_context = None
        self.speech = "The parser still needs validation."
        await self.observe("explicit", "Check the parser")
        await self.packet(lambda packet: "text" in packet)
        session = json.loads(self.requests[-1]["messages"][0]["content"])[
            "focus_session"
        ]
        self.assertIsNone(session)
        self.assertEqual(
            require(self.daemon.application.memory.context("cli:one")).session_context,
            "Parser refactor awaits validation.",
        )
        self.speech = "The parser context guides compatibility checks."
        await self.observe("routine")
        await self.packet(lambda packet: "text" in packet)
        session = json.loads(self.requests[-1]["messages"][0]["content"])[
            "focus_session"
        ]
        self.assertEqual(
            session["session_context"], "Parser refactor awaits validation."
        )
        self.assertEqual(len(self.requests), 3)

    async def test_first_request_summary_restores_bookkeeping_from_a_session_alias(
        self,
    ) -> None:
        self.config.summarize_first_request_only = True
        self.daemon.application.cognition.memory.save_context(
            "cli:old-id", "Review hooks", "Hook review", ["Review hooks"], 1, True
        )
        self.event(
            "intent", session="new-id", aliases=["old-id"], goal="Check hook edge cases"
        )
        await self.packet(lambda packet: packet.get("text") == self.speech)
        await self.wait_for(lambda: self.daemon.application.task is None)
        self.assertEqual(self.summary_requests, [])
        lane = self.daemon.application.sessions.lanes["cli:new-id"]
        self.assertEqual(lane.first_prompt, "Review hooks")
        self.assertEqual(lane.request_count, 2)
        self.assertTrue(lane.first_summary_attempted)

    async def test_first_request_summary_uses_the_initial_prompt_when_followups_are_queued_together(
        self,
    ) -> None:
        self.config.summarize_first_request_only = True
        for goal in ("Refactor the parser", "Keep its API unchanged"):
            self.daemon.receive(
                json.dumps(
                    {
                        "kind": "event",
                        "event": "intent",
                        "harness": "cli",
                        "session": "one",
                        "goal": goal,
                    }
                ).encode()
            )
        await self.packet(lambda packet: packet.get("text") == self.speech)
        await self.wait_for(lambda: self.daemon.application.task is None)
        self.assertEqual(len(self.summary_requests), 1)
        payload = json.loads(self.summary_requests[0]["messages"][0]["content"])
        self.assertEqual(payload["initial_intent"], "Refactor the parser")
        self.assertEqual(set(payload), {"initial_intent"})
        self.assertNotIn("Keep its API unchanged", json.dumps(payload))

    async def test_subagent_inherited_context_does_not_consume_its_first_request_summary(
        self,
    ) -> None:
        self.config.summarize_first_request_only = True
        self.event("intent", goal="Refactor the parser")
        await self.packet(lambda packet: packet.get("text") == self.speech)
        await self.wait_for(lambda: self.daemon.application.task is None)
        self.event("subagent_start", session="child", parent="one")
        await self.packet(lambda packet: packet.get("text") == self.speech)
        await self.wait_for(lambda: self.daemon.application.task is None)
        self.assertEqual(len(self.summary_requests), 1)
        self.event("intent", session="child", goal="Inspect parser edge cases")
        await self.packet(lambda packet: packet.get("text") == self.speech)
        await self.wait_for(lambda: self.daemon.application.task is None)
        self.assertEqual(len(self.summary_requests), 2)
        payload = json.loads(self.summary_requests[-1]["messages"][0]["content"])
        self.assertEqual(payload["initial_intent"], "Inspect parser edge cases")
        self.assertNotIn("Refactor the parser", json.dumps(payload))

    async def test_interleaved_work_keeps_separate_summaries_and_commentary(
        self,
    ) -> None:
        self.session_context = "Keep the parser API stable."
        self.event(
            "intent",
            session="parser",
            goal="Refactor the parser without changing its API",
        )
        await self.packet(lambda packet: "text" in packet)
        await self.wait_for(lambda: self.daemon.application.task is None)
        self.session_context = "Remove redundant Python hook handling."
        self.event(
            "intent", session="hooks", goal="Review Python hooks for redundant handling"
        )
        await self.packet(lambda packet: "text" in packet)
        await self.wait_for(lambda: self.daemon.application.task is None)
        commentary = json.loads(self.requests[-1]["messages"][0]["content"])
        self.assertEqual(
            commentary["focus_session"]["recent_requests"],
            ["Review Python hooks for redundant handling"],
        )
        self.assertEqual(commentary["other_sessions"], [])
        self.assertNotIn("Refactor the parser", json.dumps(commentary))
        for request, excluded in zip(self.summary_requests, ("Python hooks", "parser")):
            self.assertNotIn(excluded, request["messages"][0]["content"])
        memory = self.daemon.application.cognition.memory
        self.assertEqual(
            require(memory.context("cli:parser")).session_context,
            "Keep the parser API stable.",
        )
        self.assertEqual(
            require(memory.context("cli:hooks")).session_context,
            "Remove redundant Python hook handling.",
        )
        self.event("turn_done", session="hooks", progress="Hook cleanup passed checks")
        await self.packet(lambda packet: "text" in packet, timeout=2)
        await self.wait_for(lambda: self.daemon.application.task is None)
        commentary = json.loads(self.requests[-1]["messages"][0]["content"])
        self.assertEqual(
            commentary["focus_session"]["recent_requests"],
            ["Review Python hooks for redundant handling"],
        )
        self.assertFalse(commentary["convergence"])
        self.assertEqual(commentary["other_sessions"], [])
        self.assertTrue(self.daemon.application.sessions.lanes["cli:parser"].active)
        summary_count = len(self.summary_requests)
        await self.observe("routine")
        self.assertEqual(len(self.summary_requests), summary_count)

    async def test_summary_failure_preserves_commentary_and_saved_context(self) -> None:
        self.session_context = "Keep the parser API stable."
        self.event("intent", goal="Refactor the parser")
        await self.packet(lambda packet: "text" in packet)
        await self.wait_for(lambda: self.daemon.application.task is None)
        self.summary_status = 500
        self.event("intent", goal="Validate parser behavior")
        await self.packet(lambda packet: packet.get("text") == self.speech)
        await self.wait_for(lambda: self.daemon.application.task is None)
        self.assertEqual(
            require(
                self.daemon.application.cognition.memory.context("cli:one")
            ).session_context,
            "Keep the parser API stable.",
        )

    async def test_followup_with_the_same_native_turn_reaches_the_next_prompt(
        self,
    ) -> None:
        self.event(
            "intent", harness="codex", goal="Compress the workflow", turn="native-turn"
        )
        await self.packet(lambda packet: "text" in packet)
        await self.wait_for(lambda: self.daemon.application.task is None)
        self.event(
            "intent",
            harness="codex",
            goal="Explain the counter limits",
            turn="native-turn",
        )
        await self.packet(lambda packet: "text" in packet)
        self.assertEqual(len(self.requests), 2)
        payload = json.loads(self.requests[-1]["messages"][0]["content"])
        self.assertEqual(request_reason(self.requests[-1]), "prompt")
        self.assertNotIn("first_prompt", payload["focus_session"])
        self.assertEqual(
            payload["focus_session"]["recent_requests"], ["Explain the counter limits"]
        )
        self.assertEqual(
            payload["focus_session"]["recent_requests"][-1],
            "Explain the counter limits",
        )

    async def test_followup_during_observation_refuses_stale_summary(self) -> None:
        self.daemon.application.cognition.memory.save_context(
            "cli:one",
            "Refactor the parser",
            "Keep its API stable",
            ["Refactor the parser"],
        )
        self.slow = True
        self.session_context = "Stale parser summary."
        self.event("intent", goal="Refactor the parser", turn="first")
        await asyncio.wait_for(self.entered.wait(), 1)
        self.event(
            "intent", goal="Keep the parser API unchanged", turn="first", manual=True
        )
        await self.wait_for(
            lambda: (
                "Keep the parser API unchanged"
                in self.daemon.application.sessions.lanes["cli:one"].recent_requests
            )
        )
        self.session_context = None
        self.release.set()
        await self.wait_for(
            lambda: self.daemon.application.task is None and len(self.requests) >= 2
        )
        lane = self.daemon.application.sessions.lanes["cli:one"]
        self.assertNotEqual(lane.session_context, "Stale parser summary.")
        self.assertEqual(lane.first_prompt, "Refactor the parser")
        self.assertIn("Keep the parser API unchanged", lane.recent_requests)
        self.assertEqual(len(self.daemon.application.cognition.memory.speech()), 1)

    async def test_explicit_speech_cannot_overwrite_context_after_a_followup(
        self,
    ) -> None:
        self.event("intent", goal="Refactor the parser", turn="first")
        await self.packet(lambda packet: "text" in packet)
        await self.wait_for(lambda: self.daemon.application.task is None)
        self.entered.clear()
        self.slow = True
        self.session_context = "Stale explicit summary."
        self.send({"kind": "speak", "context": "Comment on the parser"})
        await asyncio.wait_for(self.entered.wait(), 1)
        self.event("intent", goal="Keep the parser API unchanged", turn="first")
        await self.wait_for(
            lambda: (
                "Keep the parser API unchanged"
                in self.daemon.application.sessions.lanes["cli:one"].recent_requests
            )
        )
        self.session_context = None
        self.release.set()
        await self.wait_for(
            lambda: self.daemon.application.task is None and len(self.requests) >= 3
        )
        self.assertNotEqual(
            self.daemon.application.sessions.lanes["cli:one"].session_context,
            "Stale explicit summary.",
        )
        saved = require(self.daemon.application.cognition.memory.context("cli:one"))
        self.assertNotEqual(saved.session_context, "Stale explicit summary.")

    async def test_explicit_speech_keeps_session_summary_separate_from_progress(
        self,
    ) -> None:
        self.session_context = "Keep the parser API stable."
        self.event("intent", goal="Refactor the parser", turn="first")
        await self.packet(lambda packet: packet.get("text") == self.speech)
        await self.wait_for(lambda: self.daemon.application.task is None)
        generation = self.daemon.application.sessions.lanes["cli:one"].generation
        self.config.summarize_on = ["explicit"]
        self.session_context = "Parser validation remains pending."
        self.speech = "The parser has another edge case."
        self.slow = True
        self.send({"kind": "speak", "context": "Comment on the parser"})
        await self.wait_for(lambda: len(self.requests) == 2)
        self.assertEqual(len(self.summary_requests), 1)
        self.event("progress", progress="Parser validation passed", turn="first")
        await self.wait_for(
            lambda: (
                self.daemon.application.sessions.lanes["cli:one"].progress
                == "Parser validation passed"
            )
        )
        self.assertEqual(
            self.daemon.application.sessions.lanes["cli:one"].generation, generation
        )
        self.release.set()
        await self.packet(lambda packet: packet.get("text") == self.speech)
        await self.wait_for(lambda: self.daemon.application.task is None)
        self.assertEqual(
            self.daemon.application.sessions.lanes["cli:one"].session_context,
            "Keep the parser API stable.",
        )
        self.assertEqual(
            require(self.daemon.application.memory.context("cli:one")).session_context,
            "Keep the parser API stable.",
        )

    async def test_restart_restores_context_when_hooks_arrive_without_restoring_live_calls(
        self,
    ) -> None:
        self.session_context = "Parser checks are pending."
        self.event("intent", goal="Refactor the parser", turn="first")
        await self.packet(lambda packet: "text" in packet)
        self.event("tool_start", call="checks", tool="pytest")
        await self.wait_for(lambda: self.daemon.application.sessions.working)
        await self.daemon.close()
        self.session_context = None
        self.daemon = Daemon(self.config, self.client, rng=random.Random(7))
        self.assertFalse(self.daemon.application.sessions.active)
        self.assertFalse(self.daemon.application.sessions.working)
        await self.daemon.start()
        self.send_status()
        await self.packet(lambda packet: packet.get("state") == "idle")
        self.event("intent", goal="Continue parser validation", turn="second")
        await self.packet(lambda packet: "text" in packet)
        lane = self.daemon.application.sessions.lanes["cli:one"]
        self.assertEqual(lane.first_prompt, "Refactor the parser")
        self.assertEqual(lane.session_context, "Parser checks are pending.")
        self.assertIn("Continue parser validation", lane.recent_requests)
        self.assertFalse(lane.calls)

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
                self.daemon.application.sessions.lanes["cli:one"].progress
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
                    "cli:one" in self.daemon.application.sessions.lanes
                    and f"check-{index}"
                    in self.daemon.application.sessions.lanes["cli:one"].calls
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
                    self.daemon.application.sessions.lanes["cli:one"].failure
                    is not None
                    and require(
                        self.daemon.application.sessions.lanes["cli:one"].failure
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
            require(self.daemon.application.sessions.lanes["cli:one"].failure).count, 5
        )

    async def test_recovered_session_skips_queued_failure_when_another_session_is_still_failing(
        self,
    ) -> None:
        self.use_clock()
        for session in ("one", "two"):
            self.event("tool_start", session=session, call="checks", tool="pytest")
            await self.wait_for(
                lambda session=session: (
                    f"cli:{session}" in self.daemon.application.sessions.lanes
                    and "checks"
                    in self.daemon.application.sessions.lanes[f"cli:{session}"].calls
                )
            )
            self.event("tool_error", session=session, call="checks")
            for index in range(2):
                call = f"failed-{index}"
                self.event("tool_start", session=session, call=call, tool="pytest")
                await self.wait_for(
                    lambda call=call, session=session: (
                        call
                        in self.daemon.application.sessions.lanes[
                            f"cli:{session}"
                        ].calls
                    )
                )
                self.event("tool_error", session=session, call=call)
                await self.wait_for(
                    lambda index=index, session=session: (
                        require(
                            self.daemon.application.sessions.lanes[
                                f"cli:{session}"
                            ].failure
                        ).count
                        == index + 2
                    )
                )
            await self.wait_for(
                lambda session=session: bool(
                    self.daemon.application.attention.pending
                    and isinstance(
                        self.daemon.application.attention.pending[2], EpisodeTrigger
                    )
                    and self.daemon.application.attention.pending[2].episode.session
                    == f"cli:{session}"
                )
            )
        self.event("tool_start", session="two", call="retry", tool="pytest")
        await self.wait_for(
            lambda: "retry" in self.daemon.application.sessions.lanes["cli:two"].calls
        )
        self.event("tool_done", session="two", call="retry")
        await self.wait_for(
            lambda: (
                require(
                    self.daemon.application.sessions.lanes["cli:two"].failure
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
            harness="claude",
            parent="main",
            parent_call="delegate",
            goal="Inspect parser",
            repo="/repo/parser",
        )
        await self.packet(lambda packet: "text" in packet)
        await self.wait_for(lambda: self.daemon.application.task is None)
        self.event(
            "tool_start",
            session="main",
            harness="claude",
            parent_call="delegate",
            call="check",
            tool="pytest",
        )
        await self.wait_for(
            lambda: (
                "check" in self.daemon.application.sessions.lanes["claude:child"].calls
            )
        )
        self.event(
            "tool_error",
            session="main",
            harness="claude",
            parent_call="delegate",
            call="check",
            detail="Parser assertion failed",
            error_signature="parser:error",
        )
        await self.wait_for(
            lambda: (
                self.daemon.application.sessions.lanes["claude:child"].failure
                is not None
            )
        )
        self.event("turn_error", session="child", harness="claude")
        await self.wait_for(
            lambda: self.daemon.application.sessions.lanes["claude:child"].finished
        )
        self.now += 2
        await self.packet(lambda packet: "text" in packet)
        self.assertEqual(request_reason(self.requests[-1]), "failure")
        failures = [
            row
            for row in self.daemon.application.cognition.memory.recent(limit=10)
            if row.event_type == "turn_error"
        ]
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0].session, "claude:child")
        self.assertEqual(failures[0].repo, "/repo/parser")

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
        await self.observe("failure")
        await self.packet(lambda packet: "text" in packet)
        body = self.requests[-1]
        self.assertNotIn("SECRET", json.dumps(body))
        self.assertNotIn(
            "SECRET", json.dumps(asdict(self.daemon.application.sessions.snapshot()))
        )
        self.assertNotIn(
            "SECRET",
            json.dumps(
                [
                    asdict(record)
                    for record in self.daemon.application.cognition.memory.recent()
                ]
            ),
        )
        failure = json.loads(body["messages"][0]["content"])["focus_session"]["failure"]
        self.assertEqual(failure["activity"], "testing")
        self.assertEqual(failure["count"], 1)

    async def test_failed_renderer_delivery_is_absent_from_memory(self) -> None:
        Path(self.config.overlay_socket).unlink()
        await self.daemon.application.observe("explicit", "Inspect the build")
        self.assertEqual(self.daemon.application.cognition.memory.speech(), [])
        self.assertIsNone(self.daemon.application.last_speech)

    async def test_completion_reacts_to_one_session_while_other_tools_keep_working(
        self,
    ) -> None:
        self.event("tool_start", harness="codex", call="checks", tool="pytest")
        self.event(
            "tool_start",
            session="two",
            harness="cursor",
            call="build",
            tool="cargo build",
        )
        await self.wait_for(lambda: len(self.daemon.application.sessions.lanes) == 2)
        self.event("turn_done", harness="codex", progress="Checks passed")
        speech = await self.packet(lambda packet: "text" in packet, timeout=2)
        snapshot = json.loads(self.requests[0]["messages"][-1]["content"])
        self.assertEqual(request_reason(self.requests[-1]), "completion")
        self.assertEqual(snapshot["focus_session"]["outcome"], "success")
        self.assertEqual(snapshot["focus_session"]["progress"], "Checks passed")
        self.assertNotIn("activity", snapshot)
        self.assertEqual(speech["text"], self.speech)
        self.assertEqual(self.daemon.application.events.activity(), "working")

    async def test_superseded_tool_activity_discards_pending_commentary(self) -> None:
        self.slow = True
        self.event("tool_start", harness="codex", call="checks", tool="pytest")
        await self.packet(lambda packet: packet.get("state") == "working")
        task = asyncio.create_task(self.observe("routine"))
        await asyncio.wait_for(self.entered.wait(), 1)
        self.event("tool_done", harness="codex", call="checks", progress="Tests passed")
        self.event("tool_start", harness="codex", call="lint", tool="lint")
        await self.wait_for(
            lambda: "lint" in self.daemon.application.sessions.lanes["codex:one"].calls
        )
        self.release.set()
        await task
        self.assertEqual(self.daemon.application.memory.speech(), [])
        self.assertFalse(any(packet.get("text") for packet in await self.drain()))

    async def test_check_in_uses_each_tasks_quiet_time_and_repeats_after_the_interval(
        self,
    ) -> None:
        self.use_clock()
        self.event("tool_start", harness="codex", call="checks", tool="pytest")
        self.event(
            "tool_start", session="two", harness="cursor", call="build", tool="build"
        )
        await self.wait_for(lambda: len(self.daemon.application.sessions.lanes) == 2)
        self.now += 299
        self.event(
            "progress", session="two", harness="cursor", progress="Build progressing"
        )
        await self.wait_for(
            lambda: (
                self.daemon.application.sessions.lanes["cursor:two"].progress
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
        self.assertEqual(snapshot["focus_session"]["active_tools"][0]["tool"], "pytest")
        self.assertIsNotNone(
            self.daemon.application.sessions.lanes["codex:one"].last_check_in
        )
        self.assertEqual(snapshot["other_sessions"], [])
        self.assertTrue(self.daemon.application.sessions.lanes["cursor:two"].active)
        self.assertEqual(snapshot["focus_session"]["quiet_for"], "5 minutes")
        self.now += 299
        self.event(
            "progress", session="two", harness="cursor", progress="Build continuing"
        )
        await self.wait_for(lambda: self.daemon.application.next_prune > self.now)
        self.assertEqual(len(self.requests), 1)
        self.now += 1
        await self.packet(lambda packet: "text" in packet)
        snapshot = json.loads(self.requests[1]["messages"][-1]["content"])
        self.assertEqual(snapshot["focus_session"]["active_tools"][0]["tool"], "pytest")

    async def test_failed_check_in_waits_the_interval_before_retrying(self) -> None:
        self.use_clock()
        self.http_status = 500
        self.event("tool_start", harness="codex", call="checks", tool="pytest")
        await self.wait_for(lambda: self.daemon.application.sessions.working)
        with self.assertLogs("pet-daemon", level="WARNING"):
            self.now += 300
            await asyncio.wait_for(self.entered.wait(), 1)
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
        self.slow = True
        self.event(
            "waiting",
            harness="claude",
            call="question",
            tool="AskUserQuestion",
            progress="Choose a strategy",
        )
        await asyncio.wait_for(self.entered.wait(), 1)
        snapshot = json.loads(self.requests[0]["messages"][-1]["content"])
        self.assertEqual(request_reason(self.requests[0]), "check_in")
        self.assertTrue(snapshot["focus_session"]["waiting_for_input"])
        self.event("progress", harness="claude", progress="Investigating the strategy")
        await self.wait_for(
            lambda: not self.daemon.application.sessions.lanes["claude:one"].waiting
        )
        self.release.set()
        await self.wait_for(lambda: self.daemon.application.task is None)
        self.assertEqual(self.daemon.application.cognition.memory.speech(), [])
        self.assertFalse(any("text" in packet for packet in await self.drain()))

    async def test_waiting_check_in_keeps_speech_while_another_task_progresses(
        self,
    ) -> None:
        self.use_clock()
        self.slow = True
        self.event(
            "waiting",
            harness="claude",
            call="question",
            tool="AskUserQuestion",
            progress="Choose a strategy",
        )
        self.event(
            "tool_start", session="two", harness="cursor", call="build", tool="build"
        )
        await self.wait_for(lambda: len(self.daemon.application.sessions.lanes) == 2)
        self.now += 299
        self.event(
            "progress", session="two", harness="cursor", progress="Building modules"
        )
        await self.wait_for(
            lambda: (
                self.daemon.application.sessions.lanes["cursor:two"].progress
                == "Building modules"
            )
        )
        self.now += 1
        await asyncio.wait_for(self.entered.wait(), 1)
        snapshot = json.loads(self.requests[0]["messages"][-1]["content"])
        self.assertEqual(request_reason(self.requests[-1]), "check_in")
        self.assertTrue(snapshot["focus_session"]["waiting_for_input"])
        self.assertNotIn("activity", snapshot)
        self.assertTrue(snapshot["focus_session"]["waiting_for_input"])
        self.assertTrue(snapshot["focus_session"]["waiting_for_input"])
        self.assertEqual(snapshot["focus_session"]["waiting_for"], "5 minutes")
        self.assertEqual(self.daemon.application.events.activity(), "working")
        self.now += 1
        self.event(
            "progress", session="two", harness="cursor", progress="Linking build"
        )
        await self.wait_for(
            lambda: (
                self.daemon.application.sessions.lanes["cursor:two"].progress
                == "Linking build"
            )
        )
        self.release.set()
        speech = await self.packet(lambda packet: "text" in packet)
        self.assertEqual(speech["text"], self.speech)
        self.assertEqual(self.daemon.application.events.activity(), "working")

    async def test_explicit_waiting_after_progress_replaces_the_pending_reminder(
        self,
    ) -> None:
        self.use_clock()
        self.slow = True
        self.event(
            "waiting",
            harness="claude",
            call="question",
            tool="AskUserQuestion",
            progress="Choose a strategy",
        )
        await asyncio.wait_for(self.entered.wait(), 1)
        self.now += 1
        self.event("progress", harness="claude", progress="Investigating the strategy")
        await self.wait_for(
            lambda: not self.daemon.application.sessions.lanes["claude:one"].waiting
        )
        self.now += 1
        self.event(
            "waiting",
            harness="claude",
            call="question",
            tool="AskUserQuestion",
            progress="Choose a target",
        )
        await self.wait_for(
            lambda: self.daemon.application.sessions.lanes["claude:one"].waiting
        )
        self.release.set()
        speech = await self.packet(lambda packet: "text" in packet)
        self.assertEqual(speech["text"], self.speech)
        await self.wait_for(lambda: self.daemon.application.task is None)
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(len(self.daemon.application.cognition.memory.speech()), 1)
        snapshot = self.daemon.application.sessions.snapshot().sessions[0]
        self.assertTrue(snapshot.waiting_for_input)
        self.assertEqual(snapshot.waiting_seconds, 0)
        self.assertEqual(
            json.loads(self.requests[-1]["messages"][-1]["content"])["focus_session"][
                "progress"
            ],
            "Choose a target",
        )

    async def test_check_in_discards_speech_when_the_active_call_finishes(self) -> None:
        self.use_clock()
        self.slow = True
        self.event("tool_start", harness="codex", call="checks", tool="pytest")
        await self.wait_for(lambda: self.daemon.application.sessions.working)
        self.now += 300
        await asyncio.wait_for(self.entered.wait(), 1)
        self.event("tool_done", harness="codex", call="checks", progress="Tests passed")
        await self.wait_for(
            lambda: not self.daemon.application.sessions.lanes["codex:one"].calls
        )
        self.assertEqual(self.daemon.application.events.activity(), "working")
        self.release.set()
        await self.wait_for(lambda: self.daemon.application.task is None)
        self.assertEqual(self.daemon.application.cognition.memory.speech(), [])
        self.assertFalse(any("text" in packet for packet in await self.drain()))

    async def test_check_in_discards_speech_when_a_different_call_starts(self) -> None:
        self.use_clock()
        self.slow = True
        self.event("tool_start", harness="codex", call="checks", tool="pytest")
        await self.wait_for(lambda: self.daemon.application.sessions.working)
        self.now += 300
        await asyncio.wait_for(self.entered.wait(), 1)
        self.event("tool_done", harness="codex", call="checks", progress="Tests passed")
        self.event("tool_start", harness="codex", call="lint", tool="lint")
        await self.wait_for(
            lambda: "lint" in self.daemon.application.sessions.lanes["codex:one"].calls
        )
        self.release.set()
        await self.wait_for(lambda: self.daemon.application.task is None)
        self.assertEqual(self.daemon.application.cognition.memory.speech(), [])
        self.assertFalse(any("text" in packet for packet in await self.drain()))

    async def test_recovery_discards_stale_automatic_speech_without_recording_it(
        self,
    ) -> None:
        self.slow = True
        self.event("intent", harness="claude", goal="Repair tests")
        self.event("tool_start", harness="claude", call="bad", tool="pytest")
        await self.packet(lambda packet: packet.get("state") == "working")
        self.event("tool_error", harness="claude", call="bad", detail="AssertionError")
        await self.wait_for(lambda: self.daemon.application.sessions.failed)
        self.daemon.application.attention.pending = None
        task = asyncio.create_task(self.observe("failure"))
        await asyncio.wait_for(self.entered.wait(), 1)
        self.event("tool_start", harness="claude", call="retry", tool="pytest")
        self.event("tool_done", harness="claude", call="retry")
        await self.wait_for(lambda: not self.daemon.application.sessions.failed)
        self.release.set()
        await task
        self.assertEqual(self.daemon.application.cognition.memory.speech(), [])
        self.assertFalse(any("text" in packet for packet in await self.drain()))

    async def test_trouble_speech_stays_discarded_after_recovery_and_another_failure(
        self,
    ) -> None:
        self.slow = True
        for index in range(3):
            self.event("tool_start", call=str(index), tool="pytest")
            self.event("tool_error", call=str(index))
        await asyncio.wait_for(self.entered.wait(), 2)
        self.config.behavior.automatic_speech_cooldown_seconds = 90
        self.event("tool_start", call="retry", tool="pytest")
        self.event("tool_done", call="retry")
        self.event("tool_start", call="another", tool="pytest")
        self.event("tool_error", call="another")
        await self.wait_for(
            lambda: (
                require(self.daemon.application.sessions.lanes["cli:one"].failure).count
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
        self.slow = True
        self.event("tool_start", call="bad", tool="pytest")
        self.event("tool_error", call="bad")
        self.event("tool_start", call="retry", tool="pytest")
        self.event("tool_done", call="retry")
        await asyncio.wait_for(self.entered.wait(), 2)
        self.config.behavior.automatic_speech_cooldown_seconds = 90
        self.event("tool_start", call="another", tool="pytest")
        self.event("tool_error", call="another")
        self.event("tool_start", call="retry-again", tool="pytest")
        self.event("tool_done", call="retry-again")
        await self.wait_for(
            lambda: (
                self.daemon.application.sessions.lanes["cli:one"].last_tool == "pytest"
                and not self.daemon.application.sessions.lanes["cli:one"].calls
            )
        )
        await self.wait_for(lambda: self.daemon.application.episodes.serial >= 5)
        self.release.set()
        await self.wait_for(lambda: self.daemon.application.task is None)
        self.assertEqual(self.daemon.application.cognition.memory.speech(), [])
        self.assertFalse(any("text" in packet for packet in await self.drain()))

    async def test_failed_request_leaves_fast_state_delivery_running(self) -> None:
        self.http_status = 500
        with self.assertLogs("pet-daemon", level="WARNING"):
            await self.observe("routine")
        self.assertEqual(len(self.requests), 1)
        self.event("tool_start", call="work", tool="make", manual=True)
        await self.packet(lambda packet: packet.get("state") == "working")
        self.event("tool_done", call="work", manual=True)
        await self.packet(lambda packet: packet.get("state") == "idle")
        self.http_status = 200
        await self.observe("routine")
        self.assertEqual(
            (await self.packet(lambda packet: "text" in packet))["text"], self.speech
        )

    async def test_timeout_cancels_request_and_accepts_next_commentary(self) -> None:
        self.slow = True
        self.config.request_timeout = 0.03
        with self.assertLogs("pet-daemon", level="WARNING"):
            await self.observe("routine")
        self.assertEqual(len(self.requests), 1)
        self.slow = False
        self.config.request_timeout = 2
        await self.observe("explicit", "Try again", ttl=7)
        packet = await self.packet(lambda packet: "text" in packet)
        self.assertEqual(packet["ttl"], 7)

    async def test_forwards_long_unicode_speech_without_trimming(self) -> None:
        self.speech = "One sufficiently long observation. 猫\n" * 200
        await self.observe("explicit", "Observe the parallel work")
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
                bool(self.daemon.application.sessions.lanes)
                and all(
                    lane.finished
                    for lane in self.daemon.application.sessions.lanes.values()
                )
            )
        )
        lane = next(iter(self.daemon.application.sessions.lanes.values()))
        self.assertEqual(lane.outcome, "error")
        self.assertEqual(require(lane.failure).activity, "command")
        self.assertFalse(hasattr(require(lane.failure), "detail"))
        await self.observe("failure")
        snapshot = json.loads(self.requests[-1]["messages"][-1]["content"])
        self.assertEqual(snapshot["focus_session"]["failure"]["activity"], "command")
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
        for data in (b"{", b"\xff", b'{"kind":"event","harness":[]}', b"x" * 2001):
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
            with self.subTest(name=name, value=value):
                config = Config(pet_dir=Path("fixture"))
                setattr(config.behavior, name, value)
                with self.assertRaisesRegex(ValueError, f"behavior.{name}"):
                    config.validate()

    def test_speech_history_window_is_positive_and_defaults_to_thirty_minutes(
        self,
    ) -> None:
        config = Config(pet_dir=Path("fixture"))
        self.assertEqual(config.speech_history_seconds, 1800)
        for value in (0, -1, True, "30", float("inf")):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "speech_history_seconds"),
            ):
                config = Config(pet_dir=Path("fixture"))
                set_config_value(config, "speech_history_seconds", value)
                config.validate()

    def test_summary_defaults_run_on_prompts(self) -> None:
        config = Config(pet_dir=Path("fixture"))
        self.assertEqual(config.summarize_on, ["prompt"])
        self.assertFalse(config.summarize_first_request_only)

    def test_summary_reasons_accept_every_commentary_trigger_and_an_empty_array(
        self,
    ) -> None:
        for reasons in (
            [],
            [
                "prompt",
                "completion",
                "routine",
                "failure",
                "check_in",
                "click",
                "explicit",
            ],
        ):
            with self.subTest(reasons=reasons):
                config = Config(pet_dir=Path("fixture"))
                set_config_value(config, "summarize_on", reasons)
                config.validate()

    def test_summary_configuration_rejects_unknown_reasons_and_nonboolean_flags(
        self,
    ) -> None:
        for reasons in ("prompt", None, ["intent"], [None], [[]]):
            with (
                self.subTest(reasons=reasons),
                self.assertRaisesRegex(ValueError, "summarize_on"),
            ):
                config = Config(pet_dir=Path("fixture"))
                set_config_value(config, "summarize_on", reasons)
                config.validate()
        for value in (1, 0, "true", None):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "summarize_first_request_only"),
            ):
                config = Config(pet_dir=Path("fixture"))
                set_config_value(config, "summarize_first_request_only", value)
                config.validate()

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

    def test_idle_message_interval_defaults_to_five_minutes(self) -> None:
        self.assertEqual(Config(pet_dir=Path("fixture")).idle_message_seconds, 300)

    def test_rejects_nonpositive_or_nonfinite_idle_message_intervals(self) -> None:
        for seconds in (0, -1, True, "300", None, float("nan"), float("inf")):
            with (
                self.subTest(seconds=seconds),
                self.assertRaisesRegex(ValueError, "idle_message_seconds"),
            ):
                config = Config(pet_dir=Path("fixture"))
                set_config_value(config, "idle_message_seconds", seconds)
                config.validate()

    def test_rejects_empty_or_nontext_persona(self) -> None:
        for persona in ("", "   ", None, 3):
            with self.subTest(persona=persona), self.assertRaises(ValueError):
                config = Config(pet_dir=Path("fixture"))
                set_config_value(config, "persona", persona)
                config.validate()

    def test_loads_long_unicode_persona_without_changing_the_text(self) -> None:
        persona = "Be a curious companion. 猫 🐱\n" * 200
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pet.json"
            path.write_text(json.dumps({"pet_dir": "fixture", "persona": persona}))
            config = Config.load(path)
            self.assertEqual(config.persona, persona)

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
                    config = Config(pet_dir=Path("fixture"))
                    setattr(config, field, value)
                    config.validate()

    def test_context_target_and_memory_retention_defaults(self) -> None:
        config = Config(pet_dir=Path("fixture"))
        self.assertEqual(config.prompt_target_bytes, 16000)
        self.assertEqual(config.memory_retention_days, 30)
        self.assertEqual(config.memory_max_events, 10000)
        self.assertEqual(config.max_tokens, 512)

    def test_rejects_invalid_intervals_and_shared_socket_before_startup(self) -> None:
        for config in (
            Config(pet_dir=Path("fixture"), observation_seconds=(90, 60)),
            Config(
                pet_dir=Path("fixture"),
                daemon_socket="/tmp/same",
                overlay_socket="/tmp/same",
            ),
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                config.validate()
