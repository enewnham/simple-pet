import asyncio
import json
import random
import socket
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import Unpack

import httpx2 as httpx
from anthropic import AsyncAnthropic

from pet_daemon.commentary.contracts import ModelRequest
from pet_daemon.config import Config
from pet_daemon.daemon import Daemon
from pet_daemon.domain import SessionSource
from pet_daemon.identity import SessionIdentity
from pet_daemon.runtime import OverlayPacket
from pet_daemon.sources import RepositoryTracker
from pet_daemon.sources.git import CommandOptions, command
from pet_daemon.types import JSONObject
from pet_daemon.work.observations import BranchObservation, PullRequestObservation
from Tests.fixtures import require, story_response
from Tests.test_runtime import PNG, reply


class WorkRuntimeTests(unittest.IsolatedAsyncioTestCase):
    @property
    def repositories(self) -> RepositoryTracker:
        provider = self.daemon.application.providers[0]
        assert isinstance(provider, RepositoryTracker)
        return provider

    async def asyncSetUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(dir="/tmp", prefix="pet-work-runtime-")
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
        self.config.behavior.automatic_speech_cooldown_seconds = 3600
        self.requests: list[ModelRequest] = []
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.slow = False
        self.speech = "The agents report session persistence survives a restart."
        self.github_commands = []
        self.related_prs = {}
        self.stack_entries = []
        self.repository_path = root / "repository"
        self.repository_path.mkdir()
        for args in (
            ("init", "-b", "feature"),
            ("config", "user.email", "pet@example.test"),
            ("config", "user.name", "Pet"),
            ("config", "commit.gpgsign", "false"),
        ):
            await command("git", *args, cwd=self.repository_path)
        self.head = await self.commit("initial session state")
        await command(
            "git",
            "remote",
            "add",
            "origin",
            "https://example.test/owner/repository.git",
            cwd=self.repository_path,
        )
        self.pr = {
            "number": 7,
            "url": "https://example.test/owner/repository/pull/7",
            "state": "OPEN",
            "mergedAt": None,
            "title": "Persist coding session context",
            "body": "Retain the requested session context when the daemon restarts.",
            "isDraft": True,
            "headRefOid": self.head,
            "headRefName": "feature",
            "headRepository": {"nameWithOwner": "owner/repository"},
            "headRepositoryOwner": {"login": "owner"},
        }

        async def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            if body["output_config"]["format"]["schema"]["required"] == [
                "intent",
                "decisions",
                "progress",
            ]:
                return httpx.Response(
                    200,
                    json={
                        "id": "story-fixture",
                        "type": "message",
                        "role": "assistant",
                        "model": self.config.model,
                        "content": [
                            {"type": "text", "text": json.dumps(story_response(body))}
                        ],
                        "stop_reason": "end_turn",
                        "usage": {"input_tokens": 10, "output_tokens": 10},
                    },
                )
            self.requests.append(body)
            speech = self.speech
            self.entered.set()
            if self.slow:
                await self.release.wait()
            return reply(speech)

        self.client = AsyncAnthropic(
            api_key="test-fixture",
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        self.overlay = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.overlay.bind(self.config.overlay_socket)
        self.overlay.setblocking(False)
        self.hook = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        await self.start_daemon()

    async def asyncTearDown(self) -> None:
        await self.daemon.close()
        await self.client.close()
        self.overlay.close()
        self.hook.close()
        self.temp.cleanup()

    async def commit(self, content: str) -> str:
        (self.repository_path / "session.txt").write_text(content + "\n")
        await command("git", "add", "session.txt", cwd=self.repository_path)
        await command(
            "git", "commit", "-m", "Record session state", cwd=self.repository_path
        )
        return (
            await command("git", "rev-parse", "HEAD", cwd=self.repository_path)
        ).strip()

    async def start_daemon(self) -> None:
        self.daemon = Daemon(self.config, self.client, rng=random.Random(7))
        self.repositories.run = self.repository_command
        self.daemon.application.attention.last_attempt = self.daemon.clock()
        await self.daemon.start()
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
                "sleeping": self.daemon.application.behavior.is_sleeping,
                "system_sleeping": self.daemon.application.suspended,
            }
        )
        await self.packet(lambda packet: packet.get("state") == "idle")

    async def repository_command(
        self, *args: str, **kwargs: Unpack[CommandOptions]
    ) -> str:
        if args[0] != "gh":
            return await command(*args, **kwargs)
        self.github_commands.append(args)
        if args[:3] == ("gh", "repo", "view"):
            return json.dumps({"isFork": False})
        if args[:3] == ("gh", "pr", "list"):
            return json.dumps([self.pr])
        if args[:3] == ("gh", "pr", "view"):
            return json.dumps(self.related_prs.get(args[3], self.pr))
        if args[:2] == ("gh", "api") and "/stacks?" in args[2]:
            return json.dumps(self.stack_entries)
        self.fail(f"Unexpected GitHub command: {args}")

    def send(self, packet: JSONObject) -> None:
        self.hook.sendto(json.dumps(packet).encode(), self.config.daemon_socket)

    async def packet(self, predicate: Callable[[OverlayPacket], bool]) -> OverlayPacket:
        async def receive() -> OverlayPacket:
            while True:
                data = await asyncio.get_running_loop().sock_recv(self.overlay, 65536)
                packet = json.loads(data)
                if predicate(packet):
                    return packet

        return await asyncio.wait_for(receive(), 10)

    async def wait_for(self, condition: Callable[[], bool]) -> None:
        async def wait() -> None:
            while not condition():
                await asyncio.sleep(0.005)

        await asyncio.wait_for(wait(), 10)

    async def drain(self) -> list[OverlayPacket]:
        packets = []
        while True:
            try:
                packets.append(json.loads(self.overlay.recv(65536)))
            except BlockingIOError:
                return packets

    async def refresh_sources(self, predicate: Callable[[], bool]) -> None:
        previous = len(self.github_commands)
        self.repositories.local_interval = 0
        self.repositories.github_interval = 0
        try:
            await self.wait_for(
                lambda: len(self.github_commands) > previous and predicate()
            )
        finally:
            self.repositories.local_interval = 3600
            self.repositories.github_interval = 3600
        await self.wait_for(lambda: not self.repositories.tasks)

    def schedule_routine(self) -> None:
        self.daemon.application.attention.next_observation = self.daemon.clock()
        self.daemon.application.tick()

    async def accept(self, speech: OverlayPacket) -> None:
        self.send(
            {
                "kind": "renderer",
                "event": "speech_accepted",
                "delivery_id": speech["delivery_id"],
            }
        )
        await self.wait_for(
            lambda: (
                (
                    outcome := self.daemon.application.memory.message_outcome(
                        speech["delivery_id"]
                    )
                )
                is not None
                and outcome.outcome == "accepted"
            )
        )

    def assert_human_context(self, value: object) -> None:
        if isinstance(value, dict):
            forbidden = {
                "id",
                "revision",
                "head",
                "number",
                "repository",
                "session",
                "work_id",
                "worktree",
                "checkout",
                "staged",
                "unstaged",
                "untracked",
                "ahead",
                "behind",
                "count",
                "timestamp",
                "changed_at",
                "observed_at",
                "artifacts",
                "activity",
                "summary",
                "version",
                "delivery_id",
            }
            self.assertFalse(forbidden.intersection(value))
            for item in value.values():
                self.assert_human_context(item)
        elif isinstance(value, list):
            for item in value:
                self.assert_human_context(item)

    async def native_work(self) -> SessionIdentity:
        self.send(
            {
                "kind": "event",
                "source": "codex",
                "session": "persistence",
                "event": "intent",
                "turn": "implementation",
                "repo": str(self.repository_path),
                "goal": "Preserve coding session context across daemon restarts",
            }
        )
        await self.wait_for(
            lambda: (
                SessionIdentity(source=SessionSource.CODEX, native_id="persistence")
                in self.daemon.application.sessions.by_id
            )
        )
        await self.wait_for(
            lambda: any(
                isinstance(item, (BranchObservation, PullRequestObservation))
                and item.pr
                for item in self.repositories.snapshot()
            )
        )
        self.send(
            {
                "kind": "event",
                "source": "codex",
                "session": "persistence",
                "event": "turn_done",
                "turn": "implementation",
                "repo": str(self.repository_path),
                "outcome": "success",
                "progress": "Implemented session persistence; next I will document restart behavior.",
            }
        )
        await self.wait_for(
            lambda: (
                self.daemon.application.sessions.by_id[
                    SessionIdentity(source=SessionSource.CODEX, native_id="persistence")
                ].outcome
                == "success"
            )
        )
        self.daemon.application.attention.pending = None
        await self.wait_for(
            lambda: any(
                isinstance(item, (BranchObservation, PullRequestObservation))
                and item.pr
                for item in self.repositories.snapshot()
            )
        )
        await self.wait_for(lambda: not self.repositories.tasks)
        self.assertEqual(self.requests, [])
        self.config.behavior.automatic_speech_cooldown_seconds = 0
        await self.drain()
        return next(
            subject.id
            for subject in self.daemon.application.context().work
            if SessionIdentity(source=SessionSource.CODEX, native_id="persistence")
            in subject.sessions
        )

    async def test_native_agent_report_reaches_commentary_without_publication_details(
        self,
    ) -> None:
        work_id = await self.native_work()
        self.schedule_routine()
        speech = await self.packet(lambda packet: "delivery_id" in packet)
        payload = json.loads(self.requests[-1]["messages"][-1]["content"])
        self.assertIn("next I will document restart behavior", json.dumps(payload))
        self.assertEqual(
            payload["current_work"]["request"],
            "Preserve coding session context across daemon restarts",
        )
        self.assertNotIn("Persist coding session context", json.dumps(payload))
        self.assertNotIn("draft", json.dumps(payload).lower())
        self.assert_human_context(payload)
        self.assertEqual(self.daemon.application.memory.speech(), [])
        await self.accept(speech)
        outcome = require(
            self.daemon.application.memory.message_outcome(speech["delivery_id"])
        )
        self.assertEqual((outcome.outcome, outcome.subject_id), ("accepted", work_id))
        self.assertEqual(
            [item.text for item in self.daemon.application.memory.speech()],
            [self.speech],
        )
        self.schedule_routine()
        await asyncio.sleep(0.06)
        self.assertEqual(len(self.requests), 1)
        await self.daemon.close()
        await self.start_daemon()
        self.schedule_routine()
        await asyncio.sleep(0.06)
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(self.daemon.application.memory.speech()[0].text, self.speech)

    async def test_basic_pr_state_refresh_does_not_rearm_the_agent_story(self) -> None:
        await self.native_work()
        self.schedule_routine()
        await self.accept(await self.packet(lambda packet: "delivery_id" in packet))
        self.pr.update(state="MERGED", mergedAt="2026-10-20T12:00:00Z")
        await self.refresh_sources(
            lambda: any(
                isinstance(item, (BranchObservation, PullRequestObservation))
                and item.pr
                and item.pr.state == "MERGED"
                for item in self.repositories.snapshot()
            )
        )
        self.schedule_routine()
        await asyncio.sleep(0.06)
        self.assertEqual(len(self.requests), 1)
        self.config.behavior.automatic_speech_cooldown_seconds = 3600
        self.send(
            {
                "kind": "event",
                "source": "codex",
                "session": "persistence",
                "event": "intent",
                "turn": "documentation",
                "goal": "Preserve coding session context across daemon restarts",
            }
        )
        self.send(
            {
                "kind": "event",
                "source": "codex",
                "session": "persistence",
                "event": "progress",
                "turn": "documentation",
                "progress": "Documented restart behavior; next I will update the examples.",
            }
        )
        await self.wait_for(
            lambda: (
                "update the examples"
                in self.daemon.application.sessions.by_id[
                    SessionIdentity(source=SessionSource.CODEX, native_id="persistence")
                ].progress
            )
        )
        self.daemon.application.attention.pending = None
        self.config.behavior.automatic_speech_cooldown_seconds = 0
        self.speech = "The agents have documented restart behavior."
        self.schedule_routine()
        await self.accept(await self.packet(lambda packet: "delivery_id" in packet))
        self.assertEqual(len(self.requests), 2)
        self.assertNotIn("MERGED", json.dumps(self.requests[-1]))
        self.assertIn("update the examples", json.dumps(self.requests[-1]["messages"]))

    async def test_superseded_report_discards_pending_commentary(self) -> None:
        await self.native_work()
        self.slow = True
        self.schedule_routine()
        await asyncio.wait_for(self.entered.wait(), 5)
        self.config.behavior.automatic_speech_cooldown_seconds = 3600
        self.send(
            {
                "kind": "event",
                "source": "codex",
                "session": "persistence",
                "event": "intent",
                "turn": "documentation",
                "goal": "Preserve coding session context across daemon restarts",
            }
        )
        self.send(
            {
                "kind": "event",
                "source": "codex",
                "session": "persistence",
                "event": "progress",
                "turn": "documentation",
                "progress": "The restart example needs a different approach.",
            }
        )
        await self.wait_for(
            lambda: (
                "different approach"
                in self.daemon.application.sessions.by_id[
                    SessionIdentity(source=SessionSource.CODEX, native_id="persistence")
                ].progress
            )
        )
        self.release.set()
        await self.wait_for(lambda: self.daemon.application.task is None)
        self.assertFalse(any(packet.get("text") for packet in await self.drain()))
        self.assertEqual(self.daemon.application.memory.speech(), [])

    async def test_unselected_pr_body_preserves_pending_commentary(self) -> None:
        await self.native_work()
        self.slow = True
        self.schedule_routine()
        await asyncio.wait_for(self.entered.wait(), 5)
        self.pr["body"] = "Additional context for the human author."
        await self.refresh_sources(
            lambda: any(
                isinstance(item, (BranchObservation, PullRequestObservation))
                and item.pr
                and item.pr.body == self.pr["body"]
                for item in self.repositories.snapshot()
            )
        )
        self.release.set()
        await self.accept(await self.packet(lambda packet: "delivery_id" in packet))
        self.assertEqual(len(self.daemon.application.memory.speech()), 1)
        self.assertNotIn(self.pr["body"], json.dumps(self.requests[-1]))

    async def test_sleep_cancels_retries_and_late_acceptance_records_visible_message(
        self,
    ) -> None:
        work_id = await self.native_work()
        self.schedule_routine()
        first = await self.packet(lambda packet: "delivery_id" in packet)
        await self.wait_for(lambda: self.daemon.application.task is None)
        self.send({"kind": "renderer", "event": "system_sleep"})
        await self.wait_for(lambda: self.daemon.application.suspended)
        now = self.daemon.clock() + 5
        self.daemon.application.delivery.clock = lambda: now
        self.daemon.application.tick()
        self.assertFalse(any("delivery_id" in packet for packet in await self.drain()))
        self.assertEqual(self.daemon.application.memory.speech(), [])
        await self.accept(first)
        self.assertEqual(
            require(
                self.daemon.application.memory.message_outcome(first["delivery_id"])
            ).subject_id,
            work_id,
        )


if __name__ == "__main__":
    unittest.main()
