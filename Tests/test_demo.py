import argparse
import asyncio
import json
import socket
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import BaseModel

from pet_daemon.commentary.contracts import ModelRequest
from pet_daemon.demo import backup_memory, demonstrate, positive_speed, select_work
from pet_daemon.domain import ModelPurpose, SpeechReason
from pet_daemon.replay import FixtureModel, ReplayHarness
from pet_daemon.storage import Memory
from pet_daemon.storage.migrations import SCHEMA_VERSION


class DemoModel(FixtureModel):
    def __init__(self) -> None:
        super().__init__()
        self.speech = "The quoted delimiters still need their multiline check."
        self.memory_contains = "multiline"
        self.ongoing = True
        self.requests: list[ModelRequest] = []

    async def complete(
        self,
        request: ModelRequest,
        reason: SpeechReason | ModelPurpose,
        *,
        response_type: type[BaseModel],
    ) -> dict[str, object]:
        self.requests.append(request)
        return await super().complete(request, reason, response_type=response_type)


class DemoChild:
    def __init__(self) -> None:
        self.returncode: int | None = None
        self.finished = asyncio.Event()

    async def wait(self) -> int:
        await self.finished.wait()
        assert self.returncode is not None
        return self.returncode

    def terminate(self) -> None:
        self.returncode = 0
        self.finished.set()

    def kill(self) -> None:
        self.returncode = -9
        self.finished.set()


class DemoTests(unittest.IsolatedAsyncioTestCase):
    def test_rejects_nonpositive_or_nonfinite_speeds(self) -> None:
        for value in ("0", "-1", "nan", "inf", "-inf"):
            with (
                self.subTest(value=value),
                self.assertRaises(argparse.ArgumentTypeError),
            ):
                positive_speed(value)
        self.assertEqual(positive_speed("10"), 10)

    async def seed(self, replay: ReplayHarness) -> None:
        replay.ingest(
            {
                "source": "codex",
                "session": "parser",
                "event": "intent",
                "goal": "Preserve quoted delimiters",
                "repo": "/saved/parser",
            }
        )
        replay.ingest(
            {
                "source": "codex",
                "session": "parser",
                "event": "progress",
                "progress": "Quoted delimiters still need a multiline check.",
            }
        )
        replay.ingest({"source": "codex", "session": "parser", "event": "session_end"})

    async def test_matches_real_saved_requests_and_reports(self) -> None:
        async with ReplayHarness() as replay:
            await self.seed(replay)
            replay.clock.advance(1)
            replay.ingest(
                {
                    "source": "codex",
                    "session": "renderer",
                    "event": "intent",
                    "goal": "Inspect renderer compatibility",
                    "repo": "/saved/renderer",
                }
            )
            subjects = replay.app.context().work
            self.assertEqual(select_work(subjects, "MULTILINE").id.native_id, "parser")
            self.assertEqual(
                select_work(subjects, "quoted delimiters").id.native_id, "parser"
            )
            self.assertEqual(select_work(subjects, None).id.native_id, "renderer")
            with self.assertRaises(ValueError):
                select_work(subjects, "not a saved thread")

    async def test_migrates_only_the_copy_of_saved_memory(self) -> None:
        async with ReplayHarness() as replay:
            await self.seed(replay)
            source = replay.config.memory_file
            replay.memory.db.execute(f"PRAGMA user_version = {SCHEMA_VERSION - 1}")
            replay.memory.db.commit()
            before = source.read_bytes()
            with tempfile.TemporaryDirectory() as directory:
                copied = Path(directory) / "demo.sqlite3"
                backup_memory(source, copied)
                memory = Memory(copied, replay.pack.pet)
                memory.close()
                self.assertEqual(source.read_bytes(), before)
                self.assertEqual(
                    replay.memory.db.execute("PRAGMA user_version").fetchone()[0],
                    SCHEMA_VERSION - 1,
                )
                self.assertGreater(
                    replay.memory.db.execute(
                        "SELECT COUNT(*) FROM work_reports"
                    ).fetchone()[0],
                    0,
                )
                with sqlite3.connect(copied) as connection:
                    self.assertEqual(
                        connection.execute("PRAGMA user_version").fetchone()[0],
                        SCHEMA_VERSION,
                    )

    async def test_cleans_private_sockets_when_the_renderer_exits_before_readiness(
        self,
    ) -> None:
        async with ReplayHarness() as seed:
            await self.seed(seed)
            child, model = DemoChild(), DemoModel()
            child.returncode = 3
            child.finished.set()
            arguments: tuple[str, ...] = ()

            async def launch(*argv: str) -> DemoChild:
                nonlocal arguments
                arguments = argv
                return child

            with (
                patch("pet_daemon.demo.asyncio.create_subprocess_exec", launch),
                self.assertRaisesRegex(RuntimeError, "Renderer did not become ready"),
            ):
                await demonstrate(seed.config, model, "multiline", 10, asyncio.Event())
            daemon_socket = Path(arguments[arguments.index("--daemon-socket") + 1])
            self.assertFalse(daemon_socket.parent.exists())
            self.assertEqual(model.requests, [])

    async def test_uses_saved_work_and_configured_model_with_private_renderer_lifecycle(
        self,
    ) -> None:
        async with ReplayHarness() as seed:
            await self.seed(seed)
            before = seed.config.memory_file.read_bytes()
            config = seed.config.model_copy(deep=True)
            config.model = "configured-demo-model"
            config.persona = "Speak with the configured quiet humor."
            child, model, stopping = DemoChild(), DemoModel(), asyncio.Event()
            overlay = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            overlay.setblocking(False)
            renderer_task = None
            arguments: tuple[str, ...] = ()
            loop = asyncio.get_running_loop()

            async def launch(*argv: str) -> DemoChild:
                nonlocal renderer_task, arguments
                arguments = argv
                overlay_socket = argv[argv.index("--socket") + 1]
                daemon_socket = argv[argv.index("--daemon-socket") + 1]
                overlay.bind(overlay_socket)

                async def renderer() -> None:
                    status = {
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
                        "sleeping": False,
                        "system_sleeping": False,
                    }
                    await loop.sock_sendto(
                        overlay, json.dumps(status).encode(), daemon_socket
                    )
                    while True:
                        packet = json.loads(await loop.sock_recv(overlay, 65536))
                        if "conversation" in packet and "delivery_id" in packet:
                            stopping.set()

                renderer_task = asyncio.create_task(renderer())
                return child

            try:
                with patch("pet_daemon.demo.asyncio.create_subprocess_exec", launch):
                    await asyncio.wait_for(
                        demonstrate(config, model, "multiline", 10, stopping), 5
                    )
            finally:
                if renderer_task is not None:
                    renderer_task.cancel()
                    await asyncio.gather(renderer_task, return_exceptions=True)
                overlay.close()
            self.assertEqual(child.returncode, 0)
            self.assertEqual(seed.config.memory_file.read_bytes(), before)
            self.assertEqual(seed.config.behavior.naps, config.behavior.naps)
            self.assertTrue(model.requests)
            self.assertTrue(
                all(request["model"] == config.model for request in model.requests)
            )
            self.assertTrue(
                any(
                    config.persona in entry["text"]
                    for request in model.requests
                    for entry in request["system"]
                )
            )
            self.assertTrue(
                any(
                    "multiline" in request["messages"][0]["content"]
                    for request in model.requests
                )
            )
            daemon_socket = Path(arguments[arguments.index("--daemon-socket") + 1])
            self.assertNotEqual(str(daemon_socket), config.daemon_socket)
            self.assertFalse(daemon_socket.parent.exists())
