import asyncio
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import unittest

import httpx
from anthropic import AsyncAnthropic

from pet_daemon.config import Config
from pet_daemon.daemon import Daemon


@unittest.skipUnless(os.environ.get("PET_DESKTOP_CHECK") == "1", "Run explicitly in a logged-in macOS desktop")
class DesktopTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_renderer_and_daemon_round_trip_hooks_motion_and_sdk_speech(self):
        root = Path(__file__).resolve().parents[1]
        selected = Config.load(root / "pet.config.json")
        with tempfile.TemporaryDirectory(dir="/tmp", prefix="pet-desktop-") as directory:
            config = Config(pet_dir=selected.pet_dir, overlay_binary=root / ".build/pet-overlay",
                            daemon_socket=directory + "/daemon.sock", overlay_socket=directory + "/overlay.sock",
                            observation_seconds=(3600, 3600), speech_ttl=1, memory_file=Path(directory) / "memory.sqlite3")
            config.behavior.repeat_range = (1, 1)
            config.behavior.gesture_cooldown = 0.05
            config.behavior.idle_pause = config.behavior.working_pause = (0.05, 0.05)
            config.behavior.resume_seconds = 0.05
            config.behavior.pointer_radius = 0.01
            config.behavior.walk_seconds = (0.2, 0.2)
            config.behavior.baseline_dwell = config.behavior.idle_debounce = 0.05
            config.behavior.error_seconds = 0.2
            entered, release = asyncio.Event(), asyncio.Event()
            requests = []
            speech = "The parser remains unconvinced. 猫 " * 100

            async def provider(request):
                body = json.loads(request.content)
                requests.append(body)
                snapshot = json.loads(body["messages"][-1]["content"])
                explicit = snapshot["reason"] == "explicit"
                if explicit:
                    entered.set()
                    await release.wait()
                return httpx.Response(200, json={
                    "id": "msg_desktop", "type": "message", "role": "assistant", "model": config.model,
                    "content": [{"type": "text", "text": json.dumps({
                        "thought": "Two coding streams are being observed.",
                        "speak": speech if explicit else None, "gesture": "waving" if explicit else None, "session_context": None,
                    })}],
                    "stop_reason": "end_turn", "stop_sequence": None,
                    "usage": {"input_tokens": 100, "output_tokens": 20},
                })

            async with AsyncAnthropic(api_key="desktop-test-fixture", max_retries=0,
                                      http_client=httpx.AsyncClient(transport=httpx.MockTransport(provider))) as client:
                daemon = Daemon(config, client, rng=random.Random(7))
                task = asyncio.create_task(daemon.run())
                env = dict(os.environ, PET_DAEMON_SOCKET=config.daemon_socket, PET_COMMENTARY="")

                async def wait_for(condition, timeout=5):
                    async def wait():
                        while not condition():
                            if task.done():
                                task.result()
                                self.fail("Daemon exited before desktop validation completed")
                            await asyncio.sleep(0.01)
                    await asyncio.wait_for(wait(), timeout)

                async def cli(*arguments):
                    process = await asyncio.create_subprocess_exec(sys.executable, str(root / "bin/pet-notify"), *arguments, env=env)
                    self.assertEqual(await process.wait(), 0)

                async def hook(harness, event, payload, expected=None):
                    process = await asyncio.create_subprocess_exec(
                        sys.executable, str(root / "scripts/pet-hook"), harness, event, env=env,
                        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                    )
                    output, _ = await process.communicate(json.dumps(payload).encode())
                    self.assertEqual(process.returncode, 0)
                    self.assertEqual(json.loads(output), {} if expected is None else expected)

                try:
                    await wait_for(lambda: daemon.behavior.status is not None, 10)
                    initial = (daemon.behavior.status["x"], daemon.behavior.status["y"])
                    await wait_for(lambda: (daemon.behavior.status["x"], daemon.behavior.status["y"]) != initial)
                    status = daemon.behavior.status
                    screen = status["screen"]
                    self.assertGreaterEqual(status["x"], screen["x"])
                    self.assertLessEqual(status["x"] + status["width"], screen["x"] + screen["width"])
                    config.behavior.nap_interval = (0.2, 0.2)
                    config.behavior.nap_seconds = (2, 2)
                    daemon.behavior.next_nap = daemon.clock() + 0.2
                    await wait_for(lambda: daemon.behavior.status["sleeping"])
                    config.behavior.nap_interval = (120, 240)
                    await cli("intent", "--session", "build", "--text", "Build the parser")
                    await wait_for(lambda: not daemon.behavior.status["sleeping"])
                    await cli("--session", "build", "--status", "start", "--tool", "cargo build")
                    await hook("cursor", "beforeSubmitPrompt", {"conversation_id": "cursor-tests", "prompt": "Test the parser"}, expected={"continue": True})
                    await hook("cursor", "preToolUse", {
                        "conversation_id": "cursor-tests", "generation_id": "cursor-turn", "tool_use_id": "tests",
                        "tool_name": "Shell", "tool_input": {"command": "pytest"},
                    })
                    await wait_for(lambda: len(daemon.sessions.lanes) == 2 and daemon.sessions.working)
                    await cli("speak", "--context", "Observe these parallel parser tasks", "--ttl", "1")
                    await asyncio.wait_for(entered.wait(), 2)
                    await cli("--session", "build", "--status", "done")
                    await wait_for(lambda: daemon.sessions.lanes["cli:build"].finished)
                    self.assertTrue(daemon.sessions.working)
                    await hook("cursor", "postToolUseFailure", {
                        "conversation_id": "cursor-tests", "generation_id": "cursor-turn",
                        "tool_use_id": "tests", "tool_name": "Shell", "error_message": "Parser AssertionError",
                    })
                    await wait_for(lambda: daemon.behavior.state == "error")
                    self.assertFalse(release.is_set())
                    release.set()
                    await wait_for(lambda: daemon.behavior.status["speech"])
                    self.assertGreater(len(speech.encode()), 2048)
                    await wait_for(lambda: not daemon.behavior.status["speech"])
                    self.assertTrue(daemon.cognition.memory.speech())
                    self.assertFalse(daemon.sessions.working)
                    self.assertTrue(any(json.loads(request["messages"][-1]["content"])["reason"] == "explicit" for request in requests))
                finally:
                    release.set()
                    daemon.stopping.set()
                    try:
                        await asyncio.wait_for(task, 5)
                    finally:
                        await daemon.close()
                self.assertFalse(Path(config.daemon_socket).exists())
                self.assertFalse(Path(config.overlay_socket).exists())
