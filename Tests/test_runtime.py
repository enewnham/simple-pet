import asyncio
import base64
import json
import os
from pathlib import Path
import random
import socket
import tempfile
import unittest

import httpx
from anthropic import AsyncAnthropic

from pet_daemon.config import Config
from pet_daemon.daemon import Daemon, bind_socket


PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII=")


def reply(speak=None, gesture=None):
    return httpx.Response(200, json={
        "id": "msg_fixture", "type": "message", "role": "assistant", "model": "claude-haiku-5-5",
        "content": [{"type": "text", "text": json.dumps({"thought": "Observed current work.", "speak": speak, "gesture": gesture})}],
        "stop_reason": "end_turn", "stop_sequence": None,
        "usage": {"input_tokens": 100, "output_tokens": 20},
    })


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory(dir="/tmp", prefix="pet-runtime-")
        root = Path(self.temp.name)
        (root / "pixel.png").write_bytes(PNG)
        (root / "manifest.json").write_text(json.dumps({"pet": "quiet", "rest": "still", "clips": {
            "still": {"frames": ["pixel.png"], "fps": 1, "role": "rest"},
        }}))
        self.config = Config(pet_dir=root, daemon_socket=str(root / "daemon.sock"), overlay_socket=str(root / "overlay.sock"),
                             observation_seconds=(3600, 3600), request_timeout=2)
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.requests = []
        self.http_status = 200
        self.speech = "The tests have reconsidered."
        self.slow = False

        async def handler(request):
            self.requests.append(json.loads(request.content))
            self.entered.set()
            if self.slow:
                await self.release.wait()
            if self.http_status != 200:
                return httpx.Response(self.http_status, json={"type": "error", "error": {"type": "api_error", "message": "fixture failure"}})
            return reply(self.speech)

        self.client = AsyncAnthropic(api_key="test-fixture", max_retries=0,
                                     http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        self.overlay = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.overlay.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 131072)
        self.overlay.bind(self.config.overlay_socket)
        self.overlay.setblocking(False)
        self.hook = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.daemon = Daemon(self.config, self.client, rng=random.Random(7))
        await self.daemon.start()
        self.send_status()
        await self.packet(lambda packet: packet.get("state") == "idle")

    async def asyncTearDown(self):
        await self.daemon.close()
        await self.client.close()
        self.overlay.close()
        self.hook.close()
        self.temp.cleanup()

    def send(self, packet):
        self.hook.sendto(json.dumps(packet).encode(), self.config.daemon_socket)

    def event(self, event, session="one", harness="cli", **extra):
        self.send({"kind": "event", "event": event, "harness": harness, "session": session, **extra})

    def send_status(self, **extra):
        self.send({"kind": "renderer", "event": "status", "x": 100, "y": 100, "width": 200, "height": 160,
                   "screen": {"x": 0, "y": 0, "width": 1000, "height": 800},
                   "pointer": {"x": 2000, "y": 2000, "moving": False}, "dragging": False, "menu": False, "speech": False, **extra})

    async def packet(self, predicate, timeout=1):
        async def receive():
            while True:
                data = await asyncio.get_running_loop().sock_recv(self.overlay, 65536)
                packet = json.loads(data)
                if predicate(packet):
                    return packet
        return await asyncio.wait_for(receive(), timeout)

    async def wait_for(self, condition, timeout=1):
        async def wait():
            while not condition():
                await asyncio.sleep(0.005)
        await asyncio.wait_for(wait(), timeout)

    async def drain(self):
        result = []
        await asyncio.sleep(0.01)
        while True:
            try:
                result.append(json.loads(self.overlay.recv(65536)))
            except BlockingIOError:
                return result

    async def test_real_socket_tracks_parallel_calls_without_model_latency(self):
        self.slow = True
        self.send({"kind": "speak", "context": "Watch these builds"})
        await asyncio.wait_for(self.entered.wait(), 1)
        self.event("tool_start", call="build", tool="cargo build", manual=True)
        await self.packet(lambda packet: packet.get("state") == "working")
        self.event("tool_start", session="two", harness="cursor", call="lint", tool="lint")
        await self.wait_for(lambda: len(self.daemon.sessions.lanes) == 2)
        self.event("tool_done", call="build", manual=True)
        await self.wait_for(lambda: not self.daemon.sessions.lanes["cli:one"].calls)
        self.assertTrue(self.daemon.sessions.working)
        self.assertFalse(self.release.is_set())
        self.event("tool_done", session="two", harness="cursor", call="lint")
        await self.packet(lambda packet: packet.get("state") == "idle")
        self.release.set()
        speech = await self.packet(lambda packet: "text" in packet)
        self.assertEqual(speech["text"], self.speech)
        self.assertEqual(speech["ttl"], 5)
        self.assertEqual(len(self.requests), 1)

    async def test_recovery_discards_stale_automatic_speech_but_keeps_thought(self):
        self.slow = True
        self.event("intent", harness="claude", goal="Repair tests")
        self.event("tool_start", harness="claude", call="bad", tool="pytest")
        await self.packet(lambda packet: packet.get("state") == "working")
        self.event("tool_error", harness="claude", call="bad", detail="AssertionError")
        await self.packet(lambda packet: packet.get("state") == "error")
        task = asyncio.create_task(self.daemon.observe("failure"))
        await asyncio.wait_for(self.entered.wait(), 1)
        self.event("tool_start", harness="claude", call="retry", tool="pytest")
        self.event("tool_done", harness="claude", call="retry")
        await self.wait_for(lambda: not self.daemon.sessions.failed)
        self.release.set()
        await task
        self.assertEqual(len(self.daemon.cognition.history), 2)
        self.assertFalse(any("text" in packet for packet in await self.drain()))

    async def test_failed_request_leaves_fast_state_delivery_running(self):
        self.http_status = 500
        with self.assertLogs("pet-daemon", level="WARNING"):
            await self.daemon.observe("routine")
        self.assertEqual(len(self.requests), 1)
        self.event("tool_start", call="work", tool="make", manual=True)
        await self.packet(lambda packet: packet.get("state") == "working")
        self.event("tool_done", call="work", manual=True)
        await self.packet(lambda packet: packet.get("state") == "idle")
        self.http_status = 200
        await self.daemon.observe("routine")
        self.assertEqual((await self.packet(lambda packet: "text" in packet))["text"], self.speech)

    async def test_timeout_cancels_request_and_accepts_next_commentary(self):
        self.slow = True
        self.config.request_timeout = 0.03
        with self.assertLogs("pet-daemon", level="WARNING"):
            await self.daemon.observe("routine")
        self.assertEqual(len(self.requests), 1)
        self.slow = False
        self.config.request_timeout = 2
        await self.daemon.observe("explicit", "Try again", ttl=7)
        packet = await self.packet(lambda packet: "text" in packet)
        self.assertEqual(packet["ttl"], 7)

    async def test_forwards_long_unicode_speech_without_trimming(self):
        self.speech = "One sufficiently long observation. 猫\n" * 200
        await self.daemon.observe("explicit", "Observe the parallel work")
        packet = await self.packet(lambda packet: "text" in packet)
        self.assertGreater(len(json.dumps(packet).encode()), 2048)
        self.assertEqual(packet["text"], self.speech)

    async def test_wrapper_failure_preserves_exit_detail_through_session_end(self):
        root = Path(__file__).resolve().parents[1]
        env = dict(os.environ, PET_DAEMON_SOCKET=self.config.daemon_socket, PET_COMMENTARY="")
        child = await asyncio.create_subprocess_exec(str(root / "examples/pet-run"), "/bin/sh", "-c", "exit 7", env=env)
        self.assertEqual(await child.wait(), 7)
        await self.wait_for(lambda: bool(self.daemon.sessions.lanes) and all(lane.finished for lane in self.daemon.sessions.lanes.values()))
        lane = next(iter(self.daemon.sessions.lanes.values()))
        self.assertEqual(lane.outcome, "error")
        self.assertEqual(lane.failure["detail"], "Exit code 7")
        await self.daemon.observe("failure")
        snapshot = json.loads(self.requests[-1]["messages"][-1]["content"])
        self.assertEqual(snapshot["sessions"][0]["failure"]["detail"], "Exit code 7")

    async def test_live_listener_is_preserved_and_owned_socket_cleans_up(self):
        path = self.config.daemon_socket
        with self.assertRaisesRegex(ValueError, "already listening"):
            bind_socket(path)
        self.assertEqual(os.stat(path).st_mode & 0o777, 0o600)
        self.event("tool_start", call="test", manual=True)
        await self.packet(lambda packet: packet.get("state") == "working")
        await self.daemon.close()
        self.assertFalse(Path(path).exists())

    async def test_startup_failure_cleans_only_its_own_socket_and_renderer(self):
        await self.daemon.close()
        self.config.overlay_binary = Path(self.temp.name) / "bad-renderer"
        self.config.overlay_binary.write_text("#!/bin/sh\nexit 4\n")
        self.config.overlay_binary.chmod(0o755)
        daemon = Daemon(self.config, self.client)
        with self.assertRaisesRegex(RuntimeError, "Renderer exited"):
            await daemon.run()
        self.assertFalse(Path(self.config.daemon_socket).exists())
        self.assertTrue(Path(self.config.overlay_socket).exists())
        self.assertEqual(daemon.child.returncode, 4)

    async def test_malformed_packets_do_not_change_valid_session_state(self):
        self.event("tool_start", call="real", manual=True)
        await self.packet(lambda packet: packet.get("state") == "working")
        for data in (b"{", b"\xff", b'{"kind":"event","harness":[]}', b"x" * 2001):
            self.hook.sendto(data, self.config.daemon_socket)
        self.event("tool_done", call="real", manual=True)
        await self.packet(lambda packet: packet.get("state") == "idle")
        self.assertFalse(self.daemon.sessions.working)


class ConfigurationTests(unittest.TestCase):
    def test_loads_startup_paths_relative_to_config_and_configurable_timing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pet.json"
            path.write_text(json.dumps({"pet_dir": "pets/other", "observation_seconds": [15, 25], "history_pairs": 6,
                                        "behavior": {"repeat_range": [2, 4], "roaming": False}}))
            config = Config.load(path)
            self.assertEqual(config.pet_dir, Path(directory).resolve() / "pets/other")
            self.assertEqual(config.observation_seconds, [15, 25])
            self.assertFalse(config.behavior.roaming)
            self.assertEqual(config.behavior.repeat_range, [2, 4])

    def test_rejects_invalid_intervals_and_shared_socket_before_startup(self):
        for config in (Config(observation_seconds=(90, 60)), Config(history_pairs=0),
                       Config(daemon_socket="/tmp/same", overlay_socket="/tmp/same")):
            with self.subTest(config=config), self.assertRaises(ValueError):
                config.validate()
