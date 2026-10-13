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


def reply(speak=None, gesture=None, summary=None):
    result = {"summary": summary} if summary is not None else {"thought": "Observed current work.", "speak": speak, "gesture": gesture}
    return httpx.Response(200, json={
        "id": "msg_fixture", "type": "message", "role": "assistant", "model": "claude-haiku-5-5",
        "content": [{"type": "text", "text": json.dumps(result)}],
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
                             observation_seconds=(3600, 3600), request_timeout=2, memory_file=root / "memory.json")
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.requests = []
        self.http_status = 200
        self.speech = "The tests have reconsidered."
        self.speech_by_reason = {}
        self.slow = False
        self.delay = 0

        async def handler(request):
            body = json.loads(request.content)
            self.requests.append(body)
            self.entered.set()
            if self.delay:
                await asyncio.sleep(self.delay)
            if self.slow:
                await self.release.wait()
            if self.http_status != 200:
                return httpx.Response(self.http_status, json={"type": "error", "error": {"type": "api_error", "message": "fixture failure"}})
            summary = "Remember the parser work." if "summary" in body["output_config"]["format"]["schema"]["properties"] else None
            speech = self.speech
            if summary is None:
                reason = json.loads(body["messages"][-1]["content"])["reason"]
                speech = self.speech_by_reason.get(reason, speech)
            return reply(speech, summary=summary)

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

    def use_clock(self):
        self.now = 1000.0
        clock = lambda: self.now
        self.daemon.clock = clock
        self.daemon.sessions.clock = clock
        self.daemon.cognition.clock = clock
        self.daemon.next_prune = 0
        self.daemon.next_observation = self.now + 3600

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
        self.assertEqual(speech["ttl"], 15)
        self.assertEqual(len(self.requests), 1)

    async def test_prompt_triggers_commentary_without_waiting_for_routine_timer(self):
        self.event("intent", harness="codex", goal="Run local checks")
        speech = await self.packet(lambda packet: "text" in packet)
        self.assertEqual(speech["text"], self.speech)
        snapshot = json.loads(self.requests[0]["messages"][-1]["content"])
        self.assertEqual(snapshot["reason"], "prompt")
        self.assertEqual(snapshot["trigger"]["goal"], "Run local checks")

    async def test_click_requests_fresh_speech_without_waiting_for_routine_timer(self):
        self.send({"kind": "renderer", "event": "click"})
        speech = await self.packet(lambda packet: "text" in packet)
        self.assertEqual(speech["text"], self.speech)
        self.assertEqual(speech["ttl"], 15)
        snapshot = json.loads(self.requests[0]["messages"][-1]["content"])
        self.assertEqual(snapshot["reason"], "click")
        schema = self.requests[0]["output_config"]["format"]["schema"]
        self.assertEqual(schema["properties"]["speak"], {"type": "string"})

    async def test_napping_pauses_routine_commentary_and_click_wakes_for_speech(self):
        self.use_clock()
        self.config.behavior.nap_interval = (1, 1)
        self.config.behavior.nap_seconds = (10, 10)
        self.daemon.behavior.next_nap = self.now + 1
        self.now += 1
        await self.packet(lambda packet: packet.get("sleeping") is True)
        self.daemon.next_observation = self.now
        self.now += 2
        self.send_status(x=101)
        await self.wait_for(lambda: self.daemon.behavior.status["x"] == 101)
        self.assertTrue(self.daemon.behavior.sleeping)
        self.assertEqual(self.requests, [])
        self.send({"kind": "renderer", "event": "click"})
        await self.packet(lambda packet: packet.get("sleeping") is False)
        speech = await self.packet(lambda packet: "text" in packet)
        self.assertEqual(speech["text"], self.speech)
        snapshot = json.loads(self.requests[-1]["messages"][-1]["content"])
        self.assertEqual(snapshot["reason"], "click")
        self.assertFalse(snapshot["sleeping"])

    async def test_prompt_wakes_a_nap_before_tool_work_starts(self):
        self.use_clock()
        self.config.behavior.nap_interval = (1, 1)
        self.daemon.behavior.next_nap = self.now + 1
        self.now += 1
        await self.packet(lambda packet: packet.get("sleeping") is True)
        self.event("intent", harness="codex", goal="Run checks")
        await self.packet(lambda packet: packet.get("sleeping") is False)
        speech = await self.packet(lambda packet: "text" in packet)
        self.assertEqual(speech["text"], self.speech)
        self.assertFalse(self.daemon.sessions.working)
        self.assertFalse(self.daemon.behavior.sleeping)

    async def test_click_replays_recent_speech_and_requests_fresh_speech_after_a_minute(self):
        self.use_clock()
        self.send({"kind": "speak", "context": "Inspect the build", "ttl": 1})
        original = await self.packet(lambda packet: "text" in packet)
        self.assertEqual(original["ttl"], 1)
        await self.wait_for(lambda: self.daemon.request_task is None)
        self.now += 60
        self.send({"kind": "renderer", "event": "click"})
        replay = await self.packet(lambda packet: "text" in packet)
        self.assertEqual(replay["text"], original["text"])
        self.assertEqual(replay["ttl"], 15)
        self.assertEqual(len(self.requests), 1)
        self.speech = "The build has something fresh to say."
        self.now += 1
        self.send({"kind": "renderer", "event": "click"})
        fresh = await self.packet(lambda packet: "text" in packet)
        self.assertEqual(fresh["text"], self.speech)
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(json.loads(self.requests[-1]["messages"][-1]["content"])["reason"], "click")

    async def test_clicks_during_a_request_use_its_response_without_another_model_pass(self):
        self.slow = True
        self.send({"kind": "renderer", "event": "click"})
        await asyncio.wait_for(self.entered.wait(), 1)
        for _ in range(5):
            self.send({"kind": "renderer", "event": "click"})
        self.send_status(x=101)
        await self.wait_for(lambda: self.daemon.behavior.status["x"] == 101)
        self.release.set()
        speech = await self.packet(lambda packet: "text" in packet)
        self.assertEqual(speech["text"], self.speech)
        await self.wait_for(lambda: self.daemon.request_task is None)
        self.assertFalse(self.daemon.clicked)
        self.assertEqual(len(self.requests), 1)

    async def test_click_during_a_silent_observation_gets_a_fresh_comment(self):
        self.slow = True
        self.speech_by_reason["routine"] = None
        self.daemon.next_observation = self.daemon.clock()
        await asyncio.wait_for(self.entered.wait(), 1)
        self.send({"kind": "renderer", "event": "click"})
        self.send_status(x=101)
        await self.wait_for(lambda: self.daemon.behavior.status["x"] == 101)
        self.release.set()
        speech = await self.packet(lambda packet: "text" in packet)
        self.assertEqual(speech["text"], self.speech)
        reasons = [json.loads(request["messages"][-1]["content"])["reason"] for request in self.requests]
        self.assertEqual(reasons, ["routine", "click"])

    async def test_completion_reacts_to_one_session_while_other_tools_keep_working(self):
        self.event("tool_start", harness="codex", call="checks", tool="pytest")
        self.event("tool_start", session="two", harness="cursor", call="build", tool="cargo build")
        await self.wait_for(lambda: len(self.daemon.sessions.lanes) == 2)
        self.event("turn_done", harness="codex", progress="Checks passed")
        speech = await self.packet(lambda packet: "text" in packet, timeout=2)
        snapshot = json.loads(self.requests[0]["messages"][-1]["content"])
        self.assertEqual(snapshot["reason"], "completion")
        self.assertEqual(snapshot["trigger"]["session"], "one")
        self.assertEqual(snapshot["trigger"]["progress"], "Checks passed")
        self.assertEqual(snapshot["activity"], "working")
        self.assertEqual(speech["text"], self.speech)
        self.assertEqual(self.daemon.activity(), "working")

    async def test_tool_progress_during_observation_preserves_commentary(self):
        self.slow = True
        self.event("tool_start", harness="codex", call="checks", tool="pytest")
        await self.packet(lambda packet: packet.get("state") == "working")
        task = asyncio.create_task(self.daemon.observe("routine"))
        await asyncio.wait_for(self.entered.wait(), 1)
        self.event("tool_done", harness="codex", call="checks", progress="Tests passed")
        self.event("tool_start", harness="codex", call="lint", tool="lint")
        await self.wait_for(lambda: "lint" in self.daemon.sessions.lanes["codex:one"].calls)
        self.release.set()
        await task
        speech = await self.packet(lambda packet: "text" in packet)
        self.assertEqual(speech["text"], self.speech)

    async def test_check_in_uses_each_tasks_quiet_time_and_repeats_after_the_interval(self):
        self.use_clock()
        self.event("tool_start", harness="codex", call="checks", tool="pytest")
        self.event("tool_start", session="two", harness="cursor", call="build", tool="build")
        await self.wait_for(lambda: len(self.daemon.sessions.lanes) == 2)
        self.now += 299
        self.event("progress", session="two", harness="cursor", progress="Build progressing")
        await self.wait_for(lambda: self.daemon.sessions.lanes["cursor:two"].progress == "Build progressing")
        await self.wait_for(lambda: self.daemon.next_prune > self.now)
        self.assertEqual(self.requests, [])
        self.now += 1
        speech = await self.packet(lambda packet: "text" in packet)
        self.assertEqual(speech["text"], self.speech)
        snapshot = json.loads(self.requests[0]["messages"][-1]["content"])
        self.assertEqual(snapshot["reason"], "check_in")
        self.assertEqual(snapshot["trigger"], {"sessions": ["codex:one"]})
        sessions = {lane["id"]: lane for lane in snapshot["sessions"]}
        self.assertEqual(set(sessions), {"codex:one"})
        self.assertEqual(sessions["codex:one"]["quiet_seconds"], 300)
        self.now += 299
        self.event("progress", session="two", harness="cursor", progress="Build continuing")
        await self.wait_for(lambda: self.daemon.next_prune > self.now)
        self.assertEqual(len(self.requests), 1)
        self.now += 1
        await self.packet(lambda packet: "text" in packet)
        snapshot = json.loads(self.requests[1]["messages"][-1]["content"])
        self.assertEqual(snapshot["trigger"], {"sessions": ["codex:one"]})

    async def test_failed_check_in_waits_the_interval_before_retrying(self):
        self.use_clock()
        self.http_status = 500
        self.event("tool_start", harness="codex", call="checks", tool="pytest")
        await self.wait_for(lambda: self.daemon.sessions.working)
        with self.assertLogs("pet-daemon", level="WARNING"):
            self.now += 300
            await asyncio.wait_for(self.entered.wait(), 1)
            await self.wait_for(lambda: self.daemon.request_task is None)
        self.assertEqual(len(self.requests), 1)
        self.now += 299
        await self.wait_for(lambda: self.daemon.next_prune > self.now)
        self.assertEqual(len(self.requests), 1)
        self.http_status = 200
        self.now += 1
        await self.packet(lambda packet: "text" in packet)
        self.assertEqual(len(self.requests), 2)
        snapshot = json.loads(self.requests[-1]["messages"][-1]["content"])
        self.assertEqual(snapshot["reason"], "check_in")

    async def test_waiting_check_in_survives_progress_and_discards_an_answered_question(self):
        self.use_clock()
        self.slow = True
        self.event("waiting", harness="claude", call="question", tool="AskUserQuestion", progress="Choose a strategy")
        await self.wait_for(lambda: "claude:one" in self.daemon.sessions.lanes)
        self.now += 299
        self.event("progress", harness="claude", progress="Strategy pending")
        await self.wait_for(lambda: self.daemon.sessions.lanes["claude:one"].progress == "Strategy pending")
        self.now += 1
        await asyncio.wait_for(self.entered.wait(), 1)
        snapshot = json.loads(self.requests[0]["messages"][-1]["content"])
        self.assertEqual(snapshot["reason"], "check_in")
        self.assertEqual(snapshot["trigger"], {"sessions": ["claude:one"]})
        self.assertEqual(snapshot["sessions"][0]["waiting_seconds"], 300)
        self.assertEqual(snapshot["sessions"][0]["quiet_seconds"], 1)
        self.event("tool_done", harness="claude", call="question", progress="Strategy chosen")
        await self.wait_for(lambda: not self.daemon.sessions.lanes["claude:one"].waiting)
        self.release.set()
        await self.wait_for(lambda: self.daemon.request_task is None)
        self.assertEqual(len(self.daemon.cognition.history), 2)
        self.assertFalse(any("text" in packet for packet in await self.drain()))

    async def test_waiting_check_in_keeps_speech_while_another_task_progresses(self):
        self.use_clock()
        self.slow = True
        self.event("waiting", harness="claude", call="question", tool="AskUserQuestion", progress="Choose a strategy")
        self.event("tool_start", session="two", harness="cursor", call="build", tool="build")
        await self.wait_for(lambda: len(self.daemon.sessions.lanes) == 2)
        self.now += 299
        self.event("progress", session="two", harness="cursor", progress="Building modules")
        await self.wait_for(lambda: self.daemon.sessions.lanes["cursor:two"].progress == "Building modules")
        self.now += 1
        await asyncio.wait_for(self.entered.wait(), 1)
        snapshot = json.loads(self.requests[0]["messages"][-1]["content"])
        self.assertEqual(snapshot["reason"], "check_in")
        self.assertEqual(snapshot["trigger"], {"sessions": ["claude:one"]})
        self.assertEqual(snapshot["activity"], "idle")
        self.assertEqual([lane["id"] for lane in snapshot["sessions"]], ["claude:one"])
        self.assertTrue(snapshot["sessions"][0]["waiting_for_input"])
        self.assertEqual(snapshot["sessions"][0]["waiting_seconds"], 300)
        self.assertEqual(self.daemon.activity(), "working")
        self.now += 1
        self.event("progress", session="two", harness="cursor", progress="Linking build")
        await self.wait_for(lambda: self.daemon.sessions.lanes["cursor:two"].progress == "Linking build")
        self.release.set()
        speech = await self.packet(lambda packet: "text" in packet)
        self.assertEqual(speech["text"], self.speech)
        self.assertEqual(self.daemon.activity(), "working")

    async def test_waiting_check_in_keeps_speech_after_progress_and_duplicate_waiting_events(self):
        self.use_clock()
        self.slow = True
        self.event("waiting", harness="claude", call="question", tool="AskUserQuestion", progress="Choose a strategy")
        await self.wait_for(lambda: "claude:one" in self.daemon.sessions.lanes)
        self.now += 300
        await asyncio.wait_for(self.entered.wait(), 1)
        self.now += 1
        self.event("progress", harness="claude", progress="Strategy still pending")
        await self.wait_for(lambda: self.daemon.sessions.lanes["claude:one"].progress == "Strategy still pending")
        self.now += 1
        self.event("waiting", harness="claude", call="question", tool="AskUserQuestion", progress="Still choosing a strategy")
        await self.wait_for(lambda: self.daemon.sessions.lanes["claude:one"].progress == "Still choosing a strategy")
        self.release.set()
        speech = await self.packet(lambda packet: "text" in packet)
        self.assertEqual(speech["text"], self.speech)
        snapshot = self.daemon.sessions.snapshot()["sessions"][0]
        self.assertTrue(snapshot["waiting_for_input"])
        self.assertEqual(snapshot["waiting_seconds"], 302)

    async def test_check_in_discards_speech_when_the_active_call_finishes(self):
        self.use_clock()
        self.slow = True
        self.event("tool_start", harness="codex", call="checks", tool="pytest")
        await self.wait_for(lambda: self.daemon.sessions.working)
        self.now += 300
        await asyncio.wait_for(self.entered.wait(), 1)
        self.event("tool_done", harness="codex", call="checks", progress="Tests passed")
        await self.wait_for(lambda: not self.daemon.sessions.working)
        self.release.set()
        await self.wait_for(lambda: self.daemon.request_task is None)
        self.assertEqual(len(self.daemon.cognition.history), 2)
        self.assertFalse(any("text" in packet for packet in await self.drain()))

    async def test_check_in_discards_speech_when_a_different_call_starts(self):
        self.use_clock()
        self.slow = True
        self.event("tool_start", harness="codex", call="checks", tool="pytest")
        await self.wait_for(lambda: self.daemon.sessions.working)
        self.now += 300
        await asyncio.wait_for(self.entered.wait(), 1)
        self.event("tool_done", harness="codex", call="checks", progress="Tests passed")
        self.event("tool_start", harness="codex", call="lint", tool="lint")
        await self.wait_for(lambda: "lint" in self.daemon.sessions.lanes["codex:one"].calls)
        self.release.set()
        await self.wait_for(lambda: self.daemon.request_task is None)
        self.assertEqual(len(self.daemon.cognition.history), 2)
        self.assertFalse(any("text" in packet for packet in await self.drain()))

    async def test_file_log_connects_hook_context_model_decision_and_renderer_delivery(self):
        import logging
        from pet_daemon.cognition import INSTRUCTIONS
        from pet_daemon.telemetry import configure

        logger = logging.getLogger("pet-daemon")
        handlers, level, propagate = logger.handlers[:], logger.level, logger.propagate
        path = Path(self.temp.name) / "logs" / "pet.jsonl"
        configure(path)
        try:
            self.event("intent", harness="claude", goal="Fix the quoted parser \"猫\"\ncarefully")
            self.event("tool_start", harness="claude", call="parser", tool="pytest parser")
            await self.packet(lambda packet: packet.get("state") == "working")
            await self.daemon.observe("explicit", "Inspect parser progress")
            speech = await self.packet(lambda packet: "text" in packet)
            entries = [json.loads(line) for line in path.read_text().splitlines()]
            hooks = [entry for entry in entries if entry["event"] == "hook.received"]
            self.assertTrue(any(entry["packet"].get("tool") == "pytest parser" for entry in hooks))
            request = next(entry for entry in entries if entry["event"] == "observation.request")
            self.assertEqual(request["request"]["system"][0]["text"], self.config.persona + "\n\n" + INSTRUCTIONS)
            snapshot = json.loads(request["request"]["messages"][-1]["content"])
            self.assertEqual(snapshot["sessions"][0]["last_tool"], "pytest parser")
            self.assertIn('猫', snapshot["sessions"][0]["goal"])
            response = next(entry for entry in entries if entry["event"] == "observation.response")
            self.assertEqual(response["response"]["usage"]["input_tokens"], 100)
            decision = next(entry for entry in entries if entry["event"] == "observation.decision")
            delivery = next(entry for entry in entries if entry["event"] == "renderer.command" and "text" in entry["packet"])
            self.assertEqual(decision["decision"]["speak"], speech["text"])
            self.assertEqual(delivery["packet"]["text"], speech["text"])
            self.assertEqual({entry["observation_id"] for entry in (request, response, decision, delivery)},
                             {request["observation_id"]})
            self.assertEqual(len({entry["run_id"] for entry in entries}), 1)
            self.assertTrue(all(entry["timestamp"].endswith("+00:00") for entry in entries))
            self.assertNotIn("test-fixture", path.read_text())
        finally:
            for handler in logger.handlers[:]:
                if handler not in handlers:
                    logger.removeHandler(handler)
                    handler.close()
            logger.setLevel(level)
            logger.propagate = propagate

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
        self.assertAlmostEqual(packet["ttl"], 3 + len(self.speech.split()) / 2)

    async def test_explicit_speech_duration_overrides_long_speech_reading_time(self):
        self.speech = "A detailed observation about the ongoing build. " * 100
        self.send({"kind": "speak", "context": "Inspect the build", "ttl": 2.5})
        packet = await self.packet(lambda packet: "text" in packet)
        self.assertEqual(packet["text"], self.speech)
        self.assertEqual(packet["ttl"], 2.5)

    async def test_delivers_speech_when_both_summary_levels_and_observation_use_separate_timeout_budgets(self):
        self.config.summary_interval = 1
        self.config.summary_summary_interval = 1
        for index in range(3):
            await self.daemon.cognition.observe(self.daemon.sessions.snapshot(), f"seed-{index}")
        self.assertEqual(self.daemon.cognition.compaction_requests_due, 2)
        self.config.request_timeout = 0.3
        self.delay = 0.24
        await self.daemon.observe("explicit", "Inspect progress")
        packet = await self.packet(lambda packet: "text" in packet)
        self.assertEqual(packet["text"], self.speech)
        self.assertEqual(len(self.daemon.cognition.history), 4)
        self.assertEqual(len(self.daemon.cognition.summaries), 1)
        self.assertIsNotNone(self.daemon.cognition.summary)

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
            path.write_text(json.dumps({"pet_dir": "pets/other", "observation_seconds": [15, 25], "idle_message_seconds": 450, "summary_interval": 6, "summary_summary_interval": 3, "compaction_max_tokens": 512, "persona": "A calm garden companion.", "log_file": "analysis/pet.jsonl", "memory_file": "state/companion.json",
                                        "behavior": {"repeat_range": [2, 4], "roaming": False}}))
            config = Config.load(path)
            self.assertEqual(config.pet_dir, Path(directory).resolve() / "pets/other")
            self.assertEqual(config.log_file, Path(directory).resolve() / "analysis/pet.jsonl")
            self.assertEqual(config.memory_file, Path(directory).resolve() / "state/companion.json")
            self.assertEqual(config.persona, "A calm garden companion.")
            self.assertEqual(config.observation_seconds, [15, 25])
            self.assertEqual(config.idle_message_seconds, 450)
            self.assertEqual(config.summary_interval, 6)
            self.assertEqual(config.summary_summary_interval, 3)
            self.assertEqual(config.compaction_max_tokens, 512)
            self.assertFalse(config.behavior.roaming)
            self.assertEqual(config.behavior.repeat_range, [2, 4])

    def test_idle_message_interval_defaults_to_five_minutes(self):
        self.assertEqual(Config(pet_dir=Path("fixture")).idle_message_seconds, 300)

    def test_rejects_nonpositive_or_nonfinite_idle_message_intervals(self):
        for seconds in (0, -1, True, "300", None, float("nan"), float("inf")):
            with self.subTest(seconds=seconds), self.assertRaisesRegex(ValueError, "idle_message_seconds"):
                Config(pet_dir=Path("fixture"), idle_message_seconds=seconds).validate()

    def test_rejects_empty_or_nontext_persona(self):
        for persona in ("", "   ", None, 3):
            with self.subTest(persona=persona), self.assertRaises(ValueError):
                Config(pet_dir=Path("fixture"), persona=persona).validate()

    def test_rejects_invalid_compaction_settings(self):
        for settings in ({"summary_interval": 0}, {"summary_interval": True},
                         {"summary_summary_interval": 0}, {"summary_summary_interval": 8.5},
                         {"compaction_max_tokens": 0}, {"compaction_max_tokens": True}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                Config(pet_dir=Path("fixture"), **settings).validate()

    def test_rejects_invalid_intervals_and_shared_socket_before_startup(self):
        for config in (Config(pet_dir=Path("fixture"), observation_seconds=(90, 60)), Config(pet_dir=Path("fixture"), summary_interval=0),
                       Config(pet_dir=Path("fixture"), daemon_socket="/tmp/same", overlay_socket="/tmp/same")):
            with self.subTest(config=config), self.assertRaises(ValueError):
                config.validate()
