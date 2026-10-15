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
from pet_daemon.sprites import Pack


PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII=")


def reply(speak=None, session_context=None, summary=False):
    result = {"session_context": session_context} if summary else {"speak": speak}
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
                             observation_seconds=(3600, 3600), request_timeout=2, memory_file=root / "memory.sqlite3")
        self.config.behavior.automatic_speech_cooldown_seconds = 0
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.requests = []
        self.summary_requests = []
        self.http_status = 200
        self.summary_status = 200
        self.speech = "The tests have reconsidered."
        self.speech_by_reason = {}
        self.session_context = None
        self.slow = False
        self.delay = 0

        async def handler(request):
            body = json.loads(request.content)
            response_context = self.session_context
            summary = body["output_config"]["format"]["schema"]["required"] == ["session_context"]
            requests = self.summary_requests if summary else self.requests
            requests.append(body)
            self.entered.set()
            if self.delay:
                await asyncio.sleep(self.delay)
            if self.slow:
                await self.release.wait()
            status = self.summary_status if summary else self.http_status
            if status != 200:
                return httpx.Response(status, json={"type": "error", "error": {"type": "api_error", "message": "fixture failure"}})
            if summary:
                return reply(session_context=response_context, summary=True)
            reason = json.loads(body["messages"][-1]["content"])["reason"]
            speech = self.speech_by_reason.get(reason, self.speech)
            return reply(speech, session_context=response_context)

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
                   "pointer": {"x": 2000, "y": 2000}, "dragging": False, "menu": False, "speech": False, **extra})

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

    async def test_clear_memory_cancels_commentary_and_starts_fresh_sessions(self):
        self.slow = True
        self.event("intent", goal="Improve parser", repo="/repo/parser")
        await asyncio.wait_for(self.entered.wait(), 1)
        memory = self.daemon.cognition.memory
        memory.record_speech("Remember the parser", "prompt")
        self.send({"kind": "clear_memory"})
        await self.packet(lambda packet: packet.get("text") == "" and packet.get("ttl") == 0)
        self.release.set()
        await self.wait_for(lambda: self.daemon.request_task is None)
        self.assertEqual(self.daemon.sessions.snapshot()["sessions"], [])
        self.assertEqual(memory.recent(), [])
        self.assertEqual(memory.speech(), [])
        self.assertEqual(memory.projects(["/repo/parser"]), [])
        self.assertIsNone(memory.context("cli:one"))
        self.assertIsNone(self.daemon.last_speech)
        self.assertFalse(any(packet.get("text") for packet in await self.drain()))
        self.slow = False
        self.event("intent", goal="Write documentation", repo="/repo/parser")
        await self.packet(lambda packet: packet.get("text") == self.speech)
        lane = self.daemon.sessions.lanes["cli:one"]
        self.assertEqual(lane.first_prompt, "Write documentation")
        self.assertNotIn("Improve parser", json.dumps(memory.recent()))

    def use_clock(self):
        self.now = 1000.0
        clock = lambda: self.now
        self.daemon.clock = clock
        self.daemon.sessions.clock = clock
        self.daemon.cognition.clock = clock
        self.daemon.next_prune = 0
        self.daemon.next_observation = self.now + 3600

    async def test_idle_exercise_speaks_and_click_resets_care(self):
        manifest_path = self.config.pet_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["clips"]["jumping"] = {"frames": ["pixel.png"], "fps": 1, "role": "celebrate"}
        manifest_path.write_text(json.dumps(manifest))
        self.daemon.behavior.pack = Pack(self.config.pet_dir)
        self.use_clock()
        behavior = self.daemon.behavior
        behavior.clock = lambda: self.now
        behavior.config.naps = False
        behavior.config.roaming = False
        behavior.next_care = self.now
        behavior.next_care_speech = self.now
        behavior.execution.interrupt()
        animation = await self.packet(lambda packet: packet.get("animation", {}).get("id") == "jumping")
        self.assertEqual(animation["animation"]["repeats"], 1)
        await self.packet(lambda packet: packet.get("text") == self.speech)
        payload = json.loads(self.requests[-1]["messages"][-1]["content"])
        self.assertEqual(payload["reason"], "care")
        self.assertEqual(payload["trigger"]["care"], "exercise")
        self.send({"kind": "renderer", "event": "click"})
        await self.wait_for(lambda: len(self.requests) == 2)
        self.assertEqual(json.loads(self.requests[-1]["messages"][-1]["content"])["reason"], "click")
        self.assertGreaterEqual(behavior.next_care, self.now + 300)
        self.assertIsNone(behavior.care_remark)

    async def test_new_work_discards_pending_care_speech(self):
        self.slow = True
        task = asyncio.create_task(self.daemon.observe("care", trigger={"care": "consternation"}))
        self.addCleanup(task.cancel)
        await asyncio.wait_for(self.entered.wait(), 1)
        self.event("intent", goal="Review the parser")
        await self.wait_for(lambda: self.daemon.sessions.active)
        self.release.set()
        await task
        self.assertFalse(any(packet.get("text") == self.speech for packet in await self.drain()))

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
        await self.wait_for(lambda: not self.daemon.sessions.lanes["cursor:two"].calls)
        self.assertEqual(self.daemon.activity(), "working")
        self.event("turn_done", session="two", harness="cursor")
        await self.packet(lambda packet: packet.get("state") == "idle")
        self.release.set()
        speech = await self.packet(lambda packet: "text" in packet)
        self.assertEqual(speech["text"], self.speech)
        self.assertEqual(speech["ttl"], 15)
        self.assertEqual(len(self.requests), 1)

    async def test_investigation_and_input_wait_animate_while_commentary_is_pending(self):
        manifest_path = self.config.pet_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for clip, role in (("review", "attention"), ("waiting", "attention"), ("running", "work")):
            manifest["clips"][clip] = {"frames": ["pixel.png"], "fps": 1, "role": role}
        manifest_path.write_text(json.dumps(manifest))
        self.daemon.behavior.pack = Pack(self.config.pet_dir)
        self.use_clock()
        self.daemon.behavior.clock = lambda: self.now
        self.slow = True
        self.event("intent", goal="Inspect the parser")
        await asyncio.wait_for(self.entered.wait(), 1)
        await self.packet(lambda packet: packet.get("animation", {}).get("id") == "review")
        self.now += 8
        self.event("tool_start", call="read", tool="parser.py", tool_name="Read")
        reaction = await self.packet(lambda packet: packet.get("animation", {}).get("id") == "review" and packet["animation"].get("repeats") is not None)
        self.assertGreater(reaction["animation"]["repeats"], 0)
        self.now += 60
        self.send({"kind": "renderer", "event": "animation_done", "command_id": reaction["animation"]["command_id"]})
        self.event("waiting", call="input", tool="AskUserQuestion")
        waiting = await self.packet(lambda packet: packet.get("animation", {}).get("id") == "waiting")
        self.assertGreater(waiting["animation"]["repeats"], 0)
        self.assertFalse(self.release.is_set())
        self.assertIsNotNone(self.daemon.request_task)

    async def test_recovery_celebrates_without_waiting_for_commentary(self):
        manifest_path = self.config.pet_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["clips"]["hop"] = {"frames": ["pixel.png"], "fps": 1, "role": "celebrate"}
        manifest_path.write_text(json.dumps(manifest))
        self.daemon.behavior.pack = Pack(self.config.pet_dir)
        self.slow = True
        self.event("intent", goal="Run checks")
        await asyncio.wait_for(self.entered.wait(), 1)
        self.event("tool_start", call="tests", tool="pytest")
        await self.wait_for(lambda: "tests" in self.daemon.sessions.lanes["cli:one"].calls)
        self.event("tool_error", call="tests")
        await self.wait_for(lambda: self.daemon.sessions.failed)
        self.event("tool_start", call="retry", tool="pytest")
        await self.wait_for(lambda: "retry" in self.daemon.sessions.lanes["cli:one"].calls)
        self.event("tool_done", call="retry", outcome="success")
        await self.packet(lambda packet: packet.get("animation", {}).get("id") == "hop")
        self.assertTrue(self.daemon.sessions.active)
        self.assertFalse(self.release.is_set())
        await self.drain()
        self.event("tool_start", call="cancelled", tool="pytest")
        await self.wait_for(lambda: "cancelled" in self.daemon.sessions.lanes["cli:one"].calls)
        self.event("tool_done", call="cancelled", outcome="cancelled")
        await self.wait_for(lambda: not self.daemon.sessions.lanes["cli:one"].calls)
        packets = await self.drain()
        self.assertFalse(any(packet.get("animation", {}).get("id") == "hop" for packet in packets))

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
        self.assertEqual(set(schema["properties"]), {"speak"})
        self.assertEqual(schema["required"], ["speak"])
        self.assertEqual(schema["properties"]["speak"], {"anyOf": [{"type": "string"}, {"type": "null"}]})

    async def test_napping_pauses_routine_commentary_and_click_wakes_for_speech(self):
        self.use_clock()
        self.config.behavior.nap_interval = (1, 1)
        self.config.behavior.nap_seconds = (10, 10)
        self.daemon.behavior.clock = lambda: self.now
        self.daemon.behavior.inactive_since = self.now
        self.daemon.behavior.execution.interrupt()
        self.daemon.behavior.next_nap = self.now + 1
        self.now += 240
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
        self.daemon.behavior.clock = lambda: self.now
        self.daemon.behavior.inactive_since = self.now
        self.daemon.behavior.execution.interrupt()
        self.daemon.behavior.next_nap = self.now + 1
        self.now += 240
        await self.packet(lambda packet: packet.get("sleeping") is True)
        self.event("intent", harness="codex", goal="Run checks")
        await self.packet(lambda packet: packet.get("sleeping") is False)
        speech = await self.packet(lambda packet: "text" in packet)
        self.assertEqual(speech["text"], self.speech)
        self.assertTrue(self.daemon.sessions.working)
        self.assertFalse(self.daemon.behavior.sleeping)

    async def test_click_requests_fresh_speech_even_after_recent_speech(self):
        self.send({"kind": "speak", "context": "Inspect the build", "ttl": 1})
        original = await self.packet(lambda packet: "text" in packet)
        await self.wait_for(lambda: self.daemon.request_task is None)
        self.speech = "The build has something fresh to say."
        self.send({"kind": "renderer", "event": "click"})
        fresh = await self.packet(lambda packet: "text" in packet)
        self.assertNotEqual(fresh["text"], original["text"])
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
        self.event("tool_start", call="checks", tool="pytest")
        await self.wait_for(lambda: self.daemon.sessions.active)
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

    async def test_automatic_cooldown_allows_one_waiting_reminder_and_manual_speech(self):
        self.use_clock()
        self.config.behavior.automatic_speech_cooldown_seconds = 90
        self.event("intent", goal="Inspect checks")
        await self.packet(lambda packet: "text" in packet)
        await self.wait_for(lambda: self.daemon.request_task is None)
        self.event("turn_done")
        await self.wait_for(lambda: self.daemon.pending is not None)
        self.now += 2
        await self.wait_for(lambda: self.daemon.pending is None)
        self.assertEqual(len(self.requests), 1)
        self.event("intent", goal="Choose the next step")
        await self.wait_for(lambda: self.daemon.pending is None and self.daemon.sessions.active)
        self.event("waiting", call="question")
        await self.packet(lambda packet: "text" in packet)
        await self.wait_for(lambda: self.daemon.request_task is None)
        self.event("waiting", call="question")
        await self.wait_for(lambda: self.daemon.sessions.lanes["cli:one"].waiting)
        self.assertEqual(len(self.requests), 2)
        self.send({"kind": "renderer", "event": "click"})
        await self.packet(lambda packet: "text" in packet)
        self.assertEqual(len(self.requests), 3)

    async def test_each_new_question_can_bypass_the_automatic_cooldown_once(self):
        self.config.behavior.automatic_speech_cooldown_seconds = 90
        self.event("waiting", call="first", progress="Choose a strategy")
        await self.packet(lambda packet: "text" in packet)
        await self.wait_for(lambda: self.daemon.request_task is None)
        self.event("waiting", call="second", progress="Choose a target")
        await self.packet(lambda packet: "text" in packet)
        await self.wait_for(lambda: self.daemon.request_task is None)
        self.assertEqual(len(self.requests), 2)
        self.assertEqual([json.loads(body["messages"][-1]["content"])["trigger"]["progress"] for body in self.requests], ["Choose a strategy", "Choose a target"])
        self.event("waiting", call="second", progress="Still choosing a target")
        await self.wait_for(lambda: self.daemon.sessions.lanes["cli:one"].progress == "Still choosing a target")
        self.assertEqual(len(self.requests), 2)

    async def test_new_question_discards_the_pending_reminder_for_the_first(self):
        self.slow = True
        self.event("waiting", call="first", progress="Choose a strategy")
        await asyncio.wait_for(self.entered.wait(), 1)
        self.event("waiting", call="second", progress="Choose a target")
        await self.wait_for(lambda: self.daemon.sessions.lanes["cli:one"].waiting_call == "second")
        self.release.set()
        await self.packet(lambda packet: "text" in packet)
        await self.wait_for(lambda: self.daemon.request_task is None)
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(len(self.daemon.cognition.memory.speech()), 1)
        self.assertEqual(json.loads(self.requests[-1]["messages"][-1]["content"])["trigger"]["progress"], "Choose a target")

    async def test_waiting_reminder_survives_background_call_completion_in_the_same_session(self):
        self.slow = True
        self.event("tool_start", call="background", tool="Read")
        self.event("waiting", call="question", progress="Choose a strategy")
        await asyncio.wait_for(self.entered.wait(), 1)
        self.event("tool_done", call="background")
        await self.wait_for(lambda: not self.daemon.sessions.lanes["cli:one"].calls)
        self.assertTrue(self.daemon.sessions.lanes["cli:one"].waiting)
        self.release.set()
        self.assertEqual((await self.packet(lambda packet: "text" in packet))["text"], self.speech)
        await self.wait_for(lambda: self.daemon.request_task is None)
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(len(self.daemon.cognition.memory.speech()), 1)

    async def test_manual_speech_delays_automatic_outcome_commentary(self):
        self.use_clock()
        self.config.behavior.automatic_speech_cooldown_seconds = 90
        self.event("tool_start", call="checks", tool="pytest")
        await self.wait_for(lambda: self.daemon.sessions.active)
        self.send({"kind": "renderer", "event": "click"})
        await self.packet(lambda packet: "text" in packet)
        await self.wait_for(lambda: self.daemon.request_task is None)
        self.event("turn_done")
        await self.wait_for(lambda: self.daemon.pending is not None)
        self.now += 2
        await self.wait_for(lambda: self.daemon.pending is None)
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(json.loads(self.requests[0]["messages"][-1]["content"])["reason"], "click")

    async def test_click_preempts_pending_automatic_commentary(self):
        self.slow = True
        self.speech_by_reason["prompt"] = "Superseded acknowledgment."
        self.event("intent", goal="Inspect checks")
        await asyncio.wait_for(self.entered.wait(), 1)
        self.send({"kind": "renderer", "event": "click"})
        await self.wait_for(lambda: len(self.requests) == 2)
        self.release.set()
        speech = await self.packet(lambda packet: "text" in packet)
        self.assertEqual(speech["text"], self.speech)
        self.assertEqual([json.loads(body["messages"][-1]["content"])["reason"] for body in self.requests], ["prompt", "click"])
        self.assertNotIn("Superseded acknowledgment.", json.dumps(self.daemon.cognition.memory.speech()))

    async def test_hooks_persist_without_requesting_commentary(self):
        self.event("intent", goal="Repair the parser", cwd="/repo/parser")
        await self.wait_for(lambda: bool(self.daemon.cognition.memory.recent()))
        self.assertIn("Repair the parser", json.dumps(self.daemon.cognition.memory.recent()))
        self.assertEqual(self.requests, [])

    async def test_visible_speech_persists_after_renderer_delivery(self):
        await self.daemon.observe("explicit", "Inspect the build")
        await self.packet(lambda packet: "text" in packet)
        self.assertIn(self.speech, json.dumps(self.daemon.cognition.memory.speech()))

    async def test_zero_duration_speech_is_absent_from_memory(self):
        await self.daemon.observe("explicit", "Inspect the build", ttl=0)
        await self.packet(lambda packet: "text" in packet)
        self.assertEqual(self.daemon.cognition.memory.speech(), [])

    async def test_computer_sleep_cancels_commentary_and_click_recovers_after_wake(self):
        self.slow = True
        self.send({"kind": "speak", "context": "Inspect the build"})
        await asyncio.wait_for(self.entered.wait(), 1)
        self.send({"kind": "renderer", "event": "system_sleep"})
        await self.wait_for(lambda: self.daemon.system_sleeping and self.daemon.request_task is None)
        self.release.set()
        self.assertEqual(self.daemon.cognition.memory.speech(), [])
        self.assertFalse(any("text" in packet for packet in await self.drain()))
        self.send_status(system_sleeping=False, x=102)
        await self.wait_for(lambda: not self.daemon.system_sleeping)
        self.send({"kind": "renderer", "event": "click"})
        packet = await self.packet(lambda packet: "text" in packet)
        self.assertEqual(packet["text"], self.speech)

    async def test_summary_updates_on_each_prompt_and_skips_completion_by_default(self):
        self.session_context = "Keep the parser API compatible."
        self.event("intent", goal="Refactor the parser")
        await self.packet(lambda packet: packet.get("text") == self.speech)
        await self.wait_for(lambda: self.daemon.request_task is None)
        self.event("intent", goal="Investigate the parser timeout")
        await self.packet(lambda packet: packet.get("text") == self.speech)
        await self.wait_for(lambda: self.daemon.request_task is None)
        self.assertEqual(len(self.summary_requests), 2)
        payload = json.loads(self.summary_requests[-1]["messages"][0]["content"])
        self.assertEqual(payload["recent_requests"][-1], "Investigate the parser timeout")
        self.assertEqual(payload["session_context"], self.session_context)
        await self.daemon.observe("completion")
        self.assertEqual(len(self.summary_requests), 2)
        self.assertEqual(len(self.requests), 3)

    async def test_first_request_summary_overrides_triggers_and_survives_a_failed_attempt_and_restart(self):
        self.config.summarize_first_request_only = True
        self.config.summarize_on = []
        self.summary_status = 500
        initial = "Refactor the parser while keeping the public interface unchanged. " * 8
        self.event("intent", goal=initial)
        await self.packet(lambda packet: packet.get("text") == self.speech)
        await self.wait_for(lambda: self.daemon.request_task is None)
        self.assertEqual(len(self.summary_requests), 1)
        payload = json.loads(self.summary_requests[0]["messages"][0]["content"])
        self.assertEqual(payload["first_prompt"], initial)
        self.assertEqual(payload["recent_requests"], [initial])
        self.assertNotIn("progress", payload)
        self.assertEqual(payload["session_context"], "")
        saved = self.daemon.cognition.memory.context("cli:one")
        self.assertTrue(saved["first_summary_attempted"])
        self.assertEqual(saved["first_prompt"], initial)
        await self.daemon.observe("prompt")
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
        await self.wait_for(lambda: self.daemon.request_task is None)
        lane = self.daemon.sessions.lanes["cli:one"]
        self.assertEqual(lane.first_prompt, initial)
        self.assertEqual(lane.request_count, 2)
        for reason in ("completion", "explicit"):
            await self.daemon.observe(reason, "Parser work")
        self.assertEqual(len(self.summary_requests), 1)
        await self.drain()
        self.event("intent", session="another", goal="Review hooks")
        await self.packet(lambda packet: packet.get("text") == self.speech)
        await self.wait_for(lambda: self.daemon.request_task is None)
        self.assertEqual(len(self.summary_requests), 2)
        payload = json.loads(self.summary_requests[-1]["messages"][0]["content"])
        self.assertEqual(payload["first_prompt"], "Review hooks")

    async def test_summary_update_persists_without_speech_and_reaches_the_next_prompt(self):
        self.use_clock()
        self.speech = None
        self.session_context = "Parser refactor awaits validation."
        self.event("intent", goal="Refactor the parser", turn="first")
        await self.wait_for(lambda: self.daemon.cognition.memory.context("cli:one") is not None and self.daemon.cognition.memory.context("cli:one")["session_context"] == self.session_context)
        await self.wait_for(lambda: self.daemon.request_task is None)
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(self.daemon.cognition.memory.speech(), [])
        self.now += 20
        self.daemon.next_observation = self.now
        await self.wait_for(lambda: self.daemon.next_observation > self.now)
        self.assertEqual(len(self.requests), 1)
        self.session_context = None
        self.speech = "The parser still needs validation."
        await self.daemon.observe("explicit", "Check the parser")
        await self.packet(lambda packet: "text" in packet)
        session = json.loads(self.requests[-1]["messages"][0]["content"])["focus_session"]
        self.assertEqual(session["first_prompt"], "Refactor the parser")
        self.assertEqual(session["session_context"], "Parser refactor awaits validation.")
        self.assertEqual(len(self.requests), 2)

    async def test_first_request_summary_restores_bookkeeping_from_a_session_alias(self):
        self.config.summarize_first_request_only = True
        self.daemon.cognition.memory.save_context("cli:old-id", "Review hooks", "Hook review", ["Review hooks"], 1, True)
        self.event("intent", session="new-id", aliases=["old-id"], goal="Check hook edge cases")
        await self.packet(lambda packet: packet.get("text") == self.speech)
        await self.wait_for(lambda: self.daemon.request_task is None)
        self.assertEqual(self.summary_requests, [])
        lane = self.daemon.sessions.lanes["cli:new-id"]
        self.assertEqual(lane.first_prompt, "Review hooks")
        self.assertEqual(lane.request_count, 2)
        self.assertTrue(lane.first_summary_attempted)

    async def test_first_request_summary_uses_the_initial_prompt_when_followups_are_queued_together(self):
        self.config.summarize_first_request_only = True
        for goal in ("Refactor the parser", "Keep its API unchanged"):
            self.daemon.receive(json.dumps({"kind": "event", "event": "intent", "harness": "cli",
                                           "session": "one", "goal": goal}).encode())
        await self.packet(lambda packet: packet.get("text") == self.speech)
        await self.wait_for(lambda: self.daemon.request_task is None)
        self.assertEqual(len(self.summary_requests), 1)
        payload = json.loads(self.summary_requests[0]["messages"][0]["content"])
        self.assertEqual(payload["first_prompt"], "Refactor the parser")
        self.assertEqual(payload["recent_requests"], ["Refactor the parser"])
        self.assertNotIn("Keep its API unchanged", json.dumps(payload))

    async def test_subagent_inherited_context_does_not_consume_its_first_request_summary(self):
        self.config.summarize_first_request_only = True
        self.event("intent", goal="Refactor the parser")
        await self.packet(lambda packet: packet.get("text") == self.speech)
        await self.wait_for(lambda: self.daemon.request_task is None)
        self.event("subagent_start", session="child", parent="one")
        await self.packet(lambda packet: packet.get("text") == self.speech)
        await self.wait_for(lambda: self.daemon.request_task is None)
        self.assertEqual(len(self.summary_requests), 1)
        self.event("intent", session="child", goal="Inspect parser edge cases")
        await self.packet(lambda packet: packet.get("text") == self.speech)
        await self.wait_for(lambda: self.daemon.request_task is None)
        self.assertEqual(len(self.summary_requests), 2)
        payload = json.loads(self.summary_requests[-1]["messages"][0]["content"])
        self.assertEqual(payload["first_prompt"], "Inspect parser edge cases")
        self.assertNotIn("Refactor the parser", json.dumps(payload))

    async def test_interleaved_sessions_keep_separate_summaries_and_shared_commentary(self):
        self.session_context = "Keep the parser API stable."
        self.event("intent", session="parser", goal="Refactor the parser without changing its API")
        await self.packet(lambda packet: "text" in packet)
        await self.wait_for(lambda: self.daemon.request_task is None)
        self.session_context = "Remove redundant Python hook handling."
        self.event("intent", session="hooks", goal="Review Python hooks for redundant handling")
        await self.packet(lambda packet: "text" in packet)
        await self.wait_for(lambda: self.daemon.request_task is None)
        commentary = json.loads(self.requests[-1]["messages"][0]["content"])
        self.assertEqual(commentary["focus_session"]["id"], "cli:hooks")
        self.assertEqual(commentary["other_sessions"][0]["id"], "cli:parser")
        self.assertTrue(commentary["other_sessions"][0]["turn_active"])
        self.assertIn("parser", commentary["other_sessions"][0]["goal"])
        for request, excluded in zip(self.summary_requests, ("Python hooks", "parser")):
            self.assertNotIn(excluded, request["messages"][0]["content"])
        memory = self.daemon.cognition.memory
        self.assertEqual(memory.context("cli:parser")["session_context"], "Keep the parser API stable.")
        self.assertEqual(memory.context("cli:hooks")["session_context"], "Remove redundant Python hook handling.")
        self.event("turn_done", session="hooks", progress="Hook cleanup passed checks")
        await self.packet(lambda packet: "text" in packet, timeout=2)
        await self.wait_for(lambda: self.daemon.request_task is None)
        commentary = json.loads(self.requests[-1]["messages"][0]["content"])
        self.assertEqual(commentary["focus_session"]["id"], "cli:hooks")
        self.assertFalse(commentary["convergence"])
        self.assertTrue(commentary["other_sessions"][0]["turn_active"])
        summary_count = len(self.summary_requests)
        await self.daemon.observe("routine")
        self.assertEqual(len(self.summary_requests), summary_count)

    async def test_summary_failure_preserves_commentary_and_saved_context(self):
        self.session_context = "Keep the parser API stable."
        self.event("intent", goal="Refactor the parser")
        await self.packet(lambda packet: "text" in packet)
        await self.wait_for(lambda: self.daemon.request_task is None)
        self.summary_status = 500
        self.event("intent", goal="Validate parser behavior")
        await self.packet(lambda packet: packet.get("text") == self.speech)
        await self.wait_for(lambda: self.daemon.request_task is None)
        self.assertEqual(self.daemon.cognition.memory.context("cli:one")["session_context"], "Keep the parser API stable.")

    async def test_followup_with_the_same_native_turn_reaches_the_next_prompt(self):
        self.event("intent", harness="codex", goal="Compress the workflow", turn="native-turn")
        await self.packet(lambda packet: "text" in packet)
        await self.wait_for(lambda: self.daemon.request_task is None)
        self.event("intent", harness="codex", goal="Explain the counter limits", turn="native-turn")
        await self.packet(lambda packet: "text" in packet)
        self.assertEqual(len(self.requests), 2)
        payload = json.loads(self.requests[-1]["messages"][0]["content"])
        self.assertEqual(payload["reason"], "prompt")
        self.assertEqual(payload["focus_session"]["first_prompt"], "Compress the workflow")
        self.assertEqual(payload["focus_session"]["recent_requests"], ["Compress the workflow", "Explain the counter limits"])
        self.assertEqual(payload["trigger"]["goal"], "Explain the counter limits")

    async def test_followup_during_observation_refuses_stale_summary(self):
        self.daemon.cognition.memory.save_context("cli:one", "Refactor the parser", "Keep its API stable", ["Refactor the parser"])
        self.slow = True
        self.session_context = "Stale parser summary."
        self.event("intent", goal="Refactor the parser", turn="first")
        await asyncio.wait_for(self.entered.wait(), 1)
        self.event("intent", goal="Keep the parser API unchanged", turn="first", manual=True)
        await self.wait_for(lambda: "Keep the parser API unchanged" in self.daemon.sessions.lanes["cli:one"].recent_requests)
        self.session_context = None
        self.release.set()
        await self.wait_for(lambda: self.daemon.request_task is None and len(self.requests) >= 2)
        lane = self.daemon.sessions.lanes["cli:one"]
        self.assertNotEqual(lane.session_context, "Stale parser summary.")
        self.assertEqual(lane.first_prompt, "Refactor the parser")
        self.assertIn("Keep the parser API unchanged", lane.recent_requests)
        self.assertEqual(len(self.daemon.cognition.memory.speech()), 1)

    async def test_explicit_speech_cannot_overwrite_context_after_a_followup(self):
        self.event("intent", goal="Refactor the parser", turn="first")
        await self.packet(lambda packet: "text" in packet)
        await self.wait_for(lambda: self.daemon.request_task is None)
        self.entered.clear()
        self.slow = True
        self.session_context = "Stale explicit summary."
        self.send({"kind": "speak", "context": "Comment on the parser"})
        await asyncio.wait_for(self.entered.wait(), 1)
        self.event("intent", goal="Keep the parser API unchanged", turn="first")
        await self.wait_for(lambda: "Keep the parser API unchanged" in self.daemon.sessions.lanes["cli:one"].recent_requests)
        self.session_context = None
        self.release.set()
        await self.wait_for(lambda: self.daemon.request_task is None and len(self.requests) >= 3)
        self.assertNotEqual(self.daemon.sessions.lanes["cli:one"].session_context, "Stale explicit summary.")
        saved = self.daemon.cognition.memory.context("cli:one")
        self.assertNotEqual(saved["session_context"], "Stale explicit summary.")

    async def test_restart_restores_context_when_hooks_arrive_without_restoring_live_calls(self):
        self.session_context = "Parser checks are pending."
        self.event("intent", goal="Refactor the parser", turn="first")
        await self.packet(lambda packet: "text" in packet)
        self.event("tool_start", call="checks", tool="pytest")
        await self.wait_for(lambda: self.daemon.sessions.working)
        await self.daemon.close()
        self.session_context = None
        self.daemon = Daemon(self.config, self.client, rng=random.Random(7))
        self.assertFalse(self.daemon.sessions.active)
        self.assertFalse(self.daemon.sessions.working)
        await self.daemon.start()
        self.send_status()
        await self.packet(lambda packet: packet.get("state") == "idle")
        self.event("intent", goal="Continue parser validation", turn="second")
        await self.packet(lambda packet: "text" in packet)
        lane = self.daemon.sessions.lanes["cli:one"]
        self.assertEqual(lane.first_prompt, "Refactor the parser")
        self.assertEqual(lane.session_context, "Parser checks are pending.")
        self.assertIn("Continue parser validation", lane.recent_requests)
        self.assertFalse(lane.calls)

    async def test_routine_observations_require_a_change_in_work(self):
        self.use_clock()
        self.event("tool_start", call="checks", tool="pytest")
        await self.wait_for(lambda: self.daemon.sessions.working)
        self.daemon.next_observation = self.now
        await self.packet(lambda packet: "text" in packet)
        await self.wait_for(lambda: self.daemon.request_task is None)
        self.now += 20
        self.daemon.next_observation = self.now
        self.send_status(x=101)
        await self.wait_for(lambda: self.daemon.behavior.status["x"] == 101)
        self.assertEqual(len(self.requests), 1)
        self.event("progress", progress="Parser checks passed")
        await self.wait_for(lambda: self.daemon.sessions.lanes["cli:one"].progress == "Parser checks passed")
        self.daemon.next_observation = self.now
        await self.packet(lambda packet: "text" in packet)
        self.assertEqual(len(self.requests), 2)

    async def test_repeated_failures_persist_and_request_speech_at_the_third_failure_once(self):
        self.use_clock()
        for index in range(5):
            self.event("tool_start", call=f"check-{index}", tool="pytest")
            await self.wait_for(lambda: "cli:one" in self.daemon.sessions.lanes and f"check-{index}" in self.daemon.sessions.lanes["cli:one"].calls)
            self.event("tool_error", call=f"check-{index}", tool="pytest", detail="Parser assertion failed", error_signature="parser:error")
            await self.wait_for(lambda: self.daemon.sessions.lanes["cli:one"].failure is not None and self.daemon.sessions.lanes["cli:one"].failure["count"] == index + 1)
            if index == 2:
                self.now += 2
                await self.packet(lambda packet: "text" in packet)
                await self.wait_for(lambda: self.daemon.request_task is None)
        self.assertEqual(len(self.requests), 1)
        events = self.daemon.cognition.memory.recent(limit=10)
        failures = [event for event in events if event["event_type"] == "tool_error"]
        self.assertEqual(len(failures), 5)
        self.assertTrue(all("error_signature" not in event for event in failures))
        self.assertNotIn("Parser assertion failed", json.dumps(failures))

    async def test_recovered_session_skips_queued_failure_when_another_session_is_still_failing(self):
        self.use_clock()
        for session in ("one", "two"):
            self.event("tool_start", session=session, call="checks", tool="pytest")
            await self.wait_for(lambda: f"cli:{session}" in self.daemon.sessions.lanes and "checks" in self.daemon.sessions.lanes[f"cli:{session}"].calls)
            self.event("tool_error", session=session, call="checks")
            for index in range(2):
                call = f"failed-{index}"
                self.event("tool_start", session=session, call=call, tool="pytest")
                await self.wait_for(lambda: call in self.daemon.sessions.lanes[f"cli:{session}"].calls)
                self.event("tool_error", session=session, call=call)
                await self.wait_for(lambda: self.daemon.sessions.lanes[f"cli:{session}"].failure["count"] == index + 2)
            await self.wait_for(lambda: self.daemon.pending and self.daemon.pending[2]["target_session"] == f"cli:{session}")
        self.event("tool_start", session="two", call="retry", tool="pytest")
        await self.wait_for(lambda: "retry" in self.daemon.sessions.lanes["cli:two"].calls)
        self.event("tool_done", session="two", call="retry")
        await self.wait_for(lambda: self.daemon.sessions.lanes["cli:two"].failure["resolved"])
        self.now += 2
        await self.wait_for(lambda: self.daemon.pending is None)
        self.assertTrue(self.daemon.sessions.failed)
        self.assertFalse(any(json.loads(body["messages"][-1]["content"])["reason"] == "failure" for body in self.requests))

    async def test_parent_routed_child_failure_is_persisted_under_the_child_session(self):
        self.use_clock()
        self.event("subagent_start", session="child", harness="claude", parent="main", parent_call="delegate", goal="Inspect parser", repo="/repo/parser")
        await self.packet(lambda packet: "text" in packet)
        await self.wait_for(lambda: self.daemon.request_task is None)
        self.event("tool_start", session="main", harness="claude", parent_call="delegate", call="check", tool="pytest")
        await self.wait_for(lambda: "check" in self.daemon.sessions.lanes["claude:child"].calls)
        self.event("tool_error", session="main", harness="claude", parent_call="delegate", call="check", detail="Parser assertion failed", error_signature="parser:error")
        await self.wait_for(lambda: self.daemon.sessions.lanes["claude:child"].failure is not None)
        self.event("turn_error", session="child", harness="claude")
        await self.wait_for(lambda: self.daemon.sessions.lanes["claude:child"].finished)
        self.now += 2
        await self.packet(lambda packet: "text" in packet)
        self.assertEqual(json.loads(self.requests[-1]["messages"][0]["content"])["reason"], "failure")
        failures = [row for row in self.daemon.cognition.memory.recent(limit=10) if row["event_type"] == "tool_error"]
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["session"], "claude:child")
        self.assertEqual(failures[0]["repo"], "/repo/parser")

    async def test_failure_request_and_memory_exclude_injected_diagnostics(self):
        self.use_clock()
        self.event("tool_start", call="check", tool="pytest")
        await self.wait_for(lambda: self.daemon.sessions.working)
        self.event("tool_error", call="check", detail="SECRET PRIVATE STACK", error_signature="SECRET PRIVATE HASH", progress="SECRET FAILED COMMAND OUTPUT")
        await self.wait_for(lambda: self.daemon.sessions.failed)
        await self.daemon.observe("failure")
        await self.packet(lambda packet: "text" in packet)
        body = self.requests[-1]
        self.assertNotIn("SECRET", json.dumps(body))
        self.assertNotIn("SECRET", json.dumps(self.daemon.sessions.snapshot()))
        self.assertNotIn("SECRET", json.dumps(self.daemon.cognition.memory.recent()))
        failure = json.loads(body["messages"][0]["content"])["focus_session"]["failure"]
        self.assertEqual(failure["activity"], "testing")
        self.assertEqual(failure["count"], 1)

    async def test_failed_renderer_delivery_is_absent_from_memory(self):
        Path(self.config.overlay_socket).unlink()
        await self.daemon.observe("explicit", "Inspect the build")
        self.assertEqual(self.daemon.cognition.memory.speech(), [])
        self.assertIsNone(self.daemon.last_speech)

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
        self.assertEqual(snapshot["focus_session"]["id"], "codex:one")
        sessions = {snapshot["focus_session"]["id"]: snapshot["focus_session"]}
        self.assertEqual(set(sessions), {"codex:one"})
        self.assertEqual(snapshot["other_sessions"][0]["id"], "cursor:two")
        self.assertTrue(snapshot["other_sessions"][0]["turn_active"])
        self.assertEqual(sessions["codex:one"]["quiet_seconds"], 300)
        self.now += 299
        self.event("progress", session="two", harness="cursor", progress="Build continuing")
        await self.wait_for(lambda: self.daemon.next_prune > self.now)
        self.assertEqual(len(self.requests), 1)
        self.now += 1
        await self.packet(lambda packet: "text" in packet)
        snapshot = json.loads(self.requests[1]["messages"][-1]["content"])
        self.assertEqual(snapshot["focus_session"]["id"], "codex:one")

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
        self.assertEqual(snapshot["focus_session"]["id"], "claude:one")
        self.assertEqual(snapshot["focus_session"]["waiting_seconds"], 300)
        self.assertEqual(snapshot["focus_session"]["quiet_seconds"], 1)
        self.event("tool_done", harness="claude", call="question", progress="Strategy chosen")
        await self.wait_for(lambda: not self.daemon.sessions.lanes["claude:one"].waiting)
        self.release.set()
        await self.wait_for(lambda: self.daemon.request_task is None)
        self.assertEqual(self.daemon.cognition.memory.speech(), [])
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
        self.assertEqual(snapshot["focus_session"]["id"], "claude:one")
        self.assertEqual(snapshot["activity"], "working")
        self.assertEqual(snapshot["focus_session"]["id"], "claude:one")
        self.assertTrue(snapshot["focus_session"]["waiting_for_input"])
        self.assertEqual(snapshot["focus_session"]["waiting_seconds"], 300)
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
        await self.wait_for(lambda: not self.daemon.sessions.lanes["codex:one"].calls)
        self.assertEqual(self.daemon.activity(), "working")
        self.release.set()
        await self.wait_for(lambda: self.daemon.request_task is None)
        self.assertEqual(self.daemon.cognition.memory.speech(), [])
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
        self.assertEqual(self.daemon.cognition.memory.speech(), [])
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
            self.assertEqual(request["request"]["system"][0]["text"], "<persona>\n" + self.config.persona + "\n</persona>\n\n" + INSTRUCTIONS)
            snapshot = json.loads(request["request"]["messages"][-1]["content"])
            self.assertEqual(snapshot["focus_session"]["last_tool"], "pytest parser")
            self.assertIn('猫', snapshot["focus_session"]["first_prompt"])
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

    async def test_recovery_discards_stale_automatic_speech_without_recording_it(self):
        self.slow = True
        self.event("intent", harness="claude", goal="Repair tests")
        self.event("tool_start", harness="claude", call="bad", tool="pytest")
        await self.packet(lambda packet: packet.get("state") == "working")
        self.event("tool_error", harness="claude", call="bad", detail="AssertionError")
        await self.wait_for(lambda: self.daemon.sessions.failed)
        self.daemon.pending = None
        task = asyncio.create_task(self.daemon.observe("failure"))
        await asyncio.wait_for(self.entered.wait(), 1)
        self.event("tool_start", harness="claude", call="retry", tool="pytest")
        self.event("tool_done", harness="claude", call="retry")
        await self.wait_for(lambda: not self.daemon.sessions.failed)
        self.release.set()
        await task
        self.assertEqual(self.daemon.cognition.memory.speech(), [])
        self.assertFalse(any("text" in packet for packet in await self.drain()))

    async def test_trouble_speech_stays_discarded_after_recovery_and_another_failure(self):
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
        await self.wait_for(lambda: self.daemon.sessions.lanes["cli:one"].failure["count"] == 1)
        self.release.set()
        await self.wait_for(lambda: self.daemon.request_task is None)
        self.assertEqual(self.daemon.cognition.memory.speech(), [])
        self.assertFalse(any("text" in packet for packet in await self.drain()))

    async def test_recovery_speech_stays_discarded_after_another_failure_and_recovery(self):
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
        await self.wait_for(lambda: self.daemon.sessions.lanes["cli:one"].last_tool == "pytest"
                            and not self.daemon.sessions.lanes["cli:one"].calls)
        await self.wait_for(lambda: self.daemon.episodes.serial >= 5)
        self.release.set()
        await self.wait_for(lambda: self.daemon.request_task is None)
        self.assertEqual(self.daemon.cognition.memory.speech(), [])
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

    async def test_wrapper_failure_records_command_activity_without_diagnostics(self):
        root = Path(__file__).resolve().parents[1]
        env = dict(os.environ, PET_DAEMON_SOCKET=self.config.daemon_socket)
        child = await asyncio.create_subprocess_exec(str(root / "examples/pet-run"), "/bin/sh", "-c", "exit 7", env=env)
        self.assertEqual(await child.wait(), 7)
        await self.wait_for(lambda: bool(self.daemon.sessions.lanes) and all(lane.finished for lane in self.daemon.sessions.lanes.values()))
        lane = next(iter(self.daemon.sessions.lanes.values()))
        self.assertEqual(lane.outcome, "error")
        self.assertEqual(lane.failure["activity"], "command")
        self.assertNotIn("detail", lane.failure)
        await self.daemon.observe("failure")
        snapshot = json.loads(self.requests[-1]["messages"][-1]["content"])
        self.assertEqual(snapshot["focus_session"]["failure"]["activity"], "command")
        self.assertNotIn("Exit code", json.dumps(snapshot))

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
    def test_summary_defaults_run_on_prompts(self):
        config = Config(pet_dir=Path("fixture"))
        self.assertEqual(config.summarize_on, ["prompt"])
        self.assertFalse(config.summarize_first_request_only)

    def test_summary_reasons_accept_every_commentary_trigger_and_an_empty_array(self):
        for reasons in ([], ["prompt", "completion", "routine", "failure", "check_in", "click", "explicit"]):
            with self.subTest(reasons=reasons):
                Config(pet_dir=Path("fixture"), summarize_on=reasons).validate()

    def test_summary_configuration_rejects_unknown_reasons_and_nonboolean_flags(self):
        for reasons in ("prompt", None, ["intent"], [None], [[]]):
            with self.subTest(reasons=reasons), self.assertRaisesRegex(ValueError, "summarize_on"):
                Config(pet_dir=Path("fixture"), summarize_on=reasons).validate()
        for value in (1, 0, "true", None):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "summarize_first_request_only"):
                Config(pet_dir=Path("fixture"), summarize_first_request_only=value).validate()

    def test_loads_startup_paths_relative_to_config_and_configurable_timing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pet.json"
            path.write_text(json.dumps({"pet_dir": "pets/other", "observation_seconds": [15, 25], "idle_message_seconds": 450, "prompt_max_bytes": 12000, "memory_retention_days": 14, "memory_max_events": 500, "persona": "A calm garden companion.", "log_file": "analysis/pet.jsonl", "memory_file": "state/companion.sqlite3",
                                        "behavior": {"repeat_range": [2, 4], "roaming": False}}))
            config = Config.load(path)
            self.assertEqual(config.pet_dir, Path(directory).resolve() / "pets/other")
            self.assertEqual(config.log_file, Path(directory).resolve() / "analysis/pet.jsonl")
            self.assertEqual(config.memory_file, Path(directory).resolve() / "state/companion.sqlite3")
            self.assertEqual(config.persona, "A calm garden companion.")
            self.assertEqual(config.observation_seconds, [15, 25])
            self.assertEqual(config.idle_message_seconds, 450)
            self.assertEqual(config.prompt_max_bytes, 12000)
            self.assertEqual(config.memory_retention_days, 14)
            self.assertEqual(config.memory_max_events, 500)
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

    def test_rejects_invalid_memory_limits(self):
        for field in ("prompt_max_bytes", "memory_retention_days", "memory_max_events"):
            for value in (0, -1, True):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    Config(pet_dir=Path("fixture"), **{field: value}).validate()

    def test_memory_defaults_bound_requests_and_retention(self):
        config = Config(pet_dir=Path("fixture"))
        self.assertEqual(config.prompt_max_bytes, 16000)
        self.assertEqual(config.memory_retention_days, 30)
        self.assertEqual(config.memory_max_events, 10000)
        self.assertEqual(config.max_tokens, 512)

    def test_rejects_invalid_intervals_and_shared_socket_before_startup(self):
        for config in (Config(pet_dir=Path("fixture"), observation_seconds=(90, 60)),
                       Config(pet_dir=Path("fixture"), daemon_socket="/tmp/same", overlay_socket="/tmp/same")):
            with self.subTest(config=config), self.assertRaises(ValueError):
                config.validate()
