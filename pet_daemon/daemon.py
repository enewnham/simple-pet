import asyncio
from dataclasses import asdict
import errno
import json
import logging
import math
import os
import random
import signal
import socket
import sqlite3
import stat
import time

from .behavior import Behavior
from .cognition import Cognition
from .memory import bounded
from .sessions import Sessions
from .sprites import Pack
from .telemetry import observation, trace


log = logging.getLogger("pet-daemon")
MAX_INPUT_BYTES = 2000
SPEECH_REPLAY_SECONDS = 60


def bind_socket(path):
    try:
        existing = os.lstat(path)
    except FileNotFoundError:
        existing = None
    if existing:
        if not stat.S_ISSOCK(existing.st_mode) or existing.st_uid != os.getuid():
            raise ValueError("Refusing to replace a non-socket or another user's socket")
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as probe:
            try:
                probe.connect(path)
            except OSError as error:
                if error.errno not in (errno.ECONNREFUSED, errno.ENOENT):
                    raise
            else:
                raise ValueError(f"A daemon is already listening at {path}")
        os.unlink(path)
    receiver = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    try:
        receiver.setblocking(False)
        receiver.bind(path)
        os.chmod(path, 0o600)
        identity = os.lstat(path)
        return receiver, (identity.st_dev, identity.st_ino)
    except BaseException:
        receiver.close()
        raise


def unlink_owned(path, identity):
    try:
        current = os.lstat(path)
        if (current.st_dev, current.st_ino) == identity:
            os.unlink(path)
    except FileNotFoundError:
        pass


class Receiver(asyncio.DatagramProtocol):
    def __init__(self, daemon):
        self.daemon = daemon

    def datagram_received(self, data, address):
        self.daemon.receive(data)

    def error_received(self, error):
        log.warning("Hook socket: %s", error)


class Daemon:
    def __init__(self, config, client, clock=time.monotonic, rng=None):
        self.config = config
        self.clock = clock
        self.rng = rng or random.Random()
        self.pack = Pack(config.pet_dir)
        self.sessions = Sessions(clock, config.stale_seconds, config.completed_retention_seconds)
        self.behavior = Behavior(self.pack, config.behavior, clock, self.rng, self.emit)
        self.cognition = Cognition(client, config, self.pack)
        self.transport = None
        self.identity = None
        self.sender = None
        self.child = None
        self.loop_task = None
        self.request_task = None
        self.system_sleeping = False
        self.last_reaction_state = None
        self.pending = None
        self.manual = None
        self.clicked = False
        self.last_speech = None
        self.error_until = 0
        self.next_prune = 0
        self.next_observation = clock() + self.rng.uniform(*config.observation_seconds)
        self.stopping = asyncio.Event()

    async def start(self):
        receiver, identity = bind_socket(self.config.daemon_socket)
        self.identity = identity
        try:
            self.transport, _ = await asyncio.get_running_loop().create_datagram_endpoint(lambda: Receiver(self), sock=receiver)
        except BaseException:
            receiver.close()
            unlink_owned(self.config.daemon_socket, identity)
            self.identity = None
            raise
        self.sender = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.sender.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 131072)
        self.sender.setblocking(False)
        self.loop_task = asyncio.create_task(self.loop())
        trace("daemon.started", config=asdict(self.config), pet=self.pack.pet,
              clips={key: asdict(clip) for key, clip in self.pack.clips.items()}, next_observation=self.next_observation)
        log.info("listening at %s; pet=%s model=%s", self.config.daemon_socket, self.pack.pet, self.config.model)

    def emit(self, packet):
        if self.sender is None:
            return False
        try:
            data = json.dumps(packet, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
            self.sender.sendto(data, self.config.overlay_socket)
            trace("renderer.command", packet=packet)
            return True
        except (OSError, ValueError) as error:
            trace("renderer.delivery_failed", packet=packet, error=str(error))
            log.warning("Renderer delivery failed: %s", error)
            return False

    def queue(self, reason, trigger=None):
        now = self.clock()
        if reason == "failure":
            self.pending = (reason, now + 1, trigger)
        elif not self.pending or self.pending[0] != "failure" or not self.sessions.failed:
            delay = {"prompt": 0, "completion": 1}[reason]
            self.pending = (reason, now + delay, trigger)

        trace("observation.queued", reason=reason, scheduled=self.pending)

    def activity(self):
        return "error" if self.clock() < self.error_until else "working" if self.sessions.working else "idle"

    def receive(self, data):
        if not data or len(data) > MAX_INPUT_BYTES:
            return
        try:
            packet = json.loads(data.decode("utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
            if not isinstance(packet, dict):
                return
            kind = packet.get("kind")
            if kind == "clear_memory":
                self.clear_memory()
            elif kind == "event":
                result = self.sessions.handle(packet)
                if result:
                    key = result["session"]
                    lane = self.sessions.lanes.get(key)
                    if not lane.context_loaded:
                        lane.context_loaded = True
                        try:
                            stored = self.cognition.memory.context(key)
                            if stored:
                                lane.first_prompt = stored["first_prompt"] or lane.first_prompt
                                lane.session_context = stored["session_context"]
                                lane.recent_requests = [*stored["recent_requests"], *lane.recent_requests][-4:]
                                lane.goal = lane.session_context or lane.first_prompt
                        except sqlite3.Error as error:
                            log.warning("Could not restore pet session context: %s", error)
                    if packet["event"] in ("intent", "subagent_start") and lane.first_prompt:
                        try:
                            self.cognition.memory.save_context(key, lane.first_prompt, lane.session_context, lane.recent_requests)
                        except sqlite3.Error as error:
                            log.warning("Could not save pet session context: %s", error)
                    state = {"id": key, "repo": lane.repo, "goal": lane.goal, "last_tool": lane.last_tool, "outcome": lane.outcome, "activity": result["activity"]}
                    trigger = {**packet, "target_session": key, "repo": lane.repo, "activity": result["activity"]}
                    if packet["event"] in ("turn_done", "turn_error", "subagent_done", "interrupt", "session_end"):
                        trigger["outcome"] = lane.outcome
                    try:
                        self.cognition.memory.record_event(packet, state)
                    except sqlite3.Error as error:
                        log.warning("Could not record pet event: %s", error)
                    if result["failure"]:
                        self.error_until = self.clock() + self.config.behavior.error_seconds
                        if lane.failure["count"] in (1, 3, 5):
                            self.queue("failure", {**trigger, "generation": lane.generation})
                        else:
                            trace("observation.skipped", reason="repeated_failure")
                    elif packet["event"] in ("intent", "subagent_start"):
                        self.queue("prompt", trigger)
                    elif packet["event"] in ("turn_done", "subagent_done") or result["convergence"]:
                        self.queue("completion", trigger)
                    if not self.system_sleeping:
                        self.behavior.tick(self.activity(), self.sessions)
                        tool_succeeded = packet["event"] == "tool_done" and packet.get("outcome", "success") == "success"
                        turn_succeeded = packet["event"] in ("turn_done", "subagent_done") and lane.outcome == "success"
                        if tool_succeeded or turn_succeeded:
                            self.behavior.success()
                trace("hook.received", packet=packet, result=result, snapshot=self.sessions.snapshot())
            elif kind == "renderer":
                if packet.get("event") == "status":
                    if self.behavior.renderer_status(packet):
                        sleeping = packet.get("system_sleeping")
                        if type(sleeping) is bool:
                            self.system_sleep_changed(sleeping)
                        if not self.system_sleeping:
                            self.behavior.tick(self.activity(), self.sessions)
                elif packet.get("event") in ("system_sleep", "system_wake"):
                    self.system_sleep_changed(packet["event"] == "system_sleep")
                elif packet.get("event") == "click":
                    self.click()
                else:
                    trace("renderer.feedback", packet=packet)
                    self.behavior.completion(packet)
            elif kind == "speak" and isinstance(packet.get("context"), str):
                ttl = packet.get("ttl")
                if ttl is None or (type(ttl) in (int, float) and math.isfinite(ttl) and ttl >= 0):
                    self.behavior.wake("speech_request")
                    self.manual = (packet["context"], ttl)
                    trace("speech.requested", context=packet["context"], ttl=ttl)
        except (ValueError, KeyError, TypeError, UnicodeError) as error:
            trace("input.rejected", error=str(error))
            log.debug("Ignoring malformed datagram")

    def clear_memory(self):
        try:
            self.cognition.memory.clear()
        except sqlite3.Error as error:
            log.warning("Could not clear pet memory: %s", error)
            return
        if self.request_task:
            self.request_task.cancel()
        self.sessions = Sessions(self.clock, self.config.stale_seconds, self.config.completed_retention_seconds)
        self.pending = None
        self.manual = None
        self.clicked = False
        self.last_speech = None
        self.last_reaction_state = None
        self.error_until = 0
        self.next_observation = self.clock() + self.rng.uniform(*self.config.observation_seconds)
        self.emit({"text": "", "ttl": 0})
        trace("memory.cleared", pet=self.pack.pet)
        log.info("cleared knowledge for pet=%s", self.pack.pet)

    def speech_duration(self, speech):
        return max(self.config.speech_ttl, 3 + len(speech.split()) / 2)

    def system_sleep_changed(self, sleeping):
        if sleeping == self.system_sleeping:
            return
        self.system_sleeping = sleeping
        trace("system.sleep" if sleeping else "system.wake")
        if sleeping:
            if self.request_task:
                self.request_task.cancel()
            self.last_speech = None
            self.pending = None
        else:
            self.pending = None
            self.sessions.prune()
            for lane in self.sessions.lanes.values():
                if lane.active:
                    lane.last_check_in = self.clock()
            self.last_reaction_state = self.sessions.reaction_state()
            self.next_observation = self.clock() + self.rng.uniform(*self.config.observation_seconds)

    def click(self):
        self.behavior.wake("click")
        self.behavior.greet(self.sessions)
        if self.last_speech and self.clock() - self.last_speech[1] <= SPEECH_REPLAY_SECONDS:
            speech, _ = self.last_speech
            self.emit({"text": speech, "ttl": self.speech_duration(speech)})
            trace("speech.replayed", text=speech)
        else:
            self.clicked = True
            trace("speech.requested", reason="click")

    async def observe(self, reason, context=None, ttl=None, trigger=None):
        if self.system_sleeping:
            return
        with observation():
            started = self.clock()
            sessions = self.sessions
            include_activity = reason == "check_in"
            keys = trigger["sessions"] if include_activity else None
            observation_context = self.sessions.observation_context(include_activity, keys)
            trace("observation.started", reason=reason)
            self.last_reaction_state = self.sessions.reaction_state()
            timeout = self.config.request_timeout
            try:
                result = await asyncio.wait_for(
                    self.cognition.observe({**self.sessions.snapshot(), "trigger": trigger, "sleeping": self.behavior.sleeping}, reason, context),
                    timeout=timeout,
                )
                if sessions is not self.sessions:
                    trace("observation.discarded", reason="memory_cleared")
                    return
                if context is None and observation_context != self.sessions.observation_context(include_activity, keys):
                    trace("observation.discarded", reason="session_changed", decision=result)
                    log.info("discarded stale automatic response")
                    return
                summary = result["session_context"]
                focus = self.sessions.lanes.get(result["context_session"])
                if summary is not None and summary.strip() and focus:
                    version = (focus.generation, focus.first_prompt, tuple(focus.recent_requests), focus.session_context)
                    if version == result["context_version"]:
                        focus.session_context = bounded(summary, 400)
                        focus.goal = focus.session_context or focus.first_prompt
                        try:
                            self.cognition.memory.save_context(result["context_session"], focus.first_prompt, focus.session_context, focus.recent_requests)
                        except sqlite3.Error as error:
                            log.warning("Could not save pet session context: %s", error)
                        trace("memory.context_updated", session=result["context_session"])
                    else:
                        trace("memory.context_discarded", reason="session_changed", session=result["context_session"])
                if result["speak"] is not None:
                    speech = result["speak"]
                    if speech.strip() and ttl != 0:
                        self.behavior.wake("speech")
                    reading_seconds = self.speech_duration(speech)
                    delivered = self.emit({"text": speech, "ttl": reading_seconds if ttl is None else ttl})
                    if delivered and speech.strip() and ttl != 0:
                        self.last_speech = (speech, self.clock())
                        self.clicked = False
                        self.cognition.memory.record_speech(speech, reason)
                if result["gesture"] is not None:
                    accepted = self.behavior.gesture(result["gesture"], self.sessions)
                    trace("observation.gesture", gesture=result["gesture"], accepted=accepted)
            except asyncio.CancelledError:
                trace("observation.cancelled")
                raise
            except Exception as error:
                log.warning("Observation failed: %s", error, exc_info=True)
            finally:
                trace("observation.finished", duration_seconds=self.clock() - started)

    async def loop(self):
        while True:
            now = self.clock()
            if now >= self.next_prune:
                previous = set(self.sessions.lanes)
                changed = self.sessions.prune()
                removed = sorted(previous - set(self.sessions.lanes))
                if changed or removed:
                    trace("sessions.pruned", removed=removed, snapshot=self.sessions.snapshot())
                self.next_prune = now + 1
            if not self.system_sleeping:
                self.behavior.tick(self.activity(), self.sessions)
            if self.request_task and self.request_task.done():
                if not self.request_task.cancelled():
                    self.request_task.result()
                self.request_task = None
            if not self.system_sleeping and self.request_task is None:
                request = None
                check_in = self.sessions.check_in_due(self.config.idle_message_seconds) if self.behavior.status else []
                if self.behavior.status and self.manual is not None:
                    context, ttl = self.manual
                    self.manual = None
                    request = ("explicit", context, ttl)
                elif self.behavior.status and self.clicked:
                    self.clicked = False
                    request = ("click", None, None)
                elif self.behavior.status and self.pending and now >= self.pending[1]:
                    reason, _, trigger = self.pending
                    self.pending = None
                    valid = True
                    if reason == "failure":
                        lane = self.sessions.lanes.get(trigger["target_session"])
                        failure = lane.failures.get(trigger["activity"]) if lane else None
                        valid = bool(failure and not failure["resolved"] and failure["generation"] == trigger["generation"])
                    if valid:
                        request = (reason, None, None, trigger)
                    else:
                        trace("observation.skipped", reason=reason, snapshot=self.sessions.snapshot())
                elif check_in:
                    request = ("check_in", None, None, {"sessions": check_in, "target_session": check_in[0]})
                    for key in check_in:
                        self.sessions.lanes[key].last_check_in = now
                elif self.behavior.status and self.sessions.active and now >= self.next_observation and not self.behavior.sleeping:
                    self.next_observation = now + self.rng.uniform(*self.config.observation_seconds)
                    if self.sessions.reaction_state() != self.last_reaction_state:
                        request = ("routine", None, None)
                    else:
                        trace("observation.skipped", reason="unchanged_activity")
                if request:
                    self.next_observation = now + self.rng.uniform(*self.config.observation_seconds)
                    trace("observation.scheduled", reason=request[0], next_observation=self.next_observation)
                    self.request_task = asyncio.create_task(self.observe(*request))
            await asyncio.sleep(0.05)

    async def close(self):
        trace("daemon.stopping", snapshot=self.sessions.snapshot())
        for task in (self.loop_task, self.request_task):
            if task:
                task.cancel()
        await asyncio.gather(*(task for task in (self.loop_task, self.request_task) if task), return_exceptions=True)
        if self.child and self.child.returncode is None:
            self.child.terminate()
            try:
                await asyncio.wait_for(self.child.wait(), timeout=2)
            except asyncio.TimeoutError:
                self.child.kill()
                await self.child.wait()
        if self.transport:
            self.transport.close()
            await asyncio.sleep(0)
        if self.sender:
            self.sender.close()
            self.sender = None
        if self.identity:
            unlink_owned(self.config.daemon_socket, self.identity)
            self.identity = None
        self.cognition.memory.close()

    async def run(self):
        if not self.config.overlay_binary.is_file() or not os.access(self.config.overlay_binary, os.X_OK):
            raise ValueError("Build the renderer with make build before starting the daemon")
        loop = asyncio.get_running_loop()
        for number in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(number, self.stopping.set)
        child_wait = stop_wait = None
        try:
            await self.start()
            self.child = await asyncio.create_subprocess_exec(
                str(self.config.overlay_binary), "--assets", str(self.pack.directory),
                "--socket", self.config.overlay_socket, "--daemon-socket", self.config.daemon_socket,
                "--pointer-radius", str(self.config.behavior.pointer_radius),
            )
            child_wait = asyncio.create_task(self.child.wait())
            stop_wait = asyncio.create_task(self.stopping.wait())
            finished, _ = await asyncio.wait((child_wait, stop_wait, self.loop_task), return_when=asyncio.FIRST_COMPLETED)
            if self.loop_task in finished:
                self.loop_task.result()
            if child_wait in finished and child_wait.result() != 0:
                raise RuntimeError(f"Renderer exited with status {child_wait.result()}")
        finally:
            await self.close()
            for task in (child_wait, stop_wait):
                if task and not task.done():
                    task.cancel()
            await asyncio.gather(*(task for task in (child_wait, stop_wait) if task), return_exceptions=True)
            for number in (signal.SIGINT, signal.SIGTERM):
                loop.remove_signal_handler(number)
