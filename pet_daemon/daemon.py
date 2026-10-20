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

from .attention import Attention
from .behavior import Behavior
from .cognition import Cognition
from .commentary import Commentary
from .episodes import Episodes
from .events import AgentEvents
from .memory import Memory
from .sessions import Sessions
from .processes import ProcessTracker
from .repositories import RepositoryTracker
from .sprites import Pack
from .telemetry import event, trace, verbose


log = logging.getLogger("pet-daemon")
MAX_INPUT_BYTES = 2000


def reject_constant(value):
    raise ValueError(f"Invalid JSON constant: {value}")


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
        self.sessions = Sessions(clock, config.stale_seconds, config.completed_retention_seconds, config.activity_fresh_seconds)
        self.episodes = Episodes()
        self.attention = Attention(config, clock, self.rng)
        self.processes = ProcessTracker(clock=clock, interval=config.process_probe_seconds)
        self.behavior = Behavior(self.pack, config.behavior, clock, self.rng, self.emit)
        self.memory = Memory(config.memory_file, self.pack.pet, time.time, config.memory_retention_days, config.memory_max_events, speech_history_seconds=config.speech_history_seconds)
        self.cognition = Cognition(client, config, self.pack, memory=self.memory)
        self.repositories = RepositoryTracker(self.memory, local_interval=config.repository_refresh_seconds, github_interval=config.github_refresh_seconds)
        self.commentary = Commentary(
            config, sessions=self.sessions, episodes=self.episodes, attention=self.attention,
            behavior=self.behavior, cognition=self.cognition, repositories=self.repositories,
            memory=self.memory, emit=self.emit, clock=clock, rng=self.rng,
        )
        self.events = AgentEvents(
            config, sessions=self.sessions, episodes=self.episodes, repositories=self.repositories,
            memory=self.memory, behavior=self.behavior, queue=self.commentary.queue, clock=clock, rng=self.rng,
        )
        self.process_task = None
        self.transport = None
        self.identity = None
        self.sender = None
        self.child = None
        self.loop_task = None
        self.next_prune = 0
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
        verbose("daemon.started", config=asdict(self.config), pet=self.pack.pet,
              clips={key: asdict(clip) for key, clip in self.pack.clips.items()}, next_observation=self.attention.next_observation)
        event("daemon.listening", socket=self.config.daemon_socket, pet=self.pack.pet, model=self.config.model,
              stale_seconds=self.config.stale_seconds, retention_seconds=self.config.completed_retention_seconds)

    def emit(self, packet):
        if self.sender is None:
            return False
        try:
            data = json.dumps(packet, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
            self.sender.sendto(data, self.config.overlay_socket)
            verbose("renderer.command", packet=packet)
            return True
        except (OSError, ValueError) as error:
            trace("renderer.delivery_failed", packet=packet, error=str(error))
            log.warning("Renderer delivery failed: %s", error)
            return False

    async def probe_processes(self):
        result = await self.processes.refresh(self.sessions.process_owners())
        self.sessions.apply_process_status(result)

    def receive(self, data):
        if not data or len(data) > MAX_INPUT_BYTES:
            return
        try:
            packet = json.loads(data.decode("utf-8"), parse_constant=reject_constant)
        except (ValueError, UnicodeError) as error:
            trace("input.rejected", error=str(error))
            return
        if not isinstance(packet, dict):
            return
        kind = packet.get("kind")
        if kind == "clear_speech":
            self.commentary.clear_speech()
        elif kind == "clear_memory":
            self.clear_memory()
        elif kind == "event":
            self.events.handle(packet, self.commentary.suspended)
        elif kind == "renderer":
            self.receive_renderer(packet)
        elif kind == "speak" and isinstance(packet.get("context"), str):
            ttl = packet.get("ttl")
            if ttl is None or (type(ttl) in (int, float) and math.isfinite(ttl) and ttl >= 0):
                self.commentary.speak(packet["context"], ttl)

    def receive_renderer(self, packet):
        feedback = packet.get("event")
        if feedback == "speech_accepted":
            self.commentary.delivery.acknowledge(packet)
        elif feedback == "status":
            if self.behavior.renderer_status(packet):
                sleeping = packet.get("system_sleeping")
                if type(sleeping) is bool:
                    self.system_sleep_changed(sleeping)
                if not self.commentary.suspended:
                    self.behavior.tick(self.events.activity(), self.sessions)
        elif feedback in ("system_sleep", "system_wake"):
            self.system_sleep_changed(feedback == "system_sleep")
        elif feedback == "click":
            self.commentary.click()
        else:
            trace("renderer.feedback", command_id=packet.get("command_id"), feedback=feedback,
                  result=packet.get("result"), packet=packet)
            self.behavior.completion(packet)

    def clear_memory(self):
        try:
            self.memory.clear()
        except sqlite3.Error as error:
            log.warning("Could not clear pet memory: %s", error)
            return
        self.repositories.clear()
        self.sessions = Sessions(self.clock, self.config.stale_seconds, self.config.completed_retention_seconds, self.config.activity_fresh_seconds)
        self.episodes = Episodes()
        self.events.reset(self.sessions, self.episodes)
        self.commentary.reset(self.sessions, self.episodes)
        self.emit({"text": "", "ttl": 0})
        event("memory.cleared", pet=self.pack.pet)

    def system_sleep_changed(self, sleeping):
        if sleeping == self.commentary.suspended:
            return
        event("system.sleep" if sleeping else "system.wake")
        if sleeping:
            self.behavior.suspend("system_sleep")
        else:
            self.behavior.resume()
            self.sessions.prune()
        self.commentary.sleep_changed(sleeping)

    async def loop(self):
        while True:
            now = self.clock()
            self.repositories.tick()
            if self.process_task is None or self.process_task.done():
                if self.process_task is not None:
                    self.process_task.result()
                self.process_task = asyncio.create_task(self.probe_processes())
            if now >= self.next_prune:
                previous = set(self.sessions.lanes)
                changed = self.sessions.prune()
                removed = sorted(previous - set(self.sessions.lanes))
                if changed or removed:
                    trace("sessions.pruned", removed=removed, snapshot=self.sessions.snapshot())
                self.next_prune = now + 1
            if not self.commentary.suspended:
                self.behavior.tick(self.events.activity(), self.sessions)
            self.commentary.tick()
            await asyncio.sleep(0.05)

    async def close(self):
        event("daemon.stopping", snapshot=self.sessions.snapshot(), sessions=len(self.sessions.lanes))
        for task in (self.loop_task, self.process_task):
            if task:
                task.cancel()
        await self.commentary.close()
        await asyncio.gather(*(task for task in (self.loop_task, self.process_task) if task), return_exceptions=True)
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
        await self.repositories.close()
        self.memory.close()

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
