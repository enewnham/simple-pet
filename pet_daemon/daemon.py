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
import stat
import time

from .behavior import Behavior
from .cognition import Cognition
from .sessions import Sessions
from .sprites import Pack
from .telemetry import observation, trace


log = logging.getLogger("pet-daemon")
MAX_INPUT_BYTES = 2000


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
        self.pending = None
        self.manual = None
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

    def queue(self, reason):
        now = self.clock()
        if reason == "failure":
            self.pending = (reason, now + 1)
        elif not self.pending or self.pending[0] != "failure" or not self.sessions.failed:
            self.pending = (reason, now + 3)

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
            if kind == "event":
                result = self.sessions.handle(packet)
                trace("hook.received", packet=packet, result=result, revision=self.sessions.revision, snapshot=self.sessions.snapshot())
                if result:
                    if result["failure"]:
                        self.error_until = self.clock() + self.config.behavior.error_seconds
                        self.queue("failure")
                    elif result["convergence"]:
                        self.queue("convergence")
                    self.behavior.tick(self.activity(), self.sessions)
            elif kind == "renderer":
                if packet.get("event") == "status":
                    if self.behavior.renderer_status(packet):
                        self.behavior.tick(self.activity(), self.sessions)
                else:
                    trace("renderer.feedback", packet=packet)
                    self.behavior.completion(packet)
            elif kind == "speak" and isinstance(packet.get("context"), str):
                ttl = packet.get("ttl", self.config.speech_ttl)
                if type(ttl) in (int, float) and math.isfinite(ttl) and ttl >= 0:
                    self.manual = (packet["context"], ttl)
                    trace("speech.requested", context=packet["context"], ttl=ttl)
        except (ValueError, KeyError, TypeError, UnicodeError) as error:
            trace("input.rejected", error=str(error))
            log.debug("Ignoring malformed datagram")

    async def observe(self, reason, context=None, ttl=None):
        with observation():
            started = self.clock()
            revision = self.sessions.revision
            trace("observation.started", reason=reason, revision=revision)
            try:
                result = await asyncio.wait_for(
                    self.cognition.observe(self.sessions.snapshot(), reason, context),
                    timeout=self.config.request_timeout,
                )
                if context is None and revision != self.sessions.revision:
                    trace("observation.discarded", reason="session_changed", requested_revision=revision, revision=self.sessions.revision, decision=result)
                    log.info("kept observation in history; discarded stale automatic response")
                    return
                if result["speak"] is not None:
                    self.emit({"text": result["speak"], "ttl": self.config.speech_ttl if ttl is None else ttl})
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
            self.behavior.tick(self.activity(), self.sessions)
            if self.request_task and self.request_task.done():
                self.request_task.result()
                self.request_task = None
            if self.behavior.status and self.request_task is None:
                request = None
                if self.manual is not None:
                    context, ttl = self.manual
                    self.manual = None
                    request = ("explicit", context, ttl)
                elif self.pending and now >= self.pending[1]:
                    reason, _ = self.pending
                    self.pending = None
                    valid = self.sessions.failed if reason == "failure" else self.sessions.convergence is not None and not self.sessions.active
                    if valid:
                        request = (reason, None, None)
                    else:
                        trace("observation.skipped", reason=reason, snapshot=self.sessions.snapshot())
                elif now >= self.next_observation:
                    request = ("routine", None, None)
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
