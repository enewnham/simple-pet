import asyncio
import errno
import json
import logging
import os
import random
import signal
import socket
import stat
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict
from typing import Never

from anthropic import AsyncAnthropic

from .application import Application
from .attention import Attention
from .behavior.controller import Behavior
from .behavior.sprites import Pack
from .commentary import AnthropicModelClient, Cognition
from .config import Config
from .runtime import RandomSource
from .sessions.episodes import Episodes
from .sessions.processes import ProcessTracker
from .sessions.tracking import Sessions
from .sources import RepositoryTracker
from .sources.hooks import normalize_hook
from .storage import Memory
from .telemetry import event, trace, verbose

log = logging.getLogger("pet-daemon")
MAX_INPUT_BYTES = 2000


def reject_constant(value: str) -> Never:
    raise ValueError(f"Invalid JSON constant: {value}")


def bind_socket(path: str) -> tuple[socket.socket, tuple[int, int]]:
    try:
        existing = os.lstat(path)
    except FileNotFoundError:
        existing = None
    if existing:
        if not stat.S_ISSOCK(existing.st_mode) or existing.st_uid != os.getuid():
            raise ValueError(
                "Refusing to replace a non-socket or another user's socket"
            )
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


def unlink_owned(path: str, identity: tuple[int, int]) -> None:
    try:
        current = os.lstat(path)
        if (current.st_dev, current.st_ino) == identity:
            os.unlink(path)
    except FileNotFoundError:
        pass


class Receiver(asyncio.DatagramProtocol):
    def __init__(self, daemon: "Daemon") -> None:
        self.daemon = daemon

    def datagram_received(
        self, data: bytes, addr: tuple[str, int] | str | None
    ) -> None:
        self.daemon.receive(data)

    def error_received(self, exc: Exception) -> None:
        log.warning("Hook socket: %s", exc)


class Daemon:
    def __init__(
        self,
        config: Config,
        client: AsyncAnthropic,
        clock: Callable[[], float] = time.monotonic,
        rng: RandomSource | None = None,
    ) -> None:
        self.config = config
        self.clock = clock
        self.rng = rng or random.Random()
        self.pack = Pack(config.pet_dir)
        sessions = Sessions(
            clock,
            config.stale_seconds,
            config.completed_retention_seconds,
            config.activity_fresh_seconds,
        )
        episodes = Episodes()
        attention = Attention(config, clock, self.rng)
        processes = ProcessTracker(clock=clock, interval=config.process_probe_seconds)
        behavior = Behavior(self.pack, config.behavior, clock, self.rng, self.emit)
        memory = Memory(
            config.memory_file,
            self.pack.pet,
            time.time,
            config.memory_retention_days,
            config.memory_max_events,
            speech_history_seconds=config.speech_history_seconds,
        )
        cognition = Cognition(AnthropicModelClient(client), config, memory=memory)
        repositories = RepositoryTracker(
            lambda observation: self.application.ingest(observation),
            local_interval=config.repository_refresh_seconds,
            github_interval=config.github_refresh_seconds,
            checkpoints=memory.load_repositories(),
            targets=memory.collection_targets(),
        )
        self.application = Application(
            config,
            sessions=sessions,
            episodes=episodes,
            attention=attention,
            behavior=behavior,
            cognition=cognition,
            providers=(repositories,),
            processes=processes,
            memory=memory,
            emit=self.emit,
            clock=clock,
            rng=self.rng,
            checkout_target=lambda path, refresh: repositories.track_checkout(
                path, refresh=refresh
            ),
            reconcile_work=lambda: repositories.reconcile_targets(
                memory.collection_targets()
            ),
        )
        self.application.reconcile_targets()
        self.transport: asyncio.DatagramTransport | None = None
        self.identity: tuple[int, int] | None = None
        self.sender: socket.socket | None = None
        self.child: asyncio.subprocess.Process | None = None
        self.loop_task: asyncio.Task[None] | None = None
        self.stopping = asyncio.Event()

    async def start(self) -> None:
        receiver, identity = bind_socket(self.config.daemon_socket)
        self.identity = identity
        try:
            (
                self.transport,
                _,
            ) = await asyncio.get_running_loop().create_datagram_endpoint(
                lambda: Receiver(self), sock=receiver
            )
        except BaseException:
            receiver.close()
            unlink_owned(self.config.daemon_socket, identity)
            self.identity = None
            raise
        self.sender = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.sender.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 131072)
        self.sender.setblocking(False)
        self.loop_task = asyncio.create_task(self.loop())
        verbose(
            "daemon.started",
            config=asdict(self.config),
            pet=self.pack.pet,
            clips={key: asdict(clip) for key, clip in self.pack.clips.items()},
            next_observation=self.application.attention.next_observation,
        )
        event(
            "daemon.listening",
            socket=self.config.daemon_socket,
            pet=self.pack.pet,
            model=self.config.model,
            stale_seconds=self.config.stale_seconds,
            retention_seconds=self.config.completed_retention_seconds,
        )

    def emit(self, packet: Mapping[str, object]) -> bool:
        if self.sender is None:
            return False
        try:
            data = json.dumps(
                packet, ensure_ascii=False, allow_nan=False, separators=(",", ":")
            ).encode()
            self.sender.sendto(data, self.config.overlay_socket)
            verbose("renderer.command", packet=packet)
            return True
        except (OSError, ValueError) as error:
            trace("renderer.delivery_failed", packet=packet, error=str(error))
            log.warning("Renderer delivery failed: %s", error)
            return False

    def receive(self, data: bytes) -> None:
        if not data or len(data) > MAX_INPUT_BYTES:
            return
        try:
            packet = json.loads(data.decode("utf-8"), parse_constant=reject_constant)
        except (ValueError, UnicodeError) as error:
            trace("input.rejected", error=str(error))
            return
        if not isinstance(packet, dict):
            return
        if packet.get("kind") == "event":
            hook = normalize_hook(packet)
            if hook is not None:
                self.application.ingest(hook)
        else:
            self.application.receive(packet)

    async def loop(self) -> None:
        while True:
            self.application.tick()
            await asyncio.sleep(0.05)

    async def close(self) -> None:
        event(
            "daemon.stopping",
            snapshot=asdict(self.application.sessions.activity_view()),
            sessions=len(self.application.sessions.by_id),
        )
        if self.loop_task:
            self.loop_task.cancel()
        await self.application.close()
        if self.loop_task:
            await asyncio.gather(self.loop_task, return_exceptions=True)
        if self.child and self.child.returncode is None:
            self.child.terminate()
            try:
                await asyncio.wait_for(self.child.wait(), timeout=2)
            except TimeoutError:
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
        self.application.memory.close()

    async def run(self) -> None:
        if not self.config.overlay_binary.is_file() or not os.access(
            self.config.overlay_binary, os.X_OK
        ):
            raise ValueError(
                "Build the renderer with make build before starting the daemon"
            )
        loop = asyncio.get_running_loop()
        for number in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(number, self.stopping.set)
        child_wait = stop_wait = None
        try:
            await self.start()
            self.child = await asyncio.create_subprocess_exec(
                str(self.config.overlay_binary),
                "--assets",
                str(self.pack.directory),
                "--socket",
                self.config.overlay_socket,
                "--daemon-socket",
                self.config.daemon_socket,
                "--pointer-radius",
                str(self.config.behavior.pointer_radius),
            )
            child_wait = asyncio.create_task(self.child.wait())
            stop_wait = asyncio.create_task(self.stopping.wait())
            loop_task = self.loop_task
            assert loop_task is not None
            finished, _ = await asyncio.wait(
                (child_wait, stop_wait, loop_task),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if loop_task in finished:
                loop_task.result()
            if child_wait in finished and child_wait.result() != 0:
                raise RuntimeError(f"Renderer exited with status {child_wait.result()}")
        finally:
            await self.close()
            for task in (child_wait, stop_wait):
                if task and not task.done():
                    task.cancel()
            await asyncio.gather(
                *(task for task in (child_wait, stop_wait) if task),
                return_exceptions=True,
            )
            for number in (signal.SIGINT, signal.SIGTERM):
                loop.remove_signal_handler(number)
