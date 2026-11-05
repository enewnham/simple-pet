import asyncio
import errno
import json
import os
import random
import signal
import socket
import stat
import time
from collections.abc import Callable, Mapping
from typing import Never
from urllib.parse import unquote, urlsplit

from anthropic import AsyncAnthropic
from claude_agent_sdk import query as coordinator_query

from .application import Application
from .attention import Attention
from .behavior.controller import Behavior
from .behavior.sprites import Pack
from .commentary import AnthropicModelClient, Cognition
from .companion.index import SourceChange
from .companion.sdk import SdkCompanion, SdkQuery
from .config import Config
from .conversation import Conversations
from .coordination.models import (
    CandidateCause,
    DelegatedTask,
    Responsibility,
    ResponsibilityStatus,
    SourceReference,
)
from .coordination.runtime import Coordinator
from .coordination.watches import Watches
from .coordination.workflows import Workflows
from .desktop import AdapterClient, Delegation
from .logging import get_logger
from .navigation import Navigator, Target, TargetKind
from .runtime import RandomSource
from .sessions.processes import ProcessTracker
from .sessions.reactions import SessionReactions
from .sessions.tracking import Sessions
from .sources import RepositoryTracker
from .sources.checks import GitHubChecks
from .sources.hooks import normalize_hook
from .storage import Memory
from .transport import MAX_DATAGRAM_BYTES, SOCKET_BUFFER_BYTES
from .types import JSONObject

log = get_logger("pet-daemon")


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
        receiver.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, SOCKET_BUFFER_BYTES)
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
    def __init__(self, daemon: Daemon) -> None:
        self.daemon = daemon

    def datagram_received(
        self, data: bytes, addr: tuple[str, int] | str | None
    ) -> None:
        self.daemon.receive(data)

    def error_received(self, exc: Exception) -> None:
        log.error("hook.socket_failed", error=str(exc), exc_info=exc)


class Daemon:
    def __init__(
        self,
        config: Config,
        client: AsyncAnthropic,
        clock: Callable[[], float] = time.monotonic,
        rng: RandomSource | None = None,
        wall_clock: Callable[[], float] = time.time,
        companion_query: SdkQuery = coordinator_query,
    ) -> None:
        self.config = config
        self.clock = clock
        self.wall_clock = wall_clock
        self.companion_query = companion_query
        self.rng = rng or random.Random()
        self.pack = Pack(config.pet_dir)
        sessions = Sessions(
            clock,
            config.stale_seconds,
            config.completed_retention_seconds,
            config.activity_fresh_seconds,
            wall_clock=wall_clock,
        )
        reactions = SessionReactions()
        attention = Attention(config, clock, self.rng, wall_clock=wall_clock)
        processes = ProcessTracker(clock=clock, interval=config.process_probe_seconds)
        behavior = Behavior(self.pack, config.behavior, clock, self.rng, self.emit)
        memory = Memory(
            config.memory_file,
            self.pack.pet,
            wall_clock,
            config.memory_retention_days,
            config.memory_max_events,
            speech_history_seconds=config.speech_history_seconds,
        )
        cognition = Cognition(
            AnthropicModelClient(client), config, memory=memory, clock=wall_clock
        )
        repositories = RepositoryTracker(
            lambda observation: self.application.ingest(observation),
            local_interval=config.repository_refresh_seconds,
            github_interval=config.github_refresh_seconds,
            checkpoints=memory.load_repositories(),
            targets=memory.collection_targets(),
        )
        self.configure_coordination(config, memory, wall_clock)
        conversations = (
            Conversations(
                memory.coordination,
                self.companion.reply,
                self.emit_conversation,
                clock=wall_clock,
                act=self.workflows.accept_action,
                desktop_capabilities=config.desktop.capabilities
                if config.desktop
                else (),
            )
            if self.companion is not None
            else None
        )
        self.application = Application(
            config,
            sessions=sessions,
            reactions=reactions,
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
            conversations=conversations,
            companion=self.companion,
            coordinator=self.coordinator,
            desktop=self.desktop,
        )
        self.application.reconcile_targets()
        self.transport: asyncio.DatagramTransport | None = None
        self.identity: tuple[int, int] | None = None
        self.sender: socket.socket | None = None
        self.child: asyncio.subprocess.Process | None = None
        self.loop_task: asyncio.Task[None] | None = None
        self.stopping = asyncio.Event()

    def emit_conversation(self, packet: JSONObject) -> None:
        self.emit(packet)

    def configure_coordination(
        self, config: Config, memory: Memory, clock: Callable[[], float]
    ) -> None:
        self.coordination_store = memory.coordination
        self.desktop = (
            Delegation(
                memory.coordination,
                AdapterClient(config.desktop),
                clock=clock,
                changed=self.task_changed,
                coding_slots=config.coordinator.scheduling.coding_slots,
            )
            if config.desktop is not None
            else None
        )
        effect = self.desktop.dispatch if self.desktop is not None else None
        workflows = Workflows(
            memory.coordination,
            clock,
            effect=effect,
            changed=self.responsibility_changed,
            navigate=self.open_source,
            paused=self.coordination_paused,
            revised=self.responsibility_revised,
        )
        self.workflows = workflows
        self.companion = (
            SdkCompanion(
                config.coordinator,
                memory,
                clock,
                config.persona,
                workflow_tools=workflows.tools,
                on_source_changes=self.sources_changed,
                query=self.companion_query,
            )
            if config.coordinator.enabled
            else None
        )
        self.watches = Watches(
            memory.coordination,
            config.coordinator.scheduling,
            clock,
            GitHubChecks(clock=clock).observe,
        )
        self.coordinator = (
            Coordinator(
                memory.coordination,
                config.coordinator.scheduling,
                clock,
                self.companion.investigate,
                effect=effect,
                reconcile=self.desktop.reconcile if self.desktop is not None else None,
                refresh_watches=self.refresh_coordination_sources,
                sources_current=self.sources_current,
                completion_sources_valid=self.companion.completion_sources_valid,
                completed=self.responsibility_revised,
            )
            if self.companion is not None
            else None
        )

    def sources_changed(self, changes: tuple[SourceChange, ...]) -> None:
        if self.coordinator is None or self.companion is None:
            return
        for change in changes:
            document = self.companion.index.document(change.source_id)
            scope = document.source.scope if document is not None else change.provider
            self.coordinator.queue(
                "Do these source changes affect recent work or open questions?",
                scope,
                CandidateCause(
                    change.kind, change.source_id, change.revision, change.observed_at
                ),
            )

    def task_changed(self, task: DelegatedTask) -> None:
        if self.coordinator is None:
            return
        responsibility = self.coordination_store.responsibility(task.responsibility_id)
        if responsibility is not None:
            self.coordinator.queue(
                "Assess the next bounded step for the owned task.",
                responsibility.scope,
                CandidateCause("task", task.id, str(task.revision), self.wall_clock()),
                responsibility_id=responsibility.id,
                project_id=responsibility.project_id,
                priority=responsibility.priority,
                due_at=responsibility.due_at,
                exclusive=responsibility.exclusive,
            )

    def responsibility_changed(self, responsibility: Responsibility) -> None:
        if self.coordinator is not None:
            self.coordinator.queue(
                responsibility.purpose,
                responsibility.scope,
                CandidateCause(
                    "responsibility",
                    responsibility.id,
                    str(responsibility.revision),
                    self.wall_clock(),
                ),
                responsibility_id=responsibility.id,
                project_id=responsibility.project_id,
                priority=responsibility.priority,
                due_at=responsibility.due_at,
                exclusive=responsibility.exclusive,
            )

    def responsibility_revised(self, responsibility: Responsibility) -> None:
        if self.desktop is not None:
            self.desktop.responsibility_changed(responsibility)

    def coordination_paused(self, paused: bool) -> None:
        if self.coordinator is not None:
            self.coordinator.suspended = paused
            self.coordinator.next_tick = 0
            if paused and self.coordinator.task is not None:
                self.coordinator.task.cancel()
        if self.desktop is not None:
            self.desktop.suspended = paused or self.application.suspended

    async def refresh_coordination_sources(self) -> None:
        if self.companion is not None:
            await self.companion.refresh_sources()
        await self.watches.refresh()
        for responsibility in self.coordination_store.responsibilities():
            if responsibility.status == ResponsibilityStatus.ACTIVE:
                self.responsibility_changed(responsibility)
            if (
                self.coordinator is not None
                and responsibility.followup_at is not None
                and responsibility.followup_at <= self.wall_clock()
            ):
                self.coordinator.queue(
                    responsibility.purpose,
                    responsibility.scope,
                    CandidateCause(
                        "followup",
                        f"followup:{responsibility.id}",
                        str(responsibility.followup_at),
                        self.wall_clock(),
                    ),
                    responsibility_id=responsibility.id,
                    project_id=responsibility.project_id,
                )

    def sources_current(self, sources: tuple[SourceReference, ...]) -> bool:
        if self.companion is None:
            return False
        statuses = {
            status.provider: status for status in self.companion.index.statuses()
        }
        for source in sources:
            if source.provider == "user":
                instruction = self.coordination_store.instruction(source.id)
                if (
                    instruction is None
                    or not instruction.active
                    or str(instruction.revision) != source.revision
                ):
                    return False
                continue
            status = statuses.get(source.provider)
            document = self.companion.index.document(source.id.split("#L", 1)[0])
            operational = document is not None and document.kind in (
                "observation",
                "ticket",
                "checks",
                "task_observation",
            )
            if (
                not self.companion.index.current(
                    source,
                    now=self.wall_clock() if operational else None,
                    max_age=self.config.coordinator.scheduling.source_fresh_seconds,
                )
                or status is None
                or operational
                and self.wall_clock() - status.checked_at
                > self.config.coordinator.scheduling.source_fresh_seconds
            ):
                return False
        return True

    async def open_source(self, source: SourceReference) -> None:
        if source.provider == "user":
            instruction = self.coordination_store.instruction(source.id)
            conversations = self.application.conversations
            if (
                instruction is None
                or conversations is None
                or str(instruction.revision) != source.revision
            ):
                raise ValueError("That user source changed or is unavailable")
            if instruction.message_id is not None:
                conversations.show_message(instruction.message_id)
            else:
                conversations.show_activity()
            return
        known = (
            self.companion is not None and self.companion.index.current(source)
        ) or any(
            source in record.sources for record in self.coordination_store.activity()
        )
        if not known or not source.available or not source.uri:
            raise ValueError("The source changed or is unavailable")
        parsed = urlsplit(source.uri)
        if parsed.scheme in ("https", "http"):
            target = Target(TargetKind.PULL_REQUEST, source.uri)
        elif parsed.scheme == "agent":
            target = Target(TargetKind.SESSION, parsed.path.lstrip("/"), parsed.netloc)
        elif parsed.scheme == "file":
            target = Target(TargetKind.WORKTREE, unquote(parsed.path))
        elif not parsed.scheme and source.uri.startswith("/"):
            target = Target(TargetKind.WORKTREE, source.uri)
        else:
            raise ValueError("No navigation route is configured for that source")
        result = await Navigator(self.config.navigation.handlers).open(target)
        if not result.success:
            raise RuntimeError(result.error or "Source navigation failed")

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
        log.debug(
            "daemon.started",
            detail="payload",
            config=self.config.model_dump(),
            pet=self.pack.pet,
            clips={key: clip.model_dump() for key, clip in self.pack.clips.items()},
            next_observation=self.application.attention.next_observation,
        )
        log.info(
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
            log.debug("renderer.command", detail="payload", packet=packet)
            return True
        except (OSError, ValueError) as error:
            log.exception("renderer.delivery_failed", packet=packet, error=str(error))
            return False

    def receive(self, data: bytes) -> None:
        if not data:
            return
        if len(data) > MAX_DATAGRAM_BYTES:
            log.warning("input.rejected", reason="message_too_large", bytes=len(data))
            return
        try:
            packet = json.loads(data.decode("utf-8"), parse_constant=reject_constant)
        except (ValueError, UnicodeError) as error:
            log.debug("input.rejected", error=str(error))
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
        log.info(
            "daemon.stopping",
            snapshot=self.application.sessions.activity().model_dump(),
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
