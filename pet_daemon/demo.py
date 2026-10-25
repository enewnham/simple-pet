import argparse
import asyncio
import json
import math
import os
import signal
import socket
import sqlite3
import tempfile
import time
from collections.abc import Sequence
from contextlib import closing
from pathlib import Path

from anthropic import AsyncAnthropic

from .commentary.model import AnthropicModelClient, ModelClient
from .config import Config
from .daemon import bind_socket, unlink_owned
from .domain import SpeechReason
from .logging import configure
from .replay import ReplayHarness
from .runtime import OverlayPacket
from .work.evidence import ExecutionStatus, ReportedProgress, UserIntent
from .work.shared_work import WorkSubject


def positive_speed(value: str) -> float:
    speed = float(value)
    if not math.isfinite(speed) or speed <= 0:
        raise argparse.ArgumentTypeError("Speed must be positive and finite")
    return speed


def backup_memory(source: Path, destination: Path) -> None:
    with (
        closing(
            sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True)
        ) as original,
        closing(sqlite3.connect(destination)) as copy,
    ):
        original.backup(copy)
    destination.chmod(0o600)


def select_work(subjects: Sequence[WorkSubject], query: str | None) -> WorkSubject:
    for subject in sorted(subjects, key=lambda item: -item.changed_at):
        if subject.parent is not None or not (
            subject.story.requests or subject.story.activity
        ):
            continue
        text = [subject.story.intent]
        for record in (*subject.story.requests, *subject.story.activity):
            match record.details:
                case UserIntent(text=value) | ReportedProgress(reported_progress=value):
                    text.append(value)
                case ExecutionStatus(reported_progress=value) if value:
                    text.append(value)
        if query is None or query.casefold() in "\n".join(text).casefold():
            return subject
    raise ValueError(
        "No saved work matches the requested thread"
        if query
        else "No saved work is available"
    )


async def demonstrate(
    config: Config,
    client: ModelClient,
    work: str | None,
    speed: float,
    stopping: asyncio.Event,
) -> None:
    with tempfile.TemporaryDirectory(prefix="pet-demo-", dir="/tmp") as directory:
        root = Path(directory)
        copied = root / "memory.sqlite3"
        backup_memory(config.memory_file, copied)
        settings = config.model_copy(
            deep=True,
            update={
                "memory_file": copied,
                "daemon_socket": str(root / "daemon.sock"),
                "overlay_socket": str(root / "overlay.sock"),
            },
        )
        settings.behavior.naps = False
        settings.behavior.roaming = False
        settings.behavior.care_speech_interval = (90, 150)
        receiver, identity = bind_socket(settings.daemon_socket)
        sender = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        sender.setblocking(False)
        child = None
        tasks: list[asyncio.Task[None] | asyncio.Task[int] | asyncio.Task[bool]] = []
        ready = asyncio.Event()
        loop = asyncio.get_running_loop()

        def emit(packet: OverlayPacket) -> None:
            try:
                sender.sendto(
                    json.dumps(packet, ensure_ascii=False, allow_nan=False).encode(),
                    settings.overlay_socket,
                )
            except OSError:
                pass

        try:
            async with ReplayHarness(
                settings,
                client,
                memory_file=copied,
                wall_time=time.time(),
                monotonic_time=time.monotonic(),
                on_packet=emit,
            ) as replay:
                try:
                    chosen = select_work(replay.app.context().work, work)

                    async def receive() -> None:
                        while True:
                            data = await loop.sock_recv(receiver, 65536)
                            try:
                                packet = json.loads(data)
                            except ValueError:
                                continue
                            if isinstance(packet, dict):
                                replay.receive(packet)
                                if packet.get("event") == "status":
                                    ready.set()

                    async def tick() -> None:
                        previous = time.monotonic()
                        while True:
                            current = time.monotonic()
                            replay.clock.advance((current - previous) * speed)
                            previous = current
                            replay.app.tick()
                            await asyncio.sleep(0.05)

                    child = await asyncio.create_subprocess_exec(
                        str(settings.overlay_binary),
                        "--assets",
                        str(settings.pet_dir),
                        "--socket",
                        settings.overlay_socket,
                        "--daemon-socket",
                        settings.daemon_socket,
                        "--pointer-radius",
                        str(settings.behavior.pointer_radius),
                    )
                    receive_task = asyncio.create_task(receive())
                    child_wait = asyncio.create_task(child.wait())
                    ready_wait = asyncio.create_task(ready.wait())
                    stop_wait = asyncio.create_task(stopping.wait())
                    tasks.extend((receive_task, child_wait, ready_wait, stop_wait))
                    finished, _ = await asyncio.wait(
                        (ready_wait, child_wait, receive_task, stop_wait),
                        timeout=5,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if receive_task in finished:
                        receive_task.result()
                    if stop_wait in finished:
                        return
                    if ready_wait not in finished:
                        raise RuntimeError(
                            "Renderer did not become ready within five seconds"
                        )
                    snapshot = replay.app.context()
                    selection = replay.app.attention.select_recollection(
                        snapshot, SpeechReason.CLICK, subject=chosen
                    )
                    replay.app.reason = SpeechReason.CLICK
                    replay.app.task = asyncio.create_task(
                        replay.app.comment(
                            SpeechReason.CLICK, snapshot=snapshot, selection=selection
                        )
                    )
                    print(
                        f"Thread: {chosen.story.intent}\nReal model ({settings.model}) and configured persona, {speed:g}× time, temporary memory. Press Ctrl-C to stop.",
                        flush=True,
                    )
                    tick_task = asyncio.create_task(tick())
                    tasks.append(tick_task)
                    finished, _ = await asyncio.wait(
                        (child_wait, stop_wait, tick_task, receive_task),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for task in finished:
                        result = task.result()
                        if task is child_wait and result != 0:
                            raise RuntimeError(f"Renderer exited with status {result}")
                finally:
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            if child is not None and child.returncode is None:
                child.terminate()
                try:
                    await asyncio.wait_for(child.wait(), 2)
                except TimeoutError:
                    child.kill()
                    await child.wait()
            receiver.close()
            sender.close()
            unlink_owned(settings.daemon_socket, identity)


async def run(config: Config, work: str | None, speed: float) -> None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError("Set ANTHROPIC_API_KEY before starting pet-demo")
    stopping = asyncio.Event()
    loop = asyncio.get_running_loop()
    for number in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(number, stopping.set)
    try:
        async with AsyncAnthropic(
            api_key=key, timeout=config.request_timeout, max_retries=0
        ) as client:
            await demonstrate(
                config, AnthropicModelClient(client), work, speed, stopping
            )
    finally:
        for number in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(number)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="pet-demo",
        description="Explore saved work with your configured pet and model",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "pet.config.json",
    )
    parser.add_argument(
        "--work", help="Choose a saved thread by request or report text"
    )
    parser.add_argument("--speed", type=positive_speed, default=10.0)
    args = parser.parse_args()
    configure()
    try:
        asyncio.run(run(Config.load(args.config), args.work, args.speed))
    except KeyboardInterrupt:
        pass
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as error:
        parser.exit(1, f"pet-demo: {error}\n")


if __name__ == "__main__":
    main()
