"""Drive the application headlessly with controlled inputs and model clients."""

import argparse
import asyncio
import base64
import json
import logging
import math
import random
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, Self

import structlog

from .application import Application
from .attention import Attention
from .behavior.controller import Behavior
from .behavior.sprites import Pack
from .commentary.cognition import Cognition
from .commentary.contracts import ModelRequest
from .commentary.model import AnthropicModelClient, ModelClient
from .commentary.usage import UsageTokens, collect_usage, request_usage
from .config import Config
from .domain import DeliveryStatus, ModelPurpose, ProcessStatus, SpeechReason
from .logging import json_default
from .runtime import HookEvent, OverlayPacket, ProcessIdentity, Trigger
from .sessions.processes import ProcessTracker
from .sessions.reactions import SessionReactions
from .sessions.tracking import Sessions
from .sources.hooks import normalize_hook
from .storage import Memory
from .work.observations import RepositoryObservation


class FixtureModel:
    def __init__(self) -> None:
        self.speech: str | None = None

    async def complete(
        self, request: ModelRequest, reason: SpeechReason | ModelPurpose
    ) -> dict[str, object]:
        if reason != ModelPurpose.STORY:
            return {"speak": self.speech}
        payload = json.loads(request["messages"][0]["content"])
        latest = payload["current_work"]["latest_request_source"]
        sources = payload["current_moment"]["evidence"]
        intent = next(source for source in sources if source["id"] == latest)
        return {
            "intent": {"text": intent["text"][:400], "sources": [latest]},
            "decisions": [],
            "progress": [],
        }


class RecordingModel:
    def __init__(self, client: ModelClient) -> None:
        self.client = client
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self, request: ModelRequest, reason: SpeechReason | ModelPurpose
    ) -> dict[str, object]:
        started = time.monotonic()
        call: dict[str, Any] = {
            "purpose": str(reason),
            "request": deepcopy(request),
        }
        parent_usage = request_usage.get()
        with collect_usage() as usage:
            try:
                response = await self.client.complete(request, reason)
                call["response"] = response
                return response
            except Exception as error:
                call["error"] = {
                    "type": type(error).__name__,
                    "message": str(error),
                }
                raise
            finally:
                call["latency_seconds"] = round(time.monotonic() - started, 3)
                call["usage"] = {
                    str(key): value.model_dump() for key, value in usage.items()
                }
                self.calls.append(call)
                if parent_usage is not None:
                    for purpose, tokens in usage.items():
                        parent_usage[purpose] = (
                            parent_usage.get(purpose, UsageTokens()) + tokens
                        )


class ReplayClock:
    def __init__(self, now: float = 1000, wall_now: float = 1000) -> None:
        self.now = now
        self.wall_now = wall_now

    def __call__(self) -> float:
        return self.now

    def wall(self) -> float:
        return self.wall_now

    def advance(self, seconds: float, wall_seconds: float | None = None) -> None:
        wall_seconds = seconds if wall_seconds is None else wall_seconds
        if not math.isfinite(seconds) or seconds < 0 or not math.isfinite(wall_seconds):
            raise ValueError(
                "Replay time must be finite and monotonic time cannot decrease"
            )
        self.now += seconds
        self.wall_now += wall_seconds


class ReplayProcesses(ProcessTracker):
    """Supply process observations without probing the host machine."""

    def __init__(self, clock: ReplayClock, interval: float) -> None:
        self.statuses: dict[ProcessIdentity, ProcessStatus] = {}
        super().__init__(
            clock=clock,
            interval=interval,
            probe_process=lambda identity: self.statuses.get(
                identity, ProcessStatus.UNKNOWN
            ),
        )


class _Capture(logging.Handler):
    users = 0
    previous_level = logging.NOTSET

    def __init__(self, replay: ReplayHarness) -> None:
        super().__init__()
        self.replay = replay

    def start(self) -> None:
        logger = logging.getLogger("pet-daemon")
        if _Capture.users == 0:
            _Capture.previous_level = logger.level
            logger.setLevel(logging.DEBUG)
        _Capture.users += 1
        logger.addHandler(self)

    def stop(self) -> None:
        logger = logging.getLogger("pet-daemon")
        logger.removeHandler(self)
        _Capture.users -= 1
        if _Capture.users == 0:
            logger.setLevel(_Capture.previous_level)
        self.close()

    def emit(self, record: logging.LogRecord) -> None:
        if (
            not isinstance(record.msg, dict)
            or record.msg.get("replay_id") != self.replay.id
        ):
            return
        event = deepcopy(record.msg)
        for field in ("timestamp", "monotonic_seconds", "run_id", "replay_id"):
            event.pop(field, None)
        event["at"] = self.replay.clock()
        event["wall_time"] = self.replay.clock.wall()
        exc_info = record.exc_info or event.get("exc_info")
        if exc_info is True:
            exc_info = sys.exc_info()
        if isinstance(exc_info, tuple) and exc_info[1]:
            error = exc_info[1]
            event["exception"] = {"type": type(error).__name__, "message": str(error)}
        self.replay.events.append(event)


class ReplayHarness:
    """Run production application paths with explicit external inputs."""

    def __init__(
        self,
        config: Config | None = None,
        client: ModelClient | None = None,
        *,
        seed: int = 0,
        wall_time: float = 1000,
        monotonic_time: float = 1000,
        memory_file: Path | None = None,
        pack: Pack | None = None,
    ) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="pet-replay-")
        directory = Path(self.temporary.name)
        if config is None:
            pack = pack or self._fixture_pack(directory / "sprites")
            config = Config(pet_dir=pack.directory)
        self.config = config.model_copy(
            update={"memory_file": memory_file or directory / "memory.sqlite3"}
        )
        self.pack = pack or Pack(self.config.pet_dir)
        self.seed = seed
        self.clock = ReplayClock(monotonic_time, wall_time)
        offline = FixtureModel()
        self.fixture_model = offline if client is None else None
        self.model = RecordingModel(client if client is not None else offline)
        self.packets: list[OverlayPacket] = []
        self.events: list[dict[str, Any]] = []
        self.id = str(id(self))
        self._context = structlog.contextvars.bound_contextvars(replay_id=self.id)
        self._capture = _Capture(self)
        self._build()

    @staticmethod
    def _fixture_pack(directory: Path) -> Pack:
        directory.mkdir()
        (directory / "pixel.png").write_bytes(
            base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII="
            )
        )
        from .behavior.policy import ANIMATION_ROLES

        roles = {
            **ANIMATION_ROLES,
            "walk-left": "walk-left",
            "walk-right": "walk-right",
            "sleeping": "sleep",
        }
        (directory / "manifest.json").write_text(
            json.dumps(
                {
                    "pet": "replay",
                    "rest": "idle",
                    "clips": {
                        identifier: {"frames": ["pixel.png"], "fps": 1, "role": role}
                        for identifier, role in roles.items()
                    },
                }
            )
        )
        return Pack(directory)

    def _build(self) -> None:
        rng = random.Random(self.seed)
        config = self.config
        self.memory = Memory(
            config.memory_file,
            self.pack.pet,
            self.clock.wall,
            config.memory_retention_days,
            config.memory_max_events,
            speech_history_seconds=config.speech_history_seconds,
        )
        self.processes = ReplayProcesses(self.clock, config.process_probe_seconds)
        self.app = Application(
            config,
            sessions=Sessions(
                self.clock,
                config.stale_seconds,
                config.completed_retention_seconds,
                config.activity_fresh_seconds,
                wall_clock=self.clock.wall,
            ),
            reactions=SessionReactions(),
            attention=Attention(config, self.clock, rng, wall_clock=self.clock.wall),
            behavior=Behavior(self.pack, config.behavior, self.clock, rng, self._emit),
            cognition=Cognition(
                self.model, config, memory=self.memory, clock=self.clock.wall
            ),
            providers=(),
            processes=self.processes,
            memory=self.memory,
            emit=self._emit,
            clock=self.clock,
            rng=rng,
        )
        self.receive(
            {
                "kind": "renderer",
                "event": "status",
                "x": 100,
                "y": 100,
                "width": 200,
                "height": 160,
                "screen": {"x": 0, "y": 0, "width": 1000, "height": 800},
                "pointer": {"x": 2000, "y": 2000},
                "dragging": False,
                "menu": False,
                "speech": False,
                "sleeping": False,
                "system_sleeping": False,
            }
        )

    def _emit(self, packet: OverlayPacket) -> None:
        self.packets.append(deepcopy(packet))
        self.events.append(
            {
                "event": "renderer.command",
                "packet": deepcopy(packet),
                "at": self.clock(),
                "wall_time": self.clock.wall(),
            }
        )

    async def __aenter__(self) -> Self:
        self._context.__enter__()
        self._capture.start()
        return self

    async def __aexit__(self, *args: object) -> None:
        try:
            await self.app.close()
        finally:
            self.memory.close()
            self._capture.stop()
            self._context.__exit__(None, None, None)
            self.temporary.cleanup()

    def ingest(
        self, observation: HookEvent | RepositoryObservation | Mapping[str, object]
    ) -> None:
        if isinstance(observation, Mapping):
            hook = normalize_hook(observation)
            if hook is None:
                raise ValueError("Invalid replay hook")
            observation = hook
        self.app.ingest(observation)

    def receive(self, packet: Mapping[str, object]) -> None:
        self.app.receive(packet)

    async def settle(self) -> None:
        """Wait for current application tasks without advancing simulated time."""
        tasks = [task for task in (self.app.task, self.app.process_task) if task]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException) and not isinstance(
                result, asyncio.CancelledError
            ):
                raise result

    async def advance(
        self,
        seconds: float,
        *,
        wall_seconds: float | None = None,
        tick_seconds: float = 0.05,
    ) -> None:
        """Advance both clocks and execute the application's daemon ticks."""
        if not math.isfinite(tick_seconds) or tick_seconds <= 0:
            raise ValueError("tick_seconds must be positive and finite")
        start, wall_start = self.clock(), self.clock.wall()
        self.clock.advance(seconds, wall_seconds)
        target, wall_target = self.clock(), self.clock.wall()
        self.clock.now, self.clock.wall_now = start, wall_start
        if seconds == 0:
            self.clock.wall_now = wall_target
            self.app.tick()
            await self.settle()
            return
        while self.clock() < target:
            self.clock.now = min(target, self.clock() + tick_seconds)
            fraction = (self.clock() - start) / seconds
            self.clock.wall_now = wall_start + (wall_target - wall_start) * fraction
            self.app.tick()
            await self.settle()

    async def comment(
        self,
        reason: SpeechReason,
        context: str | None = None,
        trigger: Trigger | None = None,
    ) -> None:
        await self.app.comment(reason, context, trigger=trigger)

    async def restart(self) -> None:
        statuses = dict(self.processes.statuses)
        await self.app.close()
        self.memory.close()
        self._build()
        self.processes.statuses.update(statuses)

    def result(self) -> dict[str, Any]:
        """Export actual requests, outputs, and events with replay settings."""
        return json.loads(
            json.dumps(
                {
                    "seed": self.seed,
                    "model_mode": "fixture"
                    if self.fixture_model is not None
                    else "provided_client",
                    "config": self.config.model_dump(),
                    "monotonic_time": self.clock(),
                    "wall_time": self.clock.wall(),
                    "packets": self.packets,
                    "events": self.events,
                    "calls": self.model.calls,
                },
                default=json_default,
            )
        )


async def replay_sequence(
    fixture: dict[str, Any],
    *,
    client: ModelClient | None = None,
    model: str | None = None,
    settings: Config | None = None,
) -> dict[str, Any]:
    """Adapt reason-driven commentary fixtures to production application methods."""
    steps = []
    async with ReplayHarness(settings, client) as replay:
        if model:
            replay.config.model = model
        for index, event in enumerate(fixture["events"]):
            replay.clock.advance(
                event.get("after_seconds", 1), event.get("wall_seconds")
            )
            if event.get("restart"):
                await replay.restart()
            packet_start = len(replay.packets)
            for packet in event.get("packets", []):
                replay.ingest(packet)
            if "reason" not in event:
                continue
            if replay.fixture_model is not None:
                replay.fixture_model.speech = event.get("candidate_speech")
            reason = SpeechReason(event["reason"])
            current = replay.app.context()
            selection = replay.app.attention.select(current, reason)
            step: dict[str, Any] = {
                "step": index,
                "baseline": event.get("baseline"),
                "review": event.get("review", fixture.get("review", "")),
                "expected_speech": event.get("expected_speech"),
            }
            call_start, event_start = len(replay.model.calls), len(replay.events)
            if not replay.app.attention.is_selection_eligible(selection, current):
                step.update(speech=None, decision="unchanged_work", accepted=False)
            else:
                await replay.app.comment(reason, snapshot=current, selection=selection)
                observed = replay.events[event_start:]
                finished = next(
                    item
                    for item in reversed(observed)
                    if item["event"] == "observation.finished"
                )
                outcome = finished["outcome"]
                packet = next(
                    (
                        item
                        for item in reversed(replay.packets[packet_start:])
                        if "delivery_id" in item
                    ),
                    None,
                )
                speech = packet["text"] if packet else None
                accepted = False
                if packet and event.get("accepted", True):
                    identifier = packet["delivery_id"]
                    replay.receive(
                        {
                            "kind": "renderer",
                            "event": "speech_accepted",
                            "delivery_id": identifier,
                        }
                    )
                    outcome_record = replay.memory.message_outcome(identifier)
                    accepted = bool(
                        outcome_record
                        and outcome_record.outcome == DeliveryStatus.ACCEPTED
                    )
                decision = {
                    "submitted": "speech",
                    "silent": "silence",
                    "failed": "error",
                }.get(outcome, outcome)
                step.update(speech=speech, decision=decision, accepted=accepted)
                if outcome == "deferred":
                    deferred = next(
                        item
                        for item in reversed(observed)
                        if item["event"] == "observation.deferred"
                    )
                    step["error"] = {
                        "type": "StoryUnavailable",
                        "message": deferred["reason"],
                    }
                elif outcome == "failed":
                    failed = next(
                        item
                        for item in reversed(observed)
                        if item["event"] == "observation.failed"
                    )
                    step["error"] = failed["exception"]
            calls = deepcopy(replay.model.calls[call_start:])
            if client is None:
                for call in calls:
                    call["latency_seconds"] = 0
            step["calls"] = calls
            packets = deepcopy(replay.packets[packet_start:])
            for packet in packets:
                if "delivery_id" in packet:
                    packet["delivery_id"] = f"replay-{index}"
            step["packets"] = packets
            steps.append(step)
    return {
        "name": fixture["name"],
        "mode": "dry_run" if client is None else "model_run",
        "scope": "forced_commentary",
        "steps": steps,
    }


async def run(args: argparse.Namespace) -> None:
    fixture_paths = (
        [args.fixtures]
        if args.fixtures.is_file()
        else sorted(args.fixtures.glob("*.json"))
    )
    fixtures = [json.loads(path.read_text()) for path in fixture_paths]
    if not fixtures:
        raise ValueError("No replay fixtures found")
    samples = args.samples if args.model_run else 1
    settings = Config.load(args.config) if args.config else None
    client = None
    anthropic = None
    if args.model_run:
        from anthropic import AsyncAnthropic

        anthropic = AsyncAnthropic()
        client = AnthropicModelClient(anthropic)
    try:
        results = []
        for fixture in fixtures:
            for sample in range(samples):
                result = await replay_sequence(
                    fixture, client=client, model=args.model, settings=settings
                )
                result["sample"] = sample + 1
                results.append(result)
        rendered = json.dumps(results, indent=2, ensure_ascii=False)
        if args.model_run or args.output:
            destination = args.output or Path(".cache/replays") / time.strftime(
                "%Y%m%d-%H%M%S"
            )
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "comparison.json").write_text(rendered + "\n")
            lines = [
                "# Commentary replay",
                "",
                "Fixture reasons request commentary directly. Use ReplayHarness.advance for automatic scheduling and delivery timing.",
                "",
                "Review grounding, scope, voice, and whether silence fits.",
                "",
            ]
            for result in results:
                lines.extend([f"## {result['name']} · sample {result['sample']}", ""])
                for step in result["steps"]:
                    lines.extend(
                        [
                            f"- Baseline: {step['baseline'] or '(none recorded)'}",
                            f"- Candidate: {step['speech'] or '(silence)'}",
                            f"- Decision: {step['decision']}",
                            *(
                                [
                                    f"- Error: {step['error']['type']}: {step['error']['message']}"
                                ]
                                if "error" in step
                                else []
                            ),
                            f"- Review: {step['review']}",
                            "",
                        ]
                    )
            (destination / "comparison.md").write_text("\n".join(lines))
            print(destination)
        else:
            print(rendered)
    finally:
        if anthropic:
            await anthropic.close()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "fixtures", type=Path, nargs="?", default=Path("Tests/fixtures/replay")
    )
    parser.add_argument(
        "--model-run", action="store_true", help="Call the configured API model"
    )
    parser.add_argument("--model", help="Override the configured model")
    parser.add_argument(
        "--config", type=Path, help="Use this daemon configuration for prompts"
    )
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument(
        "--output", type=Path, help="Write comparison files into this directory"
    )
    args = parser.parse_args(argv)
    if args.samples < 1:
        parser.error("--samples must be positive")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
