import base64
import json
import math
import random
from itertools import pairwise
from pathlib import Path

import pytest
from pydantic import BaseModel

from pet_daemon.attention import Attention
from pet_daemon.behavior.controller import Behavior
from pet_daemon.behavior.sprites import Pack
from pet_daemon.commentary.cognition import CannotRespond, Cognition
from pet_daemon.commentary.contracts import ModelRequest
from pet_daemon.commentary.request import encoded_size
from pet_daemon.config import Config
from pet_daemon.domain import CareKind, ModelPurpose, ProcessStatus, SpeechReason
from pet_daemon.runtime import CareTrigger, PetContext
from pet_daemon.sessions.tracking import Sessions
from pet_daemon.storage import Memory
from Tests.fixtures import capture_snapshot, require

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jWZkAAAAASUVORK5CYII="
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class Clock:
    now = 0.0

    def __call__(self) -> float:
        return self.now


class Model:
    def __init__(self) -> None:
        self.calls: list[tuple[SpeechReason | ModelPurpose, ModelRequest]] = []
        self.response: dict[str, object] = {"speak": None}

    async def complete(
        self,
        request: ModelRequest,
        reason: SpeechReason | ModelPurpose,
        *,
        response_type: type[BaseModel],
    ) -> dict[str, object]:
        self.calls.append((reason, request))
        return self.response


class IdlePet:
    def __init__(self, root: Path) -> None:
        self.clock = Clock()
        self.config = Config(pet_dir=root)
        self.sessions = Sessions(self.clock, wall_clock=self.clock)
        update = require(
            self.sessions.handle(
                {
                    "source": "claude",
                    "session": "idle-terminal",
                    "event": "intent",
                    "goal": "PRIVATE_REPOSITORY_DETAIL " * 500,
                }
            )
        )
        self.session = self.sessions.by_id[update.session]
        self.session.presence = ProcessStatus.ALIVE
        self.memory = Memory(root / "memory.sqlite3", "idle-pet", clock=self.clock)
        with self.memory.transaction():
            self.memory.work.record_session(update.observation)
            self.memory.observe_session_context(update.observation)
        self.clock.now = 600
        self.attention = Attention(
            self.config, self.clock, random.Random(4), wall_clock=self.clock
        )
        self.model = Model()
        self.cognition = Cognition(
            self.model, self.config, memory=self.memory, clock=self.clock
        )

    def snapshot(self) -> PetContext:
        activity = self.sessions.activity()
        return capture_snapshot(
            activity, memory=self.memory, work=self.memory.connected_work(activity)
        )

    def care_snapshot(self) -> PetContext:
        return self.snapshot().model_copy(
            update={"trigger": CareTrigger(care=CareKind.ATTENTION)}
        )


@pytest.mark.anyio
@pytest.mark.parametrize("outcome", ["silent", "malformed", "discarded"])
async def test_eight_idle_hours_only_generate_infrequent_small_care_requests(
    tmp_path: Path,
    outcome: str,
) -> None:
    pet = IdlePet(tmp_path)
    if outcome == "malformed":
        pet.model.response = {"speak": "Quiet.", "mentions": ["unavailable"]}
    elif outcome == "discarded":
        pet.model.response = {"speak": "A little stretch."}
    try:
        (tmp_path / "pose.png").write_bytes(PNG)
        (tmp_path / "manifest.json").write_text(
            json.dumps(
                {
                    "pet": "idle-pet",
                    "rest": "rest",
                    "clips": {
                        role: {"frames": ["pose.png"], "fps": 1, "role": role}
                        for role in ("rest", "sleep", "celebrate", "greeting", "error")
                    },
                }
            )
        )
        packets = []
        behavior = Behavior(
            Pack(tmp_path),
            pet.config.behavior,
            pet.clock,
            random.Random(5),
            packets.append,
        )
        behavior.update_renderer_status(
            {
                "x": 100,
                "y": 100,
                "width": 200,
                "height": 160,
                "screen": {"x": 0, "y": 0, "width": 1000, "height": 800},
                "pointer": {"x": 900, "y": 700},
                "dragging": False,
                "menu": False,
                "speech": False,
                "sleeping": False,
                "system_sleeping": False,
            }
        )
        started = pet.clock()
        times = []
        while pet.clock() - started < 8 * 3600:
            pet.clock.now += 30
            snapshot = pet.snapshot()
            behavior.tick(snapshot.activity, pet.sessions)
            snapshot = snapshot.model_copy(
                update={"sleeping": behavior.nap_until is not None}
            )
            request = pet.attention.request(
                snapshot,
                check_in=pet.sessions.check_in_due(pet.config.idle_message_seconds),
                care=behavior.care_remark,
            )
            if request is None:
                continue
            assert request.reason == SpeechReason.CARE
            snapshot = snapshot.model_copy(update={"trigger": request.trigger})
            selection = pet.attention.select(snapshot, request.reason)
            pet.attention.mark_scheduled()
            behavior.care_remark = None
            if outcome == "malformed":
                with pytest.raises(ValueError):
                    await pet.cognition.generate_commentary(snapshot, selection)
            else:
                result = await pet.cognition.generate_commentary(snapshot, selection)
                assert result.request.selection.subject is None
            times.append(pet.clock())
        assert any(packet.get("sleeping") is True for packet in packets)
        assert 5 <= len(times) <= 10
        assert len(pet.model.calls) == len(times)
        assert not pet.memory.speech()
        assert all(later - earlier >= 2700 for earlier, later in pairwise(times))
        assert all(reason == SpeechReason.CARE for reason, _ in pet.model.calls)
        for _, request in pet.model.calls:
            api_payload = {
                key: value for key, value in request.items() if key != "timeout"
            }
            estimated_tokens = math.ceil(encoded_size(api_payload) / 4)
            assert 500 <= estimated_tokens <= 1000
            assert request["max_tokens"] <= 128
            assert "PRIVATE_REPOSITORY_DETAIL" not in json.dumps(request)
    finally:
        pet.memory.close()


def test_stale_open_turn_ages_out_but_verified_live_tool_stays_engaged(
    tmp_path: Path,
) -> None:
    pet = IdlePet(tmp_path)
    try:
        assert not pet.sessions.is_engaged
        assert not pet.sessions.is_working
        assert not pet.sessions.check_in_due(300)
        assert pet.session.active
        pet.sessions.handle(
            {
                "source": "claude",
                "session": "idle-terminal",
                "event": "tool_start",
                "call": "build",
                "tool": "Bash",
            }
        )
        pet.clock.now += 8 * 3600
        assert pet.sessions.is_engaged
        assert pet.sessions.is_working
        assert pet.attention.is_working(pet.snapshot().sessions[0])
    finally:
        pet.memory.close()


@pytest.mark.anyio
async def test_optional_care_uses_one_attempt_and_ignores_work_drift(
    tmp_path: Path,
) -> None:
    pet = IdlePet(tmp_path)
    try:
        snapshot = pet.care_snapshot()
        selection = pet.attention.select(snapshot, SpeechReason.CARE)
        result = await pet.cognition.generate_commentary(snapshot, selection)
        assert result.request.conditions.is_valid(
            snapshot.model_copy(update={"work": ()})
        )
        sleeping = snapshot.model_copy(update={"sleeping": True})
        assert pet.attention.request(sleeping, care=CareKind.ATTENTION) is None
        with pytest.raises(CannotRespond):
            await pet.cognition.generate_commentary(sleeping, selection)
        assert len(pet.model.calls) == 1
        pet.model.response = {"speak": "Quiet moment.", "mentions": ["invented-work"]}
        with pytest.raises(ValueError):
            await pet.cognition.generate_commentary(snapshot, selection)
        assert len(pet.model.calls) == 2
        pet.sessions.handle(
            {
                "source": "claude",
                "session": "idle-terminal",
                "event": "intent",
                "goal": "Fresh work",
            }
        )
        assert not result.request.conditions.is_valid(pet.snapshot())
    finally:
        pet.memory.close()
