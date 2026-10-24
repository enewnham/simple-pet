from pathlib import Path
from typing import Annotated, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    field_validator,
    model_validator,
)

DEFAULT_PERSONA = """
You are an interested, dry, understated companion working alongside the developer.
Speak warmly and plainly. Humor is optional and grounded in the actual work.
Never sound reluctant, invent trouble, or force a punchline.
""".strip()


type Positive = Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)]
type Nonnegative = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]
type PositiveInt = Annotated[int, Field(strict=True, gt=0)]


def ordered_interval(value: tuple[float, float]) -> tuple[float, float]:
    lower, upper = value
    if lower > upper:
        raise ValueError("Expected a positive [minimum, maximum]")
    return value


type Interval = Annotated[tuple[Positive, Positive], AfterValidator(ordered_interval)]


class BehaviorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    roaming: StrictBool = True
    roaming_radius: Positive = 160
    automatic_speech_cooldown_seconds: Nonnegative = 90
    naps: StrictBool = True
    care: StrictBool = True
    sentiment_reactions: StrictBool = True
    care_interval: Interval = (300, 600)
    care_speech_interval: Interval = (2700, 5400)
    nap_interval: Interval = (120, 240)
    nap_seconds: Interval = (20, 45)
    baseline_dwell: Positive = 2
    error_seconds: Positive = 2
    walk_seconds: Interval = (2, 6)
    walk_speed: Positive = 40
    idle_pause: Interval = (20, 60)
    working_pause: Interval = (2, 5)
    work_burst_seconds: Interval = (12, 25)
    pointer_radius: Positive = 100
    resume_seconds: Positive = 3
    gaze_reset_seconds: Positive = 1.2
    gaze_deadzone: Positive = 24


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pet_dir: Path
    persona: str = DEFAULT_PERSONA
    log_file: Path = Path("logs/pet-daemon.jsonl")
    memory_file: Path = Path("state/pet-memory.sqlite3")
    overlay_binary: Path = Path(".build/pet-overlay")
    daemon_socket: str = "/tmp/pet-daemon.sock"
    overlay_socket: str = "/tmp/pet.sock"
    model: str = "claude-haiku-5-5"
    observation_seconds: Interval = (20, 40)
    idle_message_seconds: Positive = 300
    prompt_target_bytes: PositiveInt = 16000
    memory_retention_days: PositiveInt = 30
    memory_max_events: PositiveInt = 10000
    speech_history_seconds: Positive = 1800
    speech_ttl: Positive = 15
    request_timeout: Positive = 20
    process_probe_seconds: Positive = 10
    activity_fresh_seconds: Positive = 300
    repository_refresh_seconds: Positive = 30
    github_refresh_seconds: Positive = 180
    stale_seconds: Positive = 3600
    completed_retention_seconds: Positive = 7200
    max_tokens: PositiveInt = 512
    behavior: BehaviorConfig = Field(default_factory=BehaviorConfig)

    @classmethod
    def load(cls, path: str | Path) -> Config:
        path = Path(path).resolve()
        config = cls.model_validate_json(path.read_text())
        for name in ("pet_dir", "overlay_binary", "log_file", "memory_file"):
            value = Path(getattr(config, name)).expanduser()
            setattr(config, name, (path.parent / value).resolve())
        return config

    @field_validator("persona")
    @classmethod
    def nonempty_persona(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("persona must be a nonempty string")
        return value

    @field_validator("model")
    @classmethod
    def nonempty_model(cls, value: str) -> str:
        if not value:
            raise ValueError("model must be nonempty")
        return value

    @field_validator("daemon_socket", "overlay_socket")
    @classmethod
    def socket_path(cls, value: str) -> str:
        if not value or "\0" in value or len(value.encode()) >= 104:
            raise ValueError("Socket paths must fit the macOS UNIX socket address")
        return value

    @model_validator(mode="after")
    def distinct_sockets(self) -> Self:
        if self.daemon_socket == self.overlay_socket:
            raise ValueError("Daemon and overlay sockets must differ")
        return self
