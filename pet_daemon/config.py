import json
import math
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TypeGuard, get_origin

from .domain import SpeechReason

DEFAULT_PERSONA = """
You are deadpan and slightly snarky.
You have the energy of a seasoned peer tackling the work: focused, wry about your own
stubborn bugs, and fundamentally on the developer's side. Never be cruel or insult them.
""".strip()


def positive(value: object) -> TypeGuard[float]:
    return (
        (type(value) is int or type(value) is float)
        and math.isfinite(value)
        and value > 0
    )


def interval(value: object) -> TypeGuard[tuple[float, float] | list[float]]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return False
    lower, upper = value
    return positive(lower) and positive(upper) and lower <= upper


@dataclass
class BehaviorConfig:
    roaming: bool = True
    roaming_radius: float = 160
    automatic_speech_cooldown_seconds: float = 90
    naps: bool = True
    care: bool = True
    sentiment_reactions: bool = True
    care_interval: tuple[float, float] = (300, 600)
    care_speech_interval: tuple[float, float] = (2700, 5400)
    nap_interval: tuple[float, float] = (120, 240)
    nap_seconds: tuple[float, float] = (20, 45)
    baseline_dwell: float = 2
    error_seconds: float = 2
    walk_seconds: tuple[float, float] = (2, 6)
    walk_speed: float = 40
    idle_pause: tuple[float, float] = (20, 60)
    working_pause: tuple[float, float] = (2, 5)
    work_burst_seconds: tuple[float, float] = (12, 25)
    pointer_radius: float = 100
    resume_seconds: float = 3
    gaze_reset_seconds: float = 1.2
    gaze_deadzone: float = 24

    def validate(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if item.type is bool:
                valid = type(value) is bool
            elif item.name == "automatic_speech_cooldown_seconds":
                valid = (
                    type(value) in (int, float) and math.isfinite(value) and value >= 0
                )
            elif get_origin(item.type) is tuple:
                valid = interval(value)
            else:
                valid = positive(value)
            if not valid:
                raise ValueError(f"Invalid behavior.{item.name}")


@dataclass
class Config:
    pet_dir: Path
    persona: str = DEFAULT_PERSONA
    log_file: Path = Path("logs/pet-daemon.jsonl")
    memory_file: Path = Path("state/pet-memory.sqlite3")
    overlay_binary: Path = Path(".build/pet-overlay")
    daemon_socket: str = "/tmp/pet-daemon.sock"
    overlay_socket: str = "/tmp/pet.sock"
    model: str = "claude-haiku-5-5"
    summarize_on: list[SpeechReason] = field(
        default_factory=lambda: [SpeechReason.PROMPT]
    )
    summarize_first_request_only: bool = False
    observation_seconds: tuple[float, float] = (20, 40)
    idle_message_seconds: float = 300
    prompt_target_bytes: int = 16000
    memory_retention_days: int = 30
    memory_max_events: int = 10000
    speech_history_seconds: float = 1800
    speech_ttl: float = 15
    request_timeout: float = 20
    process_probe_seconds: float = 10
    activity_fresh_seconds: float = 300
    repository_refresh_seconds: float = 30
    github_refresh_seconds: float = 180
    stale_seconds: float = 3600
    completed_retention_seconds: float = 7200
    max_tokens: int = 512
    behavior: BehaviorConfig = field(default_factory=BehaviorConfig)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        path = Path(path).resolve()
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            raise ValueError("Configuration must be a JSON object")  # noqa: TRY004 - Report malformed JSON values through the validation contract.
        if "summarize_on" in data:
            reasons = data["summarize_on"]
            if not isinstance(reasons, list):
                raise ValueError("summarize_on must be an array of commentary reasons")
            try:
                data["summarize_on"] = [SpeechReason(reason) for reason in reasons]
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "summarize_on must be an array of commentary reasons"
                ) from error
        behavior = BehaviorConfig(**data.pop("behavior", {}))
        config = cls(**data, behavior=behavior)
        for name in ("pet_dir", "overlay_binary", "log_file", "memory_file"):
            value = Path(getattr(config, name)).expanduser()
            setattr(config, name, (path.parent / value).resolve())
        config.validate()
        for item in fields(behavior):
            if get_origin(item.type) is tuple:
                lower, upper = getattr(behavior, item.name)
                setattr(behavior, item.name, (lower, upper))
        lower, upper = config.observation_seconds
        config.observation_seconds = (lower, upper)
        return config

    def validate(self) -> None:
        self.behavior.validate()
        if not isinstance(self.summarize_on, list) or any(
            not isinstance(reason, SpeechReason) for reason in self.summarize_on
        ):
            raise ValueError("summarize_on must be an array of commentary reasons")
        if type(self.summarize_first_request_only) is not bool:
            raise ValueError("summarize_first_request_only must be a boolean")
        if not interval(self.observation_seconds):
            raise ValueError(
                "observation_seconds must be a positive [minimum, maximum]"
            )
        for name in (
            "prompt_target_bytes",
            "memory_retention_days",
            "memory_max_events",
            "max_tokens",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "idle_message_seconds",
            "speech_history_seconds",
            "speech_ttl",
            "request_timeout",
            "stale_seconds",
            "completed_retention_seconds",
            "process_probe_seconds",
            "activity_fresh_seconds",
            "repository_refresh_seconds",
            "github_refresh_seconds",
        ):
            if not positive(getattr(self, name)):
                raise ValueError(f"{name} must be positive")
        if not isinstance(self.persona, str) or not self.persona.strip():
            raise ValueError("persona must be a nonempty string")
        if not isinstance(self.model, str) or not self.model:
            raise ValueError("model must be nonempty")
        for path in (self.daemon_socket, self.overlay_socket):
            if (
                not isinstance(path, str)
                or not path
                or "\0" in path
                or len(path.encode()) >= 104
            ):
                raise ValueError("Socket paths must fit the macOS UNIX socket address")
        if self.daemon_socket == self.overlay_socket:
            raise ValueError("Daemon and overlay sockets must differ")
