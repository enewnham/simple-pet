import json
import math
from dataclasses import dataclass, field, fields
from pathlib import Path


DEFAULT_PERSONA = """
You are deadpan and slightly snarky.
You have the energy of a seasoned peer tackling the work: focused, wry about your own
stubborn bugs, and fundamentally on the developer's side. Never be cruel or insult them.
""".strip()

OBSERVATION_REASONS = frozenset({"prompt", "completion", "routine", "failure", "check_in", "click", "explicit", "care"})


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def interval(value):
    return isinstance(value, (list, tuple)) and len(value) == 2 and all(positive(v) for v in value) and value[0] <= value[1]


@dataclass
class BehaviorConfig:
    roaming: bool = True
    roaming_radius: float = 160
    automatic_speech_cooldown_seconds: float = 90
    naps: bool = True
    care: bool = True
    care_interval: tuple = (300, 600)
    care_speech_interval: tuple = (2700, 5400)
    nap_interval: tuple = (120, 240)
    nap_seconds: tuple = (20, 45)
    repeat_range: tuple = (1, 3)
    repeat_budget: float = 4
    gesture_cooldown: float = 15
    idle_greeting_seconds: float = 300
    baseline_dwell: float = 2
    idle_debounce: float = 1
    error_seconds: float = 2
    walk_seconds: tuple = (2, 6)
    walk_speed: float = 40
    idle_pause: tuple = (20, 60)
    working_pause: tuple = (2, 5)
    work_burst_seconds: tuple = (12, 25)
    pointer_radius: float = 100
    resume_seconds: float = 3
    gaze_reset_seconds: float = 1.2
    gaze_deadzone: float = 24

    def validate(self):
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name in ("roaming", "naps", "care"):
                valid = type(value) is bool
            elif item.name == "automatic_speech_cooldown_seconds":
                valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
            elif item.name == "repeat_range":
                valid = interval(value) and all(type(v) is int for v in value)
            elif isinstance(value, (list, tuple)):
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
    summarize_on: list[str] = field(default_factory=lambda: ["prompt"])
    summarize_first_request_only: bool = False
    observation_seconds: tuple = (20, 40)
    idle_message_seconds: float = 300
    prompt_max_bytes: int = 16000
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
    def load(cls, path):
        path = Path(path).resolve()
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            raise ValueError("Configuration must be a JSON object")
        behavior = BehaviorConfig(**data.pop("behavior", {}))
        config = cls(**data, behavior=behavior)
        for name in ("pet_dir", "overlay_binary", "log_file", "memory_file"):
            value = Path(getattr(config, name)).expanduser()
            setattr(config, name, (path.parent / value).resolve())
        config.validate()
        return config

    def validate(self):
        self.behavior.validate()
        if not isinstance(self.summarize_on, list) or any(
            not isinstance(reason, str) or reason not in OBSERVATION_REASONS for reason in self.summarize_on
        ):
            raise ValueError("summarize_on must be an array of commentary reasons")
        if type(self.summarize_first_request_only) is not bool:
            raise ValueError("summarize_first_request_only must be a boolean")
        if not interval(self.observation_seconds):
            raise ValueError("observation_seconds must be a positive [minimum, maximum]")
        for name in ("prompt_max_bytes", "memory_retention_days", "memory_max_events", "max_tokens"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("idle_message_seconds", "speech_history_seconds", "speech_ttl", "request_timeout", "stale_seconds", "completed_retention_seconds", "process_probe_seconds", "activity_fresh_seconds", "repository_refresh_seconds", "github_refresh_seconds"):
            if not positive(getattr(self, name)):
                raise ValueError(f"{name} must be positive")
        if self.prompt_max_bytes < 8000:
            raise ValueError("prompt_max_bytes must be at least 8000")
        if not isinstance(self.persona, str) or not self.persona.strip():
            raise ValueError("persona must be a nonempty string")
        if len(self.persona.encode("utf-8")) > 4000:
            raise ValueError("persona must fit within 4000 UTF-8 bytes")
        if not isinstance(self.model, str) or not self.model:
            raise ValueError("model must be nonempty")
        for path in (self.daemon_socket, self.overlay_socket):
            if not isinstance(path, str) or not path or "\0" in path or len(path.encode()) >= 104:
                raise ValueError("Socket paths must fit the macOS UNIX socket address")
        if self.daemon_socket == self.overlay_socket:
            raise ValueError("Daemon and overlay sockets must differ")
