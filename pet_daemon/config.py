import json
import math
from dataclasses import dataclass, field, fields
from pathlib import Path


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def interval(value):
    return isinstance(value, (list, tuple)) and len(value) == 2 and all(positive(v) for v in value) and value[0] <= value[1]


@dataclass
class BehaviorConfig:
    roaming: bool = True
    repeat_range: tuple = (1, 3)
    repeat_budget: float = 4
    gesture_cooldown: float = 15
    baseline_dwell: float = 2
    idle_debounce: float = 1
    error_seconds: float = 2
    walk_seconds: tuple = (2, 6)
    walk_speed: float = 40
    idle_pause: tuple = (20, 60)
    working_pause: tuple = (8, 20)
    work_burst_seconds: tuple = (2, 4)
    pointer_radius: float = 100
    resume_seconds: float = 3
    gaze_reset_seconds: float = 1.2
    gaze_deadzone: float = 24

    def validate(self):
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name == "roaming":
                valid = type(value) is bool
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
    log_file: Path = Path("logs/pet-daemon.jsonl")
    overlay_binary: Path = Path(".build/pet-overlay")
    daemon_socket: str = "/tmp/pet-daemon.sock"
    overlay_socket: str = "/tmp/pet.sock"
    model: str = "claude-haiku-5-5"
    observation_seconds: tuple = (20, 40)
    history_pairs: int = 8
    speech_ttl: float = 5
    request_timeout: float = 20
    stale_seconds: float = 3600
    completed_retention_seconds: float = 7200
    max_tokens: int = 4096
    behavior: BehaviorConfig = field(default_factory=BehaviorConfig)

    @classmethod
    def load(cls, path):
        path = Path(path).resolve()
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            raise ValueError("Configuration must be a JSON object")
        behavior = BehaviorConfig(**data.pop("behavior", {}))
        config = cls(**data, behavior=behavior)
        for name in ("pet_dir", "overlay_binary", "log_file"):
            value = Path(getattr(config, name)).expanduser()
            setattr(config, name, (path.parent / value).resolve())
        config.validate()
        return config

    def validate(self):
        self.behavior.validate()
        if not interval(self.observation_seconds):
            raise ValueError("observation_seconds must be a positive [minimum, maximum]")
        for name in ("history_pairs", "max_tokens"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("speech_ttl", "request_timeout", "stale_seconds", "completed_retention_seconds"):
            if not positive(getattr(self, name)):
                raise ValueError(f"{name} must be positive")
        if not isinstance(self.model, str) or not self.model:
            raise ValueError("model must be nonempty")
        for path in (self.daemon_socket, self.overlay_socket):
            if not isinstance(path, str) or not path or "\0" in path or len(path.encode()) >= 104:
                raise ValueError("Socket paths must fit the macOS UNIX socket address")
        if self.daemon_socket == self.overlay_socket:
            raise ValueError("Daemon and overlay sockets must differ")
