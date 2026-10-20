import json
from dataclasses import dataclass
from pathlib import Path

from .config import positive


@dataclass(frozen=True)
class Clip:
    id: str
    frames: tuple
    fps: float
    role: str
    max_repeats: int | None = None

    @property
    def seconds(self):
        return len(self.frames) / self.fps


class Pack:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        data = json.loads((self.directory / "manifest.json").read_text())
        self.pet = data["pet"]
        self.rest = data["rest"]
        self.clips = {}
        for key, value in data["clips"].items():
            if not isinstance(key, str) or not key or not positive(value["fps"]):
                raise ValueError("Sprite clips require an ID and positive FPS")
            frames = value["frames"]
            if not isinstance(frames, list) or not frames:
                raise ValueError(f"Clip {key} has no frames")
            for frame in frames:
                file = (self.directory / frame).resolve()
                if Path(frame).is_absolute() or not file.is_relative_to(self.directory) or file.suffix.lower() != ".png" or not file.is_file():
                    raise ValueError(f"Invalid frame in clip {key}")
            maximum = value.get("max_repeats")
            if maximum is not None and (type(maximum) is not int or maximum <= 0):
                raise ValueError(f"Invalid repeat limit in clip {key}")
            role = value["role"]
            if not isinstance(role, str) or not role:
                raise ValueError(f"Invalid role in clip {key}")
            self.clips[key] = Clip(key, tuple(frames), value["fps"], role, maximum)
        if not isinstance(self.pet, str) or not self.pet or self.rest not in self.clips:
            raise ValueError("Sprite pack requires a name and rest pose")
        self.gaze = data.get("gaze", [])
        if self.gaze and (len(self.gaze) != 16 or any(key not in self.clips for key in self.gaze)):
            raise ValueError("Gaze requires sixteen existing poses")
        self.gaze_anchor = data.get("gazeAnchor", {"x": 0.5, "y": 0.5})
        if not isinstance(self.gaze_anchor, dict) or any(type(self.gaze_anchor.get(k)) not in (int, float) or not 0 <= self.gaze_anchor[k] <= 1 for k in ("x", "y")):
            raise ValueError("Gaze anchor must contain normalized x and y")

    def role(self, *roles):
        return [clip for clip in self.clips.values() if clip.role in roles]
