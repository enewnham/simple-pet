import json
from dataclasses import dataclass
from pathlib import Path

from ..config import positive
from ..runtime import Point


@dataclass(frozen=True)
class Clip:
    id: str
    frames: tuple[str, ...]
    fps: float
    role: str
    max_repeats: int | None = None

    @property
    def seconds(self) -> float:
        return len(self.frames) / self.fps


class Pack:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory).resolve()
        data: object = json.loads((self.directory / "manifest.json").read_text())
        if not isinstance(data, dict):
            raise TypeError("Sprite manifest must be an object")
        pet: object = data.get("pet")
        rest: object = data.get("rest")
        clips: object = data.get("clips")
        if not isinstance(clips, dict):
            raise TypeError("Sprite manifest requires clip definitions")
        self.clips: dict[str, Clip] = {}
        for key, value in clips.items():
            if not isinstance(key, str) or not key or not isinstance(value, dict):
                raise ValueError("Sprite clips require an ID and clip definition")
            fps: object = value.get("fps")
            if not positive(fps):
                raise ValueError("Sprite clips require an ID and positive FPS")
            frames: object = value.get("frames")
            if not isinstance(frames, list) or not frames:
                raise ValueError(f"Clip {key} has no frames")
            paths: list[str] = []
            for frame in frames:
                if not isinstance(frame, str):
                    raise TypeError(f"Invalid frame in clip {key}")
                file = (self.directory / frame).resolve()
                if (
                    Path(frame).is_absolute()
                    or not file.is_relative_to(self.directory)
                    or file.suffix.lower() != ".png"
                    or not file.is_file()
                ):
                    raise ValueError(f"Invalid frame in clip {key}")
                paths.append(frame)
            maximum: object = value.get("max_repeats")
            if maximum is not None and (
                not isinstance(maximum, int)
                or isinstance(maximum, bool)
                or maximum <= 0
            ):
                raise ValueError(f"Invalid repeat limit in clip {key}")
            role: object = value.get("role")
            if not isinstance(role, str) or not role:
                raise ValueError(f"Invalid role in clip {key}")
            self.clips[key] = Clip(key, tuple(paths), fps, role, maximum)
        if (
            not isinstance(pet, str)
            or not pet
            or not isinstance(rest, str)
            or rest not in self.clips
        ):
            raise ValueError("Sprite pack requires a name and rest pose")
        self.pet = pet
        self.rest = rest
        gaze: object = data.get("gaze", [])
        if not isinstance(gaze, list) or (
            gaze
            and (
                len(gaze) != 16
                or any(
                    not isinstance(key, str) or key not in self.clips for key in gaze
                )
            )
        ):
            raise ValueError("Gaze requires sixteen existing poses")
        self.gaze: tuple[str, ...] = tuple(gaze)
        anchor: object = data.get("gazeAnchor", {"x": 0.5, "y": 0.5})
        if not isinstance(anchor, dict):
            raise TypeError("Gaze anchor must contain normalized x and y")
        x: object = anchor.get("x")
        y: object = anchor.get("y")
        if (
            not isinstance(x, (int, float))
            or isinstance(x, bool)
            or not 0 <= x <= 1
            or not isinstance(y, (int, float))
            or isinstance(y, bool)
            or not 0 <= y <= 1
        ):
            raise ValueError("Gaze anchor must contain normalized x and y")
        self.gaze_anchor = Point(x=x, y=y)

    def role(self, *roles: str) -> list[Clip]:
        return [clip for clip in self.clips.values() if clip.role in roles]
