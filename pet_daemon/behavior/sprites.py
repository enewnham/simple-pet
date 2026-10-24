from pathlib import Path
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..config import Positive, PositiveInt
from ..runtime import Point


class ManifestClip(BaseModel):
    frames: tuple[str, ...] = Field(min_length=1)
    fps: Positive
    role: str = Field(min_length=1)
    max_repeats: PositiveInt | None = None


class Clip(ManifestClip):
    id: str

    @property
    def seconds(self) -> float:
        return len(self.frames) / self.fps


class GazeAnchor(BaseModel):
    x: Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]
    y: Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]


class Manifest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    pet: str = Field(min_length=1)
    rest: str
    clips: dict[Annotated[str, Field(min_length=1)], ManifestClip]
    gaze: tuple[str, ...] = ()
    gaze_anchor: GazeAnchor = Field(
        default_factory=lambda: GazeAnchor(x=0.5, y=0.5), alias="gazeAnchor"
    )

    @model_validator(mode="after")
    def existing_poses(self) -> Self:
        if self.rest not in self.clips:
            raise ValueError("Sprite pack requires a name and rest pose")
        if self.gaze and (
            len(self.gaze) != 16 or any(key not in self.clips for key in self.gaze)
        ):
            raise ValueError("Gaze requires sixteen existing poses")
        return self


class Pack:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory).resolve()
        manifest = Manifest.model_validate_json(
            (self.directory / "manifest.json").read_text()
        )
        self.clips: dict[str, Clip] = {}
        for key, definition in manifest.clips.items():
            for frame in definition.frames:
                file = (self.directory / frame).resolve()
                if (
                    Path(frame).is_absolute()
                    or not file.is_relative_to(self.directory)
                    or file.suffix.lower() != ".png"
                    or not file.is_file()
                ):
                    raise ValueError(f"Invalid frame in clip {key}")
            self.clips[key] = Clip(id=key, **definition.model_dump())
        self.pet = manifest.pet
        self.rest = manifest.rest
        self.gaze = manifest.gaze
        self.gaze_anchor = Point(x=manifest.gaze_anchor.x, y=manifest.gaze_anchor.y)

    def role(self, *roles: str) -> list[Clip]:
        return [clip for clip in self.clips.values() if clip.role in roles]
