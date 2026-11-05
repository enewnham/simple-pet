import math

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..coordination.models import OperationKind


class DesktopConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str = Field(min_length=1, max_length=120)
    argv: tuple[str, ...]
    capabilities: tuple[OperationKind, ...] = (
        OperationKind.LAUNCH,
        OperationKind.INSPECT,
    )
    timeout_seconds: float = Field(default=30, gt=0)

    @field_validator("argv")
    @classmethod
    def executable_arguments(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or any(not argument or "\0" in argument for argument in value):
            raise ValueError(
                "Configure a nonempty adapter executable and argument vector"
            )
        return value

    @field_validator("timeout_seconds")
    @classmethod
    def finite_timeout(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("Adapter timeouts must be finite")
        return value

    @model_validator(mode="after")
    def inspection_required(self) -> DesktopConfig:
        if (
            OperationKind.LAUNCH in self.capabilities
            and OperationKind.INSPECT not in self.capabilities
        ):
            raise ValueError("Automatic launch requires adapter inspection")
        return self
