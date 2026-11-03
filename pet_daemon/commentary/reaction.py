from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..attention import AttentionSelection
from ..domain import SpeechReason
from .templates import render


class Response(BaseModel):
    model_config = ConfigDict(extra="forbid")
    speak: str | None = Field(
        description=render("schema.jinja", field="speak_optional")
    )
    mentions: tuple[str, ...] = Field(
        default=(), description=render("schema.jinja", field="speech_mentions")
    )


class RequiredResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    speak: str = Field(description=render("schema.jinja", field="speak_required"))
    mentions: tuple[str, ...] = Field(
        default=(), description=render("schema.jinja", field="speech_mentions")
    )

    @field_validator("speak")
    @classmethod
    def nonblank_speech(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Requested commentary requires speech")
        return value


def response_type(selection: AttentionSelection) -> type[Response | RequiredResponse]:
    return (
        RequiredResponse
        if selection.context is not None or selection.reason == SpeechReason.EXPLICIT
        else Response
    )
