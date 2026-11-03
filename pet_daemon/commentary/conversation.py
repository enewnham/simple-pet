from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..attention import AttentionSelection
from ..domain import SpeechSituation
from .templates import render


class ActionChoice(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    label: str = Field(min_length=1, max_length=40)

    @field_validator("label")
    @classmethod
    def nonblank_label(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("A conversation choice requires a label")
        return value.strip()


class Exchange(BaseModel):
    model_config = ConfigDict(extra="forbid")
    headline: str = Field(max_length=100)
    context: str = Field(max_length=1000)
    choices: tuple[ActionChoice, ...] = Field(default=(), max_length=2)

    @model_validator(mode="after")
    def distinct_choices(self) -> Self:
        if len({choice.id for choice in self.choices}) != len(self.choices):
            raise ValueError("Conversation choices must be distinct")
        return self


class Response(Exchange):
    speak: str | None = Field(
        description=render("schema.jinja", field="speak_optional")
    )
    mentions: tuple[str, ...] = Field(
        default=(),
        description=render("schema.jinja", field="conversation_mentions"),
    )

    @model_validator(mode="after")
    def supported_exchange(self) -> Self:
        if self.speak is not None:
            if not self.speak.strip() or not self.headline.strip():
                raise ValueError("A spoken conversation requires a remark and headline")
        elif self.headline or self.context or self.choices:
            raise ValueError("A silent conversation cannot include displayed content")
        return self


class RecollectionResponse(Response):
    memory: str | None = Field(
        default=None, description=render("schema.jinja", field="memory")
    )
    ongoing: bool = Field(
        default=False,
        description=render("schema.jinja", field="ongoing"),
    )


def response_type(selection: AttentionSelection) -> type[Response]:
    return (
        RecollectionResponse
        if selection.situation == SpeechSituation.IDLE_REFLECTION
        else Response
    )
