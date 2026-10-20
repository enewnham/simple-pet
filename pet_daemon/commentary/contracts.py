from collections.abc import Sequence
from typing import Literal, Protocol, TypedDict

from anthropic.types import TextBlockParam, ThinkingConfigDisabledParam


class UserMessage(TypedDict):
    role: Literal["user"]
    content: str


class TextSchema(TypedDict, total=False):
    type: str
    anyOf: list[dict[str, str]]
    description: str


class JSONSchema(TypedDict):
    type: Literal["object"]
    properties: dict[str, TextSchema]
    required: list[str]
    additionalProperties: bool


class OutputFormat(TypedDict):
    type: Literal["json_schema"]
    schema: JSONSchema


class OutputConfig(TypedDict):
    effort: Literal["low"]
    format: OutputFormat


class ModelRequest(TypedDict):
    model: str
    max_tokens: int
    timeout: float
    thinking: ThinkingConfigDisabledParam
    system: Sequence[TextBlockParam]
    messages: list[UserMessage]
    output_config: OutputConfig


class RequestFactory(Protocol):
    def __call__(
        self,
        payload: object,
        system: Sequence[TextBlockParam],
        speech_required: bool = False,
    ) -> ModelRequest: ...
