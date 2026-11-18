from NaturalLanguage import NLLanguageEnglish, NLTokenizer, NLTokenUnitSentence  # ty: ignore[unresolved-import]
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..attention import AttentionSelection
from ..domain import SpeechReason
from .correction import CorrectableOutput
from .templates import render

OCCASION_MAX_WORDS = 20


def validate_occasion_speech(speech: str | None) -> None:
    if speech is None:
        return
    if len(speech.split()) > OCCASION_MAX_WORDS:
        raise CorrectableOutput("occasion_length", max_words=OCCASION_MAX_WORDS)
    tokenizer = NLTokenizer.alloc().initWithUnit_(NLTokenUnitSentence)
    tokenizer.setLanguage_(NLLanguageEnglish)
    tokenizer.setString_(speech)
    utf16_length = len(speech.encode("utf-16-le")) // 2
    if len(tokenizer.tokensForRange_((0, utf16_length))) != 1:
        raise CorrectableOutput("occasion_length", max_words=OCCASION_MAX_WORDS)


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
