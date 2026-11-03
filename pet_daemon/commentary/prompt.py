from dataclasses import dataclass

from ..attention import AttentionSelection
from ..domain import SpeechReason, SpeechSituation
from . import conversation, reaction

type Response = (
    reaction.Response
    | reaction.RequiredResponse
    | conversation.Response
    | conversation.RecollectionResponse
)


@dataclass(frozen=True)
class Prompt:
    response: type[Response]
    situation: SpeechSituation
    requested_by_click: bool = False
    min_tokens: int = 0

    @property
    def template(self) -> str:
        return f"requests/{self.situation.value}.jinja"

    @property
    def instruction_template(self) -> str:
        return (
            "system/conversation.jinja"
            if self.is_conversation
            else "system/speech.jinja"
        )

    @property
    def examples_template(self) -> str:
        return (
            "examples/conversation.jinja"
            if self.is_conversation
            else "examples/speech.jinja"
        )

    @property
    def is_conversation(self) -> bool:
        return issubclass(self.response, conversation.Response)


def select_prompt(selection: AttentionSelection) -> Prompt:
    if selection.situation.is_conversation:
        return Prompt(
            conversation.response_type(selection),
            selection.situation,
            requested_by_click=selection.reason == SpeechReason.CLICK,
            min_tokens=768,
        )
    return Prompt(
        reaction.response_type(selection),
        selection.situation,
    )
