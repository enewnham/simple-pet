from dataclasses import dataclass

from ..attention import AttentionSelection
from . import conversation, reaction

type Response = (
    reaction.Response
    | reaction.RequiredResponse
    | conversation.Response
    | conversation.RecollectionResponse
)


@dataclass(frozen=True)
class Prompt:
    instructions: str
    direction: str
    response: type[Response]
    min_tokens: int = 0

    @property
    def is_conversation(self) -> bool:
        return issubclass(self.response, conversation.Response)


def select_prompt(selection: AttentionSelection) -> Prompt:
    if selection.situation.is_conversation:
        return Prompt(
            conversation.INSTRUCTIONS,
            conversation.direction(selection),
            conversation.response_type(selection),
            min_tokens=768,
        )
    return Prompt(
        reaction.INSTRUCTIONS,
        reaction.direction(selection),
        reaction.response_type(selection),
    )
