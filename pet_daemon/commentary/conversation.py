from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..attention import AttentionSelection
from ..domain import SpeechReason, SpeechSituation

INSTRUCTIONS = """
Compose a small conversation: a specific headline, one natural spoken remark, optional
context worth expanding, and zero to two response choices. Keep the remark within
45 words and context within 100 words. Give the headline, remark, explanation, and choice
labels the same configured voice. Write a headline for this moment rather than copying
a raw task request or using a generic title.

Use context to explain supplied decisions, connections, and tradeoffs when there is more
to tell. Conversation can happen while work continues. React to the supplied work without
answering an intercepted developer request or inventing technical advice.

Include in mentions every source used in the headline, remark, context, or choice labels.
recent_mentions shows sources already expressed and their age. Avoid immediately repeating
a freshly acknowledged result. Completed work can recede without inventing another chore.

Choose only IDs supplied in available_actions. Write short, truthful labels suited to
this exchange. Questions can invite the developer to explore the supplied context.
Each choice must match its offered capability and the content it reveals. Return at most
two distinct choices and omit choices that add no value. Omit tomorrow and handled unless
the selected evidence explicitly describes remaining or ongoing work. Never generate
commands, paths, URLs, or executable fields. The application owns destinations and actions.

A brief companionable remark can stand on its own during quiet time. Do not manufacture
a task or question to fill the space. For silence, return speak null and empty
headline/context/choices/mentions.
""".strip()

DIRECTIONS: dict[SpeechSituation, str] = {
    SpeechSituation.CLICK_WAITING: "Help the developer understand the observed wait in the context of this work. Do not invent a blocker or a question.",
    SpeechSituation.CLICK_WORKING: "Offer a useful perspective on the current work and related memories. Respect confirmed activity and explicitly reported remaining work.",
    SpeechSituation.CLICK_IDLE: "Give the developer a casual greeting.",
    SpeechSituation.CARE_EXERCISE: "Give a brief spoken aside about taking a stretch break during quiet time.",
    SpeechSituation.CARE_ATTENTION: "Give a brief casual greeting during quiet time.",
    SpeechSituation.CARE_CONSTERNATION: "Give a brief aside about a quiet stretch.",
    SpeechSituation.IDLE_REFLECTION: "This is a recollection of earlier work, not a reply to current_work.request. Choose one eligible memory candidate for a useful last-known decision, result, or explicitly reported remaining step, or choose silence. Any work-related speech requires that candidate's ID in memory and mentions. Use a candidate with eligible_recent true, or set ongoing true for an eligible_ongoing candidate that explicitly describes ongoing, paused, or remaining work. A silent selected recollection keeps its memory ID and has empty mentions. If no candidate supports a useful remark, return silence with memory null.",
}


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
        description="One brief spoken remark, or null for silence."
    )
    mentions: tuple[str, ...] = Field(
        default=(),
        description="Source IDs expressed in the headline, remark, context, or choices.",
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
        default=None, description="One eligible source ID chosen for idle recollection."
    )
    ongoing: bool = Field(
        default=False,
        description="Whether that source explicitly reports ongoing, paused, or remaining work.",
    )


def direction(selection: AttentionSelection) -> str:
    text = DIRECTIONS[selection.situation]
    if selection.situation == SpeechSituation.IDLE_REFLECTION:
        text += " A later resolution or cancellation disqualifies the earlier remaining step."
        if selection.reason == SpeechReason.CLICK:
            text += " This recollection was requested with a click. If no memory is useful, a brief playful greeting with memory null is appropriate."
    return text


def response_type(selection: AttentionSelection) -> type[Response]:
    return (
        RecollectionResponse
        if selection.situation == SpeechSituation.IDLE_REFLECTION
        else Response
    )
