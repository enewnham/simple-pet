from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..attention import AttentionSelection
from ..domain import ExecutionOutcome, SpeechReason, SpeechSituation

INSTRUCTIONS = """
Offer one brief, self-contained spoken reaction to the selected moment, or silence.
React to one specific detail rather than reciting a status report. A new message calls
for a reaction to its supported intent, not an answer to the developer's technical
question or an evaluation of their explanation. Omit publication mechanics, PR details,
commit hashes, and branch names. Companionship need not become a productivity prompt.
Use human-readable ages to judge freshness. Cite only observations used in the remark.
""".strip()

DIRECTIONS: dict[SpeechSituation, str] = {
    SpeechSituation.PROMPT: "React briefly to the latest message in the context of the preceding exchange. Express supported intent using current_work without answering technical questions or assuming work has started.",
    SpeechSituation.ROUTINE: "React briefly to the change in activity or progress.",
    SpeechSituation.COMPLETION: "React briefly to the result of this work segment.",
    SpeechSituation.FAILURE: "React briefly to the observed failed call.",
    SpeechSituation.EXPLICIT: "Give a brief reaction to the supplied context.",
    SpeechSituation.CHECK_IN_WAITING: "Give a brief remark about the supplied wait.",
    SpeechSituation.CHECK_IN_RUNNING: "Give a brief remark about the running call and its elapsed time.",
    SpeechSituation.CHECK_IN_QUIET: "Give a brief remark about the quiet time during this work. Companionship is enough; do not manufacture a work update or a next task.",
}


class Response(BaseModel):
    model_config = ConfigDict(extra="forbid")
    speak: str | None = Field(
        description="One brief spoken remark, or null for silence."
    )
    mentions: tuple[str, ...] = Field(
        default=(), description="Source IDs actually expressed in the spoken remark."
    )


class RequiredResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    speak: str = Field(description="One brief spoken remark.")
    mentions: tuple[str, ...] = Field(
        default=(), description="Source IDs actually expressed in the spoken remark."
    )

    @field_validator("speak")
    @classmethod
    def nonblank_speech(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Requested commentary requires speech")
        return value


def direction(selection: AttentionSelection) -> str:
    text = DIRECTIONS[selection.situation]
    if selection.situation == SpeechSituation.COMPLETION:
        outcome = selection.focus_session.outcome if selection.focus_session else None
        if outcome == ExecutionOutcome.CANCELLED:
            text += " Acknowledge the cancellation."
        elif outcome == ExecutionOutcome.ERROR:
            text += " Acknowledge the unsuccessful result."
    return text


def response_type(selection: AttentionSelection) -> type[Response | RequiredResponse]:
    return (
        RequiredResponse
        if selection.context is not None or selection.reason == SpeechReason.EXPLICIT
        else Response
    )
