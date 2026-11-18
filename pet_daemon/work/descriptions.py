from collections.abc import Sequence

from ..domain import SpeechSituation
from ..prose import duration, work_prose
from ..runtime import AgentSessionStatus, WaitingKind
from .history import report_prose
from .models import ExecutionEvidence, ProgressEvidence, SessionReport


def short_report(reports: Sequence[SessionReport], observed_at: float) -> str | None:
    for report in reversed(reports):
        if report.observed_at != observed_at:
            continue
        if not isinstance(report.payload, (ProgressEvidence, ExecutionEvidence)):
            continue
        text = report_prose(report)
        if text and len(text) <= 100 and "\n" not in text:
            return f"The agent reported: {work_prose(text, 100)}"
    return None


def describe_occasion(
    situation: SpeechSituation, focus: AgentSessionStatus, meaning: str | None
) -> str | None:
    if focus.waiting_for_input:
        description = {
            WaitingKind.PERMISSION: "The work is waiting for the developer to approve an operation.",
            WaitingKind.QUESTION: "The work is waiting for an answer to a question.",
        }.get(focus.waiting_kind, "The work is waiting for input.")
        prompt = work_prose(focus.waiting_prompt, 120)
        if prompt:
            description += f" The request is: {prompt}"
        if focus.waiting_seconds is not None:
            description += f" It has been open for {duration(focus.waiting_seconds)}."
        return description
    if situation == SpeechSituation.CHECK_IN_RUNNING:
        tool = max(
            focus.active_tools, key=lambda item: item.duration_seconds, default=None
        )
        if tool is not None:
            elapsed = duration(tool.duration_seconds)
            return f"A {tool.activity} operation is still running. It has been running for {elapsed}."
    if situation == SpeechSituation.FAILURE and focus.failure:
        return f"{meaning} The affected operation was {focus.failure.activity}."
    return meaning
