import asyncio
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from anthropic.types import TextBlockParam

from ..attention import AttentionSelection
from ..config import Config
from ..domain import ExecutionOutcome, ModelPurpose, SpeechReason
from ..logging import get_logger
from ..runtime import PetContext, SessionContext
from ..storage import Memory
from .context import PreparedCommentary, PromptBuilder
from .contracts import JSONSchema, ModelRequest, TextSchema
from .model import ModelClient
from .repetition import RepeatMatch, repeat_match
from .summarization import summary_request

log = get_logger("pet-daemon")


@dataclass(frozen=True)
class SummaryEvidence:
    recent_requests: tuple[str, ...]
    summary: str
    progress: str
    outcome: ExecutionOutcome | None
    turn_active: bool
    report_ids: tuple[int, ...]


@dataclass(frozen=True)
class SummaryVersion:
    generation: int
    initial_request: str
    evidence: SummaryEvidence | None


@dataclass(frozen=True)
class CommentaryResult:
    speak: str | None
    summary: str | None
    summary_session: str | None
    summary_version: SummaryVersion | None
    prepared: PreparedCommentary


class Cognition:
    def __init__(
        self,
        client: ModelClient,
        config: Config,
        *,
        memory: Memory,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.client = client
        self.config = config
        self.clock = clock
        self.memory = memory
        self.prompts = PromptBuilder(config, lambda: self.clock())

    def summary_request(self, focus: SessionContext) -> ModelRequest:
        return summary_request(
            self.config, self.memory, self.clock, self.request, focus
        )

    def schema(self, speech_required: bool = False) -> JSONSchema:
        speech: TextSchema = (
            {"type": "string"}
            if speech_required
            else {"anyOf": [{"type": "string"}, {"type": "null"}]}
        )
        speech["description"] = (
            "One brief spoken remark."
            if speech_required
            else "One brief spoken remark, or null for silence."
        )
        return {
            "type": "object",
            "properties": {"speak": speech},
            "required": ["speak"],
            "additionalProperties": False,
        }

    def request(
        self,
        payload: object,
        system: Sequence[TextBlockParam],
        speech_required: bool = False,
    ) -> ModelRequest:
        return {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "timeout": self.config.request_timeout,
            "thinking": {"type": "disabled"},
            "system": system,
            "messages": [
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
            ],
            "output_config": {
                "effort": "low",
                "format": {
                    "type": "json_schema",
                    "schema": self.schema(speech_required),
                },
            },
        }

    def should_summarize(
        self, focus: SessionContext | None, reason: SpeechReason
    ) -> bool:
        if not focus:
            return False
        if self.config.summarize_first_request_only:
            return (
                reason == SpeechReason.PROMPT
                and focus.request_count > 0
                and not focus.first_summary_attempted
            )
        return reason in self.config.summarize_on

    def summary_version(self, focus: SessionContext | None) -> SummaryVersion | None:
        if not focus:
            return None
        if self.config.summarize_first_request_only:
            return SummaryVersion(focus.generation, focus.initial_request, None)
        reports = self.memory.recent(
            limit=3,
            session=focus.id,
            event_types=(
                "turn_done",
                "turn_error",
                "subagent_done",
                "session_end",
                "interrupt",
            ),
        )
        return SummaryVersion(
            focus.generation,
            focus.initial_request,
            SummaryEvidence(
                tuple(focus.recent_requests),
                focus.summary,
                focus.progress,
                focus.outcome,
                focus.turn_active,
                tuple(report.id for report in reports),
            ),
        )

    async def summarize(self, focus: SessionContext) -> str | None:
        request = self.summary_request(focus)
        result = await self.complete(request, ModelPurpose.SUMMARY)
        if set(result) != {"summary"}:
            raise ValueError("Invalid session summary")
        summary = result["summary"]
        if summary is not None and not isinstance(summary, str):
            raise ValueError("Invalid session summary")
        return summary

    async def complete(
        self, request: ModelRequest, reason: SpeechReason | ModelPurpose
    ) -> dict[str, object]:
        return await self.client.complete(request, reason)

    async def generate_commentary(
        self,
        snapshot: PetContext,
        selection: AttentionSelection,
        refresh: Callable[[], PetContext] | None = None,
    ) -> CommentaryResult:
        reason = selection.reason
        explicit = selection.context is not None or reason in (
            SpeechReason.CHECK_IN,
            SpeechReason.CLICK,
            SpeechReason.EXPLICIT,
            SpeechReason.CARE,
        )
        focus = selection.focus_session
        summary_version = self.summary_version(focus)
        prepared = self.prompts.prepare(snapshot, selection, self.request, explicit)
        summary = None
        if focus is not None and self.should_summarize(focus, reason):
            raw_result, summary = await asyncio.gather(
                self.complete(prepared.request, reason),
                self.summarize(focus),
                return_exceptions=True,
            )
            if isinstance(raw_result, BaseException):
                raise raw_result
            if isinstance(summary, BaseException) and not isinstance(
                summary, Exception
            ):
                raise summary
            if isinstance(summary, Exception):
                log.warning("observation.summary_failed", error=str(summary))
                summary = None
        else:
            raw_result = await self.complete(prepared.request, reason)
        speech = self.validate_speech(raw_result, explicit)
        match = await self.repetition(speech)
        if match:
            log.info(
                "speech.repetition",
                stage="initial",
                method=match.method,
                similarity=round(match.similarity, 3),
            )
            manual = selection.context is not None or reason in (
                SpeechReason.CLICK,
                SpeechReason.EXPLICIT,
            )
            initial = speech
            current = refresh() if refresh else snapshot
            refreshed = selection.refresh(current)
            if not selection.supported(current):
                return CommentaryResult(
                    None,
                    summary,
                    focus.id if focus else None,
                    summary_version,
                    prepared,
                )
            retry = self.prompts.prepare(
                current, refreshed, self.request, explicit, retry=True
            )
            try:
                raw_result = await self.complete(retry.request, reason)
                speech = self.validate_speech(raw_result, explicit)
            except Exception as error:  # noqa: BLE001 - Keep failed model retries from discarding accepted speech.
                log.warning("speech.retry_failed", error=str(error))
                speech = initial if manual else None
            else:
                prepared = retry
                match = await self.repetition(speech)
                if match:
                    log.info(
                        "speech.repetition",
                        stage="retry",
                        method=match.method,
                        similarity=round(match.similarity, 3),
                        action="acknowledgment" if manual else "silence",
                    )
                    if not manual:
                        speech = None
        log.debug(
            "observation.decision",
            reason=reason,
            speaks=speech is not None,
            summary_updated=summary is not None,
            decision={"speak": speech, "summary": summary},
        )
        return CommentaryResult(
            speech,
            summary,
            focus.id if focus else None,
            summary_version,
            prepared,
        )

    async def repetition(self, text: str | None) -> RepeatMatch | None:
        if not text or not text.strip():
            return None
        return await repeat_match(text, [row.text for row in self.memory.speech()])

    def validate_speech(self, result: dict[str, object], required: bool) -> str | None:
        if set(result) != {"speak"}:
            raise ValueError("Observation requires speak")
        speech = result["speak"]
        if speech is not None and not isinstance(speech, str):
            raise ValueError("Invalid observation text")
        if required and (not isinstance(speech, str) or not speech.strip()):
            raise ValueError("Requested commentary requires speech")

        return speech
