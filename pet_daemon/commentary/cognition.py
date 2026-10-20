import asyncio
import json
import logging
import time

from dataclasses import dataclass

from .context import Context, PreparedCommentary
from .summarization import summary_request
from .repetition import repeat_match
from .model import ModelClient
from ..telemetry import event, trace


log = logging.getLogger("pet-daemon")


@dataclass(frozen=True)
class ObservationResult:
    speak: str | None
    session_context: str | None
    context_session: str | None
    context_version: object
    prepared: PreparedCommentary


class Cognition:
    def __init__(self, client: ModelClient, config, *, memory, clock=time.time):
        self.client = client
        self.config = config
        self.clock = clock
        self.memory = memory
        self.context = Context(config, lambda: self.clock())

    def summary_request(self, focus):
        return summary_request(
            self.config, self.memory, self.clock, self.request, focus
        )

    def schema(self, speech_required=False):
        speech = (
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

    def request(self, payload, system, speech_required=False):
        return dict(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            timeout=self.config.request_timeout,
            thinking={"type": "disabled"},
            system=system,
            messages=[
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
            ],
            output_config={
                "effort": "low",
                "format": {
                    "type": "json_schema",
                    "schema": self.schema(speech_required),
                },
            },
        )

    def should_summarize(self, focus, reason):
        if not focus:
            return False
        if self.config.summarize_first_request_only:
            return (
                reason == "prompt"
                and focus.get("request_count", 0) > 0
                and not focus.get("first_summary_attempted", False)
            )
        return reason in self.config.summarize_on

    def summary_version(self, focus):
        if not focus:
            return None
        if self.config.summarize_first_request_only:
            return focus.get("generation"), focus.get("first_prompt", "")
        reports = self.memory.recent(
            limit=3,
            session=focus["id"],
            event_types=(
                "turn_done",
                "turn_error",
                "subagent_done",
                "session_end",
                "interrupt",
            ),
        )
        return (
            focus.get("generation"),
            focus.get("first_prompt", ""),
            tuple(focus.get("recent_requests", ())),
            focus.get("session_context", ""),
            focus.get("progress", ""),
            focus.get("outcome"),
            focus.get("turn_active", False),
            tuple(report["id"] for report in reports),
        )

    async def summarize(self, focus):
        request = self.summary_request(focus)
        result = await self.complete(request, "summary")
        if set(result) != {"session_context"} or (
            result["session_context"] is not None
            and not isinstance(result["session_context"], str)
        ):
            raise ValueError("Invalid session context")
        return result["session_context"]

    async def complete(self, request, reason):
        return await self.client.complete(request, reason)

    async def observe(self, snapshot, selection, refresh=None):
        reason = selection.reason
        explicit = selection.context is not None or reason in (
            "check_in",
            "click",
            "explicit",
            "care",
        )
        focus = selection.focus_session
        context_version = self.summary_version(focus)
        prepared = self.context.prepare(snapshot, selection, self.request, explicit)
        summary = None
        if self.should_summarize(focus, reason):
            result, summary = await asyncio.gather(
                self.complete(prepared.request, reason),
                self.summarize(focus),
                return_exceptions=True,
            )
            if isinstance(result, Exception):
                raise result
            if isinstance(summary, Exception):
                log.warning("Could not summarize session: %s", summary)
                summary = None
        else:
            result = await self.complete(prepared.request, reason)
        self.validate_speech(result, explicit)
        match = await self.repetition(result["speak"])
        if match:
            event(
                "speech.repetition",
                stage="initial",
                method=match.method,
                similarity=round(match.similarity, 3),
            )
            manual = selection.context is not None or reason in ("click", "explicit")
            initial = result
            current = refresh() if refresh else snapshot
            refreshed = selection.refresh(current)
            if not selection.supported(current):
                return ObservationResult(
                    None,
                    summary,
                    focus["id"] if focus else None,
                    context_version,
                    prepared,
                )
            retry = self.context.prepare(
                current, refreshed, self.request, explicit, retry=True
            )
            try:
                result = await self.complete(retry.request, reason)
                self.validate_speech(result, explicit)
            except Exception as error:
                event("speech.retry_failed", level=logging.WARNING, error=str(error))
                result = initial if manual else {"speak": None}
            else:
                prepared = retry
                match = await self.repetition(result["speak"])
                if match:
                    event(
                        "speech.repetition",
                        stage="retry",
                        method=match.method,
                        similarity=round(match.similarity, 3),
                        action="acknowledgment" if manual else "silence",
                    )
                    if not manual:
                        result = {"speak": None}
        trace(
            "observation.decision",
            reason=reason,
            speaks=result["speak"] is not None,
            summary_updated=summary is not None,
            decision={**result, "session_context": summary},
        )
        return ObservationResult(
            result["speak"],
            summary,
            focus["id"] if focus else None,
            context_version,
            prepared,
        )

    async def repetition(self, text):
        if not text or not text.strip():
            return None
        return await repeat_match(text, [row["text"] for row in self.memory.speech()])

    def validate_speech(self, result, required):
        if set(result) != {"speak"}:
            raise ValueError("Observation requires speak")
        if result["speak"] is not None and not isinstance(result["speak"], str):
            raise ValueError("Invalid observation text")
        if required and (
            not isinstance(result["speak"], str) or not result["speak"].strip()
        ):
            raise ValueError("Requested commentary requires speech")
