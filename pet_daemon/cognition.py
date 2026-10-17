import asyncio
import json
import logging
import time

from .memory import Memory
from .context import Context, INSTRUCTIONS, SUMMARY_INSTRUCTIONS, RETRY_INSTRUCTIONS, encoded_size
from .repetition import repeat_match
from .telemetry import event, request_usage, trace


log = logging.getLogger("pet-daemon")


class Cognition(Context):
    def __init__(self, client, config, pack, clock=time.time, memory=None):
        self.client = client
        self.config = config
        self.pack = pack
        self.clock = clock
        self.memory = memory if memory is not None else Memory(config.memory_file, pack.pet, clock, config.memory_retention_days, config.memory_max_events, speech_history_seconds=config.speech_history_seconds)

    def schema(self, speech_required=False):
        speech = {"type": "string"} if speech_required else {"anyOf": [{"type": "string"}, {"type": "null"}]}
        speech["description"] = "One brief spoken remark." if speech_required else "One brief spoken remark, or null for silence."
        return {
            "type": "object",
            "properties": {"speak": speech},
            "required": ["speak"],
            "additionalProperties": False,
        }

    def request(self, payload, system, speech_required=False):
        return dict(
            model=self.config.model, max_tokens=self.config.max_tokens,
            timeout=self.config.request_timeout, thinking={"type": "disabled"},
            system=system,
            messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": self.schema(speech_required)}},
        )

    def should_summarize(self, focus, reason):
        if not focus:
            return False
        if self.config.summarize_first_request_only:
            return (reason == "prompt" and focus.get("request_count", 0) > 0
                    and not focus.get("first_summary_attempted", False))
        return reason in self.config.summarize_on

    def summary_version(self, focus):
        if not focus:
            return None
        if self.config.summarize_first_request_only:
            return focus.get("generation"), focus.get("first_prompt", "")
        reports = self.memory.recent(limit=3, session=focus["id"], event_types=(
            "turn_done", "turn_error", "subagent_done", "session_end", "interrupt"))
        return (focus.get("generation"), focus.get("first_prompt", ""), tuple(focus.get("recent_requests", ())),
                focus.get("session_context", ""), focus.get("progress", ""), focus.get("outcome"),
                focus.get("turn_active", False), tuple(report["id"] for report in reports))

    async def summarize(self, focus):
        request = self.summary_request(focus)
        result = await self.complete(request, "summary")
        if set(result) != {"session_context"} or (result["session_context"] is not None and not isinstance(result["session_context"], str)):
            raise ValueError("Invalid session context")
        return result["session_context"]

    async def complete(self, request, reason):
        trace("observation.request", reason=reason, request=request, request_bytes=encoded_size(request))
        started = time.monotonic()
        response = await self.client.messages.create(**request)
        trace("observation.response", reason=reason, response=response.model_dump(mode="json"))
        usage = response.usage
        fields = dict(input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
                      cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
                      cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0)
        collected = request_usage.get()
        if collected is not None:
            previous = collected.get(reason, {})
            collected[reason] = {key: previous.get(key, 0) + value for key, value in fields.items()}
        event("observation.usage", level=logging.DEBUG, reason=reason,
              duration_seconds=round(time.monotonic() - started, 3), **fields)
        if response.stop_reason != "end_turn":
            raise ValueError(f"Observation ended with {response.stop_reason}")
        content = "".join(block.text for block in response.content if block.type == "text")
        result = json.loads(content)
        if not isinstance(result, dict):
            raise ValueError("Observation requires an object")
        if reason == "summary":
            event("observation.summary", duration_seconds=round(time.monotonic() - started, 3), **fields)
        return result

    async def observe(self, snapshot, reason, context=None, refresh=None):
        explicit = context is not None or reason in ("check_in", "click", "explicit", "care")
        selected = self.selected_sessions(snapshot, reason)
        focus = selected[0] if selected else None
        context_version = self.summary_version(focus)
        request = self.bounded_request(self.prompt(snapshot, reason, context), self.system(snapshot, reason, context), speech_required=explicit)
        summary = None
        if self.should_summarize(focus, reason):
            result, summary = await asyncio.gather(self.complete(request, reason), self.summarize(focus), return_exceptions=True)
            if isinstance(result, Exception):
                raise result
            if isinstance(summary, Exception):
                log.warning("Could not summarize session: %s", summary)
                summary = None
        else:
            result = await self.complete(request, reason)
        self.validate_speech(result, explicit)
        match = await self.repetition(result["speak"])
        if match:
            event("speech.repetition", stage="initial", method=match.method, similarity=round(match.similarity, 3))
            manual = context is not None or reason in ("click", "explicit")
            initial = result
            snapshot = refresh() if refresh else snapshot
            system = self.system(snapshot, reason, context)
            system[-1]["text"] += "\n<reconsideration>\n" + RETRY_INSTRUCTIONS + "\n</reconsideration>"
            retry = self.bounded_request(self.prompt(snapshot, reason, context), system, speech_required=explicit)
            try:
                result = await self.complete(retry, reason)
                self.validate_speech(result, explicit)
            except Exception as error:
                event("speech.retry_failed", level=logging.WARNING, error=str(error))
                result = initial if manual else {"speak": None}
            else:
                match = await self.repetition(result["speak"])
                if match:
                    event("speech.repetition", stage="retry", method=match.method, similarity=round(match.similarity, 3),
                          action="acknowledgment" if manual else "silence")
                    if not manual:
                        result = {"speak": None}
        trace("observation.decision", reason=reason, decision={**result, "session_context": summary})
        return {**result, "session_context": summary, "context_session": focus["id"] if focus else None,
                "context_version": context_version}

    async def repetition(self, text):
        if not text or not text.strip():
            return None
        return await repeat_match(text, [row["text"] for row in self.memory.speech()])

    def validate_speech(self, result, required):
        if set(result) != {"speak"}:
            raise ValueError("Observation requires speak")
        if result["speak"] is not None and not isinstance(result["speak"], str):
            raise ValueError("Invalid observation text")
        if required and (not isinstance(result["speak"], str) or not result["speak"].strip()):
            raise ValueError("Requested commentary requires speech")
