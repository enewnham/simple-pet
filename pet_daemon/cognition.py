import asyncio
import json
import logging
import time

from .memory import Memory
from .context import Context, INSTRUCTIONS, SUMMARY_INSTRUCTIONS, encoded_size
from .telemetry import event, request_usage, trace


log = logging.getLogger("pet-daemon")


class Cognition(Context):
    def __init__(self, client, config, pack, clock=time.time, memory=None):
        self.client = client
        self.config = config
        self.pack = pack
        self.clock = clock
        self.memory = memory if memory is not None else Memory(config.memory_file, pack.pet, clock, config.memory_retention_days, config.memory_max_events, speech_history_seconds=config.speech_history_seconds)

    def schema(self):
        nullable_string = {"anyOf": [{"type": "string"}, {"type": "null"}]}
        return {
            "type": "object",
            "properties": {"speak": nullable_string},
            "required": ["speak"],
            "additionalProperties": False,
        }

    def request(self, payload, system):
        return dict(
            model=self.config.model, max_tokens=self.config.max_tokens,
            timeout=self.config.request_timeout, thinking={"type": "disabled"},
            system=system,
            messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": self.schema()}},
        )

    def should_summarize(self, focus, reason):
        if not focus:
            return False
        if self.config.summarize_first_request_only:
            return (reason == "prompt" and focus.get("request_count", 0) > 0
                    and not focus.get("first_summary_attempted", False))
        return reason in self.config.summarize_on

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
            collected[reason] = fields
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

    async def observe(self, snapshot, reason, context=None):
        explicit = context is not None or reason in ("check_in", "click", "explicit", "care")
        selected = self.selected_sessions(snapshot, reason)
        focus = selected[0] if selected else None
        request = self.bounded_request(self.prompt(snapshot, reason, context))
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
        if set(result) != {"speak"}:
            raise ValueError("Observation requires speak")
        if result["speak"] is not None and not isinstance(result["speak"], str):
            raise ValueError("Invalid observation text")
        if explicit and (not isinstance(result["speak"], str) or not result["speak"].strip()):
            raise ValueError("Requested commentary requires speech")
        trace("observation.decision", reason=reason, decision={**result, "session_context": summary})
        return {**result, "session_context": summary, "context_session": focus["id"] if focus else None,
                "context_version": (focus.get("generation"), focus.get("first_prompt", ""), tuple(focus.get("recent_requests", [])), focus.get("session_context", "")) if focus else None}
