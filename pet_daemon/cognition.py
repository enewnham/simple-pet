import asyncio
import json
import logging
import time

from .attention import selected_subject
from .memory import Memory
from .context import Context, RETRY_INSTRUCTIONS, encoded_size
from .summarization import summary_request
from .repetition import repeat_match
from .telemetry import event, request_usage, trace, verbose


log = logging.getLogger("pet-daemon")


class Cognition:
    def __init__(self, client, config, pack, clock=time.time, memory=None):
        self.client = client
        self.config = config
        self.pack = pack
        self.clock = clock
        self.memory = memory if memory is not None else Memory(config.memory_file, pack.pet, clock, config.memory_retention_days, config.memory_max_events, speech_history_seconds=config.speech_history_seconds)
        self.context = Context(config, lambda: self.clock())

    def prompt(self, snapshot, reason, context=None):
        return self.context.prompt(snapshot, reason, context)

    def system(self, snapshot, reason, context=None):
        return self.context.system(snapshot, reason, context)

    def selected_sessions(self, snapshot, reason=None):
        return self.context.selected_sessions(snapshot, reason)

    def quiet_greeting(self, snapshot, reason):
        return self.context.quiet_greeting(snapshot, reason)

    def summary_request(self, focus):
        return summary_request(self.config, self.memory, self.clock, self.request, focus)

    def commentary_payload(self, payload, system, speech_required=False):
        focus = payload.get("focus_session")
        requests = focus.get("recent_requests", []) if focus else []
        others = payload.get("other_sessions", [])
        selected = {**payload}
        if focus:
            selected["focus_session"] = {**focus, "recent_requests": requests[-1:]}
        if others:
            selected["other_sessions"] = []
            omitted = payload["omitted_sessions"]
            selected["omitted_sessions"] = {
                "total": omitted["total"] + len(others),
                "active": omitted["active"] + sum(bool(item.get("turn_active")) for item in others),
            }

        def has_room():
            request = self.request(selected, system, speech_required=speech_required)
            return encoded_size(request) < self.config.prompt_target_bytes

        for previous in reversed(requests[:-1]):
            if not has_room():
                break
            selected["focus_session"]["recent_requests"].insert(0, previous)
        for other in others:
            if not has_room():
                break
            selected["other_sessions"].append(other)
            selected["omitted_sessions"]["total"] -= 1
            selected["omitted_sessions"]["active"] -= bool(other.get("turn_active"))
        return selected

    def commentary_request(self, payload, system, speech_required=False):
        selected = self.commentary_payload(payload, system, speech_required)
        return self.request(selected, system, speech_required=speech_required)

    def observation_request(self, snapshot, reason, context, system, speech_required):
        payload = self.commentary_payload(self.prompt(snapshot, reason, context), system, speech_required)
        observed_sessions = ()
        if payload.get("focus_session") is not None:
            count = 1 + len(payload["other_sessions"])
            observed_sessions = tuple(session["id"] for session in self.selected_sessions(snapshot, reason)[:count])
        subject = selected_subject(snapshot)
        observed_opportunity = (subject.id, subject.revision) if payload.get("focus_opportunity") is not None else None
        observed_work = {session["id"]: session.get("connected_work", {}).get("revision")
                         for session in snapshot.get("sessions", ()) if session["id"] in observed_sessions}
        request = self.request(payload, system, speech_required=speech_required)
        return request, observed_sessions, observed_opportunity, observed_work

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
        verbose("observation.request", reason=reason, request=request, request_bytes=encoded_size(request))
        started = time.monotonic()
        response = await self.client.messages.create(**request)
        verbose("observation.response", reason=reason, response=response.model_dump(mode="json"))
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
        request, observed_sessions, observed_opportunity, observed_work = self.observation_request(
            snapshot, reason, context, self.system(snapshot, reason, context), explicit,
        )
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
            retry, retry_sessions, retry_opportunity, retry_work = self.observation_request(snapshot, reason, context, system, explicit)
            try:
                result = await self.complete(retry, reason)
                self.validate_speech(result, explicit)
            except Exception as error:
                event("speech.retry_failed", level=logging.WARNING, error=str(error))
                result = initial if manual else {"speak": None}
            else:
                observed_sessions = retry_sessions
                observed_opportunity = retry_opportunity
                observed_work = retry_work
                match = await self.repetition(result["speak"])
                if match:
                    event("speech.repetition", stage="retry", method=match.method, similarity=round(match.similarity, 3),
                          action="acknowledgment" if manual else "silence")
                    if not manual:
                        result = {"speak": None}
        trace("observation.decision", reason=reason, speaks=result["speak"] is not None, summary_updated=summary is not None, decision={**result, "session_context": summary})
        return {**result, "session_context": summary, "context_session": focus["id"] if focus else None,
                "context_version": context_version, "observed_sessions": observed_sessions,
                "observed_opportunity": observed_opportunity, "observed_work": observed_work}

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
