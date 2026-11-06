import asyncio
import contextvars
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field

from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    StreamEvent,
    SystemMessage,
    ToolUseBlock,
)

from ..logging import get_logger

log = get_logger("pet-daemon")
TOKEN_FIELDS = {
    "input_tokens": "input_tokens",
    "output_tokens": "output_tokens",
    "cache_read_tokens": "cache_read_input_tokens",
    "cache_write_tokens": "cache_creation_input_tokens",
}


class SdkFailure(RuntimeError):
    def __init__(
        self,
        kind: str,
        *,
        status: int | None = None,
        candidate_id: str | None = None,
        message: str | None = None,
    ) -> None:
        self.kind = kind
        self.status = status
        self.candidate_id = candidate_id
        super().__init__(message or f"Companion model failed: {kind}")


class SdkValidationError(SdkFailure, ValueError):
    def __init__(self, message: str, candidate_id: str | None) -> None:
        super().__init__("validation", message=message, candidate_id=candidate_id)


def tokens(usage: dict[str, object] | None) -> dict[str, int] | None:
    if not usage:
        return None
    return {
        name: int(value)
        if isinstance(value := usage.get(provider_name), (int, float))
        else 0
        for name, provider_name in TOKEN_FIELDS.items()
    }


@dataclass
class SdkUsage:
    request_id: str
    purpose: str
    model: str
    candidate_ids: tuple[str, ...] = ()
    focus_id: str | None = None
    conversation_id: str | None = None
    started: float = field(default_factory=time.monotonic)
    calls: dict[str, dict[str, int] | None] = field(default_factory=dict)
    active_calls: dict[str | None, str] = field(default_factory=dict)
    retries: int = 0
    tools_by_call: dict[str, int] = field(default_factory=dict)
    user_bytes: int = 0
    system_bytes: int = 0
    schema_bytes: int = 0
    tool_schema_bytes: int = 0
    terminal: ResultMessage | None = None

    def fields(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "purpose": self.purpose,
            "model": self.model,
            "candidate_ids": self.candidate_ids,
            "focus_id": self.focus_id,
            "conversation_id": self.conversation_id,
        }

    def receive(
        self, message: AssistantMessage | ResultMessage | SystemMessage | StreamEvent
    ) -> None:
        if isinstance(message, ResultMessage):
            self.terminal = message
        elif isinstance(message, StreamEvent):
            self.receive_stream(message)
        elif isinstance(message, AssistantMessage):
            identifier = (
                message.message_id
                or message.uuid
                or (
                    self.active_calls.get(message.parent_tool_use_id)
                    if not message.error
                    else None
                )
                or f"unidentified:{len(self.calls)}"
            )
            self.record_call(identifier, message.usage, message.error or "response")
            self.tools_by_call[identifier] = sum(
                isinstance(block, ToolUseBlock) for block in message.content
            )
        elif message.subtype == "api_retry":
            self.retries += 1
            log.info(
                "companion.model_retry",
                **self.fields(),
                retry=self.retries,
                attempt=message.data.get("attempt"),
                status_code=message.data.get("error_status"),
                usage_known=False,
            )

    def record_call(
        self,
        identifier: str,
        usage: dict[str, object] | None,
        outcome: str,
        *,
        partial: bool = False,
    ) -> None:
        update = identifier in self.calls
        values = tokens(usage)
        if partial and values is not None:
            values = {
                **(self.calls.get(identifier) or {}),
                **{
                    key: value
                    for key, value in values.items()
                    if TOKEN_FIELDS[key] in (usage or {})
                },
            }
        self.calls[identifier] = (
            values if values is not None else self.calls.get(identifier)
        )
        log.info(
            "companion.model_call",
            **self.fields(),
            call_id=identifier,
            update=update,
            aggregation="cumulative_per_call",
            usage_known=self.calls[identifier] is not None,
            usage_source="provider_stream" if partial else "provider_message",
            outcome=outcome,
            **(self.calls[identifier] or {}),
        )

    def receive_stream(self, message: StreamEvent) -> None:
        event = message.event
        if event.get("type") == "message_start":
            payload = event.get("message", {})
            identifier = payload.get("id")
            if isinstance(identifier, str):
                self.active_calls[message.parent_tool_use_id] = identifier
                self.record_call(
                    identifier, payload.get("usage"), "started", partial=True
                )
        elif event.get("type") == "message_delta":
            identifier = self.active_calls.get(message.parent_tool_use_id)
            if identifier is not None:
                self.record_call(
                    identifier, event.get("usage"), "streaming", partial=True
                )

    def finish(self, outcome: str, error_kind: str | None = None) -> None:
        terminal_usage = (
            tokens(self.terminal.usage) if self.terminal is not None else None
        )
        aggregate_source = "provider_result"
        if (
            terminal_usage is None
            and self.terminal is not None
            and self.terminal.model_usage
        ):
            fields = {
                "input_tokens": "inputTokens",
                "output_tokens": "outputTokens",
                "cache_read_input_tokens": "cacheReadInputTokens",
                "cache_creation_input_tokens": "cacheCreationInputTokens",
            }
            terminal_usage = tokens(
                {
                    name: sum(
                        item.get(field, 0)
                        for item in self.terminal.model_usage.values()
                    )
                    for name, field in fields.items()
                }
            )
            aggregate_source = "provider_model_usage"
        known = [value for value in self.calls.values() if value is not None]
        totals = terminal_usage or (
            {key: sum(value.get(key, 0) for value in known) for key in TOKEN_FIELDS}
            if known
            else None
        )
        request_bytes = (
            self.user_bytes
            + self.system_bytes
            + self.schema_bytes
            + self.tool_schema_bytes
        )
        log.info(
            "companion.usage",
            **self.fields(),
            outcome=outcome,
            error_kind=error_kind,
            usage_known=totals is not None,
            usage_complete=terminal_usage is not None
            or self.terminal is not None
            and bool(self.calls)
            and len(known) == len(self.calls),
            usage_source=aggregate_source
            if terminal_usage is not None
            else "deduplicated_provider_messages"
            if totals is not None
            else "unknown",
            **(
                totals
                or {
                    key: None
                    for key in (
                        "input_tokens",
                        "output_tokens",
                        "cache_read_tokens",
                        "cache_write_tokens",
                    )
                }
            ),
            total_context_tokens=totals["input_tokens"]
            + totals["cache_read_tokens"]
            + totals["cache_write_tokens"]
            if totals is not None
            else None,
            total_tokens=sum(totals.values()) if totals is not None else None,
            model_calls=len(self.calls),
            model_attempts_observed=len(self.calls) + self.retries,
            retries=self.retries,
            tool_calls=sum(self.tools_by_call.values()),
            request_bytes=request_bytes,
            sdk_invoked=self.user_bytes > 0,
            aggregation="request_total_do_not_add_model_call_updates",
            user_bytes=self.user_bytes,
            system_bytes=self.system_bytes,
            schema_bytes=self.schema_bytes,
            tool_schema_bytes=self.tool_schema_bytes,
            estimated_input_tokens=(request_bytes + 3) // 4,
            estimate_method="utf8_bytes_div_4_excludes_sdk_overhead",
            cost_usd=self.terminal.total_cost_usd
            if self.terminal is not None
            else None,
            duration_seconds=round(time.monotonic() - self.started, 3),
        )


active_usage = contextvars.ContextVar[SdkUsage | None]("companion_usage", default=None)


@contextmanager
def collect_usage(usage: SdkUsage) -> Generator[None]:
    token = active_usage.set(usage)
    outcome = "sdk_validated"
    error_kind = None
    try:
        yield
    except BaseException as error:
        outcome = (
            "cancelled" if isinstance(error, asyncio.CancelledError) else "rejected"
        )
        error_kind = (
            error.kind if isinstance(error, SdkFailure) else type(error).__name__
        )
        if isinstance(error, asyncio.CancelledError):
            error.__dict__["candidate_id"] = usage.focus_id
        elif isinstance(error, SdkFailure):
            error.candidate_id = usage.focus_id
        elif isinstance(error, ValueError):
            error_kind = "validation"
            raise SdkValidationError(str(error), usage.focus_id) from error
        raise
    finally:
        usage.finish(outcome, error_kind)
        active_usage.reset(token)
