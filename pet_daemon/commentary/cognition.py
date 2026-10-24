import json
import time
from collections.abc import Callable, Sequence
from typing import cast

from anthropic import transform_schema
from anthropic.types import TextBlockParam
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..attention import AttentionSelection
from ..config import Config
from ..domain import ModelPurpose, SpeechReason
from ..logging import get_logger
from ..runtime import PetContext
from ..storage import Memory
from ..work.narrative import WorkNarrative, semantic_revision
from . import story
from .context import PreparedCommentary, PromptBuilder
from .contracts import JSONSchema, ModelRequest
from .model import ModelClient
from .repetition import RepeatMatch, repeat_match

log = get_logger("pet-daemon")


class StoryUnavailable(RuntimeError):
    pass


class CommentaryResult(BaseModel):
    speak: str | None
    prepared: PreparedCommentary


class SpeechResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    speak: str | None = Field(
        description="One brief spoken remark, or null for silence."
    )


class RequiredSpeechResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    speak: str = Field(description="One brief spoken remark.")

    @field_validator("speak")
    @classmethod
    def nonblank_speech(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Requested commentary requires speech")
        return value


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

    def schema(self, speech_required: bool = False) -> JSONSchema:
        response = RequiredSpeechResponse if speech_required else SpeechResponse
        return cast(JSONSchema, transform_schema(response))

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

    async def complete(
        self, request: ModelRequest, reason: SpeechReason | ModelPurpose
    ) -> dict[str, object]:
        return await self.client.complete(request, reason)

    async def prepare_story(
        self,
        snapshot: PetContext,
        selection: AttentionSelection,
        refresh: Callable[[], PetContext] | None = None,
    ) -> WorkNarrative | None:
        prepared = story.prepare(
            snapshot,
            selection,
            self.memory,
            self.request,
            target_bytes=self.config.prompt_target_bytes,
        )
        if prepared is None:
            return None
        existing = self.memory.load_narrative(prepared.subject_id)
        if (
            existing is not None
            and existing.revision == prepared.revision
            and prepared.latest_request_source in existing.intent.sources
        ):
            return existing
        try:
            raw = await self.complete(prepared.request, ModelPurpose.STORY)
            narrative = story.validate(raw, prepared)
            current = refresh() if refresh else snapshot
            refreshed = selection.refresh(current)
            if (
                not selection.supported(current)
                or (
                    semantic_revision(refreshed.subject.story)
                    if refreshed.subject
                    else ""
                )
                != prepared.revision
            ):
                raise StoryUnavailable("Work changed during story generation")
            self.memory.save_narrative(narrative)
            return narrative
        except StoryUnavailable:
            raise
        except Exception as error:
            raise StoryUnavailable("Work story could not be updated") from error

    async def generate_commentary(
        self,
        snapshot: PetContext,
        selection: AttentionSelection,
        refresh: Callable[[], PetContext] | None = None,
        eligible: Callable[[AttentionSelection, PetContext], bool] | None = None,
    ) -> CommentaryResult:
        snapshot = refresh() if refresh else snapshot
        if eligible and not eligible(selection, snapshot):
            raise StoryUnavailable("Work is no longer eligible")
        reason = selection.reason
        explicit = selection.context is not None or reason in (
            SpeechReason.CLICK,
            SpeechReason.EXPLICIT,
            SpeechReason.CARE,
        )
        manual = selection.context is not None or reason in (
            SpeechReason.CLICK,
            SpeechReason.EXPLICIT,
        )
        basic = False
        try:
            narrative = await self.prepare_story(snapshot, selection, refresh)
        except StoryUnavailable:
            if not manual:
                raise
            snapshot = refresh() if refresh else snapshot
            selection = selection.refresh(snapshot)
            narrative = None
            basic = True
        snapshot = refresh() if refresh else snapshot
        selection = selection.refresh(snapshot)
        if eligible and not eligible(selection, snapshot):
            raise StoryUnavailable("Work is no longer eligible")
        prepared = self.prompts.prepare(
            snapshot,
            selection,
            self.request,
            explicit,
            narrative=narrative,
            basic=basic,
            excluded=tuple(record.text for record in self.memory.speech(limit=256)),
        )
        raw_result = await self.complete(prepared.request, reason)
        speech = self.validate_speech(raw_result, explicit)
        match = await self.repetition(speech)
        if match:
            log.info(
                "speech.repetition",
                stage="initial",
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
            decision={"speak": speech},
        )
        return CommentaryResult(speak=speech, prepared=prepared)

    async def repetition(self, text: str | None) -> RepeatMatch | None:
        if not text or not text.strip():
            return None
        return await repeat_match(text, [row.text for row in self.memory.speech()])

    def validate_speech(self, result: dict[str, object], required: bool) -> str | None:
        response = RequiredSpeechResponse if required else SpeechResponse
        return response.model_validate(result).speak
