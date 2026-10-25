import json
import time
from collections.abc import Callable, Sequence
from typing import cast

from anthropic import transform_schema
from anthropic.types import TextBlockParam
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..attention import AttentionSelection
from ..config import Config
from ..domain import ModelPurpose, SpeechReason, SpeechSituation
from ..logging import get_logger
from ..runtime import PetContext
from ..storage import Memory
from ..types import JSONObject
from ..work.narrative import SelectedMemory, WorkNarrative, semantic_revision
from . import story
from .contracts import ModelRequest
from .model import ModelClient
from .repetition import RepeatMatch, repeat_match
from .request import CommentaryRequest, PromptBuilder, SpeechConditions

log = get_logger("pet-daemon")


class StoryUnavailable(RuntimeError):
    pass


class CommentaryResult(BaseModel):
    speak: str | None
    request: CommentaryRequest
    mentions: tuple[str, ...] = ()


class SpeechResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    speak: str | None = Field(
        description="One brief spoken remark, or null for silence."
    )
    mentions: tuple[str, ...] = Field(
        default=(), description="Source IDs actually expressed in the spoken remark."
    )
    memory: str | None = Field(
        default=None, description="One eligible source ID chosen for idle recollection."
    )
    ongoing: bool = Field(
        default=False,
        description="Whether that source explicitly reports ongoing, paused, or remaining work.",
    )


class RequiredSpeechResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    speak: str = Field(description="One brief spoken remark.")
    mentions: tuple[str, ...] = ()
    memory: str | None = None
    ongoing: bool = False

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

    def schema(self, speech_required: bool = False) -> JSONObject:
        response = RequiredSpeechResponse if speech_required else SpeechResponse
        return cast(JSONObject, transform_schema(response))

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

    async def update_work_narrative(
        self,
        snapshot: PetContext,
        selection: AttentionSelection,
        refresh: Callable[[], PetContext] | None = None,
    ) -> WorkNarrative | None:
        narrative_request = story.prepare(
            snapshot,
            selection,
            self.memory,
            self.request,
            target_bytes=self.config.prompt_target_bytes,
        )
        if narrative_request is None:
            return None
        transient = selection.subject is None
        existing = (
            None
            if transient
            else self.memory.load_narrative(narrative_request.subject_id)
        )
        if (
            existing is not None
            and existing.revision == narrative_request.revision
            and existing.context_revision == narrative_request.context_revision
            and existing.sources == narrative_request.sources
            and narrative_request.latest_request_source in existing.intent.sources
        ):
            return existing
        try:
            raw = await self.complete(narrative_request.request, ModelPurpose.STORY)
            narrative = story.validate(raw, narrative_request)
            current = refresh() if refresh else snapshot
            refreshed = selection.refresh(current)
            if (
                not selection.is_valid(current)
                or selection.subject is not None
                and (
                    refreshed.subject is None
                    or semantic_revision(refreshed.subject.story, related=False)
                    != semantic_revision(selection.subject.story, related=False)
                )
            ):
                raise StoryUnavailable("Work changed during story generation")
            if (
                not transient
                and story.context_revision(refreshed, self.memory)
                == narrative_request.context_revision
            ):
                self.memory.save_narrative(narrative)
            return narrative
        except StoryUnavailable:
            raise
        except Exception as error:
            log.exception(
                "observation.story_failed",
                model=self.config.model,
                subject=selection.id,
                error=str(error),
            )
            raise StoryUnavailable("Work story could not be updated") from error

    async def generate_commentary(
        self,
        snapshot: PetContext,
        selection: AttentionSelection,
        refresh: Callable[[], PetContext] | None = None,
        eligible: Callable[[AttentionSelection, PetContext], bool] | None = None,
        memory_eligible: Callable[[SelectedMemory], bool] | None = None,
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
        if selection.situation == SpeechSituation.IDLE_REFLECTION:
            explicit = False
        manual = selection.context is not None or reason in (
            SpeechReason.CLICK,
            SpeechReason.EXPLICIT,
        )
        basic = False
        try:
            narrative = await self.update_work_narrative(snapshot, selection, refresh)
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
        commentary_request = self.prompts.prepare(
            snapshot,
            selection,
            self.request,
            explicit,
            narrative=narrative,
            basic=basic,
            excluded=tuple(record.text for record in self.memory.speech(limit=256)),
            memory_eligible=memory_eligible,
        )
        raw_result = await self.complete(commentary_request.request, reason)
        response = self.validate_speech(raw_result, explicit)
        speech = response.speak
        mentions = tuple(dict.fromkeys(response.mentions))
        if speech is None and mentions:
            raise StoryUnavailable("Silent commentary cited work")
        if response.memory is None and response.ongoing:
            raise StoryUnavailable("Ongoing work needs a cited memory")
        if any(ref not in commentary_request.source_ids for ref in mentions):
            raise StoryUnavailable("Speech cited unavailable evidence")
        if response.memory is not None:
            options = commentary_request.memory_options.get(response.memory)
            source = narrative.sources.get(response.memory) if narrative else None
            observed_at = source.get("observed_at") if source else None
            if (
                selection.situation != SpeechSituation.IDLE_REFLECTION
                or options is None
                or speech is not None
                and response.memory not in mentions
                or not isinstance(observed_at, (int, float))
                or not options[int(response.ongoing)]
            ):
                raise StoryUnavailable("Idle memory was not supported by the prompt")
            memory = SelectedMemory(
                key=response.memory,
                observed_at=observed_at,
                is_ongoing=response.ongoing,
            )
            if memory_eligible and not memory_eligible(memory):
                raise StoryUnavailable("Idle memory is no longer eligible")
            selection = selection.model_copy(update={"memory": memory})
            commentary_request = commentary_request.model_copy(
                update={
                    "selection": selection,
                    "conditions": commentary_request.conditions.model_copy(
                        update={"selection": selection}
                    ),
                }
            )
        elif (
            selection.situation == SpeechSituation.IDLE_REFLECTION
            and reason != SpeechReason.CLICK
            and speech
        ):
            raise StoryUnavailable("Idle reflection needs a supported memory")
        match = await self.find_repetition(speech)
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
                mentions = ()
        log.debug(
            "observation.decision",
            reason=reason,
            speaks=speech is not None,
            decision={"speak": speech},
        )
        supporting = (*mentions, *((selection.memory.key,) if selection.memory else ()))
        revisions = {
            key: revision
            for key in supporting
            if narrative is not None
            and isinstance(
                revision := narrative.sources.get(key, {}).get("scope_revision"), str
            )
        }
        commentary_request = commentary_request.model_copy(
            update={
                "conditions": commentary_request.conditions.model_copy(
                    update={"source_scope_revisions": revisions}
                )
            }
        )
        if not self.are_source_scopes_valid(commentary_request.conditions):
            raise StoryUnavailable("Supporting work changed during speech generation")
        return CommentaryResult(
            speak=speech, request=commentary_request, mentions=mentions
        )

    def are_source_scopes_valid(self, conditions: SpeechConditions) -> bool:
        return (
            self.memory.source_scope_revisions(tuple(conditions.source_scope_revisions))
            == conditions.source_scope_revisions
        )

    async def find_repetition(self, text: str | None) -> RepeatMatch | None:
        if not text or not text.strip():
            return None
        return await repeat_match(text, [row.text for row in self.memory.speech()])

    def validate_speech(
        self, result: dict[str, object], required: bool
    ) -> SpeechResponse | RequiredSpeechResponse:
        response = RequiredSpeechResponse if required else SpeechResponse
        return response.model_validate(result)
