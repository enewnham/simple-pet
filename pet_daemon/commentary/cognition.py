import json
import time
from collections.abc import Callable, Sequence
from typing import cast

from anthropic import transform_schema
from anthropic.types import TextBlockParam
from pydantic import BaseModel

from ..attention import AttentionSelection
from ..config import Config
from ..domain import ModelPurpose, SpeechReason
from ..logging import get_logger
from ..runtime import PetContext
from ..storage import Memory
from ..types import JSONObject
from ..work.narrative import SelectedMemory, WorkNarrative, semantic_revision
from . import conversation, story
from .contracts import ActionOffer, ModelRequest
from .conversation import Exchange
from .model import ModelClient
from .prompt import select_prompt
from .repetition import RepeatMatch, repeat_match
from .request import CommentaryRequest, PromptBuilder, SpeechConditions

log = get_logger("pet-daemon")


class StoryUnavailable(RuntimeError):
    pass


class CommentaryResult(BaseModel):
    speak: str | None
    request: CommentaryRequest
    mentions: tuple[str, ...] = ()
    exchange: Exchange | None = None


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

    def request(
        self,
        payload: object,
        system: Sequence[TextBlockParam],
        *,
        response: type[BaseModel],
        min_tokens: int = 0,
    ) -> ModelRequest:
        return {
            "model": self.config.model,
            "max_tokens": max(self.config.max_tokens, min_tokens),
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
                    "schema": cast(JSONObject, transform_schema(response)),
                },
            },
        }

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
            raw = await self.client.complete(
                narrative_request.request,
                ModelPurpose.STORY,
                response_type=story.StoryResponse,
            )
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
        actions: Sequence[ActionOffer] = (),
    ) -> CommentaryResult:
        snapshot = refresh() if refresh else snapshot
        if eligible and not eligible(selection, snapshot):
            raise StoryUnavailable("Work is no longer eligible")
        reason = selection.reason
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
        prompt = select_prompt(selection)
        commentary_request = self.prompts.prepare(
            snapshot,
            selection,
            self.request,
            prompt=prompt,
            narrative=narrative,
            basic=basic,
            excluded=tuple(record.text for record in self.memory.speech(limit=256)),
            memory_eligible=memory_eligible,
            actions=actions,
            mentioned_at={
                source: history.mentioned_at
                for source, history in self.memory.load_source_attention().items()
                if history.mentioned_at is not None
            }
            if prompt.is_conversation
            else {},
        )
        raw_result = await self.client.complete(
            commentary_request.request, reason, response_type=prompt.response
        )
        response = prompt.response.model_validate(raw_result)
        exchange = None
        if isinstance(response, conversation.Response):
            offered = {action.id for action in actions}
            if any(choice.id not in offered for choice in response.choices):
                raise StoryUnavailable("Conversation selected an unavailable action")
            if response.speak is not None:
                exchange = Exchange(
                    headline=response.headline.strip(),
                    context=response.context.strip(),
                    choices=response.choices,
                )
        try:
            mentions = commentary_request.citations.resolve_many(response.mentions)
        except ValueError as error:
            raise StoryUnavailable("Speech cited unavailable evidence") from error
        if (
            exchange is not None
            and not mentions
            and any(choice.id not in {"more", "rounds"} for choice in exchange.choices)
        ):
            raise StoryUnavailable("Conversation work choices require a cited source")
        speech = response.speak
        if speech is None and mentions:
            raise StoryUnavailable("Silent commentary cited work")
        if isinstance(response, conversation.RecollectionResponse):
            commentary_request = self.select_memory(
                response, commentary_request, narrative, mentions, memory_eligible
            )
            selection = commentary_request.selection
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
                exchange = None
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
            speak=speech,
            request=commentary_request,
            mentions=mentions,
            exchange=exchange,
        )

    def select_memory(
        self,
        response: conversation.RecollectionResponse,
        request: CommentaryRequest,
        narrative: WorkNarrative | None,
        mentions: tuple[str, ...],
        eligible: Callable[[SelectedMemory], bool] | None,
    ) -> CommentaryRequest:
        if response.memory is None:
            if response.ongoing:
                raise StoryUnavailable("Ongoing work needs a cited memory")
            if request.selection.reason != SpeechReason.CLICK and response.speak:
                raise StoryUnavailable("Idle reflection needs a supported memory")
            return request
        try:
            source_id = request.citations.resolve(response.memory)
        except ValueError as error:
            raise StoryUnavailable("Speech cited unavailable evidence") from error
        options = request.memory_options.get(source_id)
        source = narrative.sources.get(source_id) if narrative else None
        observed_at = source.get("observed_at") if source else None
        if (
            options is None
            or response.speak is not None
            and source_id not in mentions
            or not isinstance(observed_at, (int, float))
            or not options[int(response.ongoing)]
        ):
            raise StoryUnavailable("Idle memory was not supported by the prompt")
        memory = SelectedMemory(
            key=source_id, observed_at=observed_at, is_ongoing=response.ongoing
        )
        if eligible and not eligible(memory):
            raise StoryUnavailable("Idle memory is no longer eligible")
        selection = request.selection.model_copy(update={"memory": memory})
        return request.model_copy(
            update={
                "selection": selection,
                "conditions": request.conditions.model_copy(
                    update={"selection": selection}
                ),
            }
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
