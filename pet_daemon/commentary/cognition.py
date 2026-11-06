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
from . import care, conversation, story
from .contracts import ActionOffer, ModelRequest
from .conversation import Exchange
from .correction import CorrectableOutput
from .model import ModelClient
from .prompt import Response, select_prompt
from .repetition import RepeatMatch, repeat_match
from .request import CommentaryRequest, PromptBuilder, SpeechConditions
from .templates import render
from .usage import correction_attempt

log = get_logger("pet-daemon")


class CannotRespond(RuntimeError):
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
                {
                    "role": "user",
                    "content": payload
                    if isinstance(payload, str)
                    else json.dumps(payload, ensure_ascii=False),
                }
            ],
            "output_config": {
                "effort": "low",
                "format": {
                    "type": "json_schema",
                    "schema": cast(JSONObject, transform_schema(response)),
                },
            },
        }

    async def respond[T](
        self,
        request: ModelRequest,
        reason: SpeechReason | ModelPurpose,
        *,
        response_type: type[BaseModel],
        validate: Callable[[dict[str, object]], T],
        corrections_left: int = 1,
        attempt: int = 0,
    ) -> T:
        token = correction_attempt.set(attempt)
        try:
            draft = await self.client.complete(
                request, reason, response_type=response_type
            )
        finally:
            correction_attempt.reset(token)
        try:
            return validate(draft)
        except CorrectableOutput as error:
            if corrections_left <= 0:
                log.info(
                    "observation.correction_exhausted",
                    reason=reason,
                    issue=error.issue,
                    context=error.context,
                )
                raise
            log.info(
                "observation.correction",
                reason=reason,
                issue=error.issue,
                corrections_left=corrections_left,
            )
            feedback = render(
                "requests/correction.jinja",
                draft=json.dumps(draft, ensure_ascii=False, indent=2),
                issue=error.issue,
                **error.context,
            )
            corrected_request: ModelRequest = {
                **request,
                "messages": [
                    *request["messages"],
                    {"role": "user", "content": feedback},
                ],
            }
            return await self.respond(
                corrected_request,
                reason,
                response_type=response_type,
                validate=validate,
                corrections_left=corrections_left - 1,
                attempt=attempt + 1,
            )

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
            narrative = await self.respond(
                narrative_request.request,
                ModelPurpose.STORY,
                response_type=story.StoryResponse,
                validate=lambda draft: story.validate(draft, narrative_request),
            )
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
                raise CannotRespond("Work changed during story generation")
            if (
                not transient
                and story.context_revision(refreshed, self.memory)
                == narrative_request.context_revision
            ):
                self.memory.save_narrative(narrative)
            return narrative
        except CannotRespond:
            raise
        except Exception as error:
            log.exception(
                "observation.story_failed",
                model=self.config.model,
                subject=selection.id,
                error=str(error),
            )
            raise CannotRespond("Work story could not be updated") from error

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
            raise CannotRespond("Work is no longer eligible")
        if selection.reason == SpeechReason.CARE:
            if snapshot.sleeping or any(
                session.turn_active and not session.activity_uncertain
                for session in snapshot.sessions
            ):
                raise CannotRespond("Care chatter waits for an awake quiet moment")
            request = self.prompts.prepare_care(
                snapshot,
                selection,
                self.request,
                tuple(
                    row.text
                    for row in self.memory.speech(limit=3)
                    if row.reason == SpeechReason.CARE
                ),
            )
            draft = await self.client.complete(
                request.request, SpeechReason.CARE, response_type=care.Response
            )
            response = care.Response.model_validate(draft)
            speech = response.speak
            if await self.find_repetition(speech):
                speech = None
            return CommentaryResult(speak=speech, request=request)
        reason = selection.reason
        manual = selection.context is not None or reason in (
            SpeechReason.CLICK,
            SpeechReason.EXPLICIT,
        )
        basic = False
        try:
            narrative = await self.update_work_narrative(snapshot, selection, refresh)
        except CannotRespond:
            if not manual:
                raise
            snapshot = refresh() if refresh else snapshot
            selection = selection.refresh(snapshot)
            narrative = None
            basic = True
        snapshot = refresh() if refresh else snapshot
        selection = selection.refresh(snapshot)
        if eligible and not eligible(selection, snapshot):
            raise CannotRespond("Work is no longer eligible")
        prompt = select_prompt(selection)
        history = self.memory.load_attention_history().get(selection.id)
        commentary_request = self.prompts.prepare(
            snapshot,
            selection,
            self.request,
            prompt=prompt,
            narrative=narrative,
            basic=basic,
            spoken_content_revision=history.spoken_content_revision
            if history
            else None,
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
        try:
            result = await self.respond(
                commentary_request.request,
                reason,
                response_type=prompt.response,
                validate=lambda draft: self.validate_commentary(
                    draft,
                    response_type=prompt.response,
                    commentary_request=commentary_request,
                    narrative=narrative,
                    actions=actions,
                ),
            )
        except CorrectableOutput as error:
            raise CannotRespond("Commentary could not be corrected") from error
        commentary_request = result.request
        selection = commentary_request.selection
        if (
            selection.memory
            and memory_eligible
            and not memory_eligible(selection.memory)
        ):
            raise CannotRespond("Idle memory is no longer eligible")
        speech, mentions, exchange = result.speak, result.mentions, result.exchange
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
            raise CannotRespond("Supporting work changed during speech generation")
        return CommentaryResult(
            speak=speech,
            request=commentary_request,
            mentions=mentions,
            exchange=exchange,
        )

    def validate_commentary(
        self,
        draft: dict[str, object],
        *,
        response_type: type[Response],
        commentary_request: CommentaryRequest,
        narrative: WorkNarrative | None,
        actions: Sequence[ActionOffer],
    ) -> CommentaryResult:
        response = response_type.model_validate(draft)
        exchange = None
        if isinstance(response, conversation.Response):
            offered = {action.id for action in actions}
            if any(choice.id not in offered for choice in response.choices):
                raise CorrectableOutput(
                    "unavailable_action",
                    choices=tuple(
                        choice.id
                        for choice in response.choices
                        if choice.id not in offered
                    ),
                    actions=tuple(action.id for action in actions),
                )
            if response.speak is not None:
                exchange = Exchange(
                    headline=response.headline.strip(),
                    context=response.context.strip(),
                    choices=response.choices,
                )
        try:
            mentions = commentary_request.citations.resolve_many(response.mentions)
        except ValueError as error:
            raise CorrectableOutput(
                "unknown_citation",
                citations=response.mentions,
                available_citations=commentary_request.citations.labels(
                    commentary_request.citations.sources
                ),
            ) from error
        if (
            exchange is not None
            and not mentions
            and any(choice.id not in {"more", "rounds"} for choice in exchange.choices)
        ):
            raise CorrectableOutput("missing_work_citation")
        speech = response.speak
        if speech is None and mentions:
            raise CorrectableOutput("silent_citation")
        if isinstance(response, conversation.RecollectionResponse):
            commentary_request = self.select_memory(
                response, commentary_request, narrative, mentions
            )
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
    ) -> CommentaryRequest:
        if response.memory is None:
            if response.ongoing:
                raise CorrectableOutput(
                    "unsupported_memory", violation="ongoing_without_memory"
                )
            if request.selection.reason != SpeechReason.CLICK and response.speak:
                raise CorrectableOutput(
                    "unsupported_memory", violation="reflection_without_memory"
                )
            return request
        try:
            source_id = request.citations.resolve(response.memory)
        except ValueError as error:
            raise CorrectableOutput(
                "unknown_citation",
                citations=(response.memory,),
                available_citations=request.citations.labels(request.citations.sources),
            ) from error
        options = request.memory_options.get(source_id)
        source = narrative.sources.get(source_id) if narrative else None
        observed_at = source.get("observed_at") if source else None
        if options is None or not isinstance(observed_at, (int, float)):
            raise CorrectableOutput(
                "unsupported_memory",
                violation="unavailable_memory",
                memory=response.memory,
            )
        if response.speak is not None and source_id not in mentions:
            raise CorrectableOutput(
                "unsupported_memory", violation="uncited_memory", memory=response.memory
            )
        if not options[int(response.ongoing)]:
            raise CorrectableOutput(
                "unsupported_memory",
                violation="unsupported_status",
                memory=response.memory,
                ongoing=response.ongoing,
            )
        memory = SelectedMemory(
            key=source_id, observed_at=observed_at, is_ongoing=response.ongoing
        )
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
