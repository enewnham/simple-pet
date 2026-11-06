from collections.abc import Callable, Mapping, Sequence
from html import unescape
from typing import Any

from pet_daemon.attention import AttentionSelection
from pet_daemon.commentary.contracts import ActionOffer, RequestFactory
from pet_daemon.commentary.prompt import Prompt
from pet_daemon.commentary.request import CommentaryRequest, PromptBuilder
from pet_daemon.config import Config
from pet_daemon.runtime import PetContext
from pet_daemon.work.narrative import SelectedMemory, WorkNarrative


class RecordingPromptBuilder(PromptBuilder):
    def __init__(self, config: Config, clock: Callable[[], float]) -> None:
        super().__init__(config, clock)
        self.requests: dict[str, CommentaryRequest] = {}
        self.actions: dict[str, tuple[ActionOffer, ...]] = {}

    def prepare(
        self,
        snapshot: PetContext,
        selection: AttentionSelection,
        make_request: RequestFactory,
        *,
        prompt: Prompt | None = None,
        narrative: WorkNarrative | None = None,
        basic: bool = False,
        excluded: Sequence[str] = (),
        memory_eligible: Callable[[SelectedMemory], bool] | None = None,
        actions: Sequence[ActionOffer] = (),
        mentioned_at: Mapping[str, float] | None = None,
        spoken_content_revision: str | None = None,
    ) -> CommentaryRequest:
        prepared = super().prepare(
            snapshot,
            selection,
            make_request,
            prompt=prompt,
            narrative=narrative,
            basic=basic,
            excluded=excluded,
            memory_eligible=memory_eligible,
            actions=actions,
            mentioned_at=mentioned_at,
            spoken_content_revision=spoken_content_revision,
        )
        content = prepared.request["messages"][0]["content"]
        self.requests[content] = prepared
        self.actions[content] = tuple(actions)
        return prepared

    def prepare_care(
        self,
        snapshot: PetContext,
        selection: AttentionSelection,
        make_request: RequestFactory,
        recent: Sequence[str],
    ) -> CommentaryRequest:
        prepared = super().prepare_care(snapshot, selection, make_request, recent)
        content = prepared.request["messages"][0]["content"]
        self.requests[content] = prepared
        self.actions[content] = ()
        return prepared

    def prepared(self, request: Mapping[str, Any]) -> CommentaryRequest:
        return self.requests[request["messages"][0]["content"]]

    def text(self, request: Mapping[str, Any]) -> str:
        return unescape(request["messages"][0]["content"])

    def offers(self, request: Mapping[str, Any]) -> tuple[ActionOffer, ...]:
        return self.actions[request["messages"][0]["content"]]
