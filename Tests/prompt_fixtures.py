from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from html import unescape
from typing import Any

from pet_daemon.attention import AttentionSelection
from pet_daemon.commentary.context import MemoryCandidate
from pet_daemon.commentary.contracts import ActionOffer, RequestFactory
from pet_daemon.commentary.prompt import Prompt
from pet_daemon.commentary.request import CommentaryRequest, PromptBuilder, duration
from pet_daemon.config import Config
from pet_daemon.runtime import PetContext
from pet_daemon.work.narrative import SelectedMemory, WorkNarrative


class RecordingPromptBuilder(PromptBuilder):
    def __init__(self, config: Config, clock: Callable[[], float]) -> None:
        super().__init__(config, clock)
        self.requests: dict[str, CommentaryRequest] = {}
        self.actions: dict[str, list[dict[str, Any]]] = {}
        self.mentions: dict[str, list[dict[str, str]]] = {}

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
        self.actions[content] = [asdict(action) for action in actions]
        self.mentions[content] = [
            {
                "source_id": prepared.citations.label(source),
                "age": f"{duration(self.clock() - timestamp)} ago",
            }
            for source, timestamp in (mentioned_at or {}).items()
            if source in prepared.citations.sources
        ]
        return prepared

    def inspect_payload(self, request: Mapping[str, Any]) -> dict[str, Any]:
        content = request["messages"][0]["content"]
        observed_text = unescape(content)
        prepared = self.requests[content]
        for source in prepared.citations.sources:
            label = prepared.citations.label(source)
            if f"[{label}]" not in content:
                raise AssertionError(
                    f"Citation {label} is absent from the rendered request"
                )
        context = prepared.context
        texts = []
        if context.work:
            if context.work.request:
                texts.append(context.work.request.text)
            if context.work.story:
                texts.append(context.work.story.text)
        if context.moment:
            if context.moment.text:
                texts.append(context.moment.text)
            if context.moment.progress:
                texts.append(context.moment.progress.text)
        texts.extend(
            item.text if isinstance(item, MemoryCandidate) else item.statement.text
            for item in context.earlier
        )
        for text in texts:
            if text not in observed_text:
                raise AssertionError(
                    "Selected evidence is absent from the rendered request"
                )
        result: dict[str, Any] = dict(prepared.context.payload(prepared.citations))
        schema = request["output_config"]["format"]["schema"]
        if "headline" in schema["properties"]:
            result["available_actions"] = self.actions[content]
            result["recent_mentions"] = self.mentions[content]
            for action in self.actions[content]:
                description = (
                    f"- {action['id']} — {action['label']}: {action['description']}"
                )
                if description not in observed_text:
                    raise AssertionError(
                        f"Offered action {action['id']} is absent from the rendered request"
                    )
        return result
