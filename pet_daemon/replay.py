"""Replay sanitized event sequences through the commentary pipeline."""

import argparse
import asyncio
import json
import random
import tempfile
import time
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from anthropic import APIError

from .attention import Attention
from .commentary.cognition import Cognition, StoryUnavailable
from .commentary.contracts import ModelRequest
from .commentary.model import AnthropicModelClient, ModelClient
from .commentary.usage import collect_usage
from .config import Config
from .domain import ModelPurpose, SpeechReason
from .runtime import PetContext
from .sessions.tracking import Sessions
from .storage import Memory
from .work.narrative import semantic_revision


class FixtureModel:
    def __init__(self) -> None:
        self.speech: str | None = None

    async def complete(
        self, request: ModelRequest, reason: SpeechReason | ModelPurpose
    ) -> dict[str, object]:
        if reason != ModelPurpose.STORY:
            return {"speak": self.speech}
        payload = json.loads(request["messages"][0]["content"])
        latest = payload["current_work"]["latest_request_source"]
        sources = payload["current_moment"]["evidence"]
        intent = next(source for source in sources if source["id"] == latest)
        return {
            "intent": {"text": intent["text"][:400], "sources": [latest]},
            "decisions": [],
            "progress": [],
        }


class RecordingModel:
    def __init__(self, client: ModelClient) -> None:
        self.client = client
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self, request: ModelRequest, reason: SpeechReason | ModelPurpose
    ) -> dict[str, object]:
        started = time.monotonic()
        call: dict[str, Any] = {
            "purpose": str(reason),
            "request": deepcopy(request),
        }
        with collect_usage() as usage:
            try:
                response = await self.client.complete(request, reason)
                call["response"] = response
                return response
            except Exception as error:
                call["error"] = {
                    "type": type(error).__name__,
                    "message": str(error),
                }
                raise
            finally:
                call["latency_seconds"] = round(time.monotonic() - started, 3)
                call["usage"] = {
                    str(key): asdict(value) for key, value in usage.items()
                }
                self.calls.append(call)


def snapshot(sessions: Sessions, memory: Memory) -> PetContext:
    activity = sessions.activity_view()
    members = []
    for session in activity.sessions:
        context = memory.context(session.id)
        if context:
            session = replace(
                session,
                initial_request=context.initial_request,
                recent_requests=tuple(context.recent_requests),
                request_count=context.request_count,
            )
        members.append(session)
    return PetContext(
        activity=activity.activity,
        sessions=tuple(members),
        sleeping=False,
        work=tuple(
            replace(subject, revision=semantic_revision(subject.story))
            for subject in memory.connected_work(activity)
        ),
        convergence=activity.convergence,
    )


async def replay_sequence(
    fixture: dict[str, Any],
    *,
    client: ModelClient | None = None,
    model: str | None = None,
    settings: Config | None = None,
) -> dict[str, Any]:
    now = 1000.0
    offline = FixtureModel()
    recording = RecordingModel(client if client is not None else offline)
    with tempfile.TemporaryDirectory(prefix="pet-replay-") as directory:
        memory = Memory(Path(directory) / "memory.sqlite3", "replay", clock=lambda: now)
        config = (
            replace(settings, memory_file=Path(directory) / "memory.sqlite3")
            if settings
            else Config(pet_dir=Path(directory))
        )
        if model:
            config.model = model
        sessions = Sessions(lambda: now)
        attention = Attention(config, lambda: now, random.Random(0))
        cognition = Cognition(recording, config, memory=memory, clock=lambda: now)
        steps = []
        try:
            for index, event in enumerate(fixture["events"]):
                now += event.get("after_seconds", 1)
                for packet in event.get("packets", []):
                    result = sessions.handle(packet)
                    if result:
                        with memory.transaction():
                            memory.observe_session_context(result.observation)
                            memory.work.record_session(result.observation)
                if "reason" not in event:
                    continue
                current = snapshot(sessions, memory)
                reason = SpeechReason(event["reason"])
                selection = attention.select(current, reason)
                step: dict[str, Any] = {
                    "step": index,
                    "baseline": event.get("baseline"),
                    "review": event.get("review", fixture.get("review", "")),
                    "expected_speech": event.get("expected_speech"),
                }
                if client is None:
                    offline.speech = event.get("candidate_speech")
                start = len(recording.calls)
                if reason == SpeechReason.ROUTINE and (
                    selection.subject is None
                    or not attention.routine_eligible(selection.subject, current)
                ):
                    step.update(speech=None, decision="unchanged_work", accepted=False)
                else:
                    try:
                        generated = await cognition.generate_commentary(
                            current, selection
                        )
                    except (
                        StoryUnavailable,
                        APIError,
                        ValueError,
                        TimeoutError,
                    ) as error:
                        step.update(
                            speech=None,
                            decision="deferred"
                            if isinstance(error, StoryUnavailable)
                            else "error",
                            accepted=False,
                            error={
                                "type": type(error).__name__,
                                "message": str(error),
                            },
                        )
                    else:
                        selection = generated.prepared.selection
                        speech = generated.speak
                        accepted = bool(speech) and event.get("accepted", True)
                        step.update(
                            speech=speech,
                            decision="speech" if speech else "silence",
                            accepted=accepted,
                        )
                        if speech is None and selection.subject:
                            attention.mark_considered(selection.subject)
                            last_selected, considered = attention.history(
                                selection.subject.id
                            )
                            memory.save_attention_history(
                                selection.subject.id, last_selected, considered
                            )
                        if accepted:
                            if selection.subject:
                                attention.mark_considered(selection.subject)
                            subject_id = selection.id
                            last_selected, considered = (
                                attention.history(subject_id)
                                if subject_id
                                else (0, None)
                            )
                            memory.record_acceptance(
                                f"replay-{index}",
                                speech,
                                reason,
                                subject_id=subject_id,
                                revision=selection.subject.revision
                                if selection.subject
                                else None,
                                last_selected=last_selected,
                                considered_revision=considered,
                            )
                calls = deepcopy(recording.calls[start:])
                if client is None:
                    for call in calls:
                        call["latency_seconds"] = 0
                step["calls"] = calls
                steps.append(step)
        finally:
            memory.close()
    return {
        "name": fixture["name"],
        "mode": "dry_run" if client is None else "model_run",
        "steps": steps,
    }


async def run(args: argparse.Namespace) -> None:
    fixture_paths = (
        [args.fixtures]
        if args.fixtures.is_file()
        else sorted(args.fixtures.glob("*.json"))
    )
    fixtures = [json.loads(path.read_text()) for path in fixture_paths]
    if not fixtures:
        raise ValueError("No replay fixtures found")
    samples = args.samples if args.model_run else 1
    settings = Config.load(args.config) if args.config else None
    client = None
    anthropic = None
    if args.model_run:
        from anthropic import AsyncAnthropic

        anthropic = AsyncAnthropic()
        client = AnthropicModelClient(anthropic)
    try:
        results = []
        for fixture in fixtures:
            for sample in range(samples):
                result = await replay_sequence(
                    fixture, client=client, model=args.model, settings=settings
                )
                result["sample"] = sample + 1
                results.append(result)
        rendered = json.dumps(results, indent=2, ensure_ascii=False)
        if args.model_run or args.output:
            destination = args.output or Path(".cache/replays") / time.strftime(
                "%Y%m%d-%H%M%S"
            )
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "comparison.json").write_text(rendered + "\n")
            lines = [
                "# Commentary replay",
                "",
                "Review grounding, scope, voice, and whether silence fits.",
                "",
            ]
            for result in results:
                lines.extend([f"## {result['name']} · sample {result['sample']}", ""])
                for step in result["steps"]:
                    lines.extend(
                        [
                            f"- Baseline: {step['baseline'] or '(none recorded)'}",
                            f"- Candidate: {step['speech'] or '(silence)'}",
                            f"- Decision: {step['decision']}",
                            *(
                                [
                                    f"- Error: {step['error']['type']}: {step['error']['message']}"
                                ]
                                if "error" in step
                                else []
                            ),
                            f"- Review: {step['review']}",
                            "",
                        ]
                    )
            (destination / "comparison.md").write_text("\n".join(lines))
            print(destination)
        else:
            print(rendered)
    finally:
        if anthropic:
            await anthropic.close()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "fixtures", type=Path, nargs="?", default=Path("Tests/fixtures/replay")
    )
    parser.add_argument(
        "--model-run", action="store_true", help="Call the configured API model"
    )
    parser.add_argument("--model", help="Override the configured model")
    parser.add_argument(
        "--config", type=Path, help="Use this daemon configuration for prompts"
    )
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument(
        "--output", type=Path, help="Write comparison files into this directory"
    )
    args = parser.parse_args(argv)
    if args.samples < 1:
        parser.error("--samples must be positive")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
