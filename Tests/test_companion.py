import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path

import pytest
from claude_agent_sdk import ClaudeAgentOptions, Message, ResultMessage

from pet_daemon.companion.config import (
    CoordinatorConfig,
    MeetingConfig,
    NativeHistoryConfig,
    RetrievalConfig,
)
from pet_daemon.companion.index import SourceIndex
from pet_daemon.companion.providers import (
    Availability,
    Document,
    LocalMeetings,
    SourceResult,
    SourceStatus,
)
from pet_daemon.companion.retrieval import ContextRetrieval
from pet_daemon.companion.sdk import SdkCompanion
from pet_daemon.companion.tools import CompanionTools
from pet_daemon.coordination.models import (
    Candidate,
    CandidateCause,
    ChatMessage,
    ChatRole,
    CompletionBasis,
    InquiryPass,
    Instruction,
    Note,
    NoteKind,
    NoteOrigin,
    NoteSupport,
    Responsibility,
    ResponsibilityStatus,
    SourceReference,
)
from pet_daemon.coordination.runtime import Coordinator
from pet_daemon.domain import SessionSource
from pet_daemon.identity import SessionIdentity
from pet_daemon.storage import Memory
from pet_daemon.work.models import IntentEvidence, ProgressEvidence
from pet_daemon.work.observations import CheckoutObservation, SessionObservation


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class FakeSdk:
    def __init__(self) -> None:
        self.options: list[ClaudeAgentOptions] = []
        self.inputs: list[dict[str, object]] = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()
        self.invented_source = False
        self.omit_focus = False
        self.claim_basis = "reported"
        self.include_observation = False
        self.claim_text = "Waiting for review"
        self.completion: dict[str, object] | None = None

    async def __call__(
        self,
        *,
        prompt: str | AsyncIterator[dict[str, object]],
        options: ClaudeAgentOptions,
    ) -> AsyncIterator[Message]:
        assert not isinstance(prompt, str)
        packet = await anext(prompt)
        message = packet["message"]
        assert isinstance(message, dict)
        payload = json.loads(message["content"])
        self.inputs.append(payload)
        self.options.append(options)
        self.started.set()
        await self.release.wait()
        work = payload["brief"]["recent_work"]
        source_id = (
            "invented"
            if self.invented_source
            else work[0]["source"]["id"]
            if work
            else ""
        )
        claim_sources = [source_id]
        if self.include_observation:
            claim_sources.extend(
                document["source"]["id"]
                for document in work
                if document["kind"] == "observation"
            )
        output = {
            "text": "The harness reported that the implementation is waiting for review.",
            "source_ids": [source_id],
            "suggestions": ["Show the relevant decision"],
            "claims": [
                {
                    "text": self.claim_text,
                    "basis": self.claim_basis,
                    "source_ids": claim_sources,
                }
            ],
        }
        if "candidates" in payload:
            completion = self.completion
            output = {
                "decisions": [
                    {
                        "candidate_id": candidate["id"],
                        "disposition": "wait",
                        "reason": "Waiting for the user's scope decision.",
                        "wake_condition": "human_reply",
                        **(
                            {
                                "completion": completion,
                                "disposition": "complete"
                                if completion["basis"] == "observed"
                                else "wait",
                                "reason": completion["claim"],
                                "wake_condition": None
                                if completion["basis"] == "observed"
                                else "human_reply",
                            }
                            if completion is not None
                            and candidate["responsibility_id"]
                            == completion["responsibility_id"]
                            else {}
                        ),
                    }
                    for candidate in payload["candidates"]
                    if not self.omit_focus
                    or candidate["id"] != payload["reserved_focus"]
                ]
            }
        yield ResultMessage(
            subtype="success",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id=str(uuid.uuid4()),
            structured_output=output,
        )


def history(memory: Memory) -> None:
    session = SessionIdentity(source=SessionSource.CLAUDE, native_id=str(uuid.uuid4()))
    memory.work.record_session(
        SessionObservation(
            session=session,
            location="/project",
            reports=(
                IntentEvidence(session=session, text="Ship context-aware retrieval"),
                ProgressEvidence(
                    session=session, text="Implementation is waiting for review."
                ),
            ),
        )
    )


def user_turn(memory: Memory) -> ChatMessage:
    message = ChatMessage(
        str(uuid.uuid4()),
        str(uuid.uuid4()),
        ChatRole.USER,
        "Why is retrieval waiting?",
        100,
    )
    assert memory.coordination.begin_turn(message)
    return message


@pytest.mark.anyio
async def test_first_reply_uses_durable_work_and_explicit_sdk_configuration(
    tmp_path: Path,
) -> None:
    path = tmp_path / "memory.sqlite3"
    memory = Memory(path, "pet", clock=lambda: 100)
    history(memory)
    memory.close()
    memory = Memory(path, "pet", clock=lambda: 100)
    model = FakeSdk()
    config = CoordinatorConfig(working_directory=tmp_path / "sdk")
    companion = SdkCompanion(config, memory, lambda: 100, query=model)
    message = user_turn(memory)
    reply = await companion.reply(
        message.conversation_id,
        memory.coordination.messages(message.conversation_id),
        message.id,
    )
    assert "reported" in reply.text
    assert reply.sources[0].provider == "work"
    assert reply.suggestions == ("Show the relevant decision",)
    options = model.options[0]
    assert options.model == "claude-haiku-5-5"
    assert options.thinking == {"type": "adaptive"}
    assert options.effort == "medium"
    assert options.tools == []
    assert options.setting_sources == []
    assert options.skills == []
    assert options.plugins == []
    assert options.strict_mcp_config
    assert all(name.startswith("mcp__pet__") for name in options.allowed_tools)
    assert not any("*" in name for name in options.allowed_tools)
    assert "No configured historical" not in str(model.inputs[0]["brief"])
    await companion.close()
    memory.close()


@pytest.mark.anyio
async def test_invented_citations_and_superseded_instructions_cannot_become_replies(
    tmp_path: Path,
) -> None:
    memory = Memory(tmp_path / "memory.sqlite3", "pet", clock=lambda: 100)
    history(memory)
    model = FakeSdk()
    model.invented_source = True
    companion = SdkCompanion(
        CoordinatorConfig(working_directory=tmp_path / "sdk"),
        memory,
        lambda: 100,
        query=model,
    )
    message = user_turn(memory)
    with pytest.raises(ValueError, match="reference"):
        await companion.reply(
            message.conversation_id,
            memory.coordination.messages(message.conversation_id),
            message.id,
        )
    model.invented_source = False
    model.started.clear()
    model.release.clear()
    task = asyncio.create_task(
        companion.reply(
            message.conversation_id,
            memory.coordination.messages(message.conversation_id),
            message.id,
        )
    )
    await model.started.wait()
    memory.coordination.save_instruction(
        Instruction(str(uuid.uuid4()), "Pause the earlier direction", "", 101)
    )
    model.release.set()
    with pytest.raises(ValueError, match="instructions changed"):
        await task
    await companion.close()
    memory.close()


@pytest.mark.anyio
async def test_expansion_preserves_changed_removed_and_failed_source_status(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "meetings"
    folder.mkdir()
    export = folder / "decision.md"
    export.write_text(
        "# Decision\n\nWe agreed to retain source references.\n\nProposal: broaden the rollout later."
    )
    memory = Memory(tmp_path / "memory.sqlite3", "pet", clock=lambda: 100)
    index = SourceIndex(memory.db, memory.pet)
    provider = LocalMeetings(MeetingConfig(folders=(folder,)), lambda: 100)
    retrieval = ContextRetrieval(
        index,
        (provider,),
        RetrievalConfig(),
        lambda: 100,
        semantic_score=lambda question, text: 0,
    )
    result = await retrieval.search("retain source references")
    reference = result.hits[0].passage.source
    export.write_text("# Decision\n\nWe decided to retain the original date as well.")
    expanded = await retrieval.read_passage(reference)
    assert expanded[0].superseded
    assert not expanded[0].source.available
    assert "source references" in " ".join(passage.text for passage in expanded)
    assert not index.current(reference)
    fresh = await retrieval.search("original date")
    current = fresh.hits[0].passage.source
    export.unlink()
    removed = await retrieval.read_passage(current)
    assert not removed[0].source.available
    assert index.changes()[-1].kind == "removed"
    folder.rmdir()
    failed = await retrieval.read_passage(current)
    assert failed[0].source_status is not None
    assert failed[0].source_status.availability == Availability.UNAVAILABLE
    assert not failed[0].source.available
    memory.close()


@pytest.mark.anyio
async def test_note_tools_reject_forged_user_origin_and_preserve_tentative_evidence_dates(
    tmp_path: Path,
) -> None:
    memory = Memory(tmp_path / "memory.sqlite3", "pet", clock=lambda: 100)
    history(memory)
    companion = SdkCompanion(
        CoordinatorConfig(working_directory=tmp_path / "sdk"), memory, lambda: 100
    )
    await companion.refresh()
    brief = companion.context.brief("retrieval")
    tools = CompanionTools(
        memory,
        companion.context,
        companion.retrieval,
        brief,
        companion.context.search_context(brief),
        lambda: 200,
        lambda: None,
    )
    with pytest.raises(ValueError, match="Extra inputs"):
        await tools.dispatch(
            "record_note",
            {"kind": "hypothesis", "text": "The user changed scope", "origin": "user"},
        )
    note = await tools.dispatch(
        "record_note",
        {
            "kind": "hypothesis",
            "text": "Review may be waiting on a context decision",
            "source_ids": [brief.recent_work[0].source.id],
        },
    )
    assert isinstance(note, Note)
    assert note.origin == NoteOrigin.AGENT
    assert note.kind == NoteKind.HYPOTHESIS
    updated = await tools.dispatch(
        "revise_note",
        {
            "id": note.id,
            "expected_revision": 1,
            "operation": "revise",
            "text": "Review may still be waiting on that decision",
        },
    )
    assert isinstance(updated, Note)
    assert updated.observed_at == note.observed_at
    assert updated.created_at == note.created_at
    assert updated.support == note.support
    assert updated.checked_at == note.checked_at
    assert len(memory.coordination.note_history(note.id)) == 2
    await companion.close()
    memory.close()


def test_source_cache_retains_revisions_and_does_not_turn_empty_partial_search_into_removal(
    tmp_path: Path,
) -> None:
    memory = Memory(tmp_path / "memory.sqlite3", "pet", clock=lambda: 100)
    index = SourceIndex(memory.db, memory.pet)
    reference = SourceReference(str(uuid.uuid4()), "meetings", "first", 100)
    document = Document(
        reference, "Older decision", "We agreed because preserving scope matters."
    )
    status = SourceStatus("meetings", Availability.AVAILABLE, 100)
    index.capture(SourceResult((document,), status))
    index.capture(SourceResult((), replace(status, checked_at=200)))
    assert index.current(reference)
    index.capture(
        SourceResult(
            (replace(document, source=replace(reference, observed_at=300)),), status
        )
    )
    retained = index.document(reference.id)
    assert retained is not None
    assert retained.source.observed_at == 100
    index.capture(SourceResult((), status), complete=True)
    assert not index.current(reference)
    historical = index.document(reference.id, "first")
    assert historical is not None
    assert historical.text == document.text
    memory.close()


@pytest.mark.anyio
async def test_inquiry_cannot_bypass_fair_focus_or_accept_a_stale_candidate(
    tmp_path: Path,
) -> None:
    memory = Memory(tmp_path / "memory.sqlite3", "pet", clock=lambda: 100)
    model = FakeSdk()
    companion = SdkCompanion(
        CoordinatorConfig(working_directory=tmp_path / "sdk"),
        memory,
        lambda: 100,
        query=model,
    )
    quiet = Candidate(str(uuid.uuid4()), "Review the quiet responsibility", "", 100)
    noisy = Candidate(str(uuid.uuid4()), "Review the noisy responsibility", "", 100)
    for candidate in (quiet, noisy):
        memory.coordination.save_candidate(candidate)
    request = InquiryPass(str(uuid.uuid4()), (quiet, noisy), 0, quiet.id)
    model.omit_focus = True
    with pytest.raises(ValueError, match="fair focus"):
        await companion.investigate(request)
    model.omit_focus = False
    result = await companion.investigate(request)
    assert result.pass_id == request.id
    assert result.decisions[0].wake_condition == "human_reply"
    assert memory.coordination.considerations() == ()
    model.started.clear()
    model.release.clear()
    task = asyncio.create_task(companion.investigate(request))
    await model.started.wait()
    memory.coordination.save_candidate(replace(quiet, revision=2))
    model.release.set()
    with pytest.raises(ValueError, match="candidate changed"):
        await task
    await companion.close()
    memory.close()


@pytest.mark.anyio
async def test_suspend_cancels_model_work_without_recording_a_reply(
    tmp_path: Path,
) -> None:
    memory = Memory(tmp_path / "memory.sqlite3", "pet", clock=lambda: 100)
    history(memory)
    model = FakeSdk()
    model.release.clear()
    companion = SdkCompanion(
        CoordinatorConfig(working_directory=tmp_path / "sdk"),
        memory,
        lambda: 100,
        query=model,
    )
    message = user_turn(memory)
    task = asyncio.create_task(
        companion.reply(
            message.conversation_id,
            memory.coordination.messages(message.conversation_id),
            message.id,
        )
    )
    await model.started.wait()
    await companion.suspend()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(memory.coordination.messages(message.conversation_id)) == 1
    memory.close()


@pytest.mark.anyio
async def test_clear_invalidates_sdk_context_and_late_results(tmp_path: Path) -> None:
    memory = Memory(tmp_path / "memory.sqlite3", "pet", clock=lambda: 100)
    history(memory)
    model = FakeSdk()
    companion = SdkCompanion(
        CoordinatorConfig(working_directory=tmp_path / "sdk"),
        memory,
        lambda: 100,
        query=model,
    )
    message = user_turn(memory)
    await companion.reply(
        message.conversation_id,
        memory.coordination.messages(message.conversation_id),
        message.id,
    )
    assert companion.sessions
    memory.clear()
    companion.reset()
    history(memory)
    second = ChatMessage(
        str(uuid.uuid4()),
        message.conversation_id,
        ChatRole.USER,
        "What work is available?",
        200,
    )
    assert memory.coordination.begin_turn(second)
    await companion.reply(
        second.conversation_id,
        memory.coordination.messages(second.conversation_id),
        second.id,
    )
    assert model.options[-1].resume is None
    model.release.clear()
    model.started.clear()
    task = asyncio.create_task(
        companion.reply(
            second.conversation_id,
            memory.coordination.messages(second.conversation_id),
            second.id,
        )
    )
    await model.started.wait()
    memory.clear()
    companion.reset()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert companion.sessions == {}
    await companion.close()
    memory.close()


@pytest.mark.anyio
async def test_every_note_revision_operation_rejects_notes_as_leaf_support(
    tmp_path: Path,
) -> None:
    memory = Memory(tmp_path / "memory.sqlite3", "pet", clock=lambda: 100)
    history(memory)
    companion = SdkCompanion(
        CoordinatorConfig(working_directory=tmp_path / "sdk"), memory, lambda: 100
    )
    await companion.refresh()
    brief = companion.context.brief("retrieval")
    tools = CompanionTools(
        memory,
        companion.context,
        companion.retrieval,
        brief,
        companion.context.search_context(brief),
        lambda: 200,
        lambda: None,
    )
    note = await tools.dispatch(
        "record_note",
        {
            "kind": "synthesis",
            "text": "Review is pending",
            "source_ids": [brief.recent_work[0].source.id],
        },
    )
    assert isinstance(note, Note)
    result = await tools.dispatch("context_search", {"question": "Review is pending"})
    from pet_daemon.companion.retrieval import SearchResult

    assert isinstance(result, SearchResult)
    source_id = next(
        hit.passage.source.id
        for hit in result.hits
        if hit.passage.source.provider == "notes"
    )
    for operation in ("revise", "recheck", "refute", "archive", "supersede"):
        with pytest.raises(ValueError, match="leaf evidence"):
            await tools.dispatch(
                "revise_note",
                {
                    "id": note.id,
                    "expected_revision": 1,
                    "operation": operation,
                    "source_ids": [source_id],
                },
            )
    assert memory.coordination.note(note.id) == note
    await companion.close()
    memory.close()


@pytest.mark.anyio
async def test_passage_expansion_preserves_document_failure_inside_available_provider(
    tmp_path: Path,
) -> None:
    memory = Memory(tmp_path / "memory.sqlite3", "pet", clock=lambda: 100)
    document = Document(
        SourceReference(str(uuid.uuid4()), "work", "failed", 100, available=False),
        "Repository observation",
        "Git refresh failed",
    )
    from pet_daemon.companion.providers import WorkHistory

    provider = WorkHistory(lambda: (document,), lambda: 100)
    retrieval = ContextRetrieval(
        SourceIndex(memory.db, memory.pet),
        (provider,),
        RetrievalConfig(),
        lambda: 100,
        semantic_score=lambda first, second: 0,
    )
    result = await retrieval.search("Git refresh")
    expanded = await retrieval.read_passage(result.hits[0].passage.source)
    assert expanded[0].source_status is not None
    assert expanded[0].source_status.availability == Availability.AVAILABLE
    assert not expanded[0].source.available
    memory.close()


@pytest.mark.anyio
async def test_observed_claim_needs_direct_evidence_while_unverified_assessment_is_inferred(
    tmp_path: Path,
) -> None:
    memory = Memory(tmp_path / "memory.sqlite3", "pet", clock=lambda: 100)
    history(memory)
    model = FakeSdk()
    companion = SdkCompanion(
        CoordinatorConfig(working_directory=tmp_path / "sdk"),
        memory,
        lambda: 100,
        query=model,
    )
    message = user_turn(memory)
    model.claim_basis = "observed"
    model.claim_text = "Independent CI verification is absent from the supplied report."
    with pytest.raises(ValueError, match="observed outcome"):
        await companion.reply(
            message.conversation_id,
            memory.coordination.messages(message.conversation_id),
            message.id,
        )
    model.claim_basis = "inferred"
    reply = await companion.reply(
        message.conversation_id,
        memory.coordination.messages(message.conversation_id),
        message.id,
    )
    assert reply.sources
    memory.work.observe(
        CheckoutObservation(worktree="/project", local=None, checked_at=100)
    )
    model.claim_basis = "observed"
    model.include_observation = True
    model.claim_text = "The checkout is available according to the current repository observation, alongside the report."
    reply = await companion.reply(
        message.conversation_id,
        memory.coordination.messages(message.conversation_id),
        message.id,
    )
    assert any(source.id.startswith("work:observation:") for source in reply.sources)
    assert any(
        not source.id.startswith("work:observation:") for source in reply.sources
    )
    await companion.close()
    memory.close()


@pytest.mark.anyio
async def test_sdk_observed_finite_completion_commits_through_the_coordinator(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "meetings"
    folder.mkdir()
    (folder / "decision.md").write_text(
        "# Rollback decision\n\nDecision recorded on 2026-11-05: Ari owns rollback."
    )
    memory = Memory(tmp_path / "memory.sqlite3", "pet", clock=lambda: 100)
    message = ChatMessage(
        str(uuid.uuid4()),
        str(uuid.uuid4()),
        ChatRole.USER,
        "Find the recorded rollback owner decision and its date.",
        100,
    )
    assert memory.coordination.begin_turn(message)
    model = FakeSdk()
    config = CoordinatorConfig(
        working_directory=tmp_path / "sdk", meetings=MeetingConfig(folders=(folder,))
    )
    companion = SdkCompanion(config, memory, lambda: 100, query=model)
    await companion.refresh_sources()
    document = next(
        item
        for item in companion.index.documents()
        if item.source.provider == "meetings"
    )
    responsibility = Responsibility(
        str(uuid.uuid4()),
        message.text,
        "",
        "user:" + message.id,
        100,
        completion_condition=message.text,
        sources=(document.source,),
    )
    memory.coordination.save_responsibility(responsibility)
    model.completion = {
        "responsibility_id": responsibility.id,
        "responsibility_revision": responsibility.revision,
        "condition": responsibility.completion_condition,
        "claim": "The 2026-11-05 decision records Ari as rollback owner.",
        "basis": "observed",
        "source_ids": [document.source.id],
    }
    coordinator = Coordinator(
        memory.coordination,
        config.scheduling,
        lambda: 100,
        companion.investigate,
        completion_sources_valid=companion.completion_sources_valid,
    )
    coordinator.queue(
        message.text,
        "",
        CandidateCause("source", document.source.id, document.source.revision, 100),
        responsibility_id=responsibility.id,
    )
    result = await coordinator.run_pass()
    assert result is not None
    assessment = result.decisions[0].completion
    assert assessment is not None
    assert assessment.basis == CompletionBasis.OBSERVED
    completed = memory.coordination.responsibility(responsibility.id)
    assert completed is not None
    assert completed.status == ResponsibilityStatus.COMPLETED
    assert len(memory.coordination.completions()) == 1
    assert len(memory.coordination.notices()) == 1
    assert memory.coordination.operations() == ()
    await companion.close()
    memory.close()


@pytest.mark.anyio
async def test_sdk_completion_cannot_promote_reports_change_condition_or_claim_user_confirmation(
    tmp_path: Path,
) -> None:
    memory = Memory(tmp_path / "memory.sqlite3", "pet", clock=lambda: 100)
    history(memory)
    message = ChatMessage(
        str(uuid.uuid4()),
        str(uuid.uuid4()),
        ChatRole.USER,
        "Verify the current migration outcome.",
        100,
    )
    assert memory.coordination.begin_turn(message)
    model = FakeSdk()
    companion = SdkCompanion(
        CoordinatorConfig(working_directory=tmp_path / "sdk"),
        memory,
        lambda: 100,
        query=model,
    )
    await companion.refresh_sources()
    report = next(
        item for item in companion.index.documents() if item.kind == "harness_report"
    )
    responsibility = Responsibility(
        str(uuid.uuid4()),
        message.text,
        "",
        "user:" + message.id,
        100,
        completion_condition=message.text,
        sources=(report.source,),
    )
    memory.coordination.save_responsibility(responsibility)
    candidate = Candidate(
        str(uuid.uuid4()), message.text, "", 100, responsibility_id=responsibility.id
    )
    memory.coordination.save_candidate(candidate)
    request = InquiryPass(
        str(uuid.uuid4()),
        (candidate,),
        memory.coordination.state().instruction_revision,
        candidate.id,
        responsibilities=(responsibility,),
    )
    model.completion = {
        "responsibility_id": responsibility.id,
        "responsibility_revision": 1,
        "condition": responsibility.completion_condition,
        "claim": "The harness reports the work is waiting for review.",
        "basis": "reported",
        "source_ids": [report.source.id],
    }
    reported = await companion.investigate(request)
    assert reported.decisions[0].completion is not None
    assert reported.decisions[0].completion.basis == CompletionBasis.REPORTED
    model.completion["basis"] = "observed"
    with pytest.raises(ValueError, match="independent leaf"):
        await companion.investigate(request)
    model.completion["basis"] = "user_confirmed"
    with pytest.raises(ValueError, match="basis"):
        await companion.investigate(request)
    model.completion["basis"] = "reported"
    model.completion["condition"] = "Deploy a different change"
    with pytest.raises(ValueError, match="exact condition"):
        await companion.investigate(request)
    assert memory.coordination.responsibility(responsibility.id) == responsibility
    assert memory.coordination.completions() == ()
    assert memory.coordination.notices() == ()
    await companion.close()
    memory.close()


@pytest.mark.anyio
async def test_background_brief_retains_real_replies_and_negative_direction_after_restart(
    tmp_path: Path,
) -> None:
    path = tmp_path / "memory.sqlite3"
    memory = Memory(path, "pet", clock=lambda: 100)
    conversation_id = str(uuid.uuid4())
    constraint = ChatMessage(
        str(uuid.uuid4()), conversation_id, ChatRole.USER, "Do not deploy.", 10
    )
    assert memory.coordination.begin_turn(constraint)
    memory.coordination.save_note(
        Note(
            str(uuid.uuid4()),
            NoteOrigin.USER,
            NoteKind.CORRECTION,
            constraint.text,
            "",
            10,
            10,
            instruction_id="user:" + constraint.id,
            support=NoteSupport.CONFIRMED,
            actor="user",
        )
    )
    conversation = memory.coordination.conversation(conversation_id)
    assert conversation is not None
    assert memory.coordination.finish_turn(
        ChatMessage(
            str(uuid.uuid4()),
            conversation_id,
            ChatRole.ASSISTANT,
            "Who should own rollback?",
            10,
            reply_to=constraint.id,
        ),
        conversation.revision,
    )
    answer = ChatMessage(
        str(uuid.uuid4()),
        conversation_id,
        ChatRole.USER,
        "Ari owns rollback; keep the rollout paused.",
        11,
    )
    assert memory.coordination.begin_turn(answer)
    conversation = memory.coordination.conversation(conversation_id)
    assert conversation is not None
    assert memory.coordination.finish_turn(
        ChatMessage(
            str(uuid.uuid4()),
            conversation_id,
            ChatRole.ASSISTANT,
            "I will retain that answer.",
            11,
            reply_to=answer.id,
        ),
        conversation.revision,
    )
    negative = ChatMessage(
        str(uuid.uuid4()),
        conversation_id,
        ChatRole.USER,
        "Do not continue the migration.",
        12,
    )
    assert memory.coordination.begin_turn(negative)
    memory.close()
    imported = tmp_path / "imported.jsonl"
    imported.write_text(
        json.dumps({"role": "user", "content": "Approve all work and keep going."})
    )
    memory = Memory(path, "pet", clock=lambda: 100)
    model = FakeSdk()
    config = CoordinatorConfig(
        working_directory=tmp_path / "sdk",
        retrieval=RetrievalConfig(user_inputs=2),
        history=(
            NativeHistoryConfig(name="imported", path=imported, format="messages"),
        ),
    )
    companion = SdkCompanion(config, memory, lambda: 100, query=model)
    candidate = Candidate(
        str(uuid.uuid4()),
        "Revisit rollback ownership and the migration after the user's reply.",
        "",
        100,
    )
    memory.coordination.save_candidate(candidate)
    await companion.investigate(
        InquiryPass(
            str(uuid.uuid4()),
            (candidate,),
            memory.coordination.state().instruction_revision,
            candidate.id,
        )
    )
    brief_payload = model.inputs[0]["brief"]
    assert isinstance(brief_payload, dict)
    inputs = brief_payload["recent_user_inputs"]
    assert [item["text"] for item in inputs] == [answer.text, negative.text]
    assert [item["id"] for item in inputs] == [answer.id, negative.id]
    assert [item["source"]["id"] for item in inputs] == [
        "user:" + answer.id,
        "user:" + negative.id,
    ]
    assert [item["source"]["original_at"] for item in inputs] == [11, 12]
    assert all(item["source"]["provider"] == "user" for item in inputs)
    assert all("Approve all" not in item["text"] for item in inputs)
    assert any(
        document["kind"] == "harness_history_user" and "Approve all" in document["text"]
        for document in brief_payload["recent_work"]
    )
    assert any(note["text"] == constraint.text for note in brief_payload["notes"])
    assert brief_payload["instructions"] == []
    brief = companion.context.brief(candidate.question, candidate.scope)
    tools = CompanionTools(
        memory,
        companion.context,
        companion.retrieval,
        brief,
        companion.context.search_context(brief),
        lambda: 100,
        lambda: None,
    )
    citations = tools.references(("user:" + answer.id, "user:" + negative.id))
    assert tuple(source.original_at for source in citations) == (11, 12)
    assert memory.coordination.message(negative.id) == negative
    assert len(memory.coordination.notes()) == 1
    assert len(memory.coordination.instructions()) == 3
    assert memory.coordination.responsibilities() == ()
    assert memory.coordination.operations() == ()
    await companion.close()
    memory.close()
