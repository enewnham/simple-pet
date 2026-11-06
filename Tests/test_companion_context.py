import json
import uuid
from dataclasses import replace
from pathlib import Path

import pytest

from pet_daemon.companion.config import CoordinatorConfig
from pet_daemon.companion.providers import (
    Availability,
    Document,
    SourceResult,
    SourceStatus,
)
from pet_daemon.companion.sdk import SdkCompanion
from pet_daemon.companion.tools import CompanionTools, ToolAuthority, encoded
from pet_daemon.coordination.models import (
    Candidate,
    Consideration,
    Disposition,
    InquiryPass,
    SourceReference,
)
from pet_daemon.domain import PullRequestState
from pet_daemon.repository_identity import RemoteRepository
from pet_daemon.storage import Memory
from pet_daemon.work.observations import (
    BranchObservation,
    CheckoutObservation,
    GitObservation,
    PullRequest,
    PullRequestObservation,
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def checkout(
    path: str, repository: RemoteRepository, branch: str
) -> CheckoutObservation:
    return CheckoutObservation(
        worktree=path,
        checked_at=100,
        local=GitObservation(
            worktree=path,
            repository=repository,
            branch=branch,
            head="head",
            staged=0,
            unstaged=1,
            untracked=0,
            conflicts=0,
            upstream="origin/main",
            ahead=1,
            behind=0,
        ),
    )


@pytest.mark.anyio
async def test_branch_scope_selects_its_checkout_and_linked_pr_without_unrelated_payloads(
    tmp_path: Path,
) -> None:
    memory = Memory(tmp_path / "memory.sqlite3", "fixture", clock=lambda: 100)
    repository = RemoteRepository(value="github.com/fixture/pet")
    other = RemoteRepository(value="github.com/fixture/unrelated")
    for path, repo, branch in (
        ("/work/pet", repository, "main"),
        ("/work/pet-feature", repository, "feature"),
        ("/work/other", other, "main"),
    ):
        memory.work.observe(checkout(path, repo, branch))
    for number, repo, branch in (
        (1, repository, "main"),
        (2, repository, "feature"),
        (3, other, "main"),
    ):
        pr = PullRequest(
            number=number,
            url=f"https://{repo.value}/pull/{number}",
            state=PullRequestState.OPEN,
            merged_at=None,
            repository=repo,
            head_repository=repo,
            head_branch=branch,
            title="Migration owner decision"
            if number == 1
            else "Unrelated private work",
            body=("Relevant source details" if number == 1 else "UNRELATED_PAYLOAD")
            * 10000,
        )
        memory.work.observe(PullRequestObservation(ref=pr.url, pr=pr, checked_at=100))
        memory.work.observe(
            BranchObservation(repository=repo, branch=branch, pr=pr, checked_at=100)
        )
    old = PullRequest(
        number=4,
        url=f"https://{repository.value}/pull/4",
        state=PullRequestState.OPEN,
        merged_at=None,
        repository=repository,
        head_repository=repository,
        head_branch="main",
        head="old",
        title="UNRELATED_OLD_BRANCH",
        body="An older use of the same branch",
    )
    memory.work.observe(PullRequestObservation(ref=old.url, pr=old, checked_at=10))
    companion = SdkCompanion(
        CoordinatorConfig(working_directory=tmp_path / "sdk"), memory, lambda: 100
    )
    brief = companion.context.brief(
        "Do repository changes affect recent work?", "github.com/fixture/pet:main"
    )
    identifiers = {item.source.id for item in brief.recent_work}
    assert identifiers == {
        "work:observation:/work/pet",
        "work:observation:github.com/fixture/pet:main",
        "work:observation:https://github.com/fixture/pet/pull/1",
    }
    payload = encoded(brief)
    assert "UNRELATED_PAYLOAD" not in payload and "pet-feature" not in payload
    assert "UNRELATED_OLD_BRANCH" not in payload
    assert "Migration owner decision" in payload
    assert len(payload.encode()) < 10000
    pr_source = next(
        item.source for item in brief.recent_work if "/pull/1" in item.source.id
    )
    expanded = await companion.retrieval.read_passage(pr_source)
    assert any("Relevant source details" in passage.text for passage in expanded)
    await companion.close()
    memory.close()


@pytest.mark.anyio
async def test_first_checkpoint_coalesces_and_bounds_selected_history_with_explicit_expansion_pointers(
    tmp_path: Path,
) -> None:
    memory = Memory(tmp_path / "memory.sqlite3", "fixture", clock=lambda: 100)
    companion = SdkCompanion(
        CoordinatorConfig(working_directory=tmp_path / "sdk"), memory, lambda: 100
    )
    first = SourceReference("source:busy", "fixture", "0", 1, scope="/work/pet")
    for index in range(100):
        source = replace(first, revision=str(index), observed_at=index + 1)
        companion.index.capture(
            SourceResult(
                (Document(source, "Selected source", f"Revision {index}"),),
                SourceStatus("fixture", Availability.AVAILABLE, 100),
            )
        )
    for index in range(15):
        source = replace(first, id=f"source:peer:{index}")
        companion.index.capture(
            SourceResult(
                (Document(source, "Selected peer", "Meaningful source"),),
                SourceStatus("fixture", Availability.AVAILABLE, 100),
            )
        )
    last = replace(first, revision="latest")
    companion.index.capture(
        SourceResult(
            (Document(last, "Selected source", "Latest state"),),
            SourceStatus("fixture", Availability.AVAILABLE, 100),
        )
    )
    brief = companion.context.brief("Review recent changes", "/work/pet")
    assert len(brief.changes) == companion.config.retrieval.results
    assert len({change.source_id for change in brief.changes}) == len(brief.changes)
    assert (
        brief.changes[-1].source_id == first.id
        and brief.changes[-1].revision == "latest"
    )
    assert brief.changes_omitted == 116 - len(brief.changes)
    assert brief.history_since == 0 and brief.checkpoint == companion.index.checkpoint()
    tools = CompanionTools(
        memory,
        companion.context,
        companion.retrieval,
        brief,
        companion.context.search_context(brief),
        lambda: 100,
        lambda: None,
    )
    expanded = await tools.dispatch("relevant_changes", {"checkpoint": 0})
    assert (
        isinstance(expanded, dict)
        and expanded["changes_omitted"] == brief.changes_omitted
    )
    assert len(companion.index.changes()) == 116
    historical = await companion.retrieval.read_passage(first)
    assert (
        historical and historical[0].text == "Revision 0" and historical[0].superseded
    )
    await companion.close()
    memory.close()


@pytest.mark.anyio
async def test_long_lived_work_keeps_prior_excerpts_small_and_expands_exact_selected_history(
    tmp_path: Path,
) -> None:
    memory = Memory(tmp_path / "memory.sqlite3", "fixture", clock=lambda: 100)
    store = memory.coordination
    candidate = Candidate(str(uuid.uuid4()), "Review adopted work", "/work/pet", 100)
    store.save_candidate(candidate)
    for index in range(100):
        candidate = store.candidate(candidate.id)
        assert candidate is not None
        reason = (
            ("Supported prior finding. " * 80)
            if index < 99
            else "Long finding. " * 20000 + "FULL_HISTORY_TAIL"
        )
        store.record_consideration(
            Consideration(
                str(uuid.uuid4()),
                candidate.id,
                candidate.revision,
                0,
                index + 100,
                Disposition.NO_FINDING,
                reason,
                serviced_at=index + 100,
            )
        )
    candidate = store.candidate(candidate.id)
    assert candidate is not None
    companion = SdkCompanion(
        CoordinatorConfig(working_directory=tmp_path / "sdk"), memory, lambda: 100
    )
    request = InquiryPass(
        str(uuid.uuid4()), (candidate,), 0, prior=store.considerations()
    )
    orientation = companion._inquiry_orientation(request)
    prior_excerpts = orientation[0]["prior_considerations"]
    assert isinstance(prior_excerpts, tuple) and len(prior_excerpts) == 1
    assert orientation[0]["prior_considerations_omitted"] == 99
    assert len(encoded(orientation).encode()) < 10000
    brief = companion._focused_brief(request, candidate)
    assert len(brief.considerations) == companion.config.retrieval.results
    assert all(
        len(item.reason) <= companion.config.retrieval.passage_characters
        for item in brief.considerations
    )
    assert len(encoded(brief).encode()) < 40000
    latest = max(store.considerations(), key=lambda item: item.candidate_revision)
    assert latest.id in brief.consideration_excerpt_ids
    assert "FULL_HISTORY_TAIL" not in encoded(brief)
    tools = CompanionTools(
        memory,
        companion.context,
        companion.retrieval,
        brief,
        companion.context.search_context(brief),
        lambda: 100,
        lambda: None,
        authority=ToolAuthority(
            0, pass_id=request.id, candidates=((candidate.id, candidate.revision),)
        ),
    )
    expanded = await tools.dispatch(
        "current_work_context", {"consideration_id": latest.id}
    )
    assert isinstance(expanded, dict) and expanded["consideration"] == latest
    assert "FULL_HISTORY_TAIL" in latest.reason
    unrelated = Candidate(str(uuid.uuid4()), "Other work", "/work/other", 100)
    store.save_candidate(unrelated)
    other = Consideration(
        str(uuid.uuid4()),
        unrelated.id,
        unrelated.revision,
        0,
        100,
        Disposition.NO_FINDING,
        "Other finding",
        serviced_at=100,
    )
    store.record_consideration(other)
    with pytest.raises(ValueError, match="selected work"):
        await tools.dispatch("current_work_context", {"consideration_id": other.id})
    await companion.close()
    memory.close()


@pytest.mark.anyio
async def test_serviced_checkpoint_filters_changes_and_preserves_exact_prior_citations(
    tmp_path: Path,
) -> None:
    memory = Memory(tmp_path / "memory.sqlite3", "fixture", clock=lambda: 100)
    companion = SdkCompanion(
        CoordinatorConfig(working_directory=tmp_path / "sdk"), memory, lambda: 100
    )
    source = SourceReference(
        "work:observation:/work/pet", "work", "old", 50, scope="/work/pet"
    )
    companion.index.capture(
        SourceResult(
            (Document(source, "Recorded checkout", "Historical observation"),),
            SourceStatus("work", Availability.AVAILABLE, 50),
        )
    )
    checkpoint = companion.index.checkpoint()
    candidate = Candidate(
        str(uuid.uuid4()), "Review repository changes", "/work/pet", 50
    )
    prior = Consideration(
        str(uuid.uuid4()),
        candidate.id,
        1,
        0,
        60,
        Disposition.NO_FINDING,
        "No action from the recorded observation",
        serviced_at=60,
        sources=(source,),
        source_checkpoint=checkpoint,
    )
    memory.coordination.save_candidate(candidate)
    for index in range(50):
        unrelated = SourceReference(
            f"unrelated:{index}", "fixture", str(index), 100, scope="/work/other"
        )
        companion.index.capture(
            SourceResult(
                (Document(unrelated, "Unrelated", "Not selected"),),
                SourceStatus("fixture", Availability.AVAILABLE, 100),
            )
        )
    request = InquiryPass(str(uuid.uuid4()), (candidate,), 0, candidate.id, (prior,))
    brief = companion._focused_brief(request, candidate)
    assert brief.changes == ()
    assert brief.considerations == (prior,)
    tools = CompanionTools(
        memory,
        companion.context,
        companion.retrieval,
        brief,
        companion.context.search_context(brief),
        lambda: 100,
        lambda: None,
    )
    assert tools.references((source.id,)) == (source,)
    assert tools.unchanged_references((source.id,))
    new = replace(source, revision="new")
    companion.index.capture(
        SourceResult(
            (Document(new, "New observation", "Current state"),),
            SourceStatus("work", Availability.AVAILABLE, 100),
        )
    )
    brief = companion._focused_brief(request, candidate)
    assert len(brief.changes) == 1 and brief.changes[0].source_id == source.id
    assert brief.changes[0].id > checkpoint
    tools = CompanionTools(
        memory,
        companion.context,
        companion.retrieval,
        brief,
        companion.context.search_context(brief),
        lambda: 100,
        lambda: None,
    )
    citations = json.loads(encoded(tools.citations()))
    assert any(item["reference"]["revision"] == "old" for item in citations)
    assert any(item["reference"]["revision"] == "new" for item in citations)
    assert tools.references((f"{source.id}@new",)) == (new,)
    await companion.close()
    memory.close()
