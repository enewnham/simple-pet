import random
from pathlib import Path

import pytest

from pet_daemon.domain import (
    ArtifactKind,
    ExecutionOutcome,
    PullRequestState,
    SessionSource,
)
from pet_daemon.identity import SessionIdentity
from pet_daemon.repository_identity import RemoteRepository
from pet_daemon.storage import Memory
from pet_daemon.work.models import (
    ArtifactReference,
    ExecutionEvidence,
    IntentEvidence,
    ProgressEvidence,
    ReportContext,
    SessionReport,
)
from pet_daemon.work.narrative import NarrativeStatement, WorkNarrative
from pet_daemon.work.observations import (
    BranchObservation,
    CheckoutObservation,
    GitObservation,
    PullRequest,
    PullRequestObservation,
)


@pytest.mark.parametrize("kind", ["intent", "progress", "execution"])
def test_report_variants_retain_nested_context_after_restart(
    tmp_path: Path, kind: str
) -> None:
    session = SessionIdentity(source=SessionSource.CODEX, native_id="session")
    repository = RemoteRepository(value="github.com/team/project")
    payloads = {
        "intent": IntentEvidence(session=session, text="Check parser"),
        "progress": ProgressEvidence(session=session, text="Parser checked"),
        "execution": ExecutionEvidence(
            session=session,
            event="turn_done",
            outcome=ExecutionOutcome.SUCCESS,
            tool_name=None,
            progress="",
        ),
    }
    report = SessionReport(
        payload=payloads[kind],
        context=ReportContext(
            repository=repository,
            references=(
                ArtifactReference(
                    kind=ArtifactKind.BRANCH, ref="feature", repository=repository
                ),
            ),
        ),
        observed_at=1000,
    )
    memory = Memory(tmp_path / "memory.sqlite3", "fixture", clock=lambda: 1000)
    with memory.transaction():
        memory._work_repository.append_report(report)
    memory.close()
    memory = Memory(tmp_path / "memory.sqlite3", "fixture", clock=lambda: 1000)
    try:
        assert memory.narrative_reports((session,)) == (report,)
    finally:
        memory.close()


@pytest.mark.parametrize("kind", ["checkout", "branch", "pull_request"])
def test_repository_variants_retain_nested_observations_after_restart(
    tmp_path: Path, kind: str
) -> None:
    repository = RemoteRepository(value="github.com/team/project")
    pr = PullRequest(
        number=1,
        url="https://github.com/team/project/pull/1",
        state=PullRequestState.OPEN,
        merged_at=None,
        is_draft=False,
        repository=repository,
    )
    observations = {
        "checkout": CheckoutObservation(
            worktree="/project",
            local=GitObservation(
                worktree="/project",
                branch="feature",
                head="head",
                staged=0,
                unstaged=0,
                untracked=0,
                conflicts=0,
                upstream="",
                ahead=0,
                behind=0,
                repository=repository,
            ),
            checked_at=1000,
        ),
        "branch": BranchObservation(
            repository=repository, branch="feature", pr=pr, checked_at=1000
        ),
        "pull_request": PullRequestObservation(ref=pr.url, pr=pr, checked_at=1000),
    }
    observation = observations[kind]
    memory = Memory(tmp_path / "memory.sqlite3", "fixture", clock=lambda: 1000)
    with memory.transaction():
        memory._work_repository.put_observation(observation)
    memory.close()
    memory = Memory(tmp_path / "memory.sqlite3", "fixture", clock=lambda: 1000)
    try:
        assert memory.load_repositories() == (observation,)
    finally:
        memory.close()


def test_narrative_preserves_empty_and_false_source_values_after_restart(
    tmp_path: Path,
) -> None:
    narrative = WorkNarrative(
        subject_id=SessionIdentity(source=SessionSource.CODEX, native_id="session"),
        revision="revision",
        intent=NarrativeStatement(text="Check parser", sources=("report",)),
        sources={"report": {"empty": "", "false": False, "missing": None}},
    )
    memory = Memory(tmp_path / "memory.sqlite3", "fixture", clock=lambda: 1000)
    memory.save_narrative(narrative)
    memory.close()
    memory = Memory(tmp_path / "memory.sqlite3", "fixture", clock=lambda: 1000)
    try:
        assert memory.load_narrative(narrative.subject_id) == narrative
    finally:
        memory.close()


def test_prospective_consideration_retains_independent_history_values() -> None:
    from pet_daemon.attention import Attention
    from pet_daemon.config import Config
    from pet_daemon.domain import WorkSubjectKind
    from pet_daemon.storage.memory import AttentionHistory
    from pet_daemon.work.shared_work import WorkSubject, WorkUpdate
    from pet_daemon.work.stories import WorkStory

    identity = SessionIdentity(source=SessionSource.CODEX, native_id="session")
    attention = Attention(
        Config(pet_dir=Path(".")), clock=lambda: 1000, rng=random.Random(0)
    )
    attention.restore({identity: AttentionHistory(last_selected=100)})
    subject = WorkSubject(
        id=identity,
        kind=WorkSubjectKind.WORK,
        changed_at=1000,
        revision="subject",
        story=WorkStory(id=identity, intent="Check parser", revision="subject"),
    )
    update = WorkUpdate(session=identity, revision="session", observed_at=1000)
    first = attention.consideration(subject, update)[identity]
    second = attention.consideration(subject, update)[identity]
    first.last_selected = 200
    assert second.last_selected == 100
    assert first.considered_subject_revision == "subject"
    assert first.considered_session_revision == "session"
    assert attention.history(identity) == AttentionHistory(last_selected=100)
    attention.mark_considered(subject, update)
    attention.mark_selected(identity)
    assert first.last_selected == 200
    assert second.last_selected == 100
