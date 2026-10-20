from dataclasses import replace
from pathlib import Path
from typing import TypedDict, Unpack

from pet_daemon.runtime import (
    FailureSnapshot,
    ProcessIdentity,
    SentimentSnapshot,
    SessionSnapshot,
    SessionsSnapshot,
    ToolSnapshot,
    Trigger,
    WorkSnapshot,
)
from pet_daemon.sessions import Sessions
from pet_daemon.work import WorkSubject
from pet_daemon.work.models import RelatedLink
from pet_daemon.work.observations import GitObservation


class SubprocessOptions(TypedDict, total=False):
    cwd: str | Path | None
    stdout: int
    stderr: int


class StackHead(TypedDict):
    ref: str
    sha: str


class StackPullRequest(TypedDict):
    number: int
    state: str
    draft: bool
    merged_at: str | None
    head: StackHead


class StackFixture(TypedDict):
    number: int
    base: dict[str, str]
    pull_requests: list[StackPullRequest]


class BehaviorOptions(TypedDict, total=False):
    roaming: bool
    roaming_radius: float
    automatic_speech_cooldown_seconds: float
    naps: bool
    care: bool
    sentiment_reactions: bool
    care_interval: tuple[float, float]
    care_speech_interval: tuple[float, float]
    nap_interval: tuple[float, float]
    nap_seconds: tuple[float, float]
    baseline_dwell: float
    error_seconds: float
    walk_seconds: tuple[float, float]
    walk_speed: float
    idle_pause: tuple[float, float]
    working_pause: tuple[float, float]
    work_burst_seconds: tuple[float, float]
    pointer_radius: float
    resume_seconds: float
    gaze_reset_seconds: float
    gaze_deadzone: float


class PullRequestOptions(TypedDict, total=False):
    url: str
    state: str
    merged_at: str | None
    is_draft: bool | None
    head: str
    head_repository: str
    head_branch: str
    repository: str
    title: str
    body: str
    base_branch: str
    base_head: str
    related_links: tuple[RelatedLink, ...]
    related_links_error: str | None


class ObservationOptions(TypedDict, total=False):
    subject: str
    worktree: str
    branch: str
    sessions: tuple[str, ...]
    local: GitObservation | None
    current: bool
    revision: int
    changed_at: float
    local_checked_at: float
    github_checked_at: float
    github_attempted_at: float
    local_error: str | None
    github_error: str | None
    repository: str
    source: str


class SessionOptions(TypedDict, total=False):
    id: str
    changed_at: float
    parent: str | None
    goal: str
    repo: str
    phase: str
    first_prompt: str
    session_context: str
    recent_requests: tuple[str, ...]
    generation: int
    request_count: int
    first_summary_attempted: bool
    duration_seconds: float
    quiet_seconds: float
    turn_active: bool
    last_tool: str
    presence: str
    owner: ProcessIdentity | None
    owner_last_alive_seconds: float | None
    activity_uncertain: bool
    waiting_for_input: bool
    waiting_seconds: float | None
    active_tools: tuple[ToolSnapshot, ...]
    progress: str
    failure: FailureSnapshot | None
    outcome: str | None
    user_sentiment: SentimentSnapshot


def session_snapshot(**values: Unpack[SessionOptions]) -> SessionSnapshot:
    identity = values.get("id", "cli:fixture")
    harness, name = identity.split(":", 1)
    sessions = Sessions(lambda: 1000)
    sessions.handle({"event": "session_start", "harness": harness, "session": name})
    if values.get("waiting_for_input"):
        sessions.handle({"event": "waiting", "harness": harness, "session": name})
    snapshot = sessions.snapshot().sessions[0]
    return replace(snapshot, **values)


def sessions_snapshot(*sessions: SessionSnapshot) -> SessionsSnapshot:
    return SessionsSnapshot(activity="idle", sessions=sessions)


def capture_snapshot(
    snapshot: SessionsSnapshot,
    *,
    work: tuple[WorkSubject, ...] = (),
    trigger: Trigger | None = None,
) -> WorkSnapshot:
    return WorkSnapshot(
        activity=snapshot.activity,
        sessions=snapshot.sessions,
        sleeping=False,
        work=work,
        trigger=trigger,
        convergence=snapshot.convergence,
    )


def require[T](value: T | None) -> T:
    assert value is not None
    return value
