from dataclasses import replace
from pathlib import Path
from typing import TypedDict, Unpack

from pet_daemon.runtime import (
    ActiveTool,
    FailureSnapshot,
    PetContext,
    ProcessIdentity,
    SessionContext,
    SwarmActivity,
    Trigger,
    UserSentiment,
)
from pet_daemon.sessions import Sessions
from pet_daemon.storage import Memory
from pet_daemon.work import WorkSubject


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


class SessionOptions(TypedDict, total=False):
    id: str
    changed_at: float
    parent: str | None
    goal: str
    repo: str
    phase: str
    initial_request: str
    summary: str
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
    active_tools: tuple[ActiveTool, ...]
    progress: str
    failure: FailureSnapshot | None
    outcome: str | None
    user_sentiment: UserSentiment


def session_snapshot(**values: Unpack[SessionOptions]) -> SessionContext:
    identity = values.get("id", "cli:fixture")
    harness, name = identity.split(":", 1)
    sessions = Sessions(lambda: 1000)
    sessions.handle({"event": "session_start", "harness": harness, "session": name})
    if values.get("waiting_for_input"):
        sessions.handle({"event": "waiting", "harness": harness, "session": name})
    snapshot = sessions.activity_view().sessions[0]
    return replace(snapshot, **values)


def sessions_snapshot(*sessions: SessionContext) -> SwarmActivity:
    return SwarmActivity(activity="idle", sessions=sessions)


def capture_snapshot(
    snapshot: SwarmActivity,
    *,
    work: tuple[WorkSubject, ...] = (),
    memory: Memory | None = None,
    trigger: Trigger | None = None,
) -> PetContext:
    if memory is not None:
        members = []
        for session in snapshot.sessions:
            context = memory.context(session.id)
            if context:
                session = replace(
                    session,
                    initial_request=context.initial_request,
                    summary=context.summary,
                    recent_requests=tuple(context.recent_requests),
                    request_count=context.request_count,
                    first_summary_attempted=context.first_summary_attempted,
                )
            members.append(session)
        snapshot = replace(snapshot, sessions=tuple(members))
    return PetContext(
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
