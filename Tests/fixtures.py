import json
from pathlib import Path
from typing import TypedDict, Unpack

from pet_daemon.commentary.contracts import ModelRequest
from pet_daemon.domain import (
    Activity,
    ExecutionOutcome,
    PetState,
    ProcessStatus,
    SessionSource,
)
from pet_daemon.identity import SessionIdentity
from pet_daemon.runtime import (
    ActiveTool,
    AgentSessionStatus,
    FailureSnapshot,
    PetContext,
    ProcessIdentity,
    SwarmActivity,
    Trigger,
    UserSentiment,
    WaitingKind,
)
from pet_daemon.sessions.tracking import Sessions
from pet_daemon.storage import Memory
from pet_daemon.work import WorkSubject


def story_response(request: ModelRequest) -> dict[str, object]:
    payload = json.loads(request["messages"][0]["content"])
    latest = payload["current_work"]["latest_request_source"]
    evidence = payload["current_moment"]["evidence"]
    source = next(item for item in evidence if item["id"] == latest)
    progress = [
        {"text": item["text"][:400], "sources": [item["id"]]}
        for item in evidence
        if item["kind"] in ("reported_progress", "turn_ended") and item.get("text")
    ][-4:]
    return {
        "intent": {"text": source["text"][:400], "sources": [latest]},
        "decisions": [],
        "progress": progress,
    }


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
    id: SessionIdentity
    changed_at: float
    parent: SessionIdentity | None
    goal: str
    repo: str
    phase: Activity | None
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
    presence: ProcessStatus
    owner: ProcessIdentity | None
    owner_last_alive_seconds: float | None
    activity_uncertain: bool
    waiting_for_input: bool
    waiting_seconds: float | None
    waiting_kind: WaitingKind | None
    waiting_tool: str | None
    waiting_prompt: str | None
    active_tools: tuple[ActiveTool, ...]
    progress: str
    failure: FailureSnapshot | None
    outcome: ExecutionOutcome | None
    user_sentiment: UserSentiment


def session_snapshot(**values: Unpack[SessionOptions]) -> AgentSessionStatus:
    identity = values.get(
        "id", SessionIdentity(source=SessionSource.COMMAND, native_id="fixture")
    )
    source, name = identity.source.value, identity.native_id
    sessions = Sessions(lambda: 1000)
    sessions.handle({"event": "session_start", "source": source, "session": name})
    if values.get("waiting_for_input"):
        sessions.handle({"event": "waiting", "source": source, "session": name})
    snapshot = sessions.activity().sessions[0]
    return snapshot.model_copy(update=dict(**values))


def sessions_snapshot(*sessions: AgentSessionStatus) -> SwarmActivity:
    return SwarmActivity(activity=PetState.IDLE, sessions=sessions)


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
                session = session.model_copy(
                    update={
                        "initial_request": context.initial_request,
                        "summary": context.summary,
                        "recent_requests": tuple(context.recent_requests),
                        "request_count": context.request_count,
                        "first_summary_attempted": context.first_summary_attempted,
                    }
                )
            members.append(session)
        snapshot = snapshot.model_copy(update={"sessions": tuple(members)})
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
