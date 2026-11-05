from dataclasses import dataclass
from enum import StrEnum


class ChatRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class NoteOrigin(StrEnum):
    USER = "user"
    SOURCE = "source"
    HARNESS = "harness"
    AGENT = "agent"


class NoteKind(StrEnum):
    INSTRUCTION = "instruction"
    CORRECTION = "correction"
    DECISION = "decision"
    OBSERVATION = "observation"
    REPORT = "report"
    HYPOTHESIS = "hypothesis"
    SYNTHESIS = "synthesis"
    POINTER = "pointer"


class NoteStatus(StrEnum):
    ACTIVE = "active"
    NEEDS_RECHECK = "needs_recheck"
    SUPERSEDED = "superseded"
    REJECTED = "rejected"
    ARCHIVED = "archived"


class NoteSupport(StrEnum):
    TENTATIVE = "tentative"
    REPORTED = "reported"
    OBSERVED = "observed"
    SUPPORTED = "supported"
    CONFIRMED = "confirmed"


class ResponsibilityKind(StrEnum):
    FINITE = "finite"
    STANDING = "standing"


class ResponsibilityStatus(StrEnum):
    ACTIVE = "active"
    WAITING = "waiting"
    PAUSED = "paused"
    COMPLETED = "completed"


@dataclass(frozen=True)
class SourceReference:
    id: str
    provider: str
    revision: str
    observed_at: float
    uri: str = ""
    scope: str = ""
    original_at: float | None = None
    available: bool = True


@dataclass(frozen=True)
class ChatAction:
    id: str
    label: str
    kind: str
    target: str = ""
    revision: int | None = None


@dataclass(frozen=True)
class ChatMessage:
    id: str
    conversation_id: str
    role: ChatRole
    text: str
    created_at: float
    reply_to: str | None = None
    sources: tuple[SourceReference, ...] = ()
    continuations: tuple[str, ...] = ()
    actions: tuple[ChatAction, ...] = ()


@dataclass(frozen=True)
class Conversation:
    id: str
    created_at: float
    updated_at: float
    revision: int = 1
    pending_message_id: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class ChatRetry:
    id: str
    conversation_id: str
    message_id: str
    created_at: float


@dataclass(frozen=True)
class CoordinationState:
    revision: int = 0
    instruction_revision: int = 0
    paused: bool = False


@dataclass(frozen=True)
class Instruction:
    id: str
    text: str
    scope: str
    created_at: float
    message_id: str | None = None
    active: bool = True
    revision: int = 1


@dataclass(frozen=True)
class Note:
    id: str
    origin: NoteOrigin
    kind: NoteKind
    text: str
    scope: str
    created_at: float
    observed_at: float
    checked_at: float | None = None
    original_at: float | None = None
    sources: tuple[SourceReference, ...] = ()
    status: NoteStatus = NoteStatus.ACTIVE
    revision: int = 1
    supersedes: tuple[str, ...] = ()
    instruction_id: str | None = None
    condition: str | None = None
    support: NoteSupport = NoteSupport.TENTATIVE
    actor: str = "agent"


@dataclass(frozen=True)
class Project:
    id: str
    title: str
    scope: str
    aliases: tuple[str, ...] = ()
    sources: tuple[SourceReference, ...] = ()
    revision: int = 1


@dataclass(frozen=True)
class Responsibility:
    id: str
    purpose: str
    scope: str
    instruction_id: str
    created_at: float
    project_id: str | None = None
    kind: ResponsibilityKind = ResponsibilityKind.FINITE
    status: ResponsibilityStatus = ResponsibilityStatus.ACTIVE
    priority: int = 0
    completion_condition: str = ""
    permissions: tuple[str, ...] = ()
    sources: tuple[SourceReference, ...] = ()
    dependencies: tuple[str, ...] = ()
    followup_at: float | None = None
    waiting_on: str | None = None
    revision: int = 1


@dataclass(frozen=True)
class ActivityRecord:
    id: str
    kind: str
    title: str
    created_at: float
    entity_id: str | None = None
    detail: str = ""
    sources: tuple[SourceReference, ...] = ()


class TaskStatus(StrEnum):
    PREPARING = "preparing"
    RUNNING = "running"
    READY = "ready"
    WAITING = "waiting"
    ENDED = "ended"
    STOPPED = "stopped"


class OperationKind(StrEnum):
    LAUNCH = "launch"
    INSPECT = "inspect"
    CONTINUE = "continue"
    OPEN = "open"
    STOP = "stop"


class OperationOutcome(StrEnum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    FAILED = "failed"
    UNKNOWN = "unknown"


class Disposition(StrEnum):
    INVESTIGATE = "investigate"
    CONTINUE = "continue"
    ASK = "ask"
    WAIT = "wait"
    NO_FINDING = "no_finding"
    DEFER = "defer"


@dataclass(frozen=True)
class HarnessSession:
    adapter: str
    harness: str
    session: str
    workspace: str
    external_id: str


@dataclass(frozen=True)
class DelegatedTask:
    id: str
    responsibility_id: str
    objective: str
    expected_result: str
    created_at: float
    responsibility_revision: int
    repository: str = ""
    constraints: tuple[str, ...] = ()
    sources: tuple[SourceReference, ...] = ()
    status: TaskStatus = TaskStatus.PREPARING
    session: HarnessSession | None = None
    revision: int = 1


@dataclass(frozen=True)
class Operation:
    id: str
    task_id: str
    kind: OperationKind
    created_at: float
    task_revision: int
    responsibility_revision: int
    instruction_revision: int
    outcome: OperationOutcome = OperationOutcome.PENDING
    external_id: str | None = None
    error: str | None = None
    revision: int = 1


@dataclass(frozen=True)
class CandidateCause:
    kind: str
    source_id: str
    source_revision: str
    observed_at: float


@dataclass(frozen=True)
class Candidate:
    id: str
    question: str
    scope: str
    first_queued_at: float
    responsibility_id: str | None = None
    project_id: str | None = None
    causes: tuple[CandidateCause, ...] = ()
    last_serviced_at: float | None = None
    next_review_at: float | None = None
    waiting_on: str | None = None
    priority: int = 0
    revision: int = 1


@dataclass(frozen=True)
class Consideration:
    id: str
    candidate_id: str
    candidate_revision: int
    instruction_revision: int
    considered_at: float
    disposition: Disposition
    reason: str
    serviced_at: float | None = None
    sources: tuple[SourceReference, ...] = ()
    action_id: str | None = None
    next_review_at: float | None = None
    wake_condition: str | None = None


@dataclass(frozen=True)
class TaskAction:
    id: str
    kind: OperationKind
    responsibility_id: str
    task_id: str = ""
    instruction: str = ""
    objective: str = ""
    expected_result: str = ""
    repository: str = ""
    sources: tuple[SourceReference, ...] = ()
    constraints: tuple[str, ...] = ()


@dataclass(frozen=True)
class InquiryPass:
    id: str
    candidates: tuple[Candidate, ...]
    instruction_revision: int
    focus_id: str | None = None
    prior: tuple[Consideration, ...] = ()


@dataclass(frozen=True)
class CandidateDecision:
    candidate_id: str
    disposition: Disposition
    reason: str
    sources: tuple[SourceReference, ...] = ()
    action: TaskAction | None = None
    next_review_at: float | None = None
    wake_condition: str | None = None


@dataclass(frozen=True)
class InquiryResult:
    pass_id: str
    decisions: tuple[CandidateDecision, ...]
