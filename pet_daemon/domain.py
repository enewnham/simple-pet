from enum import StrEnum


class SessionSource(StrEnum):
    COMMAND = "cli"
    CLAUDE = "claude"
    CODEX = "codex"
    CURSOR = "cursor"


class Activity(StrEnum):
    INVESTIGATING = "investigating"
    EDITING = "editing"
    TESTING = "testing"
    BUILDING = "building"
    CHECKING = "checking"
    COMMAND = "command"
    HARNESS = "harness"


class ExecutionOutcome(StrEnum):
    SUCCESS = "success"
    ERROR = "error"
    CANCELLED = "cancelled"
    STALE = "stale"


class ProcessStatus(StrEnum):
    UNKNOWN = "unknown"
    EXITED = "exited"
    ALIVE = "alive"
    REPLACED = "replaced"


class PetState(StrEnum):
    IDLE = "idle"
    WORKING = "working"
    ERROR = "error"


class CareKind(StrEnum):
    EXERCISE = "exercise"
    ATTENTION = "attention"
    CONSTERNATION = "consternation"


class ExecutionPhase(StrEnum):
    IDLE = "idle"
    EXECUTING = "executing"
    SETTLING = "settling"
    SUSPENDED = "suspended"


class ExecutionTransition(StrEnum):
    START = "start"
    FINISH = "finish"
    READY = "ready"
    SUSPEND = "suspend"
    RESUME = "resume"


class ActionKind(StrEnum):
    FOCUS = "focus"
    WALK = "walk"
    NAP = "nap"
    REST = "rest"
    REACTION = "reaction"
    ERROR = "error"
    SUCCESS = "success"


class SpeechReason(StrEnum):
    PROMPT = "prompt"
    COMPLETION = "completion"
    FAILURE = "failure"
    CHECK_IN = "check_in"
    ROUTINE = "routine"
    EXPLICIT = "explicit"
    CLICK = "click"
    CARE = "care"


class SpeechSituation(StrEnum):
    PROMPT = "prompt"
    ROUTINE = "routine"
    COMPLETION = "completion"
    FAILURE = "failure"
    EXPLICIT = "explicit"
    CLICK_WAITING = "click_waiting"
    CLICK_WORKING = "click_working"
    CLICK_IDLE = "click_idle"
    CARE_EXERCISE = "care_exercise"
    CARE_ATTENTION = "care_attention"
    CARE_CONSTERNATION = "care_consternation"
    CHECK_IN_WAITING = "check_in_waiting"
    CHECK_IN_RUNNING = "check_in_running"
    CHECK_IN_QUIET = "check_in_quiet"


class DeliveryStatus(StrEnum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    CANCELLED = "cancelled"
    FAILED = "failed"


class RepeatMethod(StrEnum):
    EXACT = "exact"
    SEMANTIC = "semantic"


class ModelPurpose(StrEnum):
    STORY = "story"


class ArtifactKind(StrEnum):
    BRANCH = "branch"
    PULL_REQUEST = "pull_request"
    DOCUMENT = "document"
    TICKET = "ticket"
    PATCH = "patch"


class EvidenceSource(StrEnum):
    PR_REFERENCE = "pr_reference"
    USER = "user"
    HARNESS = "harness"
    GIT = "git"
    GITHUB = "github"
    OBSERVER = "observer"
    AGENT_REFERENCE = "agent_reference"
    GITHUB_STACK = "github_stack"


class EvidenceKind(StrEnum):
    INTENT = "intent"
    PROGRESS = "progress"
    EXECUTION = "execution"
    OBSERVATION = "observation"
    AVAILABILITY = "availability"
    RELATIONSHIP = "relationship"


class WorkSubjectKind(StrEnum):
    WORK = "work"
    SESSION = "session"


class PullRequestState(StrEnum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    MERGED = "MERGED"


class LinkRelationship(StrEnum):
    STACK_ENTRY = "stack_entry"


class CollectionKind(StrEnum):
    CHECKOUT = "checkout"
    BRANCH = "branch"
    PULL_REQUEST = "pull_request"
