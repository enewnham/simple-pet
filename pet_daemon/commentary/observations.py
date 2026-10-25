from ..domain import Activity, ExecutionOutcome
from ..types import JSONObject

EVENTS = {
    "work_update": "An observation of the selected work is available.",
    "permission_request": "The work is waiting for the developer to approve an operation.",
    "input_requested": "The work is waiting for an answer to a question.",
    "waiting": "The work is waiting for input.",
    "failed_call": "An operation reported a failure.",
    "turn_ended": "The current work segment ended.",
    "request_received": "The developer sent a message about this work.",
    "quiet_time": "There is no confirmed active work to react to.",
    "explicit_context": "The developer supplied context for a reaction.",
    "request": "The developer supplied a request or follow-up.",
    "reported_progress": "A report describes progress on this work.",
}
OUTCOMES = {
    ExecutionOutcome.SUCCESS: "The work segment ended without a reported execution error. This alone does not establish task completion.",
    ExecutionOutcome.ERROR: "The work segment ended with a reported execution error.",
    ExecutionOutcome.CANCELLED: "The work segment was cancelled. The report may describe partial progress.",
    ExecutionOutcome.STALE: "The execution status became stale; its result is unknown.",
}
ACTIVITIES = {
    Activity.INVESTIGATING: "code investigation",
    Activity.EDITING: "code editing",
    Activity.TESTING: "test execution",
    Activity.BUILDING: "build execution",
    Activity.CHECKING: "work verification",
    Activity.COMMAND: "command execution",
    Activity.SOURCE: "coding tool execution",
}
STATES = {
    "working": "Execution is active for this work.",
    "quiet": "No current execution is confirmed for this work.",
}


def describe_observation(observation: JSONObject) -> JSONObject:
    result = dict(observation)
    if "kind" in result:
        result["event"] = EVENTS[str(result.pop("kind"))]
    if "outcome" in result:
        result["execution"] = OUTCOMES[ExecutionOutcome(str(result.pop("outcome")))]
    if "state" in result:
        result["activity"] = STATES[str(result.pop("state"))]
    if "operation" in result:
        result["operation"] = ACTIVITIES[Activity(str(result["operation"]))]
    return result
