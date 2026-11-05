from dataclasses import asdict, dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..coordination.models import (
    DelegatedTask,
    HarnessSession,
    Operation,
    OperationOutcome,
    TaskStatus,
)
from ..types import JSONObject

PROTOCOL_VERSION = 1
MAX_ADAPTER_BYTES = 128 * 1024
DELEGATION_ENV = "SIMPLE_PET_TASK_ID"
COMMAND_ENV = "SIMPLE_PET_COMMAND_ID"


@dataclass(frozen=True)
class AdapterRequest:
    task: DelegatedTask
    operation: Operation

    def packet(self) -> JSONObject:
        task, operation = self.task, self.operation
        return {
            "protocol_version": PROTOCOL_VERSION,
            "operation": operation.kind.value,
            "command_id": operation.id,
            "task_id": task.id,
            "scope_revision": operation.responsibility_revision,
            "instruction_revision": operation.instruction_revision,
            "task_revision": operation.task_revision,
            "instruction": operation.instruction,
            "expected_status": operation.expected_status.value
            if operation.expected_status
            else None,
            "expected_readiness": operation.expected_readiness.value
            if operation.expected_readiness
            else None,
            "user_revision": operation.task_user_revision,
            "expected_interaction_revision": operation.expected_adapter_revision,
            "observation": {
                "task_revision": task.revision,
                "user_revision": task.user_revision,
                "status": task.status.value,
                "readiness": task.readiness.value,
                "observed_at": task.observed_at,
                "input": task.observed_input[:4000] if task.observed_input else None,
                "input_observed_at": task.input_observed_at,
                "native_session": {
                    "provider": task.native_session.source.value,
                    "id": task.native_session.native_id,
                }
                if task.native_session
                else None,
            },
            "brief": {
                "objective": task.objective,
                "expected_result": task.expected_result,
                "repository": task.repository,
                "constraints": task.constraints,
                "sources": tuple(asdict(source) for source in task.sources),
            },
            "session": asdict(task.session) if task.session else None,
            "environment": {DELEGATION_ENV: task.id, COMMAND_ENV: operation.id},
        }


class AdapterResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    protocol_version: Literal[1]
    task_id: str = Field(min_length=1, max_length=128)
    command_id: str = Field(min_length=1, max_length=128)
    outcome: OperationOutcome
    observed_at: float = Field(allow_inf_nan=False)
    status: TaskStatus | None = None
    readiness: Literal["ready", "busy", "waiting", "unknown"] = "unknown"
    session: HarnessSession | None = None
    external_id: str | None = None
    error: str | None = None
    interaction_revision: str | None = Field(
        default=None, min_length=1, max_length=1024
    )
