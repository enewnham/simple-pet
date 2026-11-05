import asyncio
import time
import uuid
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from ..coordination.models import (
    DelegatedTask,
    Operation,
    OperationKind,
    OperationOutcome,
    Responsibility,
    ResponsibilityStatus,
    TaskAction,
    TaskReadiness,
    TaskStatus,
)
from ..identity import SessionIdentity
from ..runtime import HookEvent, HookEventKind
from ..storage.coordination import SQLiteCoordinationRepository
from ..storage.transactions import transaction
from .client import AdapterClient, AdapterError
from .protocol import AdapterRequest, AdapterResult


class AuthorityError(ValueError):
    pass


class Delegation:
    def __init__(
        self,
        store: SQLiteCoordinationRepository,
        client: AdapterClient,
        *,
        clock: Callable[[], float] = time.time,
        changed: Callable[[DelegatedTask], None] | None = None,
        coding_slots: int = 1,
    ) -> None:
        self.store = store
        self.client = client
        self.clock = clock
        self.changed = changed
        self.coding_slots = coding_slots
        self.suspended = False
        self.lock = asyncio.Lock()
        self.unowned_hooks: dict[SessionIdentity, tuple[HookEvent, ...]] = {}
        self.inflight: dict[str, tuple[OperationKind, asyncio.Task[object]]] = {}

    async def dispatch(self, action: TaskAction) -> Operation:
        async with self.lock:
            previous = self.store.operation(action.id)
            if previous is not None:
                if previous.kind != action.kind or (
                    action.task_id and previous.task_id != action.task_id
                ):
                    raise AuthorityError(
                        "The command identity belongs to different work"
                    )
                return previous
            if action.kind == OperationKind.LAUNCH:
                self._require_coding_slot()
                task = self._launch_task(action)
            else:
                task = self._task(action.task_id)
            if action.kind in (OperationKind.LAUNCH, OperationKind.CONTINUE):
                task = self._authorize(action, task)
                if action.kind == OperationKind.CONTINUE:
                    self._require_coding_slot(exclude_task=task.id)
            elif (
                action.kind in (OperationKind.STOP, OperationKind.OPEN)
                and task.session is None
            ):
                raise AuthorityError(
                    "The selected task has no established native session"
                )
            operation = self._operation(action, task)
            with transaction(self.store.db):
                previous_task = self.store.task(task.id)
                if previous_task is None or previous_task.revision < task.revision:
                    self.store.save_task(task)
                self.store.save_operation(operation)
            current = asyncio.current_task()
            assert current is not None
            self.inflight[task.id] = (operation.kind, current)
            try:
                return await self._request(task, operation)
            finally:
                self.inflight.pop(task.id, None)

    def _active_coding_tasks(self) -> set[str]:
        operations = self.store.operations()
        unresolved = {
            operation.task_id
            for operation in operations
            if operation.kind
            in (OperationKind.LAUNCH, OperationKind.CONTINUE, OperationKind.STOP)
            and operation.outcome
            in (OperationOutcome.PENDING, OperationOutcome.UNKNOWN)
        }
        active = set(unresolved)
        for task in self.store.tasks():
            if task.status in (TaskStatus.ENDED, TaskStatus.STOPPED, TaskStatus.READY):
                continue
            launches = tuple(
                operation
                for operation in operations
                if operation.task_id == task.id
                and operation.kind == OperationKind.LAUNCH
            )
            failed_preparation = (
                task.session is None
                and task.native_session is None
                and launches
                and all(
                    operation.outcome == OperationOutcome.FAILED
                    for operation in launches
                )
            )
            if not failed_preparation:
                active.add(task.id)
        return active

    def available_coding_slots(self) -> int:
        return max(0, self.coding_slots - len(self._active_coding_tasks()))

    def _require_coding_slot(self, *, exclude_task: str | None = None) -> None:
        occupied = self._active_coding_tasks()
        if exclude_task is not None:
            occupied.discard(exclude_task)
        if len(occupied) >= self.coding_slots:
            raise AuthorityError(
                "An owned coding task occupies the slot. Wait for its observed execution to end or stop that task."
            )

    def _launch_task(self, action: TaskAction) -> DelegatedTask:
        responsibility = self.store.responsibility(action.responsibility_id)
        if responsibility is None:
            raise AuthorityError("Adopt a responsibility before launching work")
        if (
            not action.objective.strip()
            or not action.expected_result.strip()
            or not action.repository
        ):
            raise AuthorityError(
                "A task needs an objective, expected result, and repository"
            )
        if len(action.objective) > 16000 or len(action.expected_result) > 16000:
            raise AuthorityError("The task brief exceeds its bounded scope")
        if OperationKind.INSPECT not in self.client.config.capabilities:
            raise AuthorityError("Launching work requires adapter inspection")
        self._repository_scope(action.repository, responsibility)
        return DelegatedTask(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, "simple-pet:task:" + action.id)),
            responsibility_id=responsibility.id,
            objective=action.objective,
            expected_result=action.expected_result,
            created_at=self.clock(),
            responsibility_revision=responsibility.revision,
            repository=action.repository,
            sources=action.sources,
            constraints=action.constraints,
        )

    @staticmethod
    def _repository_scope(repository: str, responsibility: Responsibility) -> None:
        scope = responsibility.scope
        if not scope or not (
            repository == scope
            or (
                Path(scope).is_absolute()
                and Path(repository).resolve().is_relative_to(Path(scope).resolve())
            )
        ):
            raise AuthorityError("The repository is outside the adopted responsibility")

    def _authorize(self, action: TaskAction, task: DelegatedTask) -> DelegatedTask:
        responsibility = self.store.responsibility(task.responsibility_id)
        state = self.store.state()
        if responsibility is None or responsibility.id != action.responsibility_id:
            raise AuthorityError("The task requires its adopted responsibility")
        if (
            self.suspended
            or state.paused
            or responsibility.status != ResponsibilityStatus.ACTIVE
        ):
            raise AuthorityError(
                "Coordination is paused or the responsibility is inactive"
            )
        if action.kind.value not in responsibility.permissions:
            raise AuthorityError(
                "The adopted responsibility does not permit this operation"
            )
        expected = (
            (action.expected_task_revision, task.revision),
            (action.expected_responsibility_revision, responsibility.revision),
            (action.expected_instruction_revision, state.instruction_revision),
            (action.expected_user_revision, task.user_revision),
        )
        if any(
            captured is not None and captured != current
            for captured, current in expected
        ):
            raise AuthorityError("The instruction changed before dispatch")
        if action.kind == OperationKind.CONTINUE:
            if not action.instruction.strip() or len(action.instruction) > 16000:
                raise AuthorityError("Continuation requires a bounded instruction")
            if (
                task.session is None
                or task.readiness != TaskReadiness.READY
                or task.status != TaskStatus.READY
                or task.adapter_revision is None
            ):
                raise AuthorityError("The owned harness is not known to be ready")
            if action.sources and action.sources != task.sources:
                raise AuthorityError("The task evidence changed before dispatch")
            unresolved = any(
                operation.task_id == task.id
                and operation.outcome
                in (OperationOutcome.PENDING, OperationOutcome.UNKNOWN)
                and operation.kind
                in (OperationKind.LAUNCH, OperationKind.CONTINUE, OperationKind.STOP)
                for operation in self.store.operations()
            )
            if unresolved:
                raise AuthorityError(
                    "Inspect the unresolved operation before continuing"
                )

        if task.responsibility_revision != responsibility.revision:
            captured = (
                action.expected_task_revision,
                action.expected_responsibility_revision,
                action.expected_instruction_revision,
                action.expected_user_revision,
            )
            current = (
                task.revision,
                responsibility.revision,
                state.instruction_revision,
                task.user_revision,
            )
            if action.kind != OperationKind.CONTINUE or captured != current:
                raise AuthorityError(
                    "The responsibility changed after this task was adopted"
                )
            self._repository_scope(task.repository, responsibility)
            return replace(
                task,
                revision=task.revision + 1,
                responsibility_revision=responsibility.revision,
            )
        return task

    def responsibility_changed(self, responsibility: Responsibility) -> None:
        for task in self.store.tasks():
            if task.responsibility_id != responsibility.id:
                continue
            changed = replace(
                task,
                revision=task.revision + 1,
                readiness=TaskReadiness.UNKNOWN,
                adapter_revision=None,
                last_error="The adopted responsibility changed. Inspect and reassess before continuing.",
            )
            self.store.save_task(changed)
            pending = self.inflight.get(task.id)
            if pending is not None and pending[0] == OperationKind.CONTINUE:
                pending[1].cancel()
            self._changed(changed)

    def _operation(self, action: TaskAction, task: DelegatedTask) -> Operation:
        return Operation(
            id=action.id,
            task_id=task.id,
            kind=action.kind,
            created_at=self.clock(),
            task_revision=task.revision,
            responsibility_revision=task.responsibility_revision,
            instruction_revision=self.store.state().instruction_revision,
            instruction=action.instruction,
            expected_status=task.status,
            expected_readiness=task.readiness,
            evidence_sources=task.sources,
            task_user_revision=task.user_revision,
            expected_adapter_revision=task.adapter_revision,
        )

    async def _request(self, task: DelegatedTask, operation: Operation) -> Operation:
        try:
            result = await self.client.request(AdapterRequest(task, operation))
        except asyncio.CancelledError:
            self._uncertain(
                task.id,
                operation,
                "Observation interrupted before the outcome was established",
            )
            raise
        except AdapterError as error:
            return self._uncertain(task.id, operation, str(error), error.outcome)
        return self._accept(task.id, operation, result)

    def _uncertain(
        self,
        task_id: str,
        operation: Operation,
        error: str,
        outcome: OperationOutcome = OperationOutcome.UNKNOWN,
    ) -> Operation:
        task = self._task(task_id)
        updated = replace(
            operation, revision=operation.revision + 1, outcome=outcome, error=error
        )
        with transaction(self.store.db):
            self.store.save_operation(updated)
            task = replace(
                task,
                revision=task.revision + 1,
                readiness=TaskReadiness.UNKNOWN,
                adapter_revision=None,
                last_error=error,
            )
            self.store.save_task(task)
        self._changed(task)
        return updated

    def _accept(
        self, task_id: str, operation: Operation, result: AdapterResult
    ) -> Operation:
        task = self._task(task_id)
        if (
            task.session is not None
            and result.session is not None
            and task.session != result.session
        ):
            same_external = (
                replace(task.session, session=result.session.session) == result.session
            )
            if not same_external or not self._known_native(
                task, result.session.harness, result.session.session
            ):
                return self._uncertain(
                    task_id, operation, "The adapter returned a different owned session"
                )
        if (
            task.native_session is not None
            and result.session is not None
            and not self._known_native(
                task, result.session.harness, result.session.session
            )
        ):
            return self._uncertain(
                task_id,
                operation,
                "The receipt conflicts with the observed native session",
            )
        session = result.session or task.session
        updated = replace(
            operation,
            revision=operation.revision + 1,
            outcome=result.outcome,
            external_id=result.external_id
            or (session.external_id if session else None),
            error=result.error,
        )
        preserves_observation = (
            operation.kind == OperationKind.OPEN
            or result.observed_at < (task.observed_at or float("-inf"))
        )
        was_redirected = task.user_revision != operation.task_user_revision
        changed = replace(
            task,
            revision=task.revision + 1,
            session=session,
            status=task.status
            if preserves_observation
            else result.status or TaskStatus.UNKNOWN,
            readiness=task.readiness
            if preserves_observation
            else TaskReadiness.UNKNOWN
            if was_redirected
            else TaskReadiness(result.readiness),
            observed_at=task.observed_at
            if preserves_observation
            else result.observed_at,
            last_error="Native input changed during dispatch. Reassess the task before continuing."
            if was_redirected
            else result.error,
            adapter_revision=task.adapter_revision
            if preserves_observation
            else None
            if was_redirected
            else result.interaction_revision,
        )
        with transaction(self.store.db):
            self.store.save_operation(updated)
            self.store.save_task(changed)
        self._changed(changed)
        if changed.session is not None:
            for identity in tuple(self.unowned_hooks):
                if (
                    identity.source.value == changed.session.harness
                    and identity.native_id == changed.session.session
                ) or self._known_native(
                    changed, identity.source.value, identity.native_id
                ):
                    for accepted_hook in self.unowned_hooks.pop(identity):
                        self.observe_hook(accepted_hook, identity)
        return updated

    async def inspect(self, task_id: str) -> DelegatedTask:
        async with self.lock:
            task = self._task(task_id)
            unresolved = tuple(
                operation
                for operation in self.store.operations()
                if operation.task_id == task_id
                and operation.outcome
                in (OperationOutcome.PENDING, OperationOutcome.UNKNOWN)
                and operation.kind != OperationKind.INSPECT
            )
            if unresolved:
                for operation in unresolved:
                    await self._inspect_operation(self._task(task_id), operation)
            else:
                action = TaskAction(
                    str(uuid.uuid4()),
                    OperationKind.INSPECT,
                    task.responsibility_id,
                    task.id,
                )
                operation = self._operation(action, task)
                self.store.save_operation(operation)
                await self._request(task, operation)
            return self._task(task_id)

    async def _inspect_operation(
        self, task: DelegatedTask, operation: Operation
    ) -> None:
        inspection = replace(operation, kind=OperationKind.INSPECT)
        try:
            result = await self.client.request(AdapterRequest(task, inspection))
        except AdapterError as error:
            self._uncertain(task.id, operation, str(error))
            return
        self._accept(task.id, operation, result)

    async def reconcile(self) -> None:
        for task in self.store.tasks():
            if task.session is not None or any(
                operation.task_id == task.id for operation in self.store.operations()
            ):
                await self.inspect(task.id)

    async def open_task(
        self, task_id: str, *, command_id: str | None = None
    ) -> Operation:
        task = self._task(task_id)
        return await self.dispatch(
            TaskAction(
                command_id or str(uuid.uuid4()),
                OperationKind.OPEN,
                task.responsibility_id,
                task.id,
            )
        )

    async def stop_task(
        self, task_id: str, *, command_id: str | None = None
    ) -> Operation:
        task = self._task(task_id)
        return await self.dispatch(
            TaskAction(
                command_id or str(uuid.uuid4()),
                OperationKind.STOP,
                task.responsibility_id,
                task.id,
            )
        )

    @staticmethod
    def _known_native(task: DelegatedTask, harness: str, session: str) -> bool:
        identities = task.native_aliases + (
            (task.native_session,) if task.native_session is not None else ()
        )
        return any(
            identity.source.value == harness and identity.native_id == session
            for identity in identities
        )

    def observe_hook(
        self, hook: HookEvent, accepted_identity: SessionIdentity | None = None
    ) -> bool:
        if hook.parent:
            return False
        identity = accepted_identity or hook.subject
        if hook.delegation_id is not None or hook.delegation_command_id is not None:
            if not hook.delegation_id or not hook.delegation_command_id:
                return False
            task = self.store.task(hook.delegation_id)
            operation = self.store.operation(hook.delegation_command_id)
            if (
                task is None
                or operation is None
                or operation.task_id != task.id
                or operation.kind not in (OperationKind.LAUNCH, OperationKind.CONTINUE)
            ):
                return False
        else:
            owned = tuple(
                task
                for task in self.store.tasks()
                if task.session is not None
                and task.session.adapter == self.client.config.name
                and (
                    (
                        task.session.harness == identity.source.value
                        and task.session.session == identity.native_id
                    )
                    or self._known_native(
                        task, identity.source.value, identity.native_id
                    )
                )
            )
            if len(owned) != 1:
                self.unowned_hooks[identity] = (
                    self.unowned_hooks.get(identity, ()) + (hook.model_copy(deep=True),)
                )[-16:]
                if len(self.unowned_hooks) > 128:
                    self.unowned_hooks.pop(next(iter(self.unowned_hooks)))
                return False
            task = owned[0]
        if (
            identity != hook.subject
            and not self._known_native(task, identity.source.value, identity.native_id)
            and identity.native_id not in hook.aliases
        ):
            return False
        identities = tuple(
            dict.fromkeys(
                (accepted_identity or hook.subject, hook.subject)
                + tuple(
                    SessionIdentity(source=hook.source, native_id=identifier)
                    for identifier in hook.aliases
                    + ((hook.runtime_session,) if hook.runtime_session else ())
                )
            )
        )
        if task.native_session is not None and not set(identities).intersection(
            task.native_aliases + (task.native_session,)
        ):
            return False
        if task.session is not None and not any(
            task.session.harness == identity.source.value
            and task.session.session == identity.native_id
            for identity in identities + task.native_aliases
        ):
            return False
        status = TaskStatus.RUNNING
        readiness = TaskReadiness.BUSY
        if hook.event in (
            HookEventKind.TURN_DONE,
            HookEventKind.TURN_ERROR,
            HookEventKind.INTERRUPT,
        ):
            status, readiness = TaskStatus.ENDED, TaskReadiness.UNKNOWN
        elif hook.event == HookEventKind.WAITING:
            status, readiness = TaskStatus.WAITING, TaskReadiness.WAITING
        elif hook.event == HookEventKind.SESSION_END:
            status, readiness = TaskStatus.UNKNOWN, TaskReadiness.UNKNOWN
        elif hook.event == HookEventKind.SESSION_START:
            status, readiness = TaskStatus.PREPARING, TaskReadiness.UNKNOWN
        changed = replace(
            task,
            revision=task.revision + 1,
            native_session=accepted_identity or hook.subject,
            native_aliases=tuple(dict.fromkeys(task.native_aliases + identities)),
            status=status,
            readiness=readiness,
            observed_at=self.clock(),
            user_revision=task.user_revision + (hook.event == HookEventKind.INTENT),
            adapter_revision=None,
            observed_input=hook.goal[:4000]
            if hook.event == HookEventKind.INTENT and hook.goal is not None
            else task.observed_input,
            input_observed_at=self.clock()
            if hook.event == HookEventKind.INTENT
            else task.input_observed_at,
        )
        self.store.save_task(changed)
        pending = self.inflight.get(task.id)
        if (
            hook.event == HookEventKind.INTENT
            and pending is not None
            and pending[0] == OperationKind.CONTINUE
        ):
            pending[1].cancel()
        self._changed(changed)
        return True

    def _task(self, task_id: str) -> DelegatedTask:
        task = self.store.task(task_id)
        if task is None:
            raise AuthorityError("The task is not owned by this coordinator")
        if task.session is not None and task.session.adapter != self.client.config.name:
            raise AuthorityError(
                "The owned session belongs to a different configured adapter"
            )
        return task

    def _changed(self, task: DelegatedTask) -> None:
        if self.changed is not None:
            self.changed(task)
