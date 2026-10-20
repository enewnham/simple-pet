from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from ..runtime import HookEvent

from ..runtime import SessionsSnapshot
from .association import (
    collection_targets,
    observe_repository,
    record_intent,
    record_session_event,
    work_context,
)
from .models import ArtifactDescriptor, Opportunity, WorkAccount, WorkItem, WorkState
from .observations import RepositoryObservation, RepositoryTarget
from .opportunities import derive_opportunities, eligible, project
from .views import WorkSubject, WorkView, build_work, project_work


class WorkRepository(Protocol):
    def transaction(self) -> AbstractContextManager[None]: ...
    def load(self) -> WorkState: ...
    def save(self, state: WorkState) -> None: ...
    def clear(self) -> None: ...


class WorkService:
    def __init__(
        self,
        repository: WorkRepository,
        clock: Callable[[], float] = time.time,
        retention_days: int = 30,
    ) -> None:
        self.repository = repository
        self.clock = clock
        self.retention_days = retention_days

    def _change[T](self, apply: Callable[[WorkState], T]) -> T:
        with self.repository.transaction():
            state = self.repository.load()
            result = apply(state)
            derive_opportunities(
                state, self.clock(), lambda work_id: work_context(state, work_id)
            )
            self.repository.save(state)
        return result

    def session_intent(
        self, session: str, intent: str, aliases: tuple[str, ...] = ()
    ) -> str:
        return self._change(
            lambda state: record_intent(state, session, intent, aliases, self.clock())
        )

    def session_event(self, session: str, event: HookEvent) -> None:
        self._change(
            lambda state: record_session_event(state, session, event, self.clock())
        )

    def observe_repository(self, observation: RepositoryObservation) -> None:
        self._change(lambda state: observe_repository(state, observation, self.clock()))

    def session_context(self, session: str) -> WorkAccount | None:
        state = self.repository.load()
        work = state.current(session)
        return work_context(state, work.id) if work else None

    def work_context(self, work_id: str) -> WorkAccount | None:
        return work_context(self.repository.load(), work_id)

    def snapshot(self) -> WorkState:
        return self.repository.load()

    def views(self, runtime_snapshot: SessionsSnapshot) -> tuple[WorkSubject, ...]:
        return build_work(runtime_snapshot, self.repository.load())

    def session_view(self, session: str) -> WorkView | None:
        state = self.repository.load()
        work = state.current(session)
        return self._view(state, work.id) if work else None

    def work_view(self, work_id: str) -> WorkView:
        return self._view(self.repository.load(), work_id)

    def _view(self, state: WorkState, work_id: str) -> WorkView:
        actions = tuple(
            project(item)
            for item in state.opportunities.values()
            if item.work_id == work_id and eligible(item)
        )
        account = work_context(state, work_id)
        assert account is not None
        return project_work(work_id, account, actions)

    def work_for_session(self, session: str) -> WorkItem | None:
        return self.repository.load().current(session)

    def work_items(self) -> tuple[WorkItem, ...]:
        return tuple(self.repository.load().items.values())

    def session_names(self, work_id: str) -> tuple[str, ...]:
        return tuple(
            sorted(
                session
                for identifier, session in self.repository.load().memberships
                if identifier == work_id
            )
        )

    def collection_targets(self) -> tuple[RepositoryTarget, ...]:
        return collection_targets(self.repository.load())

    def tracked_artifacts(self) -> tuple[ArtifactDescriptor, ...]:
        return tuple(
            item.descriptor()
            for item in sorted(
                self.repository.load().artifacts.values(),
                key=lambda item: (item.kind, item.ref),
            )
        )

    def opportunities(self) -> tuple[Opportunity, ...]:
        records = self.repository.load().opportunities.values()
        return tuple(
            project(item)
            for item in sorted(records, key=lambda item: (-item.changed_at, item.id))
            if eligible(item)
        )

    def opportunity(self, identifier: str) -> Opportunity | None:
        item = self.repository.load().opportunities.get(identifier)
        return project(item) if item and eligible(item) else None

    def is_current(self, identifier: str, version: int) -> bool:
        item = self.opportunity(identifier)
        return bool(item and item.version == version)

    def prune(self) -> None:
        cutoff = self.clock() - self.retention_days * 86400
        with self.repository.transaction():
            state = self.repository.load()
            latest = {}
            for item in state.evidence:
                latest[item.work_id, item.artifact_id, item.source, item.kind] = item.id
            state.evidence = [
                item
                for item in state.evidence
                if item.timestamp >= cutoff
                or item.id
                == latest[item.work_id, item.artifact_id, item.source, item.kind]
            ]
            latest = {}
            for item in state.surfacings:
                latest[item.opportunity_id, item.version, item.outcome] = item.id
            state.surfacings = [
                item
                for item in state.surfacings
                if item.timestamp >= cutoff
                or item.id == latest[item.opportunity_id, item.version, item.outcome]
            ]
            self.repository.save(state)

    def clear(self) -> None:
        with self.repository.transaction():
            self.repository.clear()
