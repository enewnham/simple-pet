import time
from typing import ContextManager, Protocol

from .association import collection_targets, observe_repository, record_intent, record_session_event, work_context
from .models import WorkState
from .observations import RepositoryObservation
from .opportunities import derive_opportunities, eligible, project
from .views import build_work, project_work


class WorkRepository(Protocol):
    def transaction(self) -> ContextManager: ...
    def load(self) -> WorkState: ...
    def save(self, state: WorkState) -> None: ...
    def clear(self) -> None: ...
    def clear_legacy(self) -> None: ...


class WorkService:
    def __init__(self, repository: WorkRepository, clock=time.time, retention_days=30):
        self.repository = repository
        self.clock = clock
        self.retention_days = retention_days

    def _change(self, apply):
        with self.repository.transaction():
            state = self.repository.load()
            result = apply(state)
            derive_opportunities(state, self.clock(), lambda work_id: work_context(state, work_id))
            self.repository.save(state)
        return result

    def session_intent(self, session, intent, aliases=()):
        return self._change(lambda state: record_intent(state, session, intent, aliases, self.clock()))

    def session_event(self, session, event):
        self._change(lambda state: record_session_event(state, session, event, self.clock()))

    def observe_repository(self, observation: RepositoryObservation):
        if not isinstance(observation, RepositoryObservation):
            raise ValueError("repository observation must be a RepositoryObservation")
        self._change(lambda state: observe_repository(state, observation, self.clock()))

    def session_context(self, session):
        state = self.repository.load()
        work = state.current(session)
        return work_context(state, work.id) if work else {}

    def work_context(self, work_id):
        return work_context(self.repository.load(), work_id)

    def snapshot(self) -> WorkState:
        return self.repository.load()

    def views(self, runtime_snapshot, now=0):
        return build_work(runtime_snapshot, self.repository.load(), now)

    def session_view(self, session):
        state = self.repository.load()
        work = state.current(session)
        return self._view(state, work.id) if work else None

    def work_view(self, work_id):
        return self._view(self.repository.load(), work_id)

    def _view(self, state, work_id):
        actions = tuple(project(item) for item in state.opportunities.values() if item.work_id == work_id and eligible(item))
        return project_work(work_id, work_context(state, work_id), actions)

    def work_for_session(self, session):
        return self.repository.load().current(session)

    def work_items(self):
        return tuple(self.repository.load().items.values())

    def session_names(self, work_id):
        return tuple(sorted(session for identifier, session in self.repository.load().memberships if identifier == work_id))

    def collection_targets(self):
        return collection_targets(self.repository.load())

    def tracked_artifacts(self):
        return tuple(item.data for item in sorted(self.repository.load().artifacts.values(), key=lambda item: (item.data["kind"], item.data["ref"])))

    def opportunities(self):
        records = self.repository.load().opportunities.values()
        return tuple(project(item) for item in sorted(records, key=lambda item: (-item.changed_at, item.id)) if eligible(item))

    def opportunity(self, identifier):
        item = self.repository.load().opportunities.get(identifier)
        return project(item) if item and eligible(item) else None

    def is_current(self, identifier, version):
        item = self.opportunity(identifier)
        return bool(item and item.version == version)

    def prune(self):
        cutoff = self.clock() - self.retention_days * 86400
        with self.repository.transaction():
            state = self.repository.load()
            latest = {}
            for item in state.evidence:
                latest[item.work_id, item.artifact_id, item.source, item.kind] = item.id
            state.evidence = [item for item in state.evidence if item.timestamp >= cutoff or item.id == latest[item.work_id, item.artifact_id, item.source, item.kind]]
            latest = {}
            for item in state.surfacings:
                latest[item.opportunity_id, item.version, item.outcome] = item.id
            state.surfacings = [item for item in state.surfacings if item.timestamp >= cutoff or item.id == latest[item.opportunity_id, item.version, item.outcome]]
            self.repository.save(state)

    def clear(self):
        with self.repository.transaction():
            self.repository.clear()
            self.repository.clear_legacy()
