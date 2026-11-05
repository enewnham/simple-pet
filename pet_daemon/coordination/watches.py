from collections.abc import Awaitable, Callable
from dataclasses import replace

from ..sources.checks import ChecksObservation, ChecksStatus
from ..storage.coordination import SQLiteCoordinationRepository
from ..storage.transactions import transaction
from .config import ScheduleConfig
from .models import (
    ActivityRecord,
    Notice,
    ResponsibilityKind,
    ResponsibilityStatus,
    SourceReference,
)


class Watches:
    def __init__(
        self,
        store: SQLiteCoordinationRepository,
        config: ScheduleConfig,
        clock: Callable[[], float],
        observe: Callable[[str], Awaitable[ChecksObservation]],
    ) -> None:
        self.store = store
        self.config = config
        self.clock = clock
        self.observe = observe

    async def refresh(self) -> None:
        for watch in self.store.watches():
            if watch.completed_at is not None or watch.next_check_at > self.clock():
                continue
            responsibility = self.store.responsibility(watch.responsibility_id)
            if (
                responsibility is None
                or responsibility.status != ResponsibilityStatus.ACTIVE
            ):
                continue
            observation = await self.observe(watch.ref)
            if (
                self.store.state().paused
                or self.store.responsibility(responsibility.id) != responsibility
            ):
                continue
            current = next(
                (item for item in self.store.watches() if item.id == watch.id), None
            )
            if current != watch:
                continue
            now = self.clock()
            fresh = (
                observation.checked_at >= watch.created_at
                and 0
                <= now - observation.checked_at
                <= self.config.source_fresh_seconds
            )
            completed = (
                fresh
                and observation.ref == watch.ref
                and observation.status == ChecksStatus.PASSED
                and bool(observation.head)
                and bool(observation.revision)
                and not observation.error
                and responsibility.kind == ResponsibilityKind.FINITE
                and responsibility.completion_condition == "pull_request_checks_pass"
            )
            updated = replace(
                watch,
                revision=watch.revision + 1,
                next_check_at=now + self.config.watch_seconds,
                checked_at=observation.checked_at,
                head=observation.head,
                source_revision=observation.revision,
                error=observation.error,
                completed_at=now if completed else None,
            )
            with transaction(self.store.db):
                self.store.save_watch(updated)
                if completed:
                    source = SourceReference(
                        watch.ref,
                        "github",
                        observation.revision,
                        observation.checked_at,
                        uri=watch.ref,
                        scope=responsibility.scope,
                    )
                    self.store.save_responsibility(
                        replace(
                            responsibility,
                            revision=responsibility.revision + 1,
                            status=ResponsibilityStatus.COMPLETED,
                            sources=(source,),
                        )
                    )
                    self.store.record_activity(
                        ActivityRecord(
                            f"watch:{watch.id}:completion",
                            "watch_completed",
                            "The pull request checks passed.",
                            now,
                            entity_id=responsibility.id,
                            detail=f"Observed passing checks on {observation.head}.",
                            sources=(source,),
                        )
                    )
                    self.store.save_notice(
                        Notice(
                            f"watch:{watch.id}:report",
                            f"watch:{watch.id}:completion",
                            f"The pull request checks passed on {observation.head[:12]}.",
                            now,
                            sources=(source,),
                        )
                    )
                elif observation.error is not None:
                    self.store.record_activity(
                        ActivityRecord(
                            f"watch:{watch.id}:failure:{observation.checked_at}",
                            "source_unavailable",
                            "The pull request check refresh failed.",
                            now,
                            entity_id=responsibility.id,
                            detail=observation.error,
                        )
                    )
