import asyncio
import time
from collections.abc import Callable, Coroutine, Iterable
from pathlib import Path

from ..domain import CollectionKind
from ..logging import get_logger
from ..repository_identity import RepositoryIdentity, repository_columns
from ..work.observations import (
    BranchObservation,
    CheckoutObservation,
    GitObservation,
    PullRequestObservation,
    RepositoryObservation,
    RepositoryTarget,
)
from . import git, github
from .git import RunCommand

log = get_logger("pet-daemon")

type CollectionKey = (
    tuple[CollectionKind, str] | tuple[CollectionKind, RepositoryIdentity, str]
)


def _key(item: RepositoryObservation) -> CollectionKey:
    if isinstance(item, CheckoutObservation):
        return CollectionKind.CHECKOUT, item.worktree
    if isinstance(item, BranchObservation):
        return CollectionKind.BRANCH, item.repository, item.branch
    return CollectionKind.PULL_REQUEST, item.ref


def _sort_key(key: CollectionKey) -> tuple[str, str, str, str]:
    if len(key) == 2:
        return key[0], key[1], "", ""
    kind, value = repository_columns(key[1])
    return key[0], kind, value, key[2]


class RepositoryTracker:
    def __init__(
        self,
        emit: Callable[[RepositoryObservation], None],
        clock: Callable[[], float] = time.time,
        local_interval: float = 30,
        github_interval: float = 180,
        run: RunCommand = git.command,
        concurrency: int = 2,
        *,
        checkpoints: Iterable[RepositoryObservation] = (),
        targets: Iterable[RepositoryTarget] = (),
    ) -> None:
        self.emit = emit
        self.clock = clock
        self.local_interval = local_interval
        self.github_interval = github_interval
        self.run = run
        self.observations = {_key(item): item for item in checkpoints}
        self.locals: dict[str, GitObservation] = {
            item.worktree: item.local
            for item in self.observations.values()
            if isinstance(item, CheckoutObservation) and item.local is not None
        }
        self.paths: dict[str, float] = {}
        self.tasks: dict[CollectionKey, asyncio.Task[None]] = {}
        self.limit = asyncio.Semaphore(concurrency)
        self.closed = False
        self.paused = False
        self.epoch = 0
        self.sequence = 0
        self.applied: dict[str, int] = {}
        self.versions: dict[CollectionKey, int] = {}
        self.roots: dict[str, str] = {}
        targets = tuple(targets)
        for target in targets:
            if target.kind != CollectionKind.CHECKOUT:
                continue
            path = str(Path(target.ref).resolve())
            enclosing = [
                root
                for root in self.locals
                if path == root or path.startswith(root.rstrip("/") + "/")
            ]
            if enclosing:
                self.roots[path] = max(enclosing, key=len)
        self.missing: set[str] = {
            item.worktree
            for item in self.observations.values()
            if isinstance(item, CheckoutObservation) and not item.available
        }
        self.targets: dict[CollectionKey, RepositoryTarget] = {}
        self.subscriptions: dict[CollectionKey, RepositoryTarget] = {}
        self.attempted: dict[CollectionKey, float] = {}
        self.reconcile_targets(targets)

    def track_checkout(self, path: str | Path, *, refresh: bool = False) -> None:
        if not path:
            return
        supplied = str(Path(path).resolve())
        root = self.roots.get(supplied, supplied)
        self.missing.discard(root)
        self.missing.discard(supplied)
        if refresh or root not in self.paths:
            self.paths[root] = 0
            self._invalidate((CollectionKind.CHECKOUT, root))

    def reconcile_targets(self, targets: Iterable[RepositoryTarget]) -> None:
        desired: dict[CollectionKey, RepositoryTarget] = {}
        paths = set()
        for target in targets:
            if target.kind == CollectionKind.CHECKOUT:
                supplied = str(Path(target.ref).resolve())
                root = self.roots.get(supplied, supplied)
                if root not in self.missing:
                    paths.add(root)
                desired[(CollectionKind.CHECKOUT, root)] = target
            elif target.kind == CollectionKind.BRANCH and target.ref:
                repository = target.repository
                if repository:
                    desired[(CollectionKind.BRANCH, repository, target.ref)] = (
                        RepositoryTarget(
                            kind=CollectionKind.BRANCH,
                            ref=target.ref,
                            repository=repository,
                            include_links=target.include_links,
                        )
                    )
            elif target.kind == CollectionKind.PULL_REQUEST:
                _, number = github.pull_request_identity(target.ref)
                if number is not None:
                    desired[(CollectionKind.PULL_REQUEST, target.ref)] = target
        for path in self.paths.keys() - paths:
            self._invalidate((CollectionKind.CHECKOUT, path))
        self.paths = {path: self.paths.get(path, 0) for path in paths}
        self.targets = desired
        self._sync_subscriptions()

    def _sync_subscriptions(self) -> None:
        desired = {
            key: target
            for key, target in self.targets.items()
            if key[0] != CollectionKind.CHECKOUT
        }
        for path in self.paths:
            local = self.locals.get(path)
            if not local or not local.branch:
                continue
            key = CollectionKind.BRANCH, local.repository, local.branch
            desired.setdefault(
                key,
                RepositoryTarget(
                    kind=CollectionKind.BRANCH,
                    ref=local.branch,
                    repository=local.repository,
                ),
            )
        for key, target in self.subscriptions.items():
            if desired.get(key) != target:
                self._invalidate(key)
                self.attempted.pop(key, None)
        for key in desired.keys() - self.subscriptions.keys():
            self.attempted.pop(key, None)
        self.subscriptions = desired

    def snapshot(self) -> tuple[RepositoryObservation, ...]:
        return tuple(
            self.observations[key] for key in sorted(self.observations, key=_sort_key)
        )

    def signature(self) -> tuple[RepositoryObservation, ...]:
        return self.snapshot()

    def _save(self, item: RepositoryObservation) -> None:
        self.observations[_key(item)] = item
        self.emit(item)

    def _invalidate(self, key: CollectionKey) -> None:
        self.versions[key] = self.versions.get(key, 0) + 1
        task = self.tasks.get(key)
        if task:
            task.cancel()

    def _launch(
        self, key: CollectionKey, coroutine: Coroutine[object, object, None]
    ) -> None:
        self.tasks[key] = asyncio.create_task(coroutine)

        def completed(task: asyncio.Task[None]) -> None:
            if self.tasks.get(key) is task:
                self.tasks.pop(key, None)

        self.tasks[key].add_done_callback(completed)

    def _retire_missing(self, path: str) -> bool:
        try:
            Path(path).stat()
        except FileNotFoundError, NotADirectoryError:
            self.paths.pop(path, None)
            self.missing.add(path)
            self._save(
                CheckoutObservation(
                    worktree=path, local=None, checked_at=self.clock(), available=False
                )
            )
            self._sync_subscriptions()
            log.info("repository.retired", path=path)
            return True
        except OSError:
            pass
        return False

    def tick(self) -> None:
        if self.closed or self.paused:
            return
        now = self.clock()
        for path, checked in list(self.paths.items()):
            key = CollectionKind.CHECKOUT, path
            if key not in self.tasks and (
                not checked or now - checked >= self.local_interval
            ):
                if self._retire_missing(path):
                    continue
                self.paths[path] = now
                self.sequence += 1
                self._launch(
                    key,
                    self._local(
                        path, self.sequence, self.versions.get(key, 0), self.epoch
                    ),
                )
        for key, target in list(self.subscriptions.items()):
            attempted = self.attempted.get(key)
            if key not in self.tasks and (
                attempted is None or now - attempted >= self.github_interval
            ):
                self.attempted[key] = now
                self._launch(
                    key,
                    self._github(key, target, self.versions.get(key, 0), self.epoch),
                )

    def _is_active(self, key: CollectionKey, version: int, epoch: int) -> bool:
        return (
            epoch == self.epoch
            and version == self.versions.get(key, 0)
            and (
                key[1] in self.paths
                if key[0] == CollectionKind.CHECKOUT
                else key in self.subscriptions
            )
        )

    async def _local(self, path: str, sequence: int, version: int, epoch: int) -> None:
        key = CollectionKind.CHECKOUT, path
        checked_at = self.clock()
        try:
            async with self.limit:
                root = await git.checkout_root(path, self.run)
                local = await git.observe(root, self.run)
            if not self._is_active(key, version, epoch) or sequence < self.applied.get(
                root, 0
            ):
                return
            if self._retire_missing(root):
                self.paths.pop(path, None)
                return
            self.roots[path] = root
            self.applied[root] = sequence
            self.paths.pop(path, None)
            self.paths[root] = checked_at
            self.locals[root] = local
            self._save(
                CheckoutObservation(worktree=root, local=local, checked_at=checked_at)
            )
            self._sync_subscriptions()
        except (TimeoutError, OSError, RuntimeError, ValueError) as error:
            if not self._is_active(key, version, epoch):
                return
            self._save(
                CheckoutObservation(
                    worktree=path,
                    local=None,
                    checked_at=checked_at,
                    error=str(error)[:240],
                )
            )
            log.warning(
                "repository.refresh_failed",
                source="git",
                path=path,
                error=str(error),
                exc_info=True,
            )

    async def _github(
        self, key: CollectionKey, target: RepositoryTarget, version: int, epoch: int
    ) -> None:
        checked_at = self.clock()
        error = None
        pr = None
        try:
            async with self.limit:
                pr = await github.observe(
                    None,
                    target.ref if target.kind == CollectionKind.BRANCH else "",
                    run=self.run,
                    repository=target.repository,
                    url=target.ref
                    if target.kind == CollectionKind.PULL_REQUEST
                    else "",
                    include_links=target.include_links,
                )
        except (
            TimeoutError,
            OSError,
            RuntimeError,
            ValueError,
            KeyError,
            TypeError,
        ) as failure:
            error = str(failure)[:240]
            log.warning(
                "repository.refresh_failed",
                source="github",
                subject=key[1],
                error=error,
                exc_info=True,
            )
        if not self._is_active(key, version, epoch):
            return
        if target.kind == CollectionKind.BRANCH:
            assert target.repository is not None
            item: RepositoryObservation = BranchObservation(
                repository=target.repository,
                branch=target.ref,
                pr=pr,
                checked_at=checked_at,
                error=error,
            )
        else:
            item = PullRequestObservation(
                ref=target.ref, pr=pr, checked_at=checked_at, error=error
            )
        self._save(item)

    def pause(self) -> None:
        self.paused = True
        self.epoch += 1
        for task in self.tasks.values():
            task.cancel()

    def resume(self) -> None:
        if self.closed:
            return
        self.paused = False
        self.paths = dict.fromkeys(self.paths, 0)
        self.attempted.clear()

    async def close(self) -> None:
        self.closed = True
        self.pause()
        await asyncio.gather(*list(self.tasks.values()), return_exceptions=True)

    def clear(self) -> None:
        self.epoch += 1
        for task in self.tasks.values():
            task.cancel()
        self.observations.clear()
        self.locals.clear()
        self.paths.clear()
        self.applied.clear()
        self.roots.clear()
        self.missing.clear()
        self.targets.clear()
        self.subscriptions.clear()
        self.attempted.clear()
