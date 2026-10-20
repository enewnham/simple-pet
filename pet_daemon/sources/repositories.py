import asyncio
import time
from collections.abc import Callable, Coroutine, Iterable
from dataclasses import replace
from pathlib import Path

from .. import telemetry
from ..work.observations import RepositoryObservation, RepositoryTarget
from . import git, github
from .git import RunCommand


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
        associations: Iterable[tuple[str, str]] = (),
        targets: Iterable[RepositoryTarget] = (),
    ) -> None:
        self.emit = emit
        self.clock = clock
        self.local_interval = local_interval
        self.github_interval = github_interval
        self.run = run
        self.observations = {item.subject: item for item in checkpoints}
        self.associations: dict[str, set[str]] = {}
        self.paths: dict[str, float] = {}
        self.tasks: dict[tuple[str, str], asyncio.Task[None]] = {}
        self.limit = asyncio.Semaphore(concurrency)
        self.closed = False
        self.paused = False
        self.sequence = 0
        self.applied = {}
        self.epoch = 0
        self.roots = {}
        self.targets: dict[str, RepositoryTarget] = {}
        self.target_paths = set()
        self.subscriptions = set()
        for session, path in associations:
            self.associate(session, path)
        for target in targets:
            self.track_artifact(target)

    def associate(
        self, session: str, path: str | Path, *, refresh: bool = False
    ) -> None:
        if not path:
            return
        path = str(Path(path).resolve())
        sessions = self.associations.setdefault(path, set())
        if refresh or session not in sessions or path not in self.paths:
            self.paths[path] = 0
        sessions.add(session)

    def track_artifact(self, target: RepositoryTarget) -> None:
        repository = git.canonical_repository(target.repository)
        if target.kind == "pull_request":
            repository, number = github.pull_request_identity(target.ref)
            if number is None:
                return
            subject, branch = "pr:" + target.ref, ""
        elif target.kind == "branch" and target.ref:
            if target.checkout:
                checkout = str(Path(target.checkout).resolve())
                self.target_paths.add(checkout)
                self.paths.setdefault(checkout, 0)
            if not repository:
                return
            subject, branch = f"branch:{repository}:{target.ref}", target.ref
        else:
            return
        self.targets[subject] = target
        self.subscriptions.add(subject)
        if subject not in self.observations:
            self.observations[subject] = RepositoryObservation(
                subject, "", branch, current=False, repository=repository
            )

    def reconcile_targets(self, targets: Iterable[RepositoryTarget]) -> None:
        previous_paths = self.target_paths
        self.targets.clear()
        self.target_paths = set()
        self.subscriptions.clear()
        for target in targets:
            self.track_artifact(target)
        for path in previous_paths - self.target_paths:
            root = self.roots.get(path, path)
            retained_paths = self.target_paths | {
                name for name, sessions in self.associations.items() if sessions
            }
            if any(self.roots.get(name, name) == root for name in retained_paths):
                continue
            for retired in {
                root,
                path,
                *(name for name, value in self.roots.items() if value == root),
            }:
                self.paths.pop(retired, None)
                self.associations.pop(retired, None)
                self.roots.pop(retired, None)
            for item in self.snapshot():
                if item.worktree == root and item.current:
                    self._save(replace(item, current=False, revision=item.revision + 1))

    def snapshot(self) -> tuple[RepositoryObservation, ...]:
        return tuple(self.observations[key] for key in sorted(self.observations))

    def signature(self) -> tuple[tuple[str, int, bool, str | None, str | None], ...]:
        return tuple(
            (
                item.subject,
                item.revision,
                item.current,
                item.local_error,
                item.github_error,
            )
            for item in self.snapshot()
        )

    def _save(self, item: RepositoryObservation) -> None:
        self.observations[item.subject] = item
        self.emit(item)

    def _launch(
        self, key: tuple[str, str], coroutine: Coroutine[object, object, None]
    ) -> None:
        self.tasks[key] = asyncio.create_task(coroutine)

        def completed(task: asyncio.Task[None]) -> None:
            if self.tasks.get(key) is task:
                self.tasks.pop(key, None)

        self.tasks[key].add_done_callback(completed)

    def _retire_missing(self, path: str) -> bool:
        try:
            Path(path).stat()
        except (FileNotFoundError, NotADirectoryError):
            paths = {path, *(key for key, root in self.roots.items() if root == path)}
            for retired in paths:
                self.paths.pop(retired, None)
                self.associations.pop(retired, None)
                self.roots.pop(retired, None)
            self.applied.pop(path, None)
            for item in self.snapshot():
                if item.worktree == path:
                    self._save(
                        replace(
                            item,
                            current=False,
                            revision=item.revision + int(item.current),
                        )
                    )
            telemetry.event("repository.retired", path=path)
            return True
        except OSError:
            pass
        return False

    def tick(self) -> None:
        if self.closed or self.paused:
            return
        now = self.clock()
        for path, checked in list(self.paths.items()):
            if ("git", path) not in self.tasks and (
                not checked or now - checked >= self.local_interval
            ):
                if self._retire_missing(path):
                    continue
                self.paths[path] = now
                self.sequence += 1
                self._launch(
                    ("git", path), self._local(path, self.sequence, self.epoch)
                )
        for item in self.snapshot():
            key = ("github", item.subject)
            if (
                key not in self.tasks
                and (item.current or item.subject in self.subscriptions)
                and (
                    not item.github_attempted_at
                    or now - item.github_attempted_at >= self.github_interval
                )
            ):
                self._launch(key, self._github(item, self.epoch))

    async def _local(self, path: str, sequence: int, epoch: int) -> None:
        try:
            async with self.limit:
                checked_at = self.clock()
                root = await git.discover(path, self.run)
                local = await git.observe(root, self.run)
            if (
                epoch != self.epoch
                or path not in self.paths
                or sequence < self.applied.get(root, 0)
            ):
                return
            if self._retire_missing(root):
                return
            self.roots[path] = root
            self.applied[root] = sequence
            now = self.clock()
            self.paths.setdefault(root, now)
            sessions = self.associations.setdefault(root, set())
            sessions.update(self.associations.get(path, ()))
            subject = root + ":" + local.branch
            for old in self.snapshot():
                if old.worktree == root and old.subject != subject and old.current:
                    self._save(replace(old, current=False, revision=old.revision + 1))
            previous = self.observations.get(subject)
            changed = (
                previous is None or previous.local != local or not previous.current
            )
            item = (
                replace(
                    previous,
                    local=local,
                    current=True,
                    sessions=tuple(sorted(sessions)),
                    local_checked_at=checked_at,
                    local_error=None,
                    revision=previous.revision + int(changed),
                    changed_at=now if changed else previous.changed_at,
                    repository=local.repository,
                )
                if previous
                else RepositoryObservation(
                    subject,
                    root,
                    local.branch,
                    tuple(sorted(sessions)),
                    local=local,
                    revision=1,
                    changed_at=now,
                    local_checked_at=checked_at,
                    repository=local.repository,
                )
            )
            self._save(item)
            if changed:
                telemetry.event(
                    "repository.changed",
                    subject=subject,
                    revision=item.revision,
                    pending=item.pending,
                )
        except (TimeoutError, OSError, RuntimeError, ValueError) as error:
            if epoch != self.epoch:
                return
            telemetry.event(
                "repository.refresh_failed", source="git", path=path, error=str(error)
            )
            for item in self.snapshot():
                if item.worktree == self.roots.get(path, path) and item.current:
                    self._save(replace(item, local_error=str(error)[:240]))

    async def _github(self, captured: RepositoryObservation, epoch: int) -> None:
        try:
            async with self.limit:
                checked_at = self.clock()
                explicit_url = (
                    captured.subject.removeprefix("pr:")
                    if captured.subject.startswith("pr:")
                    else ""
                )
                follows_branch = captured.current or (
                    captured.subject in self.subscriptions
                    and captured.subject.startswith("branch:")
                )
                url = (
                    captured.pr.url
                    if captured.pr and not follows_branch
                    else explicit_url
                )
                target = self.targets.get(captured.subject)
                include_links = target.include_links if target else True
                pr = await github.observe(
                    None,
                    captured.branch,
                    run=self.run,
                    repository=captured.repository,
                    url=url,
                    include_links=include_links,
                )
                local = (
                    await git.observe(captured.worktree, self.run)
                    if captured.current
                    else None
                )
            current = self.observations.get(captured.subject)
            if (
                epoch != self.epoch
                or current is None
                or current.revision != captured.revision
                or current.current != captured.current
            ):
                return
            if not current.current and current.subject not in self.subscriptions:
                return
            if local is not None and local != captured.local:
                self.paths[captured.worktree] = 0
                return
            if pr is None and current.pr is not None:
                pr = current.pr
            if pr and current.pr and current.pr.url == pr.url:
                if not include_links:
                    pr = replace(
                        pr,
                        related_links=current.pr.related_links,
                        related_links_error=current.pr.related_links_error,
                    )
                elif pr.related_links_error:
                    pr = replace(pr, related_links=current.pr.related_links)
            changed = (
                replace(current.pr, related_links_error=None)
                != replace(pr, related_links_error=None)
                if current.pr and pr
                else current.pr != pr
            )
            if current.pr and pr and current.pr.url != pr.url:
                self._save(
                    replace(
                        current,
                        subject=current.subject + f":pr:{current.pr.url}",
                        current=False,
                        local=None,
                        revision=current.revision + 1,
                    )
                )
            self._save(
                replace(
                    current,
                    pr=pr,
                    github_checked_at=checked_at,
                    github_attempted_at=self.clock(),
                    github_error=None,
                    revision=current.revision + int(changed),
                    changed_at=self.clock() if changed else current.changed_at,
                )
            )
            if changed:
                telemetry.event(
                    "repository.changed",
                    subject=current.subject,
                    source="github",
                    state=pr.state if pr else None,
                )
        except (
            TimeoutError,
            OSError,
            RuntimeError,
            ValueError,
            KeyError,
            TypeError,
        ) as error:
            if epoch != self.epoch:
                return
            current = self.observations.get(captured.subject)
            if (
                epoch == self.epoch
                and current
                and current.revision == captured.revision
            ):
                self._save(
                    replace(
                        current,
                        github_attempted_at=self.clock(),
                        github_error=str(error)[:240],
                    )
                )
            telemetry.event(
                "repository.refresh_failed",
                source="github",
                subject=captured.subject,
                error=str(error),
            )

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
        self.observations = {
            key: replace(item, github_attempted_at=0)
            for key, item in self.observations.items()
        }

    async def close(self) -> None:
        self.closed = True
        self.pause()
        await asyncio.gather(*list(self.tasks.values()), return_exceptions=True)

    def clear(self) -> None:
        self.epoch += 1
        for task in self.tasks.values():
            task.cancel()
        self.observations.clear()
        self.associations.clear()
        self.paths.clear()
        self.applied.clear()
        self.roots.clear()
        self.targets.clear()
        self.target_paths.clear()
        self.subscriptions.clear()
