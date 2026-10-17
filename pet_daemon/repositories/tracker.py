import asyncio
from dataclasses import asdict, dataclass, replace
import time
from pathlib import Path

from .. import telemetry
from . import git, github


@dataclass(frozen=True)
class RepositoryObservation:
    subject: str
    worktree: str
    branch: str
    sessions: tuple = ()
    local: git.GitObservation | None = None
    pr: github.PullRequest | None = None
    current: bool = True
    revision: int = 0
    changed_at: float = 0
    local_checked_at: float = 0
    github_checked_at: float = 0
    github_attempted_at: float = 0
    local_error: str | None = None
    github_error: str | None = None

    @property
    def pending(self):
        return bool((self.current and self.local and self.local.pending) or (self.pr and self.pr.pending))


class RepositoryTracker:
    def __init__(self, memory=None, clock=time.time, local_interval=30, github_interval=180,
                 run=git.command, concurrency=2):
        self.memory = memory
        self.clock = clock
        self.local_interval = local_interval
        self.github_interval = github_interval
        self.run = run
        self.observations = {}
        self.associations = {}
        self.paths = {}
        self.tasks = {}
        self.limit = asyncio.Semaphore(concurrency)
        self.closed = False
        self.sequence = 0
        self.applied = {}
        self.epoch = 0
        self.roots = {}
        if memory:
            for item in memory.repository_associations():
                self.associations.setdefault(item["worktree"], set()).add(item["session"])
                self.paths[item["worktree"]] = 0
            for item in memory.load_repositories():
                item["sessions"] = tuple(item["sessions"])
                if item["local"]:
                    item["local"] = git.GitObservation(**item["local"])
                if item["pr"]:
                    item["pr"]["checks"] = tuple(tuple(check) for check in item["pr"]["checks"])
                    item["pr"] = github.PullRequest(**item["pr"])
                observation = RepositoryObservation(**item)
                self.observations[observation.subject] = observation

    def associate(self, session, path):
        if not path:
            return
        path = str(Path(path).resolve())
        self.associations.setdefault(path, set()).add(session)
        self.paths.setdefault(path, 0)

    def snapshot(self):
        return tuple(self.observations[key] for key in sorted(self.observations))

    def signature(self):
        return tuple((item.subject, item.revision, item.current, item.local_error, item.github_error) for item in self.snapshot())

    def _save(self, item):
        self.observations[item.subject] = item
        if self.memory:
            self.memory.save_repository(item.subject, asdict(item))

    def _launch(self, key, coroutine):
        self.tasks[key] = asyncio.create_task(coroutine)
        self.tasks[key].add_done_callback(lambda task: self.tasks.pop(key, None))

    def tick(self):
        if self.closed:
            return
        now = self.clock()
        for path, checked in list(self.paths.items()):
            if ("git", path) not in self.tasks and (not checked or now - checked >= self.local_interval):
                self.paths[path] = now
                self.sequence += 1
                self._launch(("git", path), self._local(path, self.sequence, self.epoch))
        for item in self.snapshot():
            key = ("github", item.subject)
            if key not in self.tasks and (item.current or (item.pr and item.pr.pending)) and (
                    not item.github_attempted_at or now - item.github_attempted_at >= self.github_interval):
                self._launch(key, self._github(item, self.epoch))

    async def _local(self, path, sequence, epoch):
        try:
            async with self.limit:
                root = await git.discover(path, self.run)
                self.roots[path] = root
                local = await git.observe(root, self.run)
            if epoch != self.epoch or sequence < self.applied.get(root, 0):
                return
            self.applied[root] = sequence
            now = self.clock()
            sessions = self.associations.setdefault(root, set())
            sessions.update(self.associations.get(path, ()))
            if self.memory:
                for session in sessions:
                    self.memory.associate_repository(session, root)
            subject = root + ":" + local.branch
            for old in self.snapshot():
                if old.worktree == root and old.subject != subject and old.current:
                    self._save(replace(old, current=False, revision=old.revision + 1))
            previous = self.observations.get(subject)
            changed = previous is None or previous.local != local or not previous.current
            item = replace(previous, local=local, current=True, sessions=tuple(sorted(sessions)), local_checked_at=now,
                           local_error=None, revision=previous.revision + int(changed),
                           changed_at=now if changed else previous.changed_at) if previous else RepositoryObservation(
                               subject, root, local.branch, tuple(sorted(sessions)), local=local,
                               revision=1, changed_at=now, local_checked_at=now)
            self._save(item)
            if changed:
                telemetry.event("repository.changed", subject=subject, revision=item.revision, pending=item.pending)
        except (OSError, RuntimeError, ValueError, asyncio.TimeoutError) as error:
            telemetry.event("repository.refresh_failed", source="git", path=path, error=str(error))
            for item in self.snapshot():
                if item.worktree == self.roots.get(path, path) and item.current:
                    self._save(replace(item, local_error=str(error)[:240]))

    async def _github(self, captured, epoch):
        try:
            async with self.limit:
                pr = await github.observe(captured.worktree, captured.branch,
                                          captured.pr.number if captured.pr and (captured.pr.pending or not captured.current) else None, self.run)
                local = await git.observe(captured.worktree, self.run) if captured.current else None
            current = self.observations.get(captured.subject)
            if epoch != self.epoch or current is None or current.revision != captured.revision or current.current != captured.current:
                return
            if local is not None and local != captured.local:
                self.paths[captured.worktree] = 0
                return
            changed = current.pr != pr
            if current.pr and pr and current.pr.number != pr.number:
                self._save(replace(current, subject=current.subject + f":pr:{current.pr.number}", current=False,
                                   local=None, revision=current.revision + 1))
            self._save(replace(current, pr=pr, github_checked_at=self.clock(), github_attempted_at=self.clock(), github_error=None,
                               revision=current.revision + int(changed),
                               changed_at=self.clock() if changed else current.changed_at))
            if changed:
                telemetry.event("repository.changed", subject=current.subject, source="github", state=pr.state if pr else None)
        except (OSError, RuntimeError, ValueError, KeyError, TypeError, asyncio.TimeoutError) as error:
            current = self.observations.get(captured.subject)
            if epoch == self.epoch and current and current.revision == captured.revision:
                self._save(replace(current, github_attempted_at=self.clock(), github_error=str(error)[:240]))
            telemetry.event("repository.refresh_failed", source="github", subject=captured.subject, error=str(error))

    async def close(self):
        self.closed = True
        tasks = list(self.tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def clear(self):
        self.epoch += 1
        for task in self.tasks.values():
            task.cancel()
        self.observations.clear()
        self.associations.clear()
        self.paths.clear()
        self.applied.clear()
        self.roots.clear()
