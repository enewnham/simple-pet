import asyncio
from collections.abc import Coroutine
from pathlib import Path
from typing import Protocol, TypedDict

from ..repository_identity import (
    LocalRepository,
    RemoteRepository,
    RepositoryIdentity,
    parse_remote_repository,
)
from ..work.observations import GitObservation

type CommandPath = str | Path | None


class CommandOptions(TypedDict, total=False):
    cwd: CommandPath
    timeout: float


class RunCommand(Protocol):
    def __call__(
        self, *args: str, cwd: CommandPath = None, timeout: float = 10
    ) -> Coroutine[object, object, str]: ...


async def _finish[T](task: asyncio.Task[T]) -> T:
    while True:
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.cancelled():
                raise


async def command(*args: str, cwd: CommandPath = None, timeout: float = 10) -> str:
    startup = asyncio.create_task(
        asyncio.create_subprocess_exec(
            *args,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    )
    try:
        process = await asyncio.shield(startup)
    except asyncio.CancelledError:
        try:
            process = await _finish(startup)
        except Exception:  # noqa: BLE001, S110 - Preserve cancellation when subprocess startup fails during cleanup.
            pass
        else:
            if process.returncode is None:
                process.kill()
            await _finish(asyncio.create_task(process.wait()))
        raise
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)
    except BaseException:
        if process.returncode is None:
            process.kill()
        await _finish(asyncio.create_task(process.wait()))
        raise
    if process.returncode:
        raise RuntimeError(stderr.decode(errors="replace").strip()[:240])
    return stdout.decode(errors="replace")


def canonical_repository(value: str) -> RemoteRepository | None:
    return parse_remote_repository(value)


async def repository_identity(
    path: CommandPath, run: RunCommand = command
) -> RepositoryIdentity:
    remotes = (await run("git", "remote", cwd=path)).splitlines()
    if remotes:
        remote = "origin" if "origin" in remotes else min(remotes)
        url = (await run("git", "remote", "get-url", remote, cwd=path)).strip()
        repository = canonical_repository(url)
        if repository:
            return repository
    common = (
        await run(
            "git", "rev-parse", "--path-format=absolute", "--git-common-dir", cwd=path
        )
    ).strip()
    return LocalRepository(path=str(Path(common).resolve()))


async def discover(path: CommandPath, run: RunCommand = command) -> str:
    return (await run("git", "rev-parse", "--show-toplevel", cwd=path)).strip()


async def observe(path: str | Path, run: RunCommand = command) -> GitObservation:
    output = await run(
        "git",
        "status",
        "--porcelain=v2",
        "--branch",
        "-z",
        "--untracked-files=normal",
        cwd=path,
    )
    head = branch = upstream = ""
    staged = unstaged = untracked = conflicts = ahead = behind = 0
    records = iter(output.split("\0"))
    for record in records:
        if record.startswith("# branch.oid "):
            head = record[13:]
        elif record.startswith("# branch.head "):
            branch = record[14:]
        elif record.startswith("# branch.upstream "):
            upstream = record[18:]
        elif record.startswith("# branch.ab "):
            plus, minus = record[12:].split()
            ahead, behind = int(plus), abs(int(minus))
        elif record.startswith("? "):
            untracked += 1
        elif record.startswith("u "):
            conflicts += 1
        elif record.startswith(("1 ", "2 ")):
            staged += record[2] != "."
            unstaged += record[3] != "."
            if record.startswith("2 "):
                next(records, None)
    identity = (
        await run("git", "rev-parse", "--verify", "HEAD", cwd=path)
        if head != "(initial)"
        else ""
    )
    if head == "(initial)":
        current_branch = (
            await run("git", "symbolic-ref", "--quiet", "--short", "HEAD", cwd=path)
        ).strip()
    else:
        current_branch = (
            await run("git", "rev-parse", "--abbrev-ref", "HEAD", cwd=path)
        ).strip()
        if current_branch == "HEAD":
            current_branch = "(detached)"
    if (head != "(initial)" and identity.strip() != head) or current_branch != branch:
        raise RuntimeError("Repository changed during observation")
    repository = await repository_identity(path, run)
    return GitObservation(
        worktree=str(path),
        branch=branch,
        head=head,
        staged=staged,
        unstaged=unstaged,
        untracked=untracked,
        conflicts=conflicts,
        upstream=upstream,
        ahead=ahead,
        behind=behind,
        repository=repository,
    )
