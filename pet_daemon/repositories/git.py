import asyncio
from dataclasses import dataclass


async def command(*args, cwd=None, timeout=10):
    process = await asyncio.create_subprocess_exec(*args, cwd=cwd, stdout=asyncio.subprocess.PIPE,
                                                 stderr=asyncio.subprocess.PIPE)
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)
    except BaseException:
        if process.returncode is None:
            process.kill()
        await process.wait()
        raise
    if process.returncode:
        raise RuntimeError(stderr.decode(errors="replace").strip()[:240])
    return stdout.decode(errors="replace")


@dataclass(frozen=True)
class GitObservation:
    worktree: str
    branch: str
    head: str
    staged: int
    unstaged: int
    untracked: int
    conflicts: int
    upstream: str
    ahead: int
    behind: int

    @property
    def pending(self):
        return bool(self.staged or self.unstaged or self.untracked or self.conflicts or self.ahead)


async def discover(path, run=command):
    return (await run("git", "rev-parse", "--show-toplevel", cwd=path)).strip()


async def observe(path, run=command):
    output = await run("git", "status", "--porcelain=v2", "--branch", "-z", "--untracked-files=normal", cwd=path)
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
    identity = await run("git", "rev-parse", "--verify", "HEAD", cwd=path) if head != "(initial)" else ""
    if head == "(initial)":
        current_branch = (await run("git", "symbolic-ref", "--quiet", "--short", "HEAD", cwd=path)).strip()
    else:
        current_branch = (await run("git", "rev-parse", "--abbrev-ref", "HEAD", cwd=path)).strip()
        if current_branch == "HEAD":
            current_branch = "(detached)"
    if (head != "(initial)" and identity.strip() != head) or current_branch != branch:
        raise RuntimeError("Repository changed during observation")
    return GitObservation(path, branch, head, staged, unstaged, untracked, conflicts, upstream, ahead, behind)
