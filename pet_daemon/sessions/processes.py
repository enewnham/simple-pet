import asyncio
import os
import shlex
import subprocess
import time
from collections.abc import Callable, Iterable
from pathlib import Path

from ..domain import ProcessStatus
from ..runtime import ProcessIdentity

type ProcessRow = tuple[ProcessIdentity, int, str, str]


def _parse_process(line: str) -> ProcessRow | None:
    fields = line.strip().split(None, 8)
    if len(fields) != 9:
        return None
    try:
        pid, parent = int(fields[0]), int(fields[1])
    except ValueError:
        return None
    started = " ".join(fields[2:7])
    try:
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        started = f"proc:{stat[19]}"
    except (OSError, IndexError):
        pass
    return ProcessIdentity(pid, started), parent, fields[7], fields[8]


def _processes(pid: int | None = None) -> dict[int, ProcessRow]:
    args = (
        ["ps", "-ww", "-axo", "pid=,ppid=,lstart=,stat=,command="]
        if pid is None
        else ["ps", "-ww", "-p", str(pid), "-o", "pid=,ppid=,lstart=,stat=,command="]
    )
    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=2,
        check=False,
        env={**os.environ, "LC_ALL": "C"},
    )
    if result.returncode not in (0, 1) or (result.returncode and result.stderr.strip()):
        raise OSError("Process query failed")
    rows: dict[int, ProcessRow] = {}
    for line in result.stdout.splitlines():
        row = _parse_process(line)
        if row is None:
            raise OSError("Process query returned unreadable identities")
        rows[row[0].pid] = row
    return rows


def discover_owner(harness: str, pid: int | None = None) -> ProcessIdentity | None:
    try:
        rows = _processes()
    except (OSError, subprocess.SubprocessError):
        return None
    current = os.getppid() if pid is None else pid
    ancestors = []
    visited = set()
    while current in rows and current not in visited and len(ancestors) < 32:
        visited.add(current)
        row = rows[current]
        ancestors.append(row)
        current = row[1]
    explicit = os.environ.get("PET_OWNER_PID")
    started = os.environ.get("PET_OWNER_STARTED")
    if explicit and started:
        for identity, _, state, _ in ancestors:
            if (
                str(identity.pid) == explicit
                and identity.started == started
                and "Z" not in state
            ):
                return identity
    for identity, _, state, command in ancestors:
        if "Z" in state:
            continue
        try:
            argv = shlex.split(command)
        except ValueError:
            continue
        executable = Path(argv[0]).name if argv else ""
        if harness in ("claude", "codex") and executable == harness:
            if harness == "codex" and "app-server" in argv[1:]:
                continue
            return identity
    return None


def process_identity(pid: int) -> ProcessIdentity | None:
    try:
        row = _processes(pid).get(pid)
    except (OSError, subprocess.SubprocessError):
        return None
    return row[0] if row and "Z" not in row[2] else None


def probe(identity: ProcessIdentity) -> ProcessStatus:
    try:
        row = _processes(identity.pid).get(identity.pid)
    except (OSError, subprocess.SubprocessError):
        return ProcessStatus.UNKNOWN
    if row is None or "Z" in row[2]:
        return ProcessStatus.EXITED
    return ProcessStatus.ALIVE if row[0] == identity else ProcessStatus.REPLACED


class ProcessTracker:
    def __init__(
        self,
        clock: Callable[[], float] = time.monotonic,
        interval: float = 10,
        concurrency: int = 4,
    ) -> None:
        self.clock = clock
        self.interval = interval
        self.semaphore = asyncio.Semaphore(concurrency)
        self.checked: dict[ProcessIdentity, float] = {}

    async def refresh(
        self, identities: Iterable[ProcessIdentity]
    ) -> dict[ProcessIdentity, ProcessStatus]:
        identities = set(identities)
        self.checked = {
            identity: at
            for identity, at in self.checked.items()
            if identity in identities
        }
        due = [
            identity
            for identity in identities
            if self.clock() - self.checked.get(identity, float("-inf")) >= self.interval
        ]

        async def check(
            identity: ProcessIdentity,
        ) -> tuple[ProcessIdentity, ProcessStatus]:
            async with self.semaphore:
                status = await asyncio.to_thread(probe, identity)
            self.checked[identity] = self.clock()
            return identity, status

        return dict(await asyncio.gather(*(check(identity) for identity in due)))
