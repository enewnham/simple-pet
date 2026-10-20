import asyncio
import os
from pathlib import Path
import shlex
import subprocess
import time

from .runtime import ProcessIdentity



def _parse_process(line):
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


def _processes(pid=None):
    args = ["ps", "-ww", "-axo", "pid=,ppid=,lstart=,stat=,command="] if pid is None else [
        "ps", "-ww", "-p", str(pid), "-o", "pid=,ppid=,lstart=,stat=,command="]
    result = subprocess.run(args, capture_output=True, text=True, timeout=2, env={**os.environ, "LC_ALL": "C"})
    if result.returncode not in (0, 1) or (result.returncode and result.stderr.strip()):
        raise OSError("Process query failed")
    rows = [_parse_process(line) for line in result.stdout.splitlines()]
    if any(row is None for row in rows):
        raise OSError("Process query returned unreadable identities")
    return {row[0].pid: row for row in rows}


def discover_owner(harness, pid=None):
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
            if str(identity.pid) == explicit and identity.started == started and "Z" not in state:
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


def process_identity(pid):
    try:
        row = _processes(pid).get(pid)
    except (OSError, subprocess.SubprocessError):
        return None
    return row[0] if row and "Z" not in row[2] else None


def probe(identity):
    try:
        row = _processes(identity.pid).get(identity.pid)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if row is None or "Z" in row[2]:
        return "exited"
    return "alive" if row[0] == identity else "replaced"


class ProcessTracker:
    def __init__(self, clock=time.monotonic, interval=10, concurrency=4):
        self.clock = clock
        self.interval = interval
        self.semaphore = asyncio.Semaphore(concurrency)
        self.checked = {}

    async def refresh(self, identities):
        identities = set(identities)
        self.checked = {identity: at for identity, at in self.checked.items() if identity in identities}
        due = [identity for identity in identities if self.clock() - self.checked.get(identity, float("-inf")) >= self.interval]

        async def check(identity):
            async with self.semaphore:
                status = await asyncio.to_thread(probe, identity)
            self.checked[identity] = self.clock()
            return identity, status

        return dict(await asyncio.gather(*(check(identity) for identity in due)))
