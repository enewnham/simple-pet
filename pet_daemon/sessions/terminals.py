"""Find the zmx session whose Kitty window shows a Codex thread."""

import glob
import json
import re
import shutil
import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path

KITTEN = "/Applications/kitty.app/Contents/MacOS/kitten"
# Codex titles show the thread ID, often cut to a prefix, as `tui.terminal_title` selects.
THREAD_ID = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}[0-9a-f-]*")

type KittyWindow = Mapping[str, object]


def window_zmx_session(window: KittyWindow) -> str | None:
    processes = window.get("foreground_processes")
    if not isinstance(processes, list):
        return None
    for process in reversed(processes):
        cmdline = process.get("cmdline") if isinstance(process, dict) else None
        if (
            isinstance(cmdline, list)
            and len(cmdline) >= 3
            and Path(str(cmdline[0])).name == "zmx"
            and cmdline[1] in ("attach", "a")
        ):
            arguments = iter(str(value) for value in cmdline[2:])
            for argument in arguments:
                if argument == "--labels":
                    next(arguments, None)
                elif not argument.startswith("-"):
                    return argument
    return None


def shows_thread(title: str, thread: str) -> bool:
    return any(thread.startswith(match) for match in THREAD_ID.findall(title))


def thread_zmx_session(thread: str, windows: Iterable[KittyWindow]) -> str | None:
    names = {
        name
        for window in windows
        if shows_thread(str(window.get("title") or ""), thread)
        and (name := window_zmx_session(window))
    }
    return names.pop() if len(names) == 1 else None


def kitty_windows() -> list[KittyWindow]:
    kitten = shutil.which("kitten") or KITTEN
    windows: list[KittyWindow] = []
    for socket in sorted(glob.glob("/tmp/kitty-*")):
        try:
            result = subprocess.run(
                [kitten, "@", "--to", f"unix:{socket}", "ls"],
                capture_output=True,
                text=True,
                timeout=1,
                check=False,
            )
            data = json.loads(result.stdout) if result.returncode == 0 else []
        except OSError, subprocess.SubprocessError, ValueError:
            continue
        windows.extend(
            window
            for os_window in data
            for tab in os_window.get("tabs", [])
            for window in tab.get("windows", [])
        )
    return windows
