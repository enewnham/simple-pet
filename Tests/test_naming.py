import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pytest
from pydantic import ValidationError

from pet_daemon.config import Config
from pet_daemon.naming import NamingConfig, SessionNaming, proposed_name
from pet_daemon.runtime import HookEvent


def test_naming_is_optional(tmp_path: Path) -> None:
    assert Config(pet_dir=tmp_path).naming is None
    for command in ([], [""], ["bad\0command"]):
        with pytest.raises(ValidationError):
            NamingConfig(command=command)


def test_name_uses_bounded_work_text() -> None:
    assert (
        proposed_name("Please fix the socket timeout.\nMore context")
        == "fix the socket timeout"
    )
    assert len(proposed_name("x" * 100)) == 64
    assert proposed_name("") == ""
    assert (
        proposed_name(
            "<instructions>" + "Setup " * 1000 + "</instructions>\nFix retries"
        )
        == "Fix retries"
    )


class NamingTests(unittest.IsolatedAsyncioTestCase):
    async def test_naming_debounces_marks_and_preserves_identifiers(self) -> None:
        now = [0.0]
        marks: dict[str, object] = {}
        payloads = []

        async def run(command: tuple[str, ...], payload: bytes, timeout: float) -> bool:
            payloads.append(json.loads(payload))
            return True

        naming = SessionNaming(
            NamingConfig(command=("callback",)), lambda hook: marks, lambda: now[0], run
        )
        naming.ingest(
            HookEvent(
                source="codex",
                session="thread",
                event="intent",
                goal="Fix socket timeout",
                repo="/repo",
                zmx_session="terminal",
            )
        )
        naming.tick()
        assert naming.task is not None
        await naming.task
        assert payloads == []
        now[0] = 10
        naming.tick()
        assert naming.task is not None
        await naming.task
        assert payloads == [
            {
                "harness": "codex",
                "session": "thread",
                "thread_id": "thread",
                "zmx_session": "terminal",
                "cwd": "/repo",
                "name": "Fix socket timeout",
            }
        ]
        naming.ingest(
            HookEvent(
                source="codex", session="thread", event="intent", goal="Another prompt"
            )
        )
        marks["title"] = "Repair retries"
        naming.tick()
        assert naming.task is not None
        await naming.task
        now[0] = 15
        marks["title"] = "Test retries"
        naming.tick()
        assert naming.task is not None
        await naming.task
        now[0] = 25
        naming.tick()
        assert naming.task is not None
        await naming.task
        assert [p["name"] for p in payloads] == ["Fix socket timeout", "Test retries"]
        assert payloads[-1]["zmx_session"] == "terminal"
        now[0] = 100
        naming.tick()
        assert naming.task is not None
        await naming.task
        assert len(payloads) == 2
        await naming.close()

    async def test_callback_receives_json_on_stdin(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        tmp_path = Path(temp.name)

        destination = tmp_path / "payload.json"
        command = (
            sys.executable,
            "-c",
            "import pathlib,sys; pathlib.Path(sys.argv[1]).write_bytes(sys.stdin.buffer.read())",
            str(destination),
        )
        naming = SessionNaming(
            NamingConfig(command=command, debounce_seconds=0), lambda hook: {}
        )
        naming.ingest(
            HookEvent(
                source="claude",
                session="session",
                event="intent",
                goal="Add naming hooks",
            )
        )
        naming.tick()
        assert naming.task is not None
        await naming.task
        assert json.loads(destination.read_text())["name"] == "Add naming hooks"
        await naming.close()

    async def test_close_cancels_callback(self) -> None:
        started = asyncio.Event()

        async def run(command: tuple[str, ...], payload: bytes, timeout: float) -> bool:
            started.set()
            await asyncio.Future()
            return True

        naming = SessionNaming(
            NamingConfig(command=("callback",), debounce_seconds=0),
            lambda hook: {},
            runner=run,
        )
        naming.ingest(
            HookEvent(
                source="codex", session="thread", event="intent", goal="Fix retries"
            )
        )
        naming.tick()
        await started.wait()
        await naming.close()
        assert naming.task is not None and naming.task.cancelled()

    async def test_failed_names_are_not_repeated_and_sessions_stay_distinct(
        self,
    ) -> None:
        payloads = []

        async def run(command: tuple[str, ...], payload: bytes, timeout: float) -> bool:
            payloads.append(json.loads(payload))
            raise OSError("command unavailable")

        naming = SessionNaming(
            NamingConfig(command=("callback",), debounce_seconds=0),
            lambda hook: {},
            runner=run,
        )
        for source in ("codex", "claude"):
            naming.ingest(
                HookEvent(
                    source=source, session="same-id", event="intent", goal="Fix retries"
                )
            )
        for _ in range(4):
            naming.tick()
            assert naming.task is not None
            await naming.task
        assert [p["harness"] for p in payloads] == ["codex", "claude"]
        await naming.close()

    async def test_zmx_example_passes_title_as_one_argument(self) -> None:
        from pet_daemon.notifications import run_command

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "args.json"
            fake_zmx = root / "zmx"
            fake_zmx.write_text(
                f"#!{sys.executable}\nimport json, pathlib, sys\n"
                f"pathlib.Path({str(output)!r}).write_text(json.dumps(sys.argv[1:]))\n"
            )
            fake_zmx.chmod(0o755)
            example = (
                Path(__file__).resolve().parents[1] / "examples/name-zmx-session.py"
            )
            command = (sys.executable, str(example), str(fake_zmx))
            payload = {"zmx_session": "work session", "name": "Fix 'quotes'; $(text)"}
            assert await run_command(command, json.dumps(payload).encode(), 5)
            assert json.loads(output.read_text()) == [
                "set",
                "work session",
                "title=Fix 'quotes'; $(text)",
            ]
            output.unlink()
            assert await run_command(
                command, b'{"zmx_session": null, "name": "Work"}', 5
            )
            assert not output.exists()
