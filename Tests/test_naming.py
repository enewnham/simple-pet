import asyncio
import base64
import json
import tempfile
import unittest
from pathlib import Path

import pytest
from anthropic import AsyncAnthropic
from pydantic import BaseModel, ValidationError

from pet_daemon.commentary.cognition import Cognition
from pet_daemon.commentary.contracts import ModelRequest
from pet_daemon.config import Config
from pet_daemon.daemon import Daemon
from pet_daemon.domain import ModelPurpose, SpeechReason
from pet_daemon.naming import NamingConfig, SessionName, SessionNaming
from pet_daemon.runtime import HookEvent
from pet_daemon.storage import Memory


class FakeModel:
    def __init__(self) -> None:
        self.calls: list[ModelRequest] = []
        self.response: dict[str, object] = {"name": "fix sockets"}
        self.error: Exception | None = None

    async def complete(
        self,
        request: ModelRequest,
        reason: SpeechReason | ModelPurpose,
        *,
        response_type: type[BaseModel],
    ) -> dict[str, object]:
        assert reason == ModelPurpose.NAME
        assert response_type is SessionName
        self.calls.append(request)
        if self.error is not None:
            raise self.error
        return self.response


def test_naming_configuration_and_name_schema(tmp_path: Path) -> None:
    assert Config(pet_dir=tmp_path).naming is None
    for command in ([], [""], ["bad\0command"]):
        with pytest.raises(ValidationError):
            NamingConfig(command=command)
    with pytest.raises(ValidationError):
        NamingConfig.model_validate({"command": ["callback"], "debounce_seconds": 0})
    for name in ("sockets", "fix sockets", "wm-730"):
        assert SessionName(name=name).name == name
    for name in (
        "",
        "Fix sockets",
        "three short words",
        "abcdefghijklm",
        "fix  sockets",
    ):
        with pytest.raises(ValidationError):
            SessionName(name=name)


class NamingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.memory = Memory(self.root / "memory.sqlite3", "test")
        self.config = Config(pet_dir=self.root)
        self.model = FakeModel()
        self.payloads: list[dict[str, object]] = []
        self.marks: dict[str, object] = {}
        self.callback_success = True
        self.callback_error: Exception | None = None
        self.naming = self.create_naming()
        self.hook = HookEvent(
            source="codex",
            session="thread",
            event="intent",
            goal="Fix socket timeout",
            repo="/repo",
            branch="fix-sockets",
            pull_request="https://github.com/example/repo/pull/7",
            zmx_session="terminal",
        )

    def create_naming(self) -> SessionNaming:
        cognition = Cognition(self.model, self.config, memory=self.memory)
        return SessionNaming(
            NamingConfig(command=("callback",)),
            lambda hook: self.marks,
            self.memory,
            self.model,
            cognition.request,
            runner=self.run_callback,
        )

    async def run_callback(
        self, command: tuple[str, ...], payload: bytes, timeout: float
    ) -> bool:
        assert command == ("callback",)
        assert timeout == 10
        self.payloads.append(json.loads(payload))
        if self.callback_error is not None:
            raise self.callback_error
        return self.callback_success

    async def finish_naming(self) -> None:
        await asyncio.gather(*tuple(self.naming.tasks))

    async def asyncTearDown(self) -> None:
        await self.naming.close()
        self.memory.close()
        self.temp.cleanup()

    async def test_first_intent_names_once_and_preserves_callback_identifiers(
        self,
    ) -> None:
        self.naming.ingest(self.hook)
        assert len(self.naming.tasks) == 1
        self.naming.ingest(self.hook.model_copy(update={"goal": "Another prompt"}))
        await self.finish_naming()
        self.marks["title"] = "Change task"
        self.naming.ingest(self.hook.model_copy(update={"goal": "Third prompt"}))
        await self.finish_naming()
        assert len(self.model.calls) == 1
        assert self.payloads == [
            {
                "harness": "codex",
                "session": "thread",
                "thread_id": "thread",
                "zmx_session": "terminal",
                "cwd": "/repo",
                "name": "fix sockets",
            }
        ]

    async def test_subagents_and_sessions_without_zmx_are_ignored(self) -> None:
        for changes in (
            {"parent": "parent"},
            {"zmx_session": None},
            {"event": "progress"},
        ):
            self.naming.ingest(self.hook.model_copy(update=changes))
        await self.finish_naming()
        assert self.model.calls == []
        assert self.payloads == []
        assert (
            self.memory.db.execute("SELECT count(*) FROM session_names").fetchone()[0]
            == 0
        )

    async def test_invalid_names_deliver_nothing_and_are_never_retried(self) -> None:
        for index, name in enumerate(
            ("", "Fix sockets", "abcdefghijklm", "too many words")
        ):
            self.model.response = {"name": name}
            hook = self.hook.model_copy(update={"session": f"invalid-{index}"})
            self.naming.ingest(hook)
            await self.finish_naming()
            self.naming.ingest(hook)
        assert len(self.model.calls) == 4
        assert self.payloads == []

    async def test_model_failures_are_never_retried(self) -> None:
        for index, error in enumerate(
            (TimeoutError("model timeout"), ValueError("bad response"))
        ):
            self.model.error = error
            hook = self.hook.model_copy(update={"session": f"failure-{index}"})
            self.naming.ingest(hook)
            await self.finish_naming()
            self.naming.ingest(hook)
        assert len(self.model.calls) == 2
        assert self.payloads == []

    async def test_prompt_is_bounded_and_all_checkout_facts_are_redacted(self) -> None:
        secret = "sk-abcdefghijklmnop"
        self.marks.update(title=f"Repair {secret}", goal=f"Build {secret}")
        prompt = "```python\nfix_socket()\n```\n" + secret + " " + "x" * 5000
        self.naming.ingest(
            self.hook.model_copy(
                update={
                    "goal": prompt,
                    "repo": f"/repo/{secret}",
                    "branch": secret,
                    "pull_request": f"https://example.com/{secret}",
                }
            )
        )
        await self.finish_naming()
        content = self.model.calls[0]["messages"][0]["content"]
        assert secret not in content
        facts = json.loads(content)
        assert facts["first_prompt"].startswith(
            "```python\nfix_socket()\n```\n[REDACTED]"
        )
        assert len(facts["first_prompt"]) == 4000 - len(secret) + len("[REDACTED]")
        assert facts["first_prompt"].endswith("x" * 100)
        assert "x" * 4000 not in facts["first_prompt"]
        assert facts["repo"] == "/repo/[REDACTED]"
        assert facts["branch"] == "[REDACTED]"
        assert facts["pull_request"] == "https://example.com/[REDACTED]"
        assert facts["title"] == "Repair [REDACTED]"
        assert facts["goal"] == "Build [REDACTED]"

    async def test_names_and_failed_attempts_survive_storage_restart(self) -> None:
        self.naming.ingest(self.hook)
        await self.finish_naming()
        self.model.error = TimeoutError()
        failed_hook = self.hook.model_copy(update={"session": "failed"})
        self.naming.ingest(failed_hook)
        await self.finish_naming()
        await self.naming.close()
        self.memory.close()
        self.memory = Memory(self.root / "memory.sqlite3", "test")
        self.naming = self.create_naming()
        rows = self.memory.db.execute(
            "SELECT session,name FROM session_names ORDER BY session"
        ).fetchall()
        assert [(row["session"], row["name"]) for row in rows] == [
            ("failed", None),
            ("thread", "fix sockets"),
        ]
        self.naming.ingest(self.hook)
        self.naming.ingest(failed_hook)
        await self.finish_naming()
        assert len(self.model.calls) == 2
        assert len(self.payloads) == 1

    async def test_failed_callback_is_never_repeated(self) -> None:
        self.callback_error = OSError("command unavailable")
        self.naming.ingest(self.hook)
        await self.finish_naming()
        self.naming.ingest(self.hook)
        await self.finish_naming()
        assert len(self.model.calls) == 1
        assert len(self.payloads) == 1

    async def test_harnesses_with_same_native_id_are_distinct(self) -> None:
        self.naming.ingest(self.hook)
        self.naming.ingest(self.hook.model_copy(update={"source": "claude"}))
        await self.finish_naming()
        assert {payload["harness"] for payload in self.payloads} == {"codex", "claude"}

    async def test_daemon_wiring_preserves_name_across_restart(self) -> None:
        (self.root / "pixel.png").write_bytes(
            base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+ip1sAAAAASUVORK5CYII="
            )
        )
        (self.root / "manifest.json").write_text(
            json.dumps(
                {
                    "pet": "test",
                    "rest": "still",
                    "clips": {
                        "still": {"frames": ["pixel.png"], "fps": 1, "role": "rest"}
                    },
                }
            )
        )
        config = Config(
            pet_dir=self.root,
            memory_file=self.root / "daemon.sqlite3",
            naming=NamingConfig(command=("callback",)),
        )
        async with AsyncAnthropic(api_key="test") as client:
            for _ in range(2):
                daemon = Daemon(config, client)
                try:
                    assert daemon.naming is not None
                    assert daemon.naming.memory is daemon.application.memory
                    daemon.naming.client = self.model
                    daemon.naming.runner = self.run_callback
                    daemon.receive(json.dumps(self.hook.packet()).encode())
                    await asyncio.gather(*tuple(daemon.naming.tasks))
                finally:
                    await daemon.close()
        assert len(self.model.calls) == 1
        assert len(self.payloads) == 1
