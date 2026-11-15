import io
import json
import os
import unittest
from collections.abc import Mapping
from unittest import mock

from pet_daemon.cli import hook
from pet_daemon.runtime import ProcessIdentity
from pet_daemon.sessions.terminals import KittyWindow, thread_zmx_session

THREAD = "01a1dcfb-9a6a-7e03-9aff-451fb781cd72"
PAYLOAD = {"session_id": THREAD, "prompt": "Fix it", "cwd": "/tmp"}
CLIENT = ProcessIdentity(pid=20, started="start")


def window(title: str, *cmdline: str) -> KittyWindow:
    return {"title": title, "foreground_processes": [{"cmdline": list(cmdline)}]}


CODEX_WINDOW = window(
    "Reply ok | 01a1dcfb-9a6a-7e03-9aff-451fb...",
    "zmx",
    "attach",
    "simple-pet-codex-72705",
    "/opt/homebrew/bin/codex",
)


def run_hook(
    source: str,
    owner: ProcessIdentity | None,
    payload: Mapping[str, object],
    windows: list[KittyWindow],
) -> list[dict[str, object]]:
    sent: list[dict[str, object]] = []
    with (
        mock.patch.dict(os.environ, {"ZMX_SESSION": "launcher-codex-1"}),
        mock.patch.object(hook, "discover_owner", return_value=owner),
        mock.patch.object(hook, "kitty_windows", return_value=windows),
        mock.patch.object(hook, "send", side_effect=sent.append),
        mock.patch("sys.argv", ["pet-hook", source, "UserPromptSubmit"]),
        mock.patch("sys.stdin", io.StringIO(json.dumps(payload))),
        mock.patch("sys.stdout", io.StringIO()),
    ):
        hook.main()
    return sent


class HookZmxSessionTests(unittest.TestCase):
    def test_codex_daemon_hook_takes_zmx_session_from_thread_window(self) -> None:
        [packet] = run_hook("codex", None, PAYLOAD, [CODEX_WINDOW])
        self.assertEqual(packet["zmx_session"], "simple-pet-codex-72705")

    def test_codex_daemon_hook_without_thread_window_omits_zmx_session(self) -> None:
        [packet] = run_hook("codex", None, PAYLOAD, [])
        self.assertNotIn("zmx_session", packet)

    def test_codex_daemon_hook_keeps_payload_zmx_session(self) -> None:
        payload = {**PAYLOAD, "zmx_session": "work"}
        [packet] = run_hook("codex", None, payload, [CODEX_WINDOW])
        self.assertEqual(packet["zmx_session"], "work")

    def test_codex_client_hook_keeps_env_zmx_session(self) -> None:
        [packet] = run_hook("codex", CLIENT, PAYLOAD, [CODEX_WINDOW])
        self.assertEqual(packet["zmx_session"], "launcher-codex-1")


class ThreadZmxSessionTests(unittest.TestCase):
    def test_full_and_cut_thread_ids_match(self) -> None:
        full = window(f"Fix | {THREAD}", "zmx", "attach", "work-codex-1")
        self.assertEqual(thread_zmx_session(THREAD, [full]), "work-codex-1")
        self.assertEqual(
            thread_zmx_session(THREAD, [CODEX_WINDOW]), "simple-pet-codex-72705"
        )

    def test_other_thread_or_plain_window_does_not_match(self) -> None:
        other = window("Fix | 01a1dc17-de2d-7e80-843a-c6c3e...", "zmx", "attach", "x")
        plain = window(f"Fix | {THREAD}", "/bin/zsh")
        self.assertIsNone(thread_zmx_session(THREAD, [other, plain]))

    def test_two_sessions_showing_the_thread_are_ambiguous(self) -> None:
        first = window(f"A | {THREAD}", "zmx", "attach", "one")
        second = window(f"B | {THREAD}", "zmx", "attach", "two")
        self.assertIsNone(thread_zmx_session(THREAD, [first, second]))


if __name__ == "__main__":
    unittest.main()
