import errno
import json
import os
from pathlib import Path
import runpy
import socket
import subprocess
import sys
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
CLIENT = ROOT / "bin" / "pet-notify"
HOOK = ROOT / "scripts" / "pet-hook"
WRAPPER = ROOT / "examples" / "pet-run"


class NotifyTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="pet-test-", dir="/tmp")
        self.path = Path(self.directory.name)
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.socket_path = str(self.path / "daemon.sock")
        self.listener.bind(self.socket_path)
        self.listener.settimeout(2)
        self.environment = os.environ.copy()
        self.environment["PET_DAEMON_SOCKET"] = self.socket_path
        self.environment.pop("PET_COMMENTARY", None)

    def tearDown(self):
        self.listener.close()
        self.directory.cleanup()

    def run_client(self, *arguments):
        return subprocess.run([sys.executable, str(CLIENT), *arguments], env=self.environment,
                              capture_output=True, text=True, timeout=3)

    def run_hook(self, harness, event, payload=None, raw=None):
        return subprocess.run([sys.executable, str(HOOK), harness, event],
                              input=raw if raw is not None else json.dumps(payload or {}),
                              env=self.environment, capture_output=True, text=True, timeout=3)

    def receive(self):
        data = self.listener.recv(8192)
        self.assertLessEqual(len(data), 2000)
        return json.loads(data)

    def assert_quiet(self):
        self.listener.settimeout(0.03)
        with self.assertRaises(socket.timeout):
            self.listener.recv(8192)

    def test_cli_preserves_calls_and_parallel_session_identity(self):
        for status, event in (("start", "tool_start"), ("done", "tool_done"), ("error", "tool_error")):
            result = self.run_client("--session", "worker-one", "--status", status,
                                     "--call", "test-one", "--tool", "cargo test", "--detail", "result")
            self.assertEqual((result.returncode, result.stdout, result.stderr), (0, "", ""))
            packet = self.receive()
            self.assertEqual(packet["session"], "worker-one")
            self.assertEqual(packet["call"], "test-one")
            self.assertEqual(packet["event"], event)
            self.assertEqual(packet["harness"], "cli")
            self.assertTrue(packet["manual"])

    def test_manual_single_call_and_intent_share_session(self):
        self.run_client("intent", "--session", "worker", "--text", 'Fix "quotes" 🐈')
        self.assertEqual(self.receive(), {"kind": "event", "harness": "cli", "session": "worker",
                                         "event": "intent", "goal": 'Fix "quotes" 🐈', "manual": True})
        self.run_client("--session", "worker", "--status", "start")
        self.assertEqual(self.receive()["call"], "single")

    def test_speech_only_enqueues_a_daemon_request(self):
        started = time.monotonic()
        result = self.run_client("speak", "--context", "A long and complicated event", "--ttl", "7.5")
        self.assertEqual(result.returncode, 0)
        self.assertLess(time.monotonic() - started, 1)
        self.assertEqual(self.receive(), {"kind": "speak", "context": "A long and complicated event", "ttl": 7.5})
        self.run_client("speak", "--context", "Use configured expiry")
        self.assertNotIn("ttl", self.receive())

    def test_unicode_and_escape_excerpts_fit_without_trimming_identifiers(self):
        for text in ("🐈" * 3000, "\x01" * 3000):
            self.run_client("--session", "session-" + "s" * 100, "--call", "call-" + "c" * 100,
                            "--status", "error", "--detail", text, "--tool", text)
            packet = self.receive()
            self.assertEqual(packet["session"], "session-" + "s" * 100)
            self.assertEqual(packet["call"], "call-" + "c" * 100)
            self.assertTrue(text.startswith(packet["tool"]))
            self.assertTrue(text.startswith(packet["detail"]))
            self.assertEqual(packet["event"], "tool_error")

    def test_oversize_identifiers_are_dropped_whole(self):
        self.run_client("--session", "s" * 3000, "--status", "start")
        self.assert_quiet()

    def test_absent_listener_does_not_fail_the_caller(self):
        self.environment["PET_DAEMON_SOCKET"] = str(self.path / "missing.sock")
        result = self.run_client("--session", "worker", "--status", "start")
        self.assertEqual((result.returncode, result.stderr), (0, ""))

    def test_full_socket_does_not_block_the_caller(self):
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sender:
            sender.setblocking(False)
            for _ in range(100000):
                try:
                    sender.sendto(b"x" * 512, self.socket_path)
                except OSError as error:
                    if error.errno in (errno.EAGAIN, errno.EWOULDBLOCK, errno.ENOBUFS):
                        break
                    raise
            else:
                self.fail("listener queue did not fill")
        started = time.monotonic()
        result = self.run_client("--session", "worker", "--status", "start")
        self.assertEqual(result.returncode, 0)
        self.assertLess(time.monotonic() - started, 1)

    def test_cli_rejects_missing_identity_and_invalid_ttl(self):
        for args in (("--status", "start"), ("--session", "worker"),
                     ("speak", "--context", "hi", "--ttl", "nan"),
                     ("speak", "--context", "hi", "--ttl", "-1"),
                     ("state", "working"), ("say", "hi")):
            self.assertEqual(self.run_client(*args).returncode, 2)

    def test_claude_tool_events_correlate_child_lane_and_command(self):
        payload = {"session_id": "parent", "agent_id": "child", "tool_use_id": "call-one",
                   "tool_name": "Bash", "tool_input": {"command": "cargo test"}}
        self.run_hook("claude", "PreToolUse", payload)
        packet = self.receive()
        self.assertEqual(packet["session"], "child")
        self.assertEqual(packet["parent"], "parent")
        self.assertEqual(packet["call"], "call-one")
        self.assertEqual(packet["tool"], "cargo test")
        payload["error"] = "compiler panic"
        result = self.run_hook("claude", "PostToolUseFailure", payload)
        self.assertEqual(json.loads(result.stdout), {})
        packet = self.receive()
        self.assertEqual((packet["event"], packet["detail"], packet["outcome"]),
                         ("tool_error", "compiler panic", "error"))

    def test_native_prompt_and_progress_excerpts_are_bounded(self):
        self.run_hook("claude", "UserPromptSubmit", {"session_id": "worker", "prompt": "x" * 1000})
        self.assertEqual(len(self.receive()["goal"]), 240)
        self.run_hook("codex", "Stop", {"session_id": "worker", "turn_id": "turn-one",
                      "last_assistant_message": "y" * 1000, "transcript_path": "/must/not/read"})
        packet = self.receive()
        self.assertEqual(len(packet["progress"]), 240)
        self.assertEqual(packet["turn"], "turn-one")
        self.assertNotIn("transcript_path", packet)

    def test_claude_stop_failure_closes_the_turn_as_error(self):
        self.run_hook("claude", "StopFailure", {"session_id": "worker", "error": "API error"})
        packet = self.receive()
        self.assertEqual((packet["event"], packet["outcome"]), ("turn_error", "error"))

    def test_codex_subagents_use_child_identity_and_parent_relationship(self):
        for event, expected in (("SubagentStart", "subagent_start"), ("SubagentStop", "subagent_done")):
            result = self.run_hook("codex", event, {"session_id": "parent", "agent_id": "child",
                                                   "turn_id": "parent-turn"})
            self.assertEqual(json.loads(result.stdout), {})
            packet = self.receive()
            self.assertEqual((packet["session"], packet["parent"], packet["event"]), ("child", "parent", expected))
            self.assertNotIn("turn", packet)

    def test_codex_detects_structured_failures_without_guessing_from_output(self):
        for response, expected in (({"exit_code": 1}, "tool_error"), ({"isError": True}, "tool_error"),
                                   ({"exit_code": 0}, "tool_done"), ({"stdout": "Exit code: 1"}, "tool_done"),
                                   ("Exit code: 1", "tool_done")):
            self.run_hook("codex", "PostToolUse", {"session_id": "worker", "turn_id": "turn",
                          "tool_use_id": "call", "tool_response": response})
            packet = self.receive()
            self.assertEqual(packet["event"], expected)
            self.assertEqual(packet["call"], "call")

    def test_cursor_subagent_aliases_link_child_tools(self):
        self.run_hook("cursor", "subagentStop", {"conversation_id": "parent", "subagent_id": "agent",
                      "child_conversation_id": "child", "tool_call_id": "spawn-call"})
        packet = self.receive()
        self.assertEqual(packet["session"], "child")
        self.assertEqual(packet["parent"], "parent")
        self.assertEqual(packet["aliases"], ["agent", "child"])
        self.assertEqual(packet["parent_call"], "spawn-call")
        self.run_hook("cursor", "preToolUse", {"conversation_id": "child", "generation_id": "generation",
                      "tool_use_id": "nested-call", "parent_tool_call_id": "spawn-call", "tool_name": "Shell"})
        packet = self.receive()
        self.assertEqual((packet["session"], packet["turn"], packet["call"], packet["parent_call"]),
                         ("child", "generation", "nested-call", "spawn-call"))

    def test_cursor_tool_output_and_subagent_summary_update_context(self):
        self.run_hook("cursor", "postToolUse", {"conversation_id": "worker", "tool_use_id": "call",
                      "tool_output": json.dumps({"exitCode": 0, "stdout": "All tests passed"})})
        self.assertEqual(self.receive()["progress"], "All tests passed")
        self.run_hook("cursor", "subagentStart", {"conversation_id": "parent", "subagent_id": "child",
                      "task": "Explore authentication", "tool_call_id": "spawn"})
        self.assertEqual(self.receive()["goal"], "Explore authentication")
        self.run_hook("cursor", "subagentStop", {"conversation_id": "parent", "subagent_id": "child",
                      "status": "error", "summary": "Authentication tests failed"})
        packet = self.receive()
        self.assertEqual((packet["event"], packet["outcome"], packet["progress"]),
                         ("subagent_done", "error", "Authentication tests failed"))

    def test_interrupted_tools_finish_without_an_error_alert(self):
        for harness, event, key in (("claude", "PostToolUseFailure", "session_id"),
                                    ("cursor", "postToolUseFailure", "conversation_id")):
            self.run_hook(harness, event, {key: "worker", "tool_use_id": "call", "is_interrupt": True})
            packet = self.receive()
            self.assertEqual((packet["event"], packet["outcome"]), ("tool_done", "cancelled"))

    def test_cursor_stop_distinguishes_failure_and_cancellation(self):
        for status, event, outcome in (("completed", "turn_done", "success"),
                                       ("error", "turn_error", "error"), ("aborted", "interrupt", "cancelled")):
            result = self.run_hook("cursor", "stop", {"conversation_id": "worker", "status": status})
            self.assertEqual(json.loads(result.stdout), {})
            packet = self.receive()
            self.assertEqual((packet["event"], packet["outcome"]), (event, outcome))

    def test_waiting_input_keeps_tool_identity_without_starting_work(self):
        for harness, event, session_key in (("claude", "PreToolUse", "session_id"),
                                            ("codex", "PreToolUse", "session_id"),
                                            ("cursor", "preToolUse", "conversation_id")):
            self.run_hook(harness, event, {session_key: "worker", "tool_name": "request_user_input",
                                         "tool_use_id": "question"})
            packet = self.receive()
            self.assertEqual((packet["event"], packet["call"]), ("waiting", "question"))

    def test_imported_claude_hooks_and_recursion_keep_neutral_response(self):
        self.run_hook("claude", "PreToolUse", {"session_id": "worker", "cursor_version": "2.0"})
        self.assert_quiet()
        self.environment["PET_COMMENTARY"] = "1"
        result = self.run_hook("cursor", "beforeSubmitPrompt", {"conversation_id": "worker", "prompt": "goal"})
        self.assertEqual(json.loads(result.stdout), {"continue": True})
        self.assert_quiet()

    def test_invalid_payloads_do_not_affect_agent_decisions(self):
        for raw in ("bad json", "[]", "{}"):
            result = self.run_hook("claude", "PreToolUse", raw=raw)
            self.assertEqual((result.returncode, json.loads(result.stdout)), (0, {}))
        self.assert_quiet()

    def test_cursor_prompt_response_allows_submission(self):
        result = self.run_hook("cursor", "beforeSubmitPrompt", {"conversation_id": "worker", "prompt": "goal"})
        self.assertEqual(json.loads(result.stdout), {"continue": True})
        self.assertEqual(self.receive()["goal"], "goal")

    def test_examples_use_supported_native_events(self):
        module = runpy.run_path(str(HOOK))
        for harness, filename in (("claude", "claude-settings.json"), ("codex", "codex-hooks.json"),
                                  ("cursor", "cursor-hooks.json")):
            config = json.loads((ROOT / "examples" / filename).read_text())
            self.assertEqual(set(config["hooks"]), set(module["EVENTS"][harness]))
            self.assertNotIn("--speak", json.dumps(config))

    def test_wrapper_preserves_piped_stdin(self):
        result = subprocess.run([str(WRAPPER), "/bin/cat"], input="piped command input\n",
                                env=self.environment, text=True, capture_output=True, timeout=4)
        self.assertEqual((result.returncode, result.stdout), (0, "piped command input\n"))
        for expected in ("intent", "tool_start", "tool_done", "session_end"):
            self.assertEqual(self.receive()["event"], expected)

    def test_wrapper_terminates_its_command_and_reports_cancellation(self):
        process = subprocess.Popen([str(WRAPPER), sys.executable, "-c",
                                    "import time; time.sleep(30)"], env=self.environment,
                                   text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertEqual(self.receive()["event"], "intent")
        self.assertEqual(self.receive()["event"], "tool_start")
        process.terminate()
        stdout, stderr = process.communicate(timeout=3)
        self.assertEqual(process.returncode, 143)
        interrupt, end = self.receive(), self.receive()
        self.assertEqual(interrupt["event"], "interrupt")
        self.assertEqual(end["event"], "session_end")
        self.assertEqual(end["outcome"], "cancelled")

    def test_wrapper_preserves_output_status_and_unique_session(self):
        sessions = []
        for code in (0, 7):
            result = subprocess.run([str(WRAPPER), sys.executable, "-c",
                                     f"import sys; print('command output'); sys.exit({code})"],
                                    env=self.environment, text=True, capture_output=True, timeout=4)
            self.assertEqual((result.returncode, result.stdout), (code, "command output\n"))
            intent, start, finish, end = (self.receive() for _ in range(4))
            self.assertEqual(intent["event"], "intent")
            self.assertEqual(start["event"], "tool_start")
            self.assertEqual(finish["event"], "tool_done" if code == 0 else "tool_error")
            self.assertEqual(intent["session"], start["session"])
            self.assertEqual(finish["session"], start["session"])
            self.assertEqual(end["event"], "session_end")
            self.assertEqual(end["outcome"], "success" if code == 0 else "error")
            sessions.append(start["session"])
        self.assertNotEqual(*sessions)


if __name__ == "__main__":
    unittest.main()
