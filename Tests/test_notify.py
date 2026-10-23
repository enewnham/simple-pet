import errno
import json
import os
import runpy
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from collections.abc import Mapping
from pathlib import Path
from typing import TypedDict

from pet_daemon.attention import AttentionSelection
from pet_daemon.commentary.cognition import Cognition
from pet_daemon.commentary.contracts import ModelRequest
from pet_daemon.commentary.story import prepare
from pet_daemon.config import Config
from pet_daemon.domain import ModelPurpose, SpeechReason, SpeechSituation
from pet_daemon.runtime import ProcessPacket
from pet_daemon.sessions.tracking import Sessions
from pet_daemon.sources.hooks import normalize_hook
from pet_daemon.storage import Memory
from Tests.fixtures import capture_snapshot

ROOT = Path(__file__).resolve().parents[1]
CLIENT = ROOT / "bin" / "pet-notify"
HOOK = ROOT / "scripts" / "pet-hook"
WRAPPER = ROOT / "examples" / "pet-run"


class NoModel:
    async def complete(
        self, request: ModelRequest, reason: SpeechReason | ModelPurpose
    ) -> dict[str, object]:
        raise AssertionError("Preparing a request must not call the model")


class ReceivedPacket(TypedDict, total=False):
    kind: str
    harness: str
    session: str
    event: str
    aliases: list[str]
    parent: str
    parent_call: str
    call: str
    turn: str
    tool: str
    goal: str
    progress: str
    repo: str
    repository: str
    branch: str
    pull_request: str
    tool_name: str
    sentiment_text: str
    waiting_kind: str
    outcome: str
    manual: bool
    owner: ProcessPacket
    exit_code: int


class NotifyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory(prefix="pet-test-", dir="/tmp")
        self.path = Path(self.directory.name)
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.socket_path = str(self.path / "daemon.sock")
        self.listener.bind(self.socket_path)
        self.listener.settimeout(10)
        self.environment = os.environ.copy()
        self.environment["PET_DAEMON_SOCKET"] = self.socket_path

    def tearDown(self) -> None:
        self.listener.close()
        self.directory.cleanup()

    def run_client(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(CLIENT), *arguments],
            env=self.environment,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

    def run_hook(
        self,
        harness: str,
        event: str,
        payload: Mapping[str, object] | None = None,
        raw: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(HOOK), harness, event],
            input=raw if raw is not None else json.dumps(payload or {}),
            env=self.environment,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

    def receive(self) -> ReceivedPacket:
        data = self.listener.recv(8192)
        self.assertLessEqual(len(data), 2000)
        return json.loads(data)

    def assert_quiet(self) -> None:
        self.listener.settimeout(0.03)
        with self.assertRaises(socket.timeout):
            self.listener.recv(8192)

    def test_cli_preserves_calls_and_parallel_session_identity(self) -> None:
        for status, event in (
            ("start", "tool_start"),
            ("done", "tool_done"),
            ("error", "tool_error"),
        ):
            result = self.run_client(
                "--session",
                "worker-one",
                "--status",
                status,
                "--call",
                "test-one",
                "--tool",
                "cargo test",
                "--exit-code",
                "1",
            )
            self.assertEqual(
                (result.returncode, result.stdout, result.stderr), (0, "", "")
            )
            packet = self.receive()
            self.assertEqual(packet["session"], "worker-one")
            self.assertEqual(packet["call"], "test-one")
            self.assertEqual(packet["event"], event)
            self.assertEqual(packet["harness"], "cli")
            self.assertTrue(packet["manual"])
            self.assertEqual(packet["exit_code"], 1)

    def test_manual_single_call_and_intent_share_session(self) -> None:
        self.run_client("intent", "--session", "worker", "--text", 'Fix "quotes" 🐈')
        self.assertEqual(
            self.receive(),
            {
                "kind": "event",
                "harness": "cli",
                "session": "worker",
                "event": "intent",
                "goal": 'Fix "quotes" 🐈',
                "manual": True,
            },
        )
        self.run_client("--session", "worker", "--status", "start")
        self.assertEqual(self.receive()["call"], "single")

    def test_hook_preserves_user_sentiment_prose_before_prompt_compression(
        self,
    ) -> None:
        for harness, event, identity in (
            ("claude", "UserPromptSubmit", {"session_id": "worker"}),
            ("codex", "UserPromptSubmit", {"session_id": "worker"}),
            ("cursor", "beforeSubmitPrompt", {"conversation_id": "worker"}),
        ):
            with self.subTest(harness=harness):
                prompt = (
                    "This is terrible.\n```\n"
                    + 'error = "awful"\n' * 500
                    + "```\nPlease fix the parser."
                )
                result = self.run_hook(harness, event, {**identity, "prompt": prompt})
                self.assertEqual(result.returncode, 0, result.stderr)
                packet = self.receive()
                self.assertEqual(
                    packet["sentiment_text"],
                    "This is terrible.\nPlease fix the parser.",
                )
                self.assertIn("Please fix the parser.", packet["goal"])

    def test_clear_memory_sends_command_and_reports_unavailable_daemon(self) -> None:
        result = self.run_client("clear-memory")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(self.receive(), {"kind": "clear_memory"})
        self.environment["PET_DAEMON_SOCKET"] = str(self.path / "missing.sock")
        result = self.run_client("clear-memory")
        self.assertEqual(result.returncode, 1)
        self.assertIn("could not send clear-memory", result.stderr)

    def test_clear_speech_sends_command_and_reports_unavailable_daemon(self) -> None:
        result = self.run_client("clear-speech")
        self.assertEqual(result.returncode, 0)
        self.assertEqual(self.receive(), {"kind": "clear_speech"})
        self.environment["PET_DAEMON_SOCKET"] = str(self.path / "missing.sock")
        result = self.run_client("clear-speech")
        self.assertEqual(result.returncode, 1)
        self.assertIn("could not send clear-speech", result.stderr)

    def test_speech_only_enqueues_a_daemon_request(self) -> None:
        result = self.run_client(
            "speak", "--context", "A long and complicated event", "--ttl", "7.5"
        )
        self.assertEqual(result.returncode, 0)
        self.assertEqual(
            self.receive(),
            {"kind": "speak", "context": "A long and complicated event", "ttl": 7.5},
        )
        self.run_client("speak", "--context", "Use configured expiry")
        self.assertNotIn("ttl", self.receive())

    def test_unicode_and_escape_excerpts_fit_without_trimming_identifiers(self) -> None:
        for text in ("🐈" * 3000, "\x01" * 3000):
            self.run_client(
                "--session",
                "session-" + "s" * 100,
                "--call",
                "call-" + "c" * 100,
                "--status",
                "error",
                "--tool",
                text,
            )
            packet = self.receive()
            self.assertEqual(packet["session"], "session-" + "s" * 100)
            self.assertEqual(packet["call"], "call-" + "c" * 100)
            self.assertTrue(text.startswith(packet["tool"]))
            self.assertEqual(packet["event"], "tool_error")

    def test_oversize_identifiers_are_dropped_whole(self) -> None:
        self.run_client("--session", "s" * 3000, "--status", "start")
        self.assert_quiet()

    def test_absent_listener_does_not_fail_the_caller(self) -> None:
        self.environment["PET_DAEMON_SOCKET"] = str(self.path / "missing.sock")
        result = self.run_client("--session", "worker", "--status", "start")
        self.assertEqual((result.returncode, result.stderr), (0, ""))

    def test_full_socket_does_not_block_the_caller(self) -> None:
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
        result = self.run_client("--session", "worker", "--status", "start")
        self.assertEqual(result.returncode, 0)

    def test_cli_rejects_missing_identity_and_invalid_ttl(self) -> None:
        for args in (
            ("--status", "start"),
            ("--session", "worker"),
            ("speak", "--context", "hi", "--ttl", "nan"),
            ("speak", "--context", "hi", "--ttl", "-1"),
            ("state", "working"),
            ("say", "hi"),
        ):
            self.assertEqual(self.run_client(*args).returncode, 2)

    def test_claude_tool_events_correlate_child_lane_and_command(self) -> None:
        payload = {
            "session_id": "parent",
            "agent_id": "child",
            "tool_use_id": "call-one",
            "tool_name": "Bash",
            "tool_input": {"command": "cargo test"},
        }
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
        self.assertEqual((packet["event"], packet["outcome"]), ("tool_error", "error"))
        self.assertNotIn("compiler panic", json.dumps(packet))

    def test_native_prompt_and_progress_excerpts_are_bounded(self) -> None:
        self.run_hook(
            "claude", "UserPromptSubmit", {"session_id": "worker", "prompt": "x" * 1000}
        )
        self.assertEqual(len(self.receive()["goal"]), 1000)
        self.run_hook(
            "codex",
            "Stop",
            {
                "session_id": "worker",
                "turn_id": "turn-one",
                "last_assistant_message": "y" * 1000,
                "transcript_path": "/must/not/read",
            },
        )
        packet = self.receive()
        self.assertLessEqual(len(packet["progress"]), 800)
        self.assertTrue(packet["progress"].endswith("[omitted]"))
        self.assertEqual(packet["turn"], "turn-one")
        self.assertNotIn("transcript_path", packet)

    def test_long_hook_prompt_keeps_the_final_request(self) -> None:
        prompt = "Background details. " * 100 + "Fix the parser for UTF-8 input."
        self.run_hook(
            "claude", "UserPromptSubmit", {"session_id": "worker", "prompt": prompt}
        )
        goal = self.receive()["goal"]
        self.assertLessEqual(len(goal), 1000)
        self.assertIn("Fix the parser for UTF-8 input.", goal)
        self.assertIn("[omitted]", goal)

    def test_cli_and_native_goals_filter_markdown_before_story_selection(self) -> None:
        prompt = (
            "Review the commentary.\n"
            + "> Fix the haunted mail.\n" * 60
            + "```python\nraise RuntimeError('invented failure')\n```\n"
            + "Request | Response\n--- | ---\nMail | Unrelated worry\n\n"
            + "Background details.\n" * 100
            + "Fix the parser for UTF-8 input."
        )
        self.run_client("intent", "--session", "cli-review", "--text", prompt)
        cli_packet = self.receive()
        self.run_hook(
            "claude",
            "UserPromptSubmit",
            {"session_id": "hook-review", "prompt": prompt},
        )
        hook_packet = self.receive()
        native_packet = {
            "harness": "codex",
            "session": "native-review",
            "event": "intent",
            "goal": prompt,
        }
        for packet in (cli_packet, hook_packet, native_packet):
            with self.subTest(harness=packet["harness"]):
                memory = Memory(
                    self.path / f"{packet['harness']}.sqlite3",
                    "pet",
                    clock=lambda: 1000,
                )
                try:
                    sessions = Sessions(lambda: 1000)
                    result = sessions.handle(packet)
                    self.assertIsNotNone(result)
                    assert result is not None
                    memory.observe_session_context(result.observation)
                    memory.work.record_session(result.observation)
                    activity = sessions.activity_view()
                    subjects = memory.connected_work(activity)
                    snapshot = capture_snapshot(activity, memory=memory, work=subjects)
                    selection = AttentionSelection(
                        subjects[0],
                        activity.sessions[0],
                        (),
                        SpeechReason.ROUTINE,
                        SpeechSituation.ROUTINE,
                    )
                    cognition = Cognition(
                        NoModel(), Config(pet_dir=self.path), memory=memory
                    )
                    prepared = prepare(snapshot, selection, memory, cognition.request)
                    self.assertIsNotNone(prepared)
                    assert prepared is not None
                    content = prepared.request["messages"][0]["content"]
                    self.assertIn("Fix the parser for UTF-8 input.", content)
                    for noise in (
                        "haunted mail",
                        "invented failure",
                        "Unrelated worry",
                    ):
                        self.assertNotIn(noise, content)
                finally:
                    memory.close()

    def test_cli_intent_keeps_unicode_request_within_packet_limit(self) -> None:
        prompt = (
            "> Fix the haunted mail.\n" * 60
            + "背景。 " * 400
            + "Fix parser/猫.py for 🐈 input."
        )
        session = "worker-" + "s" * 400
        self.run_client("intent", "--session", session, "--text", prompt)
        packet = self.receive()
        goal = packet["goal"]
        self.assertEqual(packet["session"], session)
        self.assertIn("Fix parser/猫.py for 🐈 input.", goal)
        self.assertNotIn("haunted mail", goal)
        self.assertLessEqual(len(goal), 1000)

    def test_claude_stop_failure_closes_the_turn_as_error(self) -> None:
        self.run_hook(
            "claude", "StopFailure", {"session_id": "worker", "error": "API error"}
        )
        packet = self.receive()
        self.assertEqual((packet["event"], packet["outcome"]), ("turn_error", "error"))

    def test_question_tool_outcomes_determine_turn_completion(self) -> None:
        for completion, extra, outcome in (
            ("PostToolUse", {}, "success"),
            ("PostToolUseFailure", {}, "error"),
            ("PostToolUseFailure", {"is_interrupt": True}, "cancelled"),
        ):
            with self.subTest(outcome=outcome):
                sessions = Sessions(time.monotonic)
                for event, payload in (
                    ("UserPromptSubmit", {"prompt": "Choose a strategy"}),
                    (
                        "PreToolUse",
                        {"tool_use_id": "question", "tool_name": "AskUserQuestion"},
                    ),
                    (
                        completion,
                        {
                            "tool_use_id": "question",
                            "tool_name": "AskUserQuestion",
                            **extra,
                        },
                    ),
                    ("Stop", {}),
                ):
                    result = self.run_hook(
                        "claude", event, {"session_id": "worker", **payload}
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    accepted = sessions.handle(self.receive())
                    if event == completion:
                        assert accepted is not None
                        self.assertEqual(accepted.failure, outcome == "error")
                self.assertEqual(sessions.activity_view().sessions[0].outcome, outcome)
                self.assertEqual(sessions.clean_convergence, outcome == "success")

    def test_codex_subagents_use_child_identity_and_parent_relationship(self) -> None:
        for event, expected in (
            ("SubagentStart", "subagent_start"),
            ("SubagentStop", "subagent_done"),
        ):
            result = self.run_hook(
                "codex",
                event,
                {"session_id": "parent", "agent_id": "child", "turn_id": "parent-turn"},
            )
            self.assertEqual(json.loads(result.stdout), {})
            packet = self.receive()
            self.assertEqual(
                (packet["session"], packet["parent"], packet["event"]),
                ("child", "parent", expected),
            )
            self.assertNotIn("turn", packet)

    def test_child_turns_and_tools_stay_separate_from_the_parent(self) -> None:
        for harness in ("claude", "codex"):
            with self.subTest(harness=harness):
                sessions = Sessions(time.monotonic)
                for agent in ("", "first", "second"):
                    identity = {"session_id": "parent"}
                    if agent:
                        identity["agent_id"] = agent
                    self.run_hook(
                        harness,
                        "UserPromptSubmit",
                        {**identity, "turn_id": agent or "parent", "prompt": "Inspect"},
                    )
                    self.assertIsNotNone(sessions.handle(self.receive()))
                for event in ("PreToolUse", "PostToolUse"):
                    for agent in ("first", "", "second"):
                        name = agent or "parent"
                        self.run_hook(
                            harness,
                            event,
                            {
                                "session_id": "parent",
                                "agent_id": agent,
                                "turn_id": name,
                                "tool_use_id": name + "-call",
                                "tool_name": "Bash",
                            },
                        )
                        self.assertIsNotNone(sessions.handle(self.receive()))
                    for name in ("parent", "first", "second"):
                        session = sessions.by_id[f"{harness}:{name}"]
                        self.assertEqual(session.turn, name)
                        self.assertEqual(
                            set(session.calls),
                            {name + "-call"} if event == "PreToolUse" else set(),
                        )
                        self.assertEqual(
                            session.parent,
                            None if name == "parent" else f"{harness}:parent",
                        )
                self.run_hook(
                    harness,
                    "PreToolUse",
                    {
                        "session_id": "parent",
                        "agent_id": "first",
                        "turn_id": "stale-turn",
                        "tool_use_id": "stale-call",
                    },
                )
                self.assertIsNone(sessions.handle(self.receive()))
                self.assertEqual(sessions.by_id[f"{harness}:first"].turn, "first")
                self.assertFalse(sessions.by_id[f"{harness}:first"].calls)

    def test_child_prompts_do_not_score_sentiment_and_input_requests_stay_child_scoped(
        self,
    ) -> None:
        for harness in ("claude", "codex"):
            with self.subTest(harness=harness):
                identity = {"session_id": "parent", "agent_id": "child"}
                self.run_hook(
                    harness,
                    "UserPromptSubmit",
                    {**identity, "prompt": "This is terrible. Inspect the parser."},
                )
                packet = self.receive()
                self.assertEqual(
                    (packet["session"], packet["parent"]), ("child", "parent")
                )
                self.assertNotIn("sentiment_text", packet)
                self.run_hook(
                    harness,
                    "PreToolUse",
                    {
                        **identity,
                        "tool_name": "request_user_input",
                        "tool_use_id": "ask",
                    },
                )
                packet = self.receive()
                self.assertEqual(
                    (
                        packet["session"],
                        packet["parent"],
                        packet["event"],
                        packet["call"],
                    ),
                    ("child", "parent", "waiting", "ask"),
                )
                self.assertEqual(packet["waiting_kind"], "question")

    def test_permission_waits_keep_their_command_separate_from_user_questions(
        self,
    ) -> None:
        sessions = Sessions(time.monotonic)
        for event, tool, expected in (
            ("PermissionRequest", "Bash", "permission"),
            ("PreToolUse", "request_user_input", "question"),
        ):
            with self.subTest(event=event):
                self.run_hook(
                    "codex",
                    event,
                    {
                        "session_id": "worker",
                        "tool_name": tool,
                        "tool_input": {
                            "command": "python check.py",
                            "questions": [{"question": "Which strategy should I use?"}],
                        },
                    },
                )
                packet = self.receive()
                self.assertEqual(packet["waiting_kind"], expected)
                self.assertIsNotNone(sessions.handle(packet))
                snapshot = sessions.activity_view().sessions[0]
                self.assertEqual(snapshot.waiting_kind, expected)
                self.assertEqual(snapshot.waiting_tool, "python check.py")
                self.assertEqual(
                    snapshot.waiting_prompt,
                    "Which strategy should I use?" if expected == "question" else None,
                )
                self.assertTrue(snapshot.waiting_for_input)
                self.assertIsNotNone(
                    sessions.handle(
                        {"harness": "codex", "session": "worker", "event": "progress"}
                    )
                )
                resumed = sessions.activity_view().sessions[0]
                self.assertFalse(resumed.waiting_for_input)
                self.assertIsNone(resumed.waiting_kind)
                self.assertIsNone(resumed.waiting_tool)
                self.assertIsNone(resumed.waiting_prompt)

    def test_hook_preserves_explicit_report_references(self) -> None:
        result = self.run_hook(
            "codex",
            "UserPromptSubmit",
            {
                "session_id": "worker",
                "prompt": "Cover parser escapes",
                "cwd": "/src/parser",
                "repository": "github.com/team/parser",
                "branch": "escapes",
                "pull_request": "https://github.com/team/parser/pull/42",
            },
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        event = normalize_hook(self.receive())
        assert event is not None
        self.assertEqual(
            (event.repository, event.branch, event.pull_request),
            (
                "github.com/team/parser",
                "escapes",
                "https://github.com/team/parser/pull/42",
            ),
        )

    def test_codex_detects_structured_failures_without_guessing_from_output(
        self,
    ) -> None:
        for response, expected in (
            ({"exit_code": 1}, "tool_error"),
            ({"isError": True}, "tool_error"),
            ({"exit_code": 0}, "tool_done"),
            ({"stdout": "Exit code: 1"}, "tool_done"),
            ("Exit code: 1", "tool_done"),
        ):
            self.run_hook(
                "codex",
                "PostToolUse",
                {
                    "session_id": "worker",
                    "turn_id": "turn",
                    "tool_use_id": "call",
                    "tool_response": response,
                },
            )
            packet = self.receive()
            self.assertEqual(packet["event"], expected)
            self.assertEqual(packet["call"], "call")

    def test_failure_output_is_absent_from_routed_metadata(self) -> None:
        output = (
            "Preparing dependencies\n" * 1000
            + "FAIL Tests/auth.test.ts login rejects expired token\nAssertionError: expected 401 received 200"
        )
        self.run_hook(
            "codex",
            "PostToolUse",
            {
                "session_id": "worker",
                "cwd": "/src/backend",
                "tool_name": "exec_command",
                "tool_input": {"cmd": "pnpm test auth"},
                "tool_response": {"exit_code": 1, "output": output},
            },
        )
        packet = self.receive()
        self.assertEqual(packet["repo"], "/src/backend")
        self.assertEqual(packet["tool_name"], "exec_command")
        self.assertEqual(packet["event"], "tool_error")
        self.assertEqual(packet["tool"], "pnpm test auth")
        self.assertNotIn("detail", packet)
        self.assertNotIn("error_signature", packet)
        self.assertNotIn("AssertionError", json.dumps(packet))
        self.assertNotIn("progress", packet)

    def test_successful_reads_and_patches_do_not_supply_progress(self) -> None:
        for name in (
            "Read",
            "read_file",
            "functions.read_file",
            "mcp__files__read_file",
            "Edit",
            "apply_patch",
        ):
            self.run_hook(
                "codex",
                "PostToolUse",
                {
                    "session_id": "worker",
                    "tool_name": name,
                    "tool_response": {
                        "output": "file body or diff patch",
                        "exit_code": 0,
                    },
                },
            )
            packet = self.receive()
            self.assertEqual(packet["event"], "tool_done")
            self.assertNotIn("progress", packet)

    def test_assistant_messages_supply_progress_for_successful_tools(self) -> None:
        for code in (0, 1):
            self.run_hook(
                "codex",
                "PostToolUse",
                {
                    "session_id": "worker",
                    "last_assistant_message": "Checking authentication",
                    "tool_response": {"exit_code": code, "stdout": "private output"},
                },
            )
            packet = self.receive()
            if code == 0:
                self.assertEqual(packet["progress"], "Checking authentication")
            else:
                self.assertNotIn("progress", packet)
            self.assertNotIn("private output", json.dumps(packet))

    def test_failed_content_blocks_are_absent_from_events(self) -> None:
        self.run_hook(
            "codex",
            "PostToolUse",
            {
                "session_id": "worker",
                "tool_name": "build",
                "tool_response": {
                    "isError": True,
                    "content": [
                        {"type": "text", "text": "Building targets\n" * 100},
                        {
                            "type": "text",
                            "text": "src/main.swift:42: error: cannot find Widget in scope",
                        },
                    ],
                },
            },
        )
        packet = self.receive()
        self.assertEqual(packet["outcome"], "error")
        self.assertNotIn("Widget", json.dumps(packet))

    def test_cursor_subagent_aliases_link_child_tools(self) -> None:
        sessions = Sessions(time.monotonic)
        self.run_hook(
            "cursor",
            "subagentStart",
            {
                "conversation_id": "parent",
                "subagent_id": "agent",
                "child_conversation_id": "child",
                "tool_call_id": "spawn-call",
            },
        )
        packet = self.receive()
        self.assertEqual(packet["session"], "child")
        self.assertEqual(packet["parent"], "parent")
        self.assertEqual(packet["aliases"], ["agent", "child"])
        self.assertEqual(packet["parent_call"], "spawn-call")
        self.assertIsNotNone(sessions.handle(packet))
        self.run_hook(
            "cursor",
            "preToolUse",
            {
                "conversation_id": "child",
                "generation_id": "generation",
                "tool_use_id": "nested-call",
                "parent_tool_call_id": "spawn-call",
                "tool_name": "Shell",
            },
        )
        packet = self.receive()
        self.assertEqual(
            (packet["session"], packet["turn"], packet["call"], packet["parent_call"]),
            ("child", "generation", "nested-call", "spawn-call"),
        )
        self.assertIsNotNone(sessions.handle(packet))
        self.run_hook(
            "cursor",
            "postToolUse",
            {
                "conversation_id": "agent",
                "generation_id": "generation",
                "tool_use_id": "nested-call",
            },
        )
        self.assertIsNotNone(sessions.handle(self.receive()))
        self.assertEqual(set(sessions.by_id), {"cursor:child"})
        child = sessions.by_id["cursor:child"]
        self.assertEqual(child.parent, "cursor:parent")
        self.assertEqual(child.turn, "generation")
        self.assertFalse(child.calls)

    def test_cursor_tool_output_and_subagent_summary_update_context(self) -> None:
        self.run_hook(
            "cursor",
            "postToolUse",
            {
                "conversation_id": "worker",
                "tool_use_id": "call",
                "tool_output": json.dumps(
                    {"exitCode": 0, "stdout": "All tests passed"}
                ),
            },
        )
        self.assertNotIn("progress", self.receive())
        self.run_hook(
            "cursor",
            "subagentStart",
            {
                "conversation_id": "parent",
                "subagent_id": "child",
                "task": "Explore authentication",
                "tool_call_id": "spawn",
            },
        )
        self.assertEqual(self.receive()["goal"], "Explore authentication")
        self.run_hook(
            "cursor",
            "subagentStop",
            {
                "conversation_id": "parent",
                "subagent_id": "child",
                "status": "error",
                "summary": "Authentication tests failed",
            },
        )
        packet = self.receive()
        self.assertEqual(
            (packet["event"], packet["outcome"]), ("subagent_done", "error")
        )
        self.assertNotIn("progress", packet)

    def test_interrupted_tools_finish_without_an_error_alert(self) -> None:
        for harness, event, key in (
            ("claude", "PostToolUseFailure", "session_id"),
            ("cursor", "postToolUseFailure", "conversation_id"),
        ):
            self.run_hook(
                harness,
                event,
                {key: "worker", "tool_use_id": "call", "is_interrupt": True},
            )
            packet = self.receive()
            self.assertEqual(
                (packet["event"], packet["outcome"]), ("tool_done", "cancelled")
            )

    def test_cursor_stop_distinguishes_failure_and_cancellation(self) -> None:
        for status, event, outcome in (
            ("completed", "turn_done", "success"),
            ("error", "turn_error", "error"),
            ("aborted", "interrupt", "cancelled"),
        ):
            result = self.run_hook(
                "cursor", "stop", {"conversation_id": "worker", "status": status}
            )
            self.assertEqual(json.loads(result.stdout), {})
            packet = self.receive()
            self.assertEqual((packet["event"], packet["outcome"]), (event, outcome))

    def test_waiting_input_keeps_tool_identity_without_starting_work(self) -> None:
        for harness, event, session_key in (
            ("claude", "PreToolUse", "session_id"),
            ("codex", "PreToolUse", "session_id"),
            ("cursor", "preToolUse", "conversation_id"),
        ):
            self.run_hook(
                harness,
                event,
                {
                    session_key: "worker",
                    "tool_name": "request_user_input",
                    "tool_use_id": "question",
                },
            )
            packet = self.receive()
            self.assertEqual((packet["event"], packet["call"]), ("waiting", "question"))

    def test_imported_claude_hooks_do_not_duplicate_cursor_events(self) -> None:
        self.run_hook(
            "claude", "PreToolUse", {"session_id": "worker", "cursor_version": "2.0"}
        )
        self.assert_quiet()

    def test_invalid_payloads_do_not_affect_agent_decisions(self) -> None:
        for raw in ("bad json", "[]", "{}"):
            result = self.run_hook("claude", "PreToolUse", raw=raw)
            self.assertEqual((result.returncode, json.loads(result.stdout)), (0, {}))
        self.assert_quiet()

    def test_cursor_prompt_response_allows_submission(self) -> None:
        result = self.run_hook(
            "cursor",
            "beforeSubmitPrompt",
            {"conversation_id": "worker", "prompt": "goal"},
        )
        self.assertEqual(json.loads(result.stdout), {"continue": True})
        self.assertEqual(self.receive()["goal"], "goal")

    def test_examples_use_supported_native_events(self) -> None:
        module = runpy.run_path(str(HOOK))
        for harness, filename in (
            ("claude", "claude-settings.json"),
            ("codex", "codex-hooks.json"),
            ("cursor", "cursor-hooks.json"),
        ):
            config = json.loads((ROOT / "examples" / filename).read_text())
            self.assertEqual(set(config["hooks"]), set(module["EVENTS"][harness]))
            self.assertNotIn("--speak", json.dumps(config))

    def test_wrapper_preserves_piped_stdin(self) -> None:
        result = subprocess.run(
            [str(WRAPPER), "/bin/cat"],
            input="piped command input\n",
            env=self.environment,
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(
            (result.returncode, result.stdout), (0, "piped command input\n")
        )
        for expected in ("intent", "tool_start", "tool_done", "session_end"):
            self.assertEqual(self.receive()["event"], expected)

    def test_wrapper_terminates_its_command_and_reports_cancellation(self) -> None:
        process = subprocess.Popen(
            [str(WRAPPER), sys.executable, "-c", "import time; time.sleep(30)"],
            env=self.environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            intent, start = self.receive(), self.receive()

            process.terminate()
            process.communicate(timeout=10)
        finally:
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=10)
        interrupt, end = self.receive(), self.receive()

        self.assertEqual(intent["event"], "intent")
        self.assertEqual(start["event"], "tool_start")
        self.assertEqual(process.returncode, 143)
        self.assertEqual(interrupt["event"], "interrupt")
        self.assertEqual(end["event"], "session_end")
        self.assertEqual(end["outcome"], "cancelled")

    def test_wrapper_preserves_output_status_and_unique_session(self) -> None:
        sessions = []
        for code in (0, 7):
            result = subprocess.run(
                [
                    str(WRAPPER),
                    sys.executable,
                    "-c",
                    f"import sys; print('command output'); sys.exit({code})",
                ],
                env=self.environment,
                text=True,
                capture_output=True,
                timeout=10,
                check=False,
            )
            self.assertEqual(
                (result.returncode, result.stdout), (code, "command output\n")
            )
            intent, start, finish, end = (self.receive() for _ in range(4))
            self.assertEqual(intent["event"], "intent")
            self.assertEqual(start["event"], "tool_start")
            self.assertEqual(
                finish["event"], "tool_done" if code == 0 else "tool_error"
            )
            self.assertEqual(intent["session"], start["session"])
            self.assertEqual(finish["session"], start["session"])
            self.assertEqual(end["event"], "session_end")
            self.assertEqual(end["outcome"], "success" if code == 0 else "error")
            sessions.append(start["session"])
        self.assertNotEqual(*sessions)


if __name__ == "__main__":
    unittest.main()
