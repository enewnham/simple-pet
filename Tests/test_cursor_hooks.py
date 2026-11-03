import json
import os
import socket
import subprocess
import tempfile
import time
import unittest
from collections.abc import Mapping
from pathlib import Path

from pet_daemon.cli.hook import update
from pet_daemon.domain import SessionSource
from pet_daemon.identity import SessionIdentity
from pet_daemon.runtime import HookEvent
from pet_daemon.sessions.tracking import Sessions
from pet_daemon.sources.hooks import normalize_hook
from pet_daemon.storage import Memory
from pet_daemon.transport import MAX_DATAGRAM_BYTES, SOCKET_BUFFER_BYTES
from pet_daemon.work.models import IntentEvidence, ProgressEvidence
from Tests.fixtures import require

ROOT = Path(__file__).resolve().parents[1]
LAUNCH = "3656cf28-3db5-41a5-9dff-83a792dea2a8"
RUNTIME = f"runtime:{LAUNCH}"
CONVERSATION = "57b7c005-e26f-4e6d-ab5d-74c6426a9e5f"
FIRST_TURN = "0f28389c-ea08-4c43-bf69-83a4c9809129"
SECOND_TURN = "47fff5fe-a6e1-4271-916b-f546ac12122c"


class CursorHookTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory(prefix="pet-cursor-", dir="/tmp")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.socket_path = str(self.root / "daemon.sock")
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self.addCleanup(self.listener.close)
        self.listener.setsockopt(
            socket.SOL_SOCKET, socket.SO_RCVBUF, SOCKET_BUFFER_BYTES
        )
        self.listener.bind(self.socket_path)
        self.listener.settimeout(5)
        self.environment = dict(os.environ, PET_DAEMON_SOCKET=self.socket_path)
        self.environment.pop("PET_CURSOR_SESSION_ID", None)
        self.now = time.time()
        self.sessions = Sessions(lambda: self.now, wall_clock=lambda: self.now)
        self.memory = Memory(
            self.root / "memory.sqlite3", "cursor", clock=lambda: self.now
        )
        self.addCleanup(lambda: self.memory.close())

    def hook(self, event: str, payload: Mapping[str, object]) -> HookEvent:
        result = subprocess.run(
            [str(ROOT / "scripts/pet-hook"), "cursor", event],
            input=json.dumps(payload),
            env=self.environment,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        response = json.loads(result.stdout)
        if event == "sessionStart":
            self.assertEqual(response, {"env": {"PET_CURSOR_SESSION_ID": RUNTIME}})
            self.environment.update(response["env"])
        elif event == "beforeSubmitPrompt":
            self.assertEqual(response, {"continue": True})
        packet = json.loads(self.listener.recv(MAX_DATAGRAM_BYTES + 1))
        return require(normalize_hook(packet))

    def ingest(self, hook: HookEvent) -> bool:
        self.now += 1
        result = self.sessions.apply(hook)
        if result is None:
            return False
        with self.memory.transaction():
            self.memory.work.record_session(result.observation)
            self.memory.observe_session_context(result.observation)
        return True

    def test_resumed_cli_routes_both_turns_and_preserves_stale_turn_rejection(
        self,
    ) -> None:
        workspace = str(self.root / "workspace")
        self.assertTrue(
            self.ingest(
                self.hook(
                    "sessionStart",
                    {
                        "conversation_id": LAUNCH,
                        "session_id": LAUNCH,
                        "generation_id": LAUNCH,
                        "workspace_roots": [workspace],
                    },
                )
            )
        )
        request = "Add stack navigation.\n\n```lua\nkeep_source_text()\n```"
        for turn, prompt in (
            (FIRST_TURN, request),
            (SECOND_TURN, "Check the refresh timing"),
        ):
            self.assertTrue(
                self.ingest(
                    self.hook(
                        "beforeSubmitPrompt",
                        {
                            "conversation_id": CONVERSATION,
                            "session_id": CONVERSATION,
                            "generation_id": turn,
                            "workspace_roots": [workspace],
                            "prompt": prompt,
                        },
                    )
                )
            )
            self.assertTrue(
                self.ingest(
                    self.hook(
                        "preToolUse",
                        {
                            "conversation_id": LAUNCH,
                            "session_id": LAUNCH,
                            "generation_id": turn,
                            "workspace_roots": [workspace],
                            "tool_use_id": f"call-{turn}",
                            "tool_name": "Shell",
                            "tool_input": {"command": "git status"},
                        },
                    )
                )
            )
            if turn == SECOND_TURN:
                self.assertFalse(
                    self.ingest(
                        self.hook(
                            "afterAgentThought",
                            {
                                "conversation_id": CONVERSATION,
                                "generation_id": f"{FIRST_TURN}-7-5n53",
                                "text": "Stale first-turn report",
                            },
                        )
                    )
                )
                report = "Checking refresh timing.\n    preserve_indentation()"
                thought = self.hook(
                    "afterAgentThought",
                    {
                        "conversation_id": CONVERSATION,
                        "generation_id": f"{SECOND_TURN}-0-5an2",
                        "text": report,
                    },
                )
                self.assertEqual(thought.turn, SECOND_TURN)
                self.assertTrue(self.ingest(thought))
            self.assertTrue(
                self.ingest(
                    self.hook(
                        "postToolUse",
                        {
                            "conversation_id": LAUNCH,
                            "generation_id": turn,
                            "tool_use_id": f"call-{turn}",
                            "tool_name": "Shell",
                            "tool_output": {"exit_code": 0},
                        },
                    )
                )
            )
            self.assertTrue(
                self.ingest(
                    self.hook(
                        "stop",
                        {
                            "conversation_id": CONVERSATION,
                            "generation_id": turn,
                            "status": "completed",
                        },
                    )
                )
            )
        self.assertEqual(len(self.sessions.by_id), 1)
        session = next(iter(self.sessions.by_id.values()))
        self.assertEqual(session.turn, SECOND_TURN)
        self.assertFalse(session.calls)
        self.assertEqual(session.repo, workspace)
        identity = SessionIdentity(source=SessionSource.CURSOR, native_id=CONVERSATION)
        reports = self.memory.narrative_reports((identity,))
        intents = [
            item.payload.text
            for item in reports
            if isinstance(item.payload, IntentEvidence)
        ]
        progress = [
            item.payload.text
            for item in reports
            if isinstance(item.payload, ProgressEvidence)
        ]
        self.assertEqual(intents, [request, "Check the refresh timing"])
        self.assertIn(report, progress)
        self.assertNotIn("Stale first-turn report", progress)

    def test_switching_resumed_conversations_preserves_separate_history_after_reopen(
        self,
    ) -> None:
        self.assertTrue(
            self.ingest(
                self.hook(
                    "sessionStart",
                    {
                        "conversation_id": LAUNCH,
                        "session_id": LAUNCH,
                        "generation_id": LAUNCH,
                    },
                )
            )
        )
        other = "c7aee38d-5c7f-400b-911a-e82e35d8a119"
        for conversation, turn, request in (
            (LAUNCH, FIRST_TURN, "Start launcher conversation"),
            (CONVERSATION, SECOND_TURN, "Inspect stack navigation"),
            (other, "4cb2cfec-d25a-4174-b48f-61a643b55b18", "Review renderer movement"),
        ):
            self.assertTrue(
                self.ingest(
                    self.hook(
                        "beforeSubmitPrompt",
                        {
                            "conversation_id": conversation,
                            "generation_id": turn,
                            "prompt": request,
                        },
                    )
                )
            )
            self.assertTrue(
                self.ingest(
                    self.hook(
                        "preToolUse",
                        {
                            "conversation_id": LAUNCH,
                            "generation_id": turn,
                            "tool_use_id": f"call-{turn}",
                            "tool_name": "Shell",
                            "tool_input": {"command": "git status"},
                        },
                    )
                )
            )
            self.assertTrue(
                self.ingest(
                    self.hook(
                        "postToolUse",
                        {
                            "conversation_id": LAUNCH,
                            "generation_id": turn,
                            "tool_use_id": f"call-{turn}",
                            "tool_output": {"exit_code": 0},
                        },
                    )
                )
            )
            self.assertTrue(
                self.ingest(
                    self.hook(
                        "stop",
                        {
                            "conversation_id": conversation,
                            "generation_id": turn,
                            "status": "completed",
                        },
                    )
                )
            )
        first = SessionIdentity(source=SessionSource.CURSOR, native_id=CONVERSATION)
        second = SessionIdentity(source=SessionSource.CURSOR, native_id=other)
        launcher = SessionIdentity(source=SessionSource.CURSOR, native_id=LAUNCH)
        runtime = SessionIdentity(source=SessionSource.CURSOR, native_id=RUNTIME)
        self.assertEqual(len(self.sessions.by_id), 3)
        self.assertNotEqual(
            self.sessions.aliases.get(first, first),
            self.sessions.aliases.get(second, second),
        )
        self.assertEqual(
            self.sessions.aliases[runtime], self.sessions.aliases.get(second, second)
        )
        self.memory.close()
        self.memory = Memory(
            self.root / "memory.sqlite3", "cursor", clock=lambda: self.now
        )
        original_reports = self.memory.narrative_reports((launcher,))
        self.assertEqual(
            [
                r.payload.text
                for r in original_reports
                if isinstance(r.payload, IntentEvidence)
            ],
            ["Start launcher conversation"],
        )
        first_reports = self.memory.narrative_reports((first,))
        second_reports = self.memory.narrative_reports((second,))
        self.assertEqual(
            [
                r.payload.text
                for r in first_reports
                if isinstance(r.payload, IntentEvidence)
            ],
            ["Inspect stack navigation"],
        )
        self.assertEqual(
            [
                r.payload.text
                for r in second_reports
                if isinstance(r.payload, IntentEvidence)
            ],
            ["Review renderer movement"],
        )
        self.assertEqual(
            require(self.memory.context(first)).initial_request,
            "Inspect stack navigation",
        )
        self.assertEqual(
            require(self.memory.context(second)).initial_request,
            "Review renderer movement",
        )

    def test_resumed_parent_routes_child_tools_from_launcher_callbacks(self) -> None:
        events = (
            ("sessionStart", {"conversation_id": LAUNCH, "session_id": LAUNCH}),
            (
                "beforeSubmitPrompt",
                {
                    "conversation_id": CONVERSATION,
                    "generation_id": FIRST_TURN,
                    "prompt": "Inspect stack navigation",
                },
            ),
            (
                "subagentStart",
                {
                    "conversation_id": CONVERSATION,
                    "parent_conversation_id": CONVERSATION,
                    "subagent_id": "child",
                    "tool_call_id": "spawn",
                    "task": "Check refresh timing",
                },
            ),
            (
                "preToolUse",
                {
                    "conversation_id": LAUNCH,
                    "generation_id": FIRST_TURN,
                    "parent_tool_call_id": "spawn",
                    "tool_use_id": "nested",
                    "tool_name": "Shell",
                },
            ),
        )
        for event, payload in events:
            self.assertTrue(self.ingest(self.hook(event, payload)))
        parent = SessionIdentity(source=SessionSource.CURSOR, native_id=CONVERSATION)
        child = SessionIdentity(source=SessionSource.CURSOR, native_id="child")
        self.assertEqual(tuple(self.sessions.by_id[child].calls), ("nested",))
        self.assertFalse(self.sessions.by_id[parent].calls)
        self.assertTrue(
            self.ingest(
                self.hook(
                    "postToolUse",
                    {
                        "conversation_id": LAUNCH,
                        "generation_id": FIRST_TURN,
                        "parent_tool_call_id": "spawn",
                        "tool_use_id": "nested",
                        "tool_output": {"exit_code": 0},
                    },
                )
            )
        )
        self.assertFalse(self.sessions.by_id[child].calls)
        self.assertEqual(self.sessions.by_id[child].parent, parent)

    def test_parent_alias_merge_preserves_child_tool_routing(self) -> None:
        for packet in (
            {"event": "session_start", "session": "canonical"},
            {"event": "intent", "session": "temporary", "goal": "Parent task"},
            {
                "event": "subagent_start",
                "session": "child",
                "parent": "temporary",
                "parent_call": "spawn",
                "goal": "Child task",
            },
            {
                "event": "session_start",
                "session": "canonical",
                "aliases": ["temporary"],
            },
            {
                "event": "tool_start",
                "session": "temporary",
                "parent_call": "spawn",
                "call": "nested",
                "tool": "git status",
            },
        ):
            hook = require(normalize_hook({"source": "cursor", **packet}))
            self.assertTrue(self.ingest(hook))
        parent = SessionIdentity(source=SessionSource.CURSOR, native_id="canonical")
        child = SessionIdentity(source=SessionSource.CURSOR, native_id="child")
        self.assertEqual(tuple(self.sessions.by_id[child].calls), ("nested",))
        self.assertFalse(self.sessions.by_id[parent].calls)
        self.assertEqual(self.sessions.by_id[child].parent, parent)

    def test_native_session_metadata_and_workspace_are_normalized_without_prose_changes(
        self,
    ) -> None:
        text = "Fix parser.\n    keep_indentation()"
        hook = require(
            update(
                "cursor",
                "beforeSubmitPrompt",
                {
                    "conversation_id": CONVERSATION,
                    "session_id": LAUNCH,
                    "generation_id": FIRST_TURN,
                    "prompt": text,
                    "workspace_roots": ["/work/project"],
                },
            )
        )
        self.assertEqual(hook.session, CONVERSATION)
        self.assertEqual(hook.aliases, (LAUNCH,))
        self.assertEqual(hook.repo, "/work/project")
        self.assertEqual(hook.goal, text)
        fallback = require(update("cursor", "sessionStart", {"session_id": LAUNCH}))
        self.assertEqual(fallback.session, RUNTIME)
        multiroot = require(
            update(
                "cursor",
                "beforeSubmitPrompt",
                {
                    "conversation_id": CONVERSATION,
                    "workspace_roots": ["/work/one", "/work/two"],
                    "prompt": text,
                },
            )
        )
        self.assertIsNone(multiroot.repo)
