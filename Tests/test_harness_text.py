import json
import unittest
from pathlib import Path

from pet_daemon.cli.hook import update
from pet_daemon.cli.notify import encode
from pet_daemon.domain import SessionSource
from pet_daemon.harness_text import clean_text
from pet_daemon.prose import prose_text
from pet_daemon.user_text import sentiment_excerpt
from Tests.fixtures import require

FIXTURES = Path(__file__).parent / "fixtures" / "harness_text"


class HarnessTextTests(unittest.TestCase):
    def test_claude_handback_preserves_findings_before_prose_pruning(self) -> None:
        text = (FIXTURES / "claude-handback.txt").read_text()
        hook = require(
            update(
                "claude",
                "UserPromptSubmit",
                {"session_id": "review", "prompt": text},
            )
        )
        self.assertEqual(hook.goal, text)
        goal = clean_text(hook.source, require(hook.goal))
        self.assertTrue(goal.startswith("## Recommendation\n"))
        self.assertIn("Leave `getById` as it is.", prose_text(goal))
        self.assertIn("ignore actor authorization", prose_text(goal))
        self.assertNotIn("agent-message", goal)
        self.assertNotIn("Subagent hand-back", goal)
        self.assertNotIn("permission laundering", goal)
        self.assertNotIn("Leave `getById`", sentiment_excerpt(text))
        packet = json.loads(require(encode(hook.packet())))
        self.assertEqual(packet["goal"], text)

    def test_claude_handback_preserves_code_indentation(self) -> None:
        text = (FIXTURES / "claude-handback.txt").read_text()
        start = text.index("The report follows:\n") + len("The report follows:\n")
        end = text.index("\n</agent-message>", start)
        text = (
            text[:start]
            + "  Checked the parser.\n      run_code()\n  Found a missing test."
            + text[end:]
        )
        cleaned = clean_text(SessionSource.CLAUDE, text)
        self.assertEqual(
            prose_text(cleaned),
            "Checked the parser.\n[omitted]\nFound a missing test.",
        )

    def test_claude_delivery_notification_has_no_work_prose(self) -> None:
        text = (FIXTURES / "claude-notification.txt").read_text()
        self.assertEqual(clean_text(SessionSource.CLAUDE, text), "")
        changed = text.replace("it is not repeated here", "the report follows")
        self.assertEqual(clean_text(SessionSource.CLAUDE, changed), changed)

    def test_claude_command_keeps_the_developers_arguments(self) -> None:
        text = (FIXTURES / "claude-command.txt").read_text()
        self.assertEqual(
            clean_text(SessionSource.CLAUDE, text),
            "I see a few bot-review comments, triage and check each of them, "
            "check in with me on any that need my attention",
        )

    def test_codex_routing_header_leaves_report_prose(self) -> None:
        body = "Implemented Kitty zmx integration. Added six passing tests."
        header = (
            "Message Type: FINAL_ANSWER\nTask name: /root\n"
            "Sender: /root/kitty_zmx\nPayload:\n"
        )
        for prefix in ("", "Agent message from /root/kitty_zmx:\n"):
            with self.subTest(prefix=prefix):
                text = prefix + header + body
                self.assertEqual(clean_text(SessionSource.CODEX, text), body)
                hook = require(
                    update(
                        "codex",
                        "Stop",
                        {"session_id": "review", "last_assistant_message": text},
                    )
                )
                self.assertEqual(hook.progress, text)

    def test_preserves_ordinary_prose_and_incomplete_envelopes(self) -> None:
        for text in (
            '<agent-message from=""> is a specific format. What about Claude?',
            "Message Type: FINAL_ANSWER is part of the format.",
            "Another Claude session sent a message:\n<agent-message",
            "Fix the parser.\n    Preserve indented code.",
            "",
        ):
            for source in SessionSource:
                with self.subTest(text=text, source=source):
                    self.assertEqual(clean_text(source, text), text)

    def test_harness_rules_only_apply_to_their_source(self) -> None:
        text = (FIXTURES / "claude-handback.txt").read_text()
        for source in (
            SessionSource.CODEX,
            SessionSource.CURSOR,
            SessionSource.COMMAND,
        ):
            with self.subTest(source=source):
                self.assertEqual(clean_text(source, text), text)
