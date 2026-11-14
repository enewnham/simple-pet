import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pet_daemon.config import Config
from pet_daemon.notifications import Notifications, NotifyConfig, bounded, run_command
from pet_daemon.runtime import HookEvent


class NotificationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name)
        self.now = 100.0
        self.output = self.path / "received.json"
        self.command = (
            sys.executable,
            "-c",
            f"import sys,pathlib; pathlib.Path({str(self.output)!r}).write_bytes(sys.stdin.buffer.read())",
        )
        self.config = NotifyConfig(command=self.command)
        self.queue = Notifications(
            self.config, self.path / "outbox.sqlite3", lambda: self.now
        )

    async def asyncTearDown(self) -> None:
        await self.queue.close()
        self.directory.cleanup()

    def hook(self, **values: object) -> HookEvent:
        return HookEvent.model_validate(
            {"source": "codex", "session": "one", "event": "turn_done", **values}
        )

    def count(self) -> int:
        return self.queue.db.execute("SELECT count(*) FROM outbox").fetchone()[0]

    async def test_delivers_redacted_event_with_session_metadata(self) -> None:
        self.queue.ingest(
            self.hook(
                event="session_start",
                repo="/work/project",
                zmx_session="work",
                tab_title="Review",
            )
        )
        self.queue.ingest(
            self.hook(progress="Done api_key=secret-value " + "🐈" * 1000)
        )
        await self.queue.deliver()
        payload = json.loads(self.output.read_text())
        self.assertEqual(payload["cwd"], "/work/project")
        self.assertEqual(payload["zmx_session"], "work")
        self.assertEqual(payload["event"], "turn_done")
        self.assertNotIn("secret-value", self.output.read_text())
        self.assertLessEqual(len(payload["last_assistant_message"].encode()), 2048)
        self.assertEqual(self.count(), 0)

    async def test_filters_require_matching_patterns_labels_and_attention(self) -> None:
        self.queue.config = NotifyConfig(
            command=self.command,
            filter={
                "cwd": "/work/*",
                "zmx_session": "build-*",
                "labels": {"owner": "maintainer"},
                "needs_attention": True,
            },
        )
        base = {
            "repo": "/work/repo",
            "zmx_session": "build-one",
            "labels": {"owner": "maintainer"},
        }
        self.queue.ingest(self.hook(**base, progress="All done"))
        self.queue.ingest(self.hook(**{**base, "repo": "/other"}, progress="Failed"))
        self.queue.ingest(
            self.hook(**{**base, "labels": {"owner": "someone"}}, progress="Failed")
        )
        self.queue.ingest(
            self.hook(**{**base, "zmx_session": "other"}, progress="Failed")
        )
        self.assertEqual(self.count(), 0)
        self.queue.ingest(self.hook(**base, event="waiting", waiting_kind="permission"))
        self.assertEqual(self.count(), 1)
        self.queue.ingest(self.hook(**base, session="two", progress="Build failed"))
        self.assertEqual(self.count(), 2)

    async def test_debounce_survives_restart_and_is_per_session(self) -> None:
        self.queue.ingest(self.hook())
        await self.queue.close()
        self.queue = Notifications(
            self.config, self.path / "outbox.sqlite3", lambda: self.now
        )
        self.queue.ingest(self.hook())
        self.queue.ingest(self.hook(session="two"))
        self.assertEqual(self.count(), 2)
        self.now += 11
        self.queue.ingest(self.hook())
        self.assertEqual(self.count(), 3)

    async def test_rules_match_either_owner_anywhere_or_attention_in_checkout(
        self,
    ) -> None:
        self.queue.config = NotifyConfig(
            command=self.command,
            events=("turn_done", "turn_error", "session_end", "permission_pending"),
            rules=(
                {"labels": {"owner": "reviewer"}},
                {"cwd": "/Users/me/sc/*", "needs_attention": True},
            ),
        )
        cases = (
            {"repo": "/Users/me/.dotfiles", "labels": {"owner": "reviewer"}},
            {"repo": "/Users/me/sc/project", "progress": "Build failed"},
            {
                "repo": "/Users/me/sc/other",
                "event": "waiting",
                "waiting_kind": "permission",
            },
            {"repo": "/Users/me/sc/project", "progress": "Finished"},
            {"repo": "/Users/me/.dotfiles", "progress": "Build failed"},
        )
        for index, values in enumerate(cases):
            self.queue.ingest(self.hook(session=str(index), **values))
        payloads = self.queue.db.execute("SELECT payload FROM outbox").fetchall()
        self.assertEqual(
            [json.loads(row[0])["session"] for row in payloads], ["0", "1", "2"]
        )

    async def test_mark_rules_match_either_blocked_or_focus_with_all_tag_patterns(
        self,
    ) -> None:
        self.queue.config = NotifyConfig(
            command=self.command,
            rules=(
                {"marks": {"status": "blocked"}},
                {"marks": {"priority": "focus", "tags": ["release-*", "review"]}},
            ),
        )
        marks: dict[str, dict[str, object]] = {
            "blocked": {"status": "blocked"},
            "focus": {"priority": "focus", "tags": ["release-next", "review"]},
            "missing-tag": {"priority": "focus", "tags": ["release-next"]},
            "background": {
                "priority": "background",
                "tags": ["release-next", "review"],
            },
        }
        self.queue.mark_lookup = lambda hook: marks.get(hook.session, {})
        for session in ("blocked", "focus", "missing-tag", "background", "unmarked"):
            self.queue.ingest(self.hook(session=session))
        payloads = [
            json.loads(row[0])
            for row in self.queue.db.execute("SELECT payload FROM outbox")
        ]
        self.assertEqual(
            [payload["session"] for payload in payloads], ["blocked", "focus"]
        )
        self.assertEqual(payloads[1]["workstream_marks"], marks["focus"])

    async def test_mark_lookup_uses_remembered_checkout_and_updated_marks(self) -> None:
        self.queue.config = NotifyConfig(
            command=self.command,
            filter={"marks": {"priority": "focus"}},
        )
        observed = []
        mark: dict[str, object] = {"priority": "background"}

        def lookup(hook: HookEvent) -> dict[str, object]:
            observed.append(hook.repo)
            return mark

        self.queue.mark_lookup = lookup
        self.queue.ingest(self.hook(event="session_start", repo="/work/project"))
        self.queue.ingest(self.hook())
        self.assertEqual(self.count(), 0)
        mark["priority"] = "focus"
        self.queue.ingest(self.hook())
        self.assertEqual(observed, ["/work/project", "/work/project"])
        self.assertEqual(self.count(), 1)

    async def test_mark_payload_limits_tags_and_excludes_freeform_mark_text(
        self,
    ) -> None:
        self.queue.mark_lookup = lambda hook: {
            "status": "active",
            "priority": "focus",
            "tags": ["password=secret-value"] * 40,
            "note": "private note",
            "goal": "private goal",
            "title": "private title",
        }
        self.queue.ingest(self.hook())
        await self.queue.deliver()
        marks = json.loads(self.output.read_text())["workstream_marks"]
        self.assertEqual(set(marks), {"status", "priority", "tags"})
        self.assertEqual(len(marks["tags"]), 32)
        self.assertNotIn("secret-value", str(marks))

    async def test_rules_replace_single_filter_and_respect_event_selection(
        self,
    ) -> None:
        self.queue.config = NotifyConfig(
            command=self.command,
            events=("turn_error",),
            filter={"cwd": "/excluded/*"},
            rules=({"labels": {"owner": "reviewer"}},),
        )
        self.queue.ingest(self.hook(labels={"owner": "reviewer"}))
        self.assertEqual(self.count(), 0)
        self.queue.ingest(self.hook(event="turn_error", labels={"owner": "reviewer"}))
        self.assertEqual(self.count(), 1)

    async def test_empty_rules_match_no_events(self) -> None:
        self.queue.config = NotifyConfig(command=self.command, rules=())
        self.queue.ingest(self.hook())
        self.assertEqual(self.count(), 0)

    async def test_retries_after_backoff_and_restart(self) -> None:
        self.queue.config = NotifyConfig(
            command=(sys.executable, "-c", "raise SystemExit(1)")
        )
        self.queue.ingest(self.hook())
        await self.queue.deliver()
        self.assertEqual(self.count(), 1)
        await self.queue.close()
        self.queue = Notifications(
            self.config, self.path / "outbox.sqlite3", lambda: self.now
        )
        await self.queue.deliver()
        self.assertFalse(self.output.exists())
        self.now += 3600
        await self.queue.deliver()
        self.assertTrue(self.output.exists())
        self.assertEqual(self.count(), 0)

    async def test_timeout_keeps_event_for_retry(self) -> None:
        self.queue.config = NotifyConfig(
            command=(sys.executable, "-c", "import time; time.sleep(60)"),
            timeout_seconds=0.05,
        )
        self.queue.ingest(self.hook())
        await asyncio.wait_for(self.queue.deliver(), 2)
        self.assertEqual(self.count(), 1)
        self.assertEqual(
            self.queue.db.execute("SELECT attempts FROM outbox").fetchone()[0], 1
        )

    async def test_outbox_and_retries_are_bounded(self) -> None:
        self.queue.config = NotifyConfig(
            command=("/nonexistent/pet-notify-command",), max_events=2, max_attempts=1
        )
        for session in ("one", "two", "three"):
            self.queue.ingest(self.hook(session=session))
        self.assertEqual(self.count(), 2)
        await self.queue.deliver()
        self.assertEqual(self.count(), 1)

    async def test_auto_approved_permission_check_never_delivers(self) -> None:
        self.queue.ingest(
            self.hook(
                event="waiting",
                waiting_kind="permission_check",
                call="request",
                turn="turn",
            )
        )
        await self.queue.deliver()
        self.assertFalse(self.output.exists())
        self.now += 2
        self.queue.ingest(self.hook(event="tool_done", call="request", turn="turn"))
        self.now += 10
        await self.queue.deliver()
        self.assertEqual(self.count(), 0)
        self.assertFalse(self.output.exists())

    async def test_pending_permission_delivers_after_grace_with_original_elapsed_time(
        self,
    ) -> None:
        self.queue.ingest(
            self.hook(
                event="waiting",
                waiting_kind="permission_check",
                call="request",
                capture_at=98,
            )
        )
        self.now += 4
        await self.queue.deliver()
        self.assertFalse(self.output.exists())
        self.queue.ingest(
            self.hook(
                event="waiting",
                waiting_kind="permission_check",
                call="request",
                capture_at=104,
            )
        )
        self.now += 1
        await self.queue.deliver()
        payload = json.loads(self.output.read_text())
        self.assertTrue(payload["needs_human"])
        self.assertEqual(
            payload["permission_confirmation"], "still_pending_after_grace"
        )
        self.assertEqual(payload["timestamp"], 98)
        self.assertEqual(payload["received_at"], 100)
        self.assertEqual(payload["elapsed_seconds"], 7)
        self.assertEqual(payload["delivery_attempt_at"], 105)
        self.assertEqual(payload["delivery_attempt"], 1)
        self.assertFalse(any(name.startswith("_") for name in payload))

    async def test_explicit_permission_upgrade_keeps_original_grace_and_timestamp(
        self,
    ) -> None:
        self.queue.ingest(
            self.hook(
                event="waiting",
                waiting_kind="permission_check",
                call="request",
                capture_at=98,
            )
        )
        self.now += 2
        self.queue.ingest(
            self.hook(
                event="waiting",
                waiting_kind="permission",
                call="request",
                capture_at=102,
            )
        )
        self.now += 3
        await self.queue.deliver()
        payload = json.loads(self.output.read_text())
        self.assertEqual(
            payload["permission_confirmation"], "explicit_permission_prompt"
        )
        self.assertEqual(payload["timestamp"], 98)
        self.assertEqual(payload["elapsed_seconds"], 7)

    async def test_unrelated_tool_completion_preserves_permission_wait(self) -> None:
        self.queue.ingest(
            self.hook(
                event="waiting", waiting_kind="permission", call="request", turn="turn"
            )
        )
        self.queue.ingest(self.hook(event="tool_done", call="other", turn="turn"))
        self.now += 5
        await self.queue.deliver()
        payload = json.loads(self.output.read_text())
        self.assertTrue(payload["needs_human"])
        self.assertEqual(
            payload["permission_confirmation"], "explicit_permission_prompt"
        )

    async def test_new_permission_request_replaces_old_inside_debounce(self) -> None:
        self.queue.config = self.config.model_copy(update={"debounce_seconds": 60})
        self.queue.ingest(self.hook())
        self.now += 1
        self.queue.ingest(
            self.hook(event="waiting", waiting_kind="permission_check", call="old")
        )
        self.now += 1
        self.queue.ingest(
            self.hook(event="waiting", waiting_kind="permission_check", call="new")
        )
        self.assertEqual(self.count(), 2)
        await self.queue.deliver()
        first = json.loads(self.output.read_text())
        self.assertEqual(first["event"], "turn_done")
        self.now += 5
        await self.queue.deliver()
        permission = json.loads(self.output.read_text())
        self.assertEqual(permission["event"], "permission_pending")
        self.assertEqual(permission["timestamp"], 102)
        self.assertEqual(self.count(), 0)

    async def test_authoritative_pending_callback_is_checked_on_every_attempt(
        self,
    ) -> None:
        pending = True
        requests = []

        def is_pending(hook: HookEvent) -> bool:
            requests.append((hook.session, hook.call, hook.turn))
            return pending

        self.queue.permission_pending = is_pending
        self.queue.config = self.config.model_copy(
            update={"command": ("/missing/notifier",)}
        )
        self.queue.ingest(
            self.hook(
                event="waiting",
                waiting_kind="permission_check",
                call="request",
                turn="turn",
            )
        )
        self.now += 5
        await self.queue.deliver()
        self.assertEqual(self.count(), 1)
        pending = False
        self.now += 1
        await self.queue.deliver()
        self.assertEqual(self.count(), 0)
        self.assertEqual(requests, [("one", "request", "turn")] * 3)

    async def test_ignored_permission_hook_cannot_replace_a_runtime_confirmed_wait(
        self,
    ) -> None:
        self.queue.permission_pending = lambda hook: hook.call == "current"
        self.queue.ingest(
            self.hook(event="waiting", waiting_kind="permission_check", call="current")
        )
        self.queue.ingest(
            self.hook(event="waiting", waiting_kind="permission_check", call="stale")
        )
        self.queue.ingest(self.hook(event="tool_done", call="current"))
        self.assertEqual(self.count(), 1)
        self.now += 5
        await self.queue.deliver()
        self.assertEqual(
            json.loads(self.output.read_text())["event"], "permission_pending"
        )

    async def test_resolved_permission_removes_retries_and_resume_cannot_be_debounced(
        self,
    ) -> None:
        self.queue.config = self.config.model_copy(
            update={"command": ("/missing/notifier",)}
        )
        self.queue.ingest(
            self.hook(event="waiting", waiting_kind="permission_check", call="request")
        )
        self.now += 5
        await self.queue.deliver()
        self.assertEqual(self.count(), 1)
        self.queue.ingest(self.hook(event="tool_start", call="request"))
        self.assertEqual(self.count(), 0)
        self.now += 1
        self.queue.config = self.config
        self.queue.ingest(
            self.hook(event="waiting", waiting_kind="permission_check", call="second")
        )
        self.now += 5
        await self.queue.deliver()
        self.assertTrue(self.output.exists())

    async def test_unconfirmed_permission_does_not_survive_daemon_restart(self) -> None:
        self.queue.ingest(
            self.hook(event="waiting", waiting_kind="permission_check", call="request")
        )
        await self.queue.close()
        self.queue = Notifications(
            self.config, self.path / "outbox.sqlite3", lambda: self.now
        )
        self.now += 5
        await self.queue.deliver()
        self.assertEqual(self.count(), 0)
        self.assertFalse(self.output.exists())

    async def test_new_events_precede_retries_and_retry_elapsed_time_increases(
        self,
    ) -> None:
        deliveries = []

        async def send(
            command: tuple[str, ...], payload: bytes, timeout: float
        ) -> bool:
            event = json.loads(payload)
            deliveries.append(event)
            return event["session"] != "one" or len(deliveries) == 3

        self.queue.ingest(self.hook(capture_at=90))
        with patch("pet_daemon.notifications.run_command", side_effect=send):
            await self.queue.deliver()
            self.now += 2
            self.queue.ingest(self.hook(session="two"))
            await self.queue.deliver()
            await self.queue.deliver()
        self.assertEqual(
            [event["session"] for event in deliveries], ["one", "two", "one"]
        )
        self.assertEqual([event["delivery_attempt"] for event in deliveries], [1, 1, 2])
        self.assertEqual(
            [event["elapsed_seconds"] for event in deliveries], [10, 0, 12]
        )
        self.assertEqual(self.count(), 0)

    async def test_retry_delays_start_at_one_second_and_cap_at_fifteen(self) -> None:
        self.queue.config = self.config.model_copy(
            update={"command": ("/missing/notifier",)}
        )
        self.queue.ingest(self.hook())
        waits = []
        for _ in range(7):
            await self.queue.deliver()
            due = self.queue.db.execute("SELECT due FROM outbox").fetchone()[0]
            waits.append(due - self.now)
            self.now = due
        self.assertEqual(waits, [1, 2, 4, 8, 15, 15, 15])

    async def test_command_receives_minimal_environment(self) -> None:
        command = (
            sys.executable,
            "-c",
            "import os,sys; sys.exit(0 if set(os.environ) <= {'PATH', 'LC_CTYPE', '__CF_USER_TEXT_ENCODING'} else 1)",
        )
        self.assertTrue(await run_command(command, b"{}", 2))

    def test_redacts_common_credentials_before_truncating(self) -> None:
        for secret in (
            "password=hunter2",
            '"api_key": "secret-value"',
            "Authorization: Bearer abcdef",
            "sk-abcdefghijklmnop",
            "ghp_abcdefghijklmnop",
            "https://user:password@example.com",
            "-----BEGIN PRIVATE KEY-----\nabc\n-----END PRIVATE KEY-----",
        ):
            self.assertNotIn(secret, bounded(secret))
        self.assertNotIn("abc", bounded("-----BEGIN PRIVATE KEY-----" + "abc" * 1000))

    def test_redacts_quoted_credentials(self) -> None:
        self.assertNotIn("secret-value", bounded('"api_key": "secret-value"'))

    async def test_session_end_retains_last_assistant_message(self) -> None:
        self.queue.ingest(self.hook(event="progress", progress="Build finished"))
        self.queue.ingest(self.hook(event="tool_done", progress="Tool output"))
        self.queue.ingest(self.hook(event="session_end"))
        await self.queue.deliver()
        self.assertEqual(
            json.loads(self.output.read_text())["last_assistant_message"],
            "Build finished",
        )

    def test_notifications_are_disabled_by_default(self) -> None:
        self.assertIsNone(Config(pet_dir=self.path).notify)
