import json
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch
from zoneinfo import ZoneInfo

from pydantic import ValidationError

from pet_daemon.config import NavigationHandler, NavigationHandlers
from pet_daemon.domain import (
    EvidenceKind,
    EvidenceSource,
    SessionSource,
    WorkSubjectKind,
)
from pet_daemon.identity import SessionIdentity
from pet_daemon.navigation import Navigator, Target, TargetKind, next_morning
from pet_daemon.work.evidence import CommentaryEvidence, ExecutionStatus
from pet_daemon.work.models import ReportContext
from pet_daemon.work.shared_work import WorkSubject
from pet_daemon.work.stories import WorkStory


class NavigationTests(unittest.IsolatedAsyncioTestCase):
    def test_offers_recorded_targets_without_probing_the_checkout(self) -> None:
        identity = SessionIdentity(
            source=SessionSource.CODEX, native_id="closed-session"
        )
        subject = WorkSubject(
            id=identity,
            kind=WorkSubjectKind.SESSION,
            changed_at=10,
            revision="report",
            story=WorkStory(
                id=identity,
                intent="Review the parser",
                revision="report",
                activity=(
                    CommentaryEvidence(
                        source=EvidenceSource.SOURCE,
                        kind=EvidenceKind.EXECUTION,
                        details=ExecutionStatus(
                            execution="turn ended",
                            reported_progress=None,
                            reported_execution_outcome=None,
                        ),
                        context=ReportContext(
                            checkout="/missing/recorded-checkout",
                            pull_request="https://github.com/team/repo/pull/3",
                        ),
                    ),
                ),
            ),
        )
        navigator = Navigator(NavigationHandlers())
        with patch.object(Path, "exists", side_effect=AssertionError("Path probe")):
            self.assertEqual(
                navigator.targets(subject),
                (
                    Target(TargetKind.WORKTREE, "/missing/recorded-checkout"),
                    Target(
                        TargetKind.PULL_REQUEST, "https://github.com/team/repo/pull/3"
                    ),
                ),
            )
        recalled = ReportContext(
            checkout="/recorded/earlier-checkout",
            pull_request="https://github.com/team/repo/pull/1",
        )
        self.assertEqual(
            navigator.targets(subject, recalled),
            (
                Target(TargetKind.WORKTREE, "/recorded/earlier-checkout"),
                Target(TargetKind.PULL_REQUEST, "https://github.com/team/repo/pull/1"),
            ),
        )
        navigator = Navigator(
            NavigationHandlers(
                session=NavigationHandler(uri_template="agent://{source}/{session_id}")
            )
        )
        session = navigator.targets(subject)[-1]
        self.assertEqual(session, Target(TargetKind.SESSION, "closed-session", "codex"))
        self.assertEqual(navigator.label(session), "View session")

    async def test_encodes_reserved_characters_in_uri_templates(self) -> None:
        navigator = Navigator(
            NavigationHandlers(
                worktree=NavigationHandler(uri_template="vscode://file{path}")
            )
        )
        process = AsyncMock()
        process.returncode = 0
        with patch("asyncio.create_subprocess_exec", return_value=process) as launch:
            result = await navigator.open(Target(TargetKind.WORKTREE, "/repo/a b?#&%"))
        self.assertTrue(result.success)
        self.assertEqual(
            launch.call_args.args,
            ("/usr/bin/open", "vscode://file/repo/a%20b%3F%23%26%25"),
        )

    async def test_passes_literal_arguments_and_target_json_to_executable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            captured = Path(directory) / "captured.json"
            script = "import json,sys; from pathlib import Path; Path(sys.argv[1]).write_text(json.dumps([sys.argv[2:],json.load(sys.stdin)]))"
            navigator = Navigator(
                NavigationHandlers(
                    worktree=NavigationHandler(
                        argv=(sys.executable, "-c", script, str(captured), "{target}")
                    )
                )
            )
            value = "/repo/$(touch NO_SUCH_FILE) `echo bad` ; &"
            result = await navigator.open(Target(TargetKind.WORKTREE, value))
            self.assertTrue(result.success)
            self.assertEqual(
                json.loads(captured.read_text()),
                [[value], {"kind": "worktree", "value": value, "source": ""}],
            )

    async def test_reports_handler_failure_without_launching_a_fallback(self) -> None:
        navigator = Navigator(
            NavigationHandlers(
                pull_request=NavigationHandler(argv=("/does/not/exist",))
            )
        )
        result = await navigator.open(
            Target(TargetKind.PULL_REQUEST, "https://example.com/pr/1")
        )
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)

    async def test_stops_a_handler_that_exceeds_its_timeout(self) -> None:
        navigator = Navigator(
            NavigationHandlers(
                worktree=NavigationHandler(
                    argv=(sys.executable, "-c", "import time; time.sleep(30)")
                ),
                timeout_seconds=0.05,
            )
        )
        result = await navigator.open(Target(TargetKind.WORKTREE, "/repo"))
        self.assertEqual(result.error, "Handler timed out")

    async def test_reports_a_nonzero_exit_status(self) -> None:
        navigator = Navigator(
            NavigationHandlers(
                worktree=NavigationHandler(
                    argv=(sys.executable, "-c", "raise SystemExit(7)")
                )
            )
        )
        result = await navigator.open(Target(TargetKind.WORKTREE, "/repo"))
        self.assertEqual(result.error, "Handler exited with status 7")

    def test_rejects_invalid_handler_configuration(self) -> None:
        for data in (
            {},
            {"argv": []},
            {"argv": ["open"], "uri_template": "editor://{target}"},
            {"uri_template": "{path}"},
            {"uri_template": "editor://{unknown}"},
            {"argv": ["open", "{target.__class__}"]},
            {"argv": ["open"], "unknown": True},
        ):
            with self.subTest(data=data), self.assertRaises(ValidationError):
                NavigationHandler.model_validate(data)

    def test_schedules_tomorrow_at_nine_across_daylight_saving_changes(self) -> None:
        timezone = ZoneInfo("America/New_York")
        for day, expected_hours in (
            (datetime(2026, 3, 7, 9, tzinfo=timezone), 23),
            (datetime(2026, 10, 31, 9, tzinfo=timezone), 25),
        ):
            timestamp = next_morning(day.timestamp(), timezone)
            tomorrow = datetime.fromtimestamp(timestamp, timezone)
            self.assertEqual((tomorrow.hour, tomorrow.minute), (9, 0))
            self.assertEqual(timestamp - day.timestamp(), expected_hours * 3600)
