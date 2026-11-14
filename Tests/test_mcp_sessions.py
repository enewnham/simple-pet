import os
import stat
import tempfile
import unittest
from pathlib import Path

from pet_daemon.domain import ProcessStatus, SessionSource
from pet_daemon.identity import SessionIdentity
from pet_daemon.mcp_sessions import SessionPublisher, read_snapshot, snapshot_path
from pet_daemon.runtime import WaitingKind
from Tests.fixtures import session_snapshot


class SessionSnapshotTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "memory.sqlite3"
        self.now = 1000.0
        self.wall = 10000.0
        self.publisher = SessionPublisher(
            self.path, "fixture", clock=lambda: self.now, wall_clock=lambda: self.wall
        )

    def test_publishes_live_state_and_wall_times_with_private_permissions(self) -> None:
        working = session_snapshot(
            id=SessionIdentity(source=SessionSource.CODEX, native_id="working"),
            repo="/Users/me/.dotfiles",
            goal="Fix shell startup",
            presence=ProcessStatus.ALIVE,
            owner_last_alive_seconds=2,
            turn_active=True,
            quiet_seconds=30,
        )
        permission = session_snapshot(
            id=SessionIdentity(source=SessionSource.CLAUDE, native_id="permission"),
            presence=ProcessStatus.ALIVE,
            owner_last_alive_seconds=0,
            turn_active=True,
            waiting_for_input=True,
            waiting_kind=WaitingKind.PERMISSION,
            waiting_prompt="Allow command?",
            waiting_tool="Bash",
        )
        question = permission.model_copy(
            update={
                "id": SessionIdentity(
                    source=SessionSource.CLAUDE, native_id="question"
                ),
                "waiting_kind": WaitingKind.QUESTION,
                "waiting_prompt": "Which branch?",
            }
        )
        self.publisher.publish((working, permission, question))
        snapshot = read_snapshot(self.path, "fixture", self.wall)
        assert snapshot is not None
        self.assertEqual(
            [session.state for session in snapshot.sessions],
            ["working", "waiting_on_permission", "idle"],
        )
        self.assertEqual(snapshot.sessions[0].last_activity_at, 9970)
        self.assertEqual(snapshot.sessions[0].presence_checked_at, 9998)
        self.assertEqual(snapshot.sessions[0].checkout, "/Users/me/.dotfiles")
        self.assertEqual(snapshot.sessions[1].waiting_prompt, "Allow command?")
        self.assertEqual(snapshot.observed_at, 10000)
        self.assertEqual(stat.S_IMODE(os.stat(snapshot_path(self.path)).st_mode), 0o600)

    def test_excludes_unverified_retired_and_expired_process_observations(self) -> None:
        live = session_snapshot(
            presence=ProcessStatus.ALIVE, owner_last_alive_seconds=0
        )
        candidates = (
            live.model_copy(update={"presence": ProcessStatus.UNKNOWN}),
            live.model_copy(update={"presence": ProcessStatus.EXITED}),
            live.model_copy(update={"retired": True}),
            live.model_copy(update={"owner_last_alive_seconds": 16}),
            live.model_copy(update={"owner_last_alive_seconds": None}),
        )
        self.publisher.publish(candidates)
        snapshot = read_snapshot(self.path, "fixture", self.wall)
        assert snapshot is not None
        self.assertEqual(snapshot.sessions, ())

    def test_heartbeat_is_throttled_and_snapshot_expires_without_daemon(self) -> None:
        self.publisher.publish(())
        self.wall += 2
        self.now += 2
        self.publisher.publish(())
        snapshot = read_snapshot(self.path, "fixture", self.wall)
        assert snapshot is not None
        self.assertEqual(snapshot.observed_at, 10000)
        self.wall += 4
        self.now += 4
        self.publisher.publish(())
        snapshot = read_snapshot(self.path, "fixture", self.wall)
        assert snapshot is not None
        self.assertEqual(snapshot.observed_at, 10006)
        self.assertIsNone(read_snapshot(self.path, "other", self.wall))
        self.assertIsNone(read_snapshot(self.path, "fixture", self.wall + 16))
        self.assertIsNone(read_snapshot(self.path, "fixture", self.wall - 1))
        self.publisher.close()
        self.assertFalse(snapshot_path(self.path).exists())
        self.assertIsNone(read_snapshot(self.path, "fixture", self.wall))

    def test_suspension_publishes_empty_sessions_and_preserves_status(self) -> None:
        live = session_snapshot(
            presence=ProcessStatus.ALIVE, owner_last_alive_seconds=0
        )
        self.publisher.publish((live,), suspended=True)
        snapshot = read_snapshot(self.path, "fixture", self.wall)
        assert snapshot is not None
        self.assertTrue(snapshot.suspended)
        self.assertEqual(snapshot.sessions, ())

    def test_fresh_heartbeat_does_not_extend_process_observation_freshness(
        self,
    ) -> None:
        live = session_snapshot(
            presence=ProcessStatus.ALIVE, owner_last_alive_seconds=10
        )
        self.publisher.publish((live,))
        snapshot = read_snapshot(self.path, "fixture", self.wall + 6)
        assert snapshot is not None
        self.assertEqual(snapshot.sessions, ())

    def test_close_preserves_another_publishers_snapshot(self) -> None:
        self.publisher.publish(())
        other = SessionPublisher(self.path, "fixture")
        other.close()
        self.assertIsNotNone(read_snapshot(self.path, "fixture", self.wall))
        other.publish(())
        self.publisher.close()
        self.assertTrue(snapshot_path(self.path).exists())
        other.close()
        self.assertFalse(snapshot_path(self.path).exists())

    def test_unreadable_snapshot_does_not_claim_live_sessions(self) -> None:
        snapshot_path(self.path).write_text("incomplete")
        self.assertIsNone(read_snapshot(self.path, "fixture", self.wall))
