import asyncio
import os
import subprocess
import sys
import unittest
from unittest.mock import patch

from pet_daemon.processes import (
    ProcessIdentity,
    ProcessTracker,
    discover_owner,
    probe,
    process_identity,
)
from pet_daemon.sessions import SessionResult, Sessions


class ProcessTests(unittest.IsolatedAsyncioTestCase):
    async def test_exited_owner_retires_work_without_an_exit_hook(self) -> None:
        child = await asyncio.create_subprocess_exec(
            sys.executable, "-c", "import time; time.sleep(30)"
        )
        try:
            owner = process_identity(child.pid)
            assert owner is not None
            sessions = Sessions(lambda: 1000)
            sessions.handle(
                {
                    "event": "intent",
                    "harness": "claude",
                    "session": "main",
                    "owner": owner.packet(),
                }
            )
            tracker = ProcessTracker(interval=0)
            sessions.apply_process_status(
                await tracker.refresh(sessions.process_owners())
            )
            self.assertTrue(sessions.active)
            self.assertEqual(sessions.snapshot().sessions[0].presence, "alive")
            child.terminate()
            await child.wait()
            sessions.apply_process_status(
                await tracker.refresh(sessions.process_owners())
            )
            self.assertFalse(sessions.active)
            self.assertEqual(sessions.lanes["claude:main"].outcome, "stale")
        finally:
            if child.returncode is None:
                child.kill()
                await child.wait()

    async def test_replaced_identity_retires_only_its_associated_session(self) -> None:
        owner = process_identity(os.getpid())
        assert owner is not None
        replaced = ProcessIdentity(owner.pid, "another-start")
        self.assertEqual(await asyncio.to_thread(probe, replaced), "replaced")
        sessions = Sessions(lambda: 1000)
        for name, identity in (("current", owner), ("old", replaced)):
            sessions.handle(
                {
                    "event": "intent",
                    "harness": "claude",
                    "session": name,
                    "owner": identity.packet(),
                }
            )
        tracker = ProcessTracker()
        sessions.apply_process_status(await tracker.refresh(sessions.process_owners()))
        self.assertTrue(sessions.lanes["claude:current"].active)
        self.assertFalse(sessions.lanes["claude:old"].active)
        self.assertEqual(await tracker.refresh(sessions.process_owners()), {})

    async def test_failed_probe_leaves_a_session_active_with_unknown_presence(
        self,
    ) -> None:
        owner = process_identity(os.getpid())
        assert owner is not None
        sessions = Sessions(lambda: 1000)
        sessions.handle(
            {
                "event": "intent",
                "harness": "claude",
                "session": "main",
                "owner": owner.packet(),
            }
        )
        with patch(
            "pet_daemon.processes.subprocess.run",
            side_effect=subprocess.TimeoutExpired("ps", 2),
        ):
            results = await ProcessTracker().refresh(sessions.process_owners())
        sessions.apply_process_status(results)
        self.assertTrue(sessions.active)
        self.assertEqual(sessions.lanes["claude:main"].presence, "unknown")

    def test_ancestry_crosses_shells_and_ignores_shared_hosts(self) -> None:
        def row(
            pid: int, parent: int, command: str
        ) -> tuple[ProcessIdentity, int, str, str]:
            return (ProcessIdentity(pid, f"started-{pid}"), parent, "S", command)

        rows = {
            30: row(30, 20, "sh -c pet-hook"),
            20: row(20, 10, "/usr/bin/claude"),
            10: row(10, 1, "/bin/zsh"),
        }
        with (
            patch("pet_daemon.processes._processes", return_value=rows),
            patch.dict(os.environ, {}, clear=True),
        ):
            self.assertEqual(discover_owner("claude", pid=30), rows[20][0])
            self.assertIsNone(discover_owner("cursor", pid=30))
            rows[20] = row(20, 10, "/usr/bin/codex app-server")
            self.assertIsNone(discover_owner("codex", pid=30))
            rows[20] = row(20, 10, "/usr/bin/codex exec task")
            self.assertEqual(discover_owner("codex", pid=30), rows[20][0])

    def test_explicit_wrapper_must_match_a_live_ancestor_identity(self) -> None:
        owner = ProcessIdentity(20, "wrapper-start")
        rows = {
            30: (ProcessIdentity(30, "hook-start"), 20, "S", "python hook"),
            20: (owner, 1, "S", "bash examples/pet-run"),
        }
        with patch("pet_daemon.processes._processes", return_value=rows):
            with patch.dict(
                os.environ, {"PET_OWNER_PID": "20", "PET_OWNER_STARTED": owner.started}
            ):
                self.assertEqual(discover_owner("cli", pid=30), owner)
            with patch.dict(
                os.environ, {"PET_OWNER_PID": "20", "PET_OWNER_STARTED": "wrong"}
            ):
                self.assertIsNone(discover_owner("cli", pid=30))
            with patch.dict(
                os.environ, {"PET_OWNER_PID": "40", "PET_OWNER_STARTED": owner.started}
            ):
                self.assertIsNone(discover_owner("cli", pid=30))


class FreshnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.now = 1000
        self.sessions = Sessions(lambda: self.now)

    def event(self, kind: str, **fields: object) -> SessionResult | None:
        return self.sessions.handle(
            {"event": kind, "harness": "claude", "session": "main", **fields}
        )

    def test_quiet_live_turn_permits_rest_and_resumes_when_activity_arrives(
        self,
    ) -> None:
        owner = process_identity(os.getpid())
        assert owner is not None
        self.event("intent", owner=owner.packet())
        self.sessions.apply_process_status({owner: "alive"})
        self.now += 300
        self.assertFalse(self.sessions.working)
        self.assertFalse(self.sessions.engaged)
        self.assertTrue(self.sessions.active)
        item = self.sessions.snapshot().sessions[0]
        self.assertTrue(item.activity_uncertain)
        self.assertEqual(item.presence, "alive")
        self.event("progress", progress="Still looking")
        self.assertTrue(self.sessions.working)
        self.assertTrue(self.sessions.engaged)

    def test_outstanding_tools_and_explicit_waiting_remain_engaged(self) -> None:
        self.event("intent")
        self.event("tool_start", call="long", tool="build")
        self.now += 600
        self.assertTrue(self.sessions.working)
        self.assertTrue(self.sessions.engaged)
        self.event("tool_done", call="long")
        self.event("waiting", call="answer")
        self.now += 600
        self.assertFalse(self.sessions.working)
        self.assertTrue(self.sessions.engaged)

    def test_owner_exit_retires_an_inherited_child_without_claiming_success(
        self,
    ) -> None:
        owner = process_identity(os.getpid())
        assert owner is not None
        self.event("intent", owner=owner.packet())
        self.sessions.handle(
            {
                "event": "subagent_start",
                "harness": "claude",
                "session": "child",
                "parent": "main",
            }
        )
        self.assertEqual(self.sessions.lanes["claude:child"].owner, owner)
        self.sessions.apply_process_status({owner: "exited"})
        self.assertFalse(self.sessions.active)
        self.assertFalse(self.sessions.clean_convergence)

    def test_ownerless_turn_still_expires_after_missing_terminal_signal(self) -> None:
        self.event("intent")
        self.now += 3600
        self.sessions.prune()
        self.assertFalse(self.sessions.active)
        self.assertEqual(self.sessions.lanes["claude:main"].outcome, "stale")
