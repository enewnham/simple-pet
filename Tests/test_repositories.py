import asyncio
from dataclasses import replace
import json
from pathlib import Path
import sqlite3
import subprocess
import tempfile
import unittest

from pet_daemon.memory import Memory
from pet_daemon.repositories import RepositoryTracker
from pet_daemon.repositories import git
from pet_daemon.repositories.github import PullRequest
from pet_daemon.repositories.tracker import RepositoryObservation
from pet_daemon.work import build_work


class RepositoryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "repo"
        self.path.mkdir()
        self.git("init", "-b", "feature")
        self.git("config", "user.email", "pet@example.test")
        self.git("config", "user.name", "Pet")
        (self.path / "tracked").write_text("first\n")
        self.git("add", ".")
        self.git("commit", "-m", "Initial")
        self.now = 1000
        self.pr = {"number": 1, "url": "https://example.test/pr/1", "state": "OPEN", "mergedAt": None,
                   "reviewDecision": "REVIEW_REQUIRED", "statusCheckRollup": []}
        self.fail = False
        self.memory = Memory(Path(self.temp.name) / "memory.db", "pet", clock=lambda: self.now)
        self.tracker = RepositoryTracker(self.memory, lambda: self.now, run=self.command)

    def git(self, *args):
        return subprocess.run(["git", *args], cwd=self.path, check=True, capture_output=True, text=True).stdout

    async def command(self, *args, **kwargs):
        if args[0] == "gh":
            if self.fail:
                raise RuntimeError("GitHub unavailable")
            return json.dumps([self.pr] if args[2] == "list" else self.pr)
        return await git.command(*args, **kwargs)

    async def refresh(self):
        self.tracker.tick()
        await asyncio.gather(*list(self.tracker.tasks.values()))

    async def asyncTearDown(self):
        await self.tracker.close()
        self.memory.close()
        self.temp.cleanup()

    async def test_local_changes_and_merged_pull_request_remain_separate(self):
        self.tracker.associate("codex:one", str(self.path))
        (self.path / "tracked").write_text("changed\n")
        (self.path / "new").write_text("new\n")
        (self.path / "staged").write_text("staged\n")
        self.git("add", "staged")
        await self.refresh()
        await self.refresh()
        item = self.tracker.snapshot()[0]
        self.assertEqual((item.local.unstaged, item.local.untracked), (1, 1))
        self.assertEqual(item.local.staged, 1)
        self.assertEqual(item.pr.state, "OPEN")
        self.pr["state"] = "MERGED"
        self.pr["mergedAt"] = "2026-10-17T00:00:00Z"
        self.now += 180
        await self.refresh()
        item = self.tracker.snapshot()[0]
        self.assertEqual(item.pr.state, "MERGED")
        self.assertTrue(item.pending)
        self.git("add", ".")
        self.git("commit", "-m", "Manual changes")
        self.now += 30
        await self.refresh()
        self.assertFalse(self.tracker.snapshot()[0].pending)

    async def test_unchanged_polling_preserves_recency_and_query_failure_preserves_evidence(self):
        self.tracker.associate("codex:one", str(self.path))
        await self.refresh()
        await self.refresh()
        original = self.tracker.snapshot()[0]
        self.now += 180
        await self.refresh()
        unchanged = self.tracker.snapshot()[0]
        self.assertEqual((original.changed_at, original.revision), (unchanged.changed_at, unchanged.revision))
        self.fail = True
        self.now += 180
        await self.refresh()
        failed = self.tracker.snapshot()[0]
        self.assertEqual(failed.pr, original.pr)
        self.assertEqual(failed.changed_at, original.changed_at)
        self.assertIn("unavailable", failed.github_error)

    async def test_associations_and_pending_pull_requests_survive_restart_and_branch_switch(self):
        self.tracker.associate("codex:one", str(self.path))
        self.tracker.associate("claude:two", str(self.path))
        await self.refresh()
        await self.refresh()
        self.git("switch", "-c", "another")
        self.now += 30
        await self.refresh()
        feature = next(item for item in self.tracker.snapshot() if item.branch == "feature")
        self.assertFalse(feature.current)
        self.assertTrue(feature.pending)
        await self.tracker.close()
        self.tracker = RepositoryTracker(self.memory, lambda: self.now, run=self.command)
        restored = next(item for item in self.tracker.snapshot() if item.branch == "feature")
        self.assertEqual(restored.sessions, ("claude:two", "codex:one"))
        self.assertEqual(restored.pr.number, 1)

    async def test_branch_switch_during_github_refresh_rejects_result(self):
        self.tracker.associate("codex:one", str(self.path))
        await self.refresh()
        entered, release = asyncio.Event(), asyncio.Event()
        async def delayed(*args, **kwargs):
            if args[0] == "gh":
                entered.set()
                await release.wait()
            return await self.command(*args, **kwargs)
        self.tracker.run = delayed
        self.tracker.tick()
        await entered.wait()
        self.git("switch", "-c", "another")
        release.set()
        await asyncio.gather(*list(self.tracker.tasks.values()))
        feature = self.tracker.snapshot()[0]
        self.assertIsNone(feature.pr)
        await self.refresh()
        self.assertFalse(next(item for item in self.tracker.snapshot() if item.branch == "feature").current)

    async def test_tracks_conflicts_and_locally_known_ahead_behind_counts(self):
        self.git("branch", "base")
        self.git("branch", "--set-upstream-to=base", "feature")
        (self.path / "tracked").write_text("feature change\n")
        self.git("add", ".")
        self.git("commit", "-m", "Feature change")
        self.git("switch", "base")
        (self.path / "tracked").write_text("base change\n")
        self.git("add", ".")
        self.git("commit", "-m", "Base change")
        self.git("switch", "feature")
        local = await git.observe(str(self.path))
        self.assertEqual((local.ahead, local.behind), (1, 1))
        result = subprocess.run(["git", "merge", "base"], cwd=self.path, capture_output=True)
        self.assertEqual(result.returncode, 1)
        local = await git.observe(str(self.path))
        self.assertEqual(local.conflicts, 1)
        self.assertTrue(local.pending)

    async def test_reused_branch_discovers_new_pull_request_and_preserves_merged_association(self):
        self.pr["state"] = "MERGED"
        self.pr["mergedAt"] = "2026-10-17T00:00:00Z"
        self.tracker.associate("codex:one", str(self.path))
        await self.refresh()
        await self.refresh()
        self.assertEqual(self.tracker.snapshot()[0].pr.state, "MERGED")
        self.pr = {**self.pr, "number": 2, "url": "https://example.test/pr/2", "state": "OPEN", "mergedAt": None}
        self.now += 180
        await self.refresh()
        current = next(item for item in self.tracker.snapshot() if item.current)
        archived = next(item for item in self.tracker.snapshot() if not item.current)
        self.assertEqual(current.pr.number, 2)
        self.assertTrue(current.pending)
        self.assertEqual(archived.pr.number, 1)
        self.assertEqual(archived.pr.state, "MERGED")
        self.assertFalse(archived.pending)

    async def test_local_branch_change_during_observation_preserves_previous_evidence(self):
        self.tracker.associate("codex:one", str(self.path))
        await self.refresh()
        await self.refresh()
        original = self.tracker.snapshot()[0]
        changed = False
        async def switching(*args, **kwargs):
            nonlocal changed
            output = await self.command(*args, **kwargs)
            if args[0:2] == ("git", "status") and not changed:
                self.git("switch", "-c", "another")
                changed = True
            return output
        self.tracker.run = switching
        self.now += 30
        await self.refresh()
        item = self.tracker.snapshot()[0]
        self.assertEqual(item.local, original.local)
        self.assertEqual(item.changed_at, original.changed_at)
        self.assertIn("changed during", item.local_error)

    async def test_clear_discards_observations_during_refresh(self):
        self.tracker.associate("codex:one", str(self.path))
        entered, release = asyncio.Event(), asyncio.Event()
        async def delayed(*args, **kwargs):
            entered.set()
            await release.wait()
            return await self.command(*args, **kwargs)
        self.tracker.run = delayed
        self.tracker.tick()
        await entered.wait()
        tasks = list(self.tracker.tasks.values())
        self.memory.clear()
        self.tracker.clear()
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.assertEqual(self.tracker.snapshot(), ())
        self.assertEqual(self.memory.load_repositories(), [])
        self.assertEqual(self.memory.repository_associations(), [])


class RepositoryMigrationTests(unittest.TestCase):
    def test_fresh_failed_checks_lead_attention_and_old_pending_work_follows_completion(self):
        pr = PullRequest(1, "https://example.test/pr/1", "OPEN", None, "", (("tests", "FAILURE"),))
        failed = RepositoryObservation("repo:failed", "/repo", "feature", pr=pr, changed_at=100000)
        stale = replace(failed, subject="repo:stale", changed_at=1)
        merged = replace(failed, subject="repo:merged", pr=replace(pr, state="MERGED"), changed_at=99999)
        work = build_work({"sessions": []}, (stale, merged, failed), now=100001)
        self.assertEqual([(item.id, item.priority) for item in work],
                         [("repo:failed", 0), ("repo:merged", 3), ("repo:stale", 4)])

    def test_version_four_upgrade_preserves_all_memory_and_adds_associations(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.db"
            memory = Memory(path, "pet", clock=lambda: 1000)
            memory.record_event({"event": "intent", "harness": "codex", "session": "one", "repo": "/repo", "goal": "Work"})
            memory.record_speech("A thought", "routine")
            memory.save_context("codex:one", "Work", "Summary", ["Work"])
            memory.close()
            with sqlite3.connect(path) as db:
                db.execute("PRAGMA user_version = 4")
                db.execute("DROP TABLE repository_associations")
                db.execute("DROP TABLE repositories")
            memory = Memory(path, "pet", clock=lambda: 1000)
            try:
                self.assertEqual(memory.recent()[0]["goal"], "Work")
                self.assertEqual(memory.speech()[0]["text"], "A thought")
                self.assertEqual(memory.context("codex:one")["session_context"], "Summary")
                self.assertEqual(memory.projects(["/repo"])[0]["goal"], "Work")
                memory.associate_repository("codex:one", "/repo")
                memory.associate_repository("codex:one", "/other")
                self.assertEqual(len(memory.repository_associations()), 2)
            finally:
                memory.close()
