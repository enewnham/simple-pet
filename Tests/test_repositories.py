import asyncio
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Never, Unpack
from unittest.mock import patch

from pet_daemon.sources import RepositoryTarget, RepositoryTracker, git, github
from pet_daemon.sources.git import CommandOptions, RunCommand
from pet_daemon.sources.github import GitHubPullRequest
from pet_daemon.sources.hooks import HookEvent, normalize_hook
from pet_daemon.work.observations import (
    RepositoryObservation,
    repository_observation_from_mapping,
)
from Tests.fixtures import StackFixture, SubprocessOptions


class RepositoryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "repo"
        self.path.mkdir()
        self.git("init", "-b", "feature")
        self.git("remote", "add", "origin", "git@example.test:team/repo.git")
        self.git("config", "user.email", "pet@example.test")
        self.git("config", "user.name", "Pet")
        (self.path / "tracked").write_text("first\n")
        self.git("add", ".")
        self.git("commit", "-m", "Initial")
        self.now = 1000
        self.pr: GitHubPullRequest = {
            "number": 1,
            "url": "https://example.test/team/repo/pull/1",
            "state": "OPEN",
            "mergedAt": None,
            "isDraft": False,
            "headRefOid": self.git("rev-parse", "HEAD").strip(),
            "headRefName": "feature",
            "headRepositoryOwner": {"login": "team"},
            "headRepository": {"name": "repo"},
            "title": "Improve parser",
            "body": "Handle quoted fields.",
            "baseRefName": "main",
            "baseRefOid": "base-head",
        }
        self.stacks = []
        self.related_prs = {}
        self.github_unavailable = False
        self.emitted = []
        self.tracker = RepositoryTracker(
            self.emitted.append, lambda: self.now, run=self.command
        )

    def restore_tracker(self, *, run: RunCommand | None = None) -> None:
        previous = self.tracker
        associations = [
            (session, path)
            for path, sessions in previous.associations.items()
            for session in sessions
        ]
        self.tracker = RepositoryTracker(
            self.emitted.append,
            lambda: self.now,
            run=run or self.command,
            checkpoints=previous.snapshot(),
            associations=associations,
            targets=tuple(previous.targets.values()),
        )

    def git(self, *args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=self.path, check=True, capture_output=True, text=True
        ).stdout

    async def command(self, *args: str, **kwargs: Unpack[CommandOptions]) -> str:
        if args[0] == "gh":
            if self.github_unavailable:
                raise RuntimeError("GitHub unavailable")
            if args[1] == "api" and "/stacks?pull_request=" in args[2]:
                return json.dumps(self.stacks)
            if args[1:3] == ("repo", "view"):
                return json.dumps({"isFork": False, "parent": None})
            return json.dumps(
                [self.pr]
                if args[2] == "list"
                else self.related_prs.get(args[3], self.pr)
            )
        return await git.command(*args, **kwargs)

    async def refresh(self) -> None:
        self.tracker.tick()
        await asyncio.gather(*list(self.tracker.tasks.values()))

    async def asyncTearDown(self) -> None:
        await self.tracker.close()
        self.temp.cleanup()

    async def test_local_changes_and_merged_pull_request_remain_separate(self) -> None:
        self.tracker.associate("codex:one", str(self.path))
        (self.path / "tracked").write_text("changed\n")
        (self.path / "new").write_text("new\n")
        (self.path / "staged").write_text("staged\n")
        self.git("add", "staged")
        await self.refresh()
        await self.refresh()
        item = self.tracker.snapshot()[0]
        assert item.local is not None
        self.assertEqual((item.local.unstaged, item.local.untracked), (1, 1))
        self.assertEqual(item.local.staged, 1)
        assert item.pr is not None
        self.assertEqual(item.pr.state, "OPEN")
        self.pr["state"] = "MERGED"
        self.pr["mergedAt"] = "2026-10-17T00:00:00Z"
        self.now += 180
        await self.refresh()
        item = self.tracker.snapshot()[0]
        assert item.pr is not None
        self.assertEqual(item.pr.state, "MERGED")
        self.assertTrue(item.pending)
        self.git("add", ".")
        self.git("commit", "-m", "Manual changes")
        self.now += 30
        await self.refresh()
        self.assertFalse(self.tracker.snapshot()[0].pending)

    async def test_unchanged_polling_preserves_recency_and_query_failure_preserves_evidence(
        self,
    ) -> None:
        self.tracker.associate("codex:one", str(self.path))
        await self.refresh()
        await self.refresh()
        original = self.tracker.snapshot()[0]
        self.now += 180
        await self.refresh()
        unchanged = self.tracker.snapshot()[0]
        self.assertEqual(
            (original.changed_at, original.revision),
            (unchanged.changed_at, unchanged.revision),
        )
        self.github_unavailable = True
        self.now += 180
        await self.refresh()
        failed = self.tracker.snapshot()[0]
        self.assertEqual(failed.pr, original.pr)
        self.assertEqual(failed.changed_at, original.changed_at)
        assert failed.github_error is not None
        self.assertIn("unavailable", failed.github_error)

    async def test_associations_and_pending_pull_requests_survive_restart_and_branch_switch(
        self,
    ) -> None:
        self.tracker.associate("codex:one", str(self.path))
        self.tracker.associate("claude:two", str(self.path))
        await self.refresh()
        await self.refresh()
        self.git("switch", "-c", "another")
        self.now += 30
        await self.refresh()
        feature = next(
            item for item in self.tracker.snapshot() if item.branch == "feature"
        )
        self.assertFalse(feature.current)
        self.assertTrue(feature.pending)
        await self.tracker.close()
        self.restore_tracker()
        restored = next(
            item for item in self.tracker.snapshot() if item.branch == "feature"
        )
        self.assertEqual(restored.sessions, ("claude:two", "codex:one"))
        assert restored.pr is not None
        self.assertEqual(restored.pr.number, 1)

    async def test_branch_switch_during_github_refresh_rejects_result(self) -> None:
        self.tracker.associate("codex:one", str(self.path))
        await self.refresh()
        entered, release = asyncio.Event(), asyncio.Event()

        async def delayed(*args: str, **kwargs: Unpack[CommandOptions]) -> str:
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
        self.assertFalse(
            next(
                item for item in self.tracker.snapshot() if item.branch == "feature"
            ).current
        )

    async def test_tracks_conflicts_and_locally_known_ahead_behind_counts(self) -> None:
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
        result = await asyncio.to_thread(
            subprocess.run,
            ["git", "merge", "base"],
            cwd=self.path,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 1)
        local = await git.observe(str(self.path))
        self.assertEqual(local.conflicts, 1)
        self.assertTrue(local.pending)

    async def test_reused_branch_discovers_new_pull_request_and_preserves_merged_association(
        self,
    ) -> None:
        self.pr["state"] = "MERGED"
        self.pr["mergedAt"] = "2026-10-17T00:00:00Z"
        self.tracker.associate("codex:one", str(self.path))
        await self.refresh()
        await self.refresh()
        pr = self.tracker.snapshot()[0].pr
        assert pr is not None
        self.assertEqual(pr.state, "MERGED")
        self.pr: GitHubPullRequest = {
            **self.pr,
            "number": 2,
            "url": "https://example.test/team/repo/pull/2",
            "state": "OPEN",
            "mergedAt": None,
        }
        self.now += 180
        await self.refresh()
        current = next(item for item in self.tracker.snapshot() if item.current)
        archived = next(item for item in self.tracker.snapshot() if not item.current)
        assert current.pr is not None
        self.assertEqual(current.pr.number, 2)
        self.assertTrue(current.pending)
        assert archived.pr is not None
        self.assertEqual(archived.pr.number, 1)
        self.assertEqual(archived.pr.state, "MERGED")
        self.assertFalse(archived.pending)

    async def test_reused_branch_binds_only_the_session_observed_at_the_new_head(
        self,
    ) -> None:
        self.tracker.associate("codex:one", str(self.path))
        await self.refresh()
        await self.refresh()
        self.assertEqual(self.tracker.snapshot()[0].pr_sessions, ("codex:one",))
        (self.path / "tracked").write_text("second\n")
        self.git("commit", "-am", "Second change")
        self.pr = {
            **self.pr,
            "number": 2,
            "url": "https://example.test/team/repo/pull/2",
            "headRefOid": self.git("rev-parse", "HEAD").strip(),
        }
        self.tracker.associate("claude:two", str(self.path))
        await self.refresh()
        pending = next(item for item in self.tracker.snapshot() if item.current)
        self.assertEqual(pending.pr_sessions, ("codex:one",))
        self.assertIn("claude:two", pending.sessions)
        await self.refresh()
        current = next(item for item in self.tracker.snapshot() if item.current)
        archived = next(item for item in self.tracker.snapshot() if not item.current)
        self.assertEqual(current.sessions, ("claude:two",))
        self.assertEqual(current.pr_sessions, ("claude:two",))
        self.assertEqual(archived.pr_sessions, ("codex:one",))
        assert archived.pr is not None
        self.assertEqual(archived.pr.url, "https://example.test/team/repo/pull/1")

    async def test_restored_path_sessions_do_not_claim_a_replacement_pull_request(
        self,
    ) -> None:
        self.tracker.associate("codex:one", str(self.path))
        await self.refresh()
        await self.refresh()
        await self.tracker.close()
        self.restore_tracker()
        (self.path / "tracked").write_text("second\n")
        self.git("commit", "-am", "Second change")
        self.pr = {
            **self.pr,
            "number": 2,
            "url": "https://example.test/team/repo/pull/2",
            "headRefOid": self.git("rev-parse", "HEAD").strip(),
        }
        self.now += 180
        await self.refresh()
        await self.refresh()
        current = next(item for item in self.tracker.snapshot() if item.current)
        archived = next(item for item in self.tracker.snapshot() if not item.current)
        self.assertEqual(current.sessions, ())
        self.assertEqual(current.pr_sessions, ())
        self.assertEqual(archived.pr_sessions, ("codex:one",))

    async def test_same_pull_request_keeps_its_session_when_the_head_advances(
        self,
    ) -> None:
        self.tracker.associate("codex:one", str(self.path))
        await self.refresh()
        await self.refresh()
        (self.path / "tracked").write_text("second\n")
        self.git("commit", "-am", "Second change")
        self.pr["headRefOid"] = self.git("rev-parse", "HEAD").strip()
        self.now += 180
        await self.refresh()
        await self.refresh()
        self.assertEqual(len(self.tracker.snapshot()), 1)
        self.assertEqual(self.tracker.snapshot()[0].pr_sessions, ("codex:one",))

    async def test_same_head_replacement_does_not_reuse_session_claims(self) -> None:
        self.tracker.associate("codex:one", str(self.path))
        await self.refresh()
        await self.refresh()
        self.tracker.associate("claude:two", str(self.path))
        self.pr = {
            **self.pr,
            "number": 2,
            "url": "https://example.test/team/repo/pull/2",
        }
        await self.refresh()
        await self.refresh()
        current = next(item for item in self.tracker.snapshot() if item.current)
        self.assertEqual(current.pr_sessions, ())
        self.assertEqual(current.sessions, ("claude:two",))
        self.now += 180
        await self.refresh()
        self.assertEqual(
            next(item for item in self.tracker.snapshot() if item.current).pr_sessions,
            (),
        )

    async def test_cached_mismatched_pull_request_cannot_confirm_a_session_claim(
        self,
    ) -> None:
        self.tracker.associate("codex:one", str(self.path))
        await self.refresh()
        await self.refresh()
        (self.path / "tracked").write_text("second\n")
        self.git("commit", "-am", "Second change")
        self.tracker.associate("claude:two", str(self.path))
        await self.refresh()
        await self.refresh()
        current = self.tracker.snapshot()[0]
        self.assertEqual(current.pr_sessions, ("codex:one",))
        self.assertIn("claude:two", current.sessions)
        self.github_unavailable = True
        self.now += 180
        await self.refresh()
        self.assertEqual(self.tracker.snapshot()[0].pr_sessions, ("codex:one",))

    async def test_absent_branch_pull_request_does_not_confirm_cached_session_claims(
        self,
    ) -> None:
        self.tracker.associate("codex:one", str(self.path))
        await self.refresh()
        await self.refresh()

        async def absent(*args: str, **kwargs: Unpack[CommandOptions]) -> str:
            if args[:3] == ("gh", "pr", "list"):
                return "[]"
            return await self.command(*args, **kwargs)

        self.tracker.run = absent
        self.tracker.associate("claude:two", str(self.path))
        await self.refresh()
        await self.refresh()
        current = self.tracker.snapshot()[0]
        self.assertEqual(current.pr_sessions, ("codex:one",))
        assert current.pr is not None
        self.assertEqual(current.pr.url, self.pr["url"])

    async def test_local_branch_change_during_observation_preserves_previous_evidence(
        self,
    ) -> None:
        self.tracker.associate("codex:one", str(self.path))
        await self.refresh()
        await self.refresh()
        original = self.tracker.snapshot()[0]
        changed = False

        async def switching(*args: str, **kwargs: Unpack[CommandOptions]) -> str:
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
        assert item.local_error is not None
        self.assertIn("changed during", item.local_error)

    async def test_clear_discards_observations_during_refresh(self) -> None:
        self.tracker.associate("codex:one", str(self.path))
        entered, release = asyncio.Event(), asyncio.Event()

        async def delayed(*args: str, **kwargs: Unpack[CommandOptions]) -> str:
            entered.set()
            await release.wait()
            return await self.command(*args, **kwargs)

        self.tracker.run = delayed
        self.tracker.tick()
        await entered.wait()
        tasks = list(self.tracker.tasks.values())
        self.tracker.clear()
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.assertEqual(self.tracker.snapshot(), ())
        self.assertEqual(self.tracker.associations, {})

    async def test_removed_worktree_stops_local_polling_after_restore(self) -> None:
        self.tracker.associate("codex:one", str(self.path))
        await self.refresh()
        await self.refresh()
        pr = self.tracker.snapshot()[0].pr
        assert pr is not None
        self.assertEqual(pr.state, "OPEN")
        self.tracker.track_artifact(RepositoryTarget("pull_request", self.pr["url"]))
        await self.tracker.close()
        shutil.rmtree(self.path)
        self.restore_tracker()
        await self.refresh()
        retained = self.tracker.snapshot()[0]
        self.assertFalse(retained.current)
        assert retained.pr is not None
        self.assertEqual(retained.pr.number, 1)
        self.assertEqual(self.tracker.associations, {})
        self.assertEqual(
            sum(bool(item.worktree) for item in self.tracker.snapshot()), 1
        )
        await self.tracker.close()
        self.restore_tracker()
        self.now += 360
        self.pr["state"] = "MERGED"
        await self.refresh()
        pr = next(
            item for item in self.tracker.snapshot() if item.subject.startswith("pr:")
        ).pr
        assert pr is not None
        self.assertEqual(pr.state, "MERGED")
        self.assertEqual(self.tracker.paths, {})

    async def test_git_failure_retains_an_existing_worktree_for_retry(self) -> None:
        self.tracker.associate("codex:one", str(self.path))
        await self.refresh()
        original = self.tracker.snapshot()[0]

        async def unavailable(*args: str, **kwargs: Unpack[CommandOptions]) -> Never:
            raise OSError("Temporary process failure")

        self.tracker.run = unavailable
        self.now += 30
        await self.refresh()
        self.assertEqual(self.tracker.snapshot()[0].local, original.local)
        local_error = self.tracker.snapshot()[0].local_error
        assert local_error is not None
        self.assertIn("Temporary", local_error)
        self.assertEqual(len(self.tracker.associations), 1)
        self.tracker.run = self.command
        self.now += 30
        await self.refresh()
        self.assertIsNone(self.tracker.snapshot()[0].local_error)

    async def test_pruned_worktree_retains_pull_requests_and_stops_local_polling(
        self,
    ) -> None:
        worktree = self.path.parent / "worktree"
        self.git("worktree", "add", "-b", "temporary", str(worktree))
        self.pr["headRefName"] = "temporary"
        self.tracker.associate("codex:one", str(worktree))
        await self.refresh()
        await self.refresh()
        await asyncio.to_thread(
            subprocess.run,
            ["git", "switch", "-c", "another"],
            cwd=worktree,
            check=True,
            capture_output=True,
        )
        self.now += 30
        await self.refresh()
        self.assertEqual(len(self.tracker.snapshot()), 2)
        self.tracker.track_artifact(RepositoryTarget("pull_request", self.pr["url"]))
        shutil.rmtree(worktree)
        self.git("worktree", "prune")
        self.now += 180
        await self.refresh()
        self.assertEqual(len(self.tracker.snapshot()), 3)
        self.assertTrue(all(not item.current for item in self.tracker.snapshot()))
        pr = next(item for item in self.tracker.snapshot() if item.pr).pr
        assert pr is not None
        self.assertEqual(pr.number, 1)
        self.assertEqual(len(self.tracker.snapshot()), 3)
        self.assertEqual(self.tracker.associations, {})
        self.assertEqual(self.tracker.paths, {})
        self.pr["state"] = "MERGED"
        self.now += 180
        await self.refresh()
        pr = next(
            item for item in self.tracker.snapshot() if item.subject.startswith("pr:")
        ).pr
        assert pr is not None
        self.assertEqual(pr.state, "MERGED")

    async def test_explicit_pull_request_polls_without_checkout(self) -> None:
        calls = []

        async def recording(*args: str, **kwargs: Unpack[CommandOptions]) -> str:
            calls.append((args, kwargs))
            return await self.command(*args, **kwargs)

        self.tracker.run = recording
        self.tracker.track_artifact(RepositoryTarget("pull_request", self.pr["url"]))
        await self.refresh()
        item = self.tracker.snapshot()[0]
        assert item.pr is not None
        self.assertEqual(item.pr.number, 1)
        self.assertEqual(item.repository, "example.test/team/repo")
        self.assertFalse(item.current)
        self.assertEqual(self.tracker.paths, {})
        self.assertEqual(calls[0][0][0:4], ("gh", "pr", "view", self.pr["url"]))
        self.assertIsNone(calls[0][1]["cwd"])

    async def test_explicit_branch_discovers_pull_request_without_checkout(
        self,
    ) -> None:
        self.tracker.track_artifact(
            RepositoryTarget("branch", "feature", "https://example.test/team/repo.git")
        )
        await self.refresh()
        item = self.tracker.snapshot()[0]
        assert item.pr is not None
        self.assertEqual(item.pr.head_repository, "example.test/team/repo")
        self.assertEqual(item.pr.head_branch, "feature")
        self.assertEqual(item.pr.head, self.git("rev-parse", "HEAD").strip())

    async def test_explicit_pull_request_target_survives_restore_without_checkout(
        self,
    ) -> None:
        self.tracker.track_artifact(RepositoryTarget("pull_request", self.pr["url"]))
        await self.refresh()
        await self.tracker.close()
        self.restore_tracker()
        self.now += 180
        self.pr["state"] = "MERGED"
        await self.refresh()
        pr = self.tracker.snapshot()[0].pr
        assert pr is not None
        self.assertEqual(pr.state, "MERGED")
        self.assertEqual(self.tracker.paths, {})

    async def test_new_session_receives_checkout_and_pull_request_after_fresh_poll(
        self,
    ) -> None:
        self.tracker.associate("codex:one", str(self.path))
        await self.refresh()
        await self.refresh()
        original = next(item for item in self.tracker.snapshot() if item.current)
        self.tracker.associate("claude:two", str(self.path))
        self.assertNotIn("claude:two", self.tracker.snapshot()[0].sessions)
        await self.refresh()
        updated = next(item for item in self.tracker.snapshot() if item.current)
        self.assertEqual(updated.sessions, ("claude:two", "codex:one"))
        self.assertEqual(
            (updated.revision, updated.changed_at),
            (original.revision, original.changed_at),
        )
        assert updated.pr is not None
        self.assertEqual(updated.pr.url, self.pr["url"])
        self.assertEqual(self.emitted[-1], updated)
        self.assertNotIn("claude:two", updated.pr_sessions)
        await self.refresh()
        confirmed = next(item for item in self.tracker.snapshot() if item.current)
        self.assertEqual(confirmed.pr_sessions, ("claude:two", "codex:one"))

    async def test_new_session_in_known_subdirectory_receives_root_after_fresh_poll(
        self,
    ) -> None:
        nested = self.path / "nested"
        nested.mkdir()
        self.tracker.associate("codex:one", str(nested))
        await self.refresh()
        await self.refresh()
        self.tracker.associate("claude:two", str(nested))
        await self.refresh()
        current = next(item for item in self.tracker.snapshot() if item.current)
        self.assertIn("claude:two", current.sessions)
        self.assertEqual(current.worktree, str(self.path.resolve()))

    async def test_new_session_after_branch_switch_never_inherits_cached_branch_or_pull_request(
        self,
    ) -> None:
        self.tracker.associate("codex:one", str(self.path))
        await self.refresh()
        await self.refresh()
        previous_url = self.pr["url"]
        self.related_prs[previous_url] = dict(self.pr)
        self.git("switch", "-c", "another")
        self.pr: GitHubPullRequest = {
            **self.pr,
            "number": 2,
            "url": "https://example.test/team/repo/pull/2",
            "headRefName": "another",
        }
        self.tracker.associate("claude:two", str(self.path))
        await self.refresh()
        await self.refresh()
        current = next(item for item in self.tracker.snapshot() if item.current)
        assert current.pr is not None
        self.assertEqual((current.branch, current.pr.url), ("another", self.pr["url"]))
        self.assertIn("claude:two", current.sessions)
        previous = next(
            item for item in self.tracker.snapshot() if item.branch == "feature"
        )
        self.assertNotIn("claude:two", previous.sessions)
        self.assertFalse(previous.current)
        assert previous.pr is not None
        self.assertEqual(previous.pr.url, previous_url)

    async def test_known_session_progress_claims_the_branch_at_the_next_normal_poll(
        self,
    ) -> None:
        self.tracker.associate("codex:one", str(self.path))
        self.tracker.associate("claude:two", str(self.path))
        await self.refresh()
        await self.refresh()
        self.git("switch", "-c", "another")
        self.pr = {
            **self.pr,
            "number": 2,
            "url": "https://example.test/team/repo/pull/2",
            "headRefName": "another",
        }
        self.now += 1
        self.tracker.associate("codex:one", str(self.path))
        await self.refresh()
        self.assertEqual(self.tracker.snapshot()[0].branch, "feature")
        self.now += 29
        await self.refresh()
        current = next(item for item in self.tracker.snapshot() if item.current)
        self.assertEqual(current.branch, "another")
        self.assertEqual(current.sessions, ("codex:one",))
        await self.refresh()
        current = next(item for item in self.tracker.snapshot() if item.current)
        self.assertEqual(current.pr_sessions, ("codex:one",))
        previous = next(item for item in self.tracker.snapshot() if not item.current)
        self.assertEqual(previous.pr_sessions, ("claude:two", "codex:one"))

    async def test_known_session_progress_claims_a_reused_branch_without_historical_sessions(
        self,
    ) -> None:
        self.tracker.associate("codex:one", str(self.path))
        self.tracker.associate("claude:two", str(self.path))
        await self.refresh()
        await self.refresh()
        (self.path / "tracked").write_text("second\n")
        self.git("commit", "-am", "Second change")
        self.pr = {
            **self.pr,
            "number": 2,
            "url": "https://example.test/team/repo/pull/2",
            "headRefOid": self.git("rev-parse", "HEAD").strip(),
        }
        self.tracker.associate("codex:one", str(self.path))
        self.now += 30
        await self.refresh()
        await self.refresh()
        current = next(item for item in self.tracker.snapshot() if item.current)
        self.assertEqual(current.sessions, ("codex:one",))
        self.assertEqual(current.pr_sessions, ("codex:one",))
        previous = next(item for item in self.tracker.snapshot() if not item.current)
        self.assertEqual(previous.pr_sessions, ("claude:two", "codex:one"))

    async def test_repeated_session_events_preserve_poll_interval_and_new_session_refreshes(
        self,
    ) -> None:
        self.tracker.associate("codex:one", str(self.path))
        await self.refresh()
        await self.refresh()
        polls = []

        async def counted(*args: str, **kwargs: Unpack[CommandOptions]) -> str:
            if args[:2] == ("git", "status"):
                polls.append(args)
            return await self.command(*args, **kwargs)

        self.tracker.run = counted
        for _ in range(3):
            self.now += 1
            self.tracker.associate("codex:one", str(self.path))
            await self.refresh()
        self.assertEqual(polls, [])
        attempted = self.tracker.snapshot()[0].github_attempted_at
        self.now += 27
        await self.refresh()
        current = self.tracker.snapshot()[0]
        self.assertEqual(current.github_attempted_at, attempted)
        self.assertEqual(current.session_heads, ())
        self.tracker.associate("claude:two", str(self.path))
        await self.refresh()
        self.assertEqual(len(polls), 2)
        self.assertIn(
            "claude:two",
            next(item for item in self.tracker.snapshot() if item.current).sessions,
        )

    async def test_exact_head_repository_filters_forks_and_ambiguous_open_requests(
        self,
    ) -> None:
        other_fork = {**self.pr, "number": 2, "headRepositoryOwner": {"login": "other"}}
        candidates = [self.pr, other_fork]

        async def listing(*args: str, **kwargs: Unpack[CommandOptions]) -> str:
            if args[0:3] == ("gh", "pr", "list"):
                return json.dumps(candidates)
            return await self.command(*args, **kwargs)

        observed = await github.observe(
            None, "feature", run=listing, repository="example.test/team/repo"
        )
        assert observed is not None
        self.assertEqual(observed.number, 1)
        candidates.append({**self.pr, "number": 3})
        with self.assertRaisesRegex(RuntimeError, "Multiple open"):
            await github.observe(
                None, "feature", run=listing, repository="example.test/team/repo"
            )

    async def test_incomplete_head_identity_keeps_branch_lookup_uncertain(self) -> None:
        self.pr["headRepository"] = None
        with self.assertRaisesRegex(RuntimeError, "head identity is unavailable"):
            await github.observe(
                None, "feature", run=self.command, repository="example.test/team/repo"
            )

    async def test_ambiguous_branch_refresh_preserves_known_pull_request_with_uncertainty(
        self,
    ) -> None:
        self.tracker.associate("codex:one", str(self.path))
        await self.refresh()
        await self.refresh()
        original = self.tracker.snapshot()[0]

        async def ambiguous(*args: str, **kwargs: Unpack[CommandOptions]) -> str:
            if args[0:3] == ("gh", "pr", "list"):
                return json.dumps([self.pr, {**self.pr, "number": 2}])
            return await self.command(*args, **kwargs)

        self.tracker.run = ambiguous
        self.now += 180
        await self.refresh()
        retained = self.tracker.snapshot()[0]
        self.assertEqual(retained.pr, original.pr)
        self.assertEqual(retained.github_checked_at, original.github_checked_at)
        assert retained.github_error is not None
        self.assertIn("Multiple open", retained.github_error)

    async def test_fork_discovers_upstream_pull_request_using_its_head_identity(
        self,
    ) -> None:
        self.pr["headRepositoryOwner"] = {"login": "fork-owner"}
        calls = []

        async def forked(*args: str, **kwargs: Unpack[CommandOptions]) -> str:
            calls.append(args)
            if args[0:3] == ("gh", "repo", "view"):
                return json.dumps(
                    {
                        "isFork": True,
                        "parent": {"owner": {"login": "team"}, "name": "repo"},
                    }
                )
            return await self.command(*args, **kwargs)

        observed = await github.observe(
            None, "feature", run=forked, repository="example.test/fork-owner/repo"
        )
        assert observed is not None
        self.assertEqual(observed.head_repository, "example.test/fork-owner/repo")
        listed = next(args for args in calls if args[1:3] == ("pr", "list"))
        self.assertEqual(listed[listed.index("--repo") + 1], "example.test/team/repo")

    async def test_pull_request_retains_exact_subject_and_base_without_inferring_body_links(
        self,
    ) -> None:
        self.pr["body"] = (
            "Parser changes. See https://example.test/team/repo/pull/42 for a related discussion."
        )
        observed = await github.observe(None, "", run=self.command, url=self.pr["url"])
        assert observed is not None
        self.assertEqual(
            (observed.title, observed.body), (self.pr["title"], self.pr["body"])
        )
        self.assertEqual(
            (observed.base_branch, observed.base_head), ("main", "base-head")
        )
        self.assertEqual(observed.related_links, ())
        self.assertIsNone(observed.related_links_error)

    def stack(self) -> StackFixture:
        return {
            "number": 8,
            "base": {"ref": "main"},
            "pull_requests": [
                {
                    "number": 1,
                    "state": "open",
                    "draft": False,
                    "merged_at": None,
                    "head": {"ref": "feature", "sha": self.pr["headRefOid"]},
                },
                {
                    "number": 2,
                    "state": "closed",
                    "draft": False,
                    "merged_at": "2026-10-17T00:00:00Z",
                    "head": {"ref": "base-layer", "sha": "base-layer-head"},
                },
                {
                    "number": 3,
                    "state": "open",
                    "draft": True,
                    "merged_at": None,
                    "head": {"ref": "next-layer", "sha": "next-layer-head"},
                },
            ],
        }

    async def test_native_stack_links_retain_exact_membership_and_individual_states(
        self,
    ) -> None:
        self.stacks = [self.stack()]
        observed = await github.observe(None, "", run=self.command, url=self.pr["url"])
        assert observed is not None
        self.assertEqual(
            [(link.ref, link.state, link.position) for link in observed.related_links],
            [
                ("https://example.test/team/repo/pull/2", "MERGED", 2),
                ("https://example.test/team/repo/pull/3", "OPEN", 3),
            ],
        )
        self.assertTrue(observed.related_links[1].is_draft)
        self.assertEqual(observed.related_links[1].head, "next-layer-head")
        self.assertTrue(
            all(
                link.source == "github_stack"
                and link.relationship == "stack_entry"
                and link.stack == 8
                for link in observed.related_links
            )
        )

    async def test_unsupported_stack_api_keeps_core_pull_request_evidence(self) -> None:
        async def unsupported(*args: str, **kwargs: Unpack[CommandOptions]) -> str:
            if args[0:2] == ("gh", "api") and "/stacks?pull_request=" in args[2]:
                raise RuntimeError("HTTP 404: Stacks are unavailable")
            return await self.command(*args, **kwargs)

        observed = await github.observe(None, "", run=unsupported, url=self.pr["url"])
        assert observed is not None
        self.assertEqual(
            (observed.title, observed.state, observed.head),
            (self.pr["title"], "OPEN", self.pr["headRefOid"]),
        )
        assert observed.related_links_error is not None
        self.assertIn("404", observed.related_links_error)
        self.assertEqual(observed.related_links, ())

    async def test_stack_membership_must_include_the_observed_pull_request(
        self,
    ) -> None:
        stack = self.stack()
        stack["pull_requests"] = stack["pull_requests"][1:]
        self.stacks = [stack]
        observed = await github.observe(None, "", run=self.command, url=self.pr["url"])
        assert observed is not None
        self.assertEqual(observed.related_links, ())

    async def test_explicit_stack_peer_target_skips_further_stack_metadata(
        self,
    ) -> None:
        stack = self.stack()
        stack["pull_requests"] = stack["pull_requests"][:2]
        self.stacks = [stack]
        peer_url = "https://example.test/team/repo/pull/2"
        self.related_prs[peer_url] = {
            **self.pr,
            "number": 2,
            "url": peer_url,
            "state": "MERGED",
            "mergedAt": "2026-10-17T00:00:00Z",
            "headRefName": "base-layer",
            "headRefOid": "base-layer-head",
            "title": "Add parser foundation",
        }
        calls = []

        async def recording(*args: str, **kwargs: Unpack[CommandOptions]) -> str:
            calls.append(args)
            return await self.command(*args, **kwargs)

        self.tracker.run = recording
        self.tracker.associate("codex:one", str(self.path))
        await self.refresh()
        await self.refresh()
        await self.refresh()
        root = next(item for item in self.tracker.snapshot() if item.current)
        self.assertEqual(len(self.tracker.snapshot()), 1)
        assert root.pr is not None
        self.assertEqual(root.pr.related_links[0].ref, peer_url)
        self.tracker.track_artifact(
            RepositoryTarget("pull_request", peer_url, include_links=False)
        )
        await self.refresh()
        peer = next(
            item
            for item in self.tracker.snapshot()
            if item.pr and item.pr.url == peer_url
        )
        root = next(item for item in self.tracker.snapshot() if item.current)
        assert root.pr is not None
        assert peer.pr is not None
        self.assertEqual((root.pr.state, peer.pr.state), ("OPEN", "MERGED"))
        self.assertEqual(peer.pr.title, "Add parser foundation")
        self.assertFalse(
            any(
                args[1] == "api" and args[2].endswith("pull_request=2")
                for args in calls
            )
        )
        self.assertTrue(any(args[1:4] == ("pr", "view", peer_url) for args in calls))
        await self.tracker.close()
        shutil.rmtree(self.path)
        self.restore_tracker(run=recording)
        self.now += 180
        await self.refresh()
        self.assertEqual(self.tracker.paths, {})
        peer = next(
            item
            for item in self.tracker.snapshot()
            if item.pr and item.pr.url == peer_url
        )
        assert peer.pr is not None
        self.assertEqual(peer.pr.title, "Add parser foundation")

    async def test_stack_lookup_failure_retains_known_links_without_advancing_change_recency(
        self,
    ) -> None:
        self.stacks = [self.stack()]
        self.tracker.associate("codex:one", str(self.path))
        await self.refresh()
        await self.refresh()
        original = next(item for item in self.tracker.snapshot() if item.current)

        async def unavailable(*args: str, **kwargs: Unpack[CommandOptions]) -> str:
            if args[0:2] == ("gh", "api") and "/stacks?pull_request=" in args[2]:
                raise RuntimeError("Stack lookup unavailable")
            return await self.command(*args, **kwargs)

        self.tracker.run = unavailable
        self.now += 180
        await self.refresh()
        retained = next(item for item in self.tracker.snapshot() if item.current)
        assert retained.pr is not None
        assert original.pr is not None
        self.assertEqual(retained.pr.related_links, original.pr.related_links)
        self.assertEqual(
            (retained.revision, retained.changed_at),
            (original.revision, original.changed_at),
        )
        assert retained.pr.related_links_error is not None
        self.assertIn("unavailable", retained.pr.related_links_error)
        self.assertGreater(retained.github_checked_at, original.github_checked_at)

    async def test_repository_without_hosted_identity_has_unknown_github_status(
        self,
    ) -> None:
        self.git("remote", "remove", "origin")
        self.tracker.associate("codex:one", str(self.path))
        await self.refresh()
        await self.refresh()
        item = self.tracker.snapshot()[0]
        self.assertTrue(item.repository.startswith("local:"))
        self.assertIsNone(item.pr)
        self.assertEqual(item.github_checked_at, 0)
        assert item.github_error is not None
        self.assertIn("identity is unavailable", item.github_error)

    async def test_remote_aliases_and_linked_worktrees_share_repository_identity(
        self,
    ) -> None:
        self.assertEqual(
            git.canonical_repository("ssh://git@EXAMPLE.test/Team/Repo.git/"),
            "example.test/team/repo",
        )
        self.assertEqual(
            git.canonical_repository("https://EXAMPLE.test/Team/Repo.git"),
            "example.test/team/repo",
        )
        self.git("remote", "remove", "origin")
        linked = self.path.parent / "linked"
        self.git("worktree", "add", "-b", "linked", str(linked))
        original = await git.observe(str(self.path))
        other = await git.observe(str(linked))
        self.assertEqual(original.repository, other.repository)
        self.assertEqual(
            original.repository, "local:" + str((self.path / ".git").resolve())
        )

    async def test_checkpoints_do_not_infer_repository_from_sessions(self) -> None:
        legacy = RepositoryObservation(
            "legacy", str(self.path), "feature", sessions=("codex:one",)
        )
        checkpoint = repository_observation_from_mapping(legacy.packet())
        await self.tracker.close()
        self.tracker = RepositoryTracker(
            self.emitted.append,
            lambda: self.now,
            run=self.command,
            checkpoints=(checkpoint,),
        )
        self.assertEqual(self.tracker.snapshot()[0].repository, "")
        self.assertEqual(self.tracker.paths, {})

    async def test_removed_subdirectory_keeps_its_existing_repository(self) -> None:
        nested = self.path / "nested"
        nested.mkdir()
        self.tracker.associate("codex:one", str(nested))
        await self.refresh()
        nested.rmdir()
        self.now += 30
        await self.refresh()
        self.assertEqual(self.tracker.snapshot()[0].worktree, str(self.path.resolve()))
        self.assertEqual(
            self.tracker.associations, {str(self.path.resolve()): {"codex:one"}}
        )
        (self.path / "tracked").write_text(
            "changed after the subdirectory disappeared\n"
        )
        self.now += 30
        await self.refresh()
        local = self.tracker.snapshot()[0].local
        assert local is not None
        self.assertEqual(local.unstaged, 1)

    async def test_cancelled_subprocess_startup_terminates_the_spawned_process(
        self,
    ) -> None:
        entered, release = asyncio.Event(), asyncio.Event()
        processes = []
        create = asyncio.create_subprocess_exec

        async def starting(
            *args: str, **kwargs: Unpack[SubprocessOptions]
        ) -> asyncio.subprocess.Process:
            process = await create(*args, **kwargs)
            processes.append(process)
            entered.set()
            await release.wait()
            return process

        try:
            with patch(
                "pet_daemon.sources.git.asyncio.create_subprocess_exec",
                side_effect=starting,
            ):
                task = asyncio.create_task(
                    git.command(sys.executable, "-c", "import time; time.sleep(60)")
                )
                await entered.wait()
                task.cancel()
                release.set()
                result = await asyncio.gather(task, return_exceptions=True)
            self.assertIsInstance(result[0], asyncio.CancelledError)
            self.assertIsNotNone(processes[0].returncode)
        finally:
            for process in processes:
                if process.returncode is None:
                    process.kill()
                    await process.wait()

    async def test_cancelled_command_terminates_its_subprocess(self) -> None:
        entered = asyncio.Event()
        processes = []
        create = asyncio.create_subprocess_exec

        async def started(
            *args: str, **kwargs: Unpack[SubprocessOptions]
        ) -> asyncio.subprocess.Process:
            process = await create(*args, **kwargs)
            processes.append(process)
            entered.set()
            return process

        with patch(
            "pet_daemon.sources.git.asyncio.create_subprocess_exec", side_effect=started
        ):
            task = asyncio.create_task(
                git.command(sys.executable, "-c", "import time; time.sleep(60)")
            )
            await entered.wait()
            task.cancel()
            result = await asyncio.gather(task, return_exceptions=True)
        self.assertIsInstance(result[0], asyncio.CancelledError)
        self.assertIsNotNone(processes[0].returncode)

    async def test_pause_cancels_collection_and_resume_refreshes_preserved_facts(
        self,
    ) -> None:
        self.tracker.associate("codex:one", str(self.path))
        await self.refresh()
        await self.refresh()
        previous = self.tracker.snapshot()[0]
        entered, cancelled = asyncio.Event(), asyncio.Event()

        async def delayed(*args: str, **kwargs: Unpack[CommandOptions]) -> Never:
            entered.set()
            try:
                await asyncio.Future[Never]()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        self.tracker.run = delayed
        self.now += 180
        self.tracker.tick()
        await entered.wait()
        tasks = list(self.tracker.tasks.values())
        self.tracker.pause()
        await asyncio.gather(*tasks, return_exceptions=True)
        await cancelled.wait()
        self.assertEqual(self.tracker.snapshot(), (previous,))
        self.assertIsNone(self.tracker.snapshot()[0].github_error)
        self.tracker.tick()
        self.assertEqual(self.tracker.tasks, {})
        self.tracker.run = self.command
        self.pr["state"] = "MERGED"
        self.tracker.resume()
        await self.refresh()
        pr = self.tracker.snapshot()[0].pr
        assert pr is not None
        self.assertEqual(pr.state, "MERGED")

    async def test_pause_discards_late_results_after_runner_suppresses_cancellation(
        self,
    ) -> None:
        self.tracker.associate("codex:one", str(self.path))
        await self.refresh()
        previous = self.tracker.snapshot()[0]
        entered = asyncio.Event()

        async def delayed(*args: str, **kwargs: Unpack[CommandOptions]) -> str:
            if args[0] == "gh" and not entered.is_set():
                entered.set()
                try:
                    await asyncio.Future()
                except asyncio.CancelledError:
                    pass
            return await self.command(*args, **kwargs)

        self.tracker.run = delayed
        self.tracker.tick()
        await entered.wait()
        tasks = list(self.tracker.tasks.values())
        self.tracker.pause()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.assertEqual(self.tracker.snapshot(), (previous,))
        self.assertEqual(self.emitted, [previous])

    async def test_retiring_checkout_target_stops_local_collection(self) -> None:
        target = RepositoryTarget(
            "branch", "feature", "example.test/team/repo", str(self.path)
        )
        self.tracker.reconcile_targets((target,))
        await self.refresh()
        self.assertTrue(any(item.current for item in self.tracker.snapshot()))
        self.tracker.reconcile_targets(())
        self.assertEqual(self.tracker.paths, {})
        self.assertFalse(any(item.current for item in self.tracker.snapshot()))
        self.now += 180
        await self.refresh()
        self.assertEqual(self.tracker.tasks, {})

    async def test_retiring_checkout_target_keeps_session_collection(self) -> None:
        self.tracker.associate("codex:one", str(self.path))
        target = RepositoryTarget(
            "branch", "feature", "example.test/team/repo", str(self.path)
        )
        self.tracker.reconcile_targets((target,))
        await self.refresh()
        self.tracker.reconcile_targets(())
        self.assertIn(str(self.path.resolve()), self.tracker.paths)
        self.assertTrue(any(item.current for item in self.tracker.snapshot()))

    async def test_reconciling_targets_stops_retired_pull_request_polling(self) -> None:
        target = RepositoryTarget("pull_request", self.pr["url"])
        self.tracker.reconcile_targets((target,))
        await self.refresh()
        previous = self.tracker.snapshot()[0]
        self.tracker.reconcile_targets(())
        self.now += 180
        self.pr["state"] = "MERGED"
        await self.refresh()
        self.assertEqual(self.tracker.snapshot(), (previous,))


class HookObservationTests(unittest.TestCase):
    def test_hook_preserves_session_routing_and_tool_metadata(self) -> None:
        packet = {
            "kind": "event",
            "harness": "codex",
            "session": "child",
            "event": "tool_done",
            "aliases": ["native", "transcript"],
            "parent": "parent",
            "parent_call": "parent-call",
            "call": "tool-call",
            "turn": "turn-id",
            "tool": "uv run checks",
            "tool_name": "exec",
            "repo": "/repo",
            "goal": "Check parser",
            "progress": "Checks passed",
            "sentiment_text": "Great",
            "outcome": "success",
            "manual": True,
            "exit_code": 0,
            "owner": {"pid": 42, "started": "process-start"},
        }
        observed = normalize_hook(packet)
        self.assertIsInstance(observed, HookEvent)
        assert observed is not None
        self.assertEqual(
            (observed.source, observed.subject), ("hook:codex", "codex:child")
        )
        self.assertEqual(observed.packet(), packet)
        packet["aliases"].append("later")
        self.assertEqual(observed.aliases, ("native", "transcript"))

    def test_hook_retains_absent_fields_and_empty_text(self) -> None:
        event = normalize_hook(
            {"harness": "claude", "session": "one", "event": "intent", "goal": ""}
        )
        assert event is not None
        self.assertEqual(
            event.packet(),
            {
                "kind": "event",
                "harness": "claude",
                "session": "one",
                "event": "intent",
                "goal": "",
            },
        )
        self.assertIsNone(event.outcome)

    def test_hook_rejects_invalid_routing_and_event_values(self) -> None:
        base = {"harness": "cursor", "session": "one", "event": "progress"}
        for changes in (
            {"aliases": [""]},
            {"parent_call": 2},
            {"event": "unknown"},
            {"harness": "unknown"},
            {"repo": "/repo\0"},
            {"outcome": "unknown"},
        ):
            with self.subTest(changes=changes):
                self.assertIsNone(normalize_hook({**base, **changes}))
