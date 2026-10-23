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

from pet_daemon.domain import (
    CollectionKind,
    SessionSource,
)
from pet_daemon.sessions.tracking import Sessions
from pet_daemon.sources import RepositoryTarget, RepositoryTracker, git, github
from pet_daemon.sources.git import CommandOptions
from pet_daemon.sources.github import GitHubPullRequest
from pet_daemon.sources.hooks import HookEvent, normalize_hook
from pet_daemon.work.observations import (
    BranchObservation,
    CheckoutObservation,
    PullRequestObservation,
    repository_observation_from_mapping,
)
from Tests.fixtures import StackFixture, SubprocessOptions


class RepositoryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.path = (Path(self.temp.name) / "repo").resolve()
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
        self.candidates: list[GitHubPullRequest] | None = None
        self.stacks: list[StackFixture] = []
        self.related_prs = {}
        self.github_unavailable = False
        self.emitted = []
        self.tracker = RepositoryTracker(
            self.emitted.append, lambda: self.now, run=self.command
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
            if args[1:3] == ("pr", "list"):
                return json.dumps(
                    self.candidates if self.candidates is not None else [self.pr]
                )
            if args[1:3] == ("pr", "view"):
                return json.dumps(self.related_prs.get(args[3], self.pr))
            raise AssertionError(f"Unsupported GitHub command: {args}")
        return await git.command(*args, **kwargs)

    async def refresh(self) -> None:
        self.tracker.tick()
        await asyncio.gather(*list(self.tracker.tasks.values()))

    async def asyncTearDown(self) -> None:
        await self.tracker.close()
        self.temp.cleanup()

    def checkout(self) -> CheckoutObservation:
        return next(
            item
            for item in self.tracker.snapshot()
            if isinstance(item, CheckoutObservation)
        )

    def branch(self) -> BranchObservation:
        return next(
            item
            for item in self.tracker.snapshot()
            if isinstance(item, BranchObservation)
        )

    async def test_checkout_and_pull_request_have_separate_observations(self) -> None:
        self.tracker.track_checkout(self.path)
        (self.path / "tracked").write_text("changed\n")
        (self.path / "new").write_text("new\n")
        (self.path / "staged").write_text("staged\n")
        self.git("add", "staged")
        await self.refresh()
        await self.refresh()
        local = self.checkout().local
        assert local is not None
        self.assertEqual((local.unstaged, local.untracked, local.staged), (1, 1, 1))
        pr = self.branch().pr
        assert pr is not None
        self.assertEqual(pr.number, 1)
        self.assertEqual(pr.url, "https://example.test/team/repo/pull/1")
        self.assertEqual(pr.head, self.pr["headRefOid"])
        self.assertEqual(pr.repository, "example.test/team/repo")
        self.assertEqual(pr.title, "Improve parser")
        for item in self.tracker.snapshot():
            self.assertEqual(repository_observation_from_mapping(item.packet()), item)

    async def test_unchanged_polls_keep_local_and_github_intervals(self) -> None:
        self.tracker.track_checkout(self.path)
        await self.refresh()
        await self.refresh()
        original = self.tracker.snapshot()
        for _ in range(3):
            self.now += 1
            self.tracker.track_checkout(self.path)
            await self.refresh()
        self.assertEqual(self.tracker.snapshot(), original)
        self.now += 27
        await self.refresh()
        self.assertEqual(self.checkout().checked_at, self.now)
        self.assertEqual(self.branch().checked_at, 1000)
        self.now += 150
        await self.refresh()
        self.assertEqual(self.branch().checked_at, self.now)

    async def test_multiple_checkouts_share_one_branch_lookup(self) -> None:
        linked = self.path.parent / "linked"
        self.git("worktree", "add", "--force", str(linked), "feature")
        calls = []

        async def recording(*args: str, **kwargs: Unpack[CommandOptions]) -> str:
            calls.append(args)
            return await self.command(*args, **kwargs)

        self.tracker.run = recording
        self.tracker.track_checkout(self.path)
        self.tracker.track_checkout(linked)
        await self.refresh()
        await self.refresh()
        self.assertEqual(sum(args[:3] == ("gh", "pr", "list") for args in calls), 1)
        self.assertEqual(
            sum(
                isinstance(item, CheckoutObservation)
                for item in self.tracker.snapshot()
            ),
            2,
        )
        pr = self.branch().pr
        assert pr is not None
        self.assertEqual(pr.number, 1)

    async def test_branch_switch_stops_polling_the_previous_branch(self) -> None:
        self.tracker.track_checkout(self.path)
        await self.refresh()
        await self.refresh()
        self.git("switch", "-c", "another")
        self.now += 30
        await self.refresh()
        await self.refresh()
        self.assertEqual(
            set(self.tracker.subscriptions),
            {("branch", "example.test/team/repo:another")},
        )
        another = next(
            item
            for item in self.tracker.snapshot()
            if isinstance(item, BranchObservation) and item.branch == "another"
        )
        self.assertIsNone(another.pr)
        self.assertIsNone(another.error)

    async def test_successful_empty_lookup_clears_branch_match(self) -> None:
        self.tracker.reconcile_targets(
            (
                RepositoryTarget(
                    CollectionKind.BRANCH, "feature", "example.test/team/repo"
                ),
            )
        )
        await self.refresh()
        self.assertIsNotNone(self.branch().pr)
        self.candidates = []
        self.now += 180
        await self.refresh()
        self.assertIsNone(self.branch().pr)
        self.assertIsNone(self.branch().error)

    async def test_failed_branch_lookup_emits_uncertainty(self) -> None:
        self.tracker.reconcile_targets(
            (
                RepositoryTarget(
                    CollectionKind.BRANCH, "feature", "example.test/team/repo"
                ),
            )
        )
        await self.refresh()
        previous = self.branch()
        self.github_unavailable = True
        self.now += 180
        await self.refresh()
        self.assertIsNotNone(previous.pr)
        self.assertIsNone(self.branch().pr)
        self.assertEqual(self.branch().checked_at, self.now)
        self.assertIn("unavailable", self.branch().error or "")

    async def test_ambiguous_historical_requests_emit_an_error(self) -> None:
        self.candidates = [
            {**self.pr, "state": "MERGED"},
            {**self.pr, "number": 2, "state": "CLOSED"},
        ]
        self.tracker.reconcile_targets(
            (
                RepositoryTarget(
                    CollectionKind.BRANCH, "feature", "example.test/team/repo"
                ),
            )
        )
        await self.refresh()
        self.assertIsNone(self.branch().pr)
        self.assertIn("Multiple historical", self.branch().error or "")
        self.candidates.append(self.pr)
        self.now += 180
        await self.refresh()
        pr = self.branch().pr
        assert pr is not None
        self.assertEqual(pr.number, 1)
        self.assertIsNone(self.branch().error)

    async def test_explicit_pull_request_keeps_polling_after_checkout_removal(
        self,
    ) -> None:
        targets = (
            RepositoryTarget(CollectionKind.CHECKOUT, str(self.path)),
            RepositoryTarget(CollectionKind.PULL_REQUEST, self.pr["url"]),
        )
        self.tracker.reconcile_targets(targets)
        await self.refresh()
        shutil.rmtree(self.path)
        self.pr["state"] = "MERGED"
        self.now += 180
        await self.refresh()
        self.assertFalse(self.checkout().available)
        self.assertEqual(self.tracker.paths, {})
        exact = next(
            item
            for item in self.tracker.snapshot()
            if isinstance(item, PullRequestObservation)
        )
        assert exact.pr is not None
        self.assertEqual(exact.pr.state, "MERGED")
        self.tracker.reconcile_targets(targets)
        self.assertEqual(self.tracker.paths, {})
        self.assertEqual(
            set(self.tracker.subscriptions), {("pull_request", self.pr["url"])}
        )

    async def test_missing_checkout_requires_a_fresh_track_request(self) -> None:
        target = RepositoryTarget(CollectionKind.CHECKOUT, str(self.path))
        self.tracker.reconcile_targets((target,))
        await self.refresh()
        moved = self.path.parent / "moved"
        self.path.rename(moved)
        self.now += 30
        await self.refresh()
        self.assertFalse(self.checkout().available)
        moved.rename(self.path)
        self.tracker.reconcile_targets((target,))
        self.now += 30
        await self.refresh()
        self.assertFalse(self.checkout().available)
        self.tracker.track_checkout(self.path)
        await self.refresh()
        self.assertTrue(self.checkout().available)

    async def test_removed_subdirectory_keeps_polling_the_checkout_root(self) -> None:
        nested = self.path / "nested"
        nested.mkdir()
        self.tracker.track_checkout(nested)
        await self.refresh()
        nested.rmdir()
        (self.path / "tracked").write_text("changed\n")
        self.now += 30
        await self.refresh()
        self.assertEqual(self.checkout().worktree, str(self.path))
        local = self.checkout().local
        assert local is not None
        self.assertEqual(local.unstaged, 1)
        self.assertEqual(set(self.tracker.paths), {str(self.path)})

    async def test_restart_polls_known_checkout_after_nested_location_disappears(
        self,
    ) -> None:
        nested = self.path / "src"
        nested.mkdir()
        self.tracker.track_checkout(nested)
        await self.refresh()
        checkpoints = self.tracker.snapshot()
        await self.tracker.close()
        nested.rmdir()
        (self.path / "tracked").write_text("changed after restart\n")
        self.tracker = RepositoryTracker(
            self.emitted.append,
            lambda: self.now,
            run=self.command,
            checkpoints=checkpoints,
            targets=(RepositoryTarget(CollectionKind.CHECKOUT, str(nested)),),
        )
        await self.refresh()
        local = self.checkout().local
        assert local is not None
        self.assertTrue(self.checkout().available)
        self.assertEqual(local.unstaged, 1)
        self.assertEqual(set(self.tracker.paths), {str(self.path)})
        self.assertEqual(
            [
                item.worktree
                for item in self.tracker.snapshot()
                if isinstance(item, CheckoutObservation)
            ],
            [str(self.path)],
        )

    async def test_restart_keeps_missing_root_suppressed_for_nested_location(
        self,
    ) -> None:
        nested = self.path / "src"
        nested.mkdir()
        self.tracker.track_checkout(nested)
        await self.refresh()
        previous = self.checkout()
        checkpoints = (
            CheckoutObservation(
                previous.worktree, previous.local, self.now, available=False
            ),
        )
        await self.tracker.close()
        nested.rmdir()
        self.tracker = RepositoryTracker(
            self.emitted.append,
            lambda: self.now,
            run=self.command,
            checkpoints=checkpoints,
            targets=(RepositoryTarget(CollectionKind.CHECKOUT, str(nested)),),
        )
        await self.refresh()
        self.assertFalse(self.checkout().available)
        self.assertEqual(self.tracker.paths, {})
        self.tracker.track_checkout(nested)
        await self.refresh()
        self.assertTrue(self.checkout().available)
        self.assertEqual(set(self.tracker.paths), {str(self.path)})

    async def test_targets_and_observations_restore_without_checkout(self) -> None:
        target = RepositoryTarget(CollectionKind.PULL_REQUEST, self.pr["url"])
        self.tracker.reconcile_targets((target,))
        await self.refresh()
        checkpoint = self.tracker.snapshot()
        await self.tracker.close()
        shutil.rmtree(self.path)
        self.tracker = RepositoryTracker(
            self.emitted.append,
            lambda: self.now,
            run=self.command,
            checkpoints=checkpoint,
            targets=(target,),
        )
        self.assertEqual(self.tracker.snapshot(), checkpoint)
        self.pr["state"] = "MERGED"
        await self.refresh()
        exact = next(
            item
            for item in self.tracker.snapshot()
            if isinstance(item, PullRequestObservation)
        )
        assert exact.pr is not None
        self.assertEqual(exact.pr.state, "MERGED")
        self.assertEqual(self.tracker.paths, {})

    async def test_retiring_targets_cancels_and_discards_late_callbacks(self) -> None:
        target = RepositoryTarget(CollectionKind.PULL_REQUEST, self.pr["url"])
        self.tracker.reconcile_targets((target,))
        entered, released = asyncio.Event(), asyncio.Event()

        async def delayed(*args: str, **kwargs: Unpack[CommandOptions]) -> str:
            entered.set()
            try:
                await released.wait()
            except asyncio.CancelledError:
                await released.wait()
            return await self.command(*args, **kwargs)

        self.tracker.run = delayed
        self.tracker.tick()
        await entered.wait()
        tasks = list(self.tracker.tasks.values())
        self.tracker.reconcile_targets(())
        self.tracker.reconcile_targets((target,))
        released.set()
        await asyncio.gather(*tasks)
        self.assertEqual(self.emitted, [])
        self.tracker.run = self.command
        await self.refresh()
        self.assertEqual(len(self.emitted), 1)

    async def test_pause_discards_late_results_and_resume_refreshes(self) -> None:
        target = RepositoryTarget(CollectionKind.PULL_REQUEST, self.pr["url"])
        self.tracker.reconcile_targets((target,))
        await self.refresh()
        previous = self.tracker.snapshot()
        entered = asyncio.Event()

        async def delayed(*args: str, **kwargs: Unpack[CommandOptions]) -> str:
            if not entered.is_set():
                entered.set()
                try:
                    await asyncio.Future[Never]()
                except asyncio.CancelledError:
                    pass
            return await self.command(*args, **kwargs)

        self.tracker.run = delayed
        self.now += 180
        self.tracker.tick()
        await entered.wait()
        tasks = list(self.tracker.tasks.values())
        self.tracker.pause()
        await asyncio.gather(*tasks)
        self.assertEqual(self.tracker.snapshot(), previous)
        self.tracker.tick()
        self.assertEqual(self.tracker.tasks, {})
        self.tracker.run = self.command
        self.pr["state"] = "MERGED"
        self.tracker.resume()
        await self.refresh()
        exact = next(
            item
            for item in self.tracker.snapshot()
            if isinstance(item, PullRequestObservation)
        )
        assert exact.pr is not None
        self.assertEqual(exact.pr.state, "MERGED")

    async def test_clear_cancels_collection_and_discards_late_callbacks(self) -> None:
        self.tracker.track_checkout(self.path)
        entered = asyncio.Event()

        async def delayed(*args: str, **kwargs: Unpack[CommandOptions]) -> str:
            if not entered.is_set():
                entered.set()
                try:
                    await asyncio.Future[Never]()
                except asyncio.CancelledError:
                    pass
            return await self.command(*args, **kwargs)

        self.tracker.run = delayed
        self.tracker.tick()
        await entered.wait()
        tasks = list(self.tracker.tasks.values())
        self.tracker.clear()
        await asyncio.gather(*tasks)
        self.assertEqual(self.emitted, [])
        self.assertEqual(self.tracker.snapshot(), ())
        self.assertEqual(self.tracker.paths, {})

    async def test_local_refresh_error_retries_at_the_next_interval(self) -> None:
        self.tracker.track_checkout(self.path)
        await self.refresh()
        original = self.checkout()
        ordinary = self.command

        async def unavailable(*args: str, **kwargs: Unpack[CommandOptions]) -> str:
            if args[0] == "git":
                raise RuntimeError("Git unavailable")
            return await ordinary(*args, **kwargs)

        self.tracker.run = unavailable
        self.now += 30
        await self.refresh()
        self.assertIsNotNone(original.local)
        self.assertIsNone(self.checkout().local)
        self.assertIn("unavailable", self.checkout().error or "")
        self.assertIn(str(self.path), self.tracker.paths)
        self.tracker.run = self.command
        self.now += 30
        await self.refresh()
        self.assertEqual(self.checkout().local, original.local)
        self.assertIsNone(self.checkout().error)

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
        self.assertIn(
            (
                "gh",
                "repo",
                "view",
                "example.test/fork-owner/repo",
                "--json",
                "isFork,parent",
            ),
            calls,
        )
        listed = next(args for args in calls if args[1:3] == ("pr", "list"))
        self.assertEqual(listed[listed.index("--repo") + 1], "example.test/team/repo")

    async def test_pull_request_retains_exact_subject_and_base_without_inferring_body_links(
        self,
    ) -> None:
        self.pr["body"] = (
            "Parser changes. See https://example.test/team/repo/pull/42 for a related discussion."
        )
        calls = []

        async def lookup(*args: str, **kwargs: Unpack[CommandOptions]) -> str:
            calls.append(args)
            return await self.command(*args, **kwargs)

        observed = await github.observe(None, "", run=lookup, url=self.pr["url"])
        assert observed is not None
        self.assertEqual(calls[0][:4], ("gh", "pr", "view", self.pr["url"]))
        self.assertIn("--json", calls[0])
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


class HookObservationTests(unittest.TestCase):
    def test_session_sources_preserve_wire_values_and_canonical_session_ids(
        self,
    ) -> None:
        for kind in SessionSource:
            with self.subTest(harness=kind):
                hook = normalize_hook(
                    {"harness": kind.value, "session": "one", "event": "intent"}
                )
                assert hook is not None
                self.assertIs(hook.harness, kind)
                packet = hook.packet()
                self.assertIs(type(packet["harness"]), str)
                self.assertEqual(packet["harness"], kind.value)
                self.assertEqual(hook.subject, f"{kind.value}:one")
                sessions = Sessions(lambda: 1000)
                update = sessions.apply(hook)
                assert update is not None
                self.assertEqual(update.session, hook.subject)
                self.assertIs(sessions.by_id[update.session].harness, kind)

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
