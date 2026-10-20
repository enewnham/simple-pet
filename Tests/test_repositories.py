import asyncio
from dataclasses import replace
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import tempfile
import unittest

from pet_daemon.memory import Memory
from pet_daemon.repositories import RepositoryTracker
from pet_daemon.repositories import git, github
from pet_daemon.repositories.github import PullRequest
from pet_daemon.repositories.tracker import RepositoryObservation


class RepositoryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
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
        self.pr = {"number": 1, "url": "https://example.test/team/repo/pull/1", "state": "OPEN", "mergedAt": None,
                   "reviewDecision": "REVIEW_REQUIRED", "statusCheckRollup": [], "isDraft": False,
                   "headRefOid": self.git("rev-parse", "HEAD").strip(), "headRefName": "feature",
                   "headRepositoryOwner": {"login": "team"}, "headRepository": {"name": "repo"}, "reviewRequests": [],
                   "title": "Improve parser", "body": "Handle quoted fields.", "baseRefName": "main", "baseRefOid": "base-head"}
        self.stacks = []
        self.related_prs = {}
        self.fail = False
        self.memory = Memory(Path(self.temp.name) / "memory.db", "pet", clock=lambda: self.now)
        self.tracker = RepositoryTracker(self.memory, lambda: self.now, run=self.command)

    def git(self, *args):
        return subprocess.run(["git", *args], cwd=self.path, check=True, capture_output=True, text=True).stdout

    async def command(self, *args, **kwargs):
        if args[0] == "gh":
            if self.fail:
                raise RuntimeError("GitHub unavailable")
            if args[1] == "api" and "/stacks?pull_request=" in args[2]:
                return json.dumps(self.stacks)
            if args[1:3] == ("api", "user"):
                return "reviewer\n"
            if args[1:3] == ("repo", "view"):
                return json.dumps({"isFork": False, "parent": None})
            return json.dumps([self.pr] if args[2] == "list" else self.related_prs.get(args[3], self.pr))
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
        self.pr = {**self.pr, "number": 2, "url": "https://example.test/team/repo/pull/2", "state": "OPEN", "mergedAt": None}
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

    async def test_removed_worktree_stops_polling_after_restart_and_preserves_history(self):
        self.tracker.associate("codex:one", str(self.path))
        self.memory.record_event({"event": "intent", "harness": "codex", "session": "one",
                                  "repo": str(self.path), "goal": "Review parser"})
        await self.refresh()
        await self.refresh()
        self.assertEqual(self.tracker.snapshot()[0].pr.state, "OPEN")
        await self.tracker.close()
        shutil.rmtree(self.path)
        self.tracker = RepositoryTracker(self.memory, lambda: self.now, run=self.command)
        await self.refresh()
        retained = self.tracker.snapshot()[0]
        self.assertFalse(retained.current)
        self.assertEqual(retained.pr.number, 1)
        self.assertEqual(self.memory.repository_associations(), [])
        self.assertEqual(sum(bool(item["worktree"]) for item in self.memory.load_repositories()), 1)
        self.assertEqual(self.memory.recent()[0]["goal"], "Review parser")
        await self.tracker.close()
        self.tracker = RepositoryTracker(self.memory, lambda: self.now, run=self.command)
        self.now += 360
        self.pr["state"] = "MERGED"
        await self.refresh()
        self.assertEqual(self.tracker.snapshot()[0].pr.state, "MERGED")
        self.assertEqual(self.tracker.paths, {})

    async def test_git_failure_retains_an_existing_worktree_for_retry(self):
        self.tracker.associate("codex:one", str(self.path))
        await self.refresh()
        original = self.tracker.snapshot()[0]
        async def unavailable(*args, **kwargs):
            raise OSError("Temporary process failure")
        self.tracker.run = unavailable
        self.now += 30
        await self.refresh()
        self.assertEqual(self.tracker.snapshot()[0].local, original.local)
        self.assertIn("Temporary", self.tracker.snapshot()[0].local_error)
        self.assertEqual(len(self.memory.repository_associations()), 1)
        self.tracker.run = self.command
        self.now += 30
        await self.refresh()
        self.assertIsNone(self.tracker.snapshot()[0].local_error)

    async def test_pruned_worktree_retains_pull_requests_and_stops_local_polling(self):
        worktree = self.path.parent / "worktree"
        self.git("worktree", "add", "-b", "temporary", str(worktree))
        self.pr["headRefName"] = "temporary"
        self.tracker.associate("codex:one", str(worktree))
        await self.refresh()
        await self.refresh()
        subprocess.run(["git", "switch", "-c", "another"], cwd=worktree, check=True, capture_output=True)
        self.now += 30
        await self.refresh()
        self.assertEqual(len(self.tracker.snapshot()), 2)
        shutil.rmtree(worktree)
        self.git("worktree", "prune")
        self.now += 180
        await self.refresh()
        self.assertEqual(len(self.tracker.snapshot()), 2)
        self.assertTrue(all(not item.current for item in self.tracker.snapshot()))
        self.assertEqual(next(item for item in self.tracker.snapshot() if item.pr).pr.number, 1)
        self.assertEqual(len(self.memory.load_repositories()), 2)
        self.assertEqual(self.memory.repository_associations(), [])
        self.assertEqual(self.tracker.paths, {})
        self.pr["state"] = "MERGED"
        self.now += 180
        await self.refresh()
        self.assertEqual(next(item for item in self.tracker.snapshot() if item.pr).pr.state, "MERGED")

    async def test_explicit_pull_request_polls_without_checkout(self):
        calls = []
        async def recording(*args, **kwargs):
            calls.append((args, kwargs))
            return await self.command(*args, **kwargs)
        self.tracker.run = recording
        self.tracker.track_artifact({"kind": "pull_request", "ref": self.pr["url"]})
        await self.refresh()
        item = self.tracker.snapshot()[0]
        self.assertEqual(item.pr.number, 1)
        self.assertEqual(item.repository, "example.test/team/repo")
        self.assertFalse(item.current)
        self.assertEqual(self.tracker.paths, {})
        self.assertEqual(calls[0][0][0:4], ("gh", "pr", "view", self.pr["url"]))
        self.assertIsNone(calls[0][1]["cwd"])

    async def test_explicit_branch_discovers_pull_request_without_checkout(self):
        self.tracker.track_artifact({"kind": "branch", "ref": "feature", "repository": "https://example.test/team/repo.git"})
        await self.refresh()
        item = self.tracker.snapshot()[0]
        self.assertEqual(item.pr.head_repository, "example.test/team/repo")
        self.assertEqual(item.pr.head_branch, "feature")
        self.assertEqual(item.pr.head, self.git("rev-parse", "HEAD").strip())

    async def test_session_pull_request_subscription_survives_restart_without_checkout(self):
        self.memory.work.session_intent("codex:one", "Review parser")
        self.tracker.associate("codex:one", str(self.path))
        await self.refresh()
        await self.refresh()
        self.assertTrue(any(artifact["kind"] == "pull_request"
                            for artifact in self.memory.work.session_context("codex:one")["artifacts"]))
        await self.tracker.close()
        shutil.rmtree(self.path)
        self.tracker = RepositoryTracker(self.memory, lambda: self.now, run=self.command)
        self.now += 180
        await self.refresh()
        self.assertTrue(any(item.pr and item.pr.number == 1 for item in self.tracker.snapshot()))
        self.assertEqual(self.tracker.paths, {})
        await self.tracker.close()
        self.tracker = RepositoryTracker(self.memory, lambda: self.now, run=self.command)
        self.now += 180
        self.pr["state"] = "MERGED"
        await self.refresh()
        self.assertTrue(all(item.pr.state == "MERGED" for item in self.tracker.snapshot() if item.pr))

    async def test_new_session_receives_checkout_and_pull_request_after_fresh_poll(self):
        self.tracker.associate("codex:one", str(self.path))
        await self.refresh()
        await self.refresh()
        original = next(item for item in self.tracker.snapshot() if item.current)
        self.memory.work.session_intent("claude:two", "Improve parser")
        self.tracker.associate("claude:two", str(self.path))
        self.assertNotIn("claude:two", self.tracker.snapshot()[0].sessions)
        self.assertEqual(self.memory.work.session_context("claude:two")["artifacts"], ())
        await self.refresh()
        updated = next(item for item in self.tracker.snapshot() if item.current)
        self.assertEqual(updated.sessions, ("claude:two", "codex:one"))
        self.assertEqual((updated.revision, updated.changed_at), (original.revision, original.changed_at))
        artifacts = self.memory.work.session_context("claude:two")["artifacts"]
        self.assertTrue(any(artifact["kind"] == "pull_request" and artifact["ref"] == self.pr["url"] for artifact in artifacts))

    async def test_new_session_in_known_subdirectory_receives_root_after_fresh_poll(self):
        nested = self.path / "nested"
        nested.mkdir()
        self.tracker.associate("codex:one", str(nested))
        await self.refresh()
        await self.refresh()
        self.memory.work.session_intent("claude:two", "Improve parser")
        self.tracker.associate("claude:two", str(nested))
        self.assertEqual(self.memory.work.session_context("claude:two")["artifacts"], ())
        await self.refresh()
        current = next(item for item in self.tracker.snapshot() if item.current)
        self.assertIn("claude:two", current.sessions)
        self.assertTrue(any(artifact["kind"] == "pull_request"
                            for artifact in self.memory.work.session_context("claude:two")["artifacts"]))

    async def test_new_session_after_branch_switch_never_inherits_cached_branch_or_pull_request(self):
        self.memory.work.session_intent("codex:one", "Improve parser")
        self.tracker.associate("codex:one", str(self.path))
        await self.refresh()
        await self.refresh()
        previous_url = self.pr["url"]
        self.related_prs[previous_url] = dict(self.pr)
        self.git("switch", "-c", "another")
        self.pr = {**self.pr, "number": 2, "url": "https://example.test/team/repo/pull/2", "headRefName": "another"}
        self.memory.work.session_intent("claude:two", "Improve billing")
        self.tracker.associate("claude:two", str(self.path))
        self.assertEqual(self.memory.work.session_context("claude:two")["artifacts"], ())
        await self.refresh()
        await self.refresh()
        artifacts = self.memory.work.session_context("claude:two")["artifacts"]
        self.assertEqual({artifact["ref"] for artifact in artifacts}, {"another", self.pr["url"]})
        previous = next(item for item in self.tracker.snapshot() if item.branch == "feature")
        self.assertNotIn("claude:two", previous.sessions)
        self.assertFalse(previous.current)
        self.assertEqual(previous.pr.url, previous_url)

    async def test_slower_pull_request_refresh_and_cached_checkout_preserve_newer_merge(self):
        self.memory.work.session_intent("codex:one", "Improve parser")
        self.pr["statusCheckRollup"] = [{"name": "tests", "conclusion": "FAILURE"}]
        self.tracker.associate("codex:one", str(self.path))
        await self.refresh()
        await self.refresh()
        await self.tracker.close()
        self.tracker = RepositoryTracker(self.memory, lambda: self.now, run=self.command)
        checkout = next(item for item in self.tracker.snapshot() if item.current)
        direct = next(item for item in self.tracker.snapshot() if item.subject.startswith("pr:"))
        branch = next(item for item in self.tracker.snapshot() if item.subject.startswith("branch:"))
        entered, release = asyncio.Event(), asyncio.Event()
        old = dict(self.pr)
        async def delayed(*args, **kwargs):
            if args[:3] == ("gh", "pr", "view"):
                entered.set()
                await release.wait()
                return json.dumps(old)
            return await self.command(*args, **kwargs)
        self.tracker.run = delayed
        self.now += 1
        task = asyncio.create_task(self.tracker._github(direct, self.tracker.epoch))
        await entered.wait()
        self.now += 1
        self.pr.update(state="MERGED", mergedAt="2026-10-19T00:00:00Z")
        await self.tracker._github(branch, self.tracker.epoch)
        self.now += 1
        release.set()
        await task
        await self.tracker._local(checkout.worktree, 1, self.tracker.epoch)
        artifacts = self.memory.work.session_context("codex:one")["artifacts"]
        pr = next(item for item in artifacts if item["kind"] == "pull_request")
        observed = next(item for item in pr["evidence"] if item["source"] == "github" and item["kind"] == "observation")
        self.assertEqual(observed["data"]["pr"]["state"], "MERGED")
        self.assertEqual(observed["observed_at"], 1002)
        self.assertEqual(self.memory.work.opportunities(), ())

    async def test_repeated_session_events_preserve_poll_interval_and_new_session_refreshes(self):
        self.tracker.associate("codex:one", str(self.path))
        await self.refresh()
        await self.refresh()
        polls = []
        async def counted(*args, **kwargs):
            if args[:2] == ("git", "status"):
                polls.append(args)
            return await self.command(*args, **kwargs)
        self.tracker.run = counted
        for _ in range(3):
            self.now += 1
            self.tracker.associate("codex:one", str(self.path))
            await self.refresh()
        self.assertEqual(polls, [])
        self.tracker.associate("claude:two", str(self.path))
        await self.refresh()
        self.assertEqual(len(polls), 1)
        self.assertIn("claude:two", next(item for item in self.tracker.snapshot() if item.current).sessions)

    async def test_exact_head_repository_filters_forks_and_ambiguous_open_requests(self):
        other_fork = {**self.pr, "number": 2, "headRepositoryOwner": {"login": "other"}}
        candidates = [self.pr, other_fork]
        async def listing(*args, **kwargs):
            if args[0:3] == ("gh", "pr", "list"):
                return json.dumps(candidates)
            return await self.command(*args, **kwargs)
        observed = await github.observe(None, "feature", run=listing, repository="example.test/team/repo")
        self.assertEqual(observed.number, 1)
        candidates.append({**self.pr, "number": 3})
        with self.assertRaisesRegex(RuntimeError, "Multiple open"):
            await github.observe(None, "feature", run=listing, repository="example.test/team/repo")

    async def test_incomplete_head_identity_keeps_branch_lookup_uncertain(self):
        self.pr["headRepository"] = None
        with self.assertRaisesRegex(RuntimeError, "head identity is unavailable"):
            await github.observe(None, "feature", run=self.command, repository="example.test/team/repo")

    async def test_ambiguous_branch_refresh_preserves_known_pull_request_with_uncertainty(self):
        self.tracker.associate("codex:one", str(self.path))
        await self.refresh()
        await self.refresh()
        original = self.tracker.snapshot()[0]
        async def ambiguous(*args, **kwargs):
            if args[0:3] == ("gh", "pr", "list"):
                return json.dumps([self.pr, {**self.pr, "number": 2}])
            return await self.command(*args, **kwargs)
        self.tracker.run = ambiguous
        self.now += 180
        await self.refresh()
        retained = self.tracker.snapshot()[0]
        self.assertEqual(retained.pr, original.pr)
        self.assertEqual(retained.github_checked_at, original.github_checked_at)
        self.assertIn("Multiple open", retained.github_error)

    async def test_fork_discovers_upstream_pull_request_using_its_head_identity(self):
        self.pr["headRepositoryOwner"] = {"login": "fork-owner"}
        calls = []
        async def forked(*args, **kwargs):
            calls.append(args)
            if args[0:3] == ("gh", "repo", "view"):
                return json.dumps({"isFork": True, "parent": {"owner": {"login": "team"}, "name": "repo"}})
            return await self.command(*args, **kwargs)
        observed = await github.observe(None, "feature", run=forked, repository="example.test/fork-owner/repo")
        self.assertEqual(observed.head_repository, "example.test/fork-owner/repo")
        listed = next(args for args in calls if args[1:3] == ("pr", "list"))
        self.assertEqual(listed[listed.index("--repo") + 1], "example.test/team/repo")

    async def test_pull_request_preserves_draft_review_and_check_revision(self):
        self.pr.update({"isDraft": True, "reviewRequests": [{"login": "reviewer"}, {"slug": "maintainers"}],
                        "statusCheckRollup": [{"__typename": "CheckRun", "name": "tests", "conclusion": "FAILURE",
                                               "headSha": self.pr["headRefOid"]}]})
        observed = await github.observe(None, "", run=self.command, url=self.pr["url"])
        self.assertTrue(observed.is_draft)
        self.assertEqual(observed.review_requests, ("maintainers", "reviewer"))
        self.assertEqual(observed.check_revision, observed.head)
        self.assertEqual(observed.checks, (("tests", "FAILURE", "CheckRun", observed.head),))

    async def test_pull_request_retains_exact_subject_and_base_without_inferring_body_links(self):
        self.pr["body"] = "Parser changes. See https://example.test/team/repo/pull/42 for a related discussion."
        observed = await github.observe(None, "", run=self.command, url=self.pr["url"])
        self.assertEqual((observed.title, observed.body), (self.pr["title"], self.pr["body"]))
        self.assertEqual((observed.base_branch, observed.base_head), ("main", "base-head"))
        self.assertEqual(observed.related_links, ())
        self.assertIsNone(observed.related_links_error)

    def stack(self):
        return {"number": 8, "base": {"ref": "main"}, "pull_requests": [
            {"number": 1, "state": "open", "draft": False, "merged_at": None,
             "head": {"ref": "feature", "sha": self.pr["headRefOid"]}},
            {"number": 2, "state": "closed", "draft": False, "merged_at": "2026-10-17T00:00:00Z",
             "head": {"ref": "base-layer", "sha": "base-layer-head"}},
            {"number": 3, "state": "open", "draft": True, "merged_at": None,
             "head": {"ref": "next-layer", "sha": "next-layer-head"}}]}

    async def test_native_stack_links_retain_exact_membership_and_individual_states(self):
        self.stacks = [self.stack()]
        observed = await github.observe(None, "", run=self.command, url=self.pr["url"])
        self.assertEqual([(link["ref"], link["state"], link["position"]) for link in observed.related_links],
                         [("https://example.test/team/repo/pull/2", "MERGED", 2),
                          ("https://example.test/team/repo/pull/3", "OPEN", 3)])
        self.assertTrue(observed.related_links[1]["is_draft"])
        self.assertEqual(observed.related_links[1]["head"], "next-layer-head")
        self.assertTrue(all(link["source"] == "github_stack" and link["relationship"] == "stack_entry"
                            and link["stack"] == 8 for link in observed.related_links))

    async def test_unsupported_stack_api_keeps_core_pull_request_evidence(self):
        async def unsupported(*args, **kwargs):
            if args[0:2] == ("gh", "api") and "/stacks?pull_request=" in args[2]:
                raise RuntimeError("HTTP 404: Stacks are unavailable")
            return await self.command(*args, **kwargs)
        observed = await github.observe(None, "", run=unsupported, url=self.pr["url"])
        self.assertEqual((observed.title, observed.state, observed.head),
                         (self.pr["title"], "OPEN", self.pr["headRefOid"]))
        self.assertIn("404", observed.related_links_error)
        self.assertEqual(observed.related_links, ())

    async def test_stack_membership_must_include_the_observed_pull_request(self):
        stack = self.stack()
        stack["pull_requests"] = stack["pull_requests"][1:]
        self.stacks = [stack]
        observed = await github.observe(None, "", run=self.command, url=self.pr["url"])
        self.assertEqual(observed.related_links, ())

    async def test_stack_peers_refresh_once_without_following_further_metadata(self):
        stack = self.stack()
        stack["pull_requests"] = stack["pull_requests"][:2]
        self.stacks = [stack]
        peer_url = "https://example.test/team/repo/pull/2"
        self.related_prs[peer_url] = {**self.pr, "number": 2, "url": peer_url, "state": "MERGED",
                                    "mergedAt": "2026-10-17T00:00:00Z", "headRefName": "base-layer",
                                    "headRefOid": "base-layer-head", "title": "Add parser foundation"}
        calls = []
        async def recording(*args, **kwargs):
            calls.append(args)
            return await self.command(*args, **kwargs)
        self.tracker.run = recording
        self.tracker.associate("codex:one", str(self.path))
        await self.refresh()
        await self.refresh()
        await self.refresh()
        peer = next(item for item in self.tracker.snapshot() if item.pr and item.pr.url == peer_url)
        root = next(item for item in self.tracker.snapshot() if item.current)
        self.assertEqual((root.pr.state, peer.pr.state), ("OPEN", "MERGED"))
        self.assertEqual(peer.pr.title, "Add parser foundation")
        self.assertFalse(any(args[1] == "api" and args[2].endswith("pull_request=2") for args in calls))
        self.assertTrue(any(args[1:4] == ("pr", "view", peer_url) for args in calls))
        await self.tracker.close()
        shutil.rmtree(self.path)
        self.tracker = RepositoryTracker(self.memory, lambda: self.now, run=recording)
        self.now += 180
        await self.refresh()
        self.assertEqual(self.tracker.paths, {})
        self.assertEqual(next(item for item in self.tracker.snapshot() if item.pr and item.pr.url == peer_url).pr.title,
                         "Add parser foundation")

    async def test_stack_lookup_failure_retains_known_links_without_advancing_change_recency(self):
        self.stacks = [self.stack()]
        self.tracker.associate("codex:one", str(self.path))
        await self.refresh()
        await self.refresh()
        original = next(item for item in self.tracker.snapshot() if item.current)
        async def unavailable(*args, **kwargs):
            if args[0:2] == ("gh", "api") and "/stacks?pull_request=" in args[2]:
                raise RuntimeError("Stack lookup unavailable")
            return await self.command(*args, **kwargs)
        self.tracker.run = unavailable
        self.now += 180
        await self.refresh()
        retained = next(item for item in self.tracker.snapshot() if item.current)
        self.assertEqual(retained.pr.related_links, original.pr.related_links)
        self.assertEqual((retained.revision, retained.changed_at), (original.revision, original.changed_at))
        self.assertIn("unavailable", retained.pr.related_links_error)
        self.assertGreater(retained.github_checked_at, original.github_checked_at)

    async def test_review_request_targets_the_authenticated_viewer(self):
        self.pr["reviewRequests"] = [{"login": "reviewer"}]
        requested = await github.observe(None, "", run=self.command, url=self.pr["url"])
        self.assertTrue(requested.review_requested)
        self.pr["reviewRequests"] = [{"login": "another-reviewer"}]
        other = await github.observe(None, "", run=self.command, url=self.pr["url"])
        self.assertFalse(other.review_requested)
        self.pr["reviewRequests"] = [{"slug": "maintainers"}]
        team = await github.observe(None, "", run=self.command, url=self.pr["url"])
        self.assertIsNone(team.review_requested)

    async def test_unknown_viewer_preserves_pull_request_with_uncertain_review_request(self):
        self.pr["reviewRequests"] = [{"login": "reviewer"}]
        async def unknown(*args, **kwargs):
            if args[0:3] == ("gh", "api", "user"):
                raise RuntimeError("Viewer identity unavailable")
            return await self.command(*args, **kwargs)
        observed = await github.observe(None, "", run=unknown, url=self.pr["url"])
        self.assertEqual(observed.number, 1)
        self.assertIsNone(observed.review_requested)

    async def test_repository_without_hosted_identity_has_unknown_github_status(self):
        self.git("remote", "remove", "origin")
        self.tracker.associate("codex:one", str(self.path))
        await self.refresh()
        await self.refresh()
        item = self.tracker.snapshot()[0]
        self.assertTrue(item.repository.startswith("local:"))
        self.assertIsNone(item.pr)
        self.assertEqual(item.github_checked_at, 0)
        self.assertIn("identity is unavailable", item.github_error)

    async def test_remote_aliases_and_linked_worktrees_share_repository_identity(self):
        self.assertEqual(git.canonical_repository("ssh://git@EXAMPLE.test/Team/Repo.git/"), "example.test/team/repo")
        self.assertEqual(git.canonical_repository("https://EXAMPLE.test/Team/Repo.git"), "example.test/team/repo")
        self.git("remote", "remove", "origin")
        linked = self.path.parent / "linked"
        self.git("worktree", "add", "-b", "linked", str(linked))
        original = await git.observe(str(self.path))
        other = await git.observe(str(linked))
        self.assertEqual(original.repository, other.repository)
        self.assertEqual(original.repository, "local:" + str((self.path / ".git").resolve()))

    async def test_legacy_snapshots_do_not_infer_repository_from_sessions(self):
        legacy = RepositoryObservation("legacy", str(self.path), "feature", sessions=("codex:one",))
        self.memory.save_repository(legacy.subject, {key: value for key, value in legacy.__dict__.items() if key != "repository"})
        await self.tracker.close()
        self.tracker = RepositoryTracker(self.memory, lambda: self.now, run=self.command)
        self.assertEqual(self.tracker.snapshot()[0].repository, "")
        self.assertEqual(self.tracker.paths, {})

    async def test_removed_subdirectory_keeps_its_existing_repository(self):
        nested = self.path / "nested"
        nested.mkdir()
        self.tracker.associate("codex:one", str(nested))
        await self.refresh()
        nested.rmdir()
        self.now += 30
        await self.refresh()
        self.assertEqual(self.tracker.snapshot()[0].worktree, str(self.path.resolve()))
        self.assertEqual(self.memory.repository_associations(), [{"session": "codex:one", "worktree": str(self.path.resolve())}])
        (self.path / "tracked").write_text("changed after the subdirectory disappeared\n")
        self.now += 30
        await self.refresh()
        self.assertEqual(self.tracker.snapshot()[0].local.unstaged, 1)


class RepositoryMigrationTests(unittest.TestCase):
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
