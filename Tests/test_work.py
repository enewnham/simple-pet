import sqlite3
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Unpack

from pet_daemon.runtime import HookEvent, HookTextFields
from pet_daemon.sessions.tracking import Sessions
from pet_daemon.storage.memory import Memory
from pet_daemon.storage.migrations import SCHEMA_VERSION, migrate
from pet_daemon.work.history import project_event
from pet_daemon.work.models import RelatedLink
from pet_daemon.work.observations import (
    BranchObservation,
    CheckoutObservation,
    GitObservation,
    PullRequest,
    PullRequestObservation,
)
from pet_daemon.work.views import (
    EvidenceView,
    ExecutionView,
    ProgressView,
    PullRequestEvidenceView,
    WorkStory,
)
from Tests.fixtures import require, sessions_snapshot

REPOSITORY = "github.com/team/project"


class WorkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "memory.sqlite3"
        self.now = 1000.0
        self.memory = Memory(self.path, "fixture", clock=lambda: self.now)
        self.store = self.memory.work
        self.sessions = Sessions(lambda: self.now)

    def tearDown(self) -> None:
        self.memory.close()
        self.temporary.cleanup()

    def event(
        self, event: str, session: str = "codex:one", **fields: Unpack[HookTextFields]
    ) -> None:
        harness, name = session.split(":", 1)
        hook = HookEvent(harness, name, event, **fields)
        result = self.sessions.apply(hook)
        if result is not None:
            with self.memory.transaction():
                self.store.record_session(result.observation)
                self.memory.record_event(project_event(hook))

    def checkout(
        self, branch: str = "feature", path: str = "/work/parser"
    ) -> CheckoutObservation:
        return CheckoutObservation(
            path,
            GitObservation(path, branch, "head", 0, 1, 0, 0, "", 0, 0, REPOSITORY),
            self.now,
        )

    def pr(self, number: int = 7) -> PullRequest:
        return PullRequest(
            number,
            f"https://github.com/team/project/pull/{number}",
            "OPEN",
            None,
            True,
            "head",
            REPOSITORY,
            "feature",
            REPOSITORY,
            "Handle quotes",
        )

    def branch(self, number: int = 7, branch: str = "feature") -> BranchObservation:
        return BranchObservation(
            REPOSITORY, branch, replace(self.pr(number), head_branch=branch), self.now
        )

    def story(self, session: str = "codex:one") -> WorkStory:
        return next(
            subject.story
            for subject in self.memory.connected_work(sessions_snapshot())
            if session in subject.sessions
        )

    def reports(self, session: str = "codex:one") -> tuple[EvidenceView, ...]:
        return self.story(session).activity

    def test_child_keeps_its_scope_and_report_with_the_parent_story(self) -> None:
        self.event(
            "intent",
            "codex:child",
            goal="Cover escapes",
            parent="root",
            repo="/work/child",
        )
        self.event("intent", "codex:root", goal="Improve parser", repo="/work/root")
        self.event("progress", "codex:child", progress="Escape tests pass")
        story = self.story("codex:root")
        self.assertEqual(story.intent, "Improve parser")
        self.assertEqual(
            [record.details for record in story.activity],
            [ExecutionView("in progress", None, None)],
        )
        self.assertEqual(story.sessions[0].scope, "Cover escapes")
        self.assertEqual(require(story.sessions[0].context).checkout, "/work/child")
        self.assertEqual(
            self.story("codex:root").sessions[0].activity[0].details,
            ProgressView("Escape tests pass"),
        )

    def test_refined_intent_and_alias_reports_share_one_story(self) -> None:
        self.event("intent", goal="Improve parser")
        self.event("intent", goal="Cover escapes", aliases=("alias",))
        self.event("progress", "codex:alias", progress="Covered escapes")
        subjects = self.memory.connected_work(sessions_snapshot())
        self.assertEqual(tuple(subject.id for subject in subjects), ("codex:one",))
        self.assertEqual(subjects[0].story.intent, "Cover escapes")
        self.assertEqual(
            subjects[0].story.activity[0].details, ProgressView("Covered escapes")
        )

    def test_explicit_alias_keeps_one_family_with_the_latest_report(
        self,
    ) -> None:
        sessions = Sessions(lambda: self.now)

        def ingest(hook: HookEvent) -> str:
            result = require(sessions.apply(hook))
            with self.memory.transaction():
                self.store.record_session(result.observation)
                self.memory.record_event(project_event(hook))
            return result.session

        ingest(
            HookEvent("codex", "a", "session_start", repo="/work/a", aliases=("old-a",))
        )
        ingest(
            HookEvent("codex", "b", "session_start", repo="/work/b", aliases=("old-b",))
        )
        ingest(HookEvent("codex", "b", "intent", goal="Initial parser goal"))
        self.now += 1
        ingest(HookEvent("codex", "a", "intent", goal="Refined parser goal"))
        ingest(
            HookEvent(
                "codex",
                "a",
                "progress",
                progress="First parser report",
                repository=REPOSITORY,
                branch="first",
                pull_request=self.pr(7).url,
            )
        )
        self.now += 1
        ingest(
            HookEvent(
                "codex",
                "b",
                "progress",
                progress="Second parser report",
                repo="/work/latest",
                repository=REPOSITORY,
                branch="second",
                pull_request=self.pr(8).url,
            )
        )
        ingest(HookEvent("codex", "b", "waiting"))
        ingest(
            HookEvent(
                "codex", "child", "subagent_start", parent="a", goal="Check escapes"
            )
        )
        self.assertEqual(
            ingest(HookEvent("codex", "b", "session_start", aliases=("a",))), "codex:b"
        )
        subjects = self.memory.connected_work(sessions.activity_view())
        self.assertEqual(tuple(subject.id for subject in subjects), ("codex:b",))
        story = subjects[0].story
        self.assertEqual(story.intent, "Refined parser goal")
        self.assertEqual(story.sessions[0].scope, "Check escapes")
        reports = [
            record.details
            for record in story.activity
            if isinstance(record.details, ProgressView)
        ]
        self.assertEqual(reports, [ProgressView("Second parser report")])
        self.assertTrue(
            any(
                "First parser report" in record.summary
                for record in self.memory.recent()
            )
        )
        self.assertEqual(
            {item.ref for item in story.artifacts if item.kind == "pull_request"},
            {self.pr(7).url, self.pr(8).url},
        )
        self.memory.close()
        self.memory = Memory(self.path, "fixture", clock=lambda: self.now)
        self.store = self.memory.work
        self.assertEqual(self.memory.connected_work(sessions.activity_view()), subjects)

    def test_report_resolves_stored_branch_without_reading_later_checkout(self) -> None:
        self.store.observe(self.checkout())
        self.event("intent", goal="Improve parser", repo="/work/parser")
        self.event("progress", progress="Feature tests pass")
        self.now += 1
        self.store.observe(self.checkout("other"))
        self.store.observe(self.branch(8, "other"))
        self.store.observe(self.branch(7))
        context = require(self.reports()[0].context)
        self.assertEqual(context.branch, "feature")
        self.assertEqual(context.pull_request, self.pr(7).url)
        self.assertEqual(context.repository, REPOSITORY)

    def test_unknown_report_context_stays_unknown_after_checkout_discovery(
        self,
    ) -> None:
        self.event("intent", goal="Improve parser", repo="/work/parser")
        self.event("progress", progress="Feature tests pass")
        self.store.observe(self.checkout())
        self.store.observe(self.branch())
        context = require(self.reports()[0].context)
        self.assertEqual(context.checkout, "/work/parser")
        self.assertEqual(context.repository, "")
        self.assertEqual(context.branch, "")
        self.assertEqual(context.pull_request, "")

    def test_branch_reuse_keeps_report_known_pull_request(self) -> None:
        self.store.observe(self.checkout())
        self.store.observe(self.branch(7))
        self.event("intent", goal="Improve parser", repo="/work/parser")
        self.event("progress", progress="Parser work is ready")
        self.now += 1
        self.store.observe(self.branch(8))
        self.assertEqual(
            require(self.reports()[0].context).pull_request, self.pr(7).url
        )

    def test_report_explicit_reference_takes_precedence_over_checkout(self) -> None:
        self.store.observe(self.checkout())
        self.store.observe(self.branch(7))
        self.event("intent", goal="Improve parser", repo="/work/parser")
        self.event("progress", progress=f"Validated {self.pr(8).url}")
        self.assertEqual(
            require(self.reports()[0].context).pull_request, self.pr(8).url
        )
        self.event(
            "progress",
            progress="Validated the alternate branch",
            repository=REPOSITORY,
            branch="alternate",
            pull_request=self.pr(9).url,
        )
        context = require(self.reports()[0].context)
        self.assertEqual(
            (context.branch, context.pull_request), ("alternate", self.pr(9).url)
        )

    def test_identical_poll_preserves_story_revision_and_diagnostic_history(
        self,
    ) -> None:
        self.store.observe(self.checkout())
        self.store.observe(self.branch())
        self.event("intent", goal="Improve parser", repo="/work/parser")
        before = self.story()
        history = self.memory.recent()
        self.now += 30
        self.store.observe(self.checkout())
        self.store.observe(self.branch())
        self.assertEqual(before.revision, self.story().revision)
        self.assertEqual(self.memory.recent(), history)
        checkpoints = self.memory.load_repositories()
        self.assertEqual({item.checked_at for item in checkpoints}, {self.now})

    def test_stale_response_cannot_replace_newer_current_facts(self) -> None:
        fresh = self.branch(8)
        self.store.observe(fresh)
        self.store.observe(replace(self.branch(7), checked_at=self.now - 1))
        self.store.observe(
            PullRequestObservation(
                self.pr(8).url, replace(self.pr(8), state="MERGED"), self.now + 1
            )
        )
        self.store.observe(PullRequestObservation(self.pr(8).url, self.pr(8), self.now))
        self.event(
            "intent", goal="Improve parser", repository=REPOSITORY, branch="feature"
        )
        story = self.story()
        outputs = [
            evidence.details.output
            for artifact in story.artifacts
            for evidence in artifact.evidence
            if isinstance(evidence.details, PullRequestEvidenceView)
        ]
        self.assertEqual(
            [(output.url, output.state) for output in outputs],
            [(self.pr(8).url, "MERGED")],
        )

    def test_failed_lookup_preserves_last_known_pull_request_with_uncertainty(
        self,
    ) -> None:
        self.event(
            "intent", goal="Improve parser", repository=REPOSITORY, branch="feature"
        )
        self.store.observe(self.branch())
        self.now += 1
        self.store.observe(
            BranchObservation(
                REPOSITORY, "feature", None, self.now, "network unavailable"
            )
        )
        artifact = next(
            item for item in self.story().artifacts if item.kind == "pull_request"
        )
        self.assertEqual(artifact.ref, self.pr().url)
        self.assertTrue(
            any(
                record.kind == "availability"
                for item in self.story().artifacts
                for record in item.evidence
            )
        )

    def test_failed_poll_preserves_fact_time_and_reports_latest_error_time(
        self,
    ) -> None:
        self.store.observe(self.checkout())
        self.store.observe(self.branch())
        self.event("intent", goal="Improve parser", repo="/work/parser")
        self.now += 30
        self.store.observe(
            CheckoutObservation("/work/parser", None, self.now, "local failed")
        )
        self.store.observe(
            BranchObservation(REPOSITORY, "feature", None, self.now, "branch failed")
        )
        self.store.observe(
            PullRequestObservation(self.pr().url, None, self.now, "PR failed")
        )
        account = self.story()
        facts = [
            record
            for artifact in account.artifacts
            for record in artifact.evidence
            if record.kind == "observation"
        ]
        errors = [
            record
            for artifact in account.artifacts
            for record in artifact.evidence
            if record.kind == "availability"
        ]
        self.assertEqual({record.observed_at for record in facts}, {1000})
        self.assertEqual({record.observed_at for record in errors}, {1030})
        history = self.memory.recent()
        self.now += 30
        self.store.observe(
            CheckoutObservation("/work/parser", None, self.now, "local failed")
        )
        self.store.observe(
            BranchObservation(REPOSITORY, "feature", None, self.now, "branch failed")
        )
        self.store.observe(
            PullRequestObservation(self.pr().url, None, self.now, "PR failed")
        )
        self.assertEqual(self.memory.recent(), history)
        before = self.story()
        self.store.observe(
            PullRequestObservation(self.pr().url, None, 1040, "stale error")
        )
        self.assertEqual(self.story(), before)
        account = self.story()
        self.assertEqual(
            {
                record.observed_at
                for artifact in account.artifacts
                for record in artifact.evidence
                if record.kind == "availability"
            },
            {1060},
        )

    def test_first_failed_lookup_does_not_establish_missing_pull_request(self) -> None:
        self.event("intent", goal=f"Inspect {self.pr().url}")
        self.store.observe(
            PullRequestObservation(self.pr().url, None, self.now, "network failed")
        )
        account = self.story()
        self.assertFalse(
            any(
                record.kind == "observation"
                for artifact in account.artifacts
                for record in artifact.evidence
            )
        )
        self.assertTrue(
            any(
                record.kind == "availability"
                for artifact in account.artifacts
                for record in artifact.evidence
            )
        )

    def test_metadata_changes_do_not_rearm_story_but_authored_progress_does(
        self,
    ) -> None:
        self.event("intent", goal="Improve parser", repo="/work/parser")
        before = self.story()
        self.store.observe(self.checkout())
        self.store.observe(self.branch())
        after = self.story()
        self.assertNotEqual(before.artifacts, after.artifacts)
        self.assertEqual(before.revision, after.revision)
        self.event("progress", progress="Quoted strings are covered")
        self.assertNotEqual(after.revision, self.story().revision)

    def test_authoritative_resume_keeps_raw_tool_details_out_of_the_story(self) -> None:
        self.event("intent", goal="Improve parser")
        self.event("waiting")
        waiting = self.story()
        self.event("tool_start", call="read", tool="Read", progress="Raw command")
        after = self.story()
        self.assertNotEqual(waiting.revision, after.revision)
        self.assertEqual(
            after.activity[0].details, ExecutionView("in progress", None, None)
        )
        self.assertNotIn("Raw command", str(after))
        self.assertNotIn("Raw command", str(self.memory.recent()))

    def test_unrelated_tool_result_keeps_waiting_and_story(self) -> None:
        self.event("intent", goal="Improve parser")
        self.event("waiting")
        before = self.story()
        self.event("tool_done", call="unrelated", progress="Raw output")
        self.assertEqual(before, self.story())

    def test_child_and_parent_reports_retain_distinct_scopes(self) -> None:
        self.event("intent", "codex:child", parent="root", goal="Cover escapes")
        self.event("progress", "codex:child", progress="Child tests pass")
        self.event("intent", "codex:root", goal="Improve parser")
        self.event("progress", "codex:root", progress="Parent tests pass")
        self.assertEqual(
            self.story("codex:root").sessions[0].activity[0].details,
            ProgressView("Child tests pass"),
        )
        self.assertEqual(
            self.reports("codex:root")[0].details, ProgressView("Parent tests pass")
        )

    def test_repository_observations_create_no_agent_story(self) -> None:
        self.store.observe(self.checkout())
        self.store.observe(self.branch())
        self.assertEqual(self.memory.connected_work(sessions_snapshot()), ())
        self.assertEqual(self.memory.collection_targets(), ())
        self.event("intent", goal="Improve parser")
        self.assertEqual(self.story().artifacts, ())

    def test_collection_targets_include_session_paths_and_report_exact_refs(
        self,
    ) -> None:
        self.event("intent", goal=f"Inspect {self.pr().url}", repo="/work/parser")
        targets = {(item.kind, item.ref) for item in self.memory.collection_targets()}
        self.assertEqual(
            targets, {("checkout", "/work/parser"), ("pull_request", self.pr().url)}
        )

    def test_explicit_branch_reference_is_collected_without_a_checkout(self) -> None:
        self.event(
            "intent", goal="Improve parser", repository=REPOSITORY, branch="feature"
        )
        self.assertEqual(
            {
                (item.kind, item.ref, item.repository)
                for item in self.memory.collection_targets()
            },
            {("branch", "feature", REPOSITORY)},
        )
        self.store.observe(self.branch())
        account = self.story()
        self.assertEqual(
            require(account.requests[0].context).pull_request, self.pr().url
        )
        self.assertEqual(
            {(item.kind, item.ref) for item in self.memory.collection_targets()},
            {("pull_request", self.pr().url)},
        )

    def test_failed_report_keeps_the_stored_session_and_history_unchanged(self) -> None:
        self.event("intent", goal="Improve parser", repo="/work/parser")
        self.memory.db.execute(
            "CREATE TRIGGER reject_progress BEFORE INSERT ON work_reports WHEN new.kind='progress' BEGIN SELECT RAISE(ABORT, 'reject'); END"
        )
        before = self.story()
        history = self.memory.recent()
        with self.assertRaises(sqlite3.IntegrityError):
            self.event("progress", repo="/work/other", progress="Must roll back")
        self.assertEqual(self.story(), before)
        self.assertEqual(self.memory.recent(), history)

    def test_session_story_and_captured_context_survive_restart(self) -> None:
        self.store.observe(self.checkout())
        self.store.observe(self.branch())
        self.event("intent", goal="Improve parser", repo="/work/parser")
        self.event("progress", progress="Quoted strings are covered")
        before = self.memory.connected_work(sessions_snapshot())
        self.memory.close()
        self.memory = Memory(self.path, "fixture", clock=lambda: self.now)
        self.store = self.memory.work
        self.assertEqual(self.memory.connected_work(sessions_snapshot()), before)

    def test_alias_after_restart_keeps_parent_checkout_and_report_context(self) -> None:
        self.store.observe(self.checkout())
        self.store.observe(self.branch())
        self.event("intent", "codex:root", goal="Improve parser", repo="/work/root")
        self.event(
            "subagent_start",
            "codex:old",
            goal="Cover escapes",
            parent="root",
            repo="/work/parser",
        )
        self.memory.close()
        self.memory = Memory(self.path, "fixture", clock=lambda: self.now)
        self.store = self.memory.work
        self.sessions = Sessions(lambda: self.now)
        self.event(
            "progress",
            "codex:native",
            aliases=("old",),
            progress="Escape cases pass",
        )

        subjects = self.memory.connected_work(self.sessions.activity_view())
        self.assertEqual(tuple(subject.id for subject in subjects), ("codex:root",))
        child = subjects[0].story.sessions[0]
        self.assertEqual(child.scope, "Cover escapes")
        report = next(
            record
            for record in child.activity
            if record.details == ProgressView("Escape cases pass")
        )
        context = require(report.context)
        self.assertEqual(
            (
                context.checkout,
                context.repository,
                context.branch,
                context.pull_request,
            ),
            ("/work/parser", REPOSITORY, "feature", self.pr().url),
        )
        self.assertIn(
            ("checkout", "/work/parser"),
            {(target.kind, target.ref) for target in self.memory.collection_targets()},
        )

    def test_completed_session_keeps_report_pull_request_when_checkout_is_reused(
        self,
    ) -> None:
        self.store.observe(self.checkout())
        self.store.observe(self.branch(7))
        self.event("intent", goal="Improve parser", repo="/work/parser")
        self.event("turn_done", progress="Parser work complete", outcome="success")
        self.now += 1
        self.store.observe(self.checkout("other"))
        self.store.observe(self.branch(8, "other"))
        self.event(
            "intent", "codex:two", goal="Improve other code", repo="/work/parser"
        )
        old = self.story()
        self.assertEqual(
            {item.ref for item in old.artifacts if item.kind == "pull_request"},
            {self.pr(7).url},
        )
        self.event("intent", goal="Continue parser")
        resumed = self.story()
        self.assertIn(self.pr(8).url, {item.ref for item in resumed.artifacts})

    def test_nested_location_uses_closest_known_checkout_for_report_context(
        self,
    ) -> None:
        self.store.observe(self.checkout(path="/work"))
        self.store.observe(self.checkout(path="/work/parser"))
        self.store.observe(self.branch())
        self.event("intent", goal="Improve parser", repo="/work/parser/src")
        self.event("progress", progress="Parser tests pass")
        context = require(self.reports()[0].context)
        self.assertEqual(
            (
                context.checkout,
                context.repository,
                context.branch,
                context.pull_request,
            ),
            ("/work/parser", REPOSITORY, "feature", self.pr().url),
        )

    def test_stack_lookup_failure_preserves_links_and_successful_empty_clears_them(
        self,
    ) -> None:
        link = RelatedLink(
            "pull_request", self.pr(8).url, "stack_entry", "github_stack"
        )
        first = replace(self.pr(), related_links=(link,))
        self.store.observe(PullRequestObservation(first.url, first, self.now))
        self.event("intent", goal=f"Review {first.url}")
        self.now += 1
        failed = replace(
            first,
            state="MERGED",
            related_links=(),
            related_links_error="stack lookup unavailable",
        )
        self.store.observe(PullRequestObservation(first.url, failed, self.now))
        output = next(item for item in self.story().artifacts if item.ref == first.url)
        self.assertTrue(
            any(
                isinstance(record.details, PullRequestEvidenceView)
                and record.details.output.state == "MERGED"
                for record in output.evidence
            )
        )
        self.assertIn(
            self.pr(8).url,
            {item.ref for item in self.story().artifacts},
        )
        self.now += 1
        self.store.observe(
            PullRequestObservation(
                first.url, replace(failed, related_links_error=None), self.now
            )
        )
        self.assertNotIn(
            self.pr(8).url,
            {item.ref for item in self.story().artifacts},
        )

    def test_missing_checkout_retains_last_known_facts_with_uncertainty(self) -> None:
        self.event("intent", goal="Improve parser", repo="/work/parser")
        self.store.observe(self.checkout())
        self.store.observe(self.branch())
        self.now += 1
        self.store.observe(
            CheckoutObservation("/work/parser", None, self.now, available=False)
        )
        account = self.story()
        self.assertIn(self.pr().url, {item.ref for item in account.artifacts})
        self.assertIn(
            self.pr().url, {item.ref for item in self.memory.collection_targets()}
        )
        self.assertTrue(
            any(
                record.kind == "availability"
                for item in account.artifacts
                for record in item.evidence
            )
        )

    def test_explicit_pull_request_metadata_is_validated_before_persistence(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            self.event(
                "intent", goal="Improve parser", pull_request="not a pull request"
            )
        self.assertEqual(self.memory.connected_work(sessions_snapshot()), ())
        self.event("intent", goal="Improve parser", pull_request=self.pr().url + "/")
        account = self.story()
        self.assertEqual(
            require(account.requests[0].context).pull_request, self.pr().url
        )
        self.assertIn(self.pr().url, {item.ref for item in account.artifacts})

    def test_future_schema_is_rejected_before_any_write(self) -> None:
        database = sqlite3.connect(":memory:")
        database.execute(f"PRAGMA user_version={SCHEMA_VERSION + 1}")
        statements = []
        database.set_trace_callback(statements.append)
        try:
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                migrate(database)
            self.assertEqual(statements, ["PRAGMA user_version"])
        finally:
            database.close()
