import tempfile
import unittest
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Unpack

from pet_daemon.runtime import FailureSnapshot, ToolSnapshot
from pet_daemon.sources.hooks import HookEvent
from pet_daemon.storage import Memory
from pet_daemon.work import build_work
from pet_daemon.work.models import (
    ArtifactDescriptor,
    EvidenceRecord,
    GitHubEvidence,
    IntentEvidence,
    ProgressEvidence,
    PullRequestAssociation,
    RelatedLink,
)
from pet_daemon.work.observations import (
    GitObservation,
    PullRequest,
    RepositoryObservation,
)
from pet_daemon.work.presentation import work_record
from pet_daemon.work.validation import descriptor
from pet_daemon.work.views import project_evidence
from Tests.fixtures import (
    ObservationOptions,
    PullRequestOptions,
    require,
    session_snapshot,
    sessions_snapshot,
)


class WorkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "memory.sqlite3"
        self.now = 1000
        self.memory = Memory(self.path, "fixture", clock=lambda: self.now)
        self.store = self.memory.work

    def tearDown(self) -> None:
        self.memory.close()
        self.temporary.cleanup()

    def start(self, session: str = "codex:one", intent: str = "Improve parser") -> str:
        return self.store.session_intent(session, intent)

    def progress(
        self,
        session: str = "codex:one",
        value: str = "Implemented quoted string handling",
    ) -> None:
        harness, name = session.split(":", 1)
        self.store.session_event(
            session, HookEvent(harness, name, "progress", progress=value)
        )

    def pr(self, number: int = 7, **extra: Unpack[PullRequestOptions]) -> PullRequest:
        initial = PullRequest(
            number=number,
            url=f"https://github.com/team/project/pull/{number}",
            state="OPEN",
            merged_at=None,
            review="",
            checks=(),
            is_draft=True,
            head="head-one",
            head_repository="github.com/team/project",
            head_branch="feature",
            repository="github.com/team/project",
            check_revision="head-one",
            review_requested=False,
            title="Handle parser quotes",
            body="Recognize quoted strings.",
        )
        return replace(initial, **extra)

    def observation(
        self, pr: PullRequest | None = None, **extra: Unpack[ObservationOptions]
    ) -> RepositoryObservation:
        initial = RepositoryObservation(
            subject="fixture",
            repository="github.com/team/project",
            worktree="/work/parser",
            branch="feature",
            sessions=("codex:one",),
            local=GitObservation(
                "/work/parser",
                "feature",
                "head-one",
                0,
                1,
                0,
                0,
                "",
                0,
                0,
                "github.com/team/project",
            ),
            pr=pr,
            local_checked_at=self.now,
            github_checked_at=self.now,
        )
        return replace(initial, **extra)

    def observe(self, observation: RepositoryObservation) -> None:
        self.store.observe_repository(observation)

    def kinds(self) -> set[str]:
        return {item.kind for item in self.store.opportunities()}

    def restart(self) -> None:
        self.memory.close()
        self.memory = Memory(self.path, "fixture", clock=lambda: self.now)
        self.store = self.memory.work

    def test_native_intents_and_progress_form_an_attributed_connected_account(
        self,
    ) -> None:
        self.start()
        self.progress()
        self.observe(self.observation(self.pr()))
        account = require(self.store.session_context("codex:one"))
        self.assertEqual(account.intent, "Improve parser")
        self.assertEqual(
            account.requests[0].payload,
            IntentEvidence("codex:one", "Improve parser"),
        )
        self.assertTrue(
            any(
                isinstance(item.payload, ProgressEvidence)
                and item.payload.text == "Implemented quoted string handling"
                for item in account.activity
            )
        )
        self.assertEqual(
            {item.artifact.kind for item in account.artifacts},
            {"branch", "pull_request"},
        )
        pull_request = next(
            item for item in account.artifacts if item.artifact.kind == "pull_request"
        )
        self.assertEqual(pull_request.artifact.title, "Handle parser quotes")
        observed = next(
            item for item in pull_request.evidence if item.kind == "observation"
        )
        assert isinstance(observed.payload, GitHubEvidence)
        assert observed.payload.pr is not None
        self.assertEqual(observed.payload.pr.title, "Handle parser quotes")
        self.assertTrue(
            any(
                isinstance(item.payload, PullRequestAssociation)
                for item in pull_request.evidence
            )
        )
        self.assertEqual(self.kinds(), set())

    def test_followup_intent_refines_work_without_collapsing_other_sessions(
        self,
    ) -> None:
        first = self.start()
        self.assertEqual(
            self.store.session_intent("codex:one", "Cover escape sequences"), first
        )
        second = self.start("claude:two", "Review the parser API")
        self.observe(
            self.observation(
                self.pr(),
                sessions=(
                    "codex:one",
                    "claude:two",
                ),
            )
        )
        self.assertNotEqual(first, second)
        self.assertEqual(
            require(self.store.session_context("codex:one")).intent,
            "Cover escape sequences",
        )
        self.assertEqual(
            require(self.store.session_context("claude:two")).intent,
            "Review the parser API",
        )
        self.assertEqual(len(self.store.tracked_artifacts()), 2)

    def test_shared_artifact_observations_update_each_connected_work(self) -> None:
        self.start()
        self.start("claude:two", "Review the parser API")
        self.observe(
            self.observation(
                self.pr(),
                sessions=(
                    "codex:one",
                    "claude:two",
                ),
            )
        )
        self.observe(self.observation(self.pr(state="MERGED"), sessions=()))
        for session in ("codex:one", "claude:two"):
            account = require(self.store.session_context(session))
            pull_request = next(
                item
                for item in account.artifacts
                if item.artifact.kind == "pull_request"
            )
            self.assertEqual(
                next(
                    require(item.payload.pr).state
                    for item in pull_request.evidence
                    if isinstance(item.payload, GitHubEvidence)
                    and item.kind == "observation"
                ),
                "MERGED",
            )
            work = self.store.work_for_session(session)
            assert work is not None
            self.assertEqual(work.status, "active")

    def test_successful_stop_records_execution_without_completing_work(self) -> None:
        work = self.start()
        self.store.session_event(
            "codex:one", HookEvent("codex", "one", "turn_done", outcome="success")
        )
        current_work = self.store.work_for_session("codex:one")
        assert current_work is not None
        self.assertEqual(current_work.status, "active")
        self.assertEqual(
            self.store.session_intent("codex:one", "Check compatibility"), work
        )
        self.assertEqual(self.kinds(), set())

    def test_repository_observation_cannot_create_a_user_request(self) -> None:
        self.observe(self.observation(self.pr()))
        self.assertIsNone(self.store.session_context("codex:one"))
        self.assertEqual(self.kinds(), set())

    def test_verified_current_user_review_request_creates_an_opportunity(self) -> None:
        self.start()
        self.observe(self.observation(self.pr(review_requested=True)))
        self.assertEqual(self.kinds(), {"review_output"})
        self.assertEqual(self.store.opportunities()[0].artifact.ref, self.pr().url)
        self.observe(self.observation(self.pr(review_requested=None)))
        self.assertEqual(self.kinds(), set())

    def test_only_checks_for_current_pr_head_create_check_opportunities(self) -> None:
        self.start()
        self.observe(
            self.observation(
                self.pr(
                    checks=(
                        (
                            "tests",
                            "FAILURE",
                        ),
                    ),
                    check_revision="older",
                )
            )
        )
        self.assertEqual(self.kinds(), set())
        self.observe(
            self.observation(
                self.pr(
                    checks=(
                        (
                            "tests",
                            "FAILURE",
                            "github",
                            "older",
                        ),
                    )
                )
            )
        )
        self.assertEqual(self.kinds(), set())
        self.observe(
            self.observation(
                self.pr(
                    checks=(
                        (
                            "tests",
                            "FAILURE",
                            "github",
                            "head-one",
                        ),
                    )
                )
            )
        )
        self.assertEqual(self.kinds(), {"resolve_checks"})
        self.observe(
            self.observation(
                self.pr(
                    checks=(
                        (
                            "tests",
                            "SUCCESS",
                        ),
                    )
                )
            )
        )
        self.assertEqual(self.kinds(), set())

    def test_draft_and_green_checks_alone_do_not_claim_readiness(self) -> None:
        self.start()
        self.progress()
        self.observe(
            self.observation(
                self.pr(
                    checks=(
                        (
                            "tests",
                            "SUCCESS",
                        ),
                    )
                )
            )
        )
        self.assertEqual(self.kinds(), set())
        work = self.store.work_for_session("codex:one")
        assert work is not None
        self.assertEqual(work.status, "active")

    def test_pr_preparation_requires_observed_output_progress_and_successful_absence_lookup(
        self,
    ) -> None:
        self.start()
        self.observe(self.observation(github_checked_at=0))
        self.assertEqual(self.kinds(), set())
        self.progress()
        self.assertEqual(self.kinds(), set())
        self.observe(self.observation())
        self.assertEqual(self.kinds(), {"prepare_pr"})
        opportunity = self.store.opportunities()[0]
        self.assertIn("may benefit", opportunity.next_step)
        self.assertIn(
            "Agent progress does not establish that the work is complete.",
            opportunity.uncertainty,
        )
        clean = self.observation()
        assert clean.local is not None
        clean = replace(clean, local=replace(clean.local, unstaged=0))
        self.observe(clean)
        self.assertEqual(self.kinds(), set())

    def test_absence_lookup_without_agent_progress_does_not_create_preparation(
        self,
    ) -> None:
        self.start()
        self.observe(self.observation())
        self.assertEqual(self.kinds(), set())

    def test_failed_absence_refresh_preserves_evidence_and_suppresses_pr_preparation(
        self,
    ) -> None:
        self.start()
        self.progress()
        self.observe(self.observation())
        self.assertEqual(self.kinds(), {"prepare_pr"})
        observed_at = self.now
        self.now += 100
        self.observe(
            self.observation(
                github_error="GitHub is unavailable", github_checked_at=observed_at
            )
        )
        self.assertEqual(self.kinds(), set())
        branch = require(self.store.session_context("codex:one")).artifacts[0]
        github = next(record for record in branch.evidence if record.source == "github")
        assert isinstance(github.payload, GitHubEvidence)
        self.assertIsNone(github.payload.pr)
        self.assertEqual(github.observed_at, observed_at)
        self.observe(self.observation())
        self.assertEqual(self.kinds(), {"prepare_pr"})

    def test_review_and_check_opportunities_disclose_failed_source_refresh(
        self,
    ) -> None:
        self.start()
        pull_request = self.pr(
            review_requested=True,
            checks=(
                (
                    "tests",
                    "FAILURE",
                ),
            ),
        )
        self.observe(self.observation(pull_request))
        observed_at = self.now
        self.now += 100
        self.observe(
            self.observation(
                pull_request,
                github_error="GitHub is unavailable",
                local_error="Checkout is unavailable",
            )
        )
        self.assertEqual(self.kinds(), {"review_output", "resolve_checks"})
        for opportunity in self.store.opportunities():
            self.assertIn(
                "GitHub refresh failed. Status comes from the last successful observation.",
                opportunity.uncertainty,
            )
            self.assertIn(
                "Local repository refresh failed. Output information comes from the last successful observation.",
                opportunity.uncertainty,
            )
            self.assertEqual(
                next(
                    record.observed_at
                    for record in opportunity.evidence
                    if record.source == "github" and record.kind == "observation"
                ),
                observed_at,
            )

    def test_inconsistent_independent_pr_branch_field_cannot_attach_unrelated_work(
        self,
    ) -> None:
        self.start()
        self.observe(self.observation())
        self.observe(
            RepositoryObservation(
                subject="fixture",
                repository="github.com/team/project",
                worktree="",
                branch="feature",
                local=None,
                pr=self.pr(head_branch="another", review_requested=True),
                github_checked_at=self.now,
            )
        )
        artifacts = require(self.store.session_context("codex:one")).artifacts
        self.assertEqual([item.artifact.kind for item in artifacts], ["branch"])
        self.assertEqual(self.kinds(), set())

    def test_stack_links_connect_individual_pr_states_and_exact_url_followups(
        self,
    ) -> None:
        self.start()
        link: RelatedLink = RelatedLink(
            kind="pull_request",
            ref=self.pr(8).url,
            relationship="stack_entry",
            source="github_stack",
            stack=4,
            position=1,
            state="OPEN",
            is_draft=True,
            head="head-two",
            head_branch="next",
            title="Expand parser coverage",
        )
        self.observe(self.observation(self.pr(related_links=(link,))))
        self.assertEqual(len(self.store.tracked_artifacts()), 3)
        self.assertEqual(
            next(
                item.title
                for item in self.store.tracked_artifacts()
                if item.ref == self.pr(8).url
            ),
            "Expand parser coverage",
        )
        self.observe(
            RepositoryObservation(
                subject="fixture",
                repository="github.com/team/project",
                worktree="",
                branch="",
                sessions=(),
                local=None,
                pr=self.pr(8, state="MERGED", head="head-two"),
                github_checked_at=self.now,
            )
        )
        account = require(self.store.session_context("codex:one"))
        states = {
            item.artifact.ref: next(
                require(record.payload.pr).state
                for record in item.evidence
                if isinstance(record.payload, GitHubEvidence)
                and record.kind == "observation"
            )
            for item in account.artifacts
            if item.artifact.kind == "pull_request"
        }
        self.assertEqual(states, {self.pr().url: "OPEN", self.pr(8).url: "MERGED"})
        work = self.store.work_for_session("codex:one")
        assert work is not None
        self.assertEqual(work.status, "active")

    def test_stack_peer_check_failure_remains_independent_of_merged_pr(self) -> None:
        self.start()
        link: RelatedLink = RelatedLink(
            kind="pull_request",
            ref=self.pr(8).url,
            relationship="stack_entry",
            source="github_stack",
        )
        self.observe(self.observation(self.pr(state="MERGED", related_links=(link,))))
        self.observe(
            RepositoryObservation(
                subject="fixture",
                repository="github.com/team/project",
                worktree="",
                branch="",
                local=None,
                pr=self.pr(8, checks=(("tests", "FAILURE"),)),
                github_checked_at=self.now,
            )
        )
        self.assertEqual(self.kinds(), {"resolve_checks"})
        self.assertEqual(self.store.opportunities()[0].artifact.ref, self.pr(8).url)

    def test_sibling_state_change_advances_the_primary_opportunity_version(
        self,
    ) -> None:
        self.start()
        link: RelatedLink = RelatedLink(
            kind="pull_request",
            ref=self.pr(8).url,
            relationship="stack_entry",
            source="github_stack",
        )
        self.observe(
            self.observation(self.pr(review_requested=True, related_links=(link,)))
        )
        original = self.store.opportunities()[0]
        self.assertTrue(self.store.is_current(original.id, original.version))
        self.observe(
            RepositoryObservation(
                subject="fixture",
                repository="github.com/team/project",
                worktree="",
                branch="",
                local=None,
                pr=self.pr(8, state="MERGED"),
                github_checked_at=self.now,
            )
        )
        current = self.store.opportunities()[0]
        self.assertEqual(current.id, original.id)
        self.assertGreater(current.version, original.version)
        self.assertFalse(self.store.is_current(original.id, original.version))

    def test_tool_event_names_do_not_rearm_an_unchanged_opportunity(self) -> None:
        self.start()
        self.store.session_event(
            "codex:one", HookEvent("codex", "one", "tool_start", tool_name="Read")
        )
        self.observe(self.observation(self.pr(review_requested=True)))
        original = self.store.opportunities()[0]
        self.memory.record_acceptance(
            uuid.uuid4().hex,
            "Review the parser output.",
            "opportunity",
            original.id,
            original.version,
        )
        self.store.session_event(
            "codex:one",
            HookEvent(
                "codex", "one", "tool_done", tool_name="Write", outcome="success"
            ),
        )
        self.assertEqual(self.kinds(), set())
        stored = self.memory.db.execute(
            "SELECT version FROM work_opportunities WHERE pet=? AND id=?",
            ("fixture", original.id),
        ).fetchone()
        self.assertEqual(stored["version"], original.version)
        self.store.session_event("codex:one", HookEvent("codex", "one", "waiting"))
        self.assertEqual(self.kinds(), {"review_output"})
        self.assertGreater(self.store.opportunities()[0].version, original.version)

    def test_native_aliases_share_session_context_and_survive_restart(self) -> None:
        work = self.store.session_intent(
            "codex:one", "Improve parser", aliases=("codex:alternate",)
        )
        self.store.session_event(
            "codex:one",
            HookEvent(
                "codex",
                "one",
                "progress",
                aliases=("third",),
                progress="Parser changes",
            ),
        )
        self.restart()
        current_work = self.store.work_for_session("codex:third")
        assert current_work is not None
        self.assertEqual(current_work.id, work)
        self.assertEqual(
            require(self.store.session_context("codex:alternate")).intent,
            "Improve parser",
        )
        self.assertIsNone(self.store.work_for_session("third"))

    def test_poll_times_and_nonzero_git_count_changes_do_not_change_meaningful_revision(
        self,
    ) -> None:
        self.start()
        self.progress()
        self.observe(self.observation())
        account = require(self.store.session_context("codex:one"))
        opportunity = self.store.opportunities()[0]
        self.now += 100
        observed = self.observation()
        assert observed.local is not None
        observed = replace(observed, local=replace(observed.local, unstaged=9))
        self.observe(observed)
        self.assertEqual(
            require(self.store.session_context("codex:one")).revision, account.revision
        )
        current = self.store.opportunities()[0]
        self.assertEqual(
            (current.id, current.version, current.changed_at),
            (opportunity.id, opportunity.version, opportunity.changed_at),
        )
        self.assertEqual(
            next(item.observed_at for item in current.evidence if item.source == "git"),
            self.now,
        )

    def test_optional_stack_lookup_errors_do_not_rearm_an_unchanged_suggestion(
        self,
    ) -> None:
        self.start()
        self.observe(
            self.observation(
                self.pr(
                    review_requested=True,
                    related_links_error="Stack lookup unavailable",
                )
            )
        )
        opportunity = self.store.opportunities()[0]
        account = require(self.store.session_context("codex:one"))
        self.memory.record_acceptance(
            uuid.uuid4().hex,
            "Review the parser output.",
            "opportunity",
            opportunity.id,
            opportunity.version,
        )
        self.now += 100
        self.observe(
            self.observation(
                self.pr(
                    review_requested=True,
                    related_links_error="Stack lookup rate limited",
                )
            )
        )
        self.assertEqual(
            require(self.store.session_context("codex:one")).revision, account.revision
        )
        self.assertEqual(self.store.opportunities(), ())

    def test_new_head_changes_context_and_opportunity_revision(self) -> None:
        self.start()
        self.progress()
        self.observe(self.observation())
        account = require(self.store.session_context("codex:one"))
        opportunity = self.store.opportunities()[0]
        observed = self.observation()
        assert observed.local is not None
        observed = replace(observed, local=replace(observed.local, head="head-two"))
        self.observe(observed)
        self.assertNotEqual(
            require(self.store.session_context("codex:one")).revision, account.revision
        )
        self.assertFalse(self.store.is_current(opportunity.id, opportunity.version))

    def test_only_accepted_current_delivery_marks_a_version_surfaced(self) -> None:
        self.start()
        self.observe(self.observation(self.pr(review_requested=True)))
        opportunity = self.store.opportunities()[0]
        self.memory.record_delivery_failure(
            uuid.uuid4().hex,
            "rejected",
            opportunity_id=opportunity.id,
            version=opportunity.version,
        )
        self.memory.record_acceptance(
            uuid.uuid4().hex,
            "Review the parser output.",
            "opportunity",
            opportunity.id,
            opportunity.version + 1,
        )
        self.assertTrue(self.store.is_current(opportunity.id, opportunity.version))
        self.memory.record_acceptance(
            uuid.uuid4().hex,
            "Review the parser output.",
            "opportunity",
            opportunity.id,
            opportunity.version,
        )
        self.restart()
        self.assertEqual(self.kinds(), set())
        self.observe(self.observation(self.pr(review_requested=True)))
        self.assertEqual(self.kinds(), set())

    def test_source_timestamps_survive_cached_observation_replay(self) -> None:
        self.start()
        self.now += 10000
        self.observe(
            self.observation(self.pr(), local_checked_at=500, github_checked_at=600)
        )
        artifacts = require(self.store.session_context("codex:one")).artifacts
        branch = next(item for item in artifacts if item.artifact.kind == "branch")
        times = {
            item.source: item.observed_at
            for item in branch.evidence
            if item.kind == "observation"
        }
        self.assertEqual(times, {"git": 500, "github": 600})

    def test_delayed_open_failure_cannot_replace_a_newer_merged_observation(
        self,
    ) -> None:
        self.start()
        self.observe(
            self.observation(
                self.pr(
                    checks=(
                        (
                            "tests",
                            "FAILURE",
                        ),
                    )
                ),
                github_checked_at=500,
            )
        )
        self.assertEqual(self.kinds(), {"resolve_checks"})
        merged = self.pr(
            state="MERGED", head="head-two", title="Verified parser changes"
        )
        self.observe(
            RepositoryObservation(
                subject="fixture",
                repository="github.com/team/project",
                worktree="",
                branch="",
                local=None,
                pr=merged,
                github_checked_at=700,
            )
        )
        self.assertEqual(self.kinds(), set())
        self.observe(
            self.observation(
                self.pr(
                    checks=(
                        (
                            "tests",
                            "FAILURE",
                        ),
                    )
                ),
                github_checked_at=600,
            )
        )
        self.assertEqual(self.kinds(), set())
        pull_request = next(
            item
            for item in require(self.store.session_context("codex:one")).artifacts
            if item.artifact.kind == "pull_request"
        )
        self.assertEqual(
            (pull_request.artifact.revision, pull_request.artifact.title),
            ("head-two", "Verified parser changes"),
        )
        github = next(
            record
            for record in pull_request.evidence
            if record.source == "github" and record.kind == "observation"
        )
        assert isinstance(github.payload, GitHubEvidence)
        assert github.payload.pr is not None
        self.assertEqual((github.payload.pr.state, github.observed_at), ("MERGED", 700))

    def test_cached_git_observation_cannot_roll_back_a_newer_head_or_verification_time(
        self,
    ) -> None:
        self.start()
        newer = self.observation(local_checked_at=700)
        assert newer.local is not None
        newer = replace(newer, local=replace(newer.local, head="head-two"))
        self.observe(newer)
        self.observe(self.observation(local_checked_at=500))
        branch = next(
            item
            for item in require(self.store.session_context("codex:one")).artifacts
            if item.artifact.kind == "branch"
        )
        self.assertEqual(branch.artifact.revision, "head-two")
        git = next(record for record in branch.evidence if record.source == "git")
        assert isinstance(git.payload, GitObservation)
        self.assertEqual((git.payload.head, git.observed_at), ("head-two", 700))

    def test_worktree_removal_preserves_context_and_independent_pr_updates(
        self,
    ) -> None:
        self.start()
        self.observe(self.observation(self.pr()))
        self.memory.forget_repository("/work/parser")
        self.restart()
        self.observe(
            RepositoryObservation(
                subject="fixture",
                worktree="",
                branch="",
                repository="github.com/team/project",
                local=None,
                pr=self.pr(checks=(("tests", "FAILURE"),)),
                github_checked_at=self.now,
            )
        )
        self.assertEqual(self.kinds(), {"resolve_checks"})
        self.assertEqual(
            len(require(self.store.session_context("codex:one")).artifacts), 2
        )

    def test_fork_checkout_and_independent_pr_poll_share_the_same_pr_identity(
        self,
    ) -> None:
        self.start()
        fork_pr = self.pr(head_repository="github.com/fork/project")
        observation = self.observation(fork_pr, repository="github.com/fork/project")
        assert observation.local is not None
        observation = replace(
            observation,
            local=replace(observation.local, repository="github.com/fork/project"),
        )
        self.observe(observation)
        revision = require(self.store.session_context("codex:one")).revision
        self.observe(
            RepositoryObservation(
                subject="fixture",
                repository="github.com/team/project",
                worktree="",
                branch="",
                local=None,
                pr=fork_pr,
                github_checked_at=self.now,
            )
        )
        self.assertEqual(
            require(self.store.session_context("codex:one")).revision, revision
        )

    def test_prune_preserves_connected_graph_and_current_source_evidence(self) -> None:
        self.start()
        self.progress(value="First parser changes")
        self.progress(value="Validated quoted strings")
        self.observe(self.observation(self.pr(review_requested=True)))
        before = require(self.store.session_context("codex:one"))
        self.now += 31 * 86400
        self.store.prune()
        self.restart()
        self.assertEqual(require(self.store.session_context("codex:one")), before)
        rows = self.memory.db.execute(
            "SELECT * FROM work_evidence WHERE pet=? AND kind='progress'", ("fixture",)
        ).fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual(self.kinds(), {"review_output"})

    def test_artifact_descriptors_reject_empty_references_and_unknown_kinds(
        self,
    ) -> None:
        for artifact in (
            ArtifactDescriptor("branch", "", "github.com/team/project"),
            ArtifactDescriptor("unsupported", "patch"),
        ):
            with self.subTest(artifact=artifact), self.assertRaises(ValueError):
                descriptor(artifact)

    def test_invalid_related_artifact_rolls_back_the_source_observation(self) -> None:
        self.start()
        with self.assertRaises(ValueError):
            self.observe(
                self.observation(
                    self.pr(
                        related_links=(
                            RelatedLink(
                                kind="pull_request",
                                ref="invalid",
                                relationship="stack_entry",
                                source="github_stack",
                            ),
                        )
                    )
                )
            )
        self.assertEqual(require(self.store.session_context("codex:one")).artifacts, ())
        self.assertEqual(self.store.tracked_artifacts(), ())

    def test_runtime_subjects_include_connected_context_and_meaningful_runtime_revision(
        self,
    ) -> None:
        self.start()
        self.observe(self.observation(self.pr(review_requested=True)))
        session = session_snapshot(
            id="codex:one",
            generation=1,
            turn_active=True,
            active_tools=(
                ToolSnapshot(
                    call="fixture-call",
                    tool="pytest",
                    activity="testing",
                    duration_seconds=1,
                ),
            ),
        )
        work = self.store.views(sessions_snapshot(session))
        subject = next(item for item in work if item.kind == "work")
        self.assertEqual(subject.view.intent, "Improve parser")
        self.assertEqual(subject.view.opportunities[0].kind, "review_output")
        session = replace(
            session,
            active_tools=(replace(session.active_tools[0], duration_seconds=99),),
        )
        self.assertEqual(
            next(
                item
                for item in self.store.views(sessions_snapshot(session))
                if item.kind == "work"
            ).revision,
            subject.revision,
        )
        session = replace(session, waiting_for_input=True, waiting_seconds=0)
        self.assertNotEqual(
            next(
                item
                for item in self.store.views(sessions_snapshot(session))
                if item.kind == "work"
            ).revision,
            subject.revision,
        )

    def test_graph_exposes_no_manual_report_or_decision_api(self) -> None:
        self.assertFalse(hasattr(self.store, "record_report"))
        self.assertIsNone(
            self.memory.db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='work_decisions'"
            ).fetchone()
        )

    def test_acceptance_records_the_captured_version_once_after_work_advances(
        self,
    ) -> None:
        self.start()
        self.observe(self.observation(self.pr(review_requested=True)))
        captured = self.store.opportunities()[0]
        self.observe(self.observation(self.pr(review_requested=True, head="head-two")))
        current = self.store.opportunities()[0]
        message_id = uuid.uuid4().hex
        self.assertTrue(
            self.memory.record_delivery_failure(message_id, "acknowledgement timed out")
        )
        self.assertTrue(
            self.memory.record_acceptance(
                message_id,
                "The parser output needs review.",
                "opportunity",
                captured.id,
                captured.version,
            )
        )
        self.assertFalse(
            self.memory.record_acceptance(
                message_id,
                "Repeated acknowledgement",
                "opportunity",
                captured.id,
                captured.version,
            )
        )
        self.assertEqual(self.store.opportunities()[0].version, current.version)
        self.assertEqual(
            self.memory.speech()[0].text, "The parser output needs review."
        )
        outcome = self.memory.message_outcome(message_id)
        assert outcome is not None
        self.assertEqual(outcome.outcome, "accepted")
        rows = self.memory.db.execute(
            "SELECT version,outcome FROM work_surfacings WHERE pet=?", ("fixture",)
        ).fetchall()
        self.assertEqual(
            [dict(row) for row in rows],
            [{"version": captured.version, "outcome": "accepted"}],
        )
        self.restart()
        self.assertFalse(
            self.memory.record_acceptance(
                message_id,
                "Repeated after restart",
                "opportunity",
                captured.id,
                captured.version,
            )
        )
        self.assertTrue(self.store.is_current(current.id, current.version))

    def test_acceptance_rolls_back_speech_outcome_and_surfacing_together(self) -> None:
        self.start()
        self.observe(self.observation(self.pr(review_requested=True)))
        opportunity = self.store.opportunities()[0]
        message_id = uuid.uuid4().hex
        with (
            self.assertRaisesRegex(RuntimeError, "rollback"),
            self.memory.transaction(),
        ):
            self.memory.record_acceptance(
                message_id,
                "Review the parser.",
                "opportunity",
                opportunity.id,
                opportunity.version,
            )
            raise RuntimeError("rollback")
        self.assertEqual(self.memory.speech(), [])
        self.assertIsNone(self.memory.message_outcome(message_id))
        self.assertTrue(self.store.is_current(opportunity.id, opportunity.version))
        self.assertEqual(
            self.memory.db.execute(
                "SELECT COUNT(*) FROM work_surfacings WHERE pet=?", ("fixture",)
            ).fetchone()[0],
            0,
        )

    def test_work_views_group_aliases_and_preserve_distinct_shared_artifact_intents(
        self,
    ) -> None:
        first = self.store.session_intent(
            "codex:one", "Improve parser", aliases=("codex:alias",)
        )
        second = self.start("claude:two", "Review the parser API")
        self.observe(
            self.observation(
                self.pr(),
                sessions=(
                    "codex:one",
                    "claude:two",
                ),
            )
        )
        sessions = [
            session_snapshot(id=name, turn_active=True)
            for name in ("codex:one", "codex:alias", "claude:two", "codex:unbound")
        ]
        view = self.store.views(sessions_snapshot(*sessions))
        self.assertEqual(
            {subject.id for subject in view}, {first, second, "codex:unbound"}
        )
        self.assertEqual(
            next(subject for subject in view if subject.id == first).sessions,
            ("codex:alias", "codex:one"),
        )
        self.assertEqual(
            next(subject for subject in view if subject.id == second).view.intent,
            "Review the parser API",
        )
        self.assertEqual(
            next(subject for subject in view if subject.id == "codex:unbound").kind,
            "session",
        )

    def test_tool_failure_does_not_establish_a_work_obstacle(self) -> None:
        self.start()
        self.store.session_event(
            "codex:one",
            HookEvent("codex", "one", "tool_error", outcome="error", tool_name="Bash"),
        )
        subject = self.store.views(
            sessions_snapshot(
                session_snapshot(
                    id="codex:one",
                    failure=FailureSnapshot(
                        activity="testing",
                        count=1,
                        goal="",
                        resolved=False,
                        seconds_ago=0,
                        current_turn=True,
                    ),
                )
            )
        )[0]
        self.assertFalse(subject.view.current_obstacle)
        self.assertEqual(subject.view.opportunities, ())

    def test_collection_expands_primary_stack_links_without_recursing_through_peers(
        self,
    ) -> None:
        self.start()
        related: RelatedLink = RelatedLink(
            kind="pull_request",
            ref=self.pr(8).url,
            relationship="stack_entry",
            source="github_stack",
        )
        self.observe(self.observation(self.pr(related_links=(related,))))
        targets = {target.ref: target for target in self.store.collection_targets()}
        self.assertTrue(targets[self.pr().url].include_links)
        self.assertFalse(targets[self.pr(8).url].include_links)
        self.assertEqual(targets["feature"].checkout, "/work/parser")
        self.start("claude:two", "Review the follow-up")
        self.observe(self.observation(self.pr(8), sessions=("claude:two",)))
        targets = {target.ref: target for target in self.store.collection_targets()}
        self.assertTrue(targets[self.pr(8).url].include_links)

    def test_retired_checkout_keeps_remote_work_without_resuming_local_collection(
        self,
    ) -> None:
        self.start()
        self.progress()
        self.observe(self.observation())
        self.assertEqual(self.kinds(), {"prepare_pr"})
        self.observe(self.observation(current=False))
        self.assertEqual(self.kinds(), set())
        branch = next(
            target
            for target in self.store.collection_targets()
            if target.kind == "branch"
        )
        self.assertEqual(branch.checkout, "")
        self.assertEqual(branch.ref, "feature")
        self.assertTrue(
            require(self.store.session_context("codex:one")).artifacts[0].evidence
        )
        self.restart()
        self.assertEqual(
            next(
                target
                for target in self.store.collection_targets()
                if target.kind == "branch"
            ).checkout,
            "",
        )

    def test_accepted_action_preserves_work_meaning_until_observed_facts_change(
        self,
    ) -> None:
        self.start()
        self.observe(self.observation(self.pr(review_requested=True)))
        runtime = sessions_snapshot(
            session_snapshot(id="codex:one", turn_active=True, generation=1)
        )
        before = self.store.views(runtime)[0]
        action = before.view.opportunities[0]
        self.memory.record_acceptance(
            uuid.uuid4().hex,
            "The parser output needs review.",
            "opportunity",
            action.id,
            action.version,
        )
        after = self.store.views(runtime)[0]
        self.assertEqual(after.view.opportunities, ())
        self.assertEqual(after.revision, before.revision)
        self.restart()
        self.assertEqual(self.store.views(runtime)[0].revision, before.revision)
        self.observe(self.observation(self.pr(review_requested=False)))
        self.assertNotEqual(self.store.views(runtime)[0].revision, before.revision)

    def test_model_work_projection_keeps_meaning_without_checkout_paths_or_internal_ids(
        self,
    ) -> None:
        self.start()
        self.progress()
        self.observe(self.observation(self.pr()))
        view = self.store.views(sessions_snapshot())[0].view
        rendered = str(work_record(view))
        self.assertIn("Improve parser", rendered)
        self.assertIn("Implemented quoted string handling", rendered)
        self.assertIn("The agent session used this checkout.", rendered)
        self.assertIn(self.pr().url, rendered)
        self.assertNotIn("/work/parser", rendered)
        self.assertNotIn("codex:one", rendered)
        self.assertNotIn("head-one", rendered)
        self.assertIsNone(
            project_evidence(
                EvidenceRecord(
                    "unknown",
                    "observation",
                    IntentEvidence("cli:private", "private value"),
                    self.now,
                )
            )
        )

    def test_pure_work_builder_uses_typed_records_without_persistence(self) -> None:
        self.start()
        runtime = sessions_snapshot(session_snapshot(id="codex:one", turn_active=True))
        state = self.store.snapshot()
        self.memory.close()
        self.assertEqual(build_work(runtime, state)[0].view.intent, "Improve parser")

    def test_hook_and_poll_updates_preserve_unaffected_rows_and_accepted_history(
        self,
    ) -> None:
        work_id = self.start()
        other_work = self.start("claude:two", "Review the parser API")
        self.observe(self.observation(self.pr(review_requested=True)))
        action = self.store.opportunities()[0]
        self.memory.record_acceptance(
            uuid.uuid4().hex,
            "Review the parser output.",
            "opportunity",
            action.id,
            action.version,
        )
        accepted = [
            dict(row)
            for row in self.memory.db.execute(
                "SELECT * FROM work_surfacings WHERE pet=?", ("fixture",)
            )
        ]
        unrelated = [
            dict(row)
            for row in self.memory.db.execute(
                "SELECT * FROM work_evidence WHERE pet=? AND work_id=?",
                ("fixture", other_work),
            )
        ]
        with self.memory.transaction():
            for table in ("work_items", "work_evidence", "work_surfacings"):
                self.memory.db.execute(f"""CREATE TRIGGER preserve_{table} BEFORE DELETE ON {table} BEGIN
                    SELECT RAISE(ABORT, 'unaffected history deleted'); END""")
        self.progress(value="Validated parser compatibility")
        self.store.session_intent("codex:one", "Cover parser compatibility")
        self.now += 30
        self.observe(self.observation(self.pr(review_requested=True)))
        work = self.store.work_for_session("codex:one")
        assert work is not None
        self.assertEqual(work.id, work_id)
        work = self.store.work_for_session("codex:one")
        assert work is not None
        self.assertEqual(
            work.intent,
            "Cover parser compatibility",
        )
        self.assertEqual(
            [
                dict(row)
                for row in self.memory.db.execute(
                    "SELECT * FROM work_surfacings WHERE pet=?", ("fixture",)
                )
            ],
            accepted,
        )
        self.assertEqual(
            [
                dict(row)
                for row in self.memory.db.execute(
                    "SELECT * FROM work_evidence WHERE pet=? AND work_id=?",
                    ("fixture", other_work),
                )
            ],
            unrelated,
        )

    def test_unchanged_graph_save_performs_no_writes(self) -> None:
        self.start()
        self.observe(self.observation(self.pr(review_requested=True)))
        action = self.store.opportunities()[0]
        self.memory.record_acceptance(
            uuid.uuid4().hex,
            "Review the parser output.",
            "opportunity",
            action.id,
            action.version,
        )
        state = self.store.snapshot()
        changed_before = self.memory.db.total_changes
        with self.memory.transaction():
            self.store.repository.save(state)
        self.assertEqual(self.memory.db.total_changes, changed_before)

    def test_identical_poll_updates_verification_time_without_rewriting_evidence(
        self,
    ) -> None:
        self.start()
        self.observe(self.observation(self.pr(review_requested=True)))
        records_before = [
            dict(row)
            for row in self.memory.db.execute(
                "SELECT * FROM work_evidence WHERE pet=? ORDER BY id", ("fixture",)
            )
        ]
        with self.memory.transaction():
            self.memory.db.execute("""CREATE TRIGGER preserve_evidence_data BEFORE UPDATE OF data ON work_evidence BEGIN
                SELECT RAISE(ABORT, 'unchanged evidence rewritten'); END""")
        self.now += 30
        self.observe(self.observation(self.pr(review_requested=True)))
        records_after = [
            dict(row)
            for row in self.memory.db.execute(
                "SELECT * FROM work_evidence WHERE pet=? ORDER BY id", ("fixture",)
            )
        ]
        self.assertEqual(len(records_after), len(records_before))
        for before, after in zip(records_before, records_after):
            self.assertEqual(
                {key: value for key, value in before.items() if key != "timestamp"},
                {key: value for key, value in after.items() if key != "timestamp"},
            )
            self.assertEqual(
                after["timestamp"],
                self.now if after["kind"] == "observation" else before["timestamp"],
            )
