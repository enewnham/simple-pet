import sqlite3
import tempfile
import unittest
from pathlib import Path

from pet_daemon.domain import (
    ArtifactKind,
    CollectionKind,
    EvidenceSource,
    LinkRelationship,
    PullRequestState,
    SessionSource,
)
from pet_daemon.identity import SessionIdentity
from pet_daemon.repository_identity import LocalRepository, RemoteRepository
from pet_daemon.runtime import HookEvent, HookEventKind
from pet_daemon.sessions.tracking import Sessions
from pet_daemon.sources.hooks import normalize_hook
from pet_daemon.storage.memory import Memory
from pet_daemon.storage.migrations import SCHEMA_VERSION, migrate
from pet_daemon.work.association import reconcile_observation
from pet_daemon.work.evidence import (
    CommentaryEvidence,
    ExecutionStatus,
    PullRequestStatus,
    ReportedProgress,
)
from pet_daemon.work.history import build_session_entry
from pet_daemon.work.models import RelatedLink
from pet_daemon.work.narrative import semantic_revision
from pet_daemon.work.observations import (
    BranchObservation,
    CheckoutObservation,
    GitObservation,
    PullRequest,
    PullRequestObservation,
)
from pet_daemon.work.stories import WorkStory
from Tests.fixtures import require, sessions_snapshot

REPOSITORY = RemoteRepository(value="github.com/team/project")
SESSION = SessionIdentity(source=SessionSource.CODEX, native_id="one")


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
        self,
        event: str,
        session: SessionIdentity = SESSION,
        *,
        aliases: tuple[str, ...] = (),
        **fields: object,
    ) -> None:
        source, name = session.source.value, session.native_id
        hook = require(
            normalize_hook(
                {
                    "source": source,
                    "session": name,
                    "event": event,
                    "aliases": list(aliases),
                    **fields,
                }
            )
        )
        result = self.sessions.apply(hook)
        if result is not None:
            with self.memory.transaction():
                self.store.record_session(result.observation)
                self.memory.record_event(build_session_entry(hook))

    def test_closure_and_replayed_completion_preserve_the_original_result(self) -> None:
        self.event("intent", goal="Accept spaced JSON", repo="/work/parser")
        self.now += 1
        self.event("turn_done", progress="Spaced JSON parses successfully.")
        original = self.memory.connected_work(sessions_snapshot())[0]
        original_update = original.updates[0]
        self.now += 20
        self.event("session_end")
        closed = self.memory.connected_work(sessions_snapshot())[0]
        self.assertEqual(closed.updates[0], original_update)
        self.assertEqual(
            semantic_revision(closed.story), semantic_revision(original.story)
        )
        self.assertEqual(closed.changed_at, original.changed_at)
        self.sessions = Sessions(lambda: self.now)
        self.now += 20
        self.store.observe(self.checkout(branch="different-work"))
        self.event(
            "turn_done", progress="The duplicate hook describes the same finished turn."
        )
        replayed = self.memory.connected_work(sessions_snapshot())[0]
        self.assertEqual(replayed.updates[0].revision, original_update.revision)
        self.assertEqual(replayed.updates[0].ended_at, original_update.ended_at)
        self.assertEqual(
            semantic_revision(replayed.story), semantic_revision(original.story)
        )
        result = next(
            record.details
            for record in replayed.story.activity
            if isinstance(record.details, ExecutionStatus)
        )
        self.assertEqual(result.reported_execution_outcome, "success")

    def test_repeated_request_establishes_new_work_after_closure(self) -> None:
        self.event("intent", goal="Accept spaced JSON")
        self.now += 1
        self.event("turn_done", progress="Spaced JSON parses successfully.")
        original = self.memory.connected_work(sessions_snapshot())[0].updates[0]
        self.event("session_end")
        self.now += 10
        self.event("intent", goal="Accept spaced JSON")
        active = self.memory.connected_work(sessions_snapshot())[0].updates[0]
        self.assertNotEqual(active.revision, original.revision)
        self.assertIsNone(active.ended_at)
        active_story = self.memory.connected_work(sessions_snapshot())[0].story
        self.assertEqual(active_story.requests[-1].observed_at, self.now)
        self.now += 1
        self.event("turn_done", progress="Spaced JSON parses successfully.")
        completed = self.memory.connected_work(sessions_snapshot())[0].updates[0]
        self.assertNotEqual(completed.revision, original.revision)
        self.assertEqual(completed.ended_at, self.now)

    def checkout(
        self, branch: str = "feature", path: str = "/work/parser"
    ) -> CheckoutObservation:
        return CheckoutObservation(
            worktree=path,
            local=GitObservation(
                worktree=path,
                branch=branch,
                head="head",
                staged=0,
                unstaged=1,
                untracked=0,
                conflicts=0,
                upstream="",
                ahead=0,
                behind=0,
                repository=REPOSITORY,
            ),
            checked_at=self.now,
        )

    def pr(self, number: int = 7) -> PullRequest:
        return PullRequest(
            number=number,
            url=f"https://github.com/team/project/pull/{number}",
            state=PullRequestState.OPEN,
            merged_at=None,
            is_draft=True,
            head="head",
            head_repository=REPOSITORY,
            head_branch="feature",
            repository=REPOSITORY,
            title="Handle quotes",
        )

    def branch(self, number: int = 7, branch: str = "feature") -> BranchObservation:
        return BranchObservation(
            repository=REPOSITORY,
            branch=branch,
            pr=self.pr(number).model_copy(update={"head_branch": branch}),
            checked_at=self.now,
        )

    def story(self, session: SessionIdentity = SESSION) -> WorkStory:
        return next(
            subject.story
            for subject in self.memory.connected_work(sessions_snapshot())
            if session in subject.sessions
        )

    def reports(
        self, session: SessionIdentity = SESSION
    ) -> tuple[CommentaryEvidence, ...]:
        return self.story(session).activity

    def test_child_keeps_its_scope_and_report_with_the_parent_story(self) -> None:
        self.event(
            "intent",
            SessionIdentity(source=SessionSource.CODEX, native_id="child"),
            goal="Cover escapes",
            parent="root",
            repo="/work/child",
        )
        self.event(
            "intent",
            SessionIdentity(source=SessionSource.CODEX, native_id="root"),
            goal="Improve parser",
            repo="/work/root",
        )
        self.event(
            "progress",
            SessionIdentity(source=SessionSource.CODEX, native_id="child"),
            progress="Escape tests pass",
        )
        story = self.story(
            SessionIdentity(source=SessionSource.CODEX, native_id="root")
        )
        self.assertEqual(story.intent, "Improve parser")
        self.assertEqual(
            [record.details for record in story.activity],
            [
                ExecutionStatus(
                    execution="in progress",
                    reported_progress=None,
                    reported_execution_outcome=None,
                )
            ],
        )
        self.assertEqual(story.sessions[0].scope, "Cover escapes")
        self.assertEqual(require(story.sessions[0].context).checkout, "/work/child")
        self.assertEqual(
            self.story(SessionIdentity(source=SessionSource.CODEX, native_id="root"))
            .sessions[0]
            .activity[0]
            .details,
            ReportedProgress(reported_progress="Escape tests pass"),
        )

    def test_refined_intent_and_alias_reports_share_one_story(self) -> None:
        self.event("intent", goal="Improve parser")
        self.event("intent", goal="Cover escapes", aliases=("alias",))
        self.event(
            "progress",
            SessionIdentity(source=SessionSource.CODEX, native_id="alias"),
            progress="Covered escapes",
        )
        subjects = self.memory.connected_work(sessions_snapshot())
        self.assertEqual(
            tuple(subject.id for subject in subjects),
            (SessionIdentity(source=SessionSource.CODEX, native_id="one"),),
        )
        self.assertEqual(subjects[0].story.intent, "Cover escapes")
        self.assertEqual(
            subjects[0].story.activity[0].details,
            ReportedProgress(reported_progress="Covered escapes"),
        )

    def test_explicit_alias_keeps_one_session_with_the_latest_report(
        self,
    ) -> None:
        sessions = Sessions(lambda: self.now)

        def ingest(hook: HookEvent) -> SessionIdentity:
            result = require(sessions.apply(hook))
            with self.memory.transaction():
                self.store.record_session(result.observation)
                self.memory.record_event(build_session_entry(hook))
            return result.session

        ingest(
            HookEvent(
                source=SessionSource.CODEX,
                session="a",
                event=HookEventKind.SESSION_START,
                repo="/work/a",
                aliases=("old-a",),
            )
        )
        ingest(
            HookEvent(
                source=SessionSource.CODEX,
                session="b",
                event=HookEventKind.SESSION_START,
                repo="/work/b",
                aliases=("old-b",),
            )
        )
        ingest(
            HookEvent(
                source=SessionSource.CODEX,
                session="b",
                event=HookEventKind.INTENT,
                goal="Initial parser goal",
            )
        )
        self.now += 1
        ingest(
            HookEvent(
                source=SessionSource.CODEX,
                session="a",
                event=HookEventKind.INTENT,
                goal="Refined parser goal",
            )
        )
        ingest(
            HookEvent(
                source=SessionSource.CODEX,
                session="a",
                event=HookEventKind.PROGRESS,
                progress="First parser report",
                repository=REPOSITORY.value,
                branch="first",
                pull_request=self.pr(7).url,
            )
        )
        self.now += 1
        ingest(
            HookEvent(
                source=SessionSource.CODEX,
                session="b",
                event=HookEventKind.PROGRESS,
                progress="Second parser report",
                repo="/work/latest",
                repository=REPOSITORY.value,
                branch="second",
                pull_request=self.pr(8).url,
            )
        )
        ingest(
            HookEvent(
                source=SessionSource.CODEX, session="b", event=HookEventKind.WAITING
            )
        )
        ingest(
            HookEvent(
                source=SessionSource.CODEX,
                session="child",
                event=HookEventKind.SUBAGENT_START,
                parent="a",
                goal="Check escapes",
            )
        )
        self.assertEqual(
            ingest(
                HookEvent(
                    source=SessionSource.CODEX,
                    session="b",
                    event=HookEventKind.SESSION_START,
                    aliases=("a",),
                )
            ),
            SessionIdentity(source=SessionSource.CODEX, native_id="b"),
        )
        subjects = self.memory.connected_work(sessions.activity())
        self.assertEqual(
            {subject.id for subject in subjects},
            {
                SessionIdentity(source=SessionSource.CODEX, native_id="b"),
                SessionIdentity(source=SessionSource.CODEX, native_id="child"),
            },
        )
        story = next(
            subject.story for subject in subjects if subject.id.native_id == "b"
        )
        self.assertEqual(story.intent, "Refined parser goal")
        child = next(subject for subject in subjects if subject.id.native_id == "child")
        self.assertEqual(child.story.intent, "Check escapes")
        self.assertEqual(child.sessions, (child.id,))
        reports = [
            record.details
            for record in story.activity
            if isinstance(record.details, ReportedProgress)
        ]
        self.assertEqual(
            reports, [ReportedProgress(reported_progress="Second parser report")]
        )
        self.assertTrue(
            any(
                record.details.progress == "First parser report"
                for record in self.memory.recent_events()
            )
        )
        self.assertEqual(
            {item.ref for item in story.artifacts if item.kind == "pull_request"},
            {self.pr(7).url, self.pr(8).url},
        )
        self.memory.close()
        self.memory = Memory(self.path, "fixture", clock=lambda: self.now)
        self.store = self.memory.work
        self.assertEqual(self.memory.connected_work(sessions.activity()), subjects)

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

    def test_local_repository_identity_survives_restart(self) -> None:
        repository = LocalRepository(path="/work/parser/.git")
        checkout = self.checkout()
        self.store.observe(
            checkout.model_copy(
                update={
                    "local": require(checkout.local).model_copy(
                        update={"repository": repository}
                    )
                }
            )
        )
        branch = BranchObservation(
            repository=repository, branch="feature", pr=None, checked_at=self.now
        )
        self.store.observe(branch)
        self.event("intent", goal="Improve parser", repo="/work/parser")
        self.event("progress", progress="Feature tests pass")
        self.memory.close()
        self.memory = Memory(self.path, "fixture", clock=lambda: self.now)
        story = self.memory.connected_work(sessions_snapshot())[0].story
        self.assertEqual(story.artifacts[0].repository, repository)
        self.assertIn(
            branch.model_copy(update={"last_success_at": self.now}),
            self.memory.load_repositories(),
        )
        self.assertEqual(require(story.requests[0].context).repository, repository)
        self.assertEqual(
            next(
                target.repository
                for target in self.memory.collection_targets()
                if target.kind == CollectionKind.BRANCH
            ),
            repository,
        )

    def test_unknown_report_context_stays_unknown_after_checkout_discovery(
        self,
    ) -> None:
        self.event("intent", goal="Improve parser", repo="/work/parser")
        self.event("progress", progress="Feature tests pass")
        self.store.observe(self.checkout())
        self.store.observe(self.branch())
        context = require(self.reports()[0].context)
        self.assertEqual(context.checkout, "/work/parser")
        self.assertIsNone(context.repository)
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
            repository=REPOSITORY.value,
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
        history = self.memory.recent_events()
        self.now += 30
        self.store.observe(self.checkout())
        self.store.observe(self.branch())
        self.assertEqual(before.revision, self.story().revision)
        self.assertEqual(self.memory.recent_events(), history)
        checkpoints = self.memory.load_repositories()
        self.assertEqual({item.checked_at for item in checkpoints}, {self.now})

    def test_stale_response_cannot_replace_newer_current_facts(self) -> None:
        fresh = self.branch(8)
        self.store.observe(fresh)
        self.store.observe(
            self.branch(7).model_copy(update={"checked_at": self.now - 1})
        )
        self.store.observe(
            PullRequestObservation(
                ref=self.pr(8).url,
                pr=self.pr(8).model_copy(update={"state": PullRequestState.MERGED}),
                checked_at=self.now + 1,
            )
        )
        self.store.observe(
            PullRequestObservation(
                ref=self.pr(8).url, pr=self.pr(8), checked_at=self.now
            )
        )
        self.event(
            "intent",
            goal="Improve parser",
            repository=REPOSITORY.value,
            branch="feature",
        )
        story = self.story()
        outputs = [
            evidence.details
            for artifact in story.artifacts
            for evidence in artifact.evidence
            if isinstance(evidence.details, PullRequestStatus)
        ]
        self.assertEqual(
            [(output.url, output.state) for output in outputs],
            [(self.pr(8).url, "MERGED")],
        )

    def test_failed_lookup_preserves_last_known_pull_request_with_uncertainty(
        self,
    ) -> None:
        self.event(
            "intent",
            goal="Improve parser",
            repository=REPOSITORY.value,
            branch="feature",
        )
        self.store.observe(self.branch())
        self.now += 1
        self.store.observe(
            BranchObservation(
                repository=REPOSITORY,
                branch="feature",
                pr=None,
                checked_at=self.now,
                error="network unavailable",
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
            CheckoutObservation(
                worktree="/work/parser",
                local=None,
                checked_at=self.now,
                error="local failed",
            )
        )
        self.store.observe(
            BranchObservation(
                repository=REPOSITORY,
                branch="feature",
                pr=None,
                checked_at=self.now,
                error="branch failed",
            )
        )
        self.store.observe(
            PullRequestObservation(
                ref=self.pr().url, pr=None, checked_at=self.now, error="PR failed"
            )
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
        history = self.memory.recent_events()
        self.now += 30
        self.store.observe(
            CheckoutObservation(
                worktree="/work/parser",
                local=None,
                checked_at=self.now,
                error="local failed",
            )
        )
        self.store.observe(
            BranchObservation(
                repository=REPOSITORY,
                branch="feature",
                pr=None,
                checked_at=self.now,
                error="branch failed",
            )
        )
        self.store.observe(
            PullRequestObservation(
                ref=self.pr().url, pr=None, checked_at=self.now, error="PR failed"
            )
        )
        self.assertEqual(self.memory.recent_events(), history)
        before = self.story()
        self.store.observe(
            PullRequestObservation(
                ref=self.pr().url, pr=None, checked_at=1040, error="stale error"
            )
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
            PullRequestObservation(
                ref=self.pr().url, pr=None, checked_at=self.now, error="network failed"
            )
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
            after.activity[0].details,
            ExecutionStatus(
                execution="in progress",
                reported_progress=None,
                reported_execution_outcome=None,
            ),
        )
        self.assertNotIn("Raw command", str(after))
        self.assertNotIn("Raw command", str(self.memory.recent_events()))

    def test_unrelated_tool_result_keeps_waiting_and_story(self) -> None:
        self.event("intent", goal="Improve parser")
        self.event("waiting")
        before = self.story()
        self.event("tool_done", call="unrelated", progress="Raw output")
        self.assertEqual(before, self.story())

    def test_child_and_parent_reports_retain_distinct_scopes(self) -> None:
        self.event(
            "intent",
            SessionIdentity(source=SessionSource.CODEX, native_id="child"),
            parent="root",
            goal="Cover escapes",
        )
        self.event(
            "progress",
            SessionIdentity(source=SessionSource.CODEX, native_id="child"),
            progress="Child tests pass",
        )
        self.event(
            "intent",
            SessionIdentity(source=SessionSource.CODEX, native_id="root"),
            goal="Improve parser",
        )
        self.event(
            "progress",
            SessionIdentity(source=SessionSource.CODEX, native_id="root"),
            progress="Parent tests pass",
        )
        self.assertEqual(
            self.story(SessionIdentity(source=SessionSource.CODEX, native_id="root"))
            .sessions[0]
            .activity[0]
            .details,
            ReportedProgress(reported_progress="Child tests pass"),
        )
        self.assertEqual(
            self.reports(SessionIdentity(source=SessionSource.CODEX, native_id="root"))[
                0
            ].details,
            ReportedProgress(reported_progress="Parent tests pass"),
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
            "intent",
            goal="Improve parser",
            repository=REPOSITORY.value,
            branch="feature",
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
        history = self.memory.recent_events()
        with self.assertRaises(sqlite3.IntegrityError):
            self.event("progress", repo="/work/other", progress="Must roll back")
        self.assertEqual(self.story(), before)
        self.assertEqual(self.memory.recent_events(), history)

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
        self.event(
            "intent",
            SessionIdentity(source=SessionSource.CODEX, native_id="root"),
            goal="Improve parser",
            repo="/work/root",
        )
        self.event(
            "subagent_start",
            SessionIdentity(source=SessionSource.CODEX, native_id="old"),
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
            SessionIdentity(source=SessionSource.CODEX, native_id="native"),
            aliases=("old",),
            progress="Escape cases pass",
        )

        subjects = self.memory.connected_work(self.sessions.activity())
        self.assertEqual(
            {subject.id for subject in subjects},
            {
                SessionIdentity(source=SessionSource.CODEX, native_id="root"),
                SessionIdentity(source=SessionSource.CODEX, native_id="native"),
            },
        )
        child = next(
            subject.story for subject in subjects if subject.id.native_id == "native"
        )
        self.assertEqual(child.intent, "Cover escapes")
        report = next(
            record
            for record in child.activity
            if record.details == ReportedProgress(reported_progress="Escape cases pass")
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
            "intent",
            SessionIdentity(source=SessionSource.CODEX, native_id="two"),
            goal="Improve other code",
            repo="/work/parser",
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
            kind=ArtifactKind.PULL_REQUEST,
            ref=self.pr(8).url,
            relationship=LinkRelationship.STACK_ENTRY,
            source=EvidenceSource.GITHUB_STACK,
        )
        first = self.pr().model_copy(update={"related_links": (link,)})
        self.store.observe(
            PullRequestObservation(ref=first.url, pr=first, checked_at=self.now)
        )
        self.event("intent", goal=f"Review {first.url}")
        self.now += 1
        failed = first.model_copy(
            update={
                "state": PullRequestState.MERGED,
                "related_links": (),
                "related_links_error": "stack lookup unavailable",
            }
        )
        self.store.observe(
            PullRequestObservation(ref=first.url, pr=failed, checked_at=self.now)
        )
        output = next(item for item in self.story().artifacts if item.ref == first.url)
        self.assertTrue(
            any(
                isinstance(record.details, PullRequestStatus)
                and record.details.state == "MERGED"
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
                ref=first.url,
                pr=failed.model_copy(update={"related_links_error": None}),
                checked_at=self.now,
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
            CheckoutObservation(
                worktree="/work/parser",
                local=None,
                checked_at=self.now,
                available=False,
            )
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


class ObservationDecisionTests(unittest.TestCase):
    def test_failed_stack_refresh_preserves_both_input_observations(self) -> None:
        link = RelatedLink(
            kind=ArtifactKind.PULL_REQUEST,
            ref="https://github.com/team/project/pull/2",
            relationship=LinkRelationship.STACK_ENTRY,
            source=EvidenceSource.GITHUB_STACK,
        )
        previous_pr = PullRequest(
            number=1,
            url="https://github.com/team/project/pull/1",
            state=PullRequestState.OPEN,
            merged_at=None,
            repository=REPOSITORY,
            related_links=(link,),
        )
        refreshed_pr = previous_pr.model_copy()
        refreshed_pr.related_links = ()
        refreshed_pr.related_links_error = "unavailable"
        previous = BranchObservation(
            repository=REPOSITORY,
            branch="main",
            pr=previous_pr,
            checked_at=1,
            last_success_at=1,
            resolutions=(previous_pr.url,),
        )
        refreshed = BranchObservation(
            repository=REPOSITORY,
            branch="main",
            pr=refreshed_pr,
            checked_at=2,
        )
        decided = require(reconcile_observation(previous, refreshed))
        self.assertIsInstance(decided, BranchObservation)
        assert isinstance(decided, BranchObservation)
        self.assertEqual(require(decided.pr).related_links, (link,))
        self.assertEqual(decided.last_success_at, 2)
        self.assertEqual(refreshed_pr.related_links, ())
        self.assertIsNone(refreshed.last_success_at)
        self.assertEqual(previous.last_success_at, 1)
        self.assertIsNone(previous_pr.related_links_error)
