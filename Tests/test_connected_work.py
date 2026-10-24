import random
import tempfile
import unittest
from pathlib import Path

from pet_daemon.attention import Attention
from pet_daemon.config import Config
from pet_daemon.domain import (
    ArtifactKind,
    EvidenceSource,
    ExecutionOutcome,
    LinkRelationship,
    PullRequestState,
    SessionSource,
)
from pet_daemon.identity import SessionIdentity
from pet_daemon.repository_identity import LocalRepository, RemoteRepository
from pet_daemon.runtime import HookEvent, HookEventKind
from pet_daemon.sessions.tracking import Sessions
from pet_daemon.storage import Memory
from pet_daemon.work import WorkSubject
from pet_daemon.work.evidence import ExecutionStatus, ReportedProgress
from pet_daemon.work.models import RelatedLink
from pet_daemon.work.observations import (
    BranchObservation,
    CheckoutObservation,
    GitObservation,
    PullRequest,
    PullRequestObservation,
)
from Tests.fixtures import capture_snapshot, require, sessions_snapshot

REPOSITORY = RemoteRepository(value="github.com/team/project")


class ConnectedWorkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "memory.sqlite3"
        self.now = 1000.0
        self.memory = Memory(self.path, "fixture", clock=lambda: self.now)
        self.sessions = Sessions(lambda: self.now)

    def tearDown(self) -> None:
        self.memory.close()
        self.temporary.cleanup()

    def event(self, event: HookEvent) -> None:
        result = self.sessions.apply(event)
        if result is not None:
            self.memory.work.record_session(result.observation)

    def subjects(self) -> tuple[WorkSubject, ...]:
        return self.memory.connected_work(sessions_snapshot())

    def test_connected_sessions_preserve_same_named_branches_across_repository_kinds(
        self,
    ) -> None:
        repositories = (LocalRepository(path="/local/.git"), REPOSITORY)
        for name, repository in zip(("root", "child"), repositories):
            path = f"/work/{name}"
            self.memory.work.observe(
                CheckoutObservation(
                    worktree=path,
                    local=GitObservation(
                        worktree=path,
                        branch="main",
                        head="head",
                        staged=0,
                        unstaged=0,
                        untracked=0,
                        conflicts=0,
                        upstream="",
                        ahead=0,
                        behind=0,
                        repository=repository,
                    ),
                    checked_at=self.now,
                )
            )
            self.event(
                HookEvent(
                    source=SessionSource.CODEX,
                    session=name,
                    event=HookEventKind.INTENT,
                    goal="Check the shared branch",
                    repo=path,
                    parent="root" if name == "child" else None,
                )
            )
        subjects = self.subjects()
        self.assertEqual(len(subjects), 1)
        branches = [
            artifact
            for artifact in subjects[0].story.artifacts
            if artifact.kind == ArtifactKind.BRANCH
        ]
        self.assertEqual([artifact.ref for artifact in branches], ["main", "main"])
        self.assertEqual(
            [artifact.repository for artifact in branches], list(repositories)
        )

    def stack(self) -> tuple[PullRequest, ...]:
        layers = (
            (31, "schema", "Store session records"),
            (32, "restart", "Restore session records"),
            (33, "examples", "Document restart examples"),
        )
        return tuple(
            PullRequest(
                number=number,
                url=f"https://{REPOSITORY.value}/pull/{number}",
                state=PullRequestState.OPEN,
                merged_at=None,
                head=f"head-{branch}",
                head_repository=REPOSITORY,
                head_branch=branch,
                repository=REPOSITORY,
                title=title,
                related_links=tuple(
                    RelatedLink(
                        kind=ArtifactKind.PULL_REQUEST,
                        ref=f"https://{REPOSITORY.value}/pull/{other_number}",
                        relationship=LinkRelationship.STACK_ENTRY,
                        source=EvidenceSource.GITHUB_STACK,
                        stack=6,
                        position=position,
                    )
                    for position, (other_number, _, _) in enumerate(layers, 1)
                    if other_number != number
                ),
            )
            for number, branch, title in layers
        )

    def assert_family_layout(self, paths: tuple[str, str, str]) -> None:
        prs = self.stack()
        for pr in prs:
            self.memory.work.observe(
                PullRequestObservation(ref=pr.url, pr=pr, checked_at=self.now)
            )
        checked_out = dict(zip(paths, prs))
        for path, pr in checked_out.items():
            self.memory.work.observe(
                CheckoutObservation(
                    worktree=path,
                    local=GitObservation(
                        worktree=path,
                        branch=pr.head_branch,
                        head=pr.head,
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
            )
            self.memory.work.observe(
                BranchObservation(
                    repository=REPOSITORY,
                    branch=pr.head_branch,
                    pr=pr,
                    checked_at=self.now,
                )
            )
        scopes = (
            (
                "root",
                "Preserve session context across restarts",
                "The storage format is ready.",
            ),
            (
                "restore",
                "Implement restart restoration",
                "Restart restoration passes its checks.",
            ),
            (
                "docs",
                "Document restart examples",
                "The restart example explains retained context.",
            ),
        )
        for path, (name, goal, report) in zip(paths, scopes):
            self.event(
                HookEvent(
                    source=SessionSource.CODEX,
                    session=name,
                    event=HookEventKind.INTENT,
                    goal=goal,
                    repo=path,
                    parent="root" if name != "root" else None,
                )
            )
            self.event(
                HookEvent(
                    source=SessionSource.CODEX,
                    session=name,
                    event=HookEventKind.PROGRESS,
                    progress=report,
                )
            )
        subjects = self.subjects()
        self.assertEqual(len(subjects), 1)
        subject = subjects[0]
        self.assertEqual(
            subject.id, SessionIdentity(source=SessionSource.CODEX, native_id="root")
        )
        self.assertEqual(
            subject.sessions,
            (
                SessionIdentity(source=SessionSource.CODEX, native_id="docs"),
                SessionIdentity(source=SessionSource.CODEX, native_id="restore"),
                SessionIdentity(source=SessionSource.CODEX, native_id="root"),
            ),
        )
        record = subject.story
        self.assertEqual(record.intent, scopes[0][1])
        self.assertEqual(
            [
                item.details.reported_progress
                for item in record.activity
                if isinstance(item.details, (ReportedProgress, ExecutionStatus))
                and item.details.reported_progress is not None
            ],
            [scopes[0][2]],
        )
        scoped_sessions = {session.scope: session for session in record.sessions}
        self.assertEqual(set(scoped_sessions), {scopes[1][1], scopes[2][1]})
        for _, goal, report in scopes[1:]:
            self.assertEqual(
                [
                    item.details.reported_progress
                    for item in scoped_sessions[goal].activity
                    if isinstance(item.details, (ReportedProgress, ExecutionStatus))
                    and item.details.reported_progress is not None
                ],
                [report],
            )
        outputs = {
            item.ref: item.title
            for item in record.artifacts
            if item.kind == "pull_request"
        }
        self.assertEqual(outputs, {pr.url: pr.title for pr in prs})

    def test_family_shares_one_stack_when_all_agents_use_one_checkout(self) -> None:
        self.assert_family_layout(("/work/shared", "/work/shared", "/work/shared"))

    def test_family_shares_one_stack_when_each_agent_uses_its_own_checkout(
        self,
    ) -> None:
        self.assert_family_layout(("/work/schema", "/work/restart", "/work/examples"))

    def test_family_shares_one_stack_when_agents_mix_shared_and_separate_checkouts(
        self,
    ) -> None:
        self.assert_family_layout(("/work/shared", "/work/shared", "/work/examples"))

    def test_child_before_parent_keeps_the_root_subject_and_its_own_report(
        self,
    ) -> None:
        self.event(
            HookEvent(
                source=SessionSource.CODEX,
                session="child",
                event=HookEventKind.INTENT,
                parent="root",
                goal="Document restart examples",
            )
        )
        self.event(
            HookEvent(
                source=SessionSource.CODEX,
                session="child",
                event=HookEventKind.PROGRESS,
                progress="The restart example is ready.",
            )
        )
        before = self.subjects()
        self.assertEqual(len(before), 1)
        self.assertEqual(
            before[0].id, SessionIdentity(source=SessionSource.CODEX, native_id="root")
        )
        self.event(
            HookEvent(
                source=SessionSource.CODEX,
                session="root",
                event=HookEventKind.INTENT,
                aliases=("root-conversation",),
                goal="Preserve session context across restarts",
            )
        )
        after = self.subjects()
        self.assertEqual(len(after), 1)
        self.assertEqual(after[0].id, before[0].id)
        record = after[0].story
        self.assertEqual(record.intent, "Preserve session context across restarts")
        self.assertEqual(record.sessions[0].scope, "Document restart examples")
        progress = record.sessions[0].activity[0].details
        assert isinstance(progress, ReportedProgress)
        self.assertEqual(
            progress.reported_progress,
            "The restart example is ready.",
        )

    def test_families_sharing_a_ticket_and_pr_keep_separate_subjects(self) -> None:
        pr = self.stack()[0]
        self.memory.work.observe(
            PullRequestObservation(ref=pr.url, pr=pr, checked_at=self.now)
        )
        ticket = "https://linear.app/team/issue/ENG-42/session-context"
        for name, goal, child_goal in (
            ("storage", "Review session storage", "Check serialized fields"),
            ("restore", "Review restart restoration", "Check restart behavior"),
        ):
            self.event(
                HookEvent(
                    source=SessionSource.CODEX,
                    session=name,
                    event=HookEventKind.INTENT,
                    goal=goal,
                )
            )
            self.event(
                HookEvent(
                    source=SessionSource.CODEX,
                    session=name,
                    event=HookEventKind.PROGRESS,
                    progress=f"Reviewing {pr.url} for {ticket}",
                )
            )
            self.event(
                HookEvent(
                    source=SessionSource.CODEX,
                    session=name + "-child",
                    event=HookEventKind.INTENT,
                    parent=name,
                    goal=child_goal,
                )
            )
        subjects = self.subjects()
        self.assertEqual(
            {subject.id for subject in subjects},
            {
                SessionIdentity(source=SessionSource.CODEX, native_id="storage"),
                SessionIdentity(source=SessionSource.CODEX, native_id="restore"),
            },
        )
        for subject in subjects:
            record = subject.story
            refs = {item.ref for item in record.artifacts}
            self.assertTrue({pr.url, ticket} <= refs)
            expected = (
                "Check serialized fields"
                if subject.id
                == SessionIdentity(source=SessionSource.CODEX, native_id="storage")
                else "Check restart behavior"
            )
            self.assertEqual([session.scope for session in record.sessions], [expected])

    def test_persisted_child_report_is_routine_eligible_after_runtime_pruning(
        self,
    ) -> None:
        sessions = Sessions(lambda: self.now, retention_seconds=1)
        for event in (
            HookEvent(
                source=SessionSource.CODEX,
                session="child",
                event=HookEventKind.INTENT,
                parent="root",
                goal="Document restart examples",
            ),
            HookEvent(
                source=SessionSource.CODEX,
                session="child",
                event=HookEventKind.TURN_DONE,
                progress="The restart example is ready.",
                outcome=ExecutionOutcome.SUCCESS,
            ),
        ):
            sessions.handle(event.packet())
            self.event(event)
        self.now += 2
        sessions.prune()
        self.assertEqual(sessions.activity().sessions, ())
        self.memory.close()
        self.memory = Memory(self.path, "fixture", clock=lambda: self.now)
        runtime = sessions.activity()
        subjects = self.memory.connected_work(runtime)
        self.assertEqual(len(subjects), 1)
        self.assertEqual(
            subjects[0].id,
            SessionIdentity(source=SessionSource.CODEX, native_id="root"),
        )
        self.assertEqual(subjects[0].story.intent, "")
        self.assertEqual(subjects[0].story.requests, ())
        self.assertEqual(subjects[0].story.activity, ())
        config = Config(pet_dir=Path(self.temporary.name), observation_seconds=(1, 1))
        attention = Attention(
            config, lambda: self.now, random.Random(0), wall_clock=lambda: self.now
        )
        self.now += 1
        snapshot = capture_snapshot(runtime, work=subjects)
        request = require(attention.request(snapshot))
        self.assertEqual(request.reason, "routine")
        selection = attention.select(snapshot, request.reason)
        self.assertEqual(
            selection.id, SessionIdentity(source=SessionSource.CODEX, native_id="root")
        )
        self.assertIsNone(selection.focus_session)
        record = require(selection.subject).story
        progress = record.sessions[0].activity[0].details
        assert isinstance(progress, ReportedProgress)
        self.assertEqual(
            progress.reported_progress,
            "The restart example is ready.",
        )
