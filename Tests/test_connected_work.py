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
from pet_daemon.work.evidence import (
    CommentaryEvidence,
    ExecutionStatus,
    ReportedProgress,
)
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
        self.assertEqual(len(subjects), 2)
        branches = {
            subject.id.native_id: [
                artifact
                for artifact in subject.story.artifacts
                if artifact.kind == ArtifactKind.BRANCH
            ]
            for subject in subjects
        }
        for name, repository in zip(("root", "child"), repositories):
            self.assertEqual([artifact.ref for artifact in branches[name]], ["main"])
            self.assertEqual(
                [artifact.repository for artifact in branches[name]], [repository]
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

    def assert_stack_layout(self, paths: tuple[str, str, str]) -> None:
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
        self.assertEqual(len(subjects), 3)
        for subject in subjects:
            name, goal, report = next(
                scope for scope in scopes if scope[0] == subject.id.native_id
            )
            self.assertEqual(subject.sessions, (subject.id,))
            self.assertEqual(
                [update.session for update in subject.updates], [subject.id]
            )
            record = subject.story
            self.assertEqual(record.id, subject.id)
            self.assertEqual(record.intent, goal)
            self.assertEqual(self.progress(record.activity), [report])
            related = {
                session.session.native_id: session for session in record.sessions
            }
            self.assertEqual(
                set(related), {scope[0] for scope in scopes if scope[0] != name}
            )
            for other_name, other_goal, other_report in scopes:
                if other_name != name:
                    self.assertEqual(related[other_name].scope, other_goal)
                    self.assertEqual(
                        self.progress(related[other_name].activity), [other_report]
                    )
            outputs = {
                item.ref: item.title
                for item in record.artifacts
                if item.kind == ArtifactKind.PULL_REQUEST
            }
            self.assertEqual(outputs, {pr.url: pr.title for pr in prs})

    @staticmethod
    def progress(records: tuple[CommentaryEvidence, ...]) -> list[str]:
        return [
            item.details.reported_progress
            for item in records
            if isinstance(item.details, (ReportedProgress, ExecutionStatus))
            and item.details.reported_progress is not None
        ]

    def test_stack_connects_sessions_using_one_checkout(self) -> None:
        self.assert_stack_layout(("/work/shared", "/work/shared", "/work/shared"))

    def test_stack_connects_sessions_using_separate_checkouts(self) -> None:
        self.assert_stack_layout(("/work/schema", "/work/restart", "/work/examples"))

    def test_stack_connects_sessions_using_shared_and_separate_checkouts(self) -> None:
        self.assert_stack_layout(("/work/shared", "/work/shared", "/work/examples"))

    def test_child_keeps_its_subject_when_parent_arrives(
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
            before[0].id, SessionIdentity(source=SessionSource.CODEX, native_id="child")
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
        self.assertEqual(len(after), 2)
        child = next(subject for subject in after if subject.id == before[0].id)
        self.assertEqual(child.revision, before[0].revision)
        self.assertEqual(
            self.progress(child.story.activity), ["The restart example is ready."]
        )
        record = next(
            subject.story for subject in after if subject.id.native_id == "root"
        )
        self.assertEqual(record.intent, "Preserve session context across restarts")
        self.assertEqual(record.sessions[0].scope, "Document restart examples")
        progress = record.sessions[0].activity[0].details
        assert isinstance(progress, ReportedProgress)
        self.assertEqual(
            progress.reported_progress,
            "The restart example is ready.",
        )

    def test_shared_pr_or_ticket_connects_independent_sessions(self) -> None:
        for connection, reference in (
            ("pr", self.stack()[0].url),
            ("ticket", "https://linear.app/team/issue/ENG-42/session-context"),
        ):
            with self.subTest(connection=connection):
                names = (f"storage-{connection}", f"restore-{connection}")
                for name in names:
                    self.event(
                        HookEvent(
                            source=SessionSource.CODEX,
                            session=name,
                            event=HookEventKind.INTENT,
                            goal="Review session context",
                        )
                    )
                    self.event(
                        HookEvent(
                            source=SessionSource.CODEX,
                            session=name,
                            event=HookEventKind.PROGRESS,
                            progress=f"{name} is reviewing {reference}",
                        )
                    )
                for subject in self.subjects():
                    if subject.id.native_id not in names:
                        continue
                    other = next(name for name in names if name != subject.id.native_id)
                    self.assertEqual(
                        [
                            session.session.native_id
                            for session in subject.story.sessions
                        ],
                        [other],
                    )
                    self.assertEqual(
                        self.progress(subject.story.sessions[0].activity),
                        [f"{other} is reviewing {reference}"],
                    )

    def test_parent_links_include_direct_reports(self) -> None:
        for name, parent in (
            ("root", None),
            ("child", "root"),
            ("grandchild", "child"),
        ):
            self.event(
                HookEvent(
                    source=SessionSource.CODEX,
                    session=name,
                    event=HookEventKind.INTENT,
                    parent=parent,
                    goal=f"Work on {name}",
                )
            )
            self.event(
                HookEvent(
                    source=SessionSource.CODEX,
                    session=name,
                    event=HookEventKind.PROGRESS,
                    progress=f"{name} progress",
                )
            )
        root = next(
            subject for subject in self.subjects() if subject.id.native_id == "root"
        )
        self.assertEqual(
            [session.session.native_id for session in root.story.sessions], ["child"]
        )

    def test_related_progress_does_not_change_selected_session_revision(self) -> None:
        ticket = "https://linear.app/team/issue/ENG-42/session-context"
        for name in ("storage", "restore"):
            self.event(
                HookEvent(
                    source=SessionSource.CODEX,
                    session=name,
                    event=HookEventKind.INTENT,
                    goal=f"Implement {ticket}",
                )
            )
            self.event(
                HookEvent(
                    source=SessionSource.CODEX,
                    session=name,
                    event=HookEventKind.PROGRESS,
                    progress=f"{name} progress",
                )
            )
        before = next(
            subject for subject in self.subjects() if subject.id.native_id == "storage"
        )
        self.now += 1
        self.event(
            HookEvent(
                source=SessionSource.CODEX,
                session="restore",
                event=HookEventKind.PROGRESS,
                progress="Restoration checks pass.",
            )
        )
        after = next(subject for subject in self.subjects() if subject.id == before.id)
        self.assertEqual(after.revision, before.revision)
        self.assertEqual(after.updates, before.updates)
        self.assertEqual(after.story.sessions, ())

    def test_related_context_keeps_three_recent_direct_reports(self) -> None:
        self.now = 4900
        self.event(
            HookEvent(
                source=SessionSource.CODEX,
                session="root",
                event=HookEventKind.INTENT,
                goal="Implement session context",
            )
        )
        for name, observed_at in (
            ("stale", 1000),
            *((f"child-{index}", 4995 + index) for index in range(5)),
            ("future", 5001),
        ):
            self.now = observed_at
            self.event(
                HookEvent(
                    source=SessionSource.CODEX,
                    session=name,
                    parent="root",
                    event=HookEventKind.INTENT,
                    goal=f"Implement {name}",
                )
            )
            self.event(
                HookEvent(
                    source=SessionSource.CODEX,
                    session=name,
                    event=HookEventKind.PROGRESS,
                    progress=f"{name} progress",
                )
            )
        self.now = 5000
        self.event(
            HookEvent(
                source=SessionSource.CODEX,
                session="root",
                event=HookEventKind.PROGRESS,
                progress="Selected storage update",
            )
        )
        root = next(
            subject for subject in self.subjects() if subject.id.native_id == "root"
        )
        self.assertEqual(
            [session.session.native_id for session in root.story.sessions],
            ["child-4", "child-3", "child-2"],
        )

    def test_consideration_keeps_related_progress_independently_eligible(
        self,
    ) -> None:
        ticket = "https://linear.app/team/issue/ENG-42/session-context"
        for name in ("storage", "restore"):
            self.event(
                HookEvent(
                    source=SessionSource.CODEX,
                    session=name,
                    event=HookEventKind.INTENT,
                    goal=f"Implement {ticket}",
                )
            )
            self.event(
                HookEvent(
                    source=SessionSource.CODEX,
                    session=name,
                    event=HookEventKind.TURN_DONE,
                    progress=f"{name} completed checks for {ticket}",
                    outcome=ExecutionOutcome.SUCCESS,
                )
            )
        subjects = self.subjects()
        storage = next(
            subject for subject in subjects if subject.id.native_id == "storage"
        )
        restore = next(
            subject for subject in subjects if subject.id.native_id == "restore"
        )
        self.assertEqual(
            [item.session for item in storage.story.sessions], [restore.id]
        )
        config = Config(pet_dir=Path(self.temporary.name), observation_seconds=(1, 1))
        attention = Attention(
            config, lambda: self.now, random.Random(0), wall_clock=lambda: self.now
        )
        snapshot = capture_snapshot(sessions_snapshot(), work=subjects)
        self.assertTrue(attention.is_update_eligible(restore.updates[0], snapshot))
        attention.mark_considered(storage, storage.updates[0])
        self.assertTrue(attention.is_update_eligible(restore.updates[0], snapshot))
        self.assertIsNone(attention.history(restore.id).considered_session_revision)
        self.assertFalse(attention.is_update_eligible(storage.updates[0], snapshot))

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
            SessionIdentity(source=SessionSource.CODEX, native_id="child"),
        )
        self.assertEqual(subjects[0].story.intent, "Document restart examples")
        self.assertEqual(
            [
                item.details.reported_progress
                for item in subjects[0].story.activity
                if isinstance(item.details, ReportedProgress)
            ],
            ["The restart example is ready."],
        )
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
            selection.id, SessionIdentity(source=SessionSource.CODEX, native_id="child")
        )
        self.assertIsNone(selection.focus_session)
        record = require(selection.subject).story
        self.assertEqual(
            set(self.progress(record.activity)), {"The restart example is ready."}
        )
