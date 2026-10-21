import random
import tempfile
import unittest
from pathlib import Path

from pet_daemon.attention import Attention
from pet_daemon.commentary.presentation import work_record
from pet_daemon.config import Config
from pet_daemon.runtime import HookEvent
from pet_daemon.sessions import Sessions
from pet_daemon.storage import Memory
from pet_daemon.work import WorkSubject
from pet_daemon.work.models import RelatedLink
from pet_daemon.work.observations import (
    BranchObservation,
    CheckoutObservation,
    GitObservation,
    PullRequest,
    PullRequestObservation,
)
from Tests.fixtures import capture_snapshot, require, sessions_snapshot

REPOSITORY = "github.com/team/project"


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

    def stack(self) -> tuple[PullRequest, ...]:
        layers = (
            (31, "schema", "Store session records"),
            (32, "restart", "Restore session records"),
            (33, "examples", "Document restart examples"),
        )
        return tuple(
            PullRequest(
                number,
                f"https://{REPOSITORY}/pull/{number}",
                "OPEN",
                None,
                head=f"head-{branch}",
                head_repository=REPOSITORY,
                head_branch=branch,
                repository=REPOSITORY,
                title=title,
                related_links=tuple(
                    RelatedLink(
                        "pull_request",
                        f"https://{REPOSITORY}/pull/{other_number}",
                        "stack_entry",
                        "github_stack",
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
            self.memory.work.observe(PullRequestObservation(pr.url, pr, self.now))
        checked_out = dict(zip(paths, prs))
        for path, pr in checked_out.items():
            self.memory.work.observe(
                CheckoutObservation(
                    path,
                    GitObservation(
                        path,
                        pr.head_branch,
                        pr.head,
                        0,
                        1,
                        0,
                        0,
                        "",
                        0,
                        0,
                        REPOSITORY,
                    ),
                    self.now,
                )
            )
            self.memory.work.observe(
                BranchObservation(REPOSITORY, pr.head_branch, pr, self.now)
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
                    "codex",
                    name,
                    "intent",
                    goal=goal,
                    repo=path,
                    parent="root" if name != "root" else None,
                )
            )
            self.event(HookEvent("codex", name, "progress", progress=report))
        subjects = self.subjects()
        self.assertEqual(len(subjects), 1)
        subject = subjects[0]
        self.assertEqual(subject.id, "codex:root")
        self.assertEqual(
            subject.sessions, ("codex:docs", "codex:restore", "codex:root")
        )
        record = work_record(subject.story)
        self.assertEqual(record["intent"], scopes[0][1])
        self.assertEqual(
            [
                item["reported_progress"]
                for item in record["activity"]
                if "reported_progress" in item
            ],
            [scopes[0][2]],
        )
        scoped_sessions = {session["scope"]: session for session in record["sessions"]}
        self.assertEqual(set(scoped_sessions), {scopes[1][1], scopes[2][1]})
        for _, goal, report in scopes[1:]:
            self.assertEqual(
                [
                    item["reported_progress"]
                    for item in scoped_sessions[goal]["activity"]
                    if "reported_progress" in item
                ],
                [report],
            )
        outputs = {
            item["ref"]: item["title"]
            for item in record["artifacts"]
            if item["kind"] == "pull_request"
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
                "codex",
                "child",
                "intent",
                parent="codex:root",
                goal="Document restart examples",
            )
        )
        self.event(
            HookEvent(
                "codex", "child", "progress", progress="The restart example is ready."
            )
        )
        before = self.subjects()
        self.assertEqual(len(before), 1)
        self.assertEqual(before[0].id, "codex:root")
        self.event(
            HookEvent(
                "codex",
                "root",
                "intent",
                aliases=("root-conversation",),
                goal="Preserve session context across restarts",
            )
        )
        after = self.subjects()
        self.assertEqual(len(after), 1)
        self.assertEqual(after[0].id, before[0].id)
        record = work_record(after[0].story)
        self.assertEqual(record["intent"], "Preserve session context across restarts")
        self.assertEqual(record["sessions"][0]["scope"], "Document restart examples")
        self.assertEqual(
            record["sessions"][0]["activity"][0]["reported_progress"],
            "The restart example is ready.",
        )

    def test_families_sharing_a_ticket_and_pr_keep_separate_subjects(self) -> None:
        pr = self.stack()[0]
        self.memory.work.observe(PullRequestObservation(pr.url, pr, self.now))
        ticket = "https://linear.app/team/issue/ENG-42/session-context"
        for name, goal, child_goal in (
            ("storage", "Review session storage", "Check serialized fields"),
            ("restore", "Review restart restoration", "Check restart behavior"),
        ):
            self.event(HookEvent("codex", name, "intent", goal=goal))
            self.event(
                HookEvent(
                    "codex",
                    name,
                    "progress",
                    progress=f"Reviewing {pr.url} for {ticket}",
                )
            )
            self.event(
                HookEvent(
                    "codex", name + "-child", "intent", parent=name, goal=child_goal
                )
            )
        subjects = self.subjects()
        self.assertEqual(
            {subject.id for subject in subjects}, {"codex:storage", "codex:restore"}
        )
        for subject in subjects:
            record = work_record(subject.story)
            refs = {item["ref"] for item in record["artifacts"]}
            self.assertTrue({pr.url, ticket} <= refs)
            expected = (
                "Check serialized fields"
                if subject.id == "codex:storage"
                else "Check restart behavior"
            )
            self.assertEqual(
                [session["scope"] for session in record["sessions"]], [expected]
            )

    def test_persisted_child_report_is_routine_eligible_after_runtime_pruning(
        self,
    ) -> None:
        sessions = Sessions(lambda: self.now, retention_seconds=1)
        for event in (
            HookEvent(
                "codex",
                "child",
                "intent",
                parent="root",
                goal="Document restart examples",
            ),
            HookEvent(
                "codex",
                "child",
                "turn_done",
                progress="The restart example is ready.",
                outcome="success",
            ),
        ):
            sessions.handle(event.packet())
            self.event(event)
        self.now += 2
        sessions.prune()
        self.assertEqual(sessions.activity_view().sessions, ())
        self.memory.close()
        self.memory = Memory(self.path, "fixture", clock=lambda: self.now)
        runtime = sessions.activity_view()
        subjects = self.memory.connected_work(runtime)
        self.assertEqual(len(subjects), 1)
        self.assertEqual(subjects[0].id, "codex:root")
        self.assertEqual(subjects[0].story.intent, "")
        self.assertEqual(subjects[0].story.requests, ())
        self.assertEqual(subjects[0].story.activity, ())
        config = Config(pet_dir=Path(self.temporary.name), observation_seconds=(1, 1))
        attention = Attention(config, lambda: self.now, random.Random(0))
        self.now += 1
        snapshot = capture_snapshot(runtime, work=subjects)
        request = require(attention.request(snapshot))
        self.assertEqual(request.reason, "routine")
        selection = attention.select(snapshot, request.reason)
        self.assertEqual(selection.id, "codex:root")
        self.assertIsNone(selection.focus_session)
        record = work_record(require(selection.subject).story)
        self.assertEqual(
            record["sessions"][0]["activity"][0]["reported_progress"],
            "The restart example is ready.",
        )
