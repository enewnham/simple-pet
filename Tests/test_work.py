import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from pet_daemon.runtime import HookEvent, ToolSnapshot
from pet_daemon.sessions import Sessions
from pet_daemon.storage import Memory
from pet_daemon.work import build_work
from pet_daemon.work.models import (
    GitHubEvidence,
    IntentEvidence,
    ProgressEvidence,
    PullRequestAssociation,
)
from pet_daemon.work.observations import (
    GitObservation,
    PullRequest,
    RepositoryObservation,
)
from pet_daemon.work.presentation import work_record
from Tests.fixtures import require, session_snapshot, sessions_snapshot


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

    def progress(self, text: str, session: str = "codex:one") -> None:
        harness, name = session.split(":", 1)
        self.store.session_event(
            session, HookEvent(harness, name, "progress", progress=text)
        )

    def pr(self, number: int = 7) -> PullRequest:
        return PullRequest(
            number=number,
            url=f"https://github.com/team/project/pull/{number}",
            state="OPEN",
            merged_at=None,
            is_draft=True,
            head="head-one",
            head_repository="github.com/team/project",
            head_branch="feature",
            repository="github.com/team/project",
            title="Handle parser quotes",
            body="Recognize quoted strings.",
        )

    def observation(
        self, number: int = 7, session: str = "codex:one"
    ) -> RepositoryObservation:
        return RepositoryObservation(
            subject="fixture",
            repository="github.com/team/project",
            worktree="/work/parser",
            branch="feature",
            sessions=(session,),
            pr_sessions=(session,),
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
            pr=self.pr(number),
            local_checked_at=self.now,
            github_checked_at=self.now,
        )

    def restart(self) -> None:
        self.memory.close()
        self.memory = Memory(self.path, "fixture", clock=lambda: self.now)
        self.store = self.memory.work

    def test_work_preserves_reported_story_and_artifact_association(self) -> None:
        identifier = self.start()
        self.progress("Implemented quoted strings. Next, cover escapes.")
        self.store.observe_repository(self.observation())
        self.restart()
        account = require(self.store.session_context("codex:one"))
        self.assertEqual(account.intent, "Improve parser")
        self.assertEqual(
            account.requests[0].payload, IntentEvidence("codex:one", "Improve parser")
        )
        self.assertEqual(
            account.activity[0].payload,
            ProgressEvidence(
                "codex:one", "Implemented quoted strings. Next, cover escapes."
            ),
        )
        self.assertEqual(
            {item.artifact.kind for item in account.artifacts},
            {"branch", "pull_request"},
        )
        pull_request = next(
            item for item in account.artifacts if item.artifact.kind == "pull_request"
        )
        self.assertTrue(
            any(
                isinstance(item.payload, PullRequestAssociation)
                for item in pull_request.evidence
            )
        )
        self.assertEqual(
            require(self.store.work_for_session("codex:one")).id, identifier
        )
        presented = work_record(self.store.work_view(identifier))
        output = next(
            item for item in presented["artifacts"] if item["kind"] == "pull_request"
        )
        self.assertIn('"state": "OPEN"', json.dumps(output))
        self.assertNotIn("checks", json.dumps(presented))
        self.assertNotIn("review_requested", json.dumps(presented))

    def test_followup_intent_refines_work_and_aliases_share_membership(self) -> None:
        identifier = self.start()
        self.assertEqual(
            self.store.session_intent("codex:one", "Cover escapes", ("codex:alias",)),
            identifier,
        )
        other = self.start("claude:two", "Review parser")
        self.assertEqual(
            require(self.store.work_for_session("codex:alias")).id, identifier
        )
        self.assertNotEqual(other, identifier)
        self.assertEqual(
            require(self.store.work_context(identifier)).intent, "Cover escapes"
        )

    def test_reused_branch_only_associates_new_pr_with_fresh_scoped_sessions(
        self,
    ) -> None:
        first = self.start()
        self.store.observe_repository(self.observation())
        second = self.start("codex:two", "Improve escapes")
        self.now += 30
        self.store.observe_repository(self.observation(8, "codex:two"))
        first_refs = {
            item.artifact.ref
            for item in require(self.store.work_context(first)).artifacts
        }
        second_refs = {
            item.artifact.ref
            for item in require(self.store.work_context(second)).artifacts
        }
        self.assertIn(self.pr(7).url, first_refs)
        self.assertNotIn(self.pr(8).url, first_refs)
        self.assertIn(self.pr(8).url, second_refs)
        self.assertNotIn(self.pr(7).url, second_refs)

    def test_local_session_does_not_claim_retained_pr_without_pr_scope(self) -> None:
        first = self.start()
        self.store.observe_repository(self.observation())
        second = self.start("codex:two", "Improve escapes")
        self.store.observe_repository(
            replace(self.observation(session="codex:two"), pr_sessions=("codex:one",))
        )
        self.assertEqual(
            {
                item.artifact.kind
                for item in require(self.store.work_context(second)).artifacts
            },
            {"branch"},
        )
        self.assertIn(
            self.pr().url,
            {
                item.artifact.ref
                for item in require(self.store.work_context(first)).artifacts
            },
        )

    def test_exact_pr_link_keeps_refreshing_without_session_claims(self) -> None:
        identifier = self.start()
        self.store.observe_repository(self.observation())
        self.now += 10
        observation = replace(
            self.observation(),
            sessions=(),
            pr_sessions=(),
            current=False,
            pr=replace(self.pr(), state="MERGED"),
        )
        self.store.observe_repository(observation)
        account = require(self.store.work_context(identifier))
        pull_request = next(
            item for item in account.artifacts if item.artifact.kind == "pull_request"
        )
        observation = next(
            item.payload
            for item in pull_request.evidence
            if isinstance(item.payload, GitHubEvidence)
        )
        self.assertEqual(require(observation.pr).state, "MERGED")

    def test_repository_metadata_changes_context_without_changing_story_revision(
        self,
    ) -> None:
        identifier = self.start()
        session = session_snapshot(id="codex:one", goal="Improve parser")
        before = self.store.views(sessions_snapshot(session))[0]
        self.store.observe_repository(self.observation())
        self.now += 10
        self.store.observe_repository(
            replace(
                self.observation(),
                pr=replace(self.pr(), state="MERGED", head="head-two"),
            )
        )
        after = self.store.views(sessions_snapshot(session))[0]
        self.assertEqual(before.revision, after.revision)
        self.assertEqual(before.view.revision, after.view.revision)
        self.assertNotEqual(before.view.artifacts, after.view.artifacts)
        self.progress("Quoted strings are covered")
        self.assertNotEqual(
            after.revision, self.store.views(sessions_snapshot(session))[0].revision
        )
        self.assertEqual(identifier, after.id)

    def test_tool_events_remain_outside_persisted_work_story(self) -> None:
        self.start()
        before = require(self.store.session_context("codex:one"))
        for event in ("tool_start", "tool_done", "tool_error"):
            self.store.session_event(
                "codex:one",
                HookEvent(
                    "codex",
                    "one",
                    event,
                    tool="pytest",
                    progress="Raw command output",
                    outcome="error",
                ),
            )
        self.assertEqual(before, self.store.session_context("codex:one"))
        self.assertEqual(
            self.memory.db.execute(
                "SELECT COUNT(*) FROM work_evidence WHERE source='harness'"
            ).fetchone()[0],
            0,
        )
        base = session_snapshot(id="codex:one", turn_active=True)
        active = replace(
            base,
            last_tool="pytest",
            phase="testing",
            active_tools=(ToolSnapshot("call", "pytest", "testing", 5),),
        )
        self.assertEqual(
            self.store.views(sessions_snapshot(base))[0].revision,
            self.store.views(sessions_snapshot(active))[0].revision,
        )

    def test_authoritative_resume_clears_persisted_input_request_without_command_facts(
        self,
    ) -> None:
        for event in (
            HookEvent("codex", "one", "tool_start", call="answer", tool="Read"),
            HookEvent("codex", "one", "progress", progress="Continuing parser work"),
            HookEvent("codex", "one", "intent", goal="Cover escapes"),
        ):
            with self.subTest(event=event.event):
                self.start()
                sessions = Sessions(lambda: self.now)
                sessions.apply(
                    HookEvent("codex", "one", "intent", goal="Improve parser")
                )
                waiting = HookEvent("codex", "one", "waiting", call="question")
                sessions.apply(waiting)
                self.store.session_event(
                    "codex:one",
                    waiting,
                    waiting_for_input=sessions.lanes["codex:one"].waiting,
                )
                self.assertIn(
                    "input requested",
                    json.dumps(
                        work_record(require(self.store.session_view("codex:one")))
                    ),
                )
                sessions.apply(event)
                self.assertFalse(sessions.lanes["codex:one"].waiting)
                self.store.session_event(
                    "codex:one",
                    event,
                    waiting_for_input=sessions.lanes["codex:one"].waiting,
                )
                rendered = json.dumps(
                    work_record(require(self.store.session_view("codex:one")))
                )
                self.assertNotIn("input requested", rendered)
                self.assertIn("in progress", rendered)
                self.assertEqual(
                    self.memory.db.execute(
                        "SELECT COUNT(*) FROM work_evidence WHERE kind='execution' AND data LIKE '%tool_start%'"
                    ).fetchone()[0],
                    0,
                )

    def test_unrelated_tool_result_keeps_the_authoritative_input_request(self) -> None:
        self.start()
        sessions = Sessions(lambda: self.now)
        sessions.apply(HookEvent("codex", "one", "intent", goal="Improve parser"))
        waiting = HookEvent("codex", "one", "waiting", call="question")
        sessions.apply(waiting)
        self.store.session_event(
            "codex:one", waiting, waiting_for_input=sessions.lanes["codex:one"].waiting
        )
        before = require(self.store.session_context("codex:one"))
        unrelated = HookEvent(
            "codex",
            "one",
            "tool_done",
            call="other",
            progress="Raw tool output",
            outcome="success",
        )
        self.assertIsNone(sessions.apply(unrelated))
        self.assertTrue(sessions.lanes["codex:one"].waiting)
        self.store.session_event(
            "codex:one",
            unrelated,
            waiting_for_input=sessions.lanes["codex:one"].waiting,
        )
        self.assertEqual(before, self.store.session_context("codex:one"))
        self.assertIn(
            "input requested",
            json.dumps(work_record(require(self.store.session_view("codex:one")))),
        )

    def test_turn_outcomes_and_input_requests_change_story(self) -> None:
        self.start()
        self.store.session_event("codex:one", HookEvent("codex", "one", "waiting"))
        waiting = require(self.store.session_context("codex:one"))
        self.store.session_event(
            "codex:one",
            HookEvent(
                "codex",
                "one",
                "turn_done",
                outcome="success",
                progress="Covered escapes",
            ),
        )
        completed = require(self.store.session_context("codex:one"))
        self.assertNotEqual(waiting.revision, completed.revision)
        rendered = json.dumps(
            work_record(require(self.store.session_view("codex:one")))
        )
        self.assertIn("turn ended", rendered)
        self.assertIn("success", rendered)
        self.assertIn("Covered escapes", rendered)

    def test_explicit_ticket_urls_connect_reported_work_and_pr_text(self) -> None:
        ticket = "https://linear.app/team/issue/ABC-123/parser"
        first = self.start(intent=f"Improve parser for {ticket}")
        second = self.start("claude:two", "Review parser")
        self.progress(f"Validated {ticket}", "claude:two")
        self.store.observe_repository(
            replace(
                self.observation(), pr=replace(self.pr(), body=f"Implements {ticket}")
            )
        )
        for identifier in (first, second):
            refs = {
                item.artifact.ref
                for item in require(self.store.work_context(identifier)).artifacts
            }
            self.assertIn(ticket, refs)
        ticket_ids = {
            item.id
            for item in self.store.snapshot().artifacts.values()
            if item.kind == "ticket"
        }
        self.assertEqual(len(ticket_ids), 1)
        self.progress("ABC-456 and parser title are not explicit URLs")
        self.assertEqual(
            {
                item.id
                for item in self.store.snapshot().artifacts.values()
                if item.kind == "ticket"
            },
            ticket_ids,
        )

    def test_accepted_story_keeps_its_revision_after_restart_and_session_pruning(
        self,
    ) -> None:
        identifier = self.start()
        self.progress("Quoted strings are covered")
        live = self.store.views(
            sessions_snapshot(session_snapshot(id="codex:one", turn_active=True))
        )[0]
        self.memory.record_acceptance(
            "story",
            "Parser progress",
            "routine",
            identifier,
            live.revision,
            1000,
            live.revision,
        )
        pruned = self.store.views(sessions_snapshot())[0]
        self.assertEqual(pruned.revision, live.revision)
        self.restart()
        restored = self.store.views(sessions_snapshot())[0]
        self.assertEqual(restored.revision, live.revision)
        self.assertEqual(
            self.memory.load_attention_history()[identifier].considered_revision,
            restored.revision,
        )
        self.store.session_event("codex:one", HookEvent("codex", "one", "waiting"))
        self.assertNotEqual(
            self.store.views(sessions_snapshot())[0].revision, restored.revision
        )

    def test_saved_pr_evidence_ignores_legacy_check_and_review_fields(self) -> None:
        self.start()
        self.store.observe_repository(self.observation())
        rows = self.memory.db.execute(
            "SELECT id,data FROM work_evidence WHERE source='github' AND kind='observation'"
        ).fetchall()
        for row in rows:
            data = json.loads(row["data"])
            data["pr"].update(
                checks=[["legacy", "FAILURE"]],
                review="CHANGES_REQUESTED",
                check_revision="old",
                review_requested=True,
            )
            self.memory.db.execute(
                "UPDATE work_evidence SET data=? WHERE id=?",
                (json.dumps(data), row["id"]),
            )
        self.memory.db.commit()
        self.restart()
        rendered = json.dumps(
            work_record(require(self.store.session_view("codex:one")))
        )
        self.assertIn("OPEN", rendered)
        self.assertNotIn("FAILURE", rendered)
        self.assertNotIn("CHANGES_REQUESTED", rendered)

    def test_missing_runtime_sessions_keep_persisted_work_context(self) -> None:
        identifier = self.start()
        self.progress("Parser tests pass")
        subjects = build_work(sessions_snapshot(), self.store.snapshot())
        self.assertEqual(subjects[0].id, identifier)
        self.assertEqual(subjects[0].sessions, ("codex:one",))
        self.assertTrue(subjects[0].view.activity)
