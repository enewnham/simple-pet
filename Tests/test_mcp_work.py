import tempfile
import unittest
from pathlib import Path
from typing import Any, cast

from pet_daemon.domain import ArtifactKind, PullRequestState, SessionSource
from pet_daemon.identity import SessionIdentity
from pet_daemon.mcp import WorkContext
from pet_daemon.mcp_work import CompanionContext
from pet_daemon.repository_identity import RemoteRepository
from pet_daemon.storage import Memory
from pet_daemon.work.models import (
    ArtifactReference,
    IntentEvidence,
    ProgressEvidence,
    ReportContext,
    SessionReport,
)
from pet_daemon.work.observations import (
    CheckoutObservation,
    GitObservation,
    PullRequest,
    PullRequestObservation,
)


class CompanionContextTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "memory.sqlite3"
        self.memory = Memory(self.path, "fixture")
        self.context = WorkContext(self.path, "fixture", clock=lambda: 1000)
        self.companion = CompanionContext(self.context)
        self.repository = RemoteRepository(value="github.com/example/project")

    def tearDown(self) -> None:
        self.memory.close()
        self.temporary.cleanup()

    def report(
        self,
        text: str,
        timestamp: float,
        *,
        checkout: str = "/workspace",
        session: str = "worker",
        intent: bool = False,
        branch: str = "feature",
        pr: str = "",
        ticket: str = "",
        pet: str = "fixture",
        ingested_at: float | None = None,
    ) -> None:
        memory = self.memory if pet == "fixture" else Memory(self.path, pet)
        identity = SessionIdentity(source=SessionSource.CLAUDE, native_id=session)
        payload = (
            IntentEvidence(session=identity, text=text)
            if intent
            else ProgressEvidence(session=identity, text=text)
        )
        links = (
            (ArtifactReference(kind=ArtifactKind.TICKET, ref=ticket),) if ticket else ()
        )
        with memory.transaction():
            memory._work_repository.append_report(
                SessionReport(
                    payload=payload,
                    context=ReportContext(
                        checkout=checkout,
                        branch=branch,
                        pull_request=pr,
                        references=links,
                    ),
                    observed_at=timestamp,
                    ingested_at=timestamp if ingested_at is None else ingested_at,
                )
            )
        if memory is not self.memory:
            memory.close()

    def stream(self) -> dict[str, Any]:
        return cast(list[dict[str, Any]], self.companion.workstreams()["workstreams"])[
            0
        ]

    def test_workstreams_group_checkout_and_session_without_assuming_a_root(
        self,
    ) -> None:
        self.report(
            "Set up shell",
            100,
            checkout="/home/me/.dotfiles",
            session="shell",
            intent=True,
        )
        self.report("Build parser", 110, intent=True)
        for index in range(20):
            self.report(f"Parser detail {index}", 200 + index)
        self.report(
            "Unlocated task", 150, checkout="", session="unlocated", intent=True
        )
        self.report("Other pet secret", 999, pet="other")
        result: dict[str, Any] = self.companion.workstreams(3)
        streams = result["workstreams"]
        self.assertEqual(len(streams), 3)
        self.assertEqual(
            [item["checkout"] for item in streams],
            ["/workspace", None, "/home/me/.dotfiles"],
        )
        self.assertEqual(streams[0]["summary"]["text"], "Parser detail 19")
        self.assertEqual(streams[0]["last_activity_at"], 219)
        self.assertEqual(
            streams[0]["summary"]["source"]["provenance"]["basis"], "reported"
        )
        self.assertNotIn("Other pet secret", str(result))
        self.assertFalse(result["truncated"])
        self.assertTrue(self.companion.workstreams(1)["truncated"])

    def test_workstream_preserves_first_intent_and_distinguishes_repo_evidence(
        self,
    ) -> None:
        pr = "https://github.com/example/project/pull/17"
        ticket = "https://linear.app/example/issue/ENG-17/parser"
        self.report("Build parser", 100, intent=True, pr=pr, ticket=ticket)
        self.report("Extend parser", 200, intent=True, pr=pr)
        self.report(
            "Decision: retain source offsets.\nShould we also index comments?",
            300,
            pr=pr,
        )
        with self.memory.transaction():
            self.memory._work_repository.put_observation(
                PullRequestObservation(
                    ref=pr,
                    checked_at=900,
                    pr=PullRequest(
                        number=17,
                        url=pr,
                        state=PullRequestState.OPEN,
                        merged_at=None,
                        repository=self.repository,
                        head_branch="feature",
                    ),
                )
            )
        summary = self.stream()
        result: dict[str, Any] = self.companion.workstream(summary["id"])
        self.assertEqual(result["goal"]["text"], "Build parser")
        self.assertEqual(result["recent_turns"][0]["changed_at"], 300)
        self.assertEqual(result["last_activity_at"], 300)
        self.assertEqual(
            result["repository_state"][0]["provenance"]["basis"], "observed"
        )
        self.assertEqual(result["repository_state"][0]["provenance"]["checked_at"], 900)
        self.assertEqual(
            result["decisions"][0]["text"], "Decision: retain source offsets."
        )
        self.assertEqual(
            result["open_questions"][0]["classification"],
            "unanswered_question_candidate",
        )
        self.assertEqual(result["tickets"][0]["value"], ticket)
        self.assertEqual(
            result["pull_requests"][0]["source"]["provenance"]["basis"], "observed"
        )
        self.report("Index comments too", 400, intent=True, pr=pr)
        self.assertEqual(self.companion.workstream(summary["id"])["open_questions"], [])
        self.assertEqual(self.stream()["id"], summary["id"])
        with self.assertRaisesRegex(ValueError, "Unknown workstream"):
            self.companion.workstream("thread:missing")

    def test_since_groups_changed_observations_using_older_checkout_context(
        self,
    ) -> None:
        pr = "https://github.com/example/project/pull/17"
        self.report("Build parser", 100, intent=True, pr=pr)
        with self.memory.transaction():
            self.memory._work_repository.put_observation(
                PullRequestObservation(
                    ref=pr,
                    checked_at=900,
                    pr=PullRequest(
                        number=17,
                        url=pr,
                        state=PullRequestState.OPEN,
                        merged_at=None,
                        repository=self.repository,
                    ),
                )
            )
        expected_id = self.stream()["id"]
        result: dict[str, Any] = self.companion.since(500)
        self.assertEqual(len(result["workstreams"]), 1)
        changed = result["workstreams"][0]
        self.assertEqual(changed["id"], expected_id)
        self.assertEqual(changed["checkout"], "/workspace")
        self.assertEqual(len(changed["sources"]), 1)
        self.assertEqual(changed["sources"][0]["provenance"]["checked_at"], 900)
        self.assertFalse(result["truncated"])
        self.assertEqual(self.companion.since(900)["workstreams"], [])

    def test_decision_search_returns_exact_excerpts_and_retains_where_and_when(
        self,
    ) -> None:
        self.report("Working on parser", 100)
        self.report(
            "Progress here.\n  Correction: use byte offsets.\nTests passed.", 200
        )
        self.report(
            "We decided to preserve comments.", 300, checkout="/home/me/.dotfiles"
        )
        result: dict[str, Any] = self.companion.decisions("byte offsets")
        self.assertEqual(len(result["decisions"]), 1)
        decision = result["decisions"][0]
        self.assertEqual(decision["text"], "Correction: use byte offsets.")
        self.assertEqual(decision["when"], 200)
        self.assertEqual(decision["where"]["checkout"], "/workspace")
        source = decision["source"]
        self.assertEqual(source["text"], decision["text"])
        self.assertEqual(
            self.context.read(source["id"], source["offset"], len(source["text"]))[
                "text"
            ],
            source["text"],
        )
        self.assertEqual(source["provenance"]["basis"], "reported")
        self.assertEqual(self.companion.decisions("!!!")["decisions"], [])
        self.assertTrue(self.companion.decisions(limit=1)["truncated"])

    def test_repository_refresh_does_not_reactivate_a_stale_thread(self) -> None:
        self.report("Ancient task", -700000, intent=True)
        with self.memory.transaction():
            self.memory._work_repository.put_observation(
                CheckoutObservation(
                    worktree="/workspace",
                    checked_at=999,
                    local=GitObservation(
                        worktree="/workspace",
                        branch="feature",
                        head="abc",
                        staged=0,
                        unstaged=0,
                        untracked=0,
                        conflicts=0,
                        upstream="",
                        ahead=0,
                        behind=0,
                        repository=self.repository,
                    ),
                )
            )
        self.assertEqual(self.companion.workstreams()["workstreams"], [])
        self.assertEqual(
            len(cast(list[object], self.companion.since(900)["workstreams"])), 1
        )

    def test_workstream_and_since_report_truncation_within_threads(self) -> None:
        self.report("Build parser", 100, intent=True)
        self.report("Decision: keep offsets.", 200)
        self.report("Decision: keep tokens.", 300)
        detail: dict[str, Any] = self.companion.workstream(self.stream()["id"], limit=1)
        self.assertTrue(detail["truncated"])
        self.assertEqual(len(detail["recent_turns"]), 1)
        changed: dict[str, Any] = self.companion.since(0, limit=1)
        self.assertTrue(changed["truncated"])
        self.assertTrue(changed["workstreams"][0]["truncated"])
        self.assertEqual(len(changed["workstreams"][0]["sources"]), 1)

    def test_late_ingestion_preserves_activity_time_and_turn_order(self) -> None:
        self.report("Latest turn", 300)
        self.report("Earlier turn", 100, ingested_at=999)
        summary = self.stream()
        self.assertEqual(summary["last_activity_at"], 300)
        self.assertEqual(summary["summary"]["text"], "Latest turn")
        detail: dict[str, Any] = self.companion.workstream(summary["id"])
        self.assertEqual(
            [source["provenance"]["observed_at"] for source in detail["recent_turns"]],
            [300, 100],
        )
        changed: dict[str, Any] = self.companion.since(500)
        self.assertEqual(
            changed["workstreams"][0]["sources"][0]["provenance"]["observed_at"], 100
        )

    def test_workstreams_mark_truncated_sessions_within_a_checkout(self) -> None:
        self.report("First session", 100, session="one")
        self.report("Second session", 200, session="two")
        result: dict[str, Any] = self.companion.workstreams(1)
        self.assertTrue(result["truncated"])
        self.assertTrue(result["workstreams"][0]["sessions_truncated"])
        self.assertEqual(len(result["workstreams"][0]["sessions"]), 1)

    def test_workstream_summary_selects_the_first_nonblank_line(self) -> None:
        self.report("\n\n  Parser verified.\nMore detail.", 100)
        summary = self.stream()["summary"]
        self.assertEqual(summary["text"], "Parser verified.")
        source = summary["source"]
        self.assertEqual(source["text"], "Parser verified.")
        self.assertEqual(
            self.context.read(source["id"], source["offset"], len(source["text"]))[
                "text"
            ],
            "Parser verified.",
        )
