import json
import sqlite3
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

from pet_daemon.domain import Activity, DeliveryStatus, ExecutionOutcome, SpeechReason
from pet_daemon.sessions.tracking import Sessions
from pet_daemon.sources.hooks import HookEvent, normalize_hook
from pet_daemon.storage import Memory
from pet_daemon.storage.memory import (
    ContextRecord,
    HistoryRecord,
)
from pet_daemon.storage.migrations import SCHEMA_VERSION
from pet_daemon.types import JSONValue
from pet_daemon.work.history import (
    HistoryDetails,
    SessionContext,
    build_recall_query,
    build_session_entry,
)
from pet_daemon.work.observations import (
    CheckoutObservation,
    GitObservation,
)
from pet_daemon.work.views import IntentView, ProgressView
from Tests.fixtures import require, sessions_snapshot


class MemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "memory.sqlite3"
        self.now = 1000
        self.memory = Memory(self.path, "fixture", clock=lambda: self.now, max_events=3)
        self.sessions = Sessions(lambda: self.now)

    def tearDown(self) -> None:
        self.memory.close()
        self.temporary.cleanup()

    def ingest(self, event: HookEvent, memory: Memory | None = None) -> None:
        result = require(self.sessions.apply(event))
        memory = memory or self.memory
        with memory.transaction():
            memory.work.record_session(result.observation)
            memory.record_event(build_session_entry(event))

    def context(self, session: str = "codex:one") -> ContextRecord:
        context = self.memory.context(session)
        assert context is not None
        return context

    def event(self, progress: str, **extra: JSONValue) -> None:
        self.now += 1
        self.memory.record_event(
            build_session_entry(
                require(
                    normalize_hook(
                        {
                            "harness": "codex",
                            "session": "one",
                            "event": "progress",
                            "repo": "/repo/parser",
                            "repository": "github.com/team/parser",
                            "progress": progress,
                            **extra,
                        }
                    )
                )
            )
        )

    def observe_checkout(self, memory: Memory | None = None) -> CheckoutObservation:
        memory = memory or self.memory
        self.ingest(
            HookEvent("codex", "one", "session_start", repo="/repo/parser"), memory
        )
        checkpoint = CheckoutObservation(
            "/repo/parser",
            GitObservation(
                "/repo/parser",
                "feature",
                "head",
                0,
                0,
                0,
                0,
                "",
                0,
                0,
                "github.com/team/project",
            ),
            self.now,
        )
        memory.work.observe(checkpoint)
        return next(
            item
            for item in memory.load_repositories()
            if isinstance(item, CheckoutObservation)
            and item.worktree == checkpoint.worktree
        )

    def test_events_survive_restart_and_remain_separate_from_speech(self) -> None:
        self.event("Parser checks passed")
        self.memory.record_speech("The parser checks passed.", SpeechReason.COMPLETION)
        self.memory.close()
        self.memory = Memory(self.path, "fixture", clock=lambda: self.now)
        self.assertIn(
            "Parser checks passed", json.dumps(self.memory.recent(), default=asdict)
        )
        self.assertIn(
            "The parser checks passed.",
            json.dumps(self.memory.speech(), default=asdict),
        )
        self.assertNotIn(
            "The parser checks passed.",
            json.dumps(self.memory.recent(), default=asdict),
        )
        self.assertEqual(self.path.read_bytes()[:16], b"SQLite format 3\x00")

    def test_speech_expires_at_thirty_minutes_without_a_prune(self) -> None:
        self.memory.record_speech("First remark", SpeechReason.ROUTINE)
        self.now += 1799
        self.assertEqual(self.memory.speech()[0].text, "First remark")
        self.now += 1
        self.assertEqual(self.memory.speech(), [])
        self.assertEqual(
            self.memory.db.execute("SELECT COUNT(*) FROM speech").fetchone()[0], 1
        )
        self.memory.prune()
        self.assertEqual(
            self.memory.db.execute("SELECT COUNT(*) FROM speech").fetchone()[0], 0
        )

    def test_speech_expiry_survives_restart_and_uses_its_own_retention(self) -> None:
        self.memory.record_speech("Expired remark", SpeechReason.ROUTINE)
        self.now += 1799
        self.memory.record_speech("Recent remark", SpeechReason.ROUTINE)
        self.memory.close()
        self.now += 1
        self.memory = Memory(
            self.path, "fixture", clock=lambda: self.now, retention_days=0
        )
        self.assertEqual(
            [item.text for item in self.memory.speech()], ["Recent remark"]
        )
        self.now += 1
        self.memory.prune()
        self.assertEqual(
            [item.text for item in self.memory.speech()], ["Recent remark"]
        )
        self.memory.close()
        self.memory = Memory(
            self.path, "fixture", clock=lambda: self.now, speech_history_seconds=1
        )
        self.assertEqual(self.memory.speech(), [])

    def test_short_speech_retention_preserves_another_pets_valid_history(self) -> None:
        self.memory.record_speech("Long history remark", SpeechReason.ROUTINE)
        other = Memory(
            self.path,
            "short-history-pet",
            clock=lambda: self.now,
            speech_history_seconds=60,
        )
        try:
            other.record_speech("Short history remark", SpeechReason.ROUTINE)
            self.now += 60
            other.prune()
            self.assertEqual(other.speech(), [])
            self.assertEqual(self.memory.speech()[0].text, "Long history remark")
            self.memory.close()
            self.memory = Memory(self.path, "fixture", clock=lambda: self.now)
            self.assertEqual(self.memory.speech()[0].text, "Long history remark")
        finally:
            other.close()

    def test_speech_history_keeps_the_global_row_limit(self) -> None:
        for index in range(257):
            self.memory.record_speech(f"Remark {index}", SpeechReason.ROUTINE)
        rows = self.memory.speech(limit=300)
        self.assertEqual(len(rows), 256)
        self.assertEqual(rows[0].text, "Remark 1")
        self.assertEqual(rows[-1].text, "Remark 256")

    def test_clear_speech_preserves_knowledge_and_other_pet_speech(self) -> None:
        self.event("Parser checks passed", goal="Improve parser")
        self.memory.record_speech("Parser work", SpeechReason.ROUTINE)
        self.memory.save_context(
            "codex:one", "Improve parser", "Parser summary", ["Improve parser"]
        )
        checkpoint = self.observe_checkout()
        other = Memory(self.path, "other-pet", clock=lambda: self.now)
        try:
            other.record_speech("Other pet remark", SpeechReason.ROUTINE)
            self.memory.clear_speech()
            self.assertEqual(self.memory.speech(), [])
            self.assertEqual(other.speech()[0].text, "Other pet remark")
            self.assertEqual(self.context().summary, "Parser summary")
            self.assertIn(
                "Parser checks passed", json.dumps(self.memory.recent(), default=asdict)
            )
            self.assertEqual(
                self.memory.projects(["/repo/parser"])[0].goal, "Improve parser"
            )
            self.assertEqual(self.memory.load_repositories(), (checkpoint,))
            self.memory.close()
            self.memory = Memory(self.path, "fixture", clock=lambda: self.now)
            self.assertEqual(self.memory.speech(), [])
            self.assertTrue(self.memory.recent())
        finally:
            other.close()

    def test_clear_erases_saved_knowledge_for_the_current_pet(self) -> None:
        other = Memory(self.path, "other-pet", clock=lambda: self.now)
        event = HookEvent(
            harness="codex",
            session="one",
            event="intent",
            repo="/repo/parser",
            goal="Improve parser",
        )
        try:
            for memory in (self.memory, other):
                memory.record_event(build_session_entry(event))
                memory.record_speech("Parser work", SpeechReason.PROMPT)
                memory.save_context(
                    "codex:one", "Improve parser", "Parser work", ["Improve parser"]
                )
            self.memory.clear()
            self.memory.close()
            self.memory = Memory(self.path, "fixture", clock=lambda: self.now)
            self.assertEqual(self.memory.recent(), [])
            self.assertEqual(self.memory.speech(), [])
            self.assertEqual(self.memory.projects(["/repo/parser"]), [])
            self.assertIsNone(self.memory.context("codex:one"))
            self.assertTrue(other.recent())
            self.assertTrue(other.speech())
            self.assertTrue(other.projects(["/repo/parser"]))
            self.assertIsNotNone(other.context("codex:one"))
            self.memory.record_event(build_session_entry(event))
            self.assertEqual(len(self.memory.recent()), 1)
        finally:
            other.close()

    def test_event_count_discards_oldest_facts(self) -> None:
        for progress in (
            "First result",
            "Second result",
            "Third result",
            "Fourth result",
        ):
            self.event(progress)
        events = self.memory.recent(limit=10)
        self.assertEqual(len(events), 3)
        self.assertNotIn("First result", json.dumps(events, default=asdict))
        self.assertIn("Fourth result", json.dumps(events, default=asdict))

    def test_retention_discards_expired_facts(self) -> None:
        self.event("Expired result")
        self.now += 31 * 86400
        self.event("Current result")
        events = json.dumps(self.memory.recent(limit=10), default=asdict)
        self.assertNotIn("Expired result", events)
        self.assertIn("Current result", events)

    def test_pet_identity_keeps_memory_separate(self) -> None:
        self.event("Fixture parser result")
        self.memory.record_speech("Fixture remark", SpeechReason.COMPLETION)
        other = Memory(self.path, "other-pet", clock=lambda: self.now)
        try:
            self.assertEqual(other.recent(), [])
            self.assertEqual(other.speech(), [])
        finally:
            other.close()

    def test_recent_results_and_speech_obey_requested_limits(self) -> None:
        for index in range(3):
            self.event(f"Result {index}")
            self.memory.record_speech(f"Remark {index}", SpeechReason.ROUTINE)
        self.assertEqual(len(self.memory.recent(limit=2)), 2)
        self.assertEqual(len(self.memory.speech(limit=2)), 2)

    def test_recent_history_filters_subjects_before_ranking_sessions(self) -> None:
        self.memory.max_events = 100
        self.event("Parser first", session="one", repo="/repo/parser")
        self.event("Parser second", session="two", repo="/repo/parser")
        self.event("Billing same session", session="one", repo="/repo/billing")
        for index in range(6):
            self.event(
                f"Unrelated history {index}",
                session=f"other-{index}",
                repo="/repo/other",
            )
        parser = self.memory.recent(limit=2, working_directory="/repo/parser")
        self.assertEqual([item.session for item in parser], ["codex:one", "codex:two"])
        self.assertTrue(
            all(item.working_directory == "/repo/parser" for item in parser)
        )
        session = self.memory.recent(limit=2, session="codex:one")
        self.assertEqual(
            [item.working_directory for item in session],
            ["/repo/parser", "/repo/billing"],
        )
        intersection = self.memory.recent(
            session="codex:one", working_directory="/repo/parser"
        )
        self.assertEqual(len(intersection), 1)
        self.assertIn(
            "Parser first",
            asdict(intersection[0].details)["progress"],
        )
        self.assertEqual(self.memory.recent(session="codex:missing"), [])
        self.assertEqual(self.memory.recent(working_directory="/repo/missing"), [])
        self.assertEqual(len(self.memory.recent(limit=5)), 5)

    def test_terminal_history_filters_events_before_recent_tool_activity(self) -> None:
        self.memory.max_events = 100
        for index in range(4):
            self.event(
                f"Report {index}", event="turn_done", session="one", outcome="success"
            )
        self.event(
            "Other report", event="turn_done", session="other", outcome="success"
        )
        for index in range(8):
            self.event(
                f"Later tool {index}",
                event="tool_done",
                session="one",
                tool="build",
                outcome="success",
            )
        rows = self.memory.recent(
            limit=3, session="codex:one", event_types=("turn_done", "turn_error")
        )
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row.session == "codex:one" for row in rows))
        self.assertTrue(all(row.event_type == "turn_done" for row in rows))
        self.assertEqual(
            [asdict(row.details)["progress"] for row in rows],
            ["Report 1", "Report 2", "Report 3"],
        )
        self.assertEqual(self.memory.recent(event_types=()), [])
        self.assertEqual(self.memory.recent(event_types=("turn_done') OR 1=1 --",)), [])

    def test_project_goal_and_request_time_change_only_for_explicit_requests(
        self,
    ) -> None:
        self.event("", event="intent", goal="Improve parser")
        original = self.memory.projects(["/repo/parser"])[0]
        self.now += 30
        self.memory.record_event(
            build_session_entry(
                require(
                    normalize_hook(
                        {
                            "harness": "codex",
                            "session": "one",
                            "event": "tool_done",
                            "tool": "pytest",
                            "outcome": "success",
                        }
                    )
                ),
                SessionContext(
                    session_id="codex:one",
                    working_directory="/repo/parser",
                    goal="Stale inherited goal",
                    last_tool="",
                    outcome=None,
                    activity=Activity.TESTING,
                ),
            )
        )
        self.assertEqual(self.memory.projects(["/repo/parser"])[0], original)
        self.event("Current progress")
        self.assertEqual(self.memory.projects(["/repo/parser"])[0], original)
        self.event("", event="intent", goal="Document parser")
        updated = self.memory.projects(["/repo/parser"])[0]
        self.assertEqual(updated.goal, "Document parser")
        self.assertGreater(updated.timestamp, original.timestamp)
        self.memory.record_event(
            build_session_entry(
                require(
                    normalize_hook(
                        {
                            "harness": "codex",
                            "session": "unknown",
                            "event": "progress",
                            "progress": "Working",
                        }
                    )
                ),
                SessionContext(
                    session_id="codex:unknown",
                    working_directory="/repo/new",
                    goal="Inherited task",
                    last_tool="",
                    outcome=None,
                    activity=Activity.COMMAND,
                ),
            )
        )
        self.assertEqual(self.memory.projects(["/repo/new"]), [])

    def test_terminal_hooks_report_turn_completion_without_claiming_tool_success(
        self,
    ) -> None:
        for event in ("turn_done", "subagent_done", "session_end"):
            with self.subTest(event=event):
                self.memory.record_event(
                    build_session_entry(
                        require(
                            normalize_hook(
                                {
                                    "harness": "codex",
                                    "session": "one",
                                    "event": event,
                                    "outcome": "success",
                                    "progress": "Finished reviewing parser behavior",
                                }
                            )
                        ),
                        SessionContext(
                            session_id="codex:one",
                            working_directory="",
                            goal="",
                            last_tool="pytest",
                            outcome=None,
                            activity=Activity.TESTING,
                        ),
                    )
                )
                details = asdict(self.memory.recent()[-1].details)
                self.assertEqual(details["reported_outcome"], "success")
                self.assertEqual(
                    details["progress"], "Finished reviewing parser behavior"
                )
                self.assertIsNone(details["activity"])
        self.memory.record_event(
            build_session_entry(
                require(
                    normalize_hook(
                        {
                            "harness": "codex",
                            "session": "one",
                            "event": "tool_done",
                            "tool": "pytest",
                            "outcome": "success",
                        }
                    )
                )
            )
        )
        self.assertIsNone(self.memory.recent()[-1].details.activity)

    def test_recall_matches_older_goals_within_the_repository(self) -> None:
        self.memory.close()
        self.memory = Memory(
            self.path, "fixture", clock=lambda: self.now, max_events=100
        )
        self.event("", event="intent", goal="Refactor parser tokenization")
        self.event(
            "",
            event="intent",
            repo="/repo/other",
            repository="github.com/team/other",
            goal="Refactor parser tokenization",
        )
        self.event("", event="intent", goal="Improve billing invoices")
        for index in range(6):
            self.event(f"Recent progress {index}")
        recalled = self.memory.recall(
            build_recall_query(
                normalize_hook(
                    {
                        "harness": "codex",
                        "session": "one",
                        "event": "intent",
                        "repo": "/repo/parser",
                        "goal": "Refactor parser tokenization",
                    }
                ),
                repository="github.com/team/parser",
            )
        )
        self.assertEqual(len(recalled), 1)
        self.assertIn("parser tokenization", recalled[0].goal)
        self.assertEqual(recalled[0].working_directory, "/repo/parser")
        self.assertNotIn(recalled[0].id, [event.id for event in self.memory.recent()])

    def test_recall_matches_observed_repository_across_worktrees(self) -> None:
        self.memory.max_events = 100
        checkout = self.observe_checkout()
        local = require(checkout.local)
        self.memory.work.observe(
            replace(
                checkout,
                worktree="/work/parser-review",
                local=replace(local, worktree="/work/parser-review", branch="review"),
            )
        )
        for location in ("/repo/parser/src", "/work/parser-review"):
            self.memory.record_event(
                build_session_entry(
                    HookEvent(
                        "codex",
                        "one",
                        "intent",
                        repo=location,
                        goal="Refactor parser tokenization",
                    )
                )
            )
        self.event(
            "",
            event="intent",
            repository="github.com/team/unrelated",
            goal="Refactor parser tokenization",
        )
        for index in range(6):
            self.event(f"Recent progress {index}")
        request = build_recall_query(
            HookEvent(
                "codex",
                "one",
                "intent",
                repo="/another/worktree",
                repository="github.com/team/unrelated",
                goal="Refactor parser tokenization",
            ),
            repository="git@github.com:team/project.git",
        )
        recalled = self.memory.recall(request)
        self.assertEqual(
            {entry.working_directory for entry in recalled},
            {"/repo/parser/src", "/work/parser-review"},
        )
        self.assertEqual(
            {entry.repository for entry in recalled}, {"github.com/team/project"}
        )

    def test_recall_requires_repository_identity_and_ignores_repeated_terms(
        self,
    ) -> None:
        event = HookEvent(
            "codex",
            "one",
            "intent",
            repo="/repo/parser",
            goal="Parser parser PARSER tokenization",
        )
        self.assertIsNone(build_recall_query(event, repository=""))
        request = require(
            build_recall_query(event, repository="github.com/team/parser")
        )
        self.assertEqual(request.terms, ("parser", "tokenization"))
        self.assertIsNotNone(
            build_recall_query(replace(event, repo=None), repository=request.repository)
        )
        self.memory.record_event(build_session_entry(event))
        self.assertEqual(self.memory.recent()[0].repository, "")

    def test_session_details_preserve_delimiters_and_provenance_after_restart(
        self,
    ) -> None:
        self.event("Kept | separators and 猫", event="turn_done", outcome="success")
        self.memory.close()
        self.memory = Memory(self.path, "fixture", clock=lambda: self.now)
        entry = self.memory.recent()[0]
        self.assertIs(entry.details.reported_outcome, ExecutionOutcome.SUCCESS)
        self.assertEqual(
            entry.details,
            HistoryDetails(
                ExecutionOutcome.SUCCESS, progress="Kept | separators and 猫"
            ),
        )
        self.assertEqual(entry.working_directory, "/repo/parser")
        self.assertEqual(entry.repository, "github.com/team/parser")

    def test_recall_requires_an_explicit_goal_at_task_start(self) -> None:
        self.memory.close()
        self.memory = Memory(
            self.path, "fixture", clock=lambda: self.now, max_events=100
        )
        self.event("", event="intent", goal="Refactor parser tokenization")
        for index in range(6):
            self.event(f"Recent progress {index}")
        for event in (
            {"event": "tool_error", "goal": "Refactor parser tokenization"},
            {"event": "progress", "progress": "Refactor parser tokenization"},
            {"event": "intent", "detail": "Refactor parser tokenization"},
            {"event": "intent", "goal": ""},
        ):
            with self.subTest(event=event):
                self.assertEqual(
                    self.memory.recall(
                        build_recall_query(
                            normalize_hook(
                                {
                                    "harness": "codex",
                                    "session": "one",
                                    "repo": "/repo/parser",
                                    **event,
                                }
                            ),
                            repository="github.com/team/parser",
                        )
                    ),
                    [],
                )

    def test_failure_facts_exclude_injected_diagnostics(self) -> None:
        self.event(
            "SECRET FAILED COMMAND OUTPUT",
            event="tool_error",
            detail="SECRET STACK TRACE",
            error_signature="SECRET HASH",
            tool="pytest",
        )
        rows = self.memory.recent()
        self.assertNotIn("SECRET", json.dumps(rows, default=asdict))
        self.assertEqual(rows, [])

    def test_long_factual_fields_are_bounded_before_storage(self) -> None:
        self.event("猫" * 100000)
        event = self.memory.recent()[0]
        self.assertLessEqual(
            len(asdict(event.details)["progress"]),
            240,
        )
        self.assertNotIn("thought", asdict(event))
        self.assertNotIn("speak", asdict(event))

    def test_recent_facts_keep_a_quiet_session_visible(self) -> None:
        self.memory.close()
        self.memory = Memory(
            self.path, "fixture", clock=lambda: self.now, max_events=100
        )
        self.event("Quiet session remains blocked", session="quiet")
        for index in range(8):
            self.event(f"Busy session progress {index}")
        events = self.memory.recent(limit=5)
        self.assertIn("codex:quiet", [event.session for event in events])
        self.assertIn("Busy session progress 7", json.dumps(events, default=asdict))

    def test_terminal_fact_preserves_reconciled_failure(self) -> None:
        self.memory.record_event(
            build_session_entry(
                require(
                    normalize_hook(
                        {
                            "session": "one",
                            "harness": "claude",
                            "event": "turn_done",
                            "outcome": "success",
                        }
                    )
                ),
                SessionContext(
                    session_id="claude:one",
                    working_directory="",
                    goal="",
                    last_tool="",
                    outcome=ExecutionOutcome.ERROR,
                    activity=Activity.COMMAND,
                ),
            )
        )
        self.assertEqual(
            asdict(self.memory.recent()[0].details)["reported_outcome"],
            "error",
        )

    def test_duplicate_progress_does_not_displace_distinct_facts(self) -> None:
        self.event("First result")
        self.event("Second result")
        for _ in range(5):
            self.event("Second result")
        events = self.memory.recent()
        self.assertEqual(len(events), 2)
        self.assertIn("First result", json.dumps(events, default=asdict))

    def test_tool_commands_do_not_create_persisted_facts(self) -> None:
        for kind in ("tool_start", "tool_done", "tool_error"):
            self.event("Command output", event=kind, tool="pytest", outcome="error")
        self.assertEqual(self.memory.recent(), [])

    def test_cancellation_retains_the_agent_turn_fact(self) -> None:
        self.memory.record_event(
            build_session_entry(
                HookEvent(
                    "claude", "one", "interrupt", outcome=ExecutionOutcome.CANCELLED
                )
            )
        )
        self.assertEqual(
            asdict(self.memory.recent()[-1].details)["reported_outcome"],
            "cancelled",
        )

    def test_acceptance_commits_speech_outcome_and_considered_story_together(
        self,
    ) -> None:
        self.assertTrue(
            self.memory.record_acceptance(
                "message",
                "Parser covered",
                SpeechReason.ROUTINE,
                "work",
                "revision",
                50,
                "revision",
            )
        )
        self.assertFalse(
            self.memory.record_acceptance(
                "message", "Duplicate", SpeechReason.ROUTINE, "work", "new", 60, "new"
            )
        )
        outcome = require(self.memory.message_outcome("message"))
        self.assertIs(outcome.outcome, DeliveryStatus.ACCEPTED)
        self.assertIs(outcome.reason, SpeechReason.ROUTINE)
        self.assertEqual((outcome.subject_id, outcome.revision), ("work", "revision"))
        self.assertEqual(
            [item.text for item in self.memory.speech()], ["Parser covered"]
        )
        self.assertEqual(
            self.memory.load_attention_history()["work"], HistoryRecord(50, "revision")
        )

    def test_late_acceptance_preserves_newer_consideration(self) -> None:
        self.memory.save_attention_history("work", 60, "new")
        self.memory.record_acceptance(
            "message", "Earlier report", SpeechReason.ROUTINE, "work", "old"
        )
        self.assertEqual(
            self.memory.load_attention_history()["work"], HistoryRecord(60, "new")
        )
        self.assertEqual(
            require(self.memory.message_outcome("message")).revision, "old"
        )

    def test_failed_attention_write_rolls_back_speech_and_outcome(self) -> None:
        self.memory.db.execute("""CREATE TRIGGER reject_attention BEFORE INSERT ON attention_history
            BEGIN SELECT RAISE(ABORT, 'attention unavailable'); END""")
        with self.assertRaises(sqlite3.IntegrityError):
            self.memory.record_acceptance(
                "message",
                "Parser covered",
                SpeechReason.ROUTINE,
                "work",
                "revision",
                50,
                "revision",
            )
        self.assertIsNone(self.memory.message_outcome("message"))
        self.assertEqual(self.memory.speech(), [])
        self.assertEqual(self.memory.load_attention_history(), {})
        with self.assertRaises(sqlite3.IntegrityError):
            self.memory.record_delivery_failure(
                "message",
                SpeechReason.ROUTINE,
                subject_id="work",
                revision="revision",
                last_selected=50,
                considered_revision="revision",
            )
        self.assertIsNone(self.memory.message_outcome("message"))

    def test_failed_acceptance_pruning_rolls_back_the_message_transaction(self) -> None:
        self.memory.db.execute(
            "INSERT INTO speech(pet,text,reason,timestamp) VALUES ('fixture','Expired','routine',-10000)"
        )
        self.memory.db.execute("""CREATE TRIGGER reject_speech_pruning BEFORE DELETE ON speech
            BEGIN SELECT RAISE(ABORT, 'pruning unavailable'); END""")
        with self.assertRaises(sqlite3.IntegrityError):
            self.memory.record_acceptance(
                "message",
                "Parser covered",
                SpeechReason.ROUTINE,
                "work",
                "revision",
                50,
                "revision",
            )
        self.assertIsNone(self.memory.message_outcome("message"))
        self.assertEqual(self.memory.speech(), [])
        self.assertEqual(self.memory.load_attention_history(), {})
        self.assertEqual(
            self.memory.db.execute("SELECT text FROM speech").fetchone()[0], "Expired"
        )

    def test_failed_delivery_records_consideration_without_speech(self) -> None:
        self.assertTrue(
            self.memory.record_delivery_failure(
                "message",
                SpeechReason.ROUTINE,
                "Parser covered",
                "work",
                "revision",
                50,
                "revision",
            )
        )
        self.assertFalse(
            self.memory.record_delivery_failure(
                "message",
                SpeechReason.ROUTINE,
                subject_id="work",
                considered_revision="other",
            )
        )
        self.assertEqual(
            require(self.memory.message_outcome("message")).outcome, "failed"
        )
        self.assertEqual(self.memory.speech(), [])
        self.assertEqual(
            self.memory.load_attention_history()["work"], HistoryRecord(50, "revision")
        )
        self.assertTrue(
            self.memory.record_acceptance(
                "message", "Parser covered", SpeechReason.ROUTINE, "work", "revision"
            )
        )
        self.assertEqual(
            require(self.memory.message_outcome("message")).outcome, "accepted"
        )

    def test_session_context_survives_restart_with_its_original_anchor(self) -> None:
        self.memory.save_context(
            "codex:one",
            "Refactor parser",
            "Parser refactor awaits validation",
            ["Refactor parser", "Check edge cases"],
        )
        self.memory.close()
        self.memory = Memory(self.path, "fixture", clock=lambda: self.now)
        context = self.context()
        self.assertEqual(context.initial_request, "Refactor parser")
        self.assertEqual(context.summary, "Parser refactor awaits validation")
        self.assertEqual(
            context.recent_requests, ["Refactor parser", "Check edge cases"]
        )
        self.assertEqual(self.memory.recent(), [])
        self.assertIsNone(self.memory.context("codex:other"))

    def test_full_session_summary_survives_restart(self) -> None:
        summary = (
            "The agent reported parser compatibility details " * 12
            + "and confirmed that legacy aliases remain supported.\n"
            + "Keep the public API stable during follow-up work."
        )
        self.memory.save_context(
            "codex:one", "Refactor parser", summary, ["Verify compatibility"]
        )
        self.memory.close()
        self.memory = Memory(self.path, "fixture", clock=lambda: self.now)
        self.assertEqual(self.context().summary, summary)

    def test_pet_identity_keeps_session_context_separate(self) -> None:
        self.memory.save_context(
            "codex:one",
            "Refactor parser",
            "Parser validation pending",
            ["Refactor parser"],
        )
        other = Memory(self.path, "other-pet", clock=lambda: self.now)
        try:
            self.assertIsNone(other.context("codex:one"))
        finally:
            other.close()

    def test_matching_schema_preserves_saved_memory_and_repositories(self) -> None:
        self.event("Parser inspection completed", goal="Improve parser")
        self.memory.save_context(
            "codex:one", "Improve parser", "Parser summary", ["Improve parser"]
        )
        self.memory.record_speech("Parser work", SpeechReason.PROMPT)
        checkpoint = self.observe_checkout()
        self.memory.close()
        with sqlite3.connect(self.path) as database:
            database.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        self.memory = Memory(self.path, "fixture", clock=lambda: self.now)
        self.assertTrue(self.memory.recent())
        self.assertEqual(self.memory.speech()[0].text, "Parser work")
        self.assertEqual(
            self.context().summary,
            "Parser summary",
        )
        self.assertEqual(
            self.memory.projects(["/repo/parser"])[0].goal, "Improve parser"
        )
        self.assertEqual(self.memory.load_repositories(), (checkpoint,))
        self.assertEqual(
            self.memory.db.execute("PRAGMA user_version").fetchone()[0], SCHEMA_VERSION
        )

    def test_connected_work_is_pet_scoped_and_clears_with_memory(self) -> None:
        other = Memory(self.path, "other-pet", clock=lambda: self.now)
        try:
            for memory in (self.memory, other):
                self.ingest(
                    HookEvent("codex", "one", "intent", goal="Improve parser"), memory
                )
                self.observe_checkout(memory)
            self.assertEqual(
                len(self.memory.connected_work(sessions_snapshot())[0].story.artifacts),
                1,
            )
            self.memory.clear()
            self.memory.close()
            self.memory = Memory(self.path, "fixture", clock=lambda: self.now)
            self.assertEqual(self.memory.collection_targets(), ())
            self.assertEqual(self.memory.connected_work(sessions_snapshot()), ())
            self.assertEqual(
                len(other.connected_work(sessions_snapshot())[0].story.artifacts), 1
            )
        finally:
            other.close()

    def test_session_keeps_its_intent_refinements_and_accepts_reports_after_pruning(
        self,
    ) -> None:
        self.memory.close()
        self.memory = Memory(
            self.path, "fixture", clock=lambda: self.now, retention_days=1
        )
        self.ingest(HookEvent("codex", "one", "intent", goal="Improve parser"))
        self.ingest(HookEvent("codex", "one", "intent", goal="Check parser edge cases"))
        self.now += 2 * 86400
        self.memory.prune()
        self.ingest(
            HookEvent("codex", "one", "progress", progress="Parser edge cases pass")
        )
        story = self.memory.connected_work(sessions_snapshot())[0].story
        self.assertEqual(story.intent, "Check parser edge cases")
        self.assertEqual(
            [request.details for request in story.requests],
            [IntentView("Improve parser"), IntentView("Check parser edge cases")],
        )
        self.assertEqual(
            [
                report.details
                for report in story.activity
                if isinstance(report.details, ProgressView)
            ],
            [ProgressView("Parser edge cases pass")],
        )

    def test_nested_ingestion_rolls_back_checkpoint_and_domain_changes_together(
        self,
    ) -> None:
        self.ingest(HookEvent("codex", "one", "intent", goal="Improve parser"))
        observation = CheckoutObservation(
            "/repo/parser",
            GitObservation(
                "/repo/parser",
                "feature",
                "head",
                0,
                0,
                0,
                0,
                "",
                0,
                0,
                "github.com/team/project",
            ),
            self.now,
        )
        with (
            self.assertRaisesRegex(RuntimeError, "rollback"),
            self.memory.transaction(),
        ):
            self.ingest(HookEvent("codex", "one", "session_start", repo="/repo/parser"))
            self.memory.work.observe(observation)
            raise RuntimeError("rollback")
        self.assertEqual(self.memory.load_repositories(), ())
        self.assertEqual(
            self.memory.connected_work(sessions_snapshot())[0].story.artifacts, ()
        )
        self.memory.close()
        self.memory = Memory(self.path, "fixture", clock=lambda: self.now)
        self.assertEqual(self.memory.load_repositories(), ())
        self.assertEqual(
            self.memory.connected_work(sessions_snapshot())[0].story.artifacts, ()
        )

    def test_attention_history_survives_pruning_restart_and_remains_pet_scoped(
        self,
    ) -> None:
        self.now = 2_000_000_000
        self.memory.save_attention_history("work-one", 2, "meaningful-revision")
        self.memory.prune()
        self.memory.close()
        self.memory = Memory(self.path, "fixture", clock=lambda: self.now)
        self.assertEqual(
            self.memory.load_attention_history(),
            {"work-one": HistoryRecord(2, "meaningful-revision")},
        )
        other = Memory(self.path, "other-pet", clock=lambda: self.now)
        try:
            self.assertEqual(other.load_attention_history(), {})
            other.save_attention_history("work-one", 7, "other-revision")
            self.memory.clear()
            self.assertEqual(self.memory.load_attention_history(), {})
            self.assertEqual(
                other.load_attention_history()["work-one"].last_selected, 7
            )
        finally:
            other.close()

    def test_version_changes_rebuild_local_memory_and_search(self) -> None:
        for version in (0, 6):
            with self.subTest(version=version):
                self.event("Parser inspection completed", goal="Improve parser")
                self.memory.record_speech("Parser work", SpeechReason.ROUTINE)
                self.memory.save_context(
                    "codex:one", "Improve parser", "Parser summary", ["Improve parser"]
                )
                self.ingest(HookEvent("codex", "one", "intent", goal="Improve parser"))
                self.observe_checkout()
                self.memory.close()
                with sqlite3.connect(self.path) as database:
                    database.execute("CREATE TABLE work_updates (content TEXT)")
                    database.execute("INSERT INTO work_updates VALUES ('obsolete')")
                    database.execute(f"PRAGMA user_version = {version}")
                self.memory = Memory(self.path, "fixture", clock=lambda: self.now)
                self.assertEqual(self.memory.recent(), [])
                self.assertEqual(self.memory.speech(), [])
                self.assertIsNone(self.memory.context("codex:one"))
                self.assertEqual(self.memory.connected_work(sessions_snapshot()), ())
                self.assertEqual(self.memory.projects(["/repo/parser"]), [])
                self.assertEqual(self.memory.load_repositories(), ())
                self.assertEqual(
                    self.memory.db.execute("PRAGMA user_version").fetchone()[0],
                    SCHEMA_VERSION,
                )
                self.assertIsNone(
                    self.memory.db.execute(
                        "SELECT name FROM sqlite_master WHERE name='work_updates'"
                    ).fetchone()
                )
                self.event("Parser validation completed", goal="Validate parser")
                self.assertEqual(
                    self.memory.db.execute(
                        "SELECT COUNT(*) FROM events_fts WHERE events_fts MATCH 'Validate'"
                    ).fetchone()[0],
                    1,
                )
                self.memory.clear()
                self.assertEqual(
                    self.memory.db.execute(
                        "SELECT COUNT(*) FROM events_fts WHERE events_fts MATCH 'Validate'"
                    ).fetchone()[0],
                    0,
                )

    def test_newer_schema_is_rejected_without_modifying_saved_memory(self) -> None:
        self.event("Parser inspection completed")
        self.memory.record_speech("Parser work", SpeechReason.ROUTINE)
        self.memory.close()
        newer = SCHEMA_VERSION + 1
        with sqlite3.connect(self.path) as database:
            database.execute(f"PRAGMA user_version = {newer}")
        with self.assertRaises(ValueError):
            Memory(self.path, "fixture", clock=lambda: self.now)
        with sqlite3.connect(self.path) as database:
            self.assertEqual(
                database.execute("PRAGMA user_version").fetchone()[0], newer
            )
            self.assertEqual(
                database.execute("SELECT text FROM speech").fetchone()[0], "Parser work"
            )
            database.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        self.memory = Memory(self.path, "fixture", clock=lambda: self.now)
        self.assertIn(
            "Parser inspection completed",
            asdict(self.memory.recent()[0].details)["progress"],
        )

    def test_failed_rebuild_preserves_saved_memory_and_schema_version(self) -> None:
        self.event("Parser inspection completed")
        self.memory.record_speech("Parser work", SpeechReason.ROUTINE)
        self.memory.close()
        with sqlite3.connect(self.path) as database:
            database.execute("PRAGMA user_version = 6")
        with (
            patch("pet_daemon.storage.migrations.WORK_SCHEMA", "INVALID SQL;"),
            self.assertRaises(sqlite3.OperationalError),
        ):
            Memory(self.path, "fixture", clock=lambda: self.now)
        with sqlite3.connect(self.path) as database:
            self.assertEqual(database.execute("PRAGMA user_version").fetchone()[0], 6)
            self.assertEqual(
                database.execute("SELECT text FROM speech").fetchone()[0], "Parser work"
            )
            self.assertEqual(
                json.loads(
                    database.execute("SELECT details FROM events").fetchone()[0]
                ),
                {
                    "reported_outcome": None,
                    "activity": None,
                    "progress": "Parser inspection completed",
                },
            )
        self.memory = Memory(self.path, "fixture", clock=lambda: self.now)
        self.assertEqual(self.memory.recent(), [])
