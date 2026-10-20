import json
import sqlite3
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from pet_daemon.sources.hooks import HookEvent, normalize_hook
from pet_daemon.storage import Memory
from pet_daemon.storage.memory import (
    ContextRecord,
    HistoryRecord,
    RepositoryAssociation,
)
from pet_daemon.storage.migrations import SCHEMA_VERSION
from pet_daemon.types import JSONValue
from pet_daemon.work.history import EventContext, project_event, recall_query
from pet_daemon.work.observations import (
    RepositoryObservation,
)
from Tests.fixtures import require


class MemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "memory.sqlite3"
        self.now = 1000
        self.memory = Memory(self.path, "fixture", clock=lambda: self.now, max_events=3)

    def tearDown(self) -> None:
        self.memory.close()
        self.temporary.cleanup()

    def context(self, session: str = "codex:one") -> ContextRecord:
        context = self.memory.context(session)
        assert context is not None
        return context

    def event(self, progress: str, **extra: JSONValue) -> None:
        self.now += 1
        self.memory.record_event(
            project_event(
                require(
                    normalize_hook(
                        {
                            "harness": "codex",
                            "session": "one",
                            "event": "progress",
                            "repo": "/repo/parser",
                            "progress": progress,
                            **extra,
                        }
                    )
                )
            )
        )

    def test_events_survive_restart_and_remain_separate_from_speech(self) -> None:
        self.event("Parser checks passed")
        self.memory.record_speech("The parser checks passed.", "completion")
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
        self.memory.record_speech("First remark", "routine")
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
        self.memory.record_speech("Expired remark", "routine")
        self.now += 1799
        self.memory.record_speech("Recent remark", "routine")
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
        self.memory.record_speech("Long history remark", "routine")
        other = Memory(
            self.path,
            "short-history-pet",
            clock=lambda: self.now,
            speech_history_seconds=60,
        )
        try:
            other.record_speech("Short history remark", "routine")
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
            self.memory.record_speech(f"Remark {index}", "routine")
        rows = self.memory.speech(limit=300)
        self.assertEqual(len(rows), 256)
        self.assertEqual(rows[0].text, "Remark 1")
        self.assertEqual(rows[-1].text, "Remark 256")

    def test_clear_speech_preserves_knowledge_and_other_pet_speech(self) -> None:
        self.event("Parser checks passed", goal="Improve parser")
        self.memory.record_speech("Parser work", "routine")
        self.memory.save_context(
            "codex:one", "Improve parser", "Parser summary", ["Improve parser"]
        )
        self.memory.associate_repository("codex:one", "/repo/parser")
        checkpoint = RepositoryObservation(
            subject="/repo/parser:feature", worktree="/repo/parser", branch="feature"
        )
        self.memory.save_repository(checkpoint)
        other = Memory(self.path, "other-pet", clock=lambda: self.now)
        try:
            other.record_speech("Other pet remark", "routine")
            self.memory.clear_speech()
            self.assertEqual(self.memory.speech(), [])
            self.assertEqual(other.speech()[0].text, "Other pet remark")
            self.assertEqual(self.context().session_context, "Parser summary")
            self.assertIn(
                "Parser checks passed", json.dumps(self.memory.recent(), default=asdict)
            )
            self.assertEqual(
                self.memory.projects(["/repo/parser"])[0].goal, "Improve parser"
            )
            self.assertEqual(
                self.memory.repository_associations(),
                [RepositoryAssociation("codex:one", "/repo/parser")],
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
                memory.record_event(project_event(event))
                memory.record_speech("Parser work", "prompt")
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
            self.memory.record_event(project_event(event))
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
        self.memory.record_speech("Fixture remark", "completion")
        other = Memory(self.path, "other-pet", clock=lambda: self.now)
        try:
            self.assertEqual(other.recent(), [])
            self.assertEqual(other.speech(), [])
        finally:
            other.close()

    def test_recent_results_and_speech_obey_requested_limits(self) -> None:
        for index in range(3):
            self.event(f"Result {index}")
            self.memory.record_speech(f"Remark {index}", "routine")
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
        parser = self.memory.recent(limit=2, repo="/repo/parser")
        self.assertEqual([item.session for item in parser], ["codex:one", "codex:two"])
        self.assertTrue(all(item.repo == "/repo/parser" for item in parser))
        session = self.memory.recent(limit=2, session="codex:one")
        self.assertEqual(
            [item.repo for item in session], ["/repo/parser", "/repo/billing"]
        )
        intersection = self.memory.recent(session="codex:one", repo="/repo/parser")
        self.assertEqual(len(intersection), 1)
        self.assertIn("Parser first", intersection[0].summary)
        self.assertEqual(self.memory.recent(session="codex:missing"), [])
        self.assertEqual(self.memory.recent(repo="/repo/missing"), [])
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
            [row.summary.split(" | ")[-1] for row in rows],
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
            project_event(
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
                EventContext(
                    id="codex:one",
                    repo="/repo/parser",
                    goal="Stale inherited goal",
                    last_tool="",
                    outcome=None,
                    activity="testing",
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
            project_event(
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
                EventContext(
                    id="codex:unknown",
                    repo="/repo/new",
                    goal="Inherited task",
                    last_tool="",
                    outcome=None,
                    activity="command",
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
                    project_event(
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
                        EventContext(
                            id="codex:one",
                            repo="",
                            goal="",
                            last_tool="pytest",
                            outcome=None,
                            activity="testing",
                        ),
                    )
                )
                summary = self.memory.recent()[-1].summary
                self.assertIn("Turn completed (agent reported success)", summary)
                self.assertIn("Finished reviewing parser behavior", summary)
                self.assertNotIn("Tests succeeded", summary)
        self.memory.record_event(
            project_event(
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
        self.assertNotIn("Tests succeeded", self.memory.recent()[-1].summary)

    def test_recall_matches_older_goals_within_the_repository(self) -> None:
        self.memory.close()
        self.memory = Memory(
            self.path, "fixture", clock=lambda: self.now, max_events=100
        )
        self.event("", event="intent", goal="Refactor parser tokenization")
        self.event(
            "", event="intent", repo="/repo/other", goal="Refactor parser tokenization"
        )
        self.event("", event="intent", goal="Improve billing invoices")
        for index in range(6):
            self.event(f"Recent progress {index}")
        recalled = self.memory.recall(
            recall_query(
                normalize_hook(
                    {
                        "harness": "codex",
                        "session": "one",
                        "event": "intent",
                        "repo": "/repo/parser",
                        "goal": "Refactor parser tokenization",
                    }
                )
            )
        )
        self.assertEqual(len(recalled), 1)
        self.assertIn("parser tokenization", recalled[0].goal)
        self.assertEqual(recalled[0].repo, "/repo/parser")
        self.assertNotIn(recalled[0].id, [event.id for event in self.memory.recent()])

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
                        recall_query(
                            normalize_hook(
                                {
                                    "harness": "codex",
                                    "session": "one",
                                    "repo": "/repo/parser",
                                    **event,
                                }
                            )
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
        self.assertLessEqual(len(event.summary), 320)
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
            project_event(
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
                EventContext(
                    id="claude:one",
                    repo="",
                    goal="",
                    last_tool="",
                    outcome="error",
                    activity="command",
                ),
            )
        )
        summary = self.memory.recent()[0].summary
        self.assertIn("failed", summary)
        self.assertNotIn("success", summary)

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
            project_event(HookEvent("claude", "one", "interrupt", outcome="cancelled"))
        )
        self.assertIn("Agent run cancelled", self.memory.recent()[-1].summary)

    def test_acceptance_commits_speech_outcome_and_considered_story_together(
        self,
    ) -> None:
        self.assertTrue(
            self.memory.record_acceptance(
                "message",
                "Parser covered",
                "routine",
                "work",
                "revision",
                50,
                "revision",
            )
        )
        self.assertFalse(
            self.memory.record_acceptance(
                "message", "Duplicate", "routine", "work", "new", 60, "new"
            )
        )
        outcome = require(self.memory.message_outcome("message"))
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
            "message", "Earlier report", "routine", "work", "old"
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
                "routine",
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
                "timeout",
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
                "routine",
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
                "timeout",
                "Parser covered",
                "work",
                "revision",
                50,
                "revision",
            )
        )
        self.assertFalse(
            self.memory.record_delivery_failure(
                "message", "duplicate", subject_id="work", considered_revision="other"
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
                "message", "Parser covered", "routine", "work", "revision"
            )
        )
        self.assertEqual(
            require(self.memory.message_outcome("message")).outcome, "accepted"
        )

    def test_schema_seven_upgrade_preserves_history_and_ignores_legacy_actions(
        self,
    ) -> None:
        self.assert_upgrade_preserves_history_and_ignores_legacy_actions(7)

    def test_schema_eight_upgrade_preserves_history_and_ignores_legacy_actions(
        self,
    ) -> None:
        self.assert_upgrade_preserves_history_and_ignores_legacy_actions(8)

    def assert_upgrade_preserves_history_and_ignores_legacy_actions(
        self, version: int
    ) -> None:
        from pet_daemon.storage.migrations import (
            DELIVERY_SCHEMA,
            MEMORY_SCHEMA,
            WORK_SCHEMA,
            execute_schema,
        )

        path = Path(self.temporary.name) / f"legacy-{version}.sqlite3"
        legacy_delivery = DELIVERY_SCHEMA.replace(
            "timestamp REAL NOT NULL,\n                subject_id TEXT, revision TEXT,",
            "opportunity_id TEXT, version INTEGER, timestamp REAL NOT NULL,",
        )
        with sqlite3.connect(path) as database:
            execute_schema(database, MEMORY_SCHEMA)
            execute_schema(database, WORK_SCHEMA)
            database.execute("""CREATE TABLE work_opportunities (
                pet TEXT NOT NULL, id TEXT NOT NULL, work_id TEXT NOT NULL, kind TEXT NOT NULL,
                artifact_id TEXT NOT NULL, version INTEGER NOT NULL, fingerprint TEXT NOT NULL,
                data TEXT NOT NULL, active INTEGER NOT NULL, surfaced_version INTEGER NOT NULL,
                changed_at REAL NOT NULL, PRIMARY KEY(pet,id))""")
            database.execute("""CREATE TABLE work_surfacings (
                id INTEGER PRIMARY KEY, pet TEXT NOT NULL, opportunity_id TEXT NOT NULL,
                version INTEGER NOT NULL, outcome TEXT NOT NULL, timestamp REAL NOT NULL)""")
            execute_schema(database, legacy_delivery)
            database.execute(f"PRAGMA user_version = {version}")
            database.execute(
                "INSERT INTO speech VALUES (1,'fixture','Saved remark','routine',1000)"
            )
            database.execute(
                "INSERT INTO message_outcomes VALUES ('fixture','legacy','accepted','Saved remark','routine','opportunity',3,1000)"
            )
            database.execute(
                "INSERT INTO work_items VALUES ('fixture','work','Improve parser','active',1000)"
            )
            database.execute(
                "INSERT INTO work_opportunities VALUES ('fixture','opportunity','work','resolve_checks','artifact',3,'fingerprint','not decoded',1,3,1000)"
            )
            database.execute(
                "INSERT INTO work_surfacings VALUES (1,'fixture','opportunity',3,'accepted',1000)"
            )
        upgraded = Memory(path, "fixture", clock=lambda: self.now)
        try:
            self.assertEqual(upgraded.speech()[0].text, "Saved remark")
            self.assertEqual(
                require(upgraded.message_outcome("legacy")).subject_id, None
            )
            upgraded.work.session_intent("codex:one", "More parser work")
            self.assertEqual(len(upgraded.work.work_items()), 2)
            self.assertEqual(
                upgraded.db.execute("SELECT data FROM work_opportunities").fetchone()[
                    0
                ],
                "not decoded",
            )
            self.assertEqual(
                upgraded.db.execute("SELECT COUNT(*) FROM work_surfacings").fetchone()[
                    0
                ],
                1,
            )
            upgraded.record_acceptance(
                "new", "More parser coverage", "routine", "work", "revision"
            )
            self.assertEqual(
                require(upgraded.message_outcome("new")).revision, "revision"
            )
            upgraded.clear()
            self.assertEqual(
                upgraded.db.execute(
                    "SELECT COUNT(*) FROM work_opportunities"
                ).fetchone()[0],
                0,
            )
            self.assertEqual(
                upgraded.db.execute("SELECT COUNT(*) FROM work_surfacings").fetchone()[
                    0
                ],
                0,
            )
        finally:
            upgraded.close()

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
        self.assertEqual(context.first_prompt, "Refactor parser")
        self.assertEqual(context.session_context, "Parser refactor awaits validation")
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
        self.assertEqual(self.context().session_context, summary)

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
        self.memory.record_speech("Parser work", "prompt")
        self.memory.associate_repository("codex:one", "/repo/parser")
        checkpoint = RepositoryObservation(
            subject="branch", worktree="/repo/parser", branch="feature"
        )
        self.memory.save_repository(checkpoint)
        self.memory.close()
        with sqlite3.connect(self.path) as database:
            database.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        self.memory = Memory(self.path, "fixture", clock=lambda: self.now)
        self.assertTrue(self.memory.recent())
        self.assertEqual(self.memory.speech()[0].text, "Parser work")
        self.assertEqual(
            self.context().session_context,
            "Parser summary",
        )
        self.assertEqual(
            self.memory.projects(["/repo/parser"])[0].goal, "Improve parser"
        )
        self.assertEqual(
            self.memory.repository_associations(),
            [RepositoryAssociation("codex:one", "/repo/parser")],
        )
        self.assertEqual(self.memory.load_repositories(), (checkpoint,))
        self.assertEqual(
            self.memory.db.execute("PRAGMA user_version").fetchone()[0], SCHEMA_VERSION
        )

    def test_connected_work_is_pet_scoped_and_clears_with_memory(self) -> None:
        other = Memory(self.path, "other-pet", clock=lambda: self.now)
        try:
            for memory in (self.memory, other):
                memory.work.session_intent("codex:one", "Improve parser")
                memory.work.observe_repository(
                    RepositoryObservation(
                        "fixture",
                        "/repo/parser",
                        "feature",
                        sessions=("codex:one",),
                        repository="github.com/team/project",
                    )
                )
            self.assertEqual(
                len(require(self.memory.work.session_context("codex:one")).artifacts), 1
            )
            self.memory.clear()
            self.memory.close()
            self.memory = Memory(self.path, "fixture", clock=lambda: self.now)
            self.assertEqual(self.memory.work.tracked_artifacts(), ())
            self.assertIsNone(self.memory.work.session_context("codex:one"))
            self.assertEqual(
                len(require(other.work.session_context("codex:one")).artifacts), 1
            )
        finally:
            other.close()

    def test_work_evidence_retention_uses_configured_memory_retention(self) -> None:
        self.memory.close()
        self.memory = Memory(
            self.path, "fixture", clock=lambda: self.now, retention_days=1
        )
        self.memory.work.session_intent("codex:one", "Improve parser")
        self.memory.work.session_intent("codex:one", "Check parser edge cases")
        self.now += 2 * 86400
        self.memory.prune()
        rows = self.memory.db.execute(
            "SELECT data FROM work_evidence WHERE pet=? AND kind='intent'", ("fixture",)
        ).fetchall()
        self.assertEqual(len(rows), 1)
        self.assertIn("Check parser edge cases", rows[0]["data"])
        item = self.memory.work.work_for_session("codex:one")
        assert item is not None
        self.assertEqual(
            item.intent,
            "Check parser edge cases",
        )

    def test_work_graph_creates_only_tables_with_automatic_producers(self) -> None:
        for name in (
            "work_updates",
            "work_decisions",
            "work_opportunities",
            "work_surfacings",
        ):
            self.assertIsNone(
                self.memory.db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                    (name,),
                ).fetchone()
            )
        self.memory.work.session_intent("codex:one", "Improve parser")
        self.memory.work.observe_repository(
            RepositoryObservation(
                "fixture",
                "/repo/parser",
                "feature",
                sessions=("codex:one",),
                repository="github.com/team/project",
            )
        )
        self.memory.close()
        self.memory = Memory(self.path, "fixture", clock=lambda: self.now)
        self.assertEqual(self.memory.work.tracked_artifacts()[0].ref, "feature")

    def test_nested_ingestion_rolls_back_checkpoint_and_domain_changes_together(
        self,
    ) -> None:
        self.memory.work.session_intent("codex:one", "Improve parser")
        observation = RepositoryObservation(
            "fixture",
            "/repo/parser",
            "feature",
            sessions=("codex:one",),
            repository="github.com/team/project",
        )
        with (
            self.assertRaisesRegex(RuntimeError, "rollback"),
            self.memory.transaction(),
        ):
            self.memory.save_repository(observation)
            self.memory.work.observe_repository(observation)
            raise RuntimeError("rollback")
        self.assertEqual(self.memory.load_repositories(), ())
        self.assertEqual(
            require(self.memory.work.session_context("codex:one")).artifacts, ()
        )
        self.memory.close()
        self.memory = Memory(self.path, "fixture", clock=lambda: self.now)
        self.assertEqual(self.memory.load_repositories(), ())
        self.assertEqual(self.memory.work.tracked_artifacts(), ())

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
        for version in (0, 3, 4, 5, 6, SCHEMA_VERSION + 1):
            with self.subTest(version=version):
                self.event("Parser inspection completed", goal="Improve parser")
                self.memory.record_speech("Parser work", "routine")
                self.memory.save_context(
                    "codex:one", "Improve parser", "Parser summary", ["Improve parser"]
                )
                self.memory.work.session_intent("codex:one", "Improve parser")
                self.memory.associate_repository("codex:one", "/repo/parser")
                self.memory.save_repository(
                    RepositoryObservation("checkpoint", "/repo/parser", "feature")
                )
                self.memory.close()
                with sqlite3.connect(self.path) as database:
                    database.execute("CREATE TABLE work_updates (content TEXT)")
                    database.execute("INSERT INTO work_updates VALUES ('obsolete')")
                    database.execute(f"PRAGMA user_version = {version}")
                self.memory = Memory(self.path, "fixture", clock=lambda: self.now)
                self.assertEqual(self.memory.recent(), [])
                self.assertEqual(self.memory.speech(), [])
                self.assertIsNone(self.memory.context("codex:one"))
                self.assertIsNone(self.memory.work.work_for_session("codex:one"))
                self.assertEqual(self.memory.projects(["/repo/parser"]), [])
                self.assertEqual(self.memory.repository_associations(), [])
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

    def test_failed_rebuild_preserves_saved_memory_and_schema_version(self) -> None:
        self.event("Parser inspection completed")
        self.memory.record_speech("Parser work", "routine")
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
                database.execute("SELECT summary FROM events").fetchone()[0],
                "progress | Parser inspection completed",
            )
        self.memory = Memory(self.path, "fixture", clock=lambda: self.now)
        self.assertEqual(self.memory.recent(), [])
