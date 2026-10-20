import json
import sqlite3
from pathlib import Path
import tempfile
import unittest

from pet_daemon.storage import Memory
from pet_daemon.work.history import project_event, recall_query
from pet_daemon.work.observations import RepositoryObservation


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "memory.sqlite3"
        self.now = 1000
        self.memory = Memory(self.path, "fixture", clock=lambda: self.now, max_events=3)

    def tearDown(self):
        self.memory.close()
        self.temporary.cleanup()

    def event(self, progress, **extra):
        self.now += 1
        self.memory.record_event(
            project_event(
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

    def test_events_survive_restart_and_remain_separate_from_speech(self):
        self.event("Parser checks passed")
        self.memory.record_speech("The parser checks passed.", "completion")
        self.memory.close()
        self.memory = Memory(self.path, "fixture", clock=lambda: self.now)
        self.assertIn("Parser checks passed", json.dumps(self.memory.recent()))
        self.assertIn("The parser checks passed.", json.dumps(self.memory.speech()))
        self.assertNotIn("The parser checks passed.", json.dumps(self.memory.recent()))
        self.assertEqual(self.path.read_bytes()[:16], b"SQLite format 3\x00")

    def test_speech_expires_at_thirty_minutes_without_a_prune(self):
        self.memory.record_speech("First remark", "routine")
        self.now += 1799
        self.assertEqual(self.memory.speech()[0]["text"], "First remark")
        self.now += 1
        self.assertEqual(self.memory.speech(), [])
        self.assertEqual(
            self.memory.db.execute("SELECT COUNT(*) FROM speech").fetchone()[0], 1
        )
        self.memory.prune()
        self.assertEqual(
            self.memory.db.execute("SELECT COUNT(*) FROM speech").fetchone()[0], 0
        )

    def test_speech_expiry_survives_restart_and_uses_its_own_retention(self):
        self.memory.record_speech("Expired remark", "routine")
        self.now += 1799
        self.memory.record_speech("Recent remark", "routine")
        self.memory.close()
        self.now += 1
        self.memory = Memory(
            self.path, "fixture", clock=lambda: self.now, retention_days=0
        )
        self.assertEqual(
            [item["text"] for item in self.memory.speech()], ["Recent remark"]
        )
        self.now += 1
        self.memory.prune()
        self.assertEqual(
            [item["text"] for item in self.memory.speech()], ["Recent remark"]
        )
        self.memory.close()
        self.memory = Memory(
            self.path, "fixture", clock=lambda: self.now, speech_history_seconds=1
        )
        self.assertEqual(self.memory.speech(), [])

    def test_short_speech_retention_preserves_another_pets_valid_history(self):
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
            self.assertEqual(self.memory.speech()[0]["text"], "Long history remark")
            self.memory.close()
            self.memory = Memory(self.path, "fixture", clock=lambda: self.now)
            self.assertEqual(self.memory.speech()[0]["text"], "Long history remark")
        finally:
            other.close()

    def test_speech_history_keeps_the_global_row_limit(self):
        for index in range(257):
            self.memory.record_speech(f"Remark {index}", "routine")
        rows = self.memory.speech(limit=300)
        self.assertEqual(len(rows), 256)
        self.assertEqual(rows[0]["text"], "Remark 1")
        self.assertEqual(rows[-1]["text"], "Remark 256")

    def test_clear_speech_preserves_knowledge_and_other_pet_speech(self):
        self.event("Parser checks passed", goal="Improve parser")
        self.memory.record_speech("Parser work", "routine")
        self.memory.save_context(
            "codex:one", "Improve parser", "Parser summary", ["Improve parser"]
        )
        self.memory.associate_repository("codex:one", "/repo/parser")
        self.memory.save_repository("/repo/parser:feature", {"pending": True})
        other = Memory(self.path, "other-pet", clock=lambda: self.now)
        try:
            other.record_speech("Other pet remark", "routine")
            self.memory.clear_speech()
            self.assertEqual(self.memory.speech(), [])
            self.assertEqual(other.speech()[0]["text"], "Other pet remark")
            self.assertEqual(
                self.memory.context("codex:one")["session_context"], "Parser summary"
            )
            self.assertIn("Parser checks passed", json.dumps(self.memory.recent()))
            self.assertEqual(
                self.memory.projects(["/repo/parser"])[0]["goal"], "Improve parser"
            )
            self.assertEqual(
                self.memory.repository_associations(),
                [{"session": "codex:one", "worktree": "/repo/parser"}],
            )
            self.assertEqual(self.memory.load_repositories(), [{"pending": True}])
            self.memory.close()
            self.memory = Memory(self.path, "fixture", clock=lambda: self.now)
            self.assertEqual(self.memory.speech(), [])
            self.assertTrue(self.memory.recent())
        finally:
            other.close()

    def test_clear_erases_saved_knowledge_for_the_current_pet(self):
        other = Memory(self.path, "other-pet", clock=lambda: self.now)
        event = {
            "harness": "codex",
            "session": "one",
            "event": "intent",
            "repo": "/repo/parser",
            "goal": "Improve parser",
        }
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

    def test_event_count_discards_oldest_facts(self):
        for progress in (
            "First result",
            "Second result",
            "Third result",
            "Fourth result",
        ):
            self.event(progress)
        events = self.memory.recent(limit=10)
        self.assertEqual(len(events), 3)
        self.assertNotIn("First result", json.dumps(events))
        self.assertIn("Fourth result", json.dumps(events))

    def test_retention_discards_expired_facts(self):
        self.event("Expired result")
        self.now += 31 * 86400
        self.event("Current result")
        events = json.dumps(self.memory.recent(limit=10))
        self.assertNotIn("Expired result", events)
        self.assertIn("Current result", events)

    def test_pet_identity_keeps_memory_separate(self):
        self.event("Fixture parser result")
        self.memory.record_speech("Fixture remark", "completion")
        other = Memory(self.path, "other-pet", clock=lambda: self.now)
        try:
            self.assertEqual(other.recent(), [])
            self.assertEqual(other.speech(), [])
        finally:
            other.close()

    def test_recent_results_and_speech_obey_requested_limits(self):
        for index in range(3):
            self.event(f"Result {index}")
            self.memory.record_speech(f"Remark {index}", "routine")
        self.assertEqual(len(self.memory.recent(limit=2)), 2)
        self.assertEqual(len(self.memory.speech(limit=2)), 2)

    def test_recent_history_filters_subjects_before_ranking_sessions(self):
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
        self.assertEqual(
            [item["session"] for item in parser], ["codex:one", "codex:two"]
        )
        self.assertTrue(all(item["repo"] == "/repo/parser" for item in parser))
        session = self.memory.recent(limit=2, session="codex:one")
        self.assertEqual(
            [item["repo"] for item in session], ["/repo/parser", "/repo/billing"]
        )
        intersection = self.memory.recent(session="codex:one", repo="/repo/parser")
        self.assertEqual(len(intersection), 1)
        self.assertIn("Parser first", intersection[0]["summary"])
        self.assertEqual(self.memory.recent(session="codex:missing"), [])
        self.assertEqual(self.memory.recent(repo="/repo/missing"), [])
        self.assertEqual(len(self.memory.recent(limit=5)), 5)

    def test_terminal_history_filters_events_before_recent_tool_activity(self):
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
        self.assertTrue(all(row["session"] == "codex:one" for row in rows))
        self.assertTrue(all(row["event_type"] == "turn_done" for row in rows))
        self.assertEqual(
            [row["summary"].split(" | ")[-1] for row in rows],
            ["Report 1", "Report 2", "Report 3"],
        )
        self.assertEqual(self.memory.recent(event_types=()), [])
        self.assertEqual(self.memory.recent(event_types=("turn_done') OR 1=1 --",)), [])

    def test_project_goal_and_request_time_change_only_for_explicit_requests(self):
        self.event("", event="intent", goal="Improve parser")
        original = self.memory.projects(["/repo/parser"])[0]
        self.now += 30
        self.memory.record_event(
            project_event(
                {
                    "harness": "codex",
                    "session": "one",
                    "event": "tool_done",
                    "tool": "pytest",
                    "outcome": "success",
                },
                {
                    "id": "codex:one",
                    "repo": "/repo/parser",
                    "goal": "Stale inherited goal",
                },
            )
        )
        self.assertEqual(self.memory.projects(["/repo/parser"])[0], original)
        self.event("Current progress")
        self.assertEqual(self.memory.projects(["/repo/parser"])[0], original)
        self.event("", event="intent", goal="Document parser")
        updated = self.memory.projects(["/repo/parser"])[0]
        self.assertEqual(updated["goal"], "Document parser")
        self.assertGreater(updated["timestamp"], original["timestamp"])
        self.memory.record_event(
            project_event(
                {
                    "harness": "codex",
                    "session": "unknown",
                    "event": "progress",
                    "progress": "Working",
                },
                {"id": "codex:unknown", "repo": "/repo/new", "goal": "Inherited task"},
            )
        )
        self.assertEqual(self.memory.projects(["/repo/new"]), [])

    def test_terminal_hooks_report_turn_completion_without_claiming_tool_success(self):
        for event in ("turn_done", "subagent_done", "session_end"):
            with self.subTest(event=event):
                self.memory.record_event(
                    project_event(
                        {
                            "harness": "codex",
                            "session": "one",
                            "event": event,
                            "outcome": "success",
                            "progress": "Finished reviewing parser behavior",
                        },
                        {
                            "id": "codex:one",
                            "last_tool": "pytest",
                            "activity": "testing",
                        },
                    )
                )
                summary = self.memory.recent()[-1]["summary"]
                self.assertIn("Turn completed (agent reported success)", summary)
                self.assertIn("Finished reviewing parser behavior", summary)
                self.assertNotIn("Tests succeeded", summary)
        self.memory.record_event(
            project_event(
                {
                    "harness": "codex",
                    "session": "one",
                    "event": "tool_done",
                    "tool": "pytest",
                    "outcome": "success",
                }
            )
        )
        self.assertIn("Tests succeeded", self.memory.recent()[-1]["summary"])

    def test_recall_matches_older_goals_within_the_repository(self):
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
                {
                    "event": "intent",
                    "repo": "/repo/parser",
                    "goal": "Refactor parser tokenization",
                    "harness": "codex",
                    "session": "one",
                }
            )
        )
        self.assertEqual(len(recalled), 1)
        self.assertIn("parser tokenization", recalled[0]["goal"])
        self.assertEqual(recalled[0]["repo"], "/repo/parser")
        self.assertNotIn(
            recalled[0]["id"], [event["id"] for event in self.memory.recent()]
        )

    def test_recall_requires_an_explicit_goal_at_task_start(self):
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
                    self.memory.recall(recall_query({"repo": "/repo/parser", **event})),
                    [],
                )

    def test_failure_facts_exclude_injected_diagnostics(self):
        self.event(
            "SECRET FAILED COMMAND OUTPUT",
            event="tool_error",
            detail="SECRET STACK TRACE",
            error_signature="SECRET HASH",
            tool="pytest",
        )
        rows = self.memory.recent()
        self.assertNotIn("SECRET", json.dumps(rows))
        self.assertNotIn("error_signature", rows[0])

    def test_long_factual_fields_are_bounded_before_storage(self):
        self.event("猫" * 100000)
        event = self.memory.recent()[0]
        self.assertLessEqual(len(event["summary"]), 320)
        self.assertNotIn("thought", event)
        self.assertNotIn("speak", event)

    def test_recent_facts_keep_a_quiet_session_visible(self):
        self.memory.close()
        self.memory = Memory(
            self.path, "fixture", clock=lambda: self.now, max_events=100
        )
        self.event("Quiet session remains blocked", session="quiet")
        for index in range(8):
            self.event(f"Busy session progress {index}")
        events = self.memory.recent(limit=5)
        self.assertIn("codex:quiet", [event["session"] for event in events])
        self.assertIn("Busy session progress 7", json.dumps(events))

    def test_terminal_fact_preserves_reconciled_failure(self):
        self.memory.record_event(
            project_event(
                {
                    "harness": "claude",
                    "session": "one",
                    "event": "turn_done",
                    "outcome": "success",
                },
                {"id": "claude:one", "outcome": "error"},
            )
        )
        summary = self.memory.recent()[0]["summary"]
        self.assertIn("failed", summary)
        self.assertNotIn("success", summary)

    def test_duplicate_progress_does_not_displace_distinct_facts(self):
        self.event("First result")
        self.event("Second result")
        for _ in range(5):
            self.event("Second result")
        events = self.memory.recent()
        self.assertEqual(len(events), 2)
        self.assertIn("First result", json.dumps(events))

    def test_failure_facts_describe_broad_activity_and_cancellation(self):
        for tool, category, expected in (
            ("pytest", "testing", "Tests failed"),
            ("cargo build", "building", "Build failed"),
            ("./release.sh", "command", "Command failed"),
            ("", "harness", "Agent run failed"),
        ):
            with self.subTest(activity=category):
                self.memory.record_event(
                    project_event(
                        {
                            "harness": "claude",
                            "session": "one",
                            "event": "tool_error",
                            "tool": tool,
                        },
                        {"id": "claude:one", "activity": category},
                    )
                )
                self.assertIn(expected, self.memory.recent()[-1]["summary"])
        self.memory.record_event(
            project_event(
                {
                    "harness": "claude",
                    "session": "one",
                    "event": "interrupt",
                    "outcome": "cancelled",
                },
                {"id": "claude:one", "activity": "harness", "outcome": "cancelled"},
            )
        )
        self.assertIn("Agent run cancelled", self.memory.recent()[-1]["summary"])

    def test_unsupported_schema_preserves_history_and_refuses_to_open(self):
        self.event("Parser inspection completed")
        self.memory.save_context(
            "codex:one", "Refactor parser", "Keep its API stable", ["Refactor parser"]
        )
        self.memory.record_speech("Parser work", "prompt")
        self.memory.close()
        with sqlite3.connect(self.path) as database:
            database.execute("PRAGMA user_version = 3")
        with self.assertRaisesRegex(ValueError, "Unsupported memory schema version: 3"):
            Memory(self.path, "fixture", clock=lambda: self.now)
        with sqlite3.connect(self.path) as database:
            self.assertEqual(database.execute("PRAGMA user_version").fetchone()[0], 3)
            self.assertEqual(
                database.execute("SELECT text FROM speech").fetchone()[0], "Parser work"
            )
            self.assertEqual(
                database.execute("SELECT session_context FROM contexts").fetchone()[0],
                "Keep its API stable",
            )
            database.execute("PRAGMA user_version = 7")
        self.memory = Memory(self.path, "fixture", clock=lambda: self.now)
        self.assertIn("Parser inspection completed", json.dumps(self.memory.recent()))

    def test_session_context_survives_restart_with_its_original_anchor(self):
        self.memory.save_context(
            "codex:one",
            "Refactor parser",
            "Parser refactor awaits validation",
            ["Refactor parser", "Check edge cases"],
        )
        self.memory.close()
        self.memory = Memory(self.path, "fixture", clock=lambda: self.now)
        context = self.memory.context("codex:one")
        self.assertEqual(context["first_prompt"], "Refactor parser")
        self.assertEqual(
            context["session_context"], "Parser refactor awaits validation"
        )
        self.assertEqual(
            context["recent_requests"], ["Refactor parser", "Check edge cases"]
        )
        self.assertEqual(self.memory.recent(), [])
        self.assertIsNone(self.memory.context("codex:other"))

    def test_full_session_summary_survives_restart(self):
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
        self.assertEqual(self.memory.context("codex:one")["session_context"], summary)

    def test_pet_identity_keeps_session_context_separate(self):
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

    def test_supported_migrations_preserve_saved_memory_and_repositories(self):
        for version in (4, 5, 6):
            with self.subTest(version=version):
                self.event("Parser inspection completed", goal="Improve parser")
                self.memory.save_context(
                    "codex:one", "Improve parser", "Parser summary", ["Improve parser"]
                )
                self.memory.record_speech("Parser work", "prompt")
                self.memory.associate_repository("codex:one", "/repo/parser")
                self.memory.save_repository("branch", {"worktree": "/repo/parser"})
                self.memory.close()
                with sqlite3.connect(self.path) as database:
                    database.execute(f"PRAGMA user_version = {version}")
                self.memory = Memory(self.path, "fixture", clock=lambda: self.now)
                self.assertTrue(self.memory.recent())
                self.assertEqual(self.memory.speech()[0]["text"], "Parser work")
                self.assertEqual(
                    self.memory.context("codex:one")["session_context"],
                    "Parser summary",
                )
                self.assertEqual(
                    self.memory.projects(["/repo/parser"])[0]["goal"], "Improve parser"
                )
                self.assertEqual(
                    self.memory.repository_associations(),
                    [{"session": "codex:one", "worktree": "/repo/parser"}],
                )
                self.assertEqual(
                    self.memory.load_repositories(), [{"worktree": "/repo/parser"}]
                )
                self.assertEqual(
                    self.memory.db.execute("PRAGMA user_version").fetchone()[0], 7
                )

    def test_connected_work_is_pet_scoped_and_clears_with_memory(self):
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
                len(self.memory.work.session_context("codex:one")["artifacts"]), 1
            )
            self.memory.clear()
            self.memory.close()
            self.memory = Memory(self.path, "fixture", clock=lambda: self.now)
            self.assertEqual(self.memory.work.tracked_artifacts(), ())
            self.assertEqual(self.memory.work.session_context("codex:one"), {})
            self.assertEqual(
                len(other.work.session_context("codex:one")["artifacts"]), 1
            )
        finally:
            other.close()

    def test_work_evidence_retention_uses_configured_memory_retention(self):
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
        self.assertEqual(
            self.memory.work.work_for_session("codex:one").intent,
            "Check parser edge cases",
        )

    def test_work_graph_creates_only_tables_with_automatic_producers(self):
        for name in ("work_updates", "work_decisions"):
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
        self.assertEqual(self.memory.work.tracked_artifacts()[0]["ref"], "feature")

    def test_legacy_command_history_survives_startup_and_clear_forgets_only_the_current_pet(
        self,
    ):
        self.memory.db.execute("""CREATE TABLE work_updates (
            pet TEXT NOT NULL, id TEXT NOT NULL, content TEXT NOT NULL, acceptance TEXT NOT NULL,
            timestamp REAL NOT NULL, PRIMARY KEY(pet,id))""")
        with self.memory.db:
            self.memory.db.executemany(
                "INSERT INTO work_updates VALUES (?,?,?,?,?)",
                [
                    (
                        pet,
                        "legacy-report",
                        '{"fact":"preserved"}',
                        '{"status":"accepted"}',
                        self.now,
                    )
                    for pet in ("fixture", "other-pet")
                ],
            )
        self.memory.close()
        self.memory = Memory(self.path, "fixture", clock=lambda: self.now)
        rows = self.memory.db.execute(
            "SELECT pet,content FROM work_updates ORDER BY pet"
        ).fetchall()
        self.assertEqual(
            [dict(row) for row in rows],
            [
                {"pet": "fixture", "content": '{"fact":"preserved"}'},
                {"pet": "other-pet", "content": '{"fact":"preserved"}'},
            ],
        )
        self.memory.clear()
        rows = self.memory.db.execute("SELECT pet FROM work_updates").fetchall()
        self.assertEqual([row["pet"] for row in rows], ["other-pet"])

    def test_legacy_decisions_survive_startup_and_clear_forgets_only_the_current_pet(
        self,
    ):
        self.memory.db.execute("""CREATE TABLE work_decisions (
            pet TEXT NOT NULL, work_id TEXT NOT NULL, target TEXT NOT NULL, action TEXT NOT NULL,
            state TEXT NOT NULL, until_at REAL, data TEXT NOT NULL,
            PRIMARY KEY(pet,work_id,target,action))""")
        with self.memory.db:
            self.memory.db.executemany(
                "INSERT INTO work_decisions VALUES (?,?,?,?,?,?,?)",
                [
                    (
                        pet,
                        "legacy-work",
                        "work",
                        "",
                        "deferred",
                        None,
                        '{"source":"user"}',
                    )
                    for pet in ("fixture", "other-pet")
                ],
            )
        self.memory.close()
        self.memory = Memory(self.path, "fixture", clock=lambda: self.now)
        self.assertEqual(
            self.memory.db.execute("SELECT COUNT(*) FROM work_decisions").fetchone()[0],
            2,
        )
        self.memory.clear()
        rows = self.memory.db.execute("SELECT pet FROM work_decisions").fetchall()
        self.assertEqual([row["pet"] for row in rows], ["other-pet"])

    def test_nested_ingestion_rolls_back_checkpoint_and_domain_changes_together(self):
        self.memory.work.session_intent("codex:one", "Improve parser")
        observation = RepositoryObservation(
            "fixture",
            "/repo/parser",
            "feature",
            sessions=("codex:one",),
            repository="github.com/team/project",
        )
        with self.assertRaisesRegex(RuntimeError, "rollback"):
            with self.memory.transaction():
                self.memory.save_repository("fixture", {"worktree": "/repo/parser"})
                self.memory.work.observe_repository(observation)
                raise RuntimeError("rollback")
        self.assertEqual(self.memory.load_repositories(), [])
        self.assertEqual(self.memory.work.session_context("codex:one")["artifacts"], ())
        self.memory.close()
        self.memory = Memory(self.path, "fixture", clock=lambda: self.now)
        self.assertEqual(self.memory.load_repositories(), [])
        self.assertEqual(self.memory.work.tracked_artifacts(), ())

    def test_attention_history_survives_pruning_restart_and_remains_pet_scoped(self):
        self.now = 2_000_000_000
        self.memory.save_attention_history("work-one", 2, "meaningful-revision")
        self.memory.prune()
        self.memory.close()
        self.memory = Memory(self.path, "fixture", clock=lambda: self.now)
        self.assertEqual(
            self.memory.load_attention_history(),
            {
                "work-one": {
                    "last_selected": 2,
                    "considered_revision": "meaningful-revision",
                }
            },
        )
        other = Memory(self.path, "other-pet", clock=lambda: self.now)
        try:
            self.assertEqual(other.load_attention_history(), {})
            other.save_attention_history("work-one", 7, "other-revision")
            self.memory.clear()
            self.assertEqual(self.memory.load_attention_history(), {})
            self.assertEqual(
                other.load_attention_history()["work-one"]["last_selected"], 7
            )
        finally:
            other.close()

    def test_supported_migration_adds_context_fields_without_erasing_requests(self):
        self.memory.save_context(
            "codex:one", "Improve parser", "Keep the API stable", ["Cover escapes"]
        )
        self.memory.close()
        with sqlite3.connect(self.path) as database:
            database.execute("ALTER TABLE contexts DROP COLUMN request_count")
            database.execute("ALTER TABLE contexts DROP COLUMN first_summary_attempted")
            database.execute("PRAGMA user_version = 4")
        self.memory = Memory(self.path, "fixture", clock=lambda: self.now)
        self.assertEqual(
            self.memory.context("codex:one")["recent_requests"], ["Cover escapes"]
        )
        self.assertEqual(
            self.memory.context("codex:one")["session_context"], "Keep the API stable"
        )
        self.assertEqual(self.memory.context("codex:one")["request_count"], 0)

    def test_failed_migration_rolls_back_schema_changes_and_preserves_history(self):
        self.event("Parser inspection completed")
        self.memory.record_speech("Parser work", "routine")
        self.memory.save_context(
            "codex:one", "Improve parser", "Keep the API stable", ["Cover escapes"]
        )
        work_id = self.memory.work.session_intent("codex:one", "Improve parser")
        self.memory.close()
        with sqlite3.connect(self.path) as database:
            database.execute("DROP TABLE attention_history")
            database.execute("DROP TABLE message_outcomes")
            database.execute("ALTER TABLE contexts DROP COLUMN request_count")
            database.execute("ALTER TABLE contexts DROP COLUMN first_summary_attempted")
            database.execute("DELETE FROM work_memberships")
            database.execute("""CREATE TRIGGER reject_migrated_membership BEFORE INSERT ON work_memberships BEGIN
                SELECT RAISE(ABORT, 'migration interrupted'); END""")
            database.execute("PRAGMA user_version = 6")
        with self.assertRaisesRegex(sqlite3.IntegrityError, "migration interrupted"):
            Memory(self.path, "fixture", clock=lambda: self.now)
        with sqlite3.connect(self.path) as database:
            self.assertEqual(database.execute("PRAGMA user_version").fetchone()[0], 6)
            self.assertEqual(
                database.execute("SELECT text FROM speech").fetchone()[0], "Parser work"
            )
            self.assertEqual(
                database.execute("SELECT id FROM work_items").fetchone()[0], work_id
            )
            self.assertIsNone(
                database.execute(
                    "SELECT name FROM sqlite_master WHERE name='attention_history'"
                ).fetchone()
            )
            self.assertIsNone(
                database.execute(
                    "SELECT name FROM sqlite_master WHERE name='message_outcomes'"
                ).fetchone()
            )
            self.assertNotIn(
                "request_count",
                {row[1] for row in database.execute("PRAGMA table_info(contexts)")},
            )
            database.execute("DROP TRIGGER reject_migrated_membership")
        self.memory = Memory(self.path, "fixture", clock=lambda: self.now)
        self.assertEqual(self.memory.work.work_for_session("codex:one").id, work_id)
        self.assertEqual(
            self.memory.context("codex:one")["recent_requests"], ["Cover escapes"]
        )
        self.assertIn("Parser inspection completed", json.dumps(self.memory.recent()))
