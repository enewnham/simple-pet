import json
import sqlite3
from pathlib import Path
import tempfile
import unittest

from pet_daemon.memory import Memory


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
        self.memory.record_event({"harness": "codex", "session": "one", "event": "progress", "repo": "/repo/parser", "progress": progress, **extra})

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
        self.assertEqual(self.memory.db.execute("SELECT COUNT(*) FROM speech").fetchone()[0], 1)
        self.memory.prune()
        self.assertEqual(self.memory.db.execute("SELECT COUNT(*) FROM speech").fetchone()[0], 0)

    def test_speech_expiry_survives_restart_and_uses_its_own_retention(self):
        self.memory.record_speech("Expired remark", "routine")
        self.now += 1799
        self.memory.record_speech("Recent remark", "routine")
        self.memory.close()
        self.now += 1
        self.memory = Memory(self.path, "fixture", clock=lambda: self.now, retention_days=0)
        self.assertEqual([item["text"] for item in self.memory.speech()], ["Recent remark"])
        self.now += 1
        self.memory.prune()
        self.assertEqual([item["text"] for item in self.memory.speech()], ["Recent remark"])
        self.memory.close()
        self.memory = Memory(self.path, "fixture", clock=lambda: self.now, speech_history_seconds=1)
        self.assertEqual(self.memory.speech(), [])

    def test_short_speech_retention_preserves_another_pets_valid_history(self):
        self.memory.record_speech("Long history remark", "routine")
        other = Memory(self.path, "short-history-pet", clock=lambda: self.now, speech_history_seconds=60)
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
        self.memory.save_context("codex:one", "Improve parser", "Parser summary", ["Improve parser"])
        self.memory.associate_repository("codex:one", "/repo/parser")
        self.memory.save_repository("/repo/parser:feature", {"pending": True})
        other = Memory(self.path, "other-pet", clock=lambda: self.now)
        try:
            other.record_speech("Other pet remark", "routine")
            self.memory.clear_speech()
            self.assertEqual(self.memory.speech(), [])
            self.assertEqual(other.speech()[0]["text"], "Other pet remark")
            self.assertEqual(self.memory.context("codex:one")["session_context"], "Parser summary")
            self.assertIn("Parser checks passed", json.dumps(self.memory.recent()))
            self.assertEqual(self.memory.projects(["/repo/parser"])[0]["goal"], "Improve parser")
            self.assertEqual(self.memory.repository_associations(), [{"session": "codex:one", "worktree": "/repo/parser"}])
            self.assertEqual(self.memory.load_repositories(), [{"pending": True}])
            self.memory.close()
            self.memory = Memory(self.path, "fixture", clock=lambda: self.now)
            self.assertEqual(self.memory.speech(), [])
            self.assertTrue(self.memory.recent())
        finally:
            other.close()

    def test_clear_erases_saved_knowledge_for_the_current_pet(self):
        other = Memory(self.path, "other-pet", clock=lambda: self.now)
        event = {"harness": "codex", "session": "one", "event": "intent",
                 "repo": "/repo/parser", "goal": "Improve parser"}
        try:
            for memory in (self.memory, other):
                memory.record_event(event)
                memory.record_speech("Parser work", "prompt")
                memory.save_context("codex:one", "Improve parser", "Parser work", ["Improve parser"])
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
            self.memory.record_event(event)
            self.assertEqual(len(self.memory.recent()), 1)
        finally:
            other.close()

    def test_event_count_discards_oldest_facts(self):
        for progress in ("First result", "Second result", "Third result", "Fourth result"):
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

    def test_recall_matches_older_goals_within_the_repository(self):
        self.memory.close()
        self.memory = Memory(self.path, "fixture", clock=lambda: self.now, max_events=100)
        self.event("", event="intent", goal="Refactor parser tokenization")
        self.event("", event="intent", repo="/repo/other", goal="Refactor parser tokenization")
        self.event("", event="intent", goal="Improve billing invoices")
        for index in range(6):
            self.event(f"Recent progress {index}")
        recalled = self.memory.recall({"event": "intent", "repo": "/repo/parser", "goal": "Refactor parser tokenization", "harness": "codex", "session": "one"})
        self.assertEqual(len(recalled), 1)
        self.assertIn("parser tokenization", recalled[0]["goal"])
        self.assertEqual(recalled[0]["repo"], "/repo/parser")
        self.assertNotIn(recalled[0]["id"], [event["id"] for event in self.memory.recent()])

    def test_recall_requires_an_explicit_goal_at_task_start(self):
        self.memory.close()
        self.memory = Memory(self.path, "fixture", clock=lambda: self.now, max_events=100)
        self.event("", event="intent", goal="Refactor parser tokenization")
        for index in range(6):
            self.event(f"Recent progress {index}")
        for event in ({"event": "tool_error", "goal": "Refactor parser tokenization"},
                      {"event": "progress", "progress": "Refactor parser tokenization"},
                      {"event": "intent", "detail": "Refactor parser tokenization"},
                      {"event": "intent", "goal": ""}):
            with self.subTest(event=event):
                self.assertEqual(self.memory.recall({"repo": "/repo/parser", **event}), [])

    def test_failure_facts_exclude_injected_diagnostics(self):
        self.event("SECRET FAILED COMMAND OUTPUT", event="tool_error", detail="SECRET STACK TRACE", error_signature="SECRET HASH", tool="pytest")
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
        self.memory = Memory(self.path, "fixture", clock=lambda: self.now, max_events=100)
        self.event("Quiet session remains blocked", session="quiet")
        for index in range(8):
            self.event(f"Busy session progress {index}")
        events = self.memory.recent(limit=5)
        self.assertIn("codex:quiet", [event["session"] for event in events])
        self.assertIn("Busy session progress 7", json.dumps(events))

    def test_terminal_fact_preserves_reconciled_failure(self):
        self.memory.record_event({"harness": "claude", "session": "one", "event": "turn_done", "outcome": "success"},
                                 {"id": "claude:one", "outcome": "error"})
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
        for tool, category, expected in (("pytest", "testing", "Tests failed"),
                                         ("cargo build", "building", "Build failed"),
                                         ("./release.sh", "command", "Command failed"),
                                         ("", "harness", "Agent run failed")):
            with self.subTest(activity=category):
                self.memory.record_event({"harness": "claude", "session": "one", "event": "tool_error", "tool": tool},
                                         {"id": "claude:one", "activity": category})
                self.assertIn(expected, self.memory.recent()[-1]["summary"])
        self.memory.record_event({"harness": "claude", "session": "one", "event": "interrupt", "outcome": "cancelled"},
                                 {"id": "claude:one", "activity": "harness", "outcome": "cancelled"})
        self.assertIn("Agent run cancelled", self.memory.recent()[-1]["summary"])

    def test_schema_recreation_discards_diagnostic_memory(self):
        self.memory.close()
        legacy_path = self.path.parent / "legacy.sqlite3"
        with sqlite3.connect(legacy_path) as database:
            database.execute("CREATE TABLE events (id INTEGER PRIMARY KEY, detail TEXT)")
            database.execute("INSERT INTO events(detail) VALUES ('SECRET legacy diagnostic')")
        self.memory = Memory(legacy_path, "fixture", clock=lambda: self.now)
        self.assertEqual(self.memory.recent(), [])
        self.event("Current parser result")
        self.assertIn("Current parser result", json.dumps(self.memory.recent()))
        with sqlite3.connect(legacy_path) as database:
            columns = [row[1] for row in database.execute("PRAGMA table_info(events)")]
        self.assertIn("goal", columns)
        self.assertNotIn("detail", columns)
        self.assertNotIn("error_signature", columns)

    def test_schema_version_mismatch_recreates_memory(self):
        self.event("Parser inspection completed")
        self.memory.save_context("codex:one", "Refactor parser", "Keep its API stable", ["Refactor parser"])
        self.memory.record_speech("Parser work", "prompt")
        self.memory.close()
        with sqlite3.connect(self.path) as database:
            database.execute("ALTER TABLE contexts DROP COLUMN request_count")
            database.execute("ALTER TABLE contexts DROP COLUMN first_summary_attempted")
            database.execute("PRAGMA user_version = 3")
        self.memory = Memory(self.path, "fixture", clock=lambda: self.now)
        self.assertIsNone(self.memory.context("codex:one"))
        self.assertEqual(self.memory.recent(), [])
        self.assertEqual(self.memory.speech(), [])
        self.assertEqual(self.memory.projects(["/repo/parser"]), [])
        self.memory.save_context("codex:one", "Review hooks", "Hook review", ["Review hooks"], 1, True)
        self.assertTrue(self.memory.context("codex:one")["first_summary_attempted"])

    def test_session_context_survives_restart_with_its_original_anchor(self):
        self.memory.save_context("codex:one", "Refactor parser", "Parser refactor awaits validation", ["Refactor parser", "Check edge cases"])
        self.memory.close()
        self.memory = Memory(self.path, "fixture", clock=lambda: self.now)
        context = self.memory.context("codex:one")
        self.assertEqual(context["first_prompt"], "Refactor parser")
        self.assertEqual(context["session_context"], "Parser refactor awaits validation")
        self.assertEqual(context["recent_requests"], ["Refactor parser", "Check edge cases"])
        self.assertEqual(self.memory.recent(), [])
        self.assertIsNone(self.memory.context("codex:other"))

    def test_pet_identity_keeps_session_context_separate(self):
        self.memory.save_context("codex:one", "Refactor parser", "Parser validation pending", ["Refactor parser"])
        other = Memory(self.path, "other-pet", clock=lambda: self.now)
        try:
            self.assertIsNone(other.context("codex:one"))
        finally:
            other.close()
