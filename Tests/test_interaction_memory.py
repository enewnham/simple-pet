import tempfile
import unittest
from pathlib import Path

from pet_daemon.storage import Memory


class InteractionMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "memory.sqlite3"
        self.now = 1000.0
        self.memory = Memory(self.path, "one", clock=lambda: self.now)

    def tearDown(self) -> None:
        self.memory.close()
        self.temporary.cleanup()

    def test_preserves_source_feedback_and_history_across_restart(self) -> None:
        key = "source:report:one"
        self.memory.record_source_consideration((key,), 100)
        self.memory.record_source_mentions((key,), 200)
        self.memory.record_source_feedback((key,), "handled", 300)
        self.memory.record_source_feedback((key,), "engaged", 400)
        self.memory.close()
        self.memory = Memory(self.path, "one", clock=lambda: self.now)
        state = self.memory.load_source_attention()[key]
        self.assertTrue(state.handled)
        self.assertEqual(state.engaged_at, 400)
        self.assertEqual(state.considered_at, 100)
        self.assertEqual(state.mentioned_at, 200)

    def test_scopes_feedback_by_pet_and_source(self) -> None:
        self.memory.record_source_feedback(
            ("report:first", "report:first"), "handled", 1000
        )
        self.memory.record_source_feedback(
            ("report:second",), "tomorrow", 1000, deferred_until=2000
        )
        state = self.memory.load_source_attention()
        self.assertTrue(state["report:first"].handled)
        self.assertFalse(state["report:second"].handled)
        self.assertEqual(state["report:second"].deferred_until, 2000)
        other = Memory(self.path, "two", clock=lambda: self.now)
        try:
            self.assertEqual(other.load_source_attention(), {})
            self.memory.clear()
            self.assertEqual(self.memory.load_source_attention(), {})
        finally:
            other.close()

    def test_handling_a_deferred_source_clears_its_deadline(self) -> None:
        self.memory.record_source_feedback(
            ("report",), "tomorrow", 1000, deferred_until=2000
        )
        self.memory.record_source_feedback(("report",), "handled", 1100)
        state = self.memory.load_source_attention()["report"]
        self.assertTrue(state.handled)
        self.assertIsNone(state.deferred_until)

    def test_prunes_expired_feedback_and_preserves_consideration_history(
        self,
    ) -> None:
        self.memory.record_source_feedback(("old", "recent"), "handled", 1000)
        self.now += 31 * 86400
        self.memory.record_source_consideration(("recent",), self.now)
        self.memory.prune()
        state = self.memory.load_source_attention()
        self.assertFalse(state["old"].handled)
        self.assertFalse(state["recent"].handled)
        self.assertEqual(state["recent"].considered_at, self.now)

    def test_requires_a_deadline_for_tomorrow_feedback(self) -> None:
        with self.assertRaises(ValueError):
            self.memory.record_source_feedback(("report",), "tomorrow", 1000)
