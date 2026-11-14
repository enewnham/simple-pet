import math
import sqlite3
import stat
import tempfile
import unittest
from pathlib import Path
from typing import cast

from pydantic import ValidationError

from pet_daemon.workstream_marks import (
    MarkPatch,
    MarkStore,
    is_suppressed,
    mark_rank,
    thread_id,
)


class WorkstreamMarkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "memory.sqlite3"
        self.now = 100.0
        self.store = MarkStore(self.path, "fixture", clock=lambda: self.now)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_thread_ids_preserve_the_captured_group_identity(self) -> None:
        self.assertEqual(
            thread_id("/workspace"),
            "thread:c52ddf65534b7b46035084358ab7902be4bfef220bdb503ac7039cc861905b05",
        )

    def test_reading_unmarked_work_does_not_create_storage(self) -> None:
        self.assertEqual(self.store.effective("thread:one"), {})
        self.assertEqual(self.store.all_effective(), {})
        self.assertEqual(self.store.history("thread:one"), [])
        self.assertFalse(self.store.path.exists())
        self.assertFalse(self.path.exists())

    def test_mark_updates_retain_each_fields_author_and_full_history(self) -> None:
        first = self.store.append(
            "thread:one",
            MarkPatch(
                status="blocked", title="Review parser", tags=("review", "review")
            ),
            author="companion-a",
            reason="Waiting on a dependency",
            principal="shared-client",
        )
        self.now = 200
        second = self.store.append(
            "thread:one",
            MarkPatch(
                title="Ship parser", priority="focus", goal="Finish parser review"
            ),
            author="companion-b",
            reason="Review is ready",
            principal="shared-client",
        )
        restored = MarkStore(self.path, "fixture")
        effective = restored.effective("thread:one")
        self.assertEqual(
            effective["values"],
            {
                "status": "blocked",
                "title": "Ship parser",
                "tags": ["review"],
                "priority": "focus",
                "goal": "Finish parser review",
            },
        )
        fields = cast(dict[str, dict[str, object]], effective["fields"])
        self.assertEqual(fields["status"]["provenance"], first["provenance"])
        self.assertEqual(fields["title"]["provenance"], second["provenance"])
        self.assertEqual(restored.history("thread:one"), [second, first])
        self.assertEqual(stat.S_IMODE(self.store.path.stat().st_mode), 0o600)
        self.assertFalse(self.path.exists())

    def test_unmark_appends_resets_without_deleting_history(self) -> None:
        first = self.store.append(
            "thread:one",
            MarkPatch(status="paused", title="Parser"),
            "caller",
            "Pause work",
        )
        partial = self.store.unmark(
            "thread:one", "reviewer", "Use generated title", fields=["title"]
        )
        self.assertEqual(
            self.store.effective("thread:one")["values"], {"status": "paused"}
        )
        self.assertEqual(partial["operation"], "unmark")
        self.assertEqual(partial["fields"], ["title"])
        reset = self.store.unmark("thread:one", "reviewer", "Resume default treatment")
        self.assertEqual(self.store.effective("thread:one"), {})
        self.assertEqual(self.store.all_effective(), {})
        self.assertEqual(self.store.history("thread:one"), [reset, partial, first])
        with self.store.database(write=True) as db:
            with self.assertRaisesRegex(sqlite3.IntegrityError, "append-only"):
                db.execute("DELETE FROM mark_events")
            with self.assertRaisesRegex(sqlite3.IntegrityError, "append-only"):
                db.execute("UPDATE mark_events SET author='different'")

    def test_histories_and_effective_marks_are_scoped_to_pet_and_thread(self) -> None:
        first = self.store.append(
            "thread:one", MarkPatch(status="active"), "caller", "Start work"
        )
        second = self.store.append(
            "thread:one", MarkPatch(priority="focus"), "caller", "Focus work"
        )
        self.store.append(
            "thread:two", MarkPatch(status="done"), "caller", "Finish work"
        )
        other = MarkStore(self.path, "other")
        other.append(
            "thread:one", MarkPatch(status="abandoned"), "caller", "Drop other work"
        )
        self.assertEqual(self.store.history("thread:one", limit=1), [second])
        self.assertEqual(
            self.store.history("thread:one", before=cast(int, second["id"])), [first]
        )
        self.assertEqual(
            self.store.effective("thread:one")["values"],
            {"status": "active", "priority": "focus"},
        )
        self.assertEqual(
            other.effective("thread:one")["values"], {"status": "abandoned"}
        )
        self.assertEqual(set(self.store.all_effective()), {"thread:one", "thread:two"})

    def test_newer_schema_is_rejected_before_writes(self) -> None:
        self.store.append(
            "thread:one", MarkPatch(status="active"), "caller", "Start work"
        )
        with sqlite3.connect(self.store.path) as db:
            db.execute("PRAGMA user_version=2")
        original = self.store.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "newer"):
            self.store.append(
                "thread:one", MarkPatch(status="done"), "caller", "Finish work"
            )
        self.assertEqual(self.store.path.read_bytes(), original)
        with self.assertRaisesRegex(ValueError, "newer"):
            self.store.history("thread:one")

    def test_patch_requires_valid_values_and_nonblank_attribution(self) -> None:
        for values in (
            {"status": "unknown"},
            {"priority": "urgent"},
            {"goal": "Two\nlines"},
            {"title": "Two\rtitle"},
            {"snooze_until": math.inf},
            {"snooze_until": -1},
        ):
            with self.subTest(values=values), self.assertRaises(ValidationError):
                MarkPatch.model_validate(values)
        with self.assertRaisesRegex(ValueError, "at least one"):
            self.store.append("thread:one", MarkPatch(), "caller", "Nothing to change")
        with self.assertRaises(ValidationError):
            self.store.append(
                "thread:one", MarkPatch(status="active"), " ", "Start work"
            )
        with self.assertRaises(ValidationError):
            self.store.append("thread:one", MarkPatch(status="active"), "caller", " ")
        with self.assertRaisesRegex(ValueError, "recognized"):
            self.store.unmark("thread:one", "caller", "Reset nothing", fields=[])
        self.assertFalse(self.store.path.exists())

    def test_marks_control_visibility_and_ranking_without_losing_values(self) -> None:
        for status in ("done", "abandoned"):
            with self.subTest(status=status):
                self.assertTrue(is_suppressed({"values": {"status": status}}, 100))
        self.assertTrue(is_suppressed({"values": {"snooze_until": 101}}, 100))
        self.assertFalse(is_suppressed({"values": {"snooze_until": 100}}, 100))
        self.assertFalse(is_suppressed({}, 100))
        self.assertLess(mark_rank({"values": {"status": "blocked"}}), mark_rank({}))
        self.assertLess(mark_rank({"values": {"priority": "focus"}}), mark_rank({}))
        self.assertGreater(mark_rank({"values": {"status": "paused"}}), mark_rank({}))
        self.assertGreater(
            mark_rank({"values": {"priority": "background"}}), mark_rank({})
        )
