import sqlite3
import tempfile
import unittest
from pathlib import Path

from pet_daemon.domain import SpeechReason
from pet_daemon.storage import Memory
from pet_daemon.storage.migrations import SCHEMA_VERSION
from pet_daemon.work.models import IntentEvidence
from pet_daemon.work.narrative import NarrativeStatement, WorkNarrative
from pet_daemon.work.observations import SessionObservation


class StoryStorageTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "memory.sqlite3"
        self.memory = Memory(self.path, "pet", clock=lambda: 1000)
        self.addCleanup(lambda: self.memory.close())

    def narrative(self) -> WorkNarrative:
        return WorkNarrative(
            "root",
            "revision",
            NarrativeStatement("Inspect parser", ("request",)),
            progress=(
                NarrativeStatement("Reported parser checks passed", ("report",)),
            ),
            sources={
                "request": {"text": "Inspect parser"},
                "report": {"text": "Checks passed"},
            },
        )

    def test_story_survives_restart_with_evidence_and_scope(self) -> None:
        value = self.narrative()
        self.memory.save_narrative(value)
        self.memory.close()
        self.memory = Memory(self.path, "pet", clock=lambda: 1000)
        self.assertEqual(self.memory.load_narrative("root"), value)

    def test_schema_change_resets_speech_reports_attention_and_stories(self) -> None:
        self.memory.work.record_session(
            SessionObservation(
                "root", reports=(IntentEvidence("root", "Inspect parser"),)
            )
        )
        self.memory.record_speech("Inspecting", SpeechReason.ROUTINE)
        self.memory.save_attention_history("root", 100, "acknowledged")
        self.memory.save_narrative(self.narrative())
        self.memory.close()
        with sqlite3.connect(self.path) as database:
            database.execute("DROP TABLE work_narratives")
            database.execute("PRAGMA user_version = 16")
        self.memory = Memory(self.path, "pet", clock=lambda: 1000)
        self.assertEqual(self.memory.speech(), [])
        self.assertEqual(self.memory.narrative_reports(("root",)), ())
        self.assertEqual(self.memory.load_attention_history(), {})
        self.assertIsNone(self.memory.load_narrative("root"))
        self.assertEqual(
            self.memory.db.execute("PRAGMA user_version").fetchone()[0], SCHEMA_VERSION
        )
        self.memory.save_narrative(self.narrative())

    def test_clear_and_retention_remove_only_this_pets_story(self) -> None:
        other = Memory(self.path, "other", clock=lambda: 1000)
        self.addCleanup(other.close)
        self.memory.save_narrative(self.narrative())
        other.save_narrative(self.narrative())
        self.memory.clear()
        self.assertIsNone(self.memory.load_narrative("root"))
        self.assertEqual(other.load_narrative("root"), self.narrative())
        self.memory.save_narrative(self.narrative())
        self.memory.clock = lambda: 1000 + 31 * 86400
        self.memory.prune()
        self.assertIsNone(self.memory.load_narrative("root"))
