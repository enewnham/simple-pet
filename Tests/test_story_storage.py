import sqlite3
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from pet_daemon.domain import SessionSource, SpeechReason
from pet_daemon.identity import SessionIdentity
from pet_daemon.storage import Memory
from pet_daemon.storage.memory import HistoryRecord
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
            SessionIdentity(SessionSource.CODEX, "root"),
            "revision",
            NarrativeStatement("Inspect parser", ("request",)),
            progress=(
                NarrativeStatement("Reported parser checks passed", ("report",)),
            ),
            sources={
                "request": {"text": "Inspect parser"},
                "report": {
                    "text": "Checks passed",
                    "observed_at": 880,
                    "origin": {"source": "codex", "native_id": "root"},
                },
            },
        )

    def test_story_survives_restart_with_evidence_and_scope(self) -> None:
        value = self.narrative()
        self.memory.save_narrative(value)
        self.memory.close()
        self.memory = Memory(self.path, "pet", clock=lambda: 1000)
        self.assertEqual(
            self.memory.load_narrative(SessionIdentity(SessionSource.CODEX, "root")),
            value,
        )

    def test_schema_change_resets_speech_reports_attention_and_stories(self) -> None:
        self.memory.work.record_session(
            SessionObservation(
                SessionIdentity(SessionSource.CODEX, "root"),
                reports=(
                    IntentEvidence(
                        SessionIdentity(SessionSource.CODEX, "root"), "Inspect parser"
                    ),
                ),
            )
        )
        self.memory.record_speech("Inspecting", SpeechReason.ROUTINE)
        self.memory.save_attention_history(
            SessionIdentity(SessionSource.CODEX, "root"),
            HistoryRecord(100, "acknowledged"),
        )
        self.memory.save_narrative(self.narrative())
        self.memory.close()
        with sqlite3.connect(self.path) as database:
            database.execute("DROP TABLE work_narratives")
            database.execute("PRAGMA user_version = 16")
        self.memory = Memory(self.path, "pet", clock=lambda: 1000)
        self.assertEqual(self.memory.speech(), [])
        self.assertEqual(
            self.memory.narrative_reports(
                (SessionIdentity(SessionSource.CODEX, "root"),)
            ),
            (),
        )
        self.assertEqual(self.memory.load_attention_history(), {})
        self.assertIsNone(
            self.memory.load_narrative(SessionIdentity(SessionSource.CODEX, "root"))
        )
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
        self.assertIsNone(
            self.memory.load_narrative(SessionIdentity(SessionSource.CODEX, "root"))
        )
        self.assertEqual(
            other.load_narrative(SessionIdentity(SessionSource.CODEX, "root")),
            self.narrative(),
        )
        self.memory.save_narrative(self.narrative())
        self.memory.clock = lambda: 1000 + 31 * 86400
        self.memory.prune()
        self.assertIsNone(
            self.memory.load_narrative(SessionIdentity(SessionSource.CODEX, "root"))
        )

    def test_native_ids_remain_scoped_to_their_source_after_reload(self) -> None:
        for index, source in enumerate((SessionSource.CODEX, SessionSource.CLAUDE)):
            parent = SessionIdentity(source, "parent")
            temporary = SessionIdentity(source, "temporary")
            identity = SessionIdentity(source, "shared:session")
            parent_observation = SessionObservation(
                parent, reports=(IntentEvidence(parent, f"Parent {source}"),)
            )
            child_observation = SessionObservation(
                temporary,
                parent=parent,
                reports=(IntentEvidence(temporary, f"Child {source}"),),
            )
            descendant = SessionObservation(
                SessionIdentity(source, "descendant"), parent=temporary
            )
            for observation in (parent_observation, child_observation, descendant):
                with self.memory.transaction():
                    self.memory.work.record_session(observation)
                    self.memory.observe_session_context(observation)
            renamed = SessionObservation(identity, aliases=(temporary,))
            with self.memory.transaction():
                self.memory.work.record_session(renamed)
                self.memory.observe_session_context(renamed)
            self.memory.save_attention_history(identity, HistoryRecord(index + 1))
            self.memory.save_narrative(replace(self.narrative(), subject_id=identity))
            self.memory.record_acceptance(
                source.value, f"Report {source}", SpeechReason.ROUTINE, identity
            )
        self.memory.close()
        self.memory = Memory(self.path, "pet", clock=lambda: 1000)
        facts = self.memory._work_repository.load()
        for index, source in enumerate((SessionSource.CODEX, SessionSource.CLAUDE)):
            parent = SessionIdentity(source, "parent")
            temporary = SessionIdentity(source, "temporary")
            identity = SessionIdentity(source, "shared:session")
            self.assertEqual(facts.aliases[temporary], identity)
            self.assertEqual(facts.sessions[identity].parent, parent)
            self.assertEqual(
                facts.sessions[SessionIdentity(source, "descendant")].parent, identity
            )
            reports = self.memory.narrative_reports((identity,))
            self.assertEqual(
                tuple(report.payload for report in reports),
                (IntentEvidence(temporary, f"Child {source}"),),
            )
            context = self.memory.context(identity)
            self.assertIsNotNone(context)
            assert context is not None
            self.assertEqual(context.initial_request, f"Child {source}")
            self.assertEqual(
                self.memory.load_attention_history()[identity], HistoryRecord(index + 1)
            )
            narrative = self.memory.load_narrative(identity)
            self.assertIsNotNone(narrative)
            assert narrative is not None
            self.assertEqual(narrative.subject_id, identity)
            outcome = self.memory.message_outcome(source.value)
            self.assertIsNotNone(outcome)
            assert outcome is not None
            self.assertEqual(outcome.subject_id, identity)
