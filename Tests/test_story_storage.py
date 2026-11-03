import re
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pet_daemon.domain import SessionSource, SpeechReason
from pet_daemon.identity import SessionIdentity
from pet_daemon.storage import Memory
from pet_daemon.storage.memory import AttentionHistory
from pet_daemon.storage.migrations import SCHEMA_VERSION
from pet_daemon.work.models import IntentEvidence, ProgressEvidence
from pet_daemon.work.narrative import NarrativeStatement, WorkNarrative
from pet_daemon.work.observations import SessionObservation


class StoryStorageTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "memory.sqlite3"
        self.memory = Memory(self.path, "pet", clock=lambda: 1000)
        self.addCleanup(lambda: self.memory.close())

    def test_recorded_observation_times_survive_reopen_and_inherited_requests(
        self,
    ) -> None:
        now = 1000.0
        self.memory.close()
        self.memory = Memory(self.path, "pet", clock=lambda: now)
        root = SessionIdentity(source=SessionSource.CODEX, native_id="root")
        child = SessionIdentity(source=SessionSource.CODEX, native_id="child")
        request = SessionObservation(
            session=root,
            reports=(IntentEvidence(session=root, text="Inspect parser"),),
        )
        self.memory.work.record_session(request)
        self.memory.observe_session_context(request)
        now = 1060.0
        progress = SessionObservation(
            session=root,
            capture_at=700,
            reports=(ProgressEvidence(session=root, text="Parser checks passed"),),
        )
        self.memory.work.record_session(progress)
        self.memory.observe_session_context(progress)
        inherited = SessionObservation(session=child, parent=root, capture_at=1060)
        self.memory.work.record_session(inherited)
        self.memory.observe_session_context(inherited)
        self.memory.close()
        now = 2000.0
        self.memory = Memory(self.path, "pet", clock=lambda: now)

        reports = self.memory.narrative_reports((root,))
        retained_request = next(
            report for report in reports if isinstance(report.payload, IntentEvidence)
        )
        retained_progress = next(
            report for report in reports if isinstance(report.payload, ProgressEvidence)
        )
        self.assertEqual(retained_request.record().observed_at, 1000)
        self.assertEqual(retained_request.ingested_at, 1000)
        self.assertIsNone(retained_request.capture_at)
        self.assertIsNone(retained_request.evidence_at)
        self.assertEqual(retained_progress.record().observed_at, 700)
        self.assertEqual(retained_progress.capture_at, 700)
        self.assertEqual(retained_progress.ingested_at, 1060)
        context = self.memory.context(child)
        assert context is not None
        self.assertEqual(context.initial_request_at, 1000)

    def test_retained_text_supports_new_cleaners_and_inherited_context(self) -> None:
        root = SessionIdentity(source=SessionSource.CODEX, native_id="root")
        child = SessionIdentity(source=SessionSource.CODEX, native_id="child")
        original = "Unrecognized report:\nFix parser.\n    keep_code()"
        observation = SessionObservation(
            session=root,
            capture_at=700,
            reports=(IntentEvidence(session=root, text=original),),
        )
        self.memory.work.record_session(observation)
        self.memory.observe_session_context(observation)
        inherited = SessionObservation(session=child, parent=root, capture_at=800)
        self.memory.work.record_session(inherited)
        self.memory.observe_session_context(inherited)
        self.memory.close()
        self.memory = Memory(self.path, "pet", clock=lambda: 2000)
        report = self.memory.narrative_reports((root,))[0]
        assert isinstance(report.payload, IntentEvidence)
        self.assertEqual(report.payload.text, original)
        self.assertEqual(report.evidence_at, 700)
        before = self.memory.context(child)
        assert before is not None
        self.assertIn("Unrecognized report:", before.initial_request)
        cleaner = re.compile(r"Unrecognized report:\n(?P<body>.*)", re.DOTALL)
        with patch("pet_daemon.harness_text._CODEX_MESSAGE", cleaner):
            after = self.memory.context(child)
            assert after is not None
            self.assertTrue(after.initial_request.startswith("Fix parser."))
            self.assertEqual(after.initial_report, before.initial_report)
            retained = self.memory.narrative_reports((root,))[0].payload
            assert isinstance(retained, IntentEvidence)
            self.assertEqual(retained.text, original)

    def test_alias_merge_preserves_inherited_report_identity_and_capture_time(
        self,
    ) -> None:
        alias = SessionIdentity(source=SessionSource.CODEX, native_id="alias")
        canonical = SessionIdentity(source=SessionSource.CODEX, native_id="canonical")
        child = SessionIdentity(source=SessionSource.CODEX, native_id="child")
        observation = SessionObservation(
            session=alias,
            capture_at=500,
            reports=(IntentEvidence(session=alias, text="Inspect parser"),),
        )
        self.memory.work.record_session(observation)
        self.memory.observe_session_context(observation)
        original = self.memory.context(alias)
        assert original is not None
        merged = SessionObservation(session=canonical, aliases=(alias,), capture_at=800)
        self.memory.work.record_session(merged)
        self.memory.observe_session_context(merged)
        inherited = SessionObservation(session=child, parent=canonical, capture_at=900)
        self.memory.work.record_session(inherited)
        self.memory.observe_session_context(inherited)
        context = self.memory.context(child)
        assert context is not None
        self.assertEqual(context.initial_report, original.initial_report)
        self.assertEqual(context.initial_request_at, 500)
        self.assertEqual(context.initial_request, "Inspect parser")

    def narrative(self) -> WorkNarrative:
        return WorkNarrative(
            subject_id=SessionIdentity(source=SessionSource.CODEX, native_id="root"),
            revision="revision",
            intent=NarrativeStatement(text="Inspect parser", sources=("request",)),
            progress=(
                NarrativeStatement(
                    text="Reported parser checks passed", sources=("report",)
                ),
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
            self.memory.load_narrative(
                SessionIdentity(source=SessionSource.CODEX, native_id="root")
            ),
            value,
        )

    def test_schema_change_resets_speech_reports_attention_and_stories(self) -> None:
        self.memory.work.record_session(
            SessionObservation(
                session=SessionIdentity(source=SessionSource.CODEX, native_id="root"),
                reports=(
                    IntentEvidence(
                        session=SessionIdentity(
                            source=SessionSource.CODEX, native_id="root"
                        ),
                        text="Inspect parser",
                    ),
                ),
            )
        )
        self.memory.record_speech("Inspecting", SpeechReason.ROUTINE)
        self.memory.save_attention_history(
            SessionIdentity(source=SessionSource.CODEX, native_id="root"),
            AttentionHistory(
                last_selected=100, considered_subject_revision="acknowledged"
            ),
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
                (SessionIdentity(source=SessionSource.CODEX, native_id="root"),)
            ),
            (),
        )
        self.assertEqual(self.memory.load_attention_history(), {})
        self.assertIsNone(
            self.memory.load_narrative(
                SessionIdentity(source=SessionSource.CODEX, native_id="root")
            )
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
            self.memory.load_narrative(
                SessionIdentity(source=SessionSource.CODEX, native_id="root")
            )
        )
        self.assertEqual(
            other.load_narrative(
                SessionIdentity(source=SessionSource.CODEX, native_id="root")
            ),
            self.narrative(),
        )
        self.memory.save_narrative(self.narrative())
        self.memory.clock = lambda: 1000 + 31 * 86400
        self.memory.prune()
        self.assertIsNone(
            self.memory.load_narrative(
                SessionIdentity(source=SessionSource.CODEX, native_id="root")
            )
        )

    def test_native_ids_remain_scoped_to_their_source_after_reload(self) -> None:
        for index, source in enumerate((SessionSource.CODEX, SessionSource.CLAUDE)):
            parent = SessionIdentity(source=source, native_id="parent")
            temporary = SessionIdentity(source=source, native_id="temporary")
            identity = SessionIdentity(source=source, native_id="shared:session")
            parent_observation = SessionObservation(
                session=parent,
                reports=(IntentEvidence(session=parent, text=f"Parent {source}"),),
            )
            child_observation = SessionObservation(
                session=temporary,
                parent=parent,
                reports=(IntentEvidence(session=temporary, text=f"Child {source}"),),
            )
            descendant = SessionObservation(
                session=SessionIdentity(source=source, native_id="descendant"),
                parent=temporary,
            )
            for observation in (parent_observation, child_observation, descendant):
                with self.memory.transaction():
                    self.memory.work.record_session(observation)
                    self.memory.observe_session_context(observation)
            renamed = SessionObservation(session=identity, aliases=(temporary,))
            with self.memory.transaction():
                self.memory.work.record_session(renamed)
                self.memory.observe_session_context(renamed)
            self.memory.save_attention_history(
                identity, AttentionHistory(last_selected=index + 1)
            )
            self.memory.save_narrative(
                self.narrative().model_copy(update={"subject_id": identity})
            )
            self.memory.record_acceptance(
                source.value, f"Report {source}", SpeechReason.ROUTINE, identity
            )
        self.memory.close()
        self.memory = Memory(self.path, "pet", clock=lambda: 1000)
        facts = self.memory._work_repository.load()
        for index, source in enumerate((SessionSource.CODEX, SessionSource.CLAUDE)):
            parent = SessionIdentity(source=source, native_id="parent")
            temporary = SessionIdentity(source=source, native_id="temporary")
            identity = SessionIdentity(source=source, native_id="shared:session")
            self.assertEqual(facts.aliases[temporary], identity)
            self.assertEqual(facts.sessions[identity].parent, parent)
            self.assertEqual(
                facts.sessions[
                    SessionIdentity(source=source, native_id="descendant")
                ].parent,
                identity,
            )
            reports = self.memory.narrative_reports((identity,))
            self.assertEqual(
                tuple(report.payload for report in reports),
                (IntentEvidence(session=temporary, text=f"Child {source}"),),
            )
            context = self.memory.context(identity)
            self.assertIsNotNone(context)
            assert context is not None
            self.assertEqual(context.initial_request, f"Child {source}")
            self.assertEqual(
                self.memory.load_attention_history()[identity],
                AttentionHistory(last_selected=index + 1),
            )
            narrative = self.memory.load_narrative(identity)
            self.assertIsNotNone(narrative)
            assert narrative is not None
            self.assertEqual(narrative.subject_id, identity)
            outcome = self.memory.message_outcome(source.value)
            self.assertIsNotNone(outcome)
            assert outcome is not None
            self.assertEqual(outcome.subject_id, identity)
