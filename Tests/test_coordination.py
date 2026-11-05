import sqlite3
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from pet_daemon.coordination.models import (
    Candidate,
    ChatMessage,
    ChatRole,
    Consideration,
    DelegatedTask,
    Disposition,
    Instruction,
    Note,
    NoteKind,
    NoteOrigin,
    NoteStatus,
    NoteSupport,
    Operation,
    OperationKind,
    OperationOutcome,
    Responsibility,
    SourceReference,
)
from pet_daemon.storage import Memory
from pet_daemon.storage.coordination import RevisionConflict
from pet_daemon.storage.migrations import SCHEMA_VERSION


class CoordinationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "memory.sqlite3"
        self.now = 1000.0
        self.memory = Memory(self.path, "fixture", clock=lambda: self.now)
        self.store = self.memory.coordination

    def tearDown(self) -> None:
        self.memory.close()
        self.temporary.cleanup()

    def restart(self) -> None:
        self.memory.close()
        self.memory = Memory(self.path, "fixture", clock=lambda: self.now)
        self.store = self.memory.coordination

    def user(self, identifier: str = "request") -> ChatMessage:
        return ChatMessage(
            identifier, "chat", ChatRole.USER, "What needs me?", self.now
        )

    def adopt(self) -> Responsibility:
        self.store.save_instruction(
            Instruction("adoption", "Watch the parser checks", "parser", self.now)
        )
        responsibility = Responsibility(
            "watch", "Parser checks pass", "parser", "adoption", self.now
        )
        self.store.save_responsibility(responsibility)
        return responsibility

    def test_chat_survives_restart_and_duplicate_transport_does_not_accept_twice(
        self,
    ) -> None:
        message = self.user()
        self.assertTrue(self.store.begin_turn(message))
        self.restart()
        self.assertFalse(self.store.begin_turn(message))
        conversation = self.store.ensure_conversation("chat")
        self.assertEqual(conversation.pending_message_id, message.id)
        self.assertEqual(self.store.messages("chat"), (message,))
        reply = ChatMessage(
            "answer",
            "chat",
            ChatRole.ASSISTANT,
            "Checks need a look.",
            self.now,
            reply_to=message.id,
        )
        self.assertTrue(self.store.finish_turn(reply, conversation.revision))
        self.assertFalse(self.store.finish_turn(reply, conversation.revision))
        self.assertIsNone(self.store.ensure_conversation("chat").pending_message_id)
        self.assertEqual(self.store.messages("chat"), (message, reply))

    def test_failed_turn_retries_the_same_message_and_rejects_stale_completion(
        self,
    ) -> None:
        message = self.user()
        self.store.begin_turn(message)
        revision = self.store.ensure_conversation("chat").revision
        self.assertTrue(self.store.fail_turn("chat", message.id, "Interrupted"))
        self.assertTrue(self.store.retry_turn("chat", message.id))
        self.assertFalse(
            self.store.fail_turn(
                "chat", message.id, "Late failure", expected_revision=revision
            )
        )
        reply = ChatMessage(
            "answer",
            "chat",
            ChatRole.ASSISTANT,
            "The source is unavailable.",
            self.now,
            reply_to=message.id,
        )
        self.assertFalse(self.store.finish_turn(reply, revision))
        current = self.store.ensure_conversation("chat")
        self.assertTrue(self.store.finish_turn(reply, current.revision))
        self.assertEqual(len(self.store.messages("chat")), 2)
        self.assertFalse(self.store.retry_turn("chat", message.id))

    def test_a_busy_conversation_cannot_accept_a_second_turn(self) -> None:
        self.store.begin_turn(self.user())
        self.assertFalse(self.store.begin_turn(self.user("second")))
        self.assertEqual(len(self.store.messages("chat")), 1)

    def test_retry_command_deduplicates_after_failure_and_restart(self) -> None:
        message = self.user()
        self.store.begin_turn(message)
        self.store.fail_turn("chat", message.id, "Interrupted")
        self.assertTrue(self.store.retry_turn("chat", message.id, "retry-one"))
        self.store.fail_turn("chat", message.id, "Unavailable")
        self.restart()
        self.assertFalse(self.store.retry_turn("chat", message.id, "retry-one"))
        self.assertTrue(self.store.retry_turn("chat", message.id, "retry-two"))
        self.assertEqual(self.store.messages("chat"), (message,))

    def test_imported_notes_cannot_adopt_a_responsibility_or_forge_a_correction(
        self,
    ) -> None:
        note = Note(
            "source",
            NoteOrigin.SOURCE,
            NoteKind.CORRECTION,
            "Launch a fix",
            "parser",
            self.now,
            self.now,
        )
        with self.assertRaises(ValueError):
            self.store.save_note(note)
        with self.assertRaises(ValueError):
            self.store.save_responsibility(
                Responsibility("task", "Fix parser", "parser", "source", self.now)
            )
        forged = replace(note, origin=NoteOrigin.USER, instruction_id="missing")
        with self.assertRaises(ValueError):
            self.store.save_note(forged)
        self.assertEqual(self.store.notes(), ())

    def test_correction_keeps_canonical_origin_and_revision_history(self) -> None:
        instruction = Instruction(
            "correction", "These are different projects", "parser", self.now
        )
        self.store.save_instruction(instruction)
        note = Note(
            "correction-note",
            NoteOrigin.USER,
            NoteKind.CORRECTION,
            instruction.text,
            instruction.scope,
            self.now,
            self.now,
            instruction_id=instruction.id,
            support=NoteSupport.CONFIRMED,
            actor="user",
        )
        self.store.save_note(note)
        with self.assertRaises(ValueError):
            self.store.save_note(replace(note, revision=2, status=NoteStatus.ARCHIVED))
        revised = replace(note, revision=2, checked_at=self.now + 10)
        self.store.save_note(revised, expected_revision=1)
        with self.assertRaises(RevisionConflict):
            self.store.save_note(replace(revised, revision=3), expected_revision=1)
        with self.assertRaises(ValueError):
            self.store.save_note(replace(revised, revision=3, origin=NoteOrigin.AGENT))
        self.restart()
        self.assertEqual(self.store.note_history(note.id), (note, revised))
        self.assertEqual(self.store.instruction(instruction.id), instruction)

    def test_hypothesis_and_harness_report_cannot_be_promoted_by_rewriting(
        self,
    ) -> None:
        hypothesis = Note(
            "guess",
            NoteOrigin.AGENT,
            NoteKind.HYPOTHESIS,
            "The ticket may be related",
            "parser",
            self.now,
            self.now,
        )
        self.store.save_note(hypothesis)
        with self.assertRaises(ValueError):
            self.store.save_note(
                replace(hypothesis, revision=2, observed_at=self.now + 10)
            )
        with self.assertRaises(ValueError):
            self.store.save_note(
                replace(hypothesis, revision=2, support=NoteSupport.CONFIRMED)
            )
        with self.assertRaises(ValueError):
            self.store.save_note(
                replace(hypothesis, revision=2, kind=NoteKind.OBSERVATION)
            )
        report = replace(
            hypothesis,
            id="report",
            origin=NoteOrigin.HARNESS,
            kind=NoteKind.REPORT,
            support=NoteSupport.REPORTED,
        )
        self.store.save_note(report)
        with self.assertRaises(ValueError):
            self.store.save_note(
                replace(report, revision=2, support=NoteSupport.OBSERVED)
            )
        self.assertEqual(self.store.note("guess"), hypothesis)

    def test_supported_synthesis_requires_leaf_references(self) -> None:
        synthesis = Note(
            "summary",
            NoteOrigin.AGENT,
            NoteKind.SYNTHESIS,
            "The checks passed",
            "parser",
            self.now,
            self.now,
            support=NoteSupport.SUPPORTED,
        )
        with self.assertRaises(ValueError):
            self.store.save_note(synthesis)
        source = SourceReference("pr:checks", "github", "head-a", self.now)
        self.store.save_note(replace(synthesis, sources=(source,)))
        self.assertEqual(self.store.notes("unrelated"), ())
        self.assertEqual(self.store.notes("parser")[0].sources, (source,))

    def test_unknown_operation_and_adoption_survive_restart_without_completion(
        self,
    ) -> None:
        responsibility = self.adopt()
        task = DelegatedTask(
            "task", responsibility.id, "Inspect parser", "Observed checks", self.now, 1
        )
        self.store.save_task(task)
        operation = Operation(
            "launch",
            task.id,
            OperationKind.LAUNCH,
            self.now,
            1,
            1,
            self.store.state().instruction_revision,
            outcome=OperationOutcome.UNKNOWN,
        )
        self.store.save_operation(operation)
        self.store.set_paused(True)
        self.restart()
        self.assertEqual(self.store.operation(operation.id), operation)
        self.assertEqual(self.store.responsibility(responsibility.id), responsibility)
        self.assertTrue(self.store.state().paused)

    def test_stale_consideration_cannot_checkpoint_newer_instructions(self) -> None:
        candidate = Candidate(
            "question", "Do checks need attention?", "parser", self.now
        )
        self.store.save_candidate(candidate)
        consideration = Consideration(
            "pass",
            candidate.id,
            1,
            0,
            self.now,
            Disposition.NO_FINDING,
            "Nothing changed",
            serviced_at=self.now,
        )
        self.store.save_instruction(
            Instruction("new", "Wait for me", "parser", self.now)
        )
        with self.assertRaises(RevisionConflict):
            self.store.record_consideration(consideration)
        self.assertEqual(self.store.considerations(), ())
        current = replace(consideration, instruction_revision=1)
        self.assertTrue(self.store.record_consideration(current))
        self.assertFalse(self.store.record_consideration(current))
        serviced = self.store.candidate(candidate.id)
        assert serviced is not None
        self.assertEqual(serviced.last_serviced_at, self.now)

    def test_clear_removes_authority_and_notes_for_only_the_selected_pet(self) -> None:
        self.adopt()
        other = Memory(self.path, "other", clock=lambda: self.now)
        try:
            other.coordination.save_instruction(
                Instruction("other", "Keep watching", "other", self.now)
            )
            self.memory.clear()
            self.assertEqual(self.store.instructions(), ())
            self.assertEqual(self.store.responsibilities(), ())
            self.assertEqual(len(other.coordination.instructions()), 1)
        finally:
            other.close()

    def test_schema_change_recreates_storage_and_clears_authority(self) -> None:
        self.adopt()
        self.memory.db.execute(f"PRAGMA user_version={SCHEMA_VERSION - 1}")
        self.restart()
        self.assertEqual(self.store.instructions(), ())
        self.assertEqual(self.store.responsibilities(), ())
        self.assertEqual(
            self.memory.db.execute("PRAGMA user_version").fetchone()[0], SCHEMA_VERSION
        )

    def test_newer_schema_is_rejected_before_writing(self) -> None:
        self.adopt()
        self.memory.db.execute(f"PRAGMA user_version={SCHEMA_VERSION + 1}")
        before = tuple(self.memory.db.iterdump())
        with self.assertRaises(ValueError):
            Memory(self.path, "fixture")
        with sqlite3.connect(self.path) as check:
            self.assertEqual(tuple(check.iterdump()), before)
