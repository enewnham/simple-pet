import tempfile
import unittest
from pathlib import Path

from pet_daemon.domain import ExecutionOutcome, SessionSource
from pet_daemon.identity import SessionIdentity
from pet_daemon.mcp import WorkContext
from pet_daemon.storage import Memory
from pet_daemon.work.models import (
    ExecutionEvidence,
    ProgressEvidence,
    ReportContext,
    SessionReport,
)


class McpReportDedupeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "memory.sqlite3"
        self.memory = Memory(self.path, "fixture")
        self.context = WorkContext(self.path, "fixture")

    def tearDown(self) -> None:
        self.memory.close()
        self.temporary.cleanup()

    def turn(self, text: str, timestamp: float, session: str = "parser") -> None:
        identity = SessionIdentity(source=SessionSource.CODEX, native_id=session)
        payloads = (
            ExecutionEvidence(
                session=identity,
                event="turn_start",
                outcome=None,
                tool_name=None,
                progress=None,
            ),
            ExecutionEvidence(
                session=identity,
                event="turn_done",
                outcome=ExecutionOutcome.SUCCESS,
                tool_name=None,
                progress=text,
            ),
            ProgressEvidence(session=identity, text=text),
        )
        with self.memory.transaction():
            for payload in payloads:
                self.memory._work_repository.append_report(
                    SessionReport(
                        payload=payload,
                        context=ReportContext(checkout="/parser"),
                        observed_at=timestamp,
                        capture_at=timestamp,
                        ingested_at=timestamp + 1,
                        hook_kind="turn_done",
                    )
                )

    def test_one_turn_uses_one_slot_before_the_result_limit(self) -> None:
        self.turn("Check parser compatibility", 100)
        self.turn("Parser compatibility verified", 200)
        sources = self.context.select(2)["sources"]
        self.assertEqual(
            [source["text"] for source in sources],
            ["Parser compatibility verified", "Check parser compatibility"],
        )
        self.assertFalse(self.context.select(2)["truncated"])
        self.assertTrue(
            all(source["provenance"]["kind"] == "execution" for source in sources)
        )
        self.assertEqual(
            self.context.read(sources[0]["id"], 0, 100)["text"],
            "Parser compatibility verified",
        )

    def test_search_and_changes_collapse_turn_reports(self) -> None:
        self.turn("Parser compatibility verified", 100)
        self.turn("Parser compatibility verified", 200)
        matches = self.context.select(2, query="compatibility")
        self.assertEqual(len(matches["sources"]), 2)
        self.assertFalse(matches["truncated"])
        changed = self.context.select(2, after=150)
        self.assertEqual(len(changed["sources"]), 1)
        self.assertEqual(changed["sources"][0]["provenance"]["observed_at"], 200)

    def test_identical_text_from_other_turns_and_sessions_keeps_its_slots(self) -> None:
        self.turn("Parser verified", 100)
        self.turn("Parser verified", 200)
        self.turn("Parser verified", 200, session="another-session")
        sources = self.context.select(3)["sources"]
        self.assertEqual(len(sources), 3)
        self.assertEqual(
            [source["provenance"]["observed_at"] for source in sources],
            [200, 200, 100],
        )
        self.assertFalse(self.context.select(3)["truncated"])
