import json
import tempfile
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import TypedDict, Unpack

from pet_daemon.commentary.cognition import Cognition
from pet_daemon.commentary.contracts import ModelRequest
from pet_daemon.commentary.summarization import SummaryPayload
from pet_daemon.config import Config
from pet_daemon.domain import ExecutionOutcome
from pet_daemon.prompts import bounded, select_prompt
from pet_daemon.sessions.tracking import Sessions
from pet_daemon.sources.hooks import HookEvent
from pet_daemon.storage import Memory
from pet_daemon.work.history import build_session_entry
from Tests.fixtures import capture_snapshot, require, session_snapshot


class EventOptions(TypedDict, total=False):
    outcome: ExecutionOutcome
    progress: str


class NoRequestsModel:
    async def complete(self, request: ModelRequest, reason: str) -> dict[str, object]:
        raise AssertionError("Summary preparation must not request model output")


class SummarizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = Config(
            pet_dir=Path(self.temp.name), summarize_first_request_only=False
        )
        self.memory = Memory(
            Path(self.temp.name) / "memory.sqlite3", "pet", clock=lambda: 1000
        )
        self.addCleanup(self.memory.close)
        self.context = Cognition(
            NoRequestsModel(), self.config, clock=lambda: 1000, memory=self.memory
        )
        self.focus = session_snapshot(
            id="codex:main",
            initial_request="Plan: [ ] Implement parser. [ ] Verify parser.",
            recent_requests=("Implement parser",),
            summary="Parser work remains pending.",
            progress="Parser implemented and tests passed",
            outcome=ExecutionOutcome.SUCCESS,
            turn_active=False,
        )

    def summary_request(self) -> ModelRequest:
        return self.context.summary_request(self.focus)

    def event(
        self, kind: str, session: str = "main", **values: Unpack[EventOptions]
    ) -> None:
        self.memory.record_event(
            build_session_entry(
                HookEvent(
                    harness="codex", session=session, event=kind, repo="/repo", **values
                )
            )
        )

    def payload(self, request: ModelRequest | None = None) -> SummaryPayload:
        request = request or self.summary_request()
        return json.loads(request["messages"][0]["content"])

    def test_initial_checklist_and_later_completion_keep_their_provenance(self) -> None:
        self.event(
            "turn_done",
            outcome=ExecutionOutcome.SUCCESS,
            progress="Parser implemented and tests passed",
        )
        request = self.summary_request()
        payload = self.payload(request)
        self.assertEqual(payload["current_work"]["request"], "Implement parser")
        self.assertEqual(
            payload["earlier_context"][0]["request"], self.focus.initial_request
        )
        self.assertNotIn("Parser work remains pending", json.dumps(payload))
        self.assertEqual(payload["current_moment"]["outcome"], "success")
        self.assertEqual(len(payload["earlier_context"]), 1)
        self.assertEqual(
            payload["current_moment"]["reported_progress"],
            "Parser implemented and tests passed",
        )
        self.assertNotIn("turn_done", json.dumps(payload))
        self.assertNotIn("reported_at", json.dumps(payload))
        self.assertIn("unchecked checklist", request["system"][0]["text"])
        self.assertEqual(
            request["output_config"]["format"]["schema"]["required"],
            ["summary"],
        )

    def test_next_request_retains_its_session_completion_after_later_tools(
        self,
    ) -> None:
        self.event(
            "turn_done", outcome=ExecutionOutcome.SUCCESS, progress="Parser delivered"
        )
        for index in range(10):
            self.event("progress", progress=f"Follow-up inspection {index}")
        self.event(
            "turn_done",
            session="other",
            outcome=ExecutionOutcome.SUCCESS,
            progress="Unrelated migration delivered",
        )
        self.focus = replace(
            self.focus,
            progress="Inspecting a follow-up",
            outcome=None,
            turn_active=True,
        )
        payload = self.payload()
        self.assertEqual(
            len(
                [
                    item
                    for item in payload.get("earlier_context", [])
                    if "reported_progress" in item
                ]
            ),
            1,
        )
        self.assertEqual(
            next(
                item
                for item in payload.get("earlier_context", [])
                if "reported_progress" in item
            )["reported_progress"],
            "Parser delivered",
        )
        self.assertNotIn("Unrelated migration", json.dumps(payload))
        self.assertEqual(
            payload["current_moment"],
            {"reported_progress": "Inspecting a follow-up"},
        )

    def test_first_request_only_excludes_followups_history_and_results(self) -> None:
        self.config.summarize_first_request_only = True
        self.event(
            "turn_done", outcome=ExecutionOutcome.SUCCESS, progress="Parser delivered"
        )
        payload = self.payload()
        self.assertEqual(
            payload, {"current_work": {"request": self.focus.initial_request}}
        )

    def test_next_turn_summary_preserves_failure_or_cancellation_with_progress(
        self,
    ) -> None:
        for outcome in ("error", "cancelled"):
            with self.subTest(outcome=outcome):
                sessions = Sessions(lambda: 1000)
                progress = "The parser change is ready."
                for event in (
                    HookEvent("codex", outcome, "intent", goal="Implement parser"),
                    HookEvent(
                        "codex",
                        outcome,
                        "turn_done",
                        outcome=ExecutionOutcome(outcome),
                        progress=progress,
                    ),
                    HookEvent("codex", outcome, "intent", goal="Review parser"),
                ):
                    result = require(sessions.apply(event))
                    with self.memory.transaction():
                        self.memory.work.record_session(result.observation)
                        self.memory.observe_session_context(result.observation)
                        self.memory.record_event(build_session_entry(event))
                focus = capture_snapshot(
                    sessions.activity_view(), memory=self.memory
                ).sessions[0]
                payload = self.payload(self.context.summary_request(focus))
                self.assertEqual(payload["current_work"]["request"], "Review parser")
                self.assertEqual(
                    payload["earlier_context"][0]["request"], "Implement parser"
                )
                self.assertNotIn("current_moment", payload)
                reports = [
                    item
                    for item in payload.get("earlier_context", [])
                    if "reported_progress" in item
                ]
                self.assertEqual(len(reports), 1)
                self.assertEqual(reports[0]["outcome"], outcome)
                self.assertEqual(reports[0]["reported_progress"], progress)

    def test_missing_results_preserve_an_unknown_outcome(self) -> None:
        self.focus = replace(self.focus, progress="", outcome=None, turn_active=True)
        payload = self.payload()
        self.assertEqual(
            [
                item
                for item in payload.get("earlier_context", [])
                if "reported_progress" in item
            ],
            [],
        )
        self.assertNotIn("current_moment", payload)

    def test_tiny_target_preserves_latest_request_and_current_report(
        self,
    ) -> None:
        self.config.prompt_target_bytes = 1
        self.focus = replace(
            self.focus,
            initial_request="Implement 猫 parser. " * 500,
            recent_requests=("Earlier request", "Review 猫. " * 500),
            progress="Current 猫 progress. " * 100,
        )
        self.event(
            "turn_done",
            outcome=ExecutionOutcome.SUCCESS,
            progress="Parser delivered after verification",
        )
        request = self.summary_request()
        payload = self.payload(request)
        self.assertGreater(
            len(json.dumps(request, ensure_ascii=False).encode()),
            self.config.prompt_target_bytes,
        )
        self.assertEqual(
            payload["current_work"]["request"],
            select_prompt(self.focus.recent_requests[-1]),
        )
        self.assertEqual(
            payload["current_moment"],
            {
                "reported_progress": bounded(self.focus.progress, 800),
                "outcome": "success",
            },
        )
        self.assertNotIn("historical_summary", payload)
        self.assertEqual(
            [
                item
                for item in payload.get("earlier_context", [])
                if "reported_progress" in item
            ],
            [],
        )

    def test_optional_reports_respect_the_target(
        self,
    ) -> None:
        self.event(
            "turn_done",
            outcome=ExecutionOutcome.SUCCESS,
            progress="Earlier parser result",
        )
        self.event(
            "turn_done",
            outcome=ExecutionOutcome.SUCCESS,
            progress="Latest parser verification " * 20,
        )
        self.config.prompt_target_bytes = 1
        essential = self.summary_request()
        self.config.prompt_target_bytes = (
            len(json.dumps(essential, ensure_ascii=False).encode()) + 1
        )
        request = self.summary_request()
        payload = self.payload(request)
        self.assertLessEqual(
            len(json.dumps(request, ensure_ascii=False).encode()),
            self.config.prompt_target_bytes,
        )
        self.assertNotIn("earlier_context", payload)

    def test_historical_failure_without_prose_keeps_its_outcome_and_age(self) -> None:
        self.event("turn_error", outcome=ExecutionOutcome.ERROR)
        self.focus = replace(self.focus, outcome=None, progress="", turn_active=True)
        payload = self.payload()
        report = next(item for item in payload["earlier_context"] if "outcome" in item)
        self.assertEqual(report, {"outcome": "error", "age": "less than a minute ago"})

    def test_optional_requests_retain_chronology_without_mutating_session_input(
        self,
    ) -> None:
        self.focus = replace(
            self.focus,
            recent_requests=(
                "Implement parser",
                "Keep aliases",
                "Verify compatibility",
            ),
        )
        before = deepcopy(self.focus)
        payload = self.payload()
        self.assertEqual(payload["current_work"]["request"], "Verify compatibility")
        self.assertEqual(
            [
                item["request"]
                for item in payload["earlier_context"]
                if "request" in item
            ],
            [self.focus.initial_request, *self.focus.recent_requests[:-1]],
        )
        self.assertNotIn(self.focus.summary, json.dumps(payload))
        self.assertEqual(self.focus, before)
