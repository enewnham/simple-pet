import json
import unittest

from pet_daemon.sessions import Sessions


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class SessionsTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.sessions = Sessions(self.clock, stale_seconds=60, retention_seconds=120)

    def event(self, event, session="main", harness="claude", **data):
        return self.sessions.handle(
            {"event": event, "session": session, "harness": harness, **data}
        )

    def test_malformed_events_leave_session_state_unchanged(self):
        self.event("intent", goal="Keep this request", aliases=["alternate"])
        before = self.sessions.snapshot()
        aliases = dict(self.sessions.aliases)
        for invalid in (
            {"harness": []},
            {"event": {}},
            {"session": []},
            {"aliases": "alternate"},
            {"aliases": [""]},
            {"aliases": [[]]},
            {"goal": []},
            {"parent": "new"},
            {"parent": "alternate", "aliases": ["main"]},
            {"event": "turn_done", "outcome": []},
            {"outcome": ["bad"]},
            {"outcome": "bogus"},
            {"repo": "/tmp/bad\0path"},
        ):
            with self.subTest(invalid=invalid):
                packet = {
                    "harness": "claude",
                    "event": "intent",
                    "session": "new",
                    **invalid,
                }
                self.assertIsNone(self.sessions.handle(packet))
                self.assertEqual(self.sessions.snapshot(), before)
                self.assertEqual(self.sessions.aliases, aliases)
                self.assertFalse(self.sessions.child_calls)

    def test_tracks_parallel_work_until_each_turn_finishes(self):
        self.event("intent", goal="Refactor batch inserts", turn="turn-1")
        self.event("tool_start", call="edit", tool="Edit", turn="turn-1")
        self.event("tool_start", call="test", tool="cargo test", turn="turn-1")
        self.event("tool_start", harness="codex", call="build", tool="cargo build")
        self.event("tool_start", harness="cursor", call="lint", tool="lint")
        self.assertEqual(len(self.sessions.lanes), 3)
        self.assertTrue(self.sessions.working)
        self.event("tool_done", call="edit")
        self.event("tool_done", harness="codex", call="build")
        self.event("tool_done", harness="cursor", call="lint")
        self.assertTrue(self.sessions.working)
        self.event("tool_done", call="test")
        self.assertTrue(self.sessions.working)
        self.assertTrue(self.sessions.active)
        self.assertIsNone(self.sessions.convergence)
        self.clock.advance(30)
        self.assertEqual(self.sessions.snapshot()["activity"], "working")
        self.event("turn_done")
        self.event("turn_done", harness="codex")
        self.assertTrue(self.sessions.working)
        self.event("turn_done", harness="cursor")
        self.assertFalse(self.sessions.working)
        self.assertEqual(self.sessions.snapshot()["activity"], "idle")

    def test_finishing_one_parallel_turn_does_not_claim_convergence(self):
        for name, harness in [("one", "claude"), ("two", "codex")]:
            self.event("intent", session=name, harness=harness, goal=name)
            self.event(
                "tool_start", session=name, harness=harness, call="test", tool="test"
            )
            self.event("tool_done", session=name, harness=harness, call="test")
        first = self.event("turn_done", session="one")
        self.assertFalse(first["convergence"])
        last = self.event("turn_done", session="two", harness="codex")
        self.assertTrue(last["convergence"])
        self.assertTrue(self.sessions.clean_convergence)
        self.assertEqual(
            self.sessions.convergence,
            [
                {"id": "claude:one", "outcome": "success"},
                {"id": "codex:two", "outcome": "success"},
            ],
        )

    def test_routes_child_calls_through_aliases_and_parent_tool_identifier(self):
        self.event("intent", goal="Refactor batch inserts", turn="parent-turn")
        self.event(
            "subagent_start", session="child", parent="main", parent_call="delegate"
        )
        self.event(
            "tool_start",
            session="main",
            parent_call="delegate",
            call="test",
            tool="pytest",
        )
        child = self.sessions.lanes["claude:child"]
        self.assertEqual(child.goal, "Refactor batch inserts")
        self.assertEqual(child.parent, "claude:main")
        self.assertIn("test", child.calls)
        self.assertFalse(self.sessions.lanes["claude:main"].calls)
        self.event(
            "progress",
            session="child",
            aliases=["child-conversation"],
            progress="Testing the persister",
        )
        self.event("tool_done", session="child-conversation", call="test")
        self.assertFalse(child.calls)
        self.event("turn_done", progress="Parent work complete")
        self.assertTrue(self.sessions.active)
        result = self.event(
            "subagent_done", session="child-conversation", progress="Child tests passed"
        )
        self.assertTrue(result["convergence"])
        self.assertTrue(self.sessions.clean_convergence)
        self.assertEqual(len(self.sessions.lanes), 2)
        self.assertEqual(child.progress, "Child tests passed")

    def test_merges_child_aliases_without_losing_calls(self):
        self.event("intent", goal="Update docs")
        self.event("subagent_start", session="child", parent="main")
        self.event("tool_start", session="child-thread", call="read", tool="Read")
        self.event(
            "progress",
            session="child",
            aliases=["child-thread"],
            progress="Reading the API",
        )
        self.assertEqual(len(self.sessions.lanes), 2)
        self.assertIn("read", self.sessions.lanes["claude:child"].calls)
        self.event("tool_done", session="child-thread", call="read")
        self.assertTrue(self.sessions.working)
        self.assertFalse(self.sessions.lanes["claude:child"].calls)

    def test_alias_merge_preserves_the_latest_unanswered_question(self):
        self.event("intent", session="native", goal="Choose a strategy")
        self.event("intent", session="temporary", goal="Choose a strategy")
        self.event(
            "waiting", session="native", call="first", progress="Choose a strategy"
        )
        self.clock.advance(1)
        self.event("tool_start", session="temporary", call="read", tool="Read")
        self.event("tool_done", session="temporary", call="read")
        self.event(
            "waiting", session="temporary", call="second", progress="Choose a target"
        )
        temporary = self.sessions.lanes["claude:temporary"]
        temporary.last_check_in = self.clock()
        self.event("session_start", session="native", aliases=["temporary"])
        lane = self.sessions.lanes["claude:native"]
        self.assertTrue(lane.waiting)
        self.assertFalse(lane.working)
        self.assertEqual(lane.waiting_call, "second")
        self.assertEqual(lane.waiting_since, self.clock())
        self.assertEqual(lane.last_check_in, self.clock())
        self.assertEqual(lane.phase, "investigating")
        self.assertEqual(lane.progress, "Choose a target")
        self.event("tool_done", session="temporary", call="second")
        self.assertFalse(lane.waiting)

    def test_alias_merge_preserves_active_child_failures_through_completion(self):
        self.event("session_start", session="child")
        self.event("tool_start", session="child-thread", call="tests", tool="pytest")
        self.event("tool_error", session="child-thread", call="tests")
        self.assertTrue(self.sessions.failed)
        self.event(
            "subagent_start",
            session="child",
            parent="main",
            aliases=["child-thread"],
            goal="Check parser",
        )
        child = self.sessions.lanes["claude:child"]
        self.assertTrue(child.active)
        self.assertTrue(self.sessions.failed)
        self.assertEqual(child.failure["activity"], "testing")
        self.event("subagent_done", session="child-thread")
        self.assertEqual(child.outcome, "error")
        self.assertFalse(self.sessions.clean_convergence)

    def test_alias_merge_preserves_completed_child_failure(self):
        self.event("session_start", session="child")
        self.event("tool_start", session="child-thread", call="tests", tool="pytest")
        self.event("tool_error", session="child-thread", call="tests")
        self.event("turn_done", session="child-thread")
        self.assertEqual(self.sessions.lanes["claude:child-thread"].outcome, "error")
        self.event("subagent_done", session="child", aliases=["child-thread"])
        child = self.sessions.lanes["claude:child"]
        self.assertTrue(child.finished)
        self.assertEqual(child.outcome, "error")
        self.assertEqual(child.failure["activity"], "testing")
        self.assertEqual(child.failure["generation"], child.generation)
        snapshot = self.sessions.snapshot()["sessions"]
        self.assertEqual([lane["id"] for lane in snapshot], ["claude:child"])
        self.assertEqual(snapshot[0]["outcome"], "error")
        self.assertFalse(self.sessions.clean_convergence)

    def test_known_file_tools_keep_activity_when_paths_contain_check_terms(self):
        for index, (name, path, expected) in enumerate(
            (
                ("Read", "/repo/build/tests/lint.py", "investigating"),
                ("Edit", "/repo/build/tests/lint.py", "editing"),
            )
        ):
            with self.subTest(tool=name):
                session = f"file-{index}"
                self.event(
                    "tool_start",
                    session=session,
                    call="file",
                    tool=path,
                    tool_name=name,
                )
                lane = self.sessions.lanes[f"claude:{session}"]
                self.assertEqual(lane.calls["file"].activity, expected)
                self.event("tool_error", session=session, call="file")
                self.assertEqual(lane.failure["activity"], expected)

    def test_terminal_failure_closes_pending_calls_without_diagnostic_progress(self):
        self.event("intent", goal="Run the tests")
        self.event("tool_start", call="test", tool="pytest")
        result = self.event(
            "turn_error",
            detail="API unavailable",
            progress="Agent stopped before tests completed",
        )
        self.assertTrue(result["failure"])
        self.assertTrue(result["convergence"])
        self.assertFalse(self.sessions.working)
        self.assertFalse(self.sessions.active)
        self.assertFalse(self.sessions.clean_convergence)
        lane = self.sessions.lanes["claude:main"]
        self.assertEqual(lane.outcome, "error")
        self.assertEqual(lane.progress, "")
        self.assertEqual(lane.failure["activity"], "harness")
        self.assertIsNone(self.event("tool_done", call="test"))
        self.assertIsNone(self.event("tool_start", call="late", tool="Read"))

    def test_old_turn_events_do_not_finish_new_turn(self):
        self.event("intent", goal="First goal", turn="one")
        self.event("turn_done", turn="one")
        self.event("intent", goal="Second goal", turn="two")
        self.event("tool_start", call="current", tool="Edit", turn="two")
        self.assertIsNone(self.event("turn_done", turn="one"))
        self.assertIsNone(
            self.event("tool_error", call="current", turn="one", detail="Old result")
        )
        self.assertTrue(self.sessions.working)
        self.assertEqual(
            self.sessions.lanes["claude:main"].recent_requests[-1], "Second goal"
        )

    def test_error_outcome_on_tool_completion_records_failure(self):
        self.event("tool_start", call="checks", tool="pytest")
        result = self.event("tool_done", call="checks", outcome="error")
        self.assertTrue(result["failure"])
        lane = self.sessions.lanes["claude:main"]
        self.assertEqual(lane.failure["activity"], "testing")
        self.assertFalse(lane.failure["resolved"])
        self.event("turn_done")
        self.assertEqual(lane.outcome, "error")

    def test_successful_retry_resolves_same_tool_failure(self):
        self.event("intent", goal="Fix tests")
        self.event("tool_start", call="first", tool="pytest")
        self.event("tool_error", call="first", detail="AssertionError")
        self.assertTrue(self.sessions.failed)
        self.event("tool_start", call="other", tool="git diff")
        self.event("tool_done", call="other")
        self.assertTrue(self.sessions.failed)
        self.event("tool_start", call="retry", tool="pytest")
        self.event("tool_done", call="retry", progress="Twenty tests passed")
        self.assertFalse(self.sessions.failed)
        self.assertTrue(self.sessions.lanes["claude:main"].failure["resolved"])
        self.event("turn_done")
        self.assertTrue(self.sessions.clean_convergence)

    def test_failure_counts_follow_activity_within_the_session_and_turn(self):
        self.event("intent", goal="Check parser", turn="first")
        for index, tool in enumerate(("pytest", "vitest")):
            self.event("tool_start", call=f"test-{index}", tool=tool)
            self.event(
                "tool_error",
                call=f"test-{index}",
                detail=f"SECRET {index}",
                error_signature=f"HASH {index}",
            )
        failure = self.sessions.lanes["claude:main"].failure
        self.assertEqual(failure["activity"], "testing")
        self.assertEqual(failure["count"], 2)
        self.assertNotIn("detail", failure)
        self.assertNotIn("signature", failure)
        self.event("tool_start", call="build", tool="cargo build")
        self.event("tool_error", call="build")
        self.assertEqual(
            self.sessions.lanes["claude:main"].failure["activity"], "building"
        )
        self.assertEqual(self.sessions.lanes["claude:main"].failure["count"], 1)
        self.event("tool_start", call="test-again", tool="pytest")
        self.event("tool_error", call="test-again")
        self.assertEqual(self.sessions.lanes["claude:main"].failure["count"], 3)
        self.event("tool_start", session="other", call="test", tool="pytest")
        self.event("tool_error", session="other", call="test")
        self.assertEqual(self.sessions.lanes["claude:other"].failure["count"], 1)
        self.event("intent", goal="Check another task", turn="second")
        self.event("tool_start", call="fresh", tool="pytest")
        self.event("tool_error", call="fresh")
        self.assertEqual(self.sessions.lanes["claude:main"].failure["count"], 1)

    def test_success_clears_only_its_matching_activity(self):
        self.event("intent", goal="Validate changes")
        self.event("tool_start", call="tests", tool="pytest")
        self.event("tool_error", call="tests")
        self.event("tool_start", call="build", tool="cargo build")
        self.event("tool_error", call="build")
        self.event("tool_start", call="tests-retry", tool="vitest")
        self.event("tool_done", call="tests-retry")
        self.assertTrue(self.sessions.failed)
        self.event("tool_start", call="build-retry", tool="cargo build")
        self.event("tool_done", call="build-retry")
        self.assertFalse(self.sessions.failed)

    def test_command_and_harness_failures_remain_broad_and_cancellation_is_separate(
        self,
    ):
        self.event("tool_start", call="script", tool="./release.sh")
        self.event("tool_error", call="script", detail="SECRET script diagnostics")
        self.assertEqual(
            self.sessions.lanes["claude:main"].failure["activity"], "command"
        )
        self.event("turn_error", detail="SECRET harness diagnostics")
        self.assertEqual(
            self.sessions.lanes["claude:main"].failure["activity"], "harness"
        )
        self.assertNotIn("SECRET", json.dumps(self.sessions.snapshot()))
        self.event("intent", goal="Fresh task", turn="fresh")
        self.event("interrupt")
        self.assertFalse(self.sessions.failed)
        self.assertEqual(self.sessions.lanes["claude:main"].outcome, "cancelled")

    def test_first_prompt_survives_followups_and_recent_requests_keep_four(self):
        self.event("intent", goal="Refactor the parser", turn="first")
        for index in range(6):
            self.event(
                "intent", goal=f"Followup request {index}", turn=f"followup-{index}"
            )
        snapshot = self.sessions.snapshot()["sessions"][0]
        self.assertEqual(snapshot["first_prompt"], "Refactor the parser")
        self.assertEqual(
            snapshot["recent_requests"],
            [f"Followup request {index}" for index in range(2, 6)],
        )
        self.assertEqual(snapshot["session_context"], "")

    def test_distinct_requests_with_the_same_native_turn_keep_the_anchor_and_both_requests(
        self,
    ):
        self.event("intent", goal="Compress the workflow", turn="native-turn")
        generation = self.sessions.lanes["claude:main"].generation
        result = self.event(
            "intent", goal="Explain the counter limits", turn="native-turn"
        )
        self.assertIsNotNone(result)
        lane = self.sessions.lanes["claude:main"]
        self.assertEqual(lane.first_prompt, "Compress the workflow")
        self.assertEqual(
            lane.recent_requests,
            ["Compress the workflow", "Explain the counter limits"],
        )
        self.assertGreater(lane.generation, generation)
        self.assertEqual(lane.turn, "native-turn")

    def test_delegated_task_sets_the_child_anchor(self):
        self.event("intent", goal="Review the changes")
        self.event(
            "subagent_start",
            session="child",
            parent="main",
            goal="Inspect parser behavior",
        )
        self.event(
            "intent", session="child", goal="Check edge cases", turn="child-next"
        )
        child = next(
            lane
            for lane in self.sessions.snapshot()["sessions"]
            if lane["id"] == "claude:child"
        )
        self.assertEqual(child["first_prompt"], "Inspect parser behavior")
        self.assertIn("Check edge cases", child["recent_requests"])

    def test_new_goal_retains_old_failure_context_without_failing_new_turn(self):
        self.event("intent", goal="Fix old tests", turn="one")
        self.event("tool_start", call="bad", tool="pytest")
        self.event("tool_error", call="bad", detail="Old assertion")
        self.event("turn_done", turn="one")
        self.event("intent", goal="Update documentation", turn="two")
        self.assertFalse(self.sessions.failed)
        snapshot = self.sessions.snapshot()["sessions"][0]
        self.assertEqual(snapshot["first_prompt"], "Fix old tests")
        self.assertEqual(snapshot["recent_requests"][-1], "Update documentation")
        self.assertEqual(snapshot["failure"]["goal"], "Fix old tests")
        self.assertFalse(snapshot["failure"]["current_turn"])
        self.event("turn_done", turn="two")
        self.assertTrue(self.sessions.clean_convergence)

    def test_interrupt_cancels_descendants_and_keeps_unrelated_lane_running(self):
        self.event("intent", goal="Parent task")
        self.event("subagent_start", session="child", parent="main")
        self.event("subagent_start", session="grandchild", parent="child")
        self.event("tool_start", session="grandchild", call="test", tool="test")
        self.event(
            "tool_start", session="other", harness="cursor", call="build", tool="build"
        )
        result = self.event("interrupt")
        self.assertFalse(result["convergence"])
        self.assertTrue(self.sessions.working)
        for key in ["claude:main", "claude:child", "claude:grandchild"]:
            self.assertEqual(self.sessions.lanes[key].outcome, "cancelled")
            self.assertFalse(self.sessions.lanes[key].calls)
        self.event("tool_done", session="other", harness="cursor", call="build")
        result = self.event("turn_done", session="other", harness="cursor")
        self.assertTrue(result["convergence"])
        self.assertFalse(self.sessions.clean_convergence)

    def test_manual_calls_complete_with_error_or_cancellation_outcomes(self):
        for session, outcome in [
            ("failed", "error"),
            ("cancelled", "cancelled"),
            ("clean", "success"),
        ]:
            self.event(
                "tool_start",
                session=session,
                harness="cli",
                call="command",
                tool="build",
                manual=True,
            )
            kind = "tool_error" if outcome == "error" else "tool_done"
            result = self.event(
                kind,
                session=session,
                harness="cli",
                call="command",
                manual=True,
                outcome=outcome,
            )
            self.assertTrue(result["convergence"])
            self.assertEqual(self.sessions.lanes[f"cli:{session}"].outcome, outcome)
            self.assertEqual(self.sessions.clean_convergence, outcome == "success")

    def test_native_prompt_without_turn_id_resets_cancelled_turn_before_stop(self):
        self.event("intent", goal="Try the build")
        self.event("tool_start", call="first", tool="cargo build")
        self.event("tool_done", call="first", outcome="cancelled")
        lane = self.sessions.lanes["claude:main"]
        generation = lane.generation
        self.assertTrue(lane.active)
        self.assertTrue(lane.cancelled)
        self.event("intent", goal="Try the build again")
        self.assertGreater(lane.generation, generation)
        self.assertFalse(lane.cancelled)
        self.assertEqual(lane.first_prompt, "Try the build")
        self.assertEqual(lane.recent_requests[-1], "Try the build again")
        self.event("tool_start", call="retry", tool="cargo build")
        self.event("tool_done", call="retry", progress="Build completed")
        result = self.event("turn_done")
        self.assertTrue(result["convergence"])
        self.assertEqual(lane.outcome, "success")
        self.assertTrue(self.sessions.clean_convergence)

    def test_duplicate_subagent_start_preserves_child_call_and_turn(self):
        self.event("intent", goal="Parent task", turn="parent-turn")
        self.event(
            "subagent_start",
            session="child",
            parent="main",
            parent_call="delegate",
            turn="parent-turn",
        )
        self.event(
            "tool_start", session="child", call="test", tool="pytest", turn="child-turn"
        )
        child = self.sessions.lanes["claude:child"]
        generation = child.generation
        self.event(
            "subagent_start",
            session="child",
            parent="main",
            parent_call="delegate",
            turn="parent-turn",
        )
        self.assertEqual(child.generation, generation)
        self.assertEqual(child.turn, "child-turn")
        self.assertIn("test", child.calls)
        self.assertTrue(self.sessions.working)
        self.event("tool_done", session="child", call="test", turn="child-turn")
        self.assertFalse(child.calls)
        self.assertEqual(self.sessions.lanes["claude:main"].turn, "parent-turn")

    def test_unknown_tool_result_preserves_waiting_and_matching_reply_resumes(self):
        self.event("intent", goal="Choose the migration strategy")
        self.event(
            "waiting",
            call="question",
            tool="AskUserQuestion",
            progress="Waiting for a strategy",
        )
        snapshot = self.sessions.snapshot()
        self.assertEqual(snapshot["activity"], "idle")
        self.assertTrue(snapshot["sessions"][0]["waiting_for_input"])
        self.assertEqual(snapshot["sessions"][0]["active_tools"], [])
        self.assertIsNone(self.event("tool_done", call="unrelated"))
        self.assertTrue(self.sessions.snapshot()["sessions"][0]["waiting_for_input"])
        result = self.event(
            "tool_done", call="question", progress="Developer chose batching"
        )
        self.assertFalse(result["failure"])
        self.assertFalse(result["convergence"])
        self.assertEqual(result["session"], "claude:main")
        snapshot = self.sessions.snapshot()
        self.assertFalse(snapshot["sessions"][0]["waiting_for_input"])
        self.assertEqual(
            snapshot["sessions"][0]["progress"], "Developer chose batching"
        )
        self.assertTrue(self.sessions.working)
        self.assertEqual(snapshot["activity"], "working")
        self.assertTrue(self.sessions.active)

    def test_stale_lane_expires_and_retention_removes_aliases(self):
        self.event("intent", aliases=["alternate"], goal="Build it")
        self.event("tool_start", call="build", tool="build")
        self.clock.advance(59)
        self.assertFalse(self.sessions.prune())
        self.assertTrue(self.sessions.working)
        self.clock.advance(1)
        self.assertTrue(self.sessions.prune())
        self.assertFalse(self.sessions.working)
        self.assertFalse(self.sessions.active)
        self.assertEqual(self.sessions.lanes["claude:main"].outcome, "stale")
        self.assertIsNone(self.sessions.convergence)
        self.clock.now += 120
        self.sessions.prune()
        self.assertFalse(self.sessions.lanes)
        self.assertFalse(self.sessions.aliases)

    def test_checks_each_active_tasks_quiet_time_independently(self):
        self.event("tool_start", call="build", tool="Build")
        self.clock.advance(299)
        self.event(
            "tool_start", session="other", harness="codex", call="lint", tool="Lint"
        )
        self.assertEqual(self.sessions.check_in_due(300), [])
        self.clock.advance(1)
        self.assertEqual(self.sessions.check_in_due(300), ["claude:main"])
        snapshots = {lane["id"]: lane for lane in self.sessions.snapshot()["sessions"]}
        self.assertEqual(snapshots["claude:main"]["quiet_seconds"], 300)
        self.assertEqual(snapshots["codex:other"]["quiet_seconds"], 1)
        self.assertIsNone(snapshots["claude:main"]["waiting_seconds"])
        self.event("progress", progress="Build still running")
        self.assertEqual(self.sessions.check_in_due(300), [])

    def test_duplicate_waiting_events_preserve_age_until_progress_resumes_work(self):
        self.event(
            "waiting",
            call="question",
            tool="AskUserQuestion",
            progress="Choose a strategy",
        )
        self.assertTrue(self.sessions.active)
        self.assertFalse(self.sessions.working)
        self.clock.advance(150)
        self.event(
            "waiting",
            call="question",
            tool="AskUserQuestion",
            progress="Strategy choice pending",
        )
        self.clock.advance(149)
        self.event("waiting", call="question", tool="AskUserQuestion")
        self.assertEqual(self.sessions.check_in_due(300), [])
        self.clock.advance(1)
        self.assertEqual(self.sessions.check_in_due(300), ["claude:main"])
        snapshot = self.sessions.snapshot()["sessions"][0]
        self.assertEqual(snapshot["waiting_seconds"], 300)
        self.assertEqual(snapshot["quiet_seconds"], 1)
        self.event("progress", progress="Strategy chosen")
        self.assertEqual(self.sessions.check_in_due(300), [])
        self.assertIsNone(self.sessions.snapshot()["sessions"][0]["waiting_seconds"])

    def test_each_question_starts_its_own_waiting_timer(self):
        self.event("waiting", call="first", tool="AskUserQuestion")
        self.clock.advance(299)
        self.event("waiting", call="second", tool="AskUserQuestion")
        self.clock.advance(1)
        self.assertEqual(self.sessions.check_in_due(300), [])
        self.assertEqual(self.sessions.snapshot()["sessions"][0]["waiting_seconds"], 1)
        self.clock.advance(299)
        self.assertEqual(self.sessions.check_in_due(300), ["claude:main"])

    def test_accepted_activity_resumes_waiting_without_matching_the_question(self):
        for kind in ("tool_start", "tool_done", "tool_error", "progress"):
            with self.subTest(event=kind):
                self.event("intent", session=kind, goal="Choose a strategy")
                self.event("tool_start", session=kind, call="background", tool="Read")
                self.event(
                    "waiting", session=kind, call="question", tool="AskUserQuestion"
                )
                self.clock.advance(300)
                call = "resume" if kind == "tool_start" else "background"
                result = self.event(
                    kind,
                    session=kind,
                    call=call,
                    tool="Read",
                    progress="Continuing investigation",
                )
                self.assertIsNotNone(result)
                snapshot = next(
                    item
                    for item in self.sessions.snapshot()["sessions"]
                    if item["id"] == f"claude:{kind}"
                )
                self.assertFalse(snapshot["waiting_for_input"])
                self.assertIsNone(snapshot["waiting_seconds"])
                self.assertTrue(snapshot["turn_active"])

    def test_ignored_activity_preserves_the_input_wait(self):
        self.event("intent", turn="current", goal="Choose a strategy")
        self.event("tool_start", call="background", tool="Read")
        self.event("waiting", call="question", tool="AskUserQuestion")
        ignored = (
            ("tool_start", {}),
            ("tool_start", {"call": "background"}),
            ("tool_done", {"call": "unknown"}),
            ("tool_error", {"call": "unknown"}),
            ("progress", {"turn": "another", "progress": "Outdated progress"}),
        )
        for kind, data in ignored:
            with self.subTest(event=kind, data=data):
                self.assertIsNone(self.event(kind, **data))
                self.assertTrue(
                    self.sessions.snapshot()["sessions"][0]["waiting_for_input"]
                )

    def test_explicit_waiting_after_progress_starts_a_new_wait(self):
        self.event("waiting", call="question", tool="AskUserQuestion")
        self.clock.advance(100)
        self.event("progress", progress="Investigating another option")
        self.assertFalse(self.sessions.snapshot()["sessions"][0]["waiting_for_input"])
        self.clock.advance(100)
        self.event("waiting", call="question", tool="AskUserQuestion")
        self.clock.advance(1)
        snapshot = self.sessions.snapshot()["sessions"][0]
        self.assertTrue(snapshot["waiting_for_input"])
        self.assertEqual(snapshot["waiting_seconds"], 1)
        self.assertEqual(self.sessions.check_in_due(300), [])

    def test_check_in_attempts_repeat_after_the_interval_and_reset_on_new_turn(self):
        self.event("intent", turn="first", goal="Run checks")
        self.event("tool_start", turn="first", call="checks", tool="pytest")
        self.clock.advance(300)
        self.assertEqual(self.sessions.check_in_due(300), ["claude:main"])
        lane = self.sessions.lanes["claude:main"]
        lane.last_check_in = self.clock()
        self.clock.advance(299)
        self.assertEqual(self.sessions.check_in_due(300), [])
        self.clock.advance(1)
        self.assertEqual(self.sessions.check_in_due(300), ["claude:main"])
        self.event("intent", turn="second", goal="Run another build")
        self.event("tool_start", turn="second", call="build", tool="make")
        self.assertIsNone(lane.last_check_in)
        self.assertEqual(self.sessions.check_in_due(300), [])
        self.clock.advance(300)
        self.assertEqual(self.sessions.check_in_due(300), ["claude:main"])

    def test_idle_completed_cancelled_and_stale_sessions_do_not_get_check_ins(self):
        self.assertEqual(self.sessions.check_in_due(30), [])
        self.event("session_start", session="idle")
        self.event("intent", session="completed", goal="Run checks")
        self.event("turn_done", session="completed")
        self.event("intent", session="cancelled", goal="Build")
        self.event("interrupt", session="cancelled")
        self.event("tool_start", session="stale", call="build", tool="Build")
        self.clock.advance(60)
        self.sessions.prune()
        self.assertEqual(self.sessions.lanes["claude:stale"].outcome, "stale")
        self.assertEqual(self.sessions.check_in_due(30), [])


if __name__ == "__main__":
    unittest.main()
