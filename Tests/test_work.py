from pathlib import Path
import tempfile
import unittest

from pet_daemon.memory import Memory
from pet_daemon.work import build_work


class WorkTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "memory.sqlite3"
        self.now = 1000
        self.memory = Memory(self.path, "fixture", clock=lambda: self.now)
        self.store = self.memory.work

    def tearDown(self):
        self.memory.close()
        self.temporary.cleanup()

    def start(self, session="codex:one", intent="Improve parser"):
        return self.store.session_intent(session, intent)

    def progress(self, session="codex:one", value="Implemented quoted string handling"):
        self.store.session_event(session, {"event": "progress", "progress": value, "harness": session.split(":", 1)[0]})

    def pr(self, number=7, **extra):
        return {"url": f"https://github.com/team/project/pull/{number}", "number": number, "state": "OPEN", "is_draft": True,
                "head": "head-one", "head_repository": "github.com/team/project", "head_branch": "feature",
                "repository": "github.com/team/project", "check_revision": "head-one", "checks": [], "review_requested": False,
                "title": "Handle parser quotes", "body": "Recognize quoted strings.", "related_links": [], **extra}

    def observation(self, pr=None, **extra):
        return {"repository": "github.com/team/project", "worktree": "/work/parser", "branch": "feature", "sessions": ["codex:one"],
                "current": True, "local": {"repository": "github.com/team/project", "branch": "feature", "head": "head-one",
                                           "staged": 0, "unstaged": 1, "untracked": 0, "conflicts": 0, "ahead": 0, "behind": 0},
                "pr": pr, "local_checked_at": self.now, "github_checked_at": self.now, **extra}

    def kinds(self):
        return {item.kind for item in self.store.opportunities()}

    def restart(self):
        self.memory.close()
        self.memory = Memory(self.path, "fixture", clock=lambda: self.now)
        self.store = self.memory.work

    def test_native_intents_and_progress_form_an_attributed_connected_account(self):
        self.start()
        self.progress()
        self.store.observe_repository(self.observation(self.pr()))
        account = self.store.session_context("codex:one")
        self.assertEqual(account["intent"], "Improve parser")
        self.assertEqual(account["requests"][0]["data"], {"session": "codex:one", "text": "Improve parser"})
        self.assertTrue(any(item["data"].get("text") == "Implemented quoted string handling" for item in account["activity"]))
        self.assertEqual({item["kind"] for item in account["artifacts"]}, {"branch", "pull_request"})
        pull_request = next(item for item in account["artifacts"] if item["kind"] == "pull_request")
        self.assertEqual(pull_request["title"], "Handle parser quotes")
        observed = next(item for item in pull_request["evidence"] if item["kind"] == "observation")
        self.assertEqual(observed["data"]["pr"]["title"], "Handle parser quotes")
        self.assertTrue(any(item["data"].get("association_only") for item in pull_request["evidence"]))
        self.assertEqual(self.kinds(), set())

    def test_followup_intent_refines_work_without_collapsing_other_sessions(self):
        first = self.start()
        self.assertEqual(self.store.session_intent("codex:one", "Cover escape sequences"), first)
        second = self.start("claude:two", "Review the parser API")
        self.store.observe_repository(self.observation(self.pr(), sessions=["codex:one", "claude:two"]))
        self.assertNotEqual(first, second)
        self.assertEqual(self.store.session_context("codex:one")["intent"], "Cover escape sequences")
        self.assertEqual(self.store.session_context("claude:two")["intent"], "Review the parser API")
        self.assertEqual(len(self.store.tracked_artifacts()), 2)

    def test_shared_artifact_observations_update_each_connected_work(self):
        self.start()
        self.start("claude:two", "Review the parser API")
        self.store.observe_repository(self.observation(self.pr(), sessions=["codex:one", "claude:two"]))
        self.store.observe_repository(self.observation(self.pr(state="MERGED"), sessions=[]))
        for session in ("codex:one", "claude:two"):
            account = self.store.session_context(session)
            pull_request = next(item for item in account["artifacts"] if item["kind"] == "pull_request")
            self.assertEqual(next(item["data"]["pr"]["state"] for item in pull_request["evidence"] if item["kind"] == "observation"), "MERGED")
            self.assertEqual(self.store._current(session)["status"], "active")

    def test_successful_stop_records_execution_without_completing_work(self):
        work = self.start()
        self.store.session_event("codex:one", {"event": "turn_done", "outcome": "success"})
        self.assertEqual(self.store._current("codex:one")["status"], "active")
        self.assertEqual(self.store.session_intent("codex:one", "Check compatibility"), work)
        self.assertEqual(self.kinds(), set())

    def test_repository_observation_cannot_create_a_user_request(self):
        self.store.observe_repository(self.observation(self.pr()))
        self.assertEqual(self.store.session_context("codex:one"), {})
        self.assertEqual(self.kinds(), set())

    def test_verified_current_user_review_request_creates_an_opportunity(self):
        self.start()
        self.store.observe_repository(self.observation(self.pr(review_requested=True)))
        self.assertEqual(self.kinds(), {"review_output"})
        self.assertEqual(self.store.opportunities()[0].artifact["ref"], self.pr()["url"])
        self.store.observe_repository(self.observation(self.pr(review_requested=None)))
        self.assertEqual(self.kinds(), set())

    def test_only_checks_for_current_pr_head_create_check_opportunities(self):
        self.start()
        self.store.observe_repository(self.observation(self.pr(checks=[["tests", "FAILURE"]], check_revision="older")))
        self.assertEqual(self.kinds(), set())
        self.store.observe_repository(self.observation(self.pr(checks=[["tests", "FAILURE", "github", "older"]])))
        self.assertEqual(self.kinds(), set())
        self.store.observe_repository(self.observation(self.pr(checks=[["tests", "FAILURE", "github", "head-one"]])))
        self.assertEqual(self.kinds(), {"resolve_checks"})
        self.store.observe_repository(self.observation(self.pr(checks=[["tests", "SUCCESS"]])))
        self.assertEqual(self.kinds(), set())

    def test_draft_and_green_checks_alone_do_not_claim_readiness(self):
        self.start()
        self.progress()
        self.store.observe_repository(self.observation(self.pr(checks=[["tests", "SUCCESS"]])))
        self.assertEqual(self.kinds(), set())
        self.assertEqual(self.store._current("codex:one")["status"], "active")

    def test_pr_preparation_requires_observed_output_progress_and_successful_absence_lookup(self):
        self.start()
        self.store.observe_repository(self.observation(github_checked_at=0))
        self.assertEqual(self.kinds(), set())
        self.progress()
        self.assertEqual(self.kinds(), set())
        self.store.observe_repository(self.observation())
        self.assertEqual(self.kinds(), {"prepare_pr"})
        opportunity = self.store.opportunities()[0]
        self.assertIn("may benefit", opportunity.next_step)
        self.assertIn("Agent progress does not establish that the work is complete.", opportunity.uncertainty)
        clean = self.observation()
        clean["local"]["unstaged"] = 0
        self.store.observe_repository(clean)
        self.assertEqual(self.kinds(), set())

    def test_absence_lookup_without_agent_progress_does_not_create_preparation(self):
        self.start()
        self.store.observe_repository(self.observation())
        self.assertEqual(self.kinds(), set())

    def test_failed_absence_refresh_preserves_evidence_and_suppresses_pr_preparation(self):
        self.start()
        self.progress()
        self.store.observe_repository(self.observation())
        self.assertEqual(self.kinds(), {"prepare_pr"})
        observed_at = self.now
        self.now += 100
        self.store.observe_repository(self.observation(github_error="GitHub is unavailable", github_checked_at=observed_at))
        self.assertEqual(self.kinds(), set())
        branch = self.store.session_context("codex:one")["artifacts"][0]
        github = next(record for record in branch["evidence"] if record["source"] == "github")
        self.assertIsNone(github["data"]["pr"])
        self.assertEqual(github["observed_at"], observed_at)
        self.store.observe_repository(self.observation())
        self.assertEqual(self.kinds(), {"prepare_pr"})

    def test_review_and_check_opportunities_disclose_failed_source_refresh(self):
        self.start()
        pull_request = self.pr(review_requested=True, checks=[["tests", "FAILURE"]])
        self.store.observe_repository(self.observation(pull_request))
        observed_at = self.now
        self.now += 100
        self.store.observe_repository(self.observation(pull_request, github_error="GitHub is unavailable", local_error="Checkout is unavailable"))
        self.assertEqual(self.kinds(), {"review_output", "resolve_checks"})
        for opportunity in self.store.opportunities():
            self.assertIn("GitHub refresh failed. Status comes from the last successful observation.", opportunity.uncertainty)
            self.assertIn("Local repository refresh failed. Output information comes from the last successful observation.", opportunity.uncertainty)
            self.assertEqual(next(record["observed_at"] for record in opportunity.evidence if record["source"] == "github" and record["kind"] == "observation"), observed_at)

    def test_inconsistent_independent_pr_branch_field_cannot_attach_unrelated_work(self):
        self.start()
        self.store.observe_repository(self.observation())
        self.store.observe_repository({"repository": "github.com/team/project", "worktree": "", "branch": "feature", "local": None,
                                       "pr": self.pr(head_branch="another", review_requested=True), "github_checked_at": self.now})
        artifacts = self.store.session_context("codex:one")["artifacts"]
        self.assertEqual([item["kind"] for item in artifacts], ["branch"])
        self.assertEqual(self.kinds(), set())

    def test_stack_links_connect_individual_pr_states_and_exact_url_followups(self):
        self.start()
        link = {"kind": "pull_request", "ref": self.pr(8)["url"], "relationship": "stack_entry", "source": "github_stack",
                "stack": 4, "position": 1, "state": "OPEN", "is_draft": True, "head": "head-two", "head_branch": "next", "title": "Expand parser coverage"}
        self.store.observe_repository(self.observation(self.pr(related_links=[link])))
        self.assertEqual(len(self.store.tracked_artifacts()), 3)
        self.assertEqual(next(item["title"] for item in self.store.tracked_artifacts() if item["ref"] == self.pr(8)["url"]), "Expand parser coverage")
        self.store.observe_repository({"repository": "github.com/team/project", "worktree": "", "branch": "", "sessions": [], "local": None,
                                       "pr": self.pr(8, state="MERGED", head="head-two"), "github_checked_at": self.now})
        account = self.store.session_context("codex:one")
        states = {item["ref"]: next(record["data"]["pr"]["state"] for record in item["evidence"] if record["kind"] == "observation")
                  for item in account["artifacts"] if item["kind"] == "pull_request"}
        self.assertEqual(states, {self.pr()["url"]: "OPEN", self.pr(8)["url"]: "MERGED"})
        self.assertEqual(self.store._current("codex:one")["status"], "active")

    def test_stack_peer_check_failure_remains_independent_of_merged_pr(self):
        self.start()
        link = {"kind": "pull_request", "ref": self.pr(8)["url"], "relationship": "stack_entry", "source": "github_stack"}
        self.store.observe_repository(self.observation(self.pr(state="MERGED", related_links=[link])))
        self.store.observe_repository({"repository": "github.com/team/project", "worktree": "", "branch": "", "local": None,
                                       "pr": self.pr(8, checks=[["tests", "FAILURE"]]), "github_checked_at": self.now})
        self.assertEqual(self.kinds(), {"resolve_checks"})
        self.assertEqual(self.store.opportunities()[0].artifact["ref"], self.pr(8)["url"])

    def test_sibling_state_change_advances_the_primary_opportunity_version(self):
        self.start()
        link = {"kind": "pull_request", "ref": self.pr(8)["url"], "relationship": "stack_entry", "source": "github_stack"}
        self.store.observe_repository(self.observation(self.pr(review_requested=True, related_links=[link])))
        original = self.store.opportunities()[0]
        self.assertTrue(self.store.is_current(original.id, original.version))
        self.store.observe_repository({"repository": "github.com/team/project", "worktree": "", "branch": "", "local": None,
                                       "pr": self.pr(8, state="MERGED"), "github_checked_at": self.now})
        current = self.store.opportunities()[0]
        self.assertEqual(current.id, original.id)
        self.assertGreater(current.version, original.version)
        self.assertFalse(self.store.is_current(original.id, original.version))

    def test_tool_event_names_do_not_rearm_an_unchanged_opportunity(self):
        self.start()
        self.store.session_event("codex:one", {"event": "tool_start", "tool_name": "Read"})
        self.store.observe_repository(self.observation(self.pr(review_requested=True)))
        original = self.store.opportunities()[0]
        self.store.record_surfacing(original.id, original.version, "accepted")
        self.store.session_event("codex:one", {"event": "tool_done", "tool_name": "Write", "outcome": "success"})
        self.assertEqual(self.kinds(), set())
        stored = self.store._rows("SELECT version FROM work_opportunities WHERE pet=? AND id=?", (original.id,))[0]
        self.assertEqual(stored["version"], original.version)
        self.store.session_event("codex:one", {"event": "waiting"})
        self.assertEqual(self.kinds(), {"review_output"})
        self.assertGreater(self.store.opportunities()[0].version, original.version)

    def test_native_aliases_share_session_context_and_survive_restart(self):
        work = self.store.session_intent("codex:one", "Improve parser", aliases=["codex:alternate"])
        self.store.session_event("codex:one", {"event": "progress", "harness": "codex", "aliases": ["third"], "progress": "Parser changes"})
        self.restart()
        self.assertEqual(self.store._current("codex:third")["id"], work)
        self.assertEqual(self.store.session_context("codex:alternate")["intent"], "Improve parser")
        self.assertIsNone(self.store._current("third"))

    def test_poll_times_and_nonzero_git_count_changes_do_not_change_meaningful_revision(self):
        self.start()
        self.progress()
        self.store.observe_repository(self.observation())
        account = self.store.session_context("codex:one")
        opportunity = self.store.opportunities()[0]
        self.now += 100
        observed = self.observation()
        observed["local"]["unstaged"] = 9
        self.store.observe_repository(observed)
        self.assertEqual(self.store.session_context("codex:one")["revision"], account["revision"])
        current = self.store.opportunities()[0]
        self.assertEqual((current.id, current.version, current.changed_at), (opportunity.id, opportunity.version, opportunity.changed_at))
        self.assertEqual(next(item["observed_at"] for item in current.evidence if item["source"] == "git"), self.now)

    def test_optional_stack_lookup_errors_do_not_rearm_an_unchanged_suggestion(self):
        self.start()
        self.store.observe_repository(self.observation(self.pr(review_requested=True, related_links_error="Stack lookup unavailable")))
        opportunity = self.store.opportunities()[0]
        account = self.store.session_context("codex:one")
        self.store.record_surfacing(opportunity.id, opportunity.version, "accepted")
        self.now += 100
        self.store.observe_repository(self.observation(self.pr(review_requested=True, related_links_error="Stack lookup rate limited")))
        self.assertEqual(self.store.session_context("codex:one")["revision"], account["revision"])
        self.assertEqual(self.store.opportunities(), ())

    def test_new_head_changes_context_and_opportunity_revision(self):
        self.start()
        self.progress()
        self.store.observe_repository(self.observation())
        account = self.store.session_context("codex:one")
        opportunity = self.store.opportunities()[0]
        observed = self.observation()
        observed["local"]["head"] = "head-two"
        self.store.observe_repository(observed)
        self.assertNotEqual(self.store.session_context("codex:one")["revision"], account["revision"])
        self.assertFalse(self.store.is_current(opportunity.id, opportunity.version))

    def test_only_accepted_current_delivery_marks_a_version_surfaced(self):
        self.start()
        self.store.observe_repository(self.observation(self.pr(review_requested=True)))
        opportunity = self.store.opportunities()[0]
        self.store.record_surfacing(opportunity.id, opportunity.version, "rejected")
        self.store.record_surfacing(opportunity.id, opportunity.version + 1, "accepted")
        self.assertTrue(self.store.is_current(opportunity.id, opportunity.version))
        self.store.record_surfacing(opportunity.id, opportunity.version, "accepted")
        self.restart()
        self.assertEqual(self.kinds(), set())
        self.store.observe_repository(self.observation(self.pr(review_requested=True)))
        self.assertEqual(self.kinds(), set())

    def test_source_timestamps_survive_cached_observation_replay(self):
        self.start()
        self.now += 10000
        self.store.observe_repository(self.observation(self.pr(), local_checked_at=500, github_checked_at=600))
        artifacts = self.store.session_context("codex:one")["artifacts"]
        branch = next(item for item in artifacts if item["kind"] == "branch")
        times = {item["source"]: item["observed_at"] for item in branch["evidence"] if item["kind"] == "observation"}
        self.assertEqual(times, {"git": 500, "github": 600})

    def test_delayed_open_failure_cannot_replace_a_newer_merged_observation(self):
        self.start()
        self.store.observe_repository(self.observation(self.pr(checks=[["tests", "FAILURE"]]), github_checked_at=500))
        self.assertEqual(self.kinds(), {"resolve_checks"})
        merged = self.pr(state="MERGED", head="head-two", title="Verified parser changes")
        self.store.observe_repository({"repository": "github.com/team/project", "worktree": "", "branch": "", "local": None,
                                       "pr": merged, "github_checked_at": 700})
        self.assertEqual(self.kinds(), set())
        self.store.observe_repository(self.observation(self.pr(checks=[["tests", "FAILURE"]]), github_checked_at=600))
        self.assertEqual(self.kinds(), set())
        pull_request = next(item for item in self.store.session_context("codex:one")["artifacts"] if item["kind"] == "pull_request")
        self.assertEqual((pull_request["revision"], pull_request["title"]), ("head-two", "Verified parser changes"))
        github = next(record for record in pull_request["evidence"] if record["source"] == "github" and record["kind"] == "observation")
        self.assertEqual((github["data"]["pr"]["state"], github["observed_at"]), ("MERGED", 700))

    def test_cached_git_observation_cannot_roll_back_a_newer_head_or_verification_time(self):
        self.start()
        newer = self.observation(local_checked_at=700)
        newer["local"]["head"] = "head-two"
        self.store.observe_repository(newer)
        self.store.observe_repository(self.observation(local_checked_at=500))
        branch = next(item for item in self.store.session_context("codex:one")["artifacts"] if item["kind"] == "branch")
        self.assertEqual(branch["revision"], "head-two")
        git = next(record for record in branch["evidence"] if record["source"] == "git")
        self.assertEqual((git["data"]["head"], git["observed_at"]), ("head-two", 700))

    def test_worktree_removal_preserves_context_and_independent_pr_updates(self):
        self.start()
        self.store.observe_repository(self.observation(self.pr()))
        self.memory.forget_repository("/work/parser")
        self.restart()
        self.store.observe_repository({"worktree": "", "branch": "", "repository": "github.com/team/project", "local": None,
                                       "pr": self.pr(checks=[["tests", "FAILURE"]]), "github_checked_at": self.now})
        self.assertEqual(self.kinds(), {"resolve_checks"})
        self.assertEqual(len(self.store.session_context("codex:one")["artifacts"]), 2)

    def test_fork_checkout_and_independent_pr_poll_share_the_same_pr_identity(self):
        self.start()
        fork_pr = self.pr(head_repository="github.com/fork/project")
        observation = self.observation(fork_pr, repository="github.com/fork/project")
        observation["local"]["repository"] = "github.com/fork/project"
        self.store.observe_repository(observation)
        revision = self.store.session_context("codex:one")["revision"]
        self.store.observe_repository({"repository": "github.com/team/project", "worktree": "", "branch": "", "local": None,
                                       "pr": fork_pr, "github_checked_at": self.now})
        self.assertEqual(self.store.session_context("codex:one")["revision"], revision)

    def test_prune_preserves_connected_graph_and_current_source_evidence(self):
        self.start()
        self.progress(value="First parser changes")
        self.progress(value="Validated quoted strings")
        self.store.observe_repository(self.observation(self.pr(review_requested=True)))
        before = self.store.session_context("codex:one")
        self.now += 31 * 86400
        self.store.prune()
        self.restart()
        self.assertEqual(self.store.session_context("codex:one"), before)
        rows = self.store._rows("SELECT * FROM work_evidence WHERE pet=? AND kind='progress'")
        self.assertEqual(len(rows), 1)
        self.assertEqual(self.kinds(), {"review_output"})

    def test_invalid_related_artifact_rolls_back_the_source_observation(self):
        self.start()
        with self.assertRaises(ValueError):
            self.store.observe_repository(self.observation(self.pr(related_links=[{"kind": "pull_request", "ref": "invalid", "relationship": "stack_entry", "source": "github_stack"}])))
        self.assertEqual(self.store.session_context("codex:one")["artifacts"], ())
        self.assertEqual(self.store.tracked_artifacts(), ())

    def test_runtime_subjects_include_connected_context_and_meaningful_runtime_revision(self):
        self.start()
        self.store.observe_repository(self.observation(self.pr(review_requested=True)))
        session = {"id": "codex:one", "generation": 1, "turn_active": True, "active_tools": [{"tool": "pytest", "activity": "testing", "duration_seconds": 1}]}
        work = build_work({"sessions": [session]}, self.store, now=self.now)
        subject = next(item for item in work if item.kind == "session")
        self.assertEqual(subject.payload["intent"], "Improve parser")
        self.assertTrue(next(item for item in work if item.kind == "opportunity").payload["connected_work"])
        session["active_tools"][0]["duration_seconds"] = 99
        self.assertEqual(next(item for item in build_work({"sessions": [session]}, self.store) if item.kind == "session").revision, subject.revision)
        session["waiting_for_input"] = True
        self.assertNotEqual(next(item for item in build_work({"sessions": [session]}, self.store) if item.kind == "session").revision, subject.revision)

    def test_graph_exposes_no_manual_report_or_decision_api(self):
        self.assertFalse(hasattr(self.store, "record_report"))
        self.assertIsNone(self.memory.db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='work_decisions'").fetchone())
