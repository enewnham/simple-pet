from pathlib import Path
import tempfile
import unittest
import uuid

from pet_daemon.storage import Memory
from pet_daemon.work import build_work
from pet_daemon.work.views import project_evidence
from pet_daemon.work.presentation import work_record
from pet_daemon.work.observations import repository_observation_from_mapping


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

    def observe(self, observation):
        observation = {"subject": "fixture", "worktree": "", "branch": "", **observation}
        if observation.get("local"):
            observation["local"] = {"worktree": observation["worktree"], "upstream": "", **observation["local"]}
        if observation.get("pr"):
            observation["pr"] = {"merged_at": None, "review": "", **observation["pr"]}
        self.store.observe_repository(repository_observation_from_mapping(observation))

    def kinds(self):
        return {item.kind for item in self.store.opportunities()}

    def restart(self):
        self.memory.close()
        self.memory = Memory(self.path, "fixture", clock=lambda: self.now)
        self.store = self.memory.work

    def test_native_intents_and_progress_form_an_attributed_connected_account(self):
        self.start()
        self.progress()
        self.observe(self.observation(self.pr()))
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
        self.observe(self.observation(self.pr(), sessions=["codex:one", "claude:two"]))
        self.assertNotEqual(first, second)
        self.assertEqual(self.store.session_context("codex:one")["intent"], "Cover escape sequences")
        self.assertEqual(self.store.session_context("claude:two")["intent"], "Review the parser API")
        self.assertEqual(len(self.store.tracked_artifacts()), 2)

    def test_shared_artifact_observations_update_each_connected_work(self):
        self.start()
        self.start("claude:two", "Review the parser API")
        self.observe(self.observation(self.pr(), sessions=["codex:one", "claude:two"]))
        self.observe(self.observation(self.pr(state="MERGED"), sessions=[]))
        for session in ("codex:one", "claude:two"):
            account = self.store.session_context(session)
            pull_request = next(item for item in account["artifacts"] if item["kind"] == "pull_request")
            self.assertEqual(next(item["data"]["pr"]["state"] for item in pull_request["evidence"] if item["kind"] == "observation"), "MERGED")
            self.assertEqual(self.store.work_for_session(session).status, "active")

    def test_successful_stop_records_execution_without_completing_work(self):
        work = self.start()
        self.store.session_event("codex:one", {"event": "turn_done", "outcome": "success"})
        self.assertEqual(self.store.work_for_session("codex:one").status, "active")
        self.assertEqual(self.store.session_intent("codex:one", "Check compatibility"), work)
        self.assertEqual(self.kinds(), set())

    def test_repository_observation_cannot_create_a_user_request(self):
        self.observe(self.observation(self.pr()))
        self.assertEqual(self.store.session_context("codex:one"), {})
        self.assertEqual(self.kinds(), set())

    def test_verified_current_user_review_request_creates_an_opportunity(self):
        self.start()
        self.observe(self.observation(self.pr(review_requested=True)))
        self.assertEqual(self.kinds(), {"review_output"})
        self.assertEqual(self.store.opportunities()[0].artifact["ref"], self.pr()["url"])
        self.observe(self.observation(self.pr(review_requested=None)))
        self.assertEqual(self.kinds(), set())

    def test_only_checks_for_current_pr_head_create_check_opportunities(self):
        self.start()
        self.observe(self.observation(self.pr(checks=[["tests", "FAILURE"]], check_revision="older")))
        self.assertEqual(self.kinds(), set())
        self.observe(self.observation(self.pr(checks=[["tests", "FAILURE", "github", "older"]])))
        self.assertEqual(self.kinds(), set())
        self.observe(self.observation(self.pr(checks=[["tests", "FAILURE", "github", "head-one"]])))
        self.assertEqual(self.kinds(), {"resolve_checks"})
        self.observe(self.observation(self.pr(checks=[["tests", "SUCCESS"]])))
        self.assertEqual(self.kinds(), set())

    def test_draft_and_green_checks_alone_do_not_claim_readiness(self):
        self.start()
        self.progress()
        self.observe(self.observation(self.pr(checks=[["tests", "SUCCESS"]])))
        self.assertEqual(self.kinds(), set())
        self.assertEqual(self.store.work_for_session("codex:one").status, "active")

    def test_pr_preparation_requires_observed_output_progress_and_successful_absence_lookup(self):
        self.start()
        self.observe(self.observation(github_checked_at=0))
        self.assertEqual(self.kinds(), set())
        self.progress()
        self.assertEqual(self.kinds(), set())
        self.observe(self.observation())
        self.assertEqual(self.kinds(), {"prepare_pr"})
        opportunity = self.store.opportunities()[0]
        self.assertIn("may benefit", opportunity.next_step)
        self.assertIn("Agent progress does not establish that the work is complete.", opportunity.uncertainty)
        clean = self.observation()
        clean["local"]["unstaged"] = 0
        self.observe(clean)
        self.assertEqual(self.kinds(), set())

    def test_absence_lookup_without_agent_progress_does_not_create_preparation(self):
        self.start()
        self.observe(self.observation())
        self.assertEqual(self.kinds(), set())

    def test_failed_absence_refresh_preserves_evidence_and_suppresses_pr_preparation(self):
        self.start()
        self.progress()
        self.observe(self.observation())
        self.assertEqual(self.kinds(), {"prepare_pr"})
        observed_at = self.now
        self.now += 100
        self.observe(self.observation(github_error="GitHub is unavailable", github_checked_at=observed_at))
        self.assertEqual(self.kinds(), set())
        branch = self.store.session_context("codex:one")["artifacts"][0]
        github = next(record for record in branch["evidence"] if record["source"] == "github")
        self.assertIsNone(github["data"]["pr"])
        self.assertEqual(github["observed_at"], observed_at)
        self.observe(self.observation())
        self.assertEqual(self.kinds(), {"prepare_pr"})

    def test_review_and_check_opportunities_disclose_failed_source_refresh(self):
        self.start()
        pull_request = self.pr(review_requested=True, checks=[["tests", "FAILURE"]])
        self.observe(self.observation(pull_request))
        observed_at = self.now
        self.now += 100
        self.observe(self.observation(pull_request, github_error="GitHub is unavailable", local_error="Checkout is unavailable"))
        self.assertEqual(self.kinds(), {"review_output", "resolve_checks"})
        for opportunity in self.store.opportunities():
            self.assertIn("GitHub refresh failed. Status comes from the last successful observation.", opportunity.uncertainty)
            self.assertIn("Local repository refresh failed. Output information comes from the last successful observation.", opportunity.uncertainty)
            self.assertEqual(next(record["observed_at"] for record in opportunity.evidence if record["source"] == "github" and record["kind"] == "observation"), observed_at)

    def test_inconsistent_independent_pr_branch_field_cannot_attach_unrelated_work(self):
        self.start()
        self.observe(self.observation())
        self.observe({"repository": "github.com/team/project", "worktree": "", "branch": "feature", "local": None,
                                       "pr": self.pr(head_branch="another", review_requested=True), "github_checked_at": self.now})
        artifacts = self.store.session_context("codex:one")["artifacts"]
        self.assertEqual([item["kind"] for item in artifacts], ["branch"])
        self.assertEqual(self.kinds(), set())

    def test_stack_links_connect_individual_pr_states_and_exact_url_followups(self):
        self.start()
        link = {"kind": "pull_request", "ref": self.pr(8)["url"], "relationship": "stack_entry", "source": "github_stack",
                "stack": 4, "position": 1, "state": "OPEN", "is_draft": True, "head": "head-two", "head_branch": "next", "title": "Expand parser coverage"}
        self.observe(self.observation(self.pr(related_links=[link])))
        self.assertEqual(len(self.store.tracked_artifacts()), 3)
        self.assertEqual(next(item["title"] for item in self.store.tracked_artifacts() if item["ref"] == self.pr(8)["url"]), "Expand parser coverage")
        self.observe({"repository": "github.com/team/project", "worktree": "", "branch": "", "sessions": [], "local": None,
                                       "pr": self.pr(8, state="MERGED", head="head-two"), "github_checked_at": self.now})
        account = self.store.session_context("codex:one")
        states = {item["ref"]: next(record["data"]["pr"]["state"] for record in item["evidence"] if record["kind"] == "observation")
                  for item in account["artifacts"] if item["kind"] == "pull_request"}
        self.assertEqual(states, {self.pr()["url"]: "OPEN", self.pr(8)["url"]: "MERGED"})
        self.assertEqual(self.store.work_for_session("codex:one").status, "active")

    def test_stack_peer_check_failure_remains_independent_of_merged_pr(self):
        self.start()
        link = {"kind": "pull_request", "ref": self.pr(8)["url"], "relationship": "stack_entry", "source": "github_stack"}
        self.observe(self.observation(self.pr(state="MERGED", related_links=[link])))
        self.observe({"repository": "github.com/team/project", "worktree": "", "branch": "", "local": None,
                                       "pr": self.pr(8, checks=[["tests", "FAILURE"]]), "github_checked_at": self.now})
        self.assertEqual(self.kinds(), {"resolve_checks"})
        self.assertEqual(self.store.opportunities()[0].artifact["ref"], self.pr(8)["url"])

    def test_sibling_state_change_advances_the_primary_opportunity_version(self):
        self.start()
        link = {"kind": "pull_request", "ref": self.pr(8)["url"], "relationship": "stack_entry", "source": "github_stack"}
        self.observe(self.observation(self.pr(review_requested=True, related_links=[link])))
        original = self.store.opportunities()[0]
        self.assertTrue(self.store.is_current(original.id, original.version))
        self.observe({"repository": "github.com/team/project", "worktree": "", "branch": "", "local": None,
                                       "pr": self.pr(8, state="MERGED"), "github_checked_at": self.now})
        current = self.store.opportunities()[0]
        self.assertEqual(current.id, original.id)
        self.assertGreater(current.version, original.version)
        self.assertFalse(self.store.is_current(original.id, original.version))

    def test_tool_event_names_do_not_rearm_an_unchanged_opportunity(self):
        self.start()
        self.store.session_event("codex:one", {"event": "tool_start", "tool_name": "Read"})
        self.observe(self.observation(self.pr(review_requested=True)))
        original = self.store.opportunities()[0]
        self.memory.record_acceptance(uuid.uuid4().hex, "Review the parser output.", "opportunity", original.id, original.version)
        self.store.session_event("codex:one", {"event": "tool_done", "tool_name": "Write", "outcome": "success"})
        self.assertEqual(self.kinds(), set())
        stored = self.memory.db.execute("SELECT version FROM work_opportunities WHERE pet=? AND id=?", ("fixture", original.id)).fetchone()
        self.assertEqual(stored["version"], original.version)
        self.store.session_event("codex:one", {"event": "waiting"})
        self.assertEqual(self.kinds(), {"review_output"})
        self.assertGreater(self.store.opportunities()[0].version, original.version)

    def test_native_aliases_share_session_context_and_survive_restart(self):
        work = self.store.session_intent("codex:one", "Improve parser", aliases=["codex:alternate"])
        self.store.session_event("codex:one", {"event": "progress", "harness": "codex", "aliases": ["third"], "progress": "Parser changes"})
        self.restart()
        self.assertEqual(self.store.work_for_session("codex:third").id, work)
        self.assertEqual(self.store.session_context("codex:alternate")["intent"], "Improve parser")
        self.assertIsNone(self.store.work_for_session("third"))

    def test_poll_times_and_nonzero_git_count_changes_do_not_change_meaningful_revision(self):
        self.start()
        self.progress()
        self.observe(self.observation())
        account = self.store.session_context("codex:one")
        opportunity = self.store.opportunities()[0]
        self.now += 100
        observed = self.observation()
        observed["local"]["unstaged"] = 9
        self.observe(observed)
        self.assertEqual(self.store.session_context("codex:one")["revision"], account["revision"])
        current = self.store.opportunities()[0]
        self.assertEqual((current.id, current.version, current.changed_at), (opportunity.id, opportunity.version, opportunity.changed_at))
        self.assertEqual(next(item["observed_at"] for item in current.evidence if item["source"] == "git"), self.now)

    def test_optional_stack_lookup_errors_do_not_rearm_an_unchanged_suggestion(self):
        self.start()
        self.observe(self.observation(self.pr(review_requested=True, related_links_error="Stack lookup unavailable")))
        opportunity = self.store.opportunities()[0]
        account = self.store.session_context("codex:one")
        self.memory.record_acceptance(uuid.uuid4().hex, "Review the parser output.", "opportunity", opportunity.id, opportunity.version)
        self.now += 100
        self.observe(self.observation(self.pr(review_requested=True, related_links_error="Stack lookup rate limited")))
        self.assertEqual(self.store.session_context("codex:one")["revision"], account["revision"])
        self.assertEqual(self.store.opportunities(), ())

    def test_new_head_changes_context_and_opportunity_revision(self):
        self.start()
        self.progress()
        self.observe(self.observation())
        account = self.store.session_context("codex:one")
        opportunity = self.store.opportunities()[0]
        observed = self.observation()
        observed["local"]["head"] = "head-two"
        self.observe(observed)
        self.assertNotEqual(self.store.session_context("codex:one")["revision"], account["revision"])
        self.assertFalse(self.store.is_current(opportunity.id, opportunity.version))

    def test_only_accepted_current_delivery_marks_a_version_surfaced(self):
        self.start()
        self.observe(self.observation(self.pr(review_requested=True)))
        opportunity = self.store.opportunities()[0]
        self.memory.record_delivery_failure(uuid.uuid4().hex, "rejected", opportunity_id=opportunity.id, version=opportunity.version)
        self.memory.record_acceptance(uuid.uuid4().hex, "Review the parser output.", "opportunity", opportunity.id, opportunity.version + 1)
        self.assertTrue(self.store.is_current(opportunity.id, opportunity.version))
        self.memory.record_acceptance(uuid.uuid4().hex, "Review the parser output.", "opportunity", opportunity.id, opportunity.version)
        self.restart()
        self.assertEqual(self.kinds(), set())
        self.observe(self.observation(self.pr(review_requested=True)))
        self.assertEqual(self.kinds(), set())

    def test_source_timestamps_survive_cached_observation_replay(self):
        self.start()
        self.now += 10000
        self.observe(self.observation(self.pr(), local_checked_at=500, github_checked_at=600))
        artifacts = self.store.session_context("codex:one")["artifacts"]
        branch = next(item for item in artifacts if item["kind"] == "branch")
        times = {item["source"]: item["observed_at"] for item in branch["evidence"] if item["kind"] == "observation"}
        self.assertEqual(times, {"git": 500, "github": 600})

    def test_delayed_open_failure_cannot_replace_a_newer_merged_observation(self):
        self.start()
        self.observe(self.observation(self.pr(checks=[["tests", "FAILURE"]]), github_checked_at=500))
        self.assertEqual(self.kinds(), {"resolve_checks"})
        merged = self.pr(state="MERGED", head="head-two", title="Verified parser changes")
        self.observe({"repository": "github.com/team/project", "worktree": "", "branch": "", "local": None,
                                       "pr": merged, "github_checked_at": 700})
        self.assertEqual(self.kinds(), set())
        self.observe(self.observation(self.pr(checks=[["tests", "FAILURE"]]), github_checked_at=600))
        self.assertEqual(self.kinds(), set())
        pull_request = next(item for item in self.store.session_context("codex:one")["artifacts"] if item["kind"] == "pull_request")
        self.assertEqual((pull_request["revision"], pull_request["title"]), ("head-two", "Verified parser changes"))
        github = next(record for record in pull_request["evidence"] if record["source"] == "github" and record["kind"] == "observation")
        self.assertEqual((github["data"]["pr"]["state"], github["observed_at"]), ("MERGED", 700))

    def test_cached_git_observation_cannot_roll_back_a_newer_head_or_verification_time(self):
        self.start()
        newer = self.observation(local_checked_at=700)
        newer["local"]["head"] = "head-two"
        self.observe(newer)
        self.observe(self.observation(local_checked_at=500))
        branch = next(item for item in self.store.session_context("codex:one")["artifacts"] if item["kind"] == "branch")
        self.assertEqual(branch["revision"], "head-two")
        git = next(record for record in branch["evidence"] if record["source"] == "git")
        self.assertEqual((git["data"]["head"], git["observed_at"]), ("head-two", 700))

    def test_worktree_removal_preserves_context_and_independent_pr_updates(self):
        self.start()
        self.observe(self.observation(self.pr()))
        self.memory.forget_repository("/work/parser")
        self.restart()
        self.observe({"worktree": "", "branch": "", "repository": "github.com/team/project", "local": None,
                                       "pr": self.pr(checks=[["tests", "FAILURE"]]), "github_checked_at": self.now})
        self.assertEqual(self.kinds(), {"resolve_checks"})
        self.assertEqual(len(self.store.session_context("codex:one")["artifacts"]), 2)

    def test_fork_checkout_and_independent_pr_poll_share_the_same_pr_identity(self):
        self.start()
        fork_pr = self.pr(head_repository="github.com/fork/project")
        observation = self.observation(fork_pr, repository="github.com/fork/project")
        observation["local"]["repository"] = "github.com/fork/project"
        self.observe(observation)
        revision = self.store.session_context("codex:one")["revision"]
        self.observe({"repository": "github.com/team/project", "worktree": "", "branch": "", "local": None,
                                       "pr": fork_pr, "github_checked_at": self.now})
        self.assertEqual(self.store.session_context("codex:one")["revision"], revision)

    def test_prune_preserves_connected_graph_and_current_source_evidence(self):
        self.start()
        self.progress(value="First parser changes")
        self.progress(value="Validated quoted strings")
        self.observe(self.observation(self.pr(review_requested=True)))
        before = self.store.session_context("codex:one")
        self.now += 31 * 86400
        self.store.prune()
        self.restart()
        self.assertEqual(self.store.session_context("codex:one"), before)
        rows = self.memory.db.execute("SELECT * FROM work_evidence WHERE pet=? AND kind='progress'", ("fixture",)).fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual(self.kinds(), {"review_output"})

    def test_invalid_related_artifact_rolls_back_the_source_observation(self):
        self.start()
        with self.assertRaises(ValueError):
            self.observe(self.observation(self.pr(related_links=[{"kind": "pull_request", "ref": "invalid", "relationship": "stack_entry", "source": "github_stack"}])))
        self.assertEqual(self.store.session_context("codex:one")["artifacts"], ())
        self.assertEqual(self.store.tracked_artifacts(), ())

    def test_runtime_subjects_include_connected_context_and_meaningful_runtime_revision(self):
        self.start()
        self.observe(self.observation(self.pr(review_requested=True)))
        session = {"id": "codex:one", "generation": 1, "turn_active": True, "active_tools": [{"tool": "pytest", "activity": "testing", "duration_seconds": 1}]}
        work = self.store.views({"sessions": [session]}, now=self.now)
        subject = next(item for item in work if item.kind == "work")
        self.assertEqual(subject.view.intent, "Improve parser")
        self.assertEqual(subject.view.opportunities[0].kind, "review_output")
        session["active_tools"][0]["duration_seconds"] = 99
        self.assertEqual(next(item for item in self.store.views({"sessions": [session]}) if item.kind == "work").revision, subject.revision)
        session["waiting_for_input"] = True
        self.assertNotEqual(next(item for item in self.store.views({"sessions": [session]}) if item.kind == "work").revision, subject.revision)

    def test_graph_exposes_no_manual_report_or_decision_api(self):
        self.assertFalse(hasattr(self.store, "record_report"))
        self.assertIsNone(self.memory.db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='work_decisions'").fetchone())

    def test_acceptance_records_the_captured_version_once_after_work_advances(self):
        self.start()
        self.observe(self.observation(self.pr(review_requested=True)))
        captured = self.store.opportunities()[0]
        self.observe(self.observation(self.pr(review_requested=True, head="head-two")))
        current = self.store.opportunities()[0]
        message_id = uuid.uuid4().hex
        self.assertTrue(self.memory.record_delivery_failure(message_id, "acknowledgement timed out"))
        self.assertTrue(self.memory.record_acceptance(message_id, "The parser output needs review.", "opportunity", captured.id, captured.version))
        self.assertFalse(self.memory.record_acceptance(message_id, "Repeated acknowledgement", "opportunity", captured.id, captured.version))
        self.assertEqual(self.store.opportunities()[0].version, current.version)
        self.assertEqual(self.memory.speech()[0]["text"], "The parser output needs review.")
        self.assertEqual(self.memory.message_outcome(message_id)["outcome"], "accepted")
        rows = self.memory.db.execute("SELECT version,outcome FROM work_surfacings WHERE pet=?", ("fixture",)).fetchall()
        self.assertEqual([dict(row) for row in rows], [{"version": captured.version, "outcome": "accepted"}])
        self.restart()
        self.assertFalse(self.memory.record_acceptance(message_id, "Repeated after restart", "opportunity", captured.id, captured.version))
        self.assertTrue(self.store.is_current(current.id, current.version))

    def test_acceptance_rolls_back_speech_outcome_and_surfacing_together(self):
        self.start()
        self.observe(self.observation(self.pr(review_requested=True)))
        opportunity = self.store.opportunities()[0]
        message_id = uuid.uuid4().hex
        with self.assertRaisesRegex(RuntimeError, "rollback"):
            with self.memory.transaction():
                self.memory.record_acceptance(message_id, "Review the parser.", "opportunity", opportunity.id, opportunity.version)
                raise RuntimeError("rollback")
        self.assertEqual(self.memory.speech(), [])
        self.assertIsNone(self.memory.message_outcome(message_id))
        self.assertTrue(self.store.is_current(opportunity.id, opportunity.version))
        self.assertEqual(self.memory.db.execute("SELECT COUNT(*) FROM work_surfacings WHERE pet=?", ("fixture",)).fetchone()[0], 0)

    def test_work_views_group_aliases_and_preserve_distinct_shared_artifact_intents(self):
        first = self.store.session_intent("codex:one", "Improve parser", aliases=("codex:alias",))
        second = self.start("claude:two", "Review the parser API")
        self.observe(self.observation(self.pr(), sessions=["codex:one", "claude:two"]))
        sessions = [{"id": name, "turn_active": True} for name in ("codex:one", "codex:alias", "claude:two", "codex:unbound")]
        view = self.store.views({"sessions": sessions}, now=self.now)
        self.assertEqual({subject.id for subject in view}, {first, second, "codex:unbound"})
        self.assertEqual(next(subject for subject in view if subject.id == first).sessions, ("codex:alias", "codex:one"))
        self.assertEqual(next(subject for subject in view if subject.id == second).view.intent, "Review the parser API")
        self.assertEqual(next(subject for subject in view if subject.id == "codex:unbound").kind, "session")

    def test_tool_failure_does_not_establish_a_work_obstacle(self):
        self.start()
        self.store.session_event("codex:one", {"event": "tool_error", "outcome": "error", "tool_name": "Bash"})
        subject = self.store.views({"sessions": [{"id": "codex:one", "failure": {"current_turn": True}}]})[0]
        self.assertFalse(subject.view.current_obstacle)
        self.assertEqual(subject.view.opportunities, ())

    def test_collection_expands_primary_stack_links_without_recursing_through_peers(self):
        self.start()
        related = {"kind": "pull_request", "ref": self.pr(8)["url"], "relationship": "stack_entry", "source": "github_stack"}
        self.observe(self.observation(self.pr(related_links=[related])))
        targets = {target.ref: target for target in self.store.collection_targets()}
        self.assertTrue(targets[self.pr()["url"]].include_links)
        self.assertFalse(targets[self.pr(8)["url"]].include_links)
        self.assertEqual(targets["feature"].checkout, "/work/parser")
        self.start("claude:two", "Review the follow-up")
        self.observe(self.observation(self.pr(8), sessions=["claude:two"]))
        targets = {target.ref: target for target in self.store.collection_targets()}
        self.assertTrue(targets[self.pr(8)["url"]].include_links)

    def test_retired_checkout_keeps_remote_work_without_resuming_local_collection(self):
        self.start()
        self.progress()
        self.observe(self.observation())
        self.assertEqual(self.kinds(), {"prepare_pr"})
        self.observe(self.observation(current=False))
        self.assertEqual(self.kinds(), set())
        branch = next(target for target in self.store.collection_targets() if target.kind == "branch")
        self.assertEqual(branch.checkout, "")
        self.assertEqual(branch.ref, "feature")
        self.assertTrue(self.store.session_context("codex:one")["artifacts"][0]["evidence"])
        self.restart()
        self.assertEqual(next(target for target in self.store.collection_targets() if target.kind == "branch").checkout, "")

    def test_accepted_action_preserves_work_meaning_until_observed_facts_change(self):
        self.start()
        self.observe(self.observation(self.pr(review_requested=True)))
        runtime = {"sessions": [{"id": "codex:one", "turn_active": True, "generation": 1}]}
        before = self.store.views(runtime)[0]
        action = before.view.opportunities[0]
        self.memory.record_acceptance(uuid.uuid4().hex, "The parser output needs review.", "opportunity", action.id, action.version)
        after = self.store.views(runtime)[0]
        self.assertEqual(after.view.opportunities, ())
        self.assertEqual(after.revision, before.revision)
        self.restart()
        self.assertEqual(self.store.views(runtime)[0].revision, before.revision)
        self.observe(self.observation(self.pr(review_requested=False)))
        self.assertNotEqual(self.store.views(runtime)[0].revision, before.revision)

    def test_model_work_projection_keeps_meaning_without_checkout_paths_or_internal_ids(self):
        self.start()
        self.progress()
        self.observe(self.observation(self.pr()))
        view = self.store.views({"sessions": []})[0].view
        rendered = str(work_record(view))
        self.assertIn("Improve parser", rendered)
        self.assertIn("Implemented quoted string handling", rendered)
        self.assertIn("The agent session used this checkout.", rendered)
        self.assertIn(self.pr()["url"], rendered)
        self.assertNotIn("/work/parser", rendered)
        self.assertNotIn("codex:one", rendered)
        self.assertNotIn("head-one", rendered)
        self.assertIsNone(project_evidence({"source": "unknown", "kind": "observation", "data": {"secret": "private value"}}))

    def test_pure_work_builder_uses_typed_records_without_persistence(self):
        self.start()
        runtime = {"sessions": [{"id": "codex:one", "turn_active": True}]}
        state = self.store.snapshot()
        self.memory.close()
        self.assertEqual(build_work(runtime, state)[0].view.intent, "Improve parser")

    def test_hook_and_poll_updates_preserve_unaffected_rows_and_accepted_history(self):
        work_id = self.start()
        other_work = self.start("claude:two", "Review the parser API")
        self.observe(self.observation(self.pr(review_requested=True)))
        action = self.store.opportunities()[0]
        self.memory.record_acceptance(uuid.uuid4().hex, "Review the parser output.", "opportunity", action.id, action.version)
        accepted = [dict(row) for row in self.memory.db.execute("SELECT * FROM work_surfacings WHERE pet=?", ("fixture",))]
        unrelated = [dict(row) for row in self.memory.db.execute("SELECT * FROM work_evidence WHERE pet=? AND work_id=?", ("fixture", other_work))]
        with self.memory.transaction():
            for table in ("work_items", "work_evidence", "work_surfacings"):
                self.memory.db.execute(f"""CREATE TRIGGER preserve_{table} BEFORE DELETE ON {table} BEGIN
                    SELECT RAISE(ABORT, 'unaffected history deleted'); END""")
        self.progress(value="Validated parser compatibility")
        self.store.session_intent("codex:one", "Cover parser compatibility")
        self.now += 30
        self.observe(self.observation(self.pr(review_requested=True)))
        self.assertEqual(self.store.work_for_session("codex:one").id, work_id)
        self.assertEqual(self.store.work_for_session("codex:one").intent, "Cover parser compatibility")
        self.assertEqual([dict(row) for row in self.memory.db.execute("SELECT * FROM work_surfacings WHERE pet=?", ("fixture",))], accepted)
        self.assertEqual([dict(row) for row in self.memory.db.execute("SELECT * FROM work_evidence WHERE pet=? AND work_id=?", ("fixture", other_work))], unrelated)

    def test_unchanged_graph_save_performs_no_writes(self):
        self.start()
        self.observe(self.observation(self.pr(review_requested=True)))
        action = self.store.opportunities()[0]
        self.memory.record_acceptance(uuid.uuid4().hex, "Review the parser output.", "opportunity", action.id, action.version)
        state = self.store.snapshot()
        changed_before = self.memory.db.total_changes
        with self.memory.transaction():
            self.store.repository.save(state)
        self.assertEqual(self.memory.db.total_changes, changed_before)

    def test_identical_poll_updates_verification_time_without_rewriting_evidence(self):
        self.start()
        self.observe(self.observation(self.pr(review_requested=True)))
        records_before = [dict(row) for row in self.memory.db.execute("SELECT * FROM work_evidence WHERE pet=? ORDER BY id", ("fixture",))]
        with self.memory.transaction():
            self.memory.db.execute("""CREATE TRIGGER preserve_evidence_data BEFORE UPDATE OF data ON work_evidence BEGIN
                SELECT RAISE(ABORT, 'unchanged evidence rewritten'); END""")
        self.now += 30
        self.observe(self.observation(self.pr(review_requested=True)))
        records_after = [dict(row) for row in self.memory.db.execute("SELECT * FROM work_evidence WHERE pet=? ORDER BY id", ("fixture",))]
        self.assertEqual(len(records_after), len(records_before))
        for before, after in zip(records_before, records_after):
            self.assertEqual({key: value for key, value in before.items() if key != "timestamp"},
                             {key: value for key, value in after.items() if key != "timestamp"})
            self.assertEqual(after["timestamp"], self.now if after["kind"] == "observation" else before["timestamp"])
