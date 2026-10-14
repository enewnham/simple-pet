import unittest

from pet_daemon.episodes import Episodes
from pet_daemon.sessions import Sessions


class EpisodeTests(unittest.TestCase):
    def setUp(self):
        self.now = 1000
        self.sessions = Sessions(lambda: self.now)
        self.episodes = Episodes()

    def event(self, event, session="one", **extra):
        packet = {"event": event, "session": session, "harness": "claude", **extra}
        result = self.sessions.handle(packet)
        return self.episodes.handle(packet, result, self.sessions) if result else []

    def fail(self, call, **extra):
        self.event("tool_start", call=call, tool="pytest", **extra)
        return self.event("tool_error", call=call, **extra)

    def test_promotes_third_failure_once_and_rearms_after_recovery(self):
        self.event("intent", goal="Fix checks")
        kinds = [episode.kind for index in range(5) for episode in self.fail(str(index))]
        self.assertEqual(kinds.count("trouble"), 1)
        self.event("tool_start", call="retry", tool="pytest")
        self.assertEqual([item.kind for item in self.event("tool_done", call="retry")], ["recovery"])
        kinds = [episode.kind for index in range(3) for episode in self.fail(f"again-{index}")]
        self.assertEqual(kinds.count("trouble"), 1)

    def test_failed_turn_and_cancellation_have_distinct_reactions(self):
        self.event("intent", goal="Fix checks")
        self.assertEqual([item.kind for item in self.event("turn_error")], ["failure"])
        self.event("intent", goal="Try a different task")
        self.assertEqual([item.kind for item in self.event("interrupt")], ["cancelled"])

    def test_recovered_trouble_stays_invalid_after_another_failure_and_alias_merge(self):
        self.event("intent", session="temporary", goal="Fix checks")
        episodes = [item for index in range(3) for item in self.fail(str(index), session="temporary")]
        trouble = next(item for item in episodes if item.kind == "trouble")
        trigger = {"target_session": trouble.session, "generation": trouble.generation,
                   "episode": trouble.kind, "serial": trouble.serial, "activity": trouble.activity}
        self.assertTrue(self.episodes.relevant(trigger, self.sessions))
        self.event("tool_start", session="temporary", call="retry", tool="pytest")
        self.event("tool_done", session="temporary", call="retry")
        self.fail("another", session="temporary")
        self.event("session_start", session="native", aliases=["temporary"])
        self.assertFalse(self.episodes.relevant(trigger, self.sessions))

    def test_another_recovery_supersedes_the_first_in_the_same_turn(self):
        self.event("intent", goal="Fix checks")
        self.fail("bad")
        self.event("tool_start", call="retry", tool="pytest")
        recovery = self.event("tool_done", call="retry")[0]
        trigger = {"target_session": recovery.session, "generation": recovery.generation,
                   "episode": recovery.kind, "serial": recovery.serial, "activity": recovery.activity}
        self.assertTrue(self.episodes.relevant(trigger, self.sessions))
        self.fail("another")
        self.event("tool_start", call="retry-again", tool="pytest")
        self.event("tool_done", call="retry-again")
        self.assertFalse(self.episodes.relevant(trigger, self.sessions))

    def test_waiting_episode_is_replaced_after_input_resumes(self):
        self.event("intent", goal="Choose an approach")
        waiting = self.event("waiting", call="read")[0]
        self.assertEqual(self.event("waiting", call="read"), [])
        trigger = {"target_session": waiting.session, "generation": waiting.generation,
                   "episode": waiting.kind, "serial": waiting.serial, "activity": waiting.activity}
        self.assertTrue(self.episodes.relevant(trigger, self.sessions))
        self.event("tool_start", call="read", tool="Read")
        self.event("waiting")
        self.assertFalse(self.episodes.relevant(trigger, self.sessions))

    def test_alias_merge_preserves_promoted_failure(self):
        self.event("intent", session="temporary", goal="Fix checks")
        promoted = None
        for index in range(3):
            for episode in self.fail(str(index), session="temporary"):
                if episode.kind == "trouble":
                    promoted = episode
        self.event("session_start", session="native", aliases=["temporary"])
        self.assertNotIn("trouble", [item.kind for item in self.fail("fourth", session="native")])
        trigger = {"target_session": promoted.session, "generation": promoted.generation,
                   "episode": promoted.kind, "serial": promoted.serial, "activity": promoted.activity}
        self.assertTrue(self.episodes.relevant(trigger, self.sessions))

    def test_new_turn_supersedes_outcome_while_other_session_continues(self):
        self.event("intent", goal="Fix checks")
        self.event("intent", session="two", goal="Inspect code")
        outcome = self.event("turn_done")[0]
        trigger = {"target_session": outcome.session, "generation": outcome.generation,
                   "episode": outcome.kind, "serial": outcome.serial, "activity": outcome.activity}
        self.assertTrue(self.sessions.active)
        self.assertTrue(self.episodes.relevant(trigger, self.sessions))
        self.event("intent", goal="Next task")
        self.assertFalse(self.episodes.relevant(trigger, self.sessions))

    def test_new_turn_discards_old_episode_state_before_an_alias_merge(self):
        self.event("intent", session="temporary", goal="First task")
        for index in range(3):
            self.fail(str(index), session="temporary")
        self.event("intent", session="temporary", goal="Second task")
        self.event("session_start", session="native", aliases=["temporary"])
        episodes = [item.kind for index in range(3) for item in self.fail(f"new-{index}", session="native")]
        self.assertEqual(episodes.count("trouble"), 1)
        self.assertEqual(len(self.episodes.states), 1)

    def test_replacing_a_question_starts_a_new_waiting_episode(self):
        self.event("intent", goal="Choose an approach")
        first = self.event("waiting", call="first")[0]
        second = self.event("waiting", call="second")[0]
        self.assertEqual(second.kind, "waiting")
        self.assertNotEqual(first.serial, second.serial)
        self.assertEqual(self.event("waiting", call="second"), [])
        trigger = {"target_session": first.session, "generation": first.generation,
                   "episode": first.kind, "serial": first.serial, "activity": first.activity}
        self.assertFalse(self.episodes.relevant(trigger, self.sessions))

    def test_merging_active_aliases_preserves_waiting_without_another_reminder(self):
        self.event("intent", session="native", goal="Choose a strategy")
        self.event("intent", session="temporary", goal="Choose a strategy")
        waiting = self.event("waiting", session="temporary", call="question")[0]
        self.assertEqual(self.event("session_start", session="native", aliases=["temporary"]), [])
        lane = self.sessions.lanes["claude:native"]
        self.assertTrue(lane.waiting)
        self.assertEqual(lane.waiting_call, "question")
        trigger = {"target_session": waiting.session, "generation": waiting.generation,
                   "episode": waiting.kind, "serial": waiting.serial, "activity": waiting.activity}
        self.assertTrue(self.episodes.relevant(trigger, self.sessions))
        self.assertEqual(self.event("waiting", session="native", call="question"), [])
