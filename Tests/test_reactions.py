import unittest
from typing import Unpack

from pet_daemon.domain import SessionSource
from pet_daemon.identity import SessionIdentity
from pet_daemon.runtime import HookPacket, HookTextFields, WaitingKind
from pet_daemon.sessions.reactions import (
    ReactionKind,
    SessionReaction,
    SessionReactions,
)
from pet_daemon.sessions.tracking import Sessions
from pet_daemon.sources.hooks import normalize_hook
from Tests.fixtures import require


class ReactionOptions(HookTextFields, total=False):
    aliases: list[str]


class ReactionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.now = 1000
        self.sessions = Sessions(lambda: self.now)
        self.reactions = SessionReactions()

    def event(
        self, event: str, session: str = "one", **extra: Unpack[ReactionOptions]
    ) -> list[SessionReaction]:
        packet: HookPacket = {
            "event": event,
            "session": session,
            "source": "claude",
            **extra,
        }
        hook = require(normalize_hook(packet))
        result = self.sessions.apply(hook)
        return self.reactions.handle(hook, result, self.sessions) if result else []

    def fail_tool(self, call: str, session: str = "one") -> list[SessionReaction]:
        self.event("tool_start", session=session, call=call, tool="pytest")
        return self.event("tool_error", session=session, call=call)

    def test_promotes_third_failure_once_and_rearms_after_recovery(self) -> None:
        self.event("intent", goal="Fix checks")
        kinds = [
            reaction.kind
            for index in range(5)
            for reaction in self.fail_tool(str(index))
        ]
        self.assertEqual(kinds.count(ReactionKind.TROUBLE), 1)
        self.event("tool_start", call="retry", tool="pytest")
        self.assertEqual(
            [item.kind for item in self.event("tool_done", call="retry")], ["recovery"]
        )
        kinds = [
            reaction.kind
            for index in range(3)
            for reaction in self.fail_tool(f"again-{index}")
        ]
        self.assertEqual(kinds.count(ReactionKind.TROUBLE), 1)

    def test_session_closure_does_not_create_a_terminal_reaction(self) -> None:
        self.event("intent", goal="Build it")
        self.assertEqual(self.event("session_end"), [])
        self.assertEqual(self.event("session_end"), [])
        self.assertIsNone(
            self.sessions.by_id[SessionIdentity(SessionSource.CLAUDE, "one")].outcome
        )

    def test_closure_keeps_the_existing_terminal_reaction_relevant(self) -> None:
        self.event("intent", goal="Build it")
        terminal = self.event("turn_done")[0]
        self.assertEqual(self.event("session_end"), [])
        self.assertTrue(self.reactions.relevant(terminal, self.sessions))

    def test_failed_turn_and_cancellation_have_distinct_reactions(self) -> None:
        self.event("intent", goal="Fix checks")
        self.assertEqual([item.kind for item in self.event("turn_error")], ["failure"])
        self.event("intent", goal="Try a different task")
        self.assertEqual([item.kind for item in self.event("interrupt")], ["cancelled"])

    def test_recovered_trouble_stays_invalid_after_another_failure_and_alias_merge(
        self,
    ) -> None:
        self.event("intent", session="temporary", goal="Fix checks")
        reactions = [
            item
            for index in range(3)
            for item in self.fail_tool(str(index), session="temporary")
        ]
        trouble = next(item for item in reactions if item.kind == "trouble")
        captured = trouble
        self.assertTrue(self.reactions.relevant(captured, self.sessions))
        self.event("tool_start", session="temporary", call="retry", tool="pytest")
        self.event("tool_done", session="temporary", call="retry")
        self.fail_tool("another", session="temporary")
        self.event("session_start", session="native", aliases=["temporary"])
        self.assertFalse(self.reactions.relevant(captured, self.sessions))

    def test_another_recovery_supersedes_the_first_in_the_same_turn(self) -> None:
        self.event("intent", goal="Fix checks")
        self.fail_tool("bad")
        self.event("tool_start", call="retry", tool="pytest")
        recovery = self.event("tool_done", call="retry")[0]
        captured = recovery
        self.assertTrue(self.reactions.relevant(captured, self.sessions))
        self.fail_tool("another")
        self.event("tool_start", call="retry-again", tool="pytest")
        self.event("tool_done", call="retry-again")
        self.assertFalse(self.reactions.relevant(captured, self.sessions))

    def test_waiting_reaction_is_replaced_after_input_resumes(self) -> None:
        self.event("intent", goal="Choose an approach")
        waiting = self.event("waiting", call="read")[0]
        self.assertEqual(self.event("waiting", call="read"), [])
        captured = waiting
        self.assertTrue(self.reactions.relevant(captured, self.sessions))
        self.event("tool_start", call="read", tool="Read")
        self.event("waiting")
        self.assertFalse(self.reactions.relevant(captured, self.sessions))

    def test_wait_kind_changes_restart_timing_and_replace_the_reaction(self) -> None:
        self.event("intent", goal="Inspect checks")
        internal = self.event(
            "waiting", call="check", waiting_kind=WaitingKind.PERMISSION_CHECK
        )[0]
        self.now += 10
        permission = self.event(
            "waiting", call="check", waiting_kind=WaitingKind.PERMISSION
        )[0]
        session = self.sessions.by_id[SessionIdentity(SessionSource.CLAUDE, "one")]
        self.assertTrue(session.waiting_for_input)
        self.assertEqual(session.waiting_since, self.now)
        self.assertFalse(self.reactions.relevant(internal, self.sessions))
        self.assertTrue(self.reactions.relevant(permission, self.sessions))
        self.now += 10
        self.assertEqual(
            self.event("waiting", call="check", waiting_kind=WaitingKind.PERMISSION), []
        )
        self.assertEqual(session.waiting_since, self.now - 10)
        question = self.event(
            "waiting", call="check", waiting_kind=WaitingKind.QUESTION
        )[0]
        self.assertEqual(session.waiting_since, self.now)
        self.assertFalse(self.reactions.relevant(permission, self.sessions))
        self.assertTrue(self.reactions.relevant(question, self.sessions))
        check = self.event(
            "waiting", call="check", waiting_kind=WaitingKind.PERMISSION_CHECK
        )[0]
        self.assertTrue(session.waiting)
        self.assertFalse(session.waiting_for_input)
        self.assertFalse(self.reactions.relevant(question, self.sessions))
        self.assertTrue(self.reactions.relevant(check, self.sessions))

    def test_alias_merge_preserves_promoted_failure(self) -> None:
        self.event("intent", session="temporary", goal="Fix checks")
        promoted = None
        for index in range(3):
            for reaction in self.fail_tool(str(index), session="temporary"):
                if reaction.kind == "trouble":
                    promoted = reaction
        self.event("session_start", session="native", aliases=["temporary"])
        self.assertNotIn(
            "trouble",
            [item.kind for item in self.fail_tool("fourth", session="native")],
        )
        assert promoted is not None
        captured = promoted
        self.assertTrue(self.reactions.relevant(captured, self.sessions))

    def test_new_turn_supersedes_outcome_while_other_session_continues(self) -> None:
        self.event("intent", goal="Fix checks")
        self.event("intent", session="two", goal="Inspect code")
        outcome = self.event("turn_done")[0]
        captured = outcome
        self.assertTrue(self.sessions.active)
        self.assertTrue(self.reactions.relevant(captured, self.sessions))
        self.event("intent", goal="Next task")
        self.assertFalse(self.reactions.relevant(captured, self.sessions))

    def test_new_turn_discards_old_reaction_state_before_an_alias_merge(self) -> None:
        self.event("intent", session="temporary", goal="First task")
        for index in range(3):
            self.fail_tool(str(index), session="temporary")
        self.event("intent", session="temporary", goal="Second task")
        self.event("session_start", session="native", aliases=["temporary"])
        reactions = [
            item.kind
            for index in range(3)
            for item in self.fail_tool(f"new-{index}", session="native")
        ]
        self.assertEqual(reactions.count(ReactionKind.TROUBLE), 1)
        self.assertEqual(len(self.reactions.states), 1)

    def test_replacing_a_question_starts_a_new_waiting_reaction(self) -> None:
        self.event("intent", goal="Choose an approach")
        first = self.event("waiting", call="first")[0]
        second = self.event("waiting", call="second")[0]
        self.assertEqual(second.kind, "waiting")
        self.assertNotEqual(first.serial, second.serial)
        self.assertEqual(self.event("waiting", call="second"), [])
        captured = first
        self.assertFalse(self.reactions.relevant(captured, self.sessions))

    def test_merging_active_aliases_preserves_waiting_without_another_reminder(
        self,
    ) -> None:
        self.event("intent", session="native", goal="Choose a strategy")
        self.event("intent", session="temporary", goal="Choose a strategy")
        waiting = self.event("waiting", session="temporary", call="question")[0]
        self.assertEqual(
            self.event("session_start", session="native", aliases=["temporary"]), []
        )
        lane = self.sessions.by_id[SessionIdentity(SessionSource.CLAUDE, "native")]
        self.assertTrue(lane.waiting)
        self.assertEqual(lane.waiting_call, "question")
        captured = waiting
        self.assertTrue(self.reactions.relevant(captured, self.sessions))
        self.assertEqual(self.event("waiting", session="native", call="question"), [])
