import random
import unittest
from collections.abc import Callable
from dataclasses import replace

from pet_daemon.behavior.policy import Candidate, Observation, Policy


class PolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = Policy()
        self.observation = Observation(
            active=False,
            working=False,
            investigating=False,
            waiting=False,
            interacting=False,
            pointer_recent=False,
            can_roam=False,
            can_nap=False,
            restlessness=0,
            rest_pressure=0,
        )

    def select_many(
        self,
        observation: Observation,
        last_completed: str | None = None,
        available: Callable[[Candidate], bool] = lambda _: True,
    ) -> list[str]:
        rng = random.Random(123)
        return [
            self.policy.select(observation, last_completed, rng, available).name
            for _ in range(100)
        ]

    def test_seeded_selection_varies_among_choices_with_similar_utility(self) -> None:
        observation = replace(self.observation, can_roam=True, restlessness=0.28)
        selections = self.select_many(observation)
        self.assertEqual(set(selections), {"rest", "roam"})
        self.assertEqual(selections, self.select_many(observation))

    def test_selection_excludes_choices_more_than_ten_percent_below_best(self) -> None:
        observation = replace(self.observation, can_roam=True, restlessness=1)
        self.assertEqual(set(self.select_many(observation)), {"roam"})

    def test_repetition_penalty_favors_an_available_alternative(self) -> None:
        observation = replace(self.observation, can_roam=True, restlessness=0.36)
        self.assertEqual(set(self.select_many(observation)), {"rest", "roam"})
        self.assertEqual(set(self.select_many(observation, "rest")), {"roam"})

    def test_recent_movement_keeps_active_work_focused(self) -> None:
        observation = replace(
            self.observation, active=True, working=True, can_roam=True, restlessness=0
        )
        self.assertEqual(set(self.select_many(observation)), {"focus"})
        candidates = {
            candidate.name: candidate
            for candidate in self.policy.candidates(observation)
        }
        self.assertAlmostEqual(candidates["roam"].score, 0.2)
        self.assertEqual(candidates["focus"].role, "work")
        investigating = replace(observation, investigating=True)
        selected = self.policy.select(
            investigating, None, random.Random(123), lambda _: True
        )
        self.assertEqual((selected.name, selected.role), ("focus", "attention"))

    def test_sustained_work_can_walk_and_focus(self) -> None:
        observation = replace(
            self.observation, active=True, working=True, can_roam=True, restlessness=1
        )
        self.assertEqual(set(self.select_many(observation, "focus")), {"focus", "roam"})
        after_walk = replace(observation, restlessness=0)
        self.assertEqual(set(self.select_many(after_walk, "roam")), {"focus"})

    def test_waiting_keeps_attention_without_work_or_roaming(self) -> None:
        observation = replace(
            self.observation, active=True, waiting=True, can_roam=True, restlessness=1
        )
        self.assertEqual(set(self.select_many(observation)), {"attentive"})

    def test_pointer_observation_requires_recent_movement_without_speech(self) -> None:
        observation = replace(self.observation, pointer_recent=True)
        self.assertEqual(set(self.select_many(observation)), {"observe"})
        speaking = replace(observation, interacting=True)
        self.assertEqual(set(self.select_many(speaking)), {"rest"})

    def test_unavailable_assets_leave_rest_as_a_valid_choice(self) -> None:
        observation = replace(
            self.observation, active=True, working=True, can_roam=True, restlessness=1
        )
        self.assertEqual(
            set(
                self.select_many(
                    observation, available=lambda candidate: candidate.name == "rest"
                )
            ),
            {"rest"},
        )

    def test_nap_utility_requires_eligibility_and_sufficient_rest_pressure(
        self,
    ) -> None:
        observation = replace(self.observation, rest_pressure=1)
        self.assertEqual(set(self.select_many(observation)), {"rest"})
        self.assertEqual(
            set(self.select_many(replace(observation, can_nap=True))), {"nap"}
        )
        self.assertEqual(
            set(
                self.select_many(replace(observation, can_nap=True, rest_pressure=0.5))
            ),
            {"rest"},
        )


if __name__ == "__main__":
    unittest.main()
