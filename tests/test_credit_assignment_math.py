import math
import random
import unittest

from experiments.toy_credit_assignment import (
    DEFAULT_SCENARIO,
    TabularCritic,
    Trajectory,
    StepRecord,
    action_probabilities,
    calibrated_mse,
    exact_value,
    next_score,
    pearson,
    sign_accuracy,
    simulate_trajectory,
    terminal_reward,
)


class CreditAssignmentMathTests(unittest.TestCase):
    def test_action_probabilities_are_valid_distribution(self) -> None:
        for threshold in (1, 2, 3):
            for score in range(-2, 4):
                for remaining in range(1, 5):
                    probs = action_probabilities(
                        threshold,
                        score,
                        remaining,
                        DEFAULT_SCENARIO,
                    )
                    self.assertEqual(set(probs), {"help", "harm", "wait"})
                    self.assertAlmostEqual(sum(probs.values()), 1.0)
                    self.assertTrue(all(value > 0.0 for value in probs.values()))

    def test_exact_value_satisfies_bellman_identity(self) -> None:
        threshold = 2
        score = 0
        remaining = 3
        probs = action_probabilities(threshold, score, remaining, DEFAULT_SCENARIO)
        expected = sum(
            probability
            * (
                -DEFAULT_SCENARIO.token_cost
                + exact_value(
                    threshold,
                    next_score(score, action),
                    remaining - 1,
                    DEFAULT_SCENARIO,
                )
            )
            for action, probability in probs.items()
        )

        self.assertAlmostEqual(
            exact_value(threshold, score, remaining, DEFAULT_SCENARIO),
            expected,
        )
        self.assertEqual(
            exact_value(threshold, threshold, 0, DEFAULT_SCENARIO),
            terminal_reward(threshold, threshold),
        )

    def test_oracle_advantage_has_zero_policy_expectation(self) -> None:
        threshold = 2
        score = 1
        remaining = 4
        probs = action_probabilities(threshold, score, remaining, DEFAULT_SCENARIO)
        weighted_advantage = sum(
            probability
            * (
                -DEFAULT_SCENARIO.token_cost
                + exact_value(
                    threshold,
                    next_score(score, action),
                    remaining - 1,
                    DEFAULT_SCENARIO,
                )
                - exact_value(threshold, score, remaining, DEFAULT_SCENARIO)
            )
            for action, probability in probs.items()
        )

        self.assertAlmostEqual(weighted_advantage, 0.0)

    def test_terminal_critic_value_uses_known_terminal_reward_for_unseen_states(self) -> None:
        training = [
            simulate_trajectory(
                random.Random(1),
                trajectory_id=0,
                prompt_id=0,
                threshold=1,
                max_steps=3,
                scenario=DEFAULT_SCENARIO,
            )
        ]
        critic = TabularCritic(training)

        self.assertEqual(critic.value(threshold=3, score=3, remaining=0), 1.0)
        self.assertEqual(critic.value(threshold=3, score=0, remaining=0), 0.0)

    def test_metric_helpers_reject_mismatched_lengths_and_zero_is_neutral(self) -> None:
        with self.assertRaises(ValueError):
            pearson([1.0], [1.0, 2.0])
        with self.assertRaises(ValueError):
            calibrated_mse([1.0], [1.0, 2.0])
        with self.assertRaises(ValueError):
            sign_accuracy([1.0], [1.0, 2.0])

        self.assertEqual(sign_accuracy([0.0, 1.0, -1.0], [1.0, 1.0, -1.0]), 2 / 3)


if __name__ == "__main__":
    unittest.main()

