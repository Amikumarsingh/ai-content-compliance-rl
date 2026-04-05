"""
Reward Calculator for Content Compliance RL.

Provides dense, explainable reward signals normalized to [0.0, 1.0].

Reward Components:
- Action Correctness: +0.4 for correct approve/reject
- Improvement Bonus: +0.3 for score improvement
- Efficiency: +0.3 for completing in optimal steps
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RewardComponent:
    """A single reward component with explanation."""

    name: str
    value: float
    description: str
    max_value: float = 1.0


class RewardCalculator:
    """
    Calculates meaningful, explainable rewards for content compliance tasks.

    All rewards are dense (partial credit) rather than binary.
    Normalized to [0.0, 1.0] range.
    """

    # Reward weights (sum to 1.0)
    ACTION_WEIGHT = 0.4
    IMPROVEMENT_WEIGHT = 0.3
    EFFICIENCY_WEIGHT = 0.3

    # Efficiency baseline
    OPTIMAL_STEPS = {
        "easy": 1,
        "medium": 2,
        "hard": 3,
    }

    def __init__(self, task_name: str = "easy"):
        self.task_name = task_name
        self._step_count = 0
        self._actions_taken: list[str] = []
        self._terminal_action_correct = False
        self._components: list[RewardComponent] = []

    def record_action(
        self,
        action_taken: str,
        expected_action: str,
    ) -> None:
        """Record an action decision."""
        self._step_count += 1
        self._actions_taken.append(action_taken)

        action_taken = action_taken.lower()
        expected_action = expected_action.lower()

        if action_taken == expected_action:
            action_reward = self.ACTION_WEIGHT
            description = f"Correct action: {action_taken}"
        elif self._is_action_close(action_taken, expected_action):
            action_reward = self.ACTION_WEIGHT * 0.5
            description = f"Partial credit: {action_taken} instead of {expected_action}"
        else:
            action_reward = 0.0
            description = f"Wrong action: {action_taken} instead of {expected_action}"

        self._components.append(
            RewardComponent(
                name="action_correctness",
                value=action_reward,
                description=description,
                max_value=self.ACTION_WEIGHT,
            )
        )

        if action_taken in ["approve", "reject"]:
            self._terminal_action_correct = action_taken == expected_action

    def record_improvement(
        self,
        original_score: float,
        new_score: float,
    ) -> None:
        """Record score improvement from an edit."""
        score_improvement = new_score - original_score

        if score_improvement > 0:
            improvement_reward = min(
                self.IMPROVEMENT_WEIGHT,
                self.IMPROVEMENT_WEIGHT * score_improvement
            )
            description = f"Score improved: {original_score:.2f} -> {new_score:.2f}"
        else:
            improvement_reward = 0.0
            description = f"No improvement: {original_score:.2f} -> {new_score:.2f}"

        self._components.append(
            RewardComponent(
                name="improvement",
                value=improvement_reward,
                description=description,
                max_value=self.IMPROVEMENT_WEIGHT,
            )
        )

    def finalize(self, terminal_correct: bool = True) -> None:
        """Finalize the episode and apply efficiency bonus."""
        self._terminal_action_correct = terminal_correct

        optimal = self.OPTIMAL_STEPS.get(self.task_name, 2)
        actual = self._step_count

        if actual <= optimal:
            efficiency_reward = self.EFFICIENCY_WEIGHT
            description = f"Efficient: {actual} steps (optimal: {optimal})"
        elif actual <= optimal + 2:
            efficiency_ratio = (optimal + 2 - actual) / 2
            efficiency_reward = self.EFFICIENCY_WEIGHT * efficiency_ratio
            description = f"Acceptable: {actual} steps (optimal: {optimal})"
        else:
            efficiency_reward = max(0.0, self.EFFICIENCY_WEIGHT - 0.05 * (actual - optimal))
            description = f"Inefficient: {actual} steps (optimal: {optimal})"

        self._components.append(
            RewardComponent(
                name="efficiency",
                value=efficiency_reward,
                description=description,
                max_value=self.EFFICIENCY_WEIGHT,
            )
        )

    def get_final_reward(self) -> tuple[float, str]:
        """Calculate final cumulative reward normalized to [0.0, 1.0]."""
        total_reward = sum(comp.value for comp in self._components)

        # Terminal correctness bonus/penalty
        if self._terminal_action_correct:
            total_reward += 0.1
        else:
            total_reward -= 0.1

        # Normalize to [0.0, 1.0]
        normalized = max(0.0, min(1.0, total_reward))

        explanation = self._generate_explanation(normalized)
        return normalized, explanation

    def _is_action_close(self, action_taken: str, expected_action: str) -> bool:
        """Check if action is 'close' to expected for partial credit."""
        close_pairs = [("edit", "reject"), ("reject", "edit")]
        return (action_taken, expected_action) in close_pairs

    def _generate_explanation(self, normalized: float) -> str:
        """Generate human-readable explanation of final reward."""
        parts = []

        if normalized >= 0.8:
            parts.append("Excellent performance")
        elif normalized >= 0.6:
            parts.append("Good performance")
        elif normalized >= 0.4:
            parts.append("Acceptable performance")
        elif normalized >= 0.2:
            parts.append("Poor performance")
        else:
            parts.append("Very poor performance")

        parts.append(f"({self._step_count} steps)")

        if self._terminal_action_correct:
            parts.append("correct termination")
        else:
            parts.append("incorrect termination")

        return "; ".join(parts)

    def get_reward_breakdown(self) -> dict[str, Any]:
        """Get detailed breakdown of reward components."""
        normalized, explanation = self.get_final_reward()

        by_type: dict[str, float] = {}
        for comp in self._components:
            if comp.name not in by_type:
                by_type[comp.name] = 0.0
            by_type[comp.name] += comp.value

        return {
            "total_reward": round(normalized, 3),
            "breakdown": {k: round(v, 3) for k, v in by_type.items()},
            "step_count": self._step_count,
            "explanation": explanation,
        }
