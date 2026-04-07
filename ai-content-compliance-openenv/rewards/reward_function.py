"""
Meaningful Reward Function for Content Compliance RL.

Provides dense, explainable reward signals that:
- Reflect partial progress (not binary)
- Penalize bad actions
- Encourage efficient completion

Reward Components (total 1.0):
- Violation Detection: +0.3 for correct detection
- Action Correctness: +0.3 for correct action choice
- Edit Quality: +0.2 for effective edits
- Efficiency: +0.2 for completing in optimal steps
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class RewardComponent:
    """A single reward component with explanation."""

    name: str
    value: float
    description: str
    max_value: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "value": round(self.value, 3),
            "description": self.description,
            "max_value": self.max_value,
        }


class RewardCalculator:
    """
    Calculates meaningful, explainable rewards for content compliance tasks.

    All rewards are dense (partial credit) rather than binary.
    Normalized to [0.0, 1.0] range.
    """

    DETECTION_WEIGHT = 0.3
    ACTION_WEIGHT = 0.3
    EDIT_WEIGHT = 0.2
    EFFICIENCY_WEIGHT = 0.2

    WRONG_ACTION_PENALTY = -0.2
    UNNECESSARY_STEP_PENALTY = -0.1

    OPTIMAL_STEPS = {
        "easy": 1,
        "medium": 2,
        "hard": 3,
    }

    def __init__(self, task_name: str = "easy"):
        self.task_name = task_name
        self.step_rewards: list[dict[str, Any]] = []
        self.components: list[RewardComponent] = []

        self._step_count = 0
        self._violations_detected: list[str] = []
        self._true_violations: list[str] = []
        self._actions_taken: list[str] = []
        self._is_episode_complete = False
        self._terminal_action_correct = False

    def set_ground_truth(self, true_violations: list[str]) -> None:
        """Set ground truth violations for comparison."""
        self._true_violations = true_violations

    def record_detection(self, detected_violations: list[str]) -> dict[str, Any]:
        """Record a violation detection step."""
        self._step_count += 1
        self._violations_detected.extend(detected_violations)

        true_set = set(self._true_violations)
        detected_set = set(detected_violations)

        true_positives = len(true_set & detected_set)
        false_positives = len(detected_set - true_set)

        if len(true_set) > 0:
            detection_ratio = true_positives / len(true_set)
            detection_reward = self.DETECTION_WEIGHT * detection_ratio
        else:
            detection_reward = self.DETECTION_WEIGHT if not detected_violations else 0.0

        if false_positives > 0:
            detection_reward -= min(0.1, 0.05 * false_positives)

        detection_reward = max(0.0, detection_reward)

        step_reward = {
            "step_number": self._step_count,
            "step_type": "detection",
            "reward": detection_reward,
            "explanation": f"Detected {true_positives}/{len(true_set)} violations",
        }

        self.step_rewards.append(step_reward)
        return step_reward

    def record_action(self, action_taken: str, expected_action: str) -> dict[str, Any]:
        """Record an action decision."""
        self._step_count += 1
        self._actions_taken.append(action_taken)

        action_taken = action_taken.lower()
        expected_action = expected_action.lower()

        if action_taken == expected_action:
            action_reward = self.ACTION_WEIGHT
            explanation = f"Correct action: {action_taken}"
        elif self._is_action_close(action_taken, expected_action):
            action_reward = self.ACTION_WEIGHT * 0.5
            explanation = f"Partial credit: {action_taken} instead of {expected_action}"
        else:
            action_reward = self.WRONG_ACTION_PENALTY
            explanation = f"Wrong action: {action_taken} instead of {expected_action}"

        action_reward = max(-0.3, min(self.ACTION_WEIGHT, action_reward))

        step_reward = {
            "step_number": self._step_count,
            "step_type": "action",
            "reward": action_reward,
            "explanation": explanation,
        }

        self.step_rewards.append(step_reward)

        if action_taken in ["approve", "reject"]:
            self._is_episode_complete = True
            self._terminal_action_correct = action_taken == expected_action

        return step_reward

    def record_edit(
        self,
        original_score: float,
        new_score: float,
        violations_before: list[str],
        violations_after: list[str],
    ) -> dict[str, Any]:
        """Record an edit action."""
        self._step_count += 1

        score_improvement = new_score - original_score
        violations_removed = len(violations_before) - len(violations_after)

        improvement_reward = min(0.1, max(0.0, score_improvement * 0.2))

        if len(violations_before) > 0:
            removal_ratio = violations_removed / len(violations_before)
            removal_reward = self.EDIT_WEIGHT * removal_ratio
        else:
            removal_reward = 0.1 if not violations_after else 0.0

        edit_reward = improvement_reward + removal_reward
        edit_reward = max(0.0, min(self.EDIT_WEIGHT, edit_reward))

        if score_improvement < 0:
            edit_reward = max(-0.1, edit_reward - 0.1)

        step_reward = {
            "step_number": self._step_count,
            "step_type": "edit",
            "reward": edit_reward,
            "explanation": f"Score: {original_score:.2f} -> {new_score:.2f}",
        }

        self.step_rewards.append(step_reward)
        return step_reward

    def finalize(self, terminal_correct: bool = True) -> None:
        """Finalize the episode and apply efficiency bonus/penalty."""
        self._is_episode_complete = True
        self._terminal_action_correct = terminal_correct

        optimal = self.OPTIMAL_STEPS.get(self.task_name, 2)
        actual = self._step_count

        if actual <= optimal:
            efficiency_reward = self.EFFICIENCY_WEIGHT
            explanation = f"Efficient: {actual} steps (optimal: {optimal})"
        elif actual <= optimal + 2:
            efficiency_ratio = (optimal + 2 - actual) / 2
            efficiency_reward = self.EFFICIENCY_WEIGHT * efficiency_ratio
            explanation = f"Acceptable: {actual} steps (optimal: {optimal})"
        else:
            excess_steps = actual - optimal
            efficiency_reward = max(-0.1, self.EFFICIENCY_WEIGHT - 0.05 * excess_steps)
            explanation = f"Inefficient: {actual} steps (optimal: {optimal})"

        self.components.append(
            RewardComponent(
                name="efficiency",
                value=efficiency_reward,
                description=explanation,
                max_value=self.EFFICIENCY_WEIGHT,
            )
        )

    def get_final_reward(self) -> tuple[float, str]:
        """Calculate final cumulative reward normalized to [0.0, 1.0]."""
        total_reward = sum(sr["reward"] for sr in self.step_rewards)

        for comp in self.components:
            if comp.name == "efficiency":
                total_reward += comp.value
                break

        if self._terminal_action_correct:
            total_reward += 0.1
        else:
            total_reward -= 0.2

        normalized = (total_reward + 0.6) / 1.8
        normalized = max(0.0, min(1.0, normalized))

        explanation = self._generate_explanation(normalized)
        return normalized, explanation

    def get_reward_breakdown(self) -> dict[str, Any]:
        """Get detailed breakdown of reward components."""
        normalized, explanation = self.get_final_reward()

        by_type: dict[str, float] = {}
        for sr in self.step_rewards:
            step_type = sr["step_type"]
            if step_type not in by_type:
                by_type[step_type] = 0.0
            by_type[step_type] += sr["reward"]

        for comp in self.components:
            if comp.name == "efficiency":
                by_type["efficiency"] = comp.value

        return {
            "total_reward": round(normalized, 3),
            "raw_reward": round(sum(sr["reward"] for sr in self.step_rewards), 3),
            "breakdown": {k: round(v, 3) for k, v in by_type.items()},
            "step_count": self._step_count,
            "explanation": explanation,
        }

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
