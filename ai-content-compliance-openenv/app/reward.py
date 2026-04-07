"""
Advanced Reward Calculator for Content Compliance RL.

Provides realistic, nuanced reward signals for multi-step RL workflow.
Includes penalties to simulate real agent struggle and encourage efficiency.

Reward Components:
1. Correct Action Rewards:
   - Correct reject (spam) → +1.0
   - Correct approve (clean) → +1.0

2. Penalties:
   - False positive (reject clean) → -0.5
   - False negative (approve spam) → -1.0
   - Unnecessary edit → -0.2
   - Step limit penalty → -0.1 per step over optimal
   - Repeated action → -0.2 per repeat

3. Partial Rewards:
   - Correct violation detection → +0.3
   - Correct edit (fixes violations) → +0.7

4. Efficiency Incentives:
   - Per-step penalty → -0.05 (discourages long trajectories)
   - Early termination bonus → +0.2 (correct in 1-2 steps)
   - Optimal steps bonus → +0.1 (within expected steps)

5. Final reward clamped to [0.0, 1.0]
6. Detailed explanation for each reward component
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
    delta: float = 0.0  # Change to cumulative reward


@dataclass
class RewardBreakdown:
    """Complete breakdown of reward calculation."""

    components: list[RewardComponent] = field(default_factory=list)
    cumulative_reward: float = 0.0
    final_reward: float = 0.0
    explanation: str = ""

    def add_component(self, name: str, delta: float, description: str) -> None:
        """Add a reward component and update cumulative reward."""
        self.components.append(RewardComponent(
            name=name,
            value=self.cumulative_reward + delta,
            description=description,
            delta=delta,
        ))
        self.cumulative_reward += delta

    def finalize(self) -> float:
        """Clamp final reward to [0.0, 1.0] and return it."""
        self.final_reward = max(0.0, min(1.0, self.cumulative_reward))
        return self.final_reward


class AdvancedRewardCalculator:
    """
    Advanced reward calculator with realistic incentives.

    Designed to encourage intelligent moderation behavior:
    - Correctly identifying spam (reject bad content)
    - Not over-rejecting clean content
    - Using edits purposefully (not as crutch)
    - Completing tasks efficiently
    - Not getting stuck in action loops

    All rewards normalized to [0.0, 1.0] range.
    """

    # Correct action rewards
    CORRECT_REJECT_SPAM = 1.0
    CORRECT_APPROVE_CLEAN = 1.0

    # Error penalties
    FALSE_POSITIVE = -0.5  # Reject clean content
    FALSE_NEGATIVE = -1.0  # Approve spam content
    UNNECESSARY_EDIT = -0.2  # Edit when not needed

    # Partial rewards
    VIOLATION_DETECTED = 0.3  # Correctly identified violations exist
    CORRECT_EDIT = 0.7  # Edit that fixes violations

    # Step efficiency
    STEP_PENALTY = -0.05  # Small penalty per step (discourages long trajectories)
    STEP_LIMIT_PENALTY = -0.1  # Additional penalty per step over optimal
    MAX_STEPS_BEFORE_PENALTY = 3  # Steps before step limit penalty kicks in

    # Loop prevention
    LOOP_PENALTY = -0.2  # Repeating same action
    REPEATED_ACTION_PENALTY = -0.15  # Same action twice in a row

    # Early termination bonus
    EARLY_TERMINATION_BONUS = 0.2  # Correct decision in 1-2 steps
    OPTIMAL_STEPS_BONUS = 0.1  # Completed within expected steps

    def __init__(self, task_name: str = "easy"):
        self.task_name = task_name
        self._breakdown = RewardBreakdown()
        self._action_history: list[str] = []
        self._initial_violations: list[str] = []
        self._initial_score: float = 0.0
        self._episode_ended: bool = False

    def start_episode(
        self,
        initial_violations: list[str],
        initial_score: float,
    ) -> None:
        """
        Start a new episode with initial content state.

        Args:
            initial_violations: Violations detected in original content.
            initial_score: Initial compliance score.
        """
        self._breakdown = RewardBreakdown()
        self._action_history = []
        self._initial_violations = initial_violations.copy()
        self._initial_score = initial_score
        self._episode_ended = False

    def record_step(
        self,
        action: str,
        current_violations: list[str],
        current_score: float,
        previous_violations: list[str],
        previous_score: float,
        is_terminal: bool = False,
    ) -> RewardBreakdown:
        """
        Record a step and calculate reward for this action.

        Args:
            action: Action taken ("approve", "reject", "edit").
            current_violations: Violations after action.
            current_score: Compliance score after action.
            previous_violations: Violations before action.
            previous_score: Score before action.
            is_terminal: Whether this is a terminal action.

        Returns:
            Current reward breakdown.
        """
        if self._episode_ended:
            return self._breakdown

        self._action_history.append(action)

        # 1. Step penalty (every step has small cost)
        self._breakdown.add_component(
            name="step_penalty",
            delta=self.STEP_PENALTY,
            description=f"Step {len(self._action_history)}: -0.05 efficiency cost",
        )

        # 2. Loop penalty (repeating same action)
        if len(self._action_history) >= 2:
            if action == self._action_history[-2]:
                # Immediate repeat (same as previous action)
                self._breakdown.add_component(
                    name="repeated_action_penalty",
                    delta=self.REPEATED_ACTION_PENALTY,
                    description=f"Repeated action '{action}': {self.REPEATED_ACTION_PENALTY}",
                )

            # Check for longer loops (same action 3+ times in episode)
            action_count = self._action_history.count(action)
            if action_count >= 3:
                self._breakdown.add_component(
                    name="loop_penalty",
                    delta=self.LOOP_PENALTY,
                    description=f"Action loop ({action_count}x '{action}'): {self.LOOP_PENALTY}",
                )

        # 3. Determine content state
        has_violations = len(current_violations) > 0
        had_violations = len(previous_violations) > 0
        is_compliant = not has_violations and current_score >= 0.5
        was_compliant = not had_violations and previous_score >= 0.5

        # 4. Action-specific rewards/penalties
        # Note: We evaluate based on the ACTION's effect, not just current state
        # A reject action is correct if it removes/rejects violating content
        if action == "reject":
            # Reject is correct when content had violations (we're rejecting the bad content)
            if had_violations or has_violations:
                # Correct reject: caught the spam
                self._breakdown.add_component(
                    name="correct_reject",
                    delta=self.CORRECT_REJECT_SPAM,
                    description=f"Correctly rejected content with violations: +{self.CORRECT_REJECT_SPAM}",
                )
            else:
                # False positive: rejected clean content
                self._breakdown.add_component(
                    name="false_positive",
                    delta=self.FALSE_POSITIVE,
                    description=f"FALSE POSITIVE: Rejected clean content: {self.FALSE_POSITIVE}",
                )

        elif action == "approve":
            if has_violations:
                # False negative: approved spam (worst error)
                self._breakdown.add_component(
                    name="false_negative",
                    delta=self.FALSE_NEGATIVE,
                    description=f"FALSE NEGATIVE: Approved content with violations: {self.FALSE_NEGATIVE}",
                )
            else:
                # Correct approve: clean content approved
                self._breakdown.add_component(
                    name="correct_approve",
                    delta=self.CORRECT_APPROVE_CLEAN,
                    description=f"Correctly approved clean content: +{self.CORRECT_APPROVE_CLEAN}",
                )

        elif action == "edit":
            if has_violations and had_violations:
                # Edit attempted but violations remain - partial credit
                violations_reduced = len(current_violations) < len(previous_violations)
                if violations_reduced:
                    self._breakdown.add_component(
                        name="partial_edit",
                        delta=0.3,
                        description="Edit reduced violations: +0.3 partial credit",
                    )
                else:
                    self._breakdown.add_component(
                        name="ineffective_edit",
                        delta=-0.1,
                        description="Edit did not reduce violations: -0.1",
                    )
            elif had_violations and not has_violations:
                # Perfect edit: fixed all violations
                self._breakdown.add_component(
                    name="correct_edit",
                    delta=self.CORRECT_EDIT,
                    description=f"Edit fixed all violations: +{self.CORRECT_EDIT}",
                )
            elif not had_violations:
                # Unnecessary edit: content was already clean
                self._breakdown.add_component(
                    name="unnecessary_edit",
                    delta=self.UNNECESSARY_EDIT,
                    description=f"Unnecessary edit on clean content: {self.UNNECESSARY_EDIT}",
                )

        # 5. Violation detection bonus (first time detecting violations)
        if has_violations and not had_violations and len(self._action_history) == 1:
            # This shouldn't happen normally, but if violations appear after first action
            # and agent detected them, give small bonus
            pass
        elif had_violations and len(self._action_history) == 1:
            # First action on content with violations - agent should detect them
            if action in ["reject", "edit"]:
                self._breakdown.add_component(
                    name="violation_detected",
                    delta=self.VIOLATION_DETECTED,
                    description=f"Correctly identified violation exists: +{self.VIOLATION_DETECTED}",
                )

        # 6. Improvement bonus (score went up)
        if current_score > previous_score:
            improvement = current_score - previous_score
            bonus = min(0.3, improvement * 0.3)  # Cap at 0.3
            self._breakdown.add_component(
                name="improvement_bonus",
                delta=bonus,
                description=f"Score improved by {improvement:.2f}: +{bonus:.2f}",
            )

        return self._breakdown

    def finalize_episode(
        self,
        final_action: str,
        final_violations: list[str],
        final_score: float,
    ) -> tuple[float, str]:
        """
        Finalize the episode and return normalized reward.

        Includes efficiency incentives and step penalties.

        Args:
            final_action: Final terminal action taken.
            final_violations: Violations at episode end.
            final_score: Final compliance score.

        Returns:
            Tuple of (normalized_reward, explanation).
        """
        self._episode_ended = True
        num_steps = len(self._action_history)

        # Final correctness check
        # For reject: correct if we're rejecting problematic content (final state has violations or we removed them)
        # For approve: correct if content is truly clean (no violations, good score)
        has_violations = len(final_violations) > 0
        is_compliant = not has_violations and final_score >= 0.5

        # Determine if the final action was appropriate
        # Reject is correct when dealing with content that had/has violations
        # Approve is correct when content is clean
        correct_outcome = (
            (final_action == "reject" and (has_violations or len(self._initial_violations) > 0)) or
            (final_action == "approve" and is_compliant)
        )

        if correct_outcome:
            self._breakdown.add_component(
                name="correct_outcome",
                delta=0.2,
                description="Episode ended with correct outcome: +0.2 bonus",
            )
        else:
            self._breakdown.add_component(
                name="incorrect_outcome",
                delta=-0.3,
                description="Episode ended with incorrect outcome: -0.3 penalty",
            )

        # === Efficiency Incentives ===

        # 1. Early termination bonus (correct decision in 1-2 steps)
        if correct_outcome and num_steps <= 2:
            self._breakdown.add_component(
                name="early_termination_bonus",
                delta=self.EARLY_TERMINATION_BONUS,
                description=f"Early termination ({num_steps} steps): +{self.EARLY_TERMINATION_BONUS}",
            )

        # 2. Optimal steps bonus (completed within expected steps for difficulty)
        optimal_steps_map = {"easy": 2, "medium": 3, "hard": 4}
        optimal_steps = optimal_steps_map.get(self.task_name, 3)
        if correct_outcome and num_steps <= optimal_steps:
            self._breakdown.add_component(
                name="optimal_steps_bonus",
                delta=self.OPTIMAL_STEPS_BONUS,
                description=f"Optimal steps ({num_steps} <= {optimal_steps}): +{self.OPTIMAL_STEPS_BONUS}",
            )

        # 3. Step limit penalty (more steps = lower reward)
        if num_steps > self.MAX_STEPS_BEFORE_PENALTY:
            excess_steps = num_steps - self.MAX_STEPS_BEFORE_PENALTY
            step_limit_penalty = excess_steps * self.STEP_LIMIT_PENALTY
            self._breakdown.add_component(
                name="step_limit_penalty",
                delta=step_limit_penalty,
                description=f"Step limit ({excess_steps} excess steps): {step_limit_penalty:.2f}",
            )

        # 4. Inefficiency penalty for very long trajectories
        if num_steps >= 5:
            long_trajectory_penalty = -0.15
            self._breakdown.add_component(
                name="long_trajectory_penalty",
                delta=long_trajectory_penalty,
                description=f"Long trajectory ({num_steps} steps): {long_trajectory_penalty:.2f}",
            )

        # Finalize and clamp
        final_reward = self._breakdown.finalize()
        explanation = self._generate_explanation()

        return final_reward, explanation

    def get_current_reward(self) -> tuple[float, str]:
        """Get current cumulative reward (before finalization)."""
        current = max(0.0, min(1.0, self._breakdown.cumulative_reward))
        return current, self._generate_explanation()

    def get_breakdown(self) -> dict[str, Any]:
        """Get detailed breakdown of reward components."""
        final_reward, explanation = self.get_current_reward()

        return {
            "final_reward": round(final_reward, 3),
            "cumulative_reward": round(self._breakdown.cumulative_reward, 3),
            "components": [
                {
                    "name": comp.name,
                    "delta": round(comp.delta, 3),
                    "cumulative": round(comp.value, 3),
                    "description": comp.description,
                }
                for comp in self._breakdown.components
            ],
            "steps": len(self._action_history),
            "action_history": self._action_history.copy(),
            "explanation": explanation,
        }

    def _generate_explanation(self) -> str:
        """Generate human-readable explanation of reward."""
        parts = []

        # Summary of actions
        if self._action_history:
            parts.append(f"Actions: {' -> '.join(self._action_history)}")

        # Key components
        positive_components = [c for c in self._breakdown.components if c.delta > 0]
        negative_components = [c for c in self._breakdown.components if c.delta < 0]

        if positive_components:
            positive_summary = sum(c.delta for c in positive_components)
            parts.append(f"bonuses: +{positive_summary:.2f}")

        if negative_components:
            negative_summary = sum(c.delta for c in negative_components)
            parts.append(f"penalties: {negative_summary:.2f}")

        # Final assessment
        final = self._breakdown.final_reward if self._episode_ended else self._breakdown.cumulative_reward
        if final >= 0.8:
            parts.append("EXCELLENT")
        elif final >= 0.6:
            parts.append("GOOD")
        elif final >= 0.4:
            parts.append("ACCEPTABLE")
        elif final >= 0.2:
            parts.append("POOR")
        else:
            parts.append("VERY POOR")

        return "; ".join(parts)


# Backward compatibility wrapper
class RewardCalculator:
    """
    Backward-compatible wrapper for AdvancedRewardCalculator.

    Maintains the same API as the original RewardCalculator
    while using the new advanced reward logic internally.
    """

    ACTION_WEIGHT = 0.4
    IMPROVEMENT_WEIGHT = 0.3
    EFFICIENCY_WEIGHT = 0.3

    OPTIMAL_STEPS = {
        "easy": 1,
        "medium": 2,
        "hard": 3,
    }

    def __init__(self, task_name: str = "easy"):
        self.task_name = task_name
        self._advanced = AdvancedRewardCalculator(task_name)
        self._components: list[RewardComponent] = []
        self._step_count = 0
        self._terminal_correct: Optional[bool] = None
        self.expected_output: Optional[Any] = None
        self._previous_violations: list[str] = []
        self._previous_score: float = 0.0

    def set_ground_truth(self, expected_output: Any) -> None:
        """Set expected output for reward calculation."""
        self.expected_output = expected_output

    def _get_expected_action(self) -> str:
        """Get expected action from ground truth."""
        if self.expected_output is None:
            return ""
        if isinstance(self.expected_output, str):
            return self.expected_output.lower()
        if isinstance(self.expected_output, dict):
            return self.expected_output.get("action", "").lower()
        return ""

    def start_episode(
        self,
        initial_violations: list[str],
        initial_score: float,
    ) -> None:
        """Initialize episode with initial content state."""
        self._advanced.start_episode(initial_violations, initial_score)
        self._previous_violations = initial_violations.copy()
        self._previous_score = initial_score

    def record_action(
        self,
        action_taken: str,
        expected_action: Optional[str] = None,
    ) -> None:
        """
        Record an action (compatibility method).

        For advanced rewards, use record_step() instead.
        """
        self._step_count += 1
        # This is a no-op for advanced calculator
        # record_step() handles the actual reward calculation

    def record_improvement(
        self,
        original_score: float,
        new_score: float,
    ) -> None:
        """
        Record score improvement (compatibility method).

        Improvement is automatically tracked in record_step().
        """
        # Handled automatically in record_step()
        pass

    def record_step(
        self,
        action: str,
        current_violations: list[str],
        current_score: float,
        is_terminal: bool = False,
    ) -> None:
        """
        Record a full step with advanced reward calculation.

        Args:
            action: Action taken ("approve", "reject", "edit").
            current_violations: Violations after action.
            current_score: Score after action.
            is_terminal: Whether this is a terminal action.
        """
        self._step_count += 1

        self._advanced.record_step(
            action=action,
            current_violations=current_violations,
            current_score=current_score,
            previous_violations=self._previous_violations,
            previous_score=self._previous_score,
            is_terminal=is_terminal,
        )

        # Update previous state
        self._previous_violations = current_violations.copy()
        self._previous_score = current_score

    def finalize(self, terminal_correct: Optional[bool] = None) -> None:
        """Finalize the episode."""
        if terminal_correct is not None:
            self._terminal_correct = terminal_correct

    def get_final_reward(self) -> tuple[float, str]:
        """Get final normalized reward and explanation."""
        # For compatibility, if finalize() was called without record_step(),
        # use the advanced calculator's finalization
        if self._terminal_correct is not None:
            # Dummy finalization for compatibility
            pass

        return self._advanced.get_current_reward()

    def get_reward_breakdown(self) -> dict[str, Any]:
        """Get detailed reward breakdown."""
        return self._advanced.get_breakdown()
