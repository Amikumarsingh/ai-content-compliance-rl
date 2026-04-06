"""
Graders for Content Moderation RL Environment.

Deterministic scoring logic for evaluating agent performance:
- Based on expected action + quality of reasoning
- Never returns random rewards
- Supports partial credit for near-correct actions

Scoring Principles:
1. Correct action gets high score (0.8-1.0)
2. Partially correct gets medium score (0.4-0.7)
3. Wrong action gets low score (0.0-0.3)
4. Harmful errors (approving dangerous content) get negative adjustment
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class GradingResult:
    """Result of grading an agent's action."""

    score: float  # 0.0 to 1.0
    is_correct: bool
    explanation: str
    components: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class ContentModerationGrader:
    """
    Deterministic grader for content moderation actions.

    Evaluates agent actions against expected outcomes:
    - approve_correct: Agent approved compliant content
    - reject_correct: Agent rejected violating content
    - edit_correct: Agent edited content to fix violations
    """

    # Score thresholds
    EXCELLENT_THRESHOLD = 0.8
    GOOD_THRESHOLD = 0.6
    ACCEPTABLE_THRESHOLD = 0.4

    # Penalty values
    HARMFUL_APPROVAL_PENALTY = -0.2
    UNNECESSARY_EDIT_PENALTY = -0.1
    STEP_PENALTY = -0.05

    def __init__(self, task_config: Optional[Any] = None):
        """
        Initialize grader.

        Args:
            task_config: Optional task configuration for expected actions.
        """
        self.task_config = task_config
        self._grading_history: list[GradingResult] = []

    def grade(
        self,
        action_taken: str,
        expected_action: str,
        content_score: float,
        violations: list[str],
        expected_violations: list[str],
        edit_quality: float = 0.0,
        step_count: int = 1,
    ) -> GradingResult:
        """
        Grade an agent's action.

        Args:
            action_taken: Action the agent took ("approve", "reject", "edit").
            expected_action: Expected correct action.
            content_score: Compliance score of content (0.0-1.0).
            violations: Violations detected in content.
            expected_violations: Violations expected in this task.
            edit_quality: Quality of edit if action was "edit" (0.0-1.0).
            step_count: Current step number.

        Returns:
            GradingResult with score and explanation.
        """
        components: dict[str, float] = {}
        explanation_parts: list[str] = []

        # 1. Base score for action correctness
        base_score = self._grade_action_correctness(
            action_taken, expected_action, content_score, violations
        )
        components["action_correctness"] = base_score

        if base_score >= self.EXCELLENT_THRESHOLD:
            explanation_parts.append("Correct action")
        elif base_score >= self.GOOD_THRESHOLD:
            explanation_parts.append("Mostly correct action")
        elif base_score >= self.ACCEPTABLE_THRESHOLD:
            explanation_parts.append("Partially correct action")
        else:
            explanation_parts.append("Incorrect action")

        # 2. Violation detection accuracy (for reject/edit actions)
        if action_taken in ["reject", "edit"]:
            detection_score = self._grade_violation_detection(
                violations, expected_violations
            )
            components["violation_detection"] = detection_score

            if detection_score >= 0.8:
                explanation_parts.append("all violations detected")
            elif detection_score >= 0.5:
                explanation_parts.append("some violations detected")
            else:
                explanation_parts.append("missed key violations")

        # 3. Edit quality (if applicable)
        if action_taken == "edit":
            edit_score = self._grade_edit_quality(
                edit_quality, violations, expected_violations
            )
            components["edit_quality"] = edit_score

            if edit_score >= 0.7:
                explanation_parts.append("good edit")
            elif edit_score >= 0.4:
                explanation_parts.append("partial edit")
            else:
                explanation_parts.append("poor edit")

        # 4. Step efficiency penalty
        efficiency_penalty = max(0, (step_count - 1) * self.STEP_PENALTY)
        components["efficiency_penalty"] = efficiency_penalty

        # 5. Calculate final score
        final_score = base_score - efficiency_penalty

        # Add bonuses for good performance
        if len(components) > 1:
            avg_component = sum(components.values()) / len(components)
            final_score = max(final_score, avg_component)

        # Clamp to valid range
        final_score = max(0.0, min(1.0, final_score))

        # Determine if correct
        is_correct = final_score >= self.ACCEPTABLE_THRESHOLD

        # Build explanation
        explanation = "; ".join(explanation_parts)
        if efficiency_penalty > 0:
            explanation += f"; -{efficiency_penalty:.2f} for extra steps"

        result = GradingResult(
            score=final_score,
            is_correct=is_correct,
            explanation=explanation,
            components=components,
            metadata={
                "action": action_taken,
                "expected": expected_action,
                "step_count": step_count,
            },
        )

        self._grading_history.append(result)
        return result

    def _grade_action_correctness(
        self,
        action: str,
        expected: str,
        content_score: float,
        violations: list[str],
    ) -> float:
        """Grade the correctness of the action taken."""
        is_compliant = content_score >= 0.5 and len(violations) == 0

        # Exact match
        if action == expected:
            if expected == "approve" and is_compliant:
                return 1.0  # Perfect: approved clean content
            elif expected == "reject" and not is_compliant:
                return 1.0  # Perfect: rejected violating content
            elif expected == "edit":
                return 0.8  # Good: edited when needed
            return 0.9

        # Partial matches
        if action == "edit" and expected in ["approve", "reject"]:
            # Edit is often a reasonable intermediate step
            return 0.5

        if action == "approve":
            if not is_compliant:
                # Approved violating content - serious error
                if any(v in ["hate_speech", "violence", "illegal_content"] for v in violations):
                    return 0.0 + self.HARMFUL_APPROVAL_PENALTY
                return 0.2  # Wrong but less severe
            return 0.3  # Approved clean content when shouldn't have

        if action == "reject":
            if is_compliant:
                # Rejected clean content - false positive
                return 0.3
            return 0.4  # Rejected violating content but maybe wrong reason

        return 0.1  # Unknown action

    def _grade_violation_detection(
        self,
        detected: list[str],
        expected: list[str],
    ) -> float:
        """Grade how well violations were detected."""
        if not expected:
            return 1.0  # No violations expected

        detected_set = set(detected)
        expected_set = set(expected)

        # True positives
        true_positives = len(detected_set & expected_set)

        # False negatives (missed violations)
        false_negatives = len(expected_set - detected_set)

        # False positives (wrongly flagged)
        false_positives = len(detected_set - expected_set)

        if not expected_set:
            return 1.0 if not detected_set else 0.5

        # Calculate recall
        recall = true_positives / len(expected_set) if expected_set else 1.0

        # Penalize false positives slightly
        fp_penalty = min(0.3, false_positives * 0.1)

        score = recall - fp_penalty
        return max(0.0, min(1.0, score))

    def _grade_edit_quality(
        self,
        edit_quality: float,
        violations_before: list[str],
        expected_violations: list[str],
    ) -> float:
        """Grade the quality of an edit action."""
        # Base score from edit quality metric
        base = edit_quality

        # Bonus if violations were actually present
        if violations_before:
            base += 0.1

        # Penalty if no violations existed (unnecessary edit)
        if not violations_before and not expected_violations:
            base += self.UNNECESSARY_EDIT_PENALTY

        return max(0.0, min(1.0, base))

    def get_grade_summary(self) -> dict[str, Any]:
        """Get summary of all grading results."""
        if not self._grading_history:
            return {"count": 0}

        scores = [r.score for r in self._grading_history]
        correct_count = sum(1 for r in self._grading_history if r.is_correct)

        return {
            "count": len(self._grading_history),
            "average_score": sum(scores) / len(scores),
            "correct_count": correct_count,
            "accuracy": correct_count / len(self._grading_history),
            "min_score": min(scores),
            "max_score": max(scores),
        }

    def reset(self) -> None:
        """Reset grading history."""
        self._grading_history = []


class BatchGrader:
    """
    Grader for batch evaluation of multiple episodes.

    Aggregates results across multiple grading runs.
    """

    def __init__(self):
        self.graders: list[ContentModerationGrader] = []
        self.results: list[GradingResult] = []

    def add_result(self, result: GradingResult) -> None:
        """Add a grading result."""
        self.results.append(result)

    def get_statistics(self) -> dict[str, Any]:
        """Get aggregate statistics."""
        if not self.results:
            return {"count": 0}

        scores = [r.score for r in self.results]
        correct_count = sum(1 for r in self.results if r.is_correct)

        # Component-wise analysis
        all_components: dict[str, list[float]] = {}
        for r in self.results:
            for comp_name, comp_value in r.components.items():
                if comp_name not in all_components:
                    all_components[comp_name] = []
                all_components[comp_name].append(comp_value)

        component_stats = {
            name: {
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
            }
            for name, values in all_components.items()
        }

        return {
            "count": len(self.results),
            "average_score": sum(scores) / len(scores),
            "accuracy": correct_count / len(self.results),
            "score_distribution": {
                "excellent": sum(1 for s in scores if s >= 0.8),
                "good": sum(1 for s in scores if 0.6 <= s < 0.8),
                "acceptable": sum(1 for s in scores if 0.4 <= s < 0.6),
                "poor": sum(1 for s in scores if s < 0.4),
            },
            "component_stats": component_stats,
        }


# Convenience function
def grade_action(
    action_taken: str,
    expected_action: str,
    content_score: float,
    violations: list[str],
    expected_violations: list[str] = None,
    step_count: int = 1,
) -> GradingResult:
    """
    Grade a single action using default grader.

    Args:
        action_taken: Action the agent took.
        expected_action: Expected correct action.
        content_score: Compliance score.
        violations: Detected violations.
        expected_violations: Expected violations.
        step_count: Current step.

    Returns:
        GradingResult with score and explanation.
    """
    grader = ContentModerationGrader()
    return grader.grade(
        action_taken=action_taken,
        expected_action=expected_action,
        content_score=content_score,
        violations=violations,
        expected_violations=expected_violations or [],
        step_count=step_count,
    )
