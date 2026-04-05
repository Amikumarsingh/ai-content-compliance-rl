"""
OpenEnv Task Grader System.

Provides deterministic grading functions for all content compliance tasks.
All scores are normalized to [0.0, 1.0] range with no randomness.

Features:
- Deterministic scoring (no randomness)
- Reproducible results
- Normalized scores (0.0-1.0)
- Per-task grading functions

Example:
    >>> from graders.grader import grade_task, GradeResult
    >>> result = grade_task(
    ...     task_name="easy_decision",
    ...     ground_truth={"is_compliant": True},
    ...     agent_output={"action_taken": "approve"}
    ... )
    >>> print(result.score)  # 1.0
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from graders.violation_detector import ViolationDetector


@dataclass
class GradeResult:
    """Result of grading a task."""

    score: float  # Normalized 0.0-1.0
    feedback: str
    breakdown: dict[str, float] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    passed: bool = False
    passed_threshold: float = 0.7  # Default passing threshold

    def __post_init__(self):
        # Ensure score is in valid range
        self.score = max(0.0, min(1.0, self.score))
        # Determine if passed
        self.passed = self.score >= self.passed_threshold

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "score": self.score,
            "feedback": self.feedback,
            "breakdown": self.breakdown,
            "metrics": self.metrics,
            "passed": self.passed,
        }


class TaskGrader:
    """
    Deterministic grader for content compliance tasks.

    All scoring is rule-based with no randomness to ensure
    reproducible results across runs.
    """

    # =========================================================================
    # Task 1: Easy Decision Grading
    # =========================================================================

    @staticmethod
    def grade_easy_decision(
        ground_truth: dict[str, Any],
        agent_output: dict[str, Any],
    ) -> GradeResult:
        """
        Grade Task 1: Easy binary decision.

        Args:
            ground_truth: {"is_compliant": bool, "expected_action": str}
            agent_output: {"action_taken": str}

        Returns:
            GradeResult with score 0.0 or 1.0

        Scoring:
            - Correct decision: 1.0
            - Wrong decision: 0.0
        """
        is_compliant = ground_truth.get("is_compliant", False)
        expected_action = ground_truth.get("expected_action", "")
        action_taken = agent_output.get("action_taken", "").lower()

        # Determine correctness
        if expected_action:
            correct = action_taken == expected_action.lower()
        else:
            correct = (
                (action_taken == "approve" and is_compliant) or
                (action_taken == "reject" and not is_compliant)
            )

        # Binary scoring
        score = 1.0 if correct else 0.0

        # Generate feedback
        if correct:
            feedback = f"Correct! Action '{action_taken}' matches expected decision."
        else:
            feedback = f"Incorrect. Expected '{expected_action}' but got '{action_taken}'."

        return GradeResult(
            score=score,
            feedback=feedback,
            breakdown={"correctness": score},
            metrics={
                "expected_action": expected_action,
                "action_taken": action_taken,
                "is_compliant": is_compliant,
            },
        )

    # =========================================================================
    # Task 2: Medium Edit Grading
    # =========================================================================

    @staticmethod
    def grade_medium_edit(
        ground_truth: dict[str, Any],
        agent_output: dict[str, Any],
    ) -> GradeResult:
        """
        Grade Task 2: Medium edit task.

        Args:
            ground_truth: {
                "original_content": str,
                "original_score": float,
                "violations": list[str]
            }
            agent_output: {
                "action_sequence": list[str],
                "final_content": str,
                "final_score": float
            }

        Returns:
            GradeResult with score 0.0-1.0

        Scoring Components:
            - Used edit action: 0.2 pts
            - Improvement achieved: 0.4 pts max
            - Final compliance: 0.4 pts
        """
        original_score = ground_truth.get("original_score", 0.0)
        violations = ground_truth.get("violations", [])

        action_sequence = agent_output.get("action_sequence", [])
        final_content = agent_output.get("final_content", "")
        final_score = agent_output.get("final_score", 0.0)

        # Ensure original_score indicates non-compliance
        original_score = min(0.5, original_score)

        # Component 1: Used edit action (0.2 points)
        used_edit = "edit" in [a.lower() for a in action_sequence]
        edit_credit = 0.2 if used_edit else 0.0

        # Component 2: Improvement (0.4 points max)
        improvement = final_score - original_score
        max_possible_improvement = 1.0 - original_score
        improvement_ratio = improvement / max_possible_improvement if max_possible_improvement > 0 else 0
        improvement_score = min(0.4, 0.4 * max(0, improvement_ratio))

        # Component 3: Final compliance (0.4 points)
        is_compliant = final_score >= 0.5
        compliance_score = 0.4 if is_compliant else 0.0

        # Total score
        total_score = edit_credit + improvement_score + compliance_score
        total_score = max(0.0, min(1.0, total_score))

        # Penalty: Edit but no improvement (cap score)
        if used_edit and improvement <= 0:
            total_score = min(total_score, 0.2)

        # Generate feedback
        feedback_parts = []
        if not used_edit:
            feedback_parts.append("Did not use edit action when required.")
        else:
            feedback_parts.append("Correctly used edit action.")

        if improvement < 0.3:
            feedback_parts.append(f"Improvement ({improvement:.2f}) below threshold.")
        else:
            feedback_parts.append(f"Good improvement: {original_score:.2f} to {final_score:.2f}.")

        if is_compliant:
            feedback_parts.append("Final content is compliant.")
        else:
            feedback_parts.append("Final content still non-compliant.")

        return GradeResult(
            score=round(total_score, 2),
            feedback=" ".join(feedback_parts),
            breakdown={
                "edit_credit": edit_credit,
                "improvement_score": round(improvement_score, 3),
                "compliance_score": compliance_score,
            },
            metrics={
                "original_score": original_score,
                "final_score": final_score,
                "improvement": round(improvement, 2),
                "used_edit": used_edit,
                "is_compliant": is_compliant,
                "violations": violations,
            },
        )

    # =========================================================================
    # Task 3: Hard Complex Grading
    # =========================================================================

    @staticmethod
    def grade_hard_complex(
        ground_truth: dict[str, Any],
        agent_output: dict[str, Any],
    ) -> GradeResult:
        """
        Grade Task 3: Hard complex multi-issue task.

        Args:
            ground_truth: {
                "original_content": str,
                "true_violations": list[str]
            }
            agent_output: {
                "action_sequence": list[str],
                "detected_violations": list[str],
                "final_content": str,
                "final_score": float
            }

        Returns:
            GradeResult with score 0.0-1.0

        Scoring Components:
            - Violation Detection (30%): F1 score
            - Edit Quality (40%): Violations removed, no new issues
            - Final Compliance (30%): Binary compliant/non-compliant
            - Termination Bonus: +/- 0.1
        """
        true_violations = set(ground_truth.get("true_violations", []))
        original_content = ground_truth.get("original_content", "")

        action_sequence = agent_output.get("action_sequence", [])
        detected_violations = set(agent_output.get("detected_violations", []))
        final_content = agent_output.get("final_content", "")
        final_score = agent_output.get("final_score", 0.0)

        # === Component 1: Violation Detection (30%) ===
        true_positives = len(true_violations & detected_violations)
        false_positives = len(detected_violations - true_violations)
        false_negatives = len(true_violations - detected_violations)

        precision = true_positives / len(detected_violations) if detected_violations else 0
        recall = true_positives / len(true_violations) if true_violations else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        detection_score = f1 * 0.3

        # === Component 2: Edit Quality (40%) ===
        used_edit = "edit" in [a.lower() for a in action_sequence]

        # Detect violations in final content
        final_violations = set(ViolationDetector.detect(final_content))
        violations_remaining = len(final_violations & true_violations)
        violations_removed = len(true_violations) - violations_remaining

        # Check for new violations introduced
        new_violations = final_violations - true_violations

        edit_quality = 0.0
        if used_edit:
            # Base credit for attempting edit
            edit_quality = 0.1

            # Credit for violations removed
            if len(true_violations) > 0:
                removal_ratio = violations_removed / len(true_violations)
                edit_quality += 0.2 * removal_ratio

            # Credit for no new violations
            if not new_violations:
                edit_quality += 0.1
        else:
            edit_quality = 0.0

        edit_score = edit_quality

        # === Component 3: Final Compliance (30%) ===
        is_compliant = final_score >= 0.5
        compliance_score = 0.3 if is_compliant else 0.0

        # === Termination Bonus ===
        termination_bonus = 0.0
        if action_sequence:
            last_action = action_sequence[-1].lower()
            if is_compliant and last_action == "approve":
                termination_bonus = 0.1
            elif not is_compliant and last_action == "approve":
                termination_bonus = -0.1

        # === Total Score ===
        total_score = detection_score + edit_score + compliance_score + termination_bonus
        total_score = max(0.0, min(1.0, total_score))

        # === Generate Feedback ===
        feedback_parts = []

        if recall >= 0.8:
            feedback_parts.append(f"Excellent violation detection ({recall:.0%} recall).")
        elif recall >= 0.5:
            feedback_parts.append(f"Partial violation detection ({recall:.0%} recall).")
        else:
            feedback_parts.append(f"Poor violation detection ({recall:.0%} recall).")

        if not used_edit:
            feedback_parts.append("CRITICAL: Did not edit when required.")
        elif violations_remaining > 0:
            feedback_parts.append(f"Edit incomplete: {violations_remaining} violations remain.")
        else:
            feedback_parts.append("All violations addressed.")

        if is_compliant:
            feedback_parts.append(f"Final content compliant (score: {final_score:.2f}).")
        else:
            feedback_parts.append(f"Final content non-compliant (score: {final_score:.2f}).")

        return GradeResult(
            score=round(total_score, 2),
            feedback=" ".join(feedback_parts),
            breakdown={
                "detection_score": round(detection_score, 3),
                "edit_score": round(edit_score, 3),
                "compliance_score": round(compliance_score, 3),
                "termination_bonus": round(termination_bonus, 3),
            },
            metrics={
                "detection": {
                    "precision": round(precision, 2),
                    "recall": round(recall, 2),
                    "f1": round(f1, 2),
                    "true_positives": true_positives,
                    "false_positives": false_positives,
                    "false_negatives": false_negatives,
                },
                "edit": {
                    "used_edit": used_edit,
                    "violations_removed": violations_removed,
                    "violations_remaining": violations_remaining,
                    "new_violations": len(new_violations),
                },
                "compliance": {
                    "is_compliant": is_compliant,
                    "final_score": final_score,
                },
            },
        )


# =============================================================================
# Public API
# =============================================================================

def grade_task(
    task_name: str,
    ground_truth: dict[str, Any],
    agent_output: dict[str, Any],
) -> GradeResult:
    """
    Grade a task completion.

    Args:
        task_name: One of "easy_decision", "medium_edit", "hard_complex"
        ground_truth: Task-specific ground truth data
        agent_output: Task-specific agent output data

    Returns:
        GradeResult with score (0.0-1.0), feedback, and breakdown

    Raises:
        ValueError: If task_name is not recognized

    Example:
        >>> result = grade_task(
        ...     task_name="easy_decision",
        ...     ground_truth={"is_compliant": True},
        ...     agent_output={"action_taken": "approve"}
        ... )
        >>> assert result.score == 1.0
    """
    graders = {
        "easy_decision": TaskGrader.grade_easy_decision,
        "medium_edit": TaskGrader.grade_medium_edit,
        "hard_complex": TaskGrader.grade_hard_complex,
    }

    if task_name not in graders:
        available = ", ".join(graders.keys())
        raise ValueError(f"Unknown task '{task_name}'. Available: {available}")

    grader_fn = graders[task_name]
    return grader_fn(ground_truth, agent_output)


def normalize_score(raw_score: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """
    Normalize a score to [0.0, 1.0] range.

    Args:
        raw_score: Raw score value
        min_val: Minimum possible raw score
        max_val: Maximum possible raw score

    Returns:
        Normalized score in [0.0, 1.0]
    """
    if max_val == min_val:
        return 0.5
    normalized = (raw_score - min_val) / (max_val - min_val)
    return max(0.0, min(1.0, normalized))


def batch_grade(
    task_name: str,
    test_cases: list[dict[str, Any]],
) -> list[GradeResult]:
    """
    Grade multiple test cases.

    Args:
        task_name: Task name
        test_cases: List of {"ground_truth": ..., "agent_output": ...} dicts

    Returns:
        List of GradeResult objects
    """
    results = []
    for case in test_cases:
        result = grade_task(
            task_name=task_name,
            ground_truth=case["ground_truth"],
            agent_output=case["agent_output"],
        )
        results.append(result)
    return results


def compute_statistics(results: list[GradeResult]) -> dict[str, Any]:
    """
    Compute statistics over grading results.

    Args:
        results: List of GradeResult objects

    Returns:
        Dictionary with mean, median, min, max, pass_rate
    """
    if not results:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "pass_rate": 0.0,
        }

    scores = [r.score for r in results]
    passed_count = sum(1 for r in results if r.passed)

    scores_sorted = sorted(scores)
    n = len(scores_sorted)
    median = (scores_sorted[n // 2 - 1] + scores_sorted[n // 2]) / 2 if n % 2 == 0 else scores_sorted[n // 2]

    return {
        "count": n,
        "mean": sum(scores) / n,
        "median": median,
        "min": min(scores),
        "max": max(scores),
        "pass_rate": passed_count / n,
    }
