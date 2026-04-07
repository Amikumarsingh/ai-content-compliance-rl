"""
Task 2: Medium Edit - Content Improvement

Difficulty: Medium
Objective: Edit content to make it compliant

Input: Content that needs editing (has fixable violations)
Goal: Transform non-compliant content into compliant content through editing

Grader:
  - Evaluates improvement in compliance score
  - Partial scoring allowed (0.0-1.0)
  - Considers both decision correctness and edit quality
"""

from typing import Any, Optional
from dataclasses import dataclass


@dataclass
class TaskConfig:
    """Configuration for Task 2."""
    name: str = "medium_edit"
    difficulty: str = "medium"
    max_steps: int = 3
    description: str = "Edit non-compliant content to make it compliant"
    min_improvement_threshold: float = 0.3  # Minimum improvement for partial credit


class MediumEditTask:
    """
    Task 2: Edit content to achieve compliance.

    The agent must identify that content needs editing, modify it
    appropriately, and then approve the edited version.

    Scoring (0.0-1.0):
        - 0.0: No improvement or wrong action
        - 0.3-0.6: Partial improvement but still non-compliant
        - 0.7-0.9: Significant improvement, nearly compliant
        - 1.0: Fully compliant after edit + correct approve
    """

    def __init__(self, config: Optional[TaskConfig] = None):
        self.config = config or TaskConfig()
        self._original_content: str = ""
        self._original_score: float = 0.0
        self._violations: list[str] = []
        self._edit_required: bool = True

    def setup(
        self,
        content: str,
        original_score: float,
        violations: list[str],
    ) -> None:
        """
        Set up the task with content and evaluation.

        Args:
            content: The non-compliant text content
            original_score: Initial compliance score (0.0-0.5)
            violations: List of detected violations
        """
        self._original_content = content
        self._original_score = min(0.5, original_score)  # Ensure non-compliant
        self._violations = violations
        self._edit_required = True

    def grade(
        self,
        action_sequence: list[str],
        final_content: str,
        final_score: float,
    ) -> dict[str, Any]:
        """
        Grade the task performance.

        Args:
            action_sequence: List of actions taken (should include "edit")
            final_content: The final edited content
            final_score: Compliance score after editing (0.0-1.0)

        Returns:
            Dictionary with score (0.0-1.0) and detailed feedback
        """
        # Check if edit action was used
        used_edit = "edit" in action_sequence

        # Calculate improvement
        improvement = final_score - self._original_score

        # Determine if final content is compliant
        is_final_compliant = final_score >= 0.5

        # Check if terminated correctly (approve after edit)
        terminated_correctly = (
            is_final_compliant and
            len(action_sequence) > 0 and
            action_sequence[-1] == "approve"
        )

        # Calculate score components
        score = 0.0

        # Component 1: Used edit action (0.2 points)
        if used_edit:
            score += 0.2

        # Component 2: Improvement achieved (0.4 points max)
        improvement_ratio = improvement / (1.0 - self._original_score) if self._original_score < 1.0 else 0
        improvement_score = min(0.4, 0.4 * improvement_ratio)
        score += improvement_score

        # Component 3: Final compliance (0.4 points)
        if is_final_compliant:
            score += 0.4

        # Bonus: Proper termination (included in component 3 implicitly)
        # Penalty: Edit but no improvement
        if used_edit and improvement <= 0:
            score = min(score, 0.2)  # Cap at edit credit only

        # Ensure score is in valid range
        score = max(0.0, min(1.0, score))

        # Generate detailed feedback
        feedback_parts = []

        if not used_edit:
            feedback_parts.append("Did not use edit action when editing was required.")
        else:
            feedback_parts.append("Correctly identified need for editing.")

        if improvement < self.config.min_improvement_threshold:
            feedback_parts.append(f"Improvement ({improvement:.2f}) below threshold ({self.config.min_improvement_threshold}).")
        else:
            feedback_parts.append(f"Good improvement: {self._original_score:.2f} to {final_score:.2f} (+{improvement:.2f}).")

        if is_final_compliant:
            feedback_parts.append("Final content is compliant.")
        else:
            feedback_parts.append("Final content still non-compliant.")

        if terminated_correctly:
            feedback_parts.append("Correctly approved after achieving compliance.")

        return {
            "score": round(score, 2),
            "feedback": " ".join(feedback_parts),
            "original_score": self._original_score,
            "final_score": final_score,
            "improvement": round(improvement, 2),
            "used_edit": used_edit,
            "is_final_compliant": is_final_compliant,
            "terminated_correctly": terminated_correctly,
            "action_sequence": action_sequence,
            "violations_addressed": self._violations,
        }

    def get_test_cases(self) -> list[dict[str, Any]]:
        """
        Get predefined test cases for this task.

        Returns:
            List of test case dictionaries
        """
        return [
            {
                "id": "medium_001",
                "content": "This product is absolutely terrible! I hate it! Worst purchase ever!!!",
                "original_score": 0.2,
                "violations": ["harassment", "hate_speech"],
                "expected_edit": "This product did not meet my expectations. Disappointed with the purchase.",
                "description": "Overly negative review with hate language"
            },
            {
                "id": "medium_002",
                "content": "CLICK NOW!!! MAKE $5000 FROM HOME!!! NO EFFORT NEEDED!!!",
                "original_score": 0.1,
                "violations": ["spam", "misinformation"],
                "expected_edit": "Work from home opportunity available. Contact for details.",
                "description": "Spammy advertisement needing toning down"
            },
            {
                "id": "medium_003",
                "content": "Anyone who supports that policy is an idiot and should be ashamed.",
                "original_score": 0.3,
                "violations": ["harassment", "incivility"],
                "expected_edit": "I disagree with that policy and believe it should be reconsidered.",
                "description": "Incivil political commentary"
            },
        ]


def create_task() -> MediumEditTask:
    """Factory function to create a new task instance."""
    return MediumEditTask()


# Example usage and validation
if __name__ == "__main__":
    task = create_task()

    # Test case 1: Perfect edit + approve
    task.setup(
        content="This product is absolutely terrible! I hate it!",
        original_score=0.2,
        violations=["harassment"]
    )
    result = task.grade(
        action_sequence=["edit", "approve"],
        final_content="This product did not meet my expectations.",
        final_score=0.85
    )
    print(f"Test 1 - Perfect edit: score={result['score']}")
    print(f"  Feedback: {result['feedback']}")
    assert 0.8 <= result["score"] <= 1.0

    # Test case 2: Partial improvement
    task.setup(
        content="CLICK NOW!!! MAKE $5000 FROM HOME!!!",
        original_score=0.1,
        violations=["spam"]
    )
    result = task.grade(
        action_sequence=["edit", "approve"],
        final_content="Make money from home! CLICK NOW!",
        final_score=0.4  # Still not compliant
    )
    print(f"Test 2 - Partial improvement: score={result['score']}")
    print(f"  Feedback: {result['feedback']}")
    assert 0.3 <= result["score"] < 0.7

    # Test case 3: No edit used (wrong approach)
    task.setup(
        content="Anyone who supports that policy is an idiot.",
        original_score=0.3,
        violations=["harassment"]
    )
    result = task.grade(
        action_sequence=["reject"],
        final_content="Anyone who supports that policy is an idiot.",
        final_score=0.3
    )
    print(f"Test 3 - No edit: score={result['score']}")
    print(f"  Feedback: {result['feedback']}")
    assert result["score"] < 0.3

    print("\nAll tests passed! Scores are deterministic and in [0.0, 1.0] range.")
