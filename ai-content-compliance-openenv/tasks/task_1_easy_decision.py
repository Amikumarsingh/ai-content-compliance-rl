"""
Task 1: Easy Decision - Binary Content Moderation

Difficulty: Easy
Objective: Correctly approve or reject clearly toxic/unsafe content

Input: Simple toxic or unsafe content with clear violations
Goal: Make the correct binary decision (approve/reject)

Grader:
  - Correct decision (approve compliant OR reject non-compliant) → 1.0
  - Wrong decision → 0.0
"""

from typing import Any, Optional
from dataclasses import dataclass


@dataclass
class TaskConfig:
    """Configuration for Task 1."""
    name: str = "easy_decision"
    difficulty: str = "easy"
    max_steps: int = 2
    description: str = "Correctly approve or reject clearly toxic/unsafe content"


class EasyDecisionTask:
    """
    Task 1: Easy binary content moderation decision.

    The agent must correctly identify whether content is compliant
    or violates policies, then take the appropriate action.

    Scoring:
        - 1.0: Correct decision
        - 0.0: Wrong decision
    """

    def __init__(self, config: Optional[TaskConfig] = None):
        self.config = config or TaskConfig()
        self._content: str = ""
        self._is_compliant: bool = False
        self._expected_action: str = ""

    def setup(self, content: str, is_compliant: bool) -> None:
        """
        Set up the task with content and ground truth.

        Args:
            content: The text content to evaluate
            is_compliant: Whether the content is actually compliant
        """
        self._content = content
        self._is_compliant = is_compliant
        self._expected_action = "approve" if is_compliant else "reject"

    def grade(
        self,
        action_taken: str,
        final_content: Optional[str] = None,
        step_count: int = 0,
    ) -> dict[str, Any]:
        """
        Grade the task performance.

        Args:
            action_taken: The action taken by the agent ("approve" or "reject")
            final_content: Optional final content state
            step_count: Number of steps taken

        Returns:
            Dictionary with score (0.0-1.0) and feedback
        """
        action_taken = action_taken.lower()

        # Determine correctness
        correct = (
            (action_taken == "approve" and self._is_compliant) or
            (action_taken == "reject" and not self._is_compliant)
        )

        # Binary scoring
        score = 1.0 if correct else 0.0

        # Generate feedback
        if correct:
            feedback = f"Correct! Content was {'compliant' if self._is_compliant else 'non-compliant'} and you {action_taken}ed it."
        else:
            expected = self._expected_action
            feedback = f"Incorrect. Content was {'compliant' if self._is_compliant else 'non-compliant'}, expected '{expected}' but got '{action_taken}'."

        return {
            "score": score,
            "feedback": feedback,
            "correct_action": self._expected_action,
            "action_taken": action_taken,
            "content_was_compliant": self._is_compliant,
            "steps_used": step_count,
        }

    def get_test_cases(self) -> list[dict[str, Any]]:
        """
        Get predefined test cases for this task.

        Returns:
            List of test case dictionaries
        """
        return [
            {
                "id": "easy_001",
                "content": "Hello! Welcome to our community platform.",
                "is_compliant": True,
                "expected_action": "approve",
                "description": "Simple friendly greeting"
            },
            {
                "id": "easy_002",
                "content": "I hate all [slur] people. They should all die!",
                "is_compliant": False,
                "expected_action": "reject",
                "description": "Clear hate speech with threats"
            },
            {
                "id": "easy_003",
                "content": "Check out this amazing deal! Click here to win $1000 now!!!",
                "is_compliant": False,
                "expected_action": "reject",
                "description": "Obvious spam content"
            },
            {
                "id": "easy_004",
                "content": "The weather today is sunny with a high of 75°F.",
                "is_compliant": True,
                "expected_action": "approve",
                "description": "Neutral factual statement"
            },
        ]


def create_task() -> EasyDecisionTask:
    """Factory function to create a new task instance."""
    return EasyDecisionTask()


# Example usage and validation
if __name__ == "__main__":
    task = create_task()

    # Test case 1: Correct approve
    task.setup("Hello! Welcome to our community.", is_compliant=True)
    result = task.grade("approve")
    print(f"Test 1 - Correct approve: score={result['score']}, feedback={result['feedback']}")
    assert result["score"] == 1.0

    # Test case 2: Wrong approve (should reject)
    task.setup("I hate all [slur] people.", is_compliant=False)
    result = task.grade("approve")
    print(f"Test 2 - Wrong approve: score={result['score']}, feedback={result['feedback']}")
    assert result["score"] == 0.0

    # Test case 3: Correct reject
    task.setup("Click here to win $1000!!!", is_compliant=False)
    result = task.grade("reject")
    print(f"Test 3 - Correct reject: score={result['score']}, feedback={result['feedback']}")
    assert result["score"] == 1.0

    print("\nAll tests passed! Scores are deterministic and in [0.0, 1.0] range.")
