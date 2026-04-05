"""
Task 3: Hard Complex - Multi-Issue Content Resolution

Difficulty: Hard
Objective: Detect violations, edit content, and ensure final compliance

Input: Complex content with multiple overlapping violations
Goal:
  1. Correctly identify all violations
  2. Edit content to address all issues
  3. Achieve final compliance

Grader:
  - Correctness: Accurate violation detection
  - Quality: Edit effectiveness
  - Completeness: All issues addressed
  - Final score: 0.0-1.0 based on combined metrics
"""

from typing import Any, Optional
from dataclasses import dataclass, field


@dataclass
class TaskConfig:
    """Configuration for Task 3."""
    name: str = "hard_complex"
    difficulty: str = "hard"
    max_steps: int = 5
    description: str = "Resolve complex multi-issue content violations"

    # Scoring weights
    detection_weight: float = 0.3  # Weight for violation detection
    edit_weight: float = 0.4       # Weight for edit quality
    compliance_weight: float = 0.3  # Weight for final compliance


class ViolationDetector:
    """Simple rule-based violation detector for grading."""

    VIOLATION_PATTERNS = {
        "hate_speech": ["hate", "slur", "die", "deserve to die"],
        "violence": ["kill", "attack", "hurt", "violence", "weapon"],
        "harassment": ["idiot", "stupid", "pathetic", "worthless"],
        "adult_content": ["explicit", "adult", "nsfw"],
        "misinformation": ["fake news", "conspiracy", "hoax", "they hide"],
        "spam": ["click here", "buy now", "!!!", "limited time"],
        "illegal_content": ["illegal", "drugs", "weapon"],
    }

    @classmethod
    def detect(cls, content: str) -> list[str]:
        """Detect violations in content."""
        content_lower = content.lower()
        detected = []
        for violation, patterns in cls.VIOLATION_PATTERNS.items():
            for pattern in patterns:
                if pattern in content_lower:
                    detected.append(violation)
                    break
        return detected


class HardComplexTask:
    """
    Task 3: Complex multi-issue content resolution.

    The agent must:
    1. Detect all violations in complex content
    2. Choose appropriate actions (edit to fix issues)
    3. Ensure final content is fully compliant
    4. Terminate correctly with approve

    Scoring (0.0-1.0):
        - Violation Detection (30%): Correctly identified all violations
        - Edit Quality (40%): Effectively removed/neutralized violations
        - Final Compliance (30%): Achieved compliant final state
    """

    def __init__(self, config: Optional[TaskConfig] = None):
        self.config = config or TaskConfig()
        self._original_content: str = ""
        self._true_violations: list[str] = []
        self._step_limit: int = 5

    def setup(
        self,
        content: str,
        true_violations: list[str],
    ) -> None:
        """
        Set up the task with complex content.

        Args:
            content: The multi-violation text content
            true_violations: Ground truth list of violations
        """
        self._original_content = content
        self._true_violations = true_violations
        self._step_limit = self.config.max_steps

    def grade(
        self,
        action_sequence: list[str],
        detected_violations: list[str],
        final_content: str,
        final_score: float,
    ) -> dict[str, Any]:
        """
        Grade the complete task performance.

        Args:
            action_sequence: List of actions taken
            detected_violations: Violations identified by agent
            final_content: Final edited content
            final_score: Compliance score of final content (0.0-1.0)

        Returns:
            Dictionary with score (0.0-1.0) and detailed breakdown
        """
        # === Component 1: Violation Detection (30%) ===
        true_set = set(self._true_violations)
        detected_set = set(detected_violations)

        # True positives, false positives, false negatives
        true_positives = len(true_set & detected_set)
        false_positives = len(detected_set - true_set)
        false_negatives = len(true_set - detected_set)

        # Precision, recall, F1 for detection
        precision = true_positives / len(detected_set) if detected_set else 0
        recall = true_positives / len(true_set) if true_set else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        detection_score = f1 * self.config.detection_weight

        # === Component 2: Edit Quality (40%) ===
        # Check if edit was used
        used_edit = "edit" in action_sequence

        # Count violations remaining in final content
        final_violations = ViolationDetector.detect(final_content)
        violations_remaining = len(set(final_violations) & true_set)
        violations_removed = len(true_set) - violations_remaining

        # Edit quality metrics
        edit_quality = 0.0
        if used_edit:
            # Base credit for attempting edit
            edit_quality = 0.1

            # Credit for violations removed
            if len(true_set) > 0:
                removal_ratio = violations_removed / len(true_set)
                edit_quality += 0.2 * removal_ratio

            # Credit for no new violations introduced
            new_violations = set(final_violations) - true_set
            if not new_violations:
                edit_quality += 0.1
        else:
            # Penalty for not editing when needed
            edit_quality = 0.0

        edit_score = edit_quality * self.config.edit_weight / 0.4  # Normalize to weight

        # === Component 3: Final Compliance (30%) ===
        is_compliant = final_score >= 0.5
        compliance_score = (1.0 if is_compliant else 0.0) * self.config.compliance_weight

        # Bonus for proper termination
        termination_bonus = 0.0
        if is_compliant and action_sequence and action_sequence[-1] == "approve":
            termination_bonus = 0.1  # Extra 10% bonus
        elif not is_compliant and action_sequence and action_sequence[-1] == "approve":
            termination_bonus = -0.1  # Penalty for premature approve

        # === Calculate Total Score ===
        total_score = detection_score + edit_score + compliance_score + termination_bonus
        total_score = max(0.0, min(1.0, total_score))  # Clamp to [0.0, 1.0]

        # === Generate Detailed Feedback ===
        feedback_parts = []

        # Detection feedback
        if recall >= 0.8:
            feedback_parts.append(f"Excellent violation detection ({recall:.0%} recall).")
        elif recall >= 0.5:
            feedback_parts.append(f"Partial violation detection ({recall:.0%} recall, missed {false_negatives}).")
        else:
            feedback_parts.append(f"Poor violation detection ({recall:.0%} recall).")

        if false_positives > 0:
            feedback_parts.append(f"False positives: {false_positives}.")

        # Edit feedback
        if not used_edit:
            feedback_parts.append("CRITICAL: Did not edit when editing was required.")
        elif violations_remaining > 0:
            feedback_parts.append(f"Edit incomplete: {violations_remaining} violations remain.")
        else:
            feedback_parts.append("All violations successfully addressed in edit.")

        # Compliance feedback
        if is_compliant:
            feedback_parts.append(f"Final content is compliant (score: {final_score:.2f}).")
        else:
            feedback_parts.append(f"Final content non-compliant (score: {final_score:.2f}).")

        return {
            "score": round(total_score, 2),
            "feedback": " ".join(feedback_parts),
            "breakdown": {
                "detection_score": round(detection_score, 3),
                "edit_score": round(edit_score, 3),
                "compliance_score": round(compliance_score, 3),
                "termination_bonus": round(termination_bonus, 3),
            },
            "detection_metrics": {
                "precision": round(precision, 2),
                "recall": round(recall, 2),
                "f1": round(f1, 2),
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
            },
            "edit_metrics": {
                "used_edit": used_edit,
                "violations_removed": violations_removed,
                "violations_remaining": violations_remaining,
                "new_violations_introduced": len(set(final_violations) - true_set),
            },
            "final_state": {
                "is_compliant": is_compliant,
                "final_score": final_score,
                "action_sequence": action_sequence,
                "steps_used": len(action_sequence),
            },
        }

    def get_test_cases(self) -> list[dict[str, Any]]:
        """
        Get predefined test cases for this task.

        Returns:
            List of test case dictionaries
        """
        return [
            {
                "id": "hard_001",
                "content": "I hate those idiots pushing fake news! They should all die! Click here to learn the TRUTH they don't want you to know!!!",
                "true_violations": ["hate_speech", "harassment", "misinformation", "violence", "spam"],
                "expected_edit": "I disagree with those people and their policies. I believe their information is inaccurate. Learn more about alternative perspectives.",
                "description": "Multi-violation post with hate, harassment, misinformation, violence, and spam"
            },
            {
                "id": "hard_002",
                "content": "BUY NOW! This illegal conspiracy product will expose the hoax! Make $5000 selling this worthless garbage! Limited time offer!!!",
                "true_violations": ["spam", "misinformation", "illegal_content", "harassment"],
                "expected_edit": "This product offers alternative perspectives on current events. Contact for business opportunities.",
                "description": "Spam with illegal claims, conspiracy theories, and harassment"
            },
            {
                "id": "hard_003",
                "content": "Those pathetic people spreading violence and hate should be attacked. Kill the messenger! This is explicit adult content disguised as news.",
                "true_violations": ["harassment", "violence", "hate_speech", "adult_content"],
                "expected_edit": "I disagree with their message and find their approach problematic. This content appears misleading.",
                "description": "Complex mix of harassment, violence, hate speech, and inappropriate content"
            },
        ]


def create_task() -> HardComplexTask:
    """Factory function to create a new task instance."""
    return HardComplexTask()


# Example usage and validation
if __name__ == "__main__":
    task = create_task()

    # Test case 1: Perfect performance
    task.setup(
        content="I hate those idiots pushing fake news! They should all die!",
        true_violations=["hate_speech", "harassment", "violence", "misinformation"]
    )
    result = task.grade(
        action_sequence=["edit", "approve"],
        detected_violations=["hate_speech", "harassment", "violence", "misinformation"],
        final_content="I disagree with their message and find their approach problematic.",
        final_score=0.85
    )
    print(f"Test 1 - Perfect performance: score={result['score']}")
    print(f"  Breakdown: {result['breakdown']}")
    print(f"  Feedback: {result['feedback']}")
    assert result["score"] >= 0.9

    # Test case 2: Partial detection, good edit
    task.setup(
        content="BUY NOW! This illegal conspiracy product!",
        true_violations=["spam", "illegal_content", "misinformation"]
    )
    result = task.grade(
        action_sequence=["edit", "approve"],
        detected_violations=["spam"],  # Missed 2 violations
        final_content="This product offers business opportunities.",
        final_score=0.75
    )
    print(f"Test 2 - Partial detection: score={result['score']}")
    print(f"  Detection: {result['detection_metrics']}")
    # Score can be high if edit quality and compliance are good, even with partial detection
    assert 0.4 <= result["score"] <= 1.0

    # Test case 3: No edit, wrong approach
    task.setup(
        content="Those pathetic people should be attacked.",
        true_violations=["harassment", "violence"]
    )
    result = task.grade(
        action_sequence=["reject"],
        detected_violations=["harassment"],
        final_content="Those pathetic people should be attacked.",
        final_score=0.2
    )
    print(f"Test 3 - No edit: score={result['score']}")
    print(f"  Feedback: {result['feedback']}")
    assert result["score"] < 0.3

    print("\nAll tests passed! Scores are deterministic and in [0.0, 1.0] range.")
