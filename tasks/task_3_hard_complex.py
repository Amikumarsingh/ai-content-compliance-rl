"""
Task 3: Hard Complex - Multi-Issue Content Resolution (Enhanced)

Difficulty: Hard - Challenging even for strong LLMs
Objective: Detect ALL violations, edit comprehensively, ensure compliance

Key Challenges:
1. Multiple overlapping violations (3-5 per content)
2. Subtle/ambiguous violations requiring nuanced detection
3. Partial credit scoring (detect 1/3 -> 0.3, detect all -> 1.0)
4. Edit must address ALL issues while preserving meaning
5. Deterministic grading with 0.0-1.0 score

Grader Requirements:
- Detect ALL violations correctly
- Choose correct action sequence
- Edit must remove all violations
- Final content must be compliant
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class TaskConfig:
    """Configuration for Task 3 Hard Complex."""

    name: str = "hard_complex"
    difficulty: str = "hard"
    max_steps: int = 5
    description: str = "Resolve complex multi-issue content violations"

    # Scoring weights
    detection_weight: float = 0.35  # Weight for violation detection
    edit_weight: float = 0.35  # Weight for edit quality
    compliance_weight: float = 0.30  # Weight for final compliance

    # Partial scoring thresholds
    partial_score_thresholds: dict[str, float] = field(default_factory=lambda: {
        "excellent": 1.0,   # All violations detected
        "good": 0.7,        # 70%+ violations detected
        "partial": 0.4,     # 40%+ violations detected
        "poor": 0.1,        # Less than 40% detected
    })


class ViolationDetector:
    """
    Enhanced rule-based violation detector for grading.

    Includes subtle patterns to catch ambiguous violations.
    Deterministic - same input always produces same output.
    """

    # Expanded pattern mappings with subtle indicators
    VIOLATION_PATTERNS: dict[str, list[str]] = {
        "hate_speech": [
            # Obvious
            "hate", "slur", "should all die", "deserve to die", "kill all",
            # Subtle
            "group of people", "they're all the same", "none of them",
        ],
        "violence": [
            # Obvious
            "kill", "attack", "hurt", "violence", "weapon", "should be attacked",
            # Subtle
            "take action against", "make them pay", "consequences",
        ],
        "harassment": [
            # Obvious
            "idiot", "stupid", "pathetic", "worthless", "terrible", "garbage",
            # Subtle
            "sad.", "ignorant", "delusional", "misguided", "too [a-z]+ to understand",
        ],
        "adult_content": [
            # Obvious
            "explicit", "adult content", "nsfw", "porn",
            # Subtle
            "dm for pics", "18+", "private content",
        ],
        "misinformation": [
            # Obvious
            "fake news", "conspiracy", "hoax", "they hide", "truth they don't want",
            # Subtle
            "they don't want you to know", "wake up", "mainstream media lies",
            "do your own research", "question the narrative",
        ],
        "spam": [
            # Obvious
            "click here", "buy now", "!!!", "limited time", "make $", "win $",
            # Subtle
            "visit ", "dm me", "free", "offer", "deal", "prizes", "exclusive",
            "bit.ly", "tinyurl", "linktr.ee",
        ],
        "engagement_bait": [
            # Obvious
            "tag someone", "share if", "like this post", "comment '",
            # Subtle
            "agree?", "thoughts?", "change my mind", "let me know",
        ],
        "illegal_content": [
            # Obvious
            "illegal", "drugs", "weapon",
            # Subtle
            "under the table", "off the books", "no questions asked",
        ],
    }

    # Ambiguity markers - content that MAY be violations depending on context
    AMBIGUOUS_PATTERNS: dict[str, list[str]] = {
        "potentially_hate": ["disagree with", "problem with", "issue with"],
        "potentially_spam": ["learn more", "check out", "visit our"],
        "potentially_misinfo": ["some people say", "allegedly", "reportedly"],
    }

    @classmethod
    def detect(cls, content: str) -> list[str]:
        """
        Detect violations in content.

        Args:
            content: Text content to analyze

        Returns:
            List of detected violation types (deterministic)
        """
        content_lower = content.lower()
        detected = []

        for violation, patterns in cls.VIOLATION_PATTERNS.items():
            for pattern in patterns:
                if pattern in content_lower:
                    detected.append(violation)
                    break  # One match per violation type

        return detected

    @classmethod
    def detect_with_ambiguity(
        cls,
        content: str,
    ) -> tuple[list[str], list[str]]:
        """
        Detect violations with ambiguity flagging.

        Args:
            content: Text content to analyze

        Returns:
            Tuple of (definite_violations, ambiguous_violations)
        """
        content_lower = content.lower()
        definite = []
        ambiguous = []

        # Check definite patterns
        for violation, patterns in cls.VIOLATION_PATTERNS.items():
            for pattern in patterns:
                if pattern in content_lower:
                    definite.append(violation)
                    break

        # Check ambiguous patterns
        for amb_type, patterns in cls.AMBIGUOUS_PATTERNS.items():
            for pattern in patterns:
                if pattern in content_lower:
                    # Map ambiguous type to main violation type
                    if "hate" in amb_type:
                        ambiguous.append("hate_speech")
                    elif "spam" in amb_type:
                        ambiguous.append("spam")
                    elif "misinfo" in amb_type:
                        ambiguous.append("misinformation")
                    break

        return definite, ambiguous

    @classmethod
    def count_severity(cls, content: str) -> dict[str, int]:
        """
        Count violation severity levels.

        Returns:
            Dict with severity counts
        """
        content_lower = content.lower()
        severity = {"high": 0, "medium": 0, "low": 0}

        # High severity: hate, violence, illegal
        high_severity = ["hate_speech", "violence", "illegal_content"]
        # Medium severity: harassment, misinformation, adult
        medium_severity = ["harassment", "misinformation", "adult_content"]
        # Low severity: spam, engagement bait
        low_severity = ["spam", "engagement_bait"]

        violations = cls.detect(content)

        for v in violations:
            if v in high_severity:
                severity["high"] += 1
            elif v in medium_severity:
                severity["medium"] += 1
            elif v in low_severity:
                severity["low"] += 1

        return severity


class HardComplexTask:
    """
    Task 3: Complex multi-issue content resolution.

    The agent must:
    1. Detect ALL violations (partial credit for partial detection)
    2. Choose appropriate action sequence
    3. Edit content to address ALL issues
    4. Achieve full compliance

    Scoring (0.0-1.0):
        - Violation Detection (35%): Partial credit based on recall
        - Edit Quality (35%): Violations removed, meaning preserved
        - Final Compliance (30%): Achieved compliant state

    Partial Scoring for Detection:
        - 100% of violations: 1.0 * weight
        - 70-99% of violations: 0.7 * weight
        - 40-69% of violations: 0.4 * weight
        - <40% of violations: 0.1 * weight
    """

    def __init__(self, config: Optional[TaskConfig] = None):
        self.config = config or TaskConfig()
        self._original_content: str = ""
        self._true_violations: list[str] = []
        self._step_limit: int = 5
        self._severity: dict[str, int] = {}

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
        self._true_violations = list(true_violations)  # Ensure copy
        self._step_limit = self.config.max_steps
        self._severity = ViolationDetector.count_severity(content)

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
            action_sequence: List of actions taken (e.g., ["edit", "approve"])
            detected_violations: Violations identified by agent
            final_content: Final edited content
            final_score: Compliance score of final content (0.0-1.0)

        Returns:
            Dictionary with score (0.0-1.0) and detailed breakdown
        """
        true_set = set(self._true_violations)
        detected_set = set(detected_violations)

        # === Component 1: Violation Detection (35%) with Partial Credit ===
        true_positives = len(true_set & detected_set)
        false_positives = len(detected_set - true_set)
        false_negatives = len(true_set - detected_set)

        # Precision, recall, F1 for detection
        precision = true_positives / len(detected_set) if detected_set else 0.0
        recall = true_positives / len(true_set) if true_set else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # Partial credit based on recall threshold
        if recall >= 1.0:
            detection_multiplier = 1.0  # Perfect detection
        elif recall >= 0.7:
            detection_multiplier = 0.7  # Good detection
        elif recall >= 0.4:
            detection_multiplier = 0.4  # Partial detection
        else:
            detection_multiplier = 0.1  # Poor detection

        # Adjust for false positives (penalty)
        fp_penalty = min(0.2, false_positives * 0.05)
        detection_multiplier = max(0.0, detection_multiplier - fp_penalty)

        detection_score = detection_multiplier * self.config.detection_weight

        # Severity bonus: extra credit for detecting high-severity violations
        severity_bonus = 0.0
        high_severity_types = {"hate_speech", "violence", "illegal_content"}
        detected_high_severity = true_set & detected_set & high_severity_types
        if detected_high_severity:
            severity_bonus = 0.05 * len(detected_high_severity)
            detection_score += severity_bonus

        detection_score = min(1.0, detection_score + severity_bonus)

        # === Component 2: Edit Quality (35%) ===
        used_edit = "edit" in action_sequence

        # Count violations remaining in final content
        final_violations = ViolationDetector.detect(final_content)
        final_violation_set = set(final_violations)

        # Violations that were in original but remain in final
        violations_remaining = len(true_set & final_violation_set)
        violations_removed = len(true_set) - violations_remaining

        # New violations introduced by edit
        new_violations = final_violation_set - true_set

        edit_quality = 0.0
        if used_edit:
            # Base credit for attempting edit
            edit_quality = 0.15

            # Credit for violations removed (proportional)
            if len(true_set) > 0:
                removal_ratio = violations_removed / len(true_set)
                edit_quality += 0.35 * removal_ratio

            # Credit for no new violations introduced
            if not new_violations:
                edit_quality += 0.15
            else:
                # Penalty for new violations
                edit_quality -= 0.1 * len(new_violations)

            # Bonus for complete removal
            if violations_remaining == 0 and len(true_set) > 0:
                edit_quality += 0.15  # Complete removal bonus
        else:
            # No edit used
            if len(true_set) > 0:
                # Penalty for not editing when violations exist
                edit_quality = 0.0
            else:
                # No violations to fix, edit not needed
                edit_quality = 0.30

        edit_quality = max(0.0, min(1.0, edit_quality))
        edit_score = edit_quality * self.config.edit_weight

        # === Component 3: Final Compliance (30%) ===
        is_compliant = final_score >= 0.5 and len(final_violations) == 0

        if is_compliant:
            compliance_score = self.config.compliance_weight
        else:
            # Partial credit for improvement
            if final_score >= 0.5:
                compliance_score = self.config.compliance_weight * 0.5
            else:
                compliance_score = 0.0

        # === Termination Bonus/Penalty ===
        termination_bonus = 0.0
        if action_sequence:
            last_action = action_sequence[-1]

            if is_compliant and last_action == "approve":
                termination_bonus = 0.05  # Small bonus for correct termination
            elif not is_compliant and last_action == "approve":
                termination_bonus = -0.1  # Penalty for premature approve
            elif last_action == "reject" and len(true_set) > 0:
                termination_bonus = 0.02  # Small bonus for appropriate reject

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

        if severity_bonus > 0:
            feedback_parts.append(f"Severity bonus: +{severity_bonus:.2f} for detecting high-severity violations.")

        # Edit feedback
        if not used_edit and len(true_set) > 0:
            feedback_parts.append("CRITICAL: Did not edit when editing was required.")
        elif violations_remaining > 0:
            feedback_parts.append(f"Edit incomplete: {violations_remaining} violations remain: {list(true_set & final_violation_set)}.")
        elif len(true_set) > 0:
            feedback_parts.append("All violations successfully addressed in edit.")

        if new_violations:
            feedback_parts.append(f"WARNING: Edit introduced {len(new_violations)} new violations: {list(new_violations)}.")

        # Compliance feedback
        if is_compliant:
            feedback_parts.append(f"Final content is compliant (score: {final_score:.2f}).")
        else:
            feedback_parts.append(f"Final content non-compliant (score: {final_score:.2f}).")

        # Severity info
        if self._severity["high"] > 0:
            feedback_parts.append(f"Content had {self._severity['high']} high-severity violations.")

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
                "partial_credit_multiplier": detection_multiplier,
            },
            "edit_metrics": {
                "used_edit": used_edit,
                "violations_removed": violations_removed,
                "violations_remaining": violations_remaining,
                "new_violations_introduced": len(new_violations),
                "edit_quality_raw": round(edit_quality, 3),
            },
            "severity_info": self._severity,
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

        Each case has 3-5 violations with varying severity.
        """
        return [
            {
                "id": "hard_001",
                "content": "I hate those idiots pushing fake news! They should all die! Click here to learn the TRUTH they don't want you to know!!!",
                "true_violations": ["hate_speech", "harassment", "misinformation", "violence", "spam"],
                "expected_edit": "I disagree with those people and their messaging. I believe their information is inaccurate. Learn more about alternative perspectives.",
                "description": "Multi-violation: hate, harassment, misinformation, violence, spam",
                "severity": {"high": 2, "medium": 2, "low": 1},
            },
            {
                "id": "hard_002",
                "content": "BUY NOW! This illegal conspiracy product will expose the hoax! Make $5000 selling this worthless garbage! Limited time offer!!!",
                "true_violations": ["spam", "misinformation", "illegal_content", "harassment"],
                "expected_edit": "This product offers alternative perspectives on current events. Contact for business opportunities.",
                "description": "Spam with illegal claims, conspiracy theories, and harassment",
                "severity": {"high": 1, "medium": 2, "low": 1},
            },
            {
                "id": "hard_003",
                "content": "Those pathetic people spreading violence and hate should be attacked. Kill the messenger! This is explicit adult content disguised as news.",
                "true_violations": ["harassment", "violence", "hate_speech", "adult_content"],
                "expected_edit": "I disagree with their message and find their approach problematic. This content appears misleading.",
                "description": "Complex mix of harassment, violence, hate speech, and adult content",
                "severity": {"high": 2, "medium": 2, "low": 0},
            },
            {
                "id": "hard_004",
                "content": "Wake up sheeple! They don't want you to know about this conspiracy. Tag 3 friends who need to see this! Visit bit.ly/truth for more!!!",
                "true_violations": ["misinformation", "engagement_bait", "spam"],
                "expected_edit": "Consider researching alternative perspectives on this topic. Share with friends who might be interested. Learn more at our website.",
                "description": "Misinformation with engagement bait and spam links",
                "severity": {"high": 0, "medium": 1, "low": 2},
            },
            {
                "id": "hard_005",
                "content": "These stupid politicians are pushing illegal policies. They should all be attacked. Do your own research - the mainstream media lies!",
                "true_violations": ["harassment", "illegal_content", "violence", "misinformation"],
                "expected_edit": "These politicians' policies are problematic. We should engage in peaceful discourse. I encourage looking at multiple information sources.",
                "description": "Harassment, illegal content references, violence, and misinformation",
                "severity": {"high": 2, "medium": 2, "low": 0},
            },
        ]


def create_task() -> HardComplexTask:
    """Factory function to create a new task instance."""
    return HardComplexTask()


# Example usage and validation
if __name__ == "__main__":
    task = create_task()

    # Test case 1: Perfect performance (all violations detected, good edit)
    print("Test 1: Perfect performance")
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
    print(f"  Score: {result['score']}")
    print(f"  Feedback: {result['feedback']}")
    assert result["score"] >= 0.9, f"Expected >= 0.9, got {result['score']}"
    print("  PASSED\n")

    # Test case 2: Partial detection (only 50% of violations)
    print("Test 2: Partial detection (50%)")
    task.setup(
        content="BUY NOW! This illegal conspiracy product!",
        true_violations=["spam", "illegal_content", "misinformation"]
    )
    result = task.grade(
        action_sequence=["edit", "approve"],
        detected_violations=["spam"],  # Only 1/3 = 33%
        final_content="This product offers business opportunities.",
        final_score=0.75
    )
    print(f"  Score: {result['score']}")
    print(f"  Detection recall: {result['detection_metrics']['recall']}")
    print(f"  Feedback: {result['feedback']}")
    # Should get partial credit (0.4 multiplier for detection)
    assert 0.3 <= result["score"] <= 0.8, f"Expected 0.3-0.8, got {result['score']}"
    print("  PASSED\n")

    # Test case 3: No edit when required
    print("Test 3: No edit when violations exist")
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
    print(f"  Score: {result['score']}")
    print(f"  Feedback: {result['feedback']}")
    assert result["score"] < 0.3, f"Expected < 0.3, got {result['score']}"
    print("  PASSED\n")

    # Test case 4: Edit introduces new violations
    print("Test 4: Edit introduces new violations")
    task.setup(
        content="I disagree with this policy.",
        true_violations=[]  # Originally clean
    )
    result = task.grade(
        action_sequence=["edit", "approve"],
        detected_violations=[],
        final_content="🔥 CLICK HERE for FREE PRIZES!!! bit.ly/1234 🔥",  # Made it spam!
        final_score=0.2
    )
    print(f"  Score: {result['score']}")
    print(f"  New violations: {result['edit_metrics']['new_violations_introduced']}")
    print(f"  Feedback: {result['feedback']}")
    assert result["score"] < 0.3, f"Expected < 0.3, got {result['score']}"
    print("  PASSED\n")

    # Test case 5: Perfect with severity bonus
    print("Test 5: Perfect with high-severity detection")
    task.setup(
        content="Kill them all! These idiots deserve to die!",
        true_violations=["violence", "hate_speech", "harassment"]
    )
    result = task.grade(
        action_sequence=["edit", "approve"],
        detected_violations=["violence", "hate_speech", "harassment"],
        final_content="I strongly disagree with their position.",
        final_score=0.9
    )
    print(f"  Score: {result['score']}")
    print(f"  Severity bonus applied: {result['detection_metrics'].get('partial_credit_multiplier', 'N/A')}")
    print(f"  Feedback: {result['feedback']}")
    assert result["score"] >= 0.95, f"Expected >= 0.95, got {result['score']}"
    print("  PASSED\n")

    print("=" * 60)
    print("All tests passed! Scores are deterministic and in [0.0, 1.0] range.")
    print("Partial scoring working correctly based on violation detection recall.")
