"""
Unit Tests for Content Compliance Graders.

Tests for grading logic, violation detection, and score normalization.
"""

import pytest

from graders.grader import GradeResult, TaskGrader, grade_task, normalize_score
from graders.violation_detector import ViolationDetector


class TestViolationDetector:
    """Tests for ViolationDetector."""

    def test_detect_hate_speech(self):
        """Test hate speech detection."""
        content = "I hate all those people. They should all die!"
        violations = ViolationDetector.detect(content)
        assert "hate_speech" in violations

    def test_detect_violence(self):
        """Test violence detection."""
        content = "I will attack them with a weapon!"
        violations = ViolationDetector.detect(content)
        assert "violence" in violations

    def test_detect_harassment(self):
        """Test harassment detection."""
        content = "You are such an idiot and worthless."
        violations = ViolationDetector.detect(content)
        assert "harassment" in violations

    def test_detect_spam(self):
        """Test spam detection."""
        content = "Click here to win $1000 now!!!"
        violations = ViolationDetector.detect(content)
        assert "spam" in violations

    def test_detect_multiple_violations(self):
        """Test multiple violation detection."""
        content = "I hate those idiots! They should all die! Click here!!!"
        violations = ViolationDetector.detect(content)
        assert len(violations) >= 2

    def test_no_violations(self):
        """Test clean content."""
        content = "Hello! Welcome to our community platform."
        violations = ViolationDetector.detect(content)
        assert len(violations) == 0

    def test_has_violation(self):
        """Test has_violation method."""
        content = "Click here to win money!!!"
        assert ViolationDetector.has_violation(content, "spam")
        assert not ViolationDetector.has_violation(content, "hate_speech")

    def test_count_violations(self):
        """Test count_violations method."""
        content = "I hate those idiots! Click here!!!"
        count = ViolationDetector.count_violations(content)
        assert count >= 2


class TestGradeResult:
    """Tests for GradeResult dataclass."""

    def test_score_in_range(self):
        """Test score is clamped to [0.0, 1.0]."""
        result = GradeResult(score=1.5, feedback="test")
        assert result.score == 1.0

        result = GradeResult(score=-0.5, feedback="test")
        assert result.score == 0.0

    def test_passed_threshold(self):
        """Test passed flag based on threshold."""
        result = GradeResult(score=0.8, feedback="test")
        assert result.passed is True

        result = GradeResult(score=0.5, feedback="test")
        assert result.passed is False

    def test_to_dict(self):
        """Test to_dict conversion."""
        result = GradeResult(score=0.9, feedback="Great!")
        d = result.to_dict()
        assert d["score"] == 0.9
        assert d["feedback"] == "Great!"
        assert d["passed"] is True


class TestTaskGrader:
    """Tests for TaskGrader."""

    def test_grade_easy_correct_approve(self):
        """Test easy task with correct approve."""
        ground_truth = {"is_compliant": True, "expected_action": "approve"}
        agent_output = {"action_taken": "approve"}

        result = TaskGrader.grade_easy_decision(ground_truth, agent_output)
        assert result.score == 1.0
        assert result.passed is True

    def test_grade_easy_correct_reject(self):
        """Test easy task with correct reject."""
        ground_truth = {"is_compliant": False, "expected_action": "reject"}
        agent_output = {"action_taken": "reject"}

        result = TaskGrader.grade_easy_decision(ground_truth, agent_output)
        assert result.score == 1.0

    def test_grade_easy_wrong_action(self):
        """Test easy task with wrong action."""
        ground_truth = {"is_compliant": True, "expected_action": "approve"}
        agent_output = {"action_taken": "reject"}

        result = TaskGrader.grade_easy_decision(ground_truth, agent_output)
        assert result.score == 0.0
        assert result.passed is False

    def test_grade_medium_edit_used(self):
        """Test medium task with edit used."""
        ground_truth = {
            "original_content": "Bad content",
            "original_score": 0.2,
            "violations": ["spam"],
        }
        agent_output = {
            "action_sequence": ["edit", "approve"],
            "final_content": "Good content",
            "final_score": 0.85,
        }

        result = TaskGrader.grade_medium_edit(ground_truth, agent_output)
        assert result.score > 0.5

    def test_grade_medium_no_edit(self):
        """Test medium task without edit."""
        ground_truth = {
            "original_content": "Bad content",
            "original_score": 0.2,
            "violations": ["spam"],
        }
        agent_output = {
            "action_sequence": ["approve"],
            "final_content": "Bad content",
            "final_score": 0.2,
        }

        result = TaskGrader.grade_medium_edit(ground_truth, agent_output)
        assert result.score < 0.5

    def test_grade_hard_detection(self):
        """Test hard task violation detection."""
        ground_truth = {
            "original_content": "Toxic content",
            "true_violations": ["hate_speech", "violence"],
        }
        agent_output = {
            "action_sequence": ["edit", "approve"],
            "detected_violations": ["hate_speech"],
            "final_content": "Clean content",
            "final_score": 0.8,
        }

        result = TaskGrader.grade_hard_complex(ground_truth, agent_output)
        assert result.score > 0.0


class TestNormalizeScore:
    """Tests for normalize_score function."""

    def test_normalize_standard_range(self):
        """Test normalization with standard range."""
        assert normalize_score(0.5, 0.0, 1.0) == 0.5
        assert normalize_score(1.0, 0.0, 1.0) == 1.0
        assert normalize_score(0.0, 0.0, 1.0) == 0.0

    def test_normalize_custom_range(self):
        """Test normalization with custom range."""
        assert normalize_score(6.0, 0.0, 10.0) == 0.6
        assert normalize_score(-5.0, -10.0, 10.0) == 0.25

    def test_normalize_clamping(self):
        """Test score clamping."""
        assert normalize_score(1.5, 0.0, 1.0) == 1.0
        assert normalize_score(-0.5, 0.0, 1.0) == 0.0


class TestGradeTask:
    """Tests for grade_task function."""

    def test_grade_task_easy(self):
        """Test grade_task with easy task."""
        result = grade_task(
            task_name="easy_decision",
            ground_truth={"is_compliant": True},
            agent_output={"action_taken": "approve"},
        )
        assert result.score == 1.0

    def test_grade_task_invalid_name(self):
        """Test grade_task with invalid task name."""
        with pytest.raises(ValueError):
            grade_task(
                task_name="invalid_task",
                ground_truth={},
                agent_output={},
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
