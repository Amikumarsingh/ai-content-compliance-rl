"""
Unit Tests for Reward Calculator.

Tests for reward calculation, normalization, and breakdown.
"""

import pytest

from rewards.reward_function import RewardCalculator, RewardComponent


class TestRewardComponent:
    """Tests for RewardComponent dataclass."""

    def test_component_creation(self):
        """Test creating a reward component."""
        comp = RewardComponent(
            name="test",
            value=0.5,
            description="Test component",
            max_value=1.0,
        )
        assert comp.name == "test"
        assert comp.value == 0.5

    def test_component_to_dict(self):
        """Test to_dict conversion."""
        comp = RewardComponent(name="action", value=0.3, description="Correct")
        d = comp.to_dict()
        assert d["name"] == "action"
        assert d["value"] == 0.3
        assert d["description"] == "Correct"


class TestRewardCalculator:
    """Tests for RewardCalculator."""

    def test_record_correct_action(self):
        """Test recording correct action."""
        calc = RewardCalculator(task_name="easy")
        calc.record_action("approve", "approve")
        calc.finalize(terminal_correct=True)

        reward, explanation = calc.get_final_reward()
        assert reward >= 0.5

    def test_record_wrong_action(self):
        """Test recording wrong action."""
        calc = RewardCalculator(task_name="easy")
        calc.record_action("reject", "approve")
        calc.finalize(terminal_correct=False)

        reward, explanation = calc.get_final_reward()
        assert reward < 0.5

    def test_record_improvement(self):
        """Test recording score improvement."""
        calc = RewardCalculator(task_name="medium")
        calc.record_improvement(original_score=0.2, new_score=0.8)
        calc.finalize(terminal_correct=True)

        reward, explanation = calc.get_final_reward()
        assert reward > 0.3

    def test_no_improvement(self):
        """Test recording no improvement."""
        calc = RewardCalculator(task_name="medium")
        calc.record_improvement(original_score=0.5, new_score=0.4)
        calc.finalize(terminal_correct=True)

        reward, _ = calc.get_final_reward()
        assert reward < 0.5

    def test_efficient_completion(self):
        """Test efficient episode completion."""
        calc = RewardCalculator(task_name="easy")
        calc.record_action("approve", "approve")
        calc.finalize(terminal_correct=True)

        breakdown = calc.get_reward_breakdown()
        assert "efficiency" in breakdown["breakdown"]

    def test_inefficient_completion(self):
        """Test inefficient episode completion."""
        calc = RewardCalculator(task_name="easy")
        for _ in range(5):
            calc.record_action("edit", "approve")
        calc.finalize(terminal_correct=True)

        reward, explanation = calc.get_final_reward()
        assert "Inefficient" in explanation

    def test_detection_reward(self):
        """Test violation detection reward."""
        calc = RewardCalculator(task_name="hard")
        calc.set_ground_truth(["hate_speech", "violence"])
        calc.record_detection(["hate_speech"])
        calc.finalize(terminal_correct=True)

        reward, _ = calc.get_final_reward()
        assert reward > 0.0

    def test_detection_with_false_positives(self):
        """Test detection with false positives."""
        calc = RewardCalculator(task_name="hard")
        calc.set_ground_truth(["hate_speech"])
        calc.record_detection(["hate_speech", "spam", "violence"])
        calc.finalize(terminal_correct=True)

        breakdown = calc.get_reward_breakdown()
        assert breakdown["breakdown"].get("detection", 0) < 0.3

    def test_edit_with_violations_removed(self):
        """Test edit with violations removed."""
        calc = RewardCalculator(task_name="medium")
        calc.record_edit(
            original_score=0.2,
            new_score=0.8,
            violations_before=["spam", "harassment"],
            violations_after=[],
        )
        calc.finalize(terminal_correct=True)

        reward, _ = calc.get_final_reward()
        assert reward > 0.5

    def test_edit_makes_worse(self):
        """Test edit that makes content worse."""
        calc = RewardCalculator(task_name="medium")
        calc.record_edit(
            original_score=0.6,
            new_score=0.3,
            violations_before=[],
            violations_after=["spam"],
        )
        calc.finalize(terminal_correct=True)

        reward, _ = calc.get_final_reward()
        assert reward < 0.5

    def test_get_reward_breakdown(self):
        """Test reward breakdown."""
        calc = RewardCalculator(task_name="easy")
        calc.record_action("approve", "approve")
        calc.finalize(terminal_correct=True)

        breakdown = calc.get_reward_breakdown()
        assert "total_reward" in breakdown
        assert "breakdown" in breakdown
        assert "step_count" in breakdown
        assert "explanation" in breakdown

    def test_terminal_action_tracking(self):
        """Test terminal action tracking."""
        calc = RewardCalculator(task_name="easy")
        calc.record_action("approve", "approve")

        assert calc._is_episode_complete is True
        assert calc._terminal_action_correct is True

    def test_partial_credit_action(self):
        """Test partial credit for close actions."""
        calc = RewardCalculator(task_name="medium")
        calc.record_action("edit", "reject")
        calc.finalize(terminal_correct=False)

        reward, _ = calc.get_final_reward()
        assert reward > 0.0

    def test_empty_episode(self):
        """Test empty episode."""
        calc = RewardCalculator(task_name="easy")
        calc.finalize(terminal_correct=False)

        reward, explanation = calc.get_final_reward()
        assert reward >= 0.0
        assert reward <= 1.0

    def test_reward_normalization(self):
        """Test reward is always normalized."""
        calc = RewardCalculator(task_name="hard")
        for _ in range(10):
            calc.record_action("edit", "approve")
        calc.finalize(terminal_correct=False)

        reward, _ = calc.get_final_reward()
        assert 0.0 <= reward <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
