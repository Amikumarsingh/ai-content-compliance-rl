"""
Reward functions for Content Compliance RL.

Provides dense, explainable reward signals normalized to [0.0, 1.0].
"""

from rewards.reward_function import RewardCalculator, RewardComponent

__all__ = [
    "RewardCalculator",
    "RewardComponent",
]
