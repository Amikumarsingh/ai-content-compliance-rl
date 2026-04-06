"""
Content Compliance RL Environment - OpenEnv Implementation

Production-grade environment for content moderation training:
- Simulates real-world content moderation workflows
- Three difficulty levels with progressive complexity
- Reward shaping with partial credit
- OpenAI-based content evaluation with fallback

OpenEnv Compliance:
- reset() -> Observation
- step(action) -> (Observation, Reward, done, truncated, info)
- state() -> Observation
- Typed Pydantic models throughout
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import models
from models import Observation, Action, Reward

# Import OpenAI evaluator (from parent directory)
import sys
sys.path.insert(0, '..')
from graders.openai_evaluator import OpenAIEvaluator, EvaluationResult

# Import deterministic grader (from parent directory)
from graders.graders import ContentModerationGrader

logger = logging.getLogger(__name__)

# Configuration from environment
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
API_BASE_URL = os.getenv("API_BASE_URL")
DEBUG = os.getenv("DEBUG", "false").lower() == "true"


# =============================================================================
# Content Templates
# =============================================================================

CONTENT_TEMPLATES = {
    "easy": {
        "clean": [
            "Hello! Welcome to our community platform. Looking forward to connecting!",
            "Thanks for sharing this helpful information about the topic.",
            "Great post! I really enjoyed reading your insights.",
            "Hi everyone, excited to be part of this community!",
        ],
        "spam": [
            "🔥 LIMITED OFFER! Click here NOW to claim your prize! Visit bit.ly/spam123",
            "I made $5000 working from home! DM me for the secret method!",
            "BUY NOW!!! 90% OFF everything! Click click click!",
        ],
        "hate": [
            "I hate all those people. They should all die!",
            "This group is terrible and should be banned.",
        ],
    },
    "medium": {
        "borderline_spam": [
            "Check out this amazing opportunity! (Not spam, I promise) bit.ly/opp",
            "Some people are making serious money online. Want to know how? DM me.",
            "This changed my life! You won't believe what happened next!",
        ],
        "subtle_harassment": [
            "Some people are just too delusional to understand the topic. Sad.",
            "Maybe if you actually studied instead of playing games all day...",
            "Interesting take, though clearly from someone who doesn't understand.",
        ],
    },
    "hard": {
        "multi_violation": [
            "They don't want you to know the truth! Wake up sheeple! The government is hiding everything about vaccines and 5G. Share before this gets deleted!",
            "Certain groups are pushing this agenda to control us. Don't believe the fake news! Do your own research and see the real conspiracy!",
            "I know people who know people. The elites are planning something big. Tag 3 friends who need to wake up!",
        ],
        "coded_hate": [
            "You know which group I'm talking about. They control everything. Globalists running the show. Wake up!",
            "Some people just don't belong here. Natural differences exist. Say what you want but we all know the truth.",
        ],
    },
}


# =============================================================================
# Environment Configuration
# =============================================================================

@dataclass
class EnvConfig:
    """Environment configuration."""
    max_steps: int = 5
    difficulty: str = "mixed"
    step_penalty: float = -0.05
    correct_reward: float = 0.7
    edit_reward: float = 0.8
    wrong_penalty: float = -0.3
    harmful_approval_penalty: float = -0.5


# =============================================================================
# Content Compliance Environment
# =============================================================================

class ContentComplianceEnv:
    """
    OpenEnv-compatible content moderation environment.

    Simulates real-world content moderation where agents must:
    1. Analyze user-generated content for policy violations
    2. Detect hate speech, spam, harassment, misinformation
    3. Decide: approve, reject, or edit content
    4. Learn from feedback with shaped rewards

    Attributes:
        config: Environment configuration
        evaluator: OpenAI-based content evaluator
        grader: Deterministic action grader
        current_content: Content being evaluated
        step_count: Current step in episode
        action_history: List of actions taken
        episode_data: Current episode metadata
    """

    # Violation types for detection
    VIOLATION_TYPES = [
        "hate_speech",
        "violence",
        "harassment",
        "adult_content",
        "misinformation",
        "spam",
        "suspicious_link",
        "engagement_bait",
        "illegal_content",
    ]

    def __init__(
        self,
        max_steps: int = 5,
        difficulty: str = "mixed",
        evaluator_provider: str = "openai",
    ):
        """
        Initialize the environment.

        Args:
            max_steps: Maximum steps per episode
            difficulty: Content difficulty level
            evaluator_provider: "openai" or "fallback"
        """
        self.config = EnvConfig(
            max_steps=max_steps,
            difficulty=difficulty,
        )
        self.evaluator = OpenAIEvaluator(model=LLM_MODEL, base_url=API_BASE_URL)
        self.grader = ContentModerationGrader()

        # Episode state
        self.current_content: Optional[str] = None
        self.current_evaluation: Optional[EvaluationResult] = None
        self.step_count: int = 0
        self.action_history: list[str] = []
        self.expected_action: str = "approve"
        self.expected_violations: list[str] = []
        self.is_episode_done: bool = True
        self._original_score: float = 0.5

        logger.info(f"ContentComplianceEnv initialized: difficulty={difficulty}, max_steps={max_steps}")

    def reset(self) -> Observation:
        """
        Reset environment and start new episode.

        Returns:
            Initial observation with content to evaluate.
        """
        self.step_count = 0
        self.action_history = []
        self.is_episode_done = False

        # Generate content based on difficulty
        content, violations, score, expected_action = self._generate_content()

        self.current_content = content
        self.expected_action = expected_action
        self.expected_violations = violations
        self._original_score = score

        # Evaluate content
        self.current_evaluation = self.evaluator.evaluate(content)

        if DEBUG:
            logger.info(f"[START] Generated content: score={score:.2f}, violations={violations}")

        return Observation(
            content=content,
            violations=self.current_evaluation.violations,
            score=self.current_evaluation.score,
            step_count=self.step_count,
            stage="decision",
            previous_action=None,
            action_history=[],
            feedback=None,
        )

    def step(self, action: Action) -> tuple[Observation, Reward, bool, bool, dict[str, Any]]:
        """
        Take a step in the environment.

        Args:
            action: Action to take (approve/reject/edit)

        Returns:
            Tuple of (observation, reward, done, truncated, info)
        """
        self.step_count += 1
        self.action_history.append(action.action_type)

        # Get current evaluation
        if self.current_evaluation is None:
            self.current_evaluation = self.evaluator.evaluate(self.current_content)

        violations = self.current_evaluation.violations
        score = self.current_evaluation.score
        is_compliant = self.current_evaluation.is_compliant

        # Determine correct action
        correct_action = "approve" if is_compliant else "reject"

        # Calculate reward based on action
        reward_value, explanation = self._calculate_reward(
            action=action.action_type,
            is_compliant=is_compliant,
            violations=violations,
            edited_content=action.edited_content,
        )

        # Determine if episode is done
        done = False
        truncated = False

        if action.action_type in ["approve", "reject"]:
            # Terminal actions
            done = True
        elif action.action_type == "edit":
            # Edit continues the episode
            if self.step_count >= self.config.max_steps:
                truncated = True
                done = True
            else:
                # Re-evaluate edited content
                if action.edited_content:
                    self.current_content = action.edited_content
                    self.current_evaluation = self.evaluator.evaluate(action.edited_content)
                    score = self.current_evaluation.score
                    violations = self.current_evaluation.violations
                    is_compliant = self.current_evaluation.is_compliant

        # Build observation
        observation = Observation(
            content=self.current_content,
            violations=violations,
            score=score,
            step_count=self.step_count,
            stage="decision",
            previous_action=action.action_type,
            action_history=self.action_history.copy(),
            feedback=explanation,
        )

        # Build reward
        reward = Reward(
            value=reward_value,
            raw_value=reward_value,
            explanation=explanation,
        )

        # Info dict
        info = {
            "correct_action": correct_action,
            "is_compliant": is_compliant,
            "violations": violations,
            "step_penalty": self.config.step_penalty,
            "timestamp": datetime.now().isoformat(),
        }

        return observation, reward, done, truncated, info

    def state(self) -> Observation:
        """
        Get current environment state without modifying it.

        Returns:
            Current observation.
        """
        if self.current_content is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        if self.current_evaluation is None:
            self.current_evaluation = self.evaluator.evaluate(self.current_content)

        return Observation(
            content=self.current_content,
            violations=self.current_evaluation.violations,
            score=self.current_evaluation.score,
            step_count=self.step_count,
            stage="decision",
            previous_action=self.action_history[-1] if self.action_history else None,
            action_history=self.action_history.copy(),
            feedback=None,
        )

    def _generate_content(self) -> tuple[str, list[str], float, str]:
        """
        Generate content based on difficulty level.

        Returns:
            Tuple of (content, violations, score, expected_action)
        """
        difficulty = self.config.difficulty

        if difficulty == "mixed":
            difficulty = random.choice(["easy", "medium", "hard"])

        if difficulty == "easy":
            # Binary classification: clean or violating
            content_type = random.choice(["clean", "spam", "hate"])
            if content_type == "clean":
                content = random.choice(CONTENT_TEMPLATES["easy"]["clean"])
                return content, [], 0.9, "approve"
            elif content_type == "spam":
                content = random.choice(CONTENT_TEMPLATES["easy"]["spam"])
                return content, ["spam", "engagement_bait"], 0.1, "reject"
            else:
                content = random.choice(CONTENT_TEMPLATES["easy"]["hate"])
                return content, ["hate_speech"], 0.0, "reject"

        elif difficulty == "medium":
            # Borderline content requiring edit
            content_type = random.choice(["borderline_spam", "subtle_harassment"])
            if content_type == "borderline_spam":
                content = random.choice(CONTENT_TEMPLATES["medium"]["borderline_spam"])
                return content, ["spam", "suspicious_link"], 0.3, "edit"
            else:
                content = random.choice(CONTENT_TEMPLATES["medium"]["subtle_harassment"])
                return content, ["harassment"], 0.4, "edit"

        else:  # hard
            # Complex multi-violation content
            content = random.choice(CONTENT_TEMPLATES["hard"]["multi_violation"])
            return content, ["misinformation", "engagement_bait", "conspiracy"], 0.1, "reject"

    def _calculate_reward(
        self,
        action: str,
        is_compliant: bool,
        violations: list[str],
        edited_content: Optional[str] = None,
    ) -> tuple[float, str]:
        """
        Calculate reward with shaping.

        Args:
            action: Action taken by agent
            is_compliant: Whether content is compliant
            violations: List of detected violations
            edited_content: Edited content if action was "edit"

        Returns:
            Tuple of (reward_value, explanation)
        """
        explanation_parts: list[str] = []
        reward = 0.0

        # Determine correct action
        correct_action = "approve" if is_compliant else "reject"

        # Base reward for action correctness
        if action == correct_action:
            reward = self.config.correct_reward
            explanation_parts.append("correct action")
        elif action == "edit":
            # Edit is partially correct
            reward = 0.4
            explanation_parts.append("edit attempted")
        else:
            # Wrong action
            reward = 0.1
            explanation_parts.append("incorrect action")

            # Severe penalty for harmful approval
            if action == "approve" and not is_compliant:
                if any(v in ["hate_speech", "violence", "illegal_content"] for v in violations):
                    reward += self.config.harmful_approval_penalty
                    explanation_parts.append("harmful approval penalty")

        # Edit bonus
        if action == "edit" and edited_content:
            # Check if edit improved compliance
            edit_eval = self.evaluator.evaluate(edited_content)
            improvement = edit_eval.score - self._original_score

            if improvement > 0.3:
                reward = self.config.edit_reward
                explanation_parts.append("good edit")
            elif improvement > 0.1:
                reward += 0.2
                explanation_parts.append("partial edit improvement")
            else:
                reward -= 0.1
                explanation_parts.append("minimal edit improvement")

        # Step penalty
        if self.step_count > 1:
            step_penalty = self.config.step_penalty * (self.step_count - 1)
            reward += step_penalty
            if step_penalty < 0:
                explanation_parts.append(f"step penalty: {step_penalty:.2f}")

        # Clamp reward to valid range
        reward = max(0.0, min(1.0, reward))

        explanation = "; ".join(explanation_parts)

        return reward, explanation

    def close(self) -> None:
        """Clean up environment resources."""
        pass

    def render(self) -> str:
        """Render environment state as string."""
        return (
            f"ContentComplianceEnv(\n"
            f"  content={self.current_content[:50] if self.current_content else 'None'}...\n"
            f"  step_count={self.step_count}\n"
            f"  done={self.is_episode_done}\n"
            f"  action_history={self.action_history}\n"
            f")"
        )
