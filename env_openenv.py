"""
OpenEnv Content Moderation RL Environment.

Production-grade environment for training content moderation AI agents.
Implements the OpenEnv specification with:
- Typed Pydantic models
- Deterministic reward shaping
- Three difficulty levels
- Multi-step decision workflow

Usage:
    from env_openenv import ContentModerationEnv
    env = ContentModerationEnv(difficulty="mixed")
    obs = env.reset()
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
from enum import Enum
from typing import Any, Optional

from dotenv import load_dotenv

from app.models import Action, Observation, Reward, StepResult

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class Stage(str, Enum):
    """Decision stages in the moderation workflow."""

    CLASSIFICATION = "classification"  # Analyze content
    DECISION = "decision"  # Choose approve/reject/edit
    REFINEMENT = "refinement"  # Edit and re-evaluate


class ContentModerationEnv:
    """
    OpenEnv Content Moderation RL Environment.

    Simulates real-world content moderation where agents must:
    1. Analyze user-generated content for policy violations
    2. Decide whether to approve, reject, or edit
    3. Handle borderline cases requiring nuanced judgment

    Action Space:
        - approve: Content passes (terminal)
        - reject: Content violates (terminal)
        - edit: Modify content (continues episode)

    Observation Space:
        - content: Text being evaluated
        - violations: Detected violation types
        - score: Compliance score (0.0-1.0)
        - step_count: Current step
        - stage: Current workflow stage
        - previous_action: Last action taken
        - action_history: All actions this episode
        - feedback: Environment feedback message

    Reward Structure:
        - +1.0: Correct terminal decision
        - +0.5: Partially correct (edit when appropriate)
        - 0.0: Wrong decision
        - -0.2: Harmful approval (approving dangerous content)
        - -0.05: Per-step efficiency penalty
    """

    # Action constants
    ACTION_APPROVE = "approve"
    ACTION_REJECT = "reject"
    ACTION_EDIT = "edit"

    # Violation types
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

    # Configuration
    DEFAULT_MAX_STEPS = 5
    DIFFICULTY_OPTIONS = ["easy", "medium", "hard", "mixed"]

    def __init__(
        self,
        max_steps: int = DEFAULT_MAX_STEPS,
        difficulty: str = "mixed",
        evaluator_provider: str = "openai",
        model_name: Optional[str] = None,
    ):
        """
        Initialize the content moderation environment.

        Args:
            max_steps: Maximum steps per episode (default: 5).
            difficulty: Content difficulty level.
            evaluator_provider: Evaluation backend ("openai" or "fallback").
            model_name: Model name for evaluation (default: gpt-4o-mini).
        """
        self.max_steps = max(1, min(max_steps, 10))  # Clamp 1-10
        self.difficulty = self._validate_difficulty(difficulty)
        self.evaluator_provider = evaluator_provider
        self.model_name = model_name or os.getenv("LLM_MODEL", "gpt-4o-mini")

        # Episode state
        self._episode_active = False
        self._current_step = 0
        self._current_stage = Stage.CLASSIFICATION
        self._previous_action: Optional[str] = None
        self._action_history: list[str] = []

        # Content state
        self._current_content: str = ""
        self._current_violations: list[str] = []
        self._current_score: float = 0.0
        self._initial_score: float = 0.0
        self._initial_violations: list[str] = []

        # Expected values for grading
        self._expected_action: Optional[str] = None
        self._expected_violations: list[str] = []

        # Evaluator (lazy initialization)
        self._evaluator = None

        logger.info(
            f"ContentModerationEnv initialized: "
            f"max_steps={self.max_steps}, difficulty={self.difficulty}, "
            f"evaluator={evaluator_provider}, model={self.model_name}"
        )

    def _validate_difficulty(self, difficulty: str) -> str:
        """Validate and normalize difficulty setting."""
        if difficulty.lower() in self.DIFFICULTY_OPTIONS:
            return difficulty.lower()
        logger.warning(f"Invalid difficulty '{difficulty}', using 'mixed'")
        return "mixed"

    def _get_evaluator(self):
        """Lazy-load the OpenAI evaluator."""
        if self._evaluator is None:
            from graders.openai_evaluator import OpenAIEvaluator

            self._evaluator = OpenAIEvaluator(
                model=self.model_name,
                timeout=10.0,
                max_retries=2,
            )
        return self._evaluator

    def _generate_content(self) -> tuple[str, list[str], float, str]:
        """
        Generate content for evaluation.

        Returns:
            Tuple of (content, violations, score, expected_action).
        """
        # Select difficulty for this episode
        if self.difficulty == "mixed":
            episode_difficulty = random.choice(["easy", "medium", "hard"])
        else:
            episode_difficulty = self.difficulty

        # Generate content based on difficulty
        if episode_difficulty == "easy":
            return self._generate_easy_content()
        elif episode_difficulty == "medium":
            return self._generate_medium_content()
        else:
            return self._generate_hard_content()

    def _generate_easy_content(self) -> tuple[str, list[str], float, str]:
        """Generate easy difficulty content (obvious cases)."""
        content_type = random.choice(["spam", "hate", "clean"])

        if content_type == "spam":
            content = random.choice([
                "🔥 LAST CHANCE! Click bit.ly/deal NOW for FREE PRIZES! 🔥",
                "I made $50000 working 1 hour from home! Visit tinyurl.com/rich",
                "BUY NOW!!! Limited offer!!! Click here now!!!",
            ])
            violations = ["spam", "engagement_bait"]
            expected_action = "reject"

        elif content_type == "hate":
            content = random.choice([
                "I hate those idiots. They should all die.",
                "Those people are stupid and deserve to suffer.",
            ])
            violations = ["hate_speech", "violence"]
            expected_action = "reject"

        else:  # clean
            content = random.choice([
                "Hello! Welcome to our community platform. Looking forward to connecting!",
                "Thanks for sharing this helpful information. Very appreciated!",
                "Great discussion everyone! Keep the conversation respectful.",
            ])
            violations = []
            expected_action = "approve"

        score = 0.1 if violations else 0.9
        return content, violations, score, expected_action

    def _generate_medium_content(self) -> tuple[str, list[str], float, str]:
        """Generate medium difficulty content (borderline cases)."""
        content_type = random.choice(["borderline_spam", "subtle_harassment"])

        if content_type == "borderline_spam":
            content = random.choice([
                "Check out this amazing opportunity! bit.ly/opportunity (Not spam, real!)",
                "Some people are just too closed-minded to understand this. Sad.",
                "I'm not saying you should click this link... but if you do: bit.ly/maybe",
            ])
            violations = ["spam", "suspicious_link"]
            expected_action = "edit"

        else:  # subtle_harassment
            content = random.choice([
                "Some people are just too ignorant to understand complex topics.",
                "If you disagree with me, you're probably not educated enough.",
                "I feel sorry for people who believe such misguided ideas.",
            ])
            violations = ["harassment"]
            expected_action = "edit"

        score = 0.35
        return content, violations, score, expected_action

    def _generate_hard_content(self) -> tuple[str, list[str], float, str]:
        """Generate hard difficulty content (multi-violation, subtle)."""
        content_type = random.choice(["multi_violation", "coded_hate", "misinformation"])

        if content_type == "multi_violation":
            content = random.choice([
                "Those people are delusional and should be removed. Don't believe their lies!",
                "They don't want you to know the truth! Wake up sheeple! Visit bit.ly/truth",
            ])
            violations = ["harassment", "misinformation", "engagement_bait"]
            expected_action = "reject"

        elif content_type == "coded_hate":
            content = random.choice([
                "Some groups just don't fit in with our values. Just saying.",
                "You know which people I'm talking about. They're ruining everything.",
            ])
            violations = ["hate_speech"]
            expected_action = "reject"

        else:  # misinformation
            content = random.choice([
                "They're hiding the truth about vaccines. Do your own research!",
                "The government is manipulating the data. Wake up!",
            ])
            violations = ["misinformation"]
            expected_action = "reject"

        score = 0.2
        return content, violations, score, expected_action

    def _evaluate_content(self, content: str) -> tuple[list[str], float, str]:
        """
        Evaluate content using OpenAI evaluator.

        Args:
            content: Text content to evaluate.

        Returns:
            Tuple of (violations, score, reason).
        """
        evaluator = self._get_evaluator()
        result = evaluator.evaluate(content)

        return result.violations, result.score, result.reason

    def reset(self) -> Observation:
        """
        Reset environment and start new episode.

        Returns:
            Initial observation with fresh content.
        """
        self._episode_active = True
        self._current_step = 0
        self._current_stage = Stage.CLASSIFICATION
        self._previous_action = None
        self._action_history = []

        # Generate new content
        content, violations, score, expected_action = self._generate_content()

        self._current_content = content
        self._current_violations = violations
        self._current_score = score
        self._initial_score = score
        self._initial_violations = violations.copy()
        self._expected_action = expected_action
        self._expected_violations = violations.copy()

        # Create initial observation
        observation = Observation(
            content=content,
            violations=violations,
            score=score,
            step_count=0,
            stage=self._current_stage.value,
            previous_action=None,
            action_history=[],
            feedback="Content loaded. Analyze and decide on appropriate action.",
        )

        logger.info(
            f"Episode started: difficulty={self.difficulty}, "
            f"violations={len(violations)}, score={score:.2f}, "
            f"expected_action={expected_action}"
        )

        return observation

    def step(self, action: Action) -> StepResult:
        """
        Take a step in the environment.

        Args:
            action: Action to take (approve/reject/edit).

        Returns:
            StepResult with new observation, reward, done, truncated, info.
        """
        if not self._episode_active:
            raise RuntimeError("Episode has ended. Call reset() to start new episode.")

        # Validate action
        if action.action_type == self.ACTION_EDIT and not action.edited_content:
            raise ValueError("Edit action requires edited_content")

        # Track action
        self._current_step += 1
        self._previous_action = action.action_type
        self._action_history.append(action.action_type)

        # Get content to evaluate
        content_to_eval = action.edited_content if action.edited_content else self._current_content

        # Evaluate content after action
        violations, score, reason = self._evaluate_content(content_to_eval)

        # Calculate reward
        reward_value, reward_explanation = self._calculate_reward(
            action=action.action_type,
            violations=violations,
            score=score,
            expected_action=self._expected_action,
            expected_violations=self._expected_violations,
            step_count=self._current_step,
        )

        # Determine if episode should end
        is_correct_terminal = self._is_correct_terminal_action(
            action=action.action_type,
            violations=violations,
            score=score,
        )

        done = is_correct_terminal or action.action_type in [self.ACTION_APPROVE, self.ACTION_REJECT]
        truncated = self._current_step >= self.max_steps and not done

        # Update stage
        if action.action_type == self.ACTION_EDIT:
            self._current_stage = Stage.REFINEMENT
        else:
            self._current_stage = Stage.DECISION

        # Update content state
        self._current_content = content_to_eval
        self._current_violations = violations
        self._current_score = score

        # Build feedback message
        feedback = self._build_feedback(
            action=action.action_type,
            violations=violations,
            score=score,
            reward=reward_value,
        )

        # Create new observation
        observation = Observation(
            content=content_to_eval,
            violations=violations,
            score=score,
            step_count=self._current_step,
            stage=self._current_stage.value,
            previous_action=action.action_type,
            action_history=self._action_history.copy(),
            feedback=feedback,
        )

        # Create reward
        reward = Reward(
            value=reward_value,
            raw_value=reward_value,
            explanation=reward_explanation,
        )

        # Build info dict
        info = {
            "action": action.action_type,
            "violations": violations,
            "score": score,
            "reason": reason,
            "expected_action": self._expected_action,
            "is_correct_terminal": is_correct_terminal,
            "step_count": self._current_step,
        }

        # Handle episode end
        if done or truncated:
            self._episode_active = False
            info["episode_end"] = True
            info["final_score"] = score

        logger.info(
            f"Step {self._current_step}: action={action.action_type}, "
            f"reward={reward_value:.3f}, done={done}, truncated={truncated}"
        )

        return StepResult(
            observation=observation,
            reward=reward,
            done=done,
            truncated=truncated,
            info=info,
        )

    def _calculate_reward(
        self,
        action: str,
        violations: list[str],
        score: float,
        expected_action: str,
        expected_violations: list[str],
        step_count: int,
    ) -> tuple[float, str]:
        """Calculate reward for the action taken."""
        components: list[str] = []
        raw_reward = 0.0

        is_compliant = score >= 0.5 and len(violations) == 0

        # 1. Action correctness
        if action == expected_action:
            if action == self.ACTION_APPROVE and is_compliant:
                raw_reward = 1.0
                components.append("correct_approve:+1.0")
            elif action == self.ACTION_REJECT and not is_compliant:
                raw_reward = 1.0
                components.append("correct_reject:+1.0")
            elif action == self.ACTION_EDIT:
                raw_reward = 0.6
                components.append("edit_initiated:+0.6")
        elif action in [self.ACTION_APPROVE, self.ACTION_REJECT]:
            # Wrong terminal action
            if action == self.ACTION_APPROVE and not is_compliant:
                # Harmful approval
                harmful_types = {"hate_speech", "violence", "illegal_content"}
                if violations and any(v in harmful_types for v in violations):
                    raw_reward = 0.0
                    components.append("harmful_approval:0.0")
                else:
                    raw_reward = 0.2
                    components.append("wrong_approve:+0.2")
            elif action == self.ACTION_REJECT and is_compliant:
                raw_reward = 0.3
                components.append("false_reject:+0.3")
        elif action == self.ACTION_EDIT and expected_action in [self.ACTION_APPROVE, self.ACTION_REJECT]:
            # Edit as intermediate step
            raw_reward = 0.4
            components.append("edit_intermediate:+0.4")

        # 2. Violation detection bonus (for reject/edit)
        if action in [self.ACTION_REJECT, self.ACTION_EDIT] and expected_violations:
            detected_set = set(violations)
            expected_set = set(expected_violations)
            recall = len(detected_set & expected_set) / len(expected_set) if expected_set else 1.0

            if recall >= 0.8:
                raw_reward += 0.2
                components.append("violation_detection:+0.2")
            elif recall >= 0.5:
                raw_reward += 0.1
                components.append("partial_detection:+0.1")

        # 3. Step efficiency penalty
        if step_count > 1:
            step_penalty = (step_count - 1) * 0.05
            raw_reward -= step_penalty
            components.append(f"step_penalty:-{step_penalty:.2f}")

        # Clamp reward
        final_reward = max(0.0, min(1.0, raw_reward))

        explanation = "; ".join(components)
        return final_reward, explanation

    def _is_correct_terminal_action(
        self,
        action: str,
        violations: list[str],
        score: float,
    ) -> bool:
        """Check if terminal action is correct."""
        is_compliant = score >= 0.5 and len(violations) == 0

        if action == self.ACTION_APPROVE:
            return is_compliant
        elif action == self.ACTION_REJECT:
            return not is_compliant

        return False

    def _build_feedback(
        self,
        action: str,
        violations: list[str],
        score: float,
        reward: float,
    ) -> str:
        """Build feedback message for the agent."""
        if reward >= 0.8:
            return "Excellent decision!"
        elif reward >= 0.6:
            return "Good decision."
        elif reward >= 0.4:
            return "Acceptable, but room for improvement."
        elif reward >= 0.2:
            return "Suboptimal decision."
        else:
            return "Poor decision. Review content carefully."

    def state(self) -> dict[str, Any]:
        """Get current environment state."""
        return {
            "episode_active": self._episode_active,
            "current_step": self._current_step,
            "current_stage": self._current_stage.value,
            "previous_action": self._previous_action,
            "action_history": self._action_history.copy(),
            "current_content": self._current_content,
            "current_violations": self._current_violations,
            "current_score": self._current_score,
            "initial_score": self._initial_score,
            "initial_violations": self._initial_violations,
            "expected_action": self._expected_action,
            "expected_violations": self._expected_violations,
            "difficulty": self.difficulty,
            "max_steps": self.max_steps,
        }

    async def close(self) -> None:
        """Clean up environment resources."""
        self._episode_active = False
        self._evaluator = None
        logger.info("Environment closed")

    @property
    def action_space(self) -> dict[str, Any]:
        """Get action space specification."""
        return {
            "type": "discrete",
            "actions": [self.ACTION_APPROVE, self.ACTION_REJECT, self.ACTION_EDIT],
            "descriptions": {
                self.ACTION_APPROVE: "Content passes compliance (terminal)",
                self.ACTION_REJECT: "Content violates policies (terminal)",
                self.ACTION_EDIT: "Modify content to fix violations (continues)",
            },
        }

    @property
    def observation_space(self) -> dict[str, Any]:
        """Get observation space specification."""
        return {
            "type": "dict",
            "fields": {
                "content": {"type": "str", "description": "Text content to evaluate"},
                "violations": {"type": "list[str]", "description": "Detected violation types"},
                "score": {"type": "float", "range": [0.0, 1.0]},
                "step_count": {"type": "int", "min": 0},
                "stage": {"type": "str", "enum": ["classification", "decision", "refinement"]},
                "previous_action": {"type": "str", "nullable": True},
                "action_history": {"type": "list[str]"},
                "feedback": {"type": "str", "nullable": True},
            },
        }


# Factory function
def make_env(
    max_steps: int = 5,
    difficulty: str = "mixed",
    evaluator_provider: str = "openai",
) -> ContentModerationEnv:
    """
    Create a content moderation environment.

    Args:
        max_steps: Maximum steps per episode.
        difficulty: Content difficulty level.
        evaluator_provider: Evaluation backend.

    Returns:
        Configured ContentModerationEnv instance.
    """
    return ContentModerationEnv(
        max_steps=max_steps,
        difficulty=difficulty,
        evaluator_provider=evaluator_provider,
    )
