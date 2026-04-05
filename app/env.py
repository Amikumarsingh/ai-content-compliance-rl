"""
OpenEnv Content Compliance Environment.

Async environment implementing the OpenEnv specification for
content moderation training with normalized rewards (0.0 to 1.0).

Example:
    >>> env = ContentComplianceEnv(max_steps=5)
    >>> obs = await env.reset()
    >>> action = Action(action_type="approve")
    >>> result = await env.step(action)
    >>> print(result.reward.value)  # 0.0 to 1.0
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from app.models import Action, Observation, Reward, StepResult
from app.reward import RewardCalculator
from app.state_manager import StateManager

logger = logging.getLogger(__name__)


class ContentComplianceEnv:
    """
    Async Content Compliance Environment (OpenEnv-compatible).

    Action Space:
        - approve: Content passes compliance (terminal)
        - reject: Content violates policies (terminal)
        - edit: Content needs modification (continues episode)

    Observation Space:
        - content: str (text to evaluate)
        - violations: List[str] (detected violations)
        - score: float (0.0 to 1.0 compliance)
        - step_count: int (current step)

    Reward:
        Normalized to [0.0, 1.0]:
        - 1.0: Perfect action
        - 0.5: Neutral
        - 0.0: Worst action
    """

    ACTION_APPROVE = "approve"
    ACTION_REJECT = "reject"
    ACTION_EDIT = "edit"

    def __init__(
        self,
        max_steps: int = 5,
        difficulty: str = "mixed",
        evaluator_provider: str = "mock",
    ):
        """
        Initialize the content compliance environment.

        Args:
            max_steps: Maximum steps per episode.
            difficulty: Content difficulty ("easy", "medium", "hard", "mixed").
            evaluator_provider: LLM provider ("mock" or "openai").
        """
        self.max_steps = max_steps
        self.difficulty = difficulty
        self.evaluator_provider = evaluator_provider

        self._state_manager = StateManager(difficulty=difficulty)
        self._episode_ended: bool = False
        self._reward_calculator: Optional[RewardCalculator] = None
        self._info: dict[str, Any] = {}

        logger.info(
            f"ContentComplianceEnv initialized: "
            f"max_steps={max_steps}, difficulty={difficulty}"
        )

    def _get_task_name(self) -> str:
        """Get task name based on difficulty."""
        difficulty_map = {
            "easy": "easy",
            "medium": "medium",
            "hard": "hard",
            "mixed": "medium",
        }
        return difficulty_map.get(self.difficulty, "medium")

    def _generate_content(self) -> tuple[str, list[str], float]:
        """
        Generate content for evaluation.

        Returns:
            Tuple of (content, violations, initial_score)
        """
        from utils.data_loader import ContentDataLoader

        loader = ContentDataLoader()
        item = loader.get_random_content()

        violations = []
        if not item.is_compliant:
            violations = self._infer_violations(item.text)

        score = 0.9 if item.is_compliant else 0.2

        return item.text, violations, score

    def _infer_violations(self, content: str) -> list[str]:
        """Infer violations from content using rule-based detection."""
        from graders.violation_detector import ViolationDetector

        return ViolationDetector.detect(content)

    def _normalize_reward(self, raw_reward: float) -> float:
        """Normalize reward to [0.0, 1.0] range."""
        return max(0.0, min(1.0, (raw_reward + 10) / 20))

    def _get_expected_action(self, score: float) -> str:
        """Determine the expected action based on compliance score."""
        return "approve" if score >= 0.5 else "reject"

    async def reset(self) -> Observation:
        """
        Reset the environment and start a new episode.

        Returns:
            Initial observation with fresh content.
        """
        self._episode_ended = False
        self._reward_calculator = RewardCalculator(task_name=self._get_task_name())

        content, violations, score = self._generate_content()

        self._state_manager.reset(
            content=content,
            violations=violations,
            score=score,
        )

        self._reward_calculator.set_ground_truth(violations)

        observation = Observation(
            content=content,
            violations=violations,
            score=score,
            step_count=0,
        )

        self._info = {
            "initial_score": score,
            "source": "dynamic",
        }

        logger.info(
            f"Episode started: content length={len(content)}, "
            f"violations={len(violations)}, score={score:.2f}"
        )

        return observation

    async def step(self, action: Action) -> StepResult:
        """
        Take a step in the environment.

        Args:
            action: Action to take (approve/reject/edit).

        Returns:
            StepResult with (observation, reward, done, truncated, info).
        """
        if self._episode_ended:
            raise RuntimeError("Episode has ended. Call reset() to start new episode.")

        current_state = self._state_manager.current_episode
        if current_state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        if action.action_type == "edit" and not action.edited_content:
            raise ValueError("Edit action requires edited_content")

        content_to_eval = action.edited_content if action.edited_content else current_state.content

        eval_result = self._evaluate_content(content_to_eval, action.action_type)

        self._state_manager.update(
            action=action.action_type,
            new_content=content_to_eval if action.action_type == "edit" else None,
            new_violations=eval_result.get("violations", []),
            new_score=eval_result.get("score", current_state.score),
        )

        expected_action = self._get_expected_action(eval_result["score"])

        if self._reward_calculator:
            self._reward_calculator.record_action(action.action_type, expected_action)

            if action.action_type == "edit":
                original_score = current_state.score
                new_score = eval_result["score"]
                if new_score > original_score:
                    self._reward_calculator.record_improvement(original_score, new_score)

        raw_reward = self._calculate_reward(
            action=action.action_type,
            eval_result=eval_result,
            expected_action=expected_action,
        )

        normalized_reward = max(0.0, min(1.0, raw_reward))
        reward = Reward(
            value=normalized_reward,
            raw_value=raw_reward,
            explanation=f"Action: {action.action_type}, Score: {eval_result['score']:.2f}"
        )

        done = action.action_type in [self.ACTION_APPROVE, self.ACTION_REJECT]
        truncated = current_state.step_count >= self.max_steps and not done

        if truncated:
            reward = Reward(
                value=0.1,
                raw_value=-5.0,
                explanation="Episode truncated - no terminal action"
            )
            done = True

        self._episode_ended = done or truncated

        observation = Observation(
            content=content_to_eval,
            violations=eval_result.get("violations", []),
            score=eval_result.get("score", 0.0),
            step_count=current_state.step_count,
        )

        self._info = {
            "action": action.action_type,
            "compliance_score": eval_result.get("score", 0.0),
            "violations": eval_result.get("violations", []),
            "suggestion": eval_result.get("suggestion", ""),
            "edited": action.edited_content is not None,
        }

        if done:
            self._state_manager.complete_episode()

        logger.info(
            f"Step complete: reward={normalized_reward:.3f}, "
            f"done={done}, truncated={truncated}"
        )

        return StepResult(
            observation=observation,
            reward=reward,
            done=done,
            truncated=truncated,
            info=self._info,
        )

    def _evaluate_content(
        self,
        content: str,
        action: str,
    ) -> dict[str, Any]:
        """Evaluate content using rule-based evaluator."""
        from graders.violation_detector import ViolationDetector

        violations = ViolationDetector.detect(content)

        is_compliant = len(violations) == 0
        score = 0.9 if is_compliant else 0.2

        suggestion = ""
        if violations:
            suggestion = f"Remove: {', '.join(violations)}"

        return {
            "score": score,
            "violations": violations,
            "suggestion": suggestion,
        }

    def _calculate_reward(
        self,
        action: str,
        eval_result: dict[str, Any],
        expected_action: str,
    ) -> float:
        """Calculate raw reward based on action correctness."""
        score = eval_result["score"]
        is_compliant = score >= 0.5

        if action == self.ACTION_APPROVE:
            if is_compliant:
                return 0.8 + (0.2 * score)
            else:
                return 0.1

        elif action == self.ACTION_REJECT:
            if not is_compliant:
                return 0.7 + (0.2 * (1.0 - score))
            else:
                return 0.1

        elif action == self.ACTION_EDIT:
            violations = eval_result.get("violations", [])
            if violations:
                return 0.4 + (0.1 * min(1.0, len(violations) / 3.0))
            else:
                return 0.2

        return 0.2

    def state(self) -> dict[str, Any]:
        """Get current environment state."""
        return self._state_manager.get_state()

    async def close(self) -> None:
        """Clean up environment resources."""
        self._episode_ended = True
        self._state_manager.clear()
        logger.info("Environment closed")

    @property
    def observation_space(self) -> dict[str, Any]:
        """Get observation space specification."""
        return {
            "type": "dict",
            "fields": {
                "content": {"type": "str"},
                "violations": {"type": "list[str]"},
                "score": {"type": "float", "range": [0.0, 1.0]},
                "step_count": {"type": "int", "min": 0},
            },
        }

    @property
    def action_space(self) -> dict[str, Any]:
        """Get action space specification."""
        return {
            "type": "discrete",
            "actions": ["approve", "reject", "edit"],
            "description": {
                "approve": "Content passes compliance (terminal)",
                "reject": "Content violates policies (terminal)",
                "edit": "Content needs modification (continues)",
            },
        }
