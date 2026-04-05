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

from openenv.models import Action, Observation, Reward, StepResult

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

    REWARD_MIN = -12.0
    REWARD_MAX = 12.0

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

        self._current_observation: Optional[Observation] = None
        self._step_count: int = 0
        self._episode_ended: bool = False
        self._content: str = ""
        self._violations: list[str] = []
        self._info: dict[str, Any] = {}
        self._evaluator = None

        logger.info(f"ContentComplianceEnv initialized: max_steps={max_steps}, difficulty={difficulty}")

    def _get_evaluator(self):
        """Lazy-load the LLM evaluator."""
        if self._evaluator is None:
            from llm.evaluator import LLMEvaluator
            self._evaluator = LLMEvaluator(provider=self.evaluator_provider)
        return self._evaluator

    def _generate_content(self) -> str:
        """Generate or fetch content for evaluation."""
        from utils.data_loader import ContentDataLoader
        loader = ContentDataLoader()
        return loader.get_random_content().text

    def _evaluate_content(self, content: str, action: str) -> dict[str, Any]:
        """Evaluate content using LLM evaluator."""
        evaluator = self._get_evaluator()
        result = evaluator.evaluate_content(content, action)
        return {
            "score": result.compliance_score,
            "violations": result.violations,
            "reward": result.reward,
            "suggestion": result.suggestion,
        }

    def _normalize_reward(self, raw_reward: float) -> float:
        """Normalize reward to [0.0, 1.0] range."""
        normalized = (raw_reward - self.REWARD_MIN) / (self.REWARD_MAX - self.REWARD_MIN)
        return max(0.0, min(1.0, normalized))

    async def reset(self) -> Observation:
        """Reset the environment and start a new episode."""
        self._step_count = 0
        self._episode_ended = False
        self._violations = []
        self._info = {}

        self._content = self._generate_content()
        initial_eval = self._evaluate_content(self._content, "approve")

        self._current_observation = Observation(
            content=self._content,
            violations=initial_eval["violations"],
            score=initial_eval["score"],
            step_count=self._step_count,
        )

        self._info["initial_score"] = initial_eval["score"]
        self._info["source"] = "dynamic"

        logger.info(f"Episode started: content length={len(self._content)}")
        return self._current_observation

    async def step(self, action: Action) -> StepResult:
        """Take a step in the environment."""
        if self._episode_ended:
            raise RuntimeError("Episode has ended. Call reset() to start new episode.")

        if self._current_observation is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        if action.action_type == "edit" and not action.edited_content:
            raise ValueError("Edit action requires edited_content")

        self._step_count += 1
        action_name = action.action_type
        logger.info(f"Step {self._step_count}: Action={action_name}")

        content_to_eval = action.edited_content if action.edited_content else self._content
        eval_result = self._evaluate_content(content_to_eval, action_name)

        raw_reward = self._calculate_reward(action_name, eval_result)
        normalized_reward = self._normalize_reward(raw_reward)

        reward = Reward(
            value=normalized_reward,
            raw_value=raw_reward,
            explanation=f"Action: {action_name}, Score: {eval_result['score']:.2f}"
        )

        self._content = content_to_eval
        self._violations = eval_result["violations"]

        done = action_name in [self.ACTION_APPROVE, self.ACTION_REJECT]
        truncated = self._step_count >= self.max_steps and not done

        if truncated:
            raw_reward = -5.0
            normalized_reward = self._normalize_reward(raw_reward)
            reward = Reward(
                value=normalized_reward,
                raw_value=raw_reward,
                explanation="Episode truncated - no terminal action"
            )
            done = True

        self._episode_ended = done or truncated

        self._current_observation = Observation(
            content=self._content,
            violations=self._violations,
            score=eval_result["score"],
            step_count=self._step_count,
        )

        self._info = {
            "action": action_name,
            "raw_reward": raw_reward,
            "normalized_reward": normalized_reward,
            "compliance_score": eval_result["score"],
            "violations": self._violations,
            "suggestion": eval_result["suggestion"],
            "edited": action.edited_content is not None,
        }

        logger.info(f"Step complete: reward={normalized_reward:.3f}, done={done}, truncated={truncated}")

        return StepResult(
            observation=self._current_observation,
            reward=reward,
            done=done,
            truncated=truncated,
            info=self._info,
        )

    def _calculate_reward(self, action: str, eval_result: dict[str, Any]) -> float:
        """Calculate raw reward based on action and evaluation."""
        score = eval_result["score"]
        is_compliant = score >= 0.5
        violations = eval_result["violations"]

        if action == self.ACTION_APPROVE:
            if is_compliant:
                return 8.0 + (4.0 * score)
            else:
                return -8.0 - (4.0 * (1.0 - score))

        elif action == self.ACTION_REJECT:
            if not is_compliant:
                return 6.0 + (4.0 * (1.0 - score))
            else:
                return -6.0 - (4.0 * score)

        elif action == self.ACTION_EDIT:
            if violations:
                return 2.0 + (3.0 * min(1.0, len(violations) / 3.0))
            else:
                return -2.0

        return 0.0

    def state(self) -> dict[str, Any]:
        """Get current environment state."""
        return {
            "step_count": self._step_count,
            "episode_ended": self._episode_ended,
            "content": self._content,
            "violations": self._violations.copy() if self._violations else [],
            "current_observation": self._current_observation.model_dump() if self._current_observation else None,
            "max_steps": self.max_steps,
            "difficulty": self.difficulty,
        }

    async def render(self, mode: str = "text") -> str:
        """Render the current environment state."""
        if mode == "json":
            import json
            return json.dumps(self.state(), indent=2)

        lines = [
            "=" * 50,
            f"Step: {self._step_count}/{self.max_steps}",
            f"Content: {self._content[:60]}...",
            f"Violations: {self._violations}",
            f"Score: {self._current_observation.score:.2f}" if self._current_observation else "Score: N/A",
            f"Episode Ended: {self._episode_ended}",
            "=" * 50,
        ]
        return "\n".join(lines)

    async def close(self) -> None:
        """Clean up environment resources."""
        self._evaluator = None
        self._episode_ended = True
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
