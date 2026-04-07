"""
Legacy Gymnasium Environment for Content Compliance RL.

Provides a Gymnasium-compatible wrapper around the OpenEnv environment
for compatibility with existing RL training code.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

import gymnasium as gym
import numpy as np

from openenv import Action, ContentComplianceEnv, Observation


class ContentComplianceGymEnv(gym.Env):
    """
    Gymnasium wrapper for Content Compliance Environment.

    Action Space:
        Discrete(3): [approve, reject, edit]

    Observation Space:
        Dict with content, violations, score, step_count
    """

    metadata = {"render_modes": ["text", "json"]}

    def __init__(
        self,
        max_steps: int = 5,
        difficulty: str = "mixed",
        evaluator_provider: str = "mock",
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.max_steps = max_steps
        self.difficulty = difficulty
        self.evaluator_provider = evaluator_provider
        self.render_mode = render_mode

        self._env = ContentComplianceEnv(
            max_steps=max_steps,
            difficulty=difficulty,
            evaluator_provider=evaluator_provider,
        )

        self.action_space = gym.spaces.Discrete(3)

        self.observation_space = gym.spaces.Dict({
            "content": gym.spaces.Text(5000),
            "violations": gym.spaces.Sequence(gym.spaces.Text(50)),
            "score": gym.spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32),
            "step_count": gym.spaces.Box(low=0, high=max_steps, shape=(), dtype=np.int32),
        })

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._current_observation: Optional[Observation] = None

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create event loop."""
        if self._loop is None or self._loop.is_closed():
            try:
                self._loop = asyncio.get_event_loop()
            except RuntimeError:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
        return self._loop

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed)

        loop = self._get_loop()
        self._current_observation = loop.run_until_complete(self._env.reset())

        obs = self._obs_to_dict(self._current_observation)
        info = {"initial_score": self._current_observation.score}

        return obs, info

    def step(
        self,
        action: int,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Take a step in the environment."""
        action_names = ["approve", "reject", "edit"]
        action_type = action_names[action]

        loop = self._get_loop()
        action_obj = Action(action_type=action_type)

        result = loop.run_until_complete(self._env.step(action_obj))
        self._current_observation = result.observation

        obs = self._obs_to_dict(self._current_observation)
        reward = result.reward.value
        done = result.done
        truncated = result.truncated
        info = result.info

        return obs, reward, done, truncated, info

    def _obs_to_dict(self, obs: Observation) -> dict[str, Any]:
        """Convert Observation to Gymnasium dict."""
        return {
            "content": obs.content,
            "violations": obs.violations,
            "score": np.float32(obs.score),
            "step_count": np.int32(obs.step_count),
        }

    def render(self) -> Optional[str]:
        """Render the environment."""
        if self.render_mode == "text":
            loop = self._get_loop()
            return loop.run_until_complete(self._env.render("text"))
        elif self.render_mode == "json":
            loop = self._get_loop()
            return loop.run_until_complete(self._env.render("json"))
        return None

    def close(self) -> None:
        """Clean up resources."""
        if self._loop:
            loop = self._get_loop()
            loop.run_until_complete(self._env.close())
            if not self._loop.is_closed():
                self._loop.close()
