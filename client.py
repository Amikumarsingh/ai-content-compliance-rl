"""
Typed client for Content Compliance RL Environment.
Extends EnvClient for WebSocket-based persistent sessions.
"""
from __future__ import annotations
from typing import Any, Dict
from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from models import ContentAction, ContentObservation


class ContentComplianceEnvClient(EnvClient[ContentAction, ContentObservation, Dict[str, Any]]):
    """WebSocket client for the Content Compliance RL Environment."""

    def _step_payload(self, action: ContentAction) -> Dict[str, Any]:
        return action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[ContentObservation]:
        # OpenEnv server returns: {"observation": {...}, "reward": float, "done": bool}
        obs_data = payload.get("observation", payload)
        reward   = payload.get("reward") or obs_data.get("reward")
        done     = payload.get("done", obs_data.get("done", False))
        obs = ContentObservation(
            content=obs_data.get("content", ""),
            violations=obs_data.get("violations", []),
            compliance_score=obs_data.get("compliance_score", 0.5),
            stage=obs_data.get("stage", "classification"),
            step_count=obs_data.get("step_count", 0),
            task_description=obs_data.get("task_description", ""),
            feedback=obs_data.get("feedback", ""),
            done=done,
            reward=reward,
        )
        return StepResult(observation=obs, reward=reward, done=done)

    def _parse_state(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return payload
