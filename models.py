"""
Typed models for Content Compliance RL Environment.
Extends OpenEnv base classes for full spec compliance.
"""
from __future__ import annotations
from typing import List, Optional
from pydantic import Field
from openenv.core.env_server.types import Action, Observation, State


class ContentAction(Action):
    """Action taken by the agent in the content moderation environment."""
    action_type: str = Field(
        ...,
        description="Moderation decision: 'approve', 'reject', or 'edit'",
    )
    edited_content: Optional[str] = Field(
        default=None,
        description="Rewritten content (required when action_type is 'edit')",
    )


class ContentObservation(Observation):
    """Observation returned by the content moderation environment."""
    content: str = Field(default="", description="User-generated content to evaluate")
    violations: List[str] = Field(
        default_factory=list,
        description="Detected policy violation types",
    )
    compliance_score: float = Field(
        default=0.5,
        description="Compliance score 0.0 (violating) to 1.0 (clean)",
    )
    stage: str = Field(
        default="classification",
        description="Current workflow stage: classification | decision | refinement",
    )
    step_count: int = Field(default=0, description="Steps taken in this episode")
    task_description: str = Field(
        default="", description="Instruction for the current step"
    )
    feedback: str = Field(default="", description="Feedback from previous action")


class ContentState(State):
    """Internal state of the content moderation environment."""
    difficulty: str = Field(default="mixed")
    initial_violations: List[str] = Field(default_factory=list)
    initial_score: float = Field(default=0.5)
    action_history: List[str] = Field(default_factory=list)
