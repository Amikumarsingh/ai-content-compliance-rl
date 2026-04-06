"""
OpenEnv Pydantic Models for Content Compliance RL.

Provides type-safe, validated data structures for:
- Observations (content + compliance state)
- Actions (approve/reject/edit)
- Rewards (normalized 0.0 to 1.0)
- Step results (environment transitions)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


class Observation(BaseModel):
    """
    Observation from the content compliance environment.

    Represents the current state of content being evaluated.

    Attributes:
        content: The text content being evaluated.
        violations: List of detected violation types.
        score: Compliance score from 0.0 to 1.0.
        step_count: Current step within the episode.
        stage: Current decision stage (classification/decision/refinement).
        previous_action: Previous action taken (None on first step).
        action_history: List of all actions taken in this episode.
        feedback: Feedback message from the environment.
    """

    content: str = Field(..., description="The text content being evaluated", max_length=1000)
    violations: list[str] = Field(
        default_factory=list,
        description="List of detected violation types"
    )
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Compliance score (0.0 = non-compliant, 1.0 = fully compliant)"
    )
    step_count: int = Field(
        default=0,
        ge=0,
        description="Current step within the episode"
    )
    stage: str = Field(
        default="decision",
        description="Current decision stage: classification, decision, or refinement"
    )
    previous_action: Optional[str] = Field(
        default=None,
        description="Previous action taken (None on first step)"
    )
    action_history: list[str] = Field(
        default_factory=list,
        description="List of all actions taken in this episode"
    )
    feedback: Optional[str] = Field(
        default=None,
        description="Feedback message from the environment"
    )

    @field_validator("score")
    @classmethod
    def validate_score(cls, v: float) -> float:
        """Ensure score is in valid range [0.0, 1.0]."""
        return max(0.0, min(1.0, v))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump()


class Action(BaseModel):
    """
    Action taken in the content compliance environment.

    Attributes:
        action_type: Type of action ("approve", "reject", or "edit").
        edited_content: Optional edited content if action_type is "edit".
    """

    action_type: str = Field(
        ...,
        description="Type of action: 'approve', 'reject', or 'edit'"
    )
    edited_content: Optional[str] = Field(
        default=None,
        description="Edited content (required if action_type is 'edit')"
    )

    @field_validator("action_type")
    @classmethod
    def validate_action_type(cls, v: str) -> str:
        """Ensure action_type is valid."""
        valid_types = {"approve", "reject", "edit"}
        if v.lower() not in valid_types:
            raise ValueError(f"action_type must be one of {valid_types}, got '{v}'")
        return v.lower()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump()


class Reward(BaseModel):
    """
    Reward signal from the environment.

    Normalized to [0.0, 1.0] range for consistency.

    Attributes:
        value: Normalized reward value (0.0 = worst, 1.0 = best).
        raw_value: Optional raw reward before normalization.
        explanation: Optional explanation of the reward.
    """

    value: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Normalized reward value (0.0 to 1.0)"
    )
    raw_value: Optional[float] = Field(
        default=None,
        description="Raw reward value before normalization"
    )
    explanation: Optional[str] = Field(
        default=None,
        description="Explanation of the reward"
    )

    @field_validator("value")
    @classmethod
    def validate_value(cls, v: float) -> float:
        """Ensure value is in valid range [0.0, 1.0]."""
        return max(0.0, min(1.0, v))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump()


@dataclass
class StepResult:
    """
    Result of taking a step in the environment.

    OpenEnv-compatible step result containing all information
    needed for RL training.

    Attributes:
        observation: New observation after taking the action.
        reward: Reward signal (normalized 0.0 to 1.0).
        done: Whether the episode has terminated.
        truncated: Whether the episode was truncated (time limit).
        info: Additional information dictionary.
    """

    observation: Observation
    reward: Reward
    done: bool = False
    truncated: bool = False
    info: dict[str, Any] = field(default_factory=dict)

    def to_tuple(self) -> tuple[Observation, Reward, bool, bool, dict[str, Any]]:
        """Convert to Gymnasium-style tuple."""
        return (self.observation, self.reward, self.done, self.truncated, self.info)
