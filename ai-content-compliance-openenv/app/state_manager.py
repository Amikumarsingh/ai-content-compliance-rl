"""
State Manager for Content Compliance RL Environment.

Handles episode state tracking and content history.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class EpisodeState:
    """Tracks the state of a single episode."""

    episode_id: str
    content: str = ""
    original_content: str = ""
    violations: list[str] = field(default_factory=list)
    score: float = 0.0
    step_count: int = 0
    actions_taken: list[str] = field(default_factory=list)
    is_complete: bool = False
    start_time: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "episode_id": self.episode_id,
            "content": self.content,
            "original_content": self.original_content,
            "violations": self.violations,
            "score": self.score,
            "step_count": self.step_count,
            "actions_taken": self.actions_taken,
            "is_complete": self.is_complete,
            "duration_seconds": (datetime.now() - self.start_time).total_seconds(),
        }


class StateManager:
    """
    Manages environment state across episodes.

    Features:
    - Episode lifecycle management
    - Content tracking
    - State persistence
    """

    def __init__(self, difficulty: str = "mixed"):
        self.difficulty = difficulty
        self.current_episode: Optional[EpisodeState] = None
        self.episode_history: list[EpisodeState] = []
        self._episode_counter = 0

    def generate_episode_id(self) -> str:
        """Generate a unique episode ID."""
        self._episode_counter += 1
        timestamp = datetime.now().isoformat()
        content_hash = hashlib.sha256(
            f"{timestamp}{self._episode_counter}".encode()
        ).hexdigest()[:8]
        return f"ep_{content_hash}"

    def reset(self, content: str, violations: list[str], score: float) -> EpisodeState:
        """Start a new episode with initial content."""
        episode_id = self.generate_episode_id()

        self.current_episode = EpisodeState(
            episode_id=episode_id,
            content=content,
            original_content=content,
            violations=violations.copy(),
            score=score,
        )

        return self.current_episode

    def update(
        self,
        action: str,
        new_content: Optional[str] = None,
        new_violations: Optional[list[str]] = None,
        new_score: Optional[float] = None,
    ) -> EpisodeState:
        """Update the current episode state."""
        if self.current_episode is None:
            raise RuntimeError("No active episode. Call reset() first.")

        self.current_episode.step_count += 1
        self.current_episode.actions_taken.append(action)

        if new_content:
            self.current_episode.content = new_content

        if new_violations is not None:
            self.current_episode.violations = new_violations.copy()

        if new_score is not None:
            self.current_episode.score = new_score

        return self.current_episode

    def complete_episode(self) -> EpisodeState:
        """Mark the current episode as complete."""
        if self.current_episode is None:
            raise RuntimeError("No active episode.")

        self.current_episode.is_complete = True
        self.episode_history.append(self.current_episode)

        return self.current_episode

    def get_state(self) -> dict[str, Any]:
        """Get current environment state."""
        return {
            "current_episode": self.current_episode.to_dict() if self.current_episode else None,
            "total_episodes": len(self.episode_history),
            "difficulty": self.difficulty,
        }

    def clear(self) -> None:
        """Reset state manager (keep history, clear current episode)."""
        self.current_episode = None

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics across all completed episodes."""
        if not self.episode_history:
            return {
                "total_episodes": 0,
                "avg_steps": 0,
                "avg_score": 0,
            }

        completed = [ep for ep in self.episode_history if ep.is_complete]
        if not completed:
            return {
                "total_episodes": 0,
                "avg_steps": 0,
                "avg_score": 0,
            }

        total_steps = sum(ep.step_count for ep in completed)
        total_scores = sum(ep.score for ep in completed)

        return {
            "total_episodes": len(completed),
            "avg_steps": total_steps / len(completed),
            "avg_score": total_scores / len(completed),
            "difficulty": self.difficulty,
        }
