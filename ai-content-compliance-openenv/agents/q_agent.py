"""
Q-Learning Agent for Content Compliance RL.

Implements a Q-learning agent with experience replay buffer
for training on content moderation tasks.
"""

from __future__ import annotations

import json
import random
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class Experience:
    """Single experience tuple for replay buffer."""

    state: str
    action: str
    reward: float
    next_state: str
    done: bool


class QLearningAgent:
    """
    Q-Learning agent with experience replay.

    Features:
    - Epsilon-greedy exploration
    - Experience replay buffer
    - Q-table based learning
    - Checkpoint save/load
    """

    def __init__(
        self,
        actions: list[str] | None = None,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        replay_buffer_size: int = 10000,
    ):
        self.actions = actions or ["approve", "reject", "edit"]
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self._q_table: dict[str, dict[str, float]] = {}
        self._replay_buffer: deque[Experience] = deque(maxlen=replay_buffer_size)
        self._training_history: list[dict[str, Any]] = []

    def get_q_values(self, state: str) -> dict[str, float]:
        """Get Q-values for a state."""
        if state not in self._q_table:
            self._q_table[state] = {a: 0.0 for a in self.actions}
        return self._q_table[state].copy()

    def select_action(self, state: str, training: bool = True) -> str:
        """Select action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            return random.choice(self.actions)

        q_values = self.get_q_values(state)
        max_q = max(q_values.values())
        best_actions = [a for a, q in q_values.items() if q == max_q]
        return random.choice(best_actions)

    def store_experience(
        self,
        state: str,
        action: str,
        reward: float,
        next_state: str,
        done: bool,
    ) -> None:
        """Store experience in replay buffer."""
        self._replay_buffer.append(
            Experience(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
            )
        )

    def update_q_table(
        self,
        state: str,
        action: str,
        reward: float,
        next_state: str,
        done: bool,
    ) -> float:
        """Update Q-value for state-action pair."""
        current_q = self.get_q_values(state)[action]
        next_max_q = max(self.get_q_values(next_state).values()) if not done else 0.0

        td_target = reward + self.discount_factor * next_max_q
        td_error = td_target - current_q

        new_q = current_q + self.learning_rate * td_error
        self._q_table[state][action] = new_q

        return new_q

    def train_batch(self, batch_size: int = 32) -> float:
        """Train on a batch of experiences."""
        if len(self._replay_buffer) < batch_size:
            return 0.0

        total_loss = 0.0
        batch = random.sample(list(self._replay_buffer), batch_size)

        for exp in batch:
            old_q = self.get_q_values(exp.state)[exp.action]
            self.update_q_table(
                exp.state, exp.action, exp.reward, exp.next_state, exp.done
            )
            new_q = self.get_q_values(exp.state)[exp.action]
            total_loss += abs(new_q - old_q)

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return total_loss / batch_size

    def save_checkpoint(self, path: str | Path, episode: int) -> None:
        """Save agent checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "episode": episode,
            "q_table": self._q_table,
            "epsilon": self.epsilon,
            "training_history": self._training_history,
        }

        with open(path, "w") as f:
            json.dump(checkpoint, f, indent=2)

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        """Load agent checkpoint."""
        path = Path(path)

        with open(path, "r") as f:
            checkpoint = json.load(f)

        self._q_table = checkpoint.get("q_table", {})
        self.epsilon = checkpoint.get("epsilon", self.epsilon_min)
        self._training_history = checkpoint.get("training_history", [])

        return checkpoint

    def record_episode(
        self,
        episode: int,
        reward: float,
        steps: int,
        success: bool = False,
    ) -> None:
        """Record episode statistics."""
        self._training_history.append({
            "episode": episode,
            "reward": reward,
            "steps": steps,
            "success": success,
            "epsilon": self.epsilon,
        })

    def get_statistics(self) -> dict[str, Any]:
        """Get training statistics."""
        if not self._training_history:
            return {"count": 0}

        rewards = [h["reward"] for h in self._training_history]
        successes = [h["success"] for h in self._training_history]

        return {
            "count": len(self._training_history),
            "avg_reward": sum(rewards) / len(rewards),
            "success_rate": sum(successes) / len(successes),
            "current_epsilon": self.epsilon,
            "q_table_size": len(self._q_table),
        }

    def reset(self) -> None:
        """Reset agent state (keep Q-table)."""
        self._replay_buffer.clear()
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
