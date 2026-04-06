"""
OpenEnv Content Compliance Environment - Multi-Step RL Workflow.

Async environment implementing sequential decision-making for content moderation:
- Stage 1: Classify content (detect violations)
- Stage 2: Choose action (approve/reject/edit)
- Stage 3: Refine/edit content if needed

Episode ends after correct terminal action OR max_steps reached.
Rewards accumulate across all steps.
"""

from __future__ import annotations

import logging
import os
from enum import Enum
from typing import Any, Optional

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from app.models import Action, Observation, Reward, StepResult
from app.reward import AdvancedRewardCalculator
from app.state_manager import StateManager

logger = logging.getLogger(__name__)


class Stage(str, Enum):
    """Decision stages in the multi-step workflow."""
    CLASSIFICATION = "classification"  # Detect violations
    DECISION = "decision"  # Choose approve/reject/edit
    REFINEMENT = "refinement"  # Edit content if needed


class ContentComplianceEnv:
    """
    Multi-Step Content Compliance Environment (OpenEnv-compatible).

    Sequential Decision Workflow:
    1. CLASSIFICATION: Analyze content, detect violations
    2. DECISION: Choose approve/reject/edit based on analysis
    3. REFINEMENT: Edit content if edit was chosen, then return to DECISION

    Action Space:
        - approve: Content passes compliance (terminal)
        - reject: Content violates policies (terminal)
        - edit: Content needs modification (continues to REFINEMENT)

    Observation Space:
        - content: str (text to evaluate)
        - violations: List[str] (detected violations)
        - score: float (0.0 to 1.0 compliance)
        - step_count: int (current step)
        - stage: str (current decision stage)
        - previous_action: str (last action taken)
        - action_history: List[str] (all actions this episode)

    Reward:
        Accumulated across steps, normalized to [0.0, 1.0]:
        - Action correctness at each step
        - Improvement bonus for edits that fix violations
        - Efficiency bonus for completing in optimal steps
    """

    ACTION_APPROVE = "approve"
    ACTION_REJECT = "reject"
    ACTION_EDIT = "edit"

    # Stage transition rules
    STAGE_ACTIONS = {
        Stage.CLASSIFICATION: ["approve", "reject", "edit"],  # Can skip to decision
        Stage.DECISION: ["approve", "reject", "edit"],
        Stage.REFINEMENT: ["approve", "reject", "edit"],  # After edit, must decide
    }

    # Optimal steps per difficulty
    OPTIMAL_STEPS = {
        "easy": 2,      # classify + decide
        "medium": 3,    # classify + decide + refine
        "hard": 4,      # classify + decide + refine + re-decide
    }

    def __init__(
        self,
        max_steps: int = 5,
        difficulty: str = "mixed",
        evaluator_provider: str = "mock",
    ):
        """
        Initialize the content compliance environment.

        Args:
            max_steps: Maximum steps per episode (default: 5).
            difficulty: Content difficulty ("easy", "medium", "hard", "mixed").
            evaluator_provider: LLM provider ("mock" or "openai").
        """
        self.max_steps = max_steps
        self.difficulty = difficulty
        self.evaluator_provider = evaluator_provider

        self._state_manager = StateManager(difficulty=difficulty)
        self._episode_ended: bool = False
        self._reward_calculator: Optional[AdvancedRewardCalculator] = None
        self._info: dict[str, Any] = {}

        # Multi-step state tracking
        self._current_stage: Stage = Stage.CLASSIFICATION
        self._previous_action: Optional[str] = None
        self._action_history: list[str] = []
        self._initial_violations: list[str] = []
        self._initial_score: float = 0.0
        self._previous_violations: list[str] = []
        self._previous_score: float = 0.0

        # Score evolution tracking
        self._score_history: list[float] = []
        self._final_score: float = 0.0

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
        item = loader.get_random_content_sync()

        violations = []
        if not item.is_compliant:
            violations = self._infer_violations(item.text)

        score = 0.9 if item.is_compliant else 0.2

        return item.text, violations, score

    def _infer_violations(self, content: str) -> list[str]:
        """Infer violations from content using rule-based detection."""
        from graders.violation_detector import ViolationDetector

        return ViolationDetector.detect(content)

    def _get_expected_action(self, score: float, violations: list[str]) -> str:
        """Determine the expected action based on compliance state."""
        if len(violations) == 0 and score >= 0.5:
            return "approve"
        else:
            return "reject"

    def _determine_next_stage(
        self,
        current_stage: Stage,
        action: str,
        violations: list[str],
    ) -> Stage:
        """Determine the next stage based on current stage and action."""
        if action == "edit":
            return Stage.REFINEMENT
        elif action in ["approve", "reject"]:
            return Stage.DECISION  # Terminal actions stay in decision for validation
        else:
            return Stage.DECISION

    def _is_correct_terminal_action(
        self,
        action: str,
        violations: list[str],
        score: float,
    ) -> bool:
        """Check if terminal action is correct given content state."""
        is_compliant = len(violations) == 0 and score >= 0.5

        if action == "approve":
            return is_compliant
        elif action == "reject":
            return not is_compliant
        return False

    async def reset(self) -> Observation:
        """
        Reset the environment and start a new episode.

        Returns:
            Initial observation with fresh content at CLASSIFICATION stage.
        """
        self._episode_ended = False
        self._reward_calculator = AdvancedRewardCalculator(task_name=self._get_task_name())

        content, violations, score = self._generate_content()

        self._state_manager.reset(
            content=content,
            violations=violations,
            score=score,
        )

        # Store initial state for reward calculation
        self._initial_violations = violations.copy()
        self._initial_score = score
        self._previous_violations = violations.copy()
        self._previous_score = score

        # Initialize score evolution tracking
        self._score_history = [score]
        self._final_score = score

        # Initialize advanced reward calculator
        self._reward_calculator.start_episode(
            initial_violations=violations,
            initial_score=score,
        )

        # Reset multi-step tracking
        self._current_stage = Stage.CLASSIFICATION
        self._previous_action = None
        self._action_history = []

        observation = self._create_observation(
            content=content,
            violations=violations,
            score=score,
            step_count=0,
        )

        self._info = {
            "initial_score": score,
            "initial_violations": violations,
            "expected_action": self._get_expected_action(score, violations),
            "stage": self._current_stage.value,
            "source": "dynamic",
        }

        logger.info(
            f"Episode started: content length={len(content)}, "
            f"violations={len(violations)}, score={score:.2f}, "
            f"stage={self._current_stage.value}"
        )

        return observation

    def _create_observation(
        self,
        content: str,
        violations: list[str],
        score: float,
        step_count: int,
    ) -> Observation:
        """Create observation with multi-step information."""
        obs = Observation(
            content=content,
            violations=violations,
            score=score,
            step_count=step_count,
            stage=self._current_stage.value,
            previous_action=self._previous_action,
            action_history=self._action_history.copy(),
        )
        return obs

    async def step(self, action: Action) -> StepResult:
        """
        Take a step in the multi-step environment.

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

        # Validate action
        if action.action_type == "edit" and not action.edited_content:
            raise ValueError("Edit action requires edited_content")

        # Track action
        self._previous_action = action.action_type
        self._action_history.append(action.action_type)

        # Get content to evaluate
        content_to_eval = action.edited_content if action.edited_content else current_state.content

        # Evaluate content after action
        eval_result = self._evaluate_content(content_to_eval, action.action_type)
        current_violations = eval_result.get("violations", [])
        current_score = eval_result.get("score", current_state.score)

        # Special handling for edit actions
        edit_evaluation = None
        if action.action_type == "edit" and action.edited_content:
            edit_evaluation = self._evaluate_edit(
                edited_content=action.edited_content,
                original_content=current_state.content,
                original_violations=self._previous_violations,
            )

        # Determine if this is a correct terminal action
        is_correct_terminal = self._is_correct_terminal_action(
            action.action_type,
            current_violations,
            current_score,
        )

        # Update state manager BEFORE reward calculation (so we have previous state)
        self._state_manager.update(
            action=action.action_type,
            new_content=content_to_eval if action.action_type == "edit" else None,
            new_violations=current_violations,
            new_score=current_score,
        )

        # Calculate reward using advanced calculator
        if self._reward_calculator:
            self._reward_calculator.record_step(
                action=action.action_type,
                current_violations=current_violations,
                current_score=current_score,
                previous_violations=self._previous_violations,
                previous_score=self._previous_score,
                is_terminal=action.action_type in ["approve", "reject"],
            )

            # Apply edit evaluation bonus/penalty
            if edit_evaluation:
                edit_score = edit_evaluation.get("edit_score", 0.5)
                reward_delta = edit_evaluation.get("reward_delta", 0.0)

                # Record edit quality in reward calculator
                if edit_score >= 0.8:
                    # Excellent edit - bonus
                    self._reward_calculator._breakdown.add_component(
                        name="edit_bonus",
                        delta=0.2,
                        description=f"Excellent edit (score={edit_score:.2f}): +0.2",
                    )
                elif edit_score >= 0.5:
                    # Good edit
                    self._reward_calculator._breakdown.add_component(
                        name="edit_good",
                        delta=0.1,
                        description=f"Good edit (score={edit_score:.2f}): +0.1",
                    )
                elif edit_score < 0.3:
                    # Poor edit - penalty
                    self._reward_calculator._breakdown.add_component(
                        name="edit_penalty",
                        delta=-0.2,
                        description=f"Poor edit (score={edit_score:.2f}): -0.2",
                    )

                # Penalty for new violations introduced
                if edit_evaluation.get("new_violations"):
                    self._reward_calculator._breakdown.add_component(
                        name="new_violations_penalty",
                        delta=-0.3,
                        description=f"Edit introduced new violations: -0.3",
                    )

        # Update stage
        self._current_stage = self._determine_next_stage(
            self._current_stage,
            action.action_type,
            current_violations,
        )

        # Update previous state for next step
        self._previous_violations = current_violations.copy()
        self._previous_score = current_score

        # Track score evolution
        self._score_history.append(current_score)
        self._final_score = current_score

        # Determine episode termination
        # Episode ends if:
        # 1. Correct terminal action (approve/reject with correct decision)
        # 2. Max steps reached
        correct_decision_made = action.action_type in ["approve", "reject"] and is_correct_terminal
        max_steps_reached = current_state.step_count >= self.max_steps

        done = correct_decision_made or max_steps_reached
        truncated = max_steps_reached and not correct_decision_made

        # Get reward from calculator
        if self._reward_calculator:
            if done:
                # Calculate improvement bonus
                improvement = current_score - self._initial_score
                improvement_bonus = 0.0

                if improvement > 0:
                    # Large improvement bonus (scaled by improvement magnitude)
                    if improvement >= 0.7:
                        improvement_bonus = 0.3  # Maximum bonus for dramatic improvement
                    elif improvement >= 0.5:
                        improvement_bonus = 0.2
                    elif improvement >= 0.3:
                        improvement_bonus = 0.1
                    else:
                        improvement_bonus = 0.05

                    # Add improvement bonus to reward calculator
                    self._reward_calculator._breakdown.add_component(
                        name="improvement_bonus",
                        delta=improvement_bonus,
                        description=f"Score improved by {improvement:.2f}: +{improvement_bonus}",
                    )

                # Finalize episode and get final reward
                reward_value, reward_explanation = self._reward_calculator.finalize_episode(
                    final_action=action.action_type,
                    final_violations=current_violations,
                    final_score=current_score,
                )
            else:
                # Get intermediate reward
                reward_value, reward_explanation = self._reward_calculator.get_current_reward()
        else:
            reward_value = 0.5
            reward_explanation = "No reward calculator initialized"

        # Build reward object
        if truncated:
            reward = Reward(
                value=0.1,
                raw_value=-5.0,
                explanation="Episode truncated - no correct terminal action within max steps"
            )
        else:
            reward = Reward(
                value=reward_value,
                raw_value=reward_value,
                explanation=reward_explanation
            )

        # Update observation
        observation = self._create_observation(
            content=content_to_eval,
            violations=current_violations,
            score=current_score,
            step_count=current_state.step_count,
        )

        # Update info with multi-step context
        self._info = {
            "action": action.action_type,
            "previous_action": self._previous_action,
            "action_history": self._action_history.copy(),
            "stage": self._current_stage.value,
            "compliance_score": current_score,
            "violations": current_violations,
            "suggestion": eval_result.get("suggestion", ""),
            "edited": action.edited_content is not None,
            "is_correct_terminal": is_correct_terminal,
        }

        # Add edit evaluation if applicable
        if edit_evaluation:
            self._info["edit_evaluation"] = edit_evaluation
            self._info["edit_quality"] = edit_evaluation.get("edit_quality", "unknown")
            self._info["violations_removed"] = edit_evaluation.get("violations_removed", False)
            self._info["new_violations"] = edit_evaluation.get("new_violations", [])

        if done:
            self._state_manager.complete_episode()
            self._episode_ended = True

            # Get final reward breakdown
            if self._reward_calculator:
                breakdown = self._reward_calculator.get_breakdown()
                self._info["reward_breakdown"] = breakdown
                self._info["reward_explanation"] = breakdown.get("explanation", "")

            # Add score evolution info
            self._info["score_evolution"] = {
                "initial_score": self._initial_score,
                "final_score": self._final_score,
                "improvement": self._final_score - self._initial_score,
                "score_history": self._score_history.copy(),
            }

        logger.info(
            f"Step {current_state.step_count}: action={action.action_type}, "
            f"stage={self._current_stage.value}, reward={reward_value:.3f}, "
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
        """Evaluate content using OpenAI evaluator with fallback."""
        from graders.openai_evaluator import OpenAIEvaluator

        # Use OpenAI evaluator (with automatic fallback if unavailable)
        evaluator = OpenAIEvaluator(
            timeout=10.0,  # 10 second timeout
            max_retries=2,  # Retry up to 2 times
        )

        result = evaluator.evaluate(content)

        suggestion = ""
        if result.violations:
            suggestion = f"Remove: {', '.join(result.violations)}"

        return {
            "score": result.score,
            "violations": result.violations,
            "suggestion": suggestion,
            "reason": result.reason,
            "confidence": result.confidence,
            "evaluator_source": result.source,  # "openai" or "fallback"
        }

    def _evaluate_edit(
        self,
        edited_content: str,
        original_content: str,
        original_violations: list[str],
    ) -> dict[str, Any]:
        """
        Evaluate an edit action.

        Args:
            edited_content: Content after edit.
            original_content: Content before edit.
            original_violations: Violations in original content.

        Returns:
            Dictionary with edit evaluation results.
        """
        from graders.edit_evaluator import EditEvaluator

        evaluator = EditEvaluator(provider=self.evaluator_provider)
        evaluation = evaluator.evaluate(
            edited_content=edited_content,
            original_content=original_content,
            original_violations=original_violations,
        )

        return {
            "edit_score": evaluation.edit_score,
            "reward_delta": evaluation.reward_delta,
            "violations_removed": evaluation.violations_removed,
            "violations_remaining": evaluation.violations_remaining,
            "new_violations": evaluation.new_violations,
            "meaning_preserved": evaluation.meaning_preserved,
            "edit_quality": evaluation.edit_quality,
            "explanation": evaluation.explanation,
            "details": evaluation.details,
        }

    def state(self) -> dict[str, Any]:
        """Get current environment state with multi-step info."""
        base_state = self._state_manager.get_state()
        base_state["current_stage"] = self._current_stage.value
        base_state["previous_action"] = self._previous_action
        base_state["action_history"] = self._action_history.copy()
        base_state["initial_violations"] = self._initial_violations
        base_state["initial_score"] = self._initial_score
        return base_state

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
                "stage": {"type": "str", "enum": ["classification", "decision", "refinement"]},
                "previous_action": {"type": "str", "nullable": True},
                "action_history": {"type": "list[str]"},
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
                "edit": "Content needs modification (continues to refinement)",
            },
            "workflow": {
                "classification": "Initial analysis stage",
                "decision": "Choose terminal action or edit",
                "refinement": "Edit content then re-decide",
            },
        }

    def get_score_evolution(self) -> dict[str, Any]:
        """
        Get the score evolution for the current episode.

        Returns:
            Dictionary with score tracking information.
        """
        return {
            "initial_score": self._initial_score,
            "final_score": self._final_score,
            "improvement": self._final_score - self._initial_score,
            "score_history": self._score_history.copy(),
            "steps": len(self._score_history),
        }

    @staticmethod
    def visualize_score_trend(score_history: list[float], title: str = "Score Evolution") -> str:
        """
        Create ASCII visualization of score trend.

        Args:
            score_history: List of scores over time.
            title: Title for the visualization.

        Returns:
            ASCII art chart string (ASCII-safe for Windows console).

        Example:
            >>> env = ContentComplianceEnv()
            >>> history = [0.2, 0.5, 0.7, 0.9]
            >>> print(env.visualize_score_trend(history))
        """
        if not score_history:
            return "No score data available"

        # Chart dimensions
        height = 8
        width = max(5, len(score_history) * 3)

        # Find min/max for scaling
        min_score = min(score_history)
        max_score = max(score_history)
        score_range = max(max_score - min_score, 0.1)  # Avoid division by zero

        # Build chart (ASCII-safe characters)
        lines = []
        lines.append(f"  {title}")
        lines.append(f"  {'=' * (width + 8)}")

        for row in range(height, -1, -1):
            line = "  0.0 |" if row == 0 else "      |"

            for col, score in enumerate(score_history):
                normalized = (score - min_score) / score_range
                chart_height = int(normalized * height)

                if col == 0:
                    if chart_height >= row:
                        line += "#"
                    else:
                        line += " "
                else:
                    if chart_height >= row:
                        line += "##"
                    elif chart_height == row - 1:
                        line += "#"
                    else:
                        line += " "

            # Add Y-axis label for top row
            if row == height:
                lines.append(line + " 1.0")
            else:
                lines.append(line)

        # X-axis
        lines.append(f"      +{'-' * len(score_history) * 2}")
        lines.append(f"       Step 0{' ' * (len(score_history) * 2 - 8)}Step {len(score_history) - 1}")

        # Summary
        improvement = score_history[-1] - score_history[0] if len(score_history) > 1 else 0
        if improvement > 0:
            trend = "UP"
        elif improvement < 0:
            trend = "DOWN"
        else:
            trend = "FLAT"
        lines.append(f"  Trend: {trend} ({improvement:+.2f})")

        return "\n".join(lines)
