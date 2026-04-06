"""
Edit Evaluator for Content Compliance RL.

Evaluates the quality of content edits made by the agent:
- Did the edit remove violations?
- Did it preserve the original meaning?
- Was the edit minimal and targeted?

Provides detailed feedback and scoring for edit actions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class EditEvaluation:
    """Result of evaluating an edit action."""

    original_content: str
    edited_content: str
    original_violations: list[str]
    edited_violations: list[str]

    # Evaluation results
    violations_removed: bool = False
    violations_remaining: list[str] = field(default_factory=list)
    new_violations: list[str] = field(default_factory=list)

    # Quality metrics
    meaning_preserved: bool = True
    edit_quality: str = "good"  # "excellent", "good", "partial", "poor"
    preservation_score: float = 1.0  # 0.0 to 1.0

    # Scoring
    edit_score: float = 0.0  # Final score for this edit (0.0 to 1.0)
    reward_delta: float = 0.0  # Recommended reward delta

    # Explanation
    explanation: str = ""
    details: dict[str, Any] = field(default_factory=dict)


class EditEvaluator:
    """
    Evaluates edit actions for content moderation workflow.

    Assesses:
    1. Violation removal - did the edit fix the problems?
    2. Meaning preservation - did it keep the original intent?
    3. Edit quality - was it minimal and targeted?

    Example:
        >>> evaluator = EditEvaluator()
        >>> result = evaluator.evaluate(
        ...     original="Click here for FREE PRIZES!!!",
        ...     edited="Learn more about our rewards program.",
        ...     original_violations=["spam"],
        ... )
        >>> print(result.edit_score, result.explanation)
    """

    # Keywords that indicate spam/promotional content
    SPAM_INDICATORS = [
        "click here", "buy now", "free", "limited time", "act now",
        "!!!", "🔥", "💰", "👇", "bit.ly", "tinyurl",
    ]

    # Keywords that indicate harassment/toxic content
    TOXIC_INDICATORS = [
        "idiot", "stupid", "hate", "pathetic", "worthless",
        "ignorant", "delusional", "sad.",
    ]

    # Keywords that indicate misinformation
    MISINFO_INDICATORS = [
        "they don't want", "wake up", "truth about", "conspiracy",
        "hoax", "fake news",
    ]

    # Transition phrases that preserve meaning while softening tone
    MEANING_PRESERVING_TRANSITIONS = [
        ("i hate", "i disagree with"),
        ("they are stupid", "they are mistaken"),
        ("should all die", "should reconsider their position"),
        ("click here", "learn more"),
        ("buy now", "consider purchasing"),
    ]

    def __init__(self, provider: str = "openai"):
        """
        Initialize edit evaluator.

        Args:
            provider: Evaluation provider ("openai" or "fallback").
        """
        self.provider = provider
        self._openai_evaluator = None

        if provider == "openai":
            self._init_openai()

    def _init_openai(self) -> None:
        """Initialize OpenAI evaluator for edit assessment."""
        try:
            from graders.openai_evaluator import OpenAIEvaluator
            self._openai_evaluator = OpenAIEvaluator(
                timeout=10.0,
                max_retries=2,
            )
        except Exception as e:
            logger.warning(f"OpenAI evaluator init failed: {e}")
            self._openai_evaluator = None

    def _init_llm(self) -> None:
        """Initialize LLM client for semantic evaluation."""
        try:
            import os
            from openai import OpenAI

            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self._llm_client = OpenAI(api_key=api_key)
        except ImportError:
            pass
        except Exception:
            pass

    def evaluate(
        self,
        edited_content: str,
        original_content: str,
        original_violations: list[str],
    ) -> EditEvaluation:
        """
        Evaluate an edit action.

        Args:
            edited_content: The modified content after edit.
            original_content: The original content before edit.
            original_violations: Violations in original content.

        Returns:
            EditEvaluation with detailed assessment.
        """
        from graders.violation_detector import ViolationDetector

        # Detect violations in edited content
        edited_violations = ViolationDetector.detect(edited_content)

        # Create evaluation result
        result = EditEvaluation(
            original_content=original_content,
            edited_content=edited_content,
            original_violations=original_violations.copy(),
            edited_violations=edited_violations.copy(),
        )

        # Check violations removed
        result.violations_removed = len(edited_violations) < len(original_violations)
        result.violations_remaining = [
            v for v in edited_violations if v in original_violations
        ]
        result.new_violations = [
            v for v in edited_violations if v not in original_violations
        ]

        # Check meaning preservation
        result.meaning_preserved = self._check_meaning_preserved(
            original_content, edited_content
        )
        result.preservation_score = self._calculate_preservation_score(
            original_content, edited_content
        )

        # Determine edit quality
        result.edit_quality = self._determine_edit_quality(result)

        # Calculate edit score and reward
        result.edit_score, result.reward_delta = self._calculate_edit_score(result)

        # Generate explanation
        result.explanation = self._generate_explanation(result)

        # Add details
        result.details = {
            "violations_before": len(original_violations),
            "violations_after": len(edited_violations),
            "violation_reduction": len(original_violations) - len(edited_violations),
            "text_similarity": result.preservation_score,
            "quality": result.edit_quality,
        }

        return result

    def _check_meaning_preserved(
        self,
        original: str,
        edited: str,
    ) -> bool:
        """
        Check if the edit preserved the original meaning.

        Uses simple heuristics:
        - Length ratio (not too different)
        - Word overlap
        - No toxic replacement
        """
        original_lower = original.lower()
        edited_lower = edited.lower()

        # Check if completely different topic
        original_words = set(original_lower.split())
        edited_words = set(edited_lower.split())

        # Calculate word overlap
        if original_words and edited_words:
            overlap = len(original_words & edited_words) / max(len(original_words), len(edited_words))
            if overlap < 0.2:
                # Very low overlap - might be completely different content
                # Check if lengths are vastly different
                len_ratio = len(edited) / max(len(original), 1)
                if len_ratio < 0.3 or len_ratio > 3.0:
                    return False

        # Check for toxic replacements that change meaning negatively
        for toxic in self.TOXIC_INDICATORS:
            if toxic in original_lower and toxic not in edited_lower:
                # Toxic content removed - good, meaning changed but improved
                pass

        return True

    def _calculate_preservation_score(
        self,
        original: str,
        edited: str,
    ) -> float:
        """
        Calculate how well the edit preserved the original content.

        Returns score from 0.0 (completely different) to 1.0 (identical).
        """
        original_lower = original.lower()
        edited_lower = edited.lower()

        # Exact match = 1.0
        if original_lower == edited_lower:
            return 1.0

        # Calculate character-level similarity (simple ratio)
        len_orig = len(original_lower)
        len_edit = len(edited_lower)

        if len_orig == 0 and len_edit == 0:
            return 1.0

        # Find common substring ratio
        common_chars = sum(1 for c in original_lower if c in edited_lower)
        char_similarity = common_chars / max(len_orig, len_edit)

        # Word overlap
        original_words = set(original_lower.split())
        edited_words = set(edited_lower.split())

        if original_words or edited_words:
            word_overlap = len(original_words & edited_words)
            word_similarity = word_overlap / max(len(original_words), len(edited_words), 1)
        else:
            word_similarity = 0.0

        # Combined score (weighted)
        score = 0.4 * char_similarity + 0.6 * word_similarity

        return min(1.0, max(0.0, score))

    def _determine_edit_quality(self, result: EditEvaluation) -> str:
        """Determine the quality level of the edit."""
        violations_fixed = len(result.original_violations) - len(result.edited_violations)
        violations_total = len(result.original_violations)

        # Excellent: All violations removed, meaning preserved, minimal edit
        if violations_fixed == violations_total and violations_total > 0:
            if result.meaning_preserved and result.preservation_score > 0.5:
                return "excellent"
            elif result.meaning_preserved:
                return "good"

        # Good: Most violations removed
        if violations_fixed > 0 and violations_fixed >= violations_total * 0.5:
            if not result.new_violations:
                return "good"
            return "partial"

        # Partial: Some improvement but issues remain
        if violations_fixed > 0 or (result.violations_removed and not result.new_violations):
            return "partial"

        # Poor: No improvement or made worse
        if result.new_violations:
            return "poor"

        # No violations to fix - edit was unnecessary
        if violations_total == 0:
            return "unnecessary"

        return "poor"

    def _calculate_edit_score(
        self,
        result: EditEvaluation,
    ) -> tuple[float, float]:
        """
        Calculate edit score and recommended reward delta.

        Returns:
            Tuple of (edit_score, reward_delta)
            - edit_score: 0.0 to 1.0 quality score
            - reward_delta: Recommended reward adjustment
        """
        violations_fixed = len(result.original_violations) - len(result.edited_violations)
        violations_total = len(result.original_violations)

        base_score = 0.5

        # Bonus for removing violations
        if violations_total > 0:
            removal_ratio = violations_fixed / violations_total
            base_score += 0.3 * removal_ratio

            if violations_fixed == violations_total:
                # All violations removed - excellent
                base_score += 0.2
        else:
            # No violations to fix - edit was unnecessary
            base_score -= 0.2
            result.edit_quality = "unnecessary"

        # Penalty for new violations
        if result.new_violations:
            base_score -= 0.2 * len(result.new_violations)

        # Bonus for meaning preservation
        if result.meaning_preserved:
            base_score += 0.1 * result.preservation_score
        else:
            base_score -= 0.2

        # Clamp score
        edit_score = max(0.0, min(1.0, base_score))

        # Calculate reward delta based on quality
        quality_rewards = {
            "excellent": 0.8,
            "good": 0.6,
            "partial": 0.3,
            "poor": -0.3,
            "unnecessary": -0.2,
        }
        reward_delta = quality_rewards.get(result.edit_quality, 0.0)

        # Adjust reward for new violations
        if result.new_violations:
            reward_delta -= 0.2 * len(result.new_violations)

        return edit_score, reward_delta

    def _generate_explanation(self, result: EditEvaluation) -> str:
        """Generate human-readable explanation of edit evaluation."""
        parts = []

        # Violation status
        if result.violations_removed:
            fixed = len(result.original_violations) - len(result.edited_violations)
            parts.append(f"Fixed {fixed} violation(s)")
        elif result.original_violations and not result.edited_violations:
            parts.append("All violations removed")
        elif result.original_violations:
            parts.append(f"Violations remain: {', '.join(result.violations_remaining)}")
        else:
            parts.append("No violations to fix")

        # New violations
        if result.new_violations:
            parts.append(f"NEW violations: {', '.join(result.new_violations)}")

        # Quality assessment
        parts.append(f"Quality: {result.edit_quality}")

        # Meaning preservation
        if result.meaning_preserved:
            parts.append(f"Meaning preserved ({result.preservation_score:.0%})")
        else:
            parts.append("Meaning NOT preserved")

        return "; ".join(parts)

    def compare_content(
        self,
        original: str,
        edited: str,
    ) -> dict[str, Any]:
        """
        Compare original and edited content with detailed diff.

        Args:
            original: Original content.
            edited: Edited content.

        Returns:
            Dictionary with comparison details.
        """
        original_lower = original.lower()
        edited_lower = edited.lower()

        # Find what was removed
        original_words = set(original_lower.split())
        edited_words = set(edited_lower.split())

        removed_words = original_words - edited_words
        added_words = edited_words - original_words

        # Character changes
        length_change = len(edited) - len(original)

        # Check for specific patterns removed
        patterns_removed = []
        patterns_added = []

        all_patterns = self.SPAM_INDICATORS + self.TOXIC_INDICATORS + self.MISINFO_INDICATORS

        for pattern in all_patterns:
            if pattern in original_lower and pattern not in edited_lower:
                patterns_removed.append(pattern)
            if pattern not in original_lower and pattern in edited_lower:
                patterns_added.append(pattern)

        return {
            "original_length": len(original),
            "edited_length": len(edited),
            "length_change": length_change,
            "removed_words": list(removed_words),
            "added_words": list(added_words),
            "patterns_removed": patterns_removed,
            "patterns_added": patterns_added,
            "similarity_score": self._calculate_preservation_score(original, edited),
        }


# Convenience function
def evaluate_edit(
    edited_content: str,
    original_content: str,
    original_violations: list[str],
) -> EditEvaluation:
    """Evaluate an edit using default evaluator."""
    evaluator = EditEvaluator()
    return evaluator.evaluate(edited_content, original_content, original_violations)
