"""
Violation Detector for Grading.

Rule-based violation detector used by the grader system to
deterministically identify policy violations in content.

This detector is used for grading Task 3 (Hard Complex) to
evaluate whether edits successfully removed violations.

Violations Detected:
    - hate_speech: Language targeting groups with hate/slurs
    - violence: Threats or glorification of violence
    - harassment: Personal attacks, insults, demeaning language
    - adult_content: Explicit or adult-themed content
    - misinformation: False claims, conspiracy theories
    - spam: Promotional, clickbait, excessive punctuation
    - illegal_content: References to illegal activities
"""

from __future__ import annotations

from typing import Any


class ViolationDetector:
    """
    Rule-based violation detector for grading purposes.

    Uses keyword pattern matching to detect policy violations.
    Deterministic - same input always produces same output.
    """

    # Pattern mappings for each violation type
    VIOLATION_PATTERNS: dict[str, list[str]] = {
        "hate_speech": [
            "hate",
            "slur",
            "should all die",
            "deserve to die",
            "kill all",
        ],
        "violence": [
            "kill",
            "attack",
            "hurt",
            "violence",
            "weapon",
            "should be attacked",
        ],
        "harassment": [
            "idiot",
            "stupid",
            "pathetic",
            "worthless",
            "terrible",
            "garbage",
            "ignorant",
            "delusional",
            "misguided",
            "sad.",
        ],
        "adult_content": [
            "explicit",
            "adult content",
            "nsfw",
            "porn",
            "dm for pics",
            "18+",
        ],
        "misinformation": [
            "fake news",
            "conspiracy",
            "hoax",
            "they hide",
            "truth they don't want",
            "they don't want you to know",
            "wake up",
            "truth about",
            "hiding from you",
        ],
        "spam": [
            "click here",
            "buy now",
            "!!!",
            "limited time",
            "make $",
            "win $",
            "visit ",
            "dm me",
            "free",
            "offer",
            "deal",
            "prizes",
            "exclusive",
            "limited offer",
        ],
        "suspicious_link": [
            "bit.ly",
            "tinyurl",
            "linktr.ee",
            "tinyurl.com",
        ],
        "engagement_bait": [
            "tag someone",
            "share if",
            "like this post",
            "comment '",
            "🔥",
            "💰",
            "👇",
            "📩",
            "❤️",
        ],
        "illegal_content": [
            "illegal",
            "drugs",
            "weapon",
        ],
    }

    @classmethod
    def detect(cls, content: str) -> list[str]:
        """
        Detect violations in content.

        Args:
            content: Text content to analyze

        Returns:
            List of detected violation types

        Example:
            >>> violations = ViolationDetector.detect(
            ...     "I hate those idiots! They should all die!"
            ... )
            >>> print(violations)
            ['hate_speech', 'harassment', 'violence']
        """
        content_lower = content.lower()
        detected = []

        for violation, patterns in cls.VIOLATION_PATTERNS.items():
            for pattern in patterns:
                if pattern in content_lower:
                    detected.append(violation)
                    break  # One match per violation type

        return detected

    @classmethod
    def has_violation(cls, content: str, violation_type: str) -> bool:
        """
        Check if content has a specific violation type.

        Args:
            content: Text content to analyze
            violation_type: Type of violation to check for

        Returns:
            True if violation is detected, False otherwise

        Example:
            >>> ViolationDetector.has_violation(
            ...     "Click here to win $1000!!!",
            ...     "spam"
            ... )
            True
        """
        return violation_type in cls.detect(content)

    @classmethod
    def count_violations(cls, content: str) -> int:
        """
        Count total number of violation types detected.

        Args:
            content: Text content to analyze

        Returns:
            Number of distinct violation types detected

        Example:
            >>> ViolationDetector.count_violations(
            ...     "I hate those idiots! Click here!!!"
            ... )
            3  # hate_speech, harassment, spam
        """
        return len(cls.detect(content))

    @classmethod
    def get_patterns(cls, violation_type: str) -> list[str]:
        """
        Get patterns for a specific violation type.

        Args:
            violation_type: The violation type to get patterns for

        Returns:
            List of patterns used to detect that violation

        Example:
            >>> patterns = ViolationDetector.get_patterns("spam")
            >>> print(patterns)
            ['click here', 'buy now', '!!!', ...]
        """
        return cls.VIOLATION_PATTERNS.get(violation_type, [])

    @classmethod
    def get_all_violation_types(cls) -> list[str]:
        """
        Get all supported violation types.

        Returns:
            List of all violation type names

        Example:
            >>> types = ViolationDetector.get_all_violation_types()
            >>> print(len(types))
            7
        """
        return list(cls.VIOLATION_PATTERNS.keys())
