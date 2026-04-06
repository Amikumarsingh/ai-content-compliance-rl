"""
OpenAI-Based Content Compliance Evaluator.

Production-grade evaluator using OpenAI API for realistic content moderation:
- Analyzes content for policy violations
- Scores compliance (0.0 to 1.0)
- Provides reasoning for decisions
- Falls back to rule-based evaluation if API fails

Features:
- Timeout handling
- Automatic retry (max 2 attempts)
- Graceful fallback to rule-based evaluation
- Structured JSON output
- No static labels - everything inferred via model
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result of OpenAI content evaluation."""

    content: str
    violations: list[str]
    score: float  # 0.0 to 1.0 compliance score
    reason: str
    is_compliant: bool
    confidence: float  # Model confidence (0.0 to 1.0)

    # Metadata
    model_used: str = ""
    latency_ms: float = 0.0
    source: str = "openai"  # "openai" or "fallback"
    raw_response: Optional[str] = None
    error: Optional[str] = None


class OpenAIEvaluator:
    """
    OpenAI-powered content compliance evaluator.

    Evaluates content for policy violations using LLM reasoning:
    - Detects subtle violations that rule-based systems miss
    - Provides nuanced compliance scores
    - Explains reasoning for transparency

    Falls back to rule-based evaluation if:
    - API is unavailable
    - Request times out
    - Response is invalid
    """

    # Evaluation categories
    VIOLATION_TYPES = [
        "hate_speech",
        "violence",
        "harassment",
        "adult_content",
        "misinformation",
        "spam",
        "suspicious_link",
        "engagement_bait",
        "illegal_content",
    ]

    # System prompt for consistent evaluation
    SYSTEM_PROMPT = """You are an expert content moderation evaluator.

Your task is to analyze user-generated content for policy violations.

Evaluate the content against these violation categories:
- hate_speech: Language targeting groups with hate/slurs
- violence: Threats or glorification of violence
- harassment: Personal attacks, insults, demeaning language
- adult_content: Explicit or sexually suggestive content
- misinformation: False claims, conspiracy theories
- spam: Promotional content, excessive self-promotion
- suspicious_link: Shortened URLs or suspicious links
- engagement_bait: Manipulative engagement requests
- illegal_content: References to illegal activities

Return ONLY valid JSON with this exact structure:
{
    "violations": ["list of violation types found, empty array if none"],
    "score": 0.0-1.0,
    "reason": "brief explanation of your evaluation",
    "confidence": 0.0-1.0
}

Scoring guidelines:
- 1.0: Completely clean, no concerns
- 0.8-0.99: Minor issues but acceptable
- 0.5-0.79: Borderline, some concerns
- 0.2-0.49: Clear violations present
- 0.0-0.19: Severe violations, immediate action needed

Be nuanced - content can be borderline. Consider context and intent."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 10.0,
        max_retries: int = 2,
    ):
        """
        Initialize OpenAI evaluator.

        Args:
            api_key: OpenAI API key (uses env var if None).
            base_url: Custom API base URL (for proxies).
            model: Model name to use.
            timeout: Request timeout in seconds.
            max_retries: Maximum retry attempts.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("API_BASE_URL")
        self.model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.timeout = timeout
        self.max_retries = max_retries

        # Statistics
        self._stats = {
            "total_evaluations": 0,
            "openai_success": 0,
            "fallback_used": 0,
            "timeouts": 0,
            "errors": 0,
        }

        # Initialize client
        self._client = None
        self._init_client()

        logger.info(
            f"OpenAIEvaluator initialized: model={self.model}, "
            f"timeout={timeout}s, retries={max_retries}"
        )

    def _init_client(self) -> None:
        """Initialize OpenAI client."""
        if not self.api_key:
            logger.warning("OPENAI_API_KEY not set, will use fallback")
            return

        try:
            from openai import OpenAI

            kwargs = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url

            self._client = OpenAI(**kwargs)
            logger.info("OpenAI client initialized")

        except ImportError:
            logger.warning("openai package not installed, using fallback")
            self._client = None
        except Exception as e:
            logger.warning(f"OpenAI client init failed: {e}")
            self._client = None

    def evaluate(
        self,
        content: str,
        use_fallback: bool = False,
    ) -> EvaluationResult:
        """
        Evaluate content for policy compliance.

        Args:
            content: Text content to evaluate.
            use_fallback: Force use of rule-based fallback.

        Returns:
            EvaluationResult with violations, score, and reasoning.
        """
        self._stats["total_evaluations"] += 1

        # Force fallback if requested or client unavailable
        if use_fallback or self._client is None:
            return self._evaluate_fallback(content)

        # Try OpenAI with retries
        for attempt in range(self.max_retries + 1):
            try:
                result = self._evaluate_openai(content)
                if result:
                    self._stats["openai_success"] += 1
                    return result
            except Exception as e:
                if attempt < self.max_retries:
                    logger.warning(f"OpenAI evaluation attempt {attempt + 1} failed: {e}")
                    time.sleep(0.5 * (attempt + 1))  # Exponential backoff
                else:
                    self._stats["errors"] += 1
                    logger.error(f"All {self.max_retries + 1} OpenAI attempts failed: {e}")

        # All attempts failed, use fallback
        self._stats["fallback_used"] += 1
        return self._evaluate_fallback(content)

    def _evaluate_openai(self, content: str) -> Optional[EvaluationResult]:
        """Evaluate content using OpenAI API."""
        start_time = time.time()

        try:
            # Build user prompt
            user_prompt = f"""Please evaluate the following content for policy compliance:

---
{content}
---

Return your evaluation as JSON."""

            # Make API call with timeout
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,  # Lower temperature for consistency
                max_tokens=500,
                response_format={"type": "json_object"},
                timeout=self.timeout,
            )

            latency_ms = (time.time() - start_time) * 1000

            # Parse response
            raw_content = response.choices[0].message.content
            parsed = self._parse_response(raw_content, content)

            result = EvaluationResult(
                content=content,
                violations=parsed["violations"],
                score=parsed["score"],
                reason=parsed["reason"],
                is_compliant=parsed["score"] >= 0.5,
                confidence=parsed["confidence"],
                model_used=self.model,
                latency_ms=latency_ms,
                source="openai",
                raw_response=raw_content,
            )

            logger.debug(
                f"OpenAI evaluation: score={result.score:.2f}, "
                f"violations={len(result.violations)}, latency={latency_ms:.0f}ms"
            )

            return result

        except Exception as e:
            logger.warning(f"OpenAI evaluation failed: {e}")
            return None

    def _parse_response(
        self,
        raw_response: str,
        content: str,
    ) -> dict[str, Any]:
        """Parse and validate LLM response."""
        try:
            parsed = json.loads(raw_response)

            # Validate required fields
            violations = parsed.get("violations", [])
            if not isinstance(violations, list):
                violations = []

            score = parsed.get("score", 0.5)
            if not isinstance(score, (int, float)):
                score = 0.5
            score = max(0.0, min(1.0, score))  # Clamp to [0, 1]

            reason = parsed.get("reason", "Evaluation completed")
            confidence = parsed.get("confidence", 0.7)
            if not isinstance(confidence, (int, float)):
                confidence = 0.7
            confidence = max(0.0, min(1.0, confidence))

            return {
                "violations": violations,
                "score": score,
                "reason": reason,
                "confidence": confidence,
            }

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            return self._fallback_parse(raw_response, content)

    def _fallback_parse(
        self,
        raw_response: str,
        content: str,
    ) -> dict[str, Any]:
        """Fallback parsing if JSON is invalid."""
        # Try to extract score from text
        import re

        score_match = re.search(r'"score"\s*:\s*([\d.]+)', raw_response)
        score = float(score_match.group(1)) if score_match else 0.5

        # Try to extract violations
        violations = []
        for vtype in self.VIOLATION_TYPES:
            if vtype.lower() in raw_response.lower():
                violations.append(vtype)

        # Extract reason (first sentence after "reason":)
        reason_match = re.search(r'"reason"\s*:\s*"([^"]+)"', raw_response)
        reason = reason_match.group(1) if reason_match else "Parsed from partial response"

        return {
            "violations": violations,
            "score": max(0.0, min(1.0, score)),
            "reason": reason,
            "confidence": 0.5,
        }

    def _evaluate_fallback(self, content: str) -> EvaluationResult:
        """
        Rule-based fallback evaluation.

        Uses keyword pattern matching when OpenAI is unavailable.
        """
        start_time = time.time()

        content_lower = content.lower()

        # Pattern mappings (same as ViolationDetector for consistency)
        patterns = {
            "hate_speech": ["hate", "slur", "should all die", "kill all"],
            "violence": ["kill", "attack", "hurt", "violence", "weapon"],
            "harassment": ["idiot", "stupid", "pathetic", "worthless", "ignorant", "delusional", "sad."],
            "adult_content": ["explicit", "nsfw", "porn", "dm for pics", "18+"],
            "misinformation": ["fake news", "conspiracy", "hoax", "they don't want you to know", "wake up"],
            "spam": ["click here", "buy now", "!!!", "limited time", "free", "visit ", "dm me"],
            "suspicious_link": ["bit.ly", "tinyurl", "linktr.ee"],
            "engagement_bait": ["tag someone", "share if", "🔥", "💰", "👇"],
            "illegal_content": ["illegal", "drugs"],
        }

        # Detect violations
        violations = []
        for vtype, keywords in patterns.items():
            if any(kw in content_lower for kw in keywords):
                violations.append(vtype)

        # Calculate score based on violations
        if len(violations) == 0:
            score = 0.9  # Clean content
        elif len(violations) == 1:
            score = 0.5  # Single violation - borderline
        elif len(violations) == 2:
            score = 0.3  # Multiple violations
        else:
            score = 0.1  # Severe violations

        # Build reason
        if violations:
            reason = f"Detected violations: {', '.join(violations)}"
        else:
            reason = "No violations detected - content appears compliant"

        latency_ms = (time.time() - start_time) * 1000

        self._stats["fallback_used"] += 1

        return EvaluationResult(
            content=content,
            violations=violations,
            score=score,
            reason=reason,
            is_compliant=score >= 0.5,
            confidence=0.8,  # High confidence in rule-based detection
            model_used="rule_based_fallback",
            latency_ms=latency_ms,
            source="fallback",
        )

    def evaluate_batch(
        self,
        contents: list[str],
        parallel: bool = False,
    ) -> list[EvaluationResult]:
        """
        Evaluate multiple content items.

        Args:
            contents: List of content strings to evaluate.
            parallel: Whether to evaluate in parallel.

        Returns:
            List of EvaluationResult objects.
        """
        if parallel:
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                results = list(executor.map(self.evaluate, contents))
        else:
            results = [self.evaluate(content) for content in contents]

        return results

    @property
    def stats(self) -> dict[str, Any]:
        """Get evaluation statistics."""
        return {
            **self._stats,
            "success_rate": (
                self._stats["openai_success"] / max(1, self._stats["total_evaluations"])
            ),
            "fallback_rate": (
                self._stats["fallback_used"] / max(1, self._stats["total_evaluations"])
            ),
        }


# Convenience function
def evaluate_content(
    content: str,
    use_openai: bool = True,
) -> EvaluationResult:
    """
    Evaluate content using default evaluator.

    Args:
        content: Text content to evaluate.
        use_openai: Whether to use OpenAI (falls back if unavailable).

    Returns:
        EvaluationResult with compliance assessment.
    """
    if use_openai:
        evaluator = OpenAIEvaluator()
    else:
        evaluator = OpenAIEvaluator()
        evaluator._client = None  # Force fallback

    return evaluator.evaluate(content)
