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
- Debug mode for troubleshooting
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

# Debug mode flag
DEBUG = os.getenv("DEBUG", "false").lower() == "true"


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

    @staticmethod
    def _is_valid_api_key(key: str) -> bool:
        """
        Validate API key format.

        Accepts both legacy (sk-) and new (sk-proj-) key formats.
        """
        if not key:
            return False
        # Strip quotes if present (from .env parsing)
        key = key.strip('"').strip("'")
        return key.startswith("sk-") or key.startswith("sk-proj-")

    def _init_client(self) -> None:
        """Initialize OpenAI client with proper key validation."""
        # Clean API key (remove quotes if present)
        if self.api_key:
            self.api_key = self.api_key.strip('"').strip("'")

        if not self.api_key:
            logger.warning("OPENAI_API_KEY not set, will use fallback")
            return

        if not self._is_valid_api_key(self.api_key):
            logger.warning(f"Invalid API key format (first 10 chars: {self.api_key[:10]}...), using fallback")
            return

        # Log masked API key for debugging
        masked_key = f"{self.api_key[:8]}...{self.api_key[-4:]}" if len(self.api_key) > 12 else "****"
        logger.info(f"API key loaded: {masked_key}")

        try:
            from openai import OpenAI

            kwargs = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url

            self._client = OpenAI(**kwargs)
            logger.info(f"OpenAI client initialized (model={self.model})")

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

        # Check API key validity before attempting
        if not self._is_valid_api_key(self.api_key):
            if self._stats["total_evaluations"] == 1:
                logger.warning("Invalid API key format, using fallback for all evaluations")
            return self._evaluate_fallback(content)

        # Force fallback if requested or client unavailable
        if use_fallback or self._client is None:
            return self._evaluate_fallback(content)

        # Try OpenAI with retries
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                result = self._evaluate_openai(content)
                if result:
                    self._stats["openai_success"] += 1
                    logger.info(f"Evaluator source: {result.source}, score: {result.score:.2f}")
                    return result
                last_error = "No result returned"
            except Exception as e:
                last_error = str(e)
                if attempt < self.max_retries:
                    wait_time = 0.5 * (attempt + 1)
                    logger.warning(f"OpenAI evaluation attempt {attempt + 1}/{self.max_retries + 1} failed, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    self._stats["errors"] += 1
                    logger.error(f"All {self.max_retries + 1} OpenAI attempts failed: {last_error}")

        # All attempts failed, use fallback
        self._stats["fallback_used"] += 1
        logger.info("Evaluator source: fallback (OpenAI unavailable)")
        return self._evaluate_fallback(content)

    def _evaluate_openai(self, content: str) -> Optional[EvaluationResult]:
        """Evaluate content using OpenAI API with modern responses endpoint."""
        start_time = time.time()

        try:
            # Build user prompt
            user_prompt = f"""Please evaluate the following content for policy compliance:

---
{content}
---

Return ONLY valid JSON with this exact structure:
{{
    "violations": ["list of violation types found, empty array if none"],
    "score": 0.0-1.0,
    "reason": "brief explanation of your evaluation",
    "confidence": 0.0-1.0
}}"""

            # Make API call with timeout using modern responses API
            try:
                # Try new responses API first (newer OpenAI SDK)
                response = self._client.responses.create(
                    model=self.model,
                    input=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.3,
                    max_output_tokens=500,
                    timeout=self.timeout,
                )
                # New API response structure
                raw_content = response.output[0].content[0].text if hasattr(response, 'output') else None
            except AttributeError:
                # Fallback to chat.completions API (older but still supported)
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.3,
                    max_tokens=500,
                    response_format={"type": "json_object"},
                    timeout=self.timeout,
                )
                # Legacy API response structure
                raw_content = response.choices[0].message.content if hasattr(response, 'choices') else None

            if raw_content is None:
                logger.warning("OpenAI response returned no content")
                return None

            latency_ms = (time.time() - start_time) * 1000

            # Debug mode: print raw response
            if DEBUG:
                logger.info(f"[DEBUG] Raw OpenAI response: {raw_content[:500]}...")

            # Parse response
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

            logger.info(
                f"OpenAI evaluation: score={result.score:.2f}, "
                f"violations={len(result.violations)}, latency={latency_ms:.0f}ms"
            )

            return result

        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)

            # Check for authentication errors
            if "401" in error_msg or "invalid_api_key" in error_msg.lower():
                logger.error("OpenAI API authentication failed (401 Unauthorized)")
            elif "429" in error_msg:
                logger.warning("OpenAI API rate limit exceeded (429)")
            else:
                logger.warning(f"OpenAI evaluation failed ({error_type}): {error_msg}")

            return None

    def _parse_response(
        self,
        raw_response: str,
        content: str,
    ) -> dict[str, Any]:
        """Parse and validate LLM response with robust error handling."""
        try:
            parsed = json.loads(raw_response)

            # Validate required fields with safe defaults
            violations = parsed.get("violations", [])
            if not isinstance(violations, list):
                violations = []

            score = parsed.get("score", 0.5)
            if not isinstance(score, (int, float)):
                try:
                    score = float(score)
                except (TypeError, ValueError):
                    score = 0.5
            score = max(0.0, min(1.0, score))  # Clamp to [0, 1]

            reason = parsed.get("reason", "Evaluation completed")
            if not isinstance(reason, str):
                reason = str(reason) if reason else "Evaluation completed"

            confidence = parsed.get("confidence", 0.7)
            if not isinstance(confidence, (int, float)):
                try:
                    confidence = float(confidence)
                except (TypeError, ValueError):
                    confidence = 0.7
            confidence = max(0.0, min(1.0, confidence))

            result = {
                "violations": violations,
                "score": score,
                "reason": reason,
                "confidence": confidence,
            }

            if DEBUG:
                logger.info(f"[DEBUG] Parsed response: {result}")

            return result

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            return self._fallback_parse(raw_response, content)
        except Exception as e:
            logger.warning(f"Unexpected error parsing response: {e}")
            return self._fallback_parse(raw_response, content)

    def _fallback_parse(
        self,
        raw_response: str,
        content: str,
    ) -> dict[str, Any]:
        """Fallback parsing if JSON is invalid - uses regex extraction."""
        # Try to extract score from text
        score_match = re.search(r'"score"\s*:\s*([\d.]+)', raw_response)
        score = float(score_match.group(1)) if score_match else 0.5

        # Try to extract violations
        violations = []
        for vtype in self.VIOLATION_TYPES:
            if vtype.lower() in raw_response.lower() or vtype.replace("_", " ").lower() in raw_response.lower():
                violations.append(vtype)

        # Extract reason (handle various formats)
        reason_match = re.search(r'"reason"\s*:\s*"([^"]+)"', raw_response)
        if not reason_match:
            reason_match = re.search(r'"reason"\s*:\s*([^\n,}]+)', raw_response)
        reason = reason_match.group(1) if reason_match else "Parsed from partial response"

        # Extract confidence if present
        confidence_match = re.search(r'"confidence"\s*:\s*([\d.]+)', raw_response)
        confidence = float(confidence_match.group(1)) if confidence_match else 0.5

        return {
            "violations": violations,
            "score": max(0.0, min(1.0, score)),
            "reason": reason,
            "confidence": max(0.0, min(1.0, confidence)),
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
            "hate_speech":    ["should all die", "kill all", "i hate those", "they should die"],
            "violence":       ["kill them", "attack them", "hurt people", "weapon of"],
            "harassment":     ["you are an idiot", "you are stupid", "you are pathetic",
                               "you are worthless", "too ignorant to", "so delusional"],
            "adult_content":  ["explicit content", "nsfw", "porn", "dm for pics"],
            "misinformation": ["fake news", "they don't want you to know", "wake up sheeple",
                               "mainstream media won't", "scientists are hiding"],
            "spam":           ["click here", "buy now", "limited time offer", "free prizes",
                               "free tips inside", "act now", "win $"],
            "suspicious_link":["bit.ly/", "tinyurl.com/", "linktr.ee/"],
            "engagement_bait":["tag someone who", "share if you", "comment below and i"],
            "illegal_content":["buy illegal", "sell drugs", "illegal weapons"],
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
    evaluator = OpenAIEvaluator()

    if not use_openai:
        evaluator._client = None  # Force fallback

    return evaluator.evaluate(content)


# Module-level stats for monitoring
_evaluator_stats: Optional[OpenAIEvaluator] = None


def get_evaluator_stats() -> dict[str, Any]:
    """Get current evaluator statistics."""
    global _evaluator_stats
    if _evaluator_stats is None:
        _evaluator_stats = OpenAIEvaluator()
    return _evaluator_stats.stats
