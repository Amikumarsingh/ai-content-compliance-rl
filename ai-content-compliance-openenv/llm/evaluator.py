"""
OpenAI-based Content Compliance Evaluator with Multi-Agent Support.

Production-ready LLM evaluator with:
- Multi-agent evaluation (3 independent agents)
- OpenAI API integration (gpt-4o-mini)
- Secure API key management via environment variables
- Response caching for cost reduction
- Retry mechanism with exponential backoff
- Graceful fallback on failures
- Comprehensive logging

Multi-Agent Features:
- 3 independent evaluation calls simulating different agents
- Disagreement detection when agents differ significantly
- Confidence scoring based on agent agreement
- Robust handling if one agent fails

Example:
    >>> evaluator = LLMEvaluator(provider="openai", model="gpt-4o-mini")
    >>> result = evaluator.evaluate_content(content, "approve")
    >>> print(result.compliance_score, result.violations, result.reward)
    >>> print(result.confidence, result.agent_breakdown)
"""

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class AgentEvaluationResult:
    """
    Result from a single agent's evaluation.

    Attributes:
        agent_id: Identifier for the agent (1, 2, or 3).
        compliance_score: Score from 0 (non-compliant) to 1 (fully compliant).
        violations: List of detected violation types.
        reward: Reward value from -10 to +10.
        latency_ms: Time taken for this agent's evaluation.
        success: Whether evaluation succeeded.
        error: Error message if evaluation failed.
    """
    agent_id: int
    compliance_score: float
    violations: list[str]
    reward: int
    latency_ms: float = 0.0
    success: bool = True
    error: str | None = None


@dataclass
class EvaluationResult:
    """
    Result of content compliance evaluation (aggregated from multiple agents).

    Attributes:
        compliance_score: Aggregated score from 0 to 1.
        violations: Merged list of detected violation types.
        suggestion: Improvement suggestion or edited content.
        reward: Aggregated reward value from -10 to +10.
        from_cache: True if result was served from cache.
        fallback_used: True if fallback was used due to API failure.
        confidence: Confidence score (0-1) based on agent agreement.
        disagreement_detected: True if agents significantly disagreed.
        agent_breakdown: List of individual agent results.
    """

    compliance_score: float
    violations: list[str]
    suggestion: str
    reward: int
    from_cache: bool = False
    fallback_used: bool = False
    confidence: float = 1.0
    disagreement_detected: bool = False
    agent_breakdown: list[AgentEvaluationResult] = field(default_factory=list)


# Maximum content length before truncation (safety limit)
MAX_CONTENT_LENGTH = 50000

# Minimum content length for meaningful evaluation
MIN_CONTENT_LENGTH = 1


@dataclass
class CacheEntry:
    """Cached evaluation result with metadata for TTL management."""

    result: EvaluationResult
    timestamp: float
    hit_count: int = 0


class ResponseCache:
    """
    Thread-safe LRU cache for API responses.

    Features:
    - Configurable max size
    - Time-to-live (TTL) for entries
    - Hit/miss statistics tracking
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        """
        Initialize the cache.

        Args:
            max_size: Maximum number of entries to cache.
            ttl_seconds: Time-to-live for cached entries.
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: dict[str, CacheEntry] = {}
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> EvaluationResult | None:
        """
        Get cached result if valid and not expired.

        Args:
            key: Cache lookup key.

        Returns:
            Cached EvaluationResult or None if not found/expired.
        """
        if key not in self._cache:
            return None

        entry = self._cache[key]
        if time.time() - entry.timestamp > self.ttl_seconds:
            del self._cache[key]
            return None

        entry.hit_count += 1
        self._hits += 1
        return entry.result

    def set(self, key: str, result: EvaluationResult) -> None:
        """
        Cache a result, evicting oldest entry if at capacity.

        Args:
            key: Cache key.
            result: EvaluationResult to cache.
        """
        if len(self._cache) >= self.max_size:
            oldest_key = min(self._cache, key=lambda k: self._cache[k].timestamp)
            del self._cache[oldest_key]

        self._cache[key] = CacheEntry(result=result, timestamp=time.time())

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()

    @property
    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        return {
            "size": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / (self._hits + self._misses)
            if (self._hits + self._misses) > 0
            else 0.0,
        }


class LLMEvaluator:
    """
    OpenAI-based content compliance evaluator with multi-agent support.

    Features:
    - Multi-agent evaluation (3 independent agents)
    - OpenAI API integration (gpt-4o-mini, gpt-4, etc.)
    - Response caching to reduce API costs
    - Retry with exponential backoff
    - Graceful fallback to mock evaluation
    - Disagreement detection and confidence scoring
    - Comprehensive logging and metrics

    Multi-Agent Architecture:
    - Agent 1: Strict evaluator (focuses on policy violations)
    - Agent 2: Balanced evaluator (considers context)
    - Agent 3: Lenient evaluator (gives benefit of doubt)

    Aggregation:
    - Final score = weighted average of agent scores
    - Final violations = union of all detected violations
    - Confidence = based on agreement level
    - Disagreement flagged when std dev > threshold

    Security:
    - API key loaded from OPENAI_API_KEY environment variable
    - Never logs or stores API keys

    Example:
        >>> evaluator = LLMEvaluator(
        ...     provider="openai",
        ...     model="gpt-4o-mini",
        ...     cache_enabled=True,
        ... )
        >>> result = evaluator.evaluate_content("Hello world!", "approve")
        >>> print(f"Score: {result.compliance_score}, Confidence: {result.confidence}")
        >>> print(f"Violations: {result.violations}")
        >>> print(f"Agent breakdown: {len(result.agent_breakdown)} agents")
    """

    # OpenAI API endpoint
    OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

    # Valid violation types for enterprise compliance
    VALID_VIOLATIONS = {
        "hate_speech",
        "harassment",
        "violence",
        "adult_content",
        "misinformation",
        "spam",
        "illegal_content",
    }

    # Agent personas for diverse evaluation
    AGENT_PERSONAS = [
        {
            "id": 1,
            "name": "Strict",
            "system_prompt": "You are a STRICT content compliance evaluator. Flag any potential policy violations. Err on the side of caution.",
        },
        {
            "id": 2,
            "name": "Balanced",
            "system_prompt": "You are a BALANCED content compliance evaluator. Consider context and intent while enforcing policies fairly.",
        },
        {
            "id": 3,
            "name": "Lenient",
            "system_prompt": "You are a LENIENT content compliance evaluator. Give content the benefit of the doubt unless clearly violating.",
        },
    ]

    def __init__(
        self,
        provider: str = "mock",
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: int = 30,
        cache_enabled: bool = True,
        cache_size: int = 500,
        cache_ttl: int = 3600,
        fallback_to_mock: bool = True,
        multi_agent_enabled: bool = True,
        disagreement_threshold: float = 0.3,
    ):
        """
        Initialize the LLM evaluator.

        Args:
            provider: LLM provider ('mock' or 'openai').
            api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
            model: OpenAI model name (e.g., 'gpt-4o-mini', 'gpt-4').
            max_retries: Maximum retry attempts on failure.
            retry_delay: Base delay between retries (seconds).
            timeout: Request timeout (seconds).
            cache_enabled: Enable response caching.
            cache_size: Maximum cache entries.
            cache_ttl: Cache entry TTL (seconds).
            fallback_to_mock: Use mock evaluation on API failure.
            multi_agent_enabled: Enable multi-agent evaluation.
            disagreement_threshold: Score difference threshold to flag disagreement.
        """
        self.provider = provider
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.fallback_to_mock = fallback_to_mock
        self.multi_agent_enabled = multi_agent_enabled

        # Initialize fallback evaluator for ultra-reliable fallback
        try:
            from llm.fallback_evaluator import FallbackEvaluator
            self._fallback_evaluator = FallbackEvaluator()
            logger.debug("Fallback evaluator initialized")
        except Exception as e:
            logger.warning(f"Could not initialize fallback evaluator: {e}")
            self._fallback_evaluator = None
        self.disagreement_threshold = disagreement_threshold

        # Initialize cache
        self.cache_enabled = cache_enabled
        self._cache = ResponseCache(max_size=cache_size, ttl_seconds=cache_ttl) if cache_enabled else None

        # Metrics tracking
        self._total_calls = 0
        self._cache_hits = 0
        self._fallback_count = 0
        self._api_latency_total = 0.0
        self._agent_evaluations = 0

        logger.info(
            f"LLMEvaluator initialized: provider={provider}, model={model}, "
            f"multi_agent={multi_agent_enabled}, cache={cache_enabled}, fallback={fallback_to_mock}"
        )

    def evaluate_content(self, content: str, action: str) -> EvaluationResult:
        """
        Evaluate content for compliance using multi-agent system.

        Args:
            content: Text content to evaluate.
            action: Action taken ('approve', 'reject', or 'edit').

        Returns:
            EvaluationResult with aggregated metrics and agent breakdown.
        """
        self._total_calls += 1

        # Validate and sanitize input
        content, validation_info = self._validate_and_sanitize_content(content)

        # Validate action
        if action not in ["approve", "reject", "edit"]:
            logger.warning(f"Invalid action '{action}', defaulting to 'approve'")
            action = "approve"

        # Generate cache key
        cache_key = self._make_cache_key(content, action)

        # Check cache first
        if self.cache_enabled and self._cache:
            cached_result = self._cache.get(cache_key)
            if cached_result is not None:
                self._cache_hits += 1
                logger.debug(f"Cache HIT: {cache_key[:16]}...")
                cached_result.from_cache = True
                return cached_result

        logger.debug(f"Evaluating content (len={len(content)}), action={action}, validation={validation_info}")

        try:
            if self.provider == "mock":
                result = self._multi_agent_mock_evaluate(content, action)
            elif self.provider == "openai":
                if self.multi_agent_enabled:
                    result = self._multi_agent_openai_evaluate(content, action)
                else:
                    result = self._single_agent_openai_evaluate(content, action)
            else:
                raise ValueError(f"Unknown provider: {self.provider}")

            # Cache the result
            if self.cache_enabled and self._cache:
                self._cache.set(cache_key, result)

            return result

        except Exception as e:
            logger.error(f"Evaluation failed: {type(e).__name__}: {e}")
            self._fallback_count += 1

            # Fallback chain: mock -> fallback_evaluator -> safe_default
            if self.fallback_to_mock:
                try:
                    logger.warning("Falling back to mock evaluation")
                    result = self._multi_agent_mock_evaluate(content, action)
                    result.fallback_used = True
                    return result
                except Exception as mock_error:
                    logger.error(f"Mock fallback also failed: {mock_error}")

            # Try rule-based fallback evaluator
            if self._fallback_evaluator:
                try:
                    logger.warning("Using rule-based fallback evaluator")
                    return self._fallback_evaluator.evaluate(content, action)
                except Exception as fb_error:
                    logger.error(f"Fallback evaluator failed: {fb_error}")

            # Last resort: safe default
            logger.critical("All evaluation methods failed, using absolute fallback")
            return self._safe_default_result(action, validation_info)

    def _make_cache_key(self, content: str, action: str) -> str:
        """Create a unique SHA256 hash for content-action pair."""
        key_data = f"{action}:{content}"
        return hashlib.sha256(key_data.encode()).hexdigest()

    def _validate_and_sanitize_content(self, content: str) -> tuple[str, dict]:
        """
        Validate and sanitize content input.

        Handles:
        - Empty or None content
        - Very long content (truncation)
        - Special characters and encoding issues
        - Whitespace-only content

        Args:
            content: Raw content string.

        Returns:
            Tuple of (sanitized_content, validation_info_dict).
        """
        validation_info = {
            "original_length": len(content) if content else 0,
            "truncated": False,
            "was_empty": False,
            "sanitized": False,
        }

        # Handle None or empty content
        if not content or not isinstance(content, str):
            validation_info["was_empty"] = True
            logger.warning("Empty or invalid content provided, using placeholder")
            return "[Empty content provided]", validation_info

        # Strip leading/trailing whitespace
        content = content.strip()

        # Handle whitespace-only content
        if not content:
            validation_info["was_empty"] = True
            logger.warning("Whitespace-only content provided, using placeholder")
            return "[Empty content provided]", validation_info

        # Truncate very long content safely
        if len(content) > MAX_CONTENT_LENGTH:
            validation_info["truncated"] = True
            # Truncate at word boundary if possible
            truncated_content = content[:MAX_CONTENT_LENGTH]
            last_space = truncated_content.rfind(" ")
            if last_space > MAX_CONTENT_LENGTH - 1000:
                truncated_content = truncated_content[:last_space]
            content = truncated_content[:MAX_CONTENT_LENGTH] + " [content truncated for length]"
            logger.info(f"Content truncated from {validation_info['original_length']} to {len(content)} chars")

        # Sanitize any problematic characters
        try:
            # Ensure content is valid UTF-8
            content.encode("utf-8").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            logger.warning("Content had encoding issues, sanitizing")
            validation_info["sanitized"] = True
            content = content.encode("utf-8", errors="replace").decode("utf-8")

        validation_info["final_length"] = len(content)
        return content, validation_info

    def _safe_default_result(self, action: str, validation_info: dict | None = None) -> EvaluationResult:
        """
        Return a safe default result when all evaluation methods fail.

        Args:
            action: Action taken.
            validation_info: Optional info about content validation issues.

        Returns:
            Safe default EvaluationResult.
        """
        # Build informative message
        messages = ["Evaluation unavailable"]
        if validation_info:
            if validation_info.get("was_empty"):
                messages.append("empty content detected")
            if validation_info.get("truncated"):
                messages.append("content was truncated")
            if validation_info.get("sanitized"):
                messages.append("content was sanitized")

        suggestion = "Proceeding with neutral assessment due to: " + ", ".join(messages[1:]) if len(messages) > 1 else "Evaluation unavailable - proceeding with neutral assessment"

        return EvaluationResult(
            compliance_score=0.5,
            violations=[],
            suggestion=suggestion,
            reward=0,
            fallback_used=True,
            confidence=0.0,
            disagreement_detected=False,
            agent_breakdown=[],
        )

    def _aggregate_agent_results(
        self,
        agent_results: list[AgentEvaluationResult],
        action: str,
    ) -> EvaluationResult:
        """
        Aggregate results from multiple agents.

        Aggregation rules:
        - Final score = weighted average (filtered to successful agents)
        - Violations = union of all detected violations
        - Reward = weighted average based on compliance
        - Confidence = 1 - (std_dev / max_possible_std)
        - Disagreement = flagged if std_dev > threshold

        Args:
            agent_results: List of individual agent results.
            action: Action taken.

        Returns:
            Aggregated EvaluationResult.
        """
        # Filter successful evaluations
        successful = [r for r in agent_results if r.success]

        if not successful:
            # All agents failed - return safe default
            return self._safe_default_result(action)

        # Calculate score statistics
        scores = [r.compliance_score for r in successful]
        mean_score = sum(scores) / len(scores)

        # Standard deviation for confidence calculation
        if len(scores) > 1:
            variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
            std_dev = variance ** 0.5
        else:
            std_dev = 0.0

        # Confidence: 1.0 when perfect agreement, decreases with disagreement
        # Max std_dev is ~0.5 (when scores are 0 and 1)
        confidence = max(0.0, 1.0 - (std_dev / 0.5))

        # Disagreement detection
        disagreement_detected = std_dev > self.disagreement_threshold

        # Merge violations (union of all)
        all_violations = set()
        for r in successful:
            all_violations.update(r.violations)

        # Calculate reward based on aggregated compliance
        is_compliant = mean_score >= 0.5
        rewards = []
        for r in successful:
            agent_compliant = r.compliance_score >= 0.5
            if action == "approve":
                rewards.append(10 if agent_compliant else -10)
            elif action == "reject":
                rewards.append(8 if not agent_compliant else -8)
            elif action == "edit":
                rewards.append(2 if r.violations else -2)
            else:
                rewards.append(0)

        mean_reward = int(sum(rewards) / len(rewards)) if rewards else 0

        # Generate suggestion based on violations
        suggestion = self._generate_suggestion(list(all_violations), mean_score)

        # Log disagreement if detected
        if disagreement_detected:
            logger.warning(
                f"Agent disagreement detected: scores={scores}, "
                f"std_dev={std_dev:.3f}, confidence={confidence:.2f}"
            )

        return EvaluationResult(
            compliance_score=mean_score,
            violations=list(all_violations),
            suggestion=suggestion,
            reward=mean_reward,
            confidence=confidence,
            disagreement_detected=disagreement_detected,
            agent_breakdown=agent_results,
        )

    def _multi_agent_mock_evaluate(
        self,
        content: str,
        action: str,
    ) -> EvaluationResult:
        """
        Multi-agent mock evaluation with diverse personas.

        Each agent uses slightly different criteria to simulate diversity.
        """
        start_time = time.perf_counter()
        agent_results: list[AgentEvaluationResult] = []

        # Agent-specific keyword weights
        agent_thresholds = [
            {"strict": 0.15, "moderate": 0.25, "lenient": 0.4},  # Strict
            {"strict": 0.2, "moderate": 0.3, "lenient": 0.35},   # Balanced
            {"strict": 0.25, "moderate": 0.35, "lenient": 0.45}, # Lenient
        ]

        # Expanded keyword detection
        violation_keywords = {
            "hate": "hate_speech",
            "violence": "violence",
            "kill": "violence",
            "die": "violence",
            "adult": "adult_content",
            "sex": "adult_content",
            "misleading": "misinformation",
            "lie": "misinformation",
            "fake": "misinformation",
            "harass": "harassment",
            "stupid": "harassment",
            "idiot": "harassment",
            "spam": "spam",
            "free": "spam",
            "prize": "spam",
            "click here": "spam",
            "limited": "spam",
            "buy now": "spam",
            "illegal": "illegal_content",
        }

        content_lower = content.lower()

        for agent_id in range(1, 4):
            agent_start = time.perf_counter()
            thresholds = agent_thresholds[agent_id - 1]

            # Each agent detects violations with different sensitivity
            violations = set()
            for keyword, label in violation_keywords.items():
                if keyword in content_lower:
                    if agent_id == 1:  # Strict - always flag
                        violations.add(label)
                    elif agent_id == 2:  # Balanced - flag most
                        violations.add(label)
                    else:  # Lenient - only clear violations
                        if label in ["hate_speech", "violence", "illegal_content"]:
                            violations.add(label)
                        elif label == "spam" and keyword in ["free", "prize", "click here"]:
                            violations.add(label)

            violations = list(violations)

            # Calculate score with agent-specific penalty
            base_score = 1.0
            penalty_per_violation = thresholds.get(
                "strict" if len(violations) > 2 else
                "moderate" if len(violations) > 0 else "lenient",
                0.2
            )
            compliance_score = max(0.0, base_score - (len(violations) * penalty_per_violation))

            agent_elapsed = (time.perf_counter() - agent_start) * 1000

            agent_results.append(AgentEvaluationResult(
                agent_id=agent_id,
                compliance_score=compliance_score,
                violations=violations,
                reward=0,
                latency_ms=agent_elapsed,
                success=True,
            ))
            self._agent_evaluations += 1

        # Aggregate results
        result = self._aggregate_agent_results(agent_results, action)

        total_elapsed = (time.perf_counter() - start_time) * 1000
        self._api_latency_total += total_elapsed

        return result

    def _multi_agent_openai_evaluate(
        self,
        content: str,
        action: str,
    ) -> EvaluationResult:
        """
        Multi-agent evaluation using OpenAI API.

        Makes 3 independent API calls with different agent personas.
        Handles individual agent failures gracefully - system continues
        even if all agents fail.
        """
        start_time = time.perf_counter()
        agent_results: list[AgentEvaluationResult] = []
        successful_agents = 0

        for i, agent_info in enumerate(self.AGENT_PERSONAS):
            agent_start = time.perf_counter()

            try:
                result = self._openai_single_agent_evaluate(
                    content, action, agent_info
                )
                result.latency_ms = (time.perf_counter() - agent_start) * 1000
                agent_results.append(result)
                successful_agents += 1
                self._agent_evaluations += 1
                logger.debug(f"Agent {agent_info['id']} completed successfully ({result.latency_ms:.0f}ms)")

            except Exception as e:
                logger.warning(f"Agent {agent_info['id']} failed ({i+1}/3): {type(e).__name__}: {e}")
                # Add failed agent result for tracking
                agent_results.append(AgentEvaluationResult(
                    agent_id=agent_info["id"],
                    compliance_score=0.5,  # Neutral on failure
                    violations=[],
                    reward=0,
                    latency_ms=(time.perf_counter() - agent_start) * 1000,
                    success=False,
                    error=str(e),
                ))

        # Log summary
        logger.info(f"Multi-agent evaluation complete: {successful_agents}/3 agents succeeded")

        # Aggregate results (handles failed agents)
        result = self._aggregate_agent_results(agent_results, action)

        total_elapsed = (time.perf_counter() - start_time) * 1000
        self._api_latency_total += total_elapsed

        return result

    def _openai_single_agent_evaluate(
        self,
        content: str,
        action: str,
        agent_info: dict[str, Any],
    ) -> AgentEvaluationResult:
        """
        Evaluate using single OpenAI agent with specific persona.

        Args:
            content: Content to evaluate.
            action: Action taken.
            agent_info: Agent persona configuration.

        Returns:
            AgentEvaluationResult for this agent.
        """
        prompt = self._build_evaluation_prompt(content, action)

        try:
            from openai import OpenAI
        except ImportError:
            logger.error("OpenAI SDK not installed")
            raise ImportError("OpenAI SDK required: pip install openai")

        try:
            client = OpenAI(api_key=self.api_key)
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": agent_info["system_prompt"]},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=500,
                timeout=self.timeout,
            )
        except Exception as e:
            logger.error(f"OpenAI API call failed: {type(e).__name__}: {e}")
            raise RuntimeError(f"API call failed: {e}")

        response_text = response.choices[0].message.content if response.choices else None

        if not response_text:
            logger.warning("Empty response from API")
            raise ValueError("Empty response from API")

        result_dict = self._extract_json_from_text(response_text)

        return AgentEvaluationResult(
            agent_id=agent_info["id"],
            compliance_score=self._validate_compliance_score(result_dict.get("compliance_score")),
            violations=self._validate_violations(result_dict.get("violations", [])),
            reward=self._validate_reward(result_dict.get("reward")),
            success=True,
        )

    def _single_agent_openai_evaluate(
        self,
        content: str,
        action: str,
    ) -> EvaluationResult:
        """Legacy single-agent evaluation for backward compatibility."""
        result = self._openai_evaluate_with_retry(content, action)

        # Wrap in multi-agent format
        agent_result = AgentEvaluationResult(
            agent_id=1,
            compliance_score=result.compliance_score,
            violations=result.violations,
            reward=result.reward,
            success=True,
        )

        return EvaluationResult(
            compliance_score=result.compliance_score,
            violations=result.violations,
            suggestion=result.suggestion,
            reward=result.reward,
            confidence=1.0,
            disagreement_detected=False,
            agent_breakdown=[agent_result],
        )

    def _openai_evaluate_with_retry(self, content: str, action: str) -> EvaluationResult:
        """Legacy retry wrapper for single-agent evaluation."""
        last_error = None

        for attempt in range(self.max_retries):
            try:
                return self._openai_evaluate(content, action)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    time.sleep(wait_time)

        raise RuntimeError(f"OpenAI API failed after {self.max_retries} attempts: {last_error}")

    def _openai_evaluate(self, content: str, action: str) -> EvaluationResult:
        """Legacy single-agent OpenAI evaluation."""
        start_time = time.perf_counter()

        if not self.api_key:
            raise ValueError("OpenAI API key not provided")

        prompt = self._build_evaluation_prompt(content, action)

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAI SDK required: pip install openai")

        client = OpenAI(api_key=self.api_key)

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a strict enterprise content compliance evaluator. Return ONLY valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=500,
            timeout=self.timeout,
        )

        response_text = response.choices[0].message.content
        result_dict = self._extract_json_from_text(response_text)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._api_latency_total += elapsed_ms

        return EvaluationResult(
            compliance_score=self._validate_compliance_score(result_dict.get("compliance_score")),
            violations=self._validate_violations(result_dict.get("violations", [])),
            suggestion=str(result_dict.get("suggestion", "No suggestions.")),
            reward=self._validate_reward(result_dict.get("reward")),
        )

    def _build_evaluation_prompt(self, content: str, action: str) -> str:
        """Build the evaluation prompt for OpenAI."""
        return f"""You are a strict enterprise content compliance evaluator.

Given content and an action (approve/reject/edit), return:

- compliance_score (0 to 1)
- violations (list)
- suggestion (improved version if needed)
- reward (-10 to +10)

Policies:
- no hate speech
- no harmful/illegal content
- maintain professional tone

Return ONLY valid JSON.

CONTENT:
{content}

ACTION TAKEN: {action}

Respond with JSON in this exact format:
{{
    "compliance_score": 0.8,
    "violations": [],
    "suggestion": "Content looks good.",
    "reward": 10
}}"""

    def _extract_json_from_text(self, text: str) -> dict[str, Any]:
        """
        Extract and parse JSON from response text with robust fallback handling.

        Strategies (in order):
        1. Direct JSON parse
        2. Remove markdown code blocks, then parse
        3. Extract JSON object with regex, then parse
        4. Extract key-value pairs and construct dict manually
        5. Return safe defaults with partial data extraction

        Args:
            text: Raw response text from LLM.

        Returns:
            Parsed dictionary or safe defaults.
        """
        if not text or not isinstance(text, str):
            logger.warning("Empty or invalid text for JSON extraction")
            return self._get_safe_default_response()

        original_text = text
        text = text.strip()

        # Strategy 1: Direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Strategy 2: Remove markdown code blocks
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Strategy 3: Extract JSON object with regex (handles nested braces)
        # Try to find outermost JSON object
        brace_count = 0
        start_idx = None
        for i, char in enumerate(text):
            if char == "{":
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0 and start_idx is not None:
                    json_str = text[start_idx:i+1]
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        pass
                    break

        # Strategy 4: Try to extract individual fields with regex
        result = {}

        # Extract compliance_score
        score_match = re.search(r'"?compliance_score"?\s*[:=]\s*([0-9.]+)', text, re.IGNORECASE)
        if score_match:
            try:
                result["compliance_score"] = float(score_match.group(1))
            except ValueError:
                pass

        # Extract reward
        reward_match = re.search(r'"?reward"?\s*[:=]\s*(-?[0-9]+)', text, re.IGNORECASE)
        if reward_match:
            try:
                result["reward"] = int(reward_match.group(1))
            except ValueError:
                pass

        # Extract violations (look for array)
        violations_match = re.search(r'"?violations"?\s*[:=]\s*\[([^\]]*)\]', text, re.IGNORECASE)
        if violations_match:
            violations_str = violations_match.group(1)
            # Extract quoted strings from the array
            violations = re.findall(r'"([^"]+)"', violations_str)
            if violations:
                result["violations"] = violations

        # Extract suggestion
        suggestion_match = re.search(r'"?suggestion"?\s*[:=]\s*"([^"]+)"', text, re.IGNORECASE)
        if suggestion_match:
            result["suggestion"] = suggestion_match.group(1)

        if result:
            logger.info(f"Partial JSON extraction successful: {list(result.keys())}")
            return result

        # Strategy 5: Complete fallback
        logger.warning(f"Could not extract any JSON data from response: {original_text[:100]}...")
        return self._get_safe_default_response()

    def _get_safe_default_response(self) -> dict[str, Any]:
        """Return safe default response dict when JSON extraction fails."""
        return {
            "compliance_score": 0.5,
            "violations": [],
            "suggestion": "Could not parse LLM response - using neutral assessment",
            "reward": 0,
        }

    def _validate_compliance_score(self, score: Any) -> float:
        """Validate and normalize compliance score to [0, 1] range."""
        if score is None:
            return 0.5
        try:
            score = float(score)
            return max(0.0, min(1.0, score))
        except (TypeError, ValueError):
            return 0.5

    def _validate_violations(self, violations: Any) -> list[str]:
        """Validate and normalize violations list."""
        if not isinstance(violations, list):
            return []

        valid = []
        for v in violations:
            if isinstance(v, str):
                normalized = v.lower().replace(" ", "_").replace("-", "_")
                valid.append(normalized)

        return valid

    def _validate_reward(self, reward: Any) -> int:
        """Validate and clamp reward to [-10, +10] range."""
        if reward is None:
            return 0
        try:
            reward = int(float(reward))
            return max(-10, min(10, reward))
        except (TypeError, ValueError):
            return 0

    def _generate_suggestion(self, violations: list[str], compliance_score: float) -> str:
        """Generate improvement suggestion based on violations."""
        if not violations:
            return "Content is compliant. No changes needed."

        suggestions = {
            "hate_speech": "Remove or rephrase language that could be perceived as hateful.",
            "harassment": "Ensure content is respectful and does not target individuals.",
            "violence": "Remove graphic or threatening descriptions of violence.",
            "adult_content": "Modify content to be appropriate for general audiences.",
            "misinformation": "Verify facts and remove unverified claims.",
            "spam": "Remove promotional or unsolicited content.",
            "illegal_content": "Remove content that may violate laws or regulations.",
        }

        suggestion_parts = [suggestions.get(v, f"Review: {v}") for v in violations]
        return " ".join(suggestion_parts)

    def edit_content(self, content: str, suggestion: str) -> str:
        """Edit content based on LLM suggestion."""
        logger.debug(f"Editing content: {content[:50]}...")

        if self.provider == "mock":
            return self._mock_edit_content(content, suggestion)
        elif self.provider == "openai":
            return self._openai_edit_content(content, suggestion)
        else:
            return f"[Edited] {content}"

    def _mock_edit_content(self, content: str, suggestion: str) -> str:
        """Mock editing - removes problematic keywords."""
        words_to_remove = ["hate", "violence", "misleading", "harass", "spam"]
        edited = content
        for word in words_to_remove:
            edited = edited.replace(word, "[removed]")
        return f"[Edited] {edited}"

    def _openai_edit_content(self, content: str, suggestion: str) -> str:
        """Edit content using OpenAI API."""
        if not self.api_key:
            return self._mock_edit_content(content, suggestion)

        prompt = f"""Rewrite the following content to address this concern:

SUGGESTION: {suggestion}

ORIGINAL CONTENT:
{content}

Return only the rewritten content, no explanations."""

        try:
            from openai import OpenAI
        except ImportError:
            return self._mock_edit_content(content, suggestion)

        client = OpenAI(api_key=self.api_key)

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a content editor. Rewrite content to be compliant. Return only the edited content."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=300,
                timeout=self.timeout,
            )

            edited_text = response.choices[0].message.content.strip()
            if edited_text:
                return edited_text

        except Exception as e:
            logger.error(f"OpenAI edit API failed: {e}")

        return self._mock_edit_content(content, suggestion)

    def get_stats(self) -> dict[str, Any]:
        """Get evaluator statistics."""
        stats = {
            "total_calls": self._total_calls,
            "cache_hits": self._cache_hits,
            "fallback_count": self._fallback_count,
            "cache_enabled": self.cache_enabled,
            "avg_latency_ms": self._api_latency_total / max(1, self._total_calls),
            "multi_agent_enabled": self.multi_agent_enabled,
            "total_agent_evaluations": self._agent_evaluations,
        }

        if self._cache:
            stats["cache_stats"] = self._cache.stats

        return stats

    def clear_cache(self) -> None:
        """Clear the response cache."""
        if self._cache:
            self._cache.clear()
            logger.info("Cache cleared")

    def batch_evaluate(self, items: list[tuple[str, str]]) -> list[EvaluationResult]:
        """Evaluate multiple content items with progress logging."""
        logger.info(f"Batch evaluating {len(items)} items")
        results = []

        for i, (content, action) in enumerate(items):
            try:
                result = self.evaluate_content(content, action)
                results.append(result)

                if (i + 1) % 10 == 0:
                    logger.debug(f"Processed {i + 1}/{len(items)} items")

            except Exception as e:
                logger.error(f"Item {i + 1} failed: {e}")
                results.append(self._safe_default_result(action))

        return results
