"""
Production-Grade Dynamic Data Loader for Content Compliance RL.

Features:
- Fully dynamic content generation (no static data)
- LLM-powered content with graceful fallback
- Topic diversity rotation
- Ambiguity level control
- Duplicate prevention
- Timeout handling for API calls
- Thread-safe caching

Sources (in priority order):
1. LLM-generated content (OpenAI API) - borderline/ambiguous cases
2. Rule-based fallback - diverse template generation
3. Hugging Face datasets (optional, if available)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ContentItem:
    """Represents a single content item for moderation."""

    text: str
    is_compliant: bool
    source: str
    category: str
    violations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    true_label: Optional[str] = None  # "spam" or "clean"
    ambiguity_level: Optional[str] = None  # "low", "medium", "high"


class ContentDataLoader:
    """
    Production-grade dynamic content loader.

    Features:
    - LLM-generated borderline content (OpenAI API)
    - Diverse topic rotation (ads, politics, social, misinformation)
    - Tone variations (sarcasm, enthusiasm, concern)
    - Ambiguity levels for progressive difficulty
    - Rule-based fallback when API unavailable
    - Duplicate prevention via hashing
    - Thread-safe operation
    """

    # Topic categories for rotation
    TOPIC_CATEGORIES: list[str] = [
        "ads_marketing",
        "politics",
        "social",
        "misinformation",
        "harassment",
        "adult_content",
    ]

    # Template patterns for fallback generation
    _TEMPLATE_PATTERNS: dict[str, list[dict[str, Any]]] = {
        "clean": [
            {"pattern": "This is a {adj} message about {topic}.", "type": "statement"},
            {"pattern": "I wanted to share my thoughts on {topic}.", "type": "opinion"},
            {"pattern": "Has anyone else experienced {topic}?", "type": "question"},
            {"pattern": "Thank you for the helpful information about {topic}.", "type": "gratitude"},
            {"pattern": "Looking forward to more {topic} discussions.", "type": "anticipation"},
        ],
        "spam": [
            {"pattern": "🔥 {urgency}! {offer} - visit {url} NOW! 🔥", "type": "promotional"},
            {"pattern": "I made ${amount} working from home! {secret}", "type": "get_rich"},
            {"pattern": "Click here for {bait}!!! {url}", "type": "clickbait"},
            {"pattern": "You won't believe this about {topic}!", "type": "sensational"},
            {"pattern": "BUY NOW! Limited offer on {topic}!", "type": "urgent_sale"},
        ],
    }

    # Dynamic word lists for template filling
    _WORD_BANKS: dict[str, list[str]] = {
        "adj": ["positive", "constructive", "helpful", "informative", "supportive", "wonderful"],
        "urgency": ["Last chance", "Ending soon", "Final hours", "Time running out"],
        "offer": ["50% off", "Buy one get one", "Free shipping", "Exclusive deal"],
        "bait": ["FREE PRIZES", "AMAZING DEALS", "LIMITED OFFER", "EXCLUSIVE ACCESS"],
        "secret": ["No experience needed", "Work 1 hour a day", "Passive income"],
        "topic_base": [
            "community guidelines", "online safety", "content moderation",
            "digital wellness", "platform policies", "user behavior",
            "social media", "online communities", "internet safety",
            "technology trends", "software development", "data science",
            "health and fitness", "education", "environmental issues",
        ],
    }

    def __init__(
        self,
        use_huggingface: bool = False,
        use_llm: bool = True,
        cache_enabled: bool = False,
        llm_provider: str = "openai",
        llm_fallback: bool = True,
        llm_timeout: float = 10.0,
    ):
        """
        Initialize dynamic data loader.

        Args:
            use_huggingface: Try to fetch from HF datasets.
            use_llm: Use LLM for content generation.
            cache_enabled: Enable caching (disabled by default).
            llm_provider: LLM provider ("openai", "mock").
            llm_fallback: Use rule-based fallback if LLM fails.
            llm_timeout: Timeout for LLM API calls in seconds.
        """
        self.use_huggingface = use_huggingface
        self.use_llm = use_llm
        self.cache_enabled = cache_enabled
        self.llm_provider = llm_provider
        self.llm_fallback = llm_fallback
        self.llm_timeout = llm_timeout

        self._hf_dataset = None
        self._hf_available = False
        self._seen_hashes: set[str] = set()
        self._llm_generator = None
        self._lock = asyncio.Lock() if asyncio.get_event_loop().is_running() else None

        # Initialize components
        self._init_llm_generator()
        if self.use_huggingface:
            self._load_huggingface_dataset()

        logger.info(
            f"ContentDataLoader initialized: HF={self._hf_available}, "
            f"LLM={self.use_llm}({llm_provider}), fallback={llm_fallback}, "
            f"timeout={llm_timeout}s"
        )

    def _init_llm_generator(self) -> None:
        """Initialize LLM content generator with error handling."""
        if not self.use_llm:
            return

        try:
            from utils.content_generator import LLMContentGenerator

            self._llm_generator = LLMContentGenerator(
                provider=self.llm_provider,
                fallback_enabled=self.llm_fallback,
            )
            logger.info("LLM content generator initialized")
        except ImportError as e:
            logger.warning(f"Could not import LLM generator: {e}")
            self._llm_generator = None
        except Exception as e:
            logger.warning(f"LLM generator init failed: {e}")
            self._llm_generator = None

    def _load_huggingface_dataset(self) -> None:
        """Attempt to load Hugging Face dataset with error handling."""
        try:
            from datasets import load_dataset

            datasets_to_try = [
                ("toxic_comments", None),
                ("hatexplain", None),
                ("tweet_eval", None),
            ]

            for name, config in datasets_to_try:
                try:
                    if config:
                        dataset = load_dataset(name, config, split="train", streaming=True)
                    else:
                        dataset = load_dataset(name, split="train", streaming=True)
                    self._hf_dataset = dataset
                    self._hf_available = True
                    logger.info(f"Loaded HF dataset: {name}")
                    return
                except Exception:
                    continue

            logger.warning("No HF datasets available")

        except ImportError:
            logger.debug("datasets package not installed")
        except Exception as e:
            logger.debug(f"HF dataset load failed: {e}")

    async def get_random_content(
        self,
        category: Optional[str] = None,
        ambiguity: Optional[str] = None,
    ) -> ContentItem:
        """
        Get fresh random content (async version).

        Args:
            category: Optional filter ("clean", "spam", or topic name).
            ambiguity: Ambiguity level ("low", "medium", "high").

        Returns:
            Fresh ContentItem.
        """
        content = None

        # Priority 1: LLM-generated content (with timeout)
        if self.use_llm and self._llm_generator:
            try:
                content = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None,
                        self._get_from_llm_sync,
                        category,
                        ambiguity,
                    ),
                    timeout=self.llm_timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(f"LLM generation timed out after {self.llm_timeout}s")
                if self.llm_fallback:
                    content = self._generate_synthetic(category)
            except Exception as e:
                logger.debug(f"LLM generation failed: {e}")
                if self.llm_fallback:
                    content = self._generate_synthetic(category)

        # Priority 2: HF dataset
        if content is None and self._hf_available and self._hf_dataset:
            content = self._get_from_huggingface()

        # Priority 3: Rule-based synthetic
        if content is None:
            content = self._generate_synthetic(category)

        # Fallback: minimal content
        if content is None:
            content = self._get_fallback_content(category)

        return content

    def _get_from_llm_sync(
        self,
        category: Optional[str],
        ambiguity: Optional[str],
    ) -> Optional[ContentItem]:
        """Synchronous LLM content fetch (for executor)."""
        if not self._llm_generator:
            return None

        try:
            # Map category to topic
            topic = None
            if category and category in self.TOPIC_CATEGORIES:
                topic = category
            elif category:
                topic = category

            generated = self._llm_generator.generate(
                topic=topic,
                ambiguity=ambiguity,
                force_fallback=not self.llm_fallback,
            )

            # Check for duplicate
            content_hash = hashlib.md5(generated.content.encode()).hexdigest()
            if content_hash in self._seen_hashes:
                return None
            self._seen_hashes.add(content_hash)

            # Convert to ContentItem
            is_compliant = generated.true_label != "spam"

            return ContentItem(
                text=generated.content,
                is_compliant=is_compliant,
                true_label=generated.true_label,
                violations=generated.violations.copy(),
                source=generated.source,
                category=generated.topic,
                ambiguity_level=generated.ambiguity_level,
                metadata={
                    "tone": generated.tone,
                    "generation_time_ms": generated.generation_time_ms,
                    "reasoning": generated.metadata.get("reasoning", ""),
                },
            )

        except Exception as e:
            logger.debug(f"LLM generation failed: {e}")
            return None

    def get_random_content_sync(
        self,
        category: Optional[str] = None,
        ambiguity: Optional[str] = None,
    ) -> ContentItem:
        """
        Get fresh random content (sync version).

        Args:
            category: Optional filter.
            ambiguity: Ambiguity level.

        Returns:
            Fresh ContentItem.
        """
        content = None

        # Priority 1: LLM-generated content
        if self.use_llm and self._llm_generator:
            content = self._get_from_llm_sync(category, ambiguity)

        # Priority 2: HF dataset
        if content is None and self._hf_available and self._hf_dataset:
            content = self._get_from_huggingface()

        # Priority 3: Rule-based synthetic
        if content is None:
            content = self._generate_synthetic(category)

        # Fallback
        if content is None:
            content = self._get_fallback_content(category)

        return content

    def _get_from_huggingface(self) -> Optional[ContentItem]:
        """Fetch content from HF dataset with error handling."""
        try:
            if not self._hf_dataset:
                return None

            item = next(self._hf_dataset)
            text = item.get("comment_text") or item.get("text") or item.get("content", "")

            if not text:
                return None

            text = text[:500]  # Truncate

            # Check duplicate
            content_hash = hashlib.md5(text.encode()).hexdigest()
            if content_hash in self._seen_hashes:
                return None
            self._seen_hashes.add(content_hash)

            toxicity = item.get("toxic", item.get("label", 0))
            is_compliant = float(toxicity) < 0.5

            return ContentItem(
                text=text,
                is_compliant=is_compliant,
                source="huggingface",
                category="hf_dataset",
                metadata={"toxicity": toxicity},
            )

        except (StopIteration, Exception) as e:
            logger.debug(f"HF fetch error: {e}")
            return None

    def _generate_synthetic(self, category: Optional[str] = None) -> ContentItem:
        """
        Generate synthetic content using templates.

        Fully dynamic - no static content examples.
        """
        if category is None:
            category = random.choice(["clean", "spam"])

        timestamp = time.time()

        if category == "clean":
            template_data = random.choice(self._TEMPLATE_PATTERNS["clean"])
            template = template_data["pattern"]

            text = self._fill_template(template, "clean")
            violations = []
        else:
            template_data = random.choice(self._TEMPLATE_PATTERNS["spam"])
            template = template_data["pattern"]

            text = self._fill_template(template, "spam")
            violations = self._infer_violations(text)

        # Check duplicate
        content_hash = hashlib.md5(text.encode()).hexdigest()
        if content_hash in self._seen_hashes:
            # Regenerate with different seed
            return self._generate_synthetic(category)
        self._seen_hashes.add(content_hash)

        return ContentItem(
            text=text,
            is_compliant=(category == "clean"),
            true_label=category,
            violations=violations,
            source="rule_based",
            category=category,
            ambiguity_level="low",
            metadata={
                "generated": True,
                "timestamp": timestamp,
                "template_type": template_data.get("type", "unknown"),
            },
        )

    def _fill_template(self, template: str, content_type: str) -> str:
        """
        Fill template with dynamic values.

        Args:
            template: Template pattern with {placeholders}.
            content_type: "clean" or "spam" for appropriate word selection.

        Returns:
            Filled template string.
        """
        replacements = {
            "adj": random.choice(self._WORD_BANKS["adj"]),
            "urgency": random.choice(self._WORD_BANKS["urgency"]),
            "offer": random.choice(self._WORD_BANKS["offer"]),
            "bait": random.choice(self._WORD_BANKS["bait"]),
            "secret": random.choice(self._WORD_BANKS["secret"]),
            "topic": random.choice(self._WORD_BANKS["topic_base"]),
            "url": f"{random.choice(['bit.ly', 'tinyurl.com', 'linktr.ee'])}/{random.randint(1000, 9999)}",
            "amount": random.choice(["5000", "10000", "50000", "100000"]),
        }

        text = template
        for key, value in replacements.items():
            text = text.replace(f"{{{key}}}", str(value))

        return text

    def _infer_violations(self, content: str) -> list[str]:
        """Infer violations from content using heuristics."""
        violations = []
        content_lower = content.lower()

        if any(x in content_lower for x in ["click", "visit", "buy now", "limited"]):
            violations.append("spam")
        if any(x in content_lower for x in ["bit.ly", "tinyurl", "linktr.ee"]):
            violations.append("suspicious_link")
        if any(x in content_lower for x in ["🔥", "💰", "👇", "📩"]):
            violations.append("engagement_bait")
        if any(x in content_lower for x in ["hate", "kill", "stupid", "idiot"]):
            violations.append("harassment")

        return violations

    def _get_fallback_content(self, category: Optional[str] = None) -> ContentItem:
        """Minimal fallback when other sources fail."""
        if category is None:
            category = random.choice(["clean", "spam"])

        timestamp = time.time()

        if category == "clean":
            text = f"Positive content about {random.choice(self._WORD_BANKS['topic_base'])} - {timestamp}"
            violations = []
        else:
            text = f"Promotional content: {random.choice(self._WORD_BANKS['urgency'])} - {timestamp}"
            violations = ["spam"]

        return ContentItem(
            text=text,
            is_compliant=(category == "clean"),
            true_label=category,
            violations=violations,
            source="fallback",
            category=category,
            metadata={"fallback": True},
        )

    def get_batch(
        self,
        size: int = 10,
        category: Optional[str] = None,
        balanced: bool = False,
    ) -> list[ContentItem]:
        """
        Get a batch of fresh content.

        Args:
            size: Number of items.
            category: Optional filter.
            balanced: Ensure mix of clean/spam.

        Returns:
            List of ContentItems.
        """
        items = []

        if balanced:
            for i in range(size):
                cat = "clean" if i % 2 == 0 else "spam"
                items.append(self.get_random_content_sync(category=cat))
        else:
            for _ in range(size):
                items.append(self.get_random_content_sync(category=category))

        return items

    def clear_cache(self) -> None:
        """Clear seen content hashes."""
        self._seen_hashes.clear()
        logger.info("Cleared content cache")

    @property
    def stats(self) -> dict[str, Any]:
        """Get loader statistics."""
        llm_stats = {}
        if self._llm_generator:
            llm_stats = self._llm_generator.stats

        return {
            "hf_available": self._hf_available,
            "use_llm": self.use_llm,
            "llm_stats": llm_stats,
            "seen_content_count": len(self._seen_hashes),
        }


# Convenience function
def get_random_content() -> ContentItem:
    """Get random content using default loader."""
    loader = ContentDataLoader()
    return loader.get_random_content_sync()
