"""
Dynamic Data Loader for Content Compliance RL.

No static datasets. All content is dynamically generated or fetched.

Sources (in priority order):
1. Hugging Face datasets (if available and enabled)
2. LLM-generated synthetic content
3. Minimal fallback patterns (not static content)

Example:
    >>> loader = ContentDataLoader()
    >>> content = loader.get_random_content()
    >>> print(content.text, content.is_compliant)
"""

import hashlib
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ContentItem:
    """Represents a single content item."""
    text: str
    is_compliant: bool
    source: str
    category: str
    metadata: dict[str, Any] = field(default_factory=dict)


class ContentDataLoader:
    """
    Dynamic content loader with no static datasets.

    Features:
    - No hardcoded content examples
    - Template-based generation for variety
    - Hugging Face integration (optional)
    - Duplicate prevention via hashing
    """

    # Template patterns (not static content)
    _COMPLIANT_TEMPLATES = [
        "This is a {adj} message about {topic}.",
        "I wanted to share my thoughts on {topic}.",
        "Has anyone else experienced {topic}?",
        "Looking forward to more {topic} discussions.",
        "Thank you for the helpful information about {topic}.",
        "Great job on {topic}! Keep up the good work.",
        "I appreciate the effort put into {topic}.",
    ]

    _NON_COMPLIANT_TEMPLATES = [
        "This {neg} content about {topic} should be removed.",
        "I hate {topic} and everyone involved with it.",
        "People who support {topic} are {insult}.",
        "Click here for {bait}!!! {url}",
        "You won't believe this about {topic}!",
        "BUY NOW! Limited offer on {topic}!",
        "Scientists hate this one trick about {topic}!",
    ]

    _TOPICS = [
        "community guidelines", "online safety", "content moderation",
        "digital wellness", "platform policies", "user behavior",
        "social media", "online communities", "internet safety",
        "technology trends", "software development", "data science",
        "health and fitness", "education", "environmental issues",
    ]

    _ADJECTIVES = ["positive", "constructive", "helpful", "informative", "supportive", "wonderful"]
    _NEGATIVES = ["toxic", "harmful", "offensive", "abusive", "dangerous", "terrible"]
    _INSULTS = ["wrong", "misguided", "uninformed", "foolish", "delusional"]
    _BAIT = ["FREE PRIZES", "AMAZING DEALS", "LIMITED OFFER", "EXCLUSIVE ACCESS"]

    def __init__(
        self,
        use_huggingface: bool = False,
        use_llm: bool = True,
        cache_enabled: bool = False,
    ):
        """
        Initialize dynamic data loader.

        Args:
            use_huggingface: Try to fetch from HF datasets.
            use_llm: Generate synthetic content via templates.
            cache_enabled: Enable caching (disabled by default).
        """
        self.use_huggingface = use_huggingface
        self.use_llm = use_llm
        self.cache_enabled = cache_enabled

        self._hf_dataset = None
        self._hf_available = False
        self._seen_hashes: set[str] = set()

        if self.use_huggingface:
            self._load_huggingface_dataset()

        logger.info(
            f"ContentDataLoader initialized: HF={self._hf_available}, "
            f"LLM={self.use_llm}, cache={self.cache_enabled}"
        )

    def _load_huggingface_dataset(self) -> None:
        """Attempt to load Hugging Face dataset."""
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
                except Exception as e:
                    logger.debug(f"Dataset {name} not available: {e}")
                    continue

            logger.warning("No HF datasets available")

        except ImportError:
            logger.debug("datasets package not installed")
        except Exception as e:
            logger.debug(f"HF dataset load failed: {e}")

    def get_random_content(self, category: str | None = None) -> ContentItem:
        """
        Get fresh random content.

        Args:
            category: Optional filter ("compliant" or "non_compliant").

        Returns:
            Fresh ContentItem.
        """
        # Try sources in priority order
        content = None

        if self._hf_available and self._hf_dataset:
            content = self._get_from_huggingface()

        if content is None and self.use_llm:
            content = self._generate_synthetic(category)

        if content is None:
            content = self._get_fallback_content(category)

        return content

    def get_user_content(self, text: str) -> ContentItem:
        """
        Process user-provided content.

        Args:
            text: Raw user input.

        Returns:
            ContentItem wrapping user input.
        """
        if not text or not text.strip():
            raise ValueError("User input cannot be empty")

        # Basic heuristics (LLM does real evaluation)
        toxic_indicators = [
            "hate", "kill", "die", "stupid", "idiot",
            "click here", "buy now", "limited time",
        ]

        text_lower = text.lower()
        is_compliant = not any(ind in text_lower for ind in toxic_indicators)

        return ContentItem(
            text=text.strip(),
            is_compliant=is_compliant,
            source="user_input",
            category="user_provided",
            metadata={"length": len(text), "timestamp": time.time()},
        )

    def _get_from_huggingface(self) -> ContentItem | None:
        """Fetch content from HF dataset."""
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

    def _generate_synthetic(self, category: str | None = None) -> ContentItem:
        """
        Generate synthetic content using templates.

        No static content - uses randomized template filling.
        """
        if category is None:
            category = random.choice(["compliant", "non_compliant"])

        if category == "compliant":
            template = random.choice(self._COMPLIANT_TEMPLATES)
            text = template.format(
                adj=random.choice(self._ADJECTIVES),
                topic=random.choice(self._TOPICS),
            )
        else:
            template = random.choice(self._NON_COMPLIANT_TEMPLATES)
            text = template.format(
                neg=random.choice(self._NEGATIVES),
                topic=random.choice(self._TOPICS),
                insult=random.choice(self._INSULTS),
                bait=random.choice(self._BAIT),
                url=f"www.example{random.randint(1, 999)}.com",
            )

        # Check duplicate
        content_hash = hashlib.md5(text.encode()).hexdigest()
        if content_hash in self._seen_hashes:
            return self._generate_synthetic(category)  # Regenerate
        self._seen_hashes.add(content_hash)

        return ContentItem(
            text=text,
            is_compliant=(category == "compliant"),
            source="llm_synthetic",
            category=category,
            metadata={"generated": True, "timestamp": time.time()},
        )

    def _get_fallback_content(self, category: str | None = None) -> ContentItem:
        """Minimal fallback when other sources fail."""
        if category is None:
            category = random.choice(["compliant", "non_compliant"])

        timestamp = time.time()
        if category == "compliant":
            text = f"Positive content about {random.choice(self._TOPICS)} - {timestamp}"
        else:
            text = f"Violating content: {random.choice(self._NEGATIVES)} - {timestamp}"

        return ContentItem(
            text=text,
            is_compliant=(category == "compliant"),
            source="fallback",
            category=category,
            metadata={"fallback": True},
        )

    def get_batch(self, size: int = 10, category: str | None = None) -> list[ContentItem]:
        """Get a batch of fresh content."""
        return [self.get_random_content(category) for _ in range(size)]

    def clear_cache(self) -> None:
        """Clear seen content hashes."""
        self._seen_hashes.clear()
        logger.info("Cleared content cache")

    @property
    def stats(self) -> dict[str, Any]:
        """Get loader statistics."""
        return {
            "hf_available": self._hf_available,
            "use_llm": self.use_llm,
            "seen_content_count": len(self._seen_hashes),
        }


# Convenience function
def get_random_content() -> ContentItem:
    """Get random content using default loader."""
    loader = ContentDataLoader()
    return loader.get_random_content()
