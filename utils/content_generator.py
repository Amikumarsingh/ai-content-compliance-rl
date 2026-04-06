"""
LLM-Powered Dynamic Content Generator for Content Compliance RL.

Generates ambiguous, borderline content using OpenAI API to create
realistic moderation challenges that require nuanced reasoning.

Features:
- Borderline spam detection
- Subtle policy violations
- Ambiguous tone (sarcasm, political, marketing)
- Topic diversity rotation
- Fallback to rule-based generation if API fails

Example:
    >>> generator = LLMContentGenerator()
    >>> item = generator.generate()
    >>> print(item.text, item.true_label, item.violations)
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class GeneratedContent:
    """Represents LLM-generated content with metadata."""
    content: str
    true_label: str  # "spam" or "clean"
    violations: list[str]
    topic: str
    tone: str
    ambiguity_level: str  # "low", "medium", "high"
    raw_response: Optional[str] = None
    generation_time_ms: float = 0.0
    source: str = "llm"
    metadata: dict[str, Any] = field(default_factory=dict)


class LLMContentGenerator:
    """
    LLM-powered content generator for realistic moderation training.

    Generates content that:
    - Has ambiguous violation signals
    - Requires contextual reasoning
    - Covers diverse topics (ads, politics, social, misinformation)
    - Varies in tone (sarcasm, enthusiasm, concern, marketing)
    """

    # Topic categories for diversity
    TOPICS = {
        "ads_marketing": [
            "promotional content", "product reviews", "affiliate links",
            "discount offers", "brand mentions", "sponsorship disclosure",
        ],
        "politics": [
            "election discussion", "policy debate", "political opinion",
            "candidate criticism", "partisan commentary", "governance",
        ],
        "social": [
            "community drama", "interpersonal conflict", "social commentary",
            "cultural discussion", "identity topics", "relationship advice",
        ],
        "misinformation": [
            "health claims", "conspiracy theories", "unverified news",
            "scientific disputes", "historical revision", "statistical manipulation",
        ],
        "harassment": [
            "veiled insults", "passive aggression", "dog whistles",
            "coded language", "implicit threats", "subtle intimidation",
        ],
        "adult_content": [
            "suggestive language", "innuendo", "flirtatious messages",
            "dating solicitation", "romantic advances", "personal ads",
        ],
    }

    # Tone variations
    TONES = [
        "sarcastic", "enthusiastic", "concerned", "angry", "humorous",
        "informative", "promotional", "critical", "supportive", "neutral",
    ]

    # Ambiguity levels
    AMBIGUITY_LEVELS = ["low", "medium", "high"]

    # System prompt for LLM
    SYSTEM_PROMPT = """You are a content generation assistant for training content moderation AI.

Your task is to create realistic social media posts or online content that:
1. May or may not violate community guidelines
2. Requires careful reasoning to classify
3. Represents real-world moderation challenges

Content should be ambiguous enough that:
- A human moderator would need to think carefully
- Context and nuance matter
- Clear-cut cases are rare

Return ONLY valid JSON with these fields:
{
    "content": "The generated post text (50-200 characters)",
    "true_label": "spam" or "clean",
    "violations": ["list of potential violations, empty if clean"],
    "topic": "topic category used",
    "tone": "tone of the content",
    "ambiguity_level": "low", "medium", or "high",
    "reasoning": "brief explanation of why this is challenging"
}

Guidelines:
- Make content realistic (like actual social media posts)
- Include subtle signals, not obvious violations
- Vary writing style and quality
- Some content should have typos or informal language
- Avoid generating genuinely harmful content"""

    def __init__(
        self,
        provider: str = "openai",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        fallback_enabled: bool = True,
    ):
        """
        Initialize LLM content generator.

        Args:
            provider: LLM provider ("openai", "anthropic", "mock").
            model: Model name (auto-detected if None).
            api_key: API key (uses env var if None).
            fallback_enabled: Use rule-based fallback if API fails.
        """
        self.provider = provider
        self.fallback_enabled = fallback_enabled
        self._client = None
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")

        # Statistics
        self._stats = {
            "total_generated": 0,
            "llm_success": 0,
            "fallback_used": 0,
            "errors": 0,
        }

        # Initialize client
        if self.provider == "openai":
            self._init_openai()
        elif self.provider == "mock":
            logger.info("Using mock LLM provider (rule-based generation)")

        logger.info(
            f"LLMContentGenerator initialized: provider={provider}, "
            f"model={self._model}, fallback={fallback_enabled}"
        )

    def _init_openai(self) -> None:
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=self._api_key)
            logger.info("OpenAI client initialized")
        except ImportError:
            logger.warning("openai package not installed, using fallback")
            self._client = None
        except Exception as e:
            logger.warning(f"OpenAI client init failed: {e}")
            self._client = None

    def generate(
        self,
        topic: Optional[str] = None,
        tone: Optional[str] = None,
        ambiguity: Optional[str] = None,
        force_fallback: bool = False,
    ) -> GeneratedContent:
        """
        Generate ambiguous content for moderation training.

        Args:
            topic: Topic category (random if None).
            tone: Content tone (random if None).
            ambiguity: Ambiguity level (random if None).
            force_fallback: Skip LLM, use rule-based generation.

        Returns:
            GeneratedContent with text, labels, and metadata.
        """
        self._stats["total_generated"] += 1

        # Select parameters
        if topic is None:
            topic_category = random.choice(list(self.TOPICS.keys()))
            topic = random.choice(self.TOPICS[topic_category])
        else:
            topic_category = topic
            topic = random.choice(self.TOPICS.get(topic, [topic]))

        if tone is None:
            tone = random.choice(self.TONES)

        if ambiguity is None:
            ambiguity = random.choice(self.AMBIGUITY_LEVELS)

        # Generate via LLM or fallback
        if force_fallback or self._client is None:
            content = self._generate_rule_based(topic, tone, ambiguity)
            self._stats["fallback_used"] += 1
        else:
            content = self._generate_via_llm(topic, tone, ambiguity)
            if content is None:
                content = self._generate_rule_based(topic, tone, ambiguity)
                self._stats["fallback_used"] += 1
            else:
                self._stats["llm_success"] += 1

        return content

    def _generate_via_llm(
        self,
        topic: str,
        tone: str,
        ambiguity: str,
    ) -> Optional[GeneratedContent]:
        """Generate content using LLM."""
        start_time = time.time()

        try:
            # Build user prompt
            user_prompt = f"""Generate a piece of content with these characteristics:

Topic: {topic}
Tone: {tone}
Ambiguity Level: {ambiguity}

The content should be a realistic social media post or online message that may or may not violate community guidelines. Make it ambiguous enough to require careful reasoning."""

            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.8,
                max_tokens=300,
                response_format={"type": "json_object"},
            )

            generation_time = (time.time() - start_time) * 1000

            # Parse response
            raw_content = response.choices[0].message.content
            parsed = json.loads(raw_content)

            # Validate required fields
            required = ["content", "true_label", "violations"]
            if not all(k in parsed for k in required):
                logger.warning(f"LLM response missing required fields: {parsed}")
                return None

            content = GeneratedContent(
                content=parsed["content"][:500],  # Truncate long responses
                true_label=parsed["true_label"].lower(),
                violations=parsed.get("violations", []),
                topic=parsed.get("topic", topic),
                tone=parsed.get("tone", tone),
                ambiguity_level=parsed.get("ambiguity_level", ambiguity),
                raw_response=raw_content,
                generation_time_ms=generation_time,
                source="openai",
                metadata={
                    "model": self._model,
                    "reasoning": parsed.get("reasoning", ""),
                },
            )

            logger.debug(
                f"LLM generated: label={content.true_label}, "
                f"violations={len(content.violations)}, time={generation_time:.0f}ms"
            )

            return content

        except Exception as e:
            self._stats["errors"] += 1
            logger.warning(f"LLM generation failed: {e}")
            return None

    def _generate_rule_based(
        self,
        topic: str,
        tone: str,
        ambiguity: str,
    ) -> GeneratedContent:
        """
        Rule-based content generation (fallback when LLM unavailable).

        Uses templates and patterns to create varied content.
        """
        start_time = time.time()

        # Select template based on topic/tone/ambiguity
        templates = self._get_templates_for_topic(topic)
        template = random.choice(templates)

        # Fill template with variations
        content_text = self._fill_template(template, tone, ambiguity)

        # Determine label and violations based on template type
        is_spam_template = "spam" in template["type"]
        true_label = "spam" if is_spam_template else "clean"

        violations = []
        if is_spam_template:
            violations = self._infer_violations(content_text, topic)

        generation_time = (time.time() - start_time) * 1000

        content = GeneratedContent(
            content=content_text,
            true_label=true_label,
            violations=violations,
            topic=topic,
            tone=tone,
            ambiguity_level=ambiguity,
            source="rule_based",
            metadata={
                "template_type": template["type"],
                "fallback": True,
            },
        )

        logger.debug(
            f"Rule-based generated: label={content.true_label}, "
            f"violations={len(content.violations)}"
        )

        return content

    def _get_templates_for_topic(self, topic: str) -> list[dict[str, Any]]:
        """Get templates appropriate for the topic."""
        # Spam templates
        spam_templates = [
            {
                "type": "spam_promotional",
                "patterns": [
                    "🔥 {urgency}! {offer} - visit {url} NOW! 🔥",
                    "I made ${amount} working from home! {secret} Click {url}",
                    "{celebrity} endorses {product}! See the shocking truth at {url}",
                    "FREE {item} for the first {number} people! DM me! 💰",
                    "This one weird trick will {benefit}! Scientists HATE this!",
                ],
            },
            {
                "type": "spam_engagement_bait",
                "patterns": [
                    "Tag someone who needs to see this! 👇 Share if you agree!",
                    "99% of people can't solve this! Are you a genius? 🧠",
                    "Like this post if you love {topic}! Let's see how many! ❤️",
                    "Comment '{word}' below and I'll send you the link! 📩",
                ],
            },
            {
                "type": "spam_misinformation",
                "patterns": [
                    "They don't want you to know about {conspiracy}! Wake up! 👁️",
                    "{celebrity} secretly supports {controversial}! The evidence is shocking!",
                    "The truth about {topic} that {authority} is hiding from you!",
                    "New study proves {falsehood}! Mainstream media won't report this!",
                ],
            },
            {
                "type": "spam_harassment_subtle",
                "patterns": [
                    "Some people are just too {negative} to understand {topic}. Sad.",
                    "If you support {group}, you're part of the problem. Think about it.",
                    "I feel sorry for people who believe {idea}. So misguided.",
                    "Another day, another {group} person proving my point. 🙄",
                ],
            },
        ]

        # Clean templates (ambiguous)
        clean_templates = [
            {
                "type": "clean_opinion",
                "patterns": [
                    "Unpopular opinion: {controversial_opinion}. Change my mind.",
                    "I know this is controversial, but I think {opinion}. Thoughts?",
                    "Hot take: {opinion}. I know some will disagree but here's why...",
                    "Maybe I'm wrong about {topic}, but here's my perspective...",
                ],
            },
            {
                "type": "clean_discussion",
                "patterns": [
                    "Can we talk about {topic}? I have mixed feelings.",
                    "Genuinely curious: what do people think about {issue}?",
                    "I've been researching {topic} and found some interesting things...",
                    "Does anyone else find it strange that {observation}?",
                ],
            },
            {
                "type": "clean_sarcastic",
                "patterns": [
                    "Oh great, another {topic} discussion. Just what I needed. 🙃",
                    "Sure, because that ALWAYS works. /s",
                    "Nothing says 'trustworthy' like {suspicious_thing}. /sarcasm",
                    "Ah yes, the classic '{cliche}'. Because that's not outdated at all.",
                ],
            },
            {
                "type": "clean_concern",
                "patterns": [
                    "I'm genuinely worried about {trend}. Should we be concerned?",
                    "Has anyone else noticed {concerning_thing}? Just me? Okay.",
                    "Not to be alarmist, but {concern}. What are your thoughts?",
                    "I hope I'm overthinking this, but {worry}...",
                ],
            },
        ]

        # Return mix based on topic
        if topic in ["ads_marketing"]:
            return spam_templates[:2] + clean_templates[:2]
        elif topic in ["politics", "social"]:
            return spam_templates[3:] + clean_templates
        elif topic in ["misinformation"]:
            return spam_templates[2:3] + clean_templates[1:3]
        else:
            return spam_templates + clean_templates

    def _fill_template(
        self,
        template: dict[str, Any],
        tone: str,
        ambiguity: str,
    ) -> str:
        """Fill template with varied content."""
        pattern = random.choice(template["patterns"])

        # Fill-in values
        replacements = {
            "urgency": random.choice(["Last chance", "Ending soon", "Final hours", "Time running out"]),
            "offer": random.choice(["50% off", "Buy one get one", "Free shipping", "Exclusive deal"]),
            "url": f"{random.choice(['bit.ly', 'tinyurl.com', 'linktr.ee'])}/{random.randint(1000, 9999)}",
            "amount": random.choice(["$5000", "$10000", "$50000", "six figures"]),
            "secret": random.choice(["No experience needed", "Work 1 hour a day", "Passive income"]),
            "celebrity": random.choice(["Famous actor", "Tech billionaire", "Popular influencer", "Award winner"]),
            "product": random.choice(["this supplement", "this course", "this app", "this system"]),
            "item": random.choice(["gift card", "iPhone", "cash", "prize package"]),
            "number": random.randint(5, 50),
            "benefit": random.choice(["lose weight", "make money", "get famous", "live longer"]),
            "topic": random.choice(["current events", "social issues", "trending topics", "online drama"]),
            "word": random.choice(["INFO", "LINK", "DETAILS", "ACCESS"]),
            "conspiracy": random.choice(["the truth about vaccines", "government surveillance", "media lies"]),
            "controversial": random.choice(["the earth is flat", "aliens built the pyramids", "moon landing fake"]),
            "authority": random.choice(["the government", "big pharma", "mainstream media", "tech companies"]),
            "falsehood": random.choice(["vaccines cause harm", "climate change is fake", "flat earth theory"]),
            "negative": random.choice(["close-minded", "uneducated", "biased", "ignorant"]),
            "group": random.choice(["liberal", "conservative", "activist", "skeptic"]),
            "idea": random.choice(["that conspiracy theory", "this political movement", "trend"]),
            "controversial_opinion": random.choice([
                "pineapple belongs on pizza", "summer is the worst season",
                "cats are better than dogs", "social media does more harm than good"
            ]),
            "opinion": random.choice([
                "remote work is overrated", "AI will change everything",
                "traditional education needs reform", "social media is divisive"
            ]),
            "issue": random.choice([
                "the state of online discourse", "rising costs", "tech addiction",
                "work-life balance", "mental health awareness"
            ]),
            "observation": random.choice([
                "everyone suddenly became an expert", "nobody reads articles anymore",
                "all conversations turn political", "trends recycle every 5 years"
            ]),
            "suspicious_thing": random.choice([
                "a verified account with 10 followers", "a link shortener",
                "DMs open for 'business opportunities'", "crypto in bio"
            ]),
            "cliche": random.choice([
                "thoughts and prayers", "stay positive", "good vibes only",
                "live laugh love", "hustle culture"
            ]),
            "trend": random.choice([
                "AI replacing jobs", "housing market", "dating apps", "social media algorithms"
            ]),
            "concerning_thing": random.choice([
                "how fast misinformation spreads", "decline in civil discourse",
                "privacy erosion", "attention economy"
            ]),
            "concern": random.choice([
                "young people are struggling", "trust in institutions is falling",
                "everything is monetized now", "nuance is disappearing"
            ]),
            "worry": random.choice([
                "we're heading for recession", "AI is advancing too fast",
                "social media is making us unhappy", "nobody trusts experts anymore"
            ]),
        }

        # Fill pattern
        for key, value in replacements.items():
            pattern = pattern.replace(f"{{{key}}}", str(value))

        # Add tone modifiers
        if tone == "sarcastic":
            pattern = pattern + " /s" if random.random() > 0.5 else pattern
        elif tone == "enthusiastic":
            pattern = "🎉 " + pattern + "!!!" if random.random() > 0.5 else pattern
        elif tone == "concerned":
            pattern = "Honestly worried about this... " + pattern

        # Add ambiguity markers
        if ambiguity == "high":
            markers = ["Maybe I'm wrong...", "Thoughts?", "Change my mind.", "Discuss."]
            pattern = pattern + " " + random.choice(markers)

        return pattern

    def _infer_violations(self, content: str, topic: str) -> list[str]:
        """Infer violations from content using heuristics."""
        violations = []
        content_lower = content.lower()

        # Spam indicators
        if any(x in content_lower for x in ["click", "visit", "dm me", "free", "limited"]):
            violations.append("spam")
        if any(x in content_lower for x in ["bit.ly", "tinyurl", "linktr.ee"]):
            violations.append("suspicious_link")
        if any(x in content_lower for x in ["🔥", "💰", "👇", "📩", "❤️"]):
            violations.append("engagement_bait")

        # Harassment indicators
        if any(x in content_lower for x in ["sad", "pathetic", "fool", "delusional"]):
            violations.append("harassment")

        # Misinformation indicators
        if any(x in content_lower for x in ["they don't want you", "wake up", "truth about", "hidden"]):
            violations.append("misinformation")

        # Adult content indicators
        if any(x in content_lower for x in ["dm for pics", "18+", "nsfw", "adult"]):
            violations.append("adult_content")

        return violations

    def generate_batch(
        self,
        size: int = 10,
        balanced: bool = True,
    ) -> list[GeneratedContent]:
        """
        Generate a batch of content items.

        Args:
            size: Number of items to generate.
            balanced: Ensure mix of spam/clean.

        Returns:
            List of GeneratedContent items.
        """
        items = []

        if balanced:
            # Ensure roughly equal spam/clean
            for i in range(size):
                topic = random.choice(list(self.TOPICS.keys()))
                item = self.generate(topic=topic)
                items.append(item)
        else:
            for _ in range(size):
                item = self.generate()
                items.append(item)

        return items

    @property
    def stats(self) -> dict[str, Any]:
        """Get generation statistics."""
        return {
            **self._stats,
            "success_rate": (
                self._stats["llm_success"] / max(1, self._stats["total_generated"])
            ),
        }


# Convenience function
def generate_content(
    topic: Optional[str] = None,
    tone: Optional[str] = None,
    ambiguity: Optional[str] = None,
) -> GeneratedContent:
    """Generate content using default generator."""
    generator = LLMContentGenerator()
    return generator.generate(topic, tone, ambiguity)
