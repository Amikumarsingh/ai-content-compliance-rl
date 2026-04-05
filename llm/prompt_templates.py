"""
Prompt Templates for LLM Content Evaluation.

Provides standardized prompts for:
- Content evaluation
- Content editing
- Violation detection
"""

from typing import Any

# System prompt for content evaluation
EVALUATION_SYSTEM_PROMPT = """You are a strict enterprise content compliance evaluator.

Your role is to evaluate content for policy violations and provide:
1. A compliance score (0.0 to 1.0)
2. A list of detected violations
3. An improvement suggestion
4. A reward value (-10 to +10)

Violation types to detect:
- hate_speech: Language targeting groups with hate/slurs
- violence: Threats or glorification of violence
- harassment: Personal attacks, insults, demeaning language
- adult_content: Explicit or adult-themed content
- misinformation: False claims, conspiracy theories
- spam: Promotional, clickbait, excessive punctuation
- illegal_content: References to illegal activities

Return ONLY valid JSON in this exact format:
{
    "compliance_score": 0.8,
    "violations": [],
    "suggestion": "Content looks good.",
    "reward": 8
}"""

# Agent personas for multi-agent evaluation
AGENT_PERSONAS = [
    {
        "id": 1,
        "name": "Strict",
        "system_prompt": """You are a STRICT content compliance evaluator.
Flag any potential policy violations. Err on the side of caution.
Return ONLY valid JSON with compliance_score, violations, suggestion, and reward.""",
    },
    {
        "id": 2,
        "name": "Balanced",
        "system_prompt": """You are a BALANCED content compliance evaluator.
Consider context and intent while enforcing policies fairly.
Return ONLY valid JSON with compliance_score, violations, suggestion, and reward.""",
    },
    {
        "id": 3,
        "name": "Lenient",
        "system_prompt": """You are a LENIENT content compliance evaluator.
Give content the benefit of the doubt unless clearly violating.
Return ONLY valid JSON with compliance_score, violations, suggestion, and reward.""",
    },
]

# User prompt template for evaluation
EVALUATION_USER_PROMPT = """CONTENT TO EVALUATE:
---
{content}
---

ACTION TAKEN: {action}

Evaluate this content and action. Return ONLY valid JSON."""

# Prompt template for content editing
EDIT_PROMPT = """Rewrite the following content to address this concern:

SUGGESTION: {suggestion}

ORIGINAL CONTENT:
{content}

Return only the rewritten content, no explanations."""

# Prompt template for content generation (training data)
GENERATION_PROMPT = """Generate a piece of content for compliance training.

Difficulty: {difficulty}
Violation types to include: {violations}

Generate realistic content that would appear on a content platform.
Include the specified violation types naturally within the content.

Return ONLY the generated content, no explanations."""


def build_evaluation_prompt(content: str, action: str) -> str:
    """Build evaluation prompt from template."""
    return EVALUATION_USER_PROMPT.format(content=content, action=action)


def build_edit_prompt(suggestion: str, content: str) -> str:
    """Build edit prompt from template."""
    return EDIT_PROMPT.format(suggestion=suggestion, content=content)


def build_generation_prompt(difficulty: str, violations: list[str]) -> str:
    """Build content generation prompt from template."""
    violations_str = ", ".join(violations) if violations else "none"
    return GENERATION_PROMPT.format(difficulty=difficulty, violations=violations_str)


def get_agent_system_prompt(agent_id: int) -> str:
    """Get system prompt for a specific agent."""
    for agent in AGENT_PERSONAS:
        if agent["id"] == agent_id:
            return agent["system_prompt"]
    return EVALUATION_SYSTEM_PROMPT


def get_all_templates() -> dict[str, Any]:
    """Get all prompt templates."""
    return {
        "evaluation_system": EVALUATION_SYSTEM_PROMPT,
        "evaluation_user": EVALUATION_USER_PROMPT,
        "edit": EDIT_PROMPT,
        "generation": GENERATION_PROMPT,
        "agent_personas": AGENT_PERSONAS,
    }
