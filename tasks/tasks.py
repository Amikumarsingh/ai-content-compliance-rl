"""
Task Definitions for Content Moderation RL Environment.

Defines three difficulty levels with progressively challenging content:
- easy: Obvious spam/hate speech detection (binary decisions)
- medium: Borderline content requiring edits
- hard: Complex multi-violation content requiring reasoning

Each task configuration includes:
- Content generation parameters
- Expected actions
- Success criteria
- Reward shaping parameters
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class TaskConfig:
    """Configuration for a single task."""

    task_id: str
    name: str
    description: str
    difficulty: str
    max_steps: int
    content_generator: str  # "dynamic" or "static"
    expected_action: str  # "approve", "reject", or "edit_then_approve"
    violation_types: list[str] = field(default_factory=list)
    success_criteria: dict[str, Any] = field(default_factory=dict)
    reward_shaping: dict[str, float] = field(default_factory=dict)


# Task definitions for each difficulty level
TASK_DEFINITIONS: dict[str, list[TaskConfig]] = {
    "easy": [
        TaskConfig(
            task_id="easy_001",
            name="Spam Detection",
            description="Detect obvious promotional spam content",
            difficulty="easy",
            max_steps=2,
            content_generator="dynamic",
            expected_action="reject",
            violation_types=["spam"],
            success_criteria={
                "correct_action": "reject",
                "min_score": 0.8,
            },
            reward_shaping={
                "correct_reject": 1.0,
                "incorrect_approve": 0.0,
            },
        ),
        TaskConfig(
            task_id="easy_002",
            name="Hate Speech Detection",
            description="Detect clear hate speech content",
            difficulty="easy",
            max_steps=2,
            content_generator="dynamic",
            expected_action="reject",
            violation_types=["hate_speech"],
            success_criteria={
                "correct_action": "reject",
                "min_score": 0.8,
            },
            reward_shaping={
                "correct_reject": 1.0,
                "incorrect_approve": -0.2,  # Penalty for approving harmful content
            },
        ),
        TaskConfig(
            task_id="easy_003",
            name="Clean Content Approval",
            description="Approve compliant welcome messages",
            difficulty="easy",
            max_steps=2,
            content_generator="dynamic",
            expected_action="approve",
            violation_types=[],
            success_criteria={
                "correct_action": "approve",
                "min_score": 0.8,
            },
            reward_shaping={
                "correct_approve": 1.0,
                "incorrect_reject": 0.3,
            },
        ),
    ],

    "medium": [
        TaskConfig(
            task_id="medium_001",
            name="Borderline Spam Edit",
            description="Edit borderline promotional content to remove spam signals",
            difficulty="medium",
            max_steps=3,
            content_generator="dynamic",
            expected_action="edit_then_approve",
            violation_types=["spam", "engagement_bait"],
            success_criteria={
                "correct_sequence": ["edit", "approve"],
                "final_score": 0.7,
            },
            reward_shaping={
                "correct_edit": 0.5,
                "final_approve": 0.5,
                "unnecessary_edit": -0.1,
            },
        ),
        TaskConfig(
            task_id="medium_002",
            name="Subtle Harassment",
            description="Identify and edit subtle harassment in discussion",
            difficulty="medium",
            max_steps=3,
            content_generator="dynamic",
            expected_action="edit_then_approve",
            violation_types=["harassment"],
            success_criteria={
                "correct_sequence": ["edit", "approve"],
                "final_score": 0.6,
            },
            reward_shaping={
                "correct_edit": 0.6,
                "final_approve": 0.4,
            },
        ),
    ],

    "hard": [
        TaskConfig(
            task_id="hard_001",
            name="Multi-Violation Analysis",
            description="Detect and handle content with multiple violation types",
            difficulty="hard",
            max_steps=5,
            content_generator="dynamic",
            expected_action="reject",
            violation_types=["hate_speech", "violence", "spam"],
            success_criteria={
                "detect_all_violations": True,
                "correct_action": "reject",
                "final_score": 0.5,
            },
            reward_shaping={
                "violation_detected": 0.1,  # Per violation
                "correct_reject": 0.7,
                "dangerous_approve": -0.5,
            },
        ),
        TaskConfig(
            task_id="hard_002",
            name="Coded Hate Speech",
            description="Identify subtle coded language and dog whistles",
            difficulty="hard",
            max_steps=5,
            content_generator="dynamic",
            expected_action="reject",
            violation_types=["hate_speech", "misinformation"],
            success_criteria={
                "detect_subtle_violations": True,
                "correct_action": "reject",
            },
            reward_shaping={
                "subtle_detection": 0.3,
                "correct_reject": 0.7,
            },
        ),
        TaskConfig(
            task_id="hard_003",
            name="Misinformation with Truth",
            description="Handle content mixing facts with false claims",
            difficulty="hard",
            max_steps=5,
            content_generator="dynamic",
            expected_action="edit_then_approve",
            violation_types=["misinformation"],
            success_criteria={
                "remove_falsehoods": True,
                "preserve_truth": True,
                "final_score": 0.6,
            },
            reward_shaping={
                "partial_edit": 0.3,
                "complete_edit": 0.7,
            },
        ),
    ],
}


# Content templates for dynamic generation
CONTENT_TEMPLATES: dict[str, list[dict[str, Any]]] = {
    "easy_spam": [
        {
            "template": "🔥 {urgency}! {offer} - visit {url} NOW! 🔥",
            "violations": ["spam", "engagement_bait"],
            "expected_action": "reject",
        },
        {
            "template": "I made ${amount} working from home! {secret}",
            "violations": ["spam"],
            "expected_action": "reject",
        },
    ],
    "easy_hate": [
        {
            "template": "I hate {group}. They should all {harmful_action}.",
            "violations": ["hate_speech", "violence"],
            "expected_action": "reject",
        },
    ],
    "easy_clean": [
        {
            "template": "Hello! Welcome to our community. Looking forward to {positive_activity}!",
            "violations": [],
            "expected_action": "approve",
        },
        {
            "template": "Thanks for sharing this {adj} information about {topic}.",
            "violations": [],
            "expected_action": "approve",
        },
    ],
    "medium_borderline": [
        {
            "template": "Check out this amazing opportunity! {url} (Not spam, real!)",
            "violations": ["spam", "suspicious_link"],
            "expected_action": "edit",
            "edit_target": "Remove URL and defensive language",
        },
        {
            "template": "Some people are just too {negative_adj} to understand {topic}. Sad.",
            "violations": ["harassment"],
            "expected_action": "edit",
            "edit_target": "Remove negative characterization",
        },
    ],
    "hard_multi": [
        {
            "template": "{group} are { slur } and should {harmful}. Don't believe the {conspiracy} lies!",
            "violations": ["hate_speech", "violence", "misinformation"],
            "expected_action": "reject",
        },
        {
            "template": "They don't want you to know {falsehood}! {authority} is hiding the truth about {topic}!",
            "violations": ["misinformation", "engagement_bait"],
            "expected_action": "reject",
        },
    ],
}


def get_task_config(difficulty: str, task_id: Optional[str] = None) -> TaskConfig:
    """
    Get task configuration for specified difficulty.

    Args:
        difficulty: Task difficulty ("easy", "medium", "hard").
        task_id: Optional specific task ID.

    Returns:
        TaskConfig for the specified task.
    """
    tasks = TASK_DEFINITIONS.get(difficulty, TASK_DEFINITIONS["easy"])

    if task_id:
        for task in tasks:
            if task.task_id == task_id:
                return task

    # Return random task from difficulty level
    import random
    return random.choice(tasks)


def get_content_template(content_type: str) -> dict[str, Any]:
    """
    Get content template for specified type.

    Args:
        content_type: Type of content template.

    Returns:
        Dictionary with template and metadata.
    """
    templates = CONTENT_TEMPLATES.get(content_type, CONTENT_TEMPLATES["easy_clean"])
    import random
    return random.choice(templates)


def get_all_task_ids() -> list[str]:
    """Get all task IDs across all difficulty levels."""
    all_ids = []
    for difficulty_tasks in TASK_DEFINITIONS.values():
        all_ids.extend([task.task_id for task in difficulty_tasks])
    return all_ids


def get_difficulty_for_task(task_id: str) -> str:
    """Get difficulty level for a specific task ID."""
    for difficulty, tasks in TASK_DEFINITIONS.items():
        for task in tasks:
            if task.task_id == task_id:
                return difficulty
    return "easy"
