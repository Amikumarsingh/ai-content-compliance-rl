"""
OpenEnv Tasks for Content Compliance Training.

Provides standardized tasks with deterministic graders for training
and evaluating RL agents on content moderation.

Tasks:
    - Task 1 (Easy): Binary approve/reject decision
    - Task 2 (Medium): Edit content to achieve compliance
    - Task 3 (Hard): Multi-issue detection, edit, and compliance

Example:
    >>> from tasks import create_task
    >>> task = create_task("easy_decision")
    >>> task.setup(content="Hello!", is_compliant=True)
    >>> result = task.grade(action_taken="approve")
    >>> print(result["score"])  # 1.0
"""

from typing import Any, Optional

from tasks.task_1_easy_decision import EasyDecisionTask, TaskConfig as EasyConfig
from tasks.task_2_medium_edit import MediumEditTask, TaskConfig as MediumConfig
from tasks.task_3_hard_complex import HardComplexTask, TaskConfig as HardConfig


# Task registry
TASK_REGISTRY = {
    "easy_decision": EasyDecisionTask,
    "medium_edit": MediumEditTask,
    "hard_complex": HardComplexTask,
}


def create_task(task_name: str, **kwargs: Any) -> Any:
    """
    Factory function to create a task instance.

    Args:
        task_name: Name of the task ("easy_decision", "medium_edit", "hard_complex")
        **kwargs: Additional arguments passed to task constructor

    Returns:
        Configured task instance

    Raises:
        ValueError: If task_name is not recognized

    Example:
        >>> task = create_task("easy_decision")
        >>> task = create_task("medium_edit", max_steps=5)
    """
    if task_name not in TASK_REGISTRY:
        available = ", ".join(TASK_REGISTRY.keys())
        raise ValueError(f"Unknown task '{task_name}'. Available: {available}")

    task_class = TASK_REGISTRY[task_name]
    return task_class(**kwargs)


def get_task_info(task_name: Optional[str] = None) -> dict[str, Any]:
    """
    Get information about available tasks.

    Args:
        task_name: Optional specific task name, or None for all tasks

    Returns:
        Dictionary with task information

    Example:
        >>> info = get_task_info("easy_decision")
        >>> print(info["difficulty"], info["description"])
    """
    task_info = {
        "easy_decision": {
            "difficulty": "easy",
            "description": "Correctly approve or reject clearly toxic/unsafe content",
            "max_steps": 2,
            "scoring": "Binary: 1.0 for correct, 0.0 for wrong",
            "skills": ["binary_classification", "policy_application"]
        },
        "medium_edit": {
            "difficulty": "medium",
            "description": "Edit non-compliant content to make it compliant",
            "max_steps": 3,
            "scoring": "Continuous 0.0-1.0 based on improvement and compliance",
            "skills": ["edit_decision", "content_transformation", "compliance_achievement"]
        },
        "hard_complex": {
            "difficulty": "hard",
            "description": "Resolve complex multi-issue content violations",
            "max_steps": 5,
            "scoring": "Weighted: detection (30%) + edit (40%) + compliance (30%)",
            "skills": ["violation_detection", "multi_issue_resolution", "strategic_editing"]
        }
    }

    if task_name:
        return task_info.get(task_name, {})
    return task_info


def list_tasks() -> list[str]:
    """
    List all available task names.

    Returns:
        List of task names

    Example:
        >>> tasks = list_tasks()
        >>> print(tasks)
        ['easy_decision', 'medium_edit', 'hard_complex']
    """
    return list(TASK_REGISTRY.keys())


__all__ = [
    "create_task",
    "get_task_info",
    "list_tasks",
    "EasyDecisionTask",
    "MediumEditTask",
    "HardComplexTask",
    "EasyConfig",
    "MediumConfig",
    "HardConfig",
]
