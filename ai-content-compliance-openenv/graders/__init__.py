"""
OpenEnv Task Graders.

Provides deterministic grading functions for content compliance tasks.

Example:
    >>> from graders import grade_task, GradeResult
    >>> result = grade_task(
    ...     task_name="easy_decision",
    ...     ground_truth={"is_compliant": True},
    ...     agent_output={"action_taken": "approve"}
    ... )
    >>> print(result.score)  # 1.0
"""

from graders.grader import (
    GradeResult,
    TaskGrader,
    grade_task,
    normalize_score,
    batch_grade,
    compute_statistics,
)
from graders.violation_detector import ViolationDetector

__all__ = [
    "GradeResult",
    "TaskGrader",
    "grade_task",
    "normalize_score",
    "batch_grade",
    "compute_statistics",
    "ViolationDetector",
]
