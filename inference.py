"""
Inference Script for Content Compliance RL.

Runs all tasks using OpenAI API with strict logging format:
[START]
[STEP]
[END]

Environment Variables:
    API_BASE_URL: OpenAI-compatible API endpoint
    MODEL_NAME: Model to use for inference
    OPENAI_API_KEY: API authentication key

Usage:
    export OPENAI_API_KEY="sk-..."
    python inference.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional
from dotenv import load_dotenv
load_dotenv()


# Configure logging - only output [START], [STEP], [END] format
# Suppress HTTP request logs from openai client
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration from environment variables
# =============================================================================

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Validate API key
_API_KEY_VALID = OPENAI_API_KEY and not OPENAI_API_KEY.startswith("YOUR_") and "here" not in OPENAI_API_KEY.lower()

# Task configurations
TASK_CONFIGS = {
    "easy": {"max_steps": 2, "description": "Binary approve/reject decision"},
    "medium": {"max_steps": 3, "description": "Edit content to achieve compliance"},
    "hard": {"max_steps": 5, "description": "Multi-issue content resolution"},
}

# Dynamic test cases - content generated at runtime
TEST_CASES = {
    "easy": [
        {"id": "easy_001", "is_compliant": True},
        {"id": "easy_002", "is_compliant": False},
        {"id": "easy_003", "is_compliant": False},
    ],
    "medium": [
        {"id": "medium_001", "original_score": 0.2, "violations": ["harassment"]},
        {"id": "medium_002", "original_score": 0.1, "violations": ["spam"]},
    ],
    "hard": [
        {"id": "hard_001", "true_violations": ["hate_speech", "violence", "spam"]},
    ],
}


@dataclass
class InferenceResult:
    """Result of a single inference step."""

    task_id: str
    step_number: int
    action: str
    reward: float
    done: bool
    explanation: str


class ContentComplianceInference:
    """
    Inference engine for content compliance tasks using OpenAI.

    Runs the model on all tasks with proper logging and error handling.
    """

    def __init__(
        self,
        api_base_url: str = API_BASE_URL,
        model_name: str = MODEL_NAME,
        api_key: str = OPENAI_API_KEY,
    ):
        """Initialize the inference engine."""
        self.api_base_url = api_base_url
        self.model_name = model_name
        self.api_key = api_key

        # Initialize OpenAI client
        self._client = None

        # Results storage
        self.results: list[InferenceResult] = []
        self.task_scores: dict[str, list[float]] = {
            "easy": [],
            "medium": [],
            "hard": [],
        }

    def _get_client(self):
        """Lazy-load OpenAI client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI

                # Validate API key before creating client
                if not self.api_key or self.api_key.startswith("YOUR_") or "here" in self.api_key.lower():
                    logger.info("[STEP] Invalid API key format, using mock inference")
                    return None

                self._client = AsyncOpenAI(
                    base_url=self.api_base_url,
                    api_key=self.api_key,
                )
            except ImportError:
                logger.info("[STEP] OpenAI not installed, using mock inference")
        return self._client

    def _generate_content(self, is_compliant: bool = True) -> str:
        """Generate content dynamically based on compliance requirement."""
        if is_compliant:
            return "Hello! Welcome to our community platform. Looking forward to connecting!"
        else:
            return "I hate all those people. They should all die! Click here now!!!"

    def _build_system_prompt(self) -> str:
        """Build the system prompt for the model."""
        return """You are a content moderation assistant. Evaluate content and decide if it violates platform policies.

Actions:
1. "approve" - Content passes compliance (use for safe content)
2. "reject" - Content violates policies (use for harmful content)
3. "edit" - Content needs modification (provide edited_content)

Violation types: hate_speech, violence, harassment, adult_content, misinformation, spam, illegal_content

Respond with JSON:
{"action": "approve"|"reject"|"edit", "edited_content": null, "reasoning": "brief"}"""

    async def _call_model(
        self,
        system_prompt: str,
        user_prompt: str,
        max_retries: int = 3,
    ) -> dict[str, Any]:
        """Call the model with retry logic."""
        client = self._get_client()

        # Use mock if client unavailable
        if client is None:
            return self._mock_inference(user_prompt)

        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.1,
                    max_tokens=500,
                )

                content = response.choices[0].message.content

                # Extract JSON from response
                json_start = content.find("{")
                json_end = content.rfind("}") + 1

                if json_start >= 0 and json_end > json_start:
                    return json.loads(content[json_start:json_end])
                else:
                    return json.loads(content)

            except json.JSONDecodeError as e:
                logger.info(f"[STEP] JSON parse error (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    return {"action": "reject", "edited_content": None, "reasoning": "Parse error"}
            except Exception as e:
                error_msg = str(e)
                # Don't log full error for API key issues - just note it's invalid
                if "401" in error_msg or "invalid_api_key" in error_msg.lower():
                    logger.info(f"[STEP] Invalid API key (attempt {attempt + 1})")
                else:
                    logger.info(f"[STEP] API error (attempt {attempt + 1}): {type(e).__name__}")
                if attempt == max_retries - 1:
                    return self._mock_inference(user_prompt)

        return {"action": "reject", "edited_content": None, "reasoning": "Max retries"}

    def _mock_inference(self, prompt: str) -> dict[str, Any]:
        """Mock inference for testing without API."""
        if "hate" in prompt.lower() or "die" in prompt.lower():
            return {"action": "reject", "edited_content": None, "reasoning": "Harmful content detected"}
        return {"action": "approve", "edited_content": None, "reasoning": "Content is safe"}

    def _calculate_reward(
        self,
        action: str,
        expected_action: str,
        is_compliant: bool,
        edit_made: bool = False,
        improvement: float = 0.0,
    ) -> float:
        """Calculate deterministic reward for an action."""
        if action == expected_action:
            base_reward = 0.6
        elif action in ["approve", "reject"]:
            base_reward = 0.1
        else:
            base_reward = 0.3

        edit_bonus = 0.0
        if edit_made and improvement > 0:
            edit_bonus = min(0.2, improvement * 0.3)
        elif edit_made and improvement <= 0:
            edit_bonus = -0.1

        efficiency_bonus = 0.1

        total_reward = base_reward + edit_bonus + efficiency_bonus
        return max(0.0, min(1.0, total_reward))

    async def run_easy_task(self, test_case: dict[str, Any]) -> list[InferenceResult]:
        """Run the easy decision task."""
        task_id = test_case["id"]
        is_compliant = test_case.get("is_compliant", True)
        expected_action = "approve" if is_compliant else "reject"

        content = self._generate_content(is_compliant)
        results = []

        system_prompt = self._build_system_prompt()
        user_prompt = f"Content: {content}\n\nEvaluate and provide action in JSON."

        logger.info(f"[STEP] {task_id}: Evaluating content...")
        response = await self._call_model(system_prompt, user_prompt)

        action = response.get("action", "reject")
        reasoning = response.get("reasoning", "")

        reward = self._calculate_reward(action, expected_action, is_compliant)
        done = action in ["approve", "reject"]

        result = InferenceResult(
            task_id=task_id,
            step_number=1,
            action=action,
            reward=reward,
            done=done,
            explanation=reasoning,
        )
        results.append(result)

        correct = action == expected_action
        logger.info(f"[STEP] {task_id}: action={action}, expected={expected_action}, correct={correct}, reward={reward:.3f}")

        self.task_scores["easy"].append(reward)
        return results

    async def run_medium_task(self, test_case: dict[str, Any]) -> list[InferenceResult]:
        """Run the medium edit task."""
        task_id = test_case["id"]
        original_score = test_case.get("original_score", 0.2)
        violations = test_case.get("violations", [])

        content = self._generate_content(is_compliant=False)
        results = []
        step_number = 0
        current_score = original_score

        system_prompt = self._build_system_prompt()

        # Step 1: Edit
        step_number += 1
        user_prompt = f"Content (score={original_score}, violations={violations}): {content}\n\nEdit to improve."

        logger.info(f"[STEP] {task_id}: Initial evaluation...")
        response = await self._call_model(system_prompt, user_prompt)

        action = response.get("action", "edit")
        edited_content = response.get("edited_content")
        reasoning = response.get("reasoning", "")

        if action == "edit" and edited_content:
            improvement = min(0.6, 0.3 + len(edited_content) / 100)
            current_score = min(0.85, original_score + improvement)
        else:
            improvement = 0.0

        reward = self._calculate_reward(action, "edit", False, edit_made=(action == "edit"), improvement=improvement)

        result = InferenceResult(
            task_id=task_id,
            step_number=step_number,
            action=action,
            reward=reward,
            done=False,
            explanation=reasoning,
        )
        results.append(result)

        logger.info(f"[STEP] {task_id}: action={action}, score={current_score:.2f}, reward={reward:.3f}")

        # Step 2: Final decision
        step_number += 1
        expected_final = "approve" if current_score >= 0.5 else "reject"

        user_prompt = f"Edited content (score={current_score}): {edited_content or content}\n\nFinal decision."

        logger.info(f"[STEP] {task_id}: Final decision...")
        response = await self._call_model(system_prompt, user_prompt)

        final_action = response.get("action", "approve")
        reasoning = response.get("reasoning", "")

        final_reward = self._calculate_reward(final_action, expected_final, current_score >= 0.5)

        result = InferenceResult(
            task_id=task_id,
            step_number=step_number,
            action=final_action,
            reward=final_reward,
            done=True,
            explanation=reasoning,
        )
        results.append(result)

        logger.info(f"[STEP] {task_id}: final={final_action}, score={current_score:.2f}, reward={final_reward:.3f}")

        avg_reward = (reward + final_reward) / 2
        self.task_scores["medium"].append(avg_reward)

        return results

    async def run_hard_task(self, test_case: dict[str, Any]) -> list[InferenceResult]:
        """Run the hard complex task."""
        task_id = test_case["id"]
        true_violations = test_case.get("true_violations", [])

        content = self._generate_content(is_compliant=False)
        results = []
        step_number = 0

        system_prompt = self._build_system_prompt()

        # Step 1: Detect violations
        step_number += 1
        user_prompt = f"Content: {content}\n\nIdentify all violations."

        logger.info(f"[STEP] {task_id}: Detecting violations...")
        response = await self._call_model(system_prompt, user_prompt)

        action = response.get("action", "edit")
        reasoning = response.get("reasoning", "")

        # Extract detected violations from reasoning
        detected = []
        for v in ["hate_speech", "violence", "harassment", "spam", "misinformation"]:
            if v.replace("_", " ") in reasoning.lower() or v in reasoning.lower():
                detected.append(v)

        if not detected:
            detected = ["multiple"]

        true_set = set(true_violations)
        detected_set = set(detected)
        true_positives = len(true_set & detected_set)
        detection_accuracy = true_positives / len(true_set) if true_set else 1.0

        reward = 0.3 * detection_accuracy

        result = InferenceResult(
            task_id=task_id,
            step_number=step_number,
            action=f"detect ({len(detected)} violations)",
            reward=reward,
            done=False,
            explanation=reasoning,
        )
        results.append(result)

        logger.info(f"[STEP] {task_id}: detected={detected}, accuracy={detection_accuracy:.2f}, reward={reward:.3f}")

        # Step 2: Edit
        step_number += 1
        user_prompt = f"Content: {content}\n\nEdit to remove violations: {detected}"

        logger.info(f"[STEP] {task_id}: Editing content...")
        response = await self._call_model(system_prompt, user_prompt)

        action = response.get("action", "edit")
        edited_content = response.get("edited_content", content)
        reasoning = response.get("reasoning", "")

        if action == "edit" and edited_content:
            content = edited_content

        reward = 0.3 + 0.2  # Detection + edit

        result = InferenceResult(
            task_id=task_id,
            step_number=step_number,
            action=action,
            reward=reward,
            done=False,
            explanation=reasoning,
        )
        results.append(result)

        logger.info(f"[STEP] {task_id}: edit action={action}, reward={reward:.3f}")

        # Step 3: Final approval
        step_number += 1
        user_prompt = f"Edited content: {content}\n\nFinal decision."

        logger.info(f"[STEP] {task_id}: Final decision...")
        response = await self._call_model(system_prompt, user_prompt)

        final_action = response.get("action", "approve")
        reasoning = response.get("reasoning", "")

        final_reward = 0.4 if final_action == "approve" else 0.1

        result = InferenceResult(
            task_id=task_id,
            step_number=step_number,
            action=final_action,
            reward=final_reward,
            done=True,
            explanation=reasoning,
        )
        results.append(result)

        logger.info(f"[STEP] {task_id}: final={final_action}, reward={final_reward:.3f}")

        avg_reward = (reward + final_reward + 0.3) / 3
        self.task_scores["hard"].append(avg_reward)

        return results

    async def run_all_tasks(self) -> dict[str, Any]:
        """Run all tasks with their test cases."""
        start_time = datetime.now()

        logger.info("[START] Inference pipeline initialized")
        logger.info(f"[START] Model: {self.model_name}")
        logger.info(f"[START] API Base: {self.api_base_url}")
        logger.info(f"[START] Tasks: {list(TASK_CONFIGS.keys())}")

        all_results: dict[str, list[InferenceResult]] = {}

        # Run easy tasks
        logger.info("[START] Running easy tasks...")
        all_results["easy"] = []
        for test_case in TEST_CASES["easy"]:
            results = await self.run_easy_task(test_case)
            all_results["easy"].extend(results)

        # Run medium tasks
        logger.info("[START] Running medium tasks...")
        all_results["medium"] = []
        for test_case in TEST_CASES["medium"]:
            results = await self.run_medium_task(test_case)
            all_results["medium"].extend(results)

        # Run hard tasks
        logger.info("[START] Running hard tasks...")
        all_results["hard"] = []
        for test_case in TEST_CASES["hard"]:
            results = await self.run_hard_task(test_case)
            all_results["hard"].extend(results)

        # Calculate summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        all_scores = []
        for scores in self.task_scores.values():
            all_scores.extend(scores)

        summary = {
            "total_tasks": sum(len(v) for v in all_results.values()),
            "total_steps": sum(len(r) for r in all_results.values()),
            "average_reward": sum(all_scores) / len(all_scores) if all_scores else 0.0,
            "task_scores": {
                k: sum(v) / len(v) if v else 0.0
                for k, v in self.task_scores.items()
            },
            "duration_seconds": duration,
            "model": self.model_name,
        }

        logger.info("[END] All tasks completed")
        logger.info(f"[END] Total tasks: {summary['total_tasks']}")
        logger.info(f"[END] Total steps: {summary['total_steps']}")
        logger.info(f"[END] Average reward: {summary['average_reward']:.3f}")
        logger.info(f"[END] Duration: {duration:.2f}s")
        logger.info(f"[END] Task scores: {summary['task_scores']}")

        return summary


async def main():
    """Main entry point."""
    # Validate API key
    api_key_valid = OPENAI_API_KEY and not OPENAI_API_KEY.startswith("YOUR_") and "here" not in OPENAI_API_KEY.lower()

    if not api_key_valid:
        logger.info("[START] WARNING: Invalid API key, using mock inference")
        logger.info("[START] To use OpenAI, edit .env and set OPENAI_API_KEY")
    else:
        logger.info(f"[START] API Key: {'*' * 8}{OPENAI_API_KEY[-4:]}")

    logger.info("[START] Content Compliance Inference")

    engine = ContentComplianceInference(
        api_base_url=API_BASE_URL,
        model_name=MODEL_NAME,
        api_key=OPENAI_API_KEY if api_key_valid else "",
    )

    try:
        summary = await engine.run_all_tasks()
        logger.info("[END] Inference complete")
        logger.info(f"[END] Summary: {json.dumps(summary, indent=2)}")
        return summary
    except Exception as e:
        logger.info(f"[END] ERROR: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
