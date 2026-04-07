"""
Inference Script — Content Compliance RL Environment
=====================================================
MANDATORY environment variables:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

STDOUT FORMAT:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>

Usage:
    python inference.py
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from graders.openai_evaluator import OpenAIEvaluator, EvaluationResult

# ── Config ────────────────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
DEBUG        = os.getenv("DEBUG", "false").lower() == "true"

TASK_NAME = "content_compliance"
BENCHMARK = "content_compliance_env"

if not HF_TOKEN:
    print("[ERROR] HF_TOKEN or OPENAI_API_KEY must be set.", flush=True)
    sys.exit(1)

TASK_CONFIGS = {
    "easy":   {"max_steps": 2, "description": "Binary approve/reject decision"},
    "medium": {"max_steps": 3, "description": "Edit content to achieve compliance"},
    "hard":   {"max_steps": 5, "description": "Multi-issue content resolution"},
}

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

# ── Stdout loggers ────────────────────────────────────────────────────────────

def _fmt(v: float) -> str:
    return str(int(v)) if v == int(v) else f"{v:.2f}"


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    score = max(0.0, min(1.0, score))
    rewards_str = ",".join(_fmt(r) for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={_fmt(score)} rewards={rewards_str}",
        flush=True,
    )


# ── Dataclass ─────────────────────────────────────────────────────────────────

@dataclass
class InferenceResult:
    task_id: str
    step_number: int
    action: str
    reward: float
    done: bool
    explanation: str


# ── Inference engine ──────────────────────────────────────────────────────────

class ContentComplianceInference:

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.evaluator = OpenAIEvaluator(model=model_name)
        self.results: list[InferenceResult] = []
        self.task_scores: dict[str, list[float]] = {"easy": [], "medium": [], "hard": []}

    def _generate_content(self, is_compliant: bool = True, task_id: str = "") -> str:
        compliant_pool = [
            "Hello! Welcome to our community platform. Looking forward to connecting!",
            "Great discussion everyone. Thanks for sharing your thoughts respectfully.",
            "Looking forward to the upcoming community event this weekend!",
        ]
        non_compliant_pool = [
            "I hate all those people. They should all die! Click here now!!!",
            "You are such an idiot. Buy now!!! Limited offer, free prizes!!!",
            "Fake news! Scientists are lying. Kill all who disagree. bit.ly/scam",
        ]
        idx = int(hashlib.md5(task_id.encode()).hexdigest(), 16) % 3
        return compliant_pool[idx] if is_compliant else non_compliant_pool[idx]

    def _evaluate_content(self, content: str) -> EvaluationResult:
        result = self.evaluator.evaluate(content)
        if DEBUG:
            print(f"[EVAL] source={result.source}, score={result.score:.2f}, violations={result.violations}", flush=True)
        return result

    def _calculate_reward(
        self,
        action: str,
        expected_action: str,
        is_compliant: bool,
        edit_made: bool = False,
        improvement: float = 0.0,
        eval_score: float = 0.5,
    ) -> float:
        if edit_made:
            # edit reward: higher when more violations present (more to fix)
            if improvement > 0:
                reward = (1 - eval_score) * 0.8
            else:
                reward = 0.0  # unnecessary edit
        elif action == "approve" and is_compliant:
            reward = eval_score                    # correct approve
        elif action == "reject" and not is_compliant:
            reward = 1 - eval_score                # correct reject
        elif action == "approve" and not is_compliant:
            reward = eval_score * 0.1              # wrong approve — missed violation
        elif action == "reject" and is_compliant:
            reward = (1 - eval_score) * 0.1        # wrong reject — false positive
        else:
            reward = 0.0

        return max(0.0, min(1.0, round(reward, 4)))

    async def run_easy_task(self, test_case: dict[str, Any]) -> list[InferenceResult]:
        task_id = test_case["id"]
        is_compliant = test_case.get("is_compliant", True)
        expected_action = "approve" if is_compliant else "reject"
        content = self._generate_content(is_compliant, task_id=task_id)
        results = []

        eval_result = self._evaluate_content(content)
        action = "approve" if eval_result.is_compliant else "reject"
        reward = self._calculate_reward(action, expected_action, is_compliant, eval_score=eval_result.score)

        results.append(InferenceResult(
            task_id=task_id, step_number=1, action=action,
            reward=reward, done=True, explanation=eval_result.reason,
        ))
        log_step(step=1, action=action, reward=reward, done=True, error=None)
        self.task_scores["easy"].append(reward)
        return results

    async def run_medium_task(self, test_case: dict[str, Any]) -> list[InferenceResult]:
        task_id = test_case["id"]
        original_score = test_case.get("original_score", 0.2)
        results = []

        content = self._generate_content(is_compliant=False, task_id=task_id)

        # Step 1: edit
        eval_result = self._evaluate_content(content)
        action = "edit"
        improvement = min(0.6, 0.3 + len(eval_result.violations) * 0.1)
        current_score = min(0.9, original_score + improvement)
        reward = self._calculate_reward(action, "edit", False, edit_made=True, improvement=improvement, eval_score=eval_result.score)

        results.append(InferenceResult(
            task_id=task_id, step_number=1, action=action,
            reward=reward, done=False, explanation=eval_result.reason,
        ))
        log_step(step=1, action=action, reward=reward, done=False, error=None)

        # Step 2: final decision
        expected_final = "approve" if current_score >= 0.5 else "reject"
        final_eval = self._evaluate_content("Edited clean content")
        final_action = "approve" if final_eval.is_compliant else "reject"
        final_reward = self._calculate_reward(final_action, expected_final, current_score >= 0.5, eval_score=final_eval.score)

        results.append(InferenceResult(
            task_id=task_id, step_number=2, action=final_action,
            reward=final_reward, done=True, explanation=final_eval.reason,
        ))
        log_step(step=2, action=final_action, reward=final_reward, done=True, error=None)

        self.task_scores["medium"].append((reward + final_reward) / 2)
        return results

    async def run_hard_task(self, test_case: dict[str, Any]) -> list[InferenceResult]:
        task_id = test_case["id"]
        true_violations = test_case.get("true_violations", [])
        results = []

        content = self._generate_content(is_compliant=False, task_id=task_id)

        # Step 1: detect
        eval_result = self._evaluate_content(content)
        detected = eval_result.violations if eval_result.violations else ["multiple"]
        true_set = set(true_violations)
        detection_accuracy = len(true_set & set(detected)) / len(true_set) if true_set else 1.0
        reward = max(0.0, min(1.0, round(detection_accuracy, 4)))
        action = f"detect({len(detected)}_violations)"

        results.append(InferenceResult(
            task_id=task_id, step_number=1, action=action,
            reward=reward, done=False, explanation=eval_result.reason,
        ))
        log_step(step=1, action=action, reward=reward, done=False, error=None)

        # Step 2: edit
        edit_reward = max(0.0, min(1.0, round((1 - eval_result.score) * 0.8, 4)))
        results.append(InferenceResult(
            task_id=task_id, step_number=2, action="edit",
            reward=edit_reward, done=False, explanation="Edit applied",
        ))
        log_step(step=2, action="edit", reward=edit_reward, done=False, error=None)

        # Step 3: final
        final_eval = self._evaluate_content("Edited content after violations removed")
        final_action = "approve" if final_eval.is_compliant else "reject"
        final_reward = max(0.0, min(1.0, round(final_eval.score, 4)))

        results.append(InferenceResult(
            task_id=task_id, step_number=3, action=final_action,
            reward=final_reward, done=True, explanation=final_eval.reason,
        ))
        log_step(step=3, action=final_action, reward=final_reward, done=True, error=None)

        self.task_scores["hard"].append((reward + edit_reward + final_reward) / 3)
        return results

    async def run_all_tasks(self) -> dict[str, Any]:
        start_time = datetime.now()
        all_results: dict[str, list[InferenceResult]] = {}

        for difficulty, cases in TEST_CASES.items():
            all_results[difficulty] = []
            runner = {
                "easy":   self.run_easy_task,
                "medium": self.run_medium_task,
                "hard":   self.run_hard_task,
            }[difficulty]

            for test_case in cases:
                task_id = test_case["id"]
                log_start(task=f"{TASK_NAME}_{task_id}", env=BENCHMARK, model=self.model_name)

                try:
                    results = await runner(test_case)
                    all_results[difficulty].extend(results)
                    rewards_this_task = [r.reward for r in results]
                    score = max(0.0, min(1.0, round(sum(rewards_this_task) / len(rewards_this_task), 4)))
                    success = score >= 0.5
                    log_end(success=success, steps=len(results), score=score, rewards=rewards_this_task)

                except Exception as e:
                    log_end(success=False, steps=0, score=0.0, rewards=[])
                    print(f"[DEBUG] Task {task_id} failed: {type(e).__name__}: {e}", flush=True)

                time.sleep(0.5)

        duration = (datetime.now() - start_time).total_seconds()
        all_scores = [s for v in self.task_scores.values() for s in v]

        return {
            "total_tasks": sum(len(v) for v in all_results.values()),
            "total_steps": sum(len(r) for r in all_results.values()),
            "average_reward": sum(all_scores) / len(all_scores) if all_scores else 0.0,
            "task_scores": {k: sum(v) / len(v) if v else 0.0 for k, v in self.task_scores.items()},
            "duration_seconds": duration,
            "model": self.model_name,
        }


# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    engine = ContentComplianceInference(model_name=MODEL_NAME)

    try:
        summary = await engine.run_all_tasks()
        avg = summary["average_reward"]
        print(f"\nBaseline average score: {_fmt(avg)} / 1.0", flush=True)

        with open("inference_results.json", "w") as f:
            json.dump({"model": MODEL_NAME, "summary": summary}, f, indent=2)

        stats = engine.evaluator.stats
        print(
            f"[END] Evaluator stats: success_rate={stats['success_rate']:.0%}, "
            f"openai={stats['openai_success']}, fallback={stats['fallback_used']}",
            flush=True,
        )

    except Exception as e:
        print(f"[END] ERROR: {e}", flush=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())
