"""
Inference Script — Content Compliance RL Environment
=====================================================
Mandatory environment variables:
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
import json
import os
import sys
import time
from typing import List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from openai import OpenAI

# ── Mandatory config ──────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")

if not HF_TOKEN:
    print("[ERROR] HF_TOKEN or OPENAI_API_KEY must be set.", flush=True)
    sys.exit(1)

BENCHMARK = "content_compliance_env"
TASK_NAME = "content_compliance"

# ── Log helpers ───────────────────────────────────────────────────────────────

def _fmt(v: float) -> str:
    return str(int(v)) if v == int(v) else f"{v:.3f}"


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={action} reward={_fmt(reward)} "
        f"done={str(done).lower()} error={error if error else 'null'}",
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


# ── LLM call ─────────────────────────────────────────────────────────────────

def get_moderation_action(client: OpenAI, content: str, violations: list, step: int) -> str:
    """Ask the LLM to decide: approve / reject / edit."""
    try:
        prompt = (
            f"You are a content moderator. Analyze this content and respond with ONLY one word: "
            f"'approve', 'reject', or 'edit'.\n\n"
            f"Content: {content}\n"
            f"Detected violations: {violations if violations else 'none'}\n"
            f"Step: {step}\n\n"
            f"Rules:\n"
            f"- If content has clear violations (hate, spam, violence) → reject\n"
            f"- If content is borderline or fixable → edit\n"
            f"- If content is clean → approve\n\n"
            f"Respond with exactly one word:"
        )
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.0,
        )
        action = response.choices[0].message.content.strip().lower()
        if action not in {"approve", "reject", "edit"}:
            action = "reject" if violations else "approve"
        return action
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        return "reject" if violations else "approve"


# ── Rule-based evaluator (no extra API calls) ─────────────────────────────────

def rule_evaluate(content: str) -> tuple[list, float]:
    """Fast rule-based violation detection. Returns (violations, score)."""
    c = content.lower()
    patterns = {
        "hate_speech":    ["hate", "should all die", "kill all"],
        "violence":       ["kill", "attack", "hurt", "weapon"],
        "harassment":     ["idiot", "stupid", "pathetic", "worthless", "ignorant", "delusional"],
        "misinformation": ["fake news", "conspiracy", "they don't want you to know", "wake up sheeple"],
        "spam":           ["click here", "buy now", "!!!", "limited time", "free prizes", "visit "],
        "suspicious_link":["bit.ly", "tinyurl", "linktr.ee"],
        "engagement_bait":["tag someone", "share if", "🔥", "💰"],
        "adult_content":  ["explicit", "nsfw", "porn"],
        "illegal_content":["illegal", "drugs"],
    }
    violations = [v for v, kws in patterns.items() if any(k in c for k in kws)]
    score = max(0.0, round(1.0 - len(violations) * 0.25, 2)) if violations else 0.9
    return violations, score


def calc_reward(action: str, violations: list, score: float) -> float:
    is_compliant = score >= 0.5 and not violations
    if action == "approve" and is_compliant:
        return 1.0
    if action == "reject" and not is_compliant:
        return 1.0
    if action == "edit":
        return 0.5
    if action == "approve" and not is_compliant:
        return 0.0   # harmful approval
    if action == "reject" and is_compliant:
        return 0.3   # false positive
    return 0.0


# ── Task runners ──────────────────────────────────────────────────────────────

EASY_CASES = [
    {"id": "easy_001", "content": "Hello! Welcome to our community. Looking forward to connecting!", "is_compliant": True},
    {"id": "easy_002", "content": "I hate those people. They should all die! Click here now!!!", "is_compliant": False},
    {"id": "easy_003", "content": "BUY NOW!!! Limited offer!!! Free prizes! bit.ly/scam", "is_compliant": False},
]

MEDIUM_CASES = [
    {"id": "medium_001", "content": "Some people are just too ignorant to understand complex topics.", "is_compliant": False},
    {"id": "medium_002", "content": "Check out this amazing opportunity! bit.ly/opportunity (Not spam, real!)", "is_compliant": False},
]

HARD_CASES = [
    {"id": "hard_001", "content": "They don't want you to know the truth! Wake up sheeple! Visit bit.ly/truth. Those people are delusional.", "is_compliant": False},
]


async def run_task(client: OpenAI, task_id: str, content: str, max_steps: int) -> tuple[list, float]:
    """Run one task episode. Returns (rewards, score)."""
    rewards: List[float] = []
    violations, score = rule_evaluate(content)
    done = False
    step = 0

    while not done and step < max_steps:
        step += 1
        action = get_moderation_action(client, content, violations, step)

        # edit: simulate content improvement
        if action == "edit":
            content = content  # in real env agent would provide edited content
            violations = [v for v in violations if v not in {"spam", "engagement_bait", "suspicious_link"}]
            score = min(0.9, score + 0.3)
            reward = calc_reward(action, violations, score)
            done = False
        else:
            reward = calc_reward(action, violations, score)
            done = True

        rewards.append(reward)
        log_step(step=step, action=action, reward=reward, done=done, error=None)
        time.sleep(0.3)

    final_score = sum(rewards) / len(rewards) if rewards else 0.0
    return rewards, final_score


# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    all_rewards: List[float] = []
    all_scores: List[float] = []

    task_groups = [
        ("easy",   EASY_CASES,   2),
        ("medium", MEDIUM_CASES, 3),
        ("hard",   HARD_CASES,   5),
    ]

    for difficulty, cases, max_steps in task_groups:
        for case in cases:
            task_id = case["id"]
            content = case["content"]

            log_start(task=f"{TASK_NAME}_{task_id}", env=BENCHMARK, model=MODEL_NAME)

            try:
                rewards, score = await run_task(client, task_id, content, max_steps)
                success = score >= 0.5
                all_rewards.extend(rewards)
                all_scores.append(score)
                log_end(success=success, steps=len(rewards), score=score, rewards=rewards)
            except Exception as exc:
                print(f"[DEBUG] Task {task_id} failed: {exc}", flush=True)
                log_end(success=False, steps=0, score=0.0, rewards=[])

    overall = sum(all_scores) / len(all_scores) if all_scores else 0.0
    overall = max(0.0, min(1.0, overall))
    print(f"\nBaseline average score: {_fmt(overall)} / 1.0", flush=True)

    with open("inference_results.json", "w") as f:
        json.dump({"model": MODEL_NAME, "average_score": overall, "task_scores": all_scores}, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
