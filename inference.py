"""
Inference Script — Content Compliance RL Environment
=====================================================
Mandatory environment variables:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

STDOUT FORMAT:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<r> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<s> rewards=<r1,r2,...>

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
from client import ContentComplianceEnvClient
from models import ContentAction

# ── Mandatory config ──────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
ENV_URL      = os.getenv("ENV_URL", "ws://localhost:7860")

if not HF_TOKEN:
    print("[ERROR] HF_TOKEN or OPENAI_API_KEY must be set.", flush=True)
    sys.exit(1)

BENCHMARK = "content_compliance_env"
TASK_NAME = "content_compliance"

# ── Task descriptions ─────────────────────────────────────────────────────────

TASK_DESCRIPTIONS = {
    1: (
        "STEP 1 — VIOLATION DETECTION\n"
        "Analyze the content and detect all policy violations present.\n"
        "Violation types: hate_speech, violence, harassment, adult_content,\n"
        "misinformation, spam, suspicious_link, engagement_bait, illegal_content\n\n"
        'Action format: {"type": "detect_violations", "data": ["violation_1", ...]}'
    ),
    2: (
        "STEP 2 — COMPLIANCE SCORING\n"
        "Based on detected violations, assign a compliance score from 0.0 to 1.0.\n"
        "1.0 = fully clean, 0.0 = severe violations.\n\n"
        'Action format: {"type": "score_compliance", "data": <float 0.0-1.0>}'
    ),
    3: (
        "STEP 3 — MODERATION DECISION\n"
        "Using violations and compliance score, choose the correct action.\n"
        "- approve: content is clean (score >= 0.7, no violations)\n"
        "- reject:  severe violations (hate_speech, violence, illegal_content)\n"
        "- edit:    minor/fixable violations only (spam, suspicious_link, harassment)\n\n"
        'Action format: {"type": "moderate", "data": "approve" | "reject" | "edit"}'
    ),
    4: (
        "STEP 4 — EDIT & RESUBMIT (only if Step 3 was edit)\n"
        "Rewrite the content to remove violations while preserving the core message.\n\n"
        'Action format: {"type": "submit_edit", "data": "<rewritten content>"}'
    ),
}

# ── Log helpers ───────────────────────────────────────────────────────────────

def _fmt(v: float) -> str:
    if v == int(v):
        return str(int(v))
    return f"{v:.3f}".rstrip('0')


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={action} reward={_fmt(reward)}"
        f" done={str(done).lower()} error={error if error else 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    score = max(0.01, min(0.99, score))
    rewards_str = ",".join(_fmt(max(0.01, min(0.99, r))) for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={_fmt(score)} rewards={rewards_str}",
        flush=True,
    )

# ── LLM helpers ───────────────────────────────────────────────────────────────

def llm_decide(client: OpenAI, content: str, violations: list, step_desc: str) -> str:
    """Ask LLM to make moderation decision using step description."""
    try:
        viol_str = ", ".join(violations) if violations else "none"
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": (
                f"{step_desc}\n\n"
                f"Content: {content}\n"
                f"Detected violations: {viol_str}\n\n"
                f"Respond with ONLY one word: approve, reject, or edit."
            )}],
            max_tokens=5,
            temperature=0.0,
        )
        action = response.choices[0].message.content.strip().lower().split()[0]
        return action if action in {"approve", "reject", "edit"} else ("reject" if violations else "approve")
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        return "reject" if violations else "approve"


def llm_rewrite(client: OpenAI, content: str, violations: list, step_desc: str) -> str:
    """Ask LLM to rewrite content to fix violations."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": (
                f"{step_desc}\n\n"
                f"Violations to fix: {', '.join(violations)}\n"
                f"Original: {content}\n"
                f"Return only the rewritten text, nothing else."
            )}],
            max_tokens=150,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        print(f"[DEBUG] LLM rewrite failed: {exc}", flush=True)
        return content

# ── Episode runner ────────────────────────────────────────────────────────────

async def run_episode(
    llm: OpenAI,
    env: ContentComplianceEnvClient,
    task_id: str,
    difficulty: str,
) -> tuple[List[float], float]:
    """Run one 4-step episode. Returns (rewards, avg_score)."""
    rewards: List[float] = []

    # Reset environment
    result = await env.reset()
    obs = result.observation

    for step_num in range(1, 5):
        content    = obs.content
        violations = obs.violations
        score      = obs.compliance_score
        done       = obs.done

        # Determine action for this step
        if step_num == 1:
            action_type = "detect_violations"
        elif step_num == 2:
            action_type = "score_compliance"
        elif step_num == 3:
            action_type = llm_decide(llm, content, violations, TASK_DESCRIPTIONS[3])
        else:
            if violations:
                action_type = "submit_edit"
            else:
                action_type = f"confirm_{obs.stage}"

        # Build edited content for step 4 if needed
        edited = None
        if step_num == 4 and action_type == "submit_edit":
            edited = llm_rewrite(llm, content, violations, TASK_DESCRIPTIONS[4])

        action = ContentAction(action_type=action_type, edited_content=edited)
        result = await env.step(action)
        obs    = result.observation
        reward = float(obs.reward) if obs.reward is not None else 0.05
        reward = max(0.01, min(0.99, reward))

        rewards.append(reward)
        log_step(step=step_num, action=action_type, reward=reward, done=obs.done, error=None)
        time.sleep(0.2)

        if obs.done:
            break

    avg = round(sum(rewards) / len(rewards), 3) if rewards else 0.05
    return rewards, avg

# ── Main ──────────────────────────────────────────────────────────────────────

EPISODES = [
    ("easy",   "easy_001"),
    ("easy",   "easy_002"),
    ("easy",   "easy_003"),
    ("medium", "medium_001"),
    ("medium", "medium_002"),
    ("hard",   "hard_001"),
    ("hard",   "hard_002"),
    ("hard",   "hard_003"),
]


async def main() -> None:
    llm = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    all_scores: List[float] = []

    async with ContentComplianceEnvClient(base_url=ENV_URL) as env:
        for difficulty, task_id in EPISODES:
            log_start(task=f"{TASK_NAME}_{task_id}", env=BENCHMARK, model=MODEL_NAME)
            try:
                rewards, score = await run_episode(llm, env, task_id, difficulty)
                success = score >= 0.5
                all_scores.append(score)
                log_end(success=success, steps=len(rewards), score=score, rewards=rewards)
            except Exception as exc:
                print(f"[DEBUG] Episode {task_id} failed: {exc}", flush=True)
                log_end(success=False, steps=0, score=0.05, rewards=[0.05])

    overall = round(sum(all_scores) / len(all_scores), 3) if all_scores else 0.0
    overall = max(0.0, min(1.0, overall))
    print(f"\nBaseline average score: {_fmt(overall)} / 1.0", flush=True)

    with open("inference_results.json", "w") as f:
        json.dump({"model": MODEL_NAME, "average_score": overall, "task_scores": all_scores}, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
