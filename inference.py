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

def _clamp(v: float) -> float:
    """Clamp to strictly (0, 1) — validator rejects exact 0.0 or 1.0."""
    return round(max(0.01, min(0.99, v)), 3)


def _fmt(v: float) -> str:
    s = f"{v:.3f}".rstrip('0').rstrip('.')
    return s


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={action} reward={_fmt(reward)}"
        f" done={str(done).lower()} error={error if error else 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    score = _clamp(score)
    rewards_str = ",".join(_fmt(_clamp(r)) for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={_fmt(score)} rewards={rewards_str}",
        flush=True,
    )


# ── LLM call ─────────────────────────────────────────────────────────────────

def get_moderation_action(client: OpenAI, content: str, violations: list, step: int) -> str:
    try:
        viol_str = ", ".join(violations) if violations else "none"
        prompt = (
            f"You are a content moderator. Respond with ONLY one word: 'approve', 'reject', or 'edit'.\n\n"
            f"Content: {content}\n"
            f"Detected violations: {viol_str}\n\n"
            f"Rules:\n"
            f"- approve: content is clean, no violations\n"
            f"- reject: severe violations (hate_speech, violence, illegal_content)\n"
            f"- edit: minor/fixable violations (spam, suspicious_link, harassment, misinformation)\n\n"
            f"One word only:"
        )
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0.0,
        )
        action = response.choices[0].message.content.strip().lower().split()[0]
        if action not in {"approve", "reject", "edit"}:
            action = "reject" if violations else "approve"
        return action
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        return "reject" if violations else "approve"


def llm_evaluate(client: OpenAI, content: str) -> tuple[list, float]:
    """Use OpenAI to evaluate content compliance. Returns (violations, score)."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{
                "role": "user",
                "content": (
                    f"Evaluate this content for policy violations. "
                    f"Return JSON only: {{\"violations\": [list], \"score\": 0.0-1.0}}\n"
                    f"Violation types: hate_speech, violence, harassment, misinformation, spam, "
                    f"suspicious_link, engagement_bait, adult_content, illegal_content\n"
                    f"Score: 1.0=fully clean, 0.0=severe violations\n\n"
                    f"Content: {content}"
                )
            }],
            max_tokens=100,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        import json as _json
        data = _json.loads(response.choices[0].message.content)
        violations = [v for v in data.get("violations", []) if isinstance(v, str)]
        score = float(data.get("score", 0.5))
        score = round(max(0.0, min(1.0, score)), 3)
        return violations, score
    except Exception:
        # Fall back to rule-based
        return rule_evaluate(content)

# Violations grouped by severity
SEVERE   = {"hate_speech", "violence", "illegal_content", "adult_content"}
MODERATE = {"harassment", "misinformation"}
MINOR    = {"spam", "suspicious_link", "engagement_bait"}

PATTERNS = {
    "hate_speech":    ["hate", "should all die", "kill all", "i hate those"],
    "violence":       ["kill", "attack", "hurt", "weapon"],
    "harassment":     ["idiot", "stupid", "pathetic", "worthless", "ignorant", "delusional"],
    "misinformation": ["fake news", "conspiracy", "they don't want you to know", "wake up sheeple"],
    "spam":           ["click here", "buy now", "!!!", "limited time", "free prizes", "free tips"],
    "suspicious_link":["bit.ly", "tinyurl", "linktr.ee"],
    "engagement_bait":["tag someone", "share if", "🔥", "💰"],
    "adult_content":  ["explicit", "nsfw", "porn"],
    "illegal_content":["illegal", "drugs"],
}


def rule_evaluate(content: str) -> tuple[list, float]:
    c = content.lower()
    violations = [v for v, kws in PATTERNS.items() if any(k in c for k in kws)]
    if not violations:
        return [], 0.9
    severe_count   = sum(1 for v in violations if v in SEVERE)
    moderate_count = sum(1 for v in violations if v in MODERATE)
    minor_count    = sum(1 for v in violations if v in MINOR)
    score = round(max(0.05, 1.0 - severe_count * 0.35 - moderate_count * 0.2 - minor_count * 0.1), 2)
    return violations, score


def calc_reward(action: str, violations: list, score: float, step: int) -> float:
    """
    Varied reward based on action correctness, violation severity, and step efficiency.
    Returns values across the full 0.0–1.0 range.
    """
    is_compliant = score >= 0.5 and not violations
    severe_count   = sum(1 for v in violations if v in SEVERE)
    moderate_count = sum(1 for v in violations if v in MODERATE)
    minor_count    = sum(1 for v in violations if v in MINOR)
    step_penalty   = (step - 1) * 0.05  # -0.05 per extra step

    if action == "approve":
        if is_compliant:
            r = 0.95 - step_penalty          # correct approve, never reaches 1.0
        else:
            r = max(0.02, 0.3 - severe_count * 0.15 - moderate_count * 0.05)

    elif action == "reject":
        if not is_compliant:
            r = min(0.95, 0.6 + severe_count * 0.15 + moderate_count * 0.05) - step_penalty
        else:
            r = 0.15                          # false positive

    elif action == "edit":
        if not is_compliant:
            r = min(0.6, 0.25 + minor_count * 0.1 + moderate_count * 0.08) - step_penalty
        else:
            r = 0.08                          # unnecessary edit

    else:
        r = 0.05

    return round(max(0.01, min(0.99, r)), 3)  # strictly (0, 1)


# ── Task cases ────────────────────────────────────────────────────────────────

EASY_CASES = [
    # clean → approve → high reward
    {"id": "easy_001", "content": "Hello! Welcome to our community. Looking forward to connecting with everyone here!"},
    # obvious hate → reject → high reward
    {"id": "easy_002", "content": "I hate those people. They should all die!"},
    # clean → approve → high reward
    {"id": "easy_003", "content": "Great discussion everyone. Thanks for sharing your thoughts respectfully."},
]

MEDIUM_CASES = [
    # spam + suspicious link → edit → approve (varied rewards across steps)
    {"id": "medium_001", "content": "Check out this amazing opportunity! bit.ly/opportunity — free tips inside!"},
    # harassment + misinformation → edit → approve
    {"id": "medium_002", "content": "Some people are just too ignorant to understand. Wake up sheeple, do your research!"},
]

HARD_CASES = [
    # hate + misinformation + suspicious link + harassment → reject (multi-violation, lower reward)
    {"id": "hard_001", "content": "They don't want you to know the truth! Wake up sheeple! Visit bit.ly/truth. Those people are delusional idiots."},
    # engagement bait + spam + suspicious link → edit → approve
    {"id": "hard_002", "content": "Tag someone who needs this!!! Free tips inside — visit linktr.ee/tips for exclusive access."},
    # clean formal content → approve
    {"id": "hard_003", "content": "Our platform prohibits harassment. Please review the community guidelines before posting."},
]


# ── Task runner ───────────────────────────────────────────────────────────────

async def run_task(client: OpenAI, task_id: str, content: str, max_steps: int) -> tuple[list, float]:
    rewards: List[float] = []
    # Use LLM to evaluate initial content — genuinely dynamic
    violations, score = llm_evaluate(client, content)
    done = False
    step = 0
    edited_once = False

    while not done and step < max_steps:
        step += 1

        # After one edit, force terminal decision based on re-evaluated violations
        if edited_once:
            action = "approve" if not violations else "reject"
        else:
            action = get_moderation_action(client, content, violations, step)

        if action == "edit" and not edited_once:
            reward = calc_reward(action, violations, score, step)
            # Ask LLM to produce a cleaned version of the content
            try:
                edit_resp = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{
                        "role": "user",
                        "content": (
                            f"Rewrite this content to remove policy violations while keeping the core message.\n"
                            f"Violations to fix: {', '.join(violations)}\n"
                            f"Original: {content}\n"
                            f"Return only the rewritten text, nothing else."
                        )
                    }],
                    max_tokens=150,
                    temperature=0.3,
                )
                content = edit_resp.choices[0].message.content.strip()
            except Exception:
                pass  # keep original content if edit fails
            # Re-evaluate the edited content with LLM
            violations, score = llm_evaluate(client, content)
            edited_once = True
            done = False
        else:
            reward = calc_reward(action, violations, score, step)
            done = True

        rewards.append(reward)
        log_step(step=step, action=action, reward=reward, done=done, error=None)
        time.sleep(0.2)

    final_score = _clamp(sum(rewards) / len(rewards)) if rewards else 0.05
    return rewards, final_score


# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    all_scores: List[float] = []

    task_groups = [
        ("easy",   EASY_CASES,   2),
        ("medium", MEDIUM_CASES, 3),
        ("hard",   HARD_CASES,   5),
    ]

    for _difficulty, cases, max_steps in task_groups:
        for case in cases:
            task_id = case["id"]
            content = case["content"]

            log_start(task=f"{TASK_NAME}_{task_id}", env=BENCHMARK, model=MODEL_NAME)

            try:
                rewards, score = await run_task(client, task_id, content, max_steps)
                success = score >= 0.5
                all_scores.append(score)
                log_end(success=success, steps=len(rewards), score=score, rewards=rewards)
            except Exception as exc:
                print(f"[DEBUG] Task {task_id} failed: {exc}", flush=True)
                log_end(success=False, steps=0, score=0.05, rewards=[0.05])

    overall = round(sum(all_scores) / len(all_scores), 3) if all_scores else 0.0
    overall = max(0.0, min(1.0, overall))
    print(f"\nBaseline average score: {_fmt(overall)} / 1.0", flush=True)

    with open("inference_results.json", "w") as f:
        json.dump({"model": MODEL_NAME, "average_score": overall, "task_scores": all_scores}, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
