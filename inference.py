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

import socket
import threading
import urllib.request

from openai import OpenAI
from client import ContentComplianceEnvClient
from models import ContentAction

# ── Mandatory config ──────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
ENV_URL      = os.getenv("ENV_URL", "ws://localhost:7860")
SERVER_PORT  = 7860

if not HF_TOKEN:
    print("[ERROR] HF_TOKEN or OPENAI_API_KEY must be set.", flush=True)
    sys.exit(1)

# ── Server lifecycle ──────────────────────────────────────────────────────────

def _is_port_open(port: int) -> bool:
    """Check if something is already listening on the port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0


def _start_server_background() -> None:
    """Start uvicorn server in a daemon thread."""
    import uvicorn
    from server.app import app as _app
    uvicorn.run(_app, host="0.0.0.0", port=SERVER_PORT, log_level="error")


def _ensure_server_running() -> None:
    """Start server if not already running, wait until ready."""
    if _is_port_open(SERVER_PORT):
        print(f"[DEBUG] Server already running on port {SERVER_PORT}", flush=True)
        return

    print(f"[DEBUG] Starting server on port {SERVER_PORT}...", flush=True)
    t = threading.Thread(target=_start_server_background, daemon=True)
    t.start()

    # Wait up to 30s for server to be ready
    for _ in range(30):
        time.sleep(1)
        if _is_port_open(SERVER_PORT):
            try:
                urllib.request.urlopen(f"http://localhost:{SERVER_PORT}/health", timeout=2)
                print(f"[DEBUG] Server ready.", flush=True)
                return
            except Exception:
                pass

    raise RuntimeError(f"Server failed to start on port {SERVER_PORT} within 30s")

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

def llm_detect_violations(client: OpenAI, content: str) -> list:
    """Ask LLM to detect violations. Returns list of violation type strings."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": (
                "You are a content moderation expert. "
                "Analyze the content below and identify ALL policy violations present.\n\n"
                "Violation types (use exact names):\n"
                "  hate_speech, violence, harassment, adult_content,\n"
                "  misinformation, spam, suspicious_link, engagement_bait, illegal_content\n\n"
                f"Content: {content}\n\n"
                "Return JSON with key 'violations' as a list. "
                "Example: {\"violations\": [\"spam\", \"suspicious_link\"]}\n"
                "If no violations: {\"violations\": []}"
            )}],
            max_tokens=80,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        import json as _j
        data = _j.loads(response.choices[0].message.content)
        violations = data.get("violations", [])
        if not isinstance(violations, list):
            violations = []
        # Filter to valid violation types only
        valid = {"hate_speech", "violence", "harassment", "adult_content",
                 "misinformation", "spam", "suspicious_link", "engagement_bait", "illegal_content"}
        return [v for v in violations if v in valid]
    except Exception as exc:
        print(f"[DEBUG] llm_detect_violations failed: {exc}", flush=True)
        return []


def llm_score_compliance(client: OpenAI, content: str, violations: list) -> float:
    """Ask LLM to score content compliance 0.0-1.0."""
    try:
        viol_str = ", ".join(violations) if violations else "none"
        # Determine expected range based on violations
        if not violations:
            hint = "No violations found. Score should be 0.8 to 1.0."
        elif any(v in {"hate_speech", "violence", "illegal_content"} for v in violations):
            hint = "Severe violations present. Score should be 0.0 to 0.3."
        elif any(v in {"harassment", "misinformation"} for v in violations):
            hint = "Moderate violations present. Score should be 0.2 to 0.5."
        else:
            hint = "Minor violations present. Score should be 0.4 to 0.7."

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": (
                f"Rate this content's policy compliance from 0.0 (worst) to 1.0 (clean).\n"
                f"Content: {content}\n"
                f"Detected violations: {viol_str}\n"
                f"{hint}\n\n"
                f"Return JSON: {{\"score\": <float>}}"
            )}],
            max_tokens=20,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        import json as _j
        data = _j.loads(response.choices[0].message.content)
        return round(max(0.0, min(1.0, float(data.get("score", 0.5)))), 3)
    except Exception as exc:
        print(f"[DEBUG] llm_score_compliance failed: {exc}", flush=True)
        # Fallback: derive score from violations
        if not violations:
            return 0.9
        severe = sum(1 for v in violations if v in {"hate_speech", "violence", "illegal_content"})
        return round(max(0.05, 0.9 - severe * 0.35 - (len(violations) - severe) * 0.15), 2)


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
    content = obs.content

    # Step 1: LLM detects violations
    detected_violations = llm_detect_violations(llm, content)
    action = ContentAction(
        action_type="detect_violations",
        metadata={"violations": detected_violations}
    )
    result = await env.step(action)
    obs = result.observation
    reward = float(result.reward) if result.reward is not None else 0.05
    reward = max(0.01, min(0.99, reward))
    rewards.append(reward)
    log_step(step=1, action=f"detect({len(detected_violations)}_violations)", reward=reward, done=obs.done, error=None)
    time.sleep(0.2)

    # Step 2: LLM scores compliance
    predicted_score = llm_score_compliance(llm, content, detected_violations)
    action = ContentAction(
        action_type="score_compliance",
        metadata={"score": predicted_score}
    )
    result = await env.step(action)
    obs = result.observation
    reward = float(result.reward) if result.reward is not None else 0.05
    reward = max(0.01, min(0.99, reward))
    rewards.append(reward)
    log_step(step=2, action=f"score({predicted_score})", reward=reward, done=obs.done, error=None)
    time.sleep(0.2)

    # Step 3: LLM decides action
    decision = llm_decide(llm, content, detected_violations, TASK_DESCRIPTIONS[3])
    action = ContentAction(action_type=decision)
    result = await env.step(action)
    obs = result.observation
    reward = float(result.reward) if result.reward is not None else 0.05
    reward = max(0.01, min(0.99, reward))
    rewards.append(reward)
    log_step(step=3, action=decision, reward=reward, done=obs.done, error=None)
    time.sleep(0.2)

    # Step 4: confirm or edit
    if decision == "edit":
        edited = llm_rewrite(llm, content, detected_violations, TASK_DESCRIPTIONS[4])
        action = ContentAction(action_type="submit_edit", edited_content=edited)
        step4_label = "submit_edit"
    else:
        action = ContentAction(action_type=f"confirm_{decision}")
        step4_label = f"confirm_{decision}"

    result = await env.step(action)
    obs = result.observation
    reward = float(result.reward) if result.reward is not None else 0.05
    reward = max(0.01, min(0.99, reward))
    rewards.append(reward)
    log_step(step=4, action=step4_label, reward=reward, done=obs.done, error=None)

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
    _ensure_server_running()
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
