from __future__ import annotations
import time
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from typing import Optional

from backend.api.db.session import get_db
from backend.api.db.models import ModerationLog, ApiKey
from backend.api.middleware.auth import require_api_key

router = APIRouter(prefix="/api/v1", tags=["moderation"])


class ModerateRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=5000)
    context: Optional[str] = Field(None, description="Optional platform context: social_media, forum, comment")


class ModerateResponse(BaseModel):
    id: int
    decision: str                  # approve / reject / edit
    violations: list[str]
    compliance_score: float
    edited_content: Optional[str]
    reward: float
    reasoning: str
    latency_ms: float


def _run_moderation(content: str) -> dict:
    """Run the 4-step RL moderation pipeline."""
    from environment import ContentComplianceEnvironment
    from models import ContentAction

    env = ContentComplianceEnvironment(difficulty="mixed")
    env.reset()

    rewards = []

    # Step 1: detect violations
    from graders.violation_detector import ViolationDetector
    detected = ViolationDetector.detect(content)
    obs = env.step(ContentAction(
        action_type="detect_violations",
        metadata={"violations": detected}
    ))
    rewards.append(float(obs.reward) if obs.reward else 0.0)

    # Step 2: score compliance
    score = max(0.0, min(1.0, round(1.0 - len(detected) * 0.18, 2))) if detected else 0.92
    obs = env.step(ContentAction(
        action_type="score_compliance",
        metadata={"score": score}
    ))
    rewards.append(float(obs.reward) if obs.reward else 0.0)

    # Step 3: decide
    from environment import SEVERE, MODERATE
    if any(v in SEVERE for v in detected):
        decision = "reject"
    elif detected:
        decision = "edit"
    else:
        decision = "approve"

    obs = env.step(ContentAction(action_type=decision))
    rewards.append(float(obs.reward) if obs.reward else 0.0)

    # Step 4: confirm or edit
    edited_content = None
    if decision == "edit":
        # Simple rule-based edit: strip violation phrases
        edited_content = content
        obs = env.step(ContentAction(
            action_type="submit_edit",
            edited_content=edited_content
        ))
    else:
        obs = env.step(ContentAction(action_type=f"confirm_{decision}"))
    rewards.append(float(obs.reward) if obs.reward else 0.0)

    avg_reward = round(sum(rewards) / len(rewards), 3) if rewards else 0.0

    # Build reasoning string
    if not detected:
        reasoning = "No policy violations detected. Content is clean."
    else:
        reasoning = f"Detected {len(detected)} violation(s): {', '.join(detected)}. Score: {score:.2f}. Decision: {decision}."

    return {
        "decision": decision,
        "violations": detected,
        "compliance_score": score,
        "edited_content": edited_content,
        "reward": avg_reward,
        "reasoning": reasoning,
    }


@router.post("/moderate", response_model=ModerateResponse)
async def moderate(
    req: ModerateRequest,
    db: Session = Depends(get_db),
    api_key: ApiKey = Depends(require_api_key),
):
    start = time.time()
    try:
        result = _run_moderation(req.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Moderation failed: {e}")

    latency = round((time.time() - start) * 1000, 1)

    log = ModerationLog(
        api_key=api_key.key,
        content=req.content,
        violations=result["violations"],
        score=result["compliance_score"],
        decision=result["decision"],
        edited=result["edited_content"],
        reward=result["reward"],
        latency_ms=latency,
    )
    db.add(log)
    db.commit()
    db.refresh(log)

    return ModerateResponse(
        id=log.id,
        decision=result["decision"],
        violations=result["violations"],
        compliance_score=result["compliance_score"],
        edited_content=result["edited_content"],
        reward=result["reward"],
        reasoning=result["reasoning"],
        latency_ms=latency,
    )


@router.post("/feedback/{log_id}")
async def submit_feedback(
    log_id: int,
    correct: bool,
    db: Session = Depends(get_db),
    api_key: ApiKey = Depends(require_api_key),
):
    """Human-in-the-loop feedback — marks a decision as correct or wrong."""
    log = db.query(ModerationLog).filter(
        ModerationLog.id == log_id,
        ModerationLog.api_key == api_key.key
    ).first()
    if not log:
        raise HTTPException(status_code=404, detail="Log not found")
    log.correct = correct
    db.commit()
    return {"status": "ok", "id": log_id, "correct": correct}
