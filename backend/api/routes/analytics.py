from __future__ import annotations
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from collections import Counter

from backend.api.db.session import get_db
from backend.api.db.models import ModerationLog, ApiKey
from backend.api.middleware.auth import require_api_key

router = APIRouter(prefix="/api/v1", tags=["analytics"])


@router.get("/analytics")
async def get_analytics(
    db: Session = Depends(get_db),
    api_key: ApiKey = Depends(require_api_key),
):
    logs = db.query(ModerationLog).filter(ModerationLog.api_key == api_key.key).all()

    if not logs:
        return {
            "total": 0, "approved": 0, "rejected": 0, "edited": 0,
            "avg_score": 0, "avg_reward": 0, "avg_latency_ms": 0,
            "violation_breakdown": {}, "accuracy": None,
            "recent": []
        }

    decisions = Counter(l.decision for l in logs)
    all_violations = [v for l in logs for v in (l.violations or [])]
    violation_counts = Counter(all_violations)

    scored = [l for l in logs if l.correct is not None]
    accuracy = round(sum(1 for l in scored if l.correct) / len(scored), 3) if scored else None

    recent = sorted(logs, key=lambda l: l.created_at, reverse=True)[:20]

    return {
        "total": len(logs),
        "approved": decisions.get("approve", 0),
        "rejected": decisions.get("reject", 0),
        "edited": decisions.get("edit", 0),
        "avg_score": round(sum(l.score for l in logs) / len(logs), 3),
        "avg_reward": round(sum(l.reward or 0 for l in logs) / len(logs), 3),
        "avg_latency_ms": round(sum(l.latency_ms or 0 for l in logs) / len(logs), 1),
        "violation_breakdown": dict(violation_counts.most_common()),
        "accuracy": accuracy,
        "recent": [
            {
                "id": l.id,
                "content": l.content[:80] + "..." if len(l.content) > 80 else l.content,
                "decision": l.decision,
                "violations": l.violations,
                "score": l.score,
                "reward": l.reward,
                "correct": l.correct,
                "created_at": l.created_at.isoformat(),
            }
            for l in recent
        ],
    }
