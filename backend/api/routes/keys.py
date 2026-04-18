from __future__ import annotations
import secrets
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.api.db.session import get_db
from backend.api.db.models import ApiKey

router = APIRouter(prefix="/api/v1/keys", tags=["keys"])


class CreateKeyRequest(BaseModel):
    name: str
    plan: str = "free"


@router.post("/")
async def create_key(req: CreateKeyRequest, db: Session = Depends(get_db)):
    key = "gr-" + secrets.token_hex(24)
    limit = {"free": 1000, "pro": 50000, "enterprise": 10_000_000}.get(req.plan, 1000)
    obj = ApiKey(key=key, name=req.name, plan=req.plan, limit=limit)
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return {"key": obj.key, "name": obj.name, "plan": obj.plan, "limit": obj.limit}


@router.get("/")
async def list_keys(db: Session = Depends(get_db)):
    keys = db.query(ApiKey).filter(ApiKey.active == True).all()
    return [
        {"id": k.id, "name": k.name, "plan": k.plan,
         "requests": k.requests, "limit": k.limit,
         "key_preview": k.key[:10] + "...", "created_at": k.created_at.isoformat()}
        for k in keys
    ]


@router.delete("/{key_id}")
async def revoke_key(key_id: int, db: Session = Depends(get_db)):
    key = db.query(ApiKey).filter(ApiKey.id == key_id).first()
    if not key:
        raise HTTPException(status_code=404, detail="Key not found")
    key.active = False
    db.commit()
    return {"status": "revoked"}
