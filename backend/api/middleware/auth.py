from fastapi import Header, HTTPException, Depends
from sqlalchemy.orm import Session
from backend.api.db.session import get_db
from backend.api.db.models import ApiKey


async def require_api_key(
    x_api_key: str = Header(..., alias="X-API-Key"),
    db: Session = Depends(get_db),
) -> ApiKey:
    key = db.query(ApiKey).filter(ApiKey.key == x_api_key, ApiKey.active == True).first()
    if not key:
        raise HTTPException(status_code=401, detail="Invalid or inactive API key")
    if key.requests >= key.limit:
        raise HTTPException(status_code=429, detail="Request limit reached for this plan")
    key.requests += 1
    db.commit()
    return key
