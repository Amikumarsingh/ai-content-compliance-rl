from __future__ import annotations
from datetime import datetime
from sqlalchemy import Column, String, Float, Integer, Boolean, DateTime, Text, JSON
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class ModerationLog(Base):
    __tablename__ = "moderation_logs"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    api_key     = Column(String(64), index=True, nullable=False)
    content     = Column(Text, nullable=False)
    violations  = Column(JSON, default=list)
    score       = Column(Float, nullable=False)
    decision    = Column(String(16), nullable=False)   # approve / reject / edit
    edited      = Column(Text, nullable=True)
    reward      = Column(Float, nullable=True)
    difficulty  = Column(String(8), nullable=True)
    latency_ms  = Column(Float, nullable=True)
    correct     = Column(Boolean, nullable=True)       # human feedback
    created_at  = Column(DateTime, default=datetime.utcnow)


class ApiKey(Base):
    __tablename__ = "api_keys"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    key        = Column(String(64), unique=True, index=True, nullable=False)
    name       = Column(String(128), nullable=False)
    plan       = Column(String(16), default="free")    # free / pro / enterprise
    requests   = Column(Integer, default=0)
    limit      = Column(Integer, default=1000)
    active     = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
