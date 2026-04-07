"""
FastAPI Server for Hugging Face Spaces.

Provides REST API endpoints for content compliance environment:
- POST /reset - Reset environment and start new episode
- POST /step - Take action in environment
- GET /health - Health check endpoint

OpenEnv validator compatible - no required request bodies.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting Content Compliance RL server...")
    app.state.start_time = datetime.now()
    app.state.episode_count = 0
    app.state.step_count = 0

    try:
        from app.env import ContentComplianceEnv
        app.state.env = ContentComplianceEnv(
            max_steps=5,
            difficulty="mixed",
            evaluator_provider=os.getenv("EVALUATOR_PROVIDER", "mock"),
        )
        logger.info("Environment initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize environment: {e}")
        app.state.env = None

    yield

    logger.info("Shutting down server...")
    if app.state.env:
        await app.state.env.close()


app = FastAPI(
    title="Content Compliance RL Environment",
    description="OpenEnv-compatible content moderation environment for RL training",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Content Compliance RL Environment",
        "version": "1.0.0",
        "endpoints": {
            "GET /": "API information",
            "GET /health": "Health check",
            "POST /reset": "Reset environment",
            "POST /step": "Take action in environment",
        },
    }


@app.get("/health", response_model=dict)
async def health_check():
    """Health check endpoint."""
    uptime = (datetime.now() - app.state.start_time).total_seconds()
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": uptime,
        "episode_count": app.state.episode_count,
        "step_count": app.state.step_count,
    }


@app.post("/reset", response_model=None)
async def reset(request: Request):
    max_steps = 5
    difficulty = "mixed"

    try:
        body = await request.json()
        if body:
            max_steps = int(body.get("max_steps", 5))
            difficulty = str(body.get("difficulty", "mixed"))
    except Exception:
        pass

    try:
        from app.env import ContentComplianceEnv
        app.state.env = ContentComplianceEnv(
            max_steps=max_steps,
            difficulty=difficulty,
            evaluator_provider=os.getenv("EVALUATOR_PROVIDER", "mock"),
        )
        observation = await app.state.env.reset()
        app.state.episode_count += 1
        return {
            "status": "success",
            "observation": observation.model_dump(),
            "info": getattr(app.state.env, "_info", {}),
        }
    except Exception as e:
        logger.error(f"Reset failed: {e}")
        return {
            "status": "error",
            "observation": {"content": "", "violations": [], "score": 0.0, "step_count": 0},
            "info": {"error": str(e)},
        }


@app.post("/step", response_model=None)
async def step(request: Request):
    """
    Take a step in the environment.

    Accepts JSON body with:
    - action_type: str (required) - "approve", "reject", or "edit"
    - edited_content: str (optional)

    Returns HTTP 200 with StepResult JSON.
    """
    if not app.state.env:
        return {
            "status": "error",
            "observation": {
                "content": "",
                "violations": [],
                "score": 0.0,
                "step_count": 0,
            },
            "reward": {"value": 0.0, "raw_value": 0.0, "explanation": "Environment not initialized"},
            "done": True,
            "truncated": False,
            "info": {"error": "Environment not initialized"},
        }

    try:
        body = await request.json()
    except Exception:
        body = {}

    action_type = body.get("action_type", "approve")
    edited_content = body.get("edited_content")

    valid_actions = {"approve", "reject", "edit"}
    if action_type not in valid_actions:
        action_type = "approve"

    try:
        from app.models import Action

        action = Action(
            action_type=action_type,
            edited_content=edited_content,
        )

        result = await app.state.env.step(action)
        app.state.step_count += 1

        return {
            "status": "success",
            "observation": result.observation.model_dump(),
            "reward": result.reward.model_dump(),
            "done": result.done,
            "truncated": result.truncated,
            "info": result.info,
        }

    except (ValueError, RuntimeError) as e:
        logger.warning(f"Action error: {e}")
        return {
            "status": "error",
            "observation": {
                "content": "",
                "violations": [],
                "score": 0.0,
                "step_count": 0,
            },
            "reward": {"value": 0.0, "raw_value": 0.0, "explanation": str(e)},
            "done": True,
            "truncated": False,
            "info": {"error": str(e)},
        }

    except Exception as e:
        logger.error(f"Step failed: {e}")
        return {
            "status": "error",
            "observation": {
                "content": "",
                "violations": [],
                "score": 0.0,
                "step_count": 0,
            },
            "reward": {"value": 0.0, "raw_value": 0.0, "explanation": str(e)},
            "done": True,
            "truncated": False,
            "info": {"error": str(e)},
        }


@app.get("/spec", response_model=dict)
async def get_spec():
    """Get OpenEnv specification."""
    return {
        "name": "ContentComplianceEnv",
        "version": "1.0.0",
        "action_space": {
            "type": "discrete",
            "actions": ["approve", "reject", "edit"],
        },
        "observation_space": {
            "type": "dict",
            "fields": {
                "content": {"type": "str"},
                "violations": {"type": "list[str]"},
                "score": {"type": "float", "range": [0.0, 1.0]},
                "step_count": {"type": "int", "min": 0},
            },
        },
        "reward_range": [0.0, 1.0],
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)
