"""
Content Compliance RL Environment - FastAPI Server

Production-grade OpenEnv-compatible server for Hugging Face Spaces deployment.
Exposes content moderation environment via REST API.

Endpoints:
    GET /         - API information and health
    POST /reset   - Reset environment, start new episode
    POST /step    - Take action in environment
    GET /state    - Get current environment state
    GET /health   - Health check endpoint

Usage:
    uvicorn app:app --host 0.0.0.0 --port 7860
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from datetime import datetime
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger(__name__)

# Suppress verbose HTTP logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Import environment and models
from env import ContentComplianceEnv
from models import Observation, Action, Reward


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="Content Compliance RL Environment",
    description="Production-grade OpenEnv-compatible content moderation environment for Hugging Face Spaces",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Global State
# =============================================================================

class EnvironmentState:
    """Manages environment instances per session."""

    def __init__(self):
        self.environments: dict[str, dict[str, Any]] = {}
        self.request_count: int = 0
        self.episode_count: int = 0
        self.step_count: int = 0
        self.start_time: datetime = datetime.now()

    def create_env(self, max_steps: int = 5, difficulty: str = "mixed") -> tuple[str, ContentComplianceEnv]:
        """Create a new environment instance."""
        env_id = str(uuid.uuid4())[:8]
        env = ContentComplianceEnv(max_steps=max_steps, difficulty=difficulty)
        self.environments[env_id] = {
            "env": env,
            "created_at": datetime.now(),
            "max_steps": max_steps,
            "difficulty": difficulty,
        }
        self.episode_count += 1
        return env_id, env

    def get_env(self, env_id: Optional[str] = None) -> tuple[Optional[str], Optional[ContentComplianceEnv]]:
        """Get environment by ID or most recent."""
        if env_id and env_id in self.environments:
            return env_id, self.environments[env_id]["env"]
        if self.environments:
            env_id = list(self.environments.keys())[-1]
            return env_id, self.environments[env_id]["env"]
        return None, None


# Global state instance
state = EnvironmentState()


# =============================================================================
# Helper Functions
# =============================================================================

def error_response(
    message: str,
    detail: str = "",
    status_code: int = 500,
) -> dict[str, Any]:
    """Create standardized error response."""
    return {
        "status": "error",
        "error": {
            "message": message,
            "detail": detail,
            "code": status_code,
        },
        "timestamp": datetime.now().isoformat(),
    }


def success_response(
    observation: Optional[dict[str, Any]] = None,
    reward: Optional[dict[str, Any]] = None,
    done: bool = False,
    truncated: bool = False,
    info: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Create standardized success response."""
    response: dict[str, Any] = {
        "status": "success",
        "done": done,
        "truncated": truncated,
        "info": info or {},
    }
    if observation:
        response["observation"] = observation
    if reward is not None:
        response["reward"] = reward
    return response


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/")
async def root():
    """Root endpoint with API information."""
    uptime = (datetime.now() - state.start_time).total_seconds()
    return {
        "name": "Content Compliance RL Environment",
        "version": "1.0.0",
        "status": "running",
        "uptime_seconds": uptime,
        "endpoints": {
            "GET /": "API information",
            "GET /health": "Health check",
            "POST /reset": "Reset environment",
            "POST /step": "Take action",
            "GET /state": "Get current state",
            "GET /spec": "Environment specification",
        },
        "metrics": {
            "total_episodes": state.episode_count,
            "total_steps": state.step_count,
            "active_environments": len(state.environments),
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    uptime = (datetime.now() - state.start_time).total_seconds()
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": uptime,
        "metrics": {
            "total_requests": state.request_count,
            "total_episodes": state.episode_count,
            "total_steps": state.step_count,
            "active_environments": len(state.environments),
        },
    }


@app.post("/reset")
async def reset(body: Optional[dict[str, Any]] = None):
    """
    Reset environment and start new episode.

    Request body (optional):
        max_steps: int (default: 5)
        difficulty: str - "easy", "medium", "hard", "mixed" (default: "mixed")

    Returns:
        Initial observation and episode info.
    """
    state.request_count += 1

    # Parse request body
    max_steps = 5
    difficulty = "mixed"

    if body:
        max_steps = int(body.get("max_steps", 5))
        difficulty = str(body.get("difficulty", "mixed"))

    # Validate inputs
    max_steps = max(1, min(max_steps, 20))
    valid_difficulties = ["easy", "medium", "hard", "mixed"]
    if difficulty not in valid_difficulties:
        difficulty = "mixed"

    # Create new environment
    env_id, env = state.create_env(max_steps=max_steps, difficulty=difficulty)

    try:
        # Reset environment with timeout
        observation = await asyncio.wait_for(
            env.reset(),
            timeout=30.0,
        )

        logger.info(f"[START] env_id={env_id}, difficulty={difficulty}, score={observation.score:.2f}")

        return success_response(
            observation=observation.model_dump(),
            info={
                "env_id": env_id,
                "max_steps": max_steps,
                "difficulty": difficulty,
                "timestamp": datetime.now().isoformat(),
            },
        )

    except asyncio.TimeoutError:
        logger.error(f"[STEP] Reset timeout for env_id={env_id}")
        raise HTTPException(status_code=504, detail="Environment reset timed out")

    except Exception as e:
        logger.exception(f"[STEP] Reset failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
async def step(action_data: dict[str, Any]):
    """
    Take a step in the environment.

    Request body:
        action_type: str (required) - "approve", "reject", or "edit"
        edited_content: str (optional, required for edit action)
        env_id: str (optional) - environment ID

    Returns:
        Step result with observation, reward, done, truncated, info.
    """
    state.request_count += 1

    # Validate action_type
    action_type = action_data.get("action_type", "approve")
    valid_actions = {"approve", "reject", "edit"}
    if action_type not in valid_actions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid action_type. Must be one of: {valid_actions}",
        )

    # Get edited_content
    edited_content = action_data.get("edited_content")
    if action_type == "edit" and not edited_content:
        raise HTTPException(
            status_code=400,
            detail="Edit action requires edited_content",
        )

    # Get environment
    env_id = action_data.get("env_id")
    env_id, env = state.get_env(env_id)

    if env is None:
        # Create new environment if none exists
        env_id, env = state.create_env()

    try:
        # Create action
        action = Action(
            action_type=action_type,
            edited_content=edited_content,
        )

        # Execute step with timeout
        result = await asyncio.wait_for(
            env.step(action),
            timeout=30.0,
        )

        state.step_count += 1

        # Log step
        logger.info(
            f"[STEP] env_id={env_id}, step={result.observation.step_count}, "
            f"action={action_type}, reward={result.reward.value:.3f}, done={result.done}"
        )

        return success_response(
            observation=result.observation.model_dump(),
            reward=result.reward.model_dump(),
            done=result.done,
            truncated=result.truncated,
            info={
                "env_id": env_id,
                "timestamp": datetime.now().isoformat(),
            },
        )

    except asyncio.TimeoutError:
        logger.error(f"[STEP] Timeout for env_id={env_id}")
        raise HTTPException(status_code=504, detail="Step execution timed out")

    except ValueError as ve:
        logger.warning(f"[STEP] Invalid action: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))

    except Exception as e:
        logger.exception(f"[STEP] Failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
async def get_state(env_id: Optional[str] = None):
    """
    Get current environment state.

    Query params:
        env_id: str (optional) - environment ID

    Returns:
        Current observation and episode info.
    """
    state.request_count += 1

    _, env = state.get_env(env_id)

    if env is None:
        raise HTTPException(status_code=404, detail="No active environment")

    try:
        current_state = env.state()

        return {
            "status": "success",
            "observation": current_state.model_dump(),
            "info": {
                "env_id": env_id,
                "timestamp": datetime.now().isoformat(),
            },
        }

    except Exception as e:
        logger.exception(f"[STEP] State failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/spec")
async def get_spec():
    """Get OpenEnv environment specification."""
    return {
        "name": "ContentComplianceEnv",
        "version": "1.0.0",
        "type": "content_moderation",
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
                "step_count": {"type": "int"},
                "stage": {"type": "str"},
                "previous_action": {"type": "str", "nullable": True},
                "action_history": {"type": "list[str]"},
                "feedback": {"type": "str", "nullable": True},
            },
        },
        "reward_range": [0.0, 1.0],
        "tasks": {
            "easy": {"max_steps": 2, "description": "Binary approve/reject"},
            "medium": {"max_steps": 3, "description": "Edit + re-evaluate"},
            "hard": {"max_steps": 5, "description": "Multi-step reasoning"},
        },
    }


@app.get("/metrics")
async def get_metrics():
    """Get server metrics."""
    uptime = (datetime.now() - state.start_time).total_seconds()
    return {
        "server": {
            "uptime_seconds": uptime,
            "start_time": state.start_time.isoformat(),
            "timestamp": datetime.now().isoformat(),
        },
        "usage": {
            "total_requests": state.request_count,
            "total_episodes": state.episode_count,
            "total_steps": state.step_count,
            "active_environments": len(state.environments),
        },
    }


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "7860"))
    host = os.getenv("HOST", "0.0.0.0")

    logger.info(f"[START] Server on {host}:{port}")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
    )
