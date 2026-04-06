"""
Production-Grade FastAPI Server for Content Compliance RL Environment.

Features:
- Global error handling with structured responses
- Request isolation (per-request environment state)
- Structured logging for debugging
- Async operations throughout
- Timeout handling for LLM calls
- Robust input validation
- No static content (fully dynamic)

OpenEnv-compatible API:
- POST /reset - Reset environment, start new episode
- POST /step - Take action in environment
- GET /health - Health check
- GET /spec - Environment specification
"""

from __future__ import annotations

import asyncio
import logging
import os
import traceback
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Load environment variables from .env file
load_dotenv()

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Global Error Handler & Exception Middleware
# =============================================================================

class ErrorHandlerMiddleware:
    """
    Global exception handler middleware.

    Catches all unhandled exceptions and returns structured error responses.
    Ensures the API never crashes and always returns valid JSON.
    """

    def __init__(self, app: FastAPI):
        self.app = app

    async def __call__(self, scope: dict, receive: callable, send: callable):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        try:
            return await self.app(scope, receive, send)
        except Exception as e:
            logger.exception(f"Unhandled exception: {e}")

            error_response = {
                "status": "error",
                "error": {
                    "type": type(e).__name__,
                    "message": str(e),
                    "timestamp": datetime.now().isoformat(),
                },
            }

            response = JSONResponse(
                status_code=500,
                content=error_response,
            )
            await response(scope, receive, send)


# =============================================================================
# Application Lifespan Manager
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Handles startup and shutdown:
    - Initialize shared resources
    - Clean up on shutdown
    - Log server lifecycle events
    """
    logger.info("=" * 60)
    logger.info("Starting Content Compliance RL Server...")
    logger.info(f"Environment: {os.getenv('ENVIRONMENT', 'development')}")
    logger.info(f"Port: {os.getenv('PORT', '8000')}")
    logger.info(f"Evaluator Provider: {os.getenv('EVALUATOR_PROVIDER', 'openai')}")
    logger.info("=" * 60)

    app.state.start_time = datetime.now()
    app.state.request_count = 0
    app.state.episode_count = 0
    app.state.step_count = 0
    app.state.environments: dict[str, Any] = {}  # Per-request env isolation

    # Initialize environment pool (optional, for performance)
    app.state.env_pool_size = int(os.getenv("ENV_POOL_SIZE", "5"))

    logger.info(f"Server started with environment pool size: {app.state.env_pool_size}")

    yield

    # Shutdown cleanup
    logger.info("Shutting down server...")

    # Clean up all environments
    cleanup_tasks = []
    for env_id, env_data in app.state.environments.items():
        if "env" in env_data and hasattr(env_data["env"], "close"):
            cleanup_tasks.append(env_data["env"].close())

    if cleanup_tasks:
        await asyncio.gather(*cleanup_tasks, return_exceptions=True)

    app.state.environments.clear()
    logger.info(f"Server shutdown complete. Total episodes: {app.state.episode_count}")


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="Content Compliance RL Environment",
    description="Production-grade OpenEnv-compatible content moderation environment",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add error handler middleware
app.add_middleware(ErrorHandlerMiddleware)


# =============================================================================
# Helper Functions
# =============================================================================

def get_error_response(
    error_msg: str,
    error_detail: str = "",
    status_code: int = 500,
) -> dict[str, Any]:
    """
    Create a standardized error response.

    Args:
        error_msg: Human-readable error message.
        error_detail: Additional error details.
        status_code: HTTP status code.

    Returns:
        Standardized error response dictionary.
    """
    return {
        "status": "error",
        "error": {
            "message": error_msg,
            "detail": error_detail,
            "code": status_code,
        },
        "observation": {
            "content": "",
            "violations": [],
            "score": 0.0,
            "step_count": 0,
            "stage": "unknown",
            "previous_action": None,
            "action_history": [],
        },
        "reward": {
            "value": 0.0,
            "raw_value": 0.0,
            "explanation": error_msg,
        },
        "done": True,
        "truncated": False,
        "info": {
            "error": error_msg,
            "timestamp": datetime.now().isoformat(),
        },
    }


def get_standard_response(
    observation: dict[str, Any],
    reward: Optional[dict[str, Any]] = None,
    done: bool = False,
    truncated: bool = False,
    info: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Create a standardized success response.

    Args:
        observation: Environment observation.
        reward: Reward information (optional for reset).
        done: Whether episode is complete.
        truncated: Whether episode was truncated.
        info: Additional information.

    Returns:
        Standardized response dictionary.
    """
    response = {
        "status": "success",
        "observation": observation,
        "done": done,
        "truncated": truncated,
        "info": info or {},
    }

    if reward is not None:
        response["reward"] = reward

    return response


async def create_environment(
    max_steps: int = 5,
    difficulty: str = "mixed",
) -> tuple[Any, str]:
    """
    Create a new environment instance.

    Args:
        max_steps: Maximum steps per episode.
        difficulty: Content difficulty level.

    Returns:
        Tuple of (environment instance, environment ID).
    """
    from app.env import ContentComplianceEnv

    env_id = str(uuid.uuid4())[:8]
    env = ContentComplianceEnv(
        max_steps=max_steps,
        difficulty=difficulty,
        evaluator_provider=os.getenv("EVALUATOR_PROVIDER", "openai"),
    )

    return env, env_id


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/", response_model=dict)
async def root():
    """
    Root endpoint with API information.

    Returns:
        API metadata and available endpoints.
    """
    uptime = (datetime.now() - app.state.start_time).total_seconds()

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
            "GET /spec": "Environment specification",
        },
    }


@app.get("/health", response_model=dict)
async def health_check():
    """
    Health check endpoint.

    Returns:
        Server health status and metrics.
    """
    try:
        uptime = (datetime.now() - app.state.start_time).total_seconds()

        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": uptime,
            "metrics": {
                "total_requests": getattr(app.state, "request_count", 0),
                "total_episodes": getattr(app.state, "episode_count", 0),
                "total_steps": getattr(app.state, "step_count", 0),
                "active_environments": len(getattr(app.state, "environments", {})),
            },
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "degraded",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
        }


@app.post("/reset")
async def reset(request: Request):
    """
    Reset the environment and start a new episode.

    Accepts optional JSON body with:
    - max_steps: int (default: 5)
    - difficulty: str (default: "mixed")

    Always returns valid JSON, never crashes.

    Returns:
        Initial observation and episode info.
    """
    try:
        app.state.request_count = getattr(app.state, "request_count", 0) + 1

        # Parse optional request body with timeout
        max_steps = 5
        difficulty = "mixed"

        try:
            # Add timeout for body parsing
            body = await asyncio.wait_for(request.json(), timeout=5.0)
            if body:
                max_steps = int(body.get("max_steps", 5))
                difficulty = str(body.get("difficulty", "mixed"))
        except asyncio.TimeoutError:
            logger.warning("Request body parsing timed out, using defaults")
        except Exception:
            pass  # Use defaults for empty/invalid body

        # Validate inputs
        max_steps = max(1, min(max_steps, 20))  # Clamp to 1-20
        valid_difficulties = ["easy", "medium", "hard", "mixed"]
        if difficulty not in valid_difficulties:
            difficulty = "mixed"

        # Create new environment (isolated per request)
        env, env_id = await create_environment(max_steps, difficulty)

        # Store environment for this session
        app.state.environments[env_id] = {
            "env": env,
            "created_at": datetime.now(),
            "max_steps": max_steps,
            "difficulty": difficulty,
        }

        # Reset environment
        try:
            observation = await asyncio.wait_for(
                env.reset(),
                timeout=30.0,  # LLM calls may take time
            )

            app.state.episode_count = getattr(app.state, "episode_count", 0) + 1

            # Build response
            response_data = get_standard_response(
                observation=observation.model_dump(),
                info={
                    "env_id": env_id,
                    "max_steps": max_steps,
                    "difficulty": difficulty,
                    "initial_score": observation.score,
                    "initial_violations": observation.violations,
                    "timestamp": datetime.now().isoformat(),
                },
            )

            logger.info(
                f"[RESET] env_id={env_id}, difficulty={difficulty}, "
                f"score={observation.score:.2f}, violations={len(observation.violations)}"
            )

            return response_data

        except asyncio.TimeoutError:
            logger.error(f"Environment reset timed out for env_id={env_id}")
            return get_error_response(
                error_msg="Environment reset timed out",
                error_detail="LLM call took too long",
            )
        except Exception as reset_error:
            logger.exception(f"Environment reset failed: {reset_error}")
            return get_error_response(
                error_msg="Environment reset failed",
                error_detail=str(reset_error),
            )

    except Exception as e:
        logger.exception(f"Unexpected error in /reset: {e}")
        return get_error_response(
            error_msg="Unexpected error during reset",
            error_detail=str(e),
        )


@app.post("/step")
async def step(request: Request):
    """
    Take a step in the environment.

    Accepts JSON body with:
    - action_type: str (required) - "approve", "reject", or "edit"
    - edited_content: str (optional, required for edit action)

    Always returns valid JSON, never crashes.

    Returns:
        Step result with observation, reward, done, truncated, info.
    """
    try:
        app.state.request_count = getattr(app.state, "request_count", 0) + 1

        # Parse request body with validation
        try:
            body = await asyncio.wait_for(request.json(), timeout=5.0)
        except asyncio.TimeoutError:
            return get_error_response(
                error_msg="Request body parsing timed out",
                error_detail="Please try again",
            )
        except Exception:
            body = {}

        # Validate action_type
        action_type = body.get("action_type", "approve")
        valid_actions = {"approve", "reject", "edit"}
        if action_type not in valid_actions:
            logger.warning(f"Invalid action_type: {action_type}, defaulting to 'approve'")
            action_type = "approve"

        # Get edited_content (required for edit action)
        edited_content = body.get("edited_content")
        if action_type == "edit" and not edited_content:
            return get_error_response(
                error_msg="Edit action requires edited_content",
                error_detail="Please provide edited_content in request body",
            )

        # Get environment from state (or create default)
        env_id = body.get("env_id")
        env_data = app.state.environments.get(env_id) if env_id else None

        if env_data is None:
            # Use the most recently created environment or create new one
            if app.state.environments:
                env_id, env_data = list(app.state.environments.items())[-1]
            else:
                # Create new environment with defaults
                env, new_env_id = await create_environment()
                app.state.environments[new_env_id] = {
                    "env": env,
                    "created_at": datetime.now(),
                    "max_steps": 5,
                    "difficulty": "mixed",
                }
                env_id = new_env_id
                env_data = app.state.environments[new_env_id]

        env = env_data.get("env")
        if env is None:
            return get_error_response(
                error_msg="Environment not found",
                error_detail=f"env_id={env_id} not available",
            )

        # Execute step with timeout
        try:
            from app.models import Action

            action = Action(
                action_type=action_type,
                edited_content=edited_content,
            )

            result = await asyncio.wait_for(
                env.step(action),
                timeout=30.0,  # LLM calls may take time
            )

            app.state.step_count = getattr(app.state, "step_count", 0) + 1

            # Build structured response
            info = result.info.copy() if result.info else {}
            info["env_id"] = env_id
            info["timestamp"] = datetime.now().isoformat()

            # Add structured log info
            info["step_log"] = {
                "step_number": result.observation.step_count,
                "action_taken": action_type,
                "reward_value": result.reward.value,
                "violations_detected": len(result.observation.violations),
                "stage": result.observation.stage,
            }

            response_data = get_standard_response(
                observation=result.observation.model_dump(),
                reward=result.reward.model_dump(),
                done=result.done,
                truncated=result.truncated,
                info=info,
            )

            # Log step information
            logger.info(
                f"[STEP] env_id={env_id}, step={result.observation.step_count}, "
                f"action={action_type}, reward={result.reward.value:.3f}, "
                f"done={result.done}, violations={len(result.observation.violations)}"
            )

            # Clean up ended environments
            if result.done or result.truncated:
                # Keep environment for potential reuse, but mark as ended
                pass

            return response_data

        except asyncio.TimeoutError:
            logger.error(f"Step execution timed out for env_id={env_id}")
            return get_error_response(
                error_msg="Step execution timed out",
                error_detail="LLM call took too long",
            )
        except ValueError as ve:
            logger.warning(f"Invalid action: {ve}")
            return get_error_response(
                error_msg=f"Invalid action: {ve}",
                error_detail="Please check action_type and edited_content",
            )
        except RuntimeError as re:
            logger.warning(f"Environment error: {re}")
            return get_error_response(
                error_msg=f"Environment error: {re}",
                error_detail="Environment may need to be reset",
            )
        except Exception as step_error:
            logger.exception(f"Step execution failed: {step_error}")
            return get_error_response(
                error_msg="Step execution failed",
                error_detail=str(step_error),
            )

    except Exception as e:
        logger.exception(f"Unexpected error in /step: {e}")
        return get_error_response(
            error_msg="Unexpected error during step",
            error_detail=str(e),
        )


@app.get("/spec", response_model=dict)
async def get_spec():
    """
    Get OpenEnv environment specification.

    Returns:
        Environment specification including action and observation spaces.
    """
    return {
        "name": "ContentComplianceEnv",
        "version": "1.0.0",
        "type": "content_moderation",
        "action_space": {
            "type": "discrete",
            "actions": ["approve", "reject", "edit"],
            "descriptions": {
                "approve": "Content passes compliance (terminal action)",
                "reject": "Content violates policies (terminal action)",
                "edit": "Modify content to fix violations (continues episode)",
            },
        },
        "observation_space": {
            "type": "dict",
            "fields": {
                "content": {"type": "str", "description": "Text content to evaluate"},
                "violations": {"type": "list[str]", "description": "Detected violation types"},
                "score": {"type": "float", "range": [0.0, 1.0], "description": "Compliance score"},
                "step_count": {"type": "int", "min": 0, "description": "Current step number"},
                "stage": {"type": "str", "enum": ["classification", "decision", "refinement"]},
                "previous_action": {"type": "str", "nullable": True},
                "action_history": {"type": "list[str]"},
            },
        },
        "reward_range": [0.0, 1.0],
        "reward_components": {
            "correct_action": "Correct approve/reject: +1.0",
            "false_positive": "Reject clean content: -0.5",
            "false_negative": "Approve spam: -1.0",
            "step_penalty": "Per step: -0.05",
            "loop_penalty": "Repeated action: -0.2",
            "early_termination_bonus": "Correct in 1-2 steps: +0.2",
        },
        "termination_conditions": [
            "Correct terminal action (approve clean / reject spam)",
            "Max steps reached",
        ],
    }


@app.get("/metrics", response_model=dict)
async def get_metrics():
    """
    Get server metrics and statistics.

    Returns:
        Server performance and usage metrics.
    """
    uptime = (datetime.now() - app.state.start_time).total_seconds()

    return {
        "server": {
            "uptime_seconds": uptime,
            "start_time": app.state.start_time.isoformat(),
            "timestamp": datetime.now().isoformat(),
        },
        "usage": {
            "total_requests": getattr(app.state, "request_count", 0),
            "total_episodes": getattr(app.state, "episode_count", 0),
            "total_steps": getattr(app.state, "step_count", 0),
            "active_environments": len(getattr(app.state, "environments", {})),
        },
        "configuration": {
            "env_pool_size": getattr(app.state, "env_pool_size", 5),
            "evaluator_provider": os.getenv("EVALUATOR_PROVIDER", "mock"),
            "environment": os.getenv("ENVIRONMENT", "development"),
        },
    }


# =============================================================================
# OpenEnv Entry Point
# =============================================================================

def main():
    """Run the server with uvicorn."""
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    log_level = os.getenv("LOG_LEVEL", "info")

    logger.info(f"Starting server on {host}:{port} with log_level={log_level}")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=log_level,
        access_log=True,
    )


if __name__ == "__main__":
    main()
