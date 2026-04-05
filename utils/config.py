"""
Configuration Management for Content Compliance RL.

Loads configuration from environment variables with sensible defaults.
"""

import os
from dataclasses import dataclass
from typing import Any


@dataclass
class Config:
    """Application configuration."""

    api_base_url: str = "https://api.openai.com/v1"
    model_name: str = "gpt-4o"
    openai_api_key: str = ""
    max_steps: int = 5
    difficulty: str = "mixed"
    evaluator_provider: str = "mock"
    host: str = "0.0.0.0"
    port: int = 7860
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        return cls(
            api_base_url=os.getenv("API_BASE_URL", "https://api.openai.com/v1"),
            model_name=os.getenv("MODEL_NAME", "gpt-4o"),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            max_steps=int(os.getenv("MAX_STEPS", "5")),
            difficulty=os.getenv("DIFFICULTY", "mixed"),
            evaluator_provider=os.getenv("EVALUATOR_PROVIDER", "mock"),
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "7860")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "api_base_url": self.api_base_url,
            "model_name": self.model_name,
            "openai_api_key": "***" if self.openai_api_key else "",
            "max_steps": self.max_steps,
            "difficulty": self.difficulty,
            "evaluator_provider": self.evaluator_provider,
            "host": self.host,
            "port": self.port,
            "log_level": self.log_level,
        }


_config: Config | None = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config.from_env()
    return _config
