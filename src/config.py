"""Configuration management for ConsensusX."""

from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global configuration settings for ConsensusX.
    
    Reads from environment variables and an optional .env file.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Project metadata
    project_name: str = "ConsensusX"
    random_seed: int = 42

    # LLM Settings
    default_llm_provider: str = Field(default="mock", description="mock, gemini, or openai")
    default_llm_model: str = "gemini-1.5-flash"
    gemini_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024
    llm_request_timeout: int = 30
    enable_llm_cache: bool = True
    cache_dir: Path = Path(".cache/llm_responses")

    # Logging & Paths
    log_level: str = "INFO"
    fever_raw_data_dir: Path = Path("data/raw/fever")
    fever_processed_data_dir: Path = Path("data/processed")
    outputs_dir: Path = Path("outputs")

    # Experiment settings
    default_sample_size: int = 300


# Global singleton instance
settings = Settings()

