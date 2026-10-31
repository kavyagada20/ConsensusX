"""Unit tests for ConsensusX configuration and settings."""

from src.config import Settings


def test_default_settings():
    """Verify default settings load properly with valid types and defaults."""
    cfg = Settings()
    assert cfg.project_name == "ConsensusX"
    assert cfg.default_llm_provider in ["mock", "gemini", "xai", "openai"]
    assert cfg.xai_base_url == "https://api.x.ai/v1"
    assert cfg.random_seed == 42
    assert cfg.llm_temperature >= 0.0
    assert cfg.cache_dir is not None

