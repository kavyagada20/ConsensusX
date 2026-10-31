"""Unit tests for GeminiLLMClient and LLM factory."""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from src.llm.gemini import GeminiLLMClient
from src.llm.mock import MockLLMClient
from src.llm.factory import get_llm_client


def test_factory_returns_mock():
    """Verify factory returns MockLLMClient when requested."""
    client = get_llm_client(provider="mock")
    assert isinstance(client, MockLLMClient)


def test_factory_returns_gemini():
    """Verify factory returns GeminiLLMClient when requested."""
    client = get_llm_client(provider="gemini", api_key="dummy_key_for_test")
    assert isinstance(client, GeminiLLMClient)
    assert client.model_name == "gemini-1.5-flash" or client.model_name == "gemini-2.0-flash"


def test_gemini_missing_api_key_raises_informative_error(monkeypatch):
    """Verify that calling generate() without an API key raises a clear ValueError."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    client = GeminiLLMClient(api_key=None)
    client.api_key = None  # Ensure no key

    with pytest.raises(ValueError, match="GEMINI_API_KEY is not configured"):
        client.generate(prompt="Hello")


def test_gemini_mocked_live_call(tmp_path: Path):
    """Verify GeminiLLMClient handles modern google-genai response structure and caching."""
    mock_genai = MagicMock()
    mock_client_instance = MagicMock()
    mock_response = MagicMock()

    mock_response.text = '{"verdict": "SUPPORTS", "reasoning": "Accurate."}'
    mock_response.usage_metadata.prompt_token_count = 15
    mock_response.usage_metadata.candidates_token_count = 10
    mock_response.usage_metadata.total_token_count = 25
    mock_client_instance.models.generate_content.return_value = mock_response

    client = GeminiLLMClient(
        api_key="fake-test-key",
        cache_enabled=True,
        cache_dir=tmp_path / "cache"
    )
    client._client = mock_client_instance

    res = client.generate(prompt="Is water wet?", system_instruction="Fact check.")
    assert res.provider == "gemini"
    assert res.prompt_tokens == 15
    assert res.completion_tokens == 10
    assert "SUPPORTS" in res.content

    # Second call should hit the disk cache without calling client again
    cached_res = client.generate(prompt="Is water wet?", system_instruction="Fact check.")
    assert cached_res.content == res.content
    assert mock_client_instance.models.generate_content.call_count == 1

