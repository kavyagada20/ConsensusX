"""Unit tests for xAI Grok LLM client and factory."""

import json
from pathlib import Path
import httpx
import pytest

from src.llm.base import LLMMessage
from src.llm.factory import get_llm_client
from src.llm.xai import XAILLMClient


def test_factory_returns_xai_and_grok():
    """Verify factory instantiates XAILLMClient for 'xai' and 'grok' aliases."""
    client_xai = get_llm_client(provider="xai", api_key="dummy_key_for_test")
    assert isinstance(client_xai, XAILLMClient)
    assert client_xai.provider == "xai"
    assert client_xai.model_name == "grok-2"

    client_grok = get_llm_client(provider="grok", model_name="grok-4.7", api_key="dummy_key_for_test")
    assert isinstance(client_grok, XAILLMClient)
    assert client_grok.model_name == "grok-4.7"


def test_xai_missing_api_key_raises_informative_error(monkeypatch):
    """Verify that calling generate() without an API key raises a clear ValueError without leaking secrets."""
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    client = XAILLMClient(api_key=None)
    client.api_key = None

    with pytest.raises(ValueError) as excinfo:
        client.generate(prompt="Is the sky blue?")

    msg = str(excinfo.value)
    assert "XAI_API_KEY is not configured" in msg
    assert "provider='mock'" in msg
    # Assert no secret or token is in the error message
    assert "Bearer" not in msg


def test_xai_mocked_http_generate(tmp_path: Path):
    """Verify XAILLMClient parses OpenAI-compatible response and tracks tokens, cost, and caching."""
    mock_response_payload = {
        "id": "chatcmpl-test-123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "grok-2",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": '{"predicted_label": "SUPPORTS", "confidence": 0.98, "reasoning": "Empirical evidence confirms it."}'
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 50,
            "completion_tokens": 25,
            "total_tokens": 75
        }
    }

    call_count = 0

    def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        assert request.url == "https://api.x.ai/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer mock_xai_key_12345"
        assert request.headers["Content-Type"] == "application/json"
        body = json.loads(request.content)
        assert body["model"] == "grok-2"
        assert body["messages"][-1]["content"] == "Is water wet?"
        return httpx.Response(200, json=mock_response_payload)

    mock_http = httpx.Client(transport=httpx.MockTransport(mock_handler))

    client = XAILLMClient(
        model_name="grok-2",
        api_key="mock_xai_key_12345",
        cache_enabled=True,
        cache_dir=tmp_path / "cache",
        http_client=mock_http
    )

    resp = client.generate(prompt="Is water wet?", system_instruction="Fact check.")
    assert resp.provider == "xai"
    assert resp.model == "grok-2"
    assert "SUPPORTS" in resp.content
    assert resp.prompt_tokens == 50
    assert resp.completion_tokens == 25
    assert resp.total_tokens == 75
    # Pricing for grok-2: $2.00/1M prompt, $10.00/1M completion
    # (50/1M)*2.00 = 0.0001; (25/1M)*10.00 = 0.00025 -> total = 0.00035
    assert resp.estimated_cost_usd == 0.00035
    assert resp.latency_seconds >= 0.0
    assert call_count == 1

    # Second call should hit the disk cache without calling the HTTP transport
    cached_resp = client.generate(prompt="Is water wet?", system_instruction="Fact check.")
    assert cached_resp.content == resp.content
    assert call_count == 1  # transport not called again!


def test_xai_chat_completion(tmp_path: Path):
    """Verify XAILLMClient chat method handles multi-turn LLMMessage history."""
    mock_payload = {
        "id": "chatcmpl-test-456",
        "object": "chat.completion",
        "model": "grok-4.7",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Chat response verified."},
                "finish_reason": "stop"
            }
        ],
        "usage": {"prompt_tokens": 30, "completion_tokens": 10, "total_tokens": 40}
    }

    def mock_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert len(body["messages"]) == 2
        assert body["messages"][0]["role"] == "system"
        assert body["messages"][1]["role"] == "user"
        return httpx.Response(200, json=mock_payload)

    mock_http = httpx.Client(transport=httpx.MockTransport(mock_handler))
    client = XAILLMClient(
        model_name="grok-4.7",
        api_key="mock_key",
        cache_enabled=False,
        http_client=mock_http
    )

    messages = [
        LLMMessage(role="system", content="You are a factual verifier."),
        LLMMessage(role="user", content="Verify this claim.")
    ]
    resp = client.chat(messages=messages)
    assert resp.content == "Chat response verified."
    assert resp.prompt_tokens == 30
    assert resp.completion_tokens == 10


def test_xai_http_error_handling():
    """Verify that HTTP errors (e.g. 401 Unauthorized) raise RuntimeError with clean message and no leaked tokens."""
    def error_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={"error": {"message": "Incorrect API key provided.", "type": "invalid_request_error"}}
        )

    mock_http = httpx.Client(transport=httpx.MockTransport(error_handler))
    client = XAILLMClient(
        model_name="grok-2",
        api_key="bad_secret_key_to_protect",
        cache_enabled=False,
        http_client=mock_http
    )

    with pytest.raises(RuntimeError) as excinfo:
        client.generate(prompt="Hello")

    err_str = str(excinfo.value)
    assert "xAI API error (HTTP 401)" in err_str
    assert "Incorrect API key provided" in err_str
    # Ensure raw secret key is NOT leaked in the exception text
    assert "bad_secret_key_to_protect" not in err_str

