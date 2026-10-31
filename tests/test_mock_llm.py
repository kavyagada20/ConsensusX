"""Unit tests for Mock LLM client."""

from src.llm.mock import MockLLMClient
from src.llm.base import LLMMessage


def test_mock_llm_generate():
    """Verify mock LLM generates structured response deterministically."""
    client = MockLLMClient(default_verdict="REFUTES")
    resp = client.generate(prompt="Is the sky green?")
    assert resp.provider == "mock"
    assert "REFUTES" in resp.content
    assert resp.prompt_tokens > 0
    assert resp.completion_tokens > 0
    assert len(client.call_history) == 1


def test_mock_llm_chat():
    """Verify mock LLM handles chat messages."""
    client = MockLLMClient(default_verdict="SUPPORTS")
    messages = [
        LLMMessage(role="system", content="You are a factual verifier."),
        LLMMessage(role="user", content="The earth orbits the sun.")
    ]
    resp = client.chat(messages=messages)
    assert resp.provider == "mock"
    assert "SUPPORTS" in resp.content
    assert len(client.call_history) == 1

