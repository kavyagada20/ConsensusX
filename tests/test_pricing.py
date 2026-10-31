"""Unit tests for pricing and token cost calculation."""

from src.llm.pricing import calculate_cost, MODEL_PRICING_PER_MILLION


def test_pricing_mock_client():
    """Verify mock provider always has zero cost."""
    cost = calculate_cost("mock-agent-v1", prompt_tokens=1000, completion_tokens=500, provider="mock")
    assert cost == 0.0


def test_pricing_gemini_models():
    """Verify Gemini models cost calculation matching pricing table."""
    # gemini-2.0-flash: $0.10 / 1M prompt, $0.40 / 1M completion
    # 1,000,000 prompt + 1,000,000 completion -> 0.10 + 0.40 = 0.50
    cost_2_0 = calculate_cost("gemini-2.0-flash", prompt_tokens=1_000_000, completion_tokens=1_000_000)
    assert round(cost_2_0, 2) == 0.50

    # gemini-1.5-flash: $0.075 / 1M prompt, $0.30 / 1M completion
    # 100,000 prompt + 10,000 completion -> (100,000 / 1M) * 0.075 + (10,000 / 1M) * 0.30 = 0.0105
    cost_1_5 = calculate_cost("gemini-1.5-flash", prompt_tokens=100_000, completion_tokens=10_000)
    assert cost_1_5 == 0.0105


def test_pricing_xai_grok_models():
    """Verify verified xAI Grok models cost calculation matching official rate card."""
    # grok-2: $2.00 / 1M prompt, $10.00 / 1M completion
    # 100,000 prompt + 10,000 completion -> 0.20 + 0.10 = 0.30
    cost_grok_2 = calculate_cost("grok-2", prompt_tokens=100_000, completion_tokens=10_000, provider="xai")
    assert round(cost_grok_2, 4) == 0.3000

    # grok-4.7: $2.00 / 1M prompt, $6.00 / 1M completion
    # 50,000 prompt + 20,000 completion -> (50,000/1M)*2.00 + (20,000/1M)*6.00 = 0.10 + 0.12 = 0.22
    cost_grok_47 = calculate_cost("grok-4.7", prompt_tokens=50_000, completion_tokens=20_000, provider="xai")
    assert round(cost_grok_47, 4) == 0.2200


def test_pricing_unverified_model_returns_none():
    """Verify that unverified or unknown models return None (cost marked as unavailable)."""
    cost_unknown = calculate_cost("unverified-future-model-v99", prompt_tokens=1000, completion_tokens=500)
    assert cost_unknown is None
