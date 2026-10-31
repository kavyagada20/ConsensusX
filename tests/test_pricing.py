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
    # 100,000 prompt + 10,000 completion
    # (100,000 / 1M) * 0.075 = 0.0075
    # (10,000 / 1M) * 0.30 = 0.003
    # Total = 0.0105
    cost_1_5 = calculate_cost("gemini-1.5-flash", prompt_tokens=100_000, completion_tokens=10_000)
    assert cost_1_5 == 0.0105

