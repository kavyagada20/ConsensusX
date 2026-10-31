"""Pricing and cost estimation utilities for LLM generation."""

from typing import Dict, Optional, Tuple

# Pricing in USD per 1,000,000 tokens: (input_price_per_1m, output_price_per_1m)
# Only verified official rates are included. Unverified models must return None (unavailable).
MODEL_PRICING_PER_MILLION: Dict[str, Tuple[float, float]] = {
    # Google Gemini 2.0
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-2.0-flash-exp": (0.00, 0.00),  # Experimental preview tier
    # Google Gemini 1.5
    "gemini-1.5-flash": (0.075, 0.30),
    "gemini-1.5-flash-8b": (0.0375, 0.15),
    "gemini-1.5-pro": (1.25, 5.00),
    # xAI Grok (Verified from official xAI documentation & API rate cards)
    "grok-2": (2.00, 10.00),
    "grok-2-1212": (2.00, 10.00),
    "grok-2-vision-1212": (2.00, 10.00),
    "grok-4.7": (2.00, 6.00),
    "grok-3-mini": (0.30, 0.50),
    # OpenAI reference (for future comparisons)
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    # Mock offline client
    "mock": (0.0, 0.0),
    "mock-agent-v1": (0.0, 0.0),
}


def calculate_cost(
    model_name: str,
    prompt_tokens: int,
    completion_tokens: int,
    provider: str = ""
) -> Optional[float]:
    """Calculates the estimated cost in USD for a given token usage.
    
    Args:
        model_name: Identifier of the model.
        prompt_tokens: Number of prompt/input tokens.
        completion_tokens: Number of completion/candidate tokens.
        provider: Provider name (e.g. 'mock', 'gemini', 'xai').
        
    Returns:
        Optional[float]: Estimated cost in USD rounded to 6 decimal places,
                         or None if pricing for the model cannot be verified.
    """
    if provider == "mock" or "mock" in model_name.lower():
        return 0.0

    # Normalize model name
    normalized_name = model_name.lower().strip()
    rates: Optional[Tuple[float, float]] = None

    for key, val in MODEL_PRICING_PER_MILLION.items():
        if key in normalized_name:
            rates = val
            break

    if rates is None:
        # If model pricing cannot be verified, return None (marked as unavailable)
        return None

    input_cost = (prompt_tokens / 1_000_000.0) * rates[0]
    output_cost = (completion_tokens / 1_000_000.0) * rates[1]
    return round(input_cost + output_cost, 6)
