"""Factory for creating LLM clients dynamically."""

from typing import Optional
from pathlib import Path
from src.config import settings
from src.llm.base import BaseLLMClient
from src.llm.mock import MockLLMClient
from src.llm.gemini import GeminiLLMClient


def get_llm_client(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    api_key: Optional[str] = None,
    cache_enabled: Optional[bool] = None,
    cache_dir: Optional[Path] = None
) -> BaseLLMClient:
    """Instantiates and returns the configured LLM client.
    
    Args:
        provider: 'mock' or 'gemini' (defaults to settings.default_llm_provider)
        model_name: Model identifier (e.g. 'gemini-2.0-flash', 'gemini-1.5-flash')
        temperature: Sampling temperature
        api_key: Optional explicit API key
        cache_enabled: Whether to enable response caching
        cache_dir: Directory for response cache
        
    Returns:
        BaseLLMClient: The configured LLM client instance.
    """
    provider_name = (provider or settings.default_llm_provider).lower().strip()
    temp = temperature if temperature is not None else settings.llm_temperature
    model = model_name or settings.default_llm_model
    caching = cache_enabled if cache_enabled is not None else settings.enable_llm_cache

    if provider_name == "mock":
        return MockLLMClient(model_name=model)
    elif provider_name == "gemini":
        return GeminiLLMClient(
            model_name=model,
            api_key=api_key,
            temperature=temp,
            cache_enabled=caching,
            cache_dir=cache_dir
        )
    else:
        raise ValueError(
            f"Unsupported LLM provider: '{provider_name}'. Supported providers: 'mock', 'gemini'."
        )

