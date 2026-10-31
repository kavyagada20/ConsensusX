"""Abstract base class and schemas for LLM client providers."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class LLMMessage(BaseModel):
    """Chat completion message."""
    role: str = Field(description="Role of the author: system, user, or assistant")
    content: str = Field(description="Message text content")


class LLMResponse(BaseModel):
    """Standardized response from any LLM provider."""
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: Optional[float] = 0.0
    latency_seconds: float = 0.0
    model: str = ""
    provider: str = ""
    prompt_version: str = "v1.0.0"
    raw_response: Optional[Dict[str, Any]] = None


class BaseLLMClient(ABC):
    """Abstract interface for LLM providers in ConsensusX."""

    def __init__(self, model_name: str = "mock-model", temperature: float = 0.0, provider: str = "unknown"):
        self.model_name = model_name
        self.temperature = temperature
        self.provider = provider

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        max_tokens: int = 1024,
        **kwargs: Any
    ) -> LLMResponse:
        """Generates a text completion for a single prompt."""
        pass

    @abstractmethod
    def chat(
        self,
        messages: List[LLMMessage],
        max_tokens: int = 1024,
        **kwargs: Any
    ) -> LLMResponse:
        """Generates a response for a multi-turn chat message history."""
        pass
