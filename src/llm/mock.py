"""Mock LLM client for zero-cost offline testing and CI/CD."""

import time
from typing import Any, Callable, Dict, List, Optional
from src.llm.base import BaseLLMClient, LLMMessage, LLMResponse


class MockLLMClient(BaseLLMClient):
    """Simulated LLM client that returns deterministic scripted or pattern-matched responses.
    
    Ensures the entire test suite and verification pipeline can be validated
    without network connectivity or paid API credits.
    """

    def __init__(
        self,
        default_verdict: str = "SUPPORTS",
        model_name: str = "mock-agent-v1",
        custom_handler: Optional[Callable[[str], str]] = None
    ):
        super().__init__(model_name=model_name, temperature=0.0, provider="mock")
        self.default_verdict = default_verdict
        self.custom_handler = custom_handler
        self.call_history: List[Dict[str, Any]] = []
        self.is_mock = True

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        max_tokens: int = 1024,
        **kwargs: Any
    ) -> LLMResponse:
        start_time = time.perf_counter()
        self.call_history.append({
            "prompt": prompt,
            "system_instruction": system_instruction,
            "kwargs": kwargs
        })

        if self.custom_handler:
            content = self.custom_handler(prompt)
        else:
            # Standard simulated structured JSON response for claim verification
            content = (
                f'{{"predicted_label": "{self.default_verdict}", '
                f'"confidence": 0.95, '
                f'"reasoning": "Mock verification reasoning based on simulated knowledge.", '
                f'"cited_evidence": ["Mock_Wikipedia_Page:0"]}}'
            )

        latency = time.perf_counter() - start_time
        prompt_tokens = len(prompt.split())
        completion_tokens = len(content.split())
        total_tokens = prompt_tokens + completion_tokens

        return LLMResponse(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=0.0,
            latency_seconds=latency,
            model=self.model_name,
            provider="mock",
            prompt_version="v1.0.0"
        )

    def chat(
        self,
        messages: List[LLMMessage],
        max_tokens: int = 1024,
        **kwargs: Any
    ) -> LLMResponse:
        full_text = " ".join([m.content for m in messages])
        return self.generate(prompt=full_text, max_tokens=max_tokens, **kwargs)
