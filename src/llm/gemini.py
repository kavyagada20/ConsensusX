"""Google Gemini LLM client provider using the modern official Google Gen AI SDK (google-genai)."""

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import settings
from src.llm.base import BaseLLMClient, LLMMessage, LLMResponse
from src.llm.pricing import calculate_cost
from src.utils.logging import get_logger

logger = get_logger(__name__)


class GeminiLLMClient(BaseLLMClient):
    """Client for Google Gemini models using the modern google-genai SDK.
    
    Supports gemini-2.0-flash, gemini-1.5-flash, etc., with token tracking,
    cost estimation, latency measurement, error handling, and persistent local caching.
    """

    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        cache_enabled: bool = True,
        cache_dir: Optional[Path] = None
    ):
        super().__init__(model_name=model_name, temperature=temperature, provider="gemini")
        self.api_key = api_key or settings.gemini_api_key or os.environ.get("GEMINI_API_KEY")
        self.cache_enabled = cache_enabled
        self.cache_dir = Path(cache_dir or settings.cache_dir)
        if self.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._client = None
        self.is_mock = False

    def _get_client(self):
        """Lazy initialization of the modern google.genai Client."""
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "GEMINI_API_KEY is not configured. Please set GEMINI_API_KEY in your .env "
                    "or environment variables, or use provider='mock' for offline testing."
                )
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except ImportError as e:
                raise ImportError(
                    "The modern 'google-genai' package is required. Install it via 'pip install google-genai'."
                ) from e
        return self._client

    def _compute_cache_key(self, prompt: str, system_instruction: Optional[str]) -> str:
        """Computes a deterministic hash for prompt + system instruction + model + temp."""
        key_content = f"{self.model_name}:{self.temperature}:{system_instruction or ''}:{prompt}"
        return hashlib.sha256(key_content.encode("utf-8")).hexdigest()

    def _get_cached_response(self, cache_key: str) -> Optional[LLMResponse]:
        """Retrieves cached response from disk if present."""
        if not self.cache_enabled:
            return None
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return LLMResponse(**data)
            except Exception as e:
                logger.warning("Failed to read cache file %s: %s", cache_file, e)
        return None

    def _save_cached_response(self, cache_key: str, response: LLMResponse) -> None:
        """Saves LLM response to disk cache."""
        if not self.cache_enabled:
            return
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(response.model_dump(), f, indent=2)
        except Exception as e:
            logger.warning("Failed to write cache file %s: %s", cache_file, e)

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        max_tokens: int = 2048,
        prompt_version: str = "v1.0.0",
        **kwargs: Any
    ) -> LLMResponse:
        """Generates a text completion for a single prompt."""
        cache_key = self._compute_cache_key(prompt, system_instruction)
        cached = self._get_cached_response(cache_key)
        if cached:
            return cached

        client = self._get_client()
        from google.genai import types

        config = types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=max_tokens,
            system_instruction=system_instruction,
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        )

        max_retries = 3
        for attempt in range(max_retries):
            start_time = time.perf_counter()
            try:
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config
                )
                latency = time.perf_counter() - start_time

                content = response.text or ""
                prompt_tokens = 0
                completion_tokens = 0
                total_tokens = 0

                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    prompt_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
                    completion_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0
                    total_tokens = getattr(response.usage_metadata, "total_token_count", 0) or (prompt_tokens + completion_tokens)

                cost = calculate_cost(
                    model_name=self.model_name,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    provider="gemini"
                )

                llm_response = LLMResponse(
                    content=content,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    estimated_cost_usd=cost,
                    latency_seconds=round(latency, 4),
                    model=self.model_name,
                    provider="gemini",
                    prompt_version=prompt_version
                )

                self._save_cached_response(cache_key, llm_response)
                return llm_response

            except Exception as e:
                err_str = str(e)
                if ("429" in err_str or "RESOURCE_EXHAUSTED" in err_str) and attempt < max_retries - 1:
                    delay = 16.0
                    match = re.search(r"retry in ([\d\.]+)s", err_str)
                    if match:
                        try:
                            delay = float(match.group(1)) + 1.0
                        except Exception:
                            delay = 16.0
                    logger.warning(
                        "Gemini rate limit (429 RESOURCE_EXHAUSTED) hit on Free Tier. "
                        "Backing off for %.1f seconds before retry (attempt %d/%d)...",
                        delay, attempt + 1, max_retries
                    )
                    time.sleep(delay)
                    continue
                logger.error("Gemini API error during generation: %s", e)
                raise

    def chat(
        self,
        messages: List[LLMMessage],
        max_tokens: int = 1024,
        prompt_version: str = "v1.0.0",
        **kwargs: Any
    ) -> LLMResponse:
        """Generates a response from a multi-turn chat history."""
        system_instruction = None
        user_messages: List[str] = []
        for msg in messages:
            if msg.role == "system":
                system_instruction = msg.content
            else:
                user_messages.append(f"{msg.role}: {msg.content}")

        combined_prompt = "\n".join(user_messages)
        return self.generate(
            prompt=combined_prompt,
            system_instruction=system_instruction,
            max_tokens=max_tokens,
            prompt_version=prompt_version,
            **kwargs
        )
