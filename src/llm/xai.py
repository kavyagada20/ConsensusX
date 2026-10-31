"""xAI Grok LLM client provider using OpenAI-compatible HTTP interface."""

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from src.config import settings
from src.llm.base import BaseLLMClient, LLMMessage, LLMResponse
from src.llm.pricing import calculate_cost
from src.utils.logging import get_logger

logger = get_logger(__name__)


class XAILLMClient(BaseLLMClient):
    """Client for xAI Grok models (e.g. grok-2, grok-2-1212, grok-4.7) via OpenAI-compatible API.
    
    Provides token usage tracking, cost calculation, latency measurement,
    persistent disk caching, and error handling without leaking API keys.
    """

    def __init__(
        self,
        model_name: str = "grok-2",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        cache_enabled: bool = True,
        cache_dir: Optional[Path] = None,
        timeout: Optional[int] = None,
        http_client: Optional[httpx.Client] = None
    ):
        super().__init__(model_name=model_name, temperature=temperature, provider="xai")
        self.api_key = api_key or settings.xai_api_key or os.environ.get("XAI_API_KEY")
        self.base_url = (base_url or getattr(settings, "xai_base_url", None) or os.environ.get("XAI_BASE_URL") or "https://api.x.ai/v1").rstrip("/")
        self.timeout = timeout or settings.llm_request_timeout
        self.cache_enabled = cache_enabled
        self.cache_dir = Path(cache_dir or settings.cache_dir)
        if self.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._http_client = http_client
        self.is_mock = False

    def _get_http_client(self) -> httpx.Client:
        """Returns or lazily creates an HTTP client."""
        if self._http_client is None:
            self._http_client = httpx.Client(timeout=self.timeout)
        return self._http_client

    def _compute_cache_key(self, prompt: str, system_instruction: Optional[str]) -> str:
        """Computes a deterministic SHA-256 hash for provider:model:temp:system:prompt."""
        key_content = f"xai:{self.model_name}:{self.temperature}:{system_instruction or ''}:{prompt}"
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
        max_tokens: int = 1024,
        prompt_version: str = "v1.0.0",
        **kwargs: Any
    ) -> LLMResponse:
        """Generates a completion for a prompt via xAI Grok."""
        cache_key = self._compute_cache_key(prompt, system_instruction)
        cached = self._get_cached_response(cache_key)
        if cached:
            return cached

        if not self.api_key:
            raise ValueError(
                "XAI_API_KEY is not configured. Please set XAI_API_KEY in your .env "
                "or environment variables, or use provider='mock' for offline testing."
            )

        messages: List[Dict[str, str]] = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        return self._send_request(
            messages=messages,
            max_tokens=max_tokens,
            prompt_version=prompt_version,
            cache_key=cache_key,
            **kwargs
        )

    def chat(
        self,
        messages: List[LLMMessage],
        max_tokens: int = 1024,
        prompt_version: str = "v1.0.0",
        **kwargs: Any
    ) -> LLMResponse:
        """Generates a response from a multi-turn message history."""
        prompt_repr = "\n".join(f"{m.role}: {m.content}" for m in messages)
        cache_key = self._compute_cache_key(prompt_repr, None)
        cached = self._get_cached_response(cache_key)
        if cached:
            return cached

        if not self.api_key:
            raise ValueError(
                "XAI_API_KEY is not configured. Please set XAI_API_KEY in your .env "
                "or environment variables, or use provider='mock' for offline testing."
            )

        formatted_messages = [{"role": m.role, "content": m.content} for m in messages]
        return self._send_request(
            messages=formatted_messages,
            max_tokens=max_tokens,
            prompt_version=prompt_version,
            cache_key=cache_key,
            **kwargs
        )

    def _send_request(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        prompt_version: str,
        cache_key: str,
        **kwargs: Any
    ) -> LLMResponse:
        """Sends the HTTP request to the xAI endpoint and parses the response."""
        client = self._get_http_client()
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": max_tokens,
            **kwargs
        }

        start_time = time.perf_counter()
        try:
            response = client.post(url, json=payload, headers=headers)
            latency = time.perf_counter() - start_time

            if response.status_code != 200:
                try:
                    err_json = response.json()
                    err_msg = err_json.get("error", {}).get("message", response.text)
                except Exception:
                    err_msg = response.text
                logger.error("xAI API request failed (HTTP %d): %s", response.status_code, err_msg)
                raise RuntimeError(f"xAI API error (HTTP {response.status_code}): {err_msg}")

            data = response.json()
            content = data["choices"][0]["message"]["content"] or ""
            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

            cost = calculate_cost(
                model_name=self.model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                provider="xai"
            )

            llm_response = LLMResponse(
                content=content,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                estimated_cost_usd=cost,
                latency_seconds=round(latency, 4),
                model=self.model_name,
                provider="xai",
                prompt_version=prompt_version,
                raw_response=data
            )

            self._save_cached_response(cache_key, llm_response)
            return llm_response

        except httpx.RequestError as e:
            logger.error("Network or connection error communicating with xAI API: %s", e)
            raise RuntimeError(f"xAI API connection error: {e}") from e

