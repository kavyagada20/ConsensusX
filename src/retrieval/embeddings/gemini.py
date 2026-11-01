"""Google Gen AI text embedding model implementation (e.g. text-embedding-004)."""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, List, Optional
import numpy as np

from src.config import settings
from src.retrieval.embeddings.base import BaseEmbeddingModel
from src.utils.logging import get_logger

logger = get_logger(__name__)


class GeminiEmbeddingModel(BaseEmbeddingModel):
    """Google Gemini embedding model using official google-genai SDK with disk caching."""

    def __init__(
        self,
        model_name: str = "text-embedding-004",
        dimension: int = 768,
        api_key: Optional[str] = None,
        cache_enabled: bool = True,
        cache_dir: Optional[Path] = None,
        batch_size: int = 32
    ):
        super().__init__(model_name=model_name, dimension=dimension)
        self.api_key = api_key or settings.gemini_api_key or os.environ.get("GEMINI_API_KEY")
        self.cache_enabled = cache_enabled
        self.cache_dir = Path(cache_dir or (Path(settings.cache_dir) / "embeddings"))
        self.batch_size = max(1, batch_size)
        if self.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client: Optional[Any] = None

    def _get_client(self) -> Any:
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "GEMINI_API_KEY is not configured for GeminiEmbeddingModel. "
                    "Set GEMINI_API_KEY in .env or pass provider='mock' for offline testing."
                )
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def _get_cache_path(self, text: str) -> Path:
        key_content = f"gemini_emb:{self.model_name}:{text}"
        h = hashlib.sha256(key_content.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{h}.json"

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embeds a list of texts into an L2-normalized numpy array with local caching."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        results: List[Optional[np.ndarray]] = [None] * len(texts)
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []

        if self.cache_enabled:
            for i, text in enumerate(texts):
                p = self._get_cache_path(text)
                if p.exists():
                    try:
                        with open(p, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            results[i] = np.array(data, dtype=np.float32)
                    except Exception as e:
                        logger.warning("Failed to load embedding cache %s: %s", p, e)
                        uncached_indices.append(i)
                        uncached_texts.append(text)
                else:
                    uncached_indices.append(i)
                    uncached_texts.append(text)
        else:
            uncached_indices = list(range(len(texts)))
            uncached_texts = list(texts)

        if uncached_texts:
            client = self._get_client()
            for b_start in range(0, len(uncached_texts), self.batch_size):
                b_texts = uncached_texts[b_start : b_start + self.batch_size]
                b_indices = uncached_indices[b_start : b_start + self.batch_size]

                try:
                    # In google-genai SDK, embed_content accepts contents as string or list
                    response = client.models.embed_content(
                        model=self.model_name,
                        contents=b_texts
                    )
                    embeddings_data = response.embeddings
                    for j, emb_obj in enumerate(embeddings_data):
                        idx = b_indices[j]
                        values = getattr(emb_obj, "values", None) or emb_obj
                        vec = np.array(values, dtype=np.float32)
                        results[idx] = vec
                        if self.cache_enabled:
                            p = self._get_cache_path(texts[idx])
                            try:
                                with open(p, "w", encoding="utf-8") as f:
                                    json.dump(vec.tolist(), f)
                            except Exception as e:
                                logger.warning("Failed writing embedding cache: %s", e)

                except Exception as e:
                    logger.error("Failed embedding batch with Gemini: %s", e)
                    raise RuntimeError(f"Gemini embedding API error: {e}") from e

        # Construct final 2D array
        matrix = np.array([r for r in results if r is not None], dtype=np.float32)
        return self.normalize_vectors(matrix)

