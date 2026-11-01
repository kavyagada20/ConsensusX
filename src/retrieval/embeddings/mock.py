"""Deterministic offline embedding model based on feature hashing for zero-cost testing and CI."""

import hashlib
import re
from typing import List
import numpy as np

from src.retrieval.embeddings.base import BaseEmbeddingModel


class HashEmbeddingModel(BaseEmbeddingModel):
    """Deterministic, zero-network sentence embedding model.
    
    Uses token hashing and character n-gram projection with L2 normalization to produce
    dense embeddings where shared vocabulary and entity mentions yield higher cosine similarity.
    Ensures complete bitwise reproducibility across machines and runs.
    """

    def __init__(self, dimension: int = 128, model_name: str = "hash-embedding-128"):
        super().__init__(model_name=model_name, dimension=dimension)

    def _text_to_vector(self, text: str) -> np.ndarray:
        """Converts text into a raw dense vector using signed token hashing."""
        vec = np.zeros(self.dimension, dtype=np.float32)
        cleaned = text.strip().lower()
        if not cleaned:
            return vec

        # Extract words and 3-char ngrams
        words = re.findall(r"\w+", cleaned)
        if not words:
            return vec

        tokens = list(words)
        # Add character tri-grams for subword matching
        for word in words:
            if len(word) >= 3:
                for i in range(len(word) - 2):
                    tokens.append(word[i:i+3])

        for token in tokens:
            # Deterministic SHA-256 hash of token
            h = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dimension
            sign = 1.0 if ((h >> 8) & 1) == 0 else -1.0
            vec[idx] += sign

        return vec

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embeds a list of texts into an L2-normalized float32 numpy array."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        raw_vectors = np.array([self._text_to_vector(t) for t in texts], dtype=np.float32)
        return self.normalize_vectors(raw_vectors)

