"""Abstract base class and interface for sentence embedding models."""

from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np


class BaseEmbeddingModel(ABC):
    """Abstract provider-independent embedding model interface."""

    def __init__(self, model_name: str, dimension: int):
        self.model_name = model_name
        self.dimension = dimension

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embeds a list of texts into a 2D float32 numpy array of shape (N, dimension).
        
        Returned embeddings must be L2-normalized so inner product equals cosine similarity.
        """
        pass

    def embed_query(self, query: str) -> np.ndarray:
        """Embeds a single query string into a 1D float32 numpy array of shape (dimension,)."""
        if not query or not query.strip():
            # Return zero vector for empty query
            return np.zeros(self.dimension, dtype=np.float32)
        vectors = self.embed_texts([query])
        return vectors[0]

    @staticmethod
    def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
        """L2-normalizes vectors along the last axis so cosine similarity == dot product."""
        norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0.0, 1.0, norms)
        return (vectors / norms).astype(np.float32)

