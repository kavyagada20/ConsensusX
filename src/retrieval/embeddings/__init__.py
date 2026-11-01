"""Embedding models and interfaces for ConsensusX retrieval."""

from src.retrieval.embeddings.base import BaseEmbeddingModel
from src.retrieval.embeddings.mock import HashEmbeddingModel
from src.retrieval.embeddings.gemini import GeminiEmbeddingModel
from src.retrieval.embeddings.factory import get_embedding_model

__all__ = [
    "BaseEmbeddingModel",
    "HashEmbeddingModel",
    "GeminiEmbeddingModel",
    "get_embedding_model",
]

