"""Factory for creating embedding models based on configuration."""

from typing import Optional

from src.retrieval.embeddings.base import BaseEmbeddingModel
from src.retrieval.embeddings.gemini import GeminiEmbeddingModel
from src.retrieval.embeddings.mock import HashEmbeddingModel


def get_embedding_model(
    provider: str = "mock",
    model_name: Optional[str] = None,
    dimension: Optional[int] = None,
    **kwargs
) -> BaseEmbeddingModel:
    """Instantiates a configured embedding model.
    
    Args:
        provider: Provider identifier ('mock', 'hash', 'gemini', 'sentence_transformer').
        model_name: Optional specific model name.
        dimension: Optional embedding vector dimension.
        **kwargs: Provider-specific arguments.
        
    Returns:
        BaseEmbeddingModel: Configured embedding model instance.
    """
    prov = (provider or "mock").lower().strip()

    if prov in ["mock", "hash", "offline"]:
        dim = dimension or 128
        name = model_name or f"hash-embedding-{dim}"
        return HashEmbeddingModel(dimension=dim, model_name=name)

    elif prov in ["gemini", "google"]:
        name = model_name or "text-embedding-004"
        dim = dimension or 768
        return GeminiEmbeddingModel(model_name=name, dimension=dim, **kwargs)

    elif prov in ["sentence_transformer", "huggingface", "local"]:
        try:
            from sentence_transformers import SentenceTransformer
            # Wrap in adapter if installed
            class SentenceTransformerAdapter(BaseEmbeddingModel):
                def __init__(self, m_name: str):
                    self.st = SentenceTransformer(m_name)
                    dim = self.st.get_sentence_embedding_dimension()
                    super().__init__(model_name=m_name, dimension=dim)
                def embed_texts(self, texts):
                    raw = self.st.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
                    return raw.astype("float32")

            m_name = model_name or "all-MiniLM-L6-v2"
            return SentenceTransformerAdapter(m_name)
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is not installed. Use provider='mock' or provider='gemini', "
                "or install it via 'pip install sentence-transformers'."
            ) from e

    else:
        raise ValueError(
            f"Unsupported embedding provider: '{provider}'. Supported: 'mock', 'gemini', 'sentence_transformer'."
        )

