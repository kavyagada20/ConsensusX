"""Evidence retrieval, vector store, embedding models, and RAG components for ConsensusX."""

from src.retrieval.schema import (
    EvidencePassage,
    RetrievedPassage,
    RetrievalQueryResult,
    RetrievalMetrics,
    CorpusManifest,
)
from src.retrieval.corpus import EvidenceCorpus, generate_dev_evidence_corpus
from src.retrieval.embeddings import (
    BaseEmbeddingModel,
    HashEmbeddingModel,
    GeminiEmbeddingModel,
    get_embedding_model,
)
from src.retrieval.vector_store import FAISSVectorStore
from src.retrieval.retriever import BaseRetriever, FAISSRetriever, OracleRetriever
from src.retrieval.evaluation import evaluate_retrieval

__all__ = [
    "EvidencePassage",
    "RetrievedPassage",
    "RetrievalQueryResult",
    "RetrievalMetrics",
    "CorpusManifest",
    "EvidenceCorpus",
    "generate_dev_evidence_corpus",
    "BaseEmbeddingModel",
    "HashEmbeddingModel",
    "GeminiEmbeddingModel",
    "get_embedding_model",
    "FAISSVectorStore",
    "BaseRetriever",
    "FAISSRetriever",
    "OracleRetriever",
    "evaluate_retrieval",
]

