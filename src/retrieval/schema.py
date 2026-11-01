"""Pydantic schemas for evidence passages, retrieval results, and RAG evaluation."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field


class EvidencePassage(BaseModel):
    """A single unit of evidence in the ConsensusX corpus (e.g. a Wikipedia sentence)."""
    passage_id: str = Field(
        description="Stable canonical passage identifier, e.g. '{wikipedia_title}:{sentence_id}'"
    )
    title: str = Field(description="Normalized Wikipedia article title")
    sentence_id: int = Field(ge=0, description="Sentence index within the Wikipedia article")
    text: str = Field(description="Verbatim passage text")
    source: str = Field(default="wikipedia", description="Corpus source provenance")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional provenance metadata")

    @classmethod
    def create_id(cls, title: str, sentence_id: int) -> str:
        """Constructs a deterministic, stable passage ID."""
        cleaned_title = str(title).strip()
        return f"{cleaned_title}:{sentence_id}"

    def get_coordinate(self) -> Tuple[str, int]:
        """Returns the (title, sentence_id) tuple coordinate."""
        return (self.title, self.sentence_id)


class RetrievedPassage(BaseModel):
    """A passage retrieved by the vector index or oracle retriever for a given query."""
    passage_id: str = Field(description="Stable passage ID")
    title: str = Field(description="Article title")
    sentence_id: int = Field(ge=0, description="Sentence index")
    text: str = Field(description="Passage text")
    score: float = Field(description="Similarity or relevance score (higher = more similar)")
    rank: int = Field(ge=1, description="1-based rank in the retrieved top-k list")
    is_gold: Optional[bool] = Field(default=None, description="Whether this passage matches gold evidence (for evaluation only)")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievalQueryResult(BaseModel):
    """Container for retrieval results on a single claim query."""
    claim_id: int
    query: str
    top_k: int
    retrieved_passages: List[RetrievedPassage]
    latency_seconds: float = 0.0
    retriever_type: str = "faiss"
    embedding_model: str = "default"


class RetrievalMetrics(BaseModel):
    """Comprehensive evaluation metrics for retrieval performance against gold evidence."""
    total_queries: int
    k: int
    recall_at_k: float = Field(description="Proportion of queries with at least one gold passage in top-k")
    mrr_at_k: float = Field(description="Mean Reciprocal Rank of the first relevant passage in top-k")
    ndcg_at_k: float = Field(description="Normalized Discounted Cumulative Gain at rank k")
    precision_at_k: float = Field(description="Average precision of gold passages in top-k")
    unretrieved_queries: int = Field(description="Queries with zero gold passages in top-k")


class CorpusManifest(BaseModel):
    """Provenance and metadata manifest for an evidence corpus."""
    manifest_version: str = "1.0.0"
    corpus_name: str
    total_passages: int
    unique_titles: int
    source_description: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    output_file: str
    embedding_model_compatible: Optional[str] = None

