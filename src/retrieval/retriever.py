"""Retriever interfaces: FAISS Dense Vector Retriever and Oracle Gold Retriever."""

from abc import ABC, abstractmethod
from typing import List, Optional

from src.data.schema import FEVERRecord
from src.retrieval.corpus import EvidenceCorpus
from src.retrieval.schema import EvidencePassage, RetrievedPassage
from src.retrieval.vector_store import FAISSVectorStore
from src.utils.logging import get_logger

logger = get_logger(__name__)


class BaseRetriever(ABC):
    """Abstract interface for evidence retrieval."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedPassage]:
        """Retrieves top-k evidence passages for a query claim.
        
        Note: The query must contain ONLY the claim text. Gold labels or test answers
        must never influence retrieval.
        """
        pass

    @staticmethod
    def format_evidence_context(passages: List[RetrievedPassage]) -> str:
        """Formats retrieved passages into a structured reference context for verifier and debater agents.
        
        Example output format:
        [1] [Soul_Food_-LRB-film-RRB-:0] (Score: 0.892) Soul Food is a 1997 American comedy-drama film...
        [2] [Fox_2000_Pictures:2] (Score: 0.841) Fox 2000 Pictures released the film Soul Food in 1997...
        """
        if not passages:
            return "No reference evidence retrieved."

        lines = []
        for p in passages:
            citation_tag = f"[{p.title}:{p.sentence_id}]"
            lines.append(f"[{p.rank}] {citation_tag} (Similarity: {p.score:.3f}): {p.text}")
        return "\n".join(lines)


class FAISSRetriever(BaseRetriever):
    """Dense retriever using FAISS vector store.
    
    IMPORTANT RESEARCH NOTE: High semantic similarity between a claim and a passage
    does NOT imply factual support. Contradictory statements often share 90%+ vocabulary
    and high embedding similarity with the claim. Downstream agents must evaluate
    entailment versus refutation independently.
    """

    def __init__(self, vector_store: FAISSVectorStore, default_top_k: int = 5):
        self.vector_store = vector_store
        self.default_top_k = default_top_k

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievedPassage]:
        """Queries the vector index for top-k passages matching the claim."""
        k = top_k or self.default_top_k
        return self.vector_store.search(query=query, top_k=k)


class OracleRetriever(BaseRetriever):
    """Oracle retriever that provides the ground-truth FEVER gold evidence for controlled reasoning tests.
    
    Allows isolating reasoning and debate dynamics from retrieval noise and recall errors.
    """

    def __init__(self, corpus: EvidenceCorpus):
        self.corpus = corpus

    def retrieve_for_record(self, record: FEVERRecord, top_k: int = 5) -> List[RetrievedPassage]:
        """Retrieves the gold evidence passages for a specific FEVER record."""
        gold_sets = record.get_gold_evidence_sets()
        results: List[RetrievedPassage] = []
        seen_ids = set()

        rank = 1
        for g_set in gold_sets:
            for title, sent_id in g_set:
                p_id = EvidencePassage.create_id(title, sent_id)
                if p_id in seen_ids:
                    continue
                seen_ids.add(p_id)

                passage = self.corpus.get(p_id)
                text = passage.text if passage else f"Gold evidence statement for {title} (sentence {sent_id})."
                results.append(
                    RetrievedPassage(
                        passage_id=p_id,
                        title=title,
                        sentence_id=sent_id,
                        text=text,
                        score=1.0,
                        rank=rank,
                        is_gold=True,
                        metadata={"retriever": "oracle_gold", "claim_id": record.id}
                    )
                )
                rank += 1
                if len(results) >= top_k:
                    return results

        return results

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedPassage]:
        """Fallback when record is not available (returns empty)."""
        logger.warning("OracleRetriever called with raw query text without FEVER record; returning empty.")
        return []

