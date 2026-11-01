"""FAISS vector store for dense retrieval, persistence, reload, and incremental index updates."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import faiss
import numpy as np

from src.retrieval.embeddings.base import BaseEmbeddingModel
from src.retrieval.schema import EvidencePassage, RetrievedPassage
from src.utils.logging import get_logger

logger = get_logger(__name__)


class FAISSVectorStore:
    """Vector index wrapper for FAISS IndexFlatIP (cosine similarity on L2-normalized embeddings).
    
    Provides index creation, passage addition, top-k vector search, persistence,
    safe reloading, and incremental index updates with stable passage IDs and provenance.
    """

    def __init__(
        self,
        embedding_model: BaseEmbeddingModel,
        index: Optional[faiss.Index] = None,
        id_to_passage: Optional[Dict[int, EvidencePassage]] = None,
        passage_id_to_idx: Optional[Dict[str, int]] = None
    ):
        self.embedding_model = embedding_model
        self.dimension = embedding_model.dimension
        self.index: faiss.Index = index if index is not None else faiss.IndexFlatIP(self.dimension)
        # Maps FAISS integer ID (0 .. N-1) to full EvidencePassage object
        self.id_to_passage: Dict[int, EvidencePassage] = id_to_passage or {}
        # Maps stable passage_id string ("title:sentence_id") to FAISS integer ID
        self.passage_id_to_idx: Dict[str, int] = passage_id_to_idx or {}

    def __len__(self) -> int:
        return self.index.ntotal

    def add_passages(
        self,
        passages: List[EvidencePassage],
        batch_size: int = 128
    ) -> int:
        """Embeds and indexes new passages, automatically skipping already-indexed duplicates.
        
        Args:
            passages: List of EvidencePassage objects to add.
            batch_size: Embedding batch size.
            
        Returns:
            int: Number of new unique passages indexed.
        """
        # Filter duplicates based on stable passage_id
        unique_new = [
            p for p in passages
            if p.passage_id not in self.passage_id_to_idx and p.text and p.text.strip()
        ]
        if not unique_new:
            return 0

        logger.info("Embedding and indexing %d new unique passages...", len(unique_new))

        all_vectors = []
        for i in range(0, len(unique_new), batch_size):
            batch = unique_new[i : i + batch_size]
            texts = [p.text for p in batch]
            vecs = self.embedding_model.embed_texts(texts)
            all_vectors.append(vecs)

        matrix = np.vstack(all_vectors).astype(np.float32)
        start_idx = self.index.ntotal
        self.index.add(matrix)

        for offset, p in enumerate(unique_new):
            faiss_id = start_idx + offset
            self.id_to_passage[faiss_id] = p
            self.passage_id_to_idx[p.passage_id] = faiss_id

        logger.info("Successfully added %d passages. Total index size: %d", len(unique_new), self.index.ntotal)
        return len(unique_new)

    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> List[RetrievedPassage]:
        """Performs dense vector retrieval for a query string.
        
        Args:
            query: Natural language claim or query.
            top_k: Number of highest-similarity passages to return.
            
        Returns:
            List[RetrievedPassage]: Ranked list of retrieved passages with similarity scores.
        """
        if not query or not query.strip():
            logger.debug("Received empty query; returning empty retrieval list.")
            return []

        if self.index.ntotal == 0:
            logger.warning("FAISS vector index is empty; cannot retrieve passages.")
            return []

        query_vec = self.embedding_model.embed_query(query)
        # Reshape to (1, dimension)
        query_matrix = np.expand_dims(query_vec, axis=0).astype(np.float32)

        actual_k = min(max(1, top_k), self.index.ntotal)
        scores, indices = self.index.search(query_matrix, actual_k)

        results: List[RetrievedPassage] = []
        for rank_idx, (score, faiss_id) in enumerate(zip(scores[0], indices[0]), start=1):
            if faiss_id < 0 or faiss_id not in self.id_to_passage:
                continue

            passage = self.id_to_passage[faiss_id]
            results.append(
                RetrievedPassage(
                    passage_id=passage.passage_id,
                    title=passage.title,
                    sentence_id=passage.sentence_id,
                    text=passage.text,
                    score=round(float(score), 5),
                    rank=rank_idx,
                    metadata={
                        **passage.metadata,
                        "source": passage.source,
                        "faiss_id": int(faiss_id)
                    }
                )
            )

        return results

    def save(self, directory_path: Union[str, Path]) -> None:
        """Persists the FAISS index binary and metadata dictionary to disk."""
        target_dir = Path(directory_path)
        target_dir.mkdir(parents=True, exist_ok=True)

        index_file = target_dir / "index.faiss"
        metadata_file = target_dir / "metadata.json"

        # 1. Write FAISS binary
        faiss.write_index(self.index, str(index_file))

        # 2. Write metadata JSON
        meta_payload = {
            "embedding_model": self.embedding_model.model_name,
            "dimension": self.dimension,
            "total_passages": self.index.ntotal,
            "passages": {
                str(faiss_id): passage.model_dump()
                for faiss_id, passage in self.id_to_passage.items()
            }
        }
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(meta_payload, f, indent=2)

        logger.info("Saved FAISS index (%d items) to %s", self.index.ntotal, target_dir)

    @classmethod
    def load(
        cls,
        directory_path: Union[str, Path],
        embedding_model: BaseEmbeddingModel
    ) -> "FAISSVectorStore":
        """Loads a persisted FAISS index and metadata from disk with corruption checks."""
        target_dir = Path(directory_path)
        index_file = target_dir / "index.faiss"
        metadata_file = target_dir / "metadata.json"

        if not index_file.exists() or not metadata_file.exists():
            raise FileNotFoundError(
                f"Missing vector store artifacts in '{target_dir}'. "
                f"Expected '{index_file.name}' and '{metadata_file.name}'."
            )

        # Check for zero-byte or corrupted files
        if index_file.stat().st_size == 0 or metadata_file.stat().st_size == 0:
            raise ValueError(f"Corrupted index files detected in '{target_dir}' (0 bytes).")

        try:
            index = faiss.read_index(str(index_file))
        except Exception as e:
            raise RuntimeError(f"Failed to read FAISS index binary from {index_file}: {e}") from e

        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to parse metadata JSON from {metadata_file}: {e}") from e

        # Validate dimension matches embedding model
        if index.d != embedding_model.dimension:
            raise ValueError(
                f"Dimension mismatch: Index dimension ({index.d}) does not match "
                f"embedding model dimension ({embedding_model.dimension})."
            )

        id_to_passage: Dict[int, EvidencePassage] = {}
        passage_id_to_idx: Dict[str, int] = {}

        for k_str, p_data in meta.get("passages", {}).items():
            faiss_id = int(k_str)
            passage = EvidencePassage(**p_data)
            id_to_passage[faiss_id] = passage
            passage_id_to_idx[passage.passage_id] = faiss_id

        logger.info("Loaded FAISS index with %d items from %s", index.ntotal, target_dir)
        return cls(
            embedding_model=embedding_model,
            index=index,
            id_to_passage=id_to_passage,
            passage_id_to_idx=passage_id_to_idx
        )

