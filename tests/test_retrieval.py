"""Unit tests for evidence retrieval, FAISS vector store, embedding models, and RAG evaluation."""

from pathlib import Path
import numpy as np
import pytest

from src.data.schema import FEVERLabel, FEVERRecord
from src.retrieval.corpus import EvidenceCorpus
from src.retrieval.embeddings.factory import get_embedding_model
from src.retrieval.embeddings.mock import HashEmbeddingModel
from src.retrieval.evaluation import evaluate_retrieval
from src.retrieval.retriever import FAISSRetriever, OracleRetriever
from src.retrieval.schema import EvidencePassage
from src.retrieval.vector_store import FAISSVectorStore


def test_evidence_passage_stable_id_and_coordinates():
    """Verify deterministic stable ID creation and coordinate extraction."""
    p = EvidencePassage(
        passage_id=EvidencePassage.create_id("Telemundo", 4),
        title="Telemundo",
        sentence_id=4,
        text="Telemundo broadcasts Spanish programming.",
        source="test"
    )
    assert p.passage_id == "Telemundo:4"
    assert p.get_coordinate() == ("Telemundo", 4)


def test_evidence_corpus_deduplication(tmp_path: Path):
    """Verify EvidenceCorpus deduplicates passages by stable ID and serializes to JSONL with manifest."""
    corpus = EvidenceCorpus()
    p1 = EvidencePassage(passage_id="DocA:0", title="DocA", sentence_id=0, text="First text")
    p2 = EvidencePassage(passage_id="DocA:0", title="DocA", sentence_id=0, text="Duplicate ID text")
    p3 = EvidencePassage(passage_id="DocB:1", title="DocB", sentence_id=1, text="Second text")

    added = corpus.add_passages([p1, p2, p3])
    assert added == 2
    assert len(corpus) == 2
    assert corpus.get("DocA:0").text == "First text"

    # Save and reload
    out_file = tmp_path / "corpus.jsonl"
    manifest = corpus.save_jsonl(out_file)
    assert manifest.total_passages == 2
    assert manifest.unique_titles == 2
    assert out_file.exists()

    loaded = EvidenceCorpus.load_jsonl(out_file)
    assert len(loaded) == 2
    assert loaded.get("DocB:1").text == "Second text"


def test_hash_embedding_model_properties():
    """Verify HashEmbeddingModel produces L2-normalized vectors and semantic overlap."""
    model = HashEmbeddingModel(dimension=64)
    texts = [
        "The Roman Empire fell in 476 AD.",
        "Ancient Rome and the Roman Empire collapsed in 476 AD.",
        "Photosynthesis converts sunlight into glucose in green plants."
    ]

    embs = model.embed_texts(texts)
    assert embs.shape == (3, 64)

    # Check L2 normalization (length == 1.0)
    for i in range(3):
        norm = np.linalg.norm(embs[i])
        assert abs(norm - 1.0) < 1e-5

    # Semantic similarity: Roman Empire texts should have higher similarity than photosynthesis
    sim_roman = np.dot(embs[0], embs[1])
    sim_unrelated = np.dot(embs[0], embs[2])
    assert sim_roman > sim_unrelated

    # Empty query yields zero vector
    empty_vec = model.embed_query("")
    assert np.all(empty_vec == 0.0)


def test_faiss_vector_store_add_search_persistence(tmp_path: Path):
    """Verify FAISSVectorStore indexes passages, retrieves top-k, persists, and reloads."""
    emb_model = HashEmbeddingModel(dimension=64)
    store = FAISSVectorStore(embedding_model=emb_model)

    passages = [
        EvidencePassage(passage_id="Rome:1", title="Rome", sentence_id=1, text="The Roman Empire collapsed in 476 AD."),
        EvidencePassage(passage_id="Earth:0", title="Earth", sentence_id=0, text="Earth orbits around the Sun in 365 days."),
        EvidencePassage(passage_id="Moon:0", title="Moon", sentence_id=0, text="The Moon is Earth's only natural satellite.")
    ]

    added = store.add_passages(passages)
    assert added == 3
    assert len(store) == 3

    # Query search
    results = store.search("When did the Roman Empire collapse?", top_k=2)
    assert len(results) == 2
    assert results[0].passage_id == "Rome:1"
    assert results[0].rank == 1
    assert results[0].score > 0.0

    # Persistence
    save_dir = tmp_path / "faiss_test"
    store.save(save_dir)
    assert (save_dir / "index.faiss").exists()
    assert (save_dir / "metadata.json").exists()

    # Reloading
    loaded_store = FAISSVectorStore.load(save_dir, embedding_model=emb_model)
    assert len(loaded_store) == 3

    reloaded_results = loaded_store.search("When did the Roman Empire collapse?", top_k=2)
    assert reloaded_results[0].passage_id == "Rome:1"


def test_faiss_vector_store_incremental_add():
    """Verify incremental passage addition updates FAISS index and metadata without duplicate indexing."""
    emb_model = HashEmbeddingModel(dimension=32)
    store = FAISSVectorStore(embedding_model=emb_model)

    p1 = EvidencePassage(passage_id="P1:0", title="P1", sentence_id=0, text="First statement.")
    p2 = EvidencePassage(passage_id="P2:0", title="P2", sentence_id=0, text="Second statement.")

    assert store.add_passages([p1]) == 1
    assert len(store) == 1

    # Adding p1 again should skip it; p2 should be added
    assert store.add_passages([p1, p2]) == 1
    assert len(store) == 2


def test_faiss_vector_store_edge_cases(tmp_path: Path):
    """Verify edge case handling: empty query, empty index, and corrupted index files."""
    emb_model = HashEmbeddingModel(dimension=32)
    empty_store = FAISSVectorStore(embedding_model=emb_model)

    # Empty index search
    assert empty_store.search("Any query") == []

    # Empty query string
    assert empty_store.search("") == []
    assert empty_store.search("   ") == []

    # Corrupted index detection (0-byte file)
    corrupted_dir = tmp_path / "corrupted"
    corrupted_dir.mkdir()
    (corrupted_dir / "index.faiss").touch()  # 0-byte file
    (corrupted_dir / "metadata.json").touch()

    with pytest.raises(ValueError) as exc:
        FAISSVectorStore.load(corrupted_dir, embedding_model=emb_model)
    assert "Corrupted index files detected" in str(exc.value)


def test_faiss_retriever_formatting():
    """Verify FAISSRetriever formats evidence context with citation tags."""
    emb_model = HashEmbeddingModel(dimension=32)
    store = FAISSVectorStore(embedding_model=emb_model)
    p = EvidencePassage(passage_id="TitleA:2", title="TitleA", sentence_id=2, text="Factual evidence line.")
    store.add_passages([p])

    retriever = FAISSRetriever(vector_store=store, default_top_k=1)
    retrieved = retriever.retrieve("Query line")
    context_str = retriever.format_evidence_context(retrieved)

    assert "[TitleA:2]" in context_str
    assert "Factual evidence line." in context_str
    assert "Similarity:" in context_str


def test_oracle_retriever_provides_exact_gold_passages():
    """Verify OracleRetriever extracts gold evidence pairs and skips ungrounded NEI records."""
    corpus = EvidenceCorpus([
        EvidencePassage(passage_id="PageX:0", title="PageX", sentence_id=0, text="Exact gold sentence.")
    ])
    oracle = OracleRetriever(corpus=corpus)

    rec_verifiable = FEVERRecord(
        id=701,
        claim="Claim about PageX.",
        label=FEVERLabel.SUPPORTS,
        evidence=[[[1, 1, "PageX", 0]]]
    )
    gold_results = oracle.retrieve_for_record(rec_verifiable, top_k=5)
    assert len(gold_results) == 1
    assert gold_results[0].passage_id == "PageX:0"
    assert gold_results[0].is_gold is True

    # NEI record has no gold evidence
    rec_nei = FEVERRecord(
        id=702,
        claim="Unverifiable claim.",
        label=FEVERLabel.NOT_ENOUGH_INFO,
        evidence=[[[1, None, None, None]]]
    )
    nei_results = oracle.retrieve_for_record(rec_nei, top_k=5)
    assert len(nei_results) == 0


def test_evaluate_retrieval_metrics():
    """Verify evaluate_retrieval correctly computes Recall@k, MRR@k, and nDCG@k."""
    emb_model = HashEmbeddingModel(dimension=64)
    store = FAISSVectorStore(embedding_model=emb_model)

    # Add targeted passage
    store.add_passages([
        EvidencePassage(passage_id="Biology:1", title="Biology", sentence_id=1, text="Photosynthesis occurs in plants."),
        EvidencePassage(passage_id="Physics:0", title="Physics", sentence_id=0, text="Electrons have negative charge.")
    ])
    retriever = FAISSRetriever(vector_store=store, default_top_k=2)

    records = [
        FEVERRecord(
            id=801,
            claim="Photosynthesis happens in plants.",
            label=FEVERLabel.SUPPORTS,
            evidence=[[[1, 1, "Biology", 1]]]
        ),
        FEVERRecord(
            id=802,
            claim="Electrons possess negative charge.",
            label=FEVERLabel.SUPPORTS,
            evidence=[[[2, 2, "Physics", 0]]]
        )
    ]

    metrics = evaluate_retrieval(retriever=retriever, records=records, top_k=2)
    assert metrics.total_queries == 2
    assert metrics.recall_at_k == 1.0  # Both found in top 2
    assert metrics.mrr_at_k > 0.0
    assert metrics.ndcg_at_k > 0.0
    assert metrics.unretrieved_queries == 0

