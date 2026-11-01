"""CLI script to build, persist, and query the FAISS vector index for ConsensusX RAG."""

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval.corpus import EvidenceCorpus
from src.retrieval.embeddings import get_embedding_model
from src.retrieval.retriever import FAISSRetriever
from src.retrieval.vector_store import FAISSVectorStore
from src.utils.logging import get_logger

logger = get_logger("build_vector_index")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and persist FAISS vector index for ConsensusX RAG.")
    parser.add_argument(
        "--corpus",
        type=str,
        default="data/processed/dev_corpus.jsonl",
        help="Path to evidence corpus JSONL file."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed/faiss_index",
        help="Destination directory for index.faiss and metadata.json."
    )
    parser.add_argument(
        "--embedding-provider",
        type=str,
        default="mock",
        choices=["mock", "gemini", "sentence_transformer"],
        help="Embedding model provider ('mock' for zero-cost offline, 'gemini' for live)."
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help="Optional specific embedding model ID."
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Optional test query to run after index is built."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of passages to retrieve for test query (default: 5)."
    )

    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        logger.error("Corpus file not found: %s. Run scripts/build_evidence_corpus.py first.", corpus_path)
        sys.exit(1)

    logger.info("Loading evidence corpus from: %s", corpus_path)
    corpus = EvidenceCorpus.load_jsonl(corpus_path)
    passages = corpus.to_list()
    logger.info("Loaded %d passages from corpus.", len(passages))

    # Initialize embedding model
    embedding_model = get_embedding_model(
        provider=args.embedding_provider,
        model_name=args.embedding_model
    )
    logger.info(
        "Using embedding model: '%s' (dimension=%d, provider='%s')",
        embedding_model.model_name, embedding_model.dimension, args.embedding_provider
    )

    # Build and populate vector store
    vector_store = FAISSVectorStore(embedding_model=embedding_model)
    start_time = time.perf_counter()
    added_count = vector_store.add_passages(passages)
    indexing_time = time.perf_counter() - start_time
    logger.info("Indexed %d passages in %.2f seconds.", added_count, indexing_time)

    # Persist index
    out_dir = Path(args.output_dir)
    vector_store.save(out_dir)
    logger.info("Vector index successfully saved to: %s", out_dir)

    # Test query if requested
    if args.query:
        logger.info("Testing retrieval for query: '%s'", args.query)
        retriever = FAISSRetriever(vector_store=vector_store, default_top_k=args.top_k)
        results = retriever.retrieve(args.query, top_k=args.top_k)

        print("\n" + "=" * 80)
        print(f"Retrieval Results for Query: \"{args.query}\"")
        print("=" * 80)
        if not results:
            print("No passages retrieved.")
        for r in results:
            print(f"[{r.rank}] Score: {r.score:.4f} | ID: {r.passage_id}")
            print(f"    Title: {r.title} (Sentence {r.sentence_id})")
            print(f"    Text:  {r.text}")
            print("-" * 80)
        print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

