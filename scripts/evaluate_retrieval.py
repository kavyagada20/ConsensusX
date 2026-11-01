"""CLI script to evaluate evidence retrieval performance (Recall@k, MRR@k, nDCG@k)."""

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import FEVERDataLoader
from src.retrieval.embeddings import get_embedding_model
from src.retrieval.evaluation import evaluate_retrieval
from src.retrieval.retriever import FAISSRetriever
from src.retrieval.vector_store import FAISSVectorStore
from src.utils.logging import get_logger

logger = get_logger("evaluate_retrieval")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate evidence retrieval on FEVER benchmark.")
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/dev_sample_300.jsonl",
        help="Path to evaluation dataset JSONL."
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default="data/processed/faiss_index",
        help="Path to directory containing index.faiss and metadata.json."
    )
    parser.add_argument(
        "--embedding-provider",
        type=str,
        default="mock",
        choices=["mock", "gemini", "sentence_transformer"],
        help="Embedding provider used to construct the index."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Rank threshold k for evaluation (default: 5)."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/retrieval_evaluation.json",
        help="Path to save evaluation summary JSON."
    )

    args = parser.parse_args()

    # Load evaluation records
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error("Dataset file not found: %s", data_path)
        sys.exit(1)

    logger.info("Loading records from: %s", data_path)
    loader = FEVERDataLoader(data_path)
    records = loader.load_all()

    # Load vector store
    emb_model = get_embedding_model(provider=args.embedding_provider)
    vector_store = FAISSVectorStore.load(args.index_dir, embedding_model=emb_model)
    retriever = FAISSRetriever(vector_store=vector_store, default_top_k=args.top_k)

    logger.info("Evaluating retrieval for %d records at k=%d...", len(records), args.top_k)
    metrics = evaluate_retrieval(retriever=retriever, records=records, top_k=args.top_k)

    print("\n" + "=" * 65)
    print(f"             ConsensusX Retrieval Evaluation (k={args.top_k})")
    print("=" * 65)
    print(f"Total Verifiable Queries:  {metrics.total_queries}")
    print(f"Recall@{args.top_k}:                 {metrics.recall_at_k * 100:.2f}%")
    print(f"MRR@{args.top_k}:                    {metrics.mrr_at_k:.4f}")
    print(f"nDCG@{args.top_k}:                   {metrics.ndcg_at_k:.4f}")
    print(f"Precision@{args.top_k}:              {metrics.precision_at_k:.4f}")
    print(f"Unretrieved Queries:       {metrics.unretrieved_queries} ({metrics.unretrieved_queries / metrics.total_queries * 100:.1f}%)")
    print("=" * 65 + "\n")

    out_file = Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(metrics.model_dump(), f, indent=2)

    logger.info("Saved retrieval metrics report to: %s", out_file)


if __name__ == "__main__":
    main()

