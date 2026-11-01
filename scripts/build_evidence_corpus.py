"""CLI script to build and serialize the reproducible evidence corpus for ConsensusX RAG."""

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import FEVERDataLoader
from src.retrieval.corpus import EvidenceCorpus, generate_dev_evidence_corpus
from src.utils.logging import get_logger

logger = get_logger("build_evidence_corpus")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and serialize evidence corpus for ConsensusX.")
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/dev_sample_300.jsonl",
        help="Path to evaluation sample records (default: data/processed/dev_sample_300.jsonl)."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/dev_corpus.jsonl",
        help="Destination path for evidence corpus JSONL."
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        logger.error("Sample file not found: %s", data_path)
        sys.exit(1)

    logger.info("Loading records from: %s", data_path)
    loader = FEVERDataLoader(data_path)
    records = loader.load_all()

    logger.info("Generating development evidence corpus for %d claims...", len(records))
    corpus = generate_dev_evidence_corpus(sample_records=records, output_path=args.output)
    logger.info(
        "Evidence corpus successfully built! Total passages: %d, output: %s",
        len(corpus), args.output
    )


if __name__ == "__main__":
    main()

