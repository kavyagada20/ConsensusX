"""CLI script to generate a deterministic, stratified development sample from the FEVER dataset."""

import argparse
import sys
from collections import Counter
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import FEVERDataLoader, check_data_leakage
from src.utils.logging import get_logger

logger = get_logger("prepare_data_sample")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract a balanced, stratified sample from FEVER dataset with provenance manifest."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/fever/shared_task_dev.jsonl",
        help="Path to raw FEVER jsonl file."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/dev_sample_300.jsonl",
        help="Path to output processed jsonl file."
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=300,
        help="Total number of samples to extract (divided equally among classes)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling."
    )
    parser.add_argument(
        "--check-leakage-against",
        type=str,
        default=None,
        help="Optional path to a split file (e.g. train.jsonl) to verify zero claim ID overlap."
    )
    parser.add_argument(
        "--no-manifest",
        action="store_true",
        help="Disable creation of provenance manifest JSON file."
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        logger.error("Input file does not exist: %s", input_path)
        sys.exit(1)

    logger.info("Initializing FEVER loader for: %s", input_path)
    loader = FEVERDataLoader(input_path)

    logger.info(
        "Sampling %d balanced records (seed=%d) -> %s",
        args.sample_size, args.seed, output_path
    )
    samples = loader.create_stratified_sample(
        sample_size=args.sample_size,
        random_seed=args.seed,
        output_path=output_path,
        create_manifest=not args.no_manifest
    )

    counts = Counter(s.label.value for s in samples)
    logger.info("Sample generation complete. Distribution: %s", dict(counts))
    assert len(samples) == args.sample_size, f"Expected {args.sample_size}, got {len(samples)}"
    expected_per_class = args.sample_size // 3
    for label, count in counts.items():
        assert count == expected_per_class, f"Class {label} count is {count}, expected {expected_per_class}"
    logger.info("Verification PASSED: Sample is balanced with exactly %d per class.", expected_per_class)

    # Optional Leakage Verification
    if args.check_leakage_against:
        other_path = Path(args.check_leakage_against)
        if other_path.exists():
            logger.info("Checking data leakage against: %s", other_path)
            other_loader = FEVERDataLoader(other_path)
            # Check against records in other file
            check_data_leakage(
                set_a_ids=samples,
                set_b_ids=(r.id for r in other_loader.stream_records(limit=10000)),
                set_a_name="Generated Sample",
                set_b_name=other_path.name
            )
        else:
            logger.warning("File for leakage check does not exist: %s", other_path)


if __name__ == "__main__":
    main()
