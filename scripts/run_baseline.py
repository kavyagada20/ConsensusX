"""CLI entrypoint to run the ConsensusX Single-Agent Baseline (Config 1)."""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import FEVERDataLoader
from src.llm.factory import get_llm_client
from src.pipeline.baseline import BaselinePipeline
from src.utils.logging import get_logger

logger = get_logger("run_baseline")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ConsensusX Single-Agent Baseline (Config 1)."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/dev_sample_300.jsonl",
        help="Path to evaluation jsonl file."
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="mock",
        choices=["mock", "gemini", "xai", "grok"],
        help="LLM provider: 'mock' for zero-cost offline testing, 'gemini' or 'xai' for live API."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model ID (e.g. gemini-2.0-flash, grok-2, grok-2-1212)."
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Optional API key (defaults to GEMINI_API_KEY or XAI_API_KEY from .env)."
    )
    parser.add_argument(
        "--prompt-version",
        type=str,
        default="v1.0.0",
        help="Prompt template version identifier."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of claims to evaluate."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/baseline_results.json",
        help="Destination path for evaluation results JSON."
    )

    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        logger.error("Dataset not found at: %s. Run scripts/prepare_data_sample.py first.", data_path)
        sys.exit(1)

    logger.info("Loading records from: %s", data_path)
    loader = FEVERDataLoader(data_path)
    records = loader.load_all(limit=args.limit)
    logger.info("Loaded %d records for evaluation.", len(records))

    logger.info("Initializing LLM client: provider='%s', model='%s'", args.provider, args.model or "default")
    llm_client = get_llm_client(
        provider=args.provider,
        model_name=args.model,
        api_key=args.api_key
    )

    pipeline = BaselinePipeline(
        llm_client=llm_client,
        prompt_version=args.prompt_version
    )
    summary = pipeline.evaluate_records(records=records, output_file=Path(args.output))

    print("\n" + "=" * 70)
    print("                 ConsensusX: Single-Agent Baseline Summary")
    print("=" * 70)
    if summary["is_mock_mode"]:
        print("  [!] ATTENTION: EXECUTED IN MOCK MODE (Simulated responses).")
        print("      These results validate pipeline functionality, NOT model accuracy.")
        print("-" * 70)

    print(f"Provider:              {summary['provider']}")
    print(f"Model:                 {summary['model']}")
    print(f"Prompt Version:        {summary['prompt_version']}")
    print(f"Evaluated Samples:     {summary['total_samples']}")
    print(f"Overall Accuracy:      {summary['accuracy']:.4f} ({summary['accuracy']*100:.1f}%)")
    print(f"Macro-F1:              {summary['macro_f1']:.4f}")
    print("-" * 70)
    print("Per-Class Classification Metrics:")
    for cls_name, f1_val in summary["per_class_f1"].items():
        prec = summary["per_class_precision"].get(cls_name, 0.0)
        rec = summary["per_class_recall"].get(cls_name, 0.0)
        print(f"  {cls_name:<18} Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1_val:.4f}")
    print("-" * 70)

    # Confusion Matrix display
    cm = summary["confusion_matrix"]
    print("Confusion Matrix (Rows: Ground Truth, Cols: Predicted):")
    labels = cm["labels"]
    print(f"  {'Gold \\ Pred':<18} | " + " | ".join(f"{lbl[:7]:<7}" for lbl in labels))
    print("  " + "-" * 50)
    for gold_lbl in labels:
        row_str = " | ".join(f"{cm['matrix'][gold_lbl].get(pred_lbl, 0):<7}" for pred_lbl in labels)
        print(f"  {gold_lbl:<18} | {row_str}")
    print("-" * 70)

    # Abstention Analysis display
    abst = summary["abstention_analysis"]
    print("Abstention & Insufficient-Evidence Analysis:")
    print(f"  Abstentions (NEI pred):    {abst['abstention_count']} ({abst['abstention_rate']*100:.1f}%)")
    print(f"  Decided Claims:            {abst['decided_count']} (Accuracy on decided: {abst['decided_accuracy']*100:.1f}%)")
    print(f"  Gold NEI Identification:   Recall: {abst['gold_nei_recall']:.4f} ({abst['gold_nei_correctly_abstained']}/{abst['gold_nei_count']})")
    print(f"  False Certainty Count:     {abst['false_certainty_count']} (False certainty rate: {abst['false_certainty_rate']*100:.1f}%)")
    print("-" * 70)

    # Cost & Token Usage display
    costs = summary["cost_and_tokens"]
    perf = summary["performance"]
    print("Efficiency, Latency & Token Usage:")
    print(f"  Avg Latency:               {perf['avg_latency_seconds']:.4f} s / claim")
    print(f"  Avg Tokens:                {costs['avg_tokens_per_claim']} tokens / claim (Prompt: {costs['avg_prompt_tokens']}, Output: {costs['avg_completion_tokens']})")
    print(f"  Total Estimated Cost:      ${costs['total_estimated_cost_usd']:.6f} USD")
    print(f"  Total Elapsed Wall Time:   {perf['wall_time_seconds']} s")
    print("=" * 70)
    print(f"Detailed prediction file saved to: {args.output}\n")


if __name__ == "__main__":
    main()
