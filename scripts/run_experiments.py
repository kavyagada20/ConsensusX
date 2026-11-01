"""Unified CLI runner to execute and compare ConsensusX experimental configurations:
Config 1: Single-Agent Baseline
Config 2: Multi-Agent Independent Majority Voting
Config 3: Multi-Agent Structured Debate
Config 4: ConsensusX (Structured Debate + RAG + Evidence-Based Judge + Adaptive Stopping)
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.data.loader import FEVERDataLoader
from src.data.schema import FEVERRecord
from src.llm.factory import get_llm_client
from src.pipeline.baseline import BaselinePipeline
from src.pipeline.consensus_x import ConsensusXPipeline
from src.pipeline.debate import DebatePipeline
from src.pipeline.voting import MajorityVotingPipeline
from src.utils.logging import get_logger

logger = get_logger("run_experiments")


def print_comparison_table(results_by_config: Dict[str, Dict[str, Any]]) -> None:
    """Prints a structured ASCII comparison table of benchmark configurations."""
    print("\n" + "=" * 105)
    print("                              ConsensusX Benchmark Comparison Summary")
    print("=" * 105)
    print(f"{'Configuration':<34} | {'Accuracy':<9} | {'Macro-F1':<9} | {'FCR':<7} | {'Grounding':<10} | {'Avg Lat':<8} | {'Cost':<9} | {'Rounds':<6}")
    print("-" * 105)

    for cfg_name, summary in results_by_config.items():
        acc = f"{summary.get('accuracy', 0.0) * 100:.1f}%"
        f1 = f"{summary.get('macro_f1', 0.0):.4f}"
        fcr_val = summary.get("false_consensus_rate")
        fcr = f"{fcr_val * 100:.1f}%" if fcr_val is not None else "N/A"
        grd_val = summary.get("citation_metrics", {}).get("grounding_rate")
        grd = f"{grd_val * 100:.1f}%" if grd_val is not None else "N/A"
        lat = f"{summary.get('performance', {}).get('avg_latency_per_claim', 0.0):.2f}s"
        cost = f"${summary.get('token_usage', {}).get('total_estimated_cost_usd', 0.0):.4f}"
        rounds_val = summary.get("adaptive_stopping", {}).get("avg_rounds_used")
        if rounds_val is None:
            rounds_val = 1 if "Baseline" in cfg_name or "Voting" in cfg_name else summary.get("max_rounds", "N/A")
        rounds = f"{rounds_val}"
        print(f"{cfg_name:<34} | {acc:<9} | {f1:<9} | {fcr:<7} | {grd:<10} | {lat:<8} | {cost:<9} | {rounds:<6}")

    print("=" * 105)
    print("  * FCR = False Consensus Rate (group agreement on an incorrect verdict).")
    print("  * Grounding = Proportion of decisions verified against authentic source citations.")
    print("  * Identical evaluation samples and ground-truth scoring used across all configurations.")
    print("=" * 105 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ConsensusX Experimental Configurations Comparison.")
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/dev_sample_300.jsonl",
        help="Path to evaluation JSONL file."
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["all"],
        choices=["all", "baseline", "voting", "debate", "consensus_x"],
        help="Configurations to run: 'all', 'baseline', 'voting', 'debate', 'consensus_x'."
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=settings.default_llm_provider,
        choices=["mock", "gemini", "xai", "grok"],
        help="LLM provider ('mock', 'gemini', 'xai')."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model ID (e.g. gemini-2.0-flash, grok-2)."
    )
    parser.add_argument(
        "--num-agents",
        type=int,
        default=3,
        help="Number of agents for voting and debate committees (default: 3)."
    )
    parser.add_argument(
        "--debate-rounds",
        type=int,
        default=2,
        help="Number of debate rounds for structured debate (default: 2)."
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
        default="outputs/experiment_comparison.json",
        help="Output destination path for unified benchmark results."
    )
    parser.add_argument(
        "--retriever",
        type=str,
        default="none",
        choices=["none", "faiss", "oracle"],
        help="Evidence retriever: 'none' (zero-shot), 'faiss' (dense vector), or 'oracle' (gold evidence)."
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default="data/processed/faiss_index",
        help="Directory where FAISS vector index is stored."
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default="data/processed/dev_corpus.jsonl",
        help="Path to evidence corpus JSONL (for oracle retriever or embedding)."
    )
    parser.add_argument(
        "--embedding-provider",
        type=str,
        default="mock",
        choices=["mock", "gemini", "sentence_transformer"],
        help="Embedding provider used for the FAISS index (default: mock)."
    )

    args = parser.parse_args()

    # Load evaluation records
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error("Dataset file not found: %s", data_path)
        sys.exit(1)

    logger.info("Loading records from: %s", data_path)
    loader = FEVERDataLoader(data_path)
    records: List[FEVERRecord] = loader.load_all(limit=args.limit)

    if not records:
        logger.error("No records loaded from %s", data_path)
        sys.exit(1)

    logger.info("Loaded %d records for comparative evaluation.", len(records))

    # Initialize client
    llm_client = get_llm_client(
        provider=args.provider,
        model_name=args.model,
        temperature=0.0
    )

    selected_configs = args.configs
    if "all" in selected_configs:
        selected_configs = ["baseline", "voting", "debate", "consensus_x"]

    results_by_config: Dict[str, Dict[str, Any]] = {}
    overall_start = time.perf_counter()

    # Initialize optional retriever
    evidence_retriever = None
    if args.retriever == "faiss":
        from src.retrieval.embeddings import get_embedding_model
        from src.retrieval.retriever import FAISSRetriever
        from src.retrieval.vector_store import FAISSVectorStore

        emb_model = get_embedding_model(provider=args.embedding_provider)
        vector_store = FAISSVectorStore.load(args.index_dir, embedding_model=emb_model)
        evidence_retriever = FAISSRetriever(vector_store=vector_store)
        logger.info("Initialized FAISSRetriever from %s", args.index_dir)
    elif args.retriever == "oracle":
        from src.retrieval.corpus import EvidenceCorpus
        from src.retrieval.retriever import OracleRetriever

        corpus = EvidenceCorpus.load_jsonl(args.corpus)
        evidence_retriever = OracleRetriever(corpus=corpus)
        logger.info("Initialized OracleRetriever from %s", args.corpus)

    # Config 1: Baseline
    if "baseline" in selected_configs:
        logger.info(">>> Running Config 1: Single-Agent Baseline...")
        baseline_pipeline = BaselinePipeline(llm_client=llm_client, retriever=evidence_retriever)
        baseline_summary = baseline_pipeline.evaluate_records(records=records)
        results_by_config["Config 1: Single-Agent Baseline"] = baseline_summary

    # Config 2: Majority Voting
    if "voting" in selected_configs:
        logger.info(">>> Running Config 2: Multi-Agent Majority Voting (N=%d)...", args.num_agents)
        voting_pipeline = MajorityVotingPipeline(
            llm_client=llm_client,
            num_agents=args.num_agents,
            retriever=evidence_retriever
        )
        voting_summary = voting_pipeline.evaluate_records(records=records)
        results_by_config[f"Config 2: Majority Voting (N={args.num_agents})"] = voting_summary

    # Config 3: Structured Debate
    if "debate" in selected_configs:
        logger.info(">>> Running Config 3: Structured Debate (N=%d, Rounds=%d)...", args.num_agents, args.debate_rounds)
        debate_pipeline = DebatePipeline(
            llm_client=llm_client,
            num_agents=args.num_agents,
            max_rounds=args.debate_rounds,
            retriever=evidence_retriever
        )
        debate_summary = debate_pipeline.evaluate_records(records=records)
        results_by_config[f"Config 3: Debate (N={args.num_agents}, R={args.debate_rounds})"] = debate_summary

    # Config 4: ConsensusX (Debate + RAG + Judge + Adaptive Stopping)
    if "consensus_x" in selected_configs:
        logger.info(">>> Running Config 4: ConsensusX (Debate + RAG + Judge + Adaptive Stopping)...")
        cx_pipeline = ConsensusXPipeline(
            llm_client=llm_client,
            retriever=evidence_retriever,
            num_agents=args.num_agents,
            max_rounds=args.debate_rounds,
            min_rounds=1
        )
        cx_summary = cx_pipeline.evaluate_records(records=records)
        results_by_config[f"Config 4: ConsensusX (N={args.num_agents}, R={args.debate_rounds}, Judge)"] = cx_summary

    total_wall_time = time.perf_counter() - overall_start

    # Print summary table
    print_comparison_table(results_by_config)

    # Save unified output
    unified_report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provider": args.provider,
        "model": llm_client.model_name,
        "sample_size": len(records),
        "total_elapsed_wall_time_seconds": round(total_wall_time, 2),
        "configurations": results_by_config
    }

    out_file = Path(args.output)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(unified_report, f, indent=2)

    logger.info("Saved unified experiment comparison to: %s", out_file)


if __name__ == "__main__":
    main()
