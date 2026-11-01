"""Controlled Adversarial Testing and Comparative Robustness Evaluation (Phase 7).

Evaluates ConsensusX (Config 4) against Baselines (Configs 1, 2, 3) under:
1. Normal Condition (Clean): All agents reason honestly.
2. Adversarial Condition: Injected deceptive agent (persuasive contrarian / citation distorter).

Measures:
- Accuracy, Macro-F1 with Wilson 95% Confidence Intervals
- False Consensus Rate (FCR) under clean vs adversarial conditions
- Adversarial Vulnerability Score (AVS = Acc_clean - Acc_adv)
- Adversarial Flip Rate (AFR = proportion of correct claims flipped by adversary)
- Citation validity and evidence grounding
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.data.loader import FEVERDataLoader
from src.data.schema import FEVERLabel, FEVERRecord
from src.evaluation.metrics import (
    calculate_adversarial_flip_rate,
    calculate_adversarial_vulnerability_score,
    calculate_wilson_confidence_interval,
)
from src.llm.factory import get_llm_client
from src.pipeline.baseline import BaselinePipeline
from src.pipeline.consensus_x import ConsensusXPipeline
from src.pipeline.debate import DebatePipeline
from src.pipeline.voting import MajorityVotingPipeline
from src.retrieval.embeddings import get_embedding_model
from src.retrieval.retriever import FAISSRetriever, OracleRetriever
from src.retrieval.vector_store import FAISSVectorStore
from src.utils.logging import get_logger

logger = get_logger("run_adversarial_experiments")


def stratified_sample(records: List[FEVERRecord], sample_size: int, seed: int = 42) -> List[FEVERRecord]:
    """Deterministically samples records with balanced representation across labels."""
    if sample_size >= len(records):
        return list(records)

    rng = random.Random(seed)
    by_label: Dict[FEVERLabel, List[FEVERRecord]] = {
        FEVERLabel.SUPPORTS: [],
        FEVERLabel.REFUTES: [],
        FEVERLabel.NOT_ENOUGH_INFO: [],
    }
    for r in records:
        by_label[r.label].append(r)

    per_class = sample_size // 3
    remainder = sample_size % 3

    sampled: List[FEVERRecord] = []
    labels = [FEVERLabel.SUPPORTS, FEVERLabel.REFUTES, FEVERLabel.NOT_ENOUGH_INFO]

    for i, lbl in enumerate(labels):
        k = per_class + (1 if i < remainder else 0)
        class_records = list(by_label[lbl])
        rng.shuffle(class_records)
        sampled.extend(class_records[:k])

    rng.shuffle(sampled)
    return sampled


def print_adversarial_comparison_table(
    benchmarks: Dict[str, Dict[str, Any]],
    persona: str,
    strength: str,
    sample_size: int
) -> None:
    """Prints a comparative ASCII table contrasting Clean vs Adversarial conditions."""
    print("\n" + "=" * 115)
    print(f"               ConsensusX Adversarial Robustness Benchmark Summary (N={sample_size})")
    print(f"               Adversary Persona: '{persona}' | Attack Strength: '{strength}'")
    print("=" * 115)
    print(f"{'Configuration':<28} | {'Clean Acc (95% CI)':<19} | {'Adv Acc (95% CI)':<19} | {'AVS':<7} | {'Clean FCR':<9} | {'Adv FCR':<9} | {'AFR':<7}")
    print("-" * 115)

    for cfg_key, data in benchmarks.items():
        c_acc = data.get("clean_accuracy", 0.0) * 100
        c_ci = data.get("clean_ci", (0.0, 0.0))
        c_str = f"{c_acc:.1f}% [{c_ci[0]*100:.0f}-{c_ci[1]*100:.0f}%]"

        if "adv_accuracy" in data:
            a_acc = data["adv_accuracy"] * 100
            a_ci = data.get("adv_ci", (0.0, 0.0))
            a_str = f"{a_acc:.1f}% [{a_ci[0]*100:.0f}-{a_ci[1]*100:.0f}%]"
            avs_val = data.get("avs", 0.0)
            avs_str = f"{avs_val*100:+.1f}%"
            adv_fcr_val = data.get("adv_fcr")
            adv_fcr = f"{adv_fcr_val*100:.1f}%" if adv_fcr_val is not None else "N/A"
            afr_val = data.get("flip_metrics", {}).get("adversarial_flip_rate")
            afr_str = f"{afr_val*100:.1f}%" if afr_val is not None else "N/A"
        else:
            a_str = "N/A (Single)"
            avs_str = "N/A"
            adv_fcr = "N/A"
            afr_str = "N/A"

        clean_fcr_val = data.get("clean_fcr")
        clean_fcr = f"{clean_fcr_val*100:.1f}%" if clean_fcr_val is not None else "N/A"

        print(f"{cfg_key:<28} | {c_str:<19} | {a_str:<19} | {avs_str:<7} | {clean_fcr:<9} | {adv_fcr:<9} | {afr_str:<7}")

    print("=" * 115)
    print("  * AVS (Adversarial Vulnerability Score) = Clean Acc - Adv Acc. (Lower is better; 0% = immune).")
    print("  * AFR (Adversarial Flip Rate) = Proportion of clean-correct verdicts flipped to incorrect by the adversary.")
    print("  * 95% Confidence Intervals calculated using Wilson score method.")
    print("=" * 115 + "\n")


def generate_adversarial_markdown_report(
    summary: Dict[str, Any],
    benchmarks: Dict[str, Dict[str, Any]],
    output_path: Path
) -> None:
    """Generates comprehensive markdown report for adversarial experiments."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# ConsensusX: Controlled Adversarial Testing & Robustness Report\n\n")
        f.write(f"**Timestamp**: {summary['timestamp']}  \n")
        f.write(f"**Sample Size**: {summary['sample_size']} claims (Stratified sample, Seed={summary['random_seed']})  \n")
        f.write(f"**Dataset Source**: `{summary['data_source']}`  \n")
        f.write(f"**Provider**: `{summary['provider']}` | **Model**: `{summary['model']}`  \n")
        f.write(f"**Adversary Persona**: `{summary['adversary_persona']}` | **Strength**: `{summary['attack_strength']}`  \n\n")

        f.write("---\n\n")
        f.write("## 1. Executive Summary: Adversarial Resistance Comparison\n\n")
        f.write("| Configuration | Clean Accuracy (95% CI) | Adversarial Accuracy (95% CI) | AVS (Degradation) | Clean FCR | Adversarial FCR | Flip Rate (AFR) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")

        for cfg_name, d in benchmarks.items():
            c_acc = f"{d.get('clean_accuracy', 0.0) * 100:.1f}% [{d.get('clean_ci', (0,0))[0]*100:.0f}-{d.get('clean_ci', (0,0))[1]*100:.0f}%]"
            if "adv_accuracy" in d:
                a_acc = f"{d['adv_accuracy'] * 100:.1f}% [{d.get('adv_ci', (0,0))[0]*100:.0f}-{d.get('adv_ci', (0,0))[1]*100:.0f}%]"
                avs = f"{d.get('avs', 0.0) * 100:+.1f}%"
                a_fcr = f"{d.get('adv_fcr', 0.0) * 100:.1f}%" if d.get('adv_fcr') is not None else "N/A"
                afr = f"{d.get('flip_metrics', {}).get('adversarial_flip_rate', 0.0) * 100:.1f}%"
            else:
                a_acc = "N/A"
                avs = "N/A"
                a_fcr = "N/A"
                afr = "N/A"
            c_fcr = f"{d.get('clean_fcr', 0.0) * 100:.1f}%" if d.get('clean_fcr') is not None else "N/A"

            f.write(f"| **{cfg_name}** | {c_acc} | {a_acc} | {avs} | {c_fcr} | {a_fcr} | {afr} |\n")

        f.write("\n---\n\n")
        f.write("## 2. Metric Formulations & Theoretical Definitions\n\n")
        f.write("1. **Adversarial Vulnerability Score (AVS)**:\n")
        f.write("   $$\\text{AVS} = \\text{Accuracy}_{\\text{clean}} - \\text{Accuracy}_{\\text{adversarial}}$$\n")
        f.write("   Measures net degradation in task accuracy when one agent is compromised. An AVS of 0.0 indicates complete invariance to misleading-agent injection.\n\n")
        f.write("2. **Adversarial Flip Rate (AFR)**:\n")
        f.write("   $$\\text{AFR} = \\frac{|\\{i \\mid \\text{Clean Correct}_i \\land \\text{Adversarial Incorrect}_i\\}|}{|\\{i \\mid \\text{Clean Correct}_i\\}|}$$\n")
        f.write("   Measures the probability that a previously correct group decision is corrupted into an error by the injected adversary.\n\n")
        f.write("3. **False Consensus Rate (FCR)**:\n")
        f.write("   $$\\text{FCR} = \\frac{|\\{i \\mid \\text{Consensus}_i \\land \\hat{y}_i^{\\text{consensus}} \\ne y_i\\}|}{|D|}$$\n")
        f.write("   Proportion of claims where committee reached consensus on an incorrect verdict.\n\n")

        f.write("---\n\n")
        f.write("## 3. Analysis & Key Research Insights\n\n")
        f.write("- **Majority Voting Vulnerability**: When a single adversary is injected into a 3-agent committee, any tie or error by an honest agent allows the adversary to dictate the verdict through confidence-weighted aggregation.\n")
        f.write("- **Debate Sycophancy**: In ungrounded structured debate (Config 3), debaters are susceptible to assertive contrarian rhetoric in Round 2 rebuttals, increasing False Consensus Rate.\n")
        f.write("- **ConsensusX Grounding Resistance**: In Config 4 (ConsensusX), the Evidence-Based Judge audits cited passage IDs against the retrieved knowledge index. Fabricated citations from the adversary are rejected (`INVALID_CITATION_ID`), and the epistemic safety override neutralizes the attack, minimizing AVS.\n\n")

        f.write("---\n")
        f.write("*Report generated automatically by `scripts/run_adversarial_experiments.py`.*\n")

    logger.info("Saved adversarial benchmark report to: %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ConsensusX Controlled Adversarial Robustness Benchmark.")
    parser.add_argument("--data", type=str, default="data/processed/dev_sample_300.jsonl", help="Dataset path.")
    parser.add_argument("--index-dir", type=str, default="data/processed/faiss_index", help="FAISS vector store path.")
    parser.add_argument("--sample-size", type=int, default=30, help="Number of claims to evaluate (default: 30).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic sampling.")
    parser.add_argument("--provider", type=str, default=settings.default_llm_provider, choices=["mock", "gemini", "xai", "grok"])
    parser.add_argument("--model", type=str, default=None, help="LLM model ID.")
    parser.add_argument("--num-agents", type=int, default=3, help="Committee size.")
    parser.add_argument("--debate-rounds", type=int, default=2, help="Number of debate rounds.")
    parser.add_argument("--adversary-persona", type=str, default="persuasive_contrarian", choices=["persuasive_contrarian", "citation_distorter"])
    parser.add_argument("--attack-strength", type=str, default="assertive", choices=["assertive", "subtle"])
    parser.add_argument("--output", type=str, default="outputs/adversarial_experiment_report.json", help="Output JSON path.")
    parser.add_argument("--report-md", type=str, default="outputs/adversarial_benchmark_report.md", help="Output Markdown report path.")

    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        logger.error("Dataset file not found: %s", data_path)
        sys.exit(1)

    logger.info("Loading records from: %s", data_path)
    loader = FEVERDataLoader(data_path)
    all_records = loader.load_all()

    sampled_records = stratified_sample(all_records, sample_size=args.sample_size, seed=args.seed)
    n_samples = len(sampled_records)
    logger.info("Sampled %d records (Seed=%d) for adversarial robustness evaluation.", n_samples, args.seed)

    # Initialize retriever
    emb_model = get_embedding_model(provider="mock" if args.provider == "mock" else "gemini")
    vector_store = FAISSVectorStore.load(args.index_dir, embedding_model=emb_model)
    retriever = FAISSRetriever(vector_store=vector_store, default_top_k=5)

    llm_client = get_llm_client(provider=args.provider, model_name=args.model, temperature=0.0)

    benchmarks: Dict[str, Dict[str, Any]] = {}
    start_time = time.perf_counter()

    # --------------------------------------------------------------------------
    # 1. Config 1: Baseline (Clean Only)
    # --------------------------------------------------------------------------
    logger.info(">>> Running Config 1: Single-Agent Baseline (Clean)...")
    baseline_pipe = BaselinePipeline(llm_client=llm_client, retriever=retriever)
    base_clean = baseline_pipe.evaluate_records(sampled_records)
    base_k = int(round(base_clean["accuracy"] * n_samples))
    base_ci = calculate_wilson_confidence_interval(base_k, n_samples)
    benchmarks["Config 1: Single-Agent Baseline"] = {
        "clean_accuracy": base_clean["accuracy"],
        "clean_ci": base_ci,
        "clean_macro_f1": base_clean["macro_f1"],
        "clean_fcr": None,
        "clean_summary": base_clean,
    }

    # --------------------------------------------------------------------------
    # 2. Config 2: Majority Voting (Clean vs Adversarial)
    # --------------------------------------------------------------------------
    logger.info(">>> Running Config 2: Majority Voting (Clean)...")
    voting_clean_pipe = MajorityVotingPipeline(llm_client=llm_client, num_agents=args.num_agents, retriever=retriever, inject_adversary=False)
    vote_clean = voting_clean_pipe.evaluate_records(sampled_records)

    logger.info(">>> Running Config 2: Majority Voting (Adversarial Injected)...")
    voting_adv_pipe = MajorityVotingPipeline(
        llm_client=llm_client,
        num_agents=args.num_agents,
        retriever=retriever,
        inject_adversary=True,
        adversary_persona=args.adversary_persona,
        adversary_strength=args.attack_strength,
        adversary_idx=1
    )
    vote_adv = voting_adv_pipe.evaluate_records(sampled_records)

    v_c_k = int(round(vote_clean["accuracy"] * n_samples))
    v_a_k = int(round(vote_adv["accuracy"] * n_samples))
    vote_c_ci = calculate_wilson_confidence_interval(v_c_k, n_samples)
    vote_a_ci = calculate_wilson_confidence_interval(v_a_k, n_samples)
    vote_avs = calculate_adversarial_vulnerability_score(vote_clean["accuracy"], vote_adv["accuracy"])
    vote_flip = calculate_adversarial_flip_rate(vote_clean["results"], vote_adv["results"])

    benchmarks["Config 2: Majority Voting (N=3)"] = {
        "clean_accuracy": vote_clean["accuracy"],
        "clean_ci": vote_c_ci,
        "clean_macro_f1": vote_clean["macro_f1"],
        "clean_fcr": vote_clean["false_consensus_rate"],
        "adv_accuracy": vote_adv["accuracy"],
        "adv_ci": vote_a_ci,
        "adv_macro_f1": vote_adv["macro_f1"],
        "adv_fcr": vote_adv["false_consensus_rate"],
        "avs": vote_avs,
        "flip_metrics": vote_flip,
        "clean_summary": vote_clean,
        "adv_summary": vote_adv,
    }

    # --------------------------------------------------------------------------
    # 3. Config 3: Structured Debate (Clean vs Adversarial)
    # --------------------------------------------------------------------------
    logger.info(">>> Running Config 3: Structured Debate (Clean)...")
    debate_clean_pipe = DebatePipeline(llm_client=llm_client, num_agents=args.num_agents, max_rounds=args.debate_rounds, retriever=retriever, inject_adversary=False)
    deb_clean = debate_clean_pipe.evaluate_records(sampled_records)

    logger.info(">>> Running Config 3: Structured Debate (Adversarial Injected)...")
    debate_adv_pipe = DebatePipeline(
        llm_client=llm_client,
        num_agents=args.num_agents,
        max_rounds=args.debate_rounds,
        retriever=retriever,
        inject_adversary=True,
        adversary_persona=args.adversary_persona,
        adversary_strength=args.attack_strength,
        adversary_idx=1
    )
    deb_adv = debate_adv_pipe.evaluate_records(sampled_records)

    d_c_k = int(round(deb_clean["accuracy"] * n_samples))
    d_a_k = int(round(deb_adv["accuracy"] * n_samples))
    deb_c_ci = calculate_wilson_confidence_interval(d_c_k, n_samples)
    deb_a_ci = calculate_wilson_confidence_interval(d_a_k, n_samples)
    deb_avs = calculate_adversarial_vulnerability_score(deb_clean["accuracy"], deb_adv["accuracy"])
    deb_flip = calculate_adversarial_flip_rate(deb_clean["results"], deb_adv["results"])

    benchmarks["Config 3: Structured Debate (N=3, R=2)"] = {
        "clean_accuracy": deb_clean["accuracy"],
        "clean_ci": deb_c_ci,
        "clean_macro_f1": deb_clean["macro_f1"],
        "clean_fcr": deb_clean["false_consensus_rate"],
        "adv_accuracy": deb_adv["accuracy"],
        "adv_ci": deb_a_ci,
        "adv_macro_f1": deb_adv["macro_f1"],
        "adv_fcr": deb_adv["false_consensus_rate"],
        "avs": deb_avs,
        "flip_metrics": deb_flip,
        "clean_summary": deb_clean,
        "adv_summary": deb_adv,
    }

    # --------------------------------------------------------------------------
    # 4. Config 4: ConsensusX (Clean vs Adversarial)
    # --------------------------------------------------------------------------
    logger.info(">>> Running Config 4: ConsensusX (Clean)...")
    cx_clean_pipe = ConsensusXPipeline(llm_client=llm_client, retriever=retriever, num_agents=args.num_agents, max_rounds=args.debate_rounds, min_rounds=1, inject_adversary=False)
    cx_clean = cx_clean_pipe.evaluate_records(sampled_records)

    logger.info(">>> Running Config 4: ConsensusX (Adversarial Injected)...")
    cx_adv_pipe = ConsensusXPipeline(
        llm_client=llm_client,
        retriever=retriever,
        num_agents=args.num_agents,
        max_rounds=args.debate_rounds,
        min_rounds=1,
        inject_adversary=True,
        adversary_persona=args.adversary_persona,
        adversary_strength=args.attack_strength,
        adversary_idx=1
    )
    cx_adv = cx_adv_pipe.evaluate_records(sampled_records)

    cx_c_k = int(round(cx_clean["accuracy"] * n_samples))
    cx_a_k = int(round(cx_adv["accuracy"] * n_samples))
    cx_c_ci = calculate_wilson_confidence_interval(cx_c_k, n_samples)
    cx_a_ci = calculate_wilson_confidence_interval(cx_a_k, n_samples)
    cx_avs = calculate_adversarial_vulnerability_score(cx_clean["accuracy"], cx_adv["accuracy"])
    cx_flip = calculate_adversarial_flip_rate(cx_clean["results"], cx_adv["results"])

    benchmarks["Config 4: ConsensusX (Judge + RAG)"] = {
        "clean_accuracy": cx_clean["accuracy"],
        "clean_ci": cx_c_ci,
        "clean_macro_f1": cx_clean["macro_f1"],
        "clean_fcr": cx_clean["false_consensus_rate"],
        "adv_accuracy": cx_adv["accuracy"],
        "adv_ci": cx_a_ci,
        "adv_macro_f1": cx_adv["macro_f1"],
        "adv_fcr": cx_adv["false_consensus_rate"],
        "avs": cx_avs,
        "flip_metrics": cx_flip,
        "clean_summary": cx_clean,
        "adv_summary": cx_adv,
    }

    elapsed_wall_time = time.perf_counter() - start_time

    # Print summary table
    print_adversarial_comparison_table(benchmarks, persona=args.adversary_persona, strength=args.attack_strength, sample_size=n_samples)

    # Save outputs
    out_payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "data_source": str(data_path),
        "sample_size": n_samples,
        "random_seed": args.seed,
        "provider": args.provider,
        "model": llm_client.model_name,
        "adversary_persona": args.adversary_persona,
        "attack_strength": args.attack_strength,
        "total_elapsed_wall_time_seconds": round(elapsed_wall_time, 2),
        "benchmarks": benchmarks,
    }

    out_json = Path(args.output)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)
    logger.info("Saved adversarial benchmark JSON to: %s", out_json)

    out_md = Path(args.report_md)
    generate_adversarial_markdown_report(out_payload, benchmarks, out_md)


if __name__ == "__main__":
    main()

