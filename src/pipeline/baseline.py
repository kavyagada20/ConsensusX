"""Single-agent baseline evaluation pipeline."""

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.agents.single_agent import SingleAgentVerifier
from src.data.schema import ClaimVerificationOutput, FEVERLabel, FEVERRecord
from src.evaluation.metrics import calculate_classification_metrics
from src.llm.base import BaseLLMClient
from src.utils.logging import get_logger

logger = get_logger(__name__)


class BaselinePipeline:
    """Orchestrates the evaluation of the Single-Agent Baseline on a FEVER dataset split."""

    def __init__(
        self,
        llm_client: BaseLLMClient,
        prompt_version: str = "v1.0.0"
    ):
        self.llm_client = llm_client
        self.prompt_version = prompt_version
        self.verifier = SingleAgentVerifier(
            llm_client=llm_client,
            prompt_version=prompt_version
        )

    def evaluate_records(
        self,
        records: List[FEVERRecord],
        output_file: Optional[Path] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict[str, Any]:
        """Runs the single-agent baseline on the provided records.
        
        Args:
            records: List of FEVER records to evaluate.
            output_file: Optional path to save detailed evaluation results.
            progress_callback: Optional callback(current, total) for UI / progress bar.
            
        Returns:
            Dict[str, Any]: Evaluation summary including accuracy, F1, confusion matrix,
                            abstention metrics, token counts, and estimated cost.
        """
        total = len(records)
        is_mock_run = getattr(self.llm_client, "is_mock", False)
        provider_name = getattr(self.llm_client, "provider", "unknown")

        logger.info(
            "Starting Single-Agent Baseline evaluation on %d records with provider='%s', model='%s'%s",
            total, provider_name, self.llm_client.model_name,
            " (MOCK MODE - NO API CHARGES)" if is_mock_run else ""
        )

        y_true: List[FEVERLabel] = []
        y_pred: List[FEVERLabel] = []
        results: List[Dict[str, Any]] = []

        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_cost_usd = 0.0
        total_latency = 0.0
        start_time = time.perf_counter()

        for idx, record in enumerate(records, start=1):
            pred_output = self.verifier.verify(record)

            y_true.append(record.label)
            y_pred.append(pred_output.predicted_label)

            meta = pred_output.metadata or {}
            p_tok = meta.get("prompt_tokens", 0)
            c_tok = meta.get("completion_tokens", 0)
            cost = meta.get("estimated_cost_usd", 0.0)
            lat = meta.get("latency_seconds", 0.0)

            total_prompt_tokens += p_tok
            total_completion_tokens += c_tok
            total_cost_usd += cost
            total_latency += lat

            is_correct = (record.label == pred_output.predicted_label)
            results.append({
                "id": record.id,
                "claim": record.claim,
                "gold_label": record.label.value,
                "predicted_label": pred_output.predicted_label.value,
                "is_correct": is_correct,
                "confidence": pred_output.confidence,
                "reasoning": pred_output.reasoning,
                "cited_evidence": pred_output.cited_evidence,
                "evidence_missing": pred_output.evidence_missing,
                "latency_seconds": round(lat, 4),
                "prompt_tokens": p_tok,
                "completion_tokens": c_tok,
                "estimated_cost_usd": round(cost, 6)
            })

            if progress_callback:
                progress_callback(idx, total)
            elif idx % 25 == 0 or idx == total:
                logger.info("Evaluated %d/%d claims...", idx, total)

        elapsed_wall_time = time.perf_counter() - start_time
        metrics = calculate_classification_metrics(y_true, y_pred)

        avg_latency = (total_latency / total) if total > 0 else 0.0
        avg_prompt_tokens = (total_prompt_tokens / total) if total > 0 else 0.0
        avg_completion_tokens = (total_completion_tokens / total) if total > 0 else 0.0

        summary = {
            "experiment": "Single-Agent Baseline (Config 1)",
            "provider": provider_name,
            "model": self.llm_client.model_name,
            "prompt_version": self.prompt_version,
            "is_mock_mode": is_mock_run,
            "total_samples": total,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "per_class_precision": metrics["per_class_precision"],
            "per_class_recall": metrics["per_class_recall"],
            "per_class_f1": metrics["per_class_f1"],
            "confusion_matrix": metrics["confusion_matrix"],
            "abstention_analysis": metrics["abstention_analysis"],
            "cost_and_tokens": {
                "total_prompt_tokens": total_prompt_tokens,
                "total_completion_tokens": total_completion_tokens,
                "total_tokens": total_prompt_tokens + total_completion_tokens,
                "avg_prompt_tokens": round(avg_prompt_tokens, 2),
                "avg_completion_tokens": round(avg_completion_tokens, 2),
                "avg_tokens_per_claim": round(avg_prompt_tokens + avg_completion_tokens, 2),
                "total_estimated_cost_usd": round(total_cost_usd, 6),
                "avg_cost_per_claim_usd": round(total_cost_usd / total, 6) if total > 0 else 0.0
            },
            "performance": {
                "wall_time_seconds": round(elapsed_wall_time, 2),
                "avg_latency_seconds": round(avg_latency, 4)
            }
        }

        full_output = {
            "summary": summary,
            "predictions": results
        }

        if output_file:
            out_path = Path(output_file)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(full_output, f, indent=2)
            logger.info("Saved baseline evaluation report to: %s", out_path)

        return summary
