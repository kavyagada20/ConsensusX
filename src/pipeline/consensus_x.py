"""ConsensusX Pipeline (Config 4): Structured Debate + RAG + Evidence-Based Judge + Adaptive Stopping.

Integrates dense evidence retrieval, multi-agent debate with cross-examination,
adaptive early stopping based on evidence sufficiency and decision stability, and a
final evidence-based judge to overcome superficial multi-agent consensus.
"""

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.agents.debater import DebaterAgent, DEBATE_PROMPT_VERSION
from src.agents.judge import EvidenceJudgeAgent, JUDGE_PROMPT_VERSION
from src.data.schema import (
    AdaptiveStoppingResult,
    ConsensusXOutput,
    DebateTranscript,
    DebaterTurnOutput,
    FEVERLabel,
    FEVERRecord,
    JudgeEvaluationOutput,
)
from src.evaluation.metrics import (
    calculate_citation_metrics,
    calculate_classification_metrics,
    calculate_false_consensus_rate,
)
from src.llm.base import BaseLLMClient
from src.pipeline.adaptive_stopping import AdaptiveStoppingController
from src.retrieval.retriever import BaseRetriever
from src.retrieval.schema import RetrievedPassage
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ConsensusXPipeline:
    """Orchestrator for Config 4: Multi-Agent Debate + RAG + Evidence Judge + Adaptive Stopping."""

    def __init__(
        self,
        llm_client: BaseLLMClient,
        retriever: Optional[BaseRetriever] = None,
        judge_client: Optional[BaseLLMClient] = None,
        num_agents: int = 3,
        max_rounds: int = 3,
        min_rounds: int = 1,
        stability_threshold: float = 1.0,
        require_evidence_grounding: bool = True,
        prompt_version: str = "v1.0.0",
        agent_clients: Optional[List[BaseLLMClient]] = None,
    ):
        self.llm_client = llm_client
        self.retriever = retriever
        self.num_agents = max(2, num_agents)
        self.max_rounds = max(1, max_rounds)
        self.min_rounds = max(1, min_rounds)
        self.prompt_version = prompt_version

        # Initialize debate panel
        self.debaters: List[DebaterAgent] = []
        for i in range(self.num_agents):
            client = agent_clients[i] if agent_clients and i < len(agent_clients) else llm_client
            self.debaters.append(
                DebaterAgent(
                    agent_id=f"debater_{i+1}",
                    llm_client=client,
                    prompt_version=prompt_version,
                )
            )

        # Initialize evidence-based judge
        j_client = judge_client or llm_client
        self.judge = EvidenceJudgeAgent(
            llm_client=j_client,
            agent_id="evidence_judge",
            prompt_version=JUDGE_PROMPT_VERSION,
        )

        # Initialize adaptive stopping controller
        self.stopping_controller = AdaptiveStoppingController(
            min_rounds=self.min_rounds,
            max_rounds=self.max_rounds,
            stability_threshold=stability_threshold,
            require_evidence_grounding=require_evidence_grounding,
        )

    def run_consensus_x(
        self,
        record: FEVERRecord,
        retrieved_passages: Optional[List[RetrievedPassage]] = None,
    ) -> ConsensusXOutput:
        """Executes full ConsensusX verification workflow for a single record."""
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_cost_usd = 0.0
        total_latency = 0.0

        # Step 1: Evidence Retrieval (RAG)
        passages: List[RetrievedPassage] = []
        evidence_context: Optional[str] = None

        if retrieved_passages is not None:
            passages = retrieved_passages
        elif self.retriever is not None:
            passages = self.retriever.retrieve(record.claim)

        if passages:
            evidence_context = BaseRetriever.format_evidence_context(passages)
        else:
            evidence_context = "No reference evidence retrieved."

        # Step 2: Multi-Round Debate with Adaptive Stopping
        transcript_rounds: Dict[int, List[DebaterTurnOutput]] = {}
        initial_verdicts: Dict[str, FEVERLabel] = {}
        final_verdicts: Dict[str, FEVERLabel] = {}
        opinion_shifts: List[Dict[str, Any]] = []

        # Round 1: Independent Initial Stance
        round_1_turns: List[DebaterTurnOutput] = []
        for debater in self.debaters:
            turn = debater.initial_stance(record=record, evidence_context=evidence_context)
            round_1_turns.append(turn)
            initial_verdicts[debater.agent_id] = turn.predicted_label

            meta = turn.metadata or {}
            total_prompt_tokens += meta.get("prompt_tokens", 0)
            total_completion_tokens += meta.get("completion_tokens", 0)
            cost = meta.get("estimated_cost_usd")
            if cost is not None:
                total_cost_usd += cost
            total_latency += meta.get("latency_seconds", 0.0)

        transcript_rounds[1] = round_1_turns
        current_turns = round_1_turns
        previous_turns: Optional[List[DebaterTurnOutput]] = None

        stopping_result = self.stopping_controller.evaluate_stopping(
            round_number=1,
            current_turns=current_turns,
            previous_turns=previous_turns,
            retrieved_passages=passages,
        )

        rounds_used = 1
        stopping_reason = stopping_result.reason

        # Subsequent Rounds (if not stopped at round 1)
        if not stopping_result.should_stop:
            for r in range(2, self.max_rounds + 1):
                rounds_used = r
                next_turns: List[DebaterTurnOutput] = []
                for debater, prev_turn in zip(self.debaters, current_turns):
                    peer_turns = [
                        t.model_copy(deep=True) for t in current_turns if t.agent_id != debater.agent_id
                    ]
                    turn = debater.rebuttal_and_revision(
                        record=record,
                        round_number=r,
                        peer_turns=peer_turns,
                        own_previous_turn=prev_turn.model_copy(deep=True),
                        evidence_context=evidence_context,
                    )
                    next_turns.append(turn)

                    meta = turn.metadata or {}
                    total_prompt_tokens += meta.get("prompt_tokens", 0)
                    total_completion_tokens += meta.get("completion_tokens", 0)
                    cost = meta.get("estimated_cost_usd")
                    if cost is not None:
                        total_cost_usd += cost
                    total_latency += meta.get("latency_seconds", 0.0)

                transcript_rounds[r] = next_turns
                previous_turns = current_turns
                current_turns = next_turns

                # Evaluate adaptive stopping criteria
                stopping_result = self.stopping_controller.evaluate_stopping(
                    round_number=r,
                    current_turns=current_turns,
                    previous_turns=previous_turns,
                    retrieved_passages=passages,
                )

                if stopping_result.should_stop:
                    stopping_reason = stopping_result.reason
                    break

        if not stopping_result.should_stop:
            stopping_reason = "max_rounds_reached"

        # Record opinion shifts and final stances
        for turn_r1, turn_final in zip(round_1_turns, current_turns):
            final_verdicts[turn_final.agent_id] = turn_final.predicted_label
            if turn_final.predicted_label != turn_r1.predicted_label:
                opinion_shifts.append({
                    "agent_id": turn_final.agent_id,
                    "initial_label": turn_r1.predicted_label.value,
                    "final_label": turn_final.predicted_label.value,
                    "revised_in_round": turn_final.round_number,
                })

        transcript = DebateTranscript(
            claim_id=record.id,
            claim=record.claim,
            rounds=transcript_rounds,
            initial_verdicts=initial_verdicts,
            final_verdicts=final_verdicts,
            opinion_shifts=opinion_shifts,
            total_rounds=rounds_used,
        )

        # Compute multi-agent consensus state (for research tracking)
        counts: Dict[FEVERLabel, int] = {}
        for t in current_turns:
            if t.status == "SUCCESS":
                counts[t.predicted_label] = counts.get(t.predicted_label, 0) + 1

        agent_consensus_reached = False
        agent_consensus_label: Optional[FEVERLabel] = None
        if counts:
            top_lbl, top_cnt = max(counts.items(), key=lambda x: x[1])
            if top_cnt > (len(current_turns) / 2):
                agent_consensus_reached = True
                agent_consensus_label = top_lbl

        # Step 3: Evidence-Based Judge Evaluation
        judge_output = self.judge.evaluate(
            record=record,
            retrieved_passages=passages,
            transcript=transcript,
        )

        j_meta = judge_output.metadata or {}
        total_prompt_tokens += j_meta.get("prompt_tokens", 0)
        total_completion_tokens += j_meta.get("completion_tokens", 0)
        j_cost = j_meta.get("estimated_cost_usd")
        if j_cost is not None:
            total_cost_usd += j_cost
        total_latency += j_meta.get("latency_seconds", 0.0)

        return ConsensusXOutput(
            claim_id=record.id,
            predicted_label=judge_output.predicted_label,
            confidence=judge_output.confidence,
            judge_output=judge_output,
            stopping_reason=stopping_reason,
            rounds_used=rounds_used,
            max_rounds=self.max_rounds,
            retrieved_passages_count=len(passages),
            transcript=transcript,
            agent_consensus_reached=agent_consensus_reached,
            agent_consensus_label=agent_consensus_label,
            metadata={
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_tokens": total_prompt_tokens + total_completion_tokens,
                "estimated_cost_usd": round(total_cost_usd, 6),
                "latency_seconds": round(total_latency, 4),
                "opinion_shifts_count": len(opinion_shifts),
                "is_citation_valid": judge_output.is_citation_valid,
                "conflicting_evidence": judge_output.conflicting_evidence,
                "missing_evidence": judge_output.missing_evidence,
            },
        )

    def evaluate_records(
        self,
        records: List[FEVERRecord],
        output_file: Optional[Path] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, Any]:
        """Evaluates a batch of records under Config 4 (ConsensusX) and computes full benchmark metrics."""
        total = len(records)
        is_mock_run = getattr(self.llm_client, "is_mock", False)
        provider_name = getattr(self.llm_client, "provider", "unknown")

        logger.info(
            "Starting ConsensusX evaluation (N=%d debaters, max_rounds=%d, RAG+Judge) on %d records with provider='%s', model='%s'%s",
            self.num_agents, self.max_rounds, total, provider_name, self.llm_client.model_name,
            " (MOCK MODE - NO API CHARGES)" if is_mock_run else ""
        )

        y_true: List[FEVERLabel] = []
        y_pred: List[FEVERLabel] = []
        debater_team_verdicts: List[List[FEVERLabel]] = []
        judge_outputs: List[JudgeEvaluationOutput] = []
        results: List[Dict[str, Any]] = []

        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_cost_usd = 0.0
        total_latency = 0.0
        total_rounds_used = 0
        stopping_reasons_count: Dict[str, int] = {}

        start_time = time.perf_counter()

        for idx, record in enumerate(records, start=1):
            out = self.run_consensus_x(record)

            y_true.append(record.label)
            y_pred.append(out.predicted_label)
            judge_outputs.append(out.judge_output)

            # Extract debater final round stances
            final_turn_labels = [
                turn.predicted_label for turn in out.transcript.rounds[out.rounds_used]
            ]
            debater_team_verdicts.append(final_turn_labels)

            meta = out.metadata
            total_prompt_tokens += meta.get("prompt_tokens", 0)
            total_completion_tokens += meta.get("completion_tokens", 0)
            total_cost_usd += meta.get("estimated_cost_usd", 0.0)
            total_latency += meta.get("latency_seconds", 0.0)
            total_rounds_used += out.rounds_used

            reason = out.stopping_reason
            stopping_reasons_count[reason] = stopping_reasons_count.get(reason, 0) + 1

            is_correct = (record.label == out.predicted_label)
            results.append({
                "id": record.id,
                "claim": record.claim,
                "gold_label": record.label.value,
                "predicted_label": out.predicted_label.value,
                "raw_predicted_label": out.judge_output.raw_predicted_label.value if out.judge_output.raw_predicted_label else out.predicted_label.value,
                "is_correct": is_correct,
                "confidence": out.confidence,
                "stopping_reason": out.stopping_reason,
                "rounds_used": out.rounds_used,
                "agent_consensus_reached": out.agent_consensus_reached,
                "agent_consensus_label": out.agent_consensus_label.value if out.agent_consensus_label else None,
                "judge_rationale": out.judge_output.rationale,
                "cited_evidence": out.judge_output.cited_evidence,
                "valid_citations": out.judge_output.valid_citations,
                "unsupported_citations": out.judge_output.unsupported_citations,
                "is_citation_valid": out.judge_output.is_citation_valid,
                "conflicting_evidence": out.judge_output.conflicting_evidence,
                "missing_evidence": out.judge_output.missing_evidence,
                "uncertainty": out.judge_output.uncertainty,
                "downgrade_applied": out.judge_output.downgrade_applied,
                "downgrade_reason": out.judge_output.downgrade_reason,
                "citation_validation_status": out.judge_output.citation_validation_status,
                "error_category": out.judge_output.error_category.value if out.judge_output.error_category else None,
                "needs_human_adjudication": out.judge_output.needs_human_adjudication,
                "opinion_shifts": out.transcript.opinion_shifts,
                "initial_verdicts": {k: v.value for k, v in out.transcript.initial_verdicts.items()},
                "final_verdicts": {k: v.value for k, v in out.transcript.final_verdicts.items()},
                "prompt_tokens": meta.get("prompt_tokens", 0),
                "completion_tokens": meta.get("completion_tokens", 0),
                "estimated_cost_usd": meta.get("estimated_cost_usd", 0.0),
                "latency_seconds": meta.get("latency_seconds", 0.0),
            })

            if progress_callback:
                progress_callback(idx, total)
            elif idx % 25 == 0 or idx == total:
                logger.info("ConsensusX evaluated %d/%d claims...", idx, total)

        elapsed_wall_time = time.perf_counter() - start_time
        metrics = calculate_classification_metrics(y_true, y_pred)
        citation_metrics = calculate_citation_metrics(judge_outputs)

        # Calculate FCR for debaters vs judge
        fcr_debater_details = calculate_false_consensus_rate(
            y_true, debater_team_verdicts, strict_unanimous=False, return_details=True
        )

        # False consensus rate for judge (when judge agrees with an incorrect agent consensus)
        judge_false_consensus_count = 0
        for yt, r in zip(y_true, results):
            if r["agent_consensus_reached"]:
                # If agent consensus was incorrect AND judge agreed with that incorrect consensus
                if r["agent_consensus_label"] != yt.value and r["predicted_label"] == r["agent_consensus_label"]:
                    judge_false_consensus_count += 1
        judge_fcr = round(judge_false_consensus_count / total, 4) if total > 0 else 0.0

        avg_rounds = round(total_rounds_used / total, 2) if total > 0 else 0.0

        summary = {
            "experiment": f"ConsensusX: Debate + RAG + Judge + Adaptive Stopping (Config 4, N={self.num_agents}, MaxRounds={self.max_rounds})",
            "provider": provider_name,
            "model": self.llm_client.model_name,
            "prompt_version": self.prompt_version,
            "num_agents": self.num_agents,
            "max_rounds": self.max_rounds,
            "min_rounds": self.min_rounds,
            "total_samples": total,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "per_class_precision": metrics["per_class_precision"],
            "per_class_recall": metrics["per_class_recall"],
            "per_class_f1": metrics["per_class_f1"],
            "confusion_matrix": metrics["confusion_matrix"],
            "abstention_analysis": metrics["abstention_analysis"],
            "false_consensus_rate": judge_fcr,
            "debater_false_consensus_rate": fcr_debater_details["overall_fcr"],
            "fcr_details": {
                "judge_fcr": judge_fcr,
                "debater_overall_fcr": fcr_debater_details["overall_fcr"],
                "debater_conditional_fcr": fcr_debater_details["conditional_fcr"],
                "debater_consensus_count": fcr_debater_details["consensus_count"],
            },
            "citation_metrics": citation_metrics,
            "adaptive_stopping": {
                "avg_rounds_used": avg_rounds,
                "total_rounds_used": total_rounds_used,
                "stopping_reasons_breakdown": stopping_reasons_count,
            },
            "token_usage": {
                "total_prompt_tokens": total_prompt_tokens,
                "total_completion_tokens": total_completion_tokens,
                "total_tokens": total_prompt_tokens + total_completion_tokens,
                "avg_tokens_per_claim": round((total_prompt_tokens + total_completion_tokens) / total, 1) if total > 0 else 0,
                "total_estimated_cost_usd": round(total_cost_usd, 6),
            },
            "performance": {
                "elapsed_wall_time_seconds": round(elapsed_wall_time, 2),
                "avg_latency_per_claim": round(total_latency / total, 4) if total > 0 else 0.0,
            },
            "results": results,
        }

        if output_file:
            out_path = Path(output_file)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            logger.info("Saved ConsensusX evaluation report to: %s", out_path)

        return summary

