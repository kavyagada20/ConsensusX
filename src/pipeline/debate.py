"""Multi-agent structured debate pipeline (Config 3)."""

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.agents.debater import DebaterAgent, DEBATE_PROMPT_VERSION
from src.agents.adversary import AdversaryAgent
from src.data.schema import (
    DebateOutput,
    DebateTranscript,
    DebaterTurnOutput,
    FEVERLabel,
    FEVERRecord,
)
from src.evaluation.metrics import (
    calculate_classification_metrics,
    calculate_false_consensus_rate,
)
from src.llm.base import BaseLLMClient
from src.utils.logging import get_logger

logger = get_logger(__name__)


def aggregate_debate_verdict(
    final_turns: List[DebaterTurnOutput],
    claim_id: int
) -> Dict[str, Any]:
    """Aggregates final-round debater decisions into a consensus verdict with tie-breaking."""
    valid_turns = [t for t in final_turns if t.status == "SUCCESS"]
    if not valid_turns:
        # Fall back to turns from previous round if all failed, or default NEI
        valid_turns = final_turns

    counts: Dict[FEVERLabel, int] = {
        FEVERLabel.SUPPORTS: 0,
        FEVERLabel.REFUTES: 0,
        FEVERLabel.NOT_ENOUGH_INFO: 0,
    }
    conf_sums: Dict[FEVERLabel, float] = {
        FEVERLabel.SUPPORTS: 0.0,
        FEVERLabel.REFUTES: 0.0,
        FEVERLabel.NOT_ENOUGH_INFO: 0.0,
    }

    for t in valid_turns:
        counts[t.predicted_label] += 1
        conf_sums[t.predicted_label] += t.confidence

    max_count = max(counts.values())
    tied_labels = [lbl for lbl, cnt in counts.items() if cnt == max_count]

    is_tie = len(tied_labels) > 1
    tie_break_applied = False
    winning_label: FEVERLabel

    if not is_tie:
        winning_label = tied_labels[0]
    else:
        tie_break_applied = True
        # Confidence-weighted sum
        tied_confs = {lbl: conf_sums[lbl] for lbl in tied_labels}
        max_c = max(tied_confs.values())
        highest_conf_labels = [lbl for lbl, c in tied_confs.items() if abs(c - max_c) < 1e-6]

        if len(highest_conf_labels) == 1:
            winning_label = highest_conf_labels[0]
        else:
            # Epistemic safety default
            winning_label = FEVERLabel.NOT_ENOUGH_INFO

    supporters_conf = [t.confidence for t in valid_turns if t.predicted_label == winning_label]
    avg_conf = round(sum(supporters_conf) / len(supporters_conf), 4) if supporters_conf else 0.5
    is_unanimous = (counts[winning_label] == len(valid_turns)) and (len(valid_turns) > 1)
    consensus_reached = counts[winning_label] > (len(valid_turns) / 2)

    return {
        "winning_label": winning_label,
        "confidence": avg_conf,
        "counts": {lbl.value: cnt for lbl, cnt in counts.items()},
        "is_unanimous": is_unanimous,
        "consensus_reached": consensus_reached,
        "is_tie": is_tie,
        "tie_break_applied": tie_break_applied
    }


class DebatePipeline:
    """Orchestrator for Config 3: Multi-Agent Structured Debate."""

    def __init__(
        self,
        llm_client: BaseLLMClient,
        num_agents: int = 3,
        max_rounds: int = 2,
        prompt_version: str = DEBATE_PROMPT_VERSION,
        agent_clients: Optional[List[BaseLLMClient]] = None,
        retriever: Optional[Any] = None,
        inject_adversary: bool = False,
        adversary_persona: str = "persuasive_contrarian",
        adversary_strength: str = "assertive",
        adversary_idx: int = 1,
    ):
        self.llm_client = llm_client
        self.num_agents = max(2, num_agents)
        self.max_rounds = max(1, max_rounds)
        self.prompt_version = prompt_version
        self.retriever = retriever
        self.inject_adversary = inject_adversary
        self.adversary_persona = adversary_persona
        self.adversary_strength = adversary_strength
        self.adversary_idx = adversary_idx

        # Initialize debate panel
        self.debaters: List[Any] = []
        for i in range(self.num_agents):
            if self.inject_adversary and i == self.adversary_idx:
                self.debaters.append(
                    AdversaryAgent(
                        agent_id=f"debater_{i+1}",
                        llm_client=llm_client,
                        persona=self.adversary_persona,
                        attack_strength=self.adversary_strength,
                    )
                )
            else:
                client = agent_clients[i] if agent_clients and i < len(agent_clients) else llm_client
                self.debaters.append(
                    DebaterAgent(
                        agent_id=f"debater_{i+1}",
                        llm_client=client,
                        prompt_version=prompt_version
                    )
                )

    def run_debate(
        self,
        record: FEVERRecord,
        evidence_context: Optional[str] = None
    ) -> DebateOutput:
        """Executes a full structured debate on a single record across max_rounds."""
        transcript_rounds: Dict[int, List[DebaterTurnOutput]] = {}
        initial_verdicts: Dict[str, FEVERLabel] = {}
        final_verdicts: Dict[str, FEVERLabel] = {}
        opinion_shifts: List[Dict[str, Any]] = []

        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_cost_usd = 0.0
        total_latency = 0.0

        # -------------------------------------------------------------
        # Round 1: Independent Initial Stance (Zero peer leakage)
        # -------------------------------------------------------------
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

        # -------------------------------------------------------------
        # Rounds 2 to max_rounds: Cross-Examination & Rebuttal
        # -------------------------------------------------------------
        for r in range(2, self.max_rounds + 1):
            next_turns: List[DebaterTurnOutput] = []
            for debater, prev_turn in zip(self.debaters, current_turns):
                peer_turns = [t.model_copy(deep=True) for t in current_turns if t.agent_id != debater.agent_id]
                turn = debater.rebuttal_and_revision(
                    record=record,
                    round_number=r,
                    peer_turns=peer_turns,
                    own_previous_turn=prev_turn.model_copy(deep=True),
                    evidence_context=evidence_context
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
            current_turns = next_turns

        # Record final verdicts and opinion shifts
        for turn_r1, turn_final in zip(round_1_turns, current_turns):
            final_verdicts[turn_final.agent_id] = turn_final.predicted_label
            if turn_final.predicted_label != turn_r1.predicted_label:
                opinion_shifts.append({
                    "agent_id": turn_final.agent_id,
                    "initial_label": turn_r1.predicted_label.value,
                    "final_label": turn_final.predicted_label.value,
                    "revised_in_round": turn_final.round_number
                })

        transcript = DebateTranscript(
            claim_id=record.id,
            claim=record.claim,
            rounds=transcript_rounds,
            initial_verdicts=initial_verdicts,
            final_verdicts=final_verdicts,
            opinion_shifts=opinion_shifts,
            total_rounds=self.max_rounds
        )

        agg = aggregate_debate_verdict(current_turns, claim_id=record.id)

        return DebateOutput(
            claim_id=record.id,
            predicted_label=agg["winning_label"],
            confidence=agg["confidence"],
            consensus_reached=agg["consensus_reached"],
            is_unanimous=agg["is_unanimous"],
            final_vote_counts=agg["counts"],
            transcript=transcript,
            metadata={
                "is_tie": agg["is_tie"],
                "tie_break_applied": agg["tie_break_applied"],
                "opinion_shifts_count": len(opinion_shifts),
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_tokens": total_prompt_tokens + total_completion_tokens,
                "estimated_cost_usd": round(total_cost_usd, 6),
                "latency_seconds": round(total_latency, 4)
            }
        )

    def evaluate_records(
        self,
        records: List[FEVERRecord],
        output_file: Optional[Path] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict[str, Any]:
        """Runs multi-agent structured debate across records and computes evaluation metrics."""
        total = len(records)
        is_mock_run = getattr(self.llm_client, "is_mock", False)
        provider_name = getattr(self.llm_client, "provider", "unknown")

        logger.info(
            "Starting Structured Debate evaluation (N=%d debaters, %d rounds) on %d records with provider='%s', model='%s'%s",
            self.num_agents, self.max_rounds, total, provider_name, self.llm_client.model_name,
            " (MOCK MODE - NO API CHARGES)" if is_mock_run else ""
        )

        y_true: List[FEVERLabel] = []
        y_pred: List[FEVERLabel] = []
        team_verdicts: List[List[FEVERLabel]] = []
        results: List[Dict[str, Any]] = []

        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_cost_usd = 0.0
        total_latency = 0.0
        total_opinion_shifts = 0
        unanimous_count = 0
        consensus_count = 0

        start_time = time.perf_counter()

        for idx, record in enumerate(records, start=1):
            evidence_context = None
            if self.retriever:
                passages = self.retriever.retrieve(record.claim)
                evidence_context = self.retriever.format_evidence_context(passages)

            debate_out = self.run_debate(record, evidence_context=evidence_context)

            y_true.append(record.label)
            y_pred.append(debate_out.predicted_label)
            final_turn_labels = [
                turn.predicted_label for turn in debate_out.transcript.rounds[self.max_rounds]
            ]
            team_verdicts.append(final_turn_labels)

            meta = debate_out.metadata
            total_prompt_tokens += meta.get("prompt_tokens", 0)
            total_completion_tokens += meta.get("completion_tokens", 0)
            total_cost_usd += meta.get("estimated_cost_usd", 0.0)
            total_latency += meta.get("latency_seconds", 0.0)

            shifts = len(debate_out.transcript.opinion_shifts)
            total_opinion_shifts += shifts

            if debate_out.is_unanimous:
                unanimous_count += 1
            if debate_out.consensus_reached:
                consensus_count += 1

            is_correct = (record.label == debate_out.predicted_label)

            adv_details = None
            adversary_swayed_peers = []
            if self.inject_adversary:
                adv_id = f"debater_{self.adversary_idx + 1}"
                adv_turns = [
                    t for r_turns in debate_out.transcript.rounds.values()
                    for t in r_turns
                    if t.agent_id == adv_id or (t.metadata or {}).get("is_adversarial")
                ]
                if adv_turns:
                    adv_r1 = adv_turns[0]
                    adv_target = adv_r1.predicted_label.value
                    for shift in debate_out.transcript.opinion_shifts:
                        if shift["agent_id"] != adv_id and shift["final_label"] == adv_target:
                            adversary_swayed_peers.append(shift["agent_id"])

                    adv_details = {
                        "agent_id": adv_id,
                        "persona": (adv_r1.metadata or {}).get("adversary_persona", self.adversary_persona),
                        "strength": (adv_r1.metadata or {}).get("attack_strength", self.adversary_strength),
                        "target_label": adv_target,
                        "r1_evidence": adv_r1.evidence,
                        "r1_inference": adv_r1.inference,
                        "r1_citations": adv_r1.cited_evidence,
                        "rebuttals": [t.rebuttal for t in adv_turns[1:] if t.rebuttal],
                        "delivered_to_peers": True,
                        "peers_swayed": adversary_swayed_peers,
                    }

            results.append({
                "id": record.id,
                "claim": record.claim,
                "gold_label": record.label.value,
                "predicted_label": debate_out.predicted_label.value,
                "is_correct": is_correct,
                "confidence": debate_out.confidence,
                "consensus_reached": debate_out.consensus_reached,
                "is_unanimous": debate_out.is_unanimous,
                "final_vote_counts": debate_out.final_vote_counts,
                "has_adversary": self.inject_adversary,
                "adversary_details": adv_details,
                "adversary_swayed_peers": adversary_swayed_peers,
                "opinion_shifts": debate_out.transcript.opinion_shifts,
                "initial_verdicts": {k: v.value for k, v in debate_out.transcript.initial_verdicts.items()},
                "final_verdicts": {k: v.value for k, v in debate_out.transcript.final_verdicts.items()},
                "transcript": {
                    r_num: [
                        {
                            "agent_id": t.agent_id,
                            "label": t.predicted_label.value,
                            "confidence": t.confidence,
                            "evidence": t.evidence,
                            "inference": t.inference,
                            "uncertainty": t.uncertainty,
                            "rebuttal": t.rebuttal,
                            "cited_evidence": t.cited_evidence,
                            "verdict_revised": t.verdict_revised,
                            "status": t.status,
                            "is_adversarial": bool(t.agent_id == f"debater_{self.adversary_idx + 1}" and self.inject_adversary) or bool((t.metadata or {}).get("is_adversarial")),
                        }
                        for t in turns
                    ]
                    for r_num, turns in debate_out.transcript.rounds.items()
                },
                "prompt_tokens": meta.get("prompt_tokens", 0),
                "completion_tokens": meta.get("completion_tokens", 0),
                "estimated_cost_usd": meta.get("estimated_cost_usd", 0.0),
                "latency_seconds": meta.get("latency_seconds", 0.0)
            })

            if progress_callback:
                progress_callback(idx, total)
            elif idx % 25 == 0 or idx == total:
                logger.info("Structured Debate evaluated %d/%d claims...", idx, total)

        elapsed_wall_time = time.perf_counter() - start_time
        metrics = calculate_classification_metrics(y_true, y_pred)
        fcr = calculate_false_consensus_rate(y_true, team_verdicts, strict_unanimous=False)

        summary = {
            "experiment": f"Multi-Agent Structured Debate (Config 3, N={self.num_agents}, Rounds={self.max_rounds})",
            "provider": provider_name,
            "model": self.llm_client.model_name,
            "prompt_version": self.prompt_version,
            "num_agents": self.num_agents,
            "max_rounds": self.max_rounds,
            "has_adversary": self.inject_adversary,
            "adversary_persona": self.adversary_persona if self.inject_adversary else None,
            "adversary_strength": self.adversary_strength if self.inject_adversary else None,
            "adversary_idx": self.adversary_idx if self.inject_adversary else None,
            "adversary_invocations_count": sum(len(r["adversary_details"]["rebuttals"]) + 1 for r in results if r.get("adversary_details")) if self.inject_adversary else 0,
            "total_peers_swayed_by_adversary": sum(len(r.get("adversary_swayed_peers", [])) for r in results) if self.inject_adversary else 0,
            "total_samples": total,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "per_class_precision": metrics["per_class_precision"],
            "per_class_recall": metrics["per_class_recall"],
            "per_class_f1": metrics["per_class_f1"],
            "confusion_matrix": metrics["confusion_matrix"],
            "abstention_analysis": metrics["abstention_analysis"],
            "false_consensus_rate": fcr,
            "total_opinion_shifts": total_opinion_shifts,
            "claims_with_shifts": sum(1 for r in results if r["opinion_shifts"]),
            "unanimous_decisions": unanimous_count,
            "unanimous_rate": round(unanimous_count / total, 4) if total > 0 else 0.0,
            "consensus_decisions": consensus_count,
            "consensus_rate": round(consensus_count / total, 4) if total > 0 else 0.0,
            "token_usage": {
                "total_prompt_tokens": total_prompt_tokens,
                "total_completion_tokens": total_completion_tokens,
                "total_tokens": total_prompt_tokens + total_completion_tokens,
                "avg_tokens_per_claim": round((total_prompt_tokens + total_completion_tokens) / total, 1) if total > 0 else 0,
                "total_estimated_cost_usd": round(total_cost_usd, 6)
            },
            "performance": {
                "elapsed_wall_time_seconds": round(elapsed_wall_time, 2),
                "avg_latency_per_claim": round(total_latency / total, 4) if total > 0 else 0.0
            },
            "results": results
        }

        if output_file:
            out_path = Path(output_file)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            logger.info("Saved Structured Debate evaluation report to: %s", out_path)

        return summary

