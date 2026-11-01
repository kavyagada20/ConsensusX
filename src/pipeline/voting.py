"""Multi-agent independent reasoning and majority voting pipeline (Config 2)."""

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.agents.single_agent import SingleAgentVerifier
from src.agents.adversary import AdversaryAgent
from src.data.schema import ClaimVerificationOutput, FEVERLabel, FEVERRecord, MajorityVotingOutput
from src.evaluation.metrics import (
    calculate_classification_metrics,
    calculate_false_consensus_rate,
)
from src.llm.base import BaseLLMClient
from src.utils.logging import get_logger

logger = get_logger(__name__)

VOTING_PROMPT_VERSION = "v1.0.0"


def aggregate_majority_votes(
    votes: List[ClaimVerificationOutput],
    claim_id: int
) -> MajorityVotingOutput:
    """Aggregates independent agent votes using deterministic majority voting and documented tie-breaking rules.
    
    Tie-Breaking Rules (Documented & Deterministic):
    1. Plurality: The label with the highest vote count wins.
    2. Primary Tie-Breaker ('confidence_weighted'):
       If two or more labels share the highest vote count, select the tied label with the
       highest cumulative confidence score sum: sum(confidence for label).
    3. Secondary Tie-Breaker ('epistemic_humility'):
       If the confidence sums are also equal, default to FEVERLabel.NOT_ENOUGH_INFO. Under unresolved
       evidential conflict, declaring epistemic uncertainty is mathematically and scientifically safer
       than an arbitrary binary guess.
    
    Failure Handling:
    - Votes from failed agents (e.g. confidence=0.0 and reasoning with parsing/API error) are recorded
      but excluded from majority counts so failed agents never silently influence the outcome.
    - If all agents fail, verdict defaults to NOT ENOUGH INFO with confidence 0.0.
    """
    total_agents = len(votes)
    # Filter valid votes: exclude turns where execution explicitly errored
    valid_votes = [
        v for v in votes
        if not (v.confidence == 0.0 and ("error" in v.reasoning.lower() or "parsing" in v.reasoning.lower()))
    ]
    failed_agents = total_agents - len(valid_votes)
    successful_agents = len(valid_votes)

    if not valid_votes:
        # All agents failed
        return MajorityVotingOutput(
            claim_id=claim_id,
            predicted_label=FEVERLabel.NOT_ENOUGH_INFO,
            confidence=0.0,
            vote_counts={FEVERLabel.SUPPORTS.value: 0, FEVERLabel.REFUTES.value: 0, FEVERLabel.NOT_ENOUGH_INFO.value: 0},
            votes=votes,
            is_tie=False,
            tie_break_applied=False,
            tie_break_rule="all_agents_failed",
            is_unanimous=False,
            total_agents=total_agents,
            successful_agents=0,
            failed_agents=total_agents,
            metadata={"warning": "All agents failed to provide valid responses."}
        )

    # Tally counts and confidence sums per label
    counts: Dict[FEVERLabel, int] = {
        FEVERLabel.SUPPORTS: 0,
        FEVERLabel.REFUTES: 0,
        FEVERLabel.NOT_ENOUGH_INFO: 0,
    }
    confidence_sums: Dict[FEVERLabel, float] = {
        FEVERLabel.SUPPORTS: 0.0,
        FEVERLabel.REFUTES: 0.0,
        FEVERLabel.NOT_ENOUGH_INFO: 0.0,
    }

    for v in valid_votes:
        counts[v.predicted_label] += 1
        confidence_sums[v.predicted_label] += v.confidence

    max_count = max(counts.values())
    tied_labels = [lbl for lbl, c in counts.items() if c == max_count]

    is_tie = len(tied_labels) > 1
    tie_break_applied = False
    tie_break_rule = None
    winning_label: FEVERLabel

    if not is_tie:
        winning_label = tied_labels[0]
    else:
        # Step 2: Primary tie-breaker (confidence-weighted sum)
        tie_break_applied = True
        tied_confidences = {lbl: confidence_sums[lbl] for lbl in tied_labels}
        max_conf = max(tied_confidences.values())
        highest_conf_labels = [lbl for lbl, conf in tied_confidences.items() if abs(conf - max_conf) < 1e-6]

        if len(highest_conf_labels) == 1:
            winning_label = highest_conf_labels[0]
            tie_break_rule = "confidence_weighted"
        else:
            # Step 3: Secondary tie-breaker (epistemic humility / safety default)
            winning_label = FEVERLabel.NOT_ENOUGH_INFO
            tie_break_rule = "epistemic_humility_default_nei"

    # Compute aggregate confidence as average confidence of agents who voted for winning label
    supporters_conf = [v.confidence for v in valid_votes if v.predicted_label == winning_label]
    avg_confidence = round(sum(supporters_conf) / len(supporters_conf), 4) if supporters_conf else 0.5
    is_unanimous = (counts[winning_label] == len(valid_votes)) and (len(valid_votes) > 1)

    return MajorityVotingOutput(
        claim_id=claim_id,
        predicted_label=winning_label,
        confidence=avg_confidence,
        vote_counts={lbl.value: cnt for lbl, cnt in counts.items()},
        votes=votes,
        is_tie=is_tie,
        tie_break_applied=tie_break_applied,
        tie_break_rule=tie_break_rule,
        is_unanimous=is_unanimous,
        total_agents=total_agents,
        successful_agents=successful_agents,
        failed_agents=failed_agents,
        metadata={
            "confidence_sums": {lbl.value: round(c, 4) for lbl, c in confidence_sums.items()}
        }
    )


class MajorityVotingPipeline:
    """Orchestrator for Config 2: Multi-Agent Independent Reasoning & Majority Voting."""

    def __init__(
        self,
        llm_client: BaseLLMClient,
        num_agents: int = 3,
        prompt_version: str = VOTING_PROMPT_VERSION,
        agent_clients: Optional[List[BaseLLMClient]] = None,
        retriever: Optional[Any] = None,
        inject_adversary: bool = False,
        adversary_persona: str = "persuasive_contrarian",
        adversary_strength: str = "assertive",
        adversary_idx: int = 1,
    ):
        self.llm_client = llm_client
        self.num_agents = max(1, num_agents)
        self.prompt_version = prompt_version
        self.retriever = retriever
        self.inject_adversary = inject_adversary
        self.adversary_persona = adversary_persona
        self.adversary_strength = adversary_strength
        self.adversary_idx = adversary_idx

        # Initialize committee of independent verifiers
        self.verifiers: List[Any] = []
        for i in range(self.num_agents):
            if self.inject_adversary and i == self.adversary_idx:
                self.verifiers.append(
                    AdversaryAgent(
                        agent_id=f"voting_agent_{i+1}",
                        llm_client=llm_client,
                        persona=self.adversary_persona,
                        attack_strength=self.adversary_strength,
                    )
                )
            else:
                client = agent_clients[i] if agent_clients and i < len(agent_clients) else llm_client
                self.verifiers.append(
                    SingleAgentVerifier(
                        llm_client=client,
                        agent_id=f"voting_agent_{i+1}",
                        prompt_version=prompt_version
                    )
                )

    def evaluate_records(
        self,
        records: List[FEVERRecord],
        output_file: Optional[Path] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict[str, Any]:
        """Runs multi-agent majority voting on provided records and computes metrics."""
        total = len(records)
        is_mock_run = getattr(self.llm_client, "is_mock", False)
        provider_name = getattr(self.llm_client, "provider", "unknown")

        logger.info(
            "Starting Majority Voting evaluation (N=%d) on %d records with provider='%s', model='%s'%s",
            self.num_agents, total, provider_name, self.llm_client.model_name,
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
        ties_count = 0
        unanimous_count = 0

        start_time = time.perf_counter()

        for idx, record in enumerate(records, start=1):
            evidence_context = None
            if self.retriever:
                passages = self.retriever.retrieve(record.claim)
                evidence_context = self.retriever.format_evidence_context(passages)

            # Step 1: Collect votes independently before aggregation
            agent_votes: List[ClaimVerificationOutput] = []
            claim_prompt_tokens = 0
            claim_completion_tokens = 0
            claim_cost = 0.0
            claim_latency = 0.0

            for verifier in self.verifiers:
                vote = verifier.verify(record, evidence_context=evidence_context)
                agent_votes.append(vote)

                meta = vote.metadata or {}
                claim_prompt_tokens += meta.get("prompt_tokens", 0)
                claim_completion_tokens += meta.get("completion_tokens", 0)
                cost = meta.get("estimated_cost_usd")
                if cost is not None:
                    claim_cost += cost
                claim_latency += meta.get("latency_seconds", 0.0)

            # Step 2: Aggregate votes via documented tie-breaking rules
            aggregated = aggregate_majority_votes(votes=agent_votes, claim_id=record.id)

            adv_details = None
            if self.inject_adversary:
                adv_vote = next(
                    (v for v in agent_votes if getattr(v, "is_adversarial", False) or (v.metadata or {}).get("is_adversarial")),
                    None
                )
                if adv_vote:
                    adv_meta = adv_vote.metadata or {}
                    adv_details = {
                        "agent_id": adv_vote.agent_id,
                        "persona": adv_meta.get("adversary_persona", self.adversary_persona),
                        "strength": adv_meta.get("attack_strength", self.adversary_strength),
                        "target_label": adv_vote.predicted_label.value,
                        "attack_message": adv_vote.reasoning,
                        "cited_evidence": adv_vote.cited_evidence,
                        "delivered_to_aggregation": True,
                    }

            y_true.append(record.label)
            y_pred.append(aggregated.predicted_label)
            team_verdicts.append([v.predicted_label for v in agent_votes])

            total_prompt_tokens += claim_prompt_tokens
            total_completion_tokens += claim_completion_tokens
            total_cost_usd += claim_cost
            total_latency += claim_latency

            if aggregated.is_tie:
                ties_count += 1
            if aggregated.is_unanimous:
                unanimous_count += 1

            is_correct = (record.label == aggregated.predicted_label)
            results.append({
                "id": record.id,
                "claim": record.claim,
                "gold_label": record.label.value,
                "predicted_label": aggregated.predicted_label.value,
                "is_correct": is_correct,
                "confidence": aggregated.confidence,
                "vote_counts": aggregated.vote_counts,
                "is_tie": aggregated.is_tie,
                "tie_break_applied": aggregated.tie_break_applied,
                "tie_break_rule": aggregated.tie_break_rule,
                "is_unanimous": aggregated.is_unanimous,
                "successful_agents": aggregated.successful_agents,
                "failed_agents": aggregated.failed_agents,
                "has_adversary": self.inject_adversary,
                "adversary_details": adv_details,
                "agent_votes": [
                    {
                        "agent_id": v.agent_id,
                        "label": v.predicted_label.value,
                        "confidence": v.confidence,
                        "reasoning": v.reasoning,
                        "cited_evidence": v.cited_evidence,
                        "is_adversarial": bool(getattr(v, "is_adversarial", False) or (v.metadata or {}).get("is_adversarial")),
                    }
                    for v in agent_votes
                ],
                "prompt_tokens": claim_prompt_tokens,
                "completion_tokens": claim_completion_tokens,
                "estimated_cost_usd": round(claim_cost, 6),
                "latency_seconds": round(claim_latency, 4)
            })

            if progress_callback:
                progress_callback(idx, total)
            elif idx % 50 == 0 or idx == total:
                logger.info("Majority Voting evaluated %d/%d claims...", idx, total)

        elapsed_wall_time = time.perf_counter() - start_time
        metrics = calculate_classification_metrics(y_true, y_pred)
        fcr = calculate_false_consensus_rate(y_true, team_verdicts, strict_unanimous=False)

        summary = {
            "experiment": f"Multi-Agent Majority Voting (Config 2, N={self.num_agents})",
            "provider": provider_name,
            "model": self.llm_client.model_name,
            "prompt_version": self.prompt_version,
            "num_agents": self.num_agents,
            "has_adversary": self.inject_adversary,
            "adversary_persona": self.adversary_persona if self.inject_adversary else None,
            "adversary_strength": self.adversary_strength if self.inject_adversary else None,
            "adversary_idx": self.adversary_idx if self.inject_adversary else None,
            "adversary_invocations_count": sum(1 for r in results if r.get("adversary_details")) if self.inject_adversary else 0,
            "total_samples": total,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "per_class_precision": metrics["per_class_precision"],
            "per_class_recall": metrics["per_class_recall"],
            "per_class_f1": metrics["per_class_f1"],
            "confusion_matrix": metrics["confusion_matrix"],
            "abstention_analysis": metrics["abstention_analysis"],
            "false_consensus_rate": fcr,
            "ties_recorded": ties_count,
            "ties_rate": round(ties_count / total, 4) if total > 0 else 0.0,
            "unanimous_decisions": unanimous_count,
            "unanimous_rate": round(unanimous_count / total, 4) if total > 0 else 0.0,
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
            logger.info("Saved Majority Voting evaluation report to: %s", out_path)

        return summary

