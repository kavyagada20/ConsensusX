"""Adaptive debate stopping controller for ConsensusX.

Evaluates evidence sufficiency and decision stability at each debate round to determine
whether the debate has converged with adequate evidence, avoiding unnecessary API rounds
while strictly refusing to terminate early merely because agents reach superficial consensus.
"""

from typing import Any, Dict, List, Optional, Set

from src.data.schema import (
    AdaptiveStoppingResult,
    DebaterTurnOutput,
    FEVERLabel,
)
from src.retrieval.schema import RetrievedPassage
from src.utils.logging import get_logger

logger = get_logger(__name__)


class AdaptiveStoppingController:
    """Evaluates multi-agent debate state to determine if stopping criteria are met."""

    def __init__(
        self,
        min_rounds: int = 1,
        max_rounds: int = 3,
        stability_threshold: float = 1.0,
        require_evidence_grounding: bool = True,
    ):
        """
        Args:
            min_rounds: Minimum rounds that must be executed before early stopping can trigger.
            max_rounds: Hard upper bound on debate rounds.
            stability_threshold: Proportion of debaters whose verdicts must remain unchanged (1.0 = 100%).
            require_evidence_grounding: If True, definitive verdicts (SUPPORTS/REFUTES) must be
                                       grounded in valid retrieved citations to trigger early stop.
        """
        self.min_rounds = max(1, min_rounds)
        self.max_rounds = max(1, max_rounds)
        self.stability_threshold = max(0.5, min(1.0, stability_threshold))
        self.require_evidence_grounding = require_evidence_grounding

    def _get_valid_passage_ids(self, retrieved_passages: List[RetrievedPassage]) -> Set[str]:
        """Collects all canonical and normalized passage IDs present in retrieved evidence."""
        valid_ids = set()
        for p in retrieved_passages:
            valid_ids.add(p.passage_id.strip().lower())
            valid_ids.add(f"{p.title}:{p.sentence_id}".strip().lower())
        return valid_ids

    def evaluate_stopping(
        self,
        round_number: int,
        current_turns: List[DebaterTurnOutput],
        previous_turns: Optional[List[DebaterTurnOutput]],
        retrieved_passages: List[RetrievedPassage],
    ) -> AdaptiveStoppingResult:
        """Evaluates whether the debate should terminate after round_number.
        
        Strict Research Guardrail:
        Does NOT stop merely because all agents agree. If agents agree on a definitive
        label (SUPPORTS or REFUTES) without citing valid evidence from retrieved passages,
        early stopping is rejected and the debate continues until max_rounds.
        """
        # 1. Hard bound: Maximum rounds reached
        if round_number >= self.max_rounds:
            return AdaptiveStoppingResult(
                should_stop=True,
                reason="max_rounds_reached",
                round_number=round_number,
                evidence_sufficient=False,
                stability_reached=False,
                consensus_reached=False,
                metadata={"max_rounds": self.max_rounds}
            )

        # 2. Minimum rounds guard: Cannot stop before min_rounds
        if round_number < self.min_rounds:
            return AdaptiveStoppingResult(
                should_stop=False,
                reason="min_rounds_not_met",
                round_number=round_number,
                evidence_sufficient=False,
                stability_reached=False,
                consensus_reached=False,
                metadata={"min_rounds": self.min_rounds}
            )

        # If there are no previous turns to compare against (e.g. round 1), cannot assess stability
        if not previous_turns:
            return AdaptiveStoppingResult(
                should_stop=False,
                reason="initial_round_completed",
                round_number=round_number,
                evidence_sufficient=False,
                stability_reached=False,
                consensus_reached=False,
            )

        # 3. Assess Debater Agreement & Consensus
        counts: Dict[FEVERLabel, int] = {}
        for t in current_turns:
            if t.status == "SUCCESS":
                counts[t.predicted_label] = counts.get(t.predicted_label, 0) + 1

        active_count = sum(counts.values())
        if active_count == 0:
            return AdaptiveStoppingResult(
                should_stop=False,
                reason="no_successful_turns_in_round",
                round_number=round_number,
                evidence_sufficient=False,
                stability_reached=False,
                consensus_reached=False
            )

        top_label, top_count = max(counts.items(), key=lambda x: x[1])
        consensus_reached = top_count > (active_count / 2)
        is_unanimous = top_count == active_count

        # 4. Assess Decision Stability (Opinion Shifts from prior round)
        prev_by_id = {t.agent_id: t for t in previous_turns}
        unchanged_count = 0
        comparable_count = 0

        for curr_t in current_turns:
            if curr_t.agent_id in prev_by_id:
                comparable_count += 1
                if curr_t.predicted_label == prev_by_id[curr_t.agent_id].predicted_label:
                    unchanged_count += 1

        stability_ratio = (unchanged_count / comparable_count) if comparable_count > 0 else 0.0
        stability_reached = stability_ratio >= self.stability_threshold

        # 5. Assess Evidence Sufficiency
        valid_passage_ids = self._get_valid_passage_ids(retrieved_passages)
        evidence_sufficient = False
        valid_citations_found: List[str] = []

        if top_label in (FEVERLabel.SUPPORTS, FEVERLabel.REFUTES):
            # Check if any debaters voting for top_label cited a valid passage from retrieved evidence
            for t in current_turns:
                if t.predicted_label == top_label:
                    for cite in t.cited_evidence:
                        clean_cite = cite.strip().replace("[", "").replace("]", "").strip().lower()
                        if clean_cite in valid_passage_ids:
                            valid_citations_found.append(cite)

            if len(valid_citations_found) > 0 or not self.require_evidence_grounding:
                evidence_sufficient = True
            else:
                evidence_sufficient = False
        else:
            # For NOT ENOUGH INFO: evidence sufficiency means debaters agree on lack of evidence
            # or retrieved passages are empty / lack supporting evidence
            evidence_sufficient = True

        # 6. Synthesize Stopping Decision
        metadata = {
            "top_label": top_label.value,
            "top_count": top_count,
            "active_count": active_count,
            "is_unanimous": is_unanimous,
            "consensus_reached": consensus_reached,
            "stability_ratio": round(stability_ratio, 4),
            "stability_reached": stability_reached,
            "evidence_sufficient": evidence_sufficient,
            "valid_citations_found": valid_citations_found,
            "retrieved_passages_count": len(retrieved_passages)
        }

        # CRITICAL RULE: Agreement alone is NOT sufficient to stop
        if is_unanimous and not evidence_sufficient:
            logger.info(
                "Adaptive stopping refused at round %d: Debaters unanimously agreed on %s, "
                "but no valid evidence was cited. Debate must continue.",
                round_number, top_label.value
            )
            return AdaptiveStoppingResult(
                should_stop=False,
                reason="consensus_reached_without_sufficient_evidence",
                round_number=round_number,
                evidence_sufficient=False,
                stability_reached=stability_reached,
                consensus_reached=True,
                metadata=metadata
            )

        if stability_reached and evidence_sufficient:
            logger.info(
                "Adaptive stopping triggered at round %d: Stability reached (%.2f) with verified evidence citations.",
                round_number, stability_ratio
            )
            return AdaptiveStoppingResult(
                should_stop=True,
                reason="evidence_sufficiency_and_decision_stability",
                round_number=round_number,
                evidence_sufficient=True,
                stability_reached=True,
                consensus_reached=consensus_reached,
                metadata=metadata
            )

        # Continue debate
        return AdaptiveStoppingResult(
            should_stop=False,
            reason="stability_or_evidence_criteria_not_met",
            round_number=round_number,
            evidence_sufficient=evidence_sufficient,
            stability_reached=stability_reached,
            consensus_reached=consensus_reached,
            metadata=metadata
        )

