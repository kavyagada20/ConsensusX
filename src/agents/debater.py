"""Multi-agent debater agent for ConsensusX structured debate and conflict resolution."""

import json
import re
from typing import Any, Dict, List, Optional

from src.data.schema import AgentRole, DebaterTurnOutput, FEVERLabel, FEVERRecord
from src.llm.base import BaseLLMClient
from src.utils.logging import get_logger

logger = get_logger(__name__)

DEBATE_PROMPT_VERSION = "v1.0.0"

DEBATE_SYSTEM_PROMPT = """You are an objective, rigorous fact-checking debater in the ConsensusX multi-agent research framework.
Your task is to determine the factual truth of a given claim.

Allowed labels:
- SUPPORTS: Verified factual evidence directly confirms that the claim is true.
- REFUTES: Verified factual evidence directly contradicts or disproves the claim.
- NOT ENOUGH INFO: There is insufficient evidence to determine whether the claim is true or false.

Epistemic & Reasoning Rules:
1. Strict Separation: You must explicitly distinguish between:
   - EVIDENCE: Direct factual statements or source citations. Do not invent citations if none exist.
   - INFERENCE: Step-by-step logical deductions connecting facts to the verdict.
   - UNCERTAINTY: Explicit identification of epistemic gaps, ambiguities, or missing data.
2. Peer Scrutiny: When reviewing peer arguments in debate rounds, challenge unsupported claims, logical leaps, or hallucinated evidence.
3. Rational Calibration: Defend your position if sound, but revise your verdict if peer arguments present compelling logical or factual corrections.
4. Output Format: Return ONLY a valid JSON object matching the requested schema. No conversational preamble or postscript."""

INITIAL_STANCE_PROMPT_TEMPLATE = """Claim to verify:
"{claim}"

Reference Evidence:
{evidence_context}

Provide your independent evaluation of this claim.
You must return a valid JSON object in this exact schema:
{{
  "predicted_label": "SUPPORTS" | "REFUTES" | "NOT ENOUGH INFO",
  "confidence": 0.85,
  "evidence": "Factual evidence observed or cited (or 'None' if missing)",
  "inference": "Step-by-step logical deductions derived from evidence",
  "uncertainty": "Known gaps, ambiguities, or reasons for doubt",
  "cited_evidence": ["Document_Title:Sentence_ID"]
}}"""

REBUTTAL_PROMPT_TEMPLATE = """Claim to verify:
"{claim}"

Reference Evidence:
{evidence_context}

Current Debate Round: Round {round_number}

Your previous stance in Round {previous_round}:
- Verdict: {own_verdict} (Confidence: {own_confidence:.2f})
- Evidence Cited: {own_evidence}
- Inference: {own_inference}
- Uncertainty: {own_uncertainty}

Arguments from other Debaters in Round {previous_round}:
{peer_arguments}

Debate Instructions:
1. Carefully evaluate each peer debater's position, evidence, and logical inferences.
2. Challenge any unsupported assertions, invalid leaps, or ungrounded assumptions.
3. State whether you maintain your verdict or revise it based on valid peer reasoning.
4. Output your updated response as a valid JSON object:
{{
  "predicted_label": "SUPPORTS" | "REFUTES" | "NOT ENOUGH INFO",
  "confidence": 0.85,
  "evidence": "Factual evidence you rely on",
  "inference": "Your logical deductions in light of peer arguments",
  "uncertainty": "Remaining epistemic doubts or uncertainties",
  "rebuttal": "Specific critique of peer debaters' arguments and challenge of unsupported points",
  "cited_evidence": ["Document_Title:Sentence_ID"],
  "verdict_revised": false
}}"""


class DebaterAgent:
    """Agent that participates in structured multi-agent debate and conflict resolution."""

    def __init__(
        self,
        agent_id: str,
        llm_client: BaseLLMClient,
        role: AgentRole = AgentRole.DEBATER,
        prompt_version: str = DEBATE_PROMPT_VERSION
    ):
        self.agent_id = agent_id
        self.llm_client = llm_client
        self.role = role
        self.prompt_version = prompt_version

    def _parse_json(self, raw_content: str) -> Dict[str, Any]:
        """Extracts JSON object from LLM response text."""
        content = raw_content.strip()
        # Strip markdown fences
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
        if fence_match:
            content = fence_match.group(1).strip()
        else:
            brace_match = re.search(r"\{.*\}", content, re.DOTALL)
            if brace_match:
                content = brace_match.group(0).strip()
        return json.loads(content)

    def _format_peer_arguments(self, peer_turns: List[DebaterTurnOutput]) -> str:
        """Formats peer turns from the prior round into structured debate text."""
        if not peer_turns:
            return "No peer arguments available."

        lines = []
        for turn in peer_turns:
            if turn.status == "FAILED":
                lines.append(f"[{turn.agent_id}]: [FAILED TURN - Could not provide an argument]")
                continue
            lines.append(
                f"[{turn.agent_id}]:\n"
                f"  - Verdict: {turn.predicted_label.value} (Confidence: {turn.confidence:.2f})\n"
                f"  - Evidence: {turn.evidence or 'None'}\n"
                f"  - Inference: {turn.inference or 'None'}\n"
                f"  - Uncertainty: {turn.uncertainty or 'None'}\n"
                f"  - Citations: {turn.cited_evidence or '[]'}"
            )
        return "\n\n".join(lines)

    def initial_stance(
        self,
        record: FEVERRecord,
        evidence_context: Optional[str] = None
    ) -> DebaterTurnOutput:
        """Executes Round 1 independent reasoning before seeing any peer responses."""
        ev_text = (evidence_context or "").strip() or "None provided."
        prompt = INITIAL_STANCE_PROMPT_TEMPLATE.format(
            claim=record.claim,
            evidence_context=ev_text
        )
        system_instruction = f"{DEBATE_SYSTEM_PROMPT}\nYou are Debater Agent '{self.agent_id}'."

        try:
            resp = self.llm_client.generate(
                prompt=prompt,
                system_instruction=system_instruction,
                prompt_version=self.prompt_version
            )
            data = self._parse_json(resp.content)

            raw_label = data.get("predicted_label") or data.get("verdict") or "NOT ENOUGH INFO"
            label = FEVERLabel.from_str(str(raw_label))
            confidence = max(0.0, min(1.0, float(data.get("confidence", 0.8))))
            
            # Support both debater schema and standard single-agent reasoning fallback
            evidence = str(data.get("evidence") or (data.get("reasoning") if "evidence" not in data else ""))
            inference = str(data.get("inference") or (data.get("reasoning") if "inference" not in data else ""))
            uncertainty = str(data.get("uncertainty", ""))
            
            citations = data.get("cited_evidence", [])
            if not isinstance(citations, list):
                citations = []

            meta = {
                "input_prompt": prompt,
                "system_instruction": system_instruction,
                "prompt_tokens": resp.prompt_tokens,
                "completion_tokens": resp.completion_tokens,
                "total_tokens": resp.total_tokens,
                "estimated_cost_usd": resp.estimated_cost_usd,
                "latency_seconds": resp.latency_seconds,
                "model": resp.model,
                "provider": resp.provider,
                "prompt_version": self.prompt_version,
                "is_mock": getattr(self.llm_client, "is_mock", False)
            }

            return DebaterTurnOutput(
                agent_id=self.agent_id,
                round_number=1,
                predicted_label=label,
                confidence=confidence,
                evidence=evidence,
                inference=inference,
                uncertainty=uncertainty,
                rebuttal=None,
                cited_evidence=[str(c) for c in citations if c],
                verdict_revised=False,
                status="SUCCESS",
                metadata=meta
            )

        except Exception as e:
            logger.warning("Debater %s failed initial stance on claim %d: %s", self.agent_id, record.id, e)
            return DebaterTurnOutput(
                agent_id=self.agent_id,
                round_number=1,
                predicted_label=FEVERLabel.NOT_ENOUGH_INFO,
                confidence=0.0,
                evidence="",
                inference="",
                uncertainty="Agent execution failed",
                rebuttal=None,
                cited_evidence=[],
                verdict_revised=False,
                status="FAILED",
                error_message=str(e),
                metadata={
                    "input_prompt": prompt,
                    "system_instruction": system_instruction,
                    "error": str(e),
                    "model": getattr(self.llm_client, "model_name", "unknown"),
                    "provider": getattr(self.llm_client, "provider", "unknown"),
                    "is_mock": getattr(self.llm_client, "is_mock", False)
                }
            )

    def rebuttal_and_revision(
        self,
        record: FEVERRecord,
        round_number: int,
        peer_turns: List[DebaterTurnOutput],
        own_previous_turn: DebaterTurnOutput,
        evidence_context: Optional[str] = None
    ) -> DebaterTurnOutput:
        """Executes a subsequent debate round with cross-examination and potential verdict revision."""
        ev_text = (evidence_context or "").strip() or "None provided."
        peer_text = self._format_peer_arguments(peer_turns)

        prompt = REBUTTAL_PROMPT_TEMPLATE.format(
            claim=record.claim,
            evidence_context=ev_text,
            round_number=round_number,
            previous_round=round_number - 1,
            own_verdict=own_previous_turn.predicted_label.value,
            own_confidence=own_previous_turn.confidence,
            own_evidence=own_previous_turn.evidence or "None",
            own_inference=own_previous_turn.inference or "None",
            own_uncertainty=own_previous_turn.uncertainty or "None",
            peer_arguments=peer_text
        )
        system_instruction = f"{DEBATE_SYSTEM_PROMPT}\nYou are Debater Agent '{self.agent_id}'."

        try:
            resp = self.llm_client.generate(
                prompt=prompt,
                system_instruction=system_instruction,
                prompt_version=self.prompt_version
            )
            data = self._parse_json(resp.content)

            raw_label = data.get("predicted_label") or data.get("verdict") or own_previous_turn.predicted_label.value
            label = FEVERLabel.from_str(str(raw_label))
            confidence = max(0.0, min(1.0, float(data.get("confidence", own_previous_turn.confidence))))
            
            evidence = str(data.get("evidence", own_previous_turn.evidence))
            inference = str(data.get("inference", own_previous_turn.inference))
            uncertainty = str(data.get("uncertainty", own_previous_turn.uncertainty))
            rebuttal = str(data.get("rebuttal", ""))
            
            citations = data.get("cited_evidence", [])
            if not isinstance(citations, list):
                citations = []

            revised = bool(data.get("verdict_revised", (label != own_previous_turn.predicted_label)))

            meta = {
                "input_prompt": prompt,
                "system_instruction": system_instruction,
                "prompt_tokens": resp.prompt_tokens,
                "completion_tokens": resp.completion_tokens,
                "total_tokens": resp.total_tokens,
                "estimated_cost_usd": resp.estimated_cost_usd,
                "latency_seconds": resp.latency_seconds,
                "model": resp.model,
                "provider": resp.provider,
                "prompt_version": self.prompt_version,
                "is_mock": getattr(self.llm_client, "is_mock", False)
            }

            return DebaterTurnOutput(
                agent_id=self.agent_id,
                round_number=round_number,
                predicted_label=label,
                confidence=confidence,
                evidence=evidence,
                inference=inference,
                uncertainty=uncertainty,
                rebuttal=rebuttal,
                cited_evidence=[str(c) for c in citations if c],
                verdict_revised=revised,
                status="SUCCESS",
                metadata=meta
            )

        except Exception as e:
            logger.warning(
                "Debater %s failed rebuttal round %d on claim %d: %s",
                self.agent_id, round_number, record.id, e
            )
            # On failure, fall back safely to maintaining previous turn's verdict with failed status flag
            return DebaterTurnOutput(
                agent_id=self.agent_id,
                round_number=round_number,
                predicted_label=own_previous_turn.predicted_label,
                confidence=own_previous_turn.confidence,
                evidence=own_previous_turn.evidence,
                inference=own_previous_turn.inference,
                uncertainty=f"Failed turn: {e}",
                rebuttal=None,
                cited_evidence=list(own_previous_turn.cited_evidence),
                verdict_revised=False,
                status="FAILED",
                error_message=str(e),
                metadata={
                    "input_prompt": prompt,
                    "system_instruction": system_instruction,
                    "error": str(e),
                    "model": getattr(self.llm_client, "model_name", "unknown"),
                    "provider": getattr(self.llm_client, "provider", "unknown"),
                    "is_mock": getattr(self.llm_client, "is_mock", False)
                }
            )

