"""Evidence-Based Judge agent for ConsensusX (Phase 6).

Evaluates claims based on retrieved source evidence, debater arguments, and epistemic
rigor rather than mere multi-agent consensus or agent confidence.
"""

import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from src.data.schema import (
    AgentRole,
    DebateTranscript,
    FEVERLabel,
    FEVERRecord,
    JudgeEvaluationOutput,
)
from src.llm.base import BaseLLMClient
from src.retrieval.schema import RetrievedPassage
from src.utils.logging import get_logger

logger = get_logger(__name__)

JUDGE_PROMPT_VERSION = "v1.0.0"

JUDGE_SYSTEM_PROMPT = """You are the Senior Evidence-Based Judge and Epistemic Arbiter in the ConsensusX multi-agent verification framework.

Your sole duty is to determine the factual truth of a given claim based strictly on verified source evidence from the knowledge corpus, informed by a critical evaluation of competing debater arguments.

Core Epistemic Principles:
1. Evidence Grounding Over Agent Consensus:
   - Multi-agent agreement (unanimity or supermajority) is an empirical phenomenon, NEVER proof of factual correctness.
   - Debaters may share common training biases, anchor to superficial heuristics, or succumb to persuasive adversarial arguments.
   - If debaters agree on a verdict that is NOT substantiated by the reference evidence, you must REJECT their consensus.

2. Directional Natural Language Inference:
   - SUPPORTS: Verified reference evidence explicitly and directly confirms that the claim is true.
   - REFUTES: Verified reference evidence explicitly and directly contradicts or disproves the claim.
   - NOT ENOUGH INFO: Available reference evidence is insufficient, ambiguous, or lacks decisive factual statements to prove or disprove the claim. Speculative leaps or ungrounded world knowledge are strictly forbidden.

3. Strict Provenance & Citation Integrity:
   - You may ONLY cite passage IDs that are present in the provided Reference Evidence (e.g. "[Document_Title:Sentence_ID]").
   - Fabricating citations or citing passages not present in the reference list is an epistemic violation.

4. Conflict and Uncertainty Tracking:
   - If different retrieved passages contradict one another, explicitly record the conflicting evidence.
   - If key factual predicates remain unverifiable, flag missing evidence and state the epistemic uncertainty clearly."""

JUDGE_EVALUATION_PROMPT_TEMPLATE = """Claim to verify:
"{claim}"

Reference Evidence from Knowledge Corpus:
{evidence_context}

Multi-Agent Debate Transcript and Final Positions:
{debate_transcript}

Instructions for Evidence-Based Judge:
1. Examine the claim against each retrieved Reference Evidence passage.
2. Evaluate each candidate verdict: SUPPORTS, REFUTES, NOT ENOUGH INFO.
   - Assess whether the evidence directly confirms or contradicts the claim vs. unsupported inference.
   - Check whether debaters' citations are authentic or hallucinated.
3. If the reference evidence does not provide explicit factual backing to support or refute the claim, you MUST select NOT ENOUGH INFO.
4. Record any conflicting evidence between passages and explicit reasons for epistemic doubt.

Return ONLY a valid JSON object matching this schema:
{{
  "predicted_label": "SUPPORTS" | "REFUTES" | "NOT ENOUGH INFO",
  "confidence": 0.85,
  "rationale": "Clear, concise reasoning explaining how reference evidence substantiates the verdict and why alternative verdicts or debater arguments were accepted or rejected.",
  "cited_evidence": ["Document_Title:Sentence_ID"],
  "conflicting_evidence": "Description of contradictory evidence passages if present, or null",
  "missing_evidence": false,
  "uncertainty": "Known epistemic gaps, ambiguities, or reasons for doubt"
}}"""


class EvidenceJudgeAgent:
    """Senior Judge agent that renders evidence-grounded verdicts on debated claims."""

    def __init__(
        self,
        llm_client: BaseLLMClient,
        agent_id: str = "evidence_judge",
        prompt_version: str = JUDGE_PROMPT_VERSION,
        role: AgentRole = AgentRole.JUDGE,
    ):
        self.agent_id = agent_id
        self.llm_client = llm_client
        self.prompt_version = prompt_version
        self.role = role

    @staticmethod
    def format_evidence_context(passages: List[RetrievedPassage]) -> str:
        """Formats retrieved passages into structured citation blocks for the judge."""
        if not passages:
            return "No reference evidence retrieved. (Knowledge base returned 0 passages)."

        lines = []
        for p in passages:
            citation_tag = f"[{p.passage_id}]"
            lines.append(f"{citation_tag} (Rank {p.rank}, Similarity: {p.score:.3f}): {p.text}")
        return "\n".join(lines)

    @staticmethod
    def format_debate_transcript(transcript: DebateTranscript) -> str:
        """Formats multi-round debate history into a readable summary for the judge."""
        if not transcript.rounds:
            return "No debate transcript available."

        lines = []
        for r_num in sorted(transcript.rounds.keys()):
            lines.append(f"--- Round {r_num} ---")
            for turn in transcript.rounds[r_num]:
                if turn.status == "FAILED":
                    lines.append(f"[{turn.agent_id}]: [FAILED TURN]")
                    continue
                rebuttal_str = f"\n    Rebuttal: {turn.rebuttal}" if turn.rebuttal else ""
                lines.append(
                    f"[{turn.agent_id}]: Verdict={turn.predicted_label.value} (Conf: {turn.confidence:.2f})\n"
                    f"    Evidence: {turn.evidence or 'None'}\n"
                    f"    Inference: {turn.inference or 'None'}\n"
                    f"    Uncertainty: {turn.uncertainty or 'None'}\n"
                    f"    Citations: {turn.cited_evidence or '[]'}"
                    f"{rebuttal_str}"
                )

        # Append final stances summary
        if transcript.final_verdicts:
            stances = ", ".join(f"{k}={v.value}" for k, v in transcript.final_verdicts.items())
            lines.append(f"\nFinal Debater Stances: {stances}")

        return "\n".join(lines)

    @staticmethod
    def normalize_citation_id(raw_id: str) -> str:
        """Normalizes a citation string (e.g. '[Title:0]' -> 'Title:0')."""
        cleaned = raw_id.strip()
        cleaned = re.sub(r"^\[|\]$", "", cleaned).strip()
        return cleaned

    def verify_citations(
        self,
        raw_citations: List[str],
        retrieved_passages: List[RetrievedPassage]
    ) -> Tuple[List[str], List[str], bool]:
        """Validates cited evidence IDs against available retrieved passages.
        
        Returns:
            Tuple of (valid_citations, unsupported_citations, is_citation_valid).
        """
        valid_passage_ids: Set[str] = {p.passage_id.strip() for p in retrieved_passages}
        # Also support alternate format "Title:Sentence"
        for p in retrieved_passages:
            valid_passage_ids.add(f"{p.title}:{p.sentence_id}")

        valid_citations: List[str] = []
        unsupported_citations: List[str] = []

        for cite in raw_citations:
            norm_cite = self.normalize_citation_id(cite)
            if not norm_cite:
                continue

            # Check exact match or case-insensitive match
            matched = False
            for valid_id in valid_passage_ids:
                if norm_cite.lower() == valid_id.lower():
                    valid_citations.append(valid_id)
                    matched = True
                    break
            if not matched:
                unsupported_citations.append(norm_cite)

        # Grounding check:
        # If any unsupported citations are cited, validity is False.
        is_valid = len(unsupported_citations) == 0
        return valid_citations, unsupported_citations, is_valid

    def _parse_json(self, raw_content: str) -> Dict[str, Any]:
        """Safely parses JSON response from LLM."""
        content = raw_content.strip()
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
        if fence_match:
            content = fence_match.group(1).strip()
        else:
            brace_match = re.search(r"\{.*\}", content, re.DOTALL)
            if brace_match:
                content = brace_match.group(0).strip()
        return json.loads(content)

    def evaluate(
        self,
        record: FEVERRecord,
        retrieved_passages: List[RetrievedPassage],
        transcript: DebateTranscript
    ) -> JudgeEvaluationOutput:
        """Evaluates claim, debate history, and retrieved evidence to render final verdict."""
        evidence_context = self.format_evidence_context(retrieved_passages)
        debate_context = self.format_debate_transcript(transcript)

        prompt = JUDGE_EVALUATION_PROMPT_TEMPLATE.format(
            claim=record.claim,
            evidence_context=evidence_context,
            debate_transcript=debate_context,
        )

        system_instruction = f"{JUDGE_SYSTEM_PROMPT}\nYou are Judge '{self.agent_id}'."

        try:
            resp = self.llm_client.generate(
                prompt=prompt,
                system_instruction=system_instruction,
                prompt_version=self.prompt_version,
            )

            data = self._parse_json(resp.content)
            raw_label = data.get("predicted_label") or data.get("verdict") or "NOT ENOUGH INFO"
            predicted_label = FEVERLabel.from_str(str(raw_label))
            confidence = max(0.0, min(1.0, float(data.get("confidence", 0.8))))
            rationale = str(data.get("rationale") or data.get("reasoning") or "")
            uncertainty = str(data.get("uncertainty") or "")
            conflicting_evidence = data.get("conflicting_evidence")
            if conflicting_evidence is not None:
                conflicting_evidence = str(conflicting_evidence)

            raw_citations = data.get("cited_evidence", [])
            if not isinstance(raw_citations, list):
                raw_citations = [raw_citations] if raw_citations else []

            # Verify citations against retrieved passages
            valid_cites, unsupported_cites, is_valid_citation = self.verify_citations(
                raw_citations=raw_citations,
                retrieved_passages=retrieved_passages
            )

            missing_evidence = bool(data.get("missing_evidence", False))
            override_applied = False

            # Epistemic safety guardrails:
            # 1. If verdict is SUPPORTS or REFUTES, but NO valid citations exist from the retrieved pool:
            if predicted_label in (FEVERLabel.SUPPORTS, FEVERLabel.REFUTES):
                if len(valid_cites) == 0:
                    logger.warning(
                        "Judge safety override on claim %d: %s declared without valid retrieved citation. "
                        "Downgrading to NOT ENOUGH INFO.",
                        record.id, predicted_label.value
                    )
                    predicted_label = FEVERLabel.NOT_ENOUGH_INFO
                    confidence = min(confidence, 0.5)
                    missing_evidence = True
                    is_valid_citation = False
                    rationale = (
                        f"[Epistemic Override: Downgraded to NOT ENOUGH INFO due to lack of valid retrieved citation]. "
                        f"Original rationale: {rationale}"
                    )
                    override_applied = True

            # 2. If unsupported citations were fabricated, flag citation validity
            if len(unsupported_cites) > 0:
                is_valid_citation = False
                logger.warning(
                    "Judge cited unsupported passage IDs on claim %d: %s",
                    record.id, unsupported_cites
                )

            # 3. If predicted label is NOT ENOUGH INFO, missing evidence should be True
            if predicted_label == FEVERLabel.NOT_ENOUGH_INFO:
                missing_evidence = True

            meta = {
                "prompt_tokens": resp.prompt_tokens,
                "completion_tokens": resp.completion_tokens,
                "total_tokens": resp.total_tokens,
                "estimated_cost_usd": resp.estimated_cost_usd,
                "latency_seconds": resp.latency_seconds,
                "model": resp.model,
                "provider": resp.provider,
                "override_applied": override_applied,
                "raw_citations_count": len(raw_citations),
                "is_mock": getattr(self.llm_client, "is_mock", False)
            }

            return JudgeEvaluationOutput(
                claim_id=record.id,
                predicted_label=predicted_label,
                confidence=confidence,
                rationale=rationale,
                cited_evidence=valid_cites + unsupported_cites,
                valid_citations=valid_cites,
                unsupported_citations=unsupported_cites,
                is_citation_valid=is_valid_citation,
                conflicting_evidence=conflicting_evidence,
                missing_evidence=missing_evidence,
                uncertainty=uncertainty,
                status="SUCCESS",
                metadata=meta,
            )

        except Exception as e:
            logger.error("EvidenceJudgeAgent failed on claim %d: %s", record.id, e, exc_info=True)
            return JudgeEvaluationOutput(
                claim_id=record.id,
                predicted_label=FEVERLabel.NOT_ENOUGH_INFO,
                confidence=0.0,
                rationale=f"Judge execution failure: {str(e)}",
                cited_evidence=[],
                valid_citations=[],
                unsupported_citations=[],
                is_citation_valid=False,
                conflicting_evidence=None,
                missing_evidence=True,
                uncertainty=f"Execution error encountered: {str(e)}",
                status="FAILED",
                error_message=str(e),
                metadata={"error": str(e), "is_mock": getattr(self.llm_client, "is_mock", False)},
            )

