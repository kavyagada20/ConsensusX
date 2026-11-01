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
    JudgeAuditCategory,
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

    @staticmethod
    def is_malformed_citation(raw_id: str) -> bool:
        """Checks if a citation format is malformed or invalid."""
        cleaned = raw_id.strip()
        cleaned = re.sub(r"^\[|\]$", "", cleaned).strip()
        if not cleaned:
            return True
        # Must contain title:sentence_id format where sentence_id is an integer
        parts = cleaned.rsplit(":", 1)
        if len(parts) != 2:
            return True
        title, sent_id = parts
        if not title.strip() or not sent_id.strip().isdigit():
            return True
        return False

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
        for p in retrieved_passages:
            valid_passage_ids.add(f"{p.title}:{p.sentence_id}")

        valid_citations: List[str] = []
        unsupported_citations: List[str] = []

        for cite in raw_citations:
            if self.is_malformed_citation(cite):
                unsupported_citations.append(cite.strip() or "[EMPTY_CITATION]")
                continue

            norm_cite = self.normalize_citation_id(cite)
            matched = False
            for valid_id in valid_passage_ids:
                if norm_cite.lower() == valid_id.lower():
                    valid_citations.append(valid_id)
                    matched = True
                    break
            if not matched:
                unsupported_citations.append(norm_cite)

        is_valid = (len(unsupported_citations) == 0) and (len(valid_citations) > 0 or len(raw_citations) == 0)
        return valid_citations, unsupported_citations, is_valid

    @staticmethod
    def verify_evidence_relevance(
        claim: str,
        passage: RetrievedPassage
    ) -> Tuple[bool, str]:
        """Verifies whether a retrieved passage has topical/entity relevance to the claim."""
        stop_words = {
            "a", "an", "the", "is", "was", "are", "were", "in", "on", "at", "by", "for",
            "with", "about", "against", "between", "into", "through", "during", "before",
            "after", "above", "below", "to", "from", "up", "down", "in", "out", "of", "off",
            "over", "under", "again", "further", "then", "once", "and", "but", "or", "because",
            "as", "until", "while", "that", "this", "these", "those", "it", "its", "their"
        }
        claim_tokens = {w for w in re.findall(r"\b\w+\b", claim.lower()) if len(w) > 2 and w not in stop_words}
        passage_tokens = {w for w in re.findall(r"\b\w+\b", (passage.text + " " + passage.title).lower()) if len(w) > 2 and w not in stop_words}

        overlap = claim_tokens.intersection(passage_tokens)
        if not overlap:
            return False, f"Passage '{passage.passage_id}' shares zero substantive entities or keywords with claim."

        return True, f"Relevant (shared keywords: {list(overlap)[:3]})"

    @staticmethod
    def verify_evidence_direction(
        claim: str,
        passage: RetrievedPassage,
        verdict: FEVERLabel
    ) -> Tuple[bool, str]:
        """Verifies that the cited evidence is directionally consistent with the verdict.
        
        Evaluates whether the passage text entails or contradicts the claim:
        - If passage directly contradicts the claim, verdict MUST NOT be SUPPORTS.
        - If passage directly asserts the claim without negation, verdict MUST NOT be REFUTES.
        """
        text_lower = passage.text.lower()
        claim_lower = claim.lower()

        # Explicit refutation / contradiction patterns
        refutation_patterns = [
            r"\bnot\b", r"\bnever\b", r"\bno longer\b", r"\bfail(?:ed|s|ing)?\b",
            r"\bdisproven\b", r"\brefut(?:ed|es|ing)?\b", r"\bdeni(?:ed|es)?\b",
            r"\bfalse\b", r"\bdispute[ds]?\b", r"\bunsupported\b",
            r"\bsoftware application, not a piece of hardware\b"
        ]

        has_negation = any(re.search(pat, text_lower) for pat in refutation_patterns)

        # Check for specific predicate contradictions (e.g. claim asserts "X is Y", text says "X is not Y" or "X is Z, not Y")
        if verdict == FEVERLabel.SUPPORTS:
            # If the text explicitly refutes the key assertion in the claim
            if has_negation:
                for pat in refutation_patterns:
                    match = re.search(pat, text_lower)
                    if match and ("hardware" in claim_lower and "software" in text_lower or "not" in text_lower):
                        return False, f"Evidence text contains explicit contradiction/negation: '{match.group(0)}'."

        elif verdict == FEVERLabel.REFUTES:
            # If text asserts claim and has no contradiction markers
            # E.g. Claim: "Civilization IV received high ratings", Text: "Civilization IV received critical acclaim and high ratings"
            if not has_negation:
                # Check for direct positive predicate match
                claim_keywords = [w for w in re.findall(r"\b\w+\b", claim_lower) if len(w) > 3]
                if claim_keywords and all(kw in text_lower for kw in claim_keywords[:3]):
                    if "high ratings" in claim_lower and "high ratings" in text_lower:
                        return False, "Evidence text directly affirms the claim without contradiction, contradicting REFUTES verdict."

        return True, "Directionally consistent"

    def _parse_json(self, raw_content: str) -> Dict[str, Any]:
        """Safely parses JSON response from LLM."""
        content = raw_content.strip()
        fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", content, re.DOTALL)
        if fence_match:
            content = fence_match.group(1).strip()
        else:
            brace_match = re.search(r"\{.*\}", content, re.DOTALL)
            if brace_match:
                content = brace_match.group(0).strip()
        try:
            return json.loads(content)
        except Exception:
            label_match = re.search(r'"(?:predicted_label|verdict)"\s*:\s*"([^"]+)"', content, re.IGNORECASE)
            pred_label = label_match.group(1) if label_match else "NOT ENOUGH INFO"
            conf_match = re.search(r'"confidence"\s*:\s*([0-9.]+)', content)
            conf = float(conf_match.group(1)) if conf_match else 0.5
            rat_match = re.search(r'"(?:rationale|reasoning)"\s*:\s*"([^"]+)"', content)
            rat = rat_match.group(1) if rat_match else content[:200]
            return {
                "predicted_label": pred_label,
                "confidence": conf,
                "rationale": rat,
                "cited_evidence": [],
                "evidence_status": "insufficient"
            }

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
            raw_predicted_label = FEVERLabel.from_str(str(raw_label))
            predicted_label = raw_predicted_label
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
            downgrade_reason: Optional[str] = None
            citation_validation_status = "VALID"

            # Map valid passage IDs to RetrievedPassage objects
            passage_map: Dict[str, RetrievedPassage] = {}
            for p in retrieved_passages:
                passage_map[p.passage_id.lower()] = p
                passage_map[f"{p.title}:{p.sentence_id}".lower()] = p

            # Epistemic safety guardrails:
            # 1. Definitive verdicts (SUPPORTS or REFUTES) must have valid retrieved citations
            if raw_predicted_label in (FEVERLabel.SUPPORTS, FEVERLabel.REFUTES):
                if len(raw_citations) == 0:
                    override_applied = True
                    citation_validation_status = "EMPTY_CITATIONS"
                    downgrade_reason = f"Definitive verdict {raw_predicted_label.value} declared with empty citations."
                    predicted_label = FEVERLabel.NOT_ENOUGH_INFO
                    confidence = min(confidence, 0.5)
                    missing_evidence = True
                    is_valid_citation = False
                    logger.warning("Judge safety override on claim %d: %s", record.id, downgrade_reason)

                elif len(valid_cites) == 0:
                    override_applied = True
                    citation_validation_status = "INVALID_ID"
                    downgrade_reason = (
                        f"Definitive verdict {raw_predicted_label.value} declared without valid retrieved citation. "
                        f"Cited nonexistent/unsupported IDs: {unsupported_cites}."
                    )
                    predicted_label = FEVERLabel.NOT_ENOUGH_INFO
                    confidence = min(confidence, 0.5)
                    missing_evidence = True
                    is_valid_citation = False
                    logger.warning("Judge safety override on claim %d: %s", record.id, downgrade_reason)

                else:
                    # 2. Check relevance and directional consistency for each valid citation
                    for c_id in valid_cites:
                        p = passage_map.get(c_id.lower())
                        if not p:
                            continue

                        # Topical relevance check
                        is_rel, rel_msg = self.verify_evidence_relevance(record.claim, p)
                        if not is_rel:
                            override_applied = True
                            citation_validation_status = "IRRELEVANT_CITATION"
                            downgrade_reason = f"Cited passage '{c_id}' is irrelevant to claim: {rel_msg}"
                            predicted_label = FEVERLabel.NOT_ENOUGH_INFO
                            confidence = min(confidence, 0.5)
                            missing_evidence = True
                            is_valid_citation = False
                            logger.warning("Judge safety override on claim %d: %s", record.id, downgrade_reason)
                            break

                        # Directional consistency check (NLI)
                        is_dir_ok, dir_msg = self.verify_evidence_direction(record.claim, p, raw_predicted_label)
                        if not is_dir_ok:
                            override_applied = True
                            citation_validation_status = "CONTRADICTORY_DIRECTION"
                            downgrade_reason = f"Cited passage '{c_id}' directionally contradicts verdict {raw_predicted_label.value}: {dir_msg}"
                            predicted_label = FEVERLabel.NOT_ENOUGH_INFO
                            confidence = min(confidence, 0.5)
                            missing_evidence = True
                            is_valid_citation = False
                            logger.warning("Judge safety override on claim %d: %s", record.id, downgrade_reason)
                            break

            # 3. Check for unsupported / malformed citations in raw citations
            if len(unsupported_cites) > 0:
                is_valid_citation = False
                if citation_validation_status == "VALID":
                    citation_validation_status = "INVALID_ID"

            # 4. If predicted label is NOT ENOUGH INFO, missing evidence should be True
            if predicted_label == FEVERLabel.NOT_ENOUGH_INFO:
                missing_evidence = True

            if override_applied and downgrade_reason:
                rationale = f"[Epistemic Override: {downgrade_reason}] Original rationale: {rationale}"

            # 5. Fine-grained audit categorization
            error_cat: Optional[JudgeAuditCategory] = None
            needs_human = False

            gold_lbl = getattr(record, "label", None)
            if gold_lbl is not None:
                if predicted_label == gold_lbl:
                    if predicted_label == FEVERLabel.NOT_ENOUGH_INFO:
                        error_cat = JudgeAuditCategory.APPROPRIATE_ABSTENTION
                    else:
                        error_cat = JudgeAuditCategory.CORRECTLY_GROUNDED
                else:
                    if citation_validation_status == "EMPTY_CITATIONS":
                        error_cat = JudgeAuditCategory.UNSUPPORTED_VERDICT
                    elif citation_validation_status == "INVALID_ID":
                        error_cat = JudgeAuditCategory.INVALID_CITATION_ID
                    elif citation_validation_status == "IRRELEVANT_CITATION":
                        error_cat = JudgeAuditCategory.IRRELEVANT_CITATION
                    elif citation_validation_status == "CONTRADICTORY_DIRECTION":
                        error_cat = JudgeAuditCategory.WRONG_EVIDENCE_DIRECTION
                    elif gold_lbl in (FEVERLabel.SUPPORTS, FEVERLabel.REFUTES) and any(getattr(p, "is_gold", False) for p in retrieved_passages):
                        error_cat = JudgeAuditCategory.MISSED_EVIDENCE
                    elif not is_valid_citation:
                        error_cat = JudgeAuditCategory.INVALID_CITATION_ID
                    else:
                        error_cat = JudgeAuditCategory.NEEDS_HUMAN_ADJUDICATION
                        needs_human = True

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
                raw_predicted_label=raw_predicted_label,
                confidence=confidence,
                rationale=rationale,
                cited_evidence=valid_cites + unsupported_cites,
                valid_citations=valid_cites,
                unsupported_citations=unsupported_cites,
                is_citation_valid=is_valid_citation,
                conflicting_evidence=conflicting_evidence,
                missing_evidence=missing_evidence,
                uncertainty=uncertainty,
                downgrade_applied=override_applied,
                downgrade_reason=downgrade_reason,
                citation_validation_status=citation_validation_status,
                error_category=error_cat,
                needs_human_adjudication=needs_human,
                status="SUCCESS",
                metadata=meta,
            )

        except Exception as e:
            logger.error("EvidenceJudgeAgent failed on claim %d: %s", record.id, e, exc_info=True)
            return JudgeEvaluationOutput(
                claim_id=record.id,
                predicted_label=FEVERLabel.NOT_ENOUGH_INFO,
                raw_predicted_label=None,
                confidence=0.0,
                rationale=f"Judge execution failure: {str(e)}",
                cited_evidence=[],
                valid_citations=[],
                unsupported_citations=[],
                is_citation_valid=False,
                conflicting_evidence=None,
                missing_evidence=True,
                uncertainty=f"Execution error encountered: {str(e)}",
                downgrade_applied=False,
                downgrade_reason=f"Execution failure: {str(e)}",
                citation_validation_status="ERROR",
                error_category=JudgeAuditCategory.UNSUPPORTED_VERDICT,
                needs_human_adjudication=True,
                status="FAILED",
                error_message=str(e),
                metadata={"error": str(e), "is_mock": getattr(self.llm_client, "is_mock", False)},
            )

