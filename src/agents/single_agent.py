"""Single-agent claim verifier baseline for ConsensusX."""

import json
import re
from typing import Optional

from src.data.schema import ClaimVerificationOutput, FEVERLabel, FEVERRecord
from src.llm.base import BaseLLMClient
from src.utils.logging import get_logger

logger = get_logger(__name__)

PROMPT_VERSION = "v1.0.0"

SYSTEM_PROMPT = """You are a rigorous, objective fact-checking AI agent in the ConsensusX research framework.
Your task is to verify the factual accuracy of a given claim.

You must categorize the claim into exactly one of three labels:
- SUPPORTS: Verified factual evidence directly confirms that the claim is true.
- REFUTES: Verified factual evidence directly contradicts or disproves the claim.
- NOT ENOUGH INFO: There is insufficient evidence to determine whether the claim is true or false.

Guidelines:
- Base your judgment strictly on factual knowledge and objective reasoning.
- Do not assume unstated facts. If details cannot be confirmed, select NOT ENOUGH INFO.
- If no reference evidence is provided or available, state clearly in your reasoning that evidence is missing, and do NOT fabricate citations.
- Provide a confidence score between 0.0 and 1.0 reflecting your certainty.
- Return ONLY a valid JSON object in the exact format shown below, with no additional conversational prose.

JSON Response Schema:
{
  "predicted_label": "SUPPORTS" | "REFUTES" | "NOT ENOUGH INFO",
  "confidence": 0.95,
  "reasoning": "Step-by-step justification explaining the verdict.",
  "cited_evidence": ["Document_Title:Sentence_ID"]
}"""


class SingleAgentVerifier:
    """Agent that independently verifies claims using zero-shot LLM reasoning."""

    def __init__(
        self,
        llm_client: BaseLLMClient,
        agent_id: str = "baseline_verifier",
        prompt_version: str = PROMPT_VERSION
    ):
        self.llm_client = llm_client
        self.agent_id = agent_id
        self.prompt_version = prompt_version

    def _build_prompt(self, claim: str, evidence_context: Optional[str] = None) -> str:
        """Constructs the prompt for claim verification."""
        prompt = f"Claim to verify:\n\"{claim}\"\n\n"
        if evidence_context and evidence_context.strip():
            prompt += f"Retrieved Reference Evidence:\n{evidence_context.strip()}\n\n"
        else:
            prompt += "Reference Evidence: None provided.\n\n"
        prompt += (
            "Evaluate the claim and output the JSON response containing "
            "'predicted_label', 'confidence', 'reasoning', and 'cited_evidence'."
        )
        return prompt

    def _parse_response(
        self,
        raw_content: str,
        claim_id: int,
        evidence_provided: bool
    ) -> ClaimVerificationOutput:
        """Extracts, normalizes, and validates structured output against schema."""
        content = raw_content.strip()

        # Strip markdown fences if present (```json ... ``` or ``` ...)
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
        if match:
            content = match.group(1).strip()
        else:
            brace_match = re.search(r"\{.*\}", content, re.DOTALL)
            if brace_match:
                content = brace_match.group(0).strip()

        try:
            data = json.loads(content)
            raw_label = data.get("predicted_label") or data.get("verdict") or data.get("label") or "NOT ENOUGH INFO"
            label = FEVERLabel.from_str(str(raw_label))
            confidence = float(data.get("confidence", 0.8))
            confidence = max(0.0, min(1.0, confidence))
            reasoning = str(data.get("reasoning", "No explicit reasoning provided."))
            
            raw_citations = data.get("cited_evidence", [])
            if isinstance(raw_citations, list):
                cited_evidence = [str(c) for c in raw_citations if c]
            else:
                cited_evidence = []

            # Ensure evidence is not fabricated when missing
            evidence_missing = not evidence_provided or len(cited_evidence) == 0

            return ClaimVerificationOutput(
                claim_id=claim_id,
                predicted_label=label,
                confidence=confidence,
                reasoning=reasoning,
                cited_evidence=cited_evidence if evidence_provided else [],
                evidence_missing=evidence_missing,
                prompt_version=self.prompt_version,
                agent_id=self.agent_id
            )

        except Exception as e:
            logger.warning(
                "Failed to parse LLM response for claim %d: %s. Raw: %r",
                claim_id, e, raw_content[:100]
            )
            return ClaimVerificationOutput(
                claim_id=claim_id,
                predicted_label=FEVERLabel.NOT_ENOUGH_INFO,
                confidence=0.0,
                reasoning=f"Parsing error: failed to extract valid JSON ({e}). Raw output: {raw_content[:200]}",
                cited_evidence=[],
                evidence_missing=True,
                prompt_version=self.prompt_version,
                agent_id=self.agent_id
            )

    def verify(
        self,
        record: FEVERRecord,
        evidence_context: Optional[str] = None
    ) -> ClaimVerificationOutput:
        """Verifies a single FEVER claim with error handling and metadata tracking."""
        prompt = self._build_prompt(record.claim, evidence_context=evidence_context)
        has_evidence = bool(evidence_context and evidence_context.strip())

        try:
            llm_response = self.llm_client.generate(
                prompt=prompt,
                system_instruction=SYSTEM_PROMPT,
                prompt_version=self.prompt_version
            )
            parsed = self._parse_response(
                raw_content=llm_response.content,
                claim_id=record.id,
                evidence_provided=has_evidence
            )
            parsed.metadata = {
                "prompt_tokens": llm_response.prompt_tokens,
                "completion_tokens": llm_response.completion_tokens,
                "total_tokens": llm_response.total_tokens,
                "estimated_cost_usd": getattr(llm_response, "estimated_cost_usd", 0.0),
                "latency_seconds": llm_response.latency_seconds,
                "model": llm_response.model,
                "provider": llm_response.provider,
                "prompt_version": self.prompt_version,
                "is_mock": getattr(self.llm_client, "is_mock", False)
            }
            return parsed

        except Exception as api_err:
            logger.error("API error verifying claim %d: %s", record.id, api_err)
            return ClaimVerificationOutput(
                claim_id=record.id,
                predicted_label=FEVERLabel.NOT_ENOUGH_INFO,
                confidence=0.0,
                reasoning=f"API execution error: {api_err}",
                cited_evidence=[],
                evidence_missing=True,
                prompt_version=self.prompt_version,
                agent_id=self.agent_id,
                metadata={
                    "error": str(api_err),
                    "model": getattr(self.llm_client, "model_name", "unknown"),
                    "provider": getattr(self.llm_client, "provider", "unknown"),
                    "is_mock": getattr(self.llm_client, "is_mock", False)
                }
            )
