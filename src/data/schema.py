"""Pydantic schemas and enums for FEVER dataset and ConsensusX verification."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import BaseModel, Field


class FEVERLabel(str, Enum):
    """Ground truth and predicted labels for the FEVER task."""
    SUPPORTS = "SUPPORTS"
    REFUTES = "REFUTES"
    NOT_ENOUGH_INFO = "NOT ENOUGH INFO"

    @classmethod
    def from_str(cls, value: str) -> "FEVERLabel":
        """Normalizes and maps string input to a FEVERLabel enum.
        
        Handles variants: Supported/Refuted/Not Enough Information, NEI, hyphens, underscores.
        """
        cleaned = value.strip().upper().replace("_", " ").replace("-", " ")
        if "SUPPORT" in cleaned:
            return cls.SUPPORTS
        elif "REFUTE" in cleaned:
            return cls.REFUTES
        elif "NOT ENOUGH" in cleaned or "NEI" in cleaned or "UNVERIFIABLE" in cleaned or "INSUFFICIENT" in cleaned:
            return cls.NOT_ENOUGH_INFO
        raise ValueError(f"Unrecognized FEVER label: '{value}'")


class VerifiableStatus(str, Enum):
    """Verifiability status in FEVER."""
    VERIFIABLE = "VERIFIABLE"
    NOT_VERIFIABLE = "NOT VERIFIABLE"


class EvidenceItem(BaseModel):
    """A single annotated evidence citation."""
    annotation_id: Optional[int] = None
    evidence_id: Optional[int] = None
    wikipedia_title: Optional[str] = None
    sentence_id: Optional[int] = None


class FEVERRecord(BaseModel):
    """Schema representing an individual record from the FEVER dataset."""
    id: int
    claim: str
    label: FEVERLabel
    verifiable: Optional[str] = None
    evidence: List[List[List[Any]]] = Field(default_factory=list)

    def has_gold_evidence(self) -> bool:
        """Returns True if the record contains at least one non-null gold evidence sentence."""
        for evidence_group in self.evidence:
            for item in evidence_group:
                if len(item) >= 4 and item[2] is not None and item[3] is not None:
                    return True
        return False

    def get_gold_evidence_sets(self) -> List[Set[Tuple[str, int]]]:
        """Extracts gold evidence sets as sets of (wiki_title, sentence_id).
        
        FEVER represents evidence as disjunction of conjunctions:
        evidence = [ [ [anno_id, evid_id, title, sent_id], ... ], ... ]
        Does NOT fabricate evidence when evidence is missing or null.
        """
        gold_sets: List[Set[Tuple[str, int]]] = []
        for evidence_group in self.evidence:
            current_set: Set[Tuple[str, int]] = set()
            for item in evidence_group:
                # Format: [anno_id, evid_id, wiki_title, sent_id]
                if len(item) >= 4:
                    title, sent_id = item[2], item[3]
                    if title is not None and sent_id is not None:
                        current_set.add((str(title), int(sent_id)))
            if current_set:
                gold_sets.append(current_set)
        return gold_sets


class ClaimVerificationOutput(BaseModel):
    """Structured output returned by an agent or judge verifying a claim."""
    claim_id: int
    predicted_label: FEVERLabel
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    reasoning: str = Field(description="Step-by-step reasoning behind the verdict")
    cited_evidence: List[str] = Field(default_factory=list, description="List of cited evidence references")
    evidence_missing: bool = Field(default=False, description="Flag indicating if evidence was unavailable or ungrounded")
    prompt_version: str = Field(default="v1.0.0", description="Version of the prompt template used")
    agent_id: Optional[str] = None
    is_adversarial: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DatasetManifest(BaseModel):
    """Provenance and metadata manifest for a processed dataset split or sample."""
    manifest_version: str = "1.0.0"
    source_file: str
    output_file: str
    sample_size: int
    random_seed: int
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    label_distribution: Dict[str, int]
    claim_ids: List[int]
    description: str = "ConsensusX Stratified Development Sample"
