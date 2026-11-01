"""Data handling, schema definitions, and loaders for FEVER dataset in ConsensusX."""

from src.data.schema import (
    FEVERLabel,
    FEVERRecord,
    ClaimVerificationOutput,
    DatasetManifest,
    EvidenceItem,
    VerifiableStatus,
    AgentRole,
    DebaterTurnOutput,
    DebateTranscript,
    MajorityVotingOutput,
    DebateOutput,
    JudgeEvaluationOutput,
    JudgeAuditCategory,
    AdaptiveStoppingResult,
    ConsensusXOutput,
)
from src.data.loader import FEVERDataLoader, check_data_leakage

__all__ = [
    "FEVERLabel",
    "FEVERRecord",
    "ClaimVerificationOutput",
    "DatasetManifest",
    "EvidenceItem",
    "VerifiableStatus",
    "AgentRole",
    "DebaterTurnOutput",
    "DebateTranscript",
    "MajorityVotingOutput",
    "DebateOutput",
    "JudgeEvaluationOutput",
    "JudgeAuditCategory",
    "AdaptiveStoppingResult",
    "ConsensusXOutput",
    "FEVERDataLoader",
    "check_data_leakage",
]
