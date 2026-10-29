"""Data handling and schema definitions for FEVER dataset in ConsensusX."""

from consensus_x.data.schema import FEVERLabel, FEVERRecord, ClaimVerificationOutput
from consensus_x.data.loader import FEVERDataLoader

__all__ = ["FEVERLabel", "FEVERRecord", "ClaimVerificationOutput", "FEVERDataLoader"]

