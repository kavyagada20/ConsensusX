"""Experiment pipelines and orchestrators for ConsensusX."""

from src.pipeline.baseline import BaselinePipeline
from src.pipeline.voting import MajorityVotingPipeline, aggregate_majority_votes
from src.pipeline.debate import DebatePipeline, aggregate_debate_verdict

__all__ = [
    "BaselinePipeline",
    "MajorityVotingPipeline",
    "aggregate_majority_votes",
    "DebatePipeline",
    "aggregate_debate_verdict",
]
