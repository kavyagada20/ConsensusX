"""Evaluation metrics and benchmark reporting for ConsensusX."""

from consensus_x.evaluation.metrics import (
    calculate_classification_metrics,
    calculate_false_consensus_rate,
    calculate_adversarial_vulnerability_score,
    EvaluationSummary
)

__all__ = [
    "calculate_classification_metrics",
    "calculate_false_consensus_rate",
    "calculate_adversarial_vulnerability_score",
    "EvaluationSummary"
]

