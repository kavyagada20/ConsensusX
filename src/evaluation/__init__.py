"""Evaluation metrics and benchmark reporting for ConsensusX."""

from src.evaluation.metrics import (
    calculate_classification_metrics,
    calculate_confusion_matrix,
    calculate_abstention_metrics,
    calculate_false_consensus_rate,
    calculate_adversarial_vulnerability_score,
    ConfusionMatrix,
    AbstentionMetrics,
    EvaluationSummary,
)

__all__ = [
    "calculate_classification_metrics",
    "calculate_confusion_matrix",
    "calculate_abstention_metrics",
    "calculate_false_consensus_rate",
    "calculate_adversarial_vulnerability_score",
    "ConfusionMatrix",
    "AbstentionMetrics",
    "EvaluationSummary",
]
