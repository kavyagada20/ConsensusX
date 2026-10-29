"""Evaluation metrics for claim verification, multi-agent consensus, and adversarial robustness."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from consensus_x.data.schema import FEVERLabel


class EvaluationSummary(BaseModel):
    """Container for experiment benchmark metrics."""
    total_samples: int
    accuracy: float
    macro_f1: float
    per_class_precision: Dict[str, float]
    per_class_recall: Dict[str, float]
    per_class_f1: Dict[str, float]
    false_consensus_rate: Optional[float] = None
    adversarial_vulnerability_score: Optional[float] = None
    avg_latency_seconds: float = 0.0
    avg_total_tokens: float = 0.0


def calculate_classification_metrics(
    y_true: List[FEVERLabel],
    y_pred: List[FEVERLabel]
) -> Dict[str, Any]:
    """Calculates accuracy, per-class metrics, and macro-F1 across 3 FEVER classes."""
    import numpy as np

    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true ({len(y_true)}) != y_pred ({len(y_pred)})")

    total = len(y_true)
    if total == 0:
        return {
            "total_samples": 0,
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "per_class_precision": {},
            "per_class_recall": {},
            "per_class_f1": {}
        }

    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    accuracy = correct / total

    labels = [FEVERLabel.SUPPORTS, FEVERLabel.REFUTES, FEVERLabel.NOT_ENOUGH_INFO]
    precisions = {}
    recalls = {}
    f1s = {}

    for lbl in labels:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == lbl and yp == lbl)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != lbl and yp == lbl)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == lbl and yp != lbl)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        precisions[lbl.value] = round(prec, 4)
        recalls[lbl.value] = round(rec, 4)
        f1s[lbl.value] = round(f1, 4)

    macro_f1 = float(np.mean(list(f1s.values())))

    return {
        "total_samples": total,
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class_precision": precisions,
        "per_class_recall": recalls,
        "per_class_f1": f1s
    }


def calculate_false_consensus_rate(
    y_true: List[FEVERLabel],
    team_verdicts: List[List[FEVERLabel]],
    strict_unanimous: bool = False
) -> float:
    """Calculates False Consensus Rate (FCR).
    
    Proportion of evaluated claims where multi-agent group reached agreement (majority or unanimity)
    on an INCORRECT label.
    """
    if not y_true or len(y_true) != len(team_verdicts):
        return 0.0

    false_consensus_count = 0
    total = len(y_true)

    for yt, votes in zip(y_true, team_verdicts):
        if not votes:
            continue
        # Count vote frequency
        counts: Dict[FEVERLabel, int] = {}
        for v in votes:
            counts[v] = counts.get(v, 0) + 1

        top_label, top_count = max(counts.items(), key=lambda x: x[1])
        team_size = len(votes)

        agreed = (top_count == team_size) if strict_unanimous else (top_count > team_size / 2)
        if agreed and top_label != yt:
            false_consensus_count += 1

    return round(false_consensus_count / total, 4)


def calculate_adversarial_vulnerability_score(
    clean_accuracy: float,
    adversarial_accuracy: float
) -> float:
    """Calculates Adversarial Vulnerability Score (AVS).
    
    AVS = Accuracy(clean) - Accuracy(with adversarial agent)
    Higher AVS indicates greater vulnerability to adversarial deception.
    """
    return round(clean_accuracy - adversarial_accuracy, 4)
