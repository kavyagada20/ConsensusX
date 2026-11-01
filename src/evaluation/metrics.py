"""Evaluation metrics for claim verification, confusion matrix, abstention analysis, and multi-agent robustness."""

from typing import Any, Dict, List, Optional
import numpy as np
from pydantic import BaseModel, Field
from src.data.schema import FEVERLabel


class ConfusionMatrix(BaseModel):
    """3x3 confusion matrix for FEVER claim verification."""
    labels: List[str] = Field(default=["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"])
    matrix: Dict[str, Dict[str, int]] = Field(
        description="Nested dictionary: matrix[gold_label][predicted_label] = count"
    )


class AbstentionMetrics(BaseModel):
    """Analysis of model abstention (NOT ENOUGH INFO predictions) and epistemic humility."""
    total_samples: int
    abstention_count: int = Field(description="Number of times model predicted NOT ENOUGH INFO")
    abstention_rate: float = Field(description="Proportion of claims where model abstained")
    decided_count: int = Field(description="Number of claims where model made a definitive call (SUPPORTS/REFUTES)")
    decided_accuracy: float = Field(description="Accuracy calculated exclusively on decided claims")
    gold_nei_count: int = Field(description="True count of NOT ENOUGH INFO claims in ground truth")
    gold_nei_correctly_abstained: int = Field(description="True NEI claims that the model correctly recognized as NEI")
    gold_nei_recall: float = Field(description="Recall on ground-truth NEI claims")
    false_certainty_count: int = Field(description="True NEI claims where model falsely made a definitive call")
    false_certainty_rate: float = Field(description="Proportion of true NEI claims where model hallucinated certainty")


class EvaluationSummary(BaseModel):
    """Container for experiment benchmark metrics."""
    experiment: str = "Single-Agent Baseline (Config 1)"
    provider: str
    model: str
    prompt_version: str = "v1.0.0"
    total_samples: int
    accuracy: float
    macro_f1: float
    per_class_precision: Dict[str, float]
    per_class_recall: Dict[str, float]
    per_class_f1: Dict[str, float]
    confusion_matrix: Dict[str, Any]
    abstention_analysis: Dict[str, Any]
    false_consensus_rate: Optional[float] = None
    adversarial_vulnerability_score: Optional[float] = None
    token_usage: Dict[str, Any] = Field(default_factory=dict)
    performance: Dict[str, Any] = Field(default_factory=dict)


def calculate_confusion_matrix(
    y_true: List[FEVERLabel],
    y_pred: List[FEVERLabel]
) -> Dict[str, Any]:
    """Calculates 3x3 confusion matrix for FEVER labels: matrix[gold][pred] = count."""
    labels = [FEVERLabel.SUPPORTS.value, FEVERLabel.REFUTES.value, FEVERLabel.NOT_ENOUGH_INFO.value]
    matrix: Dict[str, Dict[str, int]] = {gold: {pred: 0 for pred in labels} for gold in labels}

    for yt, yp in zip(y_true, y_pred):
        gold_val = yt.value if hasattr(yt, "value") else str(yt)
        pred_val = yp.value if hasattr(yp, "value") else str(yp)
        if gold_val in matrix and pred_val in matrix[gold_val]:
            matrix[gold_val][pred_val] += 1

    return {
        "labels": labels,
        "matrix": matrix
    }


def calculate_abstention_metrics(
    y_true: List[FEVERLabel],
    y_pred: List[FEVERLabel]
) -> Dict[str, Any]:
    """Calculates abstention (NOT ENOUGH INFO) behavior and epistemic calibration."""
    total = len(y_true)
    if total == 0:
        return {
            "total_samples": 0,
            "abstention_count": 0,
            "abstention_rate": 0.0,
            "decided_count": 0,
            "decided_accuracy": 0.0,
            "gold_nei_count": 0,
            "gold_nei_correctly_abstained": 0,
            "gold_nei_recall": 0.0,
            "false_certainty_count": 0,
            "false_certainty_rate": 0.0,
        }

    nei_label = FEVERLabel.NOT_ENOUGH_INFO

    abstention_count = sum(1 for yp in y_pred if yp == nei_label)
    abstention_rate = round(abstention_count / total, 4)

    # Decided subset (SUPPORTS or REFUTES predicted)
    decided_indices = [i for i, yp in enumerate(y_pred) if yp != nei_label]
    decided_count = len(decided_indices)
    if decided_count > 0:
        decided_correct = sum(1 for i in decided_indices if y_true[i] == y_pred[i])
        decided_accuracy = round(decided_correct / decided_count, 4)
    else:
        decided_accuracy = 0.0

    # Ground truth NEI analysis
    gold_nei_indices = [i for i, yt in enumerate(y_true) if yt == nei_label]
    gold_nei_count = len(gold_nei_indices)

    if gold_nei_count > 0:
        gold_nei_correctly_abstained = sum(1 for i in gold_nei_indices if y_pred[i] == nei_label)
        gold_nei_recall = round(gold_nei_correctly_abstained / gold_nei_count, 4)
        false_certainty_count = gold_nei_count - gold_nei_correctly_abstained
        false_certainty_rate = round(false_certainty_count / gold_nei_count, 4)
    else:
        gold_nei_correctly_abstained = 0
        gold_nei_recall = 0.0
        false_certainty_count = 0
        false_certainty_rate = 0.0

    return {
        "total_samples": total,
        "abstention_count": abstention_count,
        "abstention_rate": abstention_rate,
        "decided_count": decided_count,
        "decided_accuracy": decided_accuracy,
        "gold_nei_count": gold_nei_count,
        "gold_nei_correctly_abstained": gold_nei_correctly_abstained,
        "gold_nei_recall": gold_nei_recall,
        "false_certainty_count": false_certainty_count,
        "false_certainty_rate": false_certainty_rate,
    }


def calculate_classification_metrics(
    y_true: List[FEVERLabel],
    y_pred: List[FEVERLabel]
) -> Dict[str, Any]:
    """Calculates accuracy, per-class precision/recall/F1, confusion matrix, and abstention metrics."""
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
            "per_class_f1": {},
            "confusion_matrix": calculate_confusion_matrix([], []),
            "abstention_analysis": calculate_abstention_metrics([], [])
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
    conf_matrix = calculate_confusion_matrix(y_true, y_pred)
    abstention_stats = calculate_abstention_metrics(y_true, y_pred)

    return {
        "total_samples": total,
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class_precision": precisions,
        "per_class_recall": recalls,
        "per_class_f1": f1s,
        "confusion_matrix": conf_matrix,
        "abstention_analysis": abstention_stats
    }


def calculate_false_consensus_rate(
    y_true: List[FEVERLabel],
    team_verdicts: List[List[FEVERLabel]],
    strict_unanimous: bool = False,
    return_details: bool = False
) -> Any:
    """Calculates False Consensus Rate (FCR).
    
    Formal Mathematical Formulations:
    1. Overall FCR (FCR_total):
       FCR_total = |{i in D : ConsensusReached_i and L_consensus_i != L_gold_i}| / |D|
       The proportion of all evaluated claims where the multi-agent committee reached
       agreement (majority or unanimity) on an INCORRECT verdict.

    2. Conditional FCR (FCR_cond):
       FCR_cond = |{i in D : ConsensusReached_i and L_consensus_i != L_gold_i}| / |{i in D : ConsensusReached_i}|
       The proportion of consensus decisions that were factually incorrect (i.e. error rate
       strictly conditioned on consensus having occurred).

    Args:
        y_true: Ground truth FEVER labels.
        team_verdicts: List of agent vote lists for each claim.
        strict_unanimous: If True, requires unanimous consensus (100%); otherwise majority (>50%).
        return_details: If True, returns a dict with both overall and conditional FCR.
    """
    if not y_true or len(y_true) != len(team_verdicts):
        if return_details:
            return {
                "overall_fcr": 0.0,
                "conditional_fcr": 0.0,
                "consensus_count": 0,
                "false_consensus_count": 0,
                "total_evaluated": len(y_true) if y_true else 0
            }
        return 0.0

    false_consensus_count = 0
    consensus_count = 0
    total = len(y_true)

    for yt, votes in zip(y_true, team_verdicts):
        if not votes:
            continue
        counts: Dict[FEVERLabel, int] = {}
        for v in votes:
            counts[v] = counts.get(v, 0) + 1

        top_label, top_count = max(counts.items(), key=lambda x: x[1])
        team_size = len(votes)

        agreed = (top_count == team_size) if strict_unanimous else (top_count > team_size / 2)
        if agreed:
            consensus_count += 1
            if top_label != yt:
                false_consensus_count += 1

    overall_fcr = round(false_consensus_count / total, 4) if total > 0 else 0.0
    conditional_fcr = round(false_consensus_count / consensus_count, 4) if consensus_count > 0 else 0.0

    if return_details:
        return {
            "overall_fcr": overall_fcr,
            "conditional_fcr": conditional_fcr,
            "consensus_count": consensus_count,
            "false_consensus_count": false_consensus_count,
            "total_evaluated": total
        }

    return overall_fcr


def calculate_citation_metrics(
    judge_outputs: List[Any]
) -> Dict[str, Any]:
    """Calculates citation validity, unsupported citation rate, and evidence grounding.
    
    A claim decision is defined as grounded if:
    - For SUPPORTS/REFUTES: at least one valid retrieved passage was cited and zero unsupported citations.
    - For NOT ENOUGH INFO: no unsupported citations were fabricated.
    """
    total_evaluated = len(judge_outputs)
    if total_evaluated == 0:
        return {
            "total_citations": 0,
            "valid_citations": 0,
            "unsupported_citations": 0,
            "citation_validity_rate": 1.0,
            "unsupported_citation_rate": 0.0,
            "grounding_rate": 0.0
        }

    total_citations = 0
    valid_citations = 0
    unsupported_citations = 0
    grounded_claims = 0

    for out in judge_outputs:
        v_c = getattr(out, "valid_citations", []) or []
        u_c = getattr(out, "unsupported_citations", []) or []
        total_citations += len(v_c) + len(u_c)
        valid_citations += len(v_c)
        unsupported_citations += len(u_c)

        pred_lbl = getattr(out, "predicted_label", None)
        if pred_lbl in (FEVERLabel.SUPPORTS, FEVERLabel.REFUTES):
            if len(v_c) > 0 and len(u_c) == 0:
                grounded_claims += 1
        elif pred_lbl == FEVERLabel.NOT_ENOUGH_INFO:
            if len(u_c) == 0:
                grounded_claims += 1

    validity_rate = round(valid_citations / total_citations, 4) if total_citations > 0 else 1.0
    unsupported_rate = round(unsupported_citations / total_citations, 4) if total_citations > 0 else 0.0
    grounding_rate = round(grounded_claims / total_evaluated, 4) if total_evaluated > 0 else 0.0

    return {
        "total_citations": total_citations,
        "valid_citations": valid_citations,
        "unsupported_citations": unsupported_citations,
        "citation_validity_rate": validity_rate,
        "unsupported_citation_rate": unsupported_rate,
        "grounding_rate": grounding_rate
    }


def calculate_adversarial_vulnerability_score(
    clean_accuracy: float,
    adversarial_accuracy: float
) -> float:
    """Calculates Adversarial Vulnerability Score (AVS).
    
    AVS = Accuracy(clean) - Accuracy(with adversarial agent)
    Higher AVS indicates greater vulnerability to adversarial deception.
    """
    return round(clean_accuracy - adversarial_accuracy, 4)
