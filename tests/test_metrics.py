"""Unit tests for ConsensusX evaluation metrics: accuracy, macro-F1, confusion matrix, and abstention."""

from src.data.schema import FEVERLabel
from src.evaluation.metrics import (
    calculate_classification_metrics,
    calculate_confusion_matrix,
    calculate_abstention_metrics,
    calculate_false_consensus_rate,
    calculate_adversarial_vulnerability_score,
)


def test_classification_metrics_perfect():
    """Verify metrics when predictions are 100% accurate."""
    y_true = [FEVERLabel.SUPPORTS, FEVERLabel.REFUTES, FEVERLabel.NOT_ENOUGH_INFO]
    y_pred = [FEVERLabel.SUPPORTS, FEVERLabel.REFUTES, FEVERLabel.NOT_ENOUGH_INFO]

    res = calculate_classification_metrics(y_true, y_pred)
    assert res["accuracy"] == 1.0
    assert res["macro_f1"] == 1.0


def test_confusion_matrix_computation():
    """Verify 3x3 confusion matrix calculation matches exact ground truth and prediction pairs."""
    y_true = [
        FEVERLabel.SUPPORTS,
        FEVERLabel.SUPPORTS,
        FEVERLabel.REFUTES,
        FEVERLabel.NOT_ENOUGH_INFO
    ]
    y_pred = [
        FEVERLabel.SUPPORTS,        # True SUPPORTS, Pred SUPPORTS (correct)
        FEVERLabel.REFUTES,         # True SUPPORTS, Pred REFUTES (error)
        FEVERLabel.REFUTES,         # True REFUTES, Pred REFUTES (correct)
        FEVERLabel.SUPPORTS         # True NEI, Pred SUPPORTS (false certainty)
    ]

    cm = calculate_confusion_matrix(y_true, y_pred)
    matrix = cm["matrix"]

    assert matrix["SUPPORTS"]["SUPPORTS"] == 1
    assert matrix["SUPPORTS"]["REFUTES"] == 1
    assert matrix["SUPPORTS"]["NOT ENOUGH INFO"] == 0

    assert matrix["REFUTES"]["REFUTES"] == 1
    assert matrix["REFUTES"]["SUPPORTS"] == 0

    assert matrix["NOT ENOUGH INFO"]["SUPPORTS"] == 1
    assert matrix["NOT ENOUGH INFO"]["NOT ENOUGH INFO"] == 0


def test_abstention_metrics():
    """Verify abstention metrics and false certainty calculations."""
    y_true = [
        FEVERLabel.SUPPORTS,
        FEVERLabel.REFUTES,
        FEVERLabel.NOT_ENOUGH_INFO,
        FEVERLabel.NOT_ENOUGH_INFO
    ]
    y_pred = [
        FEVERLabel.SUPPORTS,         # Decided & correct
        FEVERLabel.NOT_ENOUGH_INFO,  # Abstained on a REFUTES claim
        FEVERLabel.NOT_ENOUGH_INFO,  # Correctly abstained on true NEI
        FEVERLabel.SUPPORTS          # Falsely decided on true NEI (false certainty!)
    ]

    abst = calculate_abstention_metrics(y_true, y_pred)
    assert abst["total_samples"] == 4
    assert abst["abstention_count"] == 2
    assert abst["abstention_rate"] == 0.50

    # Decided claims: items 0 and 3 -> item 0 correct, item 3 incorrect -> 1/2 = 0.50
    assert abst["decided_count"] == 2
    assert abst["decided_accuracy"] == 0.50

    # Gold NEI claims: items 2 and 3 (count = 2)
    # Correctly abstained: item 2 (count = 1) -> recall = 0.50
    assert abst["gold_nei_count"] == 2
    assert abst["gold_nei_correctly_abstained"] == 1
    assert abst["gold_nei_recall"] == 0.50

    # False certainty: item 3 (count = 1) -> rate = 0.50
    assert abst["false_certainty_count"] == 1
    assert abst["false_certainty_rate"] == 0.50


def test_false_consensus_rate():
    """Verify false consensus rate calculation."""
    y_true = [FEVERLabel.SUPPORTS, FEVERLabel.REFUTES, FEVERLabel.SUPPORTS]
    team_verdicts = [
        [FEVERLabel.REFUTES, FEVERLabel.REFUTES, FEVERLabel.REFUTES],
        [FEVERLabel.REFUTES, FEVERLabel.REFUTES, FEVERLabel.SUPPORTS],
        [FEVERLabel.SUPPORTS, FEVERLabel.REFUTES, FEVERLabel.NOT_ENOUGH_INFO]
    ]
    fcr = calculate_false_consensus_rate(y_true, team_verdicts)
    assert round(fcr, 2) == 0.33


def test_adversarial_vulnerability_score():
    """Verify AVS calculation."""
    clean_acc = 0.85
    adv_acc = 0.60
    avs = calculate_adversarial_vulnerability_score(clean_acc, adv_acc)
    assert round(avs, 2) == 0.25
