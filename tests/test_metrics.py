"""Unit tests for ConsensusX evaluation metrics."""

from consensus_x.data.schema import FEVERLabel
from consensus_x.evaluation.metrics import (
    calculate_classification_metrics,
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


def test_classification_metrics_imperfect():
    """Verify metrics calculation with mixed predictions."""
    y_true = [FEVERLabel.SUPPORTS, FEVERLabel.SUPPORTS, FEVERLabel.REFUTES]
    y_pred = [FEVERLabel.SUPPORTS, FEVERLabel.REFUTES, FEVERLabel.REFUTES]

    res = calculate_classification_metrics(y_true, y_pred)
    assert round(res["accuracy"], 2) == 0.67
    assert "SUPPORTS" in res["per_class_f1"]


def test_false_consensus_rate():
    """Verify false consensus rate calculation."""
    # 3 claims:
    # 1: True=SUPPORTS, Agents=[REFUTES, REFUTES, REFUTES] -> False consensus!
    # 2: True=REFUTES, Agents=[REFUTES, REFUTES, SUPPORTS] -> Correct majority consensus
    # 3: True=SUPPORTS, Agents=[SUPPORTS, REFUTES, NOT ENOUGH INFO] -> No majority consensus
    y_true = [FEVERLabel.SUPPORTS, FEVERLabel.REFUTES, FEVERLabel.SUPPORTS]
    team_verdicts = [
        [FEVERLabel.REFUTES, FEVERLabel.REFUTES, FEVERLabel.REFUTES],
        [FEVERLabel.REFUTES, FEVERLabel.REFUTES, FEVERLabel.SUPPORTS],
        [FEVERLabel.SUPPORTS, FEVERLabel.REFUTES, FEVERLabel.NOT_ENOUGH_INFO]
    ]
    fcr = calculate_false_consensus_rate(y_true, team_verdicts)
    # 1 out of 3 is a false consensus
    assert round(fcr, 2) == 0.33


def test_adversarial_vulnerability_score():
    """Verify AVS calculation."""
    clean_acc = 0.85
    adv_acc = 0.60
    avs = calculate_adversarial_vulnerability_score(clean_acc, adv_acc)
    assert round(avs, 2) == 0.25

