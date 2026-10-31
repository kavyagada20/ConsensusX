"""Unit tests for split integrity and data leakage prevention."""

import pytest
from src.data.loader import check_data_leakage
from src.data.schema import FEVERLabel, FEVERRecord


def test_split_integrity_zero_leakage():
    """Verify check_data_leakage succeeds when splits are completely disjoint."""
    split_train = [
        FEVERRecord(id=1, claim="Train claim 1", label=FEVERLabel.SUPPORTS),
        FEVERRecord(id=2, claim="Train claim 2", label=FEVERLabel.REFUTES),
        FEVERRecord(id=3, claim="Train claim 3", label=FEVERLabel.NOT_ENOUGH_INFO)
    ]
    split_eval = [
        FEVERRecord(id=101, claim="Eval claim 1", label=FEVERLabel.SUPPORTS),
        FEVERRecord(id=102, claim="Eval claim 2", label=FEVERLabel.REFUTES)
    ]

    # Should succeed without error
    leakage = check_data_leakage(split_train, split_eval, "Train", "Eval")
    assert len(leakage) == 0


def test_split_integrity_detects_leakage():
    """Verify check_data_leakage raises ValueError when an ID exists in both splits."""
    split_train = [1, 2, 3, 4, 5]
    split_eval = [5, 6, 7]  # ID 5 is leaked!

    with pytest.raises(ValueError, match="DATA LEAKAGE DETECTED"):
        check_data_leakage(split_train, split_eval, "Train", "Eval")

