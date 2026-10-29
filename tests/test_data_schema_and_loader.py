"""Unit tests for FEVER schemas, label normalization, and data loader."""

import pytest
from pathlib import Path
from consensus_x.data.schema import FEVERLabel, FEVERRecord, ClaimVerificationOutput
from consensus_x.data.loader import FEVERDataLoader


def test_fever_label_normalization():
    """Verify that different raw representations correctly map to FEVERLabel."""
    assert FEVERLabel.from_str("SUPPORTS") == FEVERLabel.SUPPORTS
    assert FEVERLabel.from_str("supports") == FEVERLabel.SUPPORTS
    assert FEVERLabel.from_str("REFUTES") == FEVERLabel.REFUTES
    assert FEVERLabel.from_str("NOT ENOUGH INFO") == FEVERLabel.NOT_ENOUGH_INFO
    assert FEVERLabel.from_str("not-enough-info") == FEVERLabel.NOT_ENOUGH_INFO
    assert FEVERLabel.from_str("NEI") == FEVERLabel.NOT_ENOUGH_INFO

    with pytest.raises(ValueError):
        FEVERLabel.from_str("UNKNOWN_LABEL")


def test_fever_record_evidence_extraction():
    """Verify parsing of FEVER evidence tuples."""
    sample = {
        "id": 1234,
        "claim": "Test claim statement.",
        "label": "SUPPORTS",
        "verifiable": "VERIFIABLE",
        "evidence": [
            [
                [1, 10, "Albert_Einstein", 3],
                [1, 10, "Physics", 0]
            ]
        ]
    }
    rec = FEVERRecord(**sample)
    assert rec.id == 1234
    assert rec.label == FEVERLabel.SUPPORTS
    gold_sets = rec.get_gold_evidence_sets()
    assert len(gold_sets) == 1
    assert ("Albert_Einstein", 3) in gold_sets[0]
    assert ("Physics", 0) in gold_sets[0]


def test_claim_verification_output_schema():
    """Verify ClaimVerificationOutput creation."""
    out = ClaimVerificationOutput(
        claim_id=1234,
        predicted_label=FEVERLabel.SUPPORTS,
        confidence=0.92,
        reasoning="Sufficient evidence found.",
        cited_evidence=["Albert_Einstein:3"]
    )
    assert out.predicted_label == FEVERLabel.SUPPORTS
    assert out.confidence == 0.92
    assert not out.is_adversarial


def test_data_loader_streaming():
    """Verify FEVERDataLoader streams records correctly from the raw dev dataset."""
    dev_path = Path("data/raw/fever/shared_task_dev.jsonl")
    if dev_path.exists():
        loader = FEVERDataLoader(dev_path)
        records = list(loader.stream_records(limit=5))
        assert len(records) == 5
        for r in records:
            assert isinstance(r.id, int)
            assert len(r.claim) > 0
            assert r.label in [FEVERLabel.SUPPORTS, FEVERLabel.REFUTES, FEVERLabel.NOT_ENOUGH_INFO]

