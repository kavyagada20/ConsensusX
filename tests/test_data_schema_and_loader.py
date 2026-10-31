"""Unit tests for FEVER schemas, label normalization, evidence preservation, and data loader."""

import json
from pathlib import Path
import pytest

from src.data.schema import (
    ClaimVerificationOutput,
    DatasetManifest,
    FEVERLabel,
    FEVERRecord,
)
from src.data.loader import FEVERDataLoader


def test_fever_label_normalization():
    """Verify that all label variants normalize correctly to FEVERLabel enum."""
    # Standard and informal string mappings
    assert FEVERLabel.from_str("SUPPORTS") == FEVERLabel.SUPPORTS
    assert FEVERLabel.from_str("Supported") == FEVERLabel.SUPPORTS
    assert FEVERLabel.from_str("supports") == FEVERLabel.SUPPORTS

    assert FEVERLabel.from_str("REFUTES") == FEVERLabel.REFUTES
    assert FEVERLabel.from_str("Refuted") == FEVERLabel.REFUTES
    assert FEVERLabel.from_str("refutes") == FEVERLabel.REFUTES

    assert FEVERLabel.from_str("NOT ENOUGH INFO") == FEVERLabel.NOT_ENOUGH_INFO
    assert FEVERLabel.from_str("Not Enough Information") == FEVERLabel.NOT_ENOUGH_INFO
    assert FEVERLabel.from_str("not-enough-info") == FEVERLabel.NOT_ENOUGH_INFO
    assert FEVERLabel.from_str("NEI") == FEVERLabel.NOT_ENOUGH_INFO
    assert FEVERLabel.from_str("unverifiable") == FEVERLabel.NOT_ENOUGH_INFO

    with pytest.raises(ValueError, match="Unrecognized FEVER label"):
        FEVERLabel.from_str("INVALID_LABEL_XYZ")


def test_fever_record_evidence_extraction_and_no_fabrication():
    """Verify gold evidence extraction preserves metadata and never fabricates evidence for NEI claims."""
    # Case 1: Verifiable record with multiple sentences
    verifiable_sample = {
        "id": 1001,
        "claim": "Albert Einstein won the Nobel Prize in Physics.",
        "label": "SUPPORTS",
        "verifiable": "VERIFIABLE",
        "evidence": [
            [
                [1, 10, "Albert_Einstein", 5],
                [1, 10, "Nobel_Prize_in_Physics", 0]
            ]
        ]
    }
    rec_v = FEVERRecord(**verifiable_sample)
    assert rec_v.id == 1001
    assert rec_v.label == FEVERLabel.SUPPORTS
    assert rec_v.has_gold_evidence() is True
    gold_sets = rec_v.get_gold_evidence_sets()
    assert len(gold_sets) == 1
    assert ("Albert_Einstein", 5) in gold_sets[0]
    assert ("Nobel_Prize_in_Physics", 0) in gold_sets[0]

    # Case 2: Unverifiable record with null evidence (NOT ENOUGH INFO)
    nei_sample = {
        "id": 1002,
        "claim": "An unknown individual lived on Mars.",
        "label": "NOT ENOUGH INFO",
        "verifiable": "NOT VERIFIABLE",
        "evidence": [[[108548, None, None, None]]]
    }
    rec_nei = FEVERRecord(**nei_sample)
    assert rec_nei.id == 1002
    assert rec_nei.label == FEVERLabel.NOT_ENOUGH_INFO
    assert rec_nei.has_gold_evidence() is False
    # Crucial rule: Do NOT silently fabricate evidence when evidence is missing
    assert rec_nei.get_gold_evidence_sets() == []


def test_claim_verification_output_schema():
    """Verify ClaimVerificationOutput creation with prompt version and evidence flags."""
    out = ClaimVerificationOutput(
        claim_id=1234,
        predicted_label=FEVERLabel.SUPPORTS,
        confidence=0.92,
        reasoning="Sufficient evidence found.",
        cited_evidence=["Albert_Einstein:3"],
        prompt_version="v1.0.0",
        evidence_missing=False
    )
    assert out.predicted_label == FEVERLabel.SUPPORTS
    assert out.confidence == 0.92
    assert out.prompt_version == "v1.0.0"
    assert not out.evidence_missing
    assert not out.is_adversarial


def test_dataset_manifest_schema():
    """Verify DatasetManifest schema serialization and fields."""
    manifest = DatasetManifest(
        source_file="data/raw/fever/shared_task_dev.jsonl",
        output_file="data/processed/dev_sample_300.jsonl",
        sample_size=300,
        random_seed=42,
        label_distribution={"SUPPORTS": 100, "REFUTES": 100, "NOT ENOUGH INFO": 100},
        claim_ids=[101, 102, 103],
        description="Test manifest"
    )
    assert manifest.sample_size == 300
    assert manifest.random_seed == 42
    assert manifest.label_distribution["SUPPORTS"] == 100
    dumped = json.loads(manifest.model_dump_json())
    assert dumped["manifest_version"] == "1.0.0"


def test_synthetic_fixture_data_loader(tmp_path: Path):
    """Verify FEVERDataLoader on a tiny synthetic JSONL fixture."""
    fixture_path = tmp_path / "synthetic_fever.jsonl"
    synthetic_records = [
        {"id": 1, "claim": "Claim 1", "label": "SUPPORTS", "verifiable": "VERIFIABLE", "evidence": [[[1, 1, "P1", 0]]]},
        {"id": 2, "claim": "Claim 2", "label": "REFUTES", "verifiable": "VERIFIABLE", "evidence": [[[2, 2, "P2", 1]]]},
        {"id": 3, "claim": "Claim 3", "label": "NOT ENOUGH INFO", "verifiable": "NOT VERIFIABLE", "evidence": [[[3, None, None, None]]]},
    ]
    with open(fixture_path, "w", encoding="utf-8") as f:
        for r in synthetic_records:
            f.write(json.dumps(r) + "\n")

    loader = FEVERDataLoader(fixture_path)
    records = list(loader.stream_records())
    assert len(records) == 3
    assert records[0].id == 1
    assert records[0].label == FEVERLabel.SUPPORTS
    assert records[1].id == 2
    assert records[1].label == FEVERLabel.REFUTES
    assert records[2].id == 3
    assert records[2].label == FEVERLabel.NOT_ENOUGH_INFO
