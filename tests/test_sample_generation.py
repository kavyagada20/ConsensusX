"""Unit tests for deterministic sample generation."""

from pathlib import Path
import json
from src.data.loader import FEVERDataLoader
from src.data.schema import FEVERLabel


def test_stratified_sampling_determinism(tmp_path: Path):
    """Verify that identical random seeds produce identical samples in the exact same order."""
    dev_path = Path("data/raw/fever/shared_task_dev.jsonl")
    if not dev_path.exists():
        return

    loader = FEVERDataLoader(dev_path)

    sample1 = loader.create_stratified_sample(sample_size=30, random_seed=42)
    sample2 = loader.create_stratified_sample(sample_size=30, random_seed=42)
    sample3 = loader.create_stratified_sample(sample_size=30, random_seed=999)

    # Identical seed -> identical IDs
    ids1 = [r.id for r in sample1]
    ids2 = [r.id for r in sample2]
    ids3 = [r.id for r in sample3]

    assert ids1 == ids2
    assert ids1 != ids3  # Different seed -> different sample

    # Check balanced counts (10 per class)
    labels = [r.label for r in sample1]
    assert labels.count(FEVERLabel.SUPPORTS) == 10
    assert labels.count(FEVERLabel.REFUTES) == 10
    assert labels.count(FEVERLabel.NOT_ENOUGH_INFO) == 10


def test_dev_sample_300_file():
    """Verify that dev_sample_300.jsonl is balanced and structurally valid."""
    sample_file = Path("data/processed/dev_sample_300.jsonl")
    if not sample_file.exists():
        return

    records = []
    with open(sample_file, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    assert len(records) == 300
    labels = [r["label"] for r in records]
    assert labels.count("SUPPORTS") == 100
    assert labels.count("REFUTES") == 100
    assert labels.count("NOT ENOUGH INFO") == 100

