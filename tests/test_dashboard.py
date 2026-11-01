"""Unit tests for ConsensusX Streamlit Research Dashboard."""

import pytest
from pathlib import Path


def test_app_file_exists():
    """Validates that app.py is present in the repository root."""
    app_path = Path("app.py")
    assert app_path.exists(), "app.py must be present in the repository root."


def test_app_imports_and_components():
    """Validates that all core modules used by the dashboard are importable."""
    from src.pipeline.baseline import BaselinePipeline
    from src.pipeline.voting import MajorityVotingPipeline
    from src.pipeline.debate import DebatePipeline
    from src.pipeline.consensus_x import ConsensusXPipeline
    from src.retrieval.retriever import FAISSRetriever
    from src.retrieval.vector_store import FAISSVectorStore
    from src.evaluation.metrics import (
        calculate_adversarial_flip_rate,
        calculate_adversarial_vulnerability_score,
        validate_evaluation_cohort,
    )
    from app import load_vector_store_retriever

    retriever, err = load_vector_store_retriever()
    # If FAISS index exists, retriever is loaded; if not, graceful error string is returned
    if Path("data/processed/faiss_index").exists():
        assert retriever is not None
        assert err is None
    else:
        assert err is not None


def test_app_recorded_experiments_parsing():
    """Validates that the dashboard can parse recorded benchmark JSON outputs."""
    import json
    report_file = Path("outputs/adversarial_experiment_report.json")
    if report_file.exists():
        with open(report_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "benchmarks" in data
        assert "sample_size" in data
        assert "cohort_claim_ids" in data
        for cfg_name, d in data["benchmarks"].items():
            if "flip_metrics" in d:
                fm = d["flip_metrics"]
                assert "categories" in fm
                assert "adversarial_flip_rate" in fm

