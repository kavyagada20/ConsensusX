"""Unit tests for SingleAgentVerifier and BaselinePipeline."""

from pathlib import Path
from unittest.mock import MagicMock
from src.agents.single_agent import SingleAgentVerifier
from src.data.schema import FEVERLabel, FEVERRecord
from src.llm.mock import MockLLMClient
from src.pipeline.baseline import BaselinePipeline


def test_single_agent_verifier_parsing():
    """Verify JSON parsing and Markdown fence stripping in SingleAgentVerifier."""
    client = MockLLMClient()
    verifier = SingleAgentVerifier(llm_client=client, prompt_version="v1.0.0")

    raw_markdown = """```json
    {
      "predicted_label": "REFUTES",
      "confidence": 0.88,
      "reasoning": "Historical records contradict this claim.",
      "cited_evidence": ["History_Doc:5"]
    }
    ```"""
    out = verifier._parse_response(raw_markdown, claim_id=101, evidence_provided=True)
    assert out.claim_id == 101
    assert out.predicted_label == FEVERLabel.REFUTES
    assert out.confidence == 0.88
    assert "contradict" in out.reasoning
    assert out.cited_evidence == ["History_Doc:5"]
    assert out.prompt_version == "v1.0.0"


def test_single_agent_verifier_malformed_fallback():
    """Verify graceful fallback when LLM returns non-JSON text."""
    client = MockLLMClient()
    verifier = SingleAgentVerifier(llm_client=client)

    raw_text = "I think this claim might be unsupported, but I'm not entirely sure."
    out = verifier._parse_response(raw_text, claim_id=202, evidence_provided=False)
    assert out.claim_id == 202
    assert out.predicted_label == FEVERLabel.NOT_ENOUGH_INFO
    assert out.confidence == 0.0
    assert "Parsing error" in out.reasoning
    assert out.evidence_missing is True


def test_single_agent_verifier_api_error_handling():
    """Verify SingleAgentVerifier catches API exceptions without crashing."""
    broken_client = MagicMock()
    broken_client.generate.side_effect = RuntimeError("Connection timeout to API")
    broken_client.model_name = "gemini-2.0-flash"
    broken_client.provider = "gemini"

    verifier = SingleAgentVerifier(llm_client=broken_client)
    rec = FEVERRecord(id=999, claim="A test claim", label=FEVERLabel.SUPPORTS)

    out = verifier.verify(rec)
    assert out.claim_id == 999
    assert out.predicted_label == FEVERLabel.NOT_ENOUGH_INFO
    assert out.confidence == 0.0
    assert "API execution error" in out.reasoning
    assert out.evidence_missing is True


def test_baseline_pipeline_offline_run(tmp_path: Path):
    """Verify BaselinePipeline runs end-to-end with MockLLMClient."""
    records = [
        FEVERRecord(id=1, claim="The Earth orbits the Sun.", label=FEVERLabel.SUPPORTS),
        FEVERRecord(id=2, claim="The Moon is made of cheese.", label=FEVERLabel.REFUTES),
        FEVERRecord(id=3, claim="An unknown actor was born in 1980.", label=FEVERLabel.NOT_ENOUGH_INFO)
    ]

    mock_client = MockLLMClient(default_verdict="SUPPORTS")
    pipeline = BaselinePipeline(llm_client=mock_client, prompt_version="v1.0.0")

    out_file = tmp_path / "baseline_out.json"
    summary = pipeline.evaluate_records(records=records, output_file=out_file)

    assert summary["total_samples"] == 3
    assert summary["is_mock_mode"] is True
    assert summary["prompt_version"] == "v1.0.0"
    # 1 out of 3 was SUPPORTS and mock always returned SUPPORTS -> accuracy = 1/3
    assert round(summary["accuracy"], 2) == 0.33
    assert "macro_f1" in summary
    assert "confusion_matrix" in summary
    assert "abstention_analysis" in summary
    assert "cost_and_tokens" in summary
    assert out_file.exists()
