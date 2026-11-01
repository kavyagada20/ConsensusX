"""Unit tests for Multi-Agent Majority Voting (Config 2) and tie-breaking logic."""

from pathlib import Path
import pytest

from src.data.schema import ClaimVerificationOutput, FEVERLabel, FEVERRecord
from src.llm.mock import MockLLMClient
from src.pipeline.voting import MajorityVotingPipeline, aggregate_majority_votes


def test_majority_voting_clear_majority():
    """Verify standard plurality victory (2 SUPPORTS vs 1 REFUTES)."""
    votes = [
        ClaimVerificationOutput(
            claim_id=1,
            predicted_label=FEVERLabel.SUPPORTS,
            confidence=0.9,
            reasoning="Fact A",
            agent_id="agent_1"
        ),
        ClaimVerificationOutput(
            claim_id=1,
            predicted_label=FEVERLabel.SUPPORTS,
            confidence=0.85,
            reasoning="Fact B",
            agent_id="agent_2"
        ),
        ClaimVerificationOutput(
            claim_id=1,
            predicted_label=FEVERLabel.REFUTES,
            confidence=0.7,
            reasoning="Fact C",
            agent_id="agent_3"
        ),
    ]

    agg = aggregate_majority_votes(votes, claim_id=1)
    assert agg.predicted_label == FEVERLabel.SUPPORTS
    assert agg.is_tie is False
    assert agg.tie_break_applied is False
    assert agg.is_unanimous is False
    assert agg.vote_counts["SUPPORTS"] == 2
    assert agg.vote_counts["REFUTES"] == 1
    assert agg.successful_agents == 3
    assert agg.failed_agents == 0


def test_majority_voting_tie_breaking_confidence_weighted():
    """Verify primary tie-breaking: 3-way tie resolved by highest sum of confidences."""
    votes = [
        ClaimVerificationOutput(
            claim_id=2,
            predicted_label=FEVERLabel.SUPPORTS,
            confidence=0.6,
            reasoning="Weak support",
            agent_id="agent_1"
        ),
        ClaimVerificationOutput(
            claim_id=2,
            predicted_label=FEVERLabel.REFUTES,
            confidence=0.95,
            reasoning="Strong refutation",
            agent_id="agent_2"
        ),
        ClaimVerificationOutput(
            claim_id=2,
            predicted_label=FEVERLabel.NOT_ENOUGH_INFO,
            confidence=0.5,
            reasoning="Uncertain",
            agent_id="agent_3"
        ),
    ]

    agg = aggregate_majority_votes(votes, claim_id=2)
    assert agg.is_tie is True
    assert agg.tie_break_applied is True
    assert agg.tie_break_rule == "confidence_weighted"
    assert agg.predicted_label == FEVERLabel.REFUTES  # Highest confidence (0.95)


def test_majority_voting_tie_breaking_epistemic_safety_default():
    """Verify secondary tie-breaking: Equal votes AND equal confidences default to NOT ENOUGH INFO."""
    votes = [
        ClaimVerificationOutput(
            claim_id=3,
            predicted_label=FEVERLabel.SUPPORTS,
            confidence=0.80,
            reasoning="Support argument",
            agent_id="agent_1"
        ),
        ClaimVerificationOutput(
            claim_id=3,
            predicted_label=FEVERLabel.REFUTES,
            confidence=0.80,
            reasoning="Refute argument",
            agent_id="agent_2"
        ),
    ]

    agg = aggregate_majority_votes(votes, claim_id=3)
    assert agg.is_tie is True
    assert agg.tie_break_applied is True
    assert agg.tie_break_rule == "epistemic_humility_default_nei"
    assert agg.predicted_label == FEVERLabel.NOT_ENOUGH_INFO


def test_majority_voting_failed_agent_does_not_distort_vote():
    """Verify that a failed agent's response is excluded from majority counts and does not count as correct."""
    votes = [
        ClaimVerificationOutput(
            claim_id=4,
            predicted_label=FEVERLabel.REFUTES,
            confidence=0.85,
            reasoning="Valid refutation",
            agent_id="agent_1"
        ),
        ClaimVerificationOutput(
            claim_id=4,
            predicted_label=FEVERLabel.NOT_ENOUGH_INFO,
            confidence=0.0,
            reasoning="Parsing error: invalid JSON payload returned",
            agent_id="agent_2"
        ),
    ]

    agg = aggregate_majority_votes(votes, claim_id=4)
    assert agg.total_agents == 2
    assert agg.successful_agents == 1
    assert agg.failed_agents == 1
    # Only agent_1's valid vote is counted
    assert agg.predicted_label == FEVERLabel.REFUTES
    assert agg.vote_counts["REFUTES"] == 1
    assert agg.vote_counts["NOT ENOUGH INFO"] == 0


def test_majority_voting_all_agents_failed():
    """Verify graceful fallback when all agents fail."""
    votes = [
        ClaimVerificationOutput(
            claim_id=5,
            predicted_label=FEVERLabel.NOT_ENOUGH_INFO,
            confidence=0.0,
            reasoning="API timeout error",
            agent_id="agent_1"
        ),
        ClaimVerificationOutput(
            claim_id=5,
            predicted_label=FEVERLabel.NOT_ENOUGH_INFO,
            confidence=0.0,
            reasoning="Parsing error: empty response",
            agent_id="agent_2"
        ),
    ]

    agg = aggregate_majority_votes(votes, claim_id=5)
    assert agg.predicted_label == FEVERLabel.NOT_ENOUGH_INFO
    assert agg.confidence == 0.0
    assert agg.successful_agents == 0
    assert agg.failed_agents == 2
    assert agg.tie_break_rule == "all_agents_failed"


def test_majority_voting_pipeline_mock(tmp_path: Path):
    """Verify MajorityVotingPipeline runs end-to-end on FEVERRecords with mock LLM and produces report."""
    mock_client = MockLLMClient(default_verdict="SUPPORTS")
    pipeline = MajorityVotingPipeline(llm_client=mock_client, num_agents=3)

    records = [
        FEVERRecord(id=101, claim="The Earth orbits the Sun.", label=FEVERLabel.SUPPORTS),
        FEVERRecord(id=102, claim="The Moon is made of cheese.", label=FEVERLabel.REFUTES),
    ]

    out_file = tmp_path / "voting_test_output.json"
    summary = pipeline.evaluate_records(records=records, output_file=out_file)

    assert summary["total_samples"] == 2
    assert summary["num_agents"] == 3
    assert summary["accuracy"] == 0.5  # 1 SUPPORTS matches, 1 REFUTES doesn't
    assert summary["unanimous_decisions"] == 2
    assert out_file.exists()

