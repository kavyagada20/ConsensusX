"""Unit tests for Multi-Agent Structured Debate (Config 3), debater agents, and transcript logging."""

import json
from pathlib import Path
import pytest

from src.agents.debater import DebaterAgent
from src.data.schema import DebaterTurnOutput, FEVERLabel, FEVERRecord
from src.llm.mock import MockLLMClient
from src.pipeline.debate import DebatePipeline, aggregate_debate_verdict


def test_debater_initial_stance_distinguishes_evidence_inference_uncertainty():
    """Verify debater parses evidence, inference, and uncertainty in initial independent stance."""
    def custom_handler(prompt: str) -> str:
        return json.dumps({
            "predicted_label": "REFUTES",
            "confidence": 0.92,
            "evidence": "Historical records show the event occurred in 1945.",
            "inference": "The claim states 1950, which directly contradicts 1945.",
            "uncertainty": "No uncertainty on date; primary sources agree.",
            "cited_evidence": ["World_War_II:12"]
        })

    client = MockLLMClient(custom_handler=custom_handler)
    agent = DebaterAgent(agent_id="debater_test", llm_client=client)

    rec = FEVERRecord(id=1, claim="World War II ended in 1950.", label=FEVERLabel.REFUTES)
    turn = agent.initial_stance(record=rec)

    assert turn.agent_id == "debater_test"
    assert turn.round_number == 1
    assert turn.predicted_label == FEVERLabel.REFUTES
    assert turn.confidence == 0.92
    assert "1945" in turn.evidence
    assert "contradicts" in turn.inference
    assert "No uncertainty" in turn.uncertainty
    assert turn.cited_evidence == ["World_War_II:12"]
    assert turn.status == "SUCCESS"


def test_debater_rebuttal_and_opinion_shift():
    """Verify debater inspects peer arguments, responds in rebuttal round, and records opinion shift."""
    # Round 1 turn: initially thought SUPPORTS
    r1_turn = DebaterTurnOutput(
        agent_id="debater_1",
        round_number=1,
        predicted_label=FEVERLabel.SUPPORTS,
        confidence=0.6,
        evidence="Initial assumption.",
        inference="Seems plausible.",
        uncertainty="Moderate",
        cited_evidence=[]
    )

    # Peer turn from Round 1
    peer_r1_turn = DebaterTurnOutput(
        agent_id="debater_2",
        round_number=1,
        predicted_label=FEVERLabel.REFUTES,
        confidence=0.95,
        evidence="Official census shows population was 5 million, not 50 million.",
        inference="Direct 10x contradiction.",
        uncertainty="None",
        cited_evidence=["Census_Record:1"]
    )

    def custom_handler(prompt: str) -> str:
        # Assert peer arguments were actually presented in prompt
        assert "debater_2" in prompt
        assert "5 million" in prompt
        return json.dumps({
            "predicted_label": "REFUTES",
            "confidence": 0.90,
            "evidence": "Census data provided by debater_2.",
            "inference": "The claim is refuted by official census statistics.",
            "uncertainty": "None remaining.",
            "rebuttal": "Debater 2's citation is authoritative; conceding my initial SUPPORTS stance.",
            "cited_evidence": ["Census_Record:1"],
            "verdict_revised": True
        })

    client = MockLLMClient(custom_handler=custom_handler)
    agent = DebaterAgent(agent_id="debater_1", llm_client=client)

    rec = FEVERRecord(id=2, claim="The city population was 50 million.", label=FEVERLabel.REFUTES)
    r2_turn = agent.rebuttal_and_revision(
        record=rec,
        round_number=2,
        peer_turns=[peer_r1_turn],
        own_previous_turn=r1_turn
    )

    assert r2_turn.round_number == 2
    assert r2_turn.predicted_label == FEVERLabel.REFUTES
    assert r2_turn.verdict_revised is True
    assert "conceding" in (r2_turn.rebuttal or "")


def test_debate_pipeline_round_limit_and_transcript(tmp_path: Path):
    """Verify debate pipeline respects max_rounds=2, records full transcript, and aggregates consensus."""
    client = MockLLMClient(default_verdict="SUPPORTS")
    pipeline = DebatePipeline(llm_client=client, num_agents=3, max_rounds=2)

    rec = FEVERRecord(id=200, claim="Water boils at 100 degrees Celsius.", label=FEVERLabel.SUPPORTS)
    debate_out = pipeline.run_debate(rec)

    assert debate_out.claim_id == 200
    assert debate_out.predicted_label == FEVERLabel.SUPPORTS
    assert debate_out.consensus_reached is True
    assert debate_out.is_unanimous is True

    # Verify transcript structure
    transcript = debate_out.transcript
    assert transcript.total_rounds == 2
    assert 1 in transcript.rounds
    assert 2 in transcript.rounds
    assert 3 not in transcript.rounds  # strictly bounded at max_rounds=2
    assert len(transcript.rounds[1]) == 3
    assert len(transcript.rounds[2]) == 3


def test_debate_pipeline_evaluates_records(tmp_path: Path):
    """Verify DebatePipeline evaluates records end-to-end and outputs structured evaluation metrics."""
    client = MockLLMClient(default_verdict="SUPPORTS")
    pipeline = DebatePipeline(llm_client=client, num_agents=3, max_rounds=2)

    records = [
        FEVERRecord(id=301, claim="Claim A", label=FEVERLabel.SUPPORTS),
        FEVERRecord(id=302, claim="Claim B", label=FEVERLabel.REFUTES),
    ]

    out_file = tmp_path / "debate_eval.json"
    summary = pipeline.evaluate_records(records=records, output_file=out_file)

    assert summary["total_samples"] == 2
    assert summary["num_agents"] == 3
    assert summary["max_rounds"] == 2
    assert summary["accuracy"] == 0.5
    assert summary["unanimous_decisions"] == 2
    assert out_file.exists()


def test_aggregate_debate_verdict_tie_breaking():
    """Verify tie-breaking on debate final-round verdicts."""
    final_turns = [
        DebaterTurnOutput(
            agent_id="d1",
            round_number=2,
            predicted_label=FEVERLabel.SUPPORTS,
            confidence=0.95,
            evidence="A",
            inference="B"
        ),
        DebaterTurnOutput(
            agent_id="d2",
            round_number=2,
            predicted_label=FEVERLabel.REFUTES,
            confidence=0.70,
            evidence="C",
            inference="D"
        ),
    ]

    agg = aggregate_debate_verdict(final_turns, claim_id=99)
    assert agg["is_tie"] is True
    assert agg["tie_break_applied"] is True
    assert agg["winning_label"] == FEVERLabel.SUPPORTS  # Higher confidence (0.95 vs 0.70)

