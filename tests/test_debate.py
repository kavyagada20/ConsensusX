"""Unit tests for Multi-Agent Structured Debate (Config 3), debater agents, and transcript logging."""

import json
from pathlib import Path
import pytest

from src.agents.debater import DebaterAgent
from src.data.schema import DebaterTurnOutput, FEVERLabel, FEVERRecord
from src.llm.base import LLMResponse
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


def test_debate_execution_order_initial_stances_before_rebuttal():
    """Verify execution order: all Round 1 initial stances complete before any Round 2 rebuttal begins."""
    call_log = []

    def spy_handler(prompt: str) -> str:
        if "Current Debate Round: Round 2" in prompt:
            call_log.append("round_2")
        else:
            call_log.append("round_1")
        return json.dumps({
            "predicted_label": "SUPPORTS",
            "confidence": 0.9,
            "evidence": "Factual evidence",
            "inference": "Logical inference",
            "uncertainty": "None"
        })

    client = MockLLMClient(custom_handler=spy_handler)
    pipeline = DebatePipeline(llm_client=client, num_agents=3, max_rounds=2)

    rec = FEVERRecord(id=500, claim="Spy trace claim", label=FEVERLabel.SUPPORTS)
    pipeline.run_debate(rec)

    # 3 debaters in Round 1, then 3 debaters in Round 2
    assert len(call_log) == 6
    assert call_log[:3] == ["round_1", "round_1", "round_1"]
    assert call_log[3:] == ["round_2", "round_2", "round_2"]


def test_round1_input_excludes_peer_outputs_and_identities():
    """Verify Round 1 prompt contains only the claim and evidence, strictly excluding peer arguments or transcripts."""
    recorded_prompts = []

    def spy_handler(prompt: str) -> str:
        recorded_prompts.append(prompt)
        return json.dumps({
            "predicted_label": "SUPPORTS",
            "confidence": 0.9,
            "evidence": "E",
            "inference": "I",
            "uncertainty": "U"
        })

    client = MockLLMClient(custom_handler=spy_handler)
    pipeline = DebatePipeline(llm_client=client, num_agents=3, max_rounds=1)

    rec = FEVERRecord(id=501, claim="Isolation audit claim", label=FEVERLabel.SUPPORTS)
    pipeline.run_debate(rec, evidence_context="Allowed reference evidence sentence.")

    assert len(recorded_prompts) == 3
    for p in recorded_prompts:
        assert "Isolation audit claim" in p
        assert "Allowed reference evidence sentence." in p
        # Must NOT contain peer names, verdicts, or arguments
        assert "Arguments from other Debaters" not in p
        assert "debater_1" not in p
        assert "debater_2" not in p
        assert "debater_3" not in p
        assert "rebuttal" not in p.lower()


def test_response_mutation_does_not_affect_other_agents_or_stored_history():
    """Verify that mutating an agent's turn output object does not mutate other agents or stored transcript."""
    client = MockLLMClient(default_verdict="SUPPORTS")
    pipeline = DebatePipeline(llm_client=client, num_agents=3, max_rounds=2)

    rec = FEVERRecord(id=502, claim="Mutation safety claim", label=FEVERLabel.SUPPORTS)
    out = pipeline.run_debate(rec)

    turn_0 = out.transcript.rounds[1][0]
    turn_1 = out.transcript.rounds[1][1]

    # Mutate turn_0's cited evidence list and metadata
    turn_0.cited_evidence.append("CORRUPTED_CITATION")
    turn_0.metadata["tampered"] = True

    # Assert turn_1 is completely unaffected
    assert "CORRUPTED_CITATION" not in turn_1.cited_evidence
    assert "tampered" not in turn_1.metadata

    # Verify that in round 2, debater_2's prompt was not corrupted by turn_0 mutation after the fact
    r2_turn_1 = out.transcript.rounds[2][1]
    assert "CORRUPTED_CITATION" not in (r2_turn_1.metadata.get("input_prompt") or "")


def test_later_rounds_receive_only_intended_peer_turns():
    """Verify that in Round 2, an agent receives peer turns only from other debaters, never its own prior turn."""
    received_peer_prompts = {}

    def spy_handler(prompt: str) -> str:
        if "Current Debate Round: Round 2" in prompt:
            # Detect which agent is being prompted based on own verdict section
            for deb_id in ["debater_1", "debater_2", "debater_3"]:
                if f"You are Debater Agent '{deb_id}'" in prompt:
                    received_peer_prompts[deb_id] = prompt
        return json.dumps({
            "predicted_label": "SUPPORTS",
            "confidence": 0.9,
            "evidence": "E",
            "inference": "I",
            "uncertainty": "U"
        })

    client = MockLLMClient(custom_handler=spy_handler)
    pipeline = DebatePipeline(llm_client=client, num_agents=3, max_rounds=2)

    rec = FEVERRecord(id=503, claim="Peer isolation claim", label=FEVERLabel.SUPPORTS)
    out = pipeline.run_debate(rec)

    # Check peer turns for each debater in Round 2
    r2_turns = out.transcript.rounds[2]
    for turn in r2_turns:
        prompt_text = turn.metadata.get("input_prompt", "")
        # Debater 1 should see debater_2 and debater_3 in peer section, NOT [debater_1]:
        if turn.agent_id == "debater_1":
            assert "[debater_2]:" in prompt_text
            assert "[debater_3]:" in prompt_text
            assert "[debater_1]:" not in prompt_text
        elif turn.agent_id == "debater_2":
            assert "[debater_1]:" in prompt_text
            assert "[debater_3]:" in prompt_text
            assert "[debater_2]:" not in prompt_text
        elif turn.agent_id == "debater_3":
            assert "[debater_1]:" in prompt_text
            assert "[debater_2]:" in prompt_text
            assert "[debater_3]:" not in prompt_text


def test_cache_key_isolation_prevents_cross_agent_cache_collision(tmp_path: Path):
    """Verify that with caching enabled, distinct debaters in Round 1 do not collide on cache keys."""
    class DummyCachingClient(MockLLMClient):
        def __init__(self, cdir: Path):
            super().__init__(model_name="caching-model")
            self.cache_dir = cdir
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache_enabled = True
            self.real_calls = 0

        def _compute_cache_key(self, prompt: str, system_instruction: str = "") -> str:
            import hashlib
            key = f"{self.model_name}:{system_instruction}:{prompt}"
            return hashlib.sha256(key.encode("utf-8")).hexdigest()

        def generate(self, prompt: str, system_instruction: str = "", **kwargs):
            ck = self._compute_cache_key(prompt, system_instruction)
            cf = self.cache_dir / f"{ck}.json"
            if cf.exists():
                with open(cf, "r", encoding="utf-8") as f:
                    return LLMResponse(**json.load(f))
            self.real_calls += 1
            resp = LLMResponse(
                content=json.dumps({
                    "predicted_label": "SUPPORTS",
                    "confidence": 0.9,
                    "evidence": f"Unique call #{self.real_calls}",
                    "inference": "Valid reasoning",
                    "uncertainty": "None"
                }),
                prompt_tokens=10,
                completion_tokens=10,
                total_tokens=20,
                latency_seconds=0.0,
                model=self.model_name,
                provider="mock",
                prompt_version="v1.0.0"
            )
            with open(cf, "w", encoding="utf-8") as f:
                json.dump(resp.model_dump(), f)
            return resp

    c_client = DummyCachingClient(tmp_path / "cache")
    c_pipeline = DebatePipeline(llm_client=c_client, num_agents=3, max_rounds=1)

    rec = FEVERRecord(id=504, claim="Cache isolation claim", label=FEVERLabel.SUPPORTS)
    out = c_pipeline.run_debate(rec)

    # In Round 1, 3 distinct debaters must execute 3 distinct calls without cache collision
    assert c_client.real_calls == 3
    r1_turns = out.transcript.rounds[1]
    evidences = [t.evidence for t in r1_turns]
    assert len(set(evidences)) == 3  # All 3 responses are unique


