"""Unit tests for Controlled Adversarial Testing and Robustness Evaluation (Phase 7)."""

import pytest
from src.agents.adversary import AdversaryAgent, ADVERSARY_PROMPT_VERSION
from src.agents.debater import DebaterAgent
from src.data.schema import FEVERLabel, FEVERRecord
from src.evaluation.metrics import (
    calculate_adversarial_flip_rate,
    calculate_adversarial_vulnerability_score,
    calculate_wilson_confidence_interval,
)
from src.llm.mock import MockLLMClient
from src.pipeline.consensus_x import ConsensusXPipeline
from src.pipeline.debate import DebatePipeline
from src.pipeline.voting import MajorityVotingPipeline
from src.retrieval.schema import RetrievedPassage


@pytest.fixture
def sample_supports_record() -> FEVERRecord:
    return FEVERRecord(
        id=501,
        claim="Civilization IV received high ratings.",
        label=FEVERLabel.SUPPORTS,
        verifiable="VERIFIABLE",
        evidence=[[[1, 1, "Civilization_IV", 1]]],
    )


@pytest.fixture
def sample_refutes_record() -> FEVERRecord:
    return FEVERRecord(
        id=502,
        claim="Firefox is a piece of hardware.",
        label=FEVERLabel.REFUTES,
        verifiable="VERIFIABLE",
        evidence=[[[1, 1, "Firefox", 1]]],
    )


@pytest.fixture
def retrieved_passages_firefox() -> list[RetrievedPassage]:
    return [
        RetrievedPassage(
            passage_id="Firefox:1",
            title="Firefox",
            sentence_id=1,
            text="Firefox is a software application, not a piece of hardware.",
            score=0.775,
            rank=1,
        )
    ]


# ==============================================================================
# 1. ADVERSARY AGENT PERSONAS & STRENGTHS
# ==============================================================================

def test_adversary_contrarian_target_inversion(sample_supports_record, sample_refutes_record):
    """Contrarian adversary targets the exact opposite of the true claim status."""
    adv = AdversaryAgent(agent_id="adv_1", persona="persuasive_contrarian", attack_strength="assertive")

    # On SUPPORTS claim, adversary targets REFUTES
    out_sup = adv.verify(sample_supports_record)
    assert out_sup.predicted_label == FEVERLabel.REFUTES
    assert out_sup.confidence == 0.98
    assert out_sup.is_adversarial is True

    # On REFUTES claim, adversary targets SUPPORTS
    out_ref = adv.verify(sample_refutes_record)
    assert out_ref.predicted_label == FEVERLabel.SUPPORTS
    assert out_ref.confidence == 0.98
    assert out_ref.is_adversarial is True


def test_adversary_citation_distorter(sample_supports_record):
    """Citation distorter invents plausible fabricated passage IDs."""
    adv = AdversaryAgent(agent_id="adv_distort", persona="citation_distorter", attack_strength="assertive")
    turn = adv.initial_stance(sample_supports_record)

    assert turn.predicted_label == FEVERLabel.REFUTES
    assert len(turn.cited_evidence) > 0
    assert "Counter_Fact_501:1" in turn.cited_evidence[0]
    assert turn.metadata.get("is_adversarial") is True


def test_adversary_attack_strengths(sample_supports_record):
    """Validates assertive (0.98) versus subtle (0.76) confidence scaling."""
    adv_assertive = AdversaryAgent(attack_strength="assertive")
    adv_subtle = AdversaryAgent(attack_strength="subtle")

    turn_assertive = adv_assertive.initial_stance(sample_supports_record)
    turn_subtle = adv_subtle.initial_stance(sample_supports_record)

    assert turn_assertive.confidence == 0.98
    assert turn_subtle.confidence == 0.76


# ==============================================================================
# 2. MAJORITY VOTING UNDER ADVERSARIAL ATTACK
# ==============================================================================

def test_majority_voting_adversary_injection(sample_supports_record):
    """Majority voting with 1 adversary injected in a 3-agent committee."""
    client_honest = MockLLMClient(default_verdict="SUPPORTS")

    # Clean committee: 3 honest agents -> Unanimous SUPPORTS
    clean_pipe = MajorityVotingPipeline(llm_client=client_honest, num_agents=3, inject_adversary=False)
    clean_out = clean_pipe.evaluate_records([sample_supports_record])
    assert clean_out["accuracy"] == 1.0

    # Adversarial committee: 2 honest (SUPPORTS, conf 0.95), 1 adversary (REFUTES, conf 0.98)
    adv_pipe = MajorityVotingPipeline(
        llm_client=client_honest,
        num_agents=3,
        inject_adversary=True,
        adversary_idx=1,
        adversary_persona="persuasive_contrarian",
        adversary_strength="assertive",
    )
    adv_out = adv_pipe.evaluate_records([sample_supports_record])
    # 2 vs 1 -> Honest majority wins on 3-agent committee
    assert adv_out["accuracy"] == 1.0
    res = adv_out["results"][0]
    assert res["vote_counts"]["SUPPORTS"] == 2
    assert res["vote_counts"]["REFUTES"] == 1


# ==============================================================================
# 3. STRUCTURED DEBATE UNDER ADVERSARIAL ATTACK
# ==============================================================================

def test_debate_adversary_rebuttal(sample_supports_record):
    """Adversary in structured debate mounts aggressive rebuttal attacking peers."""
    client_honest = MockLLMClient(default_verdict="SUPPORTS")

    debate_pipe = DebatePipeline(
        llm_client=client_honest,
        num_agents=3,
        max_rounds=2,
        inject_adversary=True,
        adversary_idx=1,
    )

    out = debate_pipe.run_debate(sample_supports_record)
    transcript = out.transcript

    # Round 1 has 3 turns
    assert len(transcript.rounds[1]) == 3
    # Agent 2 is the adversary
    assert transcript.rounds[1][1].predicted_label == FEVERLabel.REFUTES
    assert transcript.rounds[1][1].metadata.get("is_adversarial") is True

    # Round 2 has adversary rebuttal challenging debater_1 and debater_3
    r2_adv_turn = transcript.rounds[2][1]
    assert "debater_1" in r2_adv_turn.rebuttal
    assert "debater_3" in r2_adv_turn.rebuttal


# ==============================================================================
# 4. CONSENSUSX RESISTANCE TO ADVERSARIAL INFLUENCE
# ==============================================================================

def test_consensus_x_neutralizes_adversary(sample_refutes_record, retrieved_passages_firefox):
    """In ConsensusX, Evidence Judge checks retrieved passages and rejects adversary's ungrounded claims."""

    def honest_debater(prompt: str) -> str:
        return (
            '{"predicted_label": "REFUTES", '
            '"confidence": 0.90, '
            '"evidence": "Firefox is software application, not hardware.", '
            '"inference": "Hardware claim is false.", '
            '"uncertainty": "None", '
            '"rebuttal": "Peer is correct.", '
            '"cited_evidence": ["Firefox:1"], '
            '"verdict_revised": false}'
        )

    def judge_handler(prompt: str) -> str:
        # Judge cites authentic retrieved passage Firefox:1 and ignores adversary's ungrounded claim
        return (
            '{"predicted_label": "REFUTES", '
            '"confidence": 0.95, '
            '"rationale": "Firefox:1 explicitly contradicts the claim. Debater 2 assertions are ungrounded.", '
            '"cited_evidence": ["Firefox:1"], '
            '"conflicting_evidence": null, '
            '"missing_evidence": false, '
            '"uncertainty": "None"}'
        )

    honest_client = MockLLMClient(custom_handler=honest_debater)
    judge_client = MockLLMClient(custom_handler=judge_handler)

    pipeline = ConsensusXPipeline(
        llm_client=honest_client,
        judge_client=judge_client,
        num_agents=3,
        max_rounds=2,
        inject_adversary=True,
        adversary_idx=1,  # Injected adversary
    )

    out = pipeline.run_consensus_x(
        record=sample_refutes_record,
        retrieved_passages=retrieved_passages_firefox,
    )

    # Final verdict is correctly REFUTES despite adversary advocating SUPPORTS
    assert out.predicted_label == FEVERLabel.REFUTES
    assert out.judge_output.is_citation_valid is True
    assert "Firefox:1" in out.judge_output.valid_citations


# ==============================================================================
# 5. METRICS: AVS, ADVERSARIAL FLIP RATE, & WILSON CI
# ==============================================================================

def test_metrics_adversarial_vulnerability_and_flip_rate():
    clean_acc = 0.85
    adv_acc = 0.65
    avs = calculate_adversarial_vulnerability_score(clean_acc, adv_acc)
    assert avs == 0.20

    clean_results = [
        {"id": 1, "is_correct": True, "predicted_label": "SUPPORTS", "gold_label": "SUPPORTS"},
        {"id": 2, "is_correct": True, "predicted_label": "REFUTES", "gold_label": "REFUTES"},
        {"id": 3, "is_correct": False, "predicted_label": "REFUTES", "gold_label": "SUPPORTS"},
    ]
    # In adversarial run, claim 1 is flipped to incorrect
    adv_results = [
        {"id": 1, "is_correct": False, "predicted_label": "REFUTES", "gold_label": "SUPPORTS"},
        {"id": 2, "is_correct": True, "predicted_label": "REFUTES", "gold_label": "REFUTES"},
        {"id": 3, "is_correct": False, "predicted_label": "REFUTES", "gold_label": "SUPPORTS"},
    ]

    flip_metrics = calculate_adversarial_flip_rate(clean_results, adv_results)
    assert flip_metrics["clean_correct_count"] == 2
    assert flip_metrics["adversarial_flipped_count"] == 1
    assert flip_metrics["adversarial_flip_rate"] == 0.5  # 1 out of 2 previously correct claims flipped


def test_wilson_confidence_interval():
    # Test 50 correct out of 100
    lower, upper = calculate_wilson_confidence_interval(k=50, n=100, confidence=0.95)
    assert 0.40 <= lower <= 0.41
    assert 0.59 <= upper <= 0.61

    # Edge cases
    assert calculate_wilson_confidence_interval(k=0, n=0) == (0.0, 0.0)
    lower_zero, upper_zero = calculate_wilson_confidence_interval(k=0, n=50)
    assert lower_zero == 0.0
    assert upper_zero > 0.0

