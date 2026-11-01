"""Unit tests for Evidence-Based Judge, Citation Provenance, and Adaptive Stopping (Phase 6)."""

import pytest
from src.agents.judge import EvidenceJudgeAgent, JUDGE_PROMPT_VERSION
from src.data.schema import (
    DebateTranscript,
    DebaterTurnOutput,
    FEVERLabel,
    FEVERRecord,
    JudgeEvaluationOutput,
)
from src.evaluation.metrics import (
    calculate_citation_metrics,
    calculate_false_consensus_rate,
)
from src.llm.base import LLMResponse
from src.llm.mock import MockLLMClient
from src.pipeline.adaptive_stopping import AdaptiveStoppingController
from src.pipeline.consensus_x import ConsensusXPipeline
from src.retrieval.schema import RetrievedPassage


@pytest.fixture
def sample_record() -> FEVERRecord:
    return FEVERRecord(
        id=101,
        claim="Firefox is a piece of hardware.",
        label=FEVERLabel.REFUTES,
        verifiable="VERIFIABLE",
        evidence=[[[1, 1, "Firefox", 1]]],
    )


@pytest.fixture
def retrieved_passages() -> list[RetrievedPassage]:
    return [
        RetrievedPassage(
            passage_id="Firefox:1",
            title="Firefox",
            sentence_id=1,
            text="Firefox is a software application, not a piece of hardware.",
            score=0.775,
            rank=1,
        ),
        RetrievedPassage(
            passage_id="Mozilla:0",
            title="Mozilla",
            sentence_id=0,
            text="Mozilla is a free software community.",
            score=0.450,
            rank=2,
        ),
    ]


@pytest.fixture
def mock_transcript() -> DebateTranscript:
    turn1 = DebaterTurnOutput(
        agent_id="debater_1",
        round_number=1,
        predicted_label=FEVERLabel.REFUTES,
        confidence=0.9,
        evidence="Firefox is software.",
        inference="Since it is software, it cannot be hardware.",
        cited_evidence=["Firefox:1"],
        status="SUCCESS",
    )
    turn2 = DebaterTurnOutput(
        agent_id="debater_2",
        round_number=1,
        predicted_label=FEVERLabel.REFUTES,
        confidence=0.85,
        evidence="Mozilla develops open source software.",
        inference="Hardware claim is false.",
        cited_evidence=["Mozilla:0"],
        status="SUCCESS",
    )
    return DebateTranscript(
        claim_id=101,
        claim="Firefox is a piece of hardware.",
        rounds={1: [turn1, turn2]},
        initial_verdicts={"debater_1": FEVERLabel.REFUTES, "debater_2": FEVERLabel.REFUTES},
        final_verdicts={"debater_1": FEVERLabel.REFUTES, "debater_2": FEVERLabel.REFUTES},
        total_rounds=1,
    )


# ==============================================================================
# 1. CITATION PROVENANCE & UNSUPPORTED CITATIONS TESTS
# ==============================================================================

def test_citation_verification_valid_and_unsupported(retrieved_passages):
    judge = EvidenceJudgeAgent(llm_client=MockLLMClient())

    # Case A: Valid citations matching retrieved passages
    valid, unsupported, is_valid = judge.verify_citations(
        raw_citations=["Firefox:1", "[Mozilla:0]"],
        retrieved_passages=retrieved_passages,
    )
    assert "Firefox:1" in valid
    assert "Mozilla:0" in valid
    assert len(unsupported) == 0
    assert is_valid is True

    # Case B: Citations containing hallucinated / ungrounded IDs
    valid, unsupported, is_valid = judge.verify_citations(
        raw_citations=["Firefox:1", "Hallucinated_Document:42"],
        retrieved_passages=retrieved_passages,
    )
    assert "Firefox:1" in valid
    assert "Hallucinated_Document:42" in unsupported
    assert is_valid is False


def test_epistemic_safety_override_downgrades_ungrounded_verdict(sample_record, retrieved_passages, mock_transcript):
    """If judge predicts SUPPORTS or REFUTES without any valid retrieved citations,
    epistemic safety override MUST downgrade to NOT ENOUGH INFO."""

    def ungrounded_handler(prompt: str) -> str:
        return (
            '{"predicted_label": "REFUTES", '
            '"confidence": 0.95, '
            '"rationale": "I think it is false based on my internal world knowledge.", '
            '"cited_evidence": ["MadeUp_Source:99"], '
            '"missing_evidence": false}'
        )

    client = MockLLMClient(custom_handler=ungrounded_handler)
    judge = EvidenceJudgeAgent(llm_client=client)

    result = judge.evaluate(
        record=sample_record,
        retrieved_passages=retrieved_passages,
        transcript=mock_transcript,
    )

    # Must be downgraded to NOT ENOUGH INFO
    assert result.predicted_label == FEVERLabel.NOT_ENOUGH_INFO
    assert result.is_citation_valid is False
    assert "MadeUp_Source:99" in result.unsupported_citations
    assert result.missing_evidence is True
    assert result.metadata.get("override_applied") is True


# ==============================================================================
# 2. INSUFFICIENT EVIDENCE & EPISTEMIC HUMILITY
# ==============================================================================

def test_insufficient_evidence_yields_not_enough_info(sample_record, mock_transcript):
    """When no passages are retrieved or evidence is missing, judge must predict NOT ENOUGH INFO."""

    def nei_handler(prompt: str) -> str:
        return (
            '{"predicted_label": "NOT ENOUGH INFO", '
            '"confidence": 0.80, '
            '"rationale": "Retrieved evidence does not contain information to confirm or refute the claim.", '
            '"cited_evidence": [], '
            '"missing_evidence": true, '
            '"uncertainty": "No relevant Wikipedia articles were retrieved."}'
        )

    client = MockLLMClient(custom_handler=nei_handler)
    judge = EvidenceJudgeAgent(llm_client=client)

    result = judge.evaluate(
        record=sample_record,
        retrieved_passages=[],  # Empty retrieved passages
        transcript=mock_transcript,
    )

    assert result.predicted_label == FEVERLabel.NOT_ENOUGH_INFO
    assert result.missing_evidence is True
    assert len(result.valid_citations) == 0
    assert result.is_citation_valid is True  # NEI with empty citations is validly ungrounded


# ==============================================================================
# 3. CONFLICTING EVIDENCE HANDLING
# ==============================================================================

def test_conflicting_evidence_tracking(sample_record, mock_transcript):
    """Judge explicitly tracks conflicting evidence statements and residual uncertainty."""
    conflicting_passages = [
        RetrievedPassage(
            passage_id="DocA:0",
            title="DocA",
            sentence_id=0,
            text="Entity X won the award in 2020.",
            score=0.8,
            rank=1,
        ),
        RetrievedPassage(
            passage_id="DocB:0",
            title="DocB",
            sentence_id=0,
            text="Entity X was disqualified in 2020 and never received any award.",
            score=0.79,
            rank=2,
        ),
    ]

    def conflicting_handler(prompt: str) -> str:
        return (
            '{"predicted_label": "NOT ENOUGH INFO", '
            '"confidence": 0.50, '
            '"rationale": "Evidence presents direct conflict between DocA and DocB.", '
            '"cited_evidence": ["DocA:0", "DocB:0"], '
            '"conflicting_evidence": "DocA states Entity X won the award, whereas DocB states Entity X was disqualified.", '
            '"missing_evidence": false, '
            '"uncertainty": "Contradictory primary sources without corroboration."}'
        )

    client = MockLLMClient(custom_handler=conflicting_handler)
    judge = EvidenceJudgeAgent(llm_client=client)

    result = judge.evaluate(
        record=sample_record,
        retrieved_passages=conflicting_passages,
        transcript=mock_transcript,
    )

    assert result.conflicting_evidence is not None
    assert "DocA" in result.conflicting_evidence
    assert result.uncertainty != ""
    assert "DocA:0" in result.valid_citations
    assert "DocB:0" in result.valid_citations


# ==============================================================================
# 4. ADAPTIVE STOPPING CONTROLLER TESTS
# ==============================================================================

def test_adaptive_stopping_refuses_consensus_without_evidence(retrieved_passages):
    """CRITICAL TEST: Controller must NOT stop merely because agents unanimously agree,
    if evidence sufficiency is NOT satisfied (ungrounded consensus)."""
    controller = AdaptiveStoppingController(
        min_rounds=1,
        max_rounds=3,
        stability_threshold=1.0,
        require_evidence_grounding=True,
    )

    # Agents in Round 2 unanimously agree on SUPPORTS, but cite NO valid citations
    prev_turns = [
        DebaterTurnOutput(agent_id="d1", round_number=1, predicted_label=FEVERLabel.SUPPORTS, cited_evidence=[]),
        DebaterTurnOutput(agent_id="d2", round_number=1, predicted_label=FEVERLabel.SUPPORTS, cited_evidence=[]),
    ]
    curr_turns = [
        DebaterTurnOutput(agent_id="d1", round_number=2, predicted_label=FEVERLabel.SUPPORTS, cited_evidence=["Hallucinated:1"]),
        DebaterTurnOutput(agent_id="d2", round_number=2, predicted_label=FEVERLabel.SUPPORTS, cited_evidence=[]),
    ]

    res = controller.evaluate_stopping(
        round_number=2,
        current_turns=curr_turns,
        previous_turns=prev_turns,
        retrieved_passages=retrieved_passages,  # Contains Firefox:1, Mozilla:0
    )

    # Must refuse to stop early!
    assert res.should_stop is False
    assert res.reason == "consensus_reached_without_sufficient_evidence"
    assert res.consensus_reached is True
    assert res.evidence_sufficient is False


def test_adaptive_stopping_triggers_on_stability_and_valid_evidence(retrieved_passages):
    """When decision stability is achieved AND debaters cite valid retrieved evidence,
    early stopping triggers."""
    controller = AdaptiveStoppingController(
        min_rounds=1,
        max_rounds=3,
        stability_threshold=1.0,
        require_evidence_grounding=True,
    )

    prev_turns = [
        DebaterTurnOutput(agent_id="d1", round_number=1, predicted_label=FEVERLabel.REFUTES, cited_evidence=["Firefox:1"]),
        DebaterTurnOutput(agent_id="d2", round_number=1, predicted_label=FEVERLabel.REFUTES, cited_evidence=["Firefox:1"]),
    ]
    curr_turns = [
        DebaterTurnOutput(agent_id="d1", round_number=2, predicted_label=FEVERLabel.REFUTES, cited_evidence=["Firefox:1"]),
        DebaterTurnOutput(agent_id="d2", round_number=2, predicted_label=FEVERLabel.REFUTES, cited_evidence=["Firefox:1"]),
    ]

    res = controller.evaluate_stopping(
        round_number=2,
        current_turns=curr_turns,
        previous_turns=prev_turns,
        retrieved_passages=retrieved_passages,
    )

    assert res.should_stop is True
    assert res.reason == "evidence_sufficiency_and_decision_stability"
    assert res.stability_reached is True
    assert res.evidence_sufficient is True


def test_adaptive_stopping_hits_max_rounds():
    """Controller terminates when max_rounds bound is reached, even if unstable."""
    controller = AdaptiveStoppingController(min_rounds=1, max_rounds=2)

    res = controller.evaluate_stopping(
        round_number=2,
        current_turns=[],
        previous_turns=[],
        retrieved_passages=[],
    )

    assert res.should_stop is True
    assert res.reason == "max_rounds_reached"


# ==============================================================================
# 5. METRICS: FALSE CONSENSUS RATE & CITATION METRICS
# ==============================================================================

def test_false_consensus_rate_calculation():
    # 3 claims:
    # Claim 1: True SUPPORTS, Agents [REFUTES, REFUTES] -> False consensus!
    # Claim 2: True REFUTES, Agents [REFUTES, REFUTES] -> True consensus
    # Claim 3: True NEI, Agents [SUPPORTS, REFUTES] -> No consensus
    y_true = [FEVERLabel.SUPPORTS, FEVERLabel.REFUTES, FEVERLabel.NOT_ENOUGH_INFO]
    votes = [
        [FEVERLabel.REFUTES, FEVERLabel.REFUTES],
        [FEVERLabel.REFUTES, FEVERLabel.REFUTES],
        [FEVERLabel.SUPPORTS, FEVERLabel.REFUTES],
    ]

    fcr_overall = calculate_false_consensus_rate(y_true, votes, strict_unanimous=False)
    # 1 false consensus out of 3 total claims = 1/3 ~ 0.3333
    assert fcr_overall == 0.3333

    details = calculate_false_consensus_rate(y_true, votes, return_details=True)
    assert details["false_consensus_count"] == 1
    assert details["consensus_count"] == 2
    # Conditional FCR = 1 / 2 = 0.5
    assert details["conditional_fcr"] == 0.5


def test_citation_metrics():
    outputs = [
        JudgeEvaluationOutput(
            claim_id=1,
            predicted_label=FEVERLabel.SUPPORTS,
            valid_citations=["DocA:0"],
            unsupported_citations=[],
        ),
        JudgeEvaluationOutput(
            claim_id=2,
            predicted_label=FEVERLabel.REFUTES,
            valid_citations=["DocB:1"],
            unsupported_citations=["FakeDoc:0"],
        ),
        JudgeEvaluationOutput(
            claim_id=3,
            predicted_label=FEVERLabel.NOT_ENOUGH_INFO,
            valid_citations=[],
            unsupported_citations=[],
        ),
    ]

    metrics = calculate_citation_metrics(outputs)
    assert metrics["total_citations"] == 3
    assert metrics["valid_citations"] == 2
    assert metrics["unsupported_citations"] == 1
    assert metrics["citation_validity_rate"] == 0.6667
    assert metrics["grounding_rate"] == 0.6667  # Claims 1 and 3 are grounded, Claim 2 had unsupported citation


# ==============================================================================
# 6. END-TO-END CONSENSUSX PIPELINE
# ==============================================================================

def test_consensus_x_pipeline_run(sample_record, retrieved_passages):
    def debater_handler(prompt: str) -> str:
        return (
            '{"predicted_label": "REFUTES", '
            '"confidence": 0.85, '
            '"evidence": "Firefox is software application.", '
            '"inference": "Software cannot be hardware.", '
            '"uncertainty": "None", '
            '"rebuttal": "Peer is correct.", '
            '"cited_evidence": ["Firefox:1"], '
            '"verdict_revised": false}'
        )

    def judge_handler(prompt: str) -> str:
        return (
            '{"predicted_label": "REFUTES", '
            '"confidence": 0.95, '
            '"rationale": "Firefox:1 explicitly contradicts the claim.", '
            '"cited_evidence": ["Firefox:1"], '
            '"conflicting_evidence": null, '
            '"missing_evidence": false, '
            '"uncertainty": "None"}'
        )

    debater_client = MockLLMClient(custom_handler=debater_handler)
    judge_client = MockLLMClient(custom_handler=judge_handler)

    pipeline = ConsensusXPipeline(
        llm_client=debater_client,
        judge_client=judge_client,
        num_agents=2,
        max_rounds=2,
        min_rounds=1,
    )

    out = pipeline.run_consensus_x(
        record=sample_record,
        retrieved_passages=retrieved_passages,
    )

    assert out.claim_id == 101
    assert out.predicted_label == FEVERLabel.REFUTES
    assert out.judge_output.is_citation_valid is True
    assert "Firefox:1" in out.judge_output.valid_citations
    assert out.rounds_used in (1, 2)
    assert out.stopping_reason != ""

