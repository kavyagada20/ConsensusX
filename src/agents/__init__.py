"""Agent definitions and personas for ConsensusX."""

from src.agents.single_agent import SingleAgentVerifier
from src.agents.debater import DebaterAgent, DEBATE_PROMPT_VERSION
from src.agents.judge import EvidenceJudgeAgent, JUDGE_PROMPT_VERSION
from src.agents.adversary import AdversaryAgent, ADVERSARY_PROMPT_VERSION

__all__ = [
    "SingleAgentVerifier",
    "DebaterAgent",
    "DEBATE_PROMPT_VERSION",
    "EvidenceJudgeAgent",
    "JUDGE_PROMPT_VERSION",
    "AdversaryAgent",
    "ADVERSARY_PROMPT_VERSION",
]
