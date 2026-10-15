"""The single registry of the thirty sub-agents — declaration + public builder, paired.

Peer to [`leads/registry.py`](../leads/registry.py:1). ADR-0007 phase 1 established that the
true harness LEAVES are the sub-agents, not the Leads (a harness has no "call another harness"
tool type), so anything that enumerates harness-hostable leaves needs this list — and it must be
ONE list, not a second copy that can drift from the declarations.

Import direction: each sub-agent module imports :mod:`amlas.subagents.declarations`; this
registry imports the sub-agent modules. Nothing under ``tools/`` imports ``subagents``, so there
is no cycle.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from strands import Agent

from . import declarations as _decl
from .algo_hypothesis import build_algo_hypothesis_agent
from .algorithm_specialist import build_algorithm_specialist_agent
from .business_goal_advisor import build_business_goal_advisor
from .data_collection_agent import build_data_collection_agent
from .data_integrity import build_data_integrity_agent
from .declarations import SubagentDeclaration
from .deployment_agent import build_deployment_agent
from .devops_git_agent import build_devops_git_agent
from .dq_assessment import build_dq_assessment_agent
from .feature_hypothesis import build_feature_hypothesis_agent
from .fine_tune import build_fine_tune_agent
from .framing import build_framing_agent
from .governance_agent import build_governance_agent
from .improvement_agent import build_improvement_agent
from .imputation_strategist import build_imputation_strategist_agent
from .labeling_strategist import build_labeling_strategist
from .llm_extrapolation import build_llm_extrapolation_agent
from .monitoring_agent import build_monitoring_agent
from .problem_framing_advisor import build_problem_framing_advisor
from .readiness_review_agent import build_readiness_review_agent
from .research_curator import build_research_curator
from .research_discovery import build_research_discovery_agent
from .research_scout import build_research_scout
from .sagemaker_ops import build_sagemaker_ops_agent
from .solution_fit_advisor import build_solution_fit_advisor
from .solution_survey_agent import build_solution_survey_agent
from .spec_writer import build_spec_writer_agent
from .synthetic_data import build_synthetic_data_agent
from .tabular_fe import build_tabular_fe_agent
from .trainer_tuner import build_trainer_tuner_agent
from .vision_fe import build_vision_fe_agent


@dataclass(frozen=True)
class SubagentEntry:
    """One sub-agent: its offline declaration plus the public builder that realises it."""

    declaration: SubagentDeclaration
    builder: Callable[..., Agent]

    @property
    def key(self) -> str:
        return self.declaration.key


SUBAGENTS: tuple[SubagentEntry, ...] = (
    SubagentEntry(_decl.ALGO_HYPOTHESIS, build_algo_hypothesis_agent),
    SubagentEntry(_decl.ALGORITHM_SPECIALIST, build_algorithm_specialist_agent),
    SubagentEntry(_decl.BUSINESS_GOAL_ADVISOR, build_business_goal_advisor),
    SubagentEntry(_decl.DATA_COLLECTION_AGENT, build_data_collection_agent),
    SubagentEntry(_decl.DATA_INTEGRITY, build_data_integrity_agent),
    SubagentEntry(_decl.DEPLOYMENT_AGENT, build_deployment_agent),
    SubagentEntry(_decl.DEVOPS_GIT_AGENT, build_devops_git_agent),
    SubagentEntry(_decl.DQ_ASSESSMENT, build_dq_assessment_agent),
    SubagentEntry(_decl.FEATURE_HYPOTHESIS, build_feature_hypothesis_agent),
    SubagentEntry(_decl.FINE_TUNE, build_fine_tune_agent),
    SubagentEntry(_decl.FRAMING, build_framing_agent),
    SubagentEntry(_decl.GOVERNANCE_AGENT, build_governance_agent),
    SubagentEntry(_decl.IMPROVEMENT_AGENT, build_improvement_agent),
    SubagentEntry(_decl.IMPUTATION_STRATEGIST, build_imputation_strategist_agent),
    SubagentEntry(_decl.LABELING_STRATEGIST, build_labeling_strategist),
    SubagentEntry(_decl.LLM_EXTRAPOLATION, build_llm_extrapolation_agent),
    SubagentEntry(_decl.MONITORING_AGENT, build_monitoring_agent),
    SubagentEntry(_decl.PROBLEM_FRAMING_ADVISOR, build_problem_framing_advisor),
    SubagentEntry(_decl.READINESS_REVIEW_AGENT, build_readiness_review_agent),
    SubagentEntry(_decl.RESEARCH_CURATOR, build_research_curator),
    SubagentEntry(_decl.RESEARCH_DISCOVERY, build_research_discovery_agent),
    SubagentEntry(_decl.RESEARCH_SCOUT, build_research_scout),
    SubagentEntry(_decl.SAGEMAKER_OPS, build_sagemaker_ops_agent),
    SubagentEntry(_decl.SOLUTION_FIT_ADVISOR, build_solution_fit_advisor),
    SubagentEntry(_decl.SOLUTION_SURVEY_AGENT, build_solution_survey_agent),
    SubagentEntry(_decl.SPEC_WRITER, build_spec_writer_agent),
    SubagentEntry(_decl.SYNTHETIC_DATA, build_synthetic_data_agent),
    SubagentEntry(_decl.TABULAR_FE, build_tabular_fe_agent),
    SubagentEntry(_decl.TRAINER_TUNER, build_trainer_tuner_agent),
    SubagentEntry(_decl.VISION_FE, build_vision_fe_agent),
)

#: ``SUBAGENTS`` keyed by sub-agent key, for direct lookup.
SUBAGENTS_BY_KEY: dict[str, SubagentEntry] = {e.key: e for e in SUBAGENTS}


def subagent_entry(key: str) -> SubagentEntry:
    """Return the :class:`SubagentEntry` for ``key``, raising ``KeyError`` with valid keys."""
    try:
        return SUBAGENTS_BY_KEY[key]
    except KeyError:
        raise KeyError(
            f"unknown sub-agent {key!r}; known: {sorted(SUBAGENTS_BY_KEY)}"
        ) from None


def subagent_declaration(key: str) -> SubagentDeclaration:
    """The declaration behind sub-agent ``key`` — readable with no ``Agent`` built."""
    return subagent_entry(key).declaration


def subagent_keys() -> tuple[str, ...]:
    """The thirty sub-agent keys, in registry order."""
    return tuple(e.key for e in SUBAGENTS)


__all__ = [
    "SUBAGENTS",
    "SUBAGENTS_BY_KEY",
    "SubagentEntry",
    "subagent_declaration",
    "subagent_entry",
    "subagent_keys",
]
