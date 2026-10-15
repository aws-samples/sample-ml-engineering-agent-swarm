"""Agentic Pipeline Runner — LLM-driven autonomous ML pipeline orchestration.

The pipeline agent has access to ALL tools across the system and uses an LLM
(Sonnet/Opus) to reason about what to do next. It observes tool outputs,
makes decisions, and drives the full ML workflow autonomously.

Usage:
    agent = build_pipeline_agent()
    result = agent("Analyze data at /path/to/data.csv, target='label', task=classification")
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from strands import Agent

from .._build import build_agent
from ..prompts import stack_prompts

# Phase 6/7/8 data-engineering tools (Capability Groups A/B/C) — exposed so the
# autonomous pipeline agent can drive the DQ → impute → synthesize lifecycle the
# Data Engineering Lead (D0) orchestrates. Imported here only (no logic change).
from ..tools.data_quality import (
    check_expectations,
    diagnose_missingness,
    scan_dq_patterns,
    score_data_quality_6dim,
)

# Tools — imported from all modules
from ..tools.data_quality_tools import (
    detect_anomalies,
    generate_eda_report,
    score_data_quality,
)
from ..tools.data_tools import describe_dataset, load_dataset, split_dataset
from ..tools.decision_tools import (
    generate_model_card_tool,
    generate_next_steps_report,
    get_recommendation,
)
from ..tools.evaluation_tools import (
    compare_experiments,
    get_best_model,
    get_leaderboard,
    record_experiment_result,
)
from ..tools.fe_tools import auto_preprocess
from ..tools.hpo_tools import train_ensemble, tune_model
from ..tools.imputation import (
    apply_imputation_plan,
    evaluate_imputation,
    recommend_imputation_plan,
)

# Phase 10 MLOps + Lake tools (Capability Group D) — exposed so the autonomous
# pipeline agent can drive the agentic-over-SageMaker loop the MLOps Lead (M0)
# orchestrates: read a medallion silver slice → train/register/monitor on
# SageMaker → write cleaned/inferred data back to silver/gold + report KPIs.
from ..tools.lake import (
    list_lake_catalog,
    read_lake_slice,
    write_lake_dataset,
)
from ..tools.mlops import (
    approve_model_sm,
    build_sagemaker_pipeline,
    feature_store_get,
    feature_store_ingest,
    inspect_model_monitor,
    list_model_packages_sm,
    mlops_kpi_scorecard,
    register_model_sm,
    run_clarify,
    run_sagemaker_pipeline,
    setup_model_monitor,
)
from ..tools.research_tools import recommend_approaches, search_algorithms
from ..tools.spec_tools import validate_spec
from ..tools.synthetic import (
    build_synthesis_spec,
    synthesize_tabular,
    validate_synthetic,
)
from ..tools.training_tools import train_model

# ─── Constants ───────────────────────────────────────────────────────────────

PIPELINE_AGENT_ROLE = "planner"
"""Role selector — uses Opus 4.7 for complex reasoning via config."""

SYSTEM_PROMPT_SOURCES = (
    "preambles/paranoid_scientist",
    "leads/pipeline_orchestrator",
)
"""Prompt stack: paranoid scientist preamble + pipeline orchestrator instructions."""

# ─── All Tools Available to the Pipeline Agent ───────────────────────────────

PIPELINE_TOOLS = [
    # Data tools
    load_dataset,
    split_dataset,
    describe_dataset,
    # Data quality tools (new)
    score_data_quality,
    detect_anomalies,
    generate_eda_report,
    # Feature engineering
    auto_preprocess,
    # Research tools
    search_algorithms,
    recommend_approaches,
    # Training tools
    train_model,
    tune_model,
    train_ensemble,
    # Evaluation tools
    record_experiment_result,
    get_leaderboard,
    compare_experiments,
    get_best_model,
    # Decision tools
    get_recommendation,
    generate_model_card_tool,
    generate_next_steps_report,
    # Spec tools
    validate_spec,
    # Data-engineering tools (Phase 6/7/8 — Capability Groups A/B/C). These let the
    # autonomous pipeline agent run the DQ → impute → synthesize lifecycle the
    # Data Engineering Lead (D0) orchestrates as sub-agents.
    scan_dq_patterns,
    diagnose_missingness,
    check_expectations,
    score_data_quality_6dim,
    recommend_imputation_plan,
    evaluate_imputation,
    apply_imputation_plan,
    build_synthesis_spec,
    synthesize_tabular,
    validate_synthetic,
    # MLOps + Lake tools (Phase 10 — Capability Group D). These let the autonomous
    # pipeline agent drive the agentic-over-SageMaker loop the MLOps Lead (M0)
    # orchestrates as sub-agents: read a medallion silver slice → train / register /
    # monitor / explain on SageMaker → write cleaned/inferred data back to a
    # silver/gold tier (never raw) → report MLOps KPIs.
    read_lake_slice,
    write_lake_dataset,
    list_lake_catalog,
    build_sagemaker_pipeline,
    run_sagemaker_pipeline,
    register_model_sm,
    approve_model_sm,
    list_model_packages_sm,
    run_clarify,
    setup_model_monitor,
    inspect_model_monitor,
    feature_store_ingest,
    feature_store_get,
    mlops_kpi_scorecard,
]
"""Complete tool set for autonomous pipeline orchestration (45 tools)."""


# ─── Agent Factory ───────────────────────────────────────────────────────────


def build_pipeline_agent(
    *,
    model: Any | None = None,
    hooks_factory: Callable[[], Sequence[Any]] | None = None,
    name: str = "pipeline_orchestrator",
) -> Agent:
    """Build the autonomous ML pipeline agent.

    This agent has access to ALL tools and uses an LLM (role = planner → Opus
    4.7, see PIPELINE_AGENT_ROLE) to reason about what to do next. It:
    1. Assesses data quality
    2. Researches suitable approaches
    3. Runs experiments (choosing algos based on data characteristics)
    4. Evaluates and compares
    5. Makes recommendations

    Args:
        model: Optional pre-built model; None (default) uses real Bedrock. Callers may pass a real BedrockModel (real_bedrock fixture).
        hooks_factory: Optional zero-arg callable forwarded to ``build_agent``
            so the Orchestrator can attach a ``MemoryHookProvider`` scoped to
            ``(memory_id, actor_id=project_id, session_id=run_id)``.
        name: Agent name — defaults to "pipeline_orchestrator".
    """
    return build_agent(
        name=name,
        system_prompt=stack_prompts(*SYSTEM_PROMPT_SOURCES),
        tools=PIPELINE_TOOLS,
        model=model,
        role=PIPELINE_AGENT_ROLE,
        hooks_factory=hooks_factory,
        description=(
            "Autonomous ML pipeline orchestrator that drives the full workflow "
            "from data quality assessment through model training, evaluation, "
            "and recommendation using tool-calling."
        ),
    )
