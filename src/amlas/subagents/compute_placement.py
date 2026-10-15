"""Compute-placement guardrail (ADR-0002).

The 7 AgentCore/Strands leads + subagents ORCHESTRATE the ML lifecycle; they never run
compute in the AgentCore container / Code-Interpreter sandbox. All compute (training, HPO,
data cleansing, feature engineering, imputation, synthesis) dispatches to purpose-built AWS
services (SageMaker Training/AMT/Processing, Glue, Lambda, ECS/Fargate).

This module names the HEAVY in-container compute tools that must NOT appear in an
orchestrator's toolset, plus a helper to read a Strands agent's tool names. The binding test
(``validation coverage``) asserts no orchestrator subagent carries a HEAVY
compute tool. As each re-architecture phase re-points a subagent to a dispatch seam, the
corresponding tool leaves that subagent and the test moves toward green.

SCOPE, NOT REMOVAL (operator directive 2026-07-21): agents ALWAYS need LIGHT compute — quick
math, a small data sanity check, light analysis — so a general-purpose light tool like
``execute_python_tool`` is NOT banned. It is controlled by SCOPE: the agent's prompt + the
tool's hint/description bound it to light/orchestration-helper use, and HEAVY compute
(training, HPO, cleansing, FE, imputation, synthesis at scale) must dispatch to a managed AWS
service. Therefore ``execute_python_tool`` is deliberately EXCLUDED from ``HEAVY_COMPUTE_TOOL_NAMES``
below — the guardrail targets the heavy, purpose-built compute tools, not the light hatch.
See ADR-0002 (Tool Scoping section).
"""

from __future__ import annotations

from typing import Any

# Canonical names of HEAVY compute tools that perform training/cleansing/FE/imputation/
# synthesis in-container (runtime process OR Code-Interpreter sandbox) instead of dispatching
# to a managed AWS service. Sourced from the ADR-0002 inventory. An ORCHESTRATOR subagent must
# not carry any of these. ``execute_python_tool`` is intentionally NOT here (light-compute hatch,
# scoped by prompt/hint — see module docstring).
HEAVY_COMPUTE_TOOL_NAMES: frozenset[str] = frozenset(
    {
        # Training / HPO run in-process (sklearn/xgboost/optuna).
        "submit_training_trial",
        "train_model",
        "tune_model",
        "train_ensemble",
        # Feature engineering compute in-process.
        "auto_preprocess",
        # Imputation (sklearn imputers) in-process.
        "impute_numeric",
        "impute_categorical",
        "impute_timeseries",
        "evaluate_imputation",
        "apply_imputation_plan",
        # Synthesis (SDV/SMOTE) in-process.
        "synthesize_tabular",
        "augment_imbalance",
        "validate_synthetic",
        # Data-quality scans (pandas) in-process.
        "scan_dq_patterns",
        "diagnose_missingness",
        "score_data_quality_6dim",
        "check_expectations",
    }
)

# Backward-compat alias (some callers/tests may reference the old name).
COMPUTE_TOOL_NAMES = HEAVY_COMPUTE_TOOL_NAMES


def agent_tool_names(agent: Any) -> list[str]:
    """Return the tool names bound to a Strands ``Agent`` (best-effort, version-tolerant).

    Strands agents expose their tools differently across versions; we probe the common shapes
    and read each tool's ``tool_name`` (falling back to ``__name__``). Sub-agents wrapped as
    tools (``as_tool``) are included by their tool name. Never raises — returns [] if the shape
    is unrecognized (the guardrail test then simply sees no compute tools, which is safe).
    """
    tools = None
    for attr in ("tool_names",):
        val = getattr(agent, attr, None)
        if isinstance(val, (list, tuple, set)):
            return [str(t) for t in val]
    # Common: a tool registry / config carrying the tool objects.
    for attr in ("tools", "_tools"):
        val = getattr(agent, attr, None)
        if isinstance(val, (list, tuple)):
            tools = val
            break
        if isinstance(val, dict):
            tools = list(val.values())
            break
    # A ToolRegistry object (strands) — try its registry dict.
    if tools is None:
        reg = getattr(agent, "tool_registry", None) or getattr(agent, "_tool_registry", None)
        if reg is not None:
            rd = getattr(reg, "registry", None) or getattr(reg, "_registry", None)
            if isinstance(rd, dict):
                return [str(k) for k in rd.keys()]
            if isinstance(rd, (list, tuple)):
                tools = rd
    if tools is None:
        return []
    names: list[str] = []
    for t in tools:
        name = getattr(t, "tool_name", None) or getattr(t, "__name__", None) or str(t)
        names.append(str(name))
    return names


def compute_tools_in(agent: Any) -> set[str]:
    """Return the HEAVY-WORKLOAD compute tools an agent carries (empty set = ADR-0002-compliant).

    Tools that merely help the agent understand/decide (light math, inspection) are NOT counted —
    only tools that DO the deliverable workload (training/cleansing/FE/imputation/synthesis).
    """
    return {n for n in agent_tool_names(agent) if n in HEAVY_COMPUTE_TOOL_NAMES}
