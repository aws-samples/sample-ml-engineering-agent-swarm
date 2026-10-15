"""NL → schema-valid task_spec.md YAML — strands-free, client-side.

The Strands `generate_spec_from_description` tool already does heuristic NL
parsing, but importing it pulls in `strands`. The cloud-dispatch CLI must run on
a plain operator machine, so this module re-uses ONLY the pure heuristic helper
functions from `tools.spec_generation_tools` (no `@tool` import) and assembles a
spec whose front-matter validates against `SpecModel` (extra="forbid").

`build_spec_yaml(...)` returns a `task_spec.md` string (YAML front-matter + a
short body) ready to pass to `load_spec_from_text` / the app.py payload.
"""

from __future__ import annotations

import re

import yaml

# Pure heuristic helpers — `tools.spec_heuristics` is STRANDS-FREE (the @tool
# wrapper lives in `spec_generation_tools`, which we deliberately do NOT import
# so this client runs without the `agents` extra).
from ..tools.spec_heuristics import (
    _extract_metric,
    _extract_threshold,
    _infer_task_type,
)

PROJECT_ID_RE = re.compile(r"[^a-zA-Z0-9_]")


def sanitize_project_id(raw: str) -> str:
    """Coerce an arbitrary string into the SpecModel project_id rule.

    SpecModel requires ``^[a-zA-Z0-9_]+$`` (AgentCore naming constraint), ≤48 chars.
    """
    cleaned = PROJECT_ID_RE.sub("_", raw.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        cleaned = "amlas_run"
    return cleaned[:48]


def _delta_rule(threshold: float, task_type: str) -> str:
    """Turn a parsed threshold into an acceptance delta-vs-baseline comparison.

    We frame acceptance as a delta vs baseline (the Arbiter's contract). CRITICAL
    sign convention (must match arbiter.schema.evaluate_acceptance): the Arbiter
    computes ``effective_delta`` as an IMPROVEMENT that is POSITIVE when the model
    beats the baseline in BOTH directions —
        maximize: overall - baseline   (higher score = positive delta)
        minimize: baseline - overall   (lower error  = positive delta)
    so the acceptance rule must be ``>=`` for both. The prior regression default
    ``"<= -0.0"`` was a SIGN BUG: with effective_delta = baseline - overall = +27.95
    for an RMSE of 2.05 vs baseline 30, ``27.95 <= -0.0`` is False, so EVERY
    regression model that beat the baseline was wrongly REJECTED (surfaced by the
    live GHCN weather E2E — no regression run could ever pass acceptance). For
    regression "any improvement over baseline error passes" is ``>= 0.0``.
    """
    if task_type == "regression":
        return ">= 0.0"  # any improvement over baseline error passes (effective_delta>=0)
    return ">= 0.02"


# Comparison-string validator — mirrors SpecModel.AcceptanceCriteria so a bad
# override fails fast client-side (before the InvokeAgentRuntime round-trip).
_DELTA_RE = re.compile(r"^\s*(>=|<=|>|<|==)\s*-?\d+(\.\d+)?\s*$")


def build_spec_yaml(
    description: str,
    *,
    project_id: str,
    golden_dataset: str,
    target: str,
    primary_metric: str | None = None,
    baseline_metric_value: float = 0.55,
    max_retreats: int = 0,
    trials: int = 1,
    wall_clock_minutes: int = 5,
    gpu_hours: float = 0,
    acceptance_delta: str | None = None,
    no_subgroup_regression: bool = False,
    splits_strategy: str = "stratified",
) -> str:
    """Build a schema-valid ``task_spec.md`` string from a natural-language prompt.

    Args:
        description: free-text problem statement (drives task_type + metric inference).
        project_id: human label; sanitized to the SpecModel rule.
        golden_dataset: the dataset URI/path the runtime will stage (s3:// after upload).
        target: the resolved target column name.
        primary_metric: override the inferred metric (else inferred from text).
        baseline_metric_value: baseline score recorded under the metric name.
        max_retreats/trials/wall_clock_minutes/gpu_hours: compute-budget caps —
            defaults are tuned for a CHEAP verification slice.
        acceptance_delta: explicit acceptance delta-vs-baseline comparison string
            (e.g. ">= 0.15"). When provided it OVERRIDES the inferred default from
            ``_delta_rule`` — this is the Phase-18 lever for a STRICTER acceptance
            bar that forces the retreat/iterate loop. Must match the SpecModel
            comparison grammar or a ValueError is raised.
        no_subgroup_regression: when True the Arbiter additionally rejects any
            per-class/per-subgroup regression vs baseline slices.
    """
    task_type = _infer_task_type(description)
    metric = primary_metric or _extract_metric(description)
    threshold = _extract_threshold(description)
    pid = sanitize_project_id(project_id)

    if acceptance_delta is not None:
        delta_rule = acceptance_delta.strip()
        if not _DELTA_RE.match(delta_rule):
            raise ValueError(
                f"acceptance_delta must be a comparison like '>= 0.15', got: {acceptance_delta!r}"
            )
    else:
        delta_rule = _delta_rule(threshold, task_type)

    # FIX-26 wire-compat: locally "0 == unbounded", but the DEPLOYED AgentCore runtime may still
    # run the OLD schema (compute_budget.trials/wall_clock_minutes ge=1) and would REJECT a 0 with
    # a SpecLoadError before training even starts. So on the WIRE we express "unbounded" as a very
    # large finite value that is accepted by both the old (ge=1) and new (ge=0) schema and is, for
    # all practical purposes, uncapped — no redeploy needed. Positive values pass through verbatim.
    unbounded_trials = 100_000        # effectively unlimited HPO trials
    unbounded_wall_minutes = 525_600  # 1 year — effectively no wall-clock cap
    wire_trials = trials if trials and trials > 0 else unbounded_trials
    wire_wall = (
        wall_clock_minutes if wall_clock_minutes and wall_clock_minutes > 0 else unbounded_wall_minutes
    )

    front_matter: dict[str, object] = {
        "project_id": pid,
        "task_type": task_type,
        "target": target,
        "primary_metric": metric,
        "golden_dataset": golden_dataset,
        "constraints": {"no_pii_features": True},
        "splits": {"strategy": splits_strategy},
        "baseline": {"type": "gradient_boosting_default", metric: baseline_metric_value},
        "compute_budget": {
            "gpu_hours": gpu_hours,
            "trials": wire_trials,
            "wall_clock_minutes": wire_wall,
            "max_retreats": max_retreats,
        },
        "acceptance": {
            "primary_metric_delta_vs_baseline": delta_rule,
            "no_subgroup_regression": bool(no_subgroup_regression),
        },
    }

    yaml_block = yaml.safe_dump(front_matter, sort_keys=False, default_flow_style=False)
    body = (
        f"# {pid}\n\n"
        f"{description.strip()}\n\n"
        f"_Generated by the Phase 14b cloud-dispatch client (NL → spec)._\n"
    )
    return f"---\n{yaml_block}---\n{body}"
