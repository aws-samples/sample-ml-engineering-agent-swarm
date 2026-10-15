"""M2 guardrail hooks — leakage / cost-ceiling / DQ-quota (doc 05 §8.9; ADR-015).

Extends the M0/M1 ``GuardrailHooks`` mechanism (``backends/hooks.py``) with the three
M2 declarative ``PreToolUse`` guardrails the roadmap names, each fail-closed and
non-overridable by the model:

* **leakage_check guard** — blocks a split / feature-use tool when a leaking
  train/test split is detected (wraps the EXISTING ``leakage_check`` IP — row-hash dup,
  target-bleed, time-order, split-key dup). Fires BEFORE the split is consumed so no
  model ever trains on leaked data.
* **cost-ceiling guard** — gates a managed DataBrew/Glue/Processing job whose estimated
  cost exceeds a ceiling. Ties into the M1 approval mechanism: the deny is the same
  fail-closed ``Verdict`` the ``approval_gate_hook`` emits, so an over-budget managed job
  cannot launch un-approved (doc 05 §8.8 EXPENSIVE_JOB class).
* **DQ-quota / threshold guard** — blocks promotion of a dataset whose data-quality
  score is below a minimum threshold (the "don't write junk to silver/gold" rule),
  reusing the ``DataQualityBackend`` 6-dim score.

All are pure-Python, deterministic, and centrally auditable (ADR-015). They register
onto a :class:`~amlas.backends.hooks.GuardrailHooks` instance via
:func:`m2_guardrail_hooks`, which also seeds the M0 bronze-write deny + the M1 approval
gate so the full safe-by-default set is available to the control plane.
"""

from __future__ import annotations

from typing import Any

from ..backends.hooks import (
    GuardrailHooks,
    Hook,
    HookContext,
    HookEvent,
    bronze_write_deny_hook,
)
from ..backends.types import Verdict

__all__ = [
    "DEFAULT_COST_CEILING",
    "DEFAULT_DQ_MIN_SCORE",
    "cost_ceiling_hook",
    "dq_quota_hook",
    "leakage_check_hook",
    "m2_guardrail_hooks",
]

# Defaults (overridable per-hook). A managed job over $5 gates; a dataset scoring below
# 60/100 on the 6-dim scale cannot be promoted to silver/gold.
DEFAULT_COST_CEILING = 5.0
DEFAULT_DQ_MIN_SCORE = 60.0

# Tools whose invocation consumes a train/test split (leakage must be clean first).
_SPLIT_CONSUMERS = {
    "submit_training_run", "submit_training_job", "train_model", "run_hpo", "tune_model",
    "baseline_train_flow", "build_time_window_features", "feature_pit_dataset",
}
# Managed-job tools that incur per-job spend (the cost ceiling applies).
_MANAGED_JOBS = {
    "run_processing_job", "run_glue_job", "run_crawler", "apply_recipe",
    "run_profile", "run_batch_transform", "run_spark_job",
}
# Tools that promote a dataset into the curated tiers (the DQ quota applies).
_PROMOTION_TOOLS = {
    "write_lake_dataset", "write_lake_dataset_via_port", "register_dataset_version",
    "apply_imputation_plan", "apply_transform_flow",
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. leakage_check guard (PreToolUse, before splits/feature use)
# ══════════════════════════════════════════════════════════════════════════════
def leakage_check_hook() -> Hook:
    """Block a split/feature-use call when the train/test split leaks (wraps the IP).

    The hook reads ``ctx.args`` for in-memory ``train``/``test`` frames (+ ``target``,
    optional ``split_key``/``time_col``/``cutoff``) OR a precomputed ``leakage_report``
    dict (``{"blocking": bool, "summary": str}``). It runs the EXISTING
    ``tools.leakage_check.leakage_check`` and DENIES (fail-closed) when the report is
    blocking — so no model trains on leaked data. When no split is available to check,
    the hook is a no-op allow (it only fires when it can actually evaluate leakage).
    """

    def _matches(ctx: HookContext) -> bool:
        name = (ctx.tool or ctx.action or "").strip().lower()
        if name in _SPLIT_CONSUMERS or "split" in name:
            return True
        # Also match if a leakage payload was explicitly attached to the call.
        return "leakage_report" in ctx.args or ("train" in ctx.args and "test" in ctx.args)

    def _action(ctx: HookContext) -> Verdict:
        # 1) explicit precomputed report wins (cheap, no recompute).
        report = ctx.args.get("leakage_report")
        if isinstance(report, dict):
            if report.get("blocking"):
                return Verdict(
                    allowed=False,
                    reason=f"leakage detected: {report.get('summary', 'blocking findings')}",
                    detail={"rule": "leakage_check", "report": report},
                )
            return Verdict(allowed=True, reason="leakage report clean")

        # 2) in-memory frames → run the EXISTING leakage IP.
        train = ctx.args.get("train")
        test = ctx.args.get("test")
        target = ctx.args.get("target")
        if train is None or test is None or not target:
            # Nothing to evaluate — do not block (the hook only denies on real evidence).
            return Verdict(allowed=True, reason="no split available to leakage-check")
        try:
            from ..tools.leakage_check import leakage_check as _leakage_check

            rpt = _leakage_check(
                train=train,
                test=test,
                target=str(target),
                split_key=ctx.args.get("split_key"),
                time_col=ctx.args.get("time_col"),
                cutoff=ctx.args.get("cutoff"),
            )
        except Exception as e:
            return Verdict(
                allowed=False,
                reason=f"leakage check could not complete ({type(e).__name__}) — blocking fail-closed",
                detail={"rule": "leakage_check", "error": str(e)},
            )
        if rpt.blocking:
            return Verdict(
                allowed=False,
                reason=f"leakage detected: {rpt.summary()}",
                detail={"rule": "leakage_check", "findings": [f.kind for f in rpt.findings]},
            )
        return Verdict(allowed=True, reason="no leakage detected")

    return Hook(name="leakage_check", event=HookEvent.PRE_TOOL_USE, matcher=_matches, action=_action)


# ══════════════════════════════════════════════════════════════════════════════
# 2. cost-ceiling guard (PreToolUse, before managed jobs — ties to M1 approval)
# ══════════════════════════════════════════════════════════════════════════════
def cost_ceiling_hook(ceiling: float = DEFAULT_COST_CEILING) -> Hook:
    """Block a managed DataBrew/Glue/Processing job whose estimate exceeds ``ceiling``.

    Reads ``ctx.args['cost_estimate']`` (the control plane / Plan Mode supplies it) and
    DENIES fail-closed when it is over the ceiling — the same disposition the M1
    ``EXPENSIVE_JOB`` approval class enforces, expressed here as a deterministic hook so
    an over-budget managed job cannot launch ungated (doc 05 §8.8/§8.9). A per-call
    ``ctx.args['cost_ceiling']`` overrides the default ceiling.
    """

    def _matches(ctx: HookContext) -> bool:
        name = (ctx.tool or ctx.action or "").strip().lower()
        if name in _MANAGED_JOBS:
            return True
        return bool(ctx.args.get("managed")) or any(
            h in name for h in ("processing", "glue", "databrew", "batch_transform")
        )

    def _action(ctx: HookContext) -> Verdict:
        cap = float(ctx.args.get("cost_ceiling", ceiling))
        cost = float(ctx.args.get("cost_estimate", 0.0) or 0.0)
        if cost > cap:
            return Verdict(
                allowed=False,
                reason=f"estimated cost ${cost:.2f} exceeds ceiling ${cap:.2f} — approval required",
                detail={"rule": "cost_ceiling", "cost_estimate": cost, "ceiling": cap},
            )
        return Verdict(
            allowed=True,
            reason=f"estimated cost ${cost:.2f} within ceiling ${cap:.2f}",
            detail={"rule": "cost_ceiling", "cost_estimate": cost, "ceiling": cap},
        )

    return Hook(name="cost_ceiling", event=HookEvent.PRE_TOOL_USE, matcher=_matches, action=_action)


# ══════════════════════════════════════════════════════════════════════════════
# 3. DQ-quota / threshold guard (PreToolUse, before promotion to silver/gold)
# ══════════════════════════════════════════════════════════════════════════════
def dq_quota_hook(min_score: float = DEFAULT_DQ_MIN_SCORE) -> Hook:
    """Block promotion of a sub-threshold-quality dataset to the curated tiers.

    The "don't write junk to silver/gold" rule: when a promotion tool runs, the hook
    reads ``ctx.args['dq_score']`` (0-100, supplied by the control plane from a prior
    ``DataQualityBackend.profile``) and DENIES fail-closed if it is below ``min_score``.
    A per-call ``ctx.args['dq_min_score']`` overrides the default. Promotions to
    ``bronze``/``raw`` are NOT this hook's concern (the bronze-write deny handles those);
    a missing score does NOT block (the hook only denies on real evidence of low quality).
    """

    def _matches(ctx: HookContext) -> bool:
        name = (ctx.tool or ctx.action or "").strip().lower()
        if name in _PROMOTION_TOOLS or "write_lake" in name or "register_dataset" in name:
            # only gate curated-tier promotions
            tier = str(ctx.args.get("tier", "")).strip().lower()
            return tier in ("", "silver", "gold")  # default promotion target is silver
        return False

    def _action(ctx: HookContext) -> Verdict:
        if "dq_score" not in ctx.args:
            return Verdict(allowed=True, reason="no DQ score supplied — not gating on quota")
        threshold = float(ctx.args.get("dq_min_score", min_score))
        score = float(ctx.args.get("dq_score", 0.0) or 0.0)
        if score < threshold:
            return Verdict(
                allowed=False,
                reason=f"data-quality score {score:.1f} below quota {threshold:.1f} — promotion blocked",
                detail={"rule": "dq_quota", "dq_score": score, "min_score": threshold},
            )
        return Verdict(
            allowed=True,
            reason=f"data-quality score {score:.1f} meets quota {threshold:.1f}",
            detail={"rule": "dq_quota", "dq_score": score, "min_score": threshold},
        )

    return Hook(name="dq_quota", event=HookEvent.PRE_TOOL_USE, matcher=_matches, action=_action)


# ══════════════════════════════════════════════════════════════════════════════
# assembled hook set
# ══════════════════════════════════════════════════════════════════════════════
def m2_guardrail_hooks(
    *,
    cost_ceiling: float = DEFAULT_COST_CEILING,
    dq_min_score: float = DEFAULT_DQ_MIN_SCORE,
    include_approval: bool = False,
    approval_policy: Any = None,
) -> GuardrailHooks:
    """Build the M2 guardrail set: bronze-write deny + leakage + cost + DQ-quota.

    Composes the M0 bronze-write deny (defense in depth) with the three M2 guardrails.
    When ``include_approval`` is True, the M1 ``approval_gate_hook`` (T-CTL-03) is also
    registered so the full safe-by-default beat is active. The ordering matters only for
    *which* deny is reported first (fail-closed returns the FIRST deny); all denies are
    equally binding (ADR-015).

    Args:
        cost_ceiling: Dollar ceiling for the managed-job cost guard.
        dq_min_score: Minimum 6-dim score (0-100) for the DQ-quota promotion guard.
        include_approval: Also register the M1 approval gate (default off — many flows
            wrap approval explicitly via ``plan_approve_execute`` instead).
        approval_policy: The :class:`ApprovalPolicy` for the approval gate, when included.
    """
    hooks = GuardrailHooks(
        [
            bronze_write_deny_hook(),
            leakage_check_hook(),
            cost_ceiling_hook(cost_ceiling),
            dq_quota_hook(dq_min_score),
        ]
    )
    if include_approval:
        from .approvals import approval_gate_hook

        hooks.register(approval_gate_hook(approval_policy))
    return hooks
