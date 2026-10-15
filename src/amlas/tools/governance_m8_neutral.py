"""Governance/traceability neutral tools — M8 cost-report + trace-fetch surface (doc 05 §8.5/§8.8).

Backend-neutral ``@tool``-style functions (plain callables, port-resolved) the control plane /
surfaces call for the M8 hardening capabilities:

* :func:`estimate_flow_cost` (T-PR-53) — pure-reasoning cost+latency estimate for a flow/action
  via the resolved ``CostAdvisorBackend``.
* :func:`compile_cost_report` (the cost dashboard tool, §8.8) — aggregate per-run/all-run spend.
* :func:`enforce_budget_ceiling` — the budget-ceiling decision (ties to the cost-ceiling hook).
* :func:`fetch_run_trace` (§8.5) — fetch the story→flow→tool→backend→adapter→run lineage for a run.
* :func:`query_traceability_matrix` (§8.5) — the executable traceability-matrix join.

All offline + deterministic: the CostAdvisor local default + the in-process trace ledger do the
work; nothing here touches AWS (the managed CostAdvisor variant is selectable via env).
"""

from __future__ import annotations

from typing import Any

from ..backends.registry import REGISTRY
from ..control_plane.cost_governance import CostAdvisorBackend
from ..control_plane.traceability import LEDGER

__all__ = [
    "compile_cost_report",
    "enforce_budget_ceiling",
    "estimate_flow_cost",
    "fetch_run_trace",
    "query_traceability_matrix",
]


def _advisor() -> Any:
    """Resolve the CostAdvisor facet (local default offline; CloudWatch managed via env)."""
    return REGISTRY.resolve(CostAdvisorBackend).adapter


def estimate_flow_cost(*, flow_id: str = "", action: str = "", steering: Any = None) -> dict[str, Any]:
    """``T-PR-53`` — estimate the cost + latency of a flow / action (the plan-mode spend input).

    Resolves the ``CostAdvisorBackend`` and prices the flow/action from its price book. Returns the
    backend-neutral estimate dict the plan-mode preview + approval gate consult.

    The raw ``usd`` point estimate is retained verbatim (the gate/ceiling consumer). An ADDITIVE
    ``cost_band`` carries the human-facing caveated range + confidence/source label
    (:func:`~...control_plane.cost_confidence.band_for`) so a surface renders the uncertainty
    instead of a bare confident figure — the gate math is unchanged.
    """
    from ..control_plane.cost_confidence import band_for

    est = _advisor().estimate(flow_id=flow_id, action=action)
    result: dict[str, Any] = {
        "flow_id": flow_id, "action": action, "usd": est.usd, "latency_ms": est.latency_ms,
        "source": est.source, "detail": est.detail,
        "cost_band": band_for(est).to_dict(),
    }
    return result


def compile_cost_report(*, run_id: str = "", window: str = "7d") -> dict[str, Any]:
    """The cost dashboard tool (§8.8) — aggregate spend for a run (or all runs) over ``window``."""
    report: dict[str, Any] = _advisor().report(run_id=run_id, window=window)
    return report


def enforce_budget_ceiling(*, run_id: str, estimate: float, ceiling: float | None,
                           hard: bool = False) -> dict[str, Any]:
    """Budget-ceiling decision (§8.8 / §11 Q2) — would this spend keep ``run_id`` under ``ceiling``?

    Returns the :class:`~...control_plane.cost_governance.BudgetDecision` as a dict (``allowed`` /
    ``projected`` / ``reason`` / ``hard``). A hard pin hard-stops on breach; a soft ceiling
    warn+confirms. This is the programmatic twin of the ``cost_ceiling_hook`` PreToolUse guardrail.
    """
    decision = _advisor().check_budget(run_id=run_id, estimate=float(estimate), ceiling=ceiling, hard=hard)
    return {
        "allowed": decision.allowed, "ceiling": decision.ceiling, "projected": decision.projected,
        "spent": decision.spent, "estimate": decision.estimate, "reason": decision.reason,
        "hard": decision.hard,
    }


def fetch_run_trace(*, run_id: str = "", trace_id: str = "") -> dict[str, Any]:
    """``§8.5`` — fetch the story→flow→tool→backend→adapter→run lineage for a run/trace.

    Queries the in-process trace ledger; returns the :class:`~...control_plane.traceability.
    TraceRecord` as a dict (plus its human-readable ``chain``), or ``{"found": False}`` when no
    matching trace exists. This makes the doc-00 traceability matrix queryable at runtime.
    """
    rec = LEDGER.get(run_id=run_id or None, trace_id=trace_id or None)
    if rec is None:
        return {"found": False, "run_id": run_id, "trace_id": trace_id}
    out = rec.to_dict()
    out["found"] = True
    out["chain"] = rec.chain
    return out


def query_traceability_matrix(*, story_id: str = "", flow_id: str = "") -> dict[str, Any]:
    """``§8.5`` — the executable traceability-matrix join (filterable by story / flow).

    Returns ``{rows, count}`` where each row is a story→flow→tool→backend→run join (the Phase-8
    matrix), filtered by the optional ``story_id`` / ``flow_id``. With no filter, returns the full
    matrix of every recorded invocation.
    """
    if story_id or flow_id:
        recs = LEDGER.query(story_id=story_id or None, flow_id=flow_id or None)
        rows = [
            {"story_id": r.story_id, "flow_id": r.flow_id, "tool_ids": list(r.tool_ids),
             "ports": [h.port for h in r.backends], "adapters": [h.adapter for h in r.backends],
             "run_id": r.run_id, "status": r.status, "chain": r.chain}
            for r in recs
        ]
    else:
        rows = LEDGER.matrix()
    return {"rows": rows, "count": len(rows)}
