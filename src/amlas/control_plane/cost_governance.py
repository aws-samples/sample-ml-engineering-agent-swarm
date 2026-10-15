"""Cost/latency governance — the CostAdvisor surface (doc 05 §8.3/§8.8 / roadmap M8; RECON §10.12).

Closes the recon cost-governance GAP. The surface aggregates per-flow / per-run **cost + latency**
estimates and enforces **budget ceilings**, tying into the M2 hooks layer (a ``PreToolUse``
cost-ceiling guardrail) and the M1 approval gate (the plan-mode spend estimate). It is the
``CostAdvisorBackend`` facet (doc 05 §8.3): a local in-process default + a managed
CloudWatch/Cost-Explorer-backed variant (thin, offline-verified).

Three pieces (all pure-Python; the managed variant is the only one that touches boto3, injectable):

* :class:`LocalCostAdvisor` — the offline default. A small, deterministic price book
  (estimate per flow/action + a SizingRec-aware endpoint estimate) + a per-run/per-flow spend
  ledger + the ``check_budget`` ceiling decision. This is what the plan-mode spend estimate and
  the approval cost-ceiling consult — no AWS.
* :class:`CloudWatchCostAdvisor` — the managed variant: thin boto3 ``cloudwatch`` /
  ``ce`` (Cost Explorer) orchestration to read ACTUAL spend (``get_cost_and_usage`` /
  ``get_metric_statistics``) instead of the price-book estimate. Client injectable for offline substitutes.
* :func:`cost_ceiling_hook` — adapts a budget ceiling into a deterministic ``PreToolUse``
  guardrail (the M2 cost guardrail seam, ADR-015): a flow/action whose estimated spend would push
  the run over its ceiling is DENIED fail-closed (the model cannot override) — the backstop the
  approval gate + the least-priv IAM (M8) layer with.

The estimate fields are intentionally the SAME shape the M5 ``track_inference_spend`` + the
plan-mode ``estimated_cost`` already use, so cost is one consolidated surface (the §8.8 ask).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

from ..backends.hooks import Hook, HookContext, HookEvent
from ..backends.registry import register_adapter
from ..backends.types import Verdict
from ..config.aws_region import resolve_aws_region

__all__ = [
    "ENV_ABSOLUTE_BUDGET",
    "ENV_MAX_AUTO_COST",
    "BudgetCaps",
    "BudgetDecision",
    "CapDecision",
    "CloudWatchCostAdvisor",
    "CostAdvisorBackend",
    "CostEstimate",
    "LocalCostAdvisor",
    "cost_ceiling_hook",
]

# ── env seam for the APPROVED capped live run (gap-closure PREP) ──────────────
# The human approved a capped live run with two CONFIRMED caps, both env-sourced with
# SAFE BLOCKING defaults (absent config → 0.0 → never a silent unbounded auto-spend).
ENV_MAX_AUTO_COST = "AMLAS_MAX_AUTO_COST_USD"   # the $10 per-run max-auto-cost ceiling
ENV_ABSOLUTE_BUDGET = "AMLAS_ABSOLUTE_BUDGET_USD"  # the $50 absolute cumulative budget cap


def _safe_env_float(name: str, default: float = 0.0) -> float:
    """Read a non-negative USD env value; non-numeric / negative / unset → ``default`` (safe).

    Safe-by-default: a garbage or negative value can NEVER widen the cap — it collapses to the
    blocking ``default`` (0.0), so an absent/misconfigured env never silently permits spend.
    """
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return float(default)
    try:
        val = float(str(raw).strip())
    except (TypeError, ValueError):
        return float(default)
    return val if val >= 0.0 else float(default)


def _ce_cache_key(kwargs: dict[str, Any]) -> Any:
    """Build a hashable cache key from a ``get_cost_and_usage`` kwargs dict (PERF-06).

    The CE request is fully determined by its kwargs (TimePeriod / Granularity / Metrics
    / Filter / GroupBy). We serialize them deterministically (sorted, repr-stable) so two
    requests with the SAME window+filter+grouping share a cache entry and a different
    window/filter is a distinct key (never a false cache hit).
    """
    import json

    try:
        return json.dumps(kwargs, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return repr(sorted(kwargs.items(), key=lambda kv: kv[0]))


# ──────────────────────────────────────────────────────────────────────────────
# The CostAdvisorBackend port marker (a control-plane facet, doc 05 §8.3)
# ──────────────────────────────────────────────────────────────────────────────
class CostAdvisorBackend:
    """Marker base for the CostAdvisor facet so the registry can resolve it like a port.

    Not one of the 19 formal ML ports — it is the cost-governance *facet* (doc 05 §8.3): a
    control-plane backend that estimates/aggregates spend + enforces budgets. Concrete adapters
    (:class:`LocalCostAdvisor` / :class:`CloudWatchCostAdvisor`) subclass it; the registry resolves
    ``CostAdvisorBackend`` to one of them via ``COST_ADVISOR_BACKEND`` (local default, offline).
    """


# ──────────────────────────────────────────────────────────────────────────────
# Value types — a cost+latency estimate + a budget-ceiling decision
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class CostEstimate:
    """A per-flow / per-action cost + latency estimate (the consolidated §8.8 shape).

    ``usd`` is the estimated spend; ``latency_ms`` the estimated wall latency; ``source`` records
    whether it came from the local price-book estimate or a managed (CloudWatch/CE) actual. Mirrors
    the plan-mode ``estimated_cost`` + the M5 ``track_inference_spend`` fields so cost is one surface.
    """

    flow_id: str = ""
    action: str = ""
    usd: float = 0.0
    latency_ms: float = 0.0
    source: str = "estimate"
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class BudgetDecision:
    """The ceiling-enforcement decision for a flow/run (the §8.8 budget gate).

    ``allowed`` is the go/no-go; ``ceiling`` the budget cap; ``projected`` the run's spend AFTER the
    proposed action; ``reason`` the human rationale. ``hard`` marks a hard-steering budget pin
    (hard-stop) vs a soft ceiling (warn+confirm) — doc 05 §11 Q2.
    """

    allowed: bool
    ceiling: float
    projected: float
    spent: float = 0.0
    estimate: float = 0.0
    reason: str = ""
    hard: bool = False


@dataclass
class BudgetCaps:
    """The APPROVED capped-live-run budget caps (gap-closure PREP; doc 05 §8.8 / master-plan §7 R-1).

    Two env-sourced caps with SAFE BLOCKING defaults (absent → 0.0, never an unbounded permit):

    * ``max_auto_cost`` — the per-run ceiling: a single planned action costing MORE than this is
      NOT auto-approved (it gates for explicit approval). Confirmed value: ``$10``.
    * ``absolute_budget`` — the cumulative hard cap: spend that would push the WHOLE-ledger total
      over this is HARD-BLOCKED (fail-closed). Confirmed value: ``$50``.
    """

    max_auto_cost: float = 0.0
    absolute_budget: float = 0.0

    @classmethod
    def from_env(cls) -> BudgetCaps:
        """Build the caps from ``AMLAS_MAX_AUTO_COST_USD`` / ``AMLAS_ABSOLUTE_BUDGET_USD``.

        Safe-by-default: an unset / non-numeric / negative env value collapses to the blocking
        ``0.0`` (never a silent unbounded auto-spend) — see :func:`_safe_env_float`.
        """
        return cls(
            max_auto_cost=_safe_env_float(ENV_MAX_AUTO_COST, 0.0),
            absolute_budget=_safe_env_float(ENV_ABSOLUTE_BUDGET, 0.0),
        )


@dataclass
class CapDecision:
    """The combined per-run-ceiling + absolute-budget decision (the capped-live-run gate).

    ``auto_approvable`` is True ONLY when the action is BOTH within the per-run ``max_auto_cost``
    ceiling AND keeps the cumulative spend within the ``absolute_budget`` — i.e. it can stream
    without an explicit human approval. ``allowed`` is the go/no-go (False only on a HARD absolute
    breach); ``hard`` marks the fail-closed absolute-budget hard-stop; ``within_auto_cost`` records
    the per-run-ceiling check; ``projected_total`` is the cumulative spend AFTER the action.
    """

    allowed: bool
    auto_approvable: bool
    within_auto_cost: bool
    hard: bool
    estimate: float
    spent_total: float
    projected_total: float
    max_auto_cost: float
    absolute_budget: float
    reason: str = ""


# ──────────────────────────────────────────────────────────────────────────────
# LocalCostAdvisor — the offline default (price book + spend ledger + ceiling)
# ──────────────────────────────────────────────────────────────────────────────
# A small deterministic price book: estimated USD per flow id / action (the plan-mode
# ``estimated_cost`` values + per-action spend). Keys are flow ids / tool action names; the
# fallback is a conservative non-zero estimate for anything spend-incurring (safe-by-default).
_PRICE_BOOK: dict[str, float] = {
    # flows (mirror plan_mode._FLOW_TEMPLATES estimated_cost)
    "exploration_loop": 0.40,
    "ml_pipeline": 0.10,
    "baseline_train_flow": 0.40,
    "hpo_flow": 0.60,
    "deploy_flow": 0.12,
    "improve_retrain_flow": 0.50,
    "data_collection_flow": 0.08,
    "monitor_flow": 0.0,
    "data_quality_flow": 0.0,
    "imputation_flow": 0.15,
    "feature_engineering_flow": 0.05,
    "evaluation_flow": 0.0,
    # actions
    "submit_training_run": 0.40,
    "run_hpo": 0.60,
    "run_automl": 0.55,
    "run_processing_job": 0.20,
    "run_glue_job": 0.25,
    "apply_recipe": 0.18,
    "deploy_endpoint": 0.12,
    "run_batch_transform": 0.15,
    "create_labeling_job": 0.08,
}
# Per-action latency estimates (ms) — coarse, for the latency facet of the surface.
_LATENCY_BOOK: dict[str, float] = {
    "deploy_endpoint": 45.0,
    "submit_training_run": 600000.0,  # ~10 min managed training
    "run_hpo": 1800000.0,
    "run_batch_transform": 120000.0,
    "monitor_flow": 50.0,
    "evaluation_flow": 20.0,
}
_DEFAULT_USD = 0.05  # conservative non-zero fallback for an unknown spend-incurring action


class LocalCostAdvisor(CostAdvisorBackend):
    """Offline CostAdvisor — deterministic price-book estimates + a per-run spend ledger + ceiling.

    ``estimate`` prices a flow/action from the price book (or a SizingRec endpoint estimate);
    ``record_spend`` accumulates per-run/per-flow actuals; ``aggregate`` rolls a run up; ``report``
    composes the cost panel; ``check_budget`` is the ceiling decision the approval gate + the
    cost-ceiling hook consult. No AWS — this is the offline default and the offline surface.
    """

    backend_name = "local"

    def __init__(self) -> None:
        # per-run spend ledger: {run_id: [(flow/action, usd), ...]}
        self._ledger: dict[str, list[tuple[str, float]]] = {}

    # ── estimate (the plan-mode + sizing input) ─────────────────────────────────
    def estimate(self, *, flow_id: str = "", action: str = "",
                 sizing: Any = None, hours: float = 0.0) -> CostEstimate:
        """Estimate the cost + latency of a flow / action / endpoint-sizing (price-book).

        Resolution: an explicit ``sizing`` (a :class:`~...backends.types.SizingRec`) prices an
        endpoint at ``est_cost_per_hour * hours``; otherwise the ``flow_id``/``action`` price book
        applies (falling back to a conservative non-zero estimate for an unknown spender). Latency
        comes from the latency book (or the sizing's ``est_latency_ms``).
        """
        key = (action or flow_id or "").strip().lower()
        if sizing is not None and hasattr(sizing, "est_cost_per_hour"):
            usd = float(sizing.est_cost_per_hour) * float(hours or 1.0)
            return CostEstimate(flow_id=flow_id, action=action, usd=round(usd, 4),
                                latency_ms=float(getattr(sizing, "est_latency_ms", 0.0)),
                                source="sizing",
                                detail={"instance_type": getattr(sizing, "instance_type", ""), "hours": hours})
        usd = _PRICE_BOOK.get(key, _DEFAULT_USD if key else 0.0)
        latency = _LATENCY_BOOK.get(key, 0.0)
        return CostEstimate(flow_id=flow_id, action=action, usd=round(usd, 4),
                            latency_ms=latency, source="estimate")

    # ── spend ledger ────────────────────────────────────────────────────────────
    def record_spend(self, *, run_id: str, flow_id: str = "", action: str = "", usd: float = 0.0) -> float:
        """Record an incurred spend against ``run_id`` and return the run's new total."""
        self._ledger.setdefault(run_id, []).append(((action or flow_id or "spend"), float(usd)))
        return self.spent(run_id=run_id)

    def spent(self, *, run_id: str) -> float:
        """Return the total spend recorded against ``run_id`` so far."""
        return round(sum(usd for _, usd in self._ledger.get(run_id, [])), 4)

    def total_spent(self) -> float:
        """Return the CUMULATIVE spend across the WHOLE ledger (the absolute-budget denominator)."""
        return round(sum(usd for items in self._ledger.values() for _, usd in items), 4)

    # ── the combined capped-live-run gate (per-run ceiling + absolute budget) ────
    def check_caps(self, *, run_id: str, estimate: float, caps: BudgetCaps) -> CapDecision:
        """Decide a planned ``estimate`` against BOTH the per-run ceiling AND the absolute budget.

        The APPROVED capped-live-run gate (gap-closure PREP; doc 05 §8.8 / master-plan §7 R-1):

        * **per-run ceiling** (``caps.max_auto_cost``) — an action costing MORE than this is NOT
          auto-approvable (it must be explicitly approved); ``<=`` is within-ceiling. A ``0.0``
          ceiling (the safe blocking default) means NOTHING auto-approves.
        * **absolute budget** (``caps.absolute_budget``) — the cumulative whole-ledger spend AFTER
          this action (``total_spent + estimate``) must stay ``<=`` the budget; a breach HARD-blocks
          (``allowed=False``, ``hard=True``). A ``0.0`` budget hard-blocks any positive spend.

        ``auto_approvable`` is True only when within the ceiling AND within the budget — i.e. the
        action can stream without an explicit human approval under the confirmed caps. ``run_id`` is
        accepted for parity/telemetry; the absolute budget is intentionally cumulative (all runs).
        """
        spent_total = self.total_spent()
        projected_total = round(spent_total + float(estimate), 4)
        within_auto_cost = float(estimate) <= float(caps.max_auto_cost)
        within_budget = projected_total <= float(caps.absolute_budget)

        if not within_budget:
            return CapDecision(
                allowed=False, auto_approvable=False, within_auto_cost=within_auto_cost,
                hard=True, estimate=float(estimate), spent_total=spent_total,
                projected_total=projected_total, max_auto_cost=float(caps.max_auto_cost),
                absolute_budget=float(caps.absolute_budget),
                reason=(f"absolute budget ${caps.absolute_budget} breached "
                        f"(projected cumulative ${projected_total}) — hard-stop"),
            )
        if not within_auto_cost:
            return CapDecision(
                allowed=True, auto_approvable=False, within_auto_cost=False, hard=False,
                estimate=float(estimate), spent_total=spent_total,
                projected_total=projected_total, max_auto_cost=float(caps.max_auto_cost),
                absolute_budget=float(caps.absolute_budget),
                reason=(f"per-run ceiling ${caps.max_auto_cost} exceeded "
                        f"(estimate ${estimate}) — explicit approval required (not auto-approved)"),
            )
        return CapDecision(
            allowed=True, auto_approvable=True, within_auto_cost=True, hard=False,
            estimate=float(estimate), spent_total=spent_total,
            projected_total=projected_total, max_auto_cost=float(caps.max_auto_cost),
            absolute_budget=float(caps.absolute_budget),
            reason=(f"auto-approvable: ${estimate} <= ${caps.max_auto_cost}/run and projected "
                    f"cumulative ${projected_total} <= ${caps.absolute_budget} absolute"),
        )

    def aggregate(self, *, run_id: str) -> CostEstimate:
        """Roll up a run's recorded spend into a single aggregated :class:`CostEstimate`."""
        items = self._ledger.get(run_id, [])
        total = round(sum(usd for _, usd in items), 4)
        return CostEstimate(flow_id=run_id, usd=total, source="aggregate",
                            detail={"line_items": [{"action": a, "usd": u} for a, u in items], "count": len(items)})

    # ── the ceiling decision (§8.8 / §11 Q2) ─────────────────────────────────────
    def check_budget(self, *, run_id: str, estimate: float, ceiling: float | None,
                     hard: bool = False) -> BudgetDecision:
        """Decide whether an action costing ``estimate`` keeps ``run_id`` under its ``ceiling``.

        Doc 05 §11 Q2: a **hard** budget pin hard-STOPS when the projected spend would breach the
        ceiling; a **soft** ceiling warns (``allowed=True`` but flagged) so the approval gate can
        warn+confirm. ``ceiling is None`` → no budget set → allowed. ``projected = spent + estimate``.
        """
        spent = self.spent(run_id=run_id)
        projected = round(spent + float(estimate), 4)
        if ceiling is None:
            return BudgetDecision(allowed=True, ceiling=float("inf"), projected=projected,
                                  spent=spent, estimate=float(estimate), reason="no budget ceiling set")
        over = projected > float(ceiling)
        if over and hard:
            return BudgetDecision(allowed=False, ceiling=float(ceiling), projected=projected,
                                  spent=spent, estimate=float(estimate), hard=True,
                                  reason=f"hard budget ceiling ${ceiling} breached (projected ${projected}) — hard-stop")
        if over:
            return BudgetDecision(allowed=True, ceiling=float(ceiling), projected=projected,
                                  spent=spent, estimate=float(estimate), hard=False,
                                  reason=f"soft budget ceiling ${ceiling} exceeded (projected ${projected}) — warn+confirm")
        return BudgetDecision(allowed=True, ceiling=float(ceiling), projected=projected,
                              spent=spent, estimate=float(estimate),
                              reason=f"under budget (${projected} ≤ ${ceiling})")

    # ── the cost report / dashboard tool ─────────────────────────────────────────
    def report(self, *, run_id: str = "", window: str = "7d") -> dict[str, Any]:
        """Compose a cost report panel — per-run aggregate + line items (the cost dashboard tool)."""
        if run_id:
            agg = self.aggregate(run_id=run_id)
            return {"run_id": run_id, "total_usd": agg.usd, "window": window,
                    "line_items": agg.detail.get("line_items", []), "source": "local_estimate"}
        # all-runs roll-up
        runs = {rid: self.spent(run_id=rid) for rid in self._ledger}
        return {"window": window, "runs": runs, "total_usd": round(sum(runs.values()), 4),
                "source": "local_estimate"}


# ──────────────────────────────────────────────────────────────────────────────
# CloudWatchCostAdvisor — the managed variant (Cost Explorer / CloudWatch actuals)
# ──────────────────────────────────────────────────────────────────────────────
class CloudWatchCostAdvisor(LocalCostAdvisor):
    """Managed CostAdvisor — reads ACTUAL spend from Cost Explorer / CloudWatch (thin, ADR-004).

    Inherits the price-book estimate + ceiling logic from :class:`LocalCostAdvisor` (so ``estimate``
    / ``check_budget`` are identical), and OVERRIDES ``report`` to pull the real cost from a boto3
    ``ce`` (Cost Explorer) ``get_cost_and_usage`` call instead of the local ledger — the production
    actual-spend panel. The client is injectable (offline substitutes); a failure falls back to the local
    estimate so the surface never crashes. Selectable via ``COST_ADVISOR_BACKEND=cloudwatch``.
    """

    backend_name = "cloudwatch"

    def __init__(
        self,
        *,
        region: str | None = None,
        ce_client: Any = None,
        cache_ttl: float | None = None,
        clock: Any = None,
    ) -> None:
        super().__init__()
        self._region = resolve_aws_region(region, required=False)
        self._ce = ce_client
        # PERF-06: a short TTL cache at the Cost-Explorer call boundary. Each
        # ``ce.get_cost_and_usage`` is $0.01 + a network round-trip; the dashboard /
        # spend panels re-issue identical windows repeatedly. Caching the RAW CE
        # response (keyed on the request params) means a repeat read within the TTL
        # issues 0 new CE calls while post-processing still runs per-call (so the
        # run_id/budget echoes stay fresh and the payload is byte-identical). A
        # FAILED read is never cached, so the graceful-degrade fallback is preserved.
        # The TTL is env-overridable (``AMLAS_CE_CACHE_TTL``); default 5 min.
        if cache_ttl is None:
            cache_ttl = _safe_env_float("AMLAS_CE_CACHE_TTL", 300.0)
        self._cache_ttl = float(cache_ttl)
        self._clock = clock or time.monotonic
        # cache: key (params signature) → (expiry_timestamp, raw_ce_response)
        self._ce_cache: dict[Any, tuple[float, dict[str, Any]]] = {}

    def _ce_client(self) -> Any:
        if self._ce is None:
            import boto3

            self._ce = boto3.client("ce", region_name=resolve_aws_region(self._region))
        return self._ce

    def _cached_get_cost_and_usage(self, **kwargs: Any) -> dict[str, Any]:
        """TTL-cached ``ce.get_cost_and_usage`` (PERF-06).

        Returns the cached RAW response when a request with identical params was made
        within the TTL; otherwise issues the real call and caches the result. A
        ``cache_ttl <= 0`` disables caching (always a fresh call). Only SUCCESSFUL
        responses are cached — a raised exception propagates to the caller's
        graceful-degrade fallback and is NOT cached (so the next call retries).
        """
        ce = self._ce_client()
        if self._cache_ttl <= 0:
            fresh: dict[str, Any] = ce.get_cost_and_usage(**kwargs)
            return fresh
        key = _ce_cache_key(kwargs)
        now = self._clock()
        hit = self._ce_cache.get(key)
        if hit is not None and hit[0] > now:
            return hit[1]
        out: dict[str, Any] = ce.get_cost_and_usage(**kwargs)  # a failure here is NOT cached
        self._ce_cache[key] = (now + self._cache_ttl, out)
        return out

    def report(self, *, run_id: str = "", window: str = "7d", start: str = "", end: str = "") -> dict[str, Any]:
        """Pull actual spend from Cost Explorer (``get_cost_and_usage``); fall back to local estimate."""
        try:
            period = self._ce_period(window, start=start, end=end)
            out = self._cached_get_cost_and_usage(
                TimePeriod=period, Granularity="DAILY", Metrics=["UnblendedCost"],
            )
            results = out.get("ResultsByTime", [])
            total = 0.0
            for r in results:
                amt = r.get("Total", {}).get("UnblendedCost", {}).get("Amount", "0")
                total += float(amt or 0.0)
            return {"run_id": run_id, "window": window, "total_usd": round(total, 4),
                    "source": "cost_explorer", "periods": len(results)}
        except Exception as e:
            local = super().report(run_id=run_id, window=window)
            local["source"] = "local_estimate (cost_explorer unavailable)"
            local["error"] = str(e)
            return local

    # ── the shared CE window builder (reused by report + inference_spend, G4) ────
    @staticmethod
    def _ce_period(window: str, *, start: str = "", end: str = "") -> dict[str, str]:
        """Build a Cost-Explorer ``TimePeriod`` (Start/End ISO dates) from a ``Nd`` window.

        The SAME window math ``report`` uses: ``days`` parsed from the ``window`` (e.g. ``"7d"``
        → 7), the period running ``today - days`` → ``today`` unless explicit ``start``/``end``
        override. Shared so the G4 ``inference_spend`` read and the M8 cost ``report`` issue an
        identical, consistent Cost-Explorer time window.
        """
        import datetime as _dt

        today = _dt.date.today()
        days = int("".join(c for c in window if c.isdigit()) or "7")
        return {"Start": start or str(today - _dt.timedelta(days=days)), "End": end or str(today)}

    def inference_spend(
        self,
        *,
        endpoint: str = "",
        window: str = "7d",
        budget: float | None = None,
        service: str = "Amazon SageMaker",
        group_by: str = "USAGE_TYPE",
        start: str = "",
        end: str = "",
    ) -> dict[str, Any]:
        """Read REAL per-service inference spend from Cost Explorer (G4 ``ce:GetCostAndUsage``).

        The managed inference-spend read the G4 ``track_spend`` delegates to. Issues a real
        ``ce.get_cost_and_usage`` over the ``window`` (DAILY, ``UnblendedCost``), FILTERED to the
        SageMaker ``service`` and GROUPED by ``group_by`` (``USAGE_TYPE`` by default), then computes:

          * ``total_usd`` — summed unblended cost over the window;
          * ``run_rate`` — average daily cost (``total_usd / num_days``) — the inference run-rate;
          * ``spend_trend`` — ``up`` / ``down`` / ``flat`` from the later-half vs earlier-half daily
            average (a >5 % shift either way; else flat);
          * ``vs_budget`` — ``over`` / ``under`` vs the passed ``budget`` window ceiling (``unknown``
            when no budget given);
          * ``by_usage_type`` — the per-group cost breakdown (the grouped dimension totals);
          * ``daily`` — the per-day total series the trend is computed from.

        The CE client is injectable (``ce_client=`` on the advisor) so this is directly verifiable
        offline with a recording-fake (the offline AWS API emulator does NOT implement Cost Explorer). A CE failure
        degrades to a structured zero result (``source="cost_explorer_unavailable"`` + ``error``)
        — never the old P2 placeholder, never a crash.
        """
        try:
            period = self._ce_period(window, start=start, end=end)
            filt = {"Dimensions": {"Key": "SERVICE", "Values": [service]}}
            out = self._cached_get_cost_and_usage(
                TimePeriod=period,
                Granularity="DAILY",
                Metrics=["UnblendedCost"],
                Filter=filt,
                GroupBy=[{"Type": "DIMENSION", "Key": group_by}],
            )
            results = out.get("ResultsByTime", [])
            daily: list[float] = []
            by_usage_type: dict[str, float] = {}
            currency = "USD"
            for r in results:
                day_total = 0.0
                for g in r.get("Groups", []) or []:
                    keys = g.get("Keys", []) or []
                    metric = g.get("Metrics", {}).get("UnblendedCost", {})
                    amt = float(metric.get("Amount", "0") or 0.0)
                    currency = metric.get("Unit", currency)
                    day_total += amt
                    if keys:
                        by_usage_type[keys[0]] = round(by_usage_type.get(keys[0], 0.0) + amt, 6)
                # a window with no GroupBy (Total only) still contributes to the daily series
                if not r.get("Groups"):
                    amt = float(r.get("Total", {}).get("UnblendedCost", {}).get("Amount", "0") or 0.0)
                    day_total += amt
                daily.append(round(day_total, 6))

            num_days = len(daily) or 1
            total_usd = round(sum(daily), 4)
            run_rate = round(total_usd / num_days, 4)
            spend_trend = self._spend_trend(daily)
            vs_budget = self._vs_budget(total_usd, budget)

            return {
                "run_rate": run_rate,
                "spend_trend": spend_trend,
                "vs_budget": vs_budget,
                "total_usd": total_usd,
                "currency": currency,
                "window": window,
                "endpoint": endpoint,
                "service": service,
                "by_usage_type": by_usage_type,
                "daily": daily,
                "periods": len(results),
                "budget": budget,
                "source": "cost_explorer",
            }
        except Exception as e:
            return {
                "run_rate": 0.0,
                "spend_trend": "unknown",
                "vs_budget": self._vs_budget(0.0, budget),
                "total_usd": 0.0,
                "currency": "USD",
                "window": window,
                "endpoint": endpoint,
                "service": service,
                "by_usage_type": {},
                "daily": [],
                "periods": 0,
                "budget": budget,
                "source": "cost_explorer_unavailable",
                "error": str(e),
            }

    @staticmethod
    def _spend_trend(daily: list[float], *, rel_threshold: float = 0.05) -> str:
        """Bucket a daily-cost series into up | down | flat (later-half vs earlier-half avg)."""
        if len(daily) < 2:
            return "flat"
        mid = len(daily) // 2
        earlier = daily[:mid] or daily[: mid + 1]
        later = daily[mid:]
        earlier_avg = sum(earlier) / len(earlier) if earlier else 0.0
        later_avg = sum(later) / len(later) if later else 0.0
        if earlier_avg <= 0:
            return "up" if later_avg > 0 else "flat"
        delta = (later_avg - earlier_avg) / earlier_avg
        if delta > rel_threshold:
            return "up"
        if delta < -rel_threshold:
            return "down"
        return "flat"

    @staticmethod
    def _vs_budget(total_usd: float, budget: float | None) -> str:
        """Compare window spend to the passed budget ceiling → over | under | unknown."""
        if budget is None:
            return "unknown"
        return "over" if total_usd > float(budget) else "under"


# ──────────────────────────────────────────────────────────────────────────────
# The cost-ceiling PreToolUse guardrail (the M2 cost hook seam, ADR-015)
# ──────────────────────────────────────────────────────────────────────────────
def cost_ceiling_hook(advisor: LocalCostAdvisor | None = None) -> Hook:
    """Adapt a budget ceiling into a deterministic ``PreToolUse`` cost guardrail (ADR-015).

    The hook MATCHES any action carrying a ``budget_ceiling`` in its args; its ACTION prices the
    action via the advisor, checks it against the ceiling for the action's ``run_id``, and returns a
    DENYING :class:`Verdict` when a **hard** ceiling would be breached (fail-closed — the model
    cannot override). A soft ceiling allows-but-flags (the approval gate warns+confirms). This is the
    cost backstop the least-priv IAM (M8) and the approval gate layer with (doc 05 §8.8 / §11 Q2).
    """
    advisor = advisor or LocalCostAdvisor()

    def _matches(ctx: HookContext) -> bool:
        return "budget_ceiling" in ctx.args and ctx.args.get("budget_ceiling") is not None

    def _action(ctx: HookContext) -> Verdict:
        tool = ctx.tool or ctx.action
        run_id = str(ctx.args.get("run_id", "default"))
        ceiling = ctx.args.get("budget_ceiling")
        hard = bool(ctx.args.get("budget_hard", False))
        est = ctx.args.get("cost_estimate")
        if est is None:
            est = advisor.estimate(action=tool, flow_id=str(ctx.args.get("flow_id", ""))).usd
        decision = advisor.check_budget(run_id=run_id, estimate=float(est),
                                        ceiling=float(ceiling) if ceiling is not None else 0.0,
                                        hard=hard)
        if not decision.allowed:
            return Verdict(allowed=False, reason=f"cost ceiling: {decision.reason}",
                           detail={"budget": decision.__dict__})
        return Verdict(allowed=True, reason=decision.reason, detail={"budget": decision.__dict__})

    return Hook(name="cost_ceiling", event=HookEvent.PRE_TOOL_USE, matcher=_matches, action=_action)


# ── self-registration (local default offline; CloudWatch managed variant) ────────
register_adapter(CostAdvisorBackend, "local", lambda: LocalCostAdvisor(), is_default=True, is_local=True)
register_adapter(CostAdvisorBackend, "cloudwatch", lambda: CloudWatchCostAdvisor())
