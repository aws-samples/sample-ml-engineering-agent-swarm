"""Per-agent token cost traceability and shift-left estimates.

This module provides the token-attribution half of complete cost traceability. The
codebase already priced and aggregated
AWS-SERVICE spend (:mod:`amlas.control_plane.cost_governance` ``LocalCostAdvisor`` + the
CloudWatch/Cost-Explorer variant) and had plan-mode flow estimates — but there was **no per-agent
token attribution**: no way to say "the ``research_lead`` agent burned N Opus tokens costing $X."
This module closes that gap and consolidates the two into ONE cost surface.

Three pieces (all pure-Python, deterministic, offline-safe — no boto3):

* :class:`TokenPriceBook` — a small deterministic price book: per-model input/output USD per token
  (Opus > Sonnet > Haiku, output priced above input, mirroring Bedrock's published rate shape),
  keyed on the canonical :mod:`amlas.config.runtime` inference-profile ids. Unknown models fall
  back to a conservative non-zero rate (never silently free).
* :class:`TokenCostLedger` — records per-agent :class:`TokenUsage` (input/output tokens priced to
  USD, attributed to one of the 9 agents — the TUI chat agent + the 7 agentcore/strands leads),
  rolls up ``by_agent`` / ``total_tokens`` / ``total_usd``, is ``query``-able by agent/run, and
  composes a ``report`` — the per-agent cost surface the dashboard/CLI reads.
* :func:`estimate_agent_token_cost` — the SHIFT-LEFT pre-planning estimate: project an agent's
  token spend from an ``est_input_tokens`` / ``est_output_tokens`` budget BEFORE execution (does
  NOT touch the actuals ledger), so cost is projected up front, not only measured after.
* :func:`consolidate_token_spend` — folds a run's per-agent token spend INTO the
  ``LocalCostAdvisor`` run ledger as ``tokens:<agent>`` line items, so ONE run total reflects both
  AWS-service spend AND agent token spend (the "complete cost traceability" ask).

The USD estimate fields intentionally mirror the ``cost_governance.CostEstimate`` shape so cost is
one consolidated surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..config.runtime import HAIKU_4_5, OPUS_4_7, OPUS_4_8, SONNET_4_6

if TYPE_CHECKING:  # avoid a runtime import cycle; only needed for the type hint
    from .cost_governance import LocalCostAdvisor

__all__ = [
    "AGENTS",
    "ML_AUTOMATION_AGENTS",
    "TOKEN_COST_LEDGER",
    "TUI_AGENT",
    "TokenCostLedger",
    "TokenPriceBook",
    "TokenUsage",
    "consolidate_token_spend",
    "estimate_agent_token_cost",
    "record_agent_tokens",
]

# ── agent roster: TUI agent plus seven ML automation agents ───────────────────
# The single conversational TUI chat agent.
TUI_AGENT = "tui"
# The 7 agentcore/strands ML-automation leads (the ``leads/`` package — one per stage of the
# Research → DataEng → FE → ML → MLOps → Deploy → Monitor pipeline). These names match each lead
# factory's default ``name=`` so recorded token usage attributes to the SAME identity the agents
# run under.
ML_AUTOMATION_AGENTS: tuple[str, ...] = (
    "research_lead",
    "data_engineering_lead",
    "fe_lead",
    "ml_execution_lead",
    "mlops_lead",
    "deploy_serving_lead",
    "monitoring_lead",
)
# The full attributable roster: the TUI agent + the 7 ML-automation leads (9 total).
AGENTS: tuple[str, ...] = (TUI_AGENT, *ML_AUTOMATION_AGENTS)


# ──────────────────────────────────────────────────────────────────────────────
# Token price book — per-model input/output USD per token
# ──────────────────────────────────────────────────────────────────────────────
# USD PER TOKEN (i.e. per-1M-token rate / 1e6). Deterministic, offline. Output is priced above
# input (the standard Bedrock/Anthropic rate shape) and the tiers strictly order Opus > Sonnet >
# Haiku. These are coarse planning rates (the CloudWatch/Cost-Explorer actuals are the managed
# truth) — good enough for per-agent attribution + the shift-left projection. Keyed on the
# canonical inference-profile ids (config/runtime.py) so a routing change can't silently mis-price.
_PER_TOKEN: dict[str, tuple[float, float]] = {
    # model id: (input_usd_per_token, output_usd_per_token)
    OPUS_4_8: (15.0 / 1_000_000, 75.0 / 1_000_000),
    OPUS_4_7: (15.0 / 1_000_000, 75.0 / 1_000_000),
    SONNET_4_6: (3.0 / 1_000_000, 15.0 / 1_000_000),
    HAIKU_4_5: (0.80 / 1_000_000, 4.0 / 1_000_000),
}
# Conservative non-zero fallback for an unknown model (safe-by-default — never silently free).
_DEFAULT_PER_TOKEN: tuple[float, float] = (3.0 / 1_000_000, 15.0 / 1_000_000)


class TokenPriceBook:
    """Deterministic per-model token price book (USD per input/output token).

    ``cost`` prices ``input_tokens``/``output_tokens`` for a given model from the per-model rate
    (falling back to a conservative non-zero rate for an unknown model). ``rate`` exposes the raw
    ``(input, output)`` per-token rate. Offline + deterministic — the managed Cost-Explorer read is
    the actual-spend truth; this is the planning/attribution estimate.
    """

    def rate(self, model: str) -> tuple[float, float]:
        """Return the ``(input_usd_per_token, output_usd_per_token)`` rate for ``model``."""
        return _PER_TOKEN.get(model, _DEFAULT_PER_TOKEN)

    def cost(self, *, model: str, input_tokens: int, output_tokens: int) -> float:
        """Price ``input_tokens`` + ``output_tokens`` for ``model`` in USD (rounded to 6 dp)."""
        in_rate, out_rate = self.rate(model)
        usd = float(input_tokens) * in_rate + float(output_tokens) * out_rate
        return round(usd, 6)


# ──────────────────────────────────────────────────────────────────────────────
# Value type — one recorded per-agent token-usage line
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class TokenUsage:
    """One recorded per-agent token-usage line, priced to USD.

    ``agent`` is the attributed agent (``tui`` or one of the 7 leads); ``model`` the invoked
    inference profile; ``input_tokens`` / ``output_tokens`` the counts; ``usd`` the priced cost;
    ``run_id`` the optional run correlation; ``detail`` any extra structured context.
    """

    agent: str
    model: str
    input_tokens: int
    output_tokens: int
    usd: float
    run_id: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        """The combined input + output token count."""
        return int(self.input_tokens) + int(self.output_tokens)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain JSON-able dict (for the report + persistence)."""
        return {
            "agent": self.agent,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "usd": self.usd,
            "run_id": self.run_id,
            "detail": dict(self.detail),
        }


# ──────────────────────────────────────────────────────────────────────────────
# The per-agent token cost ledger
# ──────────────────────────────────────────────────────────────────────────────
class TokenCostLedger:
    """Process-wide per-agent token cost ledger — record + roll up + query + report.

    ``record`` prices a token-usage line via the :class:`TokenPriceBook` and attributes it to an
    agent; ``by_agent`` rolls tokens+cost up per agent; ``total_tokens`` / ``total_usd`` sum across
    agents; ``query`` filters by agent/run; ``report`` composes the consolidated per-agent cost
    surface. In-memory + deterministic (the test + offline surface); the managed actual-spend truth
    stays Cost Explorer.
    """

    def __init__(self, price_book: TokenPriceBook | None = None) -> None:
        self._price_book = price_book or TokenPriceBook()
        self._records: list[TokenUsage] = []

    def record(
        self,
        *,
        agent: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        run_id: str = "",
        detail: dict[str, Any] | None = None,
    ) -> TokenUsage:
        """Record + price one per-agent token-usage line; returns the :class:`TokenUsage`."""
        usd = self._price_book.cost(model=model, input_tokens=input_tokens, output_tokens=output_tokens)
        usage = TokenUsage(
            agent=agent,
            model=model,
            input_tokens=int(input_tokens),
            output_tokens=int(output_tokens),
            usd=usd,
            run_id=run_id,
            detail=dict(detail or {}),
        )
        self._records.append(usage)
        return usage

    def by_agent(self) -> dict[str, dict[str, Any]]:
        """Roll usage up per agent → ``{agent: {input_tokens, output_tokens, total_tokens, usd, calls}}``."""
        out: dict[str, dict[str, Any]] = {}
        for r in self._records:
            bucket = out.setdefault(
                r.agent,
                {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "usd": 0.0, "calls": 0},
            )
            bucket["input_tokens"] += r.input_tokens
            bucket["output_tokens"] += r.output_tokens
            bucket["total_tokens"] += r.total_tokens
            bucket["usd"] = round(bucket["usd"] + r.usd, 6)
            bucket["calls"] += 1
        return out

    def total_tokens(self) -> int:
        """Total input + output tokens recorded across all agents."""
        return sum(r.total_tokens for r in self._records)

    def total_usd(self) -> float:
        """Total token USD recorded across all agents (rounded to 6 dp)."""
        return round(sum(r.usd for r in self._records), 6)

    def query(self, *, agent: str | None = None, run_id: str | None = None) -> list[TokenUsage]:
        """Return all usage lines matching ``agent`` / ``run_id`` (AND)."""
        out = self._records
        if agent is not None:
            out = [r for r in out if r.agent == agent]
        if run_id is not None:
            out = [r for r in out if r.run_id == run_id]
        return list(out)

    def report(self, *, run_id: str | None = None) -> dict[str, Any]:
        """Compose the consolidated per-agent token cost surface (optionally scoped to ``run_id``)."""
        scope = self.query(run_id=run_id) if run_id else self._records
        by_agent: dict[str, dict[str, Any]] = {}
        for r in scope:
            bucket = by_agent.setdefault(
                r.agent,
                {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "usd": 0.0, "calls": 0},
            )
            bucket["input_tokens"] += r.input_tokens
            bucket["output_tokens"] += r.output_tokens
            bucket["total_tokens"] += r.total_tokens
            bucket["usd"] = round(bucket["usd"] + r.usd, 6)
            bucket["calls"] += 1
        return {
            "run_id": run_id or "",
            "total_usd": round(sum(b["usd"] for b in by_agent.values()), 6),
            "total_tokens": sum(b["total_tokens"] for b in by_agent.values()),
            "by_agent": by_agent,
            "source": "token_ledger",
        }

    def clear(self) -> None:
        """Reset the ledger (used by tests for isolation)."""
        self._records.clear()


# The process-wide singleton the surfaces / agents / hooks record token usage into.
TOKEN_COST_LEDGER = TokenCostLedger()


def record_agent_tokens(
    *,
    agent: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    run_id: str = "",
    detail: dict[str, Any] | None = None,
    ledger: TokenCostLedger | None = None,
) -> TokenUsage:
    """Record per-agent token usage into the process-wide :data:`TOKEN_COST_LEDGER`.

    The convenience any layer calls without threading the singleton (mirrors
    ``record_capability_trace`` / ``capture_error``). Pass ``ledger`` to target a specific
    :class:`TokenCostLedger` (tests). Returns the recorded :class:`TokenUsage`.
    """
    return (ledger or TOKEN_COST_LEDGER).record(
        agent=agent,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        run_id=run_id,
        detail=detail,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Shift-left pre-planning token cost estimate (before execution)
# ──────────────────────────────────────────────────────────────────────────────
def estimate_agent_token_cost(
    *,
    agent: str,
    model: str,
    est_input_tokens: int,
    est_output_tokens: int,
    price_book: TokenPriceBook | None = None,
    ledger: TokenCostLedger | None = None,
) -> dict[str, Any]:
    """Shift-left: project an agent's token cost from an estimated token budget BEFORE execution.

    The "shift-left pre-planning cost estimate" ask — cost is projected UP FRONT (during plan mode /
    approval), not only measured after a run. Prices the ``est_input_tokens`` / ``est_output_tokens``
    budget for ``agent`` on ``model`` via the price book and returns a ``CostEstimate``-shaped dict
    tagged ``source="shift_left_estimate"``. It deliberately does NOT record into any actuals ledger
    (the optional ``ledger`` param is accepted only for call-site symmetry and is never mutated), so
    a projection can never pollute measured spend.
    """
    pb = price_book or TokenPriceBook()
    usd = pb.cost(model=model, input_tokens=est_input_tokens, output_tokens=est_output_tokens)
    return {
        "agent": agent,
        "model": model,
        "est_input_tokens": int(est_input_tokens),
        "est_output_tokens": int(est_output_tokens),
        "est_total_tokens": int(est_input_tokens) + int(est_output_tokens),
        "usd": usd,
        "source": "shift_left_estimate",
    }


# ──────────────────────────────────────────────────────────────────────────────
# Consolidation — fold per-agent token spend into the AWS-service cost surface
# ──────────────────────────────────────────────────────────────────────────────
def consolidate_token_spend(
    ledger: TokenCostLedger,
    advisor: LocalCostAdvisor,
    *,
    run_id: str,
) -> float:
    """Fold a run's per-agent token spend INTO the ``LocalCostAdvisor`` run ledger; return the new run total.

    The "complete cost traceability" consolidation: each agent's token spend for ``run_id`` is
    recorded into the advisor's per-run ledger as a ``tokens:<agent>`` line item, so the advisor's
    ``spent(run_id=...)`` / ``aggregate(run_id=...)`` (the cost dashboard/report) reflect BOTH the
    AWS-service spend AND the agent token spend as ONE run total — the single cost surface. Returns
    the run's new total spend after consolidation.
    """
    report = ledger.report(run_id=run_id)
    by_agent: dict[str, dict[str, Any]] = report["by_agent"]
    for agent, bucket in by_agent.items():
        usd = float(bucket["usd"])
        if usd > 0.0:
            advisor.record_spend(run_id=run_id, action=f"tokens:{agent}", usd=usd)
    return advisor.spent(run_id=run_id)
