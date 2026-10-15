"""``readiness_flow`` — the Stage-9 production-readiness gate (roadmap M4; doc 06 §5.3).

The invariant Stage-9 sequence (the 10-gate go/no-go that PRECEDES deploy):

    **run the 10 readiness gates (run_readiness_gate per gate) → build risk register →
    compile readiness review → approval decision (go / no-go)**

The 10 gates (doc 04 Stage 9; T-PR-58 parameterized): Quality, Cost, Latency, Reliability,
Security, Privacy, Safety, Explainability, Operability, Maintainability. Each emits a
pass/concern/fail verdict over the evidence the caller supplies (Arbiter verdict, sizing,
guardrail/governance results). A **failing gate BLOCKS** — the review's
``approval_recommendation`` is ``no_go`` and ``deploy_flow`` must not proceed.

This flow is **read-only / pure-reasoning** (it provisions nothing; doc 05 §3.1b), so it is
NOT spend-incurring — but the PUBLIC entrypoint is still wrapped with the M1
``plan_approve_execute`` for surface symmetry with ``deploy_flow`` (readiness_flow classifies
as CHEAP_READ, so it auto-approves and runs; the gate it produces is what blocks deploy).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..control_plane.approvals import ApprovalPolicy
from ..control_plane.plan_mode import plan_approve_execute
from ..tools import deploy_neutral as D
from ._util import raw

__all__ = ["READINESS_GATES", "ReadinessFlowResult", "execute_readiness_flow", "run_readiness_flow"]

READINESS_GATES = D.READINESS_GATES  # the 10 canonical gate names (re-exported)


@dataclass
class ReadinessFlowResult:
    """The invariant output contract of one ``readiness_flow`` run."""

    status: str  # "go" | "no_go" | "failed" | "denied"
    approved: bool = True
    recommendation: str = ""  # "go" | "no_go"
    gates: list[dict[str, Any]] = field(default_factory=list)
    passed: int = 0
    concerns: int = 0
    blocking: list[dict[str, Any]] = field(default_factory=list)
    risk_register: dict[str, Any] = field(default_factory=dict)
    review: dict[str, Any] = field(default_factory=dict)
    plan: dict[str, Any] = field(default_factory=dict)
    detail: dict[str, Any] = field(default_factory=dict)


def execute_readiness_flow(
    *,
    model: Any,
    evidence: dict[str, dict[str, Any]] | None = None,
    gates: list[str] | None = None,
) -> ReadinessFlowResult:
    """The INNER executor: run the 10 gates → risk register → review → go/no-go.

    Args:
        model: The candidate (ModelRef / path / dict) under review.
        evidence: Per-gate evidence ``{gate_name: {...}}`` (e.g. ``{"quality": {"arbiter_passed": True},
            "latency": {"p99_latency_ms": 40, "latency_sla_ms": 100}, "safety": {"guardrail_passed": True}}``).
            Missing evidence for a gate → that gate emits ``concern`` (non-blocking).
        gates: Optional subset of gate names to run (defaults to all 10).
    """
    evidence = evidence or {}
    gate_names = [g.strip().lower() for g in (gates or list(READINESS_GATES))]

    # ── 1. run the 10 readiness gates (T-PR-58 per gate) ────────────────────────
    gate_results: list[dict[str, Any]] = []
    for gate in gate_names:
        try:
            res = raw(D.run_readiness_gate)(gate, evidence.get(gate, {}))
        except Exception as e:
            return ReadinessFlowResult(status="failed", detail={"stage": "gate", "gate": gate, "error": str(e)})
        gate_results.append(res)

    # ── 2. build the risk register (T-PR-60) ────────────────────────────────────
    try:
        risk_register = raw(D.build_risk_register)(model, gate_results)
    except Exception as e:
        return ReadinessFlowResult(status="failed", gates=gate_results, detail={"stage": "risk", "error": str(e)})

    # ── 3. compile the readiness review + go/no-go (T-PR-61) ────────────────────
    try:
        review = raw(D.compile_readiness_review)(model, gate_results)
    except Exception as e:
        return ReadinessFlowResult(status="failed", gates=gate_results, risk_register=risk_register,
                                   detail={"stage": "review", "error": str(e)})

    recommendation = str(review.get("approval_recommendation", "no_go"))
    blocking = list(review.get("blocking_gaps", []))
    return ReadinessFlowResult(
        status=recommendation,  # "go" | "no_go"
        approved=True,
        recommendation=recommendation,
        gates=gate_results,
        passed=int(review.get("passed", 0)),
        concerns=int(review.get("concerns", 0)),
        blocking=blocking,
        risk_register=risk_register,
        review=review,
        detail={"gates_run": gate_names},
    )


def run_readiness_flow(
    *,
    model: Any,
    evidence: dict[str, dict[str, Any]] | None = None,
    gates: list[str] | None = None,
    policy: ApprovalPolicy | None = None,
    steering: Any | None = None,
) -> ReadinessFlowResult:
    """Run ``readiness_flow`` wrapped by ``plan_approve_execute`` (surface symmetry; ADR-014).

    readiness_flow is read-only (CHEAP_READ), so the gate auto-approves and the flow runs;
    the resulting ``no_go`` recommendation is what blocks the downstream ``deploy_flow``. On
    the rare denial (e.g. an ``allowlist`` policy that excludes it) the result is
    ``status="denied"`` and no gates run.
    """
    outcome = plan_approve_execute(
        flow_id="readiness_flow",
        params={"model": model, "evidence": evidence, "gates": gates},
        executor=execute_readiness_flow,
        policy=policy,
        steering=steering,
    )
    if not outcome["approved"]:
        return ReadinessFlowResult(
            status="denied", approved=False, plan=outcome["plan"],
            detail={"decision": outcome["decision"], "reason": "approval gate denied — no gates run"},
        )
    result: ReadinessFlowResult = outcome["result"]
    result.plan = outcome["plan"]
    return result
