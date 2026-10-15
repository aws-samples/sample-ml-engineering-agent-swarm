"""``business_framing_flow`` — the Stage 1→2 business-goal → framing sequence (roadmap M6).

The invariant Stage 1–2 sequence (doc 06 §5.3 ``business_framing_flow``): **capture business
problem → define KPI → assess ML necessity → business go/no-go → select problem type → define
success metrics → build problem_card → stage-exit gate**. All CHEAP_READ pure-reasoning (no
spend), so the public entrypoint runs without an approval prompt. Composes the Stage 1–2 tools
(``tools/research_neutral.py``) — turns an NL business problem into a framed ``problem_card.json``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..tools import research_neutral as R
from ._util import raw

__all__ = ["BusinessFramingFlowResult", "run_business_framing_flow"]


@dataclass
class BusinessFramingFlowResult:
    """The invariant output contract of one ``business_framing_flow`` run."""

    status: str  # "completed" | "blocked" | "failed"
    business: dict[str, Any] = field(default_factory=dict)
    kpi: dict[str, Any] = field(default_factory=dict)
    ml_necessity: dict[str, Any] = field(default_factory=dict)
    go_no_go: dict[str, Any] = field(default_factory=dict)
    problem_type: dict[str, Any] = field(default_factory=dict)
    success_metrics: dict[str, Any] = field(default_factory=dict)
    problem_card: dict[str, Any] = field(default_factory=dict)
    stage_exit: dict[str, Any] = field(default_factory=dict)
    detail: dict[str, Any] = field(default_factory=dict)


def run_business_framing_flow(
    *,
    problem_statement: str,
    target_description: str = "",
    stakeholders: list[str] | None = None,
    pain_points: list[str] | None = None,
    current_process: str = "",
    kpi_name: str = "",
    kpi_baseline: float | None = None,
    kpi_target: float | None = None,
    kpi_unit: str = "",
    has_pattern: bool = True,
    has_data: bool = True,
    rules_sufficient: bool = False,
    num_classes: int = 0,
    inputs: list[str] | None = None,
    constraints: list[str] | None = None,
) -> BusinessFramingFlowResult:
    """Run the Stage 1–2 business-goal → framing sequence; return the framed ``problem_card``.

    Fixed steps (all CHEAP_READ pure-reasoning):
      1. **capture** the business problem (T-PR-14);
      2. **define** the business KPI (T-PR-16);
      3. **assess** ML necessity (T-PR-17);
      4. **business go/no-go** gate (T-PR-26) — if no-go, the flow returns ``blocked`` (no framing);
      5. **select** the problem type (T-PR-31);
      6. **define** the success metric (T-PR-35);
      7. **build** the problem card (T-PR-27);
      8. **stage-exit** gate (T-PR-67) for stage 2.
    """
    # ── 1. capture business problem ─────────────────────────────────────────────
    business = raw(R.capture_business_problem)(
        statement=problem_statement, stakeholders=stakeholders or [],
        current_process=current_process, pain_points=pain_points or [])

    # ── 2. define KPI ───────────────────────────────────────────────────────────
    kpi = raw(R.define_business_kpi)(objective=problem_statement, metric_name=kpi_name or "primary_kpi",
                                     baseline=kpi_baseline, target=kpi_target, unit=kpi_unit)

    # ── 3. assess ML necessity ──────────────────────────────────────────────────
    ml = raw(R.assess_ml_necessity)(problem=problem_statement, has_pattern=has_pattern,
                                    has_data=has_data, rules_sufficient=rules_sufficient)

    # ── 4. business go/no-go gate ───────────────────────────────────────────────
    go = raw(R.business_go_no_go)(clarity_score=business.get("clarity_score", 0.0),
                                  kpi_measurable=kpi.get("measurable", False),
                                  ml_recommended=ml.get("ml_recommended", False))
    if go.get("decision") != "go":
        return BusinessFramingFlowResult(status="blocked", business=business, kpi=kpi,
                                         ml_necessity=ml, go_no_go=go,
                                         detail={"blockers": go.get("blockers", [])})

    # ── 5. select problem type ──────────────────────────────────────────────────
    ptype = raw(R.select_problem_type)(target_description=target_description or problem_statement,
                                       num_classes=num_classes)

    # ── 6. define success metric ────────────────────────────────────────────────
    metrics = raw(R.define_success_metrics)(problem_type=ptype.get("problem_type", ""),
                                            threshold=kpi_target)

    # ── 7. build problem card ───────────────────────────────────────────────────
    card = raw(R.build_problem_card)(
        business_problem=problem_statement, problem_type=ptype.get("problem_type", ""),
        target=target_description or kpi_name, inputs=inputs or [],
        output_type=ptype.get("output_type", ""), success_metric=metrics.get("primary_metric", ""),
        kpi=kpi, constraints=constraints or [])

    # ── 8. stage-exit gate (stage 2) ────────────────────────────────────────────
    exit_check = raw(R.check_stage_exit)(stage="2", artifacts={
        "problem_card": card.get("problem_card"), "success_metric": metrics.get("primary_metric")})

    return BusinessFramingFlowResult(
        status="completed", business=business, kpi=kpi, ml_necessity=ml, go_no_go=go,
        problem_type=ptype, success_metrics=metrics, problem_card=card.get("problem_card", {}),
        stage_exit=exit_check,
        detail={"card_complete": card.get("complete", False), "missing": card.get("missing", [])},
    )
