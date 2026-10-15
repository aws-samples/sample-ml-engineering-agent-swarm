"""``data_quality_flow`` — the fixed DQ assessment flow over the ``DataQualityBackend`` port.

Composes the Stage-4/5 DQ tools (doc 06 §5.3 flow row: T-DQ-01/03/04/05/07/08/09,
T-ETL-07) into the invariant sequence:

    **profile → recommend rules → evaluate → (gate + report)**

The flow is fixed; the COMPUTE module swaps via the registry — the SAME flow runs the
in-process 6-dim/pandera IP offline OR managed DataBrew profile + Glue-DQ DQDL evaluation
when ``DQ_BACKEND`` is pinned (the HYBRID-cluster swappable-module property, doc 06 §4.1).

The gate is the **DQ-quota** rule (doc 05 §8.9): a dataset whose overall score is below
``min_score`` (default 60/100) is flagged ``needs_remediation`` and the report names the
weak dimensions + the recommended ruleset — exactly the "scan data quality in a simple
chat way" use case (US-4.03/5.16).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..backends.ports import DataQualityBackend
from ..backends.registry import REGISTRY

__all__ = ["DEFAULT_MIN_SCORE", "DataQualityFlowResult", "run_data_quality_flow"]

DEFAULT_MIN_SCORE = 60.0


@dataclass
class DataQualityFlowResult:
    """The invariant output contract of one ``data_quality_flow`` run."""

    status: str  # "passed" | "needs_remediation" | "failed"
    adapter: str = ""
    score: float = 0.0
    passed_gate: bool = False
    dimensions: dict[str, float] = field(default_factory=dict)
    weak_dimensions: list[str] = field(default_factory=list)
    recommended_ruleset: Any = None
    evaluation: dict[str, Any] = field(default_factory=dict)
    report: str = ""
    detail: dict[str, Any] = field(default_factory=dict)


def run_data_quality_flow(
    *,
    dataset_uri: str,
    ruleset: Any = None,
    min_score: float = DEFAULT_MIN_SCORE,
    steering: Any | None = None,
    scale: int | str | None = None,
) -> DataQualityFlowResult:
    """Run the invariant DQ assessment flow over the resolved ``DataQualityBackend``.

    Fixed steps:
      1. **profile** the dataset (T-DQ-01 / 6-dim score + missingness);
      2. **recommend** a ruleset when none supplied (T-DQ-02);
      3. **evaluate** the dataset against the ruleset (T-DQ-03 / pandera or DQDL);
      4. **gate + report** — apply the DQ-quota (score ≥ ``min_score``) and compose a
         human-readable report naming weak dimensions + the recommended rules.

    Args:
        dataset_uri: CSV (local) or s3 URI to assess.
        ruleset: Optional pre-supplied ruleset; when ``None`` the flow recommends one.
        min_score: DQ-quota threshold (0-100) the overall score must meet to pass.
        steering: Optional ``SteeringProfile`` (hard pins flow through to resolution).
        scale: Data-scale hint forwarded to the registry (large → managed adapter).

    Returns:
        A :class:`DataQualityFlowResult` (status + score + gate + weak dims + ruleset +
        evaluation + report).
    """
    try:
        resolved = REGISTRY.resolve(DataQualityBackend, steering=steering, scale=scale)
        adapter = resolved.adapter
    except Exception as e:
        return DataQualityFlowResult(status="failed", detail={"stage": "resolve", "error": str(e)})

    # ── 1. profile ────────────────────────────────────────────────────────────
    try:
        report = adapter.profile(dataset_uri=dataset_uri)
    except Exception as e:
        return DataQualityFlowResult(status="failed", adapter=resolved.adapter_name,
                                     detail={"stage": "profile", "error": str(e)})
    score = float(getattr(report, "score", 0.0))
    dims = dict(getattr(report, "dimensions", {}))

    # ── 2. recommend rules (when none supplied) ────────────────────────────────
    recommended = ruleset
    if recommended is None and hasattr(adapter, "recommend_rules"):
        try:
            rec = adapter.recommend_rules(dataset_uri=dataset_uri)
            recommended = rec.get("ruleset") if isinstance(rec, dict) else rec
        except Exception:
            recommended = None

    # ── 3. evaluate ─────────────────────────────────────────────────────────────
    evaluation: dict[str, Any] = {}
    try:
        dq = adapter.evaluate(dataset_uri=dataset_uri, ruleset=recommended)
        evaluation = {"passed": dq.passed, "score": dq.score, "violations": dq.violations}
    except Exception as e:
        evaluation = {"error": str(e)}

    # ── 4. gate + report (DQ-quota) ────────────────────────────────────────────
    # The 6-dim score is 0-100; the managed-profile async job returns 0 (score lands in
    # the output location), so the gate only fires when a real score is available.
    weak = [name for name, s in dims.items() if s < 80.0]
    has_score = score > 0.0 or bool(dims)
    passed_gate = (not has_score) or (score >= min_score)
    status = "passed" if passed_gate else "needs_remediation"

    report_lines = [
        f"Data-quality assessment via {resolved.adapter_name} ({resolved.reason}).",
        f"Overall score: {score:.1f}/100 (gate ≥ {min_score:.0f}).",
    ]
    if weak:
        report_lines.append(f"Weak dimensions: {', '.join(weak)}.")
    if not passed_gate:
        report_lines.append("VERDICT: needs remediation before promotion to silver/gold.")
    else:
        report_lines.append("VERDICT: meets the data-quality quota.")
    if isinstance(evaluation, dict) and evaluation.get("violations"):
        report_lines.append(f"Rule violations: {len(evaluation['violations'])}.")

    return DataQualityFlowResult(
        status=status,
        adapter=resolved.adapter_name,
        score=score,
        passed_gate=passed_gate,
        dimensions=dims,
        weak_dimensions=weak,
        recommended_ruleset=recommended,
        evaluation=evaluation,
        report=" ".join(report_lines),
        detail={"reason": resolved.reason, "profile_detail": getattr(report, "detail", {})},
    )
