"""Synthetic-engine steering + write-back synthesis report (master plan §3.1, §7).

Reuses the Phase-7 ``SteeringProfile`` dial for the synthetic engine:

* **allowed generators (hard)** — ``forbid_methods`` blocks a generator (falls
  back to the safe ``gaussian_copula`` default); ``pin_methods["generator"]``
  pins one. Enforced deterministically here, like ``apply_imputation_plan``.
* **value/range constraints (hard)** — ``value_ranges`` (``{col:{min,max}}``)
  clips synthesized numeric columns into a domain-valid band so the agent can't
  emit out-of-physics values.
* **privacy floors / acceptance thresholds** — ``thresholds`` feed
  ``validate_synthetic`` (the deterministic 3-pillar AND-gate grader).

Write-back: ``write_synthesis_report`` emits the synthesis-report manifest
(generator used, row count, the validation scorecard, steering applied) next to
the artifact — the auditable "prove it works" deliverable (transcript 8:04).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from strands import tool

from ...errors import tool_error_boundary
from ._common import SDV_GENERATORS, canonical_generator, read_csv_or_error

_SAFE_DEFAULT = "gaussian_copula"


def resolve_generator(
    requested: str, profile: dict[str, Any] | None
) -> tuple[str, list[dict[str, Any]]]:
    """Resolve the effective generator under HARD steering; return ``(gen, violations)``.

    A ``pin_methods["generator"]`` pin overrides the request; a generator on
    ``forbid_methods`` is rejected and falls back to the safe default. Mirrors
    the imputation pin/forbid enforcement so the agent can't violate the dial.
    """
    prof = profile or {}
    forbid = {str(m).strip().lower() for m in prof.get("forbid_methods", [])}
    pins = prof.get("pin_methods", {}) or {}
    violations: list[dict[str, Any]] = []

    requested_canon = canonical_generator(requested)
    pinned = pins.get("generator")
    effective = canonical_generator(pinned) if pinned else requested_canon

    if pinned and canonical_generator(pinned) != requested_canon:
        violations.append(
            {"field": "generator", "reason": f"pin override: '{requested_canon}' -> '{effective}'", "enforced": True}
        )

    if effective in forbid:
        violations.append(
            {"field": "generator", "reason": f"generator '{effective}' is forbidden; using '{_SAFE_DEFAULT}'", "enforced": True}
        )
        effective = _SAFE_DEFAULT

    # guard: unknown effective generator → safe default (smote/llm are valid spec
    # generators but not SDV tabular synthesizers, so coerce for synthesize_tabular)
    if effective not in SDV_GENERATORS:
        effective = _SAFE_DEFAULT

    return effective, violations


def apply_value_constraints(
    df: pd.DataFrame, value_ranges: dict[str, dict[str, float]] | None
) -> pd.DataFrame:
    """Clip synthesized numeric columns into the steered ``{col:{min,max}}`` band.

    Non-numeric and unconstrained columns are left untouched; unknown columns in
    ``value_ranges`` are ignored.
    """
    if not value_ranges:
        return df
    out = df.copy()
    for col, bounds in value_ranges.items():
        if col not in out.columns or not pd.api.types.is_numeric_dtype(out[col]):
            continue
        lo = bounds.get("min")
        hi = bounds.get("max")
        out[col] = out[col].clip(lower=lo, upper=hi)
    return out


def synthesis_thresholds(profile: dict[str, Any] | None) -> dict[str, float]:
    """Extract the synthetic acceptance thresholds from a SteeringProfile dict.

    Returns the steerable subset (``fidelity``, ``utility_delta``,
    ``detection_auroc``, ``privacy_dcr``) for ``validate_synthetic``; empty dict
    when no thresholds are set (validator then uses its §7 defaults).
    """
    prof = profile or {}
    thresholds = prof.get("thresholds", {}) or {}
    keys = ("fidelity", "utility_delta", "detection_auroc", "privacy_dcr")
    return {k: float(thresholds[k]) for k in keys if k in thresholds and thresholds[k] is not None}


def synthesis_report_path(synthetic_path: str) -> Path:
    """Deterministic report path next to the synthetic artifact (``*.synthesis_report.json``).

    Security note (CWE-22 review 2026-07): ``synthetic_path`` is the path of an
    artifact the synthesis tools THEMSELVES just wrote (tool-managed, not free-form
    user input); the report lands next to it by design.
    """
    p = Path(synthetic_path)
    return p.with_name(f"{p.stem}.synthesis_report.json")


@tool
@tool_error_boundary(source="tools.synthetic.steering")
def write_synthesis_report(
    synthetic_path: str,
    generator: str,
    validation: dict[str, Any] | str,
    steering_applied: dict[str, Any] | str | None = None,
) -> dict[str, Any]:
    """Write the synthesis-report manifest beside a synthetic artifact (§7 write-back).

    The report is the auditable "prove it works" record: which generator was
    used, how many rows were produced, the full 3-pillar validation scorecard
    (with the accept/reject verdict), and the steering that was applied.

    Args:
        synthetic_path: Path to the synthetic CSV the report describes.
        generator: The generator that produced it (e.g. ``gaussian_copula``).
        validation: The ``validate_synthetic`` scorecard dict (or its JSON).
        steering_applied: Optional steering summary dict (or its JSON).

    Returns:
        ``{"report_path","generator","row_count","accepted"}`` or
        ``{"success": False, "error": ...}``.
    """
    df, err = read_csv_or_error(synthetic_path)
    if err is not None:
        return err
    assert df is not None

    val = json.loads(validation) if isinstance(validation, str) else dict(validation)
    steer: dict[str, Any] = {}
    if steering_applied is not None:
        steer = json.loads(steering_applied) if isinstance(steering_applied, str) else dict(steering_applied)

    report = {
        "synthetic_path": str(synthetic_path),
        "generator": generator,
        "row_count": len(df),
        "columns": list(df.columns),
        "generated_at": datetime.now(UTC).isoformat(),
        "validation": val,
        "accepted": bool(val.get("accepted", False)),
        "steering_applied": steer,
    }

    report_path = synthesis_report_path(synthetic_path)
    try:
        report_path.write_text(json.dumps(report, indent=2, default=str))
    except Exception as e:
        return {"success": False, "error": f"failed to write synthesis report: {e}"}

    return {
        "report_path": str(report_path),
        "generator": generator,
        "row_count": len(df),
        "accepted": bool(val.get("accepted", False)),
    }
