"""Requirement→spec flow — "what data do I need to synthesize?" (spec sheet 2.6, §7).

The customer's explicit ask (transcript 13:73, 10:32): "feed the requirement to a
model" and get a synthesis plan. ``build_synthesis_spec`` turns a
natural-language / structured requirement (+ an optional reference dataset's
schema) into a validated **SynthesisSpec**: target columns + dtypes,
distributions/constraints, conditional balance filters, row volume, and a
generator preference.

Two paths, mirroring the Phase-7 injectable-LLM pattern
(``imputation.llm_extrapolate``):

* **LLM path** — an injectable ``llm_fn(prompt)->str`` returns a JSON spec
  (stubbed in tests; wired to Bedrock Converse in production). The returned spec
  is schema-validated before it is trusted.
* **Deterministic path** — when no LLM is available but a reference dataset is
  provided, infer columns/dtypes/distributions directly from the data (no model
  needed). This guarantees the tool always produces a usable spec.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from strands import tool

from ...errors import tool_error_boundary
from ._common import SDV_GENERATORS, canonical_generator, read_csv_or_error

LLMCallable = Callable[[str], str]

# dtype → SynthesisSpec column-type token
_NUMERIC = "numeric"
_CATEGORICAL = "categorical"
_DATETIME = "datetime"


def _column_type(series: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(series):
        return _NUMERIC
    if pd.api.types.is_datetime64_any_dtype(series):
        return _DATETIME
    return _CATEGORICAL


def infer_schema_from_frame(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Deterministically infer a per-column spec (type + distribution stats) from data.

    Numeric → ``{min,max,mean,std}``; categorical → ``{categories, frequencies}``.
    This is the no-LLM fallback the spec sheet requires.
    """
    columns: list[dict[str, Any]] = []
    for col in df.columns:
        s = df[col]
        ctype = _column_type(s)
        entry: dict[str, Any] = {"name": str(col), "type": ctype, "dtype": str(s.dtype)}
        if ctype == _NUMERIC:
            present = s.dropna()
            entry["distribution"] = {
                "min": float(present.min()) if len(present) else 0.0,
                "max": float(present.max()) if len(present) else 0.0,
                "mean": float(present.mean()) if len(present) else 0.0,
                "std": float(present.std()) if len(present) > 1 else 0.0,
            }
        else:
            vc = s.astype(str).value_counts(normalize=True)
            entry["distribution"] = {
                "categories": [str(c) for c in vc.index.tolist()[:50]],
                "frequencies": [round(float(f), 6) for f in vc.values.tolist()[:50]],
            }
        columns.append(entry)
    return columns


def _build_spec_prompt(requirement: str, schema: list[dict[str, Any]] | None) -> str:
    """Construct the requirement→spec LLM prompt."""
    lines = [
        "You are a synthetic-data planning assistant for industrial ML datasets.",
        "Given a natural-language requirement (and optionally a reference schema),",
        "produce a SynthesisSpec describing exactly what data to synthesize.",
        "",
        f"Requirement: {requirement}",
        "",
        f"Reference schema (optional): {json.dumps(schema, default=str) if schema else '(none)'}",
        "",
        "Respond with ONLY a JSON object of the form:",
        '{"columns":[{"name":...,"type":"numeric|categorical|datetime"}],',
        ' "constraints":[...], "conditions":{...}, "volume":<int>,',
        ' "generator":"gaussian_copula|ctgan|tvae|smote|llm"}',
    ]
    return "\n".join(lines)


def _normalize_spec(spec: dict[str, Any], default_volume: int) -> dict[str, Any]:
    """Coerce a raw spec dict into the canonical SynthesisSpec shape (fill defaults)."""
    columns = spec.get("columns") or []
    norm_cols: list[dict[str, Any]] = []
    for c in columns:
        if isinstance(c, dict):
            norm_cols.append({"name": str(c.get("name", "")), "type": c.get("type", _NUMERIC), **{k: v for k, v in c.items() if k not in ("name", "type")}})
        else:
            norm_cols.append({"name": str(c), "type": _NUMERIC})
    generator = canonical_generator(spec.get("generator", "gaussian_copula"))
    if generator not in SDV_GENERATORS and generator not in ("smote", "llm"):
        generator = "gaussian_copula"
    try:
        volume = int(spec.get("volume", default_volume))
    except (TypeError, ValueError):
        volume = default_volume
    return {
        "columns": norm_cols,
        "constraints": spec.get("constraints", []) or [],
        "conditions": spec.get("conditions", {}) or {},
        "volume": max(1, volume),
        "generator": generator,
    }


def validate_synthesis_spec(spec: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate a SynthesisSpec; return ``(is_valid, problems)``."""
    problems: list[str] = []
    if not isinstance(spec, dict):
        return False, ["spec is not a dict"]
    cols = spec.get("columns")
    if not cols or not isinstance(cols, list):
        problems.append("spec has no 'columns' list")
    else:
        for c in cols:
            if not isinstance(c, dict) or not c.get("name"):
                problems.append(f"column entry missing 'name': {c!r}")
            elif c.get("type") not in (_NUMERIC, _CATEGORICAL, _DATETIME):
                problems.append(f"column '{c.get('name')}' has invalid type '{c.get('type')}'")
    if int(spec.get("volume", 0)) <= 0:
        problems.append("volume must be a positive integer")
    return (len(problems) == 0), problems


def build_synthesis_spec_core(
    requirement: str,
    schema: list[dict[str, Any]] | None = None,
    seed_df: pd.DataFrame | None = None,
    default_volume: int = 1000,
    llm_fn: LLMCallable | None = None,
) -> dict[str, Any]:
    """Core requirement→spec (plain function; ``llm_fn`` injectable for isolated validation).

    Resolution order:
      1. If ``llm_fn`` is provided, ask the LLM and validate its JSON spec.
      2. Else if a reference frame/schema is available, infer deterministically.
      3. Else error (cannot plan from a bare requirement with no LLM, no data).
    """
    if schema is None and seed_df is not None:
        schema = infer_schema_from_frame(seed_df)

    # ── LLM path ─────────────────────────────────────────────────────────────
    if llm_fn is not None:
        prompt = _build_spec_prompt(requirement, schema)
        try:
            raw = llm_fn(prompt)
        except Exception as e:
            return {"success": False, "error": f"LLM call failed: {e}"}
        parsed = _parse_json(raw)
        if parsed is None:
            return {"success": False, "error": "LLM response was not valid JSON"}
        spec = _normalize_spec(parsed, default_volume)
        ok, problems = validate_synthesis_spec(spec)
        if not ok:
            return {"success": False, "error": "invalid synthesis spec", "problems": problems, "synthesis_spec": spec}
        return {"synthesis_spec": spec, "source": "llm"}

    # ── deterministic schema-inference path ──────────────────────────────────
    if schema is not None:
        spec = _normalize_spec(
            {"columns": schema, "volume": default_volume, "generator": "gaussian_copula"},
            default_volume,
        )
        ok, problems = validate_synthesis_spec(spec)
        if not ok:
            return {"success": False, "error": "inferred spec invalid", "problems": problems}
        return {"synthesis_spec": spec, "source": "schema_inference"}

    return {
        "success": False,
        "error": "cannot build a spec from a bare requirement without an LLM or a reference dataset",
    }


def _parse_json(raw: str) -> dict[str, Any] | None:
    raw = raw.strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                return None
        else:
            return None
    return parsed if isinstance(parsed, dict) else None


@tool
@tool_error_boundary(source="tools.synthetic.requirement")
def build_synthesis_spec(
    requirement: str,
    seed_path: str | None = None,
    volume: int = 1000,
) -> dict[str, Any]:
    """Turn a requirement (+ optional reference dataset) into a SynthesisSpec (§7).

    Args:
        requirement: Natural-language description of the data to synthesize
            (e.g. "more class-3 fault rows under high speed").
        seed_path: Optional CSV whose schema/distributions ground the spec; when
            present the deterministic schema-inference path is used (no LLM).
        volume: Default number of rows when the requirement doesn't specify one.

    Returns:
        ``{"synthesis_spec":{"columns","constraints","conditions","volume",
        "generator"}, "source":"schema_inference"}`` or
        ``{"success": False, "error": ...}``. (The production LLM path is wired
        via the Synthetic Data agent; offline validation inject a mock through
        ``build_synthesis_spec_core``.)
    """
    seed_df = None
    if seed_path is not None:
        df, err = read_csv_or_error(seed_path)
        if err is not None:
            return err
        seed_df = df

    return build_synthesis_spec_core(
        requirement=requirement,
        seed_df=seed_df,
        default_volume=int(volume),
        llm_fn=None,  # production LLM path wired by the agent; callers may use the core fn
    )


def sample_from_spec(spec: dict[str, Any], seed: int = 0) -> pd.DataFrame:
    """Deterministically materialise a small frame matching a SynthesisSpec.

    Used by the LLM-row classical fallback (``llm_rows``) and tests: numeric
    columns sampled from the spec's ``{mean,std,min,max}``; categorical columns
    sampled from the spec's ``{categories, frequencies}``.
    """
    rng = np.random.default_rng(seed)
    n = int(spec.get("volume", 100))
    data: dict[str, Any] = {}
    for col in spec.get("columns", []):
        name = col["name"]
        ctype = col.get("type", _NUMERIC)
        dist = col.get("distribution", {}) or {}
        if ctype == _NUMERIC:
            mean = float(dist.get("mean", 0.0))
            std = float(dist.get("std", 1.0)) or 1.0
            vals = rng.normal(mean, std, size=n)
            if "min" in dist and "max" in dist:
                vals = np.clip(vals, float(dist["min"]), float(dist["max"]))
            data[name] = np.round(vals, 4)
        else:
            cats = dist.get("categories") or ["a", "b"]
            freqs = dist.get("frequencies")
            p = None
            if freqs and len(freqs) == len(cats):
                arr = np.asarray(freqs, dtype=float)
                if arr.sum() > 0:
                    p = arr / arr.sum()
            data[name] = rng.choice(cats, size=n, p=p)
    return pd.DataFrame(data)
