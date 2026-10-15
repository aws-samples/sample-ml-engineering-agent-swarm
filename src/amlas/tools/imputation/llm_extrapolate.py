"""LLM-based extrapolation for numeric AND text/"verbiage" gaps (spec sheet 2.5).

The customer's explicit ask (transcript 7:08, "gap in verbiage ... extrapolate"):
fill semantically-rich missing strings (and hard numeric gaps) by conditioning an
LLM on the surrounding row context + free-text ``domain_notes``.

For testability the actual model call is an **injectable callable** (``llm_fn``).
In production this is wired to Bedrock Converse via a small bounded sub-agent
(master plan §3 LLM Extrapolation agent); in offline validation a mock callable is
injected so the classical fallback and integration paths run with no live AWS.
The result is ALWAYS handed to ``evaluate_imputation`` by the caller (research KB
R2 — never trust an unverified LLM fill); ``max_llm_fraction`` is enforced by the
Strategist, not here.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from strands import tool

from ...config.aws_region import resolve_aws_region
from ...errors import tool_error_boundary

# An injected LLM callable receives a single prompt string and returns the model's
# raw text response. The default (None) routes to a lazy Bedrock Converse call.
LLMCallable = Callable[[str], str]

_BATCH_CAP = 100


def _build_prompt(
    rows: list[dict[str, Any]],
    column: str,
    context_columns: list[str],
    domain_notes: str,
    dtype: str,
) -> str:
    """Construct the extrapolation prompt (context rows + domain notes + format)."""
    lines = [
        "You are a domain data-imputation assistant for long-lineage industrial records.",
        f"Infer the missing value of column '{column}' (dtype={dtype}) for each row",
        "using ONLY the provided context columns and domain notes.",
        "",
        f"Context columns: {context_columns}",
        f"Domain notes: {domain_notes or '(none)'}",
        "",
        "Rows needing a value (JSON):",
        json.dumps(rows, default=str),
        "",
        "Respond with ONLY a JSON object: "  # nosemgrep
        '{"values": [...], "confidence": [...], "rationale": "..."} '
        "where values has exactly one entry per row in order, and confidence is 0..1.",
    ]
    return "\n".join(lines)


def _default_bedrock_call(prompt: str) -> str:  # pragma: no cover — needs live AWS
    """Lazy Bedrock Converse fallback (only used when no ``llm_fn`` is injected)."""
    import boto3

    from ...config.runtime import load_config, supports_temperature

    client = boto3.client("bedrock-runtime", region_name=resolve_aws_region())
    model_id = load_config().models.bounded
    inference: dict[str, Any] = {"maxTokens": 2048}
    if supports_temperature(model_id):
        inference["temperature"] = 0.2
    resp = client.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig=inference,
    )
    text = "\n".join(
        block["text"] for block in resp["output"]["message"]["content"] if "text" in block
    )
    return text


def _coerce_values(values: list[Any], dtype: str) -> list[Any]:
    """Coerce returned values to the requested dtype (numeric → float)."""
    d = str(dtype).lower()
    if d.startswith(("float", "int", "number")):
        out: list[Any] = []
        for v in values:
            try:
                out.append(float(v))
            except (TypeError, ValueError) as e:
                raise ValueError(f"non-numeric LLM value {v!r} for numeric column") from e
        return out
    return [None if v is None else str(v) for v in values]


def extrapolate_values(
    rows: list[dict[str, Any]],
    column: str,
    context_columns: list[str],
    domain_notes: str,
    dtype: str,
    llm_fn: LLMCallable | None = None,
) -> dict[str, Any]:
    """Core extrapolation (plain function; ``llm_fn`` injectable for isolated validation).

    Returns ``{"values","confidence","rationale"}`` (values length == #rows) or an
    error dict on count-mismatch / parse failure so the caller can fall back.
    """
    if not rows:
        return {"values": [], "confidence": [], "rationale": "no missing rows"}

    if len(rows) > _BATCH_CAP:
        # chunk to respect the model context window
        chunks = [rows[i : i + _BATCH_CAP] for i in range(0, len(rows), _BATCH_CAP)]
        values: list[Any] = []
        confidence: list[float] = []
        rationales: list[str] = []
        for chunk in chunks:
            sub = extrapolate_values(chunk, column, context_columns, domain_notes, dtype, llm_fn)
            if not sub.get("values") and sub.get("success") is False:
                return sub
            values.extend(sub["values"])
            confidence.extend(sub["confidence"])
            rationales.append(sub.get("rationale", ""))
        return {"values": values, "confidence": confidence, "rationale": " | ".join(rationales)}

    call = llm_fn or _default_bedrock_call
    prompt = _build_prompt(rows, column, context_columns, domain_notes, dtype)
    try:
        raw = call(prompt)
    except Exception as e:
        return {"success": False, "error": f"LLM call failed: {e}", "values": [], "confidence": []}

    parsed = _parse_llm_json(raw)
    if parsed is None:
        return {"success": False, "error": "LLM response was not valid JSON", "values": [], "confidence": []}

    values = parsed.get("values", [])
    if len(values) != len(rows):
        return {
            "success": False,
            "error": f"count mismatch: LLM returned {len(values)} values for {len(rows)} rows",
            "values": [],
            "confidence": [],
        }
    try:
        values = _coerce_values(values, dtype)
    except ValueError as e:
        return {"success": False, "error": str(e), "values": [], "confidence": []}

    confidence = parsed.get("confidence", [0.5] * len(values))
    if len(confidence) != len(values):
        confidence = [0.5] * len(values)
    confidence = [float(c) for c in confidence]
    return {"values": values, "confidence": confidence, "rationale": str(parsed.get("rationale", ""))}


def _parse_llm_json(raw: str) -> dict[str, Any] | None:
    """Extract the first JSON object from a raw model response."""
    raw = raw.strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                return None
        else:
            return None
    return parsed if isinstance(parsed, dict) else None


@tool
@tool_error_boundary(source="tools.imputation.llm_extrapolate")
def llm_extrapolate_values(
    rows_json: str,
    column: str,
    context_columns: list[str],
    domain_notes: str = "",
    dtype: str = "str",
) -> dict[str, Any]:
    """Fill missing numeric/text values via LLM extrapolation from row context.

    Args:
        rows_json: JSON-encoded list of context-row dicts (the rows whose
            ``column`` is missing), each containing the ``context_columns``.
        column: The column to infer.
        context_columns: Columns provided to the model as evidence.
        domain_notes: Free-text domain guidance (the steering "dial" notes).
        dtype: Target dtype hint (``"float"``/``"int"`` coerce to numeric).

    Returns:
        ``{"values":[...],"confidence":[...],"rationale":str}`` (length == #rows)
        or ``{"success": False, "error": ...}`` so the caller can fall back to a
        classical method.
    """
    try:
        rows = json.loads(rows_json) if isinstance(rows_json, str) else rows_json
    except json.JSONDecodeError as e:
        return {"success": False, "error": f"rows_json is not valid JSON: {e}"}
    if not isinstance(rows, list):
        return {"success": False, "error": "rows_json must decode to a list of row dicts"}

    return extrapolate_values(
        rows=rows,
        column=column,
        context_columns=context_columns,
        domain_notes=domain_notes,
        dtype=dtype,
        llm_fn=None,  # production path; tests call extrapolate_values() with a mock
    )
