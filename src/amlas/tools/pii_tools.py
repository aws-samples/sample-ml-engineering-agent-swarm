"""PII detection / redaction / compliance-gate tools (master-plan gap **T-1**, Phase 7).

Closes the Cat-7 governance PII gap with three ``@tool``s the chat agent (and the
CLI / control plane) can invoke — **stdlib-regex + pandas only, no heavy deps, no
cloud, fully deterministic**:

  * :func:`detect_pii`          (DQ-GOV-006) — scan a CSV's columns for emails,
    SSNs, phone numbers, credit-card-like numbers, IP addresses, and name-like
    columns, returning structured findings (column, type, match count, a
    **masked** sample) — never the raw value.
  * :func:`redact_pii`          (DQ-GOV-007) — mask the detected (or explicitly
    named) PII columns and write a de-identified CSV artifact next to the input.
  * :func:`pii_compliance_gate` (DQ-GOV-008) — run :func:`detect_pii` and PASS/FAIL
    block on policy: if any un-redacted PII remains, the gate returns
    ``action="BLOCK"``; a clean/redacted dataset returns ``action="ALLOW"``.
    Mirrors the ``run_guardrail_check`` ``{"action","passed","violations"}`` shape.

Conventions (match ``tools/data_quality/`` + ``tools/synthetic/``):
  - ``@tool``-decorated, path-based, JSON-serializable dict return, fail-soft with
    ``{"success": False, "error": "..."}``.
  - Deterministic (regex heuristics only) so the Phase-8 harness + offline validation are
    reproducible offline.


Compliance note: PII handling may be subject to GDPR (EU personal data),
HIPAA (US healthcare data) and PCI-DSS (payment card data). These tools provide
detection/redaction primitives only — adopters remain responsible for the
applicable regulatory controls (lawful basis, retention, subject rights, audit).
Compliance is shared between AWS and the customer (AWS Shared Responsibility Model).
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd
from strands import tool

from ..errors import tool_error_boundary

__all__ = ["PII_PATTERNS", "detect_pii", "pii_compliance_gate", "redact_pii"]


# ─── PII value patterns (ordered: most-specific first) ───────────────────────────
# Each entry maps a ``pii_type`` → a compiled regex matching ONE value of that type.
# Ordering matters: SSN is tried before the generic phone pattern so a
# ``123-45-6789`` is classified ``ssn``, not ``phone``.
PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "ip_address": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "phone": re.compile(
        r"(?:\+?\d{1,2}[\s.\-]?)?\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}\b"
    ),
}

# Column NAMES that strongly imply PII even when the values don't match a value
# regex (e.g. free-text ``full_name``). Substring match, case-insensitive.
_NAME_COLUMN_HINTS: tuple[str, ...] = (
    "name",
    "first_name",
    "last_name",
    "fullname",
    "surname",
    "given_name",
    "email",
    "ssn",
    "social_security",
    "phone",
    "address",
    "dob",
    "birth",
    "passport",
    "license",
)

# A permissive "looks like a person's name" heuristic for a column whose NAME hints
# at a name but whose values must still resemble ``Firstname Lastname``.
# No ReDoS: the repetition alternatives are disjoint character classes ([a-z]+ vs the
# [ \-'] separator), so backtracking is linear — flagged only by a coarse length probe.
_NAME_VALUE = re.compile(r"^[A-Z][a-z]+(?:[ \-'][A-Z][a-z]+)+$")  # nosemgrep: regex_dos

# Fraction of non-null cells that must match for a value-regex column to be flagged.
_MATCH_THRESHOLD = 0.5


def _mask_value(value: str, pii_type: str) -> str:
    """Return a privacy-preserving masked rendering of one PII value.

    Keeps just enough structure to be recognizable (e.g. an email keeps its ``@``
    and TLD) while never revealing the sensitive core — the deterministic sample
    surfaced in a finding.
    """
    v = str(value)
    if pii_type == "email" and "@" in v:
        local, _, domain = v.partition("@")
        keep = local[0] if local else "*"
        return f"{keep}{'*' * max(len(local) - 1, 2)}@{domain}"
    # Numeric-ish types (ssn/phone/card/ip): reveal only the last two chars.
    tail = v[-2:] if len(v) >= 2 else v
    return f"{'*' * max(len(v) - 2, 3)}{tail}"


def _column_pii_type(series: pd.Series, column_name: str) -> tuple[str, int] | None:
    """Classify a column as PII, returning ``(pii_type, match_count)`` or ``None``.

    Detection order:
      1. VALUE regex — if ≥ :data:`_MATCH_THRESHOLD` of non-null cells match a
         :data:`PII_PATTERNS` regex, classify by that type (most-specific first).
      2. NAME heuristic — a column whose name hints at a person's name AND whose
         values look like ``Firstname Lastname`` is flagged ``name``.
    """
    non_null = series.dropna().astype(str)
    if len(non_null) == 0:
        return None

    # 1) value-based detection (specific → general, per PII_PATTERNS order)
    for pii_type, pattern in PII_PATTERNS.items():
        # Bind ``pattern`` as a default arg so the lambda captures THIS iteration's
        # compiled regex (not the last loop value) — B023 closure-safety.
        match_count = int(
            non_null.apply(lambda x, pattern=pattern: bool(pattern.search(x))).sum()
        )
        if match_count and match_count / len(non_null) >= _MATCH_THRESHOLD:
            return pii_type, match_count

    # 2) name-column heuristic (free-text names have no value regex)
    lname = column_name.lower()
    if any(hint in lname for hint in _NAME_COLUMN_HINTS):
        name_matches = int(non_null.apply(lambda x: bool(_NAME_VALUE.match(x))).sum())
        if name_matches and name_matches / len(non_null) >= _MATCH_THRESHOLD:
            return "name", name_matches
    return None


def _scan_frame(df: pd.DataFrame, columns: list[str] | None) -> list[dict[str, Any]]:
    """Return the ordered list of PII findings for ``df`` (registry-column order)."""
    scan_cols = columns if columns else list(df.columns)
    findings: list[dict[str, Any]] = []
    for col in scan_cols:
        if col not in df.columns:
            continue
        classified = _column_pii_type(df[col], col)
        if classified is None:
            continue
        pii_type, match_count = classified
        sample_raw = df[col].dropna().astype(str)
        sample = sample_raw.iloc[0] if len(sample_raw) else ""
        findings.append(
            {
                "column": col,
                "pii_type": pii_type,
                "match_count": match_count,
                "sample_masked": _mask_value(sample, pii_type),
            }
        )
    return findings


@tool
@tool_error_boundary(source="tools.pii_tools")
def detect_pii(data_path: str, columns: list[str] | None = None) -> dict[str, Any]:
    """Detect PII columns (emails / SSNs / phones / cards / IPs / names) in a CSV (DQ-GOV-006).

    Uses stdlib-regex + column-name heuristics (no heavy deps). For each PII column
    it reports the ``pii_type``, the ``match_count`` (non-null cells matching), and a
    privacy-preserving ``sample_masked`` — never the raw value.

    Args:
        data_path: Filesystem path to a CSV file.
        columns: Optional subset of columns to scan (default: all columns).

    Returns:
        ``{"has_pii": bool, "pii_columns": [...], "findings": [{column, pii_type,
        match_count, sample_masked}], "summary": str}`` or
        ``{"success": False, "error": ...}`` on failure.
    """
    # Path-traversal containment (Security review, CWE-22): agent-supplied
    # data_path must stay inside the sandbox roots / CWD / tempdir.
    from .pathref import resolve_dataset_path

    try:
        p = resolve_dataset_path(data_path)
    except PermissionError as e:
        return {"success": False, "error": str(e)}
    if not p.exists():
        return {"success": False, "error": f"File not found: {data_path}"}
    try:
        df = pd.read_csv(p)
    except Exception as e:
        return {"success": False, "error": f"Failed to read CSV: {e}"}

    findings = _scan_frame(df, columns)
    pii_columns = [f["column"] for f in findings]
    has_pii = bool(pii_columns)
    if has_pii:
        types = sorted({f["pii_type"] for f in findings})
        summary = (
            f"Detected PII in {len(pii_columns)} column(s) "
            f"({', '.join(pii_columns)}); types: {', '.join(types)}."
        )
    else:
        summary = "No PII detected — dataset appears free of regulated identifiers."
    return {
        "has_pii": has_pii,
        "pii_columns": pii_columns,
        "findings": findings,
        "summary": summary,
    }


@tool
@tool_error_boundary(source="tools.pii_tools")
def redact_pii(data_path: str, columns: list[str] | None = None) -> dict[str, Any]:
    """Mask/redact PII columns and write a de-identified CSV artifact (DQ-GOV-007).

    When ``columns`` is omitted, the PII columns are auto-discovered via
    :func:`detect_pii`. Each targeted column's values are replaced with a masked
    token so no raw PII survives; benign columns are preserved untouched. The
    de-identified frame is written next to the input as ``<stem>.redacted.csv``.

    Args:
        data_path: Filesystem path to the source CSV.
        columns: Optional explicit columns to redact (default: auto-detect PII).

    Returns:
        ``{"redacted_path": str, "redacted_columns": [...], "summary": str}`` or
        ``{"success": False, "error": ...}``.
    """
    # Path-traversal containment (Security review, CWE-22): agent-supplied
    # data_path must stay inside the sandbox roots / CWD / tempdir.
    from .pathref import resolve_dataset_path

    try:
        p = resolve_dataset_path(data_path)
    except PermissionError as e:
        return {"success": False, "error": str(e)}
    if not p.exists():
        return {"success": False, "error": f"File not found: {data_path}"}
    try:
        df = pd.read_csv(p)
    except Exception as e:
        return {"success": False, "error": f"Failed to read CSV: {e}"}

    if columns:
        target_types = {c: "explicit" for c in columns if c in df.columns}
    else:
        target_types = {f["column"]: f["pii_type"] for f in _scan_frame(df, None)}

    out = df.copy()
    redacted_columns: list[str] = []
    for col, pii_type in target_types.items():
        ptype = pii_type if pii_type != "explicit" else "name"
        # Bind ``ptype`` as a default arg so the lambda captures THIS column's type
        # (not the last loop value) — B023 closure-safety.
        out[col] = out[col].apply(
            lambda x, ptype=ptype: _mask_value(x, ptype) if pd.notna(x) else x
        )
        redacted_columns.append(col)

    out_path = p.with_name(f"{p.stem}.redacted.csv")
    try:
        out.to_csv(out_path, index=False)
    except Exception as e:
        return {"success": False, "error": f"failed to write redacted CSV: {e}"}

    summary = (
        f"Redacted {len(redacted_columns)} PII column(s): "
        f"{', '.join(redacted_columns)}."
        if redacted_columns
        else "No PII columns to redact — de-identified copy is identical."
    )
    return {
        "redacted_path": str(out_path),
        "redacted_columns": redacted_columns,
        "summary": summary,
    }


@tool
@tool_error_boundary(source="tools.pii_tools")
def pii_compliance_gate(data_path: str, columns: list[str] | None = None) -> dict[str, Any]:
    """PII compliance gate: BLOCK sharing when un-redacted PII remains (DQ-GOV-008).

    Runs :func:`detect_pii` and returns a guardrail-shaped verdict
    (``{"action","passed","violations"}`` — mirroring ``run_guardrail_check``): a
    dataset with any detected PII fails the gate (``action="BLOCK"``); a clean or
    already-redacted dataset passes (``action="ALLOW"``). This is the policy gate a
    governance officer runs before a dataset leaves de-identified.

    Args:
        data_path: Filesystem path to the CSV to gate.
        columns: Optional subset of columns to check (default: all columns).

    Returns:
        ``{"action": "ALLOW"|"BLOCK", "passed": bool, "violations": [...],
        "pii_columns": [...], "summary": str}`` or ``{"success": False, "error": ...}``.
    """
    # Ground the gate in the SAME detection core ``detect_pii`` uses (``_scan_frame``)
    # rather than re-invoking the ``@tool`` wrapper — mirrors how ``synthesize_tabular``
    # reuses ``synthesize_tabular_frame``.
    # Path-traversal containment (Security review, CWE-22): agent-supplied
    # data_path must stay inside the sandbox roots / CWD / tempdir.
    from .pathref import resolve_dataset_path

    try:
        p = resolve_dataset_path(data_path)
    except PermissionError as e:
        return {"success": False, "error": str(e)}
    if not p.exists():
        return {"success": False, "error": f"File not found: {data_path}"}
    try:
        df = pd.read_csv(p)
    except Exception as e:
        return {"success": False, "error": f"Failed to read CSV: {e}"}

    findings = _scan_frame(df, columns)
    pii_columns = [f["column"] for f in findings]
    has_pii = bool(pii_columns)

    if has_pii:
        violations = [
            {
                "column": f["column"],
                "pii_type": f["pii_type"],
                "match_count": f["match_count"],
                "reason": f"un-redacted {f['pii_type']} PII present in column '{f['column']}'",
            }
            for f in findings
        ]
        return {
            "action": "BLOCK",
            "passed": False,
            "violations": violations,
            "pii_columns": pii_columns,
            "summary": (
                f"PII compliance gate FAILED — {len(pii_columns)} column(s) carry "
                f"un-redacted PII ({', '.join(pii_columns)}); sharing blocked."
            ),
        }
    return {
        "action": "ALLOW",
        "passed": True,
        "violations": [],
        "pii_columns": [],
        "summary": "PII compliance gate PASSED — no un-redacted PII; safe to share.",
    }
