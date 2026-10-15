"""Synthetic Data Generation / Augmentation Engine (Capability Group C — Phase 8).

Implements the master-plan §7 synthetic engine:

  * :mod:`generate`     — SDV tabular synthesis (``synthesize_tabular``:
    GaussianCopula / CTGAN / TVAE) + SMOTE-family imbalance augmentation
    (``augment_imbalance``: SMOTE / SMOTENC / ADASYN / RandomOverSampler).
  * :mod:`requirement`  — the requirement→spec flow (``build_synthesis_spec``):
    natural-language / structured requirement (+ optional reference schema) →
    a validated ``SynthesisSpec`` (columns, dtypes, distributions, constraints,
    conditions, volume, generator). LLM reasoning is an injectable callable
    (mock in tests); a deterministic schema-inference path works with no LLM
    when a reference dataset is provided.
  * :mod:`llm_rows`     — LLM row synthesis (``llm_synthesize_rows``): generate
    rows via an injectable LLM callable with schema/constraint validation, and a
    classical-generator fallback when no LLM is available.
  * :mod:`validate`     — the 3-pillar synthetic-data validation
    (``validate_synthetic``): Fidelity (KS / correlation-drift / discriminator
    AUROC), Utility (TSTR vs TRTR), Privacy (DCR / membership-inference / exact
    duplicates), plus the deterministic AND-gate accept logic + scorecard.

Conventions (match ``tools/imputation/``):
  - Tools are ``@tool``-decorated, path-based, return a JSON-serializable dict,
    and fail soft with ``{"success": False, "error": "..."}``.
  - Optional heavy deps (``sdv``/``sdmetrics``/``imblearn``) are imported lazily
    inside the function so the package still imports without the ``data`` extra.
"""

from __future__ import annotations

from ._common import (
    AUGMENT_STRATEGIES,
    SDV_GENERATORS,
    canonical_generator,
)
from .generate import augment_imbalance, synthesize_tabular
from .llm_rows import llm_synthesize_rows
from .requirement import build_synthesis_spec
from .steering import (
    apply_value_constraints,
    resolve_generator,
    synthesis_thresholds,
    write_synthesis_report,
)
from .validate import validate_synthetic

__all__ = [
    "AUGMENT_STRATEGIES",
    "SDV_GENERATORS",
    "apply_value_constraints",
    "augment_imbalance",
    "build_synthesis_spec",
    "canonical_generator",
    "llm_synthesize_rows",
    "resolve_generator",
    "synthesis_thresholds",
    "synthesize_tabular",
    "validate_synthetic",
    "write_synthesis_report",
]
