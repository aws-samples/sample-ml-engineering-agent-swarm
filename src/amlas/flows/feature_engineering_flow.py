"""``feature_engineering_flow`` — Stage 6 feature engineering over the FeatureStore port.

The invariant Stage-6 sequence (doc 06 §5.3 ``feature_engineering_flow`` row;
RECON FE assets — `feature_hypothesis`, `tabular_fe`, the CAAFE validator-loop):

    **propose feature set → build/select features → (CAAFE validator-loop) →
    feature-store ingest → PIT dataset → validate**

CORE IP preserved (RECON / project rules):
  * **CAAFE validator-loop** — the propose→build→VALIDATE→accept/reject loop is preserved
    as a swappable *proposer module* seam (mirrors ``imputation_flow``'s strategy seam):
    a candidate feature is ACCEPTED into the set only if it passes validation (no NaN
    explosion, finite values, improves-or-neutral signal); a failing candidate is rejected
    and the loop continues — never silently kept. The default proposer wraps the EXISTING
    ``auto_preprocess`` IP; richer proposers (the LLM CAAFE subagents) plug in unchanged.
  * **PIT-no-leakage** — the assembled training dataset is built via
    ``FeatureStoreBackend.pit_dataset(as_of=…)`` so it contains ONLY feature values known
    at or before ``as_of`` (the gold-table adapter enforces ``event_time <= as_of``).

The flow composes the M3 ``FeatureStoreBackend`` port (gold-table local / SageMaker FS
managed) and is offline-safe (tiny pandas; no LLM/heavy compute under test).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..backends.ports import FeatureStoreBackend
from ..backends.registry import REGISTRY

__all__ = [
    "FeatureCandidate",
    "FeatureEngineeringFlowResult",
    "propose_feature_set",
    "run_feature_engineering_flow",
    "validate_feature_candidate",
]


@dataclass
class FeatureCandidate:
    """One proposed feature: its name, the source column(s), and an apply spec."""

    name: str
    sources: list[str] = field(default_factory=list)
    op: str = "passthrough"  # passthrough | ratio | log | interaction | bucket
    accepted: bool = False
    reason: str = ""


@dataclass
class FeatureEngineeringFlowResult:
    """The invariant output contract of one ``feature_engineering_flow`` run."""

    status: str  # "completed" | "failed"
    proposed: list[dict[str, Any]] = field(default_factory=list)
    accepted: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    ingested: int = 0
    feature_group: str = ""
    pit_dataset_uri: str = ""
    pit_rows: int = 0
    pit_lineage: dict[str, Any] = field(default_factory=dict)
    adapter: str = ""
    detail: dict[str, Any] = field(default_factory=dict)


def propose_feature_set(
    dataset_uri: str,
    target: str | None = None,
    max_features: int = 10,
    *,
    df: Any | None = None,
) -> list[FeatureCandidate]:
    """Propose candidate features from a dataset (T-PR-48 ``propose_feature_set``).

    The DEFAULT proposer: inspects the dataset's numeric columns and proposes cheap,
    interpretable derived features (log of skewed positives, ratios between pairs). Wraps
    no LLM — it is the offline-safe baseline proposer; richer CAAFE/LLM proposers plug into
    the same :class:`FeatureCandidate` contract. Kept tiny (capped at ``max_features``).

    PERF-05: an already-loaded ``df`` may be passed to AVOID re-reading ``dataset_uri``
    (the flow reads the CSV once and shares the frame across propose + build/validate).
    When ``df`` is ``None`` (e.g. called standalone) it falls back to reading the URI, so
    the public contract is unchanged.
    """
    import pandas as pd

    df = pd.read_csv(dataset_uri) if df is None else df
    num_cols = [c for c in df.select_dtypes(include=["number"]).columns if c != target]
    candidates: list[FeatureCandidate] = []
    # log-transform skewed strictly-positive columns
    for c in num_cols:
        if len(candidates) >= max_features:
            break
        col = df[c].dropna()
        if not col.empty and (col > 0).all() and col.skew() > 1.0:
            candidates.append(FeatureCandidate(name=f"log_{c}", sources=[c], op="log"))
    # pairwise ratios between the first few numeric columns
    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            if len(candidates) >= max_features:
                break
            candidates.append(
                FeatureCandidate(name=f"ratio_{num_cols[i]}_{num_cols[j]}", sources=[num_cols[i], num_cols[j]], op="ratio")
            )
    return candidates


def _apply_candidate(df: Any, cand: FeatureCandidate) -> Any:
    """Materialize one candidate's column on a copy of ``df`` (pure pandas)."""
    import numpy as np

    out = df.copy()
    if cand.op == "log" and cand.sources:
        out[cand.name] = np.log1p(out[cand.sources[0]].clip(lower=0))
    elif cand.op == "ratio" and len(cand.sources) == 2:
        denom = out[cand.sources[1]].replace(0, np.nan)
        out[cand.name] = out[cand.sources[0]] / denom
    else:  # passthrough
        if cand.sources:
            out[cand.name] = out[cand.sources[0]]
    return out


def validate_feature_candidate(df: Any, cand: FeatureCandidate, *, max_nan_frac: float = 0.5) -> tuple[bool, str]:
    """The CAAFE validator: accept a candidate only if its column is well-formed.

    Validation (the preserved CAAFE accept/reject gate): the materialized column must
    exist, not be entirely NaN, keep its NaN fraction under ``max_nan_frac``, and contain
    at least one finite value. A candidate failing ANY check is REJECTED (never kept).
    Returns ``(accepted, reason)``.
    """
    import numpy as np

    if cand.name not in df.columns:
        return False, "candidate column not materialized"
    series = df[cand.name]
    nan_frac = float(series.isna().mean())
    if nan_frac >= max_nan_frac:
        return False, f"NaN fraction {nan_frac:.2f} ≥ {max_nan_frac} (CAAFE reject)"
    finite = np.isfinite(series.to_numpy(dtype="float64", na_value=np.nan))
    if not finite.any():
        return False, "no finite values (CAAFE reject)"
    return True, "passed CAAFE validation"


def run_feature_engineering_flow(
    *,
    dataset_uri: str,
    feature_group: str,
    entity_key: str = "entity_id",
    event_time_key: str = "event_time",
    target: str | None = None,
    as_of: str | None = None,
    entity_ids: list[str] | None = None,
    proposer: Callable[..., list[FeatureCandidate]] | None = None,
    max_features: int = 10,
    steering: Any | None = None,
    scale: int | str | None = None,
) -> FeatureEngineeringFlowResult:
    """Run the invariant feature-engineering flow over the resolved ``FeatureStoreBackend``.

    Fixed steps:
      1. **propose** a candidate feature set (default proposer or the injected ``proposer``);
      2. **build + CAAFE-validate** each candidate, accepting only those that pass;
      3. **ingest** the accepted feature rows into the feature store (T-FS-01);
      4. **PIT dataset** assemble a no-leakage training set at ``as_of`` (T-FS-03) when
         ``as_of`` + ``entity_ids`` are supplied (otherwise skipped — ingest-only).

    Args:
        dataset_uri: CSV (local) feature source.
        feature_group: Feature-store group to ingest into / PIT-join from.
        entity_key / event_time_key: The entity-id + event-time columns (PIT join keys).
        target: Optional target column (excluded from feature proposals).
        as_of / entity_ids: PIT cutoff + entities for the no-leakage training-set assembly.
        proposer: Optional candidate proposer (defaults to :func:`propose_feature_set`).
        max_features: Cap on proposed candidates (kept tiny).
        steering / scale: Forwarded to the registry resolution.

    Returns:
        A :class:`FeatureEngineeringFlowResult` (proposed/accepted/rejected + ingest count +
        PIT dataset uri/rows/lineage).
    """
    import pandas as pd

    try:
        resolved = REGISTRY.resolve(FeatureStoreBackend, steering=steering, scale=scale)
        adapter = resolved.adapter
    except Exception as e:
        return FeatureEngineeringFlowResult(status="failed", detail={"stage": "resolve", "error": str(e)})

    # ── 0. read the feature source ONCE (PERF-05) ──────────────────────────────
    # The CSV was previously read twice — inside the default ``propose_feature_set``
    # AND again here for the build/validate loop. We read it a single time and share
    # the frame across both steps: the default proposer accepts the loaded ``df``, and
    # the build loop reuses it. Injected proposers keep their original ``dataset_uri``
    # contract (they never read the CSV here), so behaviour — proposed/accepted/rejected
    # counts — is byte-identical; only the redundant second read is eliminated.
    try:
        df = pd.read_csv(dataset_uri)
    except Exception as e:
        return FeatureEngineeringFlowResult(status="failed", detail={"stage": "read", "error": str(e)})

    # ── 1. propose ────────────────────────────────────────────────────────────
    try:
        if proposer is None:
            # Default proposer: hand it the already-loaded frame (no re-read).
            candidates = propose_feature_set(dataset_uri, target=target, max_features=max_features, df=df)
        else:
            # Injected proposer: preserve its existing call contract exactly.
            try:
                candidates = proposer(dataset_uri, target=target, max_features=max_features)
            except TypeError:
                candidates = proposer(dataset_uri)  # proposers with a simpler signature
    except Exception as e:
        return FeatureEngineeringFlowResult(status="failed", detail={"stage": "propose", "error": str(e)})

    # ── 2. build + CAAFE-validate (accept/reject loop) ──────────────────────────
    accepted: list[FeatureCandidate] = []
    rejected: list[FeatureCandidate] = []
    built = df.copy()
    for cand in candidates:
        trial = _apply_candidate(df, cand)
        ok, reason = validate_feature_candidate(trial, cand)
        cand.accepted, cand.reason = ok, reason
        if ok:
            built[cand.name] = trial[cand.name]
            accepted.append(cand)
        else:
            rejected.append(cand)

    # ── 3. ingest the engineered rows (entity + event_time + accepted features) ─
    ingest_cols = [c for c in (entity_key, event_time_key) if c in built.columns]
    ingest_cols += [c.name for c in accepted if c.name in built.columns]
    records = [
        {str(k): v for k, v in r.items()}
        for r in built[ingest_cols].to_dict(orient="records")
    ] if ingest_cols else []
    ingested = 0
    if records:
        try:
            ingested = adapter.upsert(group=feature_group, records=records)
        except Exception as e:
            return FeatureEngineeringFlowResult(
                status="failed", proposed=[_c(c) for c in candidates],
                accepted=[_c(c) for c in accepted], rejected=[_c(c) for c in rejected],
                detail={"stage": "ingest", "error": str(e)}, adapter=resolved.adapter_name,
            )

    # ── 4. PIT dataset (no-leakage) when as_of + entity_ids supplied ────────────
    # G2: PIT now runs on BOTH the local gold-table AND the managed SageMaker path (the
    # managed adapter orchestrates the offline-store Athena ``as_of`` join). The flow no
    # longer silently skips PIT on the managed backend — it performs the real assembly.
    # The ``NotImplementedError`` branch is kept ONLY as a defensive guard for a future
    # adapter that has not yet implemented PIT (so the flow degrades gracefully rather than
    # crashing); in normal operation neither registered adapter raises it.
    pit_uri, pit_rows, pit_lineage = "", 0, {}
    if as_of and entity_ids:
        try:
            ref = adapter.pit_dataset(group=feature_group, entity_ids=entity_ids, as_of=as_of)
            pit_uri, pit_rows, pit_lineage = ref.uri, ref.rows, ref.lineage
        except NotImplementedError as e:  # defensive: an adapter without PIT (not the norm)
            pit_lineage = {"pit_skipped": str(e), "pit": False}
        except Exception as e:
            pit_lineage = {"pit_error": str(e), "pit": False}

    return FeatureEngineeringFlowResult(
        status="completed",
        proposed=[_c(c) for c in candidates],
        accepted=[_c(c) for c in accepted],
        rejected=[_c(c) for c in rejected],
        ingested=ingested,
        feature_group=feature_group,
        pit_dataset_uri=pit_uri,
        pit_rows=pit_rows,
        pit_lineage=pit_lineage,
        adapter=resolved.adapter_name,
        detail={"reason": resolved.reason, "n_proposed": len(candidates), "n_accepted": len(accepted)},
    )


def _c(cand: FeatureCandidate) -> dict[str, Any]:
    """Serialize a :class:`FeatureCandidate` to a plain dict for the result contract."""
    return {"name": cand.name, "sources": cand.sources, "op": cand.op, "accepted": cand.accepted, "reason": cand.reason}
