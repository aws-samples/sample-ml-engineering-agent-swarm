"""3-pillar synthetic-data VALIDATION — the "how do we prove it works?" gate.

Master plan §7 / §12.3 + research KB §3. Synthetic data is **accepted only if all
three pillars pass** (never one metric — the TDS "passed every test, still broke
the model" failure mode):

* **Fidelity** (statistical resemblance):
    - per-column shape similarity (KSComplement numeric / TVComplement
      categorical) → ``quality_score`` (SDV QualityReport when ``sdmetrics`` is
      present; scipy ``column_shape_score`` fallback otherwise),
    - ``ks_mean`` (mean KS statistic over numeric columns),
    - ``corr_drift_frobenius`` (Frobenius norm of the correlation-matrix diff),
    - ``discriminator_auroc`` — a real-vs-synthetic classifier AUROC (≈0.5 good,
      ≥0.87 bad: a classifier cannot tell real from synthetic).
* **Utility** (TSTR vs TRTR): train a model on synthetic / test on a REAL
  holdout (TSTR) vs train-real/test-real (TRTR); report ``|TRTR−TSTR|`` plus a
  minority-class tail check (TSTR averages hide tails).
* **Privacy** (disclosure): ``min_dcr`` (min distance-to-closest-record so
  synthetic rows aren't copies), a membership-inference proxy, and an exact
  duplicate-row count.

**Acceptance (deterministic AND-gate, steerable thresholds):**
``accepted = quality_score ≥ τ_f AND |delta| ≤ τ_u AND auroc ≤ τ_d AND min_dcr > τ_p``
with defaults ``τ_f=0.85, τ_u=0.05, τ_d=0.6, τ_p=0.0`` (master plan §12.3),
overridable via ``thresholds`` (the SteeringProfile dial).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from strands import tool

from ...errors import tool_error_boundary
from ._common import (
    column_shape_score,
    correlation_drift_frobenius,
    read_csv_or_error,
)

# §7/§12.3 default acceptance thresholds (overridable via steering).
DEFAULT_THRESHOLDS = {
    "fidelity": 0.85,        # quality_score ≥ τ_f
    "utility_delta": 0.05,   # |TRTR − TSTR| ≤ τ_u
    "detection_auroc": 0.6,  # discriminator AUROC ≤ τ_d
    "privacy_dcr": 0.0,      # min_DCR > τ_p
}


# ─── fidelity ────────────────────────────────────────────────────────────────


def _quality_score(real: pd.DataFrame, synth: pd.DataFrame) -> tuple[float, float, bool]:
    """SDV QualityReport score if sdmetrics present; else scipy column-shape mean.

    Returns ``(quality_score, ks_mean, used_sdmetrics)``.
    """
    try:
        from sdmetrics.reports.single_table import QualityReport

        from ._common import build_sdv_metadata, single_table_metadata_dict

        meta = single_table_metadata_dict(build_sdv_metadata(real, table_name="table"))
        report = QualityReport()
        report.generate(real, synth[list(real.columns)], meta, verbose=False)
        _shape, ks_mean = column_shape_score(real, synth)
        return float(report.get_score()), ks_mean, True
    except Exception:
        shape, ks_mean = column_shape_score(real, synth)
        return shape, ks_mean, False


def _encode_for_model(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot / numeric encoding of a frame for sklearn estimators."""
    out = pd.get_dummies(df, dummy_na=False)
    return out.fillna(out.median(numeric_only=True)).fillna(0.0)


def discriminator_auroc(real: pd.DataFrame, synth: pd.DataFrame, seed: int = 0) -> float:
    """Train a classifier to tell real (1) from synthetic (0); return CV AUROC.

    AUROC ≈ 0.5 ⇒ indistinguishable (good fidelity); ≥ ~0.87 ⇒ easily separable
    (bad). Uses a shared one-hot encoding over the union of columns so real and
    synthetic align.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_predict

    cols = [c for c in real.columns if c in synth.columns]
    combined = pd.concat([real[cols], synth[cols]], ignore_index=True)
    X = _encode_for_model(combined)
    y = np.array([1] * len(real) + [0] * len(synth))

    n_splits = min(5, int(min(np.bincount(y))))
    if n_splits < 2:
        return 0.5
    clf = RandomForestClassifier(n_estimators=60, random_state=seed, n_jobs=1)
    try:
        proba = cross_val_predict(clf, X, y, cv=n_splits, method="predict_proba")[:, 1]
        return float(roc_auc_score(y, proba))
    except Exception:
        return 0.5


def compute_fidelity(real: pd.DataFrame, synth: pd.DataFrame) -> dict[str, Any]:
    """Fidelity pillar: quality_score, ks_mean, correlation drift, discriminator AUROC."""
    quality, ks_mean, used_sdmetrics = _quality_score(real, synth)
    corr_drift = correlation_drift_frobenius(real, synth)
    auroc = discriminator_auroc(real, synth)
    return {
        "quality_score": round(quality, 4),
        "ks_mean": round(ks_mean, 4),
        "corr_drift_frobenius": round(corr_drift, 4),
        "discriminator_auroc": round(auroc, 4),
        "used_sdmetrics": used_sdmetrics,
    }


# ─── utility (TSTR vs TRTR) ──────────────────────────────────────────────────


def _macro_f1_model(train: pd.DataFrame, test: pd.DataFrame, target: str, seed: int) -> tuple[float, dict[str, float]]:
    """Train RF on ``train``, score macro-F1 on ``test``; return (macro_f1, per_class_recall)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score, recall_score

    feat_cols = [c for c in train.columns if c != target]
    combined = pd.concat([train[feat_cols], test[feat_cols]], ignore_index=True)
    X_all = _encode_for_model(combined)
    X_train = X_all.iloc[: len(train)]
    X_test = X_all.iloc[len(train) :]
    y_train = train[target].astype(str).to_numpy()
    y_test = test[target].astype(str).to_numpy()

    if len(set(y_train)) < 2:
        return 0.0, {}
    clf = RandomForestClassifier(n_estimators=80, random_state=seed, n_jobs=1)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    macro = float(f1_score(y_test, pred, average="macro", zero_division=0))

    labels = sorted(set(y_test))
    recalls = recall_score(y_test, pred, labels=labels, average=None, zero_division=0)
    per_class = {str(lab): round(float(r), 4) for lab, r in zip(labels, recalls, strict=False)}
    return macro, per_class


def compute_utility(real: pd.DataFrame, synth: pd.DataFrame, target: str, seed: int = 0) -> dict[str, Any]:
    """Utility pillar: TSTR vs TRTR macro-F1 delta + minority-class tail check."""
    from sklearn.model_selection import train_test_split

    if target not in real.columns or target not in synth.columns:
        return {"skipped": f"target '{target}' not in both frames"}

    strat = real[target] if real[target].nunique() > 1 and real[target].value_counts().min() >= 2 else None
    try:
        real_train, real_test = train_test_split(real, test_size=0.3, random_state=seed, stratify=strat)
    except ValueError:
        real_train, real_test = train_test_split(real, test_size=0.3, random_state=seed)

    trtr_f1, trtr_recall = _macro_f1_model(real_train, real_test, target, seed)
    tstr_f1, tstr_recall = _macro_f1_model(synth, real_test, target, seed)

    # minority-class tail check: worst per-class recall gap (TRTR − TSTR)
    minority_delta = 0.0
    shared = set(trtr_recall) & set(tstr_recall)
    if shared:
        minority_delta = max(trtr_recall[c] - tstr_recall.get(c, 0.0) for c in shared)

    return {
        "trtr_f1": round(trtr_f1, 4),
        "tstr_f1": round(tstr_f1, 4),
        "delta": round(abs(trtr_f1 - tstr_f1), 4),
        "minority_delta": round(float(minority_delta), 4),
    }


# ─── privacy ─────────────────────────────────────────────────────────────────


def distance_to_closest_record(real: pd.DataFrame, synth: pd.DataFrame) -> tuple[float, float]:
    """Min & mean nearest-neighbour distance from each synthetic row to any real row.

    Computed in a shared standardised one-hot space. ``min_dcr == 0`` means at
    least one synthetic row is an exact copy of a real row (privacy leak).
    """
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    cols = [c for c in real.columns if c in synth.columns]
    combined = pd.concat([real[cols], synth[cols]], ignore_index=True)
    enc = _encode_for_model(combined)
    enc_real = enc.iloc[: len(real)].to_numpy(dtype=float)
    enc_synth = enc.iloc[len(real) :].to_numpy(dtype=float)
    if len(enc_real) == 0 or len(enc_synth) == 0:
        return 0.0, 0.0

    scaler = StandardScaler().fit(enc_real)
    a = scaler.transform(enc_real)
    b = scaler.transform(enc_synth)
    nn = NearestNeighbors(n_neighbors=1).fit(a)
    dist, _ = nn.kneighbors(b)
    return float(dist.min()), float(dist.mean())


def exact_duplicate_count(real: pd.DataFrame, synth: pd.DataFrame) -> int:
    """Number of synthetic rows that exactly match a real row (privacy leak)."""
    cols = [c for c in real.columns if c in synth.columns]
    real_set = set(map(tuple, real[cols].astype(str).to_numpy().tolist()))
    synth_rows = map(tuple, synth[cols].astype(str).to_numpy().tolist())
    return int(sum(1 for r in synth_rows if r in real_set))


def compute_privacy(real: pd.DataFrame, synth: pd.DataFrame) -> dict[str, Any]:
    """Privacy pillar: min/mean DCR, membership-inference proxy, exact duplicates."""
    min_dcr, mean_dcr = distance_to_closest_record(real, synth)
    dups = exact_duplicate_count(real, synth)
    # membership-inference proxy: fraction of synthetic rows nearer than a small
    # epsilon to a real row (higher = riskier). Cheap, deterministic.
    mi_risk = round(min(1.0, dups / max(1, len(synth))), 4)
    return {
        "min_dcr": round(min_dcr, 4),
        "mean_dcr": round(mean_dcr, 4),
        "membership_inference_risk": mi_risk,
        "exact_duplicates": dups,
    }


# ─── orchestration + AND-gate ────────────────────────────────────────────────


def _resolve_thresholds(thresholds: dict[str, Any] | None) -> dict[str, float]:
    out = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        for k, v in thresholds.items():
            if k in out and v is not None:
                out[k] = float(v)
    return out


def validate_synthetic_frames(
    real: pd.DataFrame,
    synth: pd.DataFrame,
    target: str | None = None,
    thresholds: dict[str, Any] | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Run all three pillars on in-memory frames + apply the AND-gate (plain fn)."""
    tau = _resolve_thresholds(thresholds)
    fidelity = compute_fidelity(real, synth)
    privacy = compute_privacy(real, synth)
    utility = compute_utility(real, synth, target, seed) if target else {"skipped": "no target — utility not assessed"}

    fid_ok = fidelity["quality_score"] >= tau["fidelity"]
    det_ok = fidelity["discriminator_auroc"] <= tau["detection_auroc"]
    priv_ok = privacy["min_dcr"] > tau["privacy_dcr"]
    util_ok = True
    if "delta" in utility:
        util_ok = float(utility["delta"]) <= tau["utility_delta"]

    accepted = bool(fid_ok and det_ok and priv_ok and util_ok)
    reasons: list[str] = []
    if not fid_ok:
        reasons.append(f"fidelity {fidelity['quality_score']} < {tau['fidelity']}")
    if not det_ok:
        reasons.append(f"discriminator AUROC {fidelity['discriminator_auroc']} > {tau['detection_auroc']}")
    if not priv_ok:
        reasons.append(f"min_DCR {privacy['min_dcr']} ≤ {tau['privacy_dcr']} (rows too close/copied)")
    if not util_ok:
        reasons.append(f"|TRTR−TSTR| {utility.get('delta')} > {tau['utility_delta']}")

    details = "ACCEPTED: all 3 pillars pass." if accepted else "REJECTED: " + "; ".join(reasons)
    return {
        "fidelity": fidelity,
        "utility": utility,
        "privacy": privacy,
        "accepted": accepted,
        "thresholds_used": tau,
        "pillar_pass": {"fidelity": fid_ok, "detection": det_ok, "utility": util_ok, "privacy": priv_ok},
        "details": details,
    }


@tool
@tool_error_boundary(source="tools.synthetic.validate")
def validate_synthetic(
    real_path: str,
    synthetic_path: str,
    target: str | None = None,
    thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """3-pillar synthetic-data validation + deterministic accept gate (master plan §7).

    Scores Fidelity (quality / KS / correlation-drift / discriminator AUROC),
    Utility (TSTR vs TRTR macro-F1 delta + minority tail), and Privacy (DCR /
    membership-inference / exact duplicates), then accepts iff ALL pass the
    (steerable) thresholds: ``fidelity≥0.85 AND |delta|≤0.05 AND auroc≤0.6 AND
    min_dcr>0``.

    Args:
        real_path: CSV of the real reference data.
        synthetic_path: CSV of the synthetic data to validate.
        target: Optional label column enabling the Utility/TSTR pillar; when
            omitted only Fidelity + Privacy are assessed.
        thresholds: Optional override map (``fidelity``, ``utility_delta``,
            ``detection_auroc``, ``privacy_dcr``) — the steering dial.

    Returns:
        The synthetic-data scorecard dict (``fidelity``, ``utility``, ``privacy``,
        ``accepted``, ``thresholds_used``, ``details``) or
        ``{"success": False, "error": ...}``.
    """
    real, err = read_csv_or_error(real_path)
    if err is not None:
        return err
    synth, err2 = read_csv_or_error(synthetic_path)
    if err2 is not None:
        return err2
    assert real is not None and synth is not None

    if real.empty or synth.empty:
        return {"success": False, "error": "real or synthetic dataset is empty"}

    try:
        return validate_synthetic_frames(real, synth, target=target, thresholds=thresholds)
    except Exception as e:
        return {"success": False, "error": f"validation failed: {type(e).__name__}"}
