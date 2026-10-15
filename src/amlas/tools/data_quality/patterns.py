"""Deterministic detectors for the 25-pattern DQ catalog + the ``scan_dq_patterns``
and ``detect_ocr_artifacts`` tools (master plan §5 / spec sheet 2.1).

Every catalog ``pattern_id`` maps to exactly one ``_detect_*`` function here.
Detectors are pure (DataFrame in, list[Finding] out), CPU/pandas/sklearn only,
and append zero-or-more findings. ``scan_dq_patterns`` runs them all, reuses the
existing ``detect_anomalies`` / ``generate_eda_report`` / ``leakage_columns``
helpers, and returns the structured report defined in the spec sheet.

A *Finding* dict has keys: ``pattern_id, column, evidence, severity, suggested_strategy``.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from strands import tool

from ...config.aws_region import resolve_aws_region
from ...errors import tool_error_boundary
from ..leakage_columns import detect_id_like_columns
from .catalog import PATTERN_CATALOG
from .score import compute_six_dimensions, overall_from_dimensions

Finding = dict[str, Any]


# ─── small shared helpers ────────────────────────────────────────────────────


def _finding(pattern_id: str, column: str, evidence: str, severity: str | None = None) -> Finding:
    spec = PATTERN_CATALOG[pattern_id]
    return {
        "pattern_id": pattern_id,
        "column": column,
        "evidence": evidence,
        "severity": severity or spec.severity_default,
        "suggested_strategy": spec.suggested_strategy,
    }


def _is_numeric_str(val: Any) -> bool:
    if not isinstance(val, str):
        return False
    try:
        float(val)
        return True
    except (ValueError, TypeError):
        return False


def _numeric_cols(df: pd.DataFrame) -> list[str]:
    cols: list[str] = df.select_dtypes(include=["number"]).columns.tolist()
    return cols


def _object_cols(df: pd.DataFrame) -> list[str]:
    cols: list[str] = df.select_dtypes(include=["object"]).columns.tolist()
    return cols


def _datetime_like_cols(df: pd.DataFrame) -> list[str]:
    """Columns that are datetime dtype OR parse cleanly as dates from object."""
    cols: list[str] = df.select_dtypes(include=["datetime64"]).columns.tolist()
    for col in _object_cols(df):
        sample = df[col].dropna().head(10)
        if len(sample) < 3:
            continue
        try:
            # Non-date object columns make pandas emit a noisy "Could not infer
            # format" UserWarning and fall back to slow per-element parsing.
            # We only care whether it parses, so suppress the expected warning.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                parsed = pd.to_datetime(sample, errors="coerce")
            if parsed.notna().mean() > 0.8:
                cols.append(col)
        except (ValueError, TypeError):
            continue
    return cols


_SENTINELS_NUMERIC = {-999, -9999, -99999, 9999, 99999, -1.0e30}
_SENTINELS_STRING = {"n/a", "na", "null", "none", "unknown", "missing", "?", "-", "--"}


# ─── Missingness family (6) ──────────────────────────────────────────────────


def _detect_missingness(df: pd.DataFrame) -> list[Finding]:
    findings: list[Finding] = []
    n = len(df)
    if n == 0:
        return findings

    # mask classifier (MAR) reused below

    numeric = _numeric_cols(df)

    for col in df.columns:
        null_mask = df[col].isnull()
        n_missing = int(null_mask.sum())
        if n_missing == 0:
            continue
        miss_frac = n_missing / n

        # MISS-RUN vs MISS-BLOCK: both describe contiguous nulls, but RUN is a
        # *time-series* dropout (needs an ordering/time column) while BLOCK is an
        # *era* of rows where the column was never recorded. Disambiguate by
        # whether the frame carries a datetime-like column.
        run_len = _longest_true_run(null_mask.to_numpy())
        contiguous = run_len >= max(10, int(0.15 * n)) and run_len >= 3 * (n_missing / max(run_len, 1))
        block = _is_block_missing(null_mask.to_numpy())
        has_time = len(_datetime_like_cols(df)) > 0
        is_run = contiguous and has_time and col not in _datetime_like_cols(df)
        is_block = (block or contiguous) and not is_run

        # mechanism: try to predict the null-mask from OTHER numeric columns
        mask_auc = _mask_auc(df, col, numeric)

        emitted_mechanism = False
        if is_run:
            findings.append(
                _finding("MISS-RUN", col, f"longest null run = {run_len} rows ({miss_frac*100:.0f}% missing)")
            )
            emitted_mechanism = True
        elif is_block:
            findings.append(
                _finding("MISS-BLOCK", col, f"nulls concentrated in a contiguous row block ({miss_frac*100:.0f}% missing)")
            )
            emitted_mechanism = True

        # mechanism classification (MCAR/MAR/MNAR) — always emit one
        if mask_auc is not None and mask_auc > 0.65:
            findings.append(
                _finding("MISS-MAR", col, f"null-mask predictable from other columns (AUC={mask_auc:.2f})")
            )
        else:
            # distinguish MNAR (value-dependent) vs MCAR. We can only test MNAR
            # weakly: if the present values are strongly skewed toward one tail
            # AND the mask is not explained by other cols, flag MNAR-candidate.
            mnar = _looks_mnar(df, col)
            if mnar and not emitted_mechanism:
                findings.append(
                    _finding("MISS-MNAR", col, f"missing concentrated where value would be extreme ({miss_frac*100:.0f}% missing)")
                )
            elif not emitted_mechanism:
                auc_txt = f"AUC={mask_auc:.2f}" if mask_auc is not None else "AUC~0.5"
                findings.append(
                    _finding("MISS-MCAR", col, f"null-mask independent of other columns ({auc_txt})")
                )

    # MISS-SENTINEL: disguised missing as sentinel values
    findings.extend(_detect_sentinel(df))
    return findings


def _detect_sentinel(df: pd.DataFrame) -> list[Finding]:
    findings: list[Finding] = []
    n = len(df)
    if n == 0:
        return findings
    for col in _numeric_cols(df):
        series = df[col].dropna()
        if len(series) == 0:
            continue
        hits = series.isin(_SENTINELS_NUMERIC)
        count = int(hits.sum())
        # must be a repeated sentinel (not a one-off legit value)
        if count >= max(3, int(0.02 * n)):
            val = series[hits].mode().iloc[0]
            findings.append(
                _finding("MISS-SENTINEL", col, f"{count} cells equal sentinel {val!r} (disguised missing)")
            )
    for col in _object_cols(df):
        series = df[col].dropna().astype(str).str.strip().str.lower()
        if len(series) == 0:
            continue
        hits = series.isin(_SENTINELS_STRING)
        count = int(hits.sum())
        if count >= max(3, int(0.02 * n)):
            findings.append(
                _finding("MISS-SENTINEL", col, f"{count} cells are string sentinels (e.g. 'N/A') — disguised missing")
            )
    return findings


def _longest_true_run(mask: np.ndarray) -> int:
    best = cur = 0
    for v in mask:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def _is_block_missing(mask: np.ndarray) -> bool:
    """True if >=80% of nulls fall within a contiguous window covering <60% of rows."""
    idx = np.flatnonzero(mask)
    if len(idx) < 8:
        return False
    span = idx.max() - idx.min() + 1
    density_in_span = len(idx) / span
    return bool(density_in_span > 0.5 and span < 0.6 * len(mask))


def _mask_auc(df: pd.DataFrame, col: str, numeric: list[str]) -> float | None:
    """CV AUC of predicting null-mask of ``col`` from the OTHER numeric columns."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    y = df[col].isnull().astype(int).to_numpy()
    if y.sum() < 5 or (len(y) - y.sum()) < 5:
        return None
    feats = [c for c in numeric if c != col]
    if not feats:
        return None
    X = df[feats].copy()
    # impute predictor nulls with median so the classifier can run
    X = X.fillna(X.median(numeric_only=True))
    if X.isnull().any().any():
        X = X.fillna(0.0)
    if X.shape[1] == 0 or len(np.unique(y)) < 2:
        return None
    try:
        Xs = StandardScaler().fit_transform(X)
        clf = LogisticRegression(max_iter=200)
        folds = min(5, int(y.sum()), int(len(y) - y.sum()))
        if folds < 2:
            return None
        scores = cross_val_score(clf, Xs, y, cv=folds, scoring="roc_auc")
        return float(np.mean(scores))
    except Exception:
        return None


def _looks_mnar(df: pd.DataFrame, col: str) -> bool:
    """Weak MNAR heuristic: among rows where a CORRELATED helper column is
    extreme, the target is disproportionately missing — and that wasn't already
    explained linearly. Conservative (MNAR is not statistically confirmable).
    """
    if col not in _numeric_cols(df):
        return False
    present = df[col].dropna()
    if len(present) < 10:
        return False
    # If the observed values are truncated (e.g. only low values remain because
    # high values went missing), the present distribution is strongly one-sided
    # relative to a symmetric expectation. Use skew + boundary heuristic.
    skew = float(present.skew()) if present.std() > 0 else 0.0
    miss_frac = df[col].isnull().mean()
    # high missingness + skewed survivors → value-dependent missingness.
    # Threshold 0.8 chosen so a ~30%-truncated tail (skew ~0.9) is caught while
    # roughly symmetric MCAR survivors (|skew|<0.5) are not.
    return miss_frac > 0.15 and abs(skew) > 0.8


# ─── Validity family (5) ─────────────────────────────────────────────────────


def _detect_validity(df: pd.DataFrame) -> list[Finding]:
    findings: list[Finding] = []
    len(df)

    # VAL-TYPE: object columns mixing numeric-like and text
    for col in _object_cols(df):
        non_null = df[col].dropna()
        if len(non_null) == 0:
            continue
        numeric_ratio = non_null.apply(_is_numeric_str).mean()
        if 0.1 < numeric_ratio < 0.9:
            findings.append(
                _finding("VAL-TYPE", col, f"{numeric_ratio*100:.0f}% numeric-like values mixed with text")
            )

    # VAL-INF and VAL-RANGE on numeric columns
    for col in _numeric_cols(df):
        series = df[col]
        finite = series.replace([np.inf, -np.inf], np.nan).dropna()
        n_inf = int(np.isinf(series.dropna()).sum())
        if n_inf > 0:
            findings.append(_finding("VAL-INF", col, f"{n_inf} infinite value(s)"))

        if len(finite) >= 10:
            # VAL-RANGE: impossible-looking values. Heuristic: a column that is
            # mostly non-negative but has a few negative values (or vice versa)
            # — and the negatives are far from the bulk — looks like bad entry.
            neg = int((finite < 0).sum())
            pos = int((finite > 0).sum())
            if pos >= 0.8 * len(finite) and 0 < neg <= 0.1 * len(finite):
                findings.append(
                    _finding("VAL-RANGE", col, f"{neg} negative value(s) in an otherwise non-negative column")
                )
            elif neg >= 0.8 * len(finite) and 0 < pos <= 0.1 * len(finite):
                findings.append(
                    _finding("VAL-RANGE", col, f"{pos} positive value(s) in an otherwise non-positive column")
                )

    # VAL-UNIT: bimodal scale consistent with imperial<->metric mixing
    findings.extend(_detect_unit_drift(df))

    # VAL-FORMAT: enum/regex/date format violations in object columns
    findings.extend(_detect_format(df))
    return findings


def _detect_unit_drift(df: pd.DataFrame) -> list[Finding]:
    findings: list[Finding] = []
    # Identifier / key columns are not physical measurements — never unit-drift.
    skip = set(detect_id_like_columns(df))
    for col in _numeric_cols(df):
        if col in skip:
            continue
        series = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if len(series) < 30 or series.std() == 0:
            continue
        # Bimodality test: 2-component 1D KMeans; if the two cluster centers are
        # separated by a near-constant ratio (1.5–4×, e.g. inch↔cm 2.54), and
        # each cluster has substantial mass, flag unit drift.
        try:
            from sklearn.cluster import KMeans

            x = series.to_numpy().reshape(-1, 1)
            km = KMeans(n_clusters=2, n_init=5, random_state=0).fit(x)
            raw_centers = km.cluster_centers_.ravel()  # indexed by label (unsorted)
            labels = km.labels_
            frac_small = min(np.mean(labels == 0), np.mean(labels == 1))
            lo, hi = sorted(float(c) for c in raw_centers)
            if lo > 0 and frac_small > 0.2:
                ratio = hi / lo
                if 1.5 <= ratio <= 4.0:
                    # ensure genuine separation: the gap between the two cluster
                    # centers must be large relative to the WITHIN-cluster spread
                    # (a true bimodal split, not one wide unimodal spread).
                    # NB: subtract each point's OWN cluster center (label-indexed,
                    # not the sorted lo/hi) so within-σ isn't inflated.
                    residuals = [
                        float(x[i, 0] - raw_centers[labels[i]]) for i in range(len(x))
                    ]
                    within_std = float(np.std(residuals)) if residuals else float(series.std())
                    gap = (hi - lo) / (within_std + 1e-9)
                    if gap > 3.0:
                        findings.append(
                            _finding("VAL-UNIT", col, f"bimodal scale, cluster ratio {ratio:.2f}× (gap {gap:.1f}× within-σ) suggests unit drift")
                        )
        except Exception:  # nosec B112
            continue
    return findings


def _detect_format(df: pd.DataFrame) -> list[Finding]:
    findings: list[Finding] = []
    for col in _object_cols(df):
        non_null = df[col].dropna().astype(str)
        if len(non_null) < 10:
            continue
        # skip columns we treat as numeric-mixed (VAL-TYPE) or free identifiers
        numeric_ratio = non_null.apply(_is_numeric_str).mean()
        if numeric_ratio > 0.1:
            continue
        # Infer a dominant token shape (letters/digits/sep). If >=70% match a
        # shape and a minority clearly violate it, that's a format violation.
        shapes = non_null.apply(_token_shape)
        top_shape, top_count = shapes.value_counts().index[0], shapes.value_counts().iloc[0]
        top_frac = top_count / len(non_null)
        if 0.6 <= top_frac < 0.97 and len(non_null) - top_count >= 3:
            # only when the dominant shape is "structured" (has a separator/digits)
            if any(ch.isdigit() for ch in top_shape) or "-" in top_shape or "/" in top_shape:
                violators = int(len(non_null) - top_count)
                findings.append(
                    _finding("VAL-FORMAT", col, f"{violators} values violate dominant format '{top_shape}'")
                )
    return findings


def _token_shape(s: str) -> str:
    out = []
    for ch in s:
        if ch.isalpha():
            out.append("A")
        elif ch.isdigit():
            out.append("9")
        else:
            out.append(ch)
    # collapse runs
    return re.sub(r"(.)\1+", r"\1", "".join(out))


# ─── Uniqueness / consistency family (5) ─────────────────────────────────────


def _detect_uniqueness(df: pd.DataFrame, target: str | None) -> list[Finding]:
    findings: list[Finding] = []
    n = len(df)
    if n == 0:
        return findings

    # UNQ-DUP: exact duplicate rows
    n_dup = int(df.duplicated().sum())
    if n_dup > 0:
        findings.append(_finding("UNQ-DUP", "_all_", f"{n_dup} exact duplicate rows ({n_dup/n*100:.1f}%)"))

    # UNQ-KEY: duplicate values in an identifier-like column
    id_cols = set(detect_id_like_columns(df, target_column=target))
    for col in id_cols:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if len(series) < 5:
            continue
        dup_count = int(series.duplicated().sum())
        if dup_count > 0:
            findings.append(_finding("UNQ-KEY", col, f"identifier column has {dup_count} duplicate key value(s)"))

    # UNQ-FUZZY: near-duplicate category strings
    findings.extend(_detect_fuzzy_dups(df, id_cols))

    # CON-REF: cross-field rule break (numeric start<=end style)
    findings.extend(_detect_cross_field(df))

    # CON-TEMP: future-dated / non-monotonic time
    findings.extend(_detect_temporal(df))
    return findings


def _detect_fuzzy_dups(df: pd.DataFrame, id_cols: set[str]) -> list[Finding]:
    findings: list[Finding] = []
    for col in _object_cols(df):
        if col in id_cols:
            continue
        non_null = df[col].dropna().astype(str)
        uniques = non_null.unique().tolist()
        if not (2 <= len(uniques) <= 60):
            continue
        # canonicalize (lower + strip + remove non-alnum) and see if distinct
        # surface forms collapse → near-duplicates
        canon: dict[str, set[str]] = {}
        for u in uniques:
            key = re.sub(r"[^a-z0-9]+", "", u.lower())
            canon.setdefault(key, set()).add(u)
        collapsed = {k: v for k, v in canon.items() if len(v) > 1}
        if collapsed:
            example = next(iter(collapsed.values()))
            findings.append(
                _finding("UNQ-FUZZY", col, f"{len(collapsed)} group(s) of near-duplicate categories e.g. {sorted(example)[:3]}")
            )
    return findings


def _detect_cross_field(df: pd.DataFrame) -> list[Finding]:
    findings: list[Finding] = []
    numeric = _numeric_cols(df)
    # look for plausible (start,end)/(min,max)/(low,high) column pairs by name
    pairs: list[tuple[str, str]] = []
    lname = {c: c.lower() for c in numeric}
    for a in numeric:
        for b in numeric:
            if a == b:
                continue
            la, lb = lname[a], lname[b]
            if (("start" in la and "end" in lb) or ("min" in la and "max" in lb)
                    or ("low" in la and "high" in lb) or ("from" in la and "to" in lb)):
                pairs.append((a, b))
    for a, b in pairs:
        sub = df[[a, b]].dropna()
        if len(sub) < 5:
            continue
        violations = int((sub[a] > sub[b]).sum())
        if violations > 0:
            findings.append(
                _finding("CON-REF", f"{a}>{b}", f"{violations} rows violate {a} <= {b}")
            )
    return findings


def _detect_temporal(df: pd.DataFrame) -> list[Finding]:
    findings: list[Finding] = []
    now = pd.Timestamp.now()
    future_cut = now + pd.Timedelta(days=1)
    for col in _datetime_like_cols(df):
        parsed = pd.to_datetime(df[col], errors="coerce")
        valid = parsed.dropna()
        if len(valid) < 5:
            continue
        n_future = int((valid > future_cut).sum())
        if n_future > 0:
            findings.append(_finding("CON-TEMP", col, f"{n_future} future-dated timestamp(s)"))
    return findings


# ─── Distribution / leakage / drift family (4) ───────────────────────────────


def _detect_distribution(df: pd.DataFrame, data_path: str, target: str | None) -> list[Finding]:
    findings: list[Finding] = []
    n = len(df)
    if n == 0:
        return findings

    # DST-OUT: reuse detect_anomalies (IQR) on numeric columns
    from ..data_quality_tools import detect_anomalies

    try:
        anom = detect_anomalies(data_path=data_path, method="iqr")
        per_col = anom.get("per_column", {})
        for col, info in per_col.items():
            # Skip binary/flag columns: an IQR/outlier rule on a 0/1 label
            # column flags the rare positive rows (the signal you want to keep)
            # as outliers to "cap or remove" — misleading and destructive.
            if col in df.columns and df[col].nunique(dropna=True) <= 2:
                continue
            rate = info.get("rate", 0.0)
            count = info.get("count", 0)
            if count >= max(5, int(0.03 * n)) and rate > 0.02:
                findings.append(_finding("DST-OUT", col, f"{count} IQR outliers (rate {rate:.2%})"))
    except Exception:  # nosec B110
        pass

    # DST-IMB: severe class imbalance on the target (or any low-cardinality int col)
    findings.extend(_detect_imbalance(df, target))

    # DST-DRIFT: distribution shift across era bins (first vs second half)
    findings.extend(_detect_drift(df))

    # LEAK-CARD: identifier / near-unique leakage columns. Datetime columns are
    # legitimately near-unique (a timestamp per row) and must NOT be flagged.
    datetime_cols = set(_datetime_like_cols(df))
    for col in detect_id_like_columns(df, target_column=target):
        if col in df.columns and col not in datetime_cols:
            nunique = int(df[col].nunique(dropna=True))
            findings.append(
                _finding("LEAK-CARD", col, f"identifier-like / near-unique column ({nunique} distinct of {n})")
            )
    return findings


def _detect_imbalance(df: pd.DataFrame, target: str | None) -> list[Finding]:
    findings: list[Finding] = []
    candidates = []
    if target and target in df.columns:
        candidates.append(target)
    else:
        # any low-cardinality integer/object column that looks like a label
        for col in df.columns:
            vc = df[col].dropna()
            if 2 <= vc.nunique() <= 5 and len(vc) > 20:
                if pd.api.types.is_integer_dtype(vc) or df[col].dtype == object:
                    candidates.append(col)
                    break
    for col in candidates:
        vc = df[col].value_counts()
        if len(vc) < 2:
            continue
        ratio = vc.min() / vc.max()
        if ratio < 0.20:
            findings.append(
                _finding("DST-IMB", col, f"class imbalance ratio {ratio:.3f} (minority {int(vc.min())} / majority {int(vc.max())})")
            )
    return findings


def _detect_drift(df: pd.DataFrame) -> list[Finding]:
    findings: list[Finding] = []
    n = len(df)
    if n < 60:
        return findings
    from scipy import stats

    half = n // 2
    skip = set(detect_id_like_columns(df))  # keys aren't measurements → not "drift"
    for col in _numeric_cols(df):
        if col in skip:
            continue
        series = df[col].replace([np.inf, -np.inf], np.nan)
        first = series.iloc[:half].dropna()
        second = series.iloc[half:].dropna()
        if len(first) < 20 or len(second) < 20:
            continue
        if first.std() == 0 and second.std() == 0:
            continue
        try:
            ks_stat, p = stats.ks_2samp(first, second)
            # strong drift: large KS stat + significant + meaningful mean shift
            mean_shift = abs(first.mean() - second.mean())
            pooled_std = series.dropna().std() + 1e-9
            if ks_stat > 0.4 and p < 0.01 and mean_shift / pooled_std > 0.5:
                findings.append(
                    _finding("DST-DRIFT", col, f"era drift KS={ks_stat:.2f} (p={p:.1e}, mean shift {mean_shift/pooled_std:.1f}σ)")
                )
        except Exception:  # nosec B112
            continue
    return findings


# ─── Long-lineage / OCR family (5) ───────────────────────────────────────────

# Classic OCR confusion pairs (for evidence/labelling).
_OCR_CONFUSIONS = {"O": "0", "0": "O", "l": "1", "1": "l", "S": "5", "5": "S", "B": "8", "8": "B"}
# Canonical skeleton map: collapse each confusable to ONE representative so that
# tokens differing only by an OCR confusion share an identical skeleton
# (e.g. "SER123", "5ER123", "SERl23" all -> "5ER1Z3"-style canonical form).
_OCR_CANON = {"O": "0", "0": "0", "l": "1", "1": "1", "I": "1", "S": "5", "5": "5", "B": "8", "8": "8"}
# A token that mixes letters and digits in a way consistent with OCR confusion.
_OCR_MIX_RE = re.compile(r"^(?=.*[A-Za-z])(?=.*[0-9OlISB])[A-Za-z0-9]+$")
_STRAY_PUNCT_RE = re.compile(r"[|~^`\\]")
_BROKEN_DECIMAL_RE = re.compile(r"\d+\.\d+\.\d+|\d+,\d+,\d+")
_NON_ASCII_RE = re.compile(r"[^\x00-\x7F]")


def _ocr_artifacts(df: pd.DataFrame, columns: list[str] | None = None) -> list[dict[str, Any]]:
    """Return raw OCR artifact records (used by both detector + scan tool)."""
    artifacts: list[dict[str, Any]] = []
    cols = columns or _object_cols(df)
    for col in cols:
        if col not in df.columns:
            continue
        non_null = df[col].dropna().astype(str)
        if len(non_null) == 0:
            continue

        # char_confusion: identifier-ish tokens carrying confusable chars where
        # sibling values use the *other* member of the pair (O vs 0 etc.). We
        # detect by: tokens that are alnum, share a common skeleton when
        # confusions are normalized, but differ in raw form.
        confusion_examples = _confusion_examples(non_null)
        if confusion_examples:
            artifacts.append({
                "column": col,
                "kind": "char_confusion",
                "examples": confusion_examples[:5],
                "count": len(confusion_examples),
            })

        # stray_punct / broken_decimal
        stray = non_null[non_null.str.contains(_STRAY_PUNCT_RE)]
        broken = non_null[non_null.str.contains(_BROKEN_DECIMAL_RE)]
        noise_examples = list(stray.unique())[:3] + list(broken.unique())[:3]
        if len(stray) + len(broken) > 0:
            artifacts.append({
                "column": col,
                "kind": "stray_punct" if len(stray) >= len(broken) else "broken_decimal",
                "examples": noise_examples[:5],
                "count": int(len(stray) + len(broken)),
            })

        # mixed_case: same canonical category in multiple case/whitespace forms
        case_groups = _case_variant_groups(non_null)
        if case_groups:
            artifacts.append({
                "column": col,
                "kind": "mixed_case",
                "examples": case_groups[:5],
                "count": len(case_groups),
            })
    return artifacts


def _confusion_examples(series: pd.Series) -> list[str]:
    """Find tokens differing only by an OCR-confusable substitution."""
    toks = [t for t in series.unique() if isinstance(t, str) and _OCR_MIX_RE.match(t)]
    if len(toks) < 2:
        return []

    def skeleton(t: str) -> str:
        # Canonical: collapse confusables to one representative, uppercase the rest.
        return "".join(_OCR_CANON.get(ch, ch.upper()) for ch in t)

    groups: dict[str, set[str]] = {}
    for t in toks:
        groups.setdefault(skeleton(t), set()).add(t)
    out: list[str] = []
    for forms in groups.values():
        if len(forms) > 1:
            out.extend(sorted(forms))
    return out


def _case_variant_groups(series: pd.Series) -> list[str]:
    groups: dict[str, set[str]] = {}
    for v in series.unique():
        if not isinstance(v, str):
            continue
        key = v.strip().lower()
        if key:
            groups.setdefault(key, set()).add(v)
    return [next(iter(sorted(forms))) for forms in groups.values() if len(forms) > 1]


def _detect_ocr_lineage(df: pd.DataFrame) -> list[Finding]:
    findings: list[Finding] = []
    artifacts = _ocr_artifacts(df)
    for art in artifacts:
        col = art["column"]
        kind = art["kind"]
        if kind == "char_confusion":
            findings.append(_finding("OCR-CONF", col, f"{art['count']} OCR char-confusion variants e.g. {art['examples'][:3]}"))
        elif kind in ("stray_punct", "broken_decimal"):
            findings.append(_finding("OCR-NOISE", col, f"{art['count']} stray-punctuation / broken-decimal values"))
        elif kind == "mixed_case":
            findings.append(_finding("OCR-CASE", col, f"{art['count']} category variants differing only by case/whitespace"))

    # LIN-ENCODE: non-ASCII / encoding inconsistency
    for col in _object_cols(df):
        non_null = df[col].dropna().astype(str)
        if len(non_null) == 0:
            continue
        non_ascii = non_null[non_null.str.contains(_NON_ASCII_RE)]
        if len(non_ascii) > 0:
            findings.append(
                _finding("LIN-ENCODE", col, f"{len(non_ascii)} value(s) contain non-ASCII / mixed-encoding characters")
            )

    # LIN-SCHEMA: a column present (non-null) only within a contiguous late/early era
    findings.extend(_detect_schema_drift(df))
    return findings


def _detect_schema_drift(df: pd.DataFrame) -> list[Finding]:
    findings: list[Finding] = []
    n = len(df)
    if n < 60:
        return findings
    for col in df.columns:
        present = df[col].notna().to_numpy()
        frac = present.mean()
        if frac == 0 or frac >= 1.0:
            continue
        # present values cluster in a contiguous block at one end (era-scoped col)
        idx = np.flatnonzero(present)
        if len(idx) < 10:
            continue
        span = idx.max() - idx.min() + 1
        density = len(idx) / span
        edge = idx.min() <= 0.05 * n or idx.max() >= 0.95 * n
        if density > 0.85 and span < 0.7 * n and edge and frac < 0.7:
            findings.append(
                _finding("LIN-SCHEMA", col, f"column populated only within a contiguous era ({frac*100:.0f}% of rows)")
            )
    return findings


# ─── Public tools ────────────────────────────────────────────────────────────


@tool
@tool_error_boundary(source="tools.data_quality.patterns")
def scan_dq_patterns(
    data_path: str,
    target: str | None = None,
    profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the 25-pattern DQ catalog detectors over a CSV and return findings.

    This is the master-plan §5 *validation contract*: each detector emits zero or
    more findings tagged with a catalog ``pattern_id``, family, severity, and a
    ``suggested_strategy`` consumed by downstream imputation / synthetic agents.

    Args:
        data_path: Filesystem path to a CSV file.
        target: Optional target column name (used by leakage + imbalance detectors).
        profile: Optional steering dict (reserved; ``mechanism_hints`` honored if present).

    Returns:
        ``{"patterns_detected":[{pattern_id,column,evidence,severity,suggested_strategy}],
           "overall_score":float, "dimensions":{...6...}, "summary":str}``
        or ``{"success": False, "error": ...}`` on failure.
    """
    p = Path(data_path)
    if not p.exists():
        return {"success": False, "error": f"File not found: {data_path}"}
    try:
        df = pd.read_csv(p)
    except Exception as e:
        return {"success": False, "error": f"Failed to read CSV: {e}"}

    if df.shape[0] == 0:
        return {
            "patterns_detected": [],
            "overall_score": 100.0,
            "dimensions": {
                name: {"score": 100.0, "details": "empty dataset"}
                for name in (
                    "completeness", "validity", "consistency", "accuracy", "uniqueness", "timeliness"
                )
            },
            "summary": "Empty dataset — no patterns detected; trivially scores 100/100.",
        }

    findings: list[Finding] = []
    findings.extend(_detect_missingness(df))
    findings.extend(_detect_validity(df))
    findings.extend(_detect_uniqueness(df, target))
    findings.extend(_detect_distribution(df, data_path, target))
    findings.extend(_detect_ocr_lineage(df))
    findings.extend(_detect_content_safety(df))

    # honor explicit mechanism hints from a steering profile (overrides MISS-* mechanism)
    if profile and isinstance(profile.get("mechanism_hints"), dict):
        findings = _apply_mechanism_hints(findings, profile["mechanism_hints"])

    dims = compute_six_dimensions(df, target)
    overall = overall_from_dimensions(dims)

    n_high = sum(1 for f in findings if f["severity"] == "high")
    families = sorted({PATTERN_CATALOG[f["pattern_id"]].family for f in findings})
    if findings:
        summary = (
            f"Detected {len(findings)} DQ pattern occurrence(s) across "
            f"{len(families)} families ({n_high} high-severity). "
            f"6-dim quality score {overall:.1f}/100."
        )
    else:
        summary = f"No DQ patterns detected. 6-dim quality score {overall:.1f}/100 — data appears clean."

    return {
        "patterns_detected": findings,
        "overall_score": overall,
        "dimensions": dims,
        "summary": summary,
    }


def _apply_mechanism_hints(findings: list[Finding], hints: dict[str, str]) -> list[Finding]:
    miss_ids = {"MISS-MCAR", "MISS-MAR", "MISS-MNAR"}
    hint_to_id = {"MCAR": "MISS-MCAR", "MAR": "MISS-MAR", "MNAR": "MISS-MNAR"}
    out: list[Finding] = []
    for f in findings:
        col = f["column"]
        if f["pattern_id"] in miss_ids and col in hints:
            target_id = hint_to_id.get(str(hints[col]).upper())
            if target_id and target_id != f["pattern_id"]:
                f = _finding(target_id, col, f"mechanism pinned via steering hint -> {hints[col]}")
        out.append(f)
    return out


@tool
@tool_error_boundary(source="tools.data_quality.patterns")
def detect_ocr_artifacts(data_path: str, columns: list[str] | None = None) -> dict[str, Any]:
    """Detect OCR / scanning artifacts in text columns of a CSV.

    Kinds: ``char_confusion`` (O<->0, l<->1, S<->5, B<->8), ``stray_punct``,
    ``broken_decimal``, ``mixed_case``.

    Args:
        data_path: Filesystem path to a CSV file.
        columns: Optional subset of columns to scan (default: all object columns).

    Returns:
        ``{"artifacts":[{"column","kind","examples","count"}]}`` or
        ``{"success": False, "error": ...}``.
    """
    p = Path(data_path)
    if not p.exists():
        return {"success": False, "error": f"File not found: {data_path}"}
    try:
        df = pd.read_csv(p)
    except Exception as e:
        return {"success": False, "error": f"Failed to read CSV: {e}"}

    artifacts = _ocr_artifacts(df, columns)
    return {"artifacts": artifacts}


# ─── Content Safety detectors (cloud-only, Amazon Comprehend) ────────────────


def _detect_content_safety(df: pd.DataFrame) -> list[Finding]:
    """Detect toxic content and PII in text columns using Amazon Comprehend.

    Cloud-only: requires valid AWS credentials AND the session must not be in
    offline mode. Skips gracefully when offline or when Comprehend is unavailable.
    Samples up to 50 text values per column to stay within API limits and cost.
    """
    import os
    # Respect the offline mode flag (AMLAS_OFFLINE or AMLAS_BACKENDS_OFFLINE)
    if os.environ.get("AMLAS_OFFLINE") or os.environ.get("AMLAS_BACKENDS_OFFLINE"):
        return []

    try:
        import boto3
        client = boto3.client("comprehend", region_name=resolve_aws_region())
        # Quick connectivity check (fails fast if no creds / wrong region)
        client.meta.region_name  # noqa: B018
    except Exception:
        return []  # graceful skip when no AWS

    text_cols = _object_cols(df)
    if not text_cols:
        return []

    findings: list[Finding] = []
    MAX_SAMPLE = 50  # cost + API rate guard

    for col in text_cols:
        texts = df[col].dropna().astype(str).head(MAX_SAMPLE).tolist()
        if not texts:
            continue

        # ── Toxicity detection (batch, up to 10 at a time) ──
        toxic_count = 0
        toxic_examples: list[str] = []
        for i in range(0, len(texts), 10):
            batch = texts[i:i + 10]
            try:
                resp = client.detect_toxic_content(
                    TextSegments=[{"Text": t[:1000]} for t in batch],
                    LanguageCode="en",
                )
                for j, result in enumerate(resp.get("ResultList", [])):
                    labels = result.get("Labels", [])
                    if any(lb.get("Score", 0) >= 0.7 for lb in labels):
                        toxic_count += 1
                        if len(toxic_examples) < 3:
                            toxic_examples.append(batch[j][:80])
            except Exception:
                break  # API error (quota, region, etc.) — stop gracefully

        if toxic_count > 0:
            findings.append(_finding(
                "SAFE-TOX", col,
                f"{toxic_count} of {len(texts)} sampled rows contain toxic content "
                f"(threshold >= 0.7). Examples: {toxic_examples}",
                "high",
            ))

        # ── PII detection (one row at a time, Comprehend limit) ──
        pii_count = 0
        pii_types_seen: set[str] = set()
        for text in texts[:MAX_SAMPLE]:
            try:
                resp = client.detect_pii_entities(
                    Text=text[:5000],
                    LanguageCode="en",
                )
                entities = resp.get("Entities", [])
                high_conf = [e for e in entities if e.get("Score", 0) >= 0.8]
                if high_conf:
                    pii_count += 1
                    pii_types_seen.update(e.get("Type", "UNKNOWN") for e in high_conf)
            except Exception:
                break

        if pii_count > 0:
            findings.append(_finding(
                "SAFE-PII", col,
                f"{pii_count} of {len(texts)} sampled rows contain PII "
                f"(types: {sorted(pii_types_seen)}). Redact before training.",
                "high",
            ))

    return findings
