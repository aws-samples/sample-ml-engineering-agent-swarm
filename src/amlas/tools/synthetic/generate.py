"""SDV tabular synthesis + SMOTE-family augmentation (spec sheet 2.6, master plan §7).

``synthesize_tabular`` wraps SDV's single-table synthesizers:

  * ``gaussian_copula`` — ``GaussianCopulaSynthesizer`` (safe default; highest
    composite fidelity in the 2025 MDPI study, CPU-friendly),
  * ``ctgan``           — ``CTGANSynthesizer`` (deep GAN; needs more data/epochs),
  * ``tvae``            — ``TVAESynthesizer`` (deep VAE).

``augment_imbalance`` wraps imbalanced-learn's SMOTE family for class-imbalance
oversampling (the transcript's augmentation ask):

  * ``smote``   — numeric-only SMOTE,
  * ``smotenc`` — mixed numeric+categorical SMOTENC,
  * ``adasyn``  — adaptive synthetic sampling,
  * ``random``  — RandomOverSampler fallback (when k-neighbours infeasible).

All heavy deps (``sdv``/``imblearn``) are imported lazily inside the function so
``tools.synthetic`` imports without the ``data`` extra; a missing extra returns a
clear ``{"success": False, "error": "install extra: data"}`` dict (fail-soft).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from strands import tool

from ...errors import tool_error_boundary
from ._common import (
    AUGMENT_STRATEGIES,
    SDV_GENERATORS,
    build_sdv_metadata,
    canonical_generator,
    read_csv_or_error,
    single_table_metadata_dict,
    synthetic_output_path,
)

_MISSING_SDV = {
    "success": False,
    "error": "synthetic generation dependencies are unavailable; rerun scripts/setup.sh",
}
_MISSING_IMBLEARN = {
    "success": False,
    "error": "augmentation dependencies are unavailable; rerun scripts/setup.sh",
}


# ─── SDV tabular synthesis ───────────────────────────────────────────────────


def _build_synthesizer(generator: str, metadata: Any, constraints: dict[str, Any] | None) -> Any:
    """Instantiate the requested SDV synthesizer (deep models accept ``epochs``)."""
    from sdv.single_table import (
        CTGANSynthesizer,
        GaussianCopulaSynthesizer,
        TVAESynthesizer,
    )

    params = dict(constraints or {})
    epochs = int(params.pop("epochs", 10))
    if generator == "gaussian_copula":
        return GaussianCopulaSynthesizer(metadata)
    if generator == "ctgan":
        return CTGANSynthesizer(metadata, epochs=epochs)
    if generator == "tvae":
        return TVAESynthesizer(metadata, epochs=epochs)
    raise ValueError(f"unknown generator: {generator}")  # pragma: no cover (guarded)


def synthesize_tabular_frame(
    df: pd.DataFrame,
    rows: int,
    generator: str,
    constraints: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    """Core SDV synthesis (plain function reused by the agent + validator).

    Returns ``(synthetic_df, metadata_columns_dict, warnings)``.

    ROBUSTNESS FIX (06-tui-e2e-hardening, bug #3): SDV's reject-sampling renders a
    ``tqdm`` progress bar during ``synthesizer.sample``. In any NON-TTY / captured-
    stdout context — the TUI's background worker thread, a container, AgentCore, or
    another embedded runner — ``tqdm`` can fail to initialise (``'tqdm' object has no
    attribute 'disable'``), and SDV then ABORTS with *"Sampling terminated. No
    results were saved due to unspecified output_file_path."* That made every
    ``/synth`` turn fail off-TTY even on clean data (reproduced via the Pilot
    harness; the same call SUCCEEDS at an interactive prompt — a true
    environment-dependent product bug, not a test artifact).

    Two layered defenses, both grounded in SDV's own guidance:
      1. Disable the progress bar (``show_progress_bar=False`` when supported) so
         the fragile tqdm path is never entered off-TTY.
      2. Pass an ``output_file_path`` to ``sample()`` so SDV PERSISTS partial/final
         results to disk instead of discarding them — exactly the recovery the
         error message names. We read the frame back and clean the temp file.
    """
    import tempfile
    from pathlib import Path as _Path

    warnings: list[str] = []
    if generator in ("ctgan", "tvae") and len(df) < 50:
        warnings.append(
            f"dataset has {len(df)} rows — deep generator '{generator}' may underfit; "
            "gaussian_copula is recommended for tiny datasets"
        )

    metadata = build_sdv_metadata(df, table_name="table")
    synth = _build_synthesizer(generator, metadata, constraints)
    synth.fit(df)

    # PRIMARY DEFENSE: disable SDV's ``tqdm`` reject-sampling progress bar for the
    # duration of the ``sample`` call via the env var ``tqdm`` honors globally
    # (``TQDM_DISABLE=1``). The bar is the sole failure point off-TTY: with captured
    # stdout, a worker thread (the TUI), or a container/AgentCore log
    # stream, ``tqdm`` raises ``'tqdm' object has no attribute 'disable'`` during
    # init/teardown and SDV then ABORTS the whole sample ("Sampling terminated").
    # Disabling the bar removes the fragile path entirely. We snapshot+restore the
    # env so we never leak the toggle into the rest of the process.
    # SECONDARY DEFENSE: still pass an ``output_file_path`` so that, in the unlikely
    # event the bar path is somehow re-entered, SDV persists generated rows and we
    # can recover them instead of discarding the work.
    import os as _os
    import sys as _sys

    tmp_dir = _Path(tempfile.mkdtemp())
    tmp_out = tmp_dir / "sdv_sample.csv"
    _prev_tqdm = _os.environ.get("TQDM_DISABLE")
    _os.environ["TQDM_DISABLE"] = "1"

    # The ROOT cause of the off-TTY abort: SDV builds its reject-sampling ``tqdm``
    # bar against ``sys.stderr``. When the caller has replaced stderr/stdout with a
    # NON-fd-backed object (for example, some worker-thread/container log
    # wrappers), ``tqdm.__init__`` raises ``'tqdm' object has no attribute
    # 'disable'`` and SDV aborts the whole sample BEFORE any row is generated. The
    # only bulletproof fix is to hand SDV a REAL writable stream for the duration
    # of ``sample`` — we point stdout/stderr at an ``os.devnull`` file object, then
    # restore the originals. This is safe and silent (SDV's bar output is noise we
    # don't surface anyway) and makes ``/synth`` robust in every embedded context.
    _devnull = open(_os.devnull, "w", encoding="utf-8")
    _orig_stdout, _orig_stderr = _sys.stdout, _sys.stderr
    _sys.stdout, _sys.stderr = _devnull, _devnull
    try:
        try:
            sampled = synth.sample(num_rows=rows, output_file_path=str(tmp_out))
        except TypeError:
            # Older/newer SDV without the kwarg — fall back to the plain call.
            sampled = synth.sample(num_rows=rows)
        except Exception as exc:
            # Last-resort recovery: read the rows SDV already persisted. Only
            # swallow the KNOWN off-TTY termination where a usable file exists;
            # re-raise anything else so genuine synthesis failures still surface.
            # Recovery warnings appended below are FIXED strings (no exception
            # text) so dataset characteristics can't leak into returned warnings.
            recovered = _recover_persisted_sample(exc, tmp_out)
            if recovered is None:
                raise
            sampled = recovered
            warnings.append(
                "synthesizer progress-bar aborted off-TTY; recovered "
                f"{len(sampled)} persisted rows from output_file_path"
            )
    finally:
        # Restore streams + env exactly as they were (never leak them).
        _sys.stdout, _sys.stderr = _orig_stdout, _orig_stderr
        try:
            _devnull.close()
        except Exception:  # nosec B110
            pass
        if _prev_tqdm is None:
            _os.environ.pop("TQDM_DISABLE", None)
        else:
            _os.environ["TQDM_DISABLE"] = _prev_tqdm
        try:
            if tmp_out.exists():
                tmp_out.unlink()
            tmp_dir.rmdir()
        except Exception:  # nosec B110
            pass

    meta_cols = single_table_metadata_dict(metadata).get("columns", {})
    return sampled, meta_cols, warnings


def _recover_persisted_sample(exc: Exception, out_path: Path) -> pd.DataFrame | None:
    """Read SDV's persisted rows after a progress-bar 'Sampling terminated' abort.

    SDV reports the file it flushed in its OWN message
    (``"...Partial results are stored in /tmp/.../sdv_sample.csv."``), which may
    differ from the ``output_file_path`` we passed. We therefore try BOTH the path
    we supplied AND the path parsed verbatim from the exception text, accepting the
    first CSV that contains ≥1 data row. Returns the recovered frame, or ``None``
    when ``exc`` is not the known termination / no usable file exists (caller
    re-raises so genuine synthesis failures still surface).
    """
    import re as _re

    msg = str(exc)
    low = msg.lower()
    if "sampling terminated" not in low and "tqdm" not in low:
        return None

    candidates: list[Path] = [out_path]
    # Parse "...stored in <path>." (with or without a trailing period).
    m = _re.search(r"stored in\s+(\S+?\.csv)", msg)
    if m:
        candidates.append(Path(m.group(1).rstrip(".")))

    for cand in candidates:
        try:
            if cand.exists() and cand.stat().st_size > 0:
                recovered = pd.read_csv(cand)
                if len(recovered) > 0:
                    return recovered
        except Exception:  # nosec B112
            continue
    return None


@tool
@tool_error_boundary(source="tools.synthetic.generate")
def synthesize_tabular(
    data_path: str,
    rows: int,
    generator: str = "gaussian_copula",
    constraints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate realistic synthetic rows from a reference dataset via SDV (master plan §7).

    Args:
        data_path: CSV whose distribution/structure the synthetic data should match.
        rows: Number of synthetic rows to generate.
        generator: ``gaussian_copula`` (default, safe), ``ctgan`` or ``tvae``
            (deep; aliases ``copula``/``gan``/``vae`` accepted). Deep models read
            ``constraints["epochs"]`` (default 10; use a small value in tests).
        constraints: Optional generator kwargs (e.g. ``{"epochs": 1}``).

    Returns:
        ``{"synthetic_path","generator","rows","metadata_detected","warnings"}``
        on success, or ``{"success": False, "error": ...}`` (missing file,
        unknown generator, or missing ``sdv`` extra).
    """
    gen = canonical_generator(generator)
    if gen not in SDV_GENERATORS:
        return {
            "success": False,
            "error": f"unknown generator '{generator}'; choose from {list(SDV_GENERATORS)}",
        }

    df, err = read_csv_or_error(data_path)
    if err is not None:
        return err
    assert df is not None

    try:
        import sdv  # noqa: F401
    except ImportError:
        return dict(_MISSING_SDV)

    try:
        sampled, meta_cols, warnings = synthesize_tabular_frame(df, int(rows), gen, constraints)
    except Exception as e:
        return {"success": False, "error": f"synthesis failed: {e}"}

    out_path = synthetic_output_path(data_path, gen)
    try:
        sampled.to_csv(out_path, index=False)
    except Exception as e:
        return {"success": False, "error": f"failed to write synthetic CSV: {e}"}

    return {
        "synthetic_path": str(out_path),
        "generator": gen,
        "rows": len(sampled),
        "metadata_detected": {c: v.get("sdtype") for c, v in meta_cols.items()},
        "warnings": warnings,
    }


# ─── SMOTE-family augmentation ───────────────────────────────────────────────


def _categorical_feature_indices(features: pd.DataFrame) -> list[int]:
    """Indices of non-numeric feature columns (for SMOTENC)."""
    return [i for i, c in enumerate(features.columns) if not pd.api.types.is_numeric_dtype(features[c])]


def _make_sampler(strategy: str, y: pd.Series, features: pd.DataFrame, balance: dict[str, Any] | None) -> Any:
    """Build the imblearn sampler, reducing k_neighbors for tiny minority classes."""
    from imblearn.over_sampling import ADASYN, SMOTE, SMOTENC, RandomOverSampler

    min_class = int(y.value_counts().min())
    k = max(1, min(5, min_class - 1))  # SMOTE needs k < smallest class size
    sampling_strategy = balance if balance else "auto"

    if strategy == "random" or min_class <= 1:
        return RandomOverSampler(sampling_strategy=sampling_strategy, random_state=0)
    if strategy == "smotenc":
        cat_idx = _categorical_feature_indices(features)
        if not cat_idx:
            # no categorical cols → plain SMOTE
            return SMOTE(sampling_strategy=sampling_strategy, k_neighbors=k, random_state=0)
        return SMOTENC(categorical_features=cat_idx, sampling_strategy=sampling_strategy, k_neighbors=k, random_state=0)
    if strategy == "adasyn":
        return ADASYN(sampling_strategy=sampling_strategy, n_neighbors=k, random_state=0)
    return SMOTE(sampling_strategy=sampling_strategy, k_neighbors=k, random_state=0)


@tool
@tool_error_boundary(source="tools.synthetic.generate")
def augment_imbalance(
    data_path: str,
    target: str,
    strategy: str = "smote",
    balance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Oversample minority classes via the SMOTE family (master plan §7 / C2).

    Args:
        data_path: CSV with a class-imbalanced ``target`` column.
        target: Name of the class label column to balance.
        strategy: ``smote`` (numeric), ``smotenc`` (mixed numeric+categorical),
            ``adasyn``, or ``random`` (RandomOverSampler fallback). When the
            minority class is too small for k-neighbours the tool reduces k or
            falls back to RandomOverSampler with a warning.
        balance: Optional explicit ``{class_label: target_count}`` sampling map.

    Returns:
        ``{"augmented_path","class_counts_before","class_counts_after","strategy","warnings"}``
        or ``{"success": False, "error": ...}``.
    """
    strat = str(strategy).strip().lower()
    if strat not in AUGMENT_STRATEGIES:
        return {
            "success": False,
            "error": f"unknown strategy '{strategy}'; choose from {list(AUGMENT_STRATEGIES)}",
        }

    df, err = read_csv_or_error(data_path)
    if err is not None:
        return err
    assert df is not None

    if target not in df.columns:
        return {"success": False, "error": f"target column '{target}' not found"}

    try:
        import imblearn  # noqa: F401
    except ImportError:
        return dict(_MISSING_IMBLEARN)

    warnings: list[str] = []
    X = df.drop(columns=[target])
    y = df[target]
    counts_before = {str(k): int(v) for k, v in y.value_counts().to_dict().items()}

    if y.nunique() < 2:
        return {"success": False, "error": "target has <2 classes — nothing to balance"}

    min_class = int(y.value_counts().min())
    if min_class <= 5 and strat in ("smote", "smotenc", "adasyn"):
        warnings.append(
            f"smallest class has {min_class} samples — reduced k_neighbors (or fell back to random)"
        )

    try:
        sampler = _make_sampler(strat, y, X, balance)
        X_res, y_res = sampler.fit_resample(X, y)
    except Exception as e:
        # robust fallback: RandomOverSampler never fails on ≥2 classes
        try:
            from imblearn.over_sampling import RandomOverSampler

            sampler = RandomOverSampler(random_state=0)
            X_res, y_res = sampler.fit_resample(X, y)
            warnings.append(f"'{strat}' failed ({e}); fell back to RandomOverSampler")
            strat = "random"
        except Exception as e2:
            return {"success": False, "error": f"augmentation failed: {e2}"}

    out = pd.DataFrame(X_res, columns=X.columns)
    out[target] = np.asarray(y_res)
    counts_after = {str(k): int(v) for k, v in out[target].value_counts().to_dict().items()}

    out_path = synthetic_output_path(data_path, f"augment_{strat}")
    try:
        out.to_csv(out_path, index=False)
    except Exception as e:
        return {"success": False, "error": f"failed to write augmented CSV: {e}"}

    return {
        "augmented_path": str(out_path),
        "class_counts_before": counts_before,
        "class_counts_after": counts_after,
        "strategy": strat,
        "rows_before": len(df),
        "rows_after": len(out),
        "warnings": warnings,
    }
