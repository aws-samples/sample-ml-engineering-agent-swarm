"""SageMaker tabular training entrypoint (ADR-0002 Phase 1).

Runs INSIDE a SageMaker Training job. The trainer_tuner agent DISPATCHES here instead of
fitting in the AgentCore sandbox: it reads the staged train + test CSVs, fits the agent-chosen
sklearn/HGB/XGBoost estimator on the FULL train split, predicts the test split, and writes the
canonical ``y_pred_test.npy`` to the output dir (which SageMaker uploads to S3). The Arbiter
grades that artifact — the graded contract is byte-identical to the old in-container path; only
the compute LOCATION moves off-box (where there is no ~5-min kernel wall, no idle-cycle, no
shared sandbox session quota).

``run_training`` is pure + directly verifiable (plain paths). ``main`` is the thin SageMaker shim that
resolves the ``SM_CHANNEL_*`` / ``SM_OUTPUT_DATA_DIR`` env vars and calls it. Model building +
preprocessing reuse :mod:`amlas.tools.training_tools` so the fit is IDENTICAL to what the
in-container tool did.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_YPRED_FILENAME = "y_pred_test.npy"


def run_training(
    *,
    train_path: str,
    test_path: str,
    target_column: str,
    algo: str,
    params: dict[str, Any],
    task_type: str = "classification",
    output_dir: str,
    n_splits: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """Fit ``algo`` on the full train split, predict the test split, write y_pred_test.npy.

    Returns a summary dict (algo, params, train_cv_score/std, n_test_predictions, y_pred_path).
    Reuses the same ``_build_model`` + ``_preprocess_for_training`` as the in-container tool, so
    the model + preprocessing are identical — only the compute runs here (off-box) instead of the
    sandbox. Never fabricates predictions.
    """
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

    from ..tools.training_tools import _build_model, fit_feature_pipeline, transform_features

    # Inlined from training_jobs._cv_scoring — keeps this off-box entrypoint's import chain free of
    # tools.training_jobs (which pulls tools.sagemaker → boto3) so the SageMaker source tarball stays
    # minimal. Behavior is identical: regression → r2, classification → f1_macro.
    def _cv_scoring(tt: str) -> str:
        return "r2" if tt == "regression" else "f1_macro"

    # FIX-37 (ADR-0002 Phase 6): the entrypoint does a plain ``fit(X, y)`` with NO eval_set (both the
    # CV folds and the final full-train refit). EARLY-STOPPING params (early_stopping_rounds, and the
    # XGBoost>=1.6 constructor arg / eval_metric-for-stopping) require a validation set — XGBoost's
    # early-stopping callback CRASHES at ``after_iteration`` when fit() has none (live: xgboost 1.7.6
    # "callback.py after_iteration"). Early stopping is also meaningless in a refit-on-ALL-data. So
    # strip the early-stopping family from params before building any model here. This is universal:
    # the agent may legitimately choose early-stopping HPs for its own CV reasoning; the OFF-BOX refit
    # must sanitize them. (eval_metric is kept — it is a valid, harmless metric spec on its own.)
    _EARLY_STOP_KEYS = {
        "early_stopping_rounds", "early_stopping", "n_iter_no_change",
        "callbacks", "eval_set", "validation_fraction",
    }
    safe_params = {k: v for k, v in (params or {}).items() if k not in _EARLY_STOP_KEYS}

    train_df = pd.read_csv(train_path)
    if target_column not in train_df.columns:
        raise ValueError(f"target column {target_column!r} not in train data")

    X_train, y_train, fitted = fit_feature_pipeline(train_df, target_column, task_type=task_type)

    # Report a real CV score on the train split (decision-support for the agent) — the fit that
    # PRODUCES y_pred is the full-train refit below (test labels never touch selection).
    model_cv = _build_model(algo, safe_params, seed, task_type=task_type)
    if task_type == "classification":
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scoring = _cv_scoring(task_type)
    try:
        scores = cross_val_score(model_cv, X_train, y_train, cv=cv, scoring=scoring)
        cv_mean, cv_std = float(np.mean(scores)), float(np.std(scores))
    except Exception:  # CV is best-effort decision-support; never block the y_pred write
        cv_mean, cv_std = float("nan"), float("nan")

    # Refit on the FULL train split, predict the staged test split.
    model = _build_model(algo, safe_params, seed, task_type=task_type)
    model.fit(X_train, y_train)

    # Transform the test split with the TRAIN-fitted pipeline. This used to call
    # ``_preprocess_for_training`` again, which FITS a second pipeline on the test frame — wrong
    # three ways, all measured: test-derived scaling (raw 100..102 arrived as -1.22..1.22, i.e.
    # indistinguishable from a train range of 0..4), a feature width that diverges whenever
    # categorical levels differ (4 columns vs 2), and a row count reduced by the target-NaN drop
    # so rows with a missing LABEL silently lost their prediction. The last one desynchronised
    # y_pred from the staged y_true and the Arbiter correctly refused to grade
    # (InvalidSubmissionError: shape mismatch (17964,) vs (17939,)).
    #
    # ``transform_features`` needs no target column at all — ``predict`` never does — so the
    # labelled / unlabelled branch collapses.
    test_df = pd.read_csv(test_path)
    X_test = transform_features(fitted, test_df)
    y_pred_test = model.predict(X_test)

    # The invariant, asserted where predictions are WRITTEN rather than discovered at grading
    # time: one prediction per staged test row. A mismatch here fails the JOB, with the counts in
    # the message, instead of surfacing minutes later as an opaque Arbiter refusal.
    if len(y_pred_test) != len(test_df):
        raise AssertionError(
            f"y_pred/y_true misalignment: {len(test_df)} test rows staged but "
            f"{len(y_pred_test)} predictions produced — the Arbiter grades against the staged "
            "test split, so these must match exactly"
        )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    y_pred_path = out / _YPRED_FILENAME
    np.save(y_pred_path, np.asarray(y_pred_test))
    print(f"WROTE {y_pred_path} {len(y_pred_test)}", flush=True)

    summary = {
        "algo": algo,
        "params": params,
        "params_fitted": safe_params,  # FIX-37: what was ACTUALLY fitted (early-stopping stripped)
        "task_type": task_type,
        "train_cv_score": cv_mean,
        "train_cv_std": cv_std,
        "cv_scoring": scoring,
        "n_test_predictions": len(y_pred_test),
        "n_train_rows": len(X_train),
        "y_pred_path": str(y_pred_path),
    }
    # Also drop a small summary json alongside for observability.
    try:
        (out / "training_summary.json").write_text(json.dumps(summary, default=str, indent=2))
    except Exception:  # pragma: no cover — summary write is best-effort  # nosec B110
        pass
    return summary


def resolve_sagemaker_paths() -> dict[str, str]:
    """Resolve the train/test dirs + output dir from SageMaker's SM_* env vars.

    SageMaker stages input channels at ``SM_CHANNEL_<NAME>`` and expects outputs under
    ``SM_OUTPUT_DATA_DIR`` (uploaded to S3). Defaults follow the standard container paths.
    """
    return {
        "train_dir": os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"),
        "test_dir": os.environ.get("SM_CHANNEL_TEST", "/opt/ml/input/data/test"),
        "output_dir": os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"),
    }


def _first_csv(directory: str, fallback_name: str) -> str:
    """Return the single CSV in a SageMaker channel dir (or the conventional name)."""
    d = Path(directory)
    csvs = sorted(d.glob("*.csv"))
    return str(csvs[0]) if csvs else str(d / fallback_name)


def main() -> None:  # pragma: no cover — runs only inside the SageMaker container
    """SageMaker container entrypoint: read hyperparameters + channels, run training."""
    # SageMaker writes hyperparameters to /opt/ml/input/config/hyperparameters.json (strings).
    hp_path = Path("/opt/ml/input/config/hyperparameters.json")
    hp: dict[str, Any] = {}
    if hp_path.exists():
        try:
            hp = json.loads(hp_path.read_text())
        except Exception:
            hp = {}
    # Env overrides win (AMLAS_* passed via the training job's Environment).
    algo = os.environ.get("AMLAS_ALGO", hp.get("algo", "hist_gradient_boosting"))
    target = os.environ.get("AMLAS_TARGET", hp.get("target_column", "target"))
    task_type = os.environ.get("AMLAS_TASK_TYPE", hp.get("task_type", "classification"))
    params_raw = os.environ.get("AMLAS_PARAMS", hp.get("params", "{}"))
    try:
        params = json.loads(params_raw) if isinstance(params_raw, str) else dict(params_raw)
    except Exception:
        params = {}

    paths = resolve_sagemaker_paths()
    run_training(
        train_path=_first_csv(paths["train_dir"], "train.csv"),
        test_path=_first_csv(paths["test_dir"], "test.csv"),
        target_column=target,
        algo=algo,
        params=params,
        task_type=task_type,
        output_dir=paths["output_dir"],
    )


if __name__ == "__main__":  # pragma: no cover
    main()
