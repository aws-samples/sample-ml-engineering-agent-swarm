"""Strands @tool wrappers for model training.

Tools:
  - train_model: Train a sklearn/XGBoost model given algorithm name, hyperparameters, and data path.

Supports both classification and regression (REG-01). Target encoding uses a
LabelEncoder for classification (handles multiclass string targets — YMAP-01)
and a plain float cast for regression. Leakage/identifier columns are excluded
in a data-driven way (LEAK-01).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ADR-0002 / FIX-34: this module's fit helpers (_build_model, _preprocess_for_training) are reused
# VERBATIM by the SageMaker off-box entrypoint (sagemaker_entrypoints.tabular_train) so the fit is
# byte-identical across tiers. A stock SageMaker framework container has NO ``strands`` — so the
# ``@tool`` decorator degrades to an identity decorator when strands is absent. In the AgentCore
# runtime strands IS present and the real decorator is used, unchanged.
try:  # pragma: no cover - exercised in both tiers; trivial
    from strands import tool
except ModuleNotFoundError:  # SageMaker training container (strands-free)
    from collections.abc import Callable as _Callable
    from typing import Any as _Any

    def tool(func: _Callable[..., _Any] | None = None, **_kwargs: _Any) -> _Any:  # type: ignore[no-redef]
        """Identity ``@tool`` fallback when strands is absent (training container)."""
        def _wrap(f: _Callable[..., _Any]) -> _Callable[..., _Any]:
            return f
        return _wrap(func) if callable(func) else _wrap

from amlas.security.model_artifacts import (
    UntrustedArtifactError,
    dump_model_artifact,
    load_model_artifact,
)

from ..errors import tool_error_boundary
from .leakage_columns import detect_id_like_columns

logger = logging.getLogger(__name__)


def _normalize_column_name(name: str) -> str:
    """Canonicalize a column name for tolerant matching (A1).

    Rule: lowercase, strip surrounding whitespace, and collapse spaces/hyphens to
    underscores. This lets a spec's ``machine_failure`` match a CSV column
    ``Machine failure`` (capital M + space) — the definitive AI4I data-contract
    mismatch — without ever mutating the real column name used downstream.
    """
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def resolve_target_column(columns: list[str], requested: str) -> str:
    """Resolve ``requested`` to the ACTUAL dataset column name (A1).

    Matching precedence:
      1. Exact match — returned verbatim (a real column is never normalized away).
      2. Exactly one column whose normalized form equals the normalized request →
         that actual column (the chosen name is logged).
      3. Zero matches → ``ValueError`` naming the request AND listing all columns.
      4. More than one normalized match → ``ValueError`` (ambiguous) — never
         silently pick the wrong column.

    Args:
        columns: The dataset's actual column names.
        requested: The target column name as declared in the spec.

    Returns:
        The actual column name to use for training.

    Raises:
        ValueError: on zero or ambiguous (>1) matches, with a clear message.
    """
    cols = list(columns)

    # 1. Exact match wins outright.
    if requested in cols:
        return requested

    target_norm = _normalize_column_name(requested)
    matches = [c for c in cols if _normalize_column_name(c) == target_norm]

    if len(matches) == 1:
        chosen = matches[0]
        logger.info(
            "Target column '%s' resolved to actual dataset column '%s' "
            "(case/whitespace-insensitive match).",
            requested,
            chosen,
        )
        return chosen

    if len(matches) > 1:
        raise ValueError(
            f"Target column '{requested}' is ambiguous: it normalizes to multiple "
            f"dataset columns {matches!r}. Rename the columns or set the spec "
            f"target to the exact column name."
        )

    raise ValueError(
        f"Target column '{requested}' not found. Available columns: {cols!r}"
    )

# Registry of supported algorithms
ALGORITHM_REGISTRY = {
    "logistic_regression": "sklearn.linear_model.LogisticRegression",
    "random_forest": "sklearn.ensemble.RandomForestClassifier",
    "gradient_boosting": "sklearn.ensemble.GradientBoostingClassifier",
    "hist_gradient_boosting": "sklearn.ensemble.HistGradientBoostingClassifier",
    "xgboost": "xgboost.XGBClassifier",
    "lightgbm": "lightgbm.LGBMClassifier",
    "adaboost": "sklearn.ensemble.AdaBoostClassifier",
    "decision_tree": "sklearn.tree.DecisionTreeClassifier",
    "svm": "sklearn.svm.SVC",
    # Regression variants (selected based on task_type)
    "random_forest_regressor": "sklearn.ensemble.RandomForestRegressor",
    "gradient_boosting_regressor": "sklearn.ensemble.GradientBoostingRegressor",
    "hist_gradient_boosting_regressor": "sklearn.ensemble.HistGradientBoostingRegressor",
    "xgboost_regressor": "xgboost.XGBRegressor",
    "lightgbm_regressor": "lightgbm.LGBMRegressor",
}

# Static Titanic/Kaggle-era identifier names. Retained as a baseline hint, but
# the real exclusion is data-driven via detect_id_like_columns (LEAK-01).
_ID_COLUMNS = {"PassengerId", "Name", "name", "id", "ID", "Id"}


def _build_classifier(algorithm: str, params: dict[str, Any], seed: int) -> Any:
    """Instantiate a classifier from the registry with given parameters."""
    if algorithm == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        defaults: dict[str, Any] = {"max_iter": 1000, "random_state": seed}
        defaults.update(params)
        return LogisticRegression(**defaults)

    elif algorithm == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        defaults = {"n_estimators": 100, "random_state": seed}
        defaults.update(params)
        return RandomForestClassifier(**defaults)

    elif algorithm == "gradient_boosting":
        from sklearn.ensemble import GradientBoostingClassifier
        defaults = {"n_estimators": 100, "random_state": seed}
        defaults.update(params)
        return GradientBoostingClassifier(**defaults)

    elif algorithm in ("hist_gradient_boosting", "histgradientboosting"):
        # FIX-28: the interaction-capable, fast, native-NaN sklearn learner specs mandate for
        # hard tabular tasks (bogie_compound_fault_v2). Absent here, the swarm had to hand-roll
        # a "paranoid fallback" that fought the tool and stalled — now it is a first-class algo.
        from sklearn.ensemble import HistGradientBoostingClassifier
        defaults = {"random_state": seed}
        defaults.update(params)
        return HistGradientBoostingClassifier(**defaults)

    elif algorithm == "xgboost":
        try:
            from xgboost import XGBClassifier
        except (ImportError, OSError) as e:
            raise ValueError(
                f"XGBoost is not available: {e}. Rerun scripts/setup.sh and ensure "
                "the platform OpenMP runtime is installed."
            ) from e
        defaults = {
            "n_estimators": 100,
            "random_state": seed,
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "verbosity": 0,
        }
        defaults.update(params)
        return XGBClassifier(**defaults)

    elif algorithm in ("lightgbm", "lgbm", "light_gbm"):
        # LightGBM is a top tabular learner the research/agents routinely pick, and it IS installed
        # in the SageMaker source bundle (FIX-34c) — but it was missing from this builder, so a live
        # agent choice of 'lightgbm' failed with "Unknown classification algorithm" in the managed runtime
        # net-new flow. Wire it in. Drop non-constructor kwargs the agent sometimes emits
        # (categorical_features is a fit()-time arg, not a ctor arg; early_stopping_rounds needs an
        # eval_set which the off-box refit does not provide — the entrypoint already strips it).
        try:
            from lightgbm import LGBMClassifier
        except (ImportError, OSError) as e:
            raise ValueError(
                f"LightGBM is not available: {e}. Rerun scripts/setup.sh and ensure "
                "the platform OpenMP runtime is installed."
            ) from e
        _drop = {"categorical_features", "early_stopping_rounds", "num_class", "use_label_encoder"}
        defaults = {"n_estimators": 200, "random_state": seed, "n_jobs": -1, "verbosity": -1}
        defaults.update({k: v for k, v in params.items() if k not in _drop})
        return LGBMClassifier(**defaults)

    elif algorithm == "adaboost":
        from sklearn.ensemble import AdaBoostClassifier
        defaults = {"n_estimators": 50, "random_state": seed}
        defaults.update(params)
        return AdaBoostClassifier(**defaults)

    elif algorithm == "decision_tree":
        from sklearn.tree import DecisionTreeClassifier
        defaults = {"random_state": seed}
        defaults.update(params)
        return DecisionTreeClassifier(**defaults)

    elif algorithm == "svm":
        from sklearn.svm import SVC
        defaults = {"random_state": seed, "probability": True}
        defaults.update(params)
        return SVC(**defaults)

    elif algorithm in ("knn", "k_nearest_neighbors"):
        from sklearn.neighbors import KNeighborsClassifier
        return KNeighborsClassifier(**params)

    elif algorithm in ("naive_bayes", "gaussian_nb"):
        from sklearn.naive_bayes import GaussianNB
        return GaussianNB(**params)

    else:
        raise ValueError(
            f"Unknown classification algorithm: '{algorithm}'. "
            f"Supported: {list(ALGORITHM_REGISTRY.keys())}"
        )


def _build_regressor(algorithm: str, params: dict[str, Any], seed: int) -> Any:
    """Instantiate a regressor for the given algorithm name (REG-01).

    Accepts both base names (``random_forest``) and explicit ``*_regressor``
    names from ``ALGORITHM_REGISTRY``.
    """
    # Normalize explicit regressor suffix to the base algorithm name.
    base = algorithm[: -len("_regressor")] if algorithm.endswith("_regressor") else algorithm

    if base in ("random_forest",):
        from sklearn.ensemble import RandomForestRegressor
        defaults = {"n_estimators": 100, "random_state": seed}
        defaults.update(params)
        return RandomForestRegressor(**defaults)

    elif base in ("gradient_boosting",):
        from sklearn.ensemble import GradientBoostingRegressor
        defaults = {"n_estimators": 100, "random_state": seed}
        defaults.update(params)
        return GradientBoostingRegressor(**defaults)

    elif base in ("hist_gradient_boosting", "histgradientboosting"):
        from sklearn.ensemble import HistGradientBoostingRegressor
        defaults = {"random_state": seed}
        defaults.update(params)
        return HistGradientBoostingRegressor(**defaults)

    elif base in ("xgboost",):
        try:
            from xgboost import XGBRegressor
        except (ImportError, OSError) as e:
            raise ValueError(
                f"XGBoost is not available: {e}. Rerun scripts/setup.sh and ensure "
                "the platform OpenMP runtime is installed."
            ) from e
        defaults = {"n_estimators": 100, "random_state": seed, "verbosity": 0}
        defaults.update(params)
        return XGBRegressor(**defaults)

    elif base in ("lightgbm", "lgbm", "light_gbm", "lightgbm_regressor"):
        try:
            from lightgbm import LGBMRegressor
        except (ImportError, OSError) as e:
            raise ValueError(
                f"LightGBM is not available: {e}. Rerun scripts/setup.sh and ensure "
                "the platform OpenMP runtime is installed."
            ) from e
        _drop = {"categorical_features", "early_stopping_rounds", "num_class", "use_label_encoder"}
        lgbm_params: dict[str, Any] = {"n_estimators": 200, "random_state": seed, "n_jobs": -1, "verbosity": -1}
        lgbm_params.update({k: v for k, v in params.items() if k not in _drop})
        return LGBMRegressor(**lgbm_params)

    elif base in ("decision_tree",):
        from sklearn.tree import DecisionTreeRegressor
        defaults = {"random_state": seed}
        defaults.update(params)
        return DecisionTreeRegressor(**defaults)

    elif base in ("adaboost",):
        from sklearn.ensemble import AdaBoostRegressor
        defaults = {"n_estimators": 50, "random_state": seed}
        defaults.update(params)
        return AdaBoostRegressor(**defaults)

    elif base in ("svm", "svr"):
        from sklearn.svm import SVR
        return SVR(**params)

    elif base in ("knn", "k_nearest_neighbors"):
        from sklearn.neighbors import KNeighborsRegressor
        return KNeighborsRegressor(**params)

    elif base in ("linear_regression", "linear"):
        from sklearn.linear_model import LinearRegression
        return LinearRegression(**params)

    elif base in ("ridge", "ridge_regression"):
        from sklearn.linear_model import Ridge
        defaults = {"random_state": seed}
        defaults.update(params)
        return Ridge(**defaults)

    elif base in ("lasso", "lasso_regression"):
        from sklearn.linear_model import Lasso
        defaults = {"random_state": seed}
        defaults.update(params)
        return Lasso(**defaults)

    elif base in ("elastic_net", "elasticnet"):
        from sklearn.linear_model import ElasticNet
        defaults = {"random_state": seed}
        defaults.update(params)
        return ElasticNet(**defaults)

    else:
        raise ValueError(
            f"Unknown regression algorithm: '{algorithm}'. Supported regressors: "
            "random_forest, gradient_boosting, xgboost, decision_tree, adaboost, "
            "svm, knn, linear_regression, ridge, lasso, elastic_net"
        )


def _build_model(
    algorithm: str,
    params: dict[str, Any],
    seed: int,
    task_type: str = "classification",
) -> Any:
    """Instantiate a model for the given algorithm and task type (REG-01)."""
    if task_type == "regression":
        return _build_regressor(algorithm, params, seed)
    return _build_classifier(algorithm, params, seed)


def _encode_target(y: pd.Series, task_type: str) -> tuple[np.ndarray, Any]:
    """Encode the target column for the given task type (YMAP-01).

    Classification: a ``LabelEncoder`` maps any label space (multiclass strings,
    booleans, ints) to 0..k-1, returning the fitted encoder for inverse mapping.
    Regression: the target is cast to float — never int-cast, never label-mapped.

    Returns (encoded_y, encoder_or_None).
    """
    if task_type == "regression":
        return y.astype(float).values, None

    from sklearn.preprocessing import LabelEncoder

    encoder = LabelEncoder()
    encoded = encoder.fit_transform(y.astype(str) if y.dtype == object else y)
    return encoded, encoder


def _preprocess_for_training(
    df: pd.DataFrame,
    target_column: str,
    task_type: str = "classification",
) -> tuple[np.ndarray, np.ndarray]:
    """Quick preprocessing: impute + encode for training.

    Returns (X, y) as numpy arrays. Target handling branches on task_type
    (YMAP-01); identifier/leakage columns are excluded data-driven (LEAK-01).
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    # Drop rows whose TARGET is missing before anything else (NAN-TGT-01). sklearn
    # estimators reject NaN in y with "Input y contains NaN" — a real crash hit on the
    # live GHCN weather run (134 rows have a missing TMAX). Imputing a label would be
    # wrong (fabricates supervision), so the correct fix is to drop those rows; X and y
    # are sliced from the SAME filtered frame so they stay index-aligned.
    if target_column in df.columns:
        _n_before = len(df)
        df = df[df[target_column].notna()].reset_index(drop=True)
        _n_dropped = _n_before - len(df)
        if _n_dropped:
            logger.info(
                "dropped %d/%d rows with missing target '%s' before training",
                _n_dropped, _n_before, target_column,
            )

    # Remove ID / leakage-prone columns (data-driven — LEAK-01).
    id_like = detect_id_like_columns(df, target_column=target_column)
    exclude = set(id_like) | {target_column}
    feature_cols = [c for c in df.columns if c not in exclude]
    feature_df = df[feature_cols].copy()

    # Replace inf/-inf with NaN so imputer can handle them.
    feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Convert bool columns to int (SimpleImputer doesn't support bool dtype).
    bool_cols = feature_df.select_dtypes(include=["bool"]).columns.tolist()
    for col in bool_cols:
        feature_df[col] = feature_df[col].astype(int)

    # Identify column types
    numeric_cols = feature_df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = feature_df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Build pipeline
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # PERF-14 / high-cardinality one-hot guard: cap the encoded feature space so a
    # pathological near-unique categorical column (e.g. ADSA ``Cabin`` = 6560 uniques
    # in 8693 rows) cannot explode the dense matrix to thousands of columns. Left
    # unbounded, that ~6.6k-wide matrix makes an O(n²·d) kernel model — ``SVC(rbf,
    # probability=True)`` from the research recommender's top-5 — hang past the 300s
    # per-test guard (the four e2e pipeline timeouts). ``max_categories=50`` keeps the
    # 50 most-frequent levels and folds the long rare tail into a single "infrequent"
    # bucket; ``handle_unknown="infrequent_if_exist"`` routes unseen categories there
    # too (falling back to the all-zeros row when no infrequent bucket exists — the
    # same effect the prior ``"ignore"`` gave). Columns with ≤50 distinct values are
    # encoded IDENTICALLY to before (the cap never binds), so the small in-repo
    # fixtures (ai4i, tiny CSVs) are numerically unchanged; only the pathological
    # wide-tail column is bounded, and ADSA metrics stay well above the F1 gates
    # (RF 0.79→0.79, SVM now trainable at ~0.78).
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="infrequent_if_exist",
                sparse_output=False,
                max_categories=50,
            ),
        ),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )

    X = preprocessor.fit_transform(feature_df)

    # Target encoding (task-type aware — YMAP-01)
    y, _encoder = _encode_target(df[target_column].copy(), task_type)

    _LAST_FITTED.append(FittedFeaturePipeline(preprocessor=preprocessor, feature_cols=feature_cols))
    return X, y


@dataclass(frozen=True)
class FittedFeaturePipeline:
    """A feature pipeline FITTED on the train split, for transforming other splits.

    Exists because the test split must not be preprocessed independently of train. Fitting a
    second pipeline on the test frame — which is what the SageMaker entrypoint used to do — is
    wrong in three separate ways, all measured:

    * **scaling** comes from test statistics, so a raw test range of 100..102 was handed to the
      model as -1.22..1.22 — indistinguishable from a train range of 0..4. Silently wrong
      predictions, and worst for the scale-sensitive estimators (svm / knn / ridge).
    * **feature width** diverges whenever categorical levels differ: a train frame with levels
      {a,b,c} produced 4 columns and a test frame with only {a} produced 2, so ``predict`` gets a
      differently-shaped matrix than the model was fitted on.
    * **row count** diverges because the target-NaN drop applies to the test frame too, so rows
      whose LABEL is missing lose their prediction — even though ``predict`` never needs a label.
      That desynchronised ``y_pred`` from the staged ``y_true`` and the Arbiter correctly refused
      to grade it (``InvalidSubmissionError: shape mismatch (17964,) vs (17939,)``).

    Hold the pipeline from the train fit and :func:`transform_features` every other split with it.
    """

    preprocessor: Any
    feature_cols: list[str]


#: Internal hand-off so :func:`fit_feature_pipeline` can expose the artefacts
#: ``_preprocess_for_training`` fits, without changing that function's public 2-tuple return
#: (``train_model`` and a dozen tests depend on it).
_LAST_FITTED: list[FittedFeaturePipeline] = []


def fit_feature_pipeline(
    df: pd.DataFrame,
    target_column: str,
    task_type: str = "classification",
) -> tuple[np.ndarray, np.ndarray, FittedFeaturePipeline]:
    """Preprocess the TRAIN split and return the fitted pipeline alongside ``(X, y)``.

    Identical to :func:`_preprocess_for_training` — it delegates, so there is one implementation
    and no chance of the two drifting — plus the fitted artefacts needed to transform the test
    split consistently.
    """
    _LAST_FITTED.clear()
    X, y = _preprocess_for_training(df, target_column, task_type=task_type)
    return X, y, _LAST_FITTED[-1]


def transform_features(
    fitted: FittedFeaturePipeline,
    df: pd.DataFrame,
) -> np.ndarray:
    """Transform ``df`` with the TRAIN-fitted pipeline. Never drops a row.

    One output row per input row, by construction — that is the invariant the Arbiter's shape
    check depends on. The target column is not consulted at all: ``predict`` does not need a
    label, so a missing one must not cost a prediction.
    """
    missing = [c for c in fitted.feature_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"cannot transform: the train-fitted pipeline needs feature columns {missing} "
            f"which are absent from this frame (has {list(df.columns)})"
        )

    feature_df = df[fitted.feature_cols].copy()
    # Same normalisations the fit applied, in the same order.
    feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    for col in feature_df.select_dtypes(include=["bool"]).columns.tolist():
        feature_df[col] = feature_df[col].astype(int)

    X: np.ndarray = fitted.preprocessor.transform(feature_df)
    if X.shape[0] != len(df):
        raise AssertionError(
            f"transform changed the row count ({len(df)} in, {X.shape[0]} out); predictions would "
            "no longer align with y_true"
        )
    return X


@tool
@tool_error_boundary(source="tools.training_tools")
def train_model(
    data_path: str,
    target_column: str,
    algorithm: str,
    params: dict[str, Any] | None = None,
    output_dir: str | None = None,
    seed: int = 42,
    test_size: float = 0.2,
    task_type: str = "classification",
) -> dict[str, Any]:
    """Train a sklearn/XGBoost model on a CSV dataset and return evaluation metrics.

    Performs automatic preprocessing (imputation, scaling, encoding) then
    trains the specified algorithm with given hyperparameters. Evaluates on a
    holdout test set (stratified for classification).

    Supported classification algorithms: logistic_regression, random_forest,
    gradient_boosting, xgboost, adaboost, decision_tree, svm.
    Supported regression algorithms: random_forest, gradient_boosting, xgboost,
    decision_tree, adaboost, svm, knn, linear_regression, ridge, lasso, elastic_net.

    Args:
        data_path: Filesystem path to training CSV file.
        target_column: Name of the target column.
        algorithm: Algorithm name from supported list.
        params: Hyperparameter dict for the algorithm.
        output_dir: Directory to save model artifact, predictions, and metadata.
        seed: Random seed for reproducibility.
        test_size: Fraction of data for evaluation holdout.
        task_type: "classification" (default) or "regression".
    """
    from sklearn.model_selection import train_test_split

    if params is None:
        params = {}

    start_time = time.time()

    # Load data
    p = Path(data_path)
    if not p.exists():
        return {"success": False, "algorithm": algorithm, "error": f"File not found: {data_path}"}

    try:
        df = pd.read_csv(p)
    except Exception as e:
        return {"success": False, "algorithm": algorithm, "error": f"Failed to read CSV: {e}"}

    # A1: tolerant target-column resolution (case/whitespace/underscore-insensitive)
    # so a spec's ``machine_failure`` matches a CSV column ``Machine failure``.
    try:
        target_column = resolve_target_column(list(df.columns), target_column)
    except ValueError as e:
        return {"success": False, "algorithm": algorithm, "error": str(e)}

    # Build model
    try:
        model = _build_model(algorithm, params, seed, task_type=task_type)
    except ValueError as e:
        return {"success": False, "algorithm": algorithm, "error": str(e)}

    # Preprocess
    try:
        X, y = _preprocess_for_training(df, target_column, task_type=task_type)
    except Exception as e:
        return {"success": False, "algorithm": algorithm, "error": f"Preprocessing failed: {e}"}

    # Split — stratify only for classification (regression targets are continuous)
    stratify = y if task_type == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=stratify
    )

    # Train
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        return {"success": False, "error": f"Training failed: {e}"}

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate (task-type aware)
    if task_type == "regression":
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        mse = float(mean_squared_error(y_test, y_pred))
        metrics = {
            "rmse": float(np.sqrt(mse)),
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "mse": mse,
            "r2": float(r2_score(y_test, y_pred)),
        }
    else:
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
            "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        }

    duration = time.time() - start_time

    result: dict[str, Any] = {
        "success": True,
        "algorithm": algorithm,
        "task_type": task_type,
        "params": params,
        "metrics": metrics,
        "train_size": len(y_train),
        "test_size": float(test_size),
        "n_test_samples": len(y_test),
        "duration_seconds": duration,
    }

    # Save artifacts
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Skops requires explicit type inspection before load and avoids executable
        # object deserialization. The shared loader also enforces root containment.
        model_path = out / "model.skops"
        dump_model_artifact(model, model_path)
        result["model_path"] = str(model_path)

        # Predictions
        y_pred_path = out / "y_pred.npy"
        np.save(y_pred_path, y_pred)
        result["y_pred_path"] = str(y_pred_path)

        # Metadata JSON
        metadata = {
            "algorithm": algorithm,
            "task_type": task_type,
            "params": params,
            "metrics": metrics,
            "duration_seconds": duration,
            "train_size": len(y_train),
            "test_size": len(y_test),
            "seed": seed,
            "data_path": data_path,
            "target_column": target_column,
        }
        metadata_path = out / "training_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2))
        result["metadata_path"] = str(metadata_path)

    return result


# Compatibility re-exports: callers historically imported the artifact boundary
# from this tool module. The implementation now lives in ``amlas.security``.
__all__ = [
    "UntrustedArtifactError",
    "dump_model_artifact",
    "load_model_artifact",
    "resolve_target_column",
    "train_model",
]
