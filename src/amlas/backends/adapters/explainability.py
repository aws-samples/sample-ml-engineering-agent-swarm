"""ExplainabilityBackend adapter (doc 05 §4.2 G3 / roadmap M0 work-item 9; gap-closure §G1).

* ``ClarifyBackend`` — **default**; wraps EXISTING ``tools/mlops/clarify.py``
  (``run_clarify`` — bias metrics + SHAP as a Processing job).
* ``InProcessSHAPBackend`` — **local**; a real, offline SHAP explainability backend
  (gap-closure §G1). Computes genuine global feature attributions in-process
  (``shap.TreeExplainer`` fast-path for tree models, ``shap.KernelExplainer`` fallback
  for any ``predict`` callable), writes a local ``shap_uri`` CSV of mean-abs
  attributions, and returns the SAME :class:`ExplainReport` shape ``ClarifyBackend``
  returns — so the two are interchangeable behind the port and the **Stage-9
  Explainability readiness gate is satisfiable fully offline** (no Clarify / no AWS).

The ``shap`` dependency is imported **lazily** inside the method (no import-time cost;
``backends.registry`` stays boto3/shap-free until first use). ``register_adapter(...,
"inprocess_shap", is_local=True)`` makes the offline auto path resolve to it.
"""

from __future__ import annotations

import csv
import os
import tempfile
from collections.abc import Callable
from typing import Any

from ..ports import ExplainabilityBackend
from ..registry import REGISTRY, register_adapter
from ..types import ExplainReport, ModelRef
from ._util import _csv_safe_cell, raw_tool


class ClarifyBackend:
    """Default ExplainabilityBackend — wraps EXISTING ``tools/mlops/clarify.py``.

    ``role_arn`` is the SageMaker execution role the Clarify **Processing job** runs as.
    It is resolved (in order): an explicit constructor arg → ``AMLAS_SAGEMAKER_ROLE_ARN``
    / ``AMLAS_SAGEMAKER_ROLE_ARN`` env → empty. When empty, ``run_clarify`` falls back to a
    placeholder ARN in the ``000000000000`` account, which a REAL ``create_processing_job``
    rejects with ``RoleArn: Cross-account pass role is not allowed`` (the G1 live-leg bug):
    the placeholder's account never matches the caller's. Threading a real same-account role
    through fixes the managed Clarify submit (gap-closure §G1 live).
    """

    def __init__(
        self, *, region: str | None = None, sm_client: Any = None, role_arn: str = "",
    ) -> None:
        self._region = region
        self._sm = sm_client
        self._role_arn = role_arn or os.environ.get("AMLAS_SAGEMAKER_ROLE_ARN", "") or os.environ.get(
            "AMLAS_SAGEMAKER_ROLE_ARN", ""
        )

    def bias_and_shap(
        self, *, dataset_uri: str, target: str,
        sensitive_cols: list[str] | None = None, model_ref: ModelRef | None = None,
    ) -> ExplainReport:
        """Run bias + SHAP via EXISTING ``run_clarify`` (unchanged)."""
        from ...tools.mlops.clarify import run_clarify

        config: dict[str, Any] = {"analysis": "both", "label": target}
        if sensitive_cols:
            config["facet_name"] = sensitive_cols[0]
        model = model_ref.uri if model_ref else ""
        out = raw_tool(run_clarify)(
            model, dataset_uri, config,
            role_arn=self._role_arn, region=self._region, sm_client=self._sm,
        )
        if out.get("success") is False:
            raise RuntimeError(out.get("error", "run_clarify failed"))
        return ExplainReport(
            bias_report_uri=str(out.get("bias_report", "")),
            shap_uri=str(out.get("shap_uri", "")),
            processing_job=str(out.get("processing_job", "")),
            analysis=str(out.get("analysis", "")),
            detail=dict(out.get("config", {})),
        )


class InProcessSHAPBackend:
    """Local ExplainabilityBackend — real in-process SHAP (gap-closure §G1).

    Computes genuine global SHAP feature attributions over a small estimator + dataset
    entirely offline. Resolution of the estimator mirrors the SEC-02-guarded serving path
    (``deployment.py`` ``_resolve_predict``): an explicit ``model`` object, a ``predict_fn``
    callable, or a reviewed ``.skops`` artifact (loaded under a trusted root). A tree model takes
    the fast ``TreeExplainer`` path; anything model-agnostic falls back to ``KernelExplainer``.

    Returns the SAME :class:`ExplainReport` field surface ``ClarifyBackend`` returns (so the
    adapters are interchangeable behind :class:`ExplainabilityBackend`): a populated local
    ``shap_uri`` (CSV of per-feature mean-abs attributions) + ``analysis`` + a ``detail`` bag
    carrying ``mean_abs_shap`` (the gate evidence) + the explainer kind.
    """

    #: KernelExplainer evaluation caps — keep tiny + deterministic (macOS-aarch64 safe,
    #: respects the minimal decorator single-thread OpenMP pin). Full coalition enumeration for a
    #: handful of features is exact (no sampling randomness), so attributions are stable.
    _MAX_EVAL_ROWS = 200
    _MAX_BACKGROUND = 16

    def bias_and_shap(
        self, *, dataset_uri: str, target: str,
        sensitive_cols: list[str] | None = None, model_ref: ModelRef | None = None,
        output_uri: str | None = None,
    ) -> ExplainReport:
        """Compute offline SHAP attributions → write a local ``shap_uri`` → ExplainReport.

        Args:
            dataset_uri: Local path (or ``s3://`` URI when paired with a local mirror) of the
                analysis CSV. The ``target`` column is dropped; the remaining numeric columns
                are the features SHAP attributes over, in column order.
            target: The label column name to exclude from the feature matrix.
            sensitive_cols: Optional facet columns — when supplied, a lightweight in-process
                bias summary (per-group positive rate) is written + ``analysis="both"``.
            model_ref: Carries the estimator — ``extra["model"]`` (object), ``extra["predict_fn"]``
                (callable), or a ``.skops`` ``uri`` (SEC-02-guarded load).
            output_uri: Optional explicit local CSV path for the attribution output (else a
                tempfile is used).

        Returns:
            An :class:`ExplainReport` with a populated ``shap_uri`` + ``detail["mean_abs_shap"]``.
        """
        import numpy as np
        import pandas as pd

        features, X = self._load_features(dataset_uri, target, pd, np)
        model_obj, predict_fn = self._resolve_estimator(model_ref)

        mean_abs, explainer_kind = self._compute_mean_abs_shap(
            model_obj=model_obj, predict_fn=predict_fn, X=X, n_features=len(features), np=np
        )
        attributions = {feat: float(val) for feat, val in zip(features, mean_abs, strict=False)}

        shap_uri = self._write_attribution_csv(attributions, output_uri)

        bias_report_uri = ""
        bias_summary: dict[str, Any] = {}
        if sensitive_cols:
            bias_summary = self._bias_summary(dataset_uri, target, sensitive_cols, pd)
            bias_report_uri = self._write_bias_csv(bias_summary, output_uri)

        analysis = "both" if sensitive_cols else "explainability"
        return ExplainReport(
            bias_report_uri=bias_report_uri,
            shap_uri=shap_uri,
            processing_job="",  # in-process — no managed Processing job
            analysis=analysis,
            detail={
                "backend": "inprocess_shap",
                "explainer": explainer_kind,
                "mean_abs_shap": attributions,
                "features": list(features),
                "n_rows": int(X.shape[0]),
                "bias_summary": bias_summary,
            },
        )

    # ── internals ────────────────────────────────────────────────────────────
    @staticmethod
    def _load_features(dataset_uri: str, target: str, pd: Any, np: Any) -> tuple[list[str], Any]:
        """Read the CSV, drop the target, keep numeric feature columns (column order preserved)."""
        df = pd.read_csv(dataset_uri)
        if target in df.columns:
            df = df.drop(columns=[target])
        numeric = df.select_dtypes(include=[np.number])
        if numeric.shape[1] == 0:
            raise RuntimeError(
                f"InProcessSHAPBackend: no numeric feature columns in {dataset_uri!r} after dropping "
                f"target {target!r}"
            )
        features = list(numeric.columns)
        X = numeric.to_numpy(dtype=float)
        return features, X

    def _resolve_estimator(self, model_ref: ModelRef | None) -> tuple[Any, Callable[[Any], Any] | None]:
        """Resolve a model object and/or a predict callable from the ModelRef."""
        extra = dict(getattr(model_ref, "extra", {}) or {})
        model_obj = extra.get("model")
        predict_fn = extra.get("predict_fn")
        if callable(predict_fn):
            return model_obj, predict_fn
        if model_obj is not None:
            return model_obj, None
        uri = getattr(model_ref, "uri", "") if model_ref is not None else ""
        if uri and str(uri).endswith(".skops"):
            return self._load_artifact(str(uri)), None
        raise RuntimeError(
            "InProcessSHAPBackend cannot resolve an estimator: pass model_ref.extra['model'] "
            "(a fitted estimator), model_ref.extra['predict_fn'] (callable), or a .skops model_ref.uri"
        )

    @staticmethod
    def _load_artifact(uri: str) -> Any:
        """Load a reviewed estimator under the shared SEC-02 artifact boundary."""
        from ...security.model_artifacts import UntrustedArtifactError, load_model_artifact
        from .deployment import _trusted_model_roots

        last_err: UntrustedArtifactError | None = None
        for root in _trusted_model_roots():
            try:
                return load_model_artifact(uri, trusted_root=root)
            except UntrustedArtifactError as error:
                last_err = error
        raise UntrustedArtifactError(
            f"refusing to load .skops {uri!r}: it fails every configured trusted root "
            "(set AMLAS_TRUSTED_MODEL_ROOT to the artifact staging/registry directory)"
        ) from last_err

    def _compute_mean_abs_shap(
        self, *, model_obj: Any, predict_fn: Callable[[Any], Any] | None, X: Any,
        n_features: int, np: Any,
    ) -> tuple[Any, str]:
        """Compute per-feature mean-abs SHAP (TreeExplainer fast-path; KernelExplainer fallback)."""
        import shap

        # 1) tree fast-path — a model object with no overriding predict_fn.
        if model_obj is not None and predict_fn is None:
            try:
                explainer = shap.TreeExplainer(model_obj)
                values = explainer.shap_values(X, check_additivity=False)
                return self._collapse_to_features(values, n_features, np), "tree"
            except Exception:
                if hasattr(model_obj, "predict"):
                    predict_fn = model_obj.predict
                else:
                    raise

        # 2) model-agnostic KernelExplainer fallback (any predict callable).
        predict = predict_fn if callable(predict_fn) else getattr(model_obj, "predict", None)
        if not callable(predict):
            raise RuntimeError("InProcessSHAPBackend: no predict callable available for SHAP")
        np.random.seed(0)  # determinism (with full coalition enumeration this is already exact)
        eval_rows = X[: min(self._MAX_EVAL_ROWS, X.shape[0])]
        background = X[: min(self._MAX_BACKGROUND, X.shape[0])]
        explainer = shap.KernelExplainer(predict, background, silent=True)
        values = explainer.shap_values(eval_rows, nsamples="auto", l1_reg=0, silent=True)
        return self._collapse_to_features(values, n_features, np), "kernel"

    @staticmethod
    def _collapse_to_features(values: Any, n_features: int, np: Any) -> Any:
        """Reduce any SHAP value shape (list / 2-D / 3-D multiclass) → one mean-abs per feature.

        SHAP returns different shapes by model + version: a 2-D ``(n_rows, n_features)`` for a
        single output, a list of such arrays per class, or a 3-D ``(n_rows, n_features, n_classes)``.
        We take ``abs`` then average over every axis EXCEPT the feature axis (the axis whose size
        equals ``n_features``), giving a stable per-feature global importance.
        """
        if isinstance(values, list):
            values = np.stack([np.asarray(v, dtype=float) for v in values])
        arr = np.abs(np.asarray(values, dtype=float))
        # locate the feature axis (first axis whose length == n_features)
        feature_axis = next((ax for ax, size in enumerate(arr.shape) if size == n_features), arr.ndim - 1)
        other_axes = tuple(ax for ax in range(arr.ndim) if ax != feature_axis)
        return arr.mean(axis=other_axes) if other_axes else arr

    @staticmethod
    def _write_attribution_csv(attributions: dict[str, float], output_uri: str | None) -> str:
        """Write the per-feature mean-abs attributions to a local CSV; return its path/URI."""
        path = output_uri
        if not path or str(path).startswith("s3://"):
            fd, tmp = tempfile.mkstemp(prefix="amlas-inproc-shap-", suffix=".csv")
            os.close(fd)
            path = tmp
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)  # nosemgrep: use-defusedcsv
            w.writerow(["feature", "mean_abs_shap"])
            for feat, val in attributions.items():
                w.writerow([_csv_safe_cell(feat), val])
        return path

    @staticmethod
    def _bias_summary(dataset_uri: str, target: str, sensitive_cols: list[str], pd: Any) -> dict[str, Any]:
        """A lightweight in-process bias summary: positive-label rate per sensitive group."""
        df = pd.read_csv(dataset_uri)
        out: dict[str, Any] = {}
        if target not in df.columns:
            return out
        for col in sensitive_cols:
            if col not in df.columns:
                continue
            rates = df.groupby(col)[target].mean().to_dict()
            out[col] = {str(k): float(v) for k, v in rates.items()}
        return out

    @staticmethod
    def _write_bias_csv(bias_summary: dict[str, Any], output_uri: str | None) -> str:
        """Write the per-group positive-rate bias summary to a local CSV; return its path."""
        fd, path = tempfile.mkstemp(prefix="amlas-inproc-bias-", suffix=".csv")
        os.close(fd)
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)  # nosemgrep: use-defusedcsv
            w.writerow(["facet", "group", "positive_rate"])
            for facet, groups in bias_summary.items():
                for group, rate in groups.items():
                    w.writerow([_csv_safe_cell(facet), _csv_safe_cell(group), rate])
        return path


def build_explainability_evidence(
    *, dataset_uri: str, target: str,
    sensitive_cols: list[str] | None = None, model_ref: ModelRef | None = None,
    steering: Any | None = None, output_uri: str | None = None,
) -> dict[str, Any]:
    """Stage-9 wiring helper — run the resolved ExplainabilityBackend → readiness gate evidence.

    Resolves :class:`ExplainabilityBackend` through the :class:`BackendRegistry` (offline/under
    offline → the local ``inprocess_shap`` adapter; ``EXPLAINABILITY_BACKEND=clarify`` → managed
    Clarify), runs ``bias_and_shap``, and shapes the result into the ``evidence["explainability"]``
    block ``readiness_flow``'s Explainability gate consumes (``{"shap_uri": ...}``) — so the gate
    is satisfiable fully offline (gap-closure §G1).

    Returns:
        ``{"shap_uri","bias_report_uri","analysis","adapter","reason","explain_report"}`` where
        ``explain_report`` is the backend's ``detail`` bag (carrying ``mean_abs_shap``).
    """
    resolved = REGISTRY.resolve(ExplainabilityBackend, steering=steering)
    adapter = resolved.adapter
    kwargs: dict[str, Any] = {
        "dataset_uri": dataset_uri,
        "target": target,
        "sensitive_cols": sensitive_cols,
        "model_ref": model_ref,
    }
    # the local adapter accepts an explicit output_uri; the managed one ignores extras.
    if output_uri is not None and isinstance(adapter, InProcessSHAPBackend):
        kwargs["output_uri"] = output_uri
    report = adapter.bias_and_shap(**kwargs)
    return {
        "shap_uri": report.shap_uri,
        "bias_report_uri": report.bias_report_uri,
        "analysis": report.analysis,
        "adapter": resolved.adapter_name,
        "reason": resolved.reason,
        "explain_report": dict(report.detail),
    }


register_adapter(ExplainabilityBackend, "clarify", lambda: ClarifyBackend(), is_default=True)
register_adapter(ExplainabilityBackend, "inprocess_shap", lambda: InProcessSHAPBackend(), is_local=True)
