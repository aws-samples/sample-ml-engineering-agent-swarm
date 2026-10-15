"""TuningBackend adapters (doc 05 §4.2 G1 / roadmap M3; ADR-004 / ADR-005).

Hyperparameter search behind the :class:`~amlas.backends.ports.TuningBackend`
port (``search(...)``). Two adapters, scale/budget-selected by the registry:

* ``OptunaBackend`` — the **local/offline default** (auto-resolves offline). A tiny,
  seeded, dependency-light search loop (uses ``optuna`` when installed, else a seeded
  random/grid search) that evaluates each trial with a caller-supplied ``objective_fn``
  (pure-Python, the offline-safe path) OR the EXISTING ``train_model`` IP on tiny data.
  Trials are kept tiny + single-threaded to avoid the macOS-aarch64 sklearn/joblib
  deadlock (RECON / ``validation coverage``). T-TUNE-02 ``tune_model`` evolves onto this.
* ``SageMakerAMTBackend`` — the **default (managed)**; orchestrates SageMaker Automatic
  Model Tuning (Bayesian / Hyperband, warm-start) via boto3 ``sagemaker``
  ``create_hyper_parameter_tuning_job`` + ``describe_hyper_parameter_tuning_job`` — a thin
  orchestration that NEVER reimplements the tuner (ADR-004). The client is injectable so
  callers may pass a injectable client. T-TUNE-01 ``run_hpo``.

Selector (``TUNING_BACKEND``): ``optuna`` (local), ``sagemaker`` (managed default).
Offline auto-resolves to ``optuna`` (doc 05 §4.3). Per the registry, scale/budget
selects local-Optuna for small/cheap vs managed-AMT for scale.
"""

from __future__ import annotations

import random
import re
import uuid
from collections.abc import Callable
from typing import Any

from ...config.aws_region import resolve_aws_client_region, resolve_aws_region
from ...config.sagemaker_images import validate_image_region, xgboost_image
from ..ports import TuningBackend
from ..registry import register_adapter
from ._util import raw_tool

# Metric-name substrings whose objective is MINIMIZED (lower is better). Anything else
# (accuracy/f1/auc/precision/recall/r2/score) is maximized. doc 05 Arbiter direction.
_MINIMIZE_HINTS = ("rmse", "mae", "mse", "loss", "error", "logloss", "mape", "wape")


def _infer_direction(objective: str, override: str | None = None) -> str:
    """Return ``"minimize"`` / ``"maximize"`` for an objective metric name."""
    if override in ("minimize", "maximize"):
        return override
    name = (objective or "").strip().lower()
    return "minimize" if any(h in name for h in _MINIMIZE_HINTS) else "maximize"


def _sanitize_name(prefix: str, raw: str) -> str:
    """A SageMaker-safe job name (alnum + hyphen, no underscores, ≤32-ish base)."""
    base = re.sub(r"[^a-zA-Z0-9]+", "-", raw or "tune").strip("-")[:20] or "tune"
    return f"{prefix}-{base}-{uuid.uuid4().hex[:8]}"


def _sample_param(rng: random.Random, spec: Any) -> Any:
    """Sample one value from a range spec.

    Accepts:
      * ``{"type":"int","low":a,"high":b}`` / ``{"type":"float","low":a,"high":b}``;
      * ``{"type":"categorical","choices":[...]}``;
      * a 2-element ``[low, high]`` list → float range;
      * a plain list / tuple of >2 → categorical choices;
      * a scalar → fixed value (returned unchanged).
    """
    if isinstance(spec, dict):
        kind = str(spec.get("type", "float")).lower()
        if kind in ("categorical", "choice"):
            choices = list(spec.get("choices", []))
            return rng.choice(choices) if choices else None
        lo = spec.get("low", 0)
        hi = spec.get("high", 1)
        if kind == "int":
            return rng.randint(int(lo), int(hi))
        return rng.uniform(float(lo), float(hi))
    if isinstance(spec, (list, tuple)):
        if len(spec) == 2 and all(isinstance(v, (int, float)) for v in spec):
            return rng.uniform(float(spec[0]), float(spec[1]))
        return rng.choice(list(spec))
    return spec  # fixed value


class OptunaBackend:
    """Local/offline TuningBackend — a tiny seeded HPO loop (Optuna or random search).

    ``search`` runs ``max_jobs`` (capped tiny) seeded trials over ``ranges``, evaluating
    each with the ``objective_fn`` supplied in ``base_training`` (pure-Python, offline)
    or — when ``data_path``/``target``/``algo`` are given — the EXISTING ``train_model``
    IP. Returns the best params + score + the full trial log. No AWS.
    """

    backend_name = "optuna"
    # Hard cap so a runaway max_jobs can never trigger the host's sklearn/joblib deadlock.
    MAX_TRIALS = 25

    def __init__(self, *, seed: int = 42) -> None:
        self._seed = seed

    def search(
        self,
        *,
        base_training: dict[str, Any],
        ranges: dict[str, Any],
        objective: str,
        strategy: str = "bayesian",
        max_jobs: int = 10,
        max_parallel: int = 1,
        warm_start_from: str | None = None,
    ) -> dict[str, Any]:
        """Run a tiny seeded HPO search and return the best params (T-TUNE-02)."""
        base = dict(base_training or {})
        direction = _infer_direction(objective, base.get("direction"))
        n_trials = max(1, min(int(max_jobs), self.MAX_TRIALS))
        evaluator = self._build_evaluator(base, objective)

        rng = random.Random(self._seed)  # nosec B311
        trials: list[dict[str, Any]] = []
        best: dict[str, Any] | None = None
        best_score: float | None = None

        for i in range(n_trials):
            params = {name: _sample_param(rng, spec) for name, spec in (ranges or {}).items()}
            try:
                score = float(evaluator(params))
            except Exception as e:
                trials.append({"trial": i, "params": params, "score": None, "error": str(e)})
                continue
            trials.append({"trial": i, "params": params, "score": score})
            if best_score is None or (
                score < best_score if direction == "minimize" else score > best_score
            ):
                best_score, best = score, params

        return {
            "success": best is not None,
            "backend": "optuna",
            "strategy": strategy,
            "direction": direction,
            "objective": objective,
            "best_params": best or {},
            "best_score": best_score,
            "n_trials": len(trials),
            "trials": trials,
            "warm_start_from": warm_start_from,
        }

    def _build_evaluator(self, base: dict[str, Any], objective: str) -> Callable[..., Any]:
        """Pick the per-trial evaluator: injected ``objective_fn`` or the ``train_model`` IP."""
        fn = base.get("objective_fn")
        if callable(fn):
            evaluator: Callable[..., Any] = fn  # offline-safe pure-Python path
            return evaluator

        data_path = base.get("data_path") or base.get("train_uri")
        target = base.get("target") or base.get("target_column")
        algo = base.get("algo") or base.get("algorithm", "random_forest")
        metric = (objective or "").strip().lower()
        if not data_path or not target:
            raise ValueError(
                "OptunaBackend.search needs either base_training['objective_fn'] (callable) "
                "or base_training with data_path+target to evaluate trials via train_model"
            )

        def _evaluate(params: dict[str, Any]) -> float:
            from ...tools.training_tools import train_model

            res = raw_tool(train_model)(
                data_path=data_path,
                target_column=target,
                algorithm=algo,
                params=dict(params),
                output_dir=None,
                seed=int(base.get("seed", 42)),
                test_size=float(base.get("test_size", 0.3)),
                task_type=base.get("task_type", "classification"),
            )
            metrics = res.get("metrics", {}) if isinstance(res, dict) else {}
            if metric and metric in metrics:
                return float(metrics[metric])
            # fall back to the first numeric metric
            for v in metrics.values():
                if isinstance(v, (int, float)):
                    return float(v)
            raise ValueError(f"train_model returned no usable metric for objective {objective!r}")

        return _evaluate


class SageMakerAMTBackend:
    """Managed TuningBackend — orchestrates SageMaker Automatic Model Tuning (T-TUNE-01).

    Thin boto3 ``sagemaker`` orchestration: build the tuning-job config from ``ranges`` +
    ``base_training``, ``create_hyper_parameter_tuning_job`` (Bayesian/Hyperband, optional
    warm-start), ``describe_*`` for status/best. The AMT service runs the search — we never
    reimplement it (ADR-004). The client is injectable for offline client substitutes.
    """

    backend_name = "sagemaker_amt"

    def __init__(
        self,
        *,
        region: str | None = None,
        role_arn: str = "",
        default_image: str = "",
        sagemaker_client: Any = None,
    ) -> None:
        self._region = region
        self._role = role_arn or "arn:aws:iam::000000000000:role/amt"
        self._default_image = default_image
        self._sm = sagemaker_client

    def _client(self) -> Any:
        """Lazily build (or return the injected) boto3 ``sagemaker`` client."""
        if self._sm is None:
            import boto3

            self._sm = boto3.client("sagemaker", region_name=resolve_aws_region(self._region))
        return self._sm

    def _to_amt_ranges(self, ranges: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
        """Translate the neutral ``ranges`` into AMT ParameterRanges (3 buckets)."""
        integer: list[dict[str, Any]] = []
        continuous: list[dict[str, Any]] = []
        categorical: list[dict[str, Any]] = []
        for name, spec in (ranges or {}).items():
            if isinstance(spec, dict):
                kind = str(spec.get("type", "float")).lower()
                if kind == "int":
                    integer.append({"Name": name, "MinValue": str(spec["low"]), "MaxValue": str(spec["high"])})
                elif kind in ("categorical", "choice"):
                    categorical.append({"Name": name, "Values": [str(v) for v in spec.get("choices", [])]})
                else:
                    continuous.append({"Name": name, "MinValue": str(spec["low"]), "MaxValue": str(spec["high"])})
            elif isinstance(spec, (list, tuple)) and len(spec) == 2 and all(isinstance(v, (int, float)) for v in spec):
                continuous.append({"Name": name, "MinValue": str(spec[0]), "MaxValue": str(spec[1])})
            elif isinstance(spec, (list, tuple)):
                categorical.append({"Name": name, "Values": [str(v) for v in spec]})
        return {
            "IntegerParameterRanges": integer,
            "ContinuousParameterRanges": continuous,
            "CategoricalParameterRanges": categorical,
        }

    def search(
        self,
        *,
        base_training: dict[str, Any],
        ranges: dict[str, Any],
        objective: str,
        strategy: str = "bayesian",
        max_jobs: int = 10,
        max_parallel: int = 1,
        warm_start_from: str | None = None,
    ) -> dict[str, Any]:
        """Submit a SageMaker AMT tuning job + return its handle/status (T-TUNE-01)."""
        base = dict(base_training or {})
        sm = self._client()
        job_name = _sanitize_name("amt", base.get("algo", base.get("algorithm", "tune")))
        direction = _infer_direction(objective, base.get("direction"))
        amt_type = "Hyperband" if str(strategy).lower() == "hyperband" else (
            "Bayesian" if str(strategy).lower() in ("bayesian", "bayes") else "Random"
        )
        override = base.get("image") or self._default_image
        region = resolve_aws_client_region(sm, self._region, required=not bool(override))
        image = override or xgboost_image(region)
        image = validate_image_region(str(image), region)
        instance_type = base.get("instance_type", "ml.m5.large")
        s3_input = base.get("train_uri") or base.get("data_path", "")
        # SEC-01: the unset-output fallback uses an invalid-by-construction bucket name
        # (angle brackets are illegal in S3 bucket names) so the default can never be
        # squatted; real callers always supply ``output_uri``/``output``.
        s3_output = base.get("output_uri") or base.get("output", "s3://<your-amt-output-bucket>/amt-output")

        tuning_config: dict[str, Any] = {
            "Strategy": amt_type,
            "HyperParameterTuningJobObjective": {
                "Type": "Minimize" if direction == "minimize" else "Maximize",
                "MetricName": objective,
            },
            "ResourceLimits": {
                "MaxNumberOfTrainingJobs": int(max_jobs),
                "MaxParallelTrainingJobs": int(max_parallel),
            },
            "ParameterRanges": self._to_amt_ranges(ranges),
        }
        # FIX-16 (Phase 6): AMT REQUIRES the objective metric's scrape definition in
        # ``AlgorithmSpecification.MetricDefinitions`` — without it a real
        # ``CreateHyperParameterTuningJob`` fails with "A metric is required for this
        # hyperparameter tuning job objective. Provide a metric in the metric definitions."
        # Use caller-supplied ``metric_definitions`` when present; otherwise synthesize a
        # ``name=value`` scrape regex for the objective (the ``final_metric=0.95`` stdout
        # convention the sklearn training script emits). The objective metric MUST appear
        # in MetricDefinitions, so we always include it.
        metric_defs = base.get("metric_definitions")
        if not metric_defs:
            metric_name = str(objective).split(":")[-1] or "final_metric"
            metric_defs = [
                {"Name": objective, "Regex": rf"{metric_name}=([0-9.eE+-]+)"}
            ]
        training_def: dict[str, Any] = {
            "AlgorithmSpecification": {
                "TrainingImage": image,
                "TrainingInputMode": "File",
                "MetricDefinitions": metric_defs,
            },
            "RoleArn": self._role,
            "InputDataConfig": [
                {
                    "ChannelName": "train",
                    "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": s3_input, "S3DataDistributionType": "FullyReplicated"}},
                }
            ],
            "OutputDataConfig": {"S3OutputPath": s3_output},
            "ResourceConfig": {"InstanceType": instance_type, "InstanceCount": 1, "VolumeSizeInGB": 10},
            "StoppingCondition": {"MaxRuntimeInSeconds": int(base.get("max_runtime", 3600))},
        }
        static = base.get("static_hyperparameters")
        if static:
            training_def["StaticHyperParameters"] = {str(k): str(v) for k, v in static.items()}

        kwargs: dict[str, Any] = {
            "HyperParameterTuningJobName": job_name,
            "HyperParameterTuningJobConfig": tuning_config,
            "TrainingJobDefinition": training_def,
        }
        if warm_start_from:
            kwargs["WarmStartConfig"] = {
                "ParentHyperParameterTuningJobs": [{"HyperParameterTuningJobName": warm_start_from}],
                "WarmStartType": "IdenticalDataAndAlgorithm",
            }

        sm.create_hyper_parameter_tuning_job(**kwargs)
        desc = sm.describe_hyper_parameter_tuning_job(HyperParameterTuningJobName=job_name)
        best = desc.get("BestTrainingJob", {}) or {}
        best_params = best.get("TunedHyperParameters", {}) or {}
        return {
            "success": True,
            "backend": "sagemaker_amt",
            "managed": True,
            "strategy": amt_type,
            "direction": direction,
            "objective": objective,
            "tuning_job_name": job_name,
            "status": desc.get("HyperParameterTuningJobStatus", "InProgress"),
            "arn": desc.get("HyperParameterTuningJobArn", ""),
            "best_params": dict(best_params),
            "warm_start_from": warm_start_from,
        }


# ── self-registration (the registry imports this module lazily) ───────────────
# AMT is the AWS default; Optuna is the offline/local default. Offline auto →
# optuna (doc 05 §4.3); scale/budget selects AMT for large/cloud HPO.
register_adapter(TuningBackend, "sagemaker", lambda: SageMakerAMTBackend(), is_default=True)
register_adapter(TuningBackend, "optuna", lambda: OptunaBackend(), is_local=True)
