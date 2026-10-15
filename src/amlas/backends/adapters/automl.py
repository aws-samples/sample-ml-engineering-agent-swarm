"""AutoMLBackend adapters (doc 05 §4.2 G1 / roadmap M3; ADR-004 / ADR-005).

AutoML candidate generation behind the
:class:`~amlas.backends.ports.AutoMLBackend` port (``run(...)``). Two
adapters, scale/budget-selected by the registry:

* ``LocalSweepBackend`` — the **local/offline default** (auto-resolves offline). A tiny,
  seeded "quick sweep" that trains a small fixed shortlist of cheap algorithms via the
  EXISTING ``train_model`` IP on tiny data and returns the best candidate + a ranked
  leaderboard. Kept tiny + single-threaded to avoid the macOS-aarch64 sklearn/joblib
  deadlock (RECON / ``validation coverage``). Honors ``generate_only`` (a dry-run that
  returns the candidate plan WITHOUT training — the cheap offline baseline, US-3.06).
* ``AutopilotV2Backend`` — the **default (managed)**; orchestrates SageMaker Autopilot /
  AutoML v2 via boto3 ``sagemaker`` ``create_auto_ml_job_v2`` +
  ``describe_auto_ml_job_v2`` — a thin orchestration that NEVER reimplements Autopilot
  (ADR-004). Honors ``generate_only`` (``Mode="ENSEMBLING"`` with candidate-generation
  dry-run / ``GenerateCandidateDefinitionsOnly``). The client is injectable for offline substitutes
  tests. T-AML-01 ``run_automl``.

Selector (``AUTOML_BACKEND``): ``local`` (sweep), ``autopilot`` (managed default).
Offline auto-resolves to ``local`` (doc 05 §4.3).
"""

from __future__ import annotations

import re
import uuid
from typing import Any, ClassVar

from ...config.aws_region import resolve_aws_region
from ..ports import AutoMLBackend
from ..registry import register_adapter
from ._util import raw_tool

# The cheap, fast shortlist the local sweep tries (single-threaded, tiny). Ordered
# from cheapest → richer; the sweep trains each and ranks by the objective metric.
_DEFAULT_SWEEP_ALGOS = ("logistic_regression", "decision_tree", "random_forest")
_DEFAULT_SWEEP_ALGOS_REGRESSION = ("linear_regression", "decision_tree", "random_forest")

_MINIMIZE_HINTS = ("rmse", "mae", "mse", "loss", "error", "logloss", "mape", "wape")


def _infer_direction(metric: str) -> str:
    """Return ``"minimize"`` / ``"maximize"`` for an objective metric name."""
    name = (metric or "").strip().lower()
    return "minimize" if any(h in name for h in _MINIMIZE_HINTS) else "maximize"


def _sanitize_name(prefix: str, raw: str) -> str:
    """A SageMaker-safe AutoML job name (alnum + hyphen, no underscores)."""
    base = re.sub(r"[^a-zA-Z0-9]+", "-", raw or "automl").strip("-")[:20] or "automl"
    return f"{prefix}-{base}-{uuid.uuid4().hex[:8]}"


class LocalSweepBackend:
    """Local/offline AutoMLBackend — a tiny seeded quick-sweep over the ``train_model`` IP.

    ``run`` trains a small fixed shortlist of cheap algorithms (single-threaded, tiny)
    and returns the best candidate + a ranked leaderboard. ``generate_only`` short-circuits
    to a candidate PLAN (no training) — the cheap offline baseline (US-3.06). No AWS.
    """

    backend_name = "local"
    MAX_CANDIDATES = 5  # hard cap so a big max_candidates never triggers the host deadlock

    def __init__(self, *, seed: int = 42) -> None:
        self._seed = seed

    def _shortlist(self, problem_type: str, max_candidates: int) -> list[str]:
        """The algorithm shortlist for ``problem_type``, capped tiny."""
        pt = (problem_type or "classification").strip().lower()
        base = _DEFAULT_SWEEP_ALGOS_REGRESSION if "regress" in pt else _DEFAULT_SWEEP_ALGOS
        n = max(1, min(int(max_candidates), self.MAX_CANDIDATES, len(base)))
        return list(base[:n])

    def run(
        self,
        *,
        train_uri: str,
        target: str,
        problem_type: str,
        mode: str = "ensembling",
        max_candidates: int = 10,
        generate_only: bool = False,
    ) -> dict[str, Any]:
        """Run a tiny quick-sweep AutoML baseline (T-AML-01), or a generate-only plan."""
        algos = self._shortlist(problem_type, max_candidates)

        if generate_only:
            # Dry-run: return the candidate PLAN without training (cheap baseline).
            return {
                "success": True,
                "backend": "local",
                "generate_only": True,
                "problem_type": problem_type,
                "candidates": [{"algorithm": a, "status": "proposed"} for a in algos],
                "n_candidates": len(algos),
                "best_candidate": None,
            }

        pt = (problem_type or "classification").strip().lower()
        task_type = "regression" if "regress" in pt else "classification"
        # The default objective metric per task (the sweep ranks on it).
        objective = "rmse" if task_type == "regression" else "accuracy"
        direction = _infer_direction(objective)

        candidates: list[dict[str, Any]] = []
        best: dict[str, Any] | None = None
        best_score: float | None = None
        for algo in algos:
            try:
                res = raw_tool(_train_model())(
                    data_path=train_uri,
                    target_column=target,
                    algorithm=algo,
                    params={},
                    output_dir=None,
                    seed=self._seed,
                    test_size=0.3,
                    task_type=task_type,
                )
            except Exception as e:
                candidates.append({"algorithm": algo, "status": "failed", "error": str(e)})
                continue
            metrics = res.get("metrics", {}) if isinstance(res, dict) else {}
            score = float(metrics[objective]) if objective in metrics else _first_metric(metrics)
            cand = {"algorithm": algo, "status": "trained", "score": score, "metrics": metrics}
            candidates.append(cand)
            if score is not None and (
                best_score is None or (score < best_score if direction == "minimize" else score > best_score)
            ):
                best_score, best = score, cand

        # Rank the trained candidates by score (best first).
        trained = [c for c in candidates if c.get("score") is not None]
        trained.sort(key=lambda c: c["score"], reverse=(direction == "maximize"))
        return {
            "success": best is not None,
            "backend": "local",
            "generate_only": False,
            "problem_type": problem_type,
            "objective": objective,
            "direction": direction,
            "best_candidate": best,
            "leaderboard": trained,
            "candidates": candidates,
            "n_candidates": len(candidates),
        }


def _train_model() -> Any:
    """Import the EXISTING ``train_model`` lazily (keeps module import boto3/sklearn-free)."""
    from ...tools.training_tools import train_model

    return train_model


def _first_metric(metrics: dict[str, Any]) -> float | None:
    """Return the first numeric metric value (sweep fallback when objective absent)."""
    for v in metrics.values():
        if isinstance(v, (int, float)):
            return float(v)
    return None


class AutopilotV2Backend:
    """Managed AutoMLBackend — orchestrates SageMaker Autopilot / AutoML v2 (T-AML-01).

    Thin boto3 ``sagemaker`` orchestration: build the AutoML-v2 config from the spec,
    ``create_auto_ml_job_v2`` (honoring ``generate_only`` →
    ``GenerateCandidateDefinitionsOnly``), ``describe_auto_ml_job_v2`` for status/best.
    Autopilot runs candidate generation + ensembling — we never reimplement it (ADR-004).
    The client is injectable for offline substitutes tests.
    """

    backend_name = "autopilot"

    _PROBLEM_TYPE: ClassVar[dict[str, str]] = {
        "classification": "BinaryClassification",
        "binary": "BinaryClassification",
        "binary_classification": "BinaryClassification",
        "multiclass": "MulticlassClassification",
        "multiclass_classification": "MulticlassClassification",
        "regression": "Regression",
    }

    def __init__(
        self,
        *,
        region: str | None = None,
        role_arn: str = "",
        sagemaker_client: Any = None,
    ) -> None:
        self._region = resolve_aws_region(region, required=False)
        self._role = role_arn or "arn:aws:iam::000000000000:role/autopilot"
        self._sm = sagemaker_client

    def _client(self) -> Any:
        """Lazily build (or return the injected) boto3 ``sagemaker`` client."""
        if self._sm is None:
            import boto3

            self._sm = boto3.client("sagemaker", region_name=resolve_aws_region(self._region))
        return self._sm

    def run(
        self,
        *,
        train_uri: str,
        target: str,
        problem_type: str,
        mode: str = "ensembling",
        max_candidates: int = 10,
        generate_only: bool = False,
    ) -> dict[str, Any]:
        """Submit a SageMaker Autopilot AutoML-v2 job + return its handle/status (T-AML-01)."""
        sm = self._client()
        job_name = _sanitize_name("automl", problem_type)
        pt = self._PROBLEM_TYPE.get((problem_type or "").strip().lower(), "BinaryClassification")
        # SEC-01: the no-separator fallback uses an invalid-by-construction bucket name
        # (angle brackets are illegal in S3 bucket names) so the default can never be
        # squatted; real callers always pass a concrete ``train_uri`` with a path.
        output_uri = train_uri.rsplit("/", 1)[0] + "/automl-output" if "/" in train_uri else "s3://<your-automl-output-bucket>/automl-output"
        auto_ml_mode = "ENSEMBLING" if str(mode).lower() in ("ensembling", "ensemble") else "HYPERPARAMETER_TUNING"

        problem_config: dict[str, Any] = {
            "TabularJobConfig": {
                "TargetAttributeName": target,
                "ProblemType": pt,
                "Mode": auto_ml_mode,
                "GenerateCandidateDefinitionsOnly": bool(generate_only),
                "CandidateGenerationConfig": {},
                "CompletionCriteria": {"MaxCandidates": int(max_candidates)},
            }
        }

        kwargs: dict[str, Any] = {
            "AutoMLJobName": job_name,
            "AutoMLProblemTypeConfig": problem_config,
            "AutoMLJobInputDataConfig": [
                {
                    "ChannelType": "training",
                    "ContentType": "text/csv;header=present",
                    "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": train_uri}},
                }
            ],
            "OutputDataConfig": {"S3OutputPath": output_uri},
            "RoleArn": self._role,
        }
        sm.create_auto_ml_job_v2(**kwargs)
        desc = sm.describe_auto_ml_job_v2(AutoMLJobName=job_name)
        best = desc.get("BestCandidate", {}) or {}
        return {
            "success": True,
            "backend": "autopilot",
            "managed": True,
            "generate_only": bool(generate_only),
            "problem_type": pt,
            "mode": auto_ml_mode,
            "automl_job_name": job_name,
            "status": desc.get("AutoMLJobStatus", "InProgress"),
            "arn": desc.get("AutoMLJobArn", ""),
            "best_candidate": {"name": best.get("CandidateName", "")} if best else None,
            "output_uri": output_uri,
        }


# ── self-registration (the registry imports this module lazily) ───────────────
# Autopilot is the AWS default; the local quick-sweep is the offline/local default.
# Offline auto → local (doc 05 §4.3); scale/budget selects Autopilot for cloud.
register_adapter(AutoMLBackend, "autopilot", lambda: AutopilotV2Backend(), is_default=True)
register_adapter(AutoMLBackend, "local", lambda: LocalSweepBackend(), is_local=True)
