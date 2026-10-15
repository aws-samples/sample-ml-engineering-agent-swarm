"""Real training-job @tool wrappers — the de-mock replacement for ``tools/stubs.py``.

Phase 5 of the de-mock initiative DELETED the three fake-data tools that shipped in
``tools/stubs.py`` and replaced them with REAL integrations:

  * ``submit_sagemaker_job``   — submits a REAL async SageMaker training job (real
    ``CreateTrainingJob`` via :func:`amlas.tools.sagemaker.submit_training_job`,
    real DynamoDB PENDING→RUNNING). The old stub fabricated a ``job_name``/``RUNNING``
    without touching AWS (PS-01).
  * ``fetch_training_metrics`` — reads REAL metrics back from the training job via
    ``describe_training_job`` (``FinalMetricDataList`` + status + artifacts). The old
    ``mlflow_fetch_metrics`` stub returned a fixed metrics dict from a non-existent
    MLflow server (PS-03; MLflow dropped per OPEN DECISION 6).
  * ``submit_training_trial``  — runs a REAL scikit-learn cross-validation trial on a
    REAL dataset and returns the REAL mean CV score. The old stub synthesized a score
    from ``hash(params)`` (PS-02).

No fabricated data is produced anywhere in this module. The boto3 clients are
injectable (``*_client=``) for real-AWS-backed tests; in prod they default to real
``boto3.client(...)`` built inside :mod:`amlas.tools.sagemaker`.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict
from typing import Any

from ..config.aws_region import resolve_aws_region

# FIX-34: strands is absent in the stock SageMaker framework container; the off-box entrypoint
# imports _cv_scoring from here, so degrade @tool to an identity decorator when strands is missing
# (the AgentCore runtime has strands and uses the real decorator, unchanged).
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


from amlas.config._env_compat import env_with_compat

from ..errors import tool_error_boundary
from .sagemaker import (
    TERMINAL_TRAINING_STATES,
    SageMakerSubmitError,
    get_training_metrics,
    submit_training_job,
    wait_for_training_job,
)


def _resolve_execution_role(config: dict[str, Any]) -> str:
    """Resolve the SageMaker exec-role ARN: config → env. Empty string if unset."""
    return (
        config.get("execution_role_arn")
        or os.environ.get("SAGEMAKER_EXECUTION_ROLE_ARN")
        or env_with_compat("AMLAS_SAGEMAKER_ROLE_ARN")
        or ""
    )


def _resolve_session_bucket(config: dict[str, Any]) -> str:
    """Resolve the session/artifacts bucket: config → env. Empty string if unset."""
    return (
        config.get("session_bucket")
        or env_with_compat("AMLAS_SESSION_BUCKET")
        or os.environ.get("SESSION_BUCKET")
        or ""
    )


@tool
@tool_error_boundary(source="tools.training_jobs")
def submit_sagemaker_job(
    training_type: str,
    config: dict[str, Any],
    runtime_session_id: str = "",
    region: str | None = None,
) -> dict[str, Any]:
    """Submit a REAL async SageMaker training job and return its real metadata.

    Drives the canonical 3-step submit sequence (real DynamoDB PENDING row → real
    ``CreateTrainingJob`` → DynamoDB RUNNING). The SageMaker exec-role ARN and the
    session/artifacts bucket are resolved from ``config`` or the environment
    (``SAGEMAKER_EXECUTION_ROLE_ARN`` / ``AMLAS_SESSION_BUCKET``); a clear error is
    raised when they cannot be resolved (NEVER fabricates a job).

    Args:
        training_type: one of "sft", "dpo", "grpo", "xgboost", "sklearn", etc.
            (GPU training types route to a GPU image/instance automatically).
        config: training config — may carry ``execution_role_arn``, ``session_bucket``,
            ``instance_type``, ``image_uri``, ``s3_input``, ``hyperparameters``,
            ``metric_definitions``, ``max_runtime_seconds``.
        runtime_session_id: AgentCore runtime session id to resume via the callback
            Lambda; defaults to a generated id when empty.
        region: AWS region.
    """
    role = _resolve_execution_role(config)
    bucket = _resolve_session_bucket(config)
    if not role or not bucket:
        raise RuntimeError(
            "submit_sagemaker_job: cannot submit a real SageMaker job — "
            "execution_role_arn and session_bucket must be resolvable from config or "
            "env (SAGEMAKER_EXECUTION_ROLE_ARN / AMLAS_SESSION_BUCKET). "
            f"resolved role={role!r} bucket={bucket!r}."
        )
    session_id = runtime_session_id or f"session-{training_type}-{int(time.time())}"
    try:
        result = submit_training_job(
            training_type=training_type,
            config=config,
            runtime_session_id=session_id,
            execution_role_arn=role,
            session_bucket=bucket,
            jobs_table=config.get("jobs_table"),
            region=region,
        )
    except SageMakerSubmitError as error:
        if error.accepted_job is None:
            raise
        return dict(
            asdict(error.accepted_job),
            success=True,
            accepted=True,
            tracking_error="native job accepted; DynamoDB tracking update failed",
        )
    return asdict(result)


#: Polls per job before this tool refuses to answer again. A dispatched job that has not
#: terminated in this many observations is not going to be resolved by more looking, and the loop
#: is expensive: each poll is a billed model turn. Measured on a real run before this guard,
#: ``fetch_training_metrics`` accounted for 923 of 1000 logged tool calls (92%) — ~58 polls per
#: job across 16 submissions — and burned the entire 45-minute wall clock without producing a
#: prediction ( Defect B).
MAX_POLLS_PER_JOB = 40

#: Per-job poll counter. Process-scoped on purpose: a run lives in one process, and a fresh
#: process legitimately re-observes a job it did not submit.
_POLL_COUNTS: dict[str, int] = {}


def _reset_poll_counts() -> None:
    """Test seam — clear the per-job poll ledger."""
    _POLL_COUNTS.clear()


@tool
@tool_error_boundary(source="tools.training_jobs")
def fetch_training_metrics(job_name: str, region: str | None = None) -> dict[str, Any]:
    """Fetch REAL training metrics from a SageMaker training job (no MLflow). ONE observation.

    Calls ``describe_training_job`` and returns the real status, the real
    ``FinalMetricDataList`` flattened to ``{metric_name: value}``, the model-artifacts
    S3 URI, and any failure reason. This is the de-mock replacement for the old
    ``mlflow_fetch_metrics`` fake (OPEN DECISION 6).

    This is a SNAPSHOT, not a wait. To wait for a job to finish, call
    ``await_training_job`` ONCE instead of calling this in a loop — it blocks with backoff and
    returns the terminal answer, at the cost of a single model turn.

    The result carries ``terminal`` and ``next_action`` so a caller never has to infer whether
    the job is still running. When ``terminal`` is true the answer will not change: act on
    ``metrics`` / ``model_artifacts``, or on ``failure_reason`` if it failed — do NOT poll again.

    Args:
        job_name: the SageMaker training-job name returned by ``submit_sagemaker_job``.
        region: AWS region.
    """
    polls = _POLL_COUNTS.get(job_name, 0) + 1
    _POLL_COUNTS[job_name] = polls

    if polls > MAX_POLLS_PER_JOB:
        return {
            "job_name": job_name,
            "status": "unknown",
            "terminal": True,
            "polls": polls,
            "error": (
                f"poll budget exhausted: {job_name} has been observed {polls} times "
                f"(limit {MAX_POLLS_PER_JOB}) without being acted on."
            ),
            "next_action": (
                "STOP polling this job. Each poll costs a model turn and the answer is not "
                "changing. Either call await_training_job once to block for the terminal state, "
                "or treat this job as unresolved and change approach."
            ),
        }

    result = get_training_metrics(job_name, region=region)
    return _annotate_terminality(result, polls=polls)


def _annotate_terminality(result: dict[str, Any], *, polls: int) -> dict[str, Any]:
    """Add ``terminal`` + ``next_action`` so a caller cannot mistake a finished job for a busy one.

    The status and failure reason were always on the wire; the agent polled anyway. An explicit
    instruction in the tool result is what actually stops the loop — inferring terminality from a
    status string demonstrably did not ( Defect B).
    """
    status = str(result.get("status") or "")
    terminal = status in TERMINAL_TRAINING_STATES
    annotated = dict(result)
    annotated["terminal"] = terminal
    annotated["polls"] = polls
    if not terminal:
        annotated["next_action"] = (
            f"Job is {status or 'not yet reporting'}. Do NOT poll in a loop — call "
            "await_training_job once to block until it finishes."
        )
    elif status == "Completed":
        annotated["next_action"] = (
            "Job COMPLETED and will not change. Read the predictions from the output artifact "
            "and proceed to grading. Do not poll again."
        )
    else:
        reason = str(result.get("failure_reason") or "").strip()
        annotated["next_action"] = (
            f"Job {status} and will NOT change. Do not poll again — polling cannot help. "
            + (f"Root cause: {reason} " if reason else "No failure reason was reported. ")
            + "Fix the cause or change approach, then submit a NEW job."
        )
    return annotated


@tool
@tool_error_boundary(source="tools.training_jobs")
def await_training_job(
    job_name: str,
    region: str | None = None,
    timeout_seconds: float = 1800.0,
) -> dict[str, Any]:
    """WAIT for a dispatched SageMaker training job to finish, then return its terminal result.

    Call this ONCE instead of polling ``fetch_training_metrics`` in a loop. It blocks with a
    bounded deadline and costs a single model turn, where the loop cost ~58 turns per job and
    consumed a whole run budget ( Defect B).

    ADR-0002 keeps compute off-box and the agent's role as "dispatch + observe" — awaiting a
    managed job is observing, not computing. Nothing is fitted here.

    Returns the same shape as ``fetch_training_metrics`` (status, metrics, model_artifacts,
    failure_reason) plus ``terminal`` and ``next_action``. On timeout it returns
    ``terminal: False`` with the last observed status rather than raising, so the caller can
    decide whether to give the job longer or change approach.

    Args:
        job_name: the SageMaker training-job name returned by ``submit_sagemaker_job``.
        region: AWS region.
        timeout_seconds: hard deadline for the wait.
    """
    try:
        wait_for_training_job(job_name, region=region, timeout=timeout_seconds)
    except TimeoutError:
        snapshot = get_training_metrics(job_name, region=region)
        out = dict(snapshot)
        out["terminal"] = False
        out["next_action"] = (
            f"Still running after {timeout_seconds:.0f}s. Either await again with a longer "
            "timeout, or abandon this job and change approach — do NOT switch to polling."
        )
        return out

    # Terminal: report it once, fully annotated, and retire the job from the poll ledger so a
    # later snapshot is not charged against a budget that no longer applies.
    result = get_training_metrics(job_name, region=region)
    _POLL_COUNTS.pop(job_name, None)
    return _annotate_terminality(result, polls=_POLL_COUNTS.get(job_name, 0))


def _is_s3_uri(uri: str) -> bool:
    return isinstance(uri, str) and uri.startswith("s3://")


def _stage_local_to_s3(
    local_path: str,
    *,
    bucket: str,
    key_prefix: str,
    region: str | None = None,
    s3_client: Any = None,
) -> str:
    """Upload a LOCAL csv to the session bucket and return its ``s3://`` URI.

    ADR-0002: staging the dataset for a dispatched SageMaker job is the DISPATCH TOOL's job,
    not the agent's — the trainer must never have to hand-roll an S3 upload via
    ``execute_python_tool`` (off-pattern, fragile). ``submit_tabular_training`` accepts local
    sandbox paths (``/tmp/...``) and stages them here before composing the job. Raises on a
    missing file (never fabricates a URI).
    """
    from pathlib import Path

    import boto3

    p = Path(local_path)
    if not p.is_file():
        raise FileNotFoundError(f"submit_tabular_training: local dataset not found: {local_path}")
    s3 = s3_client or boto3.client("s3", region_name=resolve_aws_region(region))
    key = f"{key_prefix.rstrip('/')}/{p.name}"
    s3.upload_file(str(p), bucket, key)
    return f"s3://{bucket}/{key}"


@tool
@tool_error_boundary(source="tools.training_jobs")
def submit_tabular_training(
    algo: str,
    params: dict[str, Any],
    target_column: str,
    train_s3_uri: str,
    test_s3_uri: str,
    task_type: str = "classification",
    runtime_session_id: str = "",
    region: str | None = None,
    session_bucket: str = "",
    s3_client: Any = None,
) -> dict[str, Any]:
    """DISPATCH a tabular training job to SageMaker (ADR-0002) — the orchestrator's off-box path.

    This is how the trainer_tuner agent runs the DELIVERABLE training WORKLOAD without fitting in
    the AgentCore sandbox: it composes a SageMaker training-job config (the chosen ``algo`` +
    ``params`` + ``target_column`` + ``task_type`` as hyperparameters; the staged ``train``/``test``
    S3 channels) and delegates to the real ``submit_sagemaker_job`` (3-step DynamoDB +
    ``CreateTrainingJob``). The SageMaker-side entrypoint
    (:mod:`amlas.sagemaker_entrypoints.tabular_train`) fits on the full train split, predicts the
    test split, and writes the canonical ``y_pred_test.npy`` to S3 that the Arbiter grades — the
    graded contract is identical to the old in-container path; only the compute location moves.
    The job is async: it resumes the agent via the EventBridge/Lambda callback (DynamoDB job state).

    Args:
        algo: estimator name (e.g. "hist_gradient_boosting", "xgboost", "random_forest").
        params: hyperparameter dict for the estimator.
        target_column: target column name.
        train_s3_uri: TRAIN dataset — an ``s3://`` URI (used as-is) OR a LOCAL sandbox path
            (e.g. ``/tmp/train.csv``), which is AUTO-STAGED to the session bucket (ADR-0002:
            staging is the dispatch tool's job, so the agent passes its local paths directly).
        test_s3_uri: TEST dataset — same contract (the graded predict split).
        task_type: "classification" (default) or "regression".
        runtime_session_id: AgentCore session id to resume via the callback Lambda.
        region: AWS region.
        session_bucket: staging bucket for local paths; resolved from env when empty
            (``AMLAS_SESSION_BUCKET`` / ``SESSION_BUCKET``).
        s3_client: injectable boto3 S3 client (tests); real client built from ``region`` when None.

    Returns:
        The ``submit_sagemaker_job`` result (``{status, job_name, sagemaker_job_name, ...}``) on
        success, or ``{"status": "error", "error": ...}`` when a dataset is missing/unstageable
        (never fabricates a job).
    """
    import json

    if not train_s3_uri or not test_s3_uri:
        return {
            "status": "error",
            "error": "submit_tabular_training requires both a train and a test dataset "
            "(s3:// URI or local path)",
        }

    # ADR-0002: auto-stage LOCAL sandbox paths to S3 so the agent never hand-rolls an upload.
    # An s3:// URI is used as-is; anything else is treated as a local file and staged.
    if not _is_s3_uri(train_s3_uri) or not _is_s3_uri(test_s3_uri):
        bucket = session_bucket or _resolve_session_bucket({})
        if not bucket:
            return {
                "status": "error",
                "error": "submit_tabular_training: a local dataset path was given but no session "
                "bucket is resolvable (set session_bucket or AMLAS_SESSION_BUCKET) to stage it.",
            }
        prefix = f"training-data/{runtime_session_id or 'adhoc'}"
        try:
            if not _is_s3_uri(train_s3_uri):
                train_s3_uri = _stage_local_to_s3(
                    train_s3_uri,
                    bucket=bucket,
                    key_prefix=f"{prefix}/train",
                    region=region,
                    s3_client=s3_client,
                )
            if not _is_s3_uri(test_s3_uri):
                test_s3_uri = _stage_local_to_s3(
                    test_s3_uri,
                    bucket=bucket,
                    key_prefix=f"{prefix}/test",
                    region=region,
                    s3_client=s3_client,
                )
        except FileNotFoundError as exc:
            return {"status": "error", "error": str(exc)}

    # Compose the training-job config. The entrypoint reads these hyperparameters; the two S3
    # channels (train + test) are staged as SageMaker input channels the container mounts.
    config: dict[str, Any] = {
        "hyperparameters": {
            "algo": algo,
            "target_column": target_column,
            "task_type": task_type,
            "params": json.dumps(params or {}),
        },
        # train channel is the canonical `s3_input`/`training` channel; test rides as an extra.
        "s3_input": train_s3_uri,
        "channel_name": "train",
        "content_type": "text/csv",
        "test_s3_uri": test_s3_uri,
        "entrypoint": "amlas.sagemaker_entrypoints.tabular_train",
    }
    # Thread the resolved staging bucket through so submit_sagemaker_job resolves the SAME bucket
    # (it reads config.session_bucket first) — avoids a mismatch between where we staged and where
    # the job's artifacts/output land.
    bucket = session_bucket or _resolve_session_bucket({})
    if bucket:
        config["session_bucket"] = bucket
    # ``training_type="sklearn"`` stays on the CPU image (not a GPU fine-tune type).
    result = submit_sagemaker_job(
        training_type="sklearn",
        config=config,
        runtime_session_id=runtime_session_id,
        region=region,
    )
    # FIX-37b (ADR-0002 Phase 6): return the DETERMINISTIC S3 y_pred URI so the agent reports the
    # OFF-BOX artifact path — NOT the legacy sandbox ``/tmp/amlas/y_pred_test.npy`` (which the
    # dispatched job never writes locally; reporting it made the Arbiter's loader FileNotFound and
    # G-METRIC fail). SageMaker writes SM_OUTPUT_DATA_DIR into
    # ``s3://{bucket}/artifacts/{job_name}/{job_name}/output/output.tar.gz``; the FIX-36 resolver
    # extracts y_pred_test.npy from that tarball. We surface the canonical .npy URI under that
    # output prefix so the trainer prompt can hand it straight to the Arbiter.
    job_name = (result or {}).get("job_name") or (result or {}).get("sagemaker_job_name")
    if isinstance(result, dict) and bucket and job_name:
        result["y_pred_s3_uri"] = f"s3://{bucket}/artifacts/{job_name}/{job_name}/output/y_pred_test.npy"
        result["output_tar_s3_uri"] = f"s3://{bucket}/artifacts/{job_name}/{job_name}/output/output.tar.gz"
    return result


# Algorithms whose CV trial scores on a regression metric (negative MSE → R²-ish).
def _cv_scoring(task_type: str) -> str:
    if task_type == "regression":
        return "r2"
    return "f1_macro"


@tool
@tool_error_boundary(source="tools.training_jobs")
def submit_training_trial(
    algo: str,
    params: dict[str, Any],
    data_path: str,
    target_column: str,
    task_type: str = "classification",
    n_splits: int = 5,
    seed: int = 42,
    write_y_pred: bool = False,
    test_data_path: str = "",
    y_pred_path: str = "/tmp/amlas/y_pred_test.npy",  # nosec B108 — sandbox path
) -> dict[str, Any]:
    """Run a REAL scikit-learn cross-validation trial and return the REAL CV score.

    De-mock replacement for the PS-02 stub (which synthesized a score from
    ``hash(params)``). This loads the REAL dataset at ``data_path``, applies the same
    real preprocessing as :func:`amlas.tools.training_tools.train_model`
    (imputation + scaling + encoding, leakage-column exclusion), builds the REAL
    estimator with ``params``, and runs real ``cross_val_score``. The returned
    ``cv_score`` is the real mean across folds and its ``std`` the real fold std.

    FIX-30 (Subtask-13) — the y_pred BRIDGE. When ``write_y_pred=True`` this tool ALSO,
    after CV, refits the estimator on the FULL training split, predicts the staged TEST
    split, and saves the predictions to ``y_pred_path`` (default the canonical
    ``/tmp/amlas/y_pred_test.npy`` the Arbiter grades). This exists because the sandbox
    ``execute_python`` kernel can lose access to the staged data after an idle/timeout
    kernel cycle (its /tmp is gone), yet this tool RE-READS the data from ``data_path`` /
    ``test_data_path`` on every call — so it is the reliable path to a persisted y_pred
    when the free-form kernel cannot reach the data. NOT leakage: the test labels never
    influence CV/selection; the test split is used ONLY for the final scored predict.

    Args:
        algo: algorithm name (e.g. "xgboost", "hist_gradient_boosting", "random_forest").
        params: hyperparameter dict passed to the real estimator.
        data_path: filesystem path to the training CSV (REAL data — required).
        target_column: name of the target column.
        task_type: "classification" (default) or "regression".
        n_splits: number of CV folds.
        seed: RNG seed for reproducibility.
        write_y_pred: FIX-30 — when True, refit on full train + predict the test split +
            save predictions to ``y_pred_path`` (the Arbiter-graded artifact).
        test_data_path: path to the staged TEST CSV (required when ``write_y_pred=True``).
        y_pred_path: where to save the test predictions (default canonical path).
    """
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from sklearn.model_selection import (
        KFold,
        StratifiedKFold,
        cross_val_score,
    )

    from .training_tools import _build_model, _preprocess_for_training

    start = time.time()
    # Path-traversal containment (Security review, CWE-22).
    from .pathref import resolve_dataset_path

    try:
        _p = resolve_dataset_path(data_path)
    except PermissionError as e:
        return {"algo": algo, "params": params, "error": str(e)}
    if not _p.exists():
        return {"algo": algo, "params": params, "error": f"File not found: {data_path}"}

    df = pd.read_csv(_p)
    if target_column not in df.columns:
        return {
            "algo": algo,
            "params": params,
            "error": f"Target column '{target_column}' not found",
        }

    X, y = _preprocess_for_training(df, target_column, task_type=task_type)
    model = _build_model(algo, params, seed, task_type=task_type)

    if task_type == "classification":
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scoring = _cv_scoring(task_type)

    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    result: dict[str, Any] = {
        "algo": algo,
        "params": params,
        "cv_score": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "scores": [float(s) for s in scores],
        "n_splits": n_splits,
        "scoring": scoring,
        "cv_strategy": "stratified_kfold" if task_type == "classification" else "kfold",
        "seed": seed,
    }

    # FIX-30: the y_pred BRIDGE — refit on full train, predict the staged test split, and
    # persist the Arbiter-graded artifact. This is the reliable path to y_pred when the
    # free-form execute_python kernel has lost data access after an idle/timeout cycle.
    if write_y_pred:
        try:
            if not test_data_path:
                result["y_pred_error"] = "write_y_pred=True but test_data_path was empty"
            else:
                test_p = resolve_dataset_path(test_data_path)
                if not test_p.exists():
                    result["y_pred_error"] = f"test file not found: {test_data_path}"
                else:
                    # Refit on the FULL training split (no CV holdout), then predict test.
                    model_full = _build_model(algo, params, seed, task_type=task_type)
                    model_full.fit(X, y)
                    df_test = pd.read_csv(test_p)
                    # Align test features to the training preprocessing. If the test CSV
                    # carries the target, drop it; otherwise use all columns as features.
                    X_test, _ = (
                        _preprocess_for_training(df_test, target_column, task_type=task_type)
                        if target_column in df_test.columns
                        else (
                            _preprocess_for_training(
                                df_test.assign(**{target_column: 0}), target_column, task_type=task_type
                            )[0],
                            None,
                        )
                    )
                    y_pred_test = model_full.predict(X_test)
                    out_p = Path(y_pred_path)
                    out_p.parent.mkdir(parents=True, exist_ok=True)
                    np.save(out_p, np.asarray(y_pred_test))
                    print(f"WROTE {y_pred_path} {len(y_pred_test)}", flush=True)
                    result["y_pred_path"] = y_pred_path
                    result["n_test_predictions"] = len(y_pred_test)
        except Exception as e:
            result["y_pred_error"] = f"{type(e).__name__}: {e}"

    result["elapsed_seconds"] = time.time() - start
    return result
