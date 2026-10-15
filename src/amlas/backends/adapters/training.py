"""TrainingBackend adapters (doc 05 §4.2 G1 / roadmap M0 work-item 4).

* ``SageMakerTrainingBackend`` — the **default**; relocates ``tools/sagemaker.py``'s
  3-step DDB submit (PENDING put → create_training_job → RUNNING update) behind the
  port. Behavior-preserving: it delegates to the EXISTING ``submit_training_job`` /
  ``get_job_status`` / ``get_training_metrics`` so the DDB+SageMaker call sequence is
  byte-for-byte today's (ADR-001).
* ``LocalSklearnBackend`` — the **local/offline** adapter; wraps the EXISTING
  ``train_model`` (sklearn/XGBoost) so small-data + the the offline AWS API emulator/offline suite train
  in-process with no AWS.

Both implement :class:`~amlas.backends.ports.TrainingBackend`. Clients /
AWS wiring are injectable on the SageMaker adapter (the existing tool's
``dynamodb_client=`` / ``sagemaker_client=`` convention) so parity callers may pass
injectable clients; the registry's no-arg factory builds the prod default.
"""

from __future__ import annotations

import os
import uuid
from typing import Any

from ...config._env_compat import env_with_compat
from ..ports import TrainingBackend
from ..registry import register_adapter
from ..types import JobHandle, JobState, JobStatus, ModelRef
from ._util import raw_tool

# Map the SageMaker DDB ``status`` string → the backend-neutral JobState.
_SM_STATE = {
    "PENDING": JobState.PENDING,
    "RUNNING": JobState.RUNNING,
    "Completed": JobState.SUCCEEDED,
    "COMPLETED": JobState.SUCCEEDED,
    "Failed": JobState.FAILED,
    "FAILED": JobState.FAILED,
    "Stopped": JobState.STOPPED,
    "STOPPED": JobState.STOPPED,
}


class LocalSklearnBackend:
    """Local/offline TrainingBackend — wraps EXISTING ``train_model`` (no AWS).

    ``submit`` runs the sklearn/XGBoost fit synchronously and returns a completed
    :class:`JobHandle` carrying the result; ``status`` reports SUCCEEDED/FAILED;
    ``metrics`` / ``artifact`` read the same result dict the tool returns today.
    """

    def submit(
        self,
        *,
        algo: str,
        train_uri: str,
        val_uri: str | None = None,
        target: str | None = None,
        hyperparams: dict[str, Any] | None = None,
        resources: dict[str, Any] | None = None,
        budget: dict[str, Any] | None = None,
    ) -> JobHandle:
        """Train ``algo`` on the local CSV at ``train_uri`` via EXISTING ``train_model``."""
        from ...tools.training_tools import train_model

        res = dict(resources or {})
        result = raw_tool(train_model)(
            data_path=train_uri,
            target_column=target or res.get("target_column", ""),
            algorithm=algo,
            params=dict(hyperparams or {}),
            output_dir=res.get("output_dir"),
            seed=int(res.get("seed", 42)),
            test_size=float(res.get("test_size", 0.2)),
            task_type=res.get("task_type", "classification"),
        )
        return JobHandle(id=f"local-{uuid.uuid4().hex[:8]}", backend="local", extra={"result": result})

    def status(self, handle: JobHandle) -> JobStatus:
        """SUCCEEDED when the wrapped ``train_model`` reported success, else FAILED."""
        result = handle.extra.get("result", {})
        ok = bool(result.get("success"))
        return JobStatus(
            state=JobState.SUCCEEDED if ok else JobState.FAILED,
            detail="" if ok else str(result.get("error", "")),
            raw=result,
        )

    def metrics(self, handle: JobHandle) -> dict[str, Any]:
        """The metrics dict the wrapped ``train_model`` returned (unchanged)."""
        return dict(handle.extra.get("result", {}).get("metrics", {}))

    def artifact(self, handle: JobHandle) -> ModelRef:
        """A :class:`ModelRef` to the pickled model (when an ``output_dir`` was given)."""
        result = handle.extra.get("result", {})
        return ModelRef(
            uri=str(result.get("model_path", "")),
            backend="local",
            algo=str(result.get("algorithm", "")),
            metrics={
                k: float(v) for k, v in result.get("metrics", {}).items() if isinstance(v, (int, float))
            },
        )


class SageMakerTrainingBackend:
    """Default TrainingBackend — wraps EXISTING ``tools/sagemaker.py`` (ADR-001).

    Relocates today's 3-step DDB submit behind the port WITHOUT changing it: the
    same ``submit_training_job`` / ``get_job_status`` / ``get_training_metrics``
    functions run, so the DDB+SageMaker call sequence is identical (the parity
    test asserts this). AWS wiring is injectable for isolated validation; prod builds default
    boto3 clients lazily inside the wrapped tool.
    """

    def __init__(
        self,
        *,
        region: str | None = None,
        execution_role_arn: str = "",
        session_bucket: str = "",
        jobs_table: str | None = None,
        runtime_session_id: str = "",
        dynamodb_client: Any = None,
        sagemaker_client: Any = None,
    ) -> None:
        self._region = region
        self._role = execution_role_arn
        self._bucket = session_bucket
        self._jobs_table = jobs_table
        self._session_id = runtime_session_id
        self._ddb = dynamodb_client
        self._sm = sagemaker_client

    def submit(
        self,
        *,
        algo: str,
        train_uri: str,
        val_uri: str | None = None,
        target: str | None = None,
        hyperparams: dict[str, Any] | None = None,
        resources: dict[str, Any] | None = None,
        budget: dict[str, Any] | None = None,
    ) -> JobHandle:
        """Submit via EXISTING ``submit_training_job`` (identical DDB+SM sequence)."""
        from ...tools.sagemaker import SageMakerSubmitError, resolve_jobs_table, submit_training_job

        res = dict(resources or {})
        config = dict(hyperparams or {})
        # Surface the training input URI through the existing config seam.
        config.setdefault("s3_input", train_uri)
        if "hyperparameters" not in config and hyperparams:
            config = {"hyperparameters": dict(hyperparams), "s3_input": train_uri}
        jobs_table = resolve_jobs_table(res.get("jobs_table") or self._jobs_table)
        tracking_error = ""
        try:
            result = submit_training_job(
                training_type=algo,
                config=config,
                runtime_session_id=res.get("runtime_session_id", self._session_id),
                execution_role_arn=(
                    res.get("execution_role_arn")
                    or self._role
                    or os.environ.get("SAGEMAKER_EXECUTION_ROLE_ARN")
                    or env_with_compat("AMLAS_SAGEMAKER_ROLE_ARN")
                    or ""
                ),
                session_bucket=(
                    res.get("session_bucket")
                    or self._bucket
                    or env_with_compat("AMLAS_SESSION_BUCKET")
                    or os.environ.get("SESSION_BUCKET")
                    or ""
                ),
                jobs_table=jobs_table,
                region=self._region,
                dynamodb_client=self._ddb,
                sagemaker_client=self._sm,
            )
        except SageMakerSubmitError as error:
            if error.accepted_job is None:
                raise
            result = error.accepted_job
            tracking_error = "native job accepted; DynamoDB tracking update failed"
        return JobHandle(
            id=result.job_name,
            backend="sagemaker",
            arn=result.training_job_arn,
            extra={
                "runtime_session_id": result.runtime_session_id,
                "status": result.status,
                "jobs_table": jobs_table,
                "tracking_error": tracking_error,
            },
        )

    def status(self, handle: JobHandle) -> JobStatus:
        """Read the DDB row via EXISTING ``get_job_status`` and map to JobState."""
        from ...tools.sagemaker import get_job_status

        row = get_job_status(
            handle.id,
            jobs_table=handle.extra.get("jobs_table") or self._jobs_table,
            region=self._region,
            dynamodb_client=self._ddb,
        )
        raw_state = str(row.get("status", ""))
        return JobStatus(state=_SM_STATE.get(raw_state, JobState.UNKNOWN), detail=raw_state, raw=row)

    def metrics(self, handle: JobHandle) -> dict[str, Any]:
        """Read real metrics via EXISTING ``get_training_metrics`` (FinalMetricDataList)."""
        from ...tools.sagemaker import get_training_metrics

        out = get_training_metrics(handle.id, region=self._region, sagemaker_client=self._sm)
        return dict(out.get("metrics", {}))

    def artifact(self, handle: JobHandle) -> ModelRef:
        """A :class:`ModelRef` to the trained model artifact (``S3ModelArtifacts``)."""
        from ...tools.sagemaker import get_training_metrics

        out = get_training_metrics(handle.id, region=self._region, sagemaker_client=self._sm)
        return ModelRef(
            uri=str(out.get("model_artifacts", "")),
            backend="sagemaker",
            metrics={k: float(v) for k, v in out.get("metrics", {}).items() if isinstance(v, (int, float))},
        )


# ── self-registration (the registry imports this module lazily) ───────────────
register_adapter(TrainingBackend, "sagemaker", lambda: SageMakerTrainingBackend(), is_default=True)
register_adapter(TrainingBackend, "local", lambda: LocalSklearnBackend(), is_local=True)
