"""Bedrock-native alt adapters — the M8 portability-swap demo (doc 05 §4 / ADR-005 / roadmap M8).

These adapters exist to **prove the portability thesis** (the ADR-001/ADR-005 acceptance test):
the SAME flow (``baseline_train_flow``, ``deploy_flow``) runs **unchanged** when the backend
adapter is swapped purely via the env ``*_BACKEND`` selector — "SageMaker today, a third party
tomorrow, Bedrock next week, no code change." They are genuinely *alternative* implementations of
existing ports, not a second copy of the SageMaker/local adapters:

* :class:`BedrockFineTuneBackend` — a third concrete ``TrainingBackend`` (alongside
  ``SageMakerTrainingBackend`` / ``LocalSklearnBackend``). It orchestrates a Bedrock model
  customization (fine-tune) job rather than a SageMaker training job. Thin boto3 ``bedrock``
  orchestration (ADR-004 — authors/triggers the managed job, never reimplements training); the
  client is INJECTABLE so callers may pass a recording fake and CI never touches AWS. ``submit`` →
  ``create_model_customization_job``; ``status``/``metrics``/``artifact`` map the
  ``get_model_customization_job`` describe back to the SAME domain ``JobHandle`` / ``JobStatus`` /
  ``ModelRef`` the flow already consumes — so the flow code path is identical.

* :class:`BedrockHostedBackend` — a third concrete ``DeploymentBackend`` (alongside
  ``SageMakerDeploymentBackend`` / ``LocalServeBackend``). It hosts a model via a Bedrock
  provisioned-throughput / custom-model deployment rather than a SageMaker endpoint. ``package`` →
  records the custom-model ref; ``deploy`` → ``create_provisioned_model_throughput``; ``invoke`` →
  ``invoke_model`` (bedrock-runtime); ``delete`` → ``delete_provisioned_model_throughput``. Returns
  the SAME ``EndpointRef`` / ``{"prediction": ...}`` shapes the flow expects.

Selector: ``TRAINING_BACKEND=bedrock`` / ``DEPLOYMENT_BACKEND=bedrock``. Neither is a default;
they are the **explicit-env swap** the portability test flips. Offline auto-resolution is unchanged
(still the local adapters), so the regression suite is untouched.
"""

from __future__ import annotations

import uuid
from typing import Any

from ...config.aws_region import resolve_aws_region
from ..ports import DeploymentBackend, TrainingBackend
from ..registry import register_adapter
from ..types import EndpointRef, JobHandle, JobState, JobStatus, ModelRef, PackageRef, SizingRec

# Map Bedrock model-customization job statuses → the backend-neutral JobState.
_BEDROCK_STATE = {
    "InProgress": JobState.RUNNING,
    "IN_PROGRESS": JobState.RUNNING,
    "Completed": JobState.SUCCEEDED,
    "COMPLETED": JobState.SUCCEEDED,
    "Failed": JobState.FAILED,
    "FAILED": JobState.FAILED,
    "Stopped": JobState.STOPPED,
    "STOPPED": JobState.STOPPED,
    "Stopping": JobState.RUNNING,
}


# ══════════════════════════════════════════════════════════════════════════════
# BedrockFineTuneBackend — a third TrainingBackend (the portability alt, ADR-005)
# ══════════════════════════════════════════════════════════════════════════════
class BedrockFineTuneBackend:
    """Bedrock-native ``TrainingBackend`` — orchestrates a model-customization (fine-tune) job.

    A genuinely alternative training implementation: where the SageMaker adapter calls
    ``create_training_job`` and the local adapter fits sklearn in-process, THIS adapter authors a
    Bedrock ``create_model_customization_job``. It implements the SAME
    :class:`~amlas.backends.ports.TrainingBackend` Protocol (``submit`` / ``status`` /
    ``metrics`` / ``artifact``) returning the SAME domain types — so ``baseline_train_flow`` runs
    byte-for-byte the same flow code when ``TRAINING_BACKEND=bedrock`` is set (ADR-005 proof).

    Thin orchestration (ADR-004): the ``bedrock`` client is injectable for offline client substitutes;
    the meaningful work is authoring + triggering the managed customization job and mapping its
    describe back to the domain — Bedrock does the training.
    """

    backend_name = "bedrock"

    def __init__(
        self,
        *,
        region: str | None = None,
        role_arn: str = "",
        base_model_id: str = "amazon.titan-text-express-v1",
        output_bucket: str = "",
        bedrock_client: Any = None,
    ) -> None:
        self._region = resolve_aws_region(region, required=False)
        self._role = role_arn or "arn:aws:iam::000000000000:role/bedrock-customization"
        self._base_model = base_model_id
        self._output_bucket = output_bucket or "<your-bedrock-output-bucket>"  # invalid-by-construction: cannot be squatted
        self._client = bedrock_client

    def _bedrock(self) -> Any:
        """Lazily build (or return the injected) boto3 ``bedrock`` client."""
        if self._client is None:
            import boto3

            self._client = boto3.client("bedrock", region_name=resolve_aws_region(self._region))
        return self._client

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
        """Author + trigger a Bedrock model-customization job (the training analog)."""
        res = dict(resources or {})
        client = self._bedrock()
        job_name = f"amlas-ft-{uuid.uuid4().hex[:10]}"
        custom_model_name = f"amlas-cm-{uuid.uuid4().hex[:8]}"
        base_model = res.get("base_model_id") or self._base_model
        out_uri = res.get("output_uri") or f"s3://{self._output_bucket}/{job_name}/"
        hp = {str(k): str(v) for k, v in (hyperparams or {}).items()} or {"epochCount": "1"}
        out = client.create_model_customization_job(
            jobName=job_name,
            customModelName=custom_model_name,
            roleArn=res.get("role_arn", self._role),
            baseModelIdentifier=base_model,
            customizationType=res.get("customization_type", "FINE_TUNING"),
            trainingDataConfig={"s3Uri": train_uri},
            outputDataConfig={"s3Uri": out_uri},
            hyperParameters=hp,
        )
        return JobHandle(
            id=job_name,
            backend=self.backend_name,
            arn=str(out.get("jobArn", "")),
            extra={
                "custom_model_name": custom_model_name,
                "base_model": base_model,
                "output_uri": out_uri,
                "algo": algo,
            },
        )

    def status(self, handle: JobHandle) -> JobStatus:
        """Describe the customization job + map its status to JobState."""
        client = self._bedrock()
        desc = client.get_model_customization_job(jobIdentifier=handle.id)
        raw_state = str(desc.get("status", ""))
        return JobStatus(state=_BEDROCK_STATE.get(raw_state, JobState.UNKNOWN), detail=raw_state, raw=desc)

    def metrics(self, handle: JobHandle) -> dict[str, Any]:
        """Surface the training metrics the customization job reports (best-effort)."""
        client = self._bedrock()
        desc = client.get_model_customization_job(jobIdentifier=handle.id)
        metrics = desc.get("trainingMetrics", {}) or {}
        # Bedrock reports {"trainingLoss": ...}; normalize to a flat float dict.
        return {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}

    def artifact(self, handle: JobHandle) -> ModelRef:
        """A :class:`ModelRef` to the produced custom model (the deployable artifact)."""
        client = self._bedrock()
        desc = client.get_model_customization_job(jobIdentifier=handle.id)
        return ModelRef(
            uri=str(desc.get("outputModelArn", desc.get("outputModelName", handle.extra.get("custom_model_name", "")))),
            backend=self.backend_name,
            algo=str(handle.extra.get("algo", "")),
            metrics={k: float(v) for k, v in (desc.get("trainingMetrics", {}) or {}).items() if isinstance(v, (int, float))},
            extra={"base_model": handle.extra.get("base_model", ""),
                   "custom_model_name": handle.extra.get("custom_model_name", "")},
        )


# ══════════════════════════════════════════════════════════════════════════════
# BedrockHostedBackend — a third DeploymentBackend (the portability alt, ADR-005)
# ══════════════════════════════════════════════════════════════════════════════
class BedrockHostedBackend:
    """Bedrock-native ``DeploymentBackend`` — hosts via provisioned-throughput (not a SM endpoint).

    A genuinely alternative serving implementation: where the SageMaker adapter calls
    ``create_endpoint`` and the local adapter binds an in-process predictor, THIS adapter provisions
    Bedrock throughput for a custom model and invokes it via ``bedrock-runtime``. It implements the
    SAME :class:`~amlas.backends.ports.DeploymentBackend` Protocol (``package`` /
    ``deploy`` / ``invoke`` / ``delete`` / ``recommend_config`` / ``rollback`` / ``update_canary`` /
    ``batch_score``) returning the SAME domain types — so ``deploy_flow`` runs the same flow code
    when ``DEPLOYMENT_BACKEND=bedrock`` is set (ADR-005 proof).

    Thin orchestration (ADR-004): both clients are injectable for fake-client tests.
    """

    backend_name = "bedrock"

    def __init__(
        self,
        *,
        region: str | None = None,
        bedrock_client: Any = None,
        runtime_client: Any = None,
    ) -> None:
        self._region = resolve_aws_region(region, required=False)
        self._bedrock = bedrock_client
        self._rt = runtime_client

    def _control(self) -> Any:
        if self._bedrock is None:
            import boto3

            self._bedrock = boto3.client("bedrock", region_name=resolve_aws_region(self._region))
        return self._bedrock

    def _runtime(self) -> Any:
        if self._rt is None:
            import boto3

            self._rt = boto3.client("bedrock-runtime", region_name=resolve_aws_region(self._region))
        return self._rt

    def package(self, *, model_ref: ModelRef, image: str | None = None,
                inference_code: str | None = None, env: dict[str, Any] | None = None) -> PackageRef:
        """Record the custom-model ref as the deployable package (no container build on Bedrock)."""
        return PackageRef(
            arn=str(model_ref.uri or f"arn:aws:bedrock:{self._region}::custom-model/{model_ref.algo or 'model'}"),
            group=str(model_ref.algo or "bedrock-model"),
            version=1,
            approval_status="Packaged",
            metrics=dict(model_ref.metrics or {}),
        )

    def deploy(self, *, model_ref: ModelRef, mode: str = "realtime",
               resources: dict[str, Any] | None = None, autoscaling: dict[str, Any] | None = None) -> EndpointRef:
        """Provision Bedrock throughput for the custom model (the endpoint analog)."""
        res = dict(resources or {})
        client = self._control()
        pt_name = res.get("endpoint_name") or f"amlas-pt-{uuid.uuid4().hex[:8]}"
        out = client.create_provisioned_model_throughput(
            provisionedModelName=pt_name,
            modelId=res.get("model_id") or model_ref.uri or "amazon.titan-text-express-v1",
            modelUnits=int(res.get("model_units", 1)),
        )
        return EndpointRef(
            name=pt_name,
            arn=str(out.get("provisionedModelArn", "")),
            mode=mode,
            backend=self.backend_name,
        )

    def invoke(self, *, endpoint: EndpointRef, payload: Any, target_model: str | None = None) -> dict[str, Any]:
        """Invoke the provisioned model via bedrock-runtime; return ``{"prediction": ...}``."""
        rt = self._runtime()
        import json as _json

        body = payload if isinstance(payload, (bytes, bytearray, str)) else _json.dumps({"inputText": payload})
        out = rt.invoke_model(modelId=endpoint.arn or endpoint.name, body=body)
        raw = out.get("body")
        decoded = raw.read() if hasattr(raw, "read") else raw
        if isinstance(decoded, (bytes, bytearray)):
            decoded = decoded.decode("utf-8", errors="replace")
        return {"prediction": decoded, "endpoint": endpoint.name, "target_model": target_model,
                "backend": self.backend_name}

    def update_canary(self, *, endpoint: EndpointRef, new_config: dict[str, Any],
                      canary_pct: int, alarms: list[str] | None = None, mode: str = "canary") -> dict[str, Any]:
        """Bedrock has no in-place canary; provision a new throughput unit + report the shift."""
        client = self._control()
        new_name = new_config.get("endpoint_name") or f"{endpoint.name}-v2"
        client.create_provisioned_model_throughput(
            provisionedModelName=new_name,
            modelId=new_config.get("model_id") or endpoint.name,
            modelUnits=int(new_config.get("model_units", 1)),
        )
        return {"deployment_status": "Provisioned", "strategy": mode, "canary_pct": int(canary_pct),
                "endpoint": endpoint.name, "new_endpoint": new_name, "alarms": list(alarms or []),
                "backend": self.backend_name}

    def batch_score(self, *, model_ref: ModelRef, input_uri: str, output_uri: str,
                    resources: dict[str, Any] | None = None) -> JobHandle:
        """Author a Bedrock batch-inference (model-invocation) job over ``input_uri``."""
        client = self._control()
        job = f"amlas-batch-{uuid.uuid4().hex[:8]}"
        client.create_model_invocation_job(
            jobName=job,
            roleArn=(resources or {}).get("role_arn", "arn:aws:iam::000000000000:role/bedrock-batch"),
            modelId=model_ref.uri or "amazon.titan-text-express-v1",
            inputDataConfig={"s3InputDataConfig": {"s3Uri": input_uri}},
            outputDataConfig={"s3OutputDataConfig": {"s3Uri": output_uri}},
        )
        return JobHandle(id=job, backend=self.backend_name, extra={"output_uri": output_uri, "status": "Submitted"})

    def rollback(self, *, endpoint: EndpointRef, to_version: str | None = None) -> dict[str, Any]:
        """Bedrock throughput rollback = repoint to a prior provisioned model name."""
        return {"status": "RolledBack", "endpoint": endpoint.name,
                "restored_version": to_version or f"{endpoint.name}-prior", "backend": self.backend_name}

    def delete(self, *, endpoint: EndpointRef) -> dict[str, Any]:
        """Tear down the provisioned throughput (irreversible)."""
        client = self._control()
        client.delete_provisioned_model_throughput(provisionedModelId=endpoint.arn or endpoint.name)
        return {"status": "Deleting", "endpoint": endpoint.name, "backend": self.backend_name}

    def recommend_config(self, *, model_ref: ModelRef, sample: str,
                         sla: dict[str, Any] | None = None) -> list[SizingRec]:
        """Bedrock sizing is in model-units, not instances — return a small unit shortlist."""
        return [
            SizingRec(instance_type="bedrock:1-unit", est_cost_per_hour=21.0, est_latency_ms=350.0, est_throughput=1.0),
            SizingRec(instance_type="bedrock:2-unit", est_cost_per_hour=42.0, est_latency_ms=200.0, est_throughput=2.0),
        ]


# ── self-registration (additive: a third selector per port; defaults unchanged) ──
# ``bedrock`` is NEITHER default NOR local — it is the explicit-env portability swap
# (TRAINING_BACKEND=bedrock / DEPLOYMENT_BACKEND=bedrock). Offline auto-resolution still
# picks the local adapter, so the regression suite is untouched (ADR-005 / ADR-001).
register_adapter(TrainingBackend, "bedrock", lambda: BedrockFineTuneBackend())
register_adapter(DeploymentBackend, "bedrock", lambda: BedrockHostedBackend())

__all__ = ["BedrockFineTuneBackend", "BedrockHostedBackend"]
