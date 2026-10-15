"""Async SageMaker training pattern — sample-mlops-agent port, Strands-native.

3-step submit sequence (sample-mlops-agent §22, MLflow DROPPED per OPEN DECISION 6):
    1. dynamodb.put_item(PENDING) — withOUT sagemaker_job_name (GSI null-string constraint)
    2. sagemaker.create_training_job() → SAGEMAKER_JOB_NAME
    3. dynamodb.update_item(RUNNING + job_name)

Phase 5 de-mock: this is the REAL integration. There is no fake MLflow run-id
synthesis anymore — training metrics are read back from the REAL training job via
``describe_training_job`` (``FinalMetricDataList`` + status + model artifacts), not
from a standing MLflow server (none is provisioned; see realinfra/01 OPEN DECISION 6).

Clients are injectable (``dynamodb_client=`` / ``sagemaker_client=``) so callers can
pass real-AWS-backed clients (or a recording client for the DDB-sequence tests whose
intent is the DynamoDB transitions, not the SageMaker side). In prod the clients
default to real ``boto3.client(...)``.

The callback Lambda handler lives in src/amlas/async_callback/lambda_handler.py.
"""

from __future__ import annotations

import os
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any

import boto3
from botocore.exceptions import ClientError

from ..config.aws_region import resolve_aws_client_region, resolve_aws_region
from ..config.deployment import deployment_job_tags, resolve_deployment_prefix
from ..config.sagemaker_images import pytorch_training_image, validate_image_region

# Retained for callers importing the legacy constant; runtime defaults resolve on every call.
DEFAULT_JOBS_TABLE = "amlas-training-jobs"


def resolve_jobs_table(explicit: str | None = None) -> str:
    """Resolve explicit table, managed-runtime env, then selected deployment default."""
    return (
        explicit
        or os.environ.get("AMLAS_JOBS_TABLE")
        or os.environ.get("JOBS_TABLE")
        or f"{resolve_deployment_prefix()}-training-jobs"
    )


# Terminal SageMaker TrainingJobStatus values.
TERMINAL_TRAINING_STATES = frozenset({"Completed", "Failed", "Stopped"})

# Default metric-extraction regexes. SageMaker scrapes the training container's
# stdout/stderr with these regexes and records the last match per metric into
# DescribeTrainingJob.FinalMetricDataList. A training script that prints e.g.
# ``final_metric=0.93`` or ``validation:f1=0.88`` is captured with no MLflow.
DEFAULT_METRIC_DEFINITIONS: list[dict[str, str]] = [
    {"Name": "final_metric", "Regex": r"final_metric=([0-9\.eE+-]+)"},
    {"Name": "train_loss", "Regex": r"train_loss=([0-9\.eE+-]+)"},
    {"Name": "val_loss", "Regex": r"val_loss=([0-9\.eE+-]+)"},
    {"Name": "accuracy", "Regex": r"accuracy=([0-9\.eE+-]+)"},
    {"Name": "f1", "Regex": r"f1=([0-9\.eE+-]+)"},
]


@dataclass(frozen=True)
class SubmitResult:
    job_name: str
    runtime_session_id: str
    status: str
    training_job_arn: str = ""


class SageMakerSubmitError(RuntimeError):
    def __init__(self, message: str, *, accepted_job: SubmitResult | None = None) -> None:
        super().__init__(message)
        self.accepted_job = accepted_job


def submit_training_job(
    *,
    training_type: str,
    config: dict[str, Any],
    runtime_session_id: str,
    execution_role_arn: str,
    session_bucket: str,
    jobs_table: str | None = None,
    region: str | None = None,
    dynamodb_client: Any = None,
    sagemaker_client: Any = None,
) -> SubmitResult:
    """Submit an async SageMaker training job following the canonical 3-step sequence.

    Clients are injectable so callers can pass real-AWS-backed clients. In prod, leave
    client args as None and we build default ones from the region. There is NO MLflow
    leg (OPEN DECISION 6) — metrics are read back from the real training job via
    :func:`get_training_metrics`.
    """
    runtime_session_id = (
        runtime_session_id or os.environ.get("RUNTIME_SESSION_ID") or f"session-{uuid.uuid4()}"
    )
    job_tags = deployment_job_tags(runtime_session_id)
    region = resolve_aws_client_region(sagemaker_client, region, required=not bool(config.get("image_uri")))
    sm = sagemaker_client or boto3.client("sagemaker", region_name=resolve_aws_region(region))
    ddb = dynamodb_client or boto3.client("dynamodb", region_name=resolve_aws_region(region))

    # uuid suffix keeps the name unique even for two submits in the same second
    # (the prior int(time.time()) alone collided under fast/parallel test submits).
    prefix = resolve_deployment_prefix()
    suffix = f"{int(time.time())}-{uuid.uuid4().hex[:6]}"
    operation = re.sub(r"[^A-Za-z0-9]+", "-", training_type).strip("-") or "training"
    name_budget = 63 - len(prefix) - len(suffix) - 2
    operation = operation[:name_budget].rstrip("-")
    job_name = f"{prefix}-{operation}-{suffix}"
    jobs_table = resolve_jobs_table(jobs_table)

    # Step 1: DynamoDB PENDING (no sagemaker_job_name yet — GSI rejects empty strings)
    try:
        ddb.put_item(
            TableName=jobs_table,
            Item={
                "job_name": {"S": job_name},
                "runtime_session_id": {"S": runtime_session_id},
                "status": {"S": "PENDING"},
                "training_type": {"S": training_type},
                "created_at": {"N": str(int(time.time()))},
            },
            ConditionExpression="attribute_not_exists(job_name)",
        )
    except ClientError as e:
        raise SageMakerSubmitError(f"DynamoDB put_item failed: {e}") from e

    # Step 2: SageMaker CreateTrainingJob (real metric-definitions + hyperparameters)
    training_job_arn = ""
    try:
        resp = _create_training_job(
            sm_client=sm,
            job_name=job_name,
            training_type=training_type,
            config=config,
            execution_role_arn=execution_role_arn,
            session_bucket=session_bucket,
            runtime_session_id=runtime_session_id,
            region=region,
            job_tags=job_tags,
        )
        training_job_arn = (resp or {}).get("TrainingJobArn", "")
    except ClientError as e:
        # Best-effort cleanup of the DynamoDB row so a failed submit leaves no PENDING.
        try:
            ddb.delete_item(TableName=jobs_table, Key={"job_name": {"S": job_name}})
        except Exception:  # nosec B110
            pass
        raise SageMakerSubmitError(f"SageMaker CreateTrainingJob failed: {e}") from e

    # Step 3: update DynamoDB → RUNNING + sagemaker_job_name
    try:
        ddb.update_item(
            TableName=jobs_table,
            Key={"job_name": {"S": job_name}},
            UpdateExpression="SET #s = :s, sagemaker_job_name = :j",
            ExpressionAttributeNames={"#s": "status"},
            ExpressionAttributeValues={
                ":s": {"S": "RUNNING"},
                ":j": {"S": job_name},
            },
        )
    except ClientError as e:
        accepted = SubmitResult(job_name, runtime_session_id, "SUBMITTED", training_job_arn)
        raise SageMakerSubmitError(
            "DynamoDB tracking update failed after native training acceptance", accepted_job=accepted
        ) from e

    return SubmitResult(
        job_name=job_name,
        runtime_session_id=runtime_session_id,
        status="RUNNING",
        training_job_arn=training_job_arn,
    )


# Training types that require GPU acceleration (fine-tuning / RL) — SM-01.
_GPU_TRAINING_TYPES = {"finetune", "fine_tune", "sft", "dpo", "grpo", "ppo", "rlhf", "gpu"}


def _select_image_and_instance(
    training_type: str, config: dict[str, Any], *, region: str | None = None
) -> tuple[str, str]:
    """Pick the ECR training image + instance type for a training type (SM-01).

    Fine-tuning / RL training types (SFT/DPO/GRPO/…) need a GPU image and a GPU
    instance; everything else stays on the CPU image. Explicit ``image_uri`` /
    ``instance_type`` in ``config`` always win.
    """
    is_gpu = training_type.lower() in _GPU_TRAINING_TYPES
    if config.get("image_uri"):
        return validate_image_region(str(config["image_uri"]), region), config.get(
            "instance_type", "ml.g5.2xlarge" if is_gpu else "ml.m5.xlarge"
        )
    region = resolve_aws_region(region)

    if is_gpu:
        default_image = pytorch_training_image(region, gpu=True)
        default_instance = "ml.g5.2xlarge"
    elif training_type.lower() == "sklearn":
        # FIX-34 (ADR-0002): tabular sklearn/HGB/XGBoost training runs in the fully-managed AWS
        # SKLearn FRAMEWORK container in SCRIPT MODE (managed-services-first — no custom image to
        # build/push). The dispatcher bundles the amlas source tarball + requirements.txt; the
        # container pip-installs them and runs sagemaker_program. Account 246618743249 is the AWS
        # sklearn DLC registry for us-* regions.
        # The image is owned by ``config.sagemaker_images`` — it must be a Python >= 3.11 family
        # because this container imports ``amlas``, and the DLC registry account
        # differs per region. Both were wrong when this line composed its own URI.
        from ..config.sagemaker_images import sklearn_training_image

        default_image = sklearn_training_image(region)
        default_instance = "ml.m5.xlarge"
    else:
        default_image = pytorch_training_image(region, gpu=False)
        default_instance = "ml.m5.xlarge"

    image_uri = config.get("image_uri", default_image)
    instance_type = config.get("instance_type", default_instance)
    return image_uri, instance_type


def _build_training_channel(config: dict[str, Any], session_bucket: str) -> list[dict[str, Any]]:
    """Build the ``InputDataConfig`` ``training`` channel from ``config`` (SM-02).

    Behavior-preserving for the existing callers (offline callers may pass neither
    ``content_type`` nor ``input_content_type`` so the channel is byte-for-byte the
    prior shape). The F2 live-AWS fix adds an OPTIONAL ``ContentType`` on the channel
    when the caller supplies ``config["content_type"]`` (or ``input_content_type``):
    SageMaker's BUILT-IN algorithms (e.g. XGBoost) REQUIRE a ``ContentType`` on the
    input channel and a real ``create_training_job`` rejects a CSV channel that omits
    it — so a live built-in-algorithm training job could never be coordinated without
    this seam. Custom-container training (the prior default) leaves it unset and is
    unchanged.
    """
    # SM-02b (LIVE-3): the input channel name defaults to ``training`` (custom-container
    # convention, byte-for-byte for the existing callers), but SageMaker BUILT-IN algorithms
    # name their channel differently — XGBoost requires ``train``. A real built-in training
    # job rejects a mis-named channel, so the caller may override it via ``config["channel_name"]``
    # (default-preserving: absent → ``training`` exactly as before).
    content_type = config.get("content_type") or config.get("input_content_type")

    def _mk(name: str, s3_uri: str) -> dict[str, Any]:
        ch: dict[str, Any] = {
            "ChannelName": name,
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": s3_uri,
                    "S3DataDistributionType": "FullyReplicated",
                }
            },
        }
        if content_type:
            ch["ContentType"] = str(content_type)
        return ch

    channels = [
        _mk(
            str(config.get("channel_name") or "training"),
            config.get("s3_input", f"s3://{session_bucket}/training-data/"),
        )
    ]
    # FIX-34: the tabular entrypoint predicts a STAGED test split to produce the graded
    # y_pred_test.npy, so wire a second ``test`` channel whenever the caller supplies its URI.
    # SageMaker mounts it at SM_CHANNEL_TEST (resolve_sagemaker_paths reads exactly that).
    test_uri = config.get("test_s3_uri")
    if test_uri:
        channels.append(_mk("test", str(test_uri)))
    return channels


def _create_training_job(
    *,
    sm_client: Any,
    job_name: str,
    training_type: str,
    config: dict[str, Any],
    execution_role_arn: str,
    session_bucket: str,
    runtime_session_id: str,
    region: str | None = None,
    job_tags: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Minimal real CreateTrainingJob.

    Per-training-type image/instance selection chooses a GPU image for fine-tune/RL
    training types (SM-01). ``MetricDefinitions`` (so the real job's metrics are
    captured into ``FinalMetricDataList``) and ``HyperParameters`` (the SageMaker SDK
    convention is JSON-encoded string values) are wired from ``config`` with sane
    defaults. Returns the boto3 ``create_training_job`` response (carries the ARN).
    """
    import json as _json

    region = resolve_aws_client_region(sm_client, region, required=not bool(config.get("image_uri")))
    image_uri, instance_type = _select_image_and_instance(training_type, config, region=region)
    metric_definitions = config.get("metric_definitions") or DEFAULT_METRIC_DEFINITIONS

    # SM-03 (LIVE-3): SageMaker BUILT-IN algorithms (XGBoost, Linear-Learner, …) own their metric
    # set and a real ``create_training_job`` REJECTS a caller-supplied ``MetricDefinitions`` with
    # ``ValidationException: You can't override the metric definitions for Amazon SageMaker
    # algorithms``. Custom-container training (the prior default) still wires them. The caller
    # signals a built-in by ``config["omit_metric_definitions"]=True`` (default-preserving: absent →
    # today's behavior, byte-for-byte for the existing custom-container callers).
    algo_spec: dict[str, Any] = {
        "TrainingImage": image_uri,
        "TrainingInputMode": "File",
    }
    if not config.get("omit_metric_definitions"):
        algo_spec["MetricDefinitions"] = list(metric_definitions)

    kwargs: dict[str, Any] = {
        "TrainingJobName": job_name,
        "RoleArn": execution_role_arn,
        "AlgorithmSpecification": algo_spec,
        "InputDataConfig": _build_training_channel(config, session_bucket),
        "OutputDataConfig": {"S3OutputPath": f"s3://{session_bucket}/artifacts/{job_name}/"},
        "ResourceConfig": {
            "InstanceType": instance_type,
            "InstanceCount": 1,
            "VolumeSizeInGB": int(config.get("volume_size_gb", 30)),
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": int(config.get("max_runtime_seconds", 3600))},
        "Environment": {
            "RUNTIME_SESSION_ID": runtime_session_id,
            "AMLAS_TRAINING_TYPE": training_type,
        },
    }

    tags = deployment_job_tags(runtime_session_id) if job_tags is None else job_tags
    if tags:
        kwargs["Tags"] = tags

    # HyperParameters — SageMaker SDK convention is JSON-encoded string values. Merge
    # any sagemaker_* program/source/region keys the caller supplied (test no-op path)
    # plus user hyperparameters from config["hyperparameters"].
    hp: dict[str, str] = {}
    for k, v in (config.get("sagemaker_kwargs") or {}).items():
        hp[k] = v if isinstance(v, str) else _json.dumps(v)
    for k, v in (config.get("hyperparameters") or {}).items():
        hp[k] = v if isinstance(v, str) else _json.dumps(v)

    # FIX-34 (ADR-0002): SCRIPT MODE for the sklearn framework container. Build + upload the amlas
    # source tarball and set the reserved sagemaker_program / sagemaker_submit_directory
    # hyperparameters so the container knows WHAT to run (the prior omission → program=None crash).
    # Gated on training_type=="sklearn" and not already provided, so every other caller is
    # byte-for-byte unchanged. Skippable via config["skip_source_bundle"] (offline offline validation).
    if (
        training_type.lower() == "sklearn"
        and "sagemaker_program" not in hp
        and not config.get("skip_source_bundle")
    ):
        from .sagemaker_source_bundle import script_mode_hyperparameters, upload_source_tarball

        submit_dir = upload_source_tarball(
            session_bucket=session_bucket,
            job_name=job_name,
            region=resolve_aws_region(region),
        )
        for k, v in script_mode_hyperparameters(submit_dir).items():
            hp[k] = v

    if hp:
        kwargs["HyperParameters"] = hp

    # WA-05 (Security/Network): when a ``vpc_config`` is supplied, run the training job INSIDE
    # the private subnets (NetworkStack) so it has no direct internet path. Absent → no
    # VpcConfig (default-preserving). Accepts the neutral ``{subnets, security_group_ids}`` shape
    # (or the verbatim SageMaker ``{Subnets, SecurityGroupIds}``).
    vpc_config = config.get("vpc_config")
    if vpc_config:
        subnets = vpc_config.get("subnets") or vpc_config.get("Subnets") or []
        sgs = vpc_config.get("security_group_ids") or vpc_config.get("SecurityGroupIds") or []
        if subnets and sgs:
            kwargs["VpcConfig"] = {"Subnets": list(subnets), "SecurityGroupIds": list(sgs)}

    result: dict[str, Any] = sm_client.create_training_job(**kwargs)
    return result


def get_job_status(
    job_name: str,
    *,
    jobs_table: str | None = None,
    region: str | None = None,
    dynamodb_client: Any = None,
) -> dict[str, Any]:
    """Return the DynamoDB row for a job. Raises KeyError if missing."""
    ddb = dynamodb_client or boto3.client("dynamodb", region_name=resolve_aws_region(region))
    r = ddb.get_item(TableName=resolve_jobs_table(jobs_table), Key={"job_name": {"S": job_name}})
    item = r.get("Item")
    if not item:
        raise KeyError(f"no such job: {job_name}")
    return {k: next(iter(v.values())) for k, v in item.items()}


def get_training_metrics(
    job_name: str,
    *,
    region: str | None = None,
    sagemaker_client: Any = None,
) -> dict[str, Any]:
    """Read REAL training metrics back from the SageMaker training job (no MLflow).

    Calls ``describe_training_job`` and returns the job status, the real
    ``FinalMetricDataList`` (flattened to ``{metric_name: value}``), the model
    artifacts S3 URI, and any failure reason. This is the de-mock replacement for the
    old ``mlflow_fetch_metrics`` fake (OPEN DECISION 6).
    """
    sm = sagemaker_client or boto3.client("sagemaker", region_name=resolve_aws_region(region))
    desc = sm.describe_training_job(TrainingJobName=job_name)
    metrics = {m["MetricName"]: float(m["Value"]) for m in desc.get("FinalMetricDataList", []) or []}
    return {
        "job_name": job_name,
        "status": desc.get("TrainingJobStatus", ""),
        "secondary_status": desc.get("SecondaryStatus", ""),
        "metrics": metrics,
        "model_artifacts": (desc.get("ModelArtifacts") or {}).get("S3ModelArtifacts", ""),
        "failure_reason": desc.get("FailureReason", ""),
        "training_job_arn": desc.get("TrainingJobArn", ""),
    }


def wait_for_training_job(
    job_name: str,
    *,
    region: str | None = None,
    sagemaker_client: Any = None,
    poll_interval: float = 30.0,
    timeout: float = 1800.0,
) -> str:
    """Poll ``describe_training_job`` until the job reaches a terminal state.

    Returns the terminal ``TrainingJobStatus`` (Completed/Failed/Stopped). Raises
    :class:`TimeoutError` if the job has not terminated within ``timeout`` seconds.
    """
    sm = sagemaker_client or boto3.client("sagemaker", region_name=resolve_aws_region(region))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status: str = sm.describe_training_job(TrainingJobName=job_name)["TrainingJobStatus"]
        if status in TERMINAL_TRAINING_STATES:
            return status
        # Deadline-bounded poll (raises TimeoutError at ``timeout``) — P3 audit H-06.
        time.sleep(poll_interval)  # nosemgrep: arbitrary-sleep
    raise TimeoutError(f"SageMaker training job {job_name} did not reach a terminal state within {timeout}s")
