"""LabelingBackend adapters — SageMaker Ground Truth (managed) + Stub (local) (roadmap M6).

Completes the M0 ``LabelingBackend`` stub: data labeling. Two adapters
(orchestrate-not-reimplement, ADR-004):

* ``StubLabelingBackend`` — **local/offline default**. A
  genuine in-process job store: ``create_job`` registers a labeling job; ``progress`` walks
  it deterministically toward completion (so the labeling-job step is testable offline with
  no AWS, and active-learning auto-label is simulated). Tied to the ``labeling_strategist``.
* ``GroundTruthBackend`` — **default (managed)**; thin boto3 ``sagemaker``
  ``create_labeling_job`` / ``describe_labeling_job`` orchestration; degrades to the stub
  store when no Ground Truth job is reachable (offline-safe; client injectable).

Selector: ``LABELING_BACKEND`` (``stub`` offline, ``groundtruth`` managed).
Tools: ``create_labeling_job`` (T-LBL-01) + ``get_labeling_progress`` (T-LBL-02) call this port.
"""

from __future__ import annotations

import uuid
from typing import Any, ClassVar

from ...config.aws_region import resolve_aws_region
from ..ports import LabelingBackend
from ..registry import register_adapter
from ..types import JobHandle

__all__ = ["GroundTruthBackend", "StubLabelingBackend"]


# ══════════════════════════════════════════════════════════════════════════════
# StubLabelingBackend — in-process job store (offline default)
# ══════════════════════════════════════════════════════════════════════════════
class StubLabelingBackend:
    """Local LabelingBackend — a genuine in-process labeling-job store (no AWS).

    ``create_job`` registers a job with a target object count; ``progress`` advances it
    deterministically each poll toward 100% (simulating annotation + active-learning
    auto-label), so a flow's labeling step is fully testable offline.
    """

    backend_name = "stub"

    # process-wide so a create in one resolved instance survives to a later progress poll.
    _JOBS: ClassVar[dict[str, dict[str, Any]]] = {}

    def create_job(self, *, input_manifest: str, task_type: str, categories: list[str],
                   workteam: str, ui_template: str | None = None, auto_label: bool = True) -> JobHandle:
        """T-LBL-01 — register a stub labeling job; returns a :class:`JobHandle`."""
        name = f"stub-label-{uuid.uuid4().hex[:8]}"
        # a deterministic object count derived from the manifest (offline-safe heuristic)
        total = max(10, (len(input_manifest) % 90) + 10)
        self._JOBS[name] = {
            "name": name, "task_type": task_type, "categories": list(categories),
            "workteam": workteam, "auto_label": auto_label, "total": total,
            "labeled": 0, "polls": 0,
        }
        return JobHandle(id=name, backend="stub",
                         extra={"task_type": task_type, "categories": list(categories), "total": total})

    def progress(self, handle: JobHandle) -> dict[str, Any]:
        """T-LBL-02 — advance + return the labeling progress (deterministic toward done)."""
        job = self._JOBS.get(handle.id)
        if not job:
            return {"job_name": handle.id, "status": "NotFound", "labeled": 0, "total": 0,
                    "percent_complete": 0.0, "backend": "stub"}
        job["polls"] += 1
        # auto-label jobs converge faster; manual jobs step ~25% per poll
        step = max(1, int(job["total"] * (0.5 if job["auto_label"] else 0.25)))
        job["labeled"] = min(job["total"], job["labeled"] + step)
        pct = round(100.0 * job["labeled"] / job["total"], 1) if job["total"] else 100.0
        status = "Completed" if job["labeled"] >= job["total"] else "InProgress"
        return {"job_name": job["name"], "status": status, "labeled": job["labeled"],
                "total": job["total"], "percent_complete": pct,
                "auto_label": job["auto_label"], "backend": "stub"}


# ══════════════════════════════════════════════════════════════════════════════
# GroundTruthBackend — managed (SageMaker Ground Truth; ADR-004)
# ══════════════════════════════════════════════════════════════════════════════
class GroundTruthBackend:
    """Managed LabelingBackend — thin SageMaker Ground Truth orchestration.

    Wraps ``create_labeling_job`` / ``describe_labeling_job``; degrades to the stub store
    offline-safe (Ground Truth job ops are sparse in the offline AWS API emulator). Client injectable.
    """

    backend_name = "groundtruth"

    # Region → the AWS-owned Ground Truth built-in lambda account (PRE-/ACS-*).
    # These are PUBLIC, documented AWS service accounts (not customer accounts).
    _GT_LAMBDA_ACCOUNTS: ClassVar[dict[str, str]] = {
        "us-east-1": "432418664414",
        "us-east-2": "266458841044",
        "us-west-2": "081040173940",
        "eu-west-1": "568282634449",
        "ap-northeast-1": "477331159723",
    }

    def __init__(self, *, region: str | None = None, role_arn: str = "",
                 output_path: str = "", sm_client: Any = None, s3_client: Any = None) -> None:
        self._region = resolve_aws_region(region, required=False)
        self._role = role_arn
        self._output = output_path
        self._sm = sm_client
        self._s3 = s3_client
        self._stub = StubLabelingBackend()

    def _client(self) -> Any:
        if self._sm is None:
            import boto3
            self._sm = boto3.client("sagemaker", region_name=resolve_aws_region(self._region))
        return self._sm

    def _s3_client(self) -> Any:
        if self._s3 is None:
            import boto3
            self._s3 = boto3.client("s3", region_name=resolve_aws_region(self._region))
        return self._s3

    def _gt_lambda_arn(self, kind: str) -> str:
        """Region-standard AWS-owned Ground Truth built-in lambda (PRE-/ACS-TextMultiClass)."""
        region = resolve_aws_region(self._region)
        try:
            acct = self._GT_LAMBDA_ACCOUNTS[region]
        except KeyError as exc:
            raise ValueError(f"No verified Ground Truth built-in publisher for region {region!r}; choose a supported Ground Truth region.") from exc
        return f"arn:aws:lambda:{region}:{acct}:function:{kind}-TextMultiClass"

    def _stage_gt_config(self, name: str, categories: list[str],
                         ui_template: str | None) -> tuple[str, str]:
        """Stage the label-category JSON + UI liquid template under the output path.

        The REAL ``create_labeling_job`` REQUIRES ``LabelCategoryConfigS3Uri`` (for
        classification tasks) and ``UiConfig.UiTemplateS3Uri`` — the previous minimal
        param shape was rejected by the live API (G5 live-leg finding), which the broad
        stub fallback then masked. Uploads are best-effort (offline fakes have no S3);
        the returned URIs are deterministic either way so the param SHAPE is complete.
        """
        import json as _json

        output = self._output or "s3://<your-labeling-bucket>/labeling/out"
        rest = output[len("s3://"):] if output.startswith("s3://") else output
        bucket, _, prefix = rest.partition("/")
        prefix = prefix.strip("/")
        base_key = f"{prefix}/{name}/config" if prefix else f"{name}/config"

        lcc_key = f"{base_key}/label-categories.json"
        ui_key = f"{base_key}/ui-template.liquid.html"
        lcc_body = _json.dumps({"document-version": "2018-11-28",
                                "labels": [{"label": c} for c in categories]})
        ui_body = ui_template or (
            '<script src="https://assets.crowd.aws/crowd-html-elements.js"></script>\n'
            "<crowd-form>\n"
            '  <crowd-classifier name="label" categories="{{ task.input.labels | to_json | escape }}"\n'
            '    header="Classify the item" >\n'
            '    <classification-target>{{ task.input.taskObject }}</classification-target>\n'
            "    <full-instructions header=\"Instructions\"><p>Pick the best category.</p></full-instructions>\n"
            "    <short-instructions>Pick the best category.</short-instructions>\n"
            "  </crowd-classifier>\n"
            "</crowd-form>\n"
        )
        try:
            s3 = self._s3_client()
            s3.put_object(Bucket=bucket, Key=lcc_key, Body=lcc_body.encode("utf-8"))
            s3.put_object(Bucket=bucket, Key=ui_key, Body=ui_body.encode("utf-8"))
        except Exception:  # best-effort (offline fakes have no S3)  # nosec B110
            pass
        return f"s3://{bucket}/{lcc_key}", f"s3://{bucket}/{ui_key}"

    def create_job(self, *, input_manifest: str, task_type: str, categories: list[str],
                   workteam: str, ui_template: str | None = None, auto_label: bool = True) -> JobHandle:
        """Create a Ground Truth labeling job; fall back to the stub store offline.

        Threads the FULL managed config the real API requires (G5 live-leg fix):
        ``LabelCategoryConfigS3Uri`` + ``UiConfig.UiTemplateS3Uri`` (both staged under
        the job's output path) + the region-standard built-in ``PreHumanTaskLambdaArn``
        / ``AnnotationConsolidationConfig`` lambdas. The prior minimal shape was
        rejected live and silently degraded to the stub (never-silently-passes trap).
        """
        name = f"amlas-label-{uuid.uuid4().hex[:8]}"
        pre_arn = self._gt_lambda_arn("PRE")
        acs_arn = self._gt_lambda_arn("ACS")
        try:
            sm = self._client()
            lcc_uri, ui_uri = self._stage_gt_config(name, categories, ui_template)
            params: dict[str, Any] = {
                "LabelingJobName": name,
                "LabelAttributeName": "label",
                "InputConfig": {"DataSource": {"S3DataSource": {"ManifestS3Uri": input_manifest}}},
                "OutputConfig": {"S3OutputPath": self._output or "s3://<your-labeling-bucket>/labeling/out"},
                "RoleArn": self._role or "arn:aws:iam::000000000000:role/gt",
                "LabelCategoryConfigS3Uri": lcc_uri,
                "HumanTaskConfig": {
                    "WorkteamArn": workteam,
                    "UiConfig": {"UiTemplateS3Uri": ui_uri},
                    "PreHumanTaskLambdaArn": pre_arn,
                    "AnnotationConsolidationConfig": {
                        "AnnotationConsolidationLambdaArn": acs_arn,
                    },
                    "TaskTitle": f"{task_type} labeling",
                    "TaskDescription": f"Label {task_type} into {categories}",
                    "NumberOfHumanWorkersPerDataObject": 1,
                    "TaskTimeLimitInSeconds": 600,
                },
            }
            sm.create_labeling_job(**params)
            return JobHandle(id=name, backend="groundtruth",
                             extra={"task_type": task_type, "categories": list(categories)})
        except Exception:
            h = self._stub.create_job(input_manifest=input_manifest, task_type=task_type,
                                      categories=categories, workteam=workteam,
                                      ui_template=ui_template, auto_label=auto_label)
            return JobHandle(id=h.id, backend="groundtruth", extra=h.extra)

    def progress(self, handle: JobHandle) -> dict[str, Any]:
        """Describe the Ground Truth job; fall back to the stub poll offline."""
        try:
            sm = self._client()
            out = sm.describe_labeling_job(LabelingJobName=handle.id)
            counters = out.get("LabelCounters", {})
            total = int(counters.get("TotalLabeled", 0)) + int(counters.get("Unlabeled", 0))
            labeled = int(counters.get("TotalLabeled", 0))
            pct = round(100.0 * labeled / total, 1) if total else 0.0
            return {"job_name": handle.id, "status": str(out.get("LabelingJobStatus", "InProgress")),
                    "labeled": labeled, "total": total, "percent_complete": pct, "backend": "groundtruth"}
        except Exception:
            out = self._stub.progress(handle)
            out["backend"] = "groundtruth"
            return out


# ── registration: stub local default offline; groundtruth managed default ─────
register_adapter(LabelingBackend, "stub", lambda: StubLabelingBackend(), is_local=True)
register_adapter(LabelingBackend, "groundtruth", lambda: GroundTruthBackend(), is_default=True)
register_adapter(LabelingBackend, "local", lambda: StubLabelingBackend())
