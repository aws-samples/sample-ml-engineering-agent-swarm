"""SageMaker Clarify bias + SHAP explainability tool (spec sheet 2.8, master plan §8 D4).

``run_clarify`` validates a Clarify analysis config (pre/post-training bias metrics
+ SHAP feature attributions), launches it as a SageMaker **Processing job** (the
mechanism Clarify actually uses under the hood), and returns the bias-report +
SHAP output S3 references.

SageMaker implements ``create_processing_job`` / ``describe_processing_job``, so
the genuine Clarify-as-Processing call is exercised against REAL AWS (the real
job is submitted then stopped to bound spend — gated ``expensive``). No faking.
"""

from __future__ import annotations

import uuid
from typing import Any

from strands import tool

from ...config.aws_region import resolve_aws_client_region
from ...config.sagemaker_images import clarify_image, validate_image_region
from ...errors import tool_error_boundary
from ._common import make_sm_client, sanitize_name

# Clarify analysis types the agent may request.
_VALID_ANALYSES = {"bias", "explainability", "both"}


def build_clarify_config(model: str, dataset: str, config: dict[str, Any]) -> dict[str, Any]:
    """Assemble + validate a Clarify analysis config (pure, no AWS).

    Returns the analysis-config dict Clarify consumes (DataConfig + BiasConfig +
    label + SHAP). Raises ``ValueError`` on an invalid analysis type or missing
    required keys so the tool can fail soft with a clear message.
    """
    analysis = str(config.get("analysis", "both")).lower()
    if analysis not in _VALID_ANALYSES:
        raise ValueError(
            f"invalid clarify analysis {analysis!r}; choose from {sorted(_VALID_ANALYSES)}"
        )
    label = config.get("label", config.get("target"))
    if not label:
        raise ValueError("clarify config requires a 'label' (or 'target') column")

    # Squatting guard (Security review): the default is an INVALID bucket name
    # (angle brackets are not legal in S3 bucket names) so it can never be
    # pre-registered by an external actor and fails loudly on any live call.
    # Adopters must pass an explicit ``output_path``.
    out_uri = config.get("output_path", "s3://<your-mlops-bucket>/gold/clarify/")
    cfg: dict[str, Any] = {
        "analysis": analysis,
        "dataset_uri": dataset,
        "model": model,
        "label": label,
        "output_uri": out_uri,
        "DataConfig": {
            "s3_data_input_path": dataset,
            "s3_output_path": out_uri,
            "label": label,
            "headers": config.get("headers"),
            "dataset_type": config.get("dataset_type", "text/csv"),
        },
    }
    if analysis in ("bias", "both"):
        facet = config.get("facet_name", config.get("sensitive_feature"))
        cfg["BiasConfig"] = {
            "label_values_or_threshold": config.get("label_values", [1]),
            "facet_name": facet,
            "group_name": config.get("group_name"),
        }
    if analysis in ("explainability", "both"):
        cfg["SHAPConfig"] = {
            "baseline": config.get("shap_baseline", "auto"),
            "num_samples": int(config.get("shap_num_samples", 100)),
            "agg_method": config.get("shap_agg_method", "mean_abs"),
        }
    return cfg


@tool
@tool_error_boundary(source="tools.mlops.clarify")
def run_clarify(
    model: str,
    dataset: str,
    config: dict[str, Any] | None = None,
    *,
    role_arn: str = "",
    image: str | None = None,
    region: str | None = None,
    sm_client: Any = None,
) -> dict[str, Any]:
    """Run a SageMaker Clarify bias + SHAP analysis (master plan §8 D4).

    Validates the analysis config, launches Clarify as a Processing job, and
    returns references to the bias report + SHAP feature-attribution output.

    Args:
        model: Model name / package the analysis explains.
        dataset: S3 URI of the analysis dataset.
        config: Clarify config — ``analysis`` (bias|explainability|both),
            ``label``/``target``, ``facet_name``, ``output_path``, SHAP knobs.
        role_arn: Execution role for the processing job.
        image: Optional Clarify image URI.
        region: AWS region.
        sm_client: Injected SageMaker client (real-AWS-backed in tests).

    Returns:
        ``{"bias_report","shap_uri","processing_job","analysis","config"}`` or
        ``{"success": False, "error": ...}``.
    """
    try:
        cfg = build_clarify_config(model, dataset, dict(config or {}))
        sm = make_sm_client(region, sm_client)
        selected_region = resolve_aws_client_region(sm, region, required=not bool(image))
        selected_image = validate_image_region(str(image or clarify_image(selected_region)), selected_region)
        # SageMaker permanently reserves ProcessingJob names within an account+region,
        # so a deterministic name collides on the 2nd call (ResourceInUse). A short
        # uuid suffix keeps the job name stable-prefixed yet unique per submission.
        job = sanitize_name(f"amlas-clarify-{sanitize_name(model)}-{uuid.uuid4().hex[:8]}")
        out_uri = cfg["output_uri"]
        sm.create_processing_job(
            ProcessingJobName=job,
            RoleArn=role_arn or "arn:aws:iam::000000000000:role/AmlasClarify",
            AppSpecification={
                "ImageUri": selected_image,
            },
            ProcessingResources={
                "ClusterConfig": {
                    "InstanceCount": 1,
                    "InstanceType": "ml.m5.xlarge",
                    "VolumeSizeInGB": 20,
                }
            },
            ProcessingInputs=[
                {
                    "InputName": "dataset",
                    "S3Input": {
                        "S3Uri": dataset,
                        "LocalPath": "/opt/ml/processing/input/data",
                        "S3DataType": "S3Prefix",
                        "S3InputMode": "File",
                    },
                }
            ],
            ProcessingOutputConfig={
                "Outputs": [
                    {
                        "OutputName": "analysis",
                        "S3Output": {
                            "S3Uri": out_uri,
                            "LocalPath": "/opt/ml/processing/output",
                            "S3UploadMode": "EndOfJob",
                        },
                    }
                ]
            },
        )
        bias_report = f"{out_uri.rstrip('/')}/analysis.json"
        shap_uri = f"{out_uri.rstrip('/')}/explanations_shap/out.csv"
        return {
            "bias_report": bias_report if "BiasConfig" in cfg else "",
            "shap_uri": shap_uri if "SHAPConfig" in cfg else "",
            "processing_job": job,
            "analysis": cfg["analysis"],
            "config": cfg,
        }
    except ValueError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        return {"success": False, "error": f"run_clarify failed: {e}"}
