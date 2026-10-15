"""Backend-neutral tools — the EVOLVE'd tool layer that calls PORTS, not boto3.

This is the doc 05 §4.4 before/after made concrete for M0: each tool here resolves
a port via the ``BackendRegistry`` and delegates to whatever adapter the registry
binds (the SageMaker/AWS adapter by default, the local adapter offline/under test).
The adapters wrap the EXISTING implementations, so the OUTPUT is identical to the
pre-seam tools — this milestone adds a seam, it does not change behavior (ADR-001).

Naming (roadmap M0 work-item 4): ``submit_training_run`` (T-TRAIN-01) is the new
backend-neutral canonical name; today's ``submit_training_job`` becomes the
``SageMakerTrainingBackend`` internals (unchanged). These NEW tools are ADDITIVE —
the existing tools/agents keep working untouched, so the full suite stays green;
flows + the control plane can adopt the port-calling tools incrementally.

The lake write tool additionally runs the **GuardrailHooks** ``PreToolUse`` layer
(``hooks.py``) so the bronze-write deny fires as a declarative guardrail (ADR-015)
in front of the port — the seam M2's cost/leakage/quota hooks plug into.
"""

from __future__ import annotations

from typing import Any

from strands import tool

from ..backends.hooks import HookContext, default_guardrail_hooks
from ..backends.ports import (
    EvaluationBackend,
    LakeBackend,
    ModelRegistryBackend,
    TrainingBackend,
)
from ..backends.registry import REGISTRY
from ..errors import tool_error_boundary

# One process-wide guardrail-hooks instance (M0: bronze-write deny only).
_HOOKS = default_guardrail_hooks()

__all__ = [
    "grade_submission",
    "register_model",
    "submit_training_run",
    "write_lake_dataset_via_port",
]


@tool
@tool_error_boundary(source="tools.backend_neutral")
def submit_training_run(
    algo: str,
    train_uri: str,
    val_uri: str | None = None,
    target: str | None = None,
    hyperparams: dict[str, Any] | None = None,
    resources: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Submit a training run for ``algo`` on ``train_uri`` (T-TRAIN-01, backend-neutral).

    Resolves :class:`TrainingBackend` via the registry (SageMaker by default, local
    sklearn offline/small) and submits — returning the job handle + initial status +
    metrics. The chat agent is unaware which adapter ran (doc 05 §4.4); the AWS
    adapter reproduces today's exact 3-step DDB submit.

    Args:
        algo: Algorithm / training type (e.g. ``xgboost``, ``sft``).
        train_uri: Training data location (S3 URI or local CSV path).
        val_uri: Optional validation data location.
        target: Target column (local adapter) — passed through ``resources`` for AWS.
        hyperparams: Hyperparameters for the run.
        resources: Adapter resources (instance type, role/bucket, seed, task_type …).

    Returns:
        ``{"job_id","backend","status","metrics","adapter","reason"}`` or
        ``{"success": False, "error": ...}`` (fail-soft, matching the tool convention).
    """
    try:
        resolved = REGISTRY.resolve(TrainingBackend)
        backend = resolved.adapter
        handle = backend.submit(
            algo=algo, train_uri=train_uri, val_uri=val_uri, target=target,
            hyperparams=hyperparams, resources=resources,
        )
        status = backend.status(handle)
        return {
            "job_id": handle.id,
            "backend": handle.backend,
            "status": status.state.value,
            "metrics": backend.metrics(handle) if status.state.value == "SUCCEEDED" else {},
            "adapter": resolved.adapter_name,
            "reason": resolved.reason,
        }
    except Exception as e:
        return {"success": False, "error": f"submit_training_run failed: {e}"}


@tool
@tool_error_boundary(source="tools.backend_neutral")
def grade_submission(
    spec: Any,
    y_true: Any,
    y_pred: Any,
    baseline_score: float | None = None,
) -> dict[str, Any]:
    """Grade ``y_pred`` vs ``y_true`` through :class:`EvaluationBackend` (T-EVAL-01).

    The decision is the deterministic Arbiter — **never delegated** (ADR-003). The
    ``DeterministicArbiter`` adapter returns the IDENTICAL ``ArbiterReport`` the
    Arbiter produces directly (the behavior-preservation test asserts equality).
    Returns the report as a dict so it is JSON-serializable for the chat surface.
    """
    try:
        backend = REGISTRY.resolve(EvaluationBackend).adapter
        report = backend.grade(spec=spec, y_true=y_true, y_pred=y_pred, baseline_score=baseline_score)
        # ArbiterReport is a pydantic model → dict; keep the rich shape unchanged.
        return report.model_dump() if hasattr(report, "model_dump") else dict(report)
    except Exception as e:
        return {"success": False, "error": f"grade_submission failed: {e}"}


@tool
@tool_error_boundary(source="tools.backend_neutral")
def write_lake_dataset_via_port(
    local_path: str,
    dest_uri: str,
    tier: str = "silver",
    lineage: dict[str, Any] | None = None,
    profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a dataset back to the lake through :class:`LakeBackend` (T-LAKE-02).

    Runs the GuardrailHooks ``PreToolUse`` layer FIRST — the bronze-write deny
    (ADR-009 → ADR-015) blocks a bronze/raw target as a declarative guardrail before
    the port is even resolved. If allowed, resolves ``LakeBackend`` (S3 medallion by
    default) and writes — the adapter wraps the EXISTING ``write_lake_dataset`` so
    the lineage sidecar + tier governance are byte-for-byte today's.

    Returns ``{"written_uri","tier","adapter","reason"}`` on success, or
    ``{"success": False, "error": ..., "denied_by": ...}`` when a guardrail denies.
    """
    # 1) PreToolUse guardrail — deterministic, non-overridable (ADR-015).
    ctx = HookContext(
        tool="write_lake_dataset",
        action="lake_write",
        args={"local_path": local_path, "dest_uri": dest_uri, "tier": tier},
    )
    verdict = _HOOKS.check_pre(ctx)
    if not verdict.allowed:
        return {"success": False, "error": verdict.reason, "denied_by": verdict.detail.get("hook", "guardrail")}

    # 2) resolve the port + write (adapter wraps the EXISTING tool — behavior-preserving).
    try:
        resolved = REGISTRY.resolve(LakeBackend)
        written_uri = resolved.adapter.write_dataset(
            local_path=local_path, dest_uri=dest_uri, tier=tier, lineage=lineage, profile=profile,
        )
        return {"written_uri": written_uri, "tier": tier, "adapter": resolved.adapter_name, "reason": resolved.reason}
    except Exception as e:
        return {"success": False, "error": f"write_lake_dataset_via_port failed: {e}"}


@tool
@tool_error_boundary(source="tools.backend_neutral")
def register_model(
    group: str,
    artifact: str,
    metrics: dict[str, Any] | None = None,
    approval: str = "PendingManualApproval",
    image: str | None = None,
) -> dict[str, Any]:
    """Register a model package through :class:`ModelRegistryBackend` (T-REG-01).

    Resolves the registry port (SageMaker Registry by default, JSON-file store
    offline) and registers a versioned package — the AWS adapter wraps the EXISTING
    ``register_model_sm`` so the call shape is unchanged.

    Returns ``{"model_package_arn","group","version","approval_status","adapter"}``
    or ``{"success": False, "error": ...}``.
    """
    try:
        resolved = REGISTRY.resolve(ModelRegistryBackend)
        ref = resolved.adapter.register(group=group, artifact=artifact, image=image, metrics=metrics, approval=approval)
        return {
            "model_package_arn": ref.arn,
            "group": ref.group,
            "version": ref.version,
            "approval_status": ref.approval_status,
            "adapter": resolved.adapter_name,
        }
    except Exception as e:
        return {"success": False, "error": f"register_model failed: {e}"}
