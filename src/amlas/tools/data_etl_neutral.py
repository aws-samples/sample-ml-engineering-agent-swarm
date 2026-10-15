"""Stage 4–5 data-quality + ETL tools that call the M2 ports (doc 06 T-DQ-*/T-ETL-*).

These ``@tool`` functions are **born backend-neutral** (roadmap M2 / doc 05 §4.4): each
resolves its port via ``BackendRegistry.resolve`` and calls the resolved adapter — never
boto3 directly. Offline/test resolves to the in-process adapter (identical to today's
behavior); an operator pins ``DQ_BACKEND``/``ETL_BACKEND`` (or a steering hard-pin) to
route the SAME tool to DataBrew/Glue-DQ/Processing/Glue at scale (the swappable-module
contract). The T-IDs are cited per-tool for the Phase-8 traceability matrix.

Tools (T-IDs):
  * ``run_profile``              — T-DQ-01 (DataQualityBackend.profile)
  * ``recommend_dq_rules``       — T-DQ-02 (DataQualityBackend.recommend_rules)
  * ``evaluate_dq``              — T-DQ-03 (DataQualityBackend.evaluate)
  * ``apply_imputation_via_port``— T-DQ-11 (DataQualityBackend.impute, swappable module)
  * ``run_processing_job``       — T-ETL-01 (ETLBackend.run_transform on Processing)
  * ``run_glue_job``             — T-ETL-02 (ETLBackend.run_transform on Glue)
  * ``run_crawler``              — T-ETL-03 (ETLBackend.run_crawl)
  * ``dedupe_dataset``           — T-ETL-07 (ETLBackend dedupe op)
  * ``apply_transform_flow``     — T-ETL-05 (ETLBackend generic transform)
  * ``auto_preprocess_via_port`` — T-ETL-14 EVOLVE (auto_preprocess through the port)
  * ``register_dataset_version`` — T-LAKE-04 (write silver/gold via LakeBackend)

Each returns a JSON-serializable dict and fails soft (``{"success": False, "error": …}``)
matching the existing tool convention.
"""

from __future__ import annotations

from typing import Any

from strands import tool

from ..backends.ports import DataQualityBackend, ETLBackend, LakeBackend
from ..backends.registry import REGISTRY
from ..errors import tool_error_boundary

__all__ = [
    "apply_imputation_via_port",
    "apply_transform_flow",
    "auto_preprocess_via_port",
    "dedupe_dataset",
    "evaluate_dq",
    "recommend_dq_rules",
    "register_dataset_version",
    "run_crawler",
    "run_glue_job",
    "run_processing_job",
    "run_profile",
]


def _resolve(port: type, steering: Any = None, scale: Any = None) -> Any:
    """Resolve ``port`` → adapter (offline → local; env/steering pins managed)."""
    return REGISTRY.resolve(port, steering=steering, scale=scale)


# ══════════════════════════════════════════════════════════════════════════════
# DataQuality tools (T-DQ-*)
# ══════════════════════════════════════════════════════════════════════════════
@tool
@tool_error_boundary(source="tools.data_etl_neutral")
def run_profile(dataset_uri: str, scale: int | str | None = None) -> dict[str, Any]:
    """T-DQ-01 — profile a dataset via the ``DataQualityBackend`` port.

    Offline → the in-process 6-dim score + missingness diagnosis (unchanged); pin
    ``DQ_BACKEND=databrew`` to run a managed DataBrew profile job at scale. Returns the
    profile report dict + the resolved adapter for traceability.
    """
    try:
        resolved = _resolve(DataQualityBackend, scale=scale)
        report = resolved.adapter.profile(dataset_uri=dataset_uri)
        return {
            "success": True,
            "adapter": resolved.adapter_name,
            "reason": resolved.reason,
            "score": getattr(report, "score", 0.0),
            "rows": getattr(report, "rows", 0),
            "columns": getattr(report, "columns", 0),
            "dimensions": getattr(report, "dimensions", {}),
            "detail": getattr(report, "detail", {}),
        }
    except Exception as e:
        return {"success": False, "error": f"run_profile failed: {e}"}


@tool
@tool_error_boundary(source="tools.data_etl_neutral")
def recommend_dq_rules(dataset_uri: str) -> dict[str, Any]:
    """T-DQ-02 — recommend a data-quality ruleset via the ``DataQualityBackend`` port.

    Offline → a simple profile-derived expectation set; pin ``DQ_BACKEND=glue_dq`` to
    trigger a managed Glue Data Quality rule-recommendation run (DQDL).
    """
    try:
        resolved = _resolve(DataQualityBackend)
        adapter = resolved.adapter
        if not hasattr(adapter, "recommend_rules"):
            return {"success": False, "error": f"adapter {resolved.adapter_name} has no recommend_rules"}
        out = adapter.recommend_rules(dataset_uri=dataset_uri)
        return {"success": True, "adapter": resolved.adapter_name, **out}
    except Exception as e:
        return {"success": False, "error": f"recommend_dq_rules failed: {e}"}


@tool
@tool_error_boundary(source="tools.data_etl_neutral")
def evaluate_dq(dataset_uri: str, ruleset: Any = None) -> dict[str, Any]:
    """T-DQ-03 — evaluate a dataset against a ruleset via the ``DataQualityBackend`` port.

    Offline → the pandera ``check_expectations`` gate; pin ``DQ_BACKEND=glue_dq`` to run
    a managed Glue DQ DQDL evaluation. Returns ``{passed, score, violations}``.
    """
    try:
        resolved = _resolve(DataQualityBackend)
        dq = resolved.adapter.evaluate(dataset_uri=dataset_uri, ruleset=ruleset)
        return {
            "success": True,
            "adapter": resolved.adapter_name,
            "passed": dq.passed,
            "score": dq.score,
            "violations": dq.violations,
        }
    except Exception as e:
        return {"success": False, "error": f"evaluate_dq failed: {e}"}


@tool
@tool_error_boundary(source="tools.data_etl_neutral")
def apply_imputation_via_port(
    dataset_uri: str, plan: dict[str, Any], scale: int | str | None = None
) -> dict[str, Any]:
    """T-DQ-11 — apply an imputation ``plan`` via the ``DataQualityBackend.impute`` port.

    THE swappable-module tool (doc 05 §4.5): the SAME call runs the in-process pandas
    apply offline OR a managed DataBrew recipe job when ``DQ_BACKEND=databrew``/at scale.
    The flow contract is invariant — "give me an imputed dataset"; the module swaps.
    """
    try:
        resolved = _resolve(DataQualityBackend, scale=scale)
        ref = resolved.adapter.impute(dataset_uri=dataset_uri, plan=plan)
        return {
            "success": True,
            "adapter": resolved.adapter_name,
            "reason": resolved.reason,
            "imputed_uri": ref.uri,
            "tier": ref.tier,
            "lineage": ref.lineage,
        }
    except Exception as e:
        return {"success": False, "error": f"apply_imputation_via_port failed: {e}"}


# ══════════════════════════════════════════════════════════════════════════════
# ETL tools (T-ETL-*)
# ══════════════════════════════════════════════════════════════════════════════
@tool
@tool_error_boundary(source="tools.data_etl_neutral")
def run_processing_job(
    input_uri: str,
    transform_spec: dict[str, Any],
    output: str,
    resources: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """T-ETL-01 — run a transform via the ``ETLBackend`` (managed SageMaker Processing).

    Pin ``ETL_BACKEND=processing`` to submit a SageMaker Processing job; offline this
    runs the equivalent in-process pandas transform. The generic Processing GAP tool.
    """
    try:
        # C1 — normalize a sparse NL-derived call at tool entry: rewrite a logical
        # ``s3://amlas/...`` input to the physical lake bucket + fill smart defaults
        # for an omitted ``output`` / ``transform_spec`` (shared resolver; the same
        # seam run_glue_job/feature_store_ingest use). Return shape is unchanged.
        from .uri_resolution import resolve_processing_io

        input_uri, output, transform_spec = resolve_processing_io(
            input_uri, output, transform_spec
        )
        resolved = _resolve(ETLBackend)
        out = resolved.adapter.run_transform(
            input_uri=input_uri, transform_spec=transform_spec, output=output, resources=resources
        )
        return {"adapter": resolved.adapter_name, "reason": resolved.reason, **out}
    except Exception as e:
        return {"success": False, "error": f"run_processing_job failed: {e}"}


@tool
@tool_error_boundary(source="tools.data_etl_neutral")
def run_glue_job(
    input_uri: str,
    transform_spec: dict[str, Any],
    output: str,
    resources: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """T-ETL-02 — run a Glue ETL job via the ``ETLBackend`` (pin ``ETL_BACKEND=glue``)."""
    try:
        # C1 — same entry normalization as run_processing_job (shared resolver):
        # logical→physical bucket rewrite + smart ``output``/``transform_spec``
        # defaults for a sparse call. Return shape unchanged.
        from .uri_resolution import resolve_processing_io

        input_uri, output, transform_spec = resolve_processing_io(
            input_uri, output, transform_spec
        )
        resolved = _resolve(ETLBackend)
        out = resolved.adapter.run_transform(
            input_uri=input_uri, transform_spec=transform_spec, output=output, resources=resources
        )
        return {"adapter": resolved.adapter_name, "reason": resolved.reason, **out}
    except Exception as e:
        return {"success": False, "error": f"run_glue_job failed: {e}"}


@tool
@tool_error_boundary(source="tools.data_etl_neutral")
def run_crawler(s3_target: str) -> dict[str, Any]:
    """T-ETL-03 — crawl + catalog ``s3_target`` via the ``ETLBackend.run_crawl`` port.

    Offline → a local schema inventory; pin ``ETL_BACKEND=glue`` to start a Glue crawler.
    """
    try:
        resolved = _resolve(ETLBackend)
        out = resolved.adapter.run_crawl(s3_target=s3_target)
        return {"adapter": resolved.adapter_name, "reason": resolved.reason, **out}
    except Exception as e:
        return {"success": False, "error": f"run_crawler failed: {e}"}


@tool
@tool_error_boundary(source="tools.data_etl_neutral")
def dedupe_dataset(dataset_uri: str, keys: list[str] | None = None, strategy: str = "exact",
                   output: str = "") -> dict[str, Any]:
    """T-ETL-07 — de-duplicate rows via the ``ETLBackend`` (in-proc / Processing at scale)."""
    try:
        resolved = _resolve(ETLBackend)
        spec = {"op": "dedupe", "keys": keys, "strategy": strategy}
        out = resolved.adapter.run_transform(
            input_uri=dataset_uri, transform_spec=spec, output=output or dataset_uri
        )
        return {"adapter": resolved.adapter_name, **out}
    except Exception as e:
        return {"success": False, "error": f"dedupe_dataset failed: {e}"}


@tool
@tool_error_boundary(source="tools.data_etl_neutral")
def apply_transform_flow(dataset_uri: str, transform_spec: dict[str, Any], output: str = "") -> dict[str, Any]:
    """T-ETL-05 — apply a generic transform spec via the ``ETLBackend`` port.

    The scale "apply" module for the preprocessing/imputation flows: in-process pandas
    offline, Data-Wrangler→Processing / Glue when a managed backend is pinned.
    """
    try:
        resolved = _resolve(ETLBackend)
        out = resolved.adapter.run_transform(
            input_uri=dataset_uri, transform_spec=transform_spec, output=output or dataset_uri
        )
        return {"adapter": resolved.adapter_name, "reason": resolved.reason, **out}
    except Exception as e:
        return {"success": False, "error": f"apply_transform_flow failed: {e}"}


@tool
@tool_error_boundary(source="tools.data_etl_neutral")
def auto_preprocess_via_port(path: str, target_column: str, output_dir: str | None = None,
                             drop_threshold: float = 0.5) -> dict[str, Any]:
    """T-ETL-14 (EVOLVE) — ``auto_preprocess`` through the ``ETLBackend`` port.

    Behavior-preserving: offline this resolves to ``InProcessPandasBackend`` which calls
    the EXISTING ``fe_tools.auto_preprocess`` — identical result — but now via the seam,
    so the SAME tool can graduate to Processing at scale.
    """
    try:
        resolved = _resolve(ETLBackend)
        spec = {
            "op": "auto_preprocess",
            "target_column": target_column,
            "output_dir": output_dir,
            "drop_threshold": drop_threshold,
        }
        out = resolved.adapter.run_transform(
            input_uri=path, transform_spec=spec, output=output_dir or path
        )
        return {"adapter": resolved.adapter_name, **out}
    except Exception as e:
        return {"success": False, "error": f"auto_preprocess_via_port failed: {e}"}


# ══════════════════════════════════════════════════════════════════════════════
# Lake versioning (T-LAKE-04) — promote a model-ready dataset to silver/gold
# ══════════════════════════════════════════════════════════════════════════════
@tool
@tool_error_boundary(source="tools.data_etl_neutral")
def register_dataset_version(local_path: str, dest_uri: str, tier: str = "silver",
                             lineage: dict[str, Any] | None = None) -> dict[str, Any]:
    """T-LAKE-04 — version a cleaned dataset to silver/gold via the ``LakeBackend`` port.

    Preserves the medallion bronze-write rejection (ADR-009): a bronze/raw target raises
    (and the M0 ``bronze_write_deny`` hook also denies). Returns the written URI.
    """
    try:
        resolved = _resolve(LakeBackend)
        uri = resolved.adapter.write_dataset(
            local_path=local_path, dest_uri=dest_uri, tier=tier, lineage=lineage
        )
        return {"success": True, "adapter": resolved.adapter_name, "written_uri": uri, "tier": tier}
    except Exception as e:
        return {"success": False, "error": f"register_dataset_version rejected: {e}"}
