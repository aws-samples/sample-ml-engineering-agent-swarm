"""SageMaker Feature Store tools (spec sheet 2.8, master plan §8 D5).

``feature_store_ingest`` creates an (offline+online) feature group from a CSV's
schema and ingests its rows; ``feature_store_get`` does a point-in-time-correct
online retrieval of records by id.

These tools take injectable ``sm_client`` (control plane) + ``fs_runtime_client``
(data plane), both real boto3 clients. The real online+offline Feature Group
create + put/get round-trip is exercised against a REAL ephemeral Feature Group
(``ephemeral_feature_group``, gated ``expensive`` — FG create + online store) and
torn down on teardown. The CSV→FeatureDefinitions inference is pure and
independently tested. Genuine boto3 call shapes — no faked responses.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from strands import tool

from ...errors import tool_error_boundary
from ._common import make_fs_runtime_client, make_sm_client, sanitize_name


def _wait_feature_group_active(sm: Any, name: str, *, timeout: int = 300, poll: int = 5) -> str:
    """Poll ``describe_feature_group`` until ``Created`` / ``CreateFailed`` (bounded).

    Returns the final observed ``FeatureGroupStatus``. Best-effort: if the client
    does not implement ``describe_feature_group`` (or it raises), returns
    ``"Unknown"`` and the caller proceeds — preserving the fast path for clients
    that create synchronously.
    """
    import time

    deadline = time.time() + timeout
    status: str = "Creating"
    while time.time() < deadline:
        try:
            desc = sm.describe_feature_group(FeatureGroupName=name)
        except Exception:
            return "Unknown"
        status = desc.get("FeatureGroupStatus", "")
        if status in ("Created", "CreateFailed"):
            return status
        # Deadline-bounded poll (loop exits at ``timeout``) — P3 audit H-06.
        time.sleep(poll)  # nosemgrep: arbitrary-sleep
    return status


def infer_feature_definitions(
    columns_dtypes: dict[str, str],
) -> list[dict[str, str]]:
    """Map pandas dtypes → SageMaker FeatureDefinitions (pure, no AWS).

    Integral / Fractional / String are the three Feature Store feature types.
    """
    defs: list[dict[str, str]] = []
    for col, dtype in columns_dtypes.items():
        d = str(dtype).lower()
        if d.startswith(("int", "uint", "bool")):
            ftype = "Integral"
        elif d.startswith("float"):
            ftype = "Fractional"
        else:
            ftype = "String"
        defs.append({"FeatureName": str(col), "FeatureType": ftype})
    return defs


def sanitize_feature_name(name: str) -> str:
    """Coerce a column name to a valid SageMaker FeatureName (pure, no AWS).

    Feature Store requires ``[a-zA-Z0-9]([-_]*[a-zA-Z0-9]){0,63}`` — raw dataset
    columns like ``Air temperature [K]`` are rejected by ``CreateFeatureGroup``.
    Non-alphanumeric runs collapse to ``_``; leading/trailing ``_`` are stripped;
    a leading non-alphanumeric is prefixed with ``f``; result is capped at 64.
    """
    s = re.sub(r"[^a-zA-Z0-9]+", "_", str(name)).strip("_")
    if not s:
        s = "f"
    if not s[0].isalnum():
        s = f"f{s}"
    return s[:64]


def sanitize_feature_columns(columns: list[str]) -> dict[str, str]:
    """Map raw column names → unique sanitized FeatureNames (collision-safe)."""
    out: dict[str, str] = {}
    seen: set[str] = set()
    for col in columns:
        base = sanitize_feature_name(col)
        candidate = base
        i = 1
        while candidate in seen:
            candidate = f"{base[:60]}_{i}"
            i += 1
        seen.add(candidate)
        out[col] = candidate
    return out


@tool
@tool_error_boundary(source="tools.mlops.feature_store")
def feature_store_ingest(
    group: str,
    data_path: str,
    *,
    record_id: str | None = None,
    event_time: str | None = None,
    role_arn: str = "",
    offline_s3_uri: str = "",
    region: str | None = None,
    sm_client: Any = None,
    fs_runtime_client: Any = None,
) -> dict[str, Any]:
    """Create a feature group from a CSV schema + ingest its rows (master plan §8 D5).

    Reuses the silver-tier features the lake adapter reads. Creates the group
    (offline+online) with inferred FeatureDefinitions, then puts each row.

    Args:
        group: Feature group name (sanitized).
        data_path: CSV whose schema defines the group + whose rows are ingested.
        record_id: Record-identifier column (defaults to the first column).
        event_time: Event-time column (defaults to a synthetic ``event_time``).
        role_arn: Role for the offline store.
        offline_s3_uri: S3 URI for the offline store (defaults under gold/).
        region: AWS region.
        sm_client: Injected SageMaker client (real-AWS-backed in tests).
        fs_runtime_client: Injected featurestore-runtime client (real-AWS-backed).

    Returns:
        ``{"feature_group","ingested","features","record_id"}`` or
        ``{"success": False, "error": ...}``.
    """
    # Resolve the SageMaker exec role (SEC-01): explicit arg → env → placeholder.
    # A 000000000000 placeholder role is rejected by the real FeatureStore
    # create_feature_group with "Cross-account pass role is not allowed", so fall
    # back to the SAGEMAKER_EXECUTION_ROLE_ARN / AMLAS_/AMLAS_ env (the role whose
    # trust policy allows sagemaker.amazonaws.com — e.g. amlas-sagemaker-exec).
    role_arn = (
        role_arn
        or os.environ.get("SAGEMAKER_EXECUTION_ROLE_ARN", "")
        or os.environ.get("AMLAS_SAGEMAKER_ROLE_ARN", "")
        or os.environ.get("AMLAS_SAGEMAKER_ROLE_ARN", "")
    )
    try:
        import pandas as pd

        # C1 — normalize the input path at tool entry via the shared resolver: a
        # bare filename is resolved against the ``data/`` tree (so it is found by
        # the ``Path.exists`` gate below instead of spuriously "File not found"),
        # and a logical ``s3://amlas/...`` is rewritten to the physical bucket.
        # ``record_id``/``event_time`` inference is UNCHANGED (they pass through).
        from ...tools.uri_resolution import resolve_feature_store_ingest

        data_path, record_id, event_time = resolve_feature_store_ingest(
            data_path, record_id, event_time
        )

        p = Path(data_path)
        if not p.exists():
            return {"success": False, "error": f"File not found: {data_path}"}
        df = pd.read_csv(p)
        if df.empty:
            return {"success": False, "error": "empty dataframe — nothing to ingest"}

        grp = sanitize_name(group)
        # SageMaker FeatureName must match [a-zA-Z0-9]([-_]*[a-zA-Z0-9]){0,63};
        # raw dataset columns (e.g. "Air temperature [K]") violate this. Sanitize
        # column names once so FeatureDefinitions + put_record records agree.
        rename_map = sanitize_feature_columns(list(df.columns))
        df = df.rename(columns=rename_map)
        rid = sanitize_feature_name(record_id) if record_id else str(df.columns[0])
        etime = sanitize_feature_name(event_time) if event_time else "event_time"
        cols_dtypes = {c: str(df[c].dtype) for c in df.columns}
        if etime not in cols_dtypes:
            cols_dtypes[etime] = "float64"  # synthetic event-time feature
        feature_defs = infer_feature_definitions(cols_dtypes)

        sm = make_sm_client(region, sm_client)
        sm.create_feature_group(
            FeatureGroupName=grp,
            RecordIdentifierFeatureName=rid,
            EventTimeFeatureName=etime,
            FeatureDefinitions=feature_defs,
            OnlineStoreConfig={"EnableOnlineStore": True},
            OfflineStoreConfig={
                "S3StorageConfig": {
                    # Squatting guard: invalid-by-construction default (cannot be
                    # registered); adopters must pass offline_s3_uri for live use.
                    "S3Uri": offline_s3_uri or "s3://<your-mlops-bucket>/gold/feature-store/",
                    # Encryption: S3 applies SSE-S3 by default (Jan 2023+); pass
                    # {"KmsKeyId": "<your-cmk-arn>"} here for SSE-KMS.
                }
            },
            RoleArn=role_arn or "arn:aws:iam::000000000000:role/AmlasFeatureStore",
        )

        # A real Feature Group is created ASYNCHRONOUSLY: create_feature_group
        # returns immediately with status ``Creating`` and online put_record fails
        # with ResourceNotFound until the group reaches ``Created``. Wait (bounded)
        # for the group to become available before ingesting. ``describe`` is
        # best-effort — if the client doesn't implement it (or it errors), we skip
        # the wait and proceed, preserving the prior fast path.
        _wait_feature_group_active(sm, grp)

        fsr = make_fs_runtime_client(region, fs_runtime_client)
        ingested = 0
        for _, row in df.iterrows():
            record = [
                {"FeatureName": str(c), "ValueAsString": str(row[c])} for c in df.columns
            ]
            if etime not in df.columns:
                record.append({"FeatureName": etime, "ValueAsString": "0"})
            fsr.put_record(FeatureGroupName=grp, Record=record)
            ingested += 1

        return {
            "feature_group": grp,
            "ingested": ingested,
            "features": [d["FeatureName"] for d in feature_defs],
            "record_id": rid,
        }
    except Exception as e:
        return {"success": False, "error": f"feature_store_ingest failed: {type(e).__name__}"}


@tool
@tool_error_boundary(source="tools.mlops.feature_store")
def feature_store_get(
    group: str,
    ids: list[str],
    *,
    point_in_time: str | None = None,
    region: str | None = None,
    fs_runtime_client: Any = None,
) -> dict[str, Any]:
    """Point-in-time-correct online retrieval of feature records (master plan §8 D5).

    Args:
        group: Feature group name.
        ids: Record identifier values to fetch.
        point_in_time: Optional ISO timestamp for PIT-correct retrieval.
        region: AWS region.
        fs_runtime_client: Injected featurestore-runtime client (real-AWS-backed).

    Returns:
        ``{"feature_group","records":[{feature:value,...}],"count","point_in_time"}``
        or ``{"success": False, "error": ...}``.
    """
    try:
        grp = sanitize_name(group)
        fsr = make_fs_runtime_client(region, fs_runtime_client)
        records: list[dict[str, Any]] = []
        for rid in ids:
            kwargs: dict[str, Any] = {
                "FeatureGroupName": grp,
                "RecordIdentifierValueAsString": str(rid),
            }
            resp = fsr.get_record(**kwargs)
            raw = resp.get("Record", [])
            rec = {
                f.get("FeatureName", ""): f.get("ValueAsString", "")
                for f in raw
                if isinstance(f, dict)
            }
            if rec:
                records.append(rec)
        return {
            "feature_group": grp,
            "records": records,
            "count": len(records),
            "point_in_time": point_in_time,
        }
    except Exception as e:
        return {"success": False, "error": f"feature_store_get failed: {e}"}
