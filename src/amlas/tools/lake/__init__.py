"""Medallion lakehouse adapter — read-subset / write-back (master plan §8, spec 2.9).

The customer's lakehouse uses the medallion model (bronze/silver/gold). The
transcript asks for exactly two operations (17:33–20:46):

  1. **Read a SUBSET** (a named silver / semantic-layer "vertical slice", NOT
     "all petabytes") — ``read_lake_slice`` resolves a ``silver://db.table`` or
     ``s3://...`` URI (optionally via the Glue Data Catalog), applies
     column/filter/limit, and returns a concrete local DataFrame artifact + the
     provenance of where it came from.
  2. **Write inferred/cleaned data BACK** to the appropriate tier — "maybe gold,
     maybe silver, probably NOT raw" — ``write_lake_dataset`` puts a cleaned /
     imputed / synthetic artifact to the configured silver/gold prefix with tier
     + lineage metadata (and **rejects bronze/raw** writes).

S3-backed (real AWS in tests) + optional Glue Catalog lookup. Reuses
``cloud/staging.parse_s3_uri`` (no duplication of the S3 URI parser).
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from strands import tool

# Reuse the existing S3 URI parser from the cloud staging path (no duplication).
from ...cloud.staging import parse_s3_uri

# The canonical lake-bucket resolver (env AMLAS_LAKE_BUCKET → SSM /amlas/lake-bucket
# → default) — the SINGLE seam for the physical bucket (config/externalized §6).
from ...config import externalized
from ...config.aws_region import resolve_aws_region

# SEC-01 / CFG-01: one region source of truth.
from ...config.runtime import DEFAULT_REGION as DEFAULT_REGION
from ...errors import tool_error_boundary

__all__ = [
    "DEFAULT_LOGICAL_LAKE_ALIASES",
    "TIERS",
    "WRITABLE_TIERS",
    "list_lake_catalog",
    "parse_lake_uri",
    "partition_dataset",
    "read_lake_slice",
    "recommend_partition_strategy",
    "resolve_lake_bucket",
    "write_lake_dataset",
]

# The medallion tiers; bronze/raw is READ-only provenance — agents never write it.
TIERS: tuple[str, ...] = ("bronze", "silver", "gold")
WRITABLE_TIERS: tuple[str, ...] = ("silver", "gold")

# ─── Logical-bucket alias (data-staging §C — keep the 100 prompts verbatim) ───
# The 100 use-case prompts hardcode a bare LOGICAL bucket ``amlas`` (e.g.
# ``s3://amlas/silver/bogie``). That name is NOT the physical lake bucket — the
# real SEC-01 medallion bucket is resolved at runtime (``amlas-lake-<acct>-<region>``).
# :func:`resolve_lake_bucket` rewrites this small, config-driven set of logical
# names to the resolved physical bucket, so every URI-parsing lake tool works
# against the real bucket WITHOUT editing a single prompt. A URI that already
# names a physical bucket (anything outside the alias set) passes through
# UNCHANGED — the real ``amlas-lake-…`` bucket and any test bucket are never
# rewritten.
DEFAULT_LOGICAL_LAKE_ALIASES: frozenset[str] = frozenset({"amlas"})

# Optional operator knob: a comma-separated override of the logical alias set.
_LOGICAL_ALIAS_ENV = "AMLAS_LAKE_LOGICAL_ALIASES"


def _logical_lake_aliases() -> frozenset[str]:
    """The active logical-bucket alias set (env override → default ``{"amlas"}``)."""
    raw = os.environ.get(_LOGICAL_ALIAS_ENV, "").strip()
    if raw:
        parsed = frozenset(a.strip() for a in raw.split(",") if a.strip())
        if parsed:
            return parsed
    return DEFAULT_LOGICAL_LAKE_ALIASES


def resolve_lake_bucket(bucket: str, *, ssm_getter: Any = None) -> str:
    """Rewrite a LOGICAL lake bucket alias to the resolved physical lake bucket.

    A bucket in the configurable alias set (default ``{"amlas"}``; overridable via
    ``AMLAS_LAKE_LOGICAL_ALIASES``) is rewritten to the physical bucket resolved by
    the canonical externalized resolver (env ``AMLAS_LAKE_BUCKET`` → SSM
    ``/amlas/lake-bucket`` → default). Any bucket OUTSIDE the alias set — including
    the real ``amlas-lake-<acct>-<region>`` and any test bucket — passes through
    UNCHANGED (regression-critical).

    Offline-safe: if the physical bucket cannot be resolved (no env/SSM), the
    logical name is returned unchanged (a deterministic default identical to the
    prior verbatim behavior) — it NEVER raises and NEVER hard-fails at import.

    Args:
        bucket: the bucket segment parsed from an ``s3://`` URI.
        ssm_getter: injectable ``(key, region, client=None) -> str`` for hermetic
            tests; defaults to the externalized best-effort SSM read ('' offline).

    Returns:
        The physical bucket when ``bucket`` is a logical alias and resolution
        succeeds; otherwise ``bucket`` unchanged.
    """
    b = str(bucket).strip()
    if b not in _logical_lake_aliases():
        return b  # already a physical bucket → never rewrite
    try:
        physical = externalized.resolve("lake_bucket", ssm_getter=ssm_getter).strip()
    except Exception:
        physical = ""  # never hard-fail: fall back to the logical name below
    return physical or b


def _guard_unresolved_alias(bucket: str) -> None:
    """Fail LOUD before any live S3 call against an UNRESOLVED logical alias.

    Bucket-squatting guard (Security review finding): if the logical alias
    (default ``amlas``) could not be rewritten to the adopter's physical lake
    bucket (no ``AMLAS_LAKE_BUCKET`` env / SSM ``/amlas/lake-bucket``), a live
    call would target the literal generic bucket name — which an external actor
    could pre-register. Offline parsing/validation still passes the alias
    through unchanged; only a REAL AWS call is blocked here.
    """
    if str(bucket).strip() in _logical_lake_aliases():
        raise ValueError(
            f"lake bucket not configured: logical alias {bucket!r} did not resolve "
            "to a physical bucket. Set AMLAS_LAKE_BUCKET or SSM /amlas/lake-bucket "
            "before running live lake operations (prevents bucket-squatting)."
        )


_MISSING_BOTO3 = {
    "success": False,
    "error": "lake dependencies are unavailable; rerun scripts/setup.sh",
}


def parse_lake_uri(uri: str) -> dict[str, Any]:
    """Parse a medallion lake URI into a structured descriptor.

    Supports two shapes:
      * ``silver://db.table`` / ``gold://db.table`` — a Glue-catalog slice on a
        named tier (``{"kind":"catalog","tier","database","table"}``).
      * ``s3://bucket/prefix`` — a direct S3 location, with the tier inferred
        from a ``/bronze/`` | ``/silver/`` | ``/gold/`` path segment if present
        (``{"kind":"s3","tier","bucket","key"}``).

    Raises ``ValueError`` on an unrecognized scheme.
    """
    u = str(uri).strip()
    for tier in TIERS:
        prefix = f"{tier}://"
        if u.startswith(prefix):
            rest = u[len(prefix):]
            database, _, table = rest.partition(".")
            return {
                "kind": "catalog",
                "tier": tier,
                "database": database,
                "table": table or database,
            }
    if u.startswith("s3://"):
        bucket, key = parse_s3_uri(u)
        # Logical-bucket alias: a bare LOGICAL bucket (default ``amlas``) is
        # rewritten to the resolved physical lake bucket here — the SINGLE
        # chokepoint every URI-parsing lake tool (read/write/register) shares —
        # so the 100 use-case prompts (``s3://amlas/silver/bogie`` …) hit the real
        # ``amlas-lake-<acct>-<region>`` bucket verbatim. A physical bucket passes
        # through unchanged (see :func:`resolve_lake_bucket`). Key/prefix preserved.
        physical_bucket = resolve_lake_bucket(bucket)
        return {
            "kind": "s3",
            "tier": _tier_from_key(key),
            "bucket": physical_bucket,
            "key": key,
        }
    raise ValueError(
        f"unrecognized lake uri {uri!r}; expected silver://db.table, gold://db.table, or s3://..."
    )


def _tier_from_key(key: str) -> str:
    """Infer the medallion tier from an S3 key's path segments (default silver)."""
    return _tier_from_key_opt(key) or "silver"


def _tier_from_key_opt(key: str) -> str | None:
    """Return the tier ONLY if a ``/bronze/`` | ``/silver/`` | ``/gold/`` segment is
    explicitly present in the key, else ``None``.

    Unlike :func:`_tier_from_key` (which defaults to ``"silver"``), this lets a
    caller distinguish "the URI explicitly names a tier" from "no tier segment, so
    fall back to the requested ``tier`` argument" — the basis of the write-back
    governance fix (06-tui-e2e-hardening bug #4/#5).
    """
    parts = {p.lower() for p in key.split("/")}
    for tier in TIERS:
        if tier in parts:
            return tier
    return None


def _read_frame(path: Any) -> Any:
    """Read a local data file into a DataFrame, dispatching on extension (PERF-10).

    ``.parquet`` → ``pd.read_parquet`` (columnar, Snappy — far fewer bytes scanned in the
    lakehouse path); everything else → ``pd.read_csv`` (the human-readable default for tiny
    fixtures). This is the read side of the CSV→Parquet capability: readers transparently
    accept Parquet so the lake path CAN use it without changing the caller. The committed
    ``data/ai4i/processed/*.csv`` artifacts are NOT migrated here (deferred to the OSS phase).
    """
    import pandas as pd

    if str(path).lower().endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _write_frame(df: Any, path: Any) -> None:
    """Write a DataFrame to a local file, dispatching on extension (PERF-10).

    ``.parquet`` → ``to_parquet`` (Snappy default); else ``to_csv(index=False)``. The write
    side of the Parquet capability — round-trips byte-equal frame content with ``_read_frame``.
    """
    if str(path).lower().endswith(".parquet"):
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def _make_s3_client(region: str, client: Any | None) -> Any:
    if client is not None:
        return client
    import boto3

    return boto3.client("s3", region_name=resolve_aws_region(region))


def _make_glue_client(region: str, client: Any | None) -> Any:
    if client is not None:
        return client
    import boto3

    return boto3.client("glue", region_name=resolve_aws_region(region))


def _resolve_catalog_s3(
    descriptor: dict[str, Any], *, region: str, glue_client: Any | None
) -> tuple[str, str]:
    """Resolve a Glue-catalog descriptor to a concrete ``(bucket, key)`` location."""
    glue = _make_glue_client(region, glue_client)
    resp = glue.get_table(DatabaseName=descriptor["database"], Name=descriptor["table"])
    location = resp["Table"]["StorageDescriptor"]["Location"]
    return parse_s3_uri(location if location.endswith((".csv", ".parquet")) else location.rstrip("/") + "/data.csv") if False else parse_s3_uri(_normalize_location(location))


def _normalize_location(location: str) -> str:
    """Ensure a Glue table Location resolves to a concrete CSV object key."""
    loc = location.rstrip("/")
    if loc.endswith((".csv", ".parquet")):
        return loc
    return f"{loc}/data.csv"


@tool
@tool_error_boundary(source="tools.lake")
def read_lake_slice(
    uri: str,
    columns: list[str] | None = None,
    filters: dict[str, Any] | None = None,
    limit: int | None = None,
    *,
    dest_dir: str | None = None,
    region: str | None = None,
    s3_client: Any = None,
    glue_client: Any = None,
) -> dict[str, Any]:
    """Read a tier-scoped SUBSET of the lakehouse to a local DataFrame artifact.

    Resolves ``silver://db.table`` / ``gold://db.table`` (via Glue) or ``s3://...``,
    downloads the object, applies ``columns`` projection / ``filters`` (equality)
    / ``limit`` row cap, writes the slice locally, and returns it with provenance.
    This is the "vertical slice, not petabytes" read (transcript 18:27).

    Args:
        uri: ``silver://db.table`` | ``gold://db.table`` | ``s3://bucket/key.csv``.
        columns: Optional subset of columns to keep.
        filters: Optional ``{column: value}`` equality filters.
        limit: Optional max rows to return.
        dest_dir: Local dir for the materialized slice.
        region: AWS region.
        s3_client: Injected S3 client (real-AWS-backed in tests).
        glue_client: Injected Glue client (real-AWS-backed in tests).

    Returns:
        ``{"local_path","schema","rows","tier","provenance":{...}}`` or
        ``{"success": False, "error": ...}``.
    """
    try:
        descriptor = parse_lake_uri(uri)
        tier = descriptor["tier"]
        if descriptor["kind"] == "catalog":
            bucket, key = _resolve_catalog_s3(descriptor, region=region, glue_client=glue_client)
        else:
            bucket, key = descriptor["bucket"], descriptor["key"]

        # R-B-003 / AMLAS-SEC-B-003 (CWE-377 / bandit B108): default the materialized-slice dir
        # to a per-run unique 0700 ``tempfile.mkdtemp()`` instead of the predictable shared
        # ``/tmp/amlas_lake`` (symlink/clobber-race prone on multi-user hosts). A caller-supplied
        # ``dest_dir`` still wins; only the default is now non-predictable + private.
        resolved_dest = dest_dir or tempfile.mkdtemp(prefix="amlas_lake_")
        dest = Path(resolved_dest)
        dest.mkdir(parents=True, exist_ok=True)
        local = dest / Path(key).name
        _guard_unresolved_alias(bucket)  # squatting guard: never hit a literal logical alias live
        s3 = _make_s3_client(region, s3_client)
        s3.download_file(bucket, str(key), str(local))

        # PERF-10: dispatch on the source extension — a ``.parquet`` object reads via the
        # columnar (Snappy) path, a CSV via read_csv. The materialized slice preserves the
        # source format so a Parquet lake slice stays Parquet (bytes-scanned win downstream).
        df = _read_frame(local)
        if filters:
            for col, val in filters.items():
                if col in df.columns:
                    df = df[df[col] == val]
        if columns:
            keep = [c for c in columns if c in df.columns]
            df = df[keep] if keep else df
        if limit is not None:
            df = df.head(int(limit))

        src_suffix = ".parquet" if str(key).lower().endswith(".parquet") else ".csv"
        slice_path = dest / f"slice_{Path(key).stem}{src_suffix}"
        _write_frame(df, slice_path)

        return {
            "local_path": str(slice_path),
            "schema": {c: str(df[c].dtype) for c in df.columns},
            "rows": len(df),
            "tier": tier,
            "provenance": {
                "uri": uri,
                "source_bucket": bucket,
                "source_key": key,
                "columns_requested": columns or "all",
                "filters": filters or {},
                "limit": limit,
            },
        }
    except ValueError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        return {"success": False, "error": f"read_lake_slice failed: {e}"}


@tool
@tool_error_boundary(source="tools.lake")
def write_lake_dataset(
    local_path: str,
    dest_uri: str,
    tier: str = "silver",
    lineage: dict[str, Any] | None = None,
    profile: dict[str, Any] | None = None,
    *,
    region: str | None = None,
    s3_client: Any = None,
) -> dict[str, Any]:
    """Write a cleaned/imputed/synthetic artifact BACK to a silver/gold tier.

    Enforces the medallion write rule: writes go to **silver or gold only** —
    **bronze/raw is rejected** ("consume from silver, put back to gold/silver,
    not raw", transcript 20:14). Tags the object with tier + a lineage manifest
    (a sidecar ``.lineage.json``) so the write is auditable.

    **Steering (the dial, master plan §3.1) — HARD, enforced here in the tool:**
    when the MLOps Lead forwards a ``profile`` (a ``SteeringProfile.as_profile_dict()``),
    a ``pin_methods["write_tier"]`` value **overrides** the requested ``tier`` and a
    ``forbid_methods`` entry naming a tier **rejects** that write — the same
    deterministic enforcement ``apply_imputation_plan`` gives the Data-Eng Lead, so
    the agent can never violate a domain operator's write-back policy.

    Args:
        local_path: Local CSV artifact to upload.
        dest_uri: Destination ``s3://bucket/prefix`` or ``silver://``/``gold://``.
        tier: ``silver`` or ``gold`` (``bronze``/``raw`` raise an error).
        lineage: Optional lineage dict (source slice, operations applied, etc.).
        profile: Optional SteeringProfile dict — ``pin_methods["write_tier"]``
            overrides the tier; ``forbid_methods`` listing a tier rejects it.
        region: AWS region.
        s3_client: Injected S3 client (real-AWS-backed in tests).

    Returns:
        ``{"written_uri","tier","lineage_id","lineage_uri","rows","steering_violations"}``
        or ``{"success": False, "error": ...}``.
    """
    try:
        steering_violations: list[dict[str, Any]] = []

        # GOVERNANCE FIX (06-tui-e2e-hardening, bugs #4+#5): the destination URI is
        # AUTHORITATIVE for the tier when it encodes one — a ``/bronze/``|``/silver/``
        # |``/gold/`` path segment, or a ``silver://``/``gold://`` catalog scheme.
        # The explicit ``tier`` arg is only a FALLBACK for a tier-less
        # ``s3://bucket/key`` dest. Previously the tool tagged the write with the
        # ``tier`` PARAM and NEVER reconciled it with ``dest_uri``; callers that omit
        # ``tier`` (the TUI ``/lake write`` path passes only local_path/dest_uri/
        # profile) therefore (a) DOWNGRADED a ``/gold/`` write to the silver default
        # and (b) BYPASSED the medallion bronze-rejection for a ``/bronze/`` dest — a
        # real data-governance hole surfaced by the real-S3 e2e turn (the same call
        # returned ``tier=silver`` for BOTH a gold and a bronze destination). Deriving
        # the tier from the dest URI closes both: ``/gold/`` → gold, ``/bronze/`` →
        # rejected. Existing callers that pass a matching ``tier=`` are unaffected
        # (the URI tier == the param), and a tier-less dest still honors ``tier``.
        descriptor = parse_lake_uri(dest_uri)
        if descriptor["kind"] == "catalog":
            uri_tier: str | None = descriptor["tier"]  # the scheme IS the tier
        else:
            uri_tier = _tier_from_key_opt(descriptor["key"])  # explicit-only (or None)
        t = (uri_tier or str(tier)).strip().lower()

        # ── HARD steering (deterministic, the dial reaches the tool) ──────────
        prof = dict(profile or {})
        pinned = (prof.get("pin_methods") or {}).get("write_tier")
        if pinned and str(pinned).strip().lower() != t:
            steering_violations.append(
                {"enforced": True, "reason": f"pin write_tier={pinned} overrode requested tier={t}"}
            )
            t = str(pinned).strip().lower()
        forbidden = {str(m).strip().lower() for m in (prof.get("forbid_methods") or [])}
        if t in forbidden:
            return {
                "success": False,
                "error": f"tier {t!r} is forbidden by steering (forbid_methods)",
                "steering_violations": [
                    {"reason": f"tier {t!r} forbidden by steering"}
                ],
            }

        if t in ("bronze", "raw"):
            return {
                "success": False,
                "error": f"refusing to write tier {t!r}: bronze/raw is read-only "
                "(medallion write-back targets silver or gold)",
            }
        if t not in WRITABLE_TIERS:
            return {
                "success": False,
                "error": f"invalid tier {t!r}; writable tiers are {list(WRITABLE_TIERS)}",
            }

        p = Path(local_path)
        if not p.exists():
            return {"success": False, "error": f"File not found: {local_path}"}

        if descriptor["kind"] == "catalog":
            # catalog dest → a conventional s3 prefix under the (URI-derived) tier
            bucket = descriptor["database"]
            key = f"{t}/{descriptor['table']}/{p.name}"
        else:
            bucket, base_key = descriptor["bucket"], descriptor["key"]
            key = base_key if base_key.endswith(".csv") else f"{base_key.rstrip('/')}/{p.name}"

        # PERF-10: read via the format-dispatching loader so a Parquet artifact round-trips
        # (the upload itself is byte-for-byte the same object — we only read to count rows).
        df = _read_frame(p)
        _guard_unresolved_alias(bucket)  # squatting guard: never hit a literal logical alias live
        s3 = _make_s3_client(region, s3_client)
        s3.upload_file(str(p), bucket, key)

        lineage_id = uuid.uuid4().hex[:12]
        lineage_doc = {
            "lineage_id": lineage_id,
            "tier": t,
            "written_uri": f"s3://{bucket}/{key}",
            "rows": len(df),
            "written_at": int(time.time()),
            **(lineage or {}),
        }
        lineage_key = f"{key}.lineage.json"
        s3.put_object(
            Bucket=bucket,
            Key=lineage_key,
            Body=json.dumps(lineage_doc).encode("utf-8"),
        )

        return {
            "written_uri": f"s3://{bucket}/{key}",
            "tier": t,
            "lineage_id": lineage_id,
            "lineage_uri": f"s3://{bucket}/{lineage_key}",
            "rows": len(df),
            "steering_violations": steering_violations,
        }
    except ValueError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        return {"success": False, "error": f"write_lake_dataset failed: {e}"}


@tool
@tool_error_boundary(source="tools.lake")
def list_lake_catalog(
    database: str | None = None,
    *,
    region: str | None = None,
    glue_client: Any = None,
) -> dict[str, Any]:
    """List datasets in the Glue Data Catalog (the "data catalog" of transcript 18:38).

    Args:
        database: Glue database to list; lists all databases' tables if None.
        region: AWS region.
        glue_client: Injected Glue client (real-AWS-backed in tests).

    Returns:
        ``{"datasets":[{"name","database","tier","location","columns"}],"count"}``
        or ``{"success": False, "error": ...}``.
    """
    try:
        glue = _make_glue_client(region, glue_client)
        databases: list[str]
        if database:
            databases = [database]
        else:
            resp = glue.get_databases()
            databases = [d["Name"] for d in resp.get("DatabaseList", [])]

        datasets: list[dict[str, Any]] = []
        for db in databases:
            tresp = glue.get_tables(DatabaseName=db)
            for tbl in tresp.get("TableList", []):
                sd = tbl.get("StorageDescriptor", {}) or {}
                location = sd.get("Location", "")
                datasets.append(
                    {
                        "name": tbl.get("Name", ""),
                        "database": db,
                        "tier": _tier_from_key(location),
                        "location": location,
                        "columns": [c.get("Name", "") for c in sd.get("Columns", [])],
                    }
                )
        return {"datasets": datasets, "count": len(datasets)}
    except Exception as e:
        return {"success": False, "error": f"list_lake_catalog failed: {e}"}


# ─── Hive partitioning helpers (master-plan gap T-2, Phase 7) ─────────────────
# Imported at module END so ``partition.py`` can reuse ``_read_frame``/``_write_frame``
# defined above (avoids a circular import at package init).
from .partition import partition_dataset, recommend_partition_strategy  # noqa: E402
