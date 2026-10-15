"""FeatureStoreBackend adapters (doc 05 §4.2 G2 / §6; roadmap M0 + M3 PIT depth).

* ``SageMakerFeatureStoreBackend`` — the **default**; wraps EXISTING
  ``tools/mlops/feature_store.py`` (``feature_store_ingest`` / ``feature_store_get``).
  Gap-closure **G2** implements the managed PIT path (offline-store + Athena `as_of` join)
  as a thin orchestration: resolve the offline Glue table via ``describe_feature_group``,
  build the no-leakage PIT SQL (:func:`build_pit_query`), run it on Athena, and return the
  SAME ``DatasetRef`` contract the gold-table adapter returns (interchangeable behind the
  port). The Athena/S3 clients are injectable so offline client substitutes drive it offline.
* ``GoldTableBackend`` — the **local** adapter; a medallion-gold feature store backed by a
  local JSON gold table. M3 implements **point-in-time (PIT) correctness** in-process: the
  ``pit_dataset(entity_ids, as_of)`` join returns, per entity, the latest feature record
  whose ``event_time <= as_of`` — **never a future record** (the project PIT-no-leakage
  rule; T-FS-03). This is the offline-safe PIT path the registry resolves in offline mode.

PIT leakage rule (CORE IP preserved, RECON / doc 05 §6): a training dataset assembled at
``as_of`` must contain ONLY feature values that were known at or before ``as_of``. The
gold-table join enforces this with a strict ``event_time <= as_of`` filter + latest-wins
per entity, so no future feature value can leak into a training row.

Selector (``FEATURE_STORE_BACKEND``): ``sagemaker`` (default managed),
``goldtable`` (local). Offline auto-resolves to ``goldtable`` (doc 05 §4.3).
"""

from __future__ import annotations

import json
import re
import tempfile
import time
from pathlib import Path
from typing import Any

from ...config.aws_region import resolve_aws_region
from ..ports import FeatureStoreBackend
from ..registry import register_adapter
from ..types import DatasetRef
from ._util import raw_tool

# The PIT no-leakage rule string — IDENTICAL across the gold-table (in-process) and the
# managed (Athena) adapters so a ``DatasetRef`` is interchangeable behind the port (T-FS-03).
_PIT_RULE = "event_time <= as_of (latest-wins, no future leakage)"
_SQL_IDENTIFIER = r"[A-Za-z_][A-Za-z0-9_]*"
_UNQUOTED_TABLE_RE = re.compile(rf"^{_SQL_IDENTIFIER}(?:\.{_SQL_IDENTIFIER})?$")
_QUOTED_TABLE_RE = re.compile(rf'^"{_SQL_IDENTIFIER}"(?:\."{_SQL_IDENTIFIER}")?$')
_COLUMN_RE = re.compile(rf"^{_SQL_IDENTIFIER}$")


def _validate_sql_identifier(value: str, *, kind: str, table: bool = False) -> str:
    """Return a catalog identifier only when it matches the closed SQL grammar.

    Athena identifiers cannot be represented by value parameters, so catalog-derived
    table and column names are validated before interpolation. Tables may be bare,
    schema-qualified, or equivalently double-quoted; columns must be a single bare
    identifier because :func:`build_pit_query` applies the SQL quoting itself.
    """
    candidate = str(value)
    pattern_ok = bool(
        (_UNQUOTED_TABLE_RE.fullmatch(candidate) or _QUOTED_TABLE_RE.fullmatch(candidate))
        if table
        else _COLUMN_RE.fullmatch(candidate)
    )
    if not pattern_ok:
        raise ValueError(f"unsafe {kind} identifier: {candidate!r}")
    return candidate


def build_count_query(table: str) -> str:
    """Build a count query after validating the catalog-derived table identifier."""
    safe_table = _validate_sql_identifier(table, kind="table", table=True)
    return f"SELECT COUNT(*) AS n FROM {safe_table}"  # nosec B608


def _sql_literal(value: Any) -> str:
    """Quote a value as a SQL string literal with single-quote escaping (anti-injection).

    Athena/Presto (and sqlite, used by the offline test) escape an embedded single quote by
    doubling it. Entity ids + the ``as_of`` cutoff flow through here so a crafted id can never
    break the predicate or inject SQL — the trust boundary for the PIT join.
    """
    return "'" + str(value).replace("'", "''") + "'"


def build_pit_query(
    *,
    table: str,
    record_id_col: str,
    event_time_col: str,
    entity_ids: list[str],
    as_of: str,
) -> str:
    """Build the point-in-time (PIT) ``as_of`` join SQL for the offline feature store (G2).

    The query encodes the SAME no-leakage contract the gold-table adapter enforces in-process:

      * **no future leakage** — a ``WHERE "event_time" <= '<as_of>'`` cutoff so NO feature row
        whose event-time is after the label timestamp can enter the assembled training set;
      * **latest-wins** — a ``ROW_NUMBER() OVER (PARTITION BY <entity> ORDER BY <event_time>
        DESC)`` window keeping only ``row_num = 1`` (the newest record known at ``as_of``);
      * **entity scoping** — ``<entity> IN (...)`` restricts to the requested entities.

    Returns a single SQL string (Athena/Presto-compatible; also valid SQLite for the offline
    end-to-end no-leakage proof). Pure — no AWS, no I/O — so the predicate is directly verifiable.
    """
    safe_table = _validate_sql_identifier(table, kind="table", table=True)
    rid = f'"{_validate_sql_identifier(record_id_col, kind="record-id column")}"'
    et = f'"{_validate_sql_identifier(event_time_col, kind="event-time column")}"'
    cutoff = _sql_literal(as_of)
    entity_filter = ""
    if entity_ids:
        id_list = ", ".join(_sql_literal(e) for e in entity_ids)
        entity_filter = f" AND {rid} IN ({id_list})"
    # Athena does not support bind parameters for catalog identifiers. The closed
    # identifier grammar and escaped SQL literals above are the reviewed trust boundary.
    return (
        "SELECT * FROM (\n"  # nosec B608
        f"  SELECT *, ROW_NUMBER() OVER (PARTITION BY {rid} ORDER BY {et} DESC) AS row_num\n"
        f"  FROM {safe_table}\n"
        f"  WHERE {et} <= {cutoff}{entity_filter}\n"
        ") ranked\n"
        "WHERE row_num = 1"
    )


def _parse_athena_results(result_set: dict[str, Any]) -> tuple[list[str], list[list[Any]]]:
    """Parse an Athena ``get_query_results`` ``ResultSet`` → (columns, data-rows).

    Athena returns the column header as the FIRST row of ``Rows`` (each cell a
    ``{"VarCharValue": ...}``); the remaining rows are the data. Returns the column names and
    the data rows as plain lists (header excluded).
    """
    rows = result_set.get("Rows", []) or []
    if not rows:
        return [], []
    header = [c.get("VarCharValue", "") for c in rows[0].get("Data", [])]
    data: list[list[Any]] = []
    for r in rows[1:]:
        data.append([c.get("VarCharValue") for c in r.get("Data", [])])
    return header, data


def _pit_rows_and_schema(athena: Any, exec_id: str) -> tuple[int, list[str]]:
    """Count PIT result rows + derive the schema WITHOUT materializing the full result (PERF-12).

    The prior path called ``_parse_athena_results`` which built a ``columns + data``
    list-of-lists (EVERY cell of EVERY row) just to compute ``rows = len(data)`` — O(rows×cols)
    client-side materialization for a result we only need a count + column names from.

    This instead:
      * derives the **schema** from the result's ``ResultSetMetadata.ColumnInfo`` (column NAMES
        only — falls back to the header row's ``VarCharValue`` if metadata is absent);
      * **counts** data rows by streaming ``get_query_results`` pages (a paginator when the
        client exposes one) and summing the per-page row count, NEVER building the cell list.

    The first page carries the header row (Athena's contract), so it is excluded from the count
    exactly as ``_parse_athena_results`` excluded ``rows[0]``. Returns ``(row_count, columns)`` —
    byte-identical to ``(len(data), header)`` from the old path, so the PIT ``DatasetRef``
    contract (rows / schema) is preserved (the no-leakage golden stays green).
    """
    columns: list[str] = []
    count = 0
    first_page = True

    def _columns_from(result_set: dict[str, Any]) -> list[str]:
        meta = (result_set.get("ResultSetMetadata", {}) or {}).get("ColumnInfo", []) or []
        names = [c.get("Name", "") for c in meta if c.get("Name")]
        if names:
            return names
        # Fallback: the header row's VarCharValues (matches _parse_athena_results' header).
        rows = result_set.get("Rows", []) or []
        if rows:
            return [c.get("VarCharValue", "") for c in rows[0].get("Data", [])]
        return []

    # Prefer a paginator (streams pages without one giant in-memory result) when available.
    pages = None
    get_paginator = getattr(athena, "get_paginator", None)
    if callable(get_paginator):
        try:
            paginator = get_paginator("get_query_results")
            pages = paginator.paginate(QueryExecutionId=exec_id)
        except Exception:
            pages = None

    if pages is not None:
        for page in pages:
            rs = page.get("ResultSet", {}) or {}
            if first_page:
                columns = _columns_from(rs)
            page_rows = rs.get("Rows", []) or []
            # The header row only appears on the FIRST page.
            count += max(0, len(page_rows) - (1 if first_page else 0))
            first_page = False
        return count, columns

    # No paginator → a single get_query_results call (the fake-client / small-result path).
    rs = (athena.get_query_results(QueryExecutionId=exec_id) or {}).get("ResultSet", {}) or {}
    columns = _columns_from(rs)
    page_rows = rs.get("Rows", []) or []
    count = max(0, len(page_rows) - 1)  # exclude the header row
    return count, columns


def _coerce_records(records: Any) -> list[dict[str, Any]]:
    """Normalize ``records`` (a CSV path, a list-of-dicts, or a DataFrame) → list-of-dicts."""
    if records is None:
        return []
    if isinstance(records, list):
        return [dict(r) for r in records]
    if isinstance(records, dict):
        return [dict(records)]
    # pandas DataFrame
    if hasattr(records, "to_dict"):
        try:
            return [dict(r) for r in records.to_dict(orient="records")]
        except Exception:  # nosec B110
            pass
    # a CSV path
    if isinstance(records, str) and records.endswith(".csv"):
        import pandas as pd

        return [{str(k): v for k, v in r.items()} for r in pd.read_csv(records).to_dict(orient="records")]
    return []


class SageMakerFeatureStoreBackend:
    """Default FeatureStoreBackend — wraps EXISTING ``tools/mlops/feature_store.py``."""

    def __init__(
        self, *, region: str | None = None,
        sm_client: Any = None, fs_runtime_client: Any = None,
        athena_client: Any = None, s3_client: Any = None,
        athena_output: str = "", athena_workgroup: str = "",
    ) -> None:
        self._region = resolve_aws_region(region, required=False)
        self._sm = sm_client
        self._fsr = fs_runtime_client
        # G2 managed PIT seam — injectable Athena/S3 clients (like the deployment/DQ adapters)
        # so offline client substitutes drive the offline-store ``as_of`` join offline. Defaults keep
        # the registry factory (``lambda: SageMakerFeatureStoreBackend()``) zero-arg.
        self._athena = athena_client
        self._s3 = s3_client
        self._athena_output = athena_output
        self._athena_workgroup = athena_workgroup

    def upsert(self, *, group: str, records: Any) -> int:
        """Ingest feature rows via EXISTING ``feature_store_ingest``.

        Accepts the SAME ``records`` shapes the gold-table adapter does — a CSV path (the
        existing tool's native input, fast-path) OR a list-of-dicts / DataFrame (what the
        ``feature_engineering_flow`` passes). Non-path records are materialized to a temp CSV
        first so the managed path runs the same flow contract as the local one (G2 parity).
        """
        from ...tools.mlops.feature_store import feature_store_ingest

        data_path = records
        if not (isinstance(records, str) and records.endswith(".csv")):
            rows = _coerce_records(records)
            if not rows:
                return 0
            import pandas as pd

            fd, tmp = tempfile.mkstemp(suffix=".csv", prefix="amlas_fs_upsert_")
            import os

            os.close(fd)
            pd.DataFrame(rows).to_csv(tmp, index=False)
            data_path = tmp

        out = raw_tool(feature_store_ingest)(
            group, str(data_path), region=self._region,
            sm_client=self._sm, fs_runtime_client=self._fsr,
        )
        if out.get("success") is False:
            raise RuntimeError(out.get("error", "feature_store_ingest failed"))
        return int(out.get("ingested", 0))

    def get(self, *, group: str, record_id: str, online: bool = True) -> dict[str, Any]:
        """Online PIT-correct retrieval via EXISTING ``feature_store_get`` (unchanged)."""
        from ...tools.mlops.feature_store import feature_store_get

        out = raw_tool(feature_store_get)(group, [record_id], region=self._region, fs_runtime_client=self._fsr)
        if out.get("success") is False:
            raise RuntimeError(out.get("error", "feature_store_get failed"))
        records = out.get("records", [])
        return dict(records[0]) if records else {}

    # ── managed PIT helpers (G2) ────────────────────────────────────────────────
    def _athena_client(self) -> Any:
        """Return the injected Athena client, else build a default boto3 one (lazy)."""
        if self._athena is not None:
            return self._athena
        import boto3

        return boto3.client("athena", region_name=resolve_aws_region(self._region))

    def _sm_client(self) -> Any:
        """Return the injected SageMaker client, else build a default boto3 one (lazy)."""
        if self._sm is not None:
            return self._sm
        import boto3

        return boto3.client("sagemaker", region_name=resolve_aws_region(self._region))

    def _resolve_offline_table(self, sm: Any, group: str) -> tuple[str, str, str, str, str]:
        """Resolve the offline-store Glue table for ``group`` via ``describe_feature_group``.

        Returns ``(qualified_table, record_id_col, event_time_col, database, s3_uri)`` where
        ``qualified_table`` is the ``"db"."table"`` reference the Athena PIT join selects from.
        The names come from the feature group created with an ``OfflineStoreConfig`` /
        ``DataCatalogConfig`` (``feature_store_ingest``). Falls back to a sanitized table name
        under the default Glue database when the catalog config is absent.
        """
        from ...tools.mlops._common import sanitize_name

        desc = sm.describe_feature_group(FeatureGroupName=sanitize_name(group))
        rid = desc.get("RecordIdentifierFeatureName") or "entity_id"
        etime = desc.get("EventTimeFeatureName") or "event_time"
        offline = desc.get("OfflineStoreConfig", {}) or {}
        catalog = offline.get("DataCatalogConfig", {}) or {}
        s3_uri = (offline.get("S3StorageConfig", {}) or {}).get("S3Uri", "")
        database = catalog.get("Database") or "sagemaker_featurestore"
        table = catalog.get("TableName") or sanitize_name(group).replace("-", "_")
        qualified = f'"{database}"."{table}"'
        return qualified, str(rid), str(etime), str(database), str(s3_uri)

    def _run_athena(self, athena: Any, sql: str, *, database: str,
                    output_location: str = "") -> tuple[str, str]:
        """Start the Athena query, poll to a terminal state, return (exec_id, output_location).

        Fails fast (``RuntimeError``) on a FAILED/CANCELLED query so a broken PIT join never
        silently returns an empty dataset (R-06 offline-store-lag aware: the caller stages rows
        before assembling). Bounded poll loop (no infinite wait).
        """
        start_kwargs: dict[str, Any] = {
            "QueryString": sql,
            "QueryExecutionContext": {"Database": database},
        }
        result_conf: dict[str, Any] = {}
        if output_location:
            result_conf["OutputLocation"] = output_location
        start_kwargs["ResultConfiguration"] = result_conf
        if self._athena_workgroup:
            start_kwargs["WorkGroup"] = self._athena_workgroup

        started = athena.start_query_execution(**start_kwargs)
        exec_id = started.get("QueryExecutionId", "")

        resolved_location = ""
        deadline = time.time() + 300  # bounded poll (5 min)
        while time.time() <= deadline:
            ex = athena.get_query_execution(QueryExecutionId=exec_id)
            qe = ex.get("QueryExecution", {})
            state = (qe.get("Status", {}) or {}).get("State", "")
            resolved_location = (qe.get("ResultConfiguration", {}) or {}).get("OutputLocation", "")
            if state == "SUCCEEDED":
                break
            if state in ("FAILED", "CANCELLED"):
                reason = (qe.get("Status", {}) or {}).get("StateChangeReason", state)
                raise RuntimeError(f"Athena PIT query {exec_id} {state}: {reason}")
            # Deadline-bounded poll (loop exits at its deadline) — P3 audit H-06.
            time.sleep(2)  # nosemgrep: arbitrary-sleep
        return exec_id, (resolved_location or output_location)

    def pit_dataset(self, *, group: str, entity_ids: list[str], as_of: str) -> DatasetRef:
        """Managed PIT dataset assembly — offline-store + Athena ``as_of`` join (G2; T-FS-03).

        Orchestrates the SageMaker Feature Store **offline** store: resolve the offline Glue
        table (``describe_feature_group`` → ``OfflineStoreConfig``/``DataCatalogConfig``), build
        the no-leakage PIT SQL (per entity, the latest record with ``event_time <= as_of`` — a
        ``ROW_NUMBER`` latest-wins window over the strict ``<= as_of`` cutoff; see
        :func:`build_pit_query`), run it on Athena, and return a :class:`DatasetRef` pointing at
        the result S3 location. The returned ref carries the SAME contract the gold-table adapter
        returns (``tier="gold"``, ``lineage["pit"]=True`` + the no-leakage rule), so the managed
        and offline PIT paths are interchangeable behind :class:`FeatureStoreBackend`.
        """
        sm = self._sm_client()
        athena = self._athena_client()
        qualified, rid_col, et_col, database, s3_uri = self._resolve_offline_table(sm, group)

        sql = build_pit_query(
            table=qualified,
            record_id_col=rid_col,
            event_time_col=et_col,
            entity_ids=entity_ids,
            as_of=str(as_of),
        )
        # Stage the Athena result under the configured output, else default to a results prefix
        # derived from the offline-store bucket (so a PIT join needs no extra wiring offline).
        staged_output = self._athena_output
        if not staged_output and s3_uri:
            staged_output = s3_uri.rstrip("/") + "/athena-pit-results/"
        exec_id, output_location = self._run_athena(
            athena, sql, database=database, output_location=staged_output,
        )

        # PERF-12: count the assembled PIT rows + derive the schema WITHOUT materializing the
        # full result set client-side (was: read every cell just to compute len(data)). The
        # row count + column names (rows / schema) are byte-identical to the old parse path, so
        # the PIT no-leakage DatasetRef contract is preserved.
        row_count, columns = _pit_rows_and_schema(athena, exec_id)
        # Drop the window helper column from the surfaced schema (it is an internal artifact).
        keep = [c for c in columns if c != "row_num"]
        schema = {c: "string" for c in keep}

        return DatasetRef(
            uri=output_location,
            tier="gold",
            rows=row_count,
            schema=schema,
            lineage={
                "feature_group": group,
                "as_of": str(as_of),
                "pit": True,
                "entity_count": len(entity_ids),
                "matched": row_count,
                "rule": _PIT_RULE,
                "engine": "athena",
                "query_execution_id": exec_id,
                "offline_table": qualified,
            },
        )


class GoldTableBackend:
    """Local FeatureStoreBackend — a medallion-gold table with in-process PIT (T-FS-03).

    Backed by a local JSON gold table (``{group: [records]}``), each record a feature row
    carrying an ``event_time`` (ISO-8601) + an entity-id column. ``pit_dataset`` performs
    the PIT join: per entity, the latest record with ``event_time <= as_of`` — strictly no
    future leakage (the core PIT IP). No AWS.
    """

    backend_name = "goldtable"

    def __init__(
        self,
        *,
        table_path: str | Path | None = None,
        entity_key: str = "entity_id",
        event_time_key: str = "event_time",
    ) -> None:
        self._path = Path(table_path) if table_path else Path(tempfile.gettempdir()) / "amlas_gold_features.json"
        self._entity_key = entity_key
        self._event_time_key = event_time_key

    # ── gold-table persistence ────────────────────────────────────────────────
    def _load(self) -> dict[str, list[dict[str, Any]]]:
        if not self._path.exists():
            return {}
        try:
            loaded: dict[str, list[dict[str, Any]]] = json.loads(self._path.read_text())
            return loaded
        except Exception:
            return {}

    def _save(self, table: dict[str, list[dict[str, Any]]]) -> None:
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(json.dumps(table, default=str))
        tmp.replace(self._path)  # atomic write (temp + rename)

    # ── port methods ──────────────────────────────────────────────────────────
    def upsert(self, *, group: str, records: Any) -> int:
        """Append feature ``records`` (list-of-dicts / DataFrame / CSV path) into the group."""
        rows = _coerce_records(records)
        if not rows:
            return 0
        table = self._load()
        table.setdefault(group, []).extend(rows)
        self._save(table)
        return len(rows)

    def get(self, *, group: str, record_id: str, online: bool = True) -> dict[str, Any]:
        """Return the LATEST record for ``record_id`` (online read = newest known value)."""
        rows = [r for r in self._load().get(group, []) if str(r.get(self._entity_key)) == str(record_id)]
        if not rows:
            return {}
        rows.sort(key=lambda r: str(r.get(self._event_time_key, "")))
        return dict(rows[-1])

    def pit_dataset(self, *, group: str, entity_ids: list[str], as_of: str) -> DatasetRef:
        """Assemble a PIT dataset at ``as_of`` (latest record per entity, no future leakage).

        For each entity in ``entity_ids``, select the record with the maximum
        ``event_time`` such that ``event_time <= as_of`` (strict — future records are
        excluded). Writes the assembled rows to a CSV under the gold table's dir and
        returns a :class:`DatasetRef` carrying the row count + PIT lineage.
        """
        import pandas as pd

        table = self._load().get(group, [])
        cutoff = str(as_of)
        picked: list[dict[str, Any]] = []
        for eid in entity_ids:
            candidates = [
                r for r in table
                if str(r.get(self._entity_key)) == str(eid)
                and str(r.get(self._event_time_key, "")) <= cutoff  # PIT: no future leakage
            ]
            if not candidates:
                continue
            candidates.sort(key=lambda r: str(r.get(self._event_time_key, "")))
            picked.append(dict(candidates[-1]))  # latest-known-at-as_of wins

        out_path = self._path.with_name(f"{group}_pit_{cutoff.replace(':', '').replace('-', '')}.csv")
        df = pd.DataFrame(picked)
        df.to_csv(out_path, index=False)
        return DatasetRef(
            uri=str(out_path),
            tier="gold",
            rows=len(picked),
            schema={c: str(df[c].dtype) for c in df.columns} if not df.empty else {},
            lineage={
                "feature_group": group,
                "as_of": cutoff,
                "pit": True,
                "entity_count": len(entity_ids),
                "matched": len(picked),
                "rule": "event_time <= as_of (latest-wins, no future leakage)",
            },
        )


register_adapter(FeatureStoreBackend, "sagemaker", lambda: SageMakerFeatureStoreBackend(), is_default=True)
register_adapter(FeatureStoreBackend, "goldtable", lambda: GoldTableBackend(), is_local=True)
