"""QueryBackend adapters — Athena (managed) + DuckDB/pandas (local) (roadmap M6; doc 05 §4.2 G2).

Completes the M0 ``QueryBackend`` stub: NL→SQL over the lake. Two adapters
(orchestrate-not-reimplement, ADR-004):

* ``DuckDBBackend`` — **local/offline default**. Runs the SQL
  in-process against registered tables (DuckDB if importable, else a pure-Python pandas/
  SQLite-ish fallback over registered dicts) so ``run_sql`` works fully offline with no AWS
  and returns the SAME ``QueryResult`` shape as Athena (parity — doc roadmap M6
  ``test_athena_duckdb_parity``).
* ``AthenaBackend`` — **default (managed)**; thin boto3 ``athena`` ``start_query_execution``
  → poll → ``get_query_results`` orchestration; degrades to the local DuckDB computation
  when no Athena workgroup is reachable (offline-safe; client injectable).

Selector: ``QUERY_BACKEND`` (``duckdb`` offline, ``athena`` managed).
Tool: ``run_sql`` (T-QRY-01) calls this port.
"""

from __future__ import annotations

import re
import time
from typing import Any

from ...config.aws_region import resolve_aws_region
from ..ports import QueryBackend
from ..registry import register_adapter

__all__ = ["AthenaBackend", "DuckDBBackend", "InvalidIdentifierError"]

# A small registry of in-memory tables the local engine can query offline. ``register_table``
# lets a flow/test seed data; ``_TABLES`` is process-wide so a seed survives to a query.
_TABLES: dict[str, list[dict[str, Any]]] = {}

# R-B-002 / AMLAS-SEC-B-002 (CWE-89 / bandit B608): the local DuckDB ``CREATE TABLE`` DDL
# interpolates the registered table name + column identifiers (values already use ``repr()``).
# Even though the engine is an ephemeral in-memory ``:memory:`` DB with no creds/persistence,
# unquoted identifier interpolation is an avoidable DDL-injection vector. We whitelist
# identifiers to a strict SQL-safe pattern and reject anything else BEFORE building the DDL —
# a malicious/typo'd registration (e.g. ``"x; DROP TABLE y"``) can no longer inject DDL.
_SQL_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class InvalidIdentifierError(ValueError):
    """Raised when a table/column identifier fails the SQL-safe whitelist (R-B-002)."""


def _validate_identifier(identifier: Any, *, kind: str) -> str:
    """Return ``identifier`` iff it matches the SQL-safe whitelist, else raise.

    Whitelist: ``^[A-Za-z_][A-Za-z0-9_]*$`` — letters/digits/underscore, not digit-leading.
    This rejects whitespace, quotes, semicolons, dots and any DDL metacharacter, so the
    identifier is safe to interpolate into the ephemeral DuckDB ``CREATE TABLE`` statement.
    """
    s = str(identifier)
    if not _SQL_IDENTIFIER.match(s):
        raise InvalidIdentifierError(
            f"unsafe {kind} identifier {identifier!r}: must match {_SQL_IDENTIFIER.pattern} "
            "(letters/digits/underscore, not digit-leading) — DDL-injection guard (R-B-002)"
        )
    return s


def register_table(name: str, rows: list[dict[str, Any]]) -> None:
    """Register an in-memory table ``name`` of row-dicts for the local query engine.

    The table name is validated against the SQL-safe identifier whitelist at registration
    time (R-B-002) so an unsafe name is rejected at the source, not just at DDL-build time.
    """
    _validate_identifier(name, kind="table")
    _TABLES[name] = list(rows or [])


# ══════════════════════════════════════════════════════════════════════════════
# DuckDBBackend — in-process SQL (offline default)
# ══════════════════════════════════════════════════════════════════════════════
class DuckDBBackend:
    """Local QueryBackend — DuckDB in-process (or a pure-Python fallback) over registered tables."""

    backend_name = "duckdb"

    def run_sql(self, *, query: str, database: str | None = None) -> dict[str, Any]:
        """T-QRY-01 — run ``query`` in-process; returns a ``QueryResult``-shaped dict."""
        try:
            return self._duckdb(query)
        except Exception:
            return self._fallback(query)

    # ── DuckDB path (genuine SQL when the dep is available) ─────────────────────
    def _duckdb(self, query: str) -> dict[str, Any]:
        # duckdb is an OPTIONAL SQL engine; absent from the offline dev venv (pyright cannot
        # resolve it there). The ImportError is caught by run_sql, which falls back to _fallback.
        import duckdb  # may raise ImportError → caught by run_sql  # pyright: ignore[reportMissingImports]
        import pandas as pd

        con = duckdb.connect(database=":memory:")
        try:
            for name, rows in _TABLES.items():
                if rows:
                    # R-B-002: validate every identifier, then use DuckDB's DataFrame
                    # registration API. No values or identifiers enter a generated SQL string.
                    safe_name = _validate_identifier(name, kind="table")
                    cols = [_validate_identifier(c, kind="column") for c in rows[0].keys()]
                    frame = pd.DataFrame(rows, columns=cols)
                    con.register(safe_name, frame)
            cur = con.execute(query)
            columns = [d[0] for d in (cur.description or [])]
            result_rows: list[list[Any]] = [list(r) for r in cur.fetchall()]
            return self._result(columns, result_rows, query, engine="duckdb")
        finally:
            con.close()

    # ── pure-Python fallback (no DuckDB): supports SELECT * / COUNT(*) over a table ─
    def _fallback(self, query: str) -> dict[str, Any]:
        q = (query or "").strip()
        m = re.search(r"from\s+([A-Za-z0-9_]+)", q, re.IGNORECASE)
        table = m.group(1) if m else ""
        rows = _TABLES.get(table, [])
        if re.search(r"count\s*\(\s*\*\s*\)", q, re.IGNORECASE):
            return self._result(["count"], [[len(rows)]], query, engine="duckdb-fallback")
        if not rows:
            return self._result([], [], query, engine="duckdb-fallback")
        # naive WHERE col = value filter (single equality), else SELECT *
        filtered = rows
        wm = re.search(r"where\s+([A-Za-z0-9_]+)\s*=\s*'?([^'\s]+)'?", q, re.IGNORECASE)
        if wm:
            col, val = wm.group(1), wm.group(2)
            filtered = [r for r in rows if str(r.get(col)) == val]
        cols = list(rows[0].keys())
        out_rows = [[r.get(c) for c in cols] for r in filtered]
        return self._result(cols, out_rows, query, engine="duckdb-fallback")

    @staticmethod
    def _result(columns: list[str], rows: list[list[Any]], query: str, *, engine: str) -> dict[str, Any]:
        return {
            "columns": columns, "rows": rows, "row_count": len(rows),
            "query": query, "engine": engine, "adapter": "duckdb",
        }


# ══════════════════════════════════════════════════════════════════════════════
# AthenaBackend — managed (Athena start→poll→results; ADR-004)
# ══════════════════════════════════════════════════════════════════════════════
class AthenaBackend:
    """Managed QueryBackend — thin Athena orchestration; local fallback offline-safe."""

    backend_name = "athena"

    def __init__(self, *, region: str | None = None, workgroup: str = "primary",
                 output_location: str = "", athena_client: Any = None, poll_max: int = 30) -> None:
        self._region = resolve_aws_region(region, required=False)
        self._workgroup = workgroup
        self._output = output_location
        self._client = athena_client
        self._poll_max = poll_max
        self._local = DuckDBBackend()

    def _athena(self) -> Any:
        if self._client is None:
            import boto3
            self._client = boto3.client("athena", region_name=resolve_aws_region(self._region))
        return self._client

    def run_sql(self, *, query: str, database: str | None = None) -> dict[str, Any]:
        """Start an Athena query → poll → fetch results; fall back to local DuckDB offline."""
        try:
            athena = self._athena()
            kwargs: dict[str, Any] = {"QueryString": query, "WorkGroup": self._workgroup}
            if database:
                kwargs["QueryExecutionContext"] = {"Database": database}
            if self._output:
                kwargs["ResultConfiguration"] = {"OutputLocation": self._output}
            qid = athena.start_query_execution(**kwargs)["QueryExecutionId"]
            state = self._poll(athena, qid)
            if state != "SUCCEEDED":
                return self._local.run_sql(query=query, database=database)
            return self._fetch(athena, qid, query)
        except Exception:
            out = self._local.run_sql(query=query, database=database)
            out["adapter"] = "athena"
            out["engine"] = "athena-fallback"
            return out

    def _poll(self, athena: Any, qid: str) -> str:
        for _ in range(self._poll_max):
            st = str(athena.get_query_execution(QueryExecutionId=qid)["QueryExecution"]["Status"]["State"])
            if st in ("SUCCEEDED", "FAILED", "CANCELLED"):
                return st
            # Deadline-bounded poll (returns "TIMEOUT" at the deadline) — P3 audit H-06.
            time.sleep(0.01)  # nosemgrep: arbitrary-sleep
        return "TIMEOUT"

    def _fetch(self, athena: Any, qid: str, query: str) -> dict[str, Any]:
        resp = athena.get_query_results(QueryExecutionId=qid)
        rows_raw = resp.get("ResultSet", {}).get("Rows", [])
        if not rows_raw:
            return {"columns": [], "rows": [], "row_count": 0, "query": query,
                    "engine": "athena", "execution_id": qid, "adapter": "athena"}
        columns = [c.get("VarCharValue", "") for c in rows_raw[0].get("Data", [])]
        data = [[c.get("VarCharValue") for c in r.get("Data", [])] for r in rows_raw[1:]]
        return {"columns": columns, "rows": data, "row_count": len(data), "query": query,
                "engine": "athena", "execution_id": qid, "adapter": "athena"}


# ── registration: duckdb local default offline; athena managed default ─────────
register_adapter(QueryBackend, "duckdb", lambda: DuckDBBackend(), is_local=True)
register_adapter(QueryBackend, "athena", lambda: AthenaBackend(), is_default=True)
register_adapter(QueryBackend, "local", lambda: DuckDBBackend())
