"""LakeBackend adapters (doc 05 §4.2 G2 / §6 / roadmap M0 work-item 7).

* ``S3MedallionBackend`` — the **default**; wraps the EXISTING ``tools/lake/``
  (``read_lake_slice`` / ``write_lake_dataset`` / ``list_lake_catalog``). The
  medallion bronze-write rejection (ADR-009) is PRESERVED — the wrapped tool still
  enforces it internally, and the registry/hooks layer (``hooks.py``) now also
  routes the deny through a declarative ``PreToolUse`` guardrail.
* ``LocalFSBackend`` — the **local** adapter for offline/dev: writes still go
  through the EXISTING tool (so the bronze rejection + lineage sidecar are
  identical), but the offline path is exercised with an injected/local S3 client by
  tests. (A pure-local-filesystem variant is a later refinement; M0 keeps behavior
  identical by delegating to the same tool.)

Both implement :class:`~amlas.backends.ports.LakeBackend` and return the
domain ``DatasetRef`` / ``CatalogView`` types — translating the existing tool's
dict shape without changing any logic.
"""

from __future__ import annotations

from typing import Any

from ...config.aws_region import resolve_aws_region
from ..ports import LakeBackend
from ..registry import register_adapter
from ..types import CatalogView, DatasetRef
from ._util import raw_tool


class S3MedallionBackend:
    """Default LakeBackend — wraps EXISTING ``tools/lake`` (bronze rejection kept).

    AWS clients are injectable (the existing tool's ``s3_client=`` / ``glue_client=``
    convention) so real-S3/offline callers may pass their own client; prod leaves them
    ``None`` and the tool builds boto3 lazily.
    """

    backend_name = "s3medallion"

    def __init__(
        self,
        *,
        region: str | None = None,
        s3_client: Any = None,
        glue_client: Any = None,
    ) -> None:
        self._region = resolve_aws_region(region, required=False)
        self._s3 = s3_client
        self._glue = glue_client

    def read_slice(
        self,
        *,
        name_or_uri: str,
        columns: list[str] | None = None,
        filters: dict[str, Any] | None = None,
        max_rows: int | None = None,
    ) -> DatasetRef:
        """Read a tier-scoped slice via EXISTING ``read_lake_slice`` (unchanged)."""
        from ...tools.lake import read_lake_slice

        out = raw_tool(read_lake_slice)(
            name_or_uri,
            columns=columns,
            filters=filters,
            limit=max_rows,
            region=self._region,
            s3_client=self._s3,
            glue_client=self._glue,
        )
        if out.get("success") is False:
            raise RuntimeError(out.get("error", "read_lake_slice failed"))
        return DatasetRef(
            uri=str(out.get("local_path", name_or_uri)),
            tier=str(out.get("tier", "")),
            rows=int(out.get("rows", 0)),
            schema=dict(out.get("schema", {})),
            lineage=dict(out.get("provenance", {})),
        )

    def write_dataset(
        self,
        *,
        local_path: str,
        dest_uri: str,
        tier: str = "silver",
        lineage: dict[str, Any] | None = None,
        profile: dict[str, Any] | None = None,
    ) -> str:
        """Write back via EXISTING ``write_lake_dataset`` (bronze rejection preserved).

        Returns the written S3 URI. Raises ``RuntimeError`` on a soft-failure (e.g.
        the bronze/raw rejection or a steering-forbidden tier), surfacing the
        existing tool's exact error message — the governance behavior is unchanged.
        """
        from ...tools.lake import write_lake_dataset

        out = raw_tool(write_lake_dataset)(
            local_path,
            dest_uri,
            tier=tier,
            lineage=lineage,
            profile=profile,
            region=self._region,
            s3_client=self._s3,
        )
        if out.get("success") is False:
            raise RuntimeError(out.get("error", "write_lake_dataset rejected"))
        return str(out.get("written_uri", ""))

    def list_catalog(self, *, database: str | None = None) -> CatalogView:
        """List the Glue catalog via EXISTING ``list_lake_catalog`` (unchanged)."""
        from ...tools.lake import list_lake_catalog

        out = raw_tool(list_lake_catalog)(database, region=self._region, glue_client=self._glue)
        if out.get("success") is False:
            raise RuntimeError(out.get("error", "list_lake_catalog failed"))
        return CatalogView(datasets=list(out.get("datasets", [])), count=int(out.get("count", 0)))


class LocalFSBackend(S3MedallionBackend):
    """Local/offline LakeBackend.

    M0 keeps behavior IDENTICAL to ``S3MedallionBackend`` by delegating to the same
    EXISTING tool (so the bronze rejection + lineage sidecar are byte-for-byte the
    same); tests drive it with an injected S3 client. A pure-filesystem store is a
    later refinement — the seam exists now so the offline default resolves here.
    """

    backend_name = "localfs"


register_adapter(LakeBackend, "s3medallion", lambda: S3MedallionBackend(), is_default=True)
register_adapter(LakeBackend, "localfs", lambda: LocalFSBackend(), is_local=True)
