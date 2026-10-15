"""ModelRegistryBackend adapters (doc 05 §4.2 G4 / roadmap M0 work-item 6).

* ``SageMakerRegistryBackend`` — **default**; wraps EXISTING ``tools/mlops/registry.py``
  (``register_model_sm`` / ``approve_model_sm`` / ``list_model_packages_sm``).
* ``JSONFileRegistryBackend`` — **local** test registry: a JSON-file model package
  store with the same register/approve/list contract (no AWS), so offline callers
  exercise the port without SageMaker.

Both implement :class:`~amlas.backends.ports.ModelRegistryBackend` and
return the domain ``PackageRef``.
"""

from __future__ import annotations

import json
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any

from ...config.aws_region import resolve_aws_region
from ..ports import ModelRegistryBackend
from ..registry import register_adapter
from ..types import PackageRef
from ._util import raw_tool


class SageMakerRegistryBackend:
    """Default ModelRegistryBackend — wraps EXISTING ``tools/mlops/registry.py``."""

    def __init__(self, *, region: str | None = None, sm_client: Any = None) -> None:
        self._region = resolve_aws_region(region, required=False)
        self._sm = sm_client

    def register(
        self,
        *,
        group: str,
        artifact: str,
        image: str | None = None,
        metrics: dict[str, Any] | None = None,
        approval: str = "PendingManualApproval",
    ) -> PackageRef:
        """Register a model package via EXISTING ``register_model_sm`` (unchanged)."""
        from ...tools.mlops.registry import register_model_sm

        out = raw_tool(register_model_sm)(
            artifact, group, metrics=metrics, approval=approval,
            image=image, region=self._region, sm_client=self._sm,
        )
        if out.get("success") is False:
            raise RuntimeError(out.get("error", "register_model_sm failed"))
        return PackageRef(
            arn=str(out.get("model_package_arn", "")),
            group=str(out.get("group", group)),
            version=int(out.get("version", 1)),
            approval_status=str(out.get("approval_status", approval)),
            metrics=dict(out.get("metrics", {})),
        )

    def approve(self, *, package: str, status: str = "Approved", reason: str | None = None) -> str:
        """Transition approval via EXISTING ``approve_model_sm`` (unchanged)."""
        from ...tools.mlops.registry import approve_model_sm

        out = raw_tool(approve_model_sm)(package, status=status, region=self._region, sm_client=self._sm)
        if out.get("success") is False:
            raise RuntimeError(out.get("error", "approve_model_sm failed"))
        return str(out.get("status", status))

    def list(self, *, group: str, status: str | None = None) -> list[PackageRef]:
        """List package versions via EXISTING ``list_model_packages_sm`` (unchanged)."""
        from ...tools.mlops.registry import list_model_packages_sm

        out = raw_tool(list_model_packages_sm)(group, region=self._region, sm_client=self._sm)
        if out.get("success") is False:
            raise RuntimeError(out.get("error", "list_model_packages_sm failed"))
        refs = [
            PackageRef(
                arn=str(v.get("model_package_arn", "")),
                group=str(out.get("group", group)),
                version=int(v.get("version", 1) or 1),
                approval_status=str(v.get("approval_status", "")),
            )
            for v in out.get("versions", [])
        ]
        if status:
            refs = [r for r in refs if r.approval_status == status]
        return refs


class JSONFileRegistryBackend:
    """Local test ModelRegistryBackend — a JSON-file model package store (no AWS).

    Same register/approve/list contract as the SageMaker adapter so offline callers
    resolve the port without SageMaker. Thread-safe atomic writes (temp + rename).
    """

    def __init__(self, *, store_path: str | Path | None = None) -> None:
        self._path = Path(store_path) if store_path else Path(tempfile.gettempdir()) / "amlas_local_registry.json"
        self._lock = threading.Lock()

    def _load(self) -> list[dict[str, Any]]:
        if self._path.exists():
            try:
                return list(json.loads(self._path.read_text()))
            except (json.JSONDecodeError, ValueError):
                return []
        return []

    def _save(self, rows: list[dict[str, Any]]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(rows, indent=2))
        tmp.rename(self._path)

    def register(
        self,
        *,
        group: str,
        artifact: str,
        image: str | None = None,
        metrics: dict[str, Any] | None = None,
        approval: str = "PendingManualApproval",
    ) -> PackageRef:
        """Append a versioned package row to the JSON store and return its ref."""
        with self._lock:
            rows = self._load()
            version = sum(1 for r in rows if r.get("group") == group) + 1
            arn = f"local:registry:{group}/{version}:{uuid.uuid4().hex[:8]}"
            rows.append({
                "arn": arn, "group": group, "version": version,
                "approval_status": approval, "artifact": artifact,
                "metrics": dict(metrics or {}),
            })
            self._save(rows)
        return PackageRef(arn=arn, group=group, version=version, approval_status=approval, metrics=dict(metrics or {}))

    def approve(self, *, package: str, status: str = "Approved", reason: str | None = None) -> str:
        """Transition the named package's approval status in the JSON store."""
        with self._lock:
            rows = self._load()
            for r in rows:
                if r.get("arn") == package:
                    r["approval_status"] = status
            self._save(rows)
        return status

    def list(self, *, group: str, status: str | None = None) -> list[PackageRef]:
        """List the JSON store's package versions for ``group`` (optional status filter)."""
        rows = [r for r in self._load() if r.get("group") == group]
        if status:
            rows = [r for r in rows if r.get("approval_status") == status]
        return [
            PackageRef(
                arn=str(r.get("arn", "")), group=group,
                version=int(r.get("version", 1)), approval_status=str(r.get("approval_status", "")),
                metrics=dict(r.get("metrics", {})),
            )
            for r in rows
        ]


register_adapter(ModelRegistryBackend, "sagemaker", lambda: SageMakerRegistryBackend(), is_default=True)
register_adapter(ModelRegistryBackend, "jsonfile", lambda: JSONFileRegistryBackend(), is_local=True)
register_adapter(ModelRegistryBackend, "local", lambda: JSONFileRegistryBackend())
