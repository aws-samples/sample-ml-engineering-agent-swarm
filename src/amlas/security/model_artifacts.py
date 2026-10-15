"""Fail-closed persistence for estimator artifacts.

Skops stores estimators without Python's executable object deserialization.  AMLAS
still treats every artifact as untrusted: the path must remain under an explicit
trusted root and every type Skops marks as unknown must belong to a reviewed ML
library before loading is attempted.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Final

import skops.io as skops_io

from amlas.config._env_compat import env_with_compat

_ARTIFACT_SUFFIX: Final = ".skops"
_REVIEWED_TYPE_PREFIXES: Final = (
    "numpy.",
    "scipy.",
    "sklearn.",
    "xgboost.",
)


class UntrustedArtifactError(RuntimeError):
    """Raised when an estimator artifact fails a trust-boundary check."""


def _default_trusted_roots() -> list[Path]:
    """Resolve operator-configured artifact roots, with cwd as the dev default."""
    configured = env_with_compat("AMLAS_TRUSTED_MODEL_ROOT", "").strip()
    if configured:
        return [Path(item).resolve() for item in configured.split(os.pathsep) if item.strip()]
    return [Path.cwd().resolve()]


def _is_within(path: Path, root: Path) -> bool:
    """Return whether ``path`` resolves beneath ``root``."""
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _validate_artifact_path(
    artifact_path: str | Path,
    *,
    trusted_root: str | Path | None,
) -> Path:
    """Validate extension, existence, and containment before parsing an artifact."""
    path = Path(artifact_path).resolve()
    roots = (
        [Path(trusted_root).resolve()]
        if trusted_root is not None
        else _default_trusted_roots()
    )

    if path.suffix != _ARTIFACT_SUFFIX:
        raise UntrustedArtifactError(f"refusing to load non-.skops artifact: {path}")
    if not path.is_file():
        raise UntrustedArtifactError(f"artifact does not exist or is not a file: {path}")
    if not any(_is_within(path, root) for root in roots):
        raise UntrustedArtifactError(
            f"artifact path {path} is outside the trusted root(s) "
            f"{[str(root) for root in roots]}"
        )
    return path


def dump_model_artifact(model: Any, artifact_path: str | Path) -> Path:
    """Persist ``model`` in Skops format and return the resolved artifact path."""
    path = Path(artifact_path)
    if path.suffix != _ARTIFACT_SUFFIX:
        raise UntrustedArtifactError(f"refusing to write non-.skops artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    skops_io.dump(model, path)
    return path.resolve()


def load_model_artifact(
    artifact_path: str | Path,
    *,
    trusted_root: str | Path | None = None,
) -> Any:
    """Load a reviewed Skops artifact after path and serialized-type validation.

    Skops has a built-in trusted set for common scikit-learn primitives.  Any
    additional type is surfaced by :func:`skops.io.get_untrusted_types`; AMLAS
    permits only reviewed numerical/ML library namespaces and rejects custom
    application types before calling :func:`skops.io.load`.
    """
    path = _validate_artifact_path(artifact_path, trusted_root=trusted_root)
    unknown_types = sorted(skops_io.get_untrusted_types(file=path))
    unreviewed = [
        type_name
        for type_name in unknown_types
        if not type_name.startswith(_REVIEWED_TYPE_PREFIXES)
    ]
    if unreviewed:
        raise UntrustedArtifactError(
            f"artifact contains unreviewed serialized types: {unreviewed}"
        )
    return skops_io.load(path, trusted=unknown_types)
