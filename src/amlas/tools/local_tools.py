"""Local (non-cloud) @tool helpers — REAL implementations, relocated from the
deleted ``tools/stubs.py`` in Phase 5 of the de-mock initiative.

These two tools were never fakes — they were real code that merely *lived* in the
``stubs`` module alongside the fake-data generators (PS-01/02/03). When the fakes
were deleted, the real tools moved here:

  * ``save_feature_manifest`` — persists a real feature manifest JSON to disk.
  * ``execute_python_tool``  — executes Python in the REAL AgentCore Code Interpreter
    kernel (state survives across calls).

``configure_artifact_dir`` sets where ``save_feature_manifest`` writes (tests point it
at a tmp path). No fabricated data is produced anywhere in this module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from strands import tool

from ..errors import tool_error_boundary

_ARTIFACT_DIR: Path | None = None


def configure_artifact_dir(path: str | Path | None) -> None:
    """Set the dir where ``save_feature_manifest`` writes (used by tests)."""
    global _ARTIFACT_DIR
    _ARTIFACT_DIR = Path(path) if path else None


# Back-compat alias for the old name used while the tools lived in stubs.py.
configure_stub_artifact_dir = configure_artifact_dir


def _artifact_path(name: str) -> Path:
    base = _ARTIFACT_DIR or Path("runs/_local_dev")
    base.mkdir(parents=True, exist_ok=True)
    return base / name


@tool
@tool_error_boundary(source="tools.local_tools")
def save_feature_manifest(
    features: list[dict[str, Any]], rejected: list[dict[str, Any]] | None = None
) -> str:
    """Persist a feature manifest JSON. Returns the filesystem path.

    Args:
        features: list of feature records ({name, derivation, rationale, cv_delta, ...}).
        rejected: list of rejected-feature records ({name, reason}).
    """
    path = _artifact_path("feature_manifest.json")
    path.write_text(
        json.dumps({"features": list(features), "rejected": list(rejected or [])}, indent=2)
    )
    return str(path)


@tool
@tool_error_boundary(source="tools.local_tools")
def execute_python_tool(code: str, description: str = "", region: str | None = None) -> dict[str, Any]:
    """Execute Python in the persistent AgentCore Code Interpreter kernel.

    State (imports, DataFrames, trained models) survives across calls.

    Args:
        code: Python source to execute.
        description: optional short description (prepended as a comment).
        region: AWS region for the Code Interpreter session.
    """
    from .code_interpreter import execute_python  # late import to avoid strands circular

    r = execute_python(code, description=description, region=region)
    return {
        "is_error": r.is_error,  # nosemgrep
        "stdout": r.stdout,
        "stderr": r.stderr,
        "exit_code": r.exit_code,
        "execution_time": r.execution_time,
        "text": r.text,
    }
