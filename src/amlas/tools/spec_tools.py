"""Strands @tool wrappers for spec generation workflow.

Tools:
  - validate_spec: Validates YAML front-matter + body against SpecModel without saving
  - save_spec: Validates + writes spec to disk
  - list_example_specs: Returns available exemplar specs from specs/ directory
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:  # Phase E2 FIX-1 — decouple the pure-filesystem tools from the agent runtime.
    from strands import tool
except ModuleNotFoundError:  # pragma: no cover - exercised via the strands-absent test
    # ``list_example_specs`` / ``validate_spec`` / ``save_spec`` are plain
    # functions (pure disk/validation work); they do NOT need the Strands agent
    # runtime. On a partial/minimal install where ``strands`` is un-importable
    # they must still be callable directly (the reported "`/spec list` doesn't
    # work offline" bug — audit §3.2 / FIX-1). This no-op passthrough decorator
    # lets the module import + the functions run without ``strands`` present,
    # while a full install still wraps them as real ``@tool``s (behavior
    # unchanged when ``strands`` IS importable).
    def tool(fn: Any = None, **_: Any) -> Any:  # type: ignore[no-redef]
        """Minimal passthrough ``@tool`` fallback (used only when ``strands`` is absent)."""
        return fn if fn is not None else (lambda f: f)

from ..errors import tool_error_boundary
from ..spec import SpecLoadError, load_spec_from_text

# Specs directory — sibling to src/ at repo root
SPECS_DIR = Path(__file__).resolve().parents[3] / "specs"


@tool
@tool_error_boundary(source="tools.spec_tools")
def validate_spec(yaml_text: str) -> dict[str, Any]:
    """Validate a spec YAML front-matter + markdown body against the SpecModel schema.

    Returns a dict with:
      - valid: bool — whether the spec is valid
      - spec: dict — the validated spec fields (if valid)
      - errors: list[str] — validation error messages (if invalid)

    Args:
        yaml_text: Full spec text with YAML front-matter delimited by --- and markdown body.
    """
    if not yaml_text or not yaml_text.strip():
        return {"valid": False, "errors": ["Empty input — spec text must contain YAML front-matter."]}

    try:
        spec = load_spec_from_text(yaml_text)
        return {
            "valid": True,
            "spec": spec.model_dump(),
            "errors": [],
        }
    except SpecLoadError as e:
        return {
            "valid": False,
            "errors": [str(e)],
        }


@tool
@tool_error_boundary(source="tools.spec_tools")
def save_spec(yaml_text: str, path: str) -> dict[str, Any]:
    """Validate a spec and save it to disk if valid.

    Will NOT write the file if validation fails. Returns success/failure status.

    Args:
        yaml_text: Full spec text with YAML front-matter delimited by --- and markdown body.
        path: Filesystem path where the spec file should be written.
    """
    if not yaml_text or not yaml_text.strip():
        return {"saved": False, "errors": ["Empty input — nothing to save."]}

    try:
        # Validate first
        load_spec_from_text(yaml_text)
    except SpecLoadError as e:
        return {"saved": False, "errors": [str(e)]}

    # Write to disk — traversal containment (Security review, CWE-22): the
    # agent-supplied path must stay inside the sandbox/CWD/tempdir roots.
    from .pathref import resolve_dataset_path

    try:
        p = resolve_dataset_path(path)
    except PermissionError as e:
        return {"saved": False, "errors": [str(e)]}
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml_text, encoding="utf-8")
    return {"saved": True, "path": str(p), "errors": []}


@tool
@tool_error_boundary(source="tools.spec_tools")
def list_example_specs() -> dict[str, Any]:
    """Return all available example specs from the specs/ directory.

    Each entry includes the spec filename and its full content, suitable for
    few-shot prompting the spec writer agent.
    """
    specs: list[dict[str, str]] = []
    if SPECS_DIR.exists():
        for spec_file in sorted(SPECS_DIR.glob("*.md")):
            specs.append({
                "name": spec_file.stem,
                "content": spec_file.read_text(encoding="utf-8"),
            })
    return {"specs": specs}
