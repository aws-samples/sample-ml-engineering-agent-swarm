"""Spec generation from natural language description using heuristic parsing.

Provides:
  - generate_spec_from_description: @tool that parses free-text into SpecModel fields

This uses regex + keyword matching (no LLM API call). The LLM agent (spec_writer)
can call this tool and validate/refine the result.
"""

from __future__ import annotations

import re
from typing import Any

from strands import tool

from ..errors import tool_error_boundary

# DRY-01 fix: the pure heuristic NL-parsing helpers now live in the
# STRANDS-FREE `spec_heuristics` module so this @tool wrapper AND the boto3-only
# cloud-dispatch client (`cloud.spec_builder`) share ONE source of truth instead
# of re-inlining identical logic. Re-imported here (and re-exported via the
# module namespace) so existing callers of these helpers keep working.
from .spec_heuristics import (
    _extract_metric,
    _extract_target_column,
    _extract_threshold,
    _extract_title,
    _infer_task_type,
    _validate_column_in_data,
)


def _project_id_from_title(title: str) -> str:
    """Derive a SpecModel-valid project_id (``[a-zA-Z0-9_]+``, <=48 chars)."""
    pid = re.sub(r"[^a-zA-Z0-9]+", "_", title.strip().lower()).strip("_")
    pid = pid or "generated_spec"
    return pid[:48].rstrip("_") or "generated_spec"


def _normalize_golden_dataset(data_path: str | None) -> str:
    """Coerce a data path into a SpecModel-valid golden_dataset string.

    SpecModel requires ``s3://``, an absolute (``/``), or relative (``./``)
    prefix. Bare paths are normalized to ``./<path>``; missing paths get a
    sensible placeholder the user is expected to edit.
    """
    if not data_path:
        return "./data/train.csv"
    if data_path.startswith(("s3://", "/", "./")):
        return data_path
    return f"./{data_path}"


def spec_dict_to_frontmatter(spec: dict[str, Any]) -> str:
    """Render a generated spec dict into SpecModel-valid front-matter text (SPEC-02).

    Maps the heuristic writer output (task_type/target/primary_metric/
    success_threshold/...) onto the full ``SpecModel`` contract — adding the
    required ``project_id``, ``golden_dataset``, ``baseline``, and
    ``acceptance`` blocks with sensible placeholders — so the spec-writer
    workflow's two ends are contract-aligned and the result passes
    ``validate_spec``.
    """
    import yaml as _yaml

    generation = spec.get("task_type") == "text_generation"
    primary_metric = spec.get("primary_metric", "" if generation else "f1_macro")
    if generation and not primary_metric:
        raise ValueError("text_generation requires a researched primary_metric before rendering")
    title = spec.get("title") or spec.get("description", "Generated Spec")
    project_id = spec.get("project_id") or _project_id_from_title(title)
    baseline = spec.get("baseline")
    if not isinstance(baseline, dict):
        baseline = (
            {"type": "untouched_base_model", primary_metric: None}
            if generation else {"type": "heuristic_default", primary_metric: 0.0}
        )
    acceptance = spec.get("acceptance")
    if not isinstance(acceptance, dict):
        if generation:
            raise ValueError("text_generation requires explicit acceptance criteria before rendering")
        acceptance = {"primary_metric_delta_vs_baseline": ">= 0.0"}

    metadata = {
        "project_id": project_id,
        "task_type": spec.get("task_type", "classification"),
        "target": spec.get("target", "target"),
        "primary_metric": primary_metric,
        "golden_dataset": _normalize_golden_dataset(
            spec.get("golden_dataset") or spec.get("data_path")
        ),
        "baseline": baseline,
        "acceptance": acceptance,
    }
    for name in ("constraints", "splits", "compute_budget", "medallion"):
        if name in spec:
            metadata[name] = spec[name]
    body = spec.get("description", "").strip() or title
    front = _yaml.dump(metadata, default_flow_style=False, sort_keys=False)
    return f"---\n{front}---\n\n{body}\n"


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


@tool
@tool_error_boundary(source="tools.spec_generation_tools")
def generate_spec_from_description(description: str, data_path: str | None = None) -> dict[str, Any]:
    """Generate a validated SpecModel from a natural language description.

    The tool:
    1. Takes free-text description of business problem
    2. Infers task_type, target_column, metric, success_threshold
    3. If data_path provided, examines data to validate column names
    4. Returns complete SpecModel dict (ready for validate_spec)

    Example input: "Predict which spaceship passengers were transported to
    another dimension. The target is the 'Transported' column.
    We need at least 75% F1 score."

    Args:
        description: Natural language description of the ML problem.
        data_path: Optional path to CSV data file to validate column names.

    Returns:
        Dict with title, description, task_type, target, primary_metric,
        success_threshold, data_path, and other SpecModel fields.
    """
    if not description or not description.strip():
        return {
            "error": "Empty description provided.",
            "task_type": "classification",
            "target": "target",
            "primary_metric": "f1_macro",
            "success_threshold": 0.80,
            "title": "Untitled ML Task",
            "description": "",
        }

    # Parse components from description
    task_type = _infer_task_type(description)
    target = _extract_target_column(description)
    primary_metric = _extract_metric(description)
    success_threshold = _extract_threshold(description)
    title = _extract_title(description)

    # Validate against data if path provided
    if data_path:
        if _validate_column_in_data(data_path, target):
            pass  # Column exists — good
        else:
            # Try to find a close match
            try:
                import pandas as pd

                df = pd.read_csv(data_path, nrows=0)
                # Look for case-insensitive match
                for col in df.columns:
                    if col.lower() == target.lower():
                        target = col
                        break
            except Exception:  # nosec B110
                pass

    result: dict[str, Any] = {
        "title": title,
        "description": description.strip(),
        "task_type": task_type,
        "target": target,
        "primary_metric": primary_metric,
        "success_threshold": success_threshold,
    }

    if data_path:
        result["data_path"] = data_path

    if task_type == "text_generation":
        result["needs_research"] = [
            name for name, value in (
                ("primary_metric", primary_metric), ("success_threshold", success_threshold),
            )
            if value is None or value == ""
        ]

    return result
