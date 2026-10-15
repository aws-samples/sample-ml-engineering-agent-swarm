"""Parse task_spec.md — YAML front-matter + markdown body — into SpecModel."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import frontmatter
from pydantic import ValidationError

from .schema import SpecModel


class SpecLoadError(Exception):
    """Raised when a task_spec.md cannot be parsed or validated."""


def load_spec_from_text(text: str) -> SpecModel:
    """Parse a spec string. Raises SpecLoadError on any parse or validation failure."""
    try:
        post = frontmatter.loads(text)
    except Exception as e:  # pragma: no cover — frontmatter raises various exceptions
        raise SpecLoadError(f"Failed to parse YAML front-matter: {e}") from e

    metadata: dict[str, Any] = dict(post.metadata)
    if not metadata:
        raise SpecLoadError(
            "No YAML front-matter found. task_spec.md must start with '---' delimited YAML."
        )
    metadata["domain_notes"] = post.content.strip()

    try:
        return SpecModel(**metadata)
    except ValidationError as e:
        raise SpecLoadError(f"Spec validation failed:\n{e}") from e


def load_spec_from_path(path: str | Path) -> SpecModel:
    """Load and validate a task_spec.md from disk."""
    p = Path(path)
    if not p.exists():
        raise SpecLoadError(f"Spec not found: {p}")
    return load_spec_from_text(p.read_text(encoding="utf-8"))


def load_spec_dict(path: str | Path) -> dict[str, Any]:
    """Load a spec file into a plain dict, handling BOTH shapes (CLI-01).

    The shipped specs are YAML front-matter + markdown body (``---`` … ``---``
    then prose), which ``yaml.safe_load`` cannot parse. Legacy specs are plain
    YAML dicts. This helper:
      1. Tries the frontmatter parser; if front-matter metadata is present,
         returns it (with the markdown body attached as ``domain_notes``).
      2. Falls back to ``yaml.safe_load`` for plain-YAML spec files.

    Returns the spec as a dict (NOT a validated SpecModel — callers that need
    validation should use ``load_spec_from_path``).

    Raises:
        SpecLoadError: if the file is missing or cannot be parsed as either shape.
    """
    p = Path(path)
    if not p.exists():
        raise SpecLoadError(f"Spec not found: {p}")

    text = p.read_text(encoding="utf-8")

    # 1. Front-matter (.md) shape — preferred for the shipped specs.
    try:
        post = frontmatter.loads(text)
        metadata: dict[str, Any] = dict(post.metadata)
        if metadata:
            body = post.content.strip()
            if body:
                metadata.setdefault("domain_notes", body)
            return metadata
    except Exception:  # nosec B110
        pass

    # 2. Plain-YAML fallback.
    import yaml

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise SpecLoadError(f"Failed to parse spec as front-matter or YAML: {e}") from e

    if not isinstance(data, dict):
        raise SpecLoadError(
            f"Spec must be a YAML mapping or front-matter document; got {type(data).__name__}."
        )
    return data
