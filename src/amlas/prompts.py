"""Prompt loader — resolves markdown prompt files under `prompts/` at import time.

All agent system prompts are verbatim-ported markdown files (with attribution comments).
Keeping them as standalone files means:
    - They can be diffed against their upstream sources across bumps.
    - Prompt engineers can edit without touching Python.
    - Tests can assert presence of specific directives.

Usage:
    >>> from amlas.prompts import load_prompt, stack_prompts
    >>> p = stack_prompts("preambles/paranoid_scientist", "subagents/data_integrity")
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"


class PromptNotFoundError(FileNotFoundError):
    pass


@lru_cache(maxsize=128)
def load_prompt(name: str) -> str:
    """Load a prompt by dotted or slash path (e.g., 'preambles/paranoid_scientist').

    Accepts with or without '.md' extension. Strips HTML-comment attribution lines so
    the returned text is agent-clean.
    """
    stem = name.removesuffix(".md")
    path = PROMPTS_DIR / f"{stem}.md"
    if not path.exists():
        raise PromptNotFoundError(f"prompt not found: {path}")
    text = path.read_text(encoding="utf-8")
    return _strip_html_comments(text).strip()


def stack_prompts(*names: str, sep: str = "\n\n---\n\n") -> str:
    """Concatenate multiple prompts in order; default separator is a markdown HR."""
    parts = [load_prompt(n) for n in names]
    return sep.join(parts)


# ── WA-09: prompt versioning / catalog seam (Bedrock-Prompt-Management-compatible) ──
# The audit found prompts were static files with NO versioning. These functions add a
# deterministic content-hash version + a catalog-record shape so prompts can be versioned and
# eval-gated like Bedrock Prompt Management, while staying offline + dependency-free. When a
# managed Bedrock Prompt Management adapter lands, it reuses ``prompt_version`` as the content
# digest and ``prompt_catalog_entry`` as the upsert payload.
def prompt_version(text: str) -> str:
    """Return a deterministic short content-hash version for a prompt body (WA-09).

    Same content → same version; any change → a new version. This is the version key a
    prompt catalog / Bedrock Prompt Management would pin and eval-gate against.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def prompt_catalog_entry(*, name: str, text: str | None = None,
                         metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a versioned prompt-catalog record for ``name`` (WA-09 catalog seam).

    Loads the prompt by ``name`` when ``text`` is not supplied, stamps the content-hash
    ``version``, and carries optional ``metadata`` (owner, eval-gate status, …). This is the
    Bedrock-Prompt-Management-compatible record the (future) managed adapter upserts.
    """
    body = text if text is not None else load_prompt(name)
    return {
        "name": name,
        "version": prompt_version(body),
        "text": body,
        "metadata": dict(metadata or {}),
    }


def _strip_html_comments(text: str) -> str:
    import re

    return re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
