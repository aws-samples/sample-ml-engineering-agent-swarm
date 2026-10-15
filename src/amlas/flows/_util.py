"""Shared helpers for the M2+ flows.

The flows compose the EXISTING ``@tool``-decorated functions. Strands wraps a ``@tool``
and exposes the original callable at ``._tool_func``; :func:`raw` returns it so a flow
calls the exact existing implementation (behavior-preserving) without tool-spec
validation — the same convention as ``backends/adapters/_util.raw_tool``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

__all__ = ["raw"]


def raw(tool: Any) -> Callable[..., Any]:
    """Return the undecorated callable behind a Strands ``@tool`` (or the fn itself)."""
    fn: Callable[..., Any] = getattr(tool, "_tool_func", tool)
    return fn
