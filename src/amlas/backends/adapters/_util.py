"""Shared helpers for the M0 adapters.

The adapters WRAP existing tool functions (the strangler pattern, doc 05 §4.4):
today's behavior becomes the default/local adapter, byte-for-byte. The existing
``@tool``-decorated functions expose their raw callable at ``._tool_func`` (the
convention the existing callers may use, e.g. ``read_lake_slice._tool_func(...)``);
:func:`raw_tool` returns it so an adapter delegates to the SAME code path the tool
ran before — guaranteeing identical outputs (the behavior-preserving invariant).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def raw_tool(tool: Any) -> Callable[..., Any]:
    """Return the undecorated callable behind a Strands ``@tool`` (or the fn itself).

    Strands wraps a ``@tool`` function and exposes the original at ``._tool_func``;
    plain functions are returned unchanged. This lets an adapter call the exact
    existing implementation (no behavior change) without going through tool-spec
    validation.
    """
    fn: Callable[..., Any] = getattr(tool, "_tool_func", tool)
    return fn


def _csv_safe_cell(value: Any) -> Any:
    """Prevent spreadsheet formula execution while preserving non-string values."""
    if not isinstance(value, str):
        return value
    candidate = value.lstrip(" ")
    if candidate.startswith(("=", "+", "-", "@", "\t", "\r")):
        return "'" + value
    return value
