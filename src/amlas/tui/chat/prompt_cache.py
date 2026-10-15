"""PERF-07 — Bedrock prompt-cache stable-prefix validation.

The single home for the two validators that decide whether an assembled prompt can
actually hit the Bedrock prompt cache: :func:`cache_prompt_segments` structures a
prompt as ``(kind, text)`` segments and :func:`prefix_is_cacheable` checks the
ordering rule (every ``stable`` segment precedes every ``volatile`` one).

They used to live in the TUI chat backend (``tui/chat/bedrock_backend.py``), which
coupled a pure, Bedrock-free ordering check to the raw-Converse fallback brain — so
deleting that brain (issue #35) would have forced a gutted module to survive just to
host them. They stay inside ``tui/chat`` because the prompt they validate is the chat
prompt, but they have no dependency on any backend, client or credential path: the
only thing they touch is a list of segments.

The function bodies moved here **verbatim** from the TUI backend, so the PERF-07
prefix contract (design Property 9) survives the move by construction.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "cache_prompt_segments",
    "prefix_is_cacheable",
]


# ── PERF-07: prompt-cache stable-prefix validation (the cache-effectiveness gate) ──
def cache_prompt_segments(system_text: str, messages: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Return the prompt assembled as ``(kind, text)`` segments for cache-prefix validation.

    ``kind`` is ``"stable"`` for the cacheable prefix (the system prompt + any prior-turn
    history that is identical across turns) and ``"volatile"`` for the per-turn tail (the new
    user message). Bedrock prompt caching only pays off when the STABLE prefix is contiguous
    and FIRST, so this is the structure :func:`prefix_is_cacheable` checks (PERF-07).
    """
    segments: list[tuple[str, str]] = [("stable", system_text)]
    # All but the last message are prior-turn history (stable across this turn); the final
    # user message is the volatile per-turn tail.
    for m in messages[:-1]:
        segments.append(("stable", str(m.get("content", ""))))
    if messages:
        segments.append(("volatile", str(messages[-1].get("content", ""))))
    return segments


def prefix_is_cacheable(segments: list[tuple[str, str]]) -> bool:
    """True iff every STABLE segment precedes every VOLATILE one (the cacheable-prefix rule).

    Prompt caching matches a stable PREFIX; if any volatile (per-turn) content is interleaved
    before a stable segment, the prefix is broken and the cache cannot hit. This validates the
    PERF-07 contract — system prompt + tool specs first, volatile context last.
    """
    seen_volatile = False
    for kind, _text in segments:
        if kind == "volatile":
            seen_volatile = True
        elif seen_volatile:  # a stable segment AFTER a volatile one breaks the prefix
            return False
    return True
