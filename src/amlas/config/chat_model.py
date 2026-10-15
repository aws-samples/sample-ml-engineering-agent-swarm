"""The single source for the TUI chat tier's model, output cap and history bound.

Issue #35 criteria 4.1 / 4.2. Three values that decide how the conversational TUI
tier talks to Bedrock were each decided in more than one place:

* the model id — ``env_with_compat("AMLAS_CHAT_MODEL", ...)`` was resolved
  independently in ``tui/chat/tool_binding_backend.py``, in the (now-deleted)
  raw-Converse ``tui/chat/bedrock_backend.py``, and a third time in
  ``tui/chat/engine.py`` for the activity panel's model label;
* the output cap — the Strands path took ``cfg.bedrock.max_tokens`` (64000) via the
  agent factory while the raw path hardcoded ``maxTokens: 2048`` twice;
* the replayed-history bound — ``MAX_HISTORY_ENTRIES = 40`` and
  ``MAX_AGENT_HISTORY_MESSAGES = 40``, the latter carrying a comment asking the
  reader to keep it equal to the former by hand.

This module is that one place. Every chat-tier caller reads from here, so the
duplicates collapse to imports and the AST single-source assertion (task 4.4) can
require exactly ONE ``env_with_compat("AMLAS_CHAT_MODEL", ...)`` call site in
``src/``. Keep it that way: add chat-tier config here rather than re-reading the
env at a call site.

Leaf-ish by design, like :mod:`.bedrock_creds`: stdlib plus :mod:`._env_compat` and
:mod:`.runtime` (both of which import only the stdlib), so there is no cycle with
the config seam, no ``strands`` import and no AWS dependency. :func:`~.runtime.load_config`
is imported from :mod:`.runtime` directly rather than from the ``amlas.config``
package re-export — same object, one hop fewer, and it keeps this module readable
as a leaf.
"""

from __future__ import annotations

import logging

from ._env_compat import env_with_compat
from .runtime import DEFAULT_AGENT_MODEL, load_config

logger = logging.getLogger(__name__)

__all__ = [
    "CHAT_HISTORY_MESSAGES",
    "chat_max_tokens",
    "chat_model_id",
]

# The single bound on how many PRIOR messages the chat tier replays to the model
# each turn. Was hand-copied as ``MAX_HISTORY_ENTRIES`` (raw backend) and
# ``MAX_AGENT_HISTORY_MESSAGES`` (engine); ``engine`` now aliases this name so the
# sliding window cannot drift between the two. Value unchanged at 40 — this is a
# consolidation, not a re-tuning.
CHAT_HISTORY_MESSAGES = 40


def chat_model_id() -> str:
    """The chat tier's Bedrock model id — the ONLY definition in ``src/``.

    Shares :data:`~amlas.config.runtime.DEFAULT_AGENT_MODEL` with hosted agents
    and swarm roles. ``AMLAS_CHAT_MODEL`` overrides it; the legacy
    ``AGENTICML_*`` prefix is still honored for one release via
    :func:`~amlas.config._env_compat.env_with_compat`.
    """
    return str(env_with_compat("AMLAS_CHAT_MODEL", DEFAULT_AGENT_MODEL))


def chat_max_tokens() -> int:
    """The chat tier's output-token cap.

    Defaults to ``cfg.bedrock.max_tokens`` (64000) — the value live users already
    get on the surviving Strands path — so unifying the two paths does NOT quietly
    re-cap chat to the raw path's hardcoded 2048 (criterion 4.2, design Property 7).
    ``AMLAS_CHAT_MAX_TOKENS`` exists so chat can be capped without touching the
    swarm tier's ``cfg.bedrock.max_tokens``.

    Fail-soft on a bad override: a non-integer, zero or negative
    ``AMLAS_CHAT_MAX_TOKENS`` logs a warning and falls back to the config value
    rather than raising. This accessor is read while an agent is being constructed,
    so a typo'd env var must not take agent construction down — the same discipline
    :mod:`.bedrock_creds` follows (never raise out of the config seam) and the
    opposite of ``int(env_with_compat(...))``, which would propagate a ``ValueError``
    from a stray ``AMLAS_CHAT_MAX_TOKENS=64k``.
    """
    default = load_config().bedrock.max_tokens
    raw = env_with_compat("AMLAS_CHAT_MAX_TOKENS", str(default))
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        logger.warning(
            "Ignoring non-integer AMLAS_CHAT_MAX_TOKENS=%r — using cfg.bedrock.max_tokens=%d",
            raw,
            default,
        )
        return default
    if value <= 0:
        logger.warning(
            "Ignoring non-positive AMLAS_CHAT_MAX_TOKENS=%r — using cfg.bedrock.max_tokens=%d",
            raw,
            default,
        )
        return default
    return value
