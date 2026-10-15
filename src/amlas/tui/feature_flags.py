"""A small, reusable feature-flag system for the TUI.

The delivered/scoped Agentic ML surface ships with ONLY the ``chat`` + ``dashboard``
tabs enabled; every other tab (specs, pipeline, results, data, steering, mlops,
lake) is a developer-surface tab that is *disabled by default* and gated behind a
feature flag. This module is the single source of truth for that gating so
``app.py`` never hard-codes the tab list.

Env contract (mirrors the ``_env_truthy`` token convention already used across
``app.py`` / ``registry.py`` — ``1/true/yes/on`` truthy, ``0/false/no/off``
falsy, anything else → the caller's default):

* ``AMLAS_TUI_ENABLE_ALL_TABS`` — master override. Truthy ⇒ every known tab is
  enabled (the full optional surface).
* ``AMLAS_TUI_TAB_<ID>`` — per-tab override (``<ID>`` upper-cased, e.g.
  ``AMLAS_TUI_TAB_SPECS``). Takes precedence over BOTH the default set and the
  master flag, so an operator can force a single tab on (without ENABLE_ALL) or
  force one off (even under ENABLE_ALL).
* ``AMLAS_TUI_FF_<NAME>`` — the generic single-feature flag read by
  :func:`is_enabled` for non-tab features.

Resolution precedence for a tab (highest first):
    1. ``AMLAS_TUI_TAB_<ID>`` (explicit per-tab)
    2. ``AMLAS_TUI_ENABLE_ALL_TABS`` (master ⇒ on)
    3. the shipped default set (:data:`DEFAULT_ENABLED_TABS`)

Pure/read-only: this module never mutates the environment. Everything is read at
call time so callers can change the environment and observe the result without any
import-order coupling.
"""

from __future__ import annotations

import os

__all__ = [
    "ALL_TAB_IDS",
    "DEFAULT_ENABLED_TABS",
    "enabled_tab_ids",
    "is_enabled",
    "is_tab_enabled",
]

# Truthy / falsy tokens (mirrors ``app._TRUTHY_TOKENS`` / ``registry.py`` semantics).
_TRUTHY_TOKENS = ("1", "true", "yes", "on")
_FALSY_TOKENS = ("0", "false", "no", "off")

# The canonical, ordered list of every tab the app can compose. Order matches the
# ``TabbedContent`` compose order in ``app.py`` (chat first = the initial tab).
ALL_TAB_IDS: tuple[str, ...] = (
    "chat",
    "dashboard",
    "specs",
    "pipeline",
    "results",
    "data",
    "steering",
    "mlops",
    "lake",
)

# The delivered/scoped default: ONLY chat + dashboard are enabled out of the box
# Everything else is opt-in via a feature flag.
DEFAULT_ENABLED_TABS: frozenset[str] = frozenset({"chat", "dashboard"})

# Master + per-tab env var names.
_ENABLE_ALL_ENV = "AMLAS_TUI_ENABLE_ALL_TABS"


def _token_state(value: str | None) -> bool | None:
    """Map an env token to ``True``/``False``/``None`` (unset or unrecognized).

    ``None`` means "no opinion" — the caller should fall through to its default.
    """
    if value is None:
        return None
    token = value.strip().lower()
    if token in _TRUTHY_TOKENS:
        return True
    if token in _FALSY_TOKENS:
        return False
    return None


def is_enabled(name: str, *, default: bool = False) -> bool:
    """Return whether a generic named feature is enabled.

    Reads ``AMLAS_TUI_FF_<NAME>`` (``<NAME>`` upper-cased). A truthy/falsy token
    wins; an unset or unrecognized value returns ``default``.
    """
    state = _token_state(os.environ.get(f"AMLAS_TUI_FF_{name.upper()}"))
    return default if state is None else state


def _enable_all_tabs() -> bool:
    """Whether the master ``AMLAS_TUI_ENABLE_ALL_TABS`` override is truthy."""
    return _token_state(os.environ.get(_ENABLE_ALL_ENV)) is True


def is_tab_enabled(tab_id: str) -> bool:
    """Return whether the given tab is enabled under the current env.

    Precedence: per-tab override → master ENABLE_ALL → shipped default set.
    An unknown ``tab_id`` follows the same rules (default-off unless a flag
    enables it), so a caller can never accidentally surface an unlisted tab.
    """
    # 1. Explicit per-tab override always wins.
    per_tab = _token_state(os.environ.get(f"AMLAS_TUI_TAB_{tab_id.upper()}"))
    if per_tab is not None:
        return per_tab
    # 2. Master override enables everything.
    if _enable_all_tabs():
        return True
    # 3. Fall back to the shipped default set.
    return tab_id in DEFAULT_ENABLED_TABS


def enabled_tab_ids(all_ids: tuple[str, ...] | None = None) -> list[str]:
    """Return the enabled tab ids, preserving the order of ``all_ids``.

    Defaults to :data:`ALL_TAB_IDS` (the app's canonical compose order) so
    ``app.compose`` can iterate it directly.
    """
    ids = all_ids if all_ids is not None else ALL_TAB_IDS
    return [tid for tid in ids if is_tab_enabled(tid)]
