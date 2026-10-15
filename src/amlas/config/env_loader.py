"""Layered ``.env`` loader — the single, DRY source of truth for env config.

THE LAYERS (the user's explicit ask: "a .env for the TUI, one for AgentCore/
Strands, and one for the actual agent swarm"):

    .env             — SHARED BASE: AWS profile/region, lake/model-group/
                       session-bucket/AgentCore-ARN/Tavily-secret-id, model tiers.
                       Loaded for EVERY entrypoint at package import.
    .env.tui         — TUI-only overlay (chat model, UX toggles, chat-backend knobs).
    .env.agentcore   — AgentCore/Strands runtime overlay (runtime ARN, OTEL/STRANDS
                       telemetry toggles, per-role model tiers, boto retry/timeout).
    .env.swarm       — Agent-swarm overlay (per-role models, retreat/budget knobs,
                       memory/KB toggles, steering defaults).

The layer files are THIN OVERLAYS over the shared base — NOT full copies (DRY).

PRECEDENCE (highest wins):
    1. Shell-exported vars   (e.g. ``AWS_PROFILE=default uv run ...``)
    2. Layer file            (.env.tui / .env.agentcore / .env.swarm)
    3. Root .env             (shared base)
    4. .agenticml.yaml       defaults (applied later by ``Settings.load()``)

Mechanism: a snapshot of ``os.environ`` is taken ONCE at module import — before
any ``.env`` file is applied — so a genuine shell export ALWAYS beats every
``.env`` layer. The root layer never clobbers an already-set value (override=
False vs root); a layer overlay DOES override root-supplied values (override=
True) but still never beats the shell snapshot.

Each entrypoint loads root first (done for all at package import) then overlays
its own layer exactly once:
    * TUI            → ``load_layered_env("tui")``       (tui/app.py:run)
    * AgentCore app  → ``load_layered_env("agentcore")`` (app.py entrypoint)
    * Agent build    → ``load_layered_env("agentcore")`` (_build.py, before model)
    * Swarm assembly → ``load_layered_env("swarm")``     (orchestrator.py)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Captured ONCE, at module import, BEFORE any .env file is applied. Anything the
# user exported in the launching shell lives here and must always win.
_SHELL_KEYS: frozenset[str] = frozenset(os.environ)

# Logical layer name → repo-root file name (thin overlays over the shared base).
LAYER_FILES: dict[str, str] = {
    "tui": ".env.tui",
    "agentcore": ".env.agentcore",
    "swarm": ".env.swarm",
}

ROOT_ENV = ".env"

# Track which layers we've already overlaid this process so a re-entrant call
# (e.g. the swarm assembled twice in one TUI session) is a cheap no-op.
_LOADED_LAYERS: set[str] = set()
_ROOT_LOADED = False


def _repo_root() -> Path:
    """Repo root = parents[3] of ``src/amlas/config/env_loader.py``."""
    return Path(__file__).resolve().parents[3]


def _candidate_paths(filename: str) -> list[Path]:
    """Look in the repo root first, then the current working directory."""
    root = _repo_root() / filename
    cwd = Path.cwd() / filename
    # De-dup while preserving order (root wins if both resolve to the same file).
    seen: set[Path] = set()
    out: list[Path] = []
    for p in (root, cwd):
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            out.append(p)
    return out


def _apply_file(filename: str, *, override_root: bool) -> dict[str, str]:
    """Apply the first existing candidate file for ``filename``.

    Args:
        override_root: When True (a LAYER overlay) the file's values replace any
            value previously set by the root ``.env`` — but NEVER a value present
            in the shell snapshot. When False (the ROOT layer) an already-present
            value (shell OR a prior load) is left untouched.

    Returns the {key: value} pairs actually written to ``os.environ`` (for the
    layered-config probe / debugging).
    """
    try:
        from dotenv import dotenv_values
    except Exception:
        return {}

    for path in _candidate_paths(filename):
        if not path.is_file():
            continue
        applied: dict[str, str] = {}
        for key, val in dotenv_values(path).items():
            if val is None:
                continue
            if key in _SHELL_KEYS:
                continue  # shell export always wins — never overridden by a file
            if not override_root and key in os.environ:
                continue  # root layer must not clobber an already-set value
            os.environ[key] = val
            applied[key] = val
        if applied:
            logger.debug(
                "env_loader: applied %d keys from %s (override_root=%s)",
                len(applied),
                path,
                override_root,
            )
        return applied  # only the FIRST existing candidate is applied
    return {}


def _normalize_empty_aws_profiles() -> None:
    """An empty standard profile selector means the default SDK credential chain."""
    for key in ("AWS_PROFILE", "AWS_DEFAULT_PROFILE"):
        if os.environ.get(key) == "":
            os.environ.pop(key, None)


def load_root_env(*, force: bool = False) -> dict[str, str]:
    """Apply the shared-base root ``.env`` (idempotent; shell still wins)."""
    global _ROOT_LOADED
    if _ROOT_LOADED and not force:
        _normalize_empty_aws_profiles()
        return {}
    applied = _apply_file(ROOT_ENV, override_root=False)
    _normalize_empty_aws_profiles()
    _ROOT_LOADED = True
    return applied


def load_layered_env(layer: str | None = None, *, force: bool = False) -> dict[str, str]:
    """Load root ``.env`` then overlay the named layer's file (once per process).

    Args:
        layer: one of ``"tui" | "agentcore" | "swarm"`` (or ``None`` for root only).
        force: re-apply even if already loaded this process (tests / explicit reload).

    Returns the union of keys applied by this call.
    """
    applied: dict[str, str] = {}
    applied.update(load_root_env(force=force))

    if layer:
        if layer not in LAYER_FILES:
            logger.warning("env_loader: unknown layer %r (no overlay applied)", layer)
            return applied
        if layer in _LOADED_LAYERS and not force:
            return applied
        applied.update(_apply_file(LAYER_FILES[layer], override_root=True))
        _LOADED_LAYERS.add(layer)

    _normalize_empty_aws_profiles()
    return applied


def describe_layers() -> dict[str, dict[str, object]]:
    """Report, per layer, the file path, existence, and keys it would contribute.

    Pure inspection (reads files via ``dotenv_values``; does NOT mutate the
    environment) — used by the layered-config verification probe to prove each
    layer is real and read.
    """
    try:
        from dotenv import dotenv_values
    except Exception:
        dotenv_values = None  # type: ignore[assignment]

    def _inspect(filename: str) -> dict[str, object]:
        for path in _candidate_paths(filename):
            if path.is_file():
                keys: list[str] = []
                if dotenv_values is not None:
                    keys = sorted(k for k, v in dotenv_values(path).items() if v is not None)
                return {"path": str(path), "exists": True, "keys": keys}
        return {"path": str(_candidate_paths(filename)[0]), "exists": False, "keys": []}

    report: dict[str, dict[str, object]] = {"root": _inspect(ROOT_ENV)}
    for layer, filename in LAYER_FILES.items():
        report[layer] = _inspect(filename)
    return report
