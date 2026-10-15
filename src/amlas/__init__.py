"""amlas — multi-agent ML engineering on AgentCore + Strands."""

from __future__ import annotations

from ._runtime_guard import apply_openmp_safety_guard

__version__ = "0.1.0"

# macOS duplicate-libomp OpenMP crash guard (found in  e2e).
# ─────────────────────────────────────────────────────────────────
# Every entrypoint — the CLI (``python -m amlas.cli`` / ``bin/agentcli``), the
# TUI (``python -m amlas.tui`` / ``bin/agenttui``) and the demo scripts — imports
# this package FIRST, before any numpy/scipy/sklearn/xgboost import is triggered
# through ``amlas.*``. On macOS a duplicate ``libomp.dylib`` runtime otherwise
# SIGSEGVs the interpreter the moment an OpenMP-parallel learner (e.g. xgboost)
# enters a parallel region — which crashed ``amlas run train --algorithm
# xgboost`` in the  A4AI e2e. The guard applies the production-safe
# neutralisation at startup: it is macOS-only (Linux/SageMaker throughput
# is untouched) and uses ``setdefault`` (an explicit operator ``OMP_NUM_THREADS``
# always wins), so it is fully behaviour-preserving. See ``_runtime_guard`` for
# the proven root-cause analysis.
apply_openmp_safety_guard()


def _load_dotenv() -> None:
    """Load the SHARED-BASE root ``.env`` into ``os.environ`` at package import.

    CENTRALIZED-ENV (project standard): every entrypoint — the TUI
    (``python -m amlas.tui`` / ``bin/agenttui``) and the CLI
    (``python -m amlas.cli`` / ``bin/agentcli``) — imports this
    package first, so honoring the root ``.env`` here means ONE gitignored file
    configures the whole app: the chat tier's AWS profile/region (which decide
    whether its Strands agent can reach Bedrock at all), plus the lake / model-group /
    AgentCore-runtime / Tavily-secret references the tools resolve at runtime.

    This is the ROOT layer of the layered-config system (see
    ``config/env_loader.py``). The per-entrypoint overlays — ``.env.tui`` (TUI),
    ``.env.agentcore`` (AgentCore/Strands runtime), ``.env.swarm`` (agent swarm)
    — are applied ON TOP of this base by each entrypoint via
    ``load_layered_env(<layer>)``.

    Discipline:
      * Shell wins — a value already exported in the launching shell (e.g.
        ``AWS_PROFILE=<your-profile> uv run ...``) ALWAYS beats every ``.env``.
      * Best-effort — if ``python-dotenv`` is missing or no ``.env`` exists,
        this is a silent no-op; the app still runs on shell env + the layered
        ``.agenticml.yaml`` config.
    """
    try:
        from .config.env_loader import load_root_env
    except Exception:
        return
    try:
        load_root_env()
    except Exception:
        return


_load_dotenv()
