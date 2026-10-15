"""Process-wide runtime safety guards applied at ``amlas`` package import.

macOS OpenMP / libomp duplicate-runtime crash guard
───────────────────────────────────────────────────
ROOT CAUSE (proven via an A4AI end-to-end run): on macOS TWO ``libomp.dylib``
runtimes get loaded into one interpreter — the Homebrew
``/opt/homebrew/.../libomp.dylib`` AND a wheel-bundled copy shipped alongside
the scipy / scikit-learn / xgboost native extensions. When a native OpenMP
parallel region spins up with MORE THAN ONE thread, the duplicate-runtime
thread-pool barrier init raises ``OMP: Error #15`` / ``__kmp_fatal`` and the
whole interpreter dies with **SIGSEGV (exit 139)** — before Python can raise
anything catchable. Symptom in the product: ``amlas run train --algorithm
xgboost`` (an OpenMP-parallel learner) crashed silently on macOS while
``random_forest`` (no OpenMP region) survived.

The production entrypoints (``python -m amlas.cli`` / ``python -m amlas.tui`` /
``bin/agentcli`` / ``bin/agenttui`` / the demo scripts) all import the ``amlas``
package first, so applying the same guard here — at package import, before any
numpy/sklearn/xgboost import is triggered through ``amlas.*`` — fixes every
product surface with one seam.

Discipline (behaviour-preserving, non-weakening):
  * **macOS-only.** Linux (CI, Docker, SageMaker training containers) ships a
    single libomp and never hits this abort, so the guard is a strict no-op
    there — real multi-core training throughput on the managed backends is
    completely untouched.
  * **``setdefault`` semantics.** An operator who explicitly exports e.g.
    ``OMP_NUM_THREADS=8`` ALWAYS wins; the guard only fills in values that are
    unset. On the tiny local-dev datasets single-threaded sklearn/xgboost is
    numerically identical, just serial.
"""

from __future__ import annotations

import os
import sys
import threading
from typing import Any

# The OpenMP / BLAS thread-pool knobs (+ the documented duplicate-libomp escape
# hatch) that make a duplicate-runtime macOS interpreter safe. Mirrors the subset
# of ``validation coverage`` that specifically neutralises the native OpenMP abort.
OPENMP_GUARD_KEYS: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "KMP_DUPLICATE_LIB_OK",
)

# The value each key is defaulted to. All thread pools pin to a single thread;
# KMP_DUPLICATE_LIB_OK enables the duplicate-runtime escape hatch.
_GUARD_VALUES: dict[str, str] = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "KMP_DUPLICATE_LIB_OK": "TRUE",
}


def apply_openmp_safety_guard(
    *,
    platform: str | None = None,
    environ: dict[str, str] | None = None,
) -> list[str]:
    """Neutralise the macOS duplicate-libomp OpenMP abort (behaviour-preserving).

    On macOS (``platform`` starts with ``"darwin"``) this ``setdefault``s the
    OpenMP / BLAS single-thread knobs + ``KMP_DUPLICATE_LIB_OK`` into ``environ``
    so a duplicate-libomp interpreter cannot SIGSEGV inside a native parallel
    region. On every other platform it is a complete no-op.

    Must run BEFORE numpy/scipy/sklearn/xgboost are first imported to take
    effect (libomp reads these at load time) — hence it is invoked at ``amlas``
    package import.

    Args:
        platform: platform string to branch on; defaults to ``sys.platform``.
            Injectable so callers can verify the contract on any host.
        environ: the environment mapping to mutate; defaults to ``os.environ``.

    Returns:
        The list of keys this call actually set (empty on non-macOS or when the
        operator has already exported every key).
    """
    plat = platform if platform is not None else sys.platform
    env = environ if environ is not None else os.environ

    # macOS-only: Linux/Windows have a single libomp and must keep full threading.
    if not plat.startswith("darwin"):
        return []

    changed: list[str] = []
    for key in OPENMP_GUARD_KEYS:
        if key not in env:
            env[key] = _GUARD_VALUES[key]
            changed.append(key)
    return changed


# ── process-level uncaught-exception funnel ───────────────────────────────────
# Guards against double-install: the CLI, TUI and AgentCore entrypoints all call
# ``install_error_excepthook`` (each imports ``amlas`` first), and only the FIRST
# install wins so we never chain our own hook onto itself.
_ERROR_EXCEPTHOOK_INSTALLED = False


def install_error_excepthook() -> bool:
    """Funnel ANY uncaught exception into the centralized ``ERROR_LOG`` (process safety net).

    Installs a ``sys.excepthook`` (main thread) + ``threading.excepthook`` (worker
    threads) that CAPTURE the escaping exception into the process-wide error surface
    (:data:`~amlas.telemetry.error_log.ERROR_LOG`) BEFORE delegating to the previously
    installed hook — so the default traceback print / any pre-existing handler behavior
    is byte-for-byte preserved (we only ADD capture, never change what the user sees).

    This is the last-resort net for the "universal error handling" ask: an error that
    escapes every component boundary in ANY runtime mode still lands in the one JSONL
    error file. Called ONCE per process from each entrypoint (CLI/TUI/AgentCore); the
    module-level guard makes a second call a no-op.

    Both hooks are individually try-guarded so the safety net can never itself raise
    (a telemetry failure must never mask the original traceback). Returns True when it
    installed the hooks this call, False when they were already installed.
    """
    global _ERROR_EXCEPTHOOK_INSTALLED
    if _ERROR_EXCEPTHOOK_INSTALLED:
        return False

    prior_sys_hook = sys.excepthook
    prior_thread_hook = threading.excepthook

    def _sys_excepthook(exc_type: type[BaseException], exc: BaseException, tb: Any) -> None:
        try:
            from .telemetry.error_log import capture_error

            if exc.__traceback__ is None and tb is not None:
                exc.__traceback__ = tb
            capture_error(exc, source="process.uncaught")
        except Exception:  # nosec B110
            pass
        try:
            prior_sys_hook(exc_type, exc, tb)
        except Exception:  # nosec B110
            pass

    def _thread_excepthook(args: Any) -> None:
        try:
            from .telemetry.error_log import capture_error

            exc = getattr(args, "exc_value", None)
            if exc is not None:
                tb = getattr(args, "exc_traceback", None)
                if exc.__traceback__ is None and tb is not None:
                    exc.__traceback__ = tb
                capture_error(exc, source="process.thread")
        except Exception:  # nosec B110
            pass
        try:
            prior_thread_hook(args)
        except Exception:  # nosec B110
            pass

    sys.excepthook = _sys_excepthook
    threading.excepthook = _thread_excepthook
    _ERROR_EXCEPTHOOK_INSTALLED = True
    return True
