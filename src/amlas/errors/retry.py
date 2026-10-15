"""Shared retry and backoff policy.

Promotes the quota-aware full-jitter exponential backoff that lived ONLY in
``amlas.tools.code_interpreter`` (FIX-27) into a general, reusable helper so every layer
(code interpreter, cloud dispatch, pollers) shares one battle-tested policy instead of
re-hand-rolling it (or, as the dispatch path did, running with ``max_attempts=1`` = no retry).

Design (unchanged from FIX-27, generalized):
* full-jitter exponential: ``uniform(0, min(cap, base * 2**attempt)]`` — AWS's recommended
  strategy, which spreads concurrent retriers across the whole window so they do not
  re-collide in lockstep on a shared quota;
* jitter is derived from a process counter + PID, NOT a seeded RNG — resume-safe (the repo
  forbids ``Math.random``/``time``-seeded randomness in resumable paths) and still
  de-synchronizes parallel sessions;
* ``sleep`` is injectable so callers can avoid real waiting;
* retryability is decided by :func:`amlas.errors.taxonomy.is_retryable` by default (transient
  category), overridable via ``is_retryable=``;
* env-tunable knobs (shared names) so operators tune all retriers at once.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import TypeVar

from .taxonomy import classify
from .taxonomy import is_retryable as _default_is_retryable

T = TypeVar("T")

# Env-tunable knobs (shared across all retriers).
_MAX_RETRIES_ENV = "AMLAS_RETRY_MAX_ATTEMPTS"
_BACKOFF_BASE_ENV = "AMLAS_RETRY_BACKOFF_BASE_SECONDS"
_BACKOFF_CAP_ENV = "AMLAS_RETRY_BACKOFF_CAP_SECONDS"
_DEFAULT_MAX_RETRIES = 6
_DEFAULT_BACKOFF_BASE = 3.0
_DEFAULT_BACKOFF_CAP = 60.0

# Monotonic jitter counter (no RNG — resume-safe; de-syncs lockstep retriers).
_jitter_counter: int = 0


def _env_int(name: str, default: int) -> int:
    try:
        v = int(os.environ.get(name, "").strip() or default)
        return v if v >= 0 else default
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        v = float(os.environ.get(name, "").strip() or default)
        return v if v > 0 else default
    except ValueError:
        return default


def backoff_delay(attempt: int, base: float, cap: float) -> float:
    """Full-jitter exponential delay for ``attempt`` (0-indexed): ``uniform(0, nominal]``.

    ``nominal = min(cap, base * 2**attempt)``. Jitter from a process counter + PID (not an
    RNG). Always returns a positive delay bounded by ``cap``.
    """
    global _jitter_counter
    nominal: float = min(float(cap), float(base) * float(2 ** max(0, attempt)))
    _jitter_counter += 1
    seed = (_jitter_counter * 2654435761 + os.getpid() * 40503) % 10_000
    frac: float = (seed + 1) / 10_001.0  # in (0, 1)
    return max(0.001, nominal * frac)


def retry_with_backoff(
    fn: Callable[[], T],
    *,
    is_retryable: Callable[[BaseException], bool] | None = None,
    max_attempts: int | None = None,
    base: float | None = None,
    cap: float | None = None,
    sleep: Callable[[float], None] = time.sleep,
    on_retry: Callable[[int, BaseException, float], None] | None = None,
) -> T:
    """Call ``fn`` with full-jitter exponential backoff on RETRYABLE failures.

    Args:
        fn: zero-arg callable to (re)invoke.
        is_retryable: predicate deciding whether an exception warrants a retry; defaults to
            the taxonomy's transient-category classifier
            (:func:`amlas.errors.taxonomy.is_retryable`).
        max_attempts: total attempts (initial + retries). Defaults to ``AMLAS_RETRY_MAX_ATTEMPTS``
            (6). ``max_attempts=1`` disables retries (one shot).
        base / cap: backoff base + ceiling seconds (env-defaulted).
        sleep: injectable wait (callers may pass a no-op → hermetic).
        on_retry: optional ``(attempt, exc, delay)`` hook (e.g. to log / capture the transient).

    Returns:
        ``fn()``'s result on success.

    Raises:
        The last exception when retries are exhausted OR the exception is not retryable — the
        caller still sees the real error (classified upstream via
        :func:`amlas.errors.taxonomy.classify` if desired).
    """
    retryable = is_retryable or _default_is_retryable  # nosemgrep: is-function-without-parentheses
    attempts = max_attempts if max_attempts is not None else _env_int(_MAX_RETRIES_ENV, _DEFAULT_MAX_RETRIES)
    attempts = max(1, attempts)
    b = base if base is not None else _env_float(_BACKOFF_BASE_ENV, _DEFAULT_BACKOFF_BASE)
    c = cap if cap is not None else _env_float(_BACKOFF_CAP_ENV, _DEFAULT_BACKOFF_CAP)

    last_exc: BaseException | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except BaseException as exc:  # re-raised below; we classify to decide (broad by design)
            last_exc = exc
            # Last attempt, or a non-retryable error → give up now (surface the real error).
            if attempt >= attempts - 1 or not retryable(exc):
                raise
            delay = backoff_delay(attempt, b, c)
            if on_retry is not None:
                try:
                    on_retry(attempt, exc, delay)
                except Exception:  # best-effort hook — never break the retry loop  # nosec B110
                    pass
            sleep(delay)
    # Unreachable (the loop either returns or raises), but keeps type-checkers happy.
    assert last_exc is not None  # nosec B101
    raise last_exc


__all__ = ["backoff_delay", "classify", "retry_with_backoff"]
