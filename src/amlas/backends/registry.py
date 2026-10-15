"""``BackendRegistry`` — resolve a port → a concrete adapter at runtime (doc 05 §4.3).

Mirrors the two patterns the team already trusts: ``arbiter/registry.py`` (task-type
→ grader) and ``config/runtime.py`` ``ModelConfig.for_role`` (role → BedrockModel).
ADR-005.

Resolution precedence (doc 05 §4.3 / §11 Q12 — strict, traced):
    1. **Hard steering pin** — a ``SteeringProfile`` hard constraint that pins a
       backend for this port WINS over everything; if the pinned adapter is
       unavailable → **fail fast** (mirrors ``cloud/dispatch`` SEC-01).
    2. **Explicit env ``*_BACKEND``** — an operator decision; honor it (bind-or-fail).
    3. **Auto** — scale + budget + soft prefs choose; defaults lean **local for
       small/offline** (cheap, deterministic) and **the AWS/SageMaker adapter for scale**.

Every resolution records ``port → adapter → reason`` (returned on the
:class:`ResolvedBackend`) for the traceability matrix (doc 05 §8.1).

Offline default (roadmap M0 + doc 05 §9.3 M2): when no env/steering selects a
backend and an explicit offline flag is set, ``auto`` resolves to the **local**
adapter and never touches AWS. Otherwise, ``auto`` leans to the **default** (AWS)
adapter.

Adapters self-register via :func:`register_adapter` (called from each
``backends/<port>/`` module). The registry lazily imports the adapters package the
first time it resolves, so importing ``backends.registry`` alone is cheap and
boto3-free.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from amlas.config._env_compat import env_with_compat

from .ports import port_key

logger = logging.getLogger(__name__)

__all__ = [
    "REGISTRY",
    "AdapterUnavailableError",
    "AwsFirstPosture",
    "BackendRegistry",
    "ResolvedBackend",
    "affirm_aws_first",
    "register_adapter",
]


class AdapterUnavailableError(RuntimeError):
    """Raised when a requested adapter is unknown or fails its capability check.

    A hard-steering pin or an explicit ``*_BACKEND`` that cannot be satisfied is a
    fail-fast condition (doc 05 §4.3), not a silent fallback.
    """


# An adapter factory takes no required args and returns the adapter instance. The
# factory is lazy so importing the registry never constructs boto3 clients.
AdapterFactory = Callable[[], Any]


@dataclass(frozen=True)
class _AdapterEntry:
    """A registered adapter: its selector name, factory, and role flags."""

    name: str
    factory: AdapterFactory
    is_default: bool = False  # the AWS/SageMaker (prod) default for the port
    is_local: bool = False  # the offline/in-process adapter (small data)


@dataclass(frozen=True)
class ResolvedBackend:
    """The result of a resolution: the bound adapter + the traced reason.

    ``adapter`` is the live instance; ``port``/``adapter_name``/``reason`` are the
    ``port → adapter (+ reason)`` triple stamped into the trace (doc 05 §8.1).
    """

    adapter: Any
    port: str
    adapter_name: str
    reason: str


@dataclass(frozen=True)
class AwsFirstPosture:
    """The result of :func:`affirm_aws_first` — the TUI AWS-first invariant, made explicit.

    User directive (2026-07-12): *"Ensure everything our TUI does kicks off via the
    AWS account by default, not locally."* Historically this held only IMPLICITLY —
    ``_auto_select`` leans to the AWS ``is_default`` adapter whenever
    :func:`_running_offline` is ``False`` (no explicit offline flag).
    This posture object turns that side effect into a directly-assertable check:

    - ``offline`` — whether the process is running with an explicit offline flag.
      When ``True`` the affirmation is an explicit **no-op** that stays LOCAL.
    - ``aws_first`` — ``True`` only when NOT offline AND every representative port's
      auto-selection resolved to its AWS (SageMaker/Clarify/S3) ``is_default`` adapter.
    - ``checked`` — ``{port_name: selected_adapter_name}`` for each probed port, so a
      caller can log/inspect exactly what the auto path resolves to (verification only).
    - ``reason`` — a short human phrase for the startup log line.

    It is a pure inspection of the adapter table via ``_auto_select`` — it does NOT
    construct any adapter/boto3 client and NEVER mutates ``os.environ`` or pins a
    selector (the auto path is left to decide; this only affirms/verifies it).
    """

    offline: bool
    aws_first: bool
    checked: dict[str, str]
    reason: str


# Per-port adapter tables: ``{port_name: {adapter_name: _AdapterEntry}}``.
_ADAPTERS: dict[str, dict[str, _AdapterEntry]] = {}


def register_adapter(
    port: type,
    name: str,
    factory: AdapterFactory,
    *,
    is_default: bool = False,
    is_local: bool = False,
) -> None:
    """Register an adapter ``factory`` for ``port`` under selector ``name``.

    Called from each ``backends/<port>/`` module at import. ``is_default`` marks
    the prod (AWS/SageMaker) adapter; ``is_local`` marks the offline/in-process
    one. Re-registering the same ``(port, name)`` overwrites, making registration
    idempotent across repeated imports.
    """
    pname = getattr(port, "__name__", str(port))
    # Validate the port is one of the known ones (fail-fast on a typo).
    port_key(port)
    _ADAPTERS.setdefault(pname, {})[name] = _AdapterEntry(
        name=name, factory=factory, is_default=is_default, is_local=is_local
    )


def _running_offline() -> bool:
    """True when the auto path should prefer the local adapter.

    Offline mode is enabled only by an explicit
    ``AMLAS_OFFLINE``/``AMLAS_BACKENDS_OFFLINE`` flag.
    """
    flag = env_with_compat("AMLAS_BACKENDS_OFFLINE") or env_with_compat("AMLAS_OFFLINE")
    return str(flag).strip().lower() in ("1", "true", "yes")


def _steering_pin_for(steering: Any, port: type) -> str | None:
    """Extract a hard backend pin for ``port`` from a steering profile, if any.

    The ``SteeringProfile`` (``data_eng/steering.py``) carries ``pin_methods``;
    a key of the form ``"<port_key_lower>_backend"`` (e.g. ``"training_backend"``)
    or the bare env key (``"TRAINING_BACKEND"``) hard-pins the adapter. Returns the
    pinned selector name or ``None``. Defensive: any steering shape that lacks
    ``pin_methods`` yields ``None`` (no pin).
    """
    if steering is None:
        return None
    pins = getattr(steering, "pin_methods", None)
    if not isinstance(pins, dict):
        return None
    key = port_key(port)  # e.g. TRAINING_BACKEND
    for candidate in (key, key.lower(), key.lower().replace("_backend", "") + "_backend"):
        if candidate in pins:
            return str(pins[candidate]).strip().lower()
    return None


def _capture_backend_error(error: Exception, *, port: str, adapter: str) -> None:
    """Funnel a backend adapter-construction error into the centralized ERROR_LOG.

    Tagged ``source="backends.<port>.<adapter>"`` so every adapter failure funnels into
    the process-wide error surface from the single ``_bind`` chokepoint. Additive +
    guarded: it never changes resolution behavior and can never itself raise (a lazy
    import keeps the registry boto3-free/import-light).
    """
    try:
        from amlas.telemetry.error_log import capture_error

        capture_error(error, source=f"backends.{port}.{adapter}")
    except Exception:  # nosec B110
        pass


class BackendRegistry:
    """Resolve ``port → adapter`` by hard-pin → env → auto (doc 05 §4.3; ADR-005)."""

    def resolve(
        self,
        port: type,
        *,
        steering: Any | None = None,
        scale: int | str | None = None,
        budget: Any | None = None,
    ) -> ResolvedBackend:
        """Bind ``port`` to a concrete adapter, recording the reason.

        Args:
            port: One of the 25 ports (a ``Protocol`` from ``ports.py``).
            steering: Optional ``SteeringProfile`` — a hard pin wins over all.
            scale: Optional data-scale hint (row count or ``"small"``/``"large"``)
                feeding the auto path.
            budget: Optional compute-budget object; low budget biases auto → local.

        Returns:
            A :class:`ResolvedBackend` (the bound adapter + ``port→adapter→reason``).

        Raises:
            AdapterUnavailableError: a hard pin or explicit env names an adapter
                that is not registered / fails its capability check (fail-fast).
            KeyError: ``port`` is not one of the 19 registered ports.
        """
        pname = getattr(port, "__name__", str(port))
        key = port_key(port)  # validates the port; raises KeyError on a typo
        self._ensure_adapters_loaded()
        table = _ADAPTERS.get(pname, {})

        # 1) hard steering pin — wins over everything, fail-fast if unavailable.
        pin = _steering_pin_for(steering, port)
        if pin:
            entry = table.get(pin)
            if entry is None:
                raise AdapterUnavailableError(
                    f"hard-steering pin {key}={pin!r} has no registered adapter for {pname}; "
                    f"available: {sorted(table)}"
                )
            return self._bind(entry, port=pname, reason=f"hard-steering pin {key}={pin}")

        # 2) explicit env *_BACKEND — operator decision, fail-fast if unavailable.
        env_val = os.environ.get(key)
        if env_val:
            sel = env_val.strip().lower()
            entry = table.get(sel)
            if entry is None:
                raise AdapterUnavailableError(
                    f"env {key}={env_val!r} selects an unknown adapter for {pname}; "
                    f"available: {sorted(table)}"
                )
            return self._bind(entry, port=pname, reason=f"explicit env {key}={sel}")

        # 3) auto — scale + budget + offline → local; else the AWS default.
        entry, reason = self._auto_select(pname, table, scale=scale, budget=budget)
        if entry is None:
            raise AdapterUnavailableError(
                f"no adapter registered for {pname} (port {key}); did its backends/"
                f"<port> module import?"
            )
        return self._bind(entry, port=pname, reason=reason)

    # ── internals ─────────────────────────────────────────────────────────────
    def _auto_select(
        self,
        pname: str,
        table: dict[str, _AdapterEntry],
        *,
        scale: int | str | None,
        budget: Any | None,
    ) -> tuple[_AdapterEntry | None, str]:
        """Pick an adapter when nothing pinned: local for small/offline, else default."""
        local = next((e for e in table.values() if e.is_local), None)  # nosemgrep
        default = next((e for e in table.values() if e.is_default), None)  # nosemgrep

        prefer_local = _running_offline() or self._is_small(scale) or self._is_low_budget(budget)

        if prefer_local and local is not None:
            why = "offline" if _running_offline() else "small-scale/low-budget"
            return local, f"auto → local ({why})"
        if default is not None:
            return default, "auto → default (AWS) adapter"
        # No default registered (e.g. a stubbed future-milestone port) — fall back
        # to local if present, else the single registered adapter, else nothing.
        if local is not None:
            return local, "auto → local (no default adapter registered)"
        only = next(iter(table.values()), None)
        return only, ("auto → sole registered adapter" if only else "no adapter")

    @staticmethod
    def _is_small(scale: int | str | None, *, threshold: int = 100_000) -> bool:
        """True when the data-scale hint indicates small data (rows < threshold)."""
        if scale is None:
            return False
        if isinstance(scale, str):
            return scale.strip().lower() in ("small", "tiny", "local")
        try:
            return int(scale) < threshold
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _is_low_budget(budget: Any | None) -> bool:
        """True when a compute budget signals 'prefer cheap/local' (best-effort)."""
        if budget is None:
            return False
        # Accept a dict {"prefer": "local"} or an object with .prefer_local / .local.
        if isinstance(budget, dict):
            return str(budget.get("prefer", "")).lower() == "local" or bool(budget.get("local"))
        return bool(getattr(budget, "prefer_local", False)) or bool(getattr(budget, "local", False))

    def _bind(self, entry: _AdapterEntry, *, port: str, reason: str) -> ResolvedBackend:
        """Instantiate the adapter via its factory + a capability check (fail-fast)."""
        try:
            adapter = entry.factory()
        except Exception as e:
            # Universal error handling: capture the adapter-construction
            # failure into the centralized ERROR_LOG at this single _bind chokepoint,
            # tagged ``source="backends.<port>.<adapter>"`` — so every backend adapter
            # error funnels in one place without touching each adapter. Additive +
            # guarded; the fail-fast AdapterUnavailableError is still raised unchanged.
            _capture_backend_error(e, port=port, adapter=entry.name)
            raise AdapterUnavailableError(
                f"adapter {entry.name!r} for {port} failed to construct: {e}"
            ) from e
        logger.debug("BackendRegistry: %s → %s (%s)", port, entry.name, reason)
        return ResolvedBackend(adapter=adapter, port=port, adapter_name=entry.name, reason=reason)

    def _ensure_adapters_loaded(self) -> None:
        """Lazily import the adapters package so its modules self-register.

        Cheap + boto3-free until first ``resolve`` — the adapter modules import
        boto3 lazily inside their methods (the existing ``make_*_client`` pattern).
        """
        if getattr(self, "_loaded", False):
            return
        try:
            from . import adapters  # noqa: F401 — import triggers self-registration
        except Exception as e:
            logger.warning("BackendRegistry: adapter package import failed: %s", e)
        self._loaded = True


# The process-wide singleton the tools call: ``REGISTRY.resolve(TrainingBackend, …)``.
REGISTRY = BackendRegistry()


# Representative ports whose PROD DEFAULT is a genuine AWS adapter (SageMaker /
# Clarify / S3), used by :func:`affirm_aws_first` to make the AWS-first invariant
# directly assertable. These mirror ``DEFAULT_BACKEND_SELECTORS`` in
# ``config/runtime.py`` (SageMaker for training/pipeline/monitoring/registry,
# Clarify for explainability, S3-medallion for the lake) — the exact
# SageMaker/Clarify/S3 posture the directive names. Each of these ports registers a
# distinct ``is_default`` (AWS) adapter AND an ``is_local`` one, so "resolves to AWS,
# not local" is a meaningful assertion (ports whose default IS also local — e.g.
# Evaluation's Arbiter, ExperimentTracking's local leaderboard — are deliberately
# excluded because they can't distinguish the two).
_AWS_FIRST_CHECK_PORTS: tuple[str, ...] = (
    "TrainingBackend",  # sagemaker
    "ExplainabilityBackend",  # clarify
    "LakeBackend",  # s3medallion
    "PipelineBackend",  # sagemaker
    "MonitoringBackend",  # sagemaker
    "ModelRegistryBackend",  # sagemaker
)


def affirm_aws_first(*, registry: BackendRegistry | None = None) -> AwsFirstPosture:
    """Affirm the TUI's AWS-first-by-default backend posture (user directive 2026-07-12).

    Makes the previously implicit invariant explicit: for each
    representative AWS-default port (:data:`_AWS_FIRST_CHECK_PORTS`), it asks the
    ``auto`` path (``_auto_select`` with no scale/budget hint) which adapter it would
    pick and records it. When NOT running hermetic-offline, every one must resolve to
    its AWS (SageMaker/Clarify/S3) ``is_default`` adapter → ``aws_first=True``.

    In explicit offline mode, the affirmation is a no-op that reports
    ``offline=True, aws_first=False`` and leaves the auto path resolving LOCAL.
    This function is verification-only: it inspects the adapter table via
    ``_auto_select`` (no adapter/boto3 client is constructed) and NEVER mutates
    ``os.environ`` or pins a ``*_BACKEND`` selector.

    Args:
        registry: Optional registry to inspect (defaults to the process singleton).

    Returns:
        An :class:`AwsFirstPosture` describing what the auto path resolves to.
    """
    reg = registry if registry is not None else REGISTRY
    reg._ensure_adapters_loaded()

    offline = _running_offline()
    checked: dict[str, str] = {}
    all_aws = True
    for pname in _AWS_FIRST_CHECK_PORTS:
        table = _ADAPTERS.get(pname, {})
        if not table:
            # Port's adapter module didn't register (unexpected) — can't affirm it.
            all_aws = False
            continue
        entry, _reason = reg._auto_select(pname, table, scale=None, budget=None)
        if entry is None:
            all_aws = False
            continue
        checked[pname] = entry.name
        if not entry.is_default:  # nosemgrep
            all_aws = False

    if offline:
        # Sacred hermetic path — explicit no-op, stays local.
        posture = AwsFirstPosture(
            offline=True,
            aws_first=False,
            checked=checked,
            reason="offline flag set — AWS-first affirmation is a no-op; auto stays local",
        )
    else:
        posture = AwsFirstPosture(
            offline=False,
            aws_first=all_aws,
            checked=checked,
            reason=(
                "AWS-first affirmed — auto resolves to AWS (SageMaker/Clarify/S3) defaults"
                if all_aws
                else "AWS-first NOT satisfied — an auto port resolved to a non-AWS adapter"
            ),
        )
    logger.debug("affirm_aws_first: %s (%s)", posture.reason, posture.checked)
    return posture
