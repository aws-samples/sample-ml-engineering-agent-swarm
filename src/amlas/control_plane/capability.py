"""One-capability-across-surfaces dispatch — the ADR-010 single funnel (roadmap M7).

This module is the **shared capability-dispatch layer**: the SINGLE place every
surface (Chat / TUI / CLI / Python-API) funnels through to invoke a registered flow.
It realizes ADR-010 ("one capability exposed identically across all surfaces") and
ADR-013 ("conversational/terminal-first; surfaces are renderers, never
reimplementations"): no capability logic is duplicated per surface — each surface is a
thin adapter that builds a params dict + an :class:`ApprovalPolicy` and calls
:func:`invoke_capability`.

The funnel composes the EXISTING M0-M6 control plane + flow registry without rebuilding
any flow logic:

  * the **flow registry** (``flows/__init__.py`` — ``FLOWS``/``get_flow``) is the source
    of truth for which capabilities exist and their spend/irreversibility disposition;
  * each registered flow's ``run_<flow_id>`` callable is the executor (resolved lazily
    by convention or an explicit override table);
  * **gated** capabilities (spend-incurring or irreversible) route through
    ``plan_approve_execute`` (T-CTL-04 → T-CTL-03; the plan-preview → approval-gate →
    execute beat); **cheap-read** capabilities stream straight through (never prompt).

Because the gate decision and execution live HERE (not in any surface), the SAME flow id
+ params + policy yields an equivalent result regardless of which surface called it —
that is the ADR-010 parity guarantee the M7 tests assert.

Pure-Python, boto3-free: the registry/ports/flows do the actual AWS work; this layer only
resolves + gates + dispatches.
"""

from __future__ import annotations

import importlib
import inspect
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# R-B-004 / AMLAS-SEC-B-004 (CWE-706, semgrep non-literal-import): the convention path below
# resolves an executor via ``importlib.import_module(f"amlas.flows.{fid}")``. ``fid``
# is normally a registry-controlled flow id, but to make the trust boundary explicit (and to
# foreclose any path where a crafted id could reach this call) we constrain ``fid`` to a single
# safe module-segment — ``^[a-z][a-z0-9_]*$`` — BEFORE importing. This rejects dots (package
# traversal), dashes, slashes and any relative/absolute import trickery; only a flat, lower-snake
# module name under the fixed ``amlas.flows`` package can ever be imported.
_SAFE_FLOW_MODULE = re.compile(r"^[a-z][a-z0-9_]*$")

from ..flows import Flow, get_flow, list_flows
from .approvals import ActionClass, ApprovalPolicy
from .plan_mode import plan_approve_execute, preview_flow_plan
from .traceability import BackendHop, record_capability_trace

__all__ = [
    "CapabilityResult",
    "describe_capability",
    "invoke_capability",
    "list_capabilities",
    "register_capability_executor",
    "required_params_for",
    "resolve_executor",
]


# ── executor resolution (flow_id → the run_<flow_id> callable) ────────────────
# Most flows expose ``run_<flow_id>`` in ``flows/<flow_id>.py`` (the M2-M6 convention),
# so resolution is by convention. A few cross-cutting flows live elsewhere or under a
# different name — register those explicitly via ``register_capability_executor`` /
# this override table. The table maps flow_id → (module path, attribute name).
_EXECUTOR_OVERRIDES: dict[str, tuple[str, str]] = {
    # project_init_flow lives in control_plane.project_memory (not flows/).
    "project_init_flow": ("..control_plane.project_memory", "project_init_flow"),
    # deploy_flow is SELF-gating (``run_deploy_flow`` wraps ``plan_approve_execute`` itself). The
    # funnel ALSO gates every gated capability, so resolving ``run_deploy_flow`` would double-gate
    # (the funnel's approving policy approves the outer gate, then the inner gate re-runs with the
    # default safe ASK policy and DENIES). The funnel is the SINGLE authoritative gate (ADR-010:
    # "the gate decision and execution live HERE, not in any surface"), so it resolves the INNER
    # executor ``execute_deploy_flow`` (which does not self-gate) — the surface's approving policy
    # is the only gate. (G3: deploy_flow is the first GATED flow exposed through the surfaces.)
    "deploy_flow": ("..flows.deploy_flow", "execute_deploy_flow"),
    # improve_retrain_flow + data_collection_flow are SELF-gating exactly like deploy_flow
    # (their ``run_<flow_id>`` wraps ``plan_approve_execute(executor=execute_<flow_id>)``). The
    # funnel ALSO gates every gated capability, so resolving the OUTER ``run_*`` wrapper here would
    # DOUBLE-gate: the funnel's approving policy approves the outer gate, then the inner gate re-runs
    # with the default safe-ASK policy and DENIES (``status="denied"`` even though the funnel
    # approved). The funnel is the SINGLE authoritative gate (ADR-010), so — exactly as for
    # deploy_flow — we resolve the INNER executor (which does not self-gate). Discovered live in
    # LIVE-3 Stage-12 (the first time improve_retrain_flow was dispatched through the funnel).
    "improve_retrain_flow": ("..flows.improve_retrain_flow", "execute_improve_retrain_flow"),
    # baseline_train_flow is SELF-gating too (``run_baseline_train_flow`` wraps
    # ``plan_approve_execute(flow_id="baseline_train_flow", executor=execute_baseline_train_flow)``
    # because training is spend-incurring). Same double-gate trap as deploy/improve: the funnel
    # approves the outer gate, then the inner gate re-runs with the default safe-ASK policy and
    # DENIES (``status="denied"``, ``arbiter_verdict=None`` even though the funnel approved + the
    # REAL training Completed). The funnel is the SINGLE authoritative gate (ADR-010), so resolve
    # the INNER executor (which does not self-gate) → train→grade(Arbiter)→log→register runs.
    # Discovered live in LIVE-3 Stage-8 (first dispatch of baseline_train_flow through the funnel).
    "baseline_train_flow": ("..flows.baseline_train_flow", "execute_baseline_train_flow"),
}

# Runtime-registered executors (tests / cross-cutting flows can inject a callable).
_EXECUTOR_REGISTRY: dict[str, Callable[..., Any]] = {}

# PERF-16: memoize the RESOLVED executor per flow_id. ``resolve_executor`` otherwise
# re-runs the allow-list regex + dict lookups (and, on the convention path, an
# ``importlib.import_module``) on EVERY dispatch. The import is already cached in
# ``sys.modules``, but the per-call validation + getattr is pure overhead once the
# (deterministic) resolution is known. We cache only SUCCESSFUL resolutions (a callable)
# — never a ``None`` — so an unknown id stays re-resolvable after a later registration.
# ``register_capability_executor`` invalidates the cache (last-write-wins preserved).
_EXECUTOR_CACHE: dict[str, Callable[..., Any]] = {}


def _clear_executor_cache() -> None:
    """Reset the PERF-16 executor memo (used by tests and on (re)registration)."""
    _EXECUTOR_CACHE.clear()


def register_capability_executor(flow_id: str, executor: Callable[..., Any]) -> None:
    """Register an explicit executor callable for ``flow_id`` (last write wins).

    Lets cross-cutting flows (or tests) bind a flow id to a callable that does not
    follow the ``run_<flow_id>`` convention. Surfaces never see this — they only ever
    call :func:`invoke_capability`.

    PERF-16: busts the resolution memo so a (re)registered executor is resolved fresh
    (never a stale cached callable) — last-write-wins is preserved.
    """
    _EXECUTOR_REGISTRY[(flow_id or "").strip().lower()] = executor
    _clear_executor_cache()


def resolve_executor(flow_id: str) -> Callable[..., Any] | None:
    """Resolve the executor callable for ``flow_id`` (registry → override → convention).

    Resolution order:
      1. an executor registered via :func:`register_capability_executor`;
      2. an entry in ``_EXECUTOR_OVERRIDES`` (module + attr);
      3. the convention ``amlas.flows.<flow_id>:run_<flow_id>``.

    Returns ``None`` when nothing resolves (an unknown / non-runnable capability).
    """
    fid = (flow_id or "").strip().lower()
    # PERF-16: a previously-resolved executor for this id is memoized — skip the
    # allow-list regex + override/convention lookup (resolution is deterministic per
    # id; ``register_capability_executor`` busts this cache so last-write-wins holds).
    cached = _EXECUTOR_CACHE.get(fid)
    if cached is not None:
        return cached

    resolved = _resolve_executor_uncached(fid)
    # Cache only a SUCCESSFUL resolution (never None) so an unknown id stays
    # re-resolvable after a later registration.
    if resolved is not None:
        _EXECUTOR_CACHE[fid] = resolved
    return resolved


def _resolve_executor_uncached(fid: str) -> Callable[..., Any] | None:
    """The actual (uncached) executor resolution — registry → override → convention."""
    if fid in _EXECUTOR_REGISTRY:
        return _EXECUTOR_REGISTRY[fid]
    if fid in _EXECUTOR_OVERRIDES:
        mod_path, attr = _EXECUTOR_OVERRIDES[fid]
        try:
            mod = importlib.import_module(mod_path, package=__package__)  # nosemgrep
            return getattr(mod, attr, None)
        except Exception:
            return None
    # convention: flows.<flow_id>:run_<flow_id>
    # R-B-004: constrain the dynamic import to a single safe module-segment under the fixed
    # ``amlas.flows`` package. A crafted id with dots/dashes/traversal cannot reach
    # importlib (returns None — an unknown capability) — the non-literal-import trust boundary.
    if not _SAFE_FLOW_MODULE.match(fid):
        return None
    try:
        mod = importlib.import_module(f"amlas.flows.{fid}")  # nosemgrep
    except Exception:
        return None
    return getattr(mod, f"run_{fid}", None)


def _is_gated(flow: Flow) -> bool:
    """A capability is gated when its flow descriptor is spend-incurring or irreversible.

    This mirrors the Plan-Mode disposition: cheap-read flows (both flags False) stream
    through; spend/irreversible flows route through ``plan_approve_execute``.
    """
    return bool(flow.spend_incurring or flow.irreversible)


@dataclass
class CapabilityResult:
    """The surface-neutral result of one capability invocation (the ADR-010 contract).

    Every surface receives this SAME shape, so Chat/TUI/CLI/Python-API render identical
    outcomes. ``result`` carries the flow's native return (a flow ``*Result`` dataclass
    or dict) untouched.

    Args:
        capability_id: The dispatched flow id.
        surface: Which surface adapter invoked it (``"api"``/``"cli"``/``"chat"``/``"tui"``/``"headless"``).
        status: ``"completed"`` | ``"denied"`` | ``"planned"`` | ``"error"``.
        approved: Whether the approval gate approved (True for un-gated cheap reads).
        gated: Whether the capability was routed through the approval gate.
        executed: Whether the flow executor actually ran.
        plan: The read-only plan preview (steps + cost + flags).
        decision: The approval decision dict (None for un-gated reads).
        result: The flow's native return value (or None when denied / errored).
        error: An error message when ``status == "error"``.
    """

    capability_id: str
    surface: str = "api"
    status: str = "completed"
    approved: bool = True
    gated: bool = False
    executed: bool = False
    plan: dict[str, Any] = field(default_factory=dict)
    decision: dict[str, Any] | None = None
    result: Any = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict (for CLI/headless JSON output)."""
        return {
            "capability_id": self.capability_id,
            "surface": self.surface,
            "status": self.status,
            "approved": self.approved,
            "gated": self.gated,
            "executed": self.executed,
            "plan": self.plan,
            "decision": self.decision,
            "result": _jsonable(self.result),
            "error": self.error,
        }


def _jsonable(obj: Any) -> Any:
    """Best-effort convert a flow ``*Result`` dataclass / object into a JSON-able dict."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    # dataclass / object with __dict__
    if hasattr(obj, "__dict__"):
        return {k: _jsonable(v) for k, v in vars(obj).items()}
    return str(obj)


def _extract_run_id(result: Any) -> str:
    """Best-effort pull a run/experiment id out of a flow's native result (for the trace)."""
    if result is None:
        return ""
    src: dict[str, Any] = result if isinstance(result, dict) else getattr(result, "__dict__", {})
    for key in ("run_id", "experiment_id", "execution_id", "job_id", "endpoint", "pipeline_id"):
        val = src.get(key)
        if val:
            return str(val)
    return ""


def _capture_flow_error(error: Exception, *, flow_id: str, surface: str) -> None:
    """Funnel a flow-dispatch error into the centralized ERROR_LOG.

    The single place the capability funnel routes an executor/gate failure into the
    process-wide error surface, tagged ``source="flows.<flow_id>"`` with the calling
    surface in ``context``. Additive + guarded: it never changes the dispatch outcome
    and can never itself raise (a telemetry failure must not break a capability call).
    """
    try:
        from ..telemetry.error_log import capture_error

        capture_error(error, source=f"flows.{flow_id}", context={"surface": surface})
    except Exception:  # nosec B110
        pass


def _emit_capability_trace(
    *, flow: Flow, params: dict[str, Any], surface: str, status: str, result: Any
) -> None:
    """Record the §8.5 story→flow→tool→backend→run lineage for one invocation (boto3-free).

    Aggregates the flow's declared ``backend_port`` into a single :class:`BackendHop` (the M0
    registry resolution-trace is per-port; the capability funnel records the end-to-end chain).
    ``story_id`` comes from ``params['story_id']`` when a surface supplies it (else ``"-"``). Any
    failure here is swallowed — traceability must never break a capability invocation.
    """
    try:
        story_id = str(params.get("story_id") or params.get("story") or "-")
        port = getattr(flow, "backend_port", "") or ""
        backends = [BackendHop(port=str(port), adapter="", reason="capability-funnel")] if port else []
        record_capability_trace(
            story_id=story_id,
            flow_id=flow.flow_id,
            tool_ids=[t for t in (getattr(flow, "tool_ids", None) or []) if t],
            backends=backends,
            run_id=_extract_run_id(result),
            status=status,
            detail={"surface": surface, "milestone": getattr(flow, "milestone", "")},
        )
    except Exception:  # nosec B110
        pass


def list_capabilities() -> list[dict[str, Any]]:
    """Return all registered capabilities as surface-neutral dicts (for menus/panes).

    Each entry: ``{capability_id, description, backend_port, spend_incurring,
    irreversible, cross_cutting, milestone, gated, runnable}``. Surfaces (a TUI pane, a
    CLI ``capability list``, the chat ``/flows`` command) render this WITHOUT importing
    the flow modules — the registry is the single source of truth.
    """
    out: list[dict[str, Any]] = []
    for flow in list_flows():
        out.append(
            {
                "capability_id": flow.flow_id,
                "description": flow.description,
                "backend_port": flow.backend_port,
                "spend_incurring": flow.spend_incurring,
                "irreversible": flow.irreversible,
                "cross_cutting": flow.cross_cutting,
                "milestone": flow.milestone,
                "gated": _is_gated(flow),
                "runnable": resolve_executor(flow.flow_id) is not None,
            }
        )
    return sorted(out, key=lambda d: d["capability_id"])


def required_params_for(capability_id: str) -> tuple[list[str], list[str]]:
    """Return ``(required, optional)`` param names for a capability's executor.

    Derived from the executor's SIGNATURE, so it can never drift from the flows: a
    keyword-only param with no default is required; everything else (defaults,
    ``*args``/``**kwargs``, and the internal ``steering``/``proposer``-style injection
    hooks) is optional. Returns ``([], [])`` when the capability or its executor cannot
    be resolved, so callers can treat "unknown" as "nothing to validate" rather than
    having to special-case it.

    Used by the ``/flow`` chat path to reject a missing required param with a friendly,
    copy-pasteable message instead of letting the raw ``TypeError`` from the call
    signature reach the user (the reported bug), and to print usage for discoverability.
    """
    executor = resolve_executor(capability_id)
    if executor is None:
        return ([], [])
    try:
        sig = inspect.signature(executor)
    except (TypeError, ValueError):  # pragma: no cover - builtins/C callables
        return ([], [])
    required: list[str] = []
    optional: list[str] = []
    for name, p in sig.parameters.items():
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if name == "self":
            continue
        if p.default is inspect.Parameter.empty:
            required.append(name)
        else:
            optional.append(name)
    return (required, optional)


def describe_capability(capability_id: str) -> dict[str, Any] | None:
    """Return the single capability descriptor for ``capability_id`` (or ``None``)."""
    flow = get_flow(capability_id)
    if flow is None:
        return None
    return {
        "capability_id": flow.flow_id,
        "description": flow.description,
        "backend_port": flow.backend_port,
        "spend_incurring": flow.spend_incurring,
        "irreversible": flow.irreversible,
        "cross_cutting": flow.cross_cutting,
        "milestone": flow.milestone,
        "gated": _is_gated(flow),
        "runnable": resolve_executor(capability_id) is not None,
    }


def invoke_capability(
    *,
    capability_id: str,
    params: dict[str, Any] | None = None,
    policy: ApprovalPolicy | None = None,
    steering: Any | None = None,
    surface: str = "api",
    plan_only: bool = False,
) -> CapabilityResult:
    """The ADR-010 single funnel — invoke a registered capability (flow) from ANY surface.

    Every surface (Chat/TUI/CLI/Python-API/headless) calls THIS. The funnel:

      1. resolves the capability from the shared flow registry (``get_flow``);
      2. resolves its executor (``run_<flow_id>``) lazily;
      3. when ``plan_only`` — returns the read-only plan preview and executes NOTHING
         (the Plan-Mode "think first" beat, ADR-013);
      4. **gated** capabilities (spend/irreversible) route through ``plan_approve_execute``
         (preview → approval-gate → execute; aborts before any spend on denial);
      5. **cheap-read** capabilities execute directly (stream like any read).

    Args:
        capability_id: The flow id to invoke (e.g. ``"data_quality_flow"``).
        params: The keyword params forwarded to the flow's ``run_<flow_id>`` executor.
        policy: The approval policy (default safe ``ask``; non-interactive ask DENIES gated).
        steering: Optional steering profile forwarded to the plan/flow.
        surface: The calling surface label (recorded on the result for traceability).
        plan_only: When True, return only the plan preview (no execution) — Plan Mode.

    Returns:
        A :class:`CapabilityResult` — the SAME shape for every surface (the parity contract).
    """
    params = dict(params or {})
    flow = get_flow(capability_id)
    if flow is None:
        return CapabilityResult(
            capability_id=capability_id, surface=surface, status="error",
            approved=False, error=f"unknown capability {capability_id!r}",
        )

    plan = preview_flow_plan(flow_id=flow.flow_id, params=params, steering=steering)

    if plan_only:
        return CapabilityResult(
            capability_id=flow.flow_id, surface=surface, status="planned",
            approved=False, gated=_is_gated(flow), executed=False, plan=plan,
        )

    executor = resolve_executor(flow.flow_id)
    if executor is None:
        return CapabilityResult(
            capability_id=flow.flow_id, surface=surface, status="error",
            approved=False, gated=_is_gated(flow), plan=plan,
            error=f"no executor resolvable for capability {flow.flow_id!r}",
        )

    # ── gated path: preview → approval gate → execute (aborts before spend on deny) ──
    if _is_gated(flow):
        try:
            outcome = plan_approve_execute(
                flow_id=flow.flow_id, params=params, executor=executor,
                policy=policy, steering=steering,
            )
        except Exception as e:
            # Universal error handling: capture the flow error into the
            # centralized ERROR_LOG with ``source="flows.<flow_id>"`` at this single
            # dispatch funnel — additive, semantics unchanged (still returns an error
            # CapabilityResult). Guarded so telemetry never breaks a dispatch.
            _capture_flow_error(e, flow_id=flow.flow_id, surface=surface)
            return CapabilityResult(
                capability_id=flow.flow_id, surface=surface, status="error",
                approved=True, gated=True, executed=True, plan=plan,
                error=str(e),
            )
        approved = bool(outcome.get("approved"))
        gated_status = "completed" if approved else "denied"
        _emit_capability_trace(flow=flow, params=params, surface=surface,
                               status=gated_status, result=outcome.get("result"))
        return CapabilityResult(
            capability_id=flow.flow_id,
            surface=surface,
            status=gated_status,
            approved=approved,
            gated=True,
            executed=bool(outcome.get("executed")),
            plan=outcome.get("plan", plan),
            decision=outcome.get("decision"),
            result=outcome.get("result"),
        )

    # ── cheap-read path: stream straight through (never prompts) ─────────────────
    try:
        result = executor(**params)
    except Exception as e:
        _capture_flow_error(e, flow_id=flow.flow_id, surface=surface)
        _emit_capability_trace(flow=flow, params=params, surface=surface,
                               status="error", result=None)
        return CapabilityResult(
            capability_id=flow.flow_id, surface=surface, status="error",
            approved=True, gated=False, executed=True, plan=plan, error=str(e),
        )
    _emit_capability_trace(flow=flow, params=params, surface=surface,
                           status="completed", result=result)
    return CapabilityResult(
        capability_id=flow.flow_id, surface=surface, status="completed",
        approved=True, gated=False, executed=True, plan=plan, result=result,
        decision={"decision": "approved", "reason": "cheap read — auto-approved",
                  "action_class": ActionClass.CHEAP_READ.value, "gated": False},
    )
