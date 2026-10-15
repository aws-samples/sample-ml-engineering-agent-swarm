"""Strands @tool wrappers that let the TUI chat agent ROUTE work to the ML system.

The TUI's conversational agent is the always-engaged "brain" of the surface. These
tools give it first-class access to the SAME capability funnel
(``control_plane.capability``) that the CLI / Python-API / TUI panes use — the single
ADR-010 entrypoint to every registered ML flow (data quality, feature engineering,
baseline train, HPO, evaluation, deploy/serving, monitoring), which in the managed
deployment run on SageMaker / Bedrock / the AgentCore swarm.

So instead of the chat agent only owning small read/train tools, it can:
  * ``list_ml_capabilities`` — discover the full menu of registered flows (read-only), and
  * ``run_ml_capability`` — dispatch a chosen flow through the funnel (Plan-Mode preview by
    default; execution is gated by the SAME safe-by-default approval gate as every surface).

This is the "tools to route to the various bespoke ML agents" seam: the agent reasons about
intent, picks the right capability, and hands the heavy work to the flow/lead behind the funnel.

LAZY IMPORTS: like the other tool modules, the control-plane import lives inside each tool so
importing this module never drags the control plane into the chat hot path / TUI launch.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any

from strands import tool

logger = logging.getLogger(__name__)

# ── H2-5 cloud-dispatch client injection seam ─────────────────────────────────
# The ``dispatch_cloud_experiment`` tool reaches the deployed AgentCore swarm via
# :func:`amlas.cloud.dispatch.run_cloud_experiment`, whose S3 + bedrock-agentcore
# clients are INJECTABLE (SEC-01 / Phase-6 de-mock). These process-local handles let
# a offline validation register FAKE clients so the whole cloud path is offline-hermetic (no
# real AWS / network). ``None`` (the production default) → ``run_cloud_experiment``
# builds the real boto3 clients itself. Guarded like the active-bus seam in bus.py.
_CLOUD_S3_CLIENT: Any | None = None
_CLOUD_AGENTCORE_CLIENT: Any | None = None

# ── FIX-16 dispatch-CONFIG injection seam ─────────────────────────────────────
# ``dispatch_cloud_experiment`` calls ``run_cloud_experiment`` with NO explicit config, so it
# uses ``DispatchConfig()`` defaults (trials=1, wall_clock_minutes=5, baseline 0.55, inferred
# delta) — a CHEAP-SMOKE budget. A mandated-path E2E harness that must dispatch a REAL
# multi-minute experiment THROUGH the tool (never a direct ``run_cloud_experiment`` shortcut)
# needs to thread the spec's real budget (trials/wall-clock/baseline/acceptance-delta) in.
# This process-local handle lets such a harness register a full ``DispatchConfig``; the tool
# passes it to ``run_cloud_experiment``. ``None`` (production default) ⇒ the tool builds the
# default config exactly as before (byte-identical, no behavior change for the chat agent).
_CLOUD_DISPATCH_CONFIG: Any | None = None

# ── FIX-16 dispatch-result observability seam ─────────────────────────────────
# ``dispatch_cloud_experiment`` (the chat agent's ONLY cloud-dispatch tool) returns a
# COMPACT summary ``{status, session_id, dataset_uri, run_status}`` to the LLM — it
# deliberately does NOT surface the full streamed event list or the terminal
# ``arbiter_report`` (those would bloat the tool result the model reasons over). But a
# mandated-path E2E harness (``scripts/e2e_string_label_classification_run.py`` / ``e2e_weather_run.py``)
# that dispatches THROUGH this tool still needs the streamed ``DispatchResult`` to grade
# the ML/metric/lead gates. We therefore capture the LAST full ``DispatchResult`` in a
# process-local handle (same pattern/guarantees as the client-injection seam above) and
# expose it via :func:`get_last_dispatch_result`. Null-safe, best-effort, cleared on
# demand; the tool's own return value is UNCHANGED (no behavior/contract change for the
# chat agent).
_LAST_DISPATCH_RESULT: Any | None = None

# ──  background-dispatch seam ─────────────────────────────────────────
# ``dispatch_cloud_experiment`` calls ``run_cloud_experiment`` SYNCHRONOUSLY, and on the
# live Strands path tools run inline inside ``agent(prompt)`` — so a multi-minute
# InvokeAgentRuntime stream BLOCKS the chat worker thread for the whole run (the chat turn
# can't return until a terminal event; the UI looks stuck). Merely adding the tool to
# ``LONG_RUNNING_TOOLS`` is insufficient because that JobManager seam only fires in
# ``execute_tool_use`` (the offline confirmation-gate path), NOT the live ``agent(prompt)``
# loop. So the tool itself submits the invoke as a background JobManager job and returns
# immediately with the session_id. The app injects its shared ThreadExecutor-backed
# JobManager here (next to ``tool_backend.job_manager``). ``None`` (headless / tests / CLI)
# ⇒ the dispatch runs synchronously exactly as before (byte-identical) — graceful
# degradation, same pattern as the InlineExecutor default.
_CLOUD_JOB_MANAGER: Any | None = None


def set_cloud_job_manager(job_manager: Any | None = None) -> None:
    """Register (or clear) the JobManager used to BACKGROUND cloud dispatch.

    The TUI app injects its shared ``ThreadExecutor``-backed manager so
    ``dispatch_cloud_experiment`` submits the blocking InvokeAgentRuntime stream as a
    background job and the chat turn returns instantly. ``None`` clears the seam back to
    the synchronous default (headless / CLI / offline callers) — behavior byte-identical to
    previous. Callers may inject a ``DeferredExecutor``-backed manager to drive it
    deterministically.
    """
    global _CLOUD_JOB_MANAGER
    _CLOUD_JOB_MANAGER = job_manager


def set_cloud_dispatch_clients(
    *, s3_client: Any | None = None, agentcore_client: Any | None = None
) -> None:
    """Register (or clear) the injectable S3 + AgentCore clients for cloud dispatch.

    Tests call this with fakes so ``dispatch_cloud_experiment`` never touches real AWS;
    passing ``None`` for both clears the seam back to the production default (real boto3).
    """
    global _CLOUD_S3_CLIENT, _CLOUD_AGENTCORE_CLIENT
    _CLOUD_S3_CLIENT = s3_client
    _CLOUD_AGENTCORE_CLIENT = agentcore_client


def set_cloud_dispatch_config(config: Any | None = None) -> None:
    """Register (or clear) an explicit ``DispatchConfig`` for ``dispatch_cloud_experiment``.

    FIX-16: the tool otherwise calls ``run_cloud_experiment`` with the ``DispatchConfig()``
    cheap-smoke defaults. A mandated-path E2E harness registers the spec's REAL budget here
    (trials / wall-clock / baseline / acceptance-delta / splits) so a full multi-minute run
    dispatches THROUGH the tool. ``None`` clears the seam back to the default config.
    """
    global _CLOUD_DISPATCH_CONFIG
    _CLOUD_DISPATCH_CONFIG = config


def get_last_dispatch_result() -> Any | None:
    """Return the full ``DispatchResult`` from the most recent ``dispatch_cloud_experiment``.

    FIX-16 observability accessor: the tool returns only a compact summary to the LLM, so a
    mandated-path E2E harness that dispatches through the tool reads the full streamed
    result (events + terminal ``arbiter_report`` + session id) from here to grade its ML /
    metric / lead gates. Returns ``None`` when no dispatch has run in this process (or after
    :func:`clear_last_dispatch_result`). Best-effort observability — never authoritative for
    the chat agent's own control flow.
    """
    return _LAST_DISPATCH_RESULT


def clear_last_dispatch_result() -> None:
    """Clear the captured last-dispatch result (between-run hygiene / between-run isolation)."""
    global _LAST_DISPATCH_RESULT
    _LAST_DISPATCH_RESULT = None


# FIX-25: the chat surface is THE command center — a bare chat dispatch must be able to run a
# REAL experiment, not a smoke test. ``DispatchConfig()`` alone defaults to trials=1 /
# wall_clock_minutes=5 (a cheap smoke), so a real run budget-exhausted before execution began.
# When the chat agent dispatches with NO harness-injected config, we seed a usable DEFAULT
# budget (NOT a cap — the operator/agent can raise it via explicit tool args, and nothing here
# clamps their value). This matches the proven e2e_live_run.py budget (20 trials / 30 min).
_CHAT_DEFAULT_BUDGET: dict[str, Any] = {
    "trials": 20,
    "wall_clock_minutes": 30,
    "max_retreats": 3,
    "max_rows": None,  # FULL dataset — a slice is what tripped D1 in the smoke.
    "read_timeout": 2400,  # socket must outlast the whole run.
}


def _resolve_dispatch_config(base: Any | None, budget_overrides: dict[str, Any] | None = None) -> Any:
    """Return a ``DispatchConfig`` with the ARN/bucket resolved AND a usable budget (FIX-23/25).

    Runtime ARN + session bucket resolve via the CLI's precedence — **explicit → env (incl.
    ``.env``) → SSM → default** (FIX-23) — filling only empty fields so an explicit ARN wins and
    an unreachable SSM degrades to the SEC-01 fail-fast (never a live default, never raises).

    Budget (FIX-25 — chat is the command center, so a chat dispatch can run a REAL experiment):
    * ``base`` is a harness-injected :class:`DispatchConfig` (via ``set_cloud_dispatch_config``):
      its budget is authoritative — kept verbatim (a mandated-path E2E owns its own budget).
    * ``base is None`` (the chat-agent path): seed :data:`_CHAT_DEFAULT_BUDGET` — a usable
      DEFAULT (20 trials / 30 min), NOT the cheap-smoke 1/5. This is a default, not a cap.
    * ``budget_overrides`` (an explicit chat tool arg — e.g. the user said "run it with a
      20-minute budget"): ALWAYS wins, over both the default and an injected config, and is
      applied verbatim with NO clamping/ceiling — the operator's number is authoritative.
    """
    from dataclasses import replace

    from amlas.cloud.dispatch import (
        DispatchConfig,
        resolve_runtime_arn,
        resolve_session_bucket,
    )

    # Base config + the chat DEFAULT budget when the agent dispatches with no injected config.
    if base is not None:
        cfg = base
    else:
        cfg = DispatchConfig(**{k: v for k, v in _CHAT_DEFAULT_BUDGET.items()})

    region = getattr(cfg, "region", None)
    try:
        arn = resolve_runtime_arn(getattr(cfg, "runtime_arn", "") or "", region=region)
    except Exception:
        arn = getattr(cfg, "runtime_arn", "") or ""
    try:
        bucket = resolve_session_bucket(getattr(cfg, "session_bucket", "") or "", region=region)
    except Exception:
        bucket = getattr(cfg, "session_bucket", "") or ""

    # Explicit budget overrides (from the chat tool call) win over everything, verbatim — NO
    # clamping. Only the fields the caller actually set (non-None) are applied.
    overrides = {
        k: v for k, v in (budget_overrides or {}).items() if v is not None
    }
    try:
        return replace(cfg, runtime_arn=arn, session_bucket=bucket, **overrides)
    except Exception:
        # Non-dataclass / replace failure → hand back the base untouched (degrade gracefully).
        return cfg


# ── H2-B context→param mapping (the ADR-010 dispatch enrichment) ──────────────
# Per-capability map of ``{flow_param_name: dotted AppContextState path}``. It is
# EXPLICIT (not global) because flows name the same concept differently — a dataset
# is ``dataset_uri`` (DQ/imputation/FE), ``dataset`` (data_collection) or ``train_uri``
# (baseline_train); the spec target is always ``target``. An entry here is only ever
# applied when (a) the flow's executor ACTUALLY declares that param (verified live via
# signature introspection — :func:`_accepted_params`) and (b) the LLM omitted it, so a
# mapping entry can never shove a key a flow rejects and the model's intent always wins.
# A capability absent from this table receives NO context defaults (the unknown-capability
# guard) — steering (H2-A) is handled separately for any flow whose executor accepts it.
_CONTEXT_PARAM_MAP: dict[str, dict[str, str]] = {
    # dataset-consuming flows → default the active dataset path into their URI param.
    "data_quality_flow": {"dataset_uri": "dataset.path"},
    "imputation_flow": {"dataset_uri": "dataset.path", "target": "spec.target"},
    "feature_engineering_flow": {"dataset_uri": "dataset.path", "target": "spec.target"},
    "data_collection_flow": {"dataset": "dataset.path"},
    # spec/train flows → default the active dataset + the active spec's target.
    "baseline_train_flow": {"train_uri": "dataset.path", "target": "spec.target"},
}


def _context_value(state: Any, dotted: str) -> Any:
    """Read a dotted attribute path off the ``AppContextState`` snapshot (null-safe)."""
    obj: Any = state
    for part in dotted.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def _accepted_params(capability_id: str) -> set[str]:
    """Return the keyword params the capability's flow executor actually declares.

    Resolves the SAME executor the funnel will run (``resolve_executor`` — registry →
    override → convention, so a gated flow resolves its INNER ``execute_*``) and reads its
    signature. This is the schema whitelist: a context default is injected ONLY if the flow
    truly accepts that param, so ``executor(**params)`` can never receive a rejected key.
    Returns an empty set (inject nothing) when the executor is unresolvable — fully safe.
    """
    try:
        from amlas.control_plane.capability import resolve_executor

        executor = resolve_executor(capability_id)
        if executor is None:
            return set()
        sig = inspect.signature(executor)
        return {
            name
            for name, p in sig.parameters.items()
            if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        }
    except (ValueError, TypeError, ImportError):
        return set()


def _context_defaults_for(
    capability_id: str, params: dict[str, Any], state: Any
) -> dict[str, Any]:
    """Compute the context-derived param DEFAULTS to fill on dispatch (H2-B; pure).

    Returns a NEW dict of ``{param: value}`` to ADD — it NEVER overwrites a key the LLM
    already set (explicit-param precedence: the model's intent wins; active context only
    fills gaps) and only names params the flow's executor declares (the schema whitelist).
    Null-safe: no state / unknown capability / unresolvable executor → ``{}``.
    """
    if state is None:
        return {}
    mapping = _CONTEXT_PARAM_MAP.get((capability_id or "").strip().lower(), {})
    if not mapping:
        return {}
    accepted = _accepted_params(capability_id)
    if not accepted:
        return {}
    out: dict[str, Any] = {}
    for param_name, dotted in mapping.items():
        if param_name in params:  # LLM provided it → the model's intent wins
            continue
        if param_name not in accepted:  # flow does not declare it → never inject
            continue
        val = _context_value(state, dotted)
        if val is not None and val != "":
            out[param_name] = val
    return out


@tool
def list_ml_capabilities() -> dict[str, Any]:
    """List every ML capability (flow) the system can run, with its gating + description.

    Read-only discovery over the shared capability registry (ADR-010) — the same menu the
    CLI ``capability list`` and the chat ``/flows`` command show. Call this FIRST when the
    user asks what the system can do, or before dispatching work, so you pick a real
    capability id (never invent one).

    Returns:
        A dict ``{"count": int, "capabilities": [{capability_id, description, gated,
        spend_incurring, irreversible, milestone, runnable}, ...]}``.
    """
    from amlas.control_plane.capability import list_capabilities

    caps = list_capabilities()
    slim = [
        {
            "capability_id": c["capability_id"],
            "description": c["description"],
            "gated": c["gated"],
            "spend_incurring": c["spend_incurring"],
            "irreversible": c["irreversible"],
            "milestone": c["milestone"],
            "runnable": c["runnable"],
        }
        for c in caps
    ]
    return {"count": len(slim), "capabilities": slim}


@tool(context=True)
def run_ml_capability(
    capability_id: str,
    params_json: str = "{}",
    plan_only: bool = True,
    tool_context: Any = None,
) -> dict[str, Any]:
    """Dispatch an ML capability (flow) through the shared funnel — the route to bespoke ML work.

    This is how you hand heavy/bespoke ML work (training, HPO, evaluation, deploy, monitoring,
    data-quality, feature-engineering, …) to the system's flows — which in the managed
    deployment run on SageMaker / Bedrock / the AgentCore swarm. It calls the SAME
    ``invoke_capability`` funnel every surface uses, so the safe-by-default approval gate
    applies identically here.

    SAFE DEFAULT: ``plan_only=True`` returns a read-only Plan-Mode PREVIEW (steps, cost
    estimate, whether it is gated/spend/irreversible) and executes NOTHING. Prefer previewing
    first, show the user the plan, and only pass ``plan_only=False`` to actually run it — a
    gated (spend/irreversible) capability will still route through the approval gate and may
    be denied by policy.

    Args:
        capability_id: A real flow id from ``list_ml_capabilities`` (e.g. ``"data_quality_flow"``).
        params_json: JSON object string of keyword params for the flow (e.g. ``'{"path": "..."}'``).
        plan_only: True (default) → preview only; False → attempt execution through the gate.

    CONTEXT PROPAGATION (Hop-2, additive + null-safe): beyond the LLM-authored
    ``params_json``, the dispatch is enriched from the live cross-tab session context —
    the SAME process-local snapshot the read-only ``get_tui_context`` tool reads
    (:func:`~amlas.tui.context.bus.get_active_bus`):

      * **H2-A — steering:** the session steering dial (Steering tab / ``/steer set``) is
        forwarded into ``invoke_capability(steering=…)`` (cost-aware preview + the approval
        gate) AND, for a flow whose executor declares it, into ``params['steering']`` so the
        gated ``executor(**params)`` actually receives it — no more steering-blind backend work.
      * **H2-B — active dataset/spec:** the active dataset URI + the active spec's target
        DEFAULT into the flow's declared params (per :data:`_CONTEXT_PARAM_MAP`) when the LLM
        omitted them, so "run data quality" uses the dataset the user selected in the UI.

    PRECEDENCE: an explicit value in ``params_json`` ALWAYS wins — the model's intent is
    authoritative; the active context only fills gaps. When there is no active bus
    (headless / no app), behavior is byte-identical to before (steering ``None``, no defaults).
    This is purely additive context on the payload — it does NOT bypass or weaken the
    ADR-010 approval gate / capability funnel; the enriched payload flows through the SAME
    ``plan_only=True`` default → plan→approve→execute path.

    Returns:
        A dict with ``{capability_id, status, gated, approved, executed, plan, error}`` where
        ``status`` is ``"planned"`` | ``"completed"`` | ``"denied"`` | ``"error"``.
    """
    import json

    from amlas.control_plane.approvals import ApprovalPolicy
    from amlas.control_plane.capability import invoke_capability

    try:
        params = json.loads(params_json) if params_json and params_json.strip() else {}
        if not isinstance(params, dict):
            return {"capability_id": capability_id, "status": "error",
                    "error": f"params_json must be a JSON object, got {type(params).__name__}"}
    except json.JSONDecodeError as exc:
        return {"capability_id": capability_id, "status": "error",
                "error": f"invalid params_json: {exc}"}

    # ── Hop-2 context enrichment (additive, null-safe; LLM params always win) ──
    # Read the CURRENT active session context — the same singleton get_tui_context uses.
    from amlas.tui.context.bus import get_active_bus

    bus = get_active_bus()
    state = bus.state() if bus is not None else None

    # H2-B: fill the active dataset/spec defaults for params THIS flow declares (LLM wins).
    params.update(_context_defaults_for(capability_id, params, state))

    # H2-A: resolve the steering dial — explicit LLM param wins, else the session dial.
    steering = params.get("steering")
    if steering is None and state is not None:
        session_steering = getattr(state, "steering", None)
        if session_steering is not None:
            steering = session_steering
            # Inject into params ONLY when the flow's executor accepts ``steering`` so the
            # gated ``executor(**params)`` receives it (schema-whitelisted; never a rejected key).
            if "steering" in _accepted_params(capability_id):
                params["steering"] = session_steering

    # SDK-injected context is excluded from the model argument schema. Only an
    # actual caller policy object is authority; prompt/params/model JSON never is.
    state = getattr(tool_context, "invocation_state", None)
    policy = state.get("_amlas_approval_policy") if isinstance(state, dict) else None
    if not isinstance(policy, ApprovalPolicy):
        policy = None

    result = invoke_capability(
        capability_id=capability_id, params=params, steering=steering,
        surface="chat", plan_only=plan_only, policy=policy,
    )
    return result.to_dict()


@tool
def get_tui_context() -> dict[str, Any]:
    """Return the active TUI session state: active spec, dataset, job, run, model, problem, steering.

    Read-only. Call this to ground answers in what the user has loaded/selected across the
    UI — e.g. the spec open on the Specs tab, the dataset selected on the Data tab, or the
    job currently running — so you never assume state the user did not set. Prefer this over
    guessing "the current spec"; it reflects the live cross-tab context bus, not a per-turn
    disk re-scan.

    Returns:
        A dict snapshot of the shared ``AppContext``:
        ``{"available": true, "spec": {...}, "dataset": {...}, "job": {...}, "run": {...},
        "model": {...}, "problem": {...}, "steering": ..., "agents": {...}, "selection": {...}}``.
        When called outside a running TUI (headless / no app registered) it returns
        ``{"available": false}`` — there is no active UI context to report.

        Phase C2 (design §D.5): when a session ``SessionLedger`` is also active, a bounded
        ``"session"`` timeline is merged under a ``"session"`` key (the HISTORY layer of what
        has happened this session across ALL modes — turns, tools, commands, run/dataset/spec
        transitions, artifacts, mode switches). The live bus snapshot above remains
        authoritative for CURRENT state; the ledger is history. Absent a ledger the tool is
        byte-identical to before (no ``"session"`` key).
    """
    from amlas.tui.context.bus import get_active_bus, get_active_ledger

    bus = get_active_bus()
    if bus is None:
        return {"available": False}
    snapshot = bus.snapshot_dict()
    # Phase C2: merge the bounded session-history timeline when a ledger is active
    # (null-safe — absent a ledger the tool is byte-identical to pre-C2). Best-effort:
    # a ledger read must never break the read-only context tool.
    ledger = get_active_ledger()
    if ledger is not None:
        try:
            snapshot["session"] = ledger.snapshot_dict()
        except Exception:  # nosec B110
            pass
    return snapshot


# ── H2-5 cloud-dispatch helpers ───────────────────────────────────────────────


def _steering_block_for_cloud(steering: Any) -> str:
    """Render the active steering dial as a text block to BAKE INTO the cloud spec body.

    Reuses :meth:`SteeringProfile.as_prompt_block` (duck-typed) so the deployed AgentCore
    swarm honors the SAME dial the user set on the Steering tab. Returns ``""`` when there
    is no dial or no active signal — so the no-steering cloud dispatch is unchanged.
    """
    if steering is None or not hasattr(steering, "as_prompt_block"):
        return ""
    try:
        return steering.as_prompt_block() or ""
    except Exception:
        return ""


def _problem_block_for_cloud(problem: Any) -> str:
    """Render the active problem framing (``ActiveProblem``) as a text block for the spec body."""
    if problem is None:
        return ""
    kpi = getattr(problem, "kpi", None)
    pid = getattr(problem, "problem_id", None)
    lines = []
    if pid:
        lines.append(f"Problem id: {pid}")
    if kpi:
        lines.append(f"KPI / objective: {kpi}")
    return ("Active problem framing — " + "; ".join(lines)) if lines else ""


# Max chars of the ledger history baked into a dispatched cloud spec (bounded so the
# multi-agent request stays small — R3 prompt-size discipline). ~12 recent entries *
# ~200 chars ≈ 2400; cap defensively at a hard 2000 clamp.
_CLOUD_LEDGER_MAX_CHARS = 2000


def _ledger_block_for_cloud(ledger: Any) -> str:
    """Render a BOUNDED session-history block to bake into the cloud spec body (Phase D2).

    Reuses ``SessionLedger.render_context`` (the SAME bounded renderer the local
    ``_build_context`` fold uses) so the deployed AgentCore swarm inherits the session
    timeline. The block is labeled "prior context" and appended LAST in
    :func:`dispatch_cloud_experiment` (after the explicit description + steering +
    problem) so the swarm weights the authoritative intent above the history. Returns
    ``""`` when there is no ledger or no history — so a no-history dispatch is
    byte-identical to today. Best-effort: never raises. Hard-clamped to
    :data:`_CLOUD_LEDGER_MAX_CHARS`.
    """
    if ledger is None:
        return ""
    try:
        body = ledger.render_context(max_entries=12)
    except Exception:
        return ""
    if not body:
        return ""
    block = f"## Session history (prior context, all modes)\n{body}"
    return block[:_CLOUD_LEDGER_MAX_CHARS]


@tool
def estimate_dispatch_cost(
    trials: int = 0,
    wall_clock_minutes: int = 0,
    n_leads: int = 7,
) -> dict[str, Any]:
    """Estimate the RUNTIME + AWS COST of a cloud dispatch BEFORE running it (read-only, no spend).

    FIX-25 — chat is the command center: call this FIRST, show the user the estimate + your
    assumptions, and get ONE confirmation before ``dispatch_cloud_experiment``. This is an
    ESTIMATE, not a bill — present it as a range/ballpark with its assumptions, never a
    confident single figure.

    Prices come from the self-refreshing local price book (``knowledge-base/pricing/
    aws-pricing-sample-<date>.json``); if the newest snapshot is >30 days old it is refreshed
    via Tavily on demand (best-effort, degrades to the last snapshot). The cost model is a
    transparent ballpark: Bedrock tokens for ``n_leads`` Opus leads over the wall-clock window
    + SageMaker compute for ``trials`` HPO trials + negligible S3.

    Args:
        trials: HPO trial budget (0 ⇒ the chat default, 20). NO cap.
        wall_clock_minutes: wall-clock budget (0 ⇒ the chat default, 30). NO cap.
        n_leads: number of Opus leads active during the run (default 7).

    Returns:
        ``{"is_estimate": True, "est_runtime_minutes": int, "est_cost_usd_low": float,
        "est_cost_usd_high": float, "breakdown": {...}, "assumptions": [...],
        "price_book": {"checked_at","source","stale","refreshed"}, "warnings": [...]}``.
        Never raises — a pricing failure still returns a caveated estimate from defaults.
    """
    from ..pricing import load_price_book

    eff_trials = trials if trials and trials > 0 else int(_CHAT_DEFAULT_BUDGET["trials"])
    eff_wall = (
        wall_clock_minutes
        if wall_clock_minutes and wall_clock_minutes > 0
        else int(_CHAT_DEFAULT_BUDGET["wall_clock_minutes"])
    )
    eff_leads = n_leads if n_leads and n_leads > 0 else 7

    try:
        book = load_price_book()
    except Exception:
        # Absolute fallback — load_price_book already never raises, but be defensive.
        from ..pricing import DEFAULT_PRICE_BOOK, PriceBook

        book = PriceBook(
            prices=dict(DEFAULT_PRICE_BOOK["prices"]),
            checked_at=str(DEFAULT_PRICE_BOOK["checked_at"]),
            source="baked-in default",
            stale=True,
            warnings=["pricing load failed — using baked-in default estimate basis"],
        )

    # ── transparent ballpark cost model (all assumptions surfaced) ──
    # Bedrock: assume each lead sustains a modest token throughput over the wall-clock window.
    tokens_per_lead_min = 1500        # combined in+out tokens/min/lead (rough)
    out_frac = 0.4                    # fraction of tokens that are output (pricier)
    total_tokens_k = (eff_leads * tokens_per_lead_min * eff_wall) / 1000.0
    in_price = book.price("bedrock_opus_input_per_1k")
    out_price = book.price("bedrock_opus_output_per_1k")
    bedrock_cost = total_tokens_k * ((1 - out_frac) * in_price + out_frac * out_price)

    # SageMaker: assume each trial occupies a small training instance for a short slice.
    minutes_per_trial = 3.0
    sm_hourly = book.price("sagemaker_ml_m5_xlarge_hour")
    sagemaker_cost = eff_trials * (minutes_per_trial / 60.0) * sm_hourly

    point = bedrock_cost + sagemaker_cost
    # Estimates are uncertain → present a band (±60%), and honor pricing staleness.
    low = round(point * 0.4, 2)
    high = round(point * 1.6, 2)

    assumptions = [
        f"{eff_leads} Opus leads sustaining ~{tokens_per_lead_min} tokens/min each over "  # nosemgrep: string-concat-in-list
        f"{eff_wall} min ({out_frac:.0%} output tokens)",
        f"{eff_trials} HPO trials × ~{minutes_per_trial:.0f} min on ml.m5.xlarge "  # nosemgrep: string-concat-in-list
        f"(${sm_hourly}/hr)",
        f"Bedrock Opus ${in_price}/1K in, ${out_price}/1K out; S3 negligible",
        "±60% band — this is a rough ESTIMATE, not a billing quote; actuals vary with "  # nosemgrep: string-concat-in-list
        "model behavior, retries, and instance choice",
    ]
    warnings = list(book.warnings)
    if book.stale:
        warnings.append(
            f"price book last checked {book.checked_at or 'never'} (>30d) — figures may be dated"
        )

    return {
        "is_estimate": True,
        "est_runtime_minutes": eff_wall,
        "est_cost_usd_low": low,
        "est_cost_usd_high": high,
        "est_cost_usd_point": round(point, 2),
        "breakdown": {
            "bedrock_usd": round(bedrock_cost, 2),
            "sagemaker_usd": round(sagemaker_cost, 2),
        },
        "budget": {"trials": eff_trials, "wall_clock_minutes": eff_wall, "n_leads": eff_leads},
        "assumptions": assumptions,
        "price_book": {
            "checked_at": book.checked_at,
            "source": book.source,
            "stale": book.stale,
            "refreshed": book.refreshed,
            "region": book.region,
        },
        "warnings": warnings,
    }


@tool
def dispatch_cloud_experiment(
    description: str,
    dataset_path: str = "",
    target: str = "",
    project_id: str = "",
    primary_metric: str = "",
    trials: int = 0,
    wall_clock_minutes: int = 0,
    max_rows: int = 0,
) -> dict[str, Any]:
    """Dispatch a REAL cloud experiment to the deployed AgentCore swarm (spend — confirm first).

    This is the ONLY seam by which the chat agent can hand a full experiment to the deployed
    AgentCore runtime (the swarm of Strands leads/subagents that run on Bedrock + SageMaker).
    It invokes a live runtime → it INCURS REAL AWS SPEND and is IRREVERSIBLE, so it is
    confirmation-gated (state-changing, NEVER auto-executed): only run it after the user has
    explicitly approved the cloud run.

    CONTEXT PROPAGATION (Hop-2 H2-5, additive + null-safe): the payload is built from the
    SAME live cross-tab session snapshot ``run_ml_capability`` reads
    (:func:`~amlas.tui.context.bus.get_active_bus`). When the LLM omits a field it DEFAULTS
    from the active context so "run this in the cloud" uses what the user set in the UI:

      * dataset → the active dataset selected on the Data tab (``state.dataset.path``);
      * target / primary_metric / project_id → the active spec (``state.spec``);
      * steering → the session dial (Steering tab) is BAKED INTO the dispatched spec text so
        the cloud swarm honors the SAME hard constraints;
      * problem → the active problem framing is appended to the spec body.

    PRECEDENCE: an explicit argument ALWAYS wins over the active-context default (the same
    rule as ``run_ml_capability`` — the model's intent is authoritative; context fills gaps).

    Args:
        description: Natural-language problem statement for the run (required).
        dataset_path: Local dataset path to stage; defaults to the active dataset when omitted.
        target: Target column; defaults to the active spec's target when omitted.
        project_id: Human label for the run; defaults to the active spec's project_id / a
            sanitized name derived from the dataset when omitted.
        primary_metric: Optional metric override; defaults to the active spec's metric.
        trials: HPO trial budget for the run. 0 (default) ⇒ the usable chat default (20);
            set it to run more/fewer. There is NO cap — the value is honored verbatim.
        wall_clock_minutes: Wall-clock budget in minutes. 0 (default) ⇒ the chat default
            (30). NO cap — honored verbatim (estimate cost/runtime and confirm with the user
            BEFORE dispatching a large budget; call ``estimate_dispatch_cost`` first).
        max_rows: Cap the dataset to N rows for a faster/cheaper run. 0 (default) ⇒ FULL
            dataset. NO ceiling imposed by the system.

    Returns:
        A dict ``{"status": "dispatched"|"error", "session_id": str, "dataset_uri": str,
        "run_status": str|None, "error": str}``. ``session_id`` correlates the run to its
        CloudWatch trace (published as agent activity for the Dashboard).
    """
    from pathlib import Path

    from amlas.tui.context.bus import get_active_bus, get_active_ledger

    # ── read the CURRENT active session context (same singleton get_tui_context uses) ──
    bus = get_active_bus()
    state = bus.state() if bus is not None else None
    spec = getattr(state, "spec", None) if state is not None else None
    dataset = getattr(state, "dataset", None) if state is not None else None
    steering = getattr(state, "steering", None) if state is not None else None
    problem = getattr(state, "problem", None) if state is not None else None

    # ── resolve each field: explicit LLM arg WINS, else the active-context default ──
    resolved_dataset = dataset_path or (getattr(dataset, "path", None) if dataset else None) or ""
    if not resolved_dataset:
        return {
            "status": "error",
            "error": "no dataset to dispatch — select a dataset on the Data tab or pass dataset_path",
        }
    resolved_target = target or (getattr(spec, "target", None) if spec else None) or ""
    resolved_metric = primary_metric or (getattr(spec, "primary_metric", None) if spec else None) or None
    resolved_project = (
        project_id
        or (getattr(spec, "project_id", None) if spec else None)
        or Path(resolved_dataset).stem
        or "amlas_run"
    )

    # ── bake the session steering + problem framing INTO the dispatched spec body ──
    effective_description = description
    steering_block = _steering_block_for_cloud(steering)
    if steering_block:
        effective_description = f"{effective_description}\n\n{steering_block}"
    problem_block = _problem_block_for_cloud(problem)
    if problem_block:
        effective_description = f"{effective_description}\n\n{problem_block}"

    # ── bake the bounded SESSION HISTORY (prior context, all modes) LAST (Phase D2) ──
    # The ledger timeline is appended AFTER the authoritative description + steering +
    # problem so the deployed swarm weights the explicit intent above the history
    # (design §D2/§C precedence). Null-safe: absent an active ledger (headless / no TUI)
    # ``get_active_ledger()`` is ``None`` → ``""`` → the dispatch is byte-identical to
    # pre-D2. Bounded/clamped by ``_ledger_block_for_cloud`` (R3 prompt-size discipline).
    ledger_block = _ledger_block_for_cloud(get_active_ledger())
    if ledger_block:
        effective_description = f"{effective_description}\n\n{ledger_block}"

    # ── resolve the runtime config + budget synchronously (fail-fast BEFORE backgrounding) ──
    # FIX-23: resolve the runtime ARN + session bucket the SAME way the CLI does —
    # explicit → env (incl. the loaded ``.env``) → SSM ``/amlas/*`` → default. ``setup.sh``
    # publishes the deploy-time ARN to SSM (NOT to ``.env``), and a bare ``DispatchConfig()``
    # is env-ONLY (``os.environ.get``), so before this every chat dispatch fail-fasted with
    # "AGENTCORE_RUNTIME_ARN not configured" even with the runtime fully wired. Routing
    # through the resolvers means a fresh ``agenttui`` after setup works with ZERO env
    # exports, while an explicit env var (on-the-fly override) still wins. SSM-unreachable
    # safely returns '' → the SEC-01 fail-fast below is preserved (no live default).
    # FIX-25: honor any explicit budget the LLM/user set in the chat call (0 ⇒ "unset",
    # fall back to the chat default). No clamping — the operator's number is authoritative.
    from amlas.cloud.dispatch import make_session_id, run_cloud_experiment

    budget_overrides = {
        "trials": trials if trials and trials > 0 else None,
        "wall_clock_minutes": (
            wall_clock_minutes if wall_clock_minutes and wall_clock_minutes > 0 else None
        ),
        "max_rows": max_rows if max_rows and max_rows > 0 else None,
    }
    try:
        dispatch_config = _resolve_dispatch_config(
            _CLOUD_DISPATCH_CONFIG, budget_overrides=budget_overrides
        )
    except (ValueError, FileNotFoundError) as exc:
        # SEC-01 fail-fast (unset runtime ARN / session bucket) → clean tool error.
        return {"status": "error", "error": str(exc)}

    events_path = __import__("os").environ.get("AMLAS_DISPATCH_EVENTS_PATH") or None

    # : PRE-MINT the session id so the chat turn can return it immediately and the
    # background job / bus job / CloudWatch poller all key off the SAME id.
    session_id = make_session_id(resolved_project)

    def _invoke() -> Any:
        """The blocking InvokeAgentRuntime stream — runs synchronously here (fallback) or
        inside the background JobManager job. Passing the pre-minted ``session_id`` keeps the
        run correlated to what the chat turn already returned."""
        return run_cloud_experiment(
            effective_description,
            dataset_path=resolved_dataset,
            project_id=resolved_project,
            target=resolved_target,
            primary_metric=resolved_metric,
            config=dispatch_config,
            session_id=session_id,
            s3_client=_CLOUD_S3_CLIENT,
            agentcore_client=_CLOUD_AGENTCORE_CLIENT,
            # FIX-39: durable event stream (env-driven). When AMLAS_DISPATCH_EVENTS_PATH is set,
            # every streamed event (incl. the terminal run_completed + arbiter_report) is written
            # to that JSONL as it arrives — so a killed/timed-out caller recovers the graded terminal
            # from disk instead of losing it. run_cloud_experiment also reads the env directly.
            events_path=events_path,
        )

    def _capture_result(result: Any) -> None:
        """Capture the FULL DispatchResult for the observability accessor (FIX-16) and flip the
        ``cloud_swarm`` agent activity to its terminal status. Best-effort, never raises."""
        try:
            global _LAST_DISPATCH_RESULT
            _LAST_DISPATCH_RESULT = result
        except Exception:  # nosec B110
            pass
        if bus is not None:
            try:
                from amlas.tui.context.models import AgentActivity

                run_status = getattr(result, "status", None)
                # Map the run's terminal status onto the activity: accepted/completed →
                # complete; anything else (error/budget_exhausted/None) → failed.
                act_status = "complete" if run_status in ("accepted", "completed", "run_completed") else "failed"
                bus.set_agent_activity(
                    AgentActivity(
                        agent_name="cloud_swarm",
                        session_id=session_id,
                        run_id=session_id,
                        last_model="AgentCore swarm",
                        aws_services=["Bedrock", "AgentCore", "S3"],
                        status=act_status,
                    )
                )
            except Exception:  # nosec B110
                pass

    # ── publish RUNNING state BEFORE dispatch so the Dashboard lights up instantly ──────
    # Previously these fired only after the blocking run returned, so the
    # Dashboard didn't engage until the run was already done.) The ActiveJob + agent
    # activity are keyed on the pre-minted session_id; the FIX-24 trace poller engages on
    # the bus "job" event with status=running + run_id, tailing the session's CloudWatch
    # trace while chat stays interactive. Best-effort, never raises.
    if bus is not None:
        try:
            from amlas.tui.context.models import AgentActivity

            bus.set_agent_activity(
                AgentActivity(
                    agent_name="cloud_swarm",
                    session_id=session_id,
                    run_id=session_id,
                    last_model="AgentCore swarm",
                    aws_services=["Bedrock", "AgentCore", "S3"],
                    status="running",
                )
            )
        except Exception:  # nosec B110
            pass
        try:
            bus.set_job(
                job_id=session_id,
                kind="cloud_swarm",
                status="running",
                run_id=session_id,
                started_by="agent",
            )
        except Exception:  # nosec B110
            pass

    # ── background the blocking invoke as a JobManager job ──────────────────
    # When the app injected its ThreadExecutor-backed manager, submit the invoke and return
    # instantly: the chat worker frees up, the Dashboard tracks progress, and the terminal
    # event folds back onto the bus when the job completes (app._on_job_event → set_job,
    # keyed on run_id in the job args). When NO manager is registered (headless / CLI /
    # tests), run synchronously — behavior byte-identical to previous.
    jm = _CLOUD_JOB_MANAGER
    if jm is not None:
        def _job_body(progress_cb: Any, _cancel_token: Any) -> Any:
            try:
                progress_cb(0.05, "dispatching to AgentCore swarm")
            except Exception:  # nosec B110
                pass
            result = _invoke()
            _capture_result(result)
            return result

        try:
            jm.submit(
                "cloud_swarm",
                _job_body,
                args={"run_id": session_id, "started_by": "agent"},
                job_id=session_id,
            )
        except Exception as exc:
            # If submission itself fails, fall back to synchronous so the user still gets a run.
            logger.warning("cloud dispatch job submit failed (%s); running inline", exc)
        else:
            return {
                "status": "dispatched",
                "session_id": session_id,
                "dataset_uri": f"(staging to session {session_id})",
                "run_status": "running",
            }

    # ── synchronous fallback (no job manager) — blocks until terminal, as before ────────
    try:
        result = _invoke()
    except (ValueError, FileNotFoundError) as exc:
        return {"status": "error", "error": str(exc)}
    _capture_result(result)

    return {
        "status": "dispatched",
        "session_id": result.session_id,
        "dataset_uri": result.dataset_uri,
        "run_status": result.status,
    }
