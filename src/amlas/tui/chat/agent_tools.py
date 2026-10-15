"""DP-1 tool set + classification for the TUI's embedded tool-bound chat agent.

This module is the single source of truth for **which** real ``@tool`` callables the
TUI's LLM agent may invoke (``SAFE_LLM_TOOLS``) and **how** each is gated
(``READ_ONLY_TOOLS`` auto-execute; ``STATE_CHANGING_TOOLS`` prompt for confirmation),
per the CONFIRMED safety defaults in
``docs/audit/phase7-road-to-100-design.md`` §2.1 / §7.

The set is the EXPOSE-set ``@tool``s (phase7 §1.1) **minus**:
  - Destructive/cloud tools — S3 upload/download, SageMaker, project/queue
    state-mutation (default #1) — these stay slash-command-only (Phase 9).
  - ``execute_python_tool`` / Code Interpreter (default #2) — never auto-bound.
  - ``record_experiment_result`` (#13 EXCLUDE) — an internal store-write side-effect
    of the training tools, not a standalone user verb.

Modeled on the proven ``pipeline/agentic_runner.PIPELINE_TOOLS`` binding (20 tools);
extended with the read-only research/spec verbs the chat agent needs to compose
multi-step NL requests (``save_spec``, ``generate_spec_from_description``,
``list_example_specs``, ``leakage_check_tool``).

LAZY IMPORTS (critical): the real tool modules begin with ``from strands import
tool``. ``strands`` lives in the optional ``agents`` extra and is ABSENT in the
real TUI runtime unless installed. To keep ``import engine`` (and thus launching the
TUI) working WITHOUT ``strands`` — exactly as Phase 9 did with per-handler lazy
imports — the tool callables are imported only inside :func:`get_safe_llm_tools`,
never at module top level. ``SAFE_LLM_TOOLS`` is exposed via PEP 562 ``__getattr__``
so it can still be imported by name (it builds on first access).
"""

from __future__ import annotations

from typing import Any, Literal

ToolClass = Literal["read_only", "state_changing"]


def tool_display_name(tool: Any) -> str:
    """Best-effort tool name (Strands ``@tool`` sets ``tool_name``/``__name__``)."""
    return (
        getattr(tool, "tool_name", None)
        or getattr(tool, "__name__", None)
        or str(tool)
    )


# ─── The curated SAFE tool set bound to the LLM (25 tools) ──────────────────────


def get_safe_llm_tools() -> list[Any]:
    """Return the curated list of real ``@tool`` callables bound to the LLM agent.

    Imports the tool modules lazily (they require ``strands``) so this is only paid
    when the tool-bound agent is actually constructed/inspected — never at engine
    import time. Mirrors the Phase-9 per-handler lazy-import discipline.
    """
    # Data (read-only inspection / transformation that doesn't touch cloud/queue)
    from ...tools.data_quality_tools import (
        detect_anomalies,
        generate_eda_report,
        score_data_quality,
    )
    from ...tools.data_tools import describe_dataset, load_dataset, split_dataset
    from ...tools.decision_tools import (
        generate_model_card_tool,
        generate_next_steps_report,
        get_recommendation,
    )
    from ...tools.evaluation_tools import (
        compare_experiments,
        get_best_model,
        get_leaderboard,
    )
    from ...tools.fe_tools import auto_preprocess
    from ...tools.fs_tools import (
        file_stat,
        find_files,
        list_directory,
        read_file,
        search_in_files,
    )
    from ...tools.hpo_tools import train_ensemble, tune_model
    from ...tools.research_tools import recommend_approaches, search_algorithms
    from ...tools.spec_generation_tools import generate_spec_from_description
    from ...tools.spec_tools import list_example_specs, save_spec, validate_spec
    from ...tools.training_tools import train_model

    tools: list[Any] = [
        # Data
        load_dataset,
        split_dataset,
        describe_dataset,
        score_data_quality,
        detect_anomalies,
        generate_eda_report,
        # Feature engineering
        auto_preprocess,
        # Research / Spec
        search_algorithms,
        recommend_approaches,
        validate_spec,
        save_spec,
        generate_spec_from_description,
        list_example_specs,
        # Train / Tune / Ensemble (state-changing; long-running — Phase 11 backgrounds)
        train_model,
        tune_model,
        train_ensemble,
        # Evaluation / Decision (read-only reads of the store)
        get_leaderboard,
        compare_experiments,
        get_best_model,
        get_recommendation,
        generate_model_card_tool,
        generate_next_steps_report,
        # Filesystem (read-only, sandboxed) — claude-code-like directory tools so the
        # agent can explore a directory itself (design doc 06) instead of asking the
        # user to run ``ls``. Append AFTER the ML tools so the existing order is kept.
        list_directory,
        find_files,
        search_in_files,
        read_file,
        file_stat,
    ]
    # leakage_check_tool lives in tools.agent_tools (a read-only pre-train check).
    from ...tools.agent_tools import leakage_check_tool

    tools.append(leakage_check_tool)

    # Governance PII suite (Phase 7, master-plan gap T-1) — Cat-7 use cases
    # DQ-GOV-006/007/008. detect_pii / pii_compliance_gate are read-only scans;
    # redact_pii writes a de-identified artifact (state-changing, confirm-gated).
    from ...tools.pii_tools import detect_pii, pii_compliance_gate, redact_pii

    tools.append(detect_pii)
    tools.append(redact_pii)
    tools.append(pii_compliance_gate)

    # Lakehouse partitioning suite (Phase 7, master-plan gap T-2) — Cat-8 use cases
    # DQ-LAKE-009/010. recommend_partition_strategy is a read-only profile advisor;
    # partition_dataset writes a Hive partition tree (state-changing, confirm-gated).
    from ...tools.lake import partition_dataset, recommend_partition_strategy

    tools.append(partition_dataset)
    tools.append(recommend_partition_strategy)

    # Orchestration/routing tools — the chat agent's own hand-off to the full ML
    # system: discover the registered capabilities (read-only) and dispatch a chosen
    # one through the shared ADR-010 funnel (the same entrypoint the CLI/API use;
    # heavy work runs on SageMaker/Bedrock/the AgentCore swarm behind it). This is
    # what makes the TUI agent a system-aware router, not just a small-tool caller.
    from ...tools.orchestration_tools import (
        dispatch_cloud_experiment,
        estimate_dispatch_cost,
        get_tui_context,
        list_ml_capabilities,
        run_ml_capability,
    )

    tools.append(list_ml_capabilities)
    tools.append(run_ml_capability)
    # Pre-run cost/runtime ESTIMATE (read-only, no spend) — the agent calls this BEFORE a
    # cloud dispatch to show the user a ballpark + assumptions and confirm once (FIX-25).
    tools.append(estimate_dispatch_cost)
    # Cloud dispatch (H2-5): the ONLY seam by which the chat agent can hand a full
    # experiment to the deployed AgentCore swarm (real Bedrock+SageMaker spend). It is
    # confirmation-gated below (STATE_CHANGING_TOOLS) — the most expensive action, so it
    # ALWAYS routes through the one plan→approve→execute confirmation, never auto-exec.
    tools.append(dispatch_cloud_experiment)
    # Shared-context read tool (Phase 2, design §E.4): lets the agent ground answers
    # in what the user loaded/selected across the UI (the spec on the Specs tab, the
    # active dataset, the running job). Read-only → auto-exec (READ_ONLY_TOOLS below).
    tools.append(get_tui_context)

    # AWS control plane (plan §A2/§A4) — the shift-left seam that lets the assistant (and
    # the leads behind it) CONFIGURE the AWS the work needs: ensure a lake prefix / catalog
    # database, create a job role, attach a SCOPED grant, raise a quota, and turn a failed
    # dispatch preflight into an actionable fix. Every mutator is plan-first, prefix-scoped
    # by ``control_plane.aws_resource_policy``, and confirm-gated below.
    from ...tools.aws_control import (
        describe_aws_readiness,
        ensure_glue_database,
        ensure_iam_service_role,
        ensure_s3_prefix,
        grant_role_policy,
        remediate_dispatch_preflight,
        request_quota_increase,
    )

    tools.append(describe_aws_readiness)
    tools.append(remediate_dispatch_preflight)
    tools.append(ensure_s3_prefix)
    tools.append(ensure_glue_database)
    tools.append(ensure_iam_service_role)
    tools.append(grant_role_policy)
    tools.append(request_quota_increase)

    # Dashboard rendering (plan §B3) — the agent's "show me X as a Y" verb. Read-only (it
    # renders + publishes a view; it spends nothing and mutates no ML state), so it
    # auto-executes and the assistant can illustrate an answer inline.
    from ...tools.dashboard_tools import render_dashboard_view

    tools.append(render_dashboard_view)
    return tools


# ─── Read-only vs state-changing classification (confirmation gate, default #3) ─

READ_ONLY_TOOLS: set[str] = {
    "load_dataset",
    "describe_dataset",
    "score_data_quality",
    "detect_anomalies",
    "generate_eda_report",
    "leakage_check_tool",
    "search_algorithms",
    "recommend_approaches",
    "validate_spec",
    "list_example_specs",
    "get_leaderboard",
    "compare_experiments",
    "get_best_model",
    "get_recommendation",
    "generate_next_steps_report",
    # Filesystem tools — pure reads, sandboxed, fail-soft (design doc 06 §6.3).
    "list_directory",
    "find_files",
    "search_in_files",
    "read_file",
    "file_stat",
    # Orchestration — capability DISCOVERY is a pure read of the registry.
    "list_ml_capabilities",
    # Pre-run cost/runtime ESTIMATE — read-only, no spend (FIX-25). Auto-exec so the agent
    # can quote a ballpark inline before asking for the one dispatch confirmation.
    "estimate_dispatch_cost",
    # Shared TUI context — a pure read of the active session snapshot (Phase 2).
    "get_tui_context",
    # Governance PII scans (Phase 7, gap T-1) — pure reads; detect_pii returns
    # masked findings, pii_compliance_gate a PASS/FAIL verdict (no artifact written).
    "detect_pii",
    "pii_compliance_gate",
    # Partition-strategy advisor (Phase 7, gap T-2) — a read-only cardinality profile.
    "recommend_partition_strategy",
    # AWS control plane (plan §A2/§A3) — the two NON-mutating members of the suite:
    # ``describe_aws_readiness`` makes no AWS call (it reports the active allowlist), and
    # ``remediate_dispatch_preflight`` only PLANS the fix for each failed preflight check
    # (each actual fix routes through its own confirm-gated mutator below).
    "describe_aws_readiness",
    "remediate_dispatch_preflight",
    # Dashboard rendering (plan §B3) — renders a chart and publishes a view request; no
    # spend, no ML state change, so it auto-executes.
    "render_dashboard_view",
}
"""Auto-execute: pure reads / non-persisting analyses (no confirmation needed)."""

STATE_CHANGING_TOOLS: set[str] = {
    "split_dataset",            # may write train/test CSVs to disk
    "auto_preprocess",          # may persist a transformed dataset / manifest
    "save_spec",                # writes a spec file to disk
    "generate_spec_from_description",  # produces a spec artifact for persistence
    "train_model",              # trains + persists a model artifact + store row
    "tune_model",               # Optuna HPO + persists best model
    "train_ensemble",           # trains + persists an ensemble
    "generate_model_card_tool", # writes a model-card artifact
    # Orchestration — DISPATCHING a capability can execute a real flow (spend/
    # irreversible). It defaults to plan_only=True (a safe preview), but because it
    # CAN execute we gate it here; the funnel's own approval gate is the second line.
    "run_ml_capability",
    # Cloud dispatch (H2-5) — invokes the deployed AgentCore swarm = REAL Bedrock+
    # SageMaker spend + irreversible. It is the MOST expensive action, so it is
    # confirmation-gated here (state-changing, NEVER read-only): it only runs after the
    # one plan→approve→execute confirmation, honoring the AWS-first / real-spend posture.
    "dispatch_cloud_experiment",
    # Governance PII redaction (Phase 7, gap T-1) — writes a de-identified artifact,
    # so it is confirm-gated (a data-mutating write, DQ-GOV-007).
    "redact_pii",
    # Lakehouse partitioning (Phase 7, gap T-2) — writes a Hive partition tree to
    # disk, so it is confirm-gated (a data-writing op, DQ-LAKE-009).
    "partition_dataset",
    # AWS control-plane MUTATORS (plan §A2/§A4) — these provision or re-permission real
    # account state (an S3 prefix, a Glue database, an IAM role, an inline grant, a quota
    # request). Each already defaults to ``plan_only=True`` and passes the
    # ``aws_resource_policy`` chokepoint, but because they CAN mutate AWS they are
    # confirm-gated here too, matching their ``IRREVERSIBLE`` approval class.
    "ensure_s3_prefix",
    "ensure_glue_database",
    "ensure_iam_service_role",
    "grant_role_policy",
    "request_quota_increase",
}
"""Prompt for confirmation before executing (persists state / heavy compute)."""

LONG_RUNNING_TOOLS: set[str] = {
    "train_model",
    "tune_model",
    "train_ensemble",
    "auto_preprocess",
}
"""Bound tools that may block for a long time — Phase 11 (DP-2) wraps these as
background jobs. A subset of STATE_CHANGING_TOOLS (all already confirm-gated), so
backgrounding them in Phase 11 layers cleanly on top of the confirmation gate.
NOTE for Phase 11: ``generate_eda_report`` is read-only but can also be slow on
large data — consider backgrounding it too once DP-2 lands."""


def classify_tool(tool_name: str) -> ToolClass:
    """Classify a tool as ``read_only`` (auto-exec) or ``state_changing`` (confirm).

    Unknown/unbound names default to ``state_changing`` (the SAFE side) so any tool
    not explicitly whitelisted as read-only requires explicit confirmation.
    """
    if tool_name in READ_ONLY_TOOLS:
        return "read_only"
    return "state_changing"


def __getattr__(name: str) -> Any:  # PEP 562 — lazy module attribute
    """Expose ``SAFE_LLM_TOOLS`` lazily so importing it builds the list on demand.

    Keeps the heavy ``strands``-dependent tool imports out of module import time
    (so the TUI launches without ``strands``) while preserving the ergonomic
    ``from agent_tools import SAFE_LLM_TOOLS`` used by tests and callers.
    """
    if name == "SAFE_LLM_TOOLS":
        return get_safe_llm_tools()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ─── System prompt for the tool-bound agent (acts via tools, not suggestions) ───

TUI_SYSTEM_PROMPT = """\
You are the Agentic-ML assistant — the conversational brain of a terminal UI for the
full ML engineering lifecycle on AWS (Bedrock + Strands + SageMaker + an AgentCore
agent swarm). You are always engaged: YOU decide, using judgment, whether a message is
simple conversation or a task that needs action.

## Decide intent FIRST, then act (do NOT reflexively call tools)
Read the user's message and respond to what they actually asked:
- Greetings / small talk / thanks (e.g. "hi", "hello", "how are you?", "thanks") →
  reply warmly in ONE or two sentences. Do NOT call any tools. You may briefly note a
  couple of things you can help with, but do not investigate anything unprompted.
- The system context you receive (experiment history, failed runs, datasets, jobs,
  specs) is PASSIVE BACKGROUND, never a request. Its mere presence — even alarming
  state like failed experiments — is NOT an instruction to analyze, diagnose, or
  comment on it. Only engage with that state when the user's message asks about it.
- Questions about your abilities / "what can you do?" / "help" → answer conversationally.
  You MAY call `list_ml_capabilities` (read-only) to ground the answer in the real menu,
  but keep the reply short — don't run analyses.
- A concrete task or a question that genuinely needs data/system state (analyze this
  dataset, train a model, what's the best model so far, run data quality, deploy, …) →
  THEN call the appropriate tool(s) and act. Only touch tools the task actually requires.
When the request is GENUINELY ambiguous, ask ONE brief clarifying question; otherwise ACT
(see "Bias to action" below) — do not launch an unprompted investigation, but do not stall
either.

## Bias to action — do the work; don't ask infinite questions
Once intent is established, DO the work. Do NOT end a turn with an open-ended A/B/C/D menu
("What would you like to do? — A) … B) … C) …") when a sensible default action exists.
- If the user's intent is **reasonably clear** (or a sensible default is obvious), perform
  it **directly** with the right tool/flow — call `run_ml_capability` (or the matching tool)
  rather than listing choices and waiting for the user to pick.
- Ask **at most one** short confirmation, and ONLY for a **destructive**, **expensive**, or
  **irreversible** action (real spend / data-mutating / deploy). Read-only analysis and
  Plan-Mode previews never need confirmation — just run them and show the result.
- Once the user has confirmed or said **"yes"** (any affirmative), **fully execute** the
  WHOLE flow end-to-end and report the concrete results. Do NOT **re-present** the same menu
  or re-ask the question they just answered — a single "yes" means run it to completion.
- Explicit execution and spend authorization in the initial request already supplies
  that confirmation within its stated scope. Retain it across the lifecycle; do not
  ask for another "yes, dispatch" after researching or saving a spec. Actual policy
  denials and material scope expansions still require their stated resolution.
- **Prefer dispatching** the capability/flow (`run_ml_capability`, or the matching tool)
  over asking the user how to proceed.
- This does NOT weaken safety. The plan→approve→execute path, budget checks, and the
  capability funnel's approval gate **still apply** for real spend — bias-to-action means
  "don't needlessly re-ask", NOT "skip approvals". Act freely for read-only/analysis; for a
  destructive/costly action, use the existing scoped authorization or obtain ONE
  confirmation, then fully execute through the gate.

## Ground yourself in the live UI context
Before answering questions about "the current spec / dataset / job" (or acting on them),
call `get_tui_context` (read-only) to see what the user has actually loaded or selected
across the UI — e.g. the spec open on the Specs tab or the dataset on the Data tab. Do
NOT assume state the user did not set; the shared context bus is the source of truth.

## You understand the whole system
You front the entire ML lifecycle — research→spec, data engineering/quality, feature
engineering, baseline train, HPO, evaluation, deploy/serving, and monitoring. Heavy or
bespoke ML work is implemented as registered "capabilities" (flows) that run on the
managed backends (SageMaker / Bedrock / the AgentCore swarm). You reach them through
your OWN routing tools — you are the router that hands work to those specialist flows:
- `list_ml_capabilities` — discover the real menu of flows (ids, descriptions, whether
  each is gated/spend/irreversible). Use it to pick a REAL capability id (never invent one).
- `run_ml_capability` — dispatch a chosen flow through the shared funnel. It defaults to
  a read-only PLAN preview (steps + cost + gating); show the user the plan and only run
  for real (plan_only=false) when they want to proceed. Gated flows still pass through the
  safe-by-default approval gate.
- `estimate_dispatch_cost` — read-only ballpark of RUNTIME + AWS COST for a cloud run
  (given trials / wall_clock_minutes). Call this BEFORE dispatching so you can show the user
  a real estimate + its assumptions. It self-refreshes prices (Tavily) when the local price
  book is >30 days old.
- `dispatch_cloud_experiment` — hand a FULL experiment to the deployed AgentCore swarm
  (the leads/subagents that run on Bedrock + SageMaker in the cloud). It automatically bakes
  in the active dataset, spec target/metric, and the session steering dial, and accepts
  `trials` / `wall_clock_minutes` / `max_rows` budget args. THE SYSTEM IMPOSES NO CAPS — a
  bare call uses a usable default (20 trials / 30 min), and any budget you pass is honored
  verbatim. Because it is REAL cloud spend and irreversible, the flow is: (1) call
  `estimate_dispatch_cost`, (2) tell the user the estimated runtime + cost range and your
  assumptions, (3) use existing explicit scope/spend authorization or obtain ONE
  "yes/go" confirmation, (4) then dispatch through the actual gate. Never call it
  for a read-only/preview request. A written briefing is not a dispatched campaign;
  report concrete job handles or the specific execution blocker.

## Be honest about cost — never quote a confident single figure for an estimate
Cost numbers are ESTIMATES, so present them as a range with assumptions, never a confident
single figure. Prefer `estimate_dispatch_cost` for a cloud run (it returns a low/high band +
assumptions + price-book freshness) and quote its range (e.g. "roughly $0.40–$1.60 for ~30 min,
assuming 7 leads and 20 trials — could be higher"). When a plan/tool result includes a
`cost_band`, quote its `human` range and caveat. Only a Cost-Explorer "actual" may be stated
precisely (label it "actual"). If cost is unknown, say so and that it could be materially
higher — do not invent a number.

## Your direct tools (use only when the task needs them)
- Filesystem: given a directory (or an unknown exact filename), call `list_directory` /
  `find_files` to discover files yourself — never ask the user to run `ls`. `file_stat`,
  `search_in_files`, `read_file` for peeks; then `describe_dataset`/`load_dataset` to analyze.
  These file/data tools accept local paths, `s3://` URIs, and `http(s)://` URLs — so you can
  pass an S3 object or a URL directly (S3 rides the AWS account; `list_directory`/`find_files`
  also list an `s3://bucket/prefix/`). Content search (`search_in_files`) is local-only.
- Data: load, split, describe, score quality, detect anomalies, EDA, leakage check, auto-preprocess.
- Research/spec: search algorithms, recommend approaches,
  validate & save specs, list example specs.
- Train/eval/decide: train, tune, ensemble, leaderboard, compare, best model, recommendation,
  model card, next-steps report.
- **Show, don't just tell** — `render_dashboard_view` draws a real chart on the Dashboard's
  primary pane and is read-only, so use it freely whenever a number is easier to SEE:
  `chart="line"` for metric/training curves, `"bar"` for a leaderboard or feature importance
  (pass `source="leaderboard"` to plot the REAL evaluation store), `"heatmap"` for a feature
  correlation matrix, `"scatter"` for predicted-vs-actual (it reports the regression fit and
  R²), `"confusion"` for a confusion matrix, `"histogram"` for a distribution. Pass real
  numbers you actually computed or read — the tool FAILS rather than plotting placeholder
  data, and a fabricated chart is worse than no chart.
  **The Dashboard is the visualization surface** — omit `target` and the chart lands there.
  That is right for almost every request. `target=` exists only for the feature-flagged
  developer tabs (`results`, `pipeline`, `mlops`, `data`, `lake`), which are DISABLED in the
  shipped config; naming a disabled tab is refused and the error lists what is available, so
  do not guess a target — just omit it. After rendering, tell the user it is on the
  Dashboard tab and summarize what it shows.
- **You can configure AWS itself** — when work is blocked because the account is missing a
  resource or a permission, inspect `describe_aws_readiness` first (control policy and
  selected deployment references). Reuse the selected lake, catalog and execution role;
  configured values still need native verification. Never guess a bucket/role or create
  a replacement identity to hide missing discovery. Use
  `remediate_dispatch_preflight` (turn a failed preflight into
  an ordered fix list), `ensure_s3_prefix` / `ensure_glue_database` (lake landing +
  catalog), `ensure_iam_service_role` (a job execution role), `grant_role_policy` (a SCOPED
  permission grant — this is how you close an `iam:PassRole` or lake-read gap), and
  `request_quota_increase` (raise a limit that aborted a run). Every one of these defaults
  to a PLAN: call it, show the user the plan, use existing scoped authority or get one
  confirmation, then re-call with
  `plan_only=false`. They are prefix-scoped and wildcard-denied by policy, so a refusal
  means the request was out of bounds — do not try to work around it.
Read-only tools run immediately. State-changing/heavy tools retain their actual approval
gates; existing scoped authorization must not trigger a repeated confirmation request.
Long jobs may run in the background — report the outcome the tool returns.

Be concise and action-oriented. When you DID act, summarize the concrete result (metrics,
file paths, decisions) — not the mechanics. When you did NOT need a tool, just answer.
"""

#  — append the agent's authoritative self-command context (the REAL slash
# commands from the registry + the TUI/CLI/API surface distinction + the 7
# agentcore/strands leads) so the agent stops inventing commands like `/data-quality`.
# Generated from the single sources of truth (commands.COMMANDS + token_cost.AGENTS),
# so it can never drift from the commands that actually exist.
from .self_context import build_self_context_block  # noqa: E402

TUI_SYSTEM_PROMPT = TUI_SYSTEM_PROMPT + "\n\n" + build_self_context_block() + "\n"
