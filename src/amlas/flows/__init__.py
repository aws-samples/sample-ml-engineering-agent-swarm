"""Flow registry scaffold (roadmap M1 work-item 6; doc 05 §3.2 / §9.1 ``flows/``).

A flow is an *opinionated, multi-step ML sequence* the control plane dispatches to the
orchestration plane (doc 05 §3.2). M1 lands the **registry scaffold** (``flow_id → Flow``)
and registers the canonical M1 flows; later milestones add the lifecycle/stage flows
(``imputation_flow`` M2, ``baseline_train_flow`` M3, ``deploy_flow`` M4, …).

A ``Flow`` is a thin descriptor — ``flow_id`` + a human description + the
``OrchestrationBackend`` it runs on + spend/irreversibility metadata (the same flags
Plan Mode previews). The actual execution goes through ``OrchestrationBackend.run_flow``
(the unified LocalDriver, doc 05 §5); this registry just makes flows discoverable +
dispatchable by id. No boto3 / no execution here.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["FLOWS", "Flow", "get_flow", "list_flows", "register_flow"]


@dataclass(frozen=True)
class Flow:
    """A registered flow descriptor (``flow_id`` → metadata).

    Args:
        flow_id: Stable id the control plane dispatches by (e.g. ``"exploration_loop"``).
        description: One-line human description (doc 06 flow table).
        backend_port: The port the flow runs on (``"OrchestrationBackend"`` at M1).
        spend_incurring: Whether the flow spends money (wrapped by ``plan_approve_execute``).
        irreversible: Whether the flow changes production / is irreversible.
        cross_cutting: True for the cross-cutting wrappers (plan_approve_execute,
            project_init_flow) that are not a lifecycle stage.
    """

    flow_id: str
    description: str
    backend_port: str = "OrchestrationBackend"
    spend_incurring: bool = False
    irreversible: bool = False
    cross_cutting: bool = False
    milestone: str = "M1"


# The process-wide flow registry (``flow_id → Flow``).
FLOWS: dict[str, Flow] = {}


def register_flow(flow: Flow) -> Flow:
    """Register ``flow`` under its ``flow_id`` (idempotent — last write wins)."""
    FLOWS[flow.flow_id] = flow
    return flow


def get_flow(flow_id: str) -> Flow | None:
    """Return the registered :class:`Flow` for ``flow_id`` (or ``None``)."""
    return FLOWS.get((flow_id or "").strip().lower())


def list_flows() -> list[Flow]:
    """Return all registered flows (registration order is not guaranteed)."""
    return list(FLOWS.values())


# ── M1 flow registrations ────────────────────────────────────────────────────
# The end-to-end driver (the unified exploration loop on the LocalDriver, doc 05 §5).
register_flow(Flow(
    flow_id="exploration_loop",
    description="End-to-end R→D1→D2→Arbiter retreat loop (the unified OrchestrationBackend driver)",
    spend_incurring=True,
    irreversible=False,
))
# The absorbed MLPipeline deterministic path (no-LLM mode of the same driver).
register_flow(Flow(
    flow_id="ml_pipeline",
    description="Deterministic no-LLM pipeline (research→plan→train→evaluate→decide) — absorbed MLPipeline",
    spend_incurring=True,
    irreversible=False,
))
# The cross-cutting Claude-Code-native wrappers (doc 06 §5.3).
register_flow(Flow(
    flow_id="plan_approve_execute",
    description="Cross-cutting wrapper: preview plan → approval gate → execute (T-CTL-04 → T-CTL-03)",
    backend_port="GuardrailHooks",
    spend_incurring=False,
    cross_cutting=True,
))
register_flow(Flow(
    flow_id="project_init_flow",
    description="Cross-cutting /init: scan repo+data → draft .agenticml.yaml project memory (T-CTL-02)",
    backend_port="ControlPlane",
    spend_incurring=False,
    cross_cutting=True,
))
# ``custom_command_flow`` (M6, T-CTL-01/US-CC.02): the cross-cutting authored-slash-command beat —
# author/load a markdown ``.agenticml/commands/*.md`` command then EXPAND it into its flow/tool
# sequence for the control plane to dispatch. Authoring/expansion is read/write of a local file +
# pure reasoning (CHEAP); the EXPANDED flow it dispatches carries its OWN approval gate, so the
# custom-command beat itself is not spend-incurring (ADR-016). doc 05 §3.1d.
register_flow(Flow(
    flow_id="custom_command_flow",
    description="Cross-cutting: author/load a markdown custom command (.agenticml/commands) → expand to its flow/tool sequence (T-CTL-01, ADR-016)",
    backend_port="ControlPlane",
    spend_incurring=False,
    cross_cutting=True,
    milestone="M6",
))

# ── M2 flow registrations (DQ + ETL orchestration GAP) ────────────────────────
# ``data_quality_flow``: profile → recommend rules → evaluate → gate/report over the
# DataQualityBackend port (in-proc 6-dim/pandera offline ↔ DataBrew/Glue-DQ at scale).
register_flow(Flow(
    flow_id="data_quality_flow",
    description="Profile → recommend DQ rules → evaluate → gate/report (DataQualityBackend port)",
    backend_port="DataQualityBackend",
    spend_incurring=False,
    milestone="M2",
))
# ``imputation_flow``: the signature fixed-flow / swappable-module pattern (doc 05 §4.5).
# diagnose missingness → recommend → apply (SWAPPABLE module) → validate; synthetic-fill
# enforces the 3-pillar AND-gate. Apply-at-scale (port/DataBrew) is spend-incurring.
register_flow(Flow(
    flow_id="imputation_flow",
    description="diagnose missingness → recommend → apply (swappable module) → validate (DataQualityBackend.impute)",
    backend_port="DataQualityBackend",
    spend_incurring=True,
    milestone="M2",
))

# ── M3 flow registrations (feature + train/tune/eval depth) ───────────────────
# ``feature_engineering_flow``: propose → build/select → CAAFE validator-loop →
# feature-store ingest → PIT dataset → validate (FeatureStoreBackend port; PIT no-leakage).
register_flow(Flow(
    flow_id="feature_engineering_flow",
    description="propose → build/select → CAAFE validate → ingest → PIT dataset (FeatureStoreBackend, no-leakage)",
    backend_port="FeatureStoreBackend",
    spend_incurring=False,
    milestone="M3",
))
# ``baseline_train_flow``: train baseline → deterministic Arbiter grade → log → register.
# Spend-incurring (training) → wrapped with plan_approve_execute; Arbiter non-delegated (ADR-003).
register_flow(Flow(
    flow_id="baseline_train_flow",
    description="train baseline (TrainingBackend) → Arbiter grade (non-delegated) → log → register; plan_approve_execute-gated",
    backend_port="TrainingBackend",
    spend_incurring=True,
    irreversible=False,
    milestone="M3",
))
# ``hpo_flow``: define search space → run HPO (Optuna local / AMT managed) → eval best →
# log → compare. Spend-incurring → wrapped with plan_approve_execute.
register_flow(Flow(
    flow_id="hpo_flow",
    description="search space → run HPO (TuningBackend: Optuna/AMT) → eval best → log → compare; plan_approve_execute-gated",
    backend_port="TuningBackend",
    spend_incurring=True,
    irreversible=False,
    milestone="M3",
))
# ``evaluation_flow``: offline metrics + the deterministic Arbiter verdict + model-comparison
# table over ExperimentTracking. Read-only / cheap → not approval-gated. Arbiter non-delegated.
register_flow(Flow(
    flow_id="evaluation_flow",
    description="offline metrics + Arbiter verdict (non-delegated) + model-comparison table (ExperimentTrackingBackend)",
    backend_port="EvaluationBackend",
    spend_incurring=False,
    milestone="M3",
))

# ── M4 flow registrations (Deployment GAP — the priority capability, ADR-007) ──
# ``readiness_flow``: the Stage-9 go/no-go gate — run the 10 production-readiness gates
# (run_readiness_gate per gate) → build risk register → compile readiness review. Read-only/
# pure-reasoning (it provisions nothing); a failing gate BLOCKS the downstream deploy_flow.
register_flow(Flow(
    flow_id="readiness_flow",
    description="10 production-readiness gates → risk register → review go/no-go (DeploymentBackend/Governance/Guardrail)",
    backend_port="DeploymentBackend",
    spend_incurring=False,
    irreversible=False,
    milestone="M4",
))
# ``deploy_flow``: (readiness pass) → package_model → choose strategy → deploy_endpoint(mode) →
# smoke invoke_endpoint → (optional) recommend_inference_config → cut_release → wire monitoring,
# with ROLLBACK on failure. Deploy/delete are expensive + IRREVERSIBLE → wrapped with
# plan_approve_execute (the prime approval-gate consumer; doc 05 §8.8; ADR-007/014).
register_flow(Flow(
    flow_id="deploy_flow",
    description="package → strategy → deploy_endpoint(mode) → smoke invoke → release → wire monitoring; rollback on fail; plan_approve_execute-gated (irreversible)",
    backend_port="DeploymentBackend",
    spend_incurring=True,
    irreversible=True,
    milestone="M4",
))

# ── M5 flow registrations (Monitoring + the closed-loop retrain→retreat, ADR-008) ─
# ``monitor_flow``: the Stage-11 observe sequence — setup baseline/schedule → collect metrics
# → analyze_drift (4 families) → configure alerts → compile report → evaluate_retrain_trigger.
# Read/observe-heavy (CHEAP_READ — it provisions nothing irreversible; the managed schedule-
# setup sub-step is the only spend, gated within the flow), so it runs without a blocking prompt.
# Its terminal output is the closed-loop retrain-trigger DECISION the Monitoring Lead routes to
# ``improve_retrain_flow`` (the production signal re-entering the unified retreat loop, ADR-008).
register_flow(Flow(
    flow_id="monitor_flow",
    description="setup/schedule → metrics → analyze_drift(4 families) → alerts → report → evaluate_retrain_trigger (MonitoringBackend; the closed-loop decision, ADR-008)",
    backend_port="MonitoringBackend",
    spend_incurring=False,
    irreversible=False,
    milestone="M5",
))
# ``improve_retrain_flow``: the Stage-12 loop-closer — analyze_production_failures → carry
# learnings to Memory → RETRAIN by RE-ENTERING the M1 unified exploration retreat loop (ADR-008;
# NOT a parallel path) with the deterministic Arbiter as the non-delegated gate (ADR-003) →
# readiness re-gate (M4 readiness_flow) → redeploy (M4 deploy_flow) — or retire_model. Retrain +
# redeploy are EXPENSIVE + IRREVERSIBLE → wrapped with plan_approve_execute (ADR-014; aborts
# before any retrain/redeploy on denial). It runs on the OrchestrationBackend (the retreat loop).
register_flow(Flow(
    flow_id="improve_retrain_flow",
    description="failures → carry learnings to Memory → retrain by RE-ENTERING the unified retreat loop (Arbiter non-delegated) → readiness re-gate → redeploy / retire; plan_approve_execute-gated (irreversible, ADR-008/003)",
    backend_port="OrchestrationBackend",
    spend_incurring=True,
    irreversible=True,
    milestone="M5",
))

# ── M6 flow registrations (Research/framing front-end + RetrievalBackend, ADR-017) ─
# ``research_intel_flow`` (Stage 0): index research corpus (RetrievalBackend) → scan external +
# indexed research → triage by reproducibility → generate digest → backlog. All CHEAP_READ
# (read/reasoning; the RetrievalBackend is local TF-IDF offline ↔ Bedrock KB managed) — runs
# without an approval prompt; its terminal output is the curated research digest + backlog.
register_flow(Flow(
    flow_id="research_intel_flow",
    description="index corpus (RetrievalBackend) → scan external+indexed research → triage reproducibility → digest → backlog (CHEAP_READ, ADR-017)",
    backend_port="RetrievalBackend",
    spend_incurring=False,
    irreversible=False,
    milestone="M6",
))
# ``research_scan_flow`` (Stage 0, gap-closure §G5): the SCHEDULED/recurring sibling of
# ``research_intel_flow`` — run ONE cadence-driven tick of the ResearchScanScheduler (invoke the
# existing Stage-0 scan per query → aggregate a deduped digest → record/deliver). Beyond on-demand,
# this closes the "every Monday scan + digest" story; the hosted cadence is the deploy-time
# EventBridge rule (infra/cdk_stacks/research_scan_stack.py). All CHEAP_READ (read-only public
# research sources; no AWS spend — ``schedule_research_scan`` is already CHEAP_READ, approvals.py:91),
# so it never routes through the approval gate. Dispatchable on all four surfaces by convention
# (the capability funnel resolves ``run_research_scan_flow``).
register_flow(Flow(
    flow_id="research_scan_flow",
    description="scheduled Stage-0 research scan tick: run scan per query → aggregate deduped digest → record/deliver; cadence-driven (EventBridge), idempotent, CHEAP_READ (gap-closure §G5)",
    backend_port="RetrievalBackend",
    spend_incurring=False,
    irreversible=False,
    milestone="M6",
))
# ``business_framing_flow`` (Stage 1→2): capture business problem → KPI → ML-necessity →
# business go/no-go gate → problem-type → success metrics → problem_card → stage-exit. All
# CHEAP_READ pure-reasoning — turns an NL business problem into a framed problem_card.json.
register_flow(Flow(
    flow_id="business_framing_flow",
    description="capture problem → KPI → ML-necessity → go/no-go → problem-type → metrics → problem_card → stage-exit (CHEAP_READ pure-reasoning)",
    backend_port="ControlPlane",
    spend_incurring=False,
    irreversible=False,
    milestone="M6",
))
# ``solution_survey_flow`` (Stage 3): list JumpStart/Bedrock FM (ModelHubBackend) → survey
# managed services → research-gate candidates → ranked candidate shortlist. CHEAP_READ (the
# ModelHub list is a catalog read; the rest is pure reasoning) — the prototype-stage shortlist.
register_flow(Flow(
    flow_id="solution_survey_flow",
    description="list JumpStart/FM (ModelHubBackend) → survey managed services → research-gate → ranked candidate shortlist (CHEAP_READ)",
    backend_port="ModelHubBackend",
    spend_incurring=False,
    irreversible=False,
    milestone="M6",
))
# ``data_collection_flow`` (Stage 4): inventory via run_sql (QueryBackend, CHEAP_READ) → decide
# labeling strategy (pure reasoning) → create labeling job (LabelingBackend — the IRREVERSIBLE/
# spend step) → poll progress. The labeling-job creation is WRAPPED with plan_approve_execute
# (ADR-014; aborts before any Ground-Truth spend on denial), so the flow is gated/irreversible.
register_flow(Flow(
    flow_id="data_collection_flow",
    description="inventory (QueryBackend run_sql) → labeling strategy → create labeling job (LabelingBackend) → poll progress; labeling-job APPROVAL-gated (irreversible, ADR-014)",
    backend_port="LabelingBackend",
    spend_incurring=True,
    irreversible=True,
    milestone="M6",
))

# ── M7 flow registration (Surfaces parity — the cross-cutting git/PR flow) ────
# ``devops_git_flow`` (T-GIT-02 → T-GIT-01): the conversational "open a PR for this
# spec/pipeline/model-card" sequence the NEW devops_git_agent owns (US-CC.08; doc 06 §1.3).
# commit_artifact stages+commits locally (BOUNDED_COMPUTE), then open_pr PUSHES the branch +
# opens the GitLab MR — a REAL external side-effect → the flow is spend-incurring + irreversible
# so the M7 capability funnel routes it through plan_approve_execute (a denial aborts BEFORE the
# push). It is ORCHESTRATE-only (git/glab via subprocess; no formal AWS port). doc 06 Group-6.
register_flow(Flow(
    flow_id="devops_git_flow",
    description="commit_artifact (T-GIT-02) → open_pr (T-GIT-01): commit ML artifacts then push+open a GitLab MR; push/MR APPROVAL-gated (irreversible, US-CC.08, devops_git_agent)",
    backend_port="VCS",
    spend_incurring=True,
    irreversible=True,
    cross_cutting=True,
    milestone="M7",
))
