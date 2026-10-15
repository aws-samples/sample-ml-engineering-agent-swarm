"""Sub-agent DECLARATIONS — the single source of truth for what each sub-agent is made of.

Peer to [`leads/declarations.py`](../leads/declarations.py:1), and the same argument applies one
level down. A sub-agent's tool set, prompt stack, role and description used to be literals
inside its ``build_*`` body, reachable only by CONSTRUCTING the agent — which, with
``model=None``, constructs a real ``BedrockModel``. That made two things impossible:

* an offline harness spec for a sub-agent (the harness ``inlineFunction`` type requires a
  description and an input schema per tool, and ADR-0007 phase 1's true leaves ARE the
  sub-agents); and
* a single source for the tool descriptions the Leads need, which is why
  ``leads/declarations.SUBAGENT_TOOL_DESCRIPTIONS`` had to duplicate all 30 of them.

Both are fixed by lifting the declaration out into data that every consumer reads.

The layer turned out to be highly uniform — verified across all 30 builders, not assumed:
two signature variants and two ``build_agent`` call shapes, and the four builders that accept
``messages`` / ``steering`` / ``problem`` are EXACTLY the four that compose their prompt through
:func:`compose_system_prompt`. So one :attr:`SubagentDeclaration.context_aware` flag captures
the whole difference.

Nothing here constructs an ``Agent``, a model, or a boto3 client.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from strands import Agent

from .._build import build_agent, compose_system_prompt
from ..prompts import stack_prompts
from ..tools.agent_tools import data_verify_tool, leakage_check_tool
from ..tools.data_etl_neutral import (
    apply_imputation_via_port,
    evaluate_dq,
    recommend_dq_rules,
    run_processing_job,
    run_profile,
)
from ..tools.deploy_neutral import (
    build_risk_register,
    compile_model,
    compile_readiness_review,
    cut_release,
    delete_endpoint,
    deploy_endpoint,
    design_production_architecture,
    draft_operational_runbook,
    invoke_endpoint,
    package_model,
    recommend_inference_config,
    rollback_deployment,
    run_ab_test,
    run_batch_transform,
    run_load_test,
    run_readiness_gate,
    run_shadow_deployment,
    select_deployment_strategy,
    update_endpoint_canary,
)
from ..tools.dispatch_preflight import dispatch_preflight
from ..tools.git_tools import commit_artifact, open_pr
from ..tools.governance_neutral import check_data_access, create_model_card, query_lineage
from ..tools.imputation import llm_extrapolate_values
from ..tools.local_tools import execute_python_tool, save_feature_manifest
from ..tools.mlops import (
    approve_model_sm,
    build_sagemaker_pipeline,
    feature_store_get,
    feature_store_ingest,
    inspect_model_monitor,
    list_model_packages_sm,
    register_model_sm,
    run_clarify,
    run_sagemaker_pipeline,
    setup_model_monitor,
)
from ..tools.monitor_neutral import (
    analyze_drift,
    analyze_production_failures,
    build_monitoring_dashboard,
    carry_forward_learnings,
    compile_monitoring_report,
    configure_alert,
    evaluate_retrain_trigger,
    get_metrics,
    get_recent_logs,
    put_custom_metric,
    retire_model,
    track_inference_spend,
)
from ..tools.research.strands_tools import arxiv_only, research_fetch_urls, research_search, web_retrieve
from ..tools.research_neutral import (
    apply_research_gate,
    assess_ml_necessity,
    build_candidate_shortlist,
    build_problem_card,
    business_go_no_go,
    capture_business_problem,
    check_stage_exit,
    create_labeling_job,
    define_business_kpi,
    define_success_metrics,
    generate_research_digest,
    get_labeling_progress,
    list_jumpstart_models,
    run_sql,
    scan_research,
    schedule_research_scan,
    select_problem_type,
    survey_managed_services,
    triage_reproducibility,
)
from ..tools.research_tools import recommend_approaches, search_algorithms
from ..tools.retrieval_neutral import (
    build_metadata_filters,
    build_retrieval_index,
    chunk_documents,
    index_research_corpus,
    query_research_corpus,
    tune_retrieval,
)
from ..tools.spec_tools import list_example_specs, save_spec, validate_spec
from ..tools.synthetic import build_synthesis_spec, llm_synthesize_rows
from ..tools.training_jobs import (
    await_training_job,
    fetch_training_metrics,
    submit_sagemaker_job,
    submit_tabular_training,
)


def tool_name_of(tool: Any) -> str:
    """The name a ``@tool`` object registers itself under."""
    name: str | None = getattr(tool, "tool_name", None)
    return name or str(getattr(tool, "__name__", tool))


def tool_contract_of(tool: Any) -> tuple[str, str, dict[str, Any]]:
    """``(name, description, input_schema)`` for a ``@tool``, readable offline.

    Strands wraps the JSON Schema as ``{"json": {...}}``; the harness ``inlineFunction`` type
    wants the schema itself, so the envelope is unwrapped here rather than at every call site.
    """
    spec = getattr(tool, "tool_spec", None) or {}
    schema = spec.get("inputSchema") or {}
    if isinstance(schema, dict) and set(schema) == {"json"}:
        schema = schema["json"]
    return (
        tool_name_of(tool),
        str(spec.get("description") or "").strip(),
        dict(schema if isinstance(schema, dict) else {}),
    )


@dataclass(frozen=True)
class SubagentDeclaration:
    """Everything needed to compose one sub-agent — readable without building anything.

    Attributes:
        key: The sub-agent's name (also the tool name it registers under on a Lead).
        prompt_sources: Prompt files stacked, in order, into the system prompt.
        tools: ``@tool`` objects bound to the sub-agent, in order.
        role: Model tier.
        description: The a2a/routing description, and the tool description a Lead sees.
        context_aware: When true the builder accepts ``messages`` / ``steering`` / ``problem``
            and the prompt is composed through :func:`compose_system_prompt`. Verified to
            coincide exactly with the composed-prompt set across all 30 builders.
    """

    key: str
    prompt_sources: tuple[str, ...]
    tools: tuple[Any, ...]
    role: str
    description: str
    context_aware: bool = False

    @property
    def declared_tool_names(self) -> tuple[str, ...]:
        """Tool names this sub-agent registers, in order — no ``Agent`` needed."""
        return tuple(tool_name_of(t) for t in self.tools)

    @property
    def tool_contracts(self) -> tuple[tuple[str, str, dict[str, Any]], ...]:
        """Per-tool ``(name, description, input_schema)``, as ``inlineFunction`` requires."""
        return tuple(tool_contract_of(t) for t in self.tools)

    def system_prompt(self, *, steering: Any | None = None, problem: Any | None = None) -> str:
        """Render the system prompt. Pure file reads — no AWS, no network, no ``Agent``.

        ``steering`` / ``problem`` are only meaningful when :attr:`context_aware`; passing them
        to a non-context-aware sub-agent would silently change its prompt away from what its
        builder produces, so they are ignored there and the composition is byte-identical.
        """
        stacked = stack_prompts(*self.prompt_sources)
        if not self.context_aware:
            return stacked
        return compose_system_prompt(stacked, steering=steering, problem=problem)


def build_subagent(
    declaration: SubagentDeclaration,
    *,
    model: Any | None = None,
    name: str | None = None,
    hooks_factory: Callable[[], Sequence[Any]] | None = None,
    messages: Sequence[Any] | None = None,
    steering: Any | None = None,
    problem: Any | None = None,
) -> Agent:
    """Build the real ``Agent`` for ``declaration`` — the single ``build_agent`` call site.

    Behaviour-preserving with the hand-written bodies this replaces. ``messages`` is forwarded
    only for a context-aware sub-agent, because the 26 others never passed it and including it
    would change their ``build_agent`` kwargs.
    """
    kwargs: dict[str, Any] = {
        "name": name or declaration.key,
        "system_prompt": declaration.system_prompt(steering=steering, problem=problem),
        "tools": list(declaration.tools),
        "model": model,
        "role": declaration.role,
        "hooks_factory": hooks_factory,
        "description": declaration.description,
    }
    if declaration.context_aware:
        kwargs["messages"] = messages
    return build_agent(**kwargs)


# -- The thirty sub-agent declarations ------------------------------------
# Extracted verbatim from each builder body rather than transcribed: 30 hand-copied
# entries is a transcription-error surface the equivalence tests would only catch
# after the fact. Several builders already exposed a module-level
# SYSTEM_PROMPT_SOURCES tuple, resolved here so every declaration carries literal
# prompt paths. Ordered by key.

ALGO_HYPOTHESIS = SubagentDeclaration(
    key="algo_hypothesis",
    prompt_sources=(
        "preambles/paranoid_scientist",
        "preambles/thought_guidance",
        "research/algo_hypothesis",
    ),
    tools=(
        research_search,
        research_fetch_urls,
        arxiv_only,
        web_retrieve,
    ),
    role="specialist",
    description="Shortlists 3-5 algorithms with rationale + Optuna search spaces.",
)

ALGORITHM_SPECIALIST = SubagentDeclaration(
    key="algorithm_specialist",
    prompt_sources=(
        "preambles/paranoid_scientist",
        "preambles/thought_guidance",
        "subagents/algo_specialist",
    ),
    tools=(),
    role="planner",
    description="Selects one algorithm from shortlist, fixes config, names ablation target.",
)

BUSINESS_GOAL_ADVISOR = SubagentDeclaration(
    key="business_goal_advisor",
    prompt_sources=(
        "preambles/paranoid_scientist",
        "preambles/thought_guidance",
        "subagents/business_goal_advisor",
    ),
    tools=(
        capture_business_problem,
        define_business_kpi,
        business_go_no_go,
    ),
    role="specialist",
    description="Stage-1 business-goal advisor: captures the business problem, defines a measurable KPI, and renders the business go/no-go gate (pure reasoning — no spend).",
    context_aware=True,
)

DATA_COLLECTION_AGENT = SubagentDeclaration(
    key="data_collection_agent",
    prompt_sources=(
        "preambles/paranoid_scientist",
        "preambles/thought_guidance",
        "subagents/data_collection_agent",
    ),
    tools=(run_sql,),
    role="bounded",
    description="Bounded Stage-4 data-collection agent: runs NL→SQL over the lake (QueryBackend) to inventory available + labeled data (CHEAP_READ); informs the labeling strategy — creates no labeling job.",
)

DATA_INTEGRITY = SubagentDeclaration(
    key="data_integrity",
    prompt_sources=(
        "preambles/paranoid_scientist",
        "preambles/data_verification_checklist",
        "subagents/data_integrity",
    ),
    tools=(
        data_verify_tool,
        leakage_check_tool,
    ),
    role="reviewer",
    description="Produces a block-or-pass data integrity verdict. Calls data_verify + leakage_check tools only.",
)

DEPLOYMENT_AGENT = SubagentDeclaration(
    key="deployment_agent",
    prompt_sources=(
        "preambles/paranoid_scientist",
        "preambles/thought_guidance",
        "subagents/deployment_agent",
    ),
    tools=(
        package_model,
        deploy_endpoint,
        invoke_endpoint,
        update_endpoint_canary,
        run_batch_transform,
        run_shadow_deployment,
        run_ab_test,
        rollback_deployment,
        delete_endpoint,
        compile_model,
        select_deployment_strategy,
        cut_release,
    ),
    role="bounded",
    description="Bounded executor of the Stage-10 deployment tools (package/deploy/invoke/canary/rollback/release) over the DeploymentBackend port; returns endpoint refs + predictions.",
)

DEVOPS_GIT_AGENT = SubagentDeclaration(
    key="devops_git_agent",
    prompt_sources=(
        "preambles/paranoid_scientist",
        "preambles/thought_guidance",
        "subagents/devops_git_agent",
    ),
    tools=(
        commit_artifact,
        open_pr,
    ),
    role="bounded",
    description="Conversational git/PR authoring for ML artifacts (US-CC.08): commits specs/pipelines/model-cards (commit_artifact) + opens a GitLab MR with an auto-written title/description/checklist (open_pr), linking the MR to the run. Push/MR is approval-gated; refuses force-push to protected branches.",
)

DQ_ASSESSMENT = SubagentDeclaration(
    key="dq_assessment",
    prompt_sources=(
        "preambles/paranoid_scientist",
        "preambles/data_verification_checklist",
        "subagents/dq_assessment",
    ),
    tools=(
        run_profile,
        recommend_dq_rules,
        evaluate_dq,
    ),
    role="reviewer",
    description="Orchestrates DQ assessment by DISPATCHING profiling + rule evaluation to Glue DQ / SageMaker Processing (run_profile → recommend_dq_rules → evaluate_dq); returns a DQ-report JSON. Does not scan data in the sandbox (ADR-0002).",
)

FEATURE_HYPOTHESIS = SubagentDeclaration(
    key="feature_hypothesis",
    prompt_sources=(
        "preambles/paranoid_scientist",
        "preambles/thought_guidance",
        "research/feature_hypothesis",
    ),
    tools=(),
    role="specialist",
    description="Proposes initial feature set with CAAFE-style justifications.",
)

FINE_TUNE = SubagentDeclaration(
    key="fine_tune",
    prompt_sources=(
        "preambles/paranoid_scientist",
        "preambles/thought_guidance",
        "subagents/fine_tune",
    ),
    tools=(
        submit_sagemaker_job,
        fetch_training_metrics,
        await_training_job,
    ),
    role="finetune",
    description="Submits async SageMaker training jobs; resumes via EventBridge callback.",
)

FRAMING = SubagentDeclaration(
    key="framing",
    prompt_sources=(
        "preambles/paranoid_scientist",
        "preambles/thought_guidance",
        "research/framing",
    ),
    tools=(),
    role="specialist",
    description="Distills task spec into a compact problem card JSON.",
    context_aware=True,
)

GOVERNANCE_AGENT = SubagentDeclaration(
    key="governance_agent",
    prompt_sources=(
        "preambles/paranoid_scientist",
        "preambles/thought_guidance",
        "subagents/governance_agent",
    ),
    tools=(
        create_model_card,
        query_lineage,
        check_data_access,
    ),
    role="bounded",
    description="Owner of model-governance touchpoints (model cards, lineage, data-access checks) over the GovernanceBackend port; informs the readiness gates, does not deploy.",
)

IMPROVEMENT_AGENT = SubagentDeclaration(
    key="improvement_agent",
    prompt_sources=(
        "preambles/paranoid_scientist",
        "preambles/thought_guidance",
        "subagents/improvement_agent",
    ),
    tools=(
        analyze_production_failures,
        carry_forward_learnings,
        evaluate_retrain_trigger,
        retire_model,
    ),
    role="bounded",
    description="Bounded executor of the Stage-12 improvement tools (failure triage, AgentCore-Memory carry-over, lineage-preserving retire); prepares the retrain carry-context, never trains.",
)

IMPUTATION_STRATEGIST = SubagentDeclaration(
    key="imputation_strategist",
    prompt_sources=(
        "preambles/paranoid_scientist",
        "preambles/data_steering",
        "subagents/imputation_strategist",
    ),
    tools=(apply_imputation_via_port,),
    role="specialist",
    description="Chooses a per-column imputation plan honoring hard steering, then DISPATCHES the apply to SageMaker Processing (apply_imputation_via_port) which writes the imputed dataset + audit manifest to S3. Does not run imputers in the sandbox (ADR-0002).",
)

LABELING_STRATEGIST = SubagentDeclaration(
    key="labeling_strategist",
    prompt_sources=(
        "preambles/paranoid_scientist",
        "preambles/thought_guidance",
        "subagents/labeling_strategist",
    ),
    tools=(
        create_labeling_job,
        get_labeling_progress,
    ),
    role="bounded",
    description="Bounded Stage-4 labeling strategist: creates a Ground-Truth labeling job + polls progress (LabelingBackend). Job creation is IRREVERSIBLE/spend — approval-gated via data_collection_flow.",
)

LLM_EXTRAPOLATION = SubagentDeclaration(
    key="llm_extrapolation",
    prompt_sources=(
        "preambles/paranoid_scientist",
        "preambles/data_steering",
        "subagents/llm_extrapolation",
    ),
    tools=(
        llm_extrapolate_values,
        execute_python_tool,
    ),
    role="bounded",
    description="Bounded numeric AND text/verbiage gap fill from row context + domain notes; values+confidence always verified downstream by evaluate_imputation.",
)

MONITORING_AGENT = SubagentDeclaration(
    key="monitoring_agent",
    prompt_sources=(
        "preambles/paranoid_scientist",
        "preambles/thought_guidance",
        "subagents/monitoring_agent",
    ),
    tools=(
        get_metrics,
        put_custom_metric,
        analyze_drift,
        configure_alert,
        build_monitoring_dashboard,
        get_recent_logs,
        track_inference_spend,
        compile_monitoring_report,
        evaluate_retrain_trigger,
    ),
    role="bounded",
    description="Bounded executor of the Stage-11 monitoring tools (metrics/drift/alerts/dashboard/report) over the MonitoringBackend port; emits the closed-loop retrain-trigger decision.",
)

PROBLEM_FRAMING_ADVISOR = SubagentDeclaration(
    key="problem_framing_advisor",
    prompt_sources=(
        "preambles/paranoid_scientist",
        "preambles/thought_guidance",
        "subagents/problem_framing_advisor",
    ),
    tools=(
        select_problem_type,
        define_success_metrics,
        build_problem_card,
        check_stage_exit,
    ),
    role="specialist",
    description="Stage-2 framing advisor: selects the ML problem type, defines success metrics + guardrails, assembles the problem_card.json, and checks the stage-2 exit gate (pure reasoning).",
    context_aware=True,
)

READINESS_REVIEW_AGENT = SubagentDeclaration(
    key="readiness_review_agent",
    prompt_sources=(
        "preambles/paranoid_scientist",
        "preambles/thought_guidance",
        "subagents/readiness_review_agent",
    ),
    tools=(
        run_readiness_gate,
        build_risk_register,
        compile_readiness_review,
        design_production_architecture,
        draft_operational_runbook,
        recommend_inference_config,
        run_load_test,
    ),
    role="bounded",
    description="Bounded executor of the Stage-9 production-readiness gates (the 10 gates + risk register + go/no-go review); read-only — assesses, never deploys.",
)

RESEARCH_CURATOR = SubagentDeclaration(
    key="research_curator",
    prompt_sources=(
        "preambles/paranoid_scientist",
        "preambles/thought_guidance",
        "subagents/research_curator",
    ),
    tools=(
        triage_reproducibility,
        generate_research_digest,
        schedule_research_scan,
    ),
    role="bounded",
    description="Bounded Stage-0 research curator: triages findings by reproducibility + compiles the research digest + backlog (persisted to memory-bank/research) and proposes a scan cadence.",
)

RESEARCH_DISCOVERY = SubagentDeclaration(
    key="research_discovery",
    prompt_sources=(
        "preambles/paranoid_scientist",
        "subagents/research_discovery",
    ),
    tools=(
        search_algorithms,
        recommend_approaches,
    ),
    role="specialist",
    description="Discovers relevant ML algorithms and experiment suggestions for a task specification.",
)

RESEARCH_SCOUT = SubagentDeclaration(
    key="research_scout",
    prompt_sources=(
        "preambles/paranoid_scientist",
        "preambles/thought_guidance",
        "subagents/research_scout",
    ),
    tools=(
        index_research_corpus,
        chunk_documents,
        build_retrieval_index,
        query_research_corpus,
        build_metadata_filters,
        tune_retrieval,
        scan_research,
        schedule_research_scan,
    ),
    role="bounded",
    description="Bounded Stage-0 research scout: indexes the research corpus + scans external/indexed sources (RetrievalBackend), returning ranked findings for the research_curator to triage.",
)

SAGEMAKER_OPS = SubagentDeclaration(
    key="sagemaker_ops",
    prompt_sources=(
        "preambles/paranoid_scientist",
        "preambles/thought_guidance",
        "subagents/sagemaker_ops",
    ),
    tools=(
        build_sagemaker_pipeline,
        run_sagemaker_pipeline,
        register_model_sm,
        approve_model_sm,
        list_model_packages_sm,
        setup_model_monitor,
        inspect_model_monitor,
        run_clarify,
        feature_store_ingest,
        feature_store_get,
    ),
    role="bounded",
    description="Bounded executor of the SageMaker MLOps tools (Pipelines, Model Registry, Model Monitor, Clarify, Feature Store); returns resource refs.",
)

SOLUTION_FIT_ADVISOR = SubagentDeclaration(
    key="solution_fit_advisor",
    prompt_sources=(
        "preambles/paranoid_scientist",
        "preambles/thought_guidance",
        "subagents/solution_fit_advisor",
    ),
    tools=(
        assess_ml_necessity,
        survey_managed_services,
    ),
    role="specialist",
    description="Solution-fit advisor: assesses whether ML is warranted (vs rules) and surveys the AWS managed-service options for the problem/modality (pure reasoning — steers away from over-build).",
    context_aware=True,
)

SOLUTION_SURVEY_AGENT = SubagentDeclaration(
    key="solution_survey_agent",
    prompt_sources=(
        "preambles/paranoid_scientist",
        "preambles/thought_guidance",
        "subagents/solution_survey_agent",
    ),
    tools=(
        list_jumpstart_models,
        apply_research_gate,
        build_candidate_shortlist,
    ),
    role="bounded",
    description="Bounded Stage-3 solution-survey agent: lists JumpStart/Bedrock FM cards (ModelHubBackend), gates candidates against research evidence, and builds the ranked candidate shortlist.",
)

SPEC_WRITER = SubagentDeclaration(
    key="spec_writer",
    prompt_sources=(
        "preambles/paranoid_scientist",
        "subagents/spec_writer",
    ),
    tools=(
        validate_spec,
        save_spec,
        list_example_specs,
    ),
    role="reviewer",
    description="Generates a validated ML task specification (SpecModel) from a natural-language business challenge description. Outputs YAML front-matter + markdown domain notes.",
)

SYNTHETIC_DATA = SubagentDeclaration(
    key="synthetic_data",
    prompt_sources=(
        "preambles/paranoid_scientist",
        "preambles/data_steering",
        "subagents/synthetic_data",
    ),
    tools=(
        build_synthesis_spec,
        run_processing_job,
        llm_synthesize_rows,
    ),
    role="specialist",
    description="Builds a SynthesisSpec, generates realistic synthetic data, and validates it across the 3 pillars (fidelity/utility/privacy).",
)

TABULAR_FE = SubagentDeclaration(
    key="tabular_fe",
    prompt_sources=(
        "preambles/paranoid_scientist",
        "preambles/thought_guidance",
        "subagents/tabular_fe",
    ),
    tools=(
        run_processing_job,
        save_feature_manifest,
        execute_python_tool,
    ),
    role="planner",
    description="Decides features (light hypothesis checks), DISPATCHES the full-dataset FE transform to SageMaker Processing (run_processing_job), and emits feature_manifest.json.",
)

TRAINER_TUNER = SubagentDeclaration(
    key="trainer_tuner",
    prompt_sources=(
        "preambles/paranoid_scientist",
        "preambles/thought_guidance",
        "subagents/trainer_tuner",
    ),
    tools=(
        dispatch_preflight,
        submit_tabular_training,
        fetch_training_metrics,
        await_training_job,
        submit_sagemaker_job,
        execute_python_tool,
    ),
    role="bounded",
    description="Chooses algo/params, DISPATCHES training to SageMaker, awaits the graded y_pred.",
)

VISION_FE = SubagentDeclaration(
    key="vision_fe",
    prompt_sources=(
        "preambles/paranoid_scientist",
        "preambles/thought_guidance",
        "subagents/vision_fe",
    ),
    tools=(),
    role="bounded",
    description="Emits augmentation manifest + ROI/normalization config for vision tasks.",
)
