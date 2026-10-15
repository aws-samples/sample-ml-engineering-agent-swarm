"""Lead DECLARATIONS — the single source of truth for what each Lead is made of.

Why this module exists (ADR-0007 phase 1, T-HARN-04.5): a Lead's tool names and
system prompt used to be literals buried inside its ``build_*_lead`` body, so the
only way to read them was to actually construct all of its sub-agents — which, with
``model=None``, constructs a real ``BedrockModel``. That makes an offline
``HarnessSpec`` impossible to derive from the builder (T-HARN-04.1 requires
construction with no AWS credentials and no network call).

So the declaration is lifted OUT of the builder body into data that both paths read:

- :func:`build_lead` (the local-construction path) consumes a :class:`LeadDeclaration`
  to build the real ``Agent`` — every ``build_*_lead`` in this package is now a thin
  signature-preserving wrapper over it.
- A future ``harness/spec.py`` consumes the SAME declaration to derive
  ``declared_tool_names`` + ``system_prompt`` with no ``Agent`` construction at all.

Because both derive from one declaration, the harness configuration and the local
peer cannot drift.

Nothing here constructs an ``Agent``, a model, or a boto3 client: reading
:attr:`LeadDeclaration.declared_tool_names` and calling
:meth:`LeadDeclaration.system_prompt` is pure (prompt files are read off disk).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from strands import Agent

from .._build import build_agent
from ..prompts import stack_prompts
from ..subagents.algo_hypothesis import build_algo_hypothesis_agent
from ..subagents.algorithm_specialist import build_algorithm_specialist_agent
from ..subagents.business_goal_advisor import build_business_goal_advisor
from ..subagents.data_collection_agent import build_data_collection_agent
from ..subagents.data_integrity import build_data_integrity_agent
from ..subagents.deployment_agent import build_deployment_agent
from ..subagents.devops_git_agent import build_devops_git_agent
from ..subagents.dq_assessment import build_dq_assessment_agent
from ..subagents.feature_hypothesis import build_feature_hypothesis_agent
from ..subagents.fine_tune import build_fine_tune_agent
from ..subagents.framing import build_framing_agent
from ..subagents.governance_agent import build_governance_agent
from ..subagents.improvement_agent import build_improvement_agent
from ..subagents.imputation_strategist import build_imputation_strategist_agent
from ..subagents.labeling_strategist import build_labeling_strategist
from ..subagents.llm_extrapolation import build_llm_extrapolation_agent
from ..subagents.monitoring_agent import build_monitoring_agent
from ..subagents.problem_framing_advisor import build_problem_framing_advisor
from ..subagents.readiness_review_agent import build_readiness_review_agent
from ..subagents.research_curator import build_research_curator
from ..subagents.research_scout import build_research_scout
from ..subagents.sagemaker_ops import build_sagemaker_ops_agent
from ..subagents.solution_fit_advisor import build_solution_fit_advisor
from ..subagents.solution_survey_agent import build_solution_survey_agent
from ..subagents.synthetic_data import build_synthetic_data_agent
from ..subagents.tabular_fe import build_tabular_fe_agent
from ..subagents.trainer_tuner import build_trainer_tuner_agent
from ..subagents.vision_fe import build_vision_fe_agent
from ..tools.aws_control import (
    describe_aws_readiness,
    ensure_glue_database,
    ensure_iam_service_role,
    ensure_s3_prefix,
    grant_role_policy,
    remediate_dispatch_preflight,
    request_quota_increase,
)
from ..tools.lake import list_lake_catalog, read_lake_slice, write_lake_dataset
from ..tools.mlops import mlops_kpi_scorecard
from ..tools.research.strands_tools import research_fetch_urls


def declared_tool_contract(tool: Any) -> tuple[str, dict[str, Any]]:
    """Return ``(description, input_schema)`` for a directly-bound ``@tool`` object.

    Strands ``DecoratedFunctionTool`` objects carry a ``tool_spec`` with ``description`` and
    ``inputSchema`` — readable with no ``Agent`` built and no AWS call. The schema arrives
    WRAPPED as ``{"json": {...}}``; the harness ``inlineFunction`` type wants the JSON Schema
    itself, so the wrapper is unwrapped here rather than at every call site.
    """
    spec = getattr(tool, "tool_spec", None) or {}
    description = str(spec.get("description") or "").strip()
    schema = spec.get("inputSchema") or {}
    if isinstance(schema, dict) and set(schema) == {"json"}:
        schema = schema["json"]
    return description, dict(schema if isinstance(schema, dict) else {})


def declared_tool_name(tool: Any) -> str:
    """Return the name a directly-bound ``@tool`` object registers itself under.

    Strands ``@tool`` functions are ``DecoratedFunctionTool`` objects carrying an
    explicit ``.tool_name``; plain callables fall back to ``__name__``. This is the
    offline equivalent of reading ``Agent.tool_names`` for a bound tool.
    """
    name: str | None = getattr(tool, "tool_name", None)
    return name or str(tool.__name__)


# The input schema every agent-as-tool exposes. ``.as_tool()`` wraps a sub-agent behind a
# single ``input`` string, so the shape is FIXED and identical for all of them — verified
# against a built peer by ``test_lead_declarations.py`` rather than assumed.
SUBAGENT_TOOL_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"input": {"type": "string", "description": "The input to send to the agent tool."}},
    "required": ["input"],
}

# DEPRECATED fallback — kept only for a tool name with no sub-agent declaration.
#
# This table used to be the source of these strings, duplicating all 30 from the sub-agent
# builders because a sub-agent's description was reachable only by CONSTRUCTING it. The
# sub-agents now have their own declaration seam
# ([`subagents/declarations.py`](../subagents/declarations.py:1)), so
# :attr:`SubAgentDeclaration.description` reads the ONE declaration instead and the duplication
# is gone. A test asserts this table is now empty of anything the registry can answer, so it
# cannot quietly become a second source of truth again.
_LEGACY_SUBAGENT_TOOL_DESCRIPTIONS: dict[str, str] = {}


@dataclass(frozen=True)
class SubAgentDeclaration:
    """One sub-agent bound as a Lead tool via ``.as_tool(preserve_context=True)``.

    Attributes:
        tool_name: The name the sub-agent registers under on the Lead.
        builder: The sub-agent factory (``build_*`` in :mod:`amlas.subagents`).
        model_kwarg: The ``build_*_lead`` keyword that overrides this sub's model
            (e.g. ``"framing_model"`` drives the ``framing`` tool).
    """

    tool_name: str
    builder: Callable[..., Agent]
    model_kwarg: str

    @property
    def description(self) -> str:
        """Tool description, as the harness ``inlineFunction`` type requires.

        Read from the sub-agent's OWN declaration — one source of truth, so it cannot drift
        from what the built sub-agent reports. Falls back to the legacy table only for a tool
        name the sub-agent registry does not know, and raises if neither has it: a harness tool
        with an invented description silently degrades the model's tool selection, which is
        worse than a loud failure at declaration time.
        """
        from ..subagents.registry import SUBAGENTS_BY_KEY

        entry = SUBAGENTS_BY_KEY.get(self.tool_name)
        if entry is not None:
            return entry.declaration.description
        try:
            return _LEGACY_SUBAGENT_TOOL_DESCRIPTIONS[self.tool_name]
        except KeyError:
            raise KeyError(
                f"no sub-agent declaration for {self.tool_name!r}; register it in "
                "amlas.subagents.registry (a description is required by the harness "
                "inlineFunction tool type)"
            ) from None

    @property
    def input_schema(self) -> dict[str, Any]:
        """The fixed agent-as-tool input schema (a single ``input`` string)."""
        return dict(SUBAGENT_TOOL_INPUT_SCHEMA)


@dataclass(frozen=True)
class LeadDeclaration:
    """Everything needed to compose a Lead — readable without building anything.

    Attributes:
        key: Registry key for the Lead (matches the builder's ``name=`` default).
        builder_name: The public ``build_*_lead`` function that consumes this.
        prompt_sources: Prompt files stacked, in order, into the system prompt.
        sub_agents: Sub-agents bound as tools, in tool-registration order.
        direct_tools: ``@tool`` objects bound straight onto the Lead, in order,
            appended AFTER the sub-agent tools.
        description: The a2a/routing description passed to ``build_agent``.
        role: Model tier — every Lead is a ``planner``.
    """

    key: str
    builder_name: str
    prompt_sources: tuple[str, ...]
    sub_agents: tuple[SubAgentDeclaration, ...]
    direct_tools: tuple[Any, ...]
    description: str
    role: str = "planner"

    @property
    def declared_tool_names(self) -> tuple[str, ...]:
        """Tool names this Lead will register, in order — no ``Agent`` needed."""
        return tuple(
            [sub.tool_name for sub in self.sub_agents] + [declared_tool_name(t) for t in self.direct_tools]
        )

    @property
    def model_kwargs(self) -> tuple[str, ...]:
        """The per-sub model-override kwargs the builder must expose."""
        return tuple(sub.model_kwarg for sub in self.sub_agents)

    @property
    def tool_contracts(self) -> tuple[tuple[str, str, dict[str, Any]], ...]:
        """``(name, description, input_schema)`` per tool, in registration order.

        This is what the harness ``inlineFunction`` tool type needs: it REQUIRES both a
        description and an input schema for every tool. Sub-agents read theirs from
        :data:`SUBAGENT_TOOL_DESCRIPTIONS` plus the fixed agent-as-tool schema; directly-bound
        ``@tool`` objects carry theirs in ``tool_spec``. Both are obtainable offline.
        """
        contracts: list[tuple[str, str, dict[str, Any]]] = [
            (sub.tool_name, sub.description, sub.input_schema) for sub in self.sub_agents
        ]
        for tool in self.direct_tools:
            description, schema = declared_tool_contract(tool)
            contracts.append((declared_tool_name(tool), description, schema))
        return tuple(contracts)

    def system_prompt(self) -> str:
        """Render the Lead's system prompt by stacking :attr:`prompt_sources`.

        Pure file reads through :func:`amlas.prompts.stack_prompts` — no AWS, no
        network, no ``Agent``.
        """
        return stack_prompts(*self.prompt_sources)


def build_lead(
    declaration: LeadDeclaration,
    *,
    model: Any | None = None,
    name: str,
    hooks_factory: Callable[[], Sequence[Any]] | None = None,
    **sub_models: Any,
) -> Agent:
    """Build the real ``Agent`` for ``declaration`` — the local-construction path.

    Behaviour-preserving with the hand-written builder bodies this replaces:
    sub-agents are constructed in declaration order and wrapped with
    ``.as_tool(name=..., preserve_context=True)``, the direct ``@tool`` objects are
    appended after them, and ``hooks_factory`` is forwarded to ``build_agent``
    unconditionally but to the sub-agent builders ONLY when non-None (sub-agent
    factories do not all accept the kwarg).

    Args:
        declaration: The Lead declaration to realise.
        model: Lead model; ``None`` → real Bedrock. Callers may inject a real BedrockModel.
        name: Agent name (each ``build_*_lead`` supplies its own default).
        hooks_factory: Optional zero-arg memory-hook factory fanned out to subs.
        **sub_models: Per-sub model overrides keyed by the declaration's
            ``model_kwarg`` names.

    Raises:
        TypeError: If ``sub_models`` carries a key the declaration does not declare
            — the guard that keeps a builder signature and its declaration in sync.
    """
    unexpected = sorted(set(sub_models) - set(declaration.model_kwargs))
    if unexpected:
        raise TypeError(
            f"{declaration.builder_name}: unknown sub-agent model override(s) "
            f"{unexpected}; declared: {sorted(declaration.model_kwargs)}"
        )

    # Conditional forwarding: sub-agent factory signatures do not all accept
    # hooks_factory, so it is only passed when the Orchestrator supplied one.
    sub_kwargs: dict[str, Any] = {}
    if hooks_factory is not None:
        sub_kwargs["hooks_factory"] = hooks_factory

    tools: list[Any] = [
        sub.builder(model=sub_models.get(sub.model_kwarg), **sub_kwargs).as_tool(
            name=sub.tool_name, preserve_context=True
        )
        for sub in declaration.sub_agents
    ]
    tools.extend(declaration.direct_tools)

    return build_agent(
        name=name,
        system_prompt=declaration.system_prompt(),
        tools=tools,
        model=model,
        role=declaration.role,
        hooks_factory=hooks_factory,
        description=declaration.description,
    )


# ── The seven Lead declarations ───────────────────────────────────────────────
# Each is consumed by exactly one ``build_*_lead`` (see the module it is named for)
# and, from ADR-0007 phase 1 onward, by the harness spec builder.

RESEARCH_LEAD = LeadDeclaration(
    key="research_lead",
    builder_name="build_research_lead",
    prompt_sources=(
        "preambles/paranoid_scientist",
        "preambles/thought_guidance",
        "leads/research_lead",
    ),
    sub_agents=(
        SubAgentDeclaration("framing", build_framing_agent, "framing_model"),
        SubAgentDeclaration("feature_hypothesis", build_feature_hypothesis_agent, "feature_model"),
        SubAgentDeclaration("algo_hypothesis", build_algo_hypothesis_agent, "algo_model"),
        # M6 Stage 0–3: research-intelligence → business-goal → framing → solution-survey.
        # ADDITIVE — the canonical R1/R2/R3 wiring above is untouched.
        SubAgentDeclaration("research_scout", build_research_scout, "research_scout_model"),
        SubAgentDeclaration("research_curator", build_research_curator, "research_curator_model"),
        SubAgentDeclaration("business_goal_advisor", build_business_goal_advisor, "business_goal_model"),
        SubAgentDeclaration("solution_fit_advisor", build_solution_fit_advisor, "solution_fit_model"),
        SubAgentDeclaration(
            "problem_framing_advisor", build_problem_framing_advisor, "problem_framing_model"
        ),
        SubAgentDeclaration("solution_survey_agent", build_solution_survey_agent, "solution_survey_model"),
    ),
    direct_tools=(research_fetch_urls,),
    description=(
        "Orchestrates R1+R2+R3 subs to produce a signed hypothesis.json; also owns the M6 "
        "Stage 0–3 front-end (research scan/curate → business-goal → framing → solution-survey)."
    ),
)

FE_LEAD = LeadDeclaration(
    key="fe_lead",
    builder_name="build_fe_lead",
    prompt_sources=(
        "preambles/paranoid_scientist",
        "preambles/thought_guidance",
        "leads/fe_lead",
    ),
    sub_agents=(
        SubAgentDeclaration("data_integrity", build_data_integrity_agent, "data_integrity_model"),
        SubAgentDeclaration("tabular_fe", build_tabular_fe_agent, "tabular_fe_model"),
        SubAgentDeclaration("vision_fe", build_vision_fe_agent, "vision_fe_model"),
    ),
    direct_tools=(),
    description="D1 orchestrator — validates data, implements features, emits feature manifest.",
)

ML_EXECUTION_LEAD = LeadDeclaration(
    key="ml_execution_lead",
    builder_name="build_ml_execution_lead",
    prompt_sources=(
        "preambles/paranoid_scientist",
        "preambles/thought_guidance",
        "leads/ml_execution_lead",
    ),
    sub_agents=(
        SubAgentDeclaration(
            "algorithm_specialist", build_algorithm_specialist_agent, "algo_specialist_model"
        ),
        SubAgentDeclaration("trainer_tuner", build_trainer_tuner_agent, "trainer_tuner_model"),
        SubAgentDeclaration("fine_tune", build_fine_tune_agent, "fine_tune_model"),
    ),
    direct_tools=(),
    description="D2 orchestrator — selects algo, tunes, or fine-tunes; emits model artifact.",
)

DATA_ENGINEERING_LEAD = LeadDeclaration(
    key="data_engineering_lead",
    builder_name="build_data_engineering_lead",
    prompt_sources=(
        "preambles/paranoid_scientist",
        "preambles/thought_guidance",
        "preambles/data_steering",
        "leads/data_engineering_lead",
    ),
    sub_agents=(
        SubAgentDeclaration("dq_assessment", build_dq_assessment_agent, "dq_assessment_model"),
        SubAgentDeclaration("imputation_strategist", build_imputation_strategist_agent, "imputation_model"),
        SubAgentDeclaration("synthetic_data", build_synthetic_data_agent, "synthetic_model"),
        SubAgentDeclaration("llm_extrapolation", build_llm_extrapolation_agent, "llm_extrapolation_model"),
        # M6 Stage-4: lake inventory (QueryBackend) → labeling job (LabelingBackend, gated).
        # ADDITIVE — the existing DQ wiring above is untouched.
        SubAgentDeclaration("data_collection_agent", build_data_collection_agent, "data_collection_model"),
        SubAgentDeclaration("labeling_strategist", build_labeling_strategist, "labeling_model"),
    ),
    direct_tools=(
        # ADR-0002: `score_data_quality_6dim` USED to be bound here. It reads a local CSV with
        # pandas in-process (`score.py:207`) and takes a filesystem `data_path` rather than a
        # `dataset_uri`, so it could not even address a lake dataset — a Lead orchestrates, it
        # never runs compute. Removed, NOT swapped for a dispatch tool: dispatch tools live on
        # the sub-agent that owns them, never on a Lead, and this Lead's own `dq_assessment`
        # sub-agent already carries `run_profile`. That routes through
        # `DataQualityBackend.profile`, whose local adapter runs THE SAME
        # `score_data_quality_6dim` (`adapters/dataquality.py:84`) and whose managed adapter
        # runs a DataBrew profile job at scale. So the 6-dimension scorecard is unchanged and
        # still reachable — it just dispatches now instead of executing here.
        # AWS control plane (plan §A2/§A4) — the LANDING half: this lead owns getting
        # data into the medallion lake, so it can ensure the destination prefix and the
        # catalog database exist before a write/crawl instead of failing on a missing
        # target. Scoped deliberately narrower than the MLOps Lead's set: NO IAM and no
        # quota verbs here (permissions are platform plumbing, not data landing).
        describe_aws_readiness,
        ensure_s3_prefix,
        ensure_glue_database,
    ),
    description=(
        "D0 orchestrator — assesses data quality, imputes gaps, synthesizes "
        "data, re-validates, and emits a data_engineering_manifest; also owns the M6 "
        "Stage-4 data-collection (lake inventory) + labeling-job (approval-gated) subs."
    ),
)

DEPLOY_SERVING_LEAD = LeadDeclaration(
    key="deploy_serving_lead",
    builder_name="build_deploy_serving_lead",
    # NOTE: no ``preambles/thought_guidance`` — this Lead (and the Monitoring Lead)
    # intentionally stack only paranoid_scientist + data_steering.
    prompt_sources=(
        "preambles/paranoid_scientist",
        "preambles/data_steering",
        "leads/deploy_serving_lead",
    ),
    sub_agents=(
        SubAgentDeclaration("readiness_review_agent", build_readiness_review_agent, "readiness_model"),
        SubAgentDeclaration("deployment_agent", build_deployment_agent, "deployment_model"),
        SubAgentDeclaration("governance_agent", build_governance_agent, "governance_model"),
    ),
    direct_tools=(),
    description=(
        "Deploy/Serving Lead (D3): owns the Stage-9 readiness review → Stage-10 deploy "
        "hand-off over the DeploymentBackend port; drives readiness_flow + deploy_flow "
        "(approval-gated, irreversible-safe). Closes the deploy GAP (ADR-007)."
    ),
)

MLOPS_LEAD = LeadDeclaration(
    key="mlops_lead",
    builder_name="build_mlops_lead",
    prompt_sources=(
        "preambles/paranoid_scientist",
        "preambles/thought_guidance",
        "preambles/data_steering",
        "leads/mlops_lead",
    ),
    sub_agents=(
        SubAgentDeclaration("sagemaker_ops", build_sagemaker_ops_agent, "sagemaker_ops_model"),
        SubAgentDeclaration("devops_git_agent", build_devops_git_agent, "devops_git_model"),
    ),
    direct_tools=(
        read_lake_slice,
        write_lake_dataset,
        list_lake_catalog,
        mlops_kpi_scorecard,
        # AWS control plane (plan §A2/§A4). The MLOps Lead owns platform plumbing, so
        # it is the lead that fixes the account itself instead of reporting a missing
        # role/prefix/quota to a human: ensure the S3 output prefix, create the
        # SageMaker job role, attach a SCOPED grant (this is what finally closes the
        # ``iam:PassRole`` gap ``dispatch_preflight`` has only ever *reported*), raise a
        # quota, and turn a failed preflight into an ordered fix list. Every mutator is
        # plan-first + prefix-scoped (``control_plane.aws_resource_policy``) and gates
        # as IRREVERSIBLE. Control-plane configuration only — ADR-0002 still holds:
        # all COMPUTE offloads to SageMaker/Glue/Lambda/Fargate.
        describe_aws_readiness,
        remediate_dispatch_preflight,
        ensure_s3_prefix,
        ensure_iam_service_role,
        grant_role_policy,
        request_quota_increase,
    ),
    description=(
        "M0 orchestrator — agent framework over SageMaker: reads a medallion "
        "silver slice, trains/registers/monitors via SageMaker Ops, writes "
        "cleaned/inferred data back to silver/gold with lineage, reports MLOps KPIs. "
        "Authors git commits + GitLab MRs for ML artifacts via the devops_git_agent "
        "(approval-gated push/MR; specs/pipelines/model-cards linked to the run)."
    ),
)

MONITORING_LEAD = LeadDeclaration(
    key="monitoring_lead",
    builder_name="build_monitoring_lead",
    # NOTE: no ``preambles/thought_guidance`` — mirrors the Deploy/Serving Lead.
    prompt_sources=(
        "preambles/paranoid_scientist",
        "preambles/data_steering",
        "leads/monitoring_lead",
    ),
    sub_agents=(
        SubAgentDeclaration("monitoring_agent", build_monitoring_agent, "monitoring_model"),
        SubAgentDeclaration("improvement_agent", build_improvement_agent, "improvement_model"),
    ),
    direct_tools=(),
    description=(
        "Monitoring Lead (M1): owns the Stage-11 monitoring → Stage-12 improvement close over "
        "the MonitoringBackend port; drives monitor_flow + improve_retrain_flow (approval-gated). "
        "Closes the lifecycle loop — the retrain trigger re-enters the unified retreat loop (ADR-008)."
    ),
)

#: Every Lead declaration, keyed by :attr:`LeadDeclaration.key`. The single
#: enumeration of the 7 Leads — ``leads/registry.py`` pairs these with their public
#: builders, and the ADR-0007 harness registry consumes it to enumerate
#: harness-hosted leads.
LEAD_DECLARATIONS: dict[str, LeadDeclaration] = {
    d.key: d
    for d in (
        RESEARCH_LEAD,
        DATA_ENGINEERING_LEAD,
        FE_LEAD,
        ML_EXECUTION_LEAD,
        DEPLOY_SERVING_LEAD,
        MLOPS_LEAD,
        MONITORING_LEAD,
    )
}


def lead_declaration(key: str) -> LeadDeclaration:
    """Return the declaration for ``key``, raising ``KeyError`` with the valid keys."""
    try:
        return LEAD_DECLARATIONS[key]
    except KeyError:
        raise KeyError(f"unknown lead {key!r}; known leads: {sorted(LEAD_DECLARATIONS)}") from None
