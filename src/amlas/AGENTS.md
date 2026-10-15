# src/amlas

## Purpose

Provide the shared application, agent factory and runtime invocation surfaces for Agentic Machine Learning Engineering.

## Ownership

Inherits [../AGENTS.md](../AGENTS.md); this document owns the files directly in this directory.

## Local Contracts

- `_build.py:build_agent` — Build a Strands Agent with Amlas canonical defaults.
- `_build.py` filters provider sampling/cache fields through shared model
  capabilities; explicit model overrides retain precedence over role defaults.
- `_build.py:context_preamble` — Compose a null-safe steering/problem PREAMBLE for a subagent system prompt (H2-6).
- `_runtime_guard.py:apply_openmp_safety_guard` — Neutralise the macOS duplicate-libomp OpenMP abort (behaviour-preserving).
- `_runtime_guard.py:install_error_excepthook` — Funnel ANY uncaught exception into the centralized ``ERROR_LOG`` (process safety net).
- `api.py:preview` — Plan-Mode preview a capability from Python — build the read-only plan, run nothing.
- `api.py:run_prompt` delegates exact natural input to the normal ChatEngine.
  Caller-supplied ApprovalPolicy is trusted code input, not model-authored JSON.
- `app.py:agent_invocation` — Entry point invoked by AgentCore Runtime.
- Owned files: `__init__.py`, `_build.py`, `_runtime_guard.py`, `api.py`, `app.py`, `orchestrator.py`, `prompts.py`, `thesis_experiment.py`.

## Work Guidance

Update this contract when owned interfaces, configuration, side effects or boundaries change. Read indexed child contracts before editing their files.

## Verification

Use the root verification contract and the owning regressions for changed behavior.

## Child DOX Index

| Path | Covers |
| --- | --- |
| `arbiter/AGENTS.md` | Grade predictive results deterministically and produce acceptance reports without delegating the decision to an LLM. |
| `async_callback/AGENTS.md` | Translate managed training events into callback/job-state handling and optional runtime resumption. |
| `backends/AGENTS.md` | Define typed backend ports, domain handles and adapter resolution at the workflow-to-compute boundary. |
| `cli/AGENTS.md` | Expose the command-line application and its shared configuration/display entry points. |
| `cloud/AGENTS.md` | Stage datasets/specifications, dispatch managed-runtime calls and resolve configured research secrets. |
| `config/AGENTS.md` | Resolve runtime configuration, selected AWS Region/profile and deployment namespaces at call time. |
| `control_plane/AGENTS.md` | Own capability dispatch, approval/plan policy, cost boundaries, project memory, sessions and integration hooks. |
| `data_eng/AGENTS.md` | Represent domain steering constraints and preferences used by the data-engineering tools. |
| `decisions/AGENTS.md` | Produce rules-based experiment recommendations, next-step decisions and model-card content. |
| `demos/AGENTS.md` | Provide the optional code-interpreter demonstration runner and its explicit staging/execution boundary. |
| `errors/AGENTS.md` | Classify failures, produce user-facing error messages and apply bounded recovery/retry behavior. |
| `evaluation/AGENTS.md` | Store and compare experiment results, leaderboards and statistical evidence. |
| `experiments/AGENTS.md` | Configure and execute experiment plans and hyperparameter selections. |
| `flows/AGENTS.md` | Declare and register reusable ML lifecycle flows with execution and approval metadata. |
| `golden/AGENTS.md` | Load evaluation fixtures and retain frozen train/test split manifests for repeatable grading. |
| `harness/AGENTS.md` | Provision and invoke opted-in managed Lead hosting and resolve its supported contracts. |
| `knowledge/AGENTS.md` | Provide algorithm knowledge and persisted learning used by recommendations. |
| `leads/AGENTS.md` | Declare and build the seven lifecycle Lead agents and their shared tool/prompt contracts. |
| `memory/AGENTS.md` | Connect agent hooks to managed memory and session state. |
| `pipeline/AGENTS.md` | Coordinate local and agentic experiment pipeline execution and result summaries. |
| `pricing/AGENTS.md` | Cache public AWS pricing inputs used for cost estimates. |
| `sagemaker_entrypoints/AGENTS.md` | Run managed tabular training/processing entrypoints and write their expected result artifacts. |
| `scheduler/AGENTS.md` | Manage project queues, research cadence and scheduler/dashboard state. |
| `security/AGENTS.md` | Enforce shared model-artifact loading boundaries across application layers. |
| `spec/AGENTS.md` | Parse task specifications and acceptance configuration into validated models. |
| `subagents/AGENTS.md` | Declare bounded specialist agents and their reusable tools/prompts. |
| `telemetry/AGENTS.md` | Write run traces, error records and execution state for inspection. |
| `tools/AGENTS.md` | Expose agent-callable data, research, ML, deployment and operational tools. |
| `tui/AGENTS.md` | Provide the Textual application, feature flags and shared capability presentation. |
