# Agentic ML Engineering

[![Python 3.11+](https://img.shields.io/badge/python-3.11--3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT-0](https://img.shields.io/badge/license-MIT--0-green.svg)](LICENSE)
[![Architecture](https://img.shields.io/badge/architecture-ports_%26_adapters-informational.svg)](#️-architecture--the-two-planes--the-seam)
[![CI](https://github.com/aws-samples/sample-ml-engineering-agent-swarm/actions/workflows/public-release.yml/badge.svg)](https://github.com/aws-samples/sample-ml-engineering-agent-swarm/actions/workflows/public-release.yml)
[![AWS](https://img.shields.io/badge/platform-AWS-FF9900.svg)](#️-backend-selection--live-aws-setup)
[![Strands Agents](https://img.shields.io/badge/agents-Strands_Agents-6F42C1.svg)](https://github.com/strands-agents/sdk-python)

**Agentic automation from hypothesis to spec driven ML engineering.**

You run this sample application locally or deploy it into your own AWS account. It uses AWS services under your configuration and permissions.

> **A terminal-native, conversational agent for the AWS ML lifecycle.** You describe intent in plain language (*"clean and impute this dataset, train a baseline, and tell me if it beats RMSE 12"*); a control-plane chat agent turns that into transparent, streamed tool calls that kick off opinionated agentic flows spanning the full ML lifecycle — research → business framing → data collection → data-quality/ETL → feature engineering → train → HPO/AutoML → evaluate (deterministic Arbiter) → deploy → monitor → closed-loop retrain. Heavy managed compute runs in your AWS account (SageMaker, Glue, Bedrock) behind pluggable **ports** and spins down when done; the surface owns the conversation, the flows, the accept/reject decision, the steering dial, the medallion opinion, and the end-to-end traceability.

> **Disclaimer.** This is sample/reference software. The managed-AWS backends provision real
> AWS resources (SageMaker jobs, endpoints, Glue jobs, S3, Bedrock calls) that **incur cost**.
> **Offline mode is intentionally limited**: it requires no AWS credentials or spend and provides
> an opinionated local subset focused on **(1)** dataset exploration and data-quality profiling,
> **(2)** missingness diagnosis and imputation, **(3)** synthetic-data generation and validation,
> **(4)** local training/tuning with deterministic Arbiter evaluation, and **(5)** local experiment
> tracking and model serving. Managed data collection, SageMaker/Glue/Bedrock backends, production
> deployment, and AWS monitoring require live AWS mode. Review the
> [cost & approval guardrails](#️-backend-selection--live-aws-setup) before enabling it.
>
> **This is sample code, for informational and educational purposes only — not for production
> usage. You should work with your security and legal teams to meet your organizational security,
> regulatory, and compliance requirements before deployment.** You are responsible for
> securing your own AWS account, credentials, IAM scoping, network egress, and budget caps. Review
> the [Security](#-security) section
> before enabling any live (billable) leg.

---

## Table of Contents

- [Agentic ML Engineering](#agentic-ml-engineering)
  - [Table of Contents](#table-of-contents)
  - [🎯 What this is](#-what-this-is)
  - [🏗️ Architecture — the two planes + the seam](#️-architecture--the-two-planes--the-seam)
  - [🧩 Feature \& capability catalog](#-feature--capability-catalog)
    - [The 25 backend ports](#the-25-backend-ports)
    - [The 21 opinionated flows](#the-21-opinionated-flows)
    - [The 7 Leads + the swarm](#the-7-leads--the-swarm)
    - [Data-quality / imputation / synthetic-data engine](#data-quality--imputation--synthetic-data-engine)
  - [🚀 Quickstart](#-quickstart)
  - [⚙️ Configuration](#️-configuration)
    - [`.agenticml.yaml` — project memory](#agenticmlyaml--project-memory)
    - [`.env` layers — credentials \& selectors](#env-layers--credentials--selectors)
    - [The SSM `/amlas` config model](#the-ssm-amlas-config-model)
  - [📖 Usage / user guide](#-usage--user-guide)
    - [1. Interactive chat / TUI](#1-interactive-chat--tui)
    - [2. CLI (`amlas`)](#2-cli-amlas)
    - [3. Headless mode + Plan Mode + session resume (`capability`)](#3-headless-mode--plan-mode--session-resume-capability)
    - [4. Python API](#4-python-api)
    - [Spec file format](#spec-file-format)
  - [☁️ Backend selection \& live-AWS setup](#️-backend-selection--live-aws-setup)
  - [🚀 Deploy the AWS infrastructure (one-click)](#-deploy-the-aws-infrastructure-one-click)
    - [Installation prerequisites and host preparation](#installation-prerequisites-and-host-preparation)
    - [Setup configuration](#setup-configuration)
    - [Optional research access](#optional-research-access)
    - [What gets deployed](#what-gets-deployed)
    - [Verify the deploy](#verify-the-deploy)
    - [Developer verification](#developer-verification)
    - [Teardown](#teardown)
    - [Profiles (headless vs SSO)](#profiles-headless-vs-sso)
  - [📁 Project layout](#-project-layout)
  - [🔐 Security](#-security)
  - [📄 License \& attribution](#-license--attribution)

---

## 🎯 What this is

This repository is a **user-deployed sample application providing a modular agentic abstraction surface over AWS SageMaker, Glue, Data Wrangler, Bedrock, and related ML services**, spanning the full 13-stage ML lifecycle (stages 0–12). The defining design decision: **the surface owns the *conversation*, the *opinion*, and the *decision*; your AWS services provide the *compute*, the *storage*, and the *algorithms*** — orchestrated behind pluggable backend ports so the *same flow* can target SageMaker today, a third party tomorrow, or Bedrock next week **without changing the flow**.

**Key properties**

- **Chat is the primary surface** — a terminal-native conversational REPL. The TUI, CLI, headless print-mode, and Python API are secondary renderers of the *same* capabilities (one capability, all surfaces).
- **Safe-by-default** — expensive/irreversible AWS actions (launch a training job, deploy an endpoint, delete a resource) are gated behind an explicit approval/permission model with a read-only **Plan Mode** preview + cost/impact estimate; cheap reads auto-approve.
- **Backend-portable** — a Bedrock-style ports-and-adapters seam: **25 Protocol ports**, each with a SageMaker (default) and a local (offline) adapter, plus a proven **Bedrock** alternative for Training & Deployment. Selected via `*_BACKEND` env selectors, resolved by `BackendRegistry.resolve(...)`.
- **Deterministic hand on the decision** — the no-LLM Arbiter makes the reproducible accept/reject call; it is **never delegated** to an LLM or managed service for predictive tasks.

**Verified headline counts** (every count below is asserted against source code, not a plan target):

| Construct | Count | Source of truth |
|---|---:|---|
| Backend ports (`ALL_PORTS`) | **25** | [`backends/ports.py`](src/amlas/backends/ports.py#L572) |
| Registered flows (`FLOWS`) | **21** | [`flows/__init__.py`](src/amlas/flows/__init__.py#L46) |
| Lead agents | **7** | [`leads/`](src/amlas/leads) |
| Control-plane modules | **11** | [`control_plane/`](src/amlas/control_plane) |
| CLI command groups | **10** | [`cli/main.py`](src/amlas/cli/main.py#L37) |
| Surfaces | **4** (Chat · TUI · CLI · Python-API) | [`surfaces`](#-usage--user-guide) |
| Lifecycle stages | **13** (0–12) | [`flows/`](src/amlas/flows) |

---

## 🏗️ Architecture — the two planes + the seam

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  SURFACES  (thin renderers of ONE capability set)                              │
│  Chat REPL (primary) · TUI (9 panes) · CLI (amlas) · headless · Python API     │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                 │  natural language / params
┌───────────────────────────────▼──────────────────────────────────────────────┐
│  CONTROL PLANE  (NL → tool calls; "this plane IS the product")                 │
│  chat engine · approval/permission gates · Plan Mode · project memory          │
│  (.agenticml.yaml) · custom slash commands · hooks-as-guardrails · MCP bridge  │
│  · capability funnel (invoke_capability) · headless/session-resume · cost gov  │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                 │  dispatch a flow_id (gated if spend/irreversible)
┌───────────────────────────────▼──────────────────────────────────────────────┐
│  ORCHESTRATION PLANE  (the 21 opinionated flows)                               │
│  7 Leads composing the subagent swarm · the DETERMINISTIC ARBITER (never       │
│  delegated) · the retreat loop · the closed-loop retrain                       │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                 │  flows call PORTS, never boto3 directly
┌───────────────────────────────▼──────────────────────────────────────────────┐
│  THE SEAM  —  25 ports + BackendRegistry.resolve(port, steering, scale, budget)│
│  precedence: hard-pin  >  explicit *_BACKEND env  >  auto (offline→local)      │
└───────────────────────────────┬──────────────────────────────────────────────┘
        ┌────────────────────────┼────────────────────────┐
        ▼                        ▼                        ▼
┌───────────────┐      ┌──────────────────┐      ┌────────────────────────┐
│ AWS (default) │      │ LOCAL (offline)  │      │ BEDROCK / 3rd-party     │
│ SageMaker/Glue│      │ sklearn/pandas/  │      │ alt adapters (proven    │
│ /Bedrock/CW   │      │ DuckDB/SHAP/...   │      │ portability)            │
└───────────────┘      └──────────────────┘      └────────────────────────┘
```

- **Control plane** — turns NL into intent and routes it to a single tool call or an orchestration-plane flow; hosts the conversation, slash commands, approval gates, Plan Mode, project memory, and the [`capability`](src/amlas/control_plane/capability.py) funnel. **This plane is the product.**
- **Orchestration plane** — executes the opinionated multi-step flows: the 7 Leads composing the subagent swarm, the deterministic Arbiter, and the retreat loop. Flows call **ports**, not boto3.
- **The seam** — [`BackendRegistry`](src/amlas/backends/registry.py#L153) binds each port to a concrete adapter at runtime with strict, traced precedence: **(1)** a hard steering pin wins; **(2)** an explicit `*_BACKEND` env selector is honored (bind-or-fail); **(3)** otherwise `auto` chooses — with `AMLAS_OFFLINE`/`AMLAS_BACKENDS_OFFLINE` it resolves to the **local** adapter, while normal operation leans to the AWS adapter. Every resolution records `port → adapter → reason` for the traceability ledger.

The **closed loop**: [`monitor_flow`](src/amlas/flows/__init__.py#L202) detects drift → emits a retrain-trigger decision → the Monitoring Lead routes it into [`improve_retrain_flow`](src/amlas/flows/__init__.py#L216), which **re-enters the unified exploration retreat loop** (not a parallel path) with the Arbiter as the non-delegated gate, then re-gates readiness and redeploys — or retires the model.

---

## 🧩 Feature & capability catalog

### The 25 backend ports

The ports are domain-typed `typing.Protocol` interfaces ([`backends/ports.py`](src/amlas/backends/ports.py#L1)); an adapter "is a" port by implementing its methods (structural typing, no inheritance). They are also the **MCP-style pluggable seam**. Each row below is verified against [`ALL_PORTS`](src/amlas/backends/ports.py#L572) and the adapters' `register_adapter(...)` calls.

**Group 1 — ML-compute (4)**

| Port | `*_BACKEND` selector | Adapters (default ✦ / local ◦ / alt) |
|---|---|---|
| `TrainingBackend` | `TRAINING_BACKEND` | `sagemaker` ✦ · `local` ◦ · `bedrock` (BedrockFineTune) |
| `TuningBackend` | `TUNING_BACKEND` | `sagemaker` ✦ (AMT) · `optuna` ◦ |
| `AutoMLBackend` | `AUTOML_BACKEND` | `autopilot` ✦ · `local` ◦ (sweep) |
| `ModelHubBackend` | `MODELHUB_BACKEND` | `jumpstart` ✦ · `local` ◦ · `bedrock` |

**Group 2 — data / feature (7)**

| Port | `*_BACKEND` selector | Adapters (default ✦ / local ◦ / alt) |
|---|---|---|
| `ETLBackend` | `ETL_BACKEND` | `processing` ✦ · `inprocess` ◦ (pandas) · `glue` |
| `DataQualityBackend` | `DQ_BACKEND` | `databrew` ✦ · `inprocess` ◦ (6-dim) · `glue_dq` |
| `FeatureStoreBackend` | `FEATURE_STORE_BACKEND` | `sagemaker` ✦ · `goldtable` ◦ |
| `LakeBackend` | `LAKE_BACKEND` | `s3medallion` ✦ · `localfs` ◦ |
| `QueryBackend` | `QUERY_BACKEND` | `athena` ✦ · `duckdb` ◦ |
| `LabelingBackend` | `LABELING_BACKEND` | `groundtruth` ✦ · `stub` ◦ |
| `RetrievalBackend` | `RETRIEVAL_BACKEND` | `bedrock_kb` ✦ · `local` ◦ (TF-IDF) · `opensearch` |

**Group 3 — deploy / serve / eval (3)**

| Port | `*_BACKEND` selector | Adapters (default ✦ / local ◦ / alt) |
|---|---|---|
| `DeploymentBackend` | `DEPLOYMENT_BACKEND` | `sagemaker` ✦ · `local` ◦ (LocalServe) · `bedrock` (hosted) |
| `ExplainabilityBackend` | `EXPLAINABILITY_BACKEND` | `clarify` ✦ · `inprocess_shap` ◦ |
| `EvaluationBackend` | `EVALUATION_BACKEND` | `arbiter` ✦ / `local` ◦ — **both are the DeterministicArbiter** (never delegated) |

**Group 4 — MLOps / governance / orchestration + Guardrail (7)**

| Port | `*_BACKEND` selector | Adapters (default ✦ / local ◦ / alt) |
|---|---|---|
| `ModelRegistryBackend` | `MODEL_REGISTRY_BACKEND` | `sagemaker` ✦ · `jsonfile` ◦ |
| `MonitoringBackend` | `MONITORING_BACKEND` | `sagemaker` ✦ (Model Monitor) · `otel` ◦ |
| `ExperimentTrackingBackend` | `EXPERIMENT_TRACKING_BACKEND` | `local` ✦◦ (leaderboard) |
| `PipelineBackend` | `PIPELINE_BACKEND` | `sagemaker` ✦ (Pipelines) · `local` ◦ |
| `OrchestrationBackend` | `ORCHESTRATION_BACKEND` | `local_driver` ✦◦ (retreat loop) · `sagemaker_pipeline` (graduation) |
| `GovernanceBackend` | `GOVERNANCE_BACKEND` | `sagemaker` ✦ · `local` ◦ |
| `GuardrailBackend` | `GUARDRAIL_BACKEND` | `bedrock` ✦ (Guardrails) · `inprocess` ◦ |

**Group 5 — agent-runtime (4, mostly pre-existing seams)**

| Port | `*_BACKEND` selector | Notes |
|---|---|---|
| `ModelBackend` | `MODEL_BACKEND` | the Strands LLM-provider seam (Bedrock/OpenAI/custom); no `@tool` surface |
| `MemoryBackend` | `MEMORY_BACKEND` | `local` ✦◦ (carry-over) · `agentcore` (managed Memory) |
| `SandboxBackend` | `SANDBOX_BACKEND` | AgentCore Code Interpreter (stateful) |
| `RuntimeBackend` | `RUNTIME_BACKEND` | AgentCore Runtime invoke (in-process locally) |

> A 26th selector, `COST_ADVISOR_BACKEND` (the `CostAdvisorBackend` cost-governance **facet**), is resolved through the same registry but is intentionally **not** in `ALL_PORTS` — it is a control-plane facet ([`cost_governance.py`](src/amlas/control_plane/cost_governance.py)), not one of the 25 formal ports.

### The 21 opinionated flows

A flow is an *invariant, opinionated, multi-step ML sequence* the control plane dispatches; the module that swaps is the adapter beneath the port. Each flow is a [`Flow`](src/amlas/flows/__init__.py#L22) descriptor carrying spend/irreversibility metadata that Plan Mode + the approval gate read. List them at runtime with `list_flows()` / the `/flows` chat command / `./bin/amlas capability list`. Every `flow_id` below is verified against the [`register_flow(...)`](src/amlas/flows/__init__.py#L67) calls.

| `flow_id` | Lifecycle stage | Port | Gate | One-line purpose |
|---|---|---|---|---|
| `exploration_loop` | loop driver | Orchestration | spend | End-to-end R→D1→D2→Arbiter retreat loop (the unified driver) |
| `ml_pipeline` | loop driver | Orchestration | spend | Deterministic no-LLM pipeline (research→plan→train→evaluate→decide) |
| `plan_approve_execute` | cross-cutting | GuardrailHooks | — | Preview plan → approval gate → execute (the safe-by-default wrapper) |
| `project_init_flow` | cross-cutting | ControlPlane | — | `/init`: scan repo+data → draft `.agenticml.yaml` project memory |
| `custom_command_flow` | cross-cutting | ControlPlane | — | Author/load a markdown custom command → expand to its flow/tool sequence |
| `devops_git_flow` | cross-cutting | VCS | spend·irrev | `commit_artifact` → `open_pr`: commit ML artifacts then push + open a PR (approval-gated) |
| `data_quality_flow` | 4–5 Pre-process | DataQuality | — | Profile → recommend DQ rules → evaluate → gate/report |
| `imputation_flow` | 4–5 Pre-process | DataQuality | spend | Diagnose missingness → recommend → apply (swappable module) → validate |
| `feature_engineering_flow` | 6 Features | FeatureStore | — | Propose → build/select → CAAFE validate → ingest → PIT dataset (no-leakage) |
| `baseline_train_flow` | 8 Train | Training | spend | Train baseline → Arbiter grade (non-delegated) → log → register (approval-gated) |
| `hpo_flow` | 8 Tune | Tuning | spend | Search space → run HPO (Optuna/AMT) → eval best → log → compare (approval-gated) |
| `evaluation_flow` | 8 Eval | Evaluation | — | Offline metrics + Arbiter verdict + model-comparison table |
| `readiness_flow` | 9 Readiness | Deployment | — | 10 production-readiness gates → risk register → go/no-go review |
| `deploy_flow` | 10 Deploy | Deployment | spend·irrev | Package → strategy → deploy_endpoint(mode) → smoke invoke → release → wire monitoring; rollback on fail (approval-gated) |
| `monitor_flow` | 11 Monitor | Monitoring | — | Setup/schedule → metrics → analyze_drift(4 families) → alerts → report → evaluate_retrain_trigger (the closed-loop decision) |
| `improve_retrain_flow` | 12 Improve | Orchestration | spend·irrev | Failures → carry learnings to Memory → retrain by RE-ENTERING the retreat loop → readiness re-gate → redeploy / retire (approval-gated) |
| `research_intel_flow` | 0 Research | Retrieval | — | Index corpus → scan research → triage reproducibility → digest → backlog |
| `research_scan_flow` | 0 Research | Retrieval | — | Scheduled sibling of `research_intel_flow`: scan per query → aggregate deduped digest → record/deliver (cadence-driven, read-only) |
| `business_framing_flow` | 1–2 Frame | ControlPlane | — | Capture problem → KPI → ML-necessity → go/no-go → problem-type → metrics → problem_card |
| `solution_survey_flow` | 3 Solution | ModelHub | — | List JumpStart/Bedrock FM → survey managed services → research-gate → ranked shortlist |
| `data_collection_flow` | 4 Collect | Labeling | spend·irrev | Inventory (run_sql) → labeling strategy → create labeling job → poll (approval-gated) |

### The 7 Leads + the swarm

The orchestration plane composes a Strands agent swarm. Each subagent declares a bounded tool allowlist (the Claude-Code subagent analog) and is invoked automatically (description-driven delegation) or explicitly; read-heavy stages fan out as parallel agent-teams. The 7 Lead modules ([`leads/`](src/amlas/leads)):

| Lead | Module | Coordinates |
|---|---|---|
| Research Lead | [`research_lead.py`](src/amlas/leads/research_lead.py) | framing + hypothesis generation (stages 0–3) |
| Data-Engineering Lead | [`data_eng.py`](src/amlas/leads/data_eng.py) | DQ → impute → synthesize; threads the steering dial |
| Feature-Engineering Lead | [`feature_eng.py`](src/amlas/leads/feature_eng.py) | data integrity + feature engineering (stage 6) |
| ML-Execution Lead | [`ml_execution.py`](src/amlas/leads/ml_execution.py) | algorithm selection + training/tuning (stage 8) |
| Deploy/Serving Lead | [`deploy_serving_lead.py`](src/amlas/leads/deploy_serving_lead.py) | readiness + deploy + serve (stages 9–10) |
| MLOps Lead | [`mlops_lead.py`](src/amlas/leads/mlops_lead.py) | pipelines/registry/monitor over SageMaker |
| Monitoring Lead | [`monitoring_lead.py`](src/amlas/leads/monitoring_lead.py) | drift + the closed-loop retrain trigger (stages 11–12) |

The **deterministic Arbiter** ([`arbiter/`](src/amlas/arbiter)) is the one non-delegable decision step. It grades classification (F1-macro + per-class slices), regression (RMSE/MAE), detection (mAP), and segmentation (mIoU) in pure Python, with no LLM.

### Data-quality / imputation / synthetic-data engine

The data-engineering surface (reachable from chat `/dqscan`, `/diagnose`, `/impute`, `/synth`, the `data` CLI group, and the `data_quality_flow` / `imputation_flow`) ships:

- A **25-pattern DQ catalog** (5 families) + a **6-dimension** scorecard (completeness/uniqueness/consistency/validity/accuracy/timeliness).
- **MCAR / MAR / MNAR missingness diagnosis** (Little's chi-square + a mask-AUC classifier).
- A **multi-strategy imputation engine** (mean/median/mode, KNN, IterativeImputer/MICE, RandomForest/MissForest, time-series interpolation + seasonal decomposition, injectable LLM gap-fill for numeric **and** text) with a **mask-and-score** self-evaluation harness.
- A **synthetic-data engine** (SDV GaussianCopula/CTGAN/TVAE + SMOTE-family augmentation) gated by a **3-pillar AND-gate**: **Fidelity** (KS / corr-drift / discriminator AUROC) **AND** **Utility** (TSTR-vs-TRTR) **AND** **Privacy** (distance-to-closest-record / membership-inference / exact-dups) — so a verbatim copy with perfect fidelity is still **rejected on privacy**.
- A domain-steering **dial** ([`SteeringProfile`](src/amlas/data_eng/steering.py)): hard constraints (`forbid_methods`/`pin_methods`/`mechanism_hints`/thresholds) enforced deterministically inside the tools (the agent cannot override a pin/forbid), soft preferences injected into prompts, and acceptance thresholds enforced by the deterministic graders.
- A **medallion** (bronze/silver/gold) lake adapter: read a named tier-scoped slice, write cleaned/inferred data back to silver/gold with a lineage manifest — **bronze/raw writes are rejected**.

---

## 🚀 Quickstart

[`scripts/setup.sh`](scripts/setup.sh) is the single supported setup entry point. It prepares
the project environment, creates or loads `.env`, prompts for missing AWS settings, validates
the selected profile, region, and target account before dependency installation or AWS writes,
bootstraps CDK, deploys the managed infrastructure, and configures the local CLI/TUI.

Before starting, configure AWS credentials for your selected account through a named
profile or the standard credential chain. Setup checks the AWS CLI’s AgentCore API
support and prepares Docker/buildx through its host helper for the default runtime
image build. Read-only preflight does not need Docker. See
[installation prerequisites](#installation-prerequisites-and-host-preparation) for host support.

> **Runtime image build requirements:** The managed-runtime image targets **Linux ARM64**.
> Use a Docker builder that supports that platform and allow space for downloads and build layers.
> Build time and disk use depend on the builder and cache. The Dockerfile selects `agents` plus
> `runtime-cpu`, which retains the data APIs while selecting CPU PyTorch wheels. Ordinary `data`
> installations retain their normal PyTorch source; uv rejects selecting both variants together.

```bash
git clone https://github.com/aws-samples/sample-ml-engineering-agent-swarm.git
cd sample-ml-engineering-agent-swarm
scripts/setup.sh
```

No separate environment, package-manager, or bootstrap command is required. Setup prints each
phase, writes a timestamped log, and can be rerun safely after a recoverable failure. The initial
dependency preparation can take 30 seconds or more; the AWS deployment takes additional time.

After setup completes:

```bash
./bin/amlas-ui
./bin/amlas --help
```

> **Cost and safety:** setup creates real, billable AWS resources. Review the
> [one-click deployment details](#-deploy-the-aws-infrastructure-one-click) before running it.

---

## ⚙️ Configuration

Two layered config sources, both with committed `.example` templates (real values never committed — account-bearing values resolve from env / SSM `/amlas/*` at runtime).

### `.agenticml.yaml` — project memory

The version-controlled project-memory file, loaded every session. Copy [`.agenticml.yaml.example`](.agenticml.yaml.example) to `.agenticml.yaml`. Resolution order (highest first): **env vars (`AMLAS_*`, `AWS_PROFILE`, `AWS_REGION`) → project `.agenticml.yaml` → user `~/.agenticml/config.yaml` → defaults**. Blocks: `aws` (profile/region/runtime ARN/session bucket), `local` (workspace/output/specs/data dirs), `lake` (medallion catalog + silver/gold prefixes), `mlops` (model-package group, feature-store DB, monitor cadence).

### `.env` layers — credentials & selectors

A shared-base root [`.env.example`](.env.example) plus thin per-component overlays loaded *after* it (a shell export always wins):

| Template | Layer | Sets |
|---|---|---|
| [`.env.example`](.env.example) | shared base | `AWS_PROFILE`/`AWS_REGION`, chat model tier, lake/MLOps names, base `*_BACKEND` selectors |
| [`.env.agentcore.example`](.env.agentcore.example) | hosted AWS prod | per-role Bedrock model tiers + **AWS** selectors (`TRAINING_BACKEND=sagemaker`, …) |
| [`.env.swarm.example`](.env.swarm.example) | local-leaning dev | budget/retreat knobs + **LOCAL** selectors (`TRAINING_BACKEND=local`, `LAKE_BACKEND=localfs`, …) |
| [`.env.tui.example`](.env.tui.example) | TUI UX | chat model + TUI theme/trace toggles |

> **Env prefix.** The canonical prefix is `AMLAS_*`. Generic `AGENTICML_*`
> aliases remain available for compatibility.

### The SSM `/amlas` config model

Account/org-bearing values are **never hardcoded**. They resolve through [`config/externalized.py`](src/amlas/config/externalized.py) with fixed precedence:

```
ENV VAR (first non-empty)  →  SSM Parameter Store (selected prefix; default /amlas/*)  →  safe default
```

- **env** for runtime selectors/tunables;
- **SSM** (`/amlas/lake-bucket`, `/amlas/guardrail-id`, `/amlas/session-bucket`, … for the default prefix) for stack-published outputs; `AMLAS_SSM_PREFIX` selects another installation;
- **Secrets Manager** for secret *values* (only the secret *id* flows through this layer);
- **CDK `self.account` / CfnParameter** for synth-time IaC inputs (the account id is regenerated from the CDK token, never hardcoded).

Required keys fail loudly (e.g. `aws_account_id` raises `MissingRequiredConfigError` when unset) — never a silent hardcoded fallback.

**The `*_BACKEND` selector model.** The registry resolves each port by **hard-pin → explicit `*_BACKEND` env → auto**. Offline vs managed in one switch:

```bash
# Offline (local, no AWS) — forces every auto resolution to the local adapter:
export AMLAS_OFFLINE=1            # or AMLAS_BACKENDS_OFFLINE=1

# Managed (real AWS) — flip individual ports to their managed adapter:
export AWS_PROFILE="<YOUR_AWS_PROFILE>" AWS_REGION="<YOUR_AWS_REGION>"
export TRAINING_BACKEND=sagemaker DEPLOYMENT_BACKEND=sagemaker LAKE_BACKEND=s3medallion
```

> `EVALUATION_BACKEND=arbiter` is the deterministic Arbiter and is **never delegated** regardless of layer.

---

## 📖 Usage / user guide

All four surfaces render the **same** capabilities through the [`capability`](src/amlas/control_plane/capability.py) funnel. Chat is primary; the others are renderers.

### 1. Interactive chat / TUI

```bash
./bin/amlas-ui          # full-screen Textual TUI (chat is the default tab)
```

The TUI is a `TabbedContent` with single-key bindings: `c` Chat · `d` Dashboard · `s` Specs · `p` Pipeline · `r` Results · `e` Data · `g` Steering · `m` MLOps · `l` Lake · `q` Quit. Key slash-commands:

| Command | Purpose |
|---|---|
| `/help` | Show available commands |
| `/flows` | List every registered ML capability (the shared flow registry) |
| `/flow <id> [--plan] [-p k=v]` | Run (or Plan-Mode preview) any registered capability via the shared funnel |
| `/dqscan <csv>` | Run the 25-pattern DQ catalog scan (honors the `/steer` dial) |
| `/diagnose <csv>` | Diagnose missingness mechanism (MCAR/MAR/MNAR) per column |
| `/impute <csv> [--method auto] [--llm] [--evaluate]` | Recommend + apply a per-column plan (honors hard steering) |
| `/synth <csv> <N> [--balance --target Y] [--validate]` | Generate / SMOTE-augment rows; optional 3-pillar validate |
| `/steer set … \| /steer show` | Set/show the session steering dial (forbid/pin/notes/thresholds) |
| `/lake read\|write\|catalog …` | Medallion read-slice / write-back / catalog |
| `/mlops pipeline\|register\|monitor\|status` | MLOps actions (`/mlops status` returns the KPI scorecard inline) |

A representative chat session (the MVP lifecycle):

```text
You: Frame this as an ML problem: predict RUL from /data/sensors.csv, target=rul, beat RMSE 12.
You: Clean and impute the missing values and prove it didn't distort the data.
You: Train the baseline and tell me if it beats RMSE 12.
     → baseline_train_flow: train → deterministic Arbiter grade → "RMSE 10.4, ACCEPTED, registered pending approval"
You: Deploy the approved model as a real-time endpoint and invoke it.
     → readiness_flow → deploy_flow  (Plan Mode preview + approval prompt, since deploy is spend·irreversible)
```

### 2. CLI (`amlas`)

Ten command groups (verified in [`cli/main.py`](src/amlas/cli/main.py#L37)): `spec`, `data`, `research`, `run`, `status`, `results`, `config`, `info`, `mlops`, `capability`.

```bash
./bin/amlas --help
./bin/amlas info health                                  # AWS/Bedrock/local readiness
./bin/amlas data quality data/ai4i/ai4i2020.csv --patterns
./bin/amlas data diagnose-missing data/ai4i/ai4i2020.csv
./bin/amlas data impute data/ai4i/ai4i2020.csv --method auto --evaluate
./bin/amlas mlops status --events output/mlops_e2e_results.json
```

### 3. Headless mode + Plan Mode + session resume (`capability`)

The `capability` group is the one-capability CLI surface over the shared funnel:

```bash
# Discover every registered capability (id · gated · runnable · milestone):
./bin/amlas capability list

# Plan-Mode preview (read-only; provisions nothing):
./bin/amlas capability plan deploy_flow

# Run through the funnel (gated flows prompt for approval):
./bin/amlas capability run data_quality_flow -p dataset_uri=data/x.csv

# HEADLESS print-mode: non-interactive, JSON out. A gated flow is DENIED unless
# allowlisted via --allow (capped by --max-budget) or --bypass:
./bin/amlas capability run data_quality_flow -p dataset_uri=data/x.csv --print
./bin/amlas capability run deploy_flow -p model_ref=… --print --allow deploy_flow --max-budget 5.0

# Resume a prior session / resolve an @-mention:
./bin/amlas capability resume my-session
./bin/amlas capability mention @dataset:silver/clean
```

> **Dataset paths:** The examples above use local CSV paths and require the in-process backend (`DQ_BACKEND=inprocess` or offline mode). When `DataQualityBackend` resolves to DataBrew, the initial profile step requires `dataset_uri` to be an `s3://` URI.

### 4. Python API

```python
# Discover + run capabilities through the shared funnel
from amlas.control_plane.capability import invoke_capability, list_capabilities

for cap in list_capabilities():
    print(cap["capability_id"], "GATED" if cap["gated"] else "read")

res = invoke_capability(capability_id="data_quality_flow",
                        params={"dataset_uri": "data/x.csv"}, surface="api")
print(res.to_dict())

# The flow registry directly
from amlas.flows import FLOWS, get_flow, list_flows
print(sorted(FLOWS))                      # all 21 flow ids
flow = get_flow("imputation_flow")        # → Flow(spend_incurring=True, …)

# Resolve a port → adapter through the registry (the seam)
from amlas.backends.registry import REGISTRY
from amlas.backends.ports import TrainingBackend, ALL_PORTS
resolved = REGISTRY.resolve(TrainingBackend)            # offline → LocalSklearnBackend
print(resolved.adapter_name, "—", resolved.reason)      # traced port→adapter→reason
print(len(ALL_PORTS), "ports")                          # 25

# The deterministic programmatic pipeline (no LLM) is still available:
from amlas.pipeline.orchestrator import MLPipeline
result = MLPipeline(output_dir="./output").run(
    spec_dict={"project_id": "ai4i", "task_type": "classification",
               "target": "Machine failure", "primary_metric": "f1_macro",
               "threshold": 0.75, "baseline_score": 0.50},
    data_path="data/ai4i/ai4i2020.csv")
print(result.decision.recommendation)
```

### Spec file format

```yaml
project_id: ai4i
task_type: classification          # classification | regression
target: Machine failure
primary_metric: f1_macro           # f1_macro | accuracy | rmse | mae
golden_dataset: data/ai4i/ai4i2020.csv
threshold: 0.75                    # acceptance threshold
baseline_score: 0.50               # score to beat
```

---

## ☁️ Backend selection & live-AWS setup

The whole point of the seam is that **the same flow runs against local or managed compute with
no code change**. The quickstart's `scripts/setup.sh` path establishes the validated AWS
profile and region, deploys the managed resources, and leaves the CLI/TUI configured for the
next launch. Advanced adapter pins remain available through the documented `.env` selectors;
they do not require a separate installation or activation flow.

The portability is an executable test, not an assertion: `BedrockFineTuneBackend` (Training)
and `BedrockHostedBackend` (Deployment) are genuine third adapters selectable with
`TRAINING_BACKEND=bedrock` / `DEPLOYMENT_BACKEND=bedrock`, returning the **same** domain
`JobHandle`/`EndpointRef` shapes the flow already consumes.

**Cost / approval guardrails against runaway spend.** Spend/irreversible flows route through `plan_approve_execute` (Plan-Mode preview + approval prompt); the `cost_ceiling_hook` (a `PreToolUse` guardrail) **fail-closes** a hard budget breach and warns+confirms a soft ceiling; per-backend least-privilege IAM ([`backend_iam_stack.py`](infra/cdk_stacks/backend_iam_stack.py)) bounds the blast radius even if a gate is bypassed; and headless runs deny gated actions unless explicitly allowlisted. The default infrastructure also includes a NAT gateway and VPC endpoints, which can incur charges while no experiment is running.

A no-credentials checkout never provisions anything by itself.

---

## 🚀 Deploy the AWS infrastructure (one-click)

The managed backends above read their account-scoped inputs (lake bucket, guardrail id,
VPC/subnets, model-package group, …) from SSM `/<deployment-prefix>/*` parameters that a small set of **AWS
CDK** stacks under [`infra/`](infra) publish. Two config-first scripts make standing that
infrastructure up — and tearing it fully back down — a single command each:

- [`scripts/setup.sh`](scripts/setup.sh) — **one-click deploy** and the authoritative setup path.
- [`scripts/teardown.sh`](scripts/teardown.sh) — **one-click teardown** of the exact owned resources recorded by setup.

Both source [`scripts/lib/deploy_common.sh`](scripts/lib/deploy_common.sh) so their config
resolution, AWS environment, and the **account safety gate** cannot drift.

> **Cost & safety.** These stacks create **real, billable** AWS resources (a NAT gateway, KMS
> CMKs, S3, a CloudTrail, VPC endpoints, …). Select the account, permissions and budget
> appropriate for your use. Before any mutation, both scripts run an **account safety gate** that aborts
> unless `aws sts get-caller-identity` **exactly** equals the configured `AMLAS_AWS_ACCOUNT_ID`.

What `setup.sh` does, in order (all teed to `output/deploy/logs/setup-<timestamp>.log`):
load `.env` → resolve profile/account/region (explicit flags override `.env`) → check the AWS CLI
prerequisite and live account identity → prepare dependencies → persist validated local selectors →
check service and model metadata → create or update the Tavily secret → bootstrap CDK when needed →
synthesize and deploy the selected stack set → verify and print the resulting status.
The local CLI/TUI keeps the validated profile and region even if a later deploy phase fails.
Managed runtimes use their IAM role rather than the developer's local named profile.

For automation, select the destination explicitly:

```bash
scripts/setup.sh --yes --profile "<YOUR_AWS_PROFILE>" \
  --account-id "<YOUR_AWS_ACCOUNT_ID>" --region "<YOUR_AWS_REGION>" --preflight-only
```

`--preflight-only` makes read-only identity, service, and model metadata checks. It does not
prove model invocation permission, available capacity, or application execution. It also verifies
local CLI API-shape support for the selected runtime; upgrade the AWS CLI when those fields are
missing, or use `--skip-agent-runtime` when you do not need that feature. Replace the
placeholders with your selected account settings; omit `--preflight-only` when
ready to deploy. Use `--skip-harness` when you do not need HarnessStack.

**Account safety gate behavior** — if the live account is missing or ≠ your configured
`AMLAS_AWS_ACCOUNT_ID`, the run aborts loudly **before dependency installation or AWS writes**:

```text
===== ACCOUNT SAFETY GATE =====
[ERROR] ACCOUNT MISMATCH — refusing to proceed.
[ERROR]   expected account: <EXPECTED_ACCOUNT_ID>
[ERROR]   actual account:   <ACTUAL_ACCOUNT_ID>  (profile=some-other-profile)
[ERROR] Safety gate FAILED — wrong account. No AWS resources were changed.
```

**`setup.sh` flags:** `--profile NAME`, `--account-id ACCOUNT`, `--region REGION`,
`--runtime-name NAME`, `--runtime-role-name NAME`, `--deployment-prefix PREFIX`,
`--ask-tavily-key`,
`--yes`/`--non-interactive`, `--skip-deps`, `--skip-bootstrap`, `--preflight-only`,
`--synth-only`, `--skip-harness`, `--skip-agent-runtime`, `--agent-runtime-only`, and `--help`.
`--synth-only` prepares local config, dependencies, logs, and CDK output; it skips Tavily secret
writes, bootstrap, deployment, image publication, and managed-runtime creation.
The setup flags select different scopes; `--synth-only` and `--preflight-only` cannot be combined.
Run `scripts/setup.sh --help` for restrictions. Runtime deployment pins its discovered platform,
enforces MMDSv2, verifies the requested configuration and runtime version, and waits
for the service-managed DEFAULT endpoint to report that live version before publishing its ARN. Set `AGENTCORE_PLATFORM_VERSION` only when selecting a
supported alternative platform. Failed operations or readiness checks return nonzero.

Use `--runtime-name` and `--runtime-role-name` to select the managed runtime and its execution
role; explicit flags override their environment or `.env` settings. Runtime names start with a
letter and contain only letters, digits, and underscores (up to 48 characters). Role names use
IAM role-name characters (up to 64). The default execution role is `<deployment-prefix>-runtime-exec-<region>`.
An existing selected role is reused only if its trust and inline policy match the requested
deployment. A mismatch fails without changing either policy; select a distinct role name or
have its owner reconcile the role explicitly before retrying. This preserves historical roles
when installing in a second Region.

### Installation prerequisites and host preparation

Start with a repository checkout, outbound HTTPS access, and an authenticated AWS
identity for your selected account. Use an EC2 instance role or your normal
short-lived AWS profile. An EC2 role supplies credentials automatically;
no copied operator credentials or persistent passwords are required. Git and
repository access are prerequisites for obtaining this checkout.

On Amazon Linux 2023 with authorized noninteractive sudo, run:

```bash
./scripts/setup.sh --yes --account-id <YOUR_ACCOUNT_ID> --region <YOUR_REGION>
```

The entrypoint verifies the selected account before host preparation and cloud
writes. It prepares Node.js/npm, Docker/buildx, CMake/compiler tools, Python 3.11,
uv and CDK through the checked-in host helper, using a writable tool/cache
workspace. It preserves a compatible AWS CLI and checks the required runtime API
fields. Docker access uses the existing daemon or a scoped sudo client wrapper.

Use `--profile <YOUR_PROFILE>` when selecting a named AWS profile. Select the
Region deliberately; there is no deployment Region default. For an isolated
installation alongside existing resources, add
`--deployment-prefix <YOUR_LOWERCASE_PREFIX>` (maximum 24 characters). The prefix
selects separate stack, IAM, storage, registry and configuration names. The
legacy default is `amlas`.

The AWS identity must allow the normal CDK bootstrap/deployment workflow and
service/model metadata reads, image publication, runtime operations and passing
the selected execution roles. The bootstrap CloudFormation execution role must
allow both creation and deletion of the deployed resources, including IAM.
Runtime execution roles created by setup are managed by a CloudFormation stack;
existing compatible execution roles are preserved.

The default security stack needs a CloudTrail trail slot in the selected Region.
Setup checks the fixed five-trail limit, including shadow trails, before cloud
writes. At capacity, retire only an explicitly authorized obsolete project trail
or select a target with room. Setup never deletes existing trails or
requests a quota increase automatically.

Amazon Linux 2 is not an automatic native installation target because its old
glibc cannot run the supported Node toolchain. Use an AL2023 test/development
host or a documented compatible container environment. Other platforms must
satisfy the helper's prerequisite diagnostics before setup; a failed check does
not authorize cloud deployment. Inspect without installing:

```bash
bash scripts/lib/prepare_host.sh --check
```

### Setup configuration

`.env` supplies persisted configuration. On first run, `setup.sh` creates it from
[`.env.example`](.env.example) and prompts for missing values. Explicit selector flags win over
shell and `.env` settings. Region lookup then uses the selected profile's local AWS config;
there is no built-in deployment region. `--yes` requires usable configured values without prompts.
The setup keys are:

| `.env` key | Purpose | Default | Req? |
|---|---|---|---|
| `AMLAS_AWS_ACCOUNT_ID` | The target account the **safety gate** asserts against (`sts get-caller-identity` must equal it) | *(none — fails loud)* | **REQUIRED** |
| `AWS_PROFILE` (or `AMLAS_AWS_PROFILE`) | Local AWS CLI profile that maps to that account; **headless preferred** (see Profiles) | standard credential chain or prompt | optional |
| `AWS_REGION` (or `AMLAS_AWS_REGION`) | Region for every `aws`/`cdk` call; service and model metadata are checked here | selected profile, then prompt | **REQUIRED** |
| `AGENTCORE_RUNTIME_NAME` | Managed AgentCore runtime name | selected deployment prefix, with hyphens converted to underscores | optional |
| `AGENTCORE_EXEC_ROLE_NAME` | IAM role for the selected runtime; existing policies must match | `<deployment-prefix>-runtime-exec-<region>` | optional |
| `AMLAS_DEPLOYMENT_PREFIX` | Isolated physical resource namespace; up to 24 lowercase characters | `amlas` | optional |
| `AMLAS_SSM_PREFIX` | Configuration path for the selected installation | `/<deployment-prefix>` | optional |
| `TAVILY_SECRET_ID` | Secrets Manager secret **id** that `SecretsStack` imports | `<deployment-prefix>/tavily-api-key` | optional |
| `AMLAS_TAVILY_API_KEY` | Optional hidden setup input; use the prompt rather than putting the value in `.env` | *(blank ⇒ no-key research)* | optional |
| `AMLAS_KMS_KEY_ARN` | BYO CMK ARN — **must stay EMPTY** so CDK mints a per-stack rotation-enabled CMK (the secure default) | *(empty)* | **leave empty** |
| `AMLAS_ENV` | `dev`/`prod`; `prod` flips the data stacks to `RemovalPolicy.RETAIN` | `dev` | optional |

> **Why `AMLAS_KMS_KEY_ARN` must stay EMPTY (FIX-08 / F15).** `.env.example` deliberately ships
> this key **empty**. If an ARN-shaped *placeholder* reaches `.env`, the CDK app could import
> that fake ARN as a BYO key — no real CMK is minted and
> `DetectionsStack` fails CFN `KmsKeyId` validation (this exact bug broke the first deploy).
> Leave it empty unless you set it to a **real, existing** `arn:aws:kms:` value;
> [`infra/app.py`](infra/app.py) additionally rejects any value containing `<` or not starting
> with `arn:aws:kms:` and falls back to the per-stack CMK.

### Optional research access

Tavily is optional. Interactive setup asks for a key with terminal echo disabled;
leave it blank to use no-key research. To request that one prompt while keeping
other defaults, add `--ask-tavily-key` to the setup command. Existing configured
secret values are preserved unless you supply a replacement. The key is sent
through stdin to Secrets Manager and is not stored by setup in `.env`, source,
command arguments or logs.

No-key operation keeps the free research sources and tries rate-limited keyless
Tavily Search. Set `AMLAS_TAVILY_KEYLESS=false` to disable that optional source.
When search is unavailable, provide public source URLs to `research_fetch_urls`
or the `urls` argument of `research_search`. Retrieval covers only supplied URLs:
at most three pages, 256 KiB per page, 15 seconds total and three validated
redirects, with no discovered-link crawl. Private/metadata addresses and URL
userinfo are rejected. Retrieved text is untrusted source data and does not
authorize tool actions. Failed sources and missing URL seeds remain explicit
coverage limits. Retrieved text is a bounded 20,000-character excerpt; each page
reports returned and available text lengths and marks truncated content as partial.

The optional secret is imported by `SecretsStack`. To configure or replace it later,
rerun setup with `--ask-tavily-key`; setup sends the value through stdin and preserves
an existing configured value when no replacement is supplied.

### What gets deployed

`setup.sh` deploys these 9 stacks by default, in dependency order (from
[`scripts/lib/deploy_common.sh`](scripts/lib/deploy_common.sh)) — **never `cdk deploy --all`**
(`RuntimeStack` collides with `AsyncTrainingStack` on fixed-name resources and needs an
out-of-band image; `BackendIamStack` is synth-only — both are intentionally excluded).
`--skip-harness` removes HarnessStack from synthesis, preflight, deployment, and verification:

| # | Stack | Provides |
|---|---|---|
| 1 | `AgentCoreRuntimeStack` | ECR repo `amlas-runtime` for the runtime image + `/amlas/runtime-ecr-uri` |
| 2 | `LakeStack` | Medallion S3 lake (`amlas-lake-<acct>-<region>`, own CMK) + Glue DB + `/amlas/lake-*` |
| 3 | `ModelRegistryStack` | Governed SageMaker Model Registry group `amlas-models` |
| 4 | `SecretsStack` | Imports the Tavily secret + publishes `/amlas/tavily-secret-id` |
| 5 | `AsyncTrainingStack` | Async-training plane: session bucket, DDB `amlas-training-jobs`, SageMaker exec role, callback Lambda |
| 6 | `NetworkStack` | PrivateLink VPC + interface/gateway endpoints (+ 1 NAT) + `/amlas/vpc-id` |
| 7 | `DetectionsStack` | CloudTrail security detections: trail bucket (own CMK), alarms, SNS |
| 8 | `GuardrailStack` | Bedrock Guardrail (`amlas-guardrail`) + `/amlas/guardrail-*` |
| 9 | `HarnessStack` | Managed AgentCore harnesses and their SSM discovery parameters |

The table shows names for the default `amlas` prefix. A selected
`--deployment-prefix` changes physical stack/resource names and the SSM path
while retaining the nine logical stack selections.

**Bootstrap runs automatically.** A fresh account needs `cdk bootstrap` first; `setup.sh` does
it for you (STEP 8) and **skips** it when the `/cdk-bootstrap/hnb659fds/version` SSM parameter
already exists — so you never run `cdk bootstrap` by hand.

### Verify the deploy

`setup.sh` prints per-stack status and the selected SSM namespace at the end.
Use the same profile, Region and prefix to inspect the installation independently:

```bash
# Check the selected installation stacks for CREATE_COMPLETE / UPDATE_COMPLETE:
aws cloudformation list-stacks --profile "$AWS_PROFILE" --region "$AWS_REGION" \
  --query "StackSummaries[?StackStatus=='CREATE_COMPLETE'||StackStatus=='UPDATE_COMPLETE'].StackName"

# Read the configuration path selected by setup (default /amlas):
aws ssm get-parameters-by-path --path "${AMLAS_SSM_PREFIX:-/amlas}" --recursive --profile "$AWS_PROFILE" --region "$AWS_REGION" \
  --query "Parameters[].[Name,Value]" --output text
```

### Developer verification

Developer verification uses a separate environment; it is not required to run
the installed application. A new shell must activate the tools installed by the
checked-in host helper; setup activates them only in its own process. From the
verified checkout, use the existing activation seam (no package installation or
global shell-profile change):

```bash
source scripts/lib/prepare_host.sh
_amlas_host_configure_paths
command -v uv node npm docker
```

Keep TMPDIR and caches in the dedicated temporary workspace outside the checkout.
Select a separate verifier path and prepare the declared locked dependencies:

```bash
UV_PROJECT_ENVIRONMENT=<VERIFY_ENV_PATH> uv sync --frozen \
  --extra dev --extra cdk --extra agents --extra runtime-cpu
UV_PROJECT_ENVIRONMENT=<VERIFY_ENV_PATH> uv run --frozen --no-sync python -m pytest tests
```

### Teardown

Setup records its owned resources in `output/deploy/deployment-manifest.json`.
Retain the manifest and preview the exact installation before cleanup:

```bash
scripts/teardown.sh --manifest output/deploy/deployment-manifest.json --dry-run
scripts/teardown.sh --manifest output/deploy/deployment-manifest.json --yes
```

Cleanup validates recorded native identities and ownership tags, stops recorded
runtime sessions, removes the managed runtime and its auxiliary role stack,
deletes the selected stacks in reverse order, and removes exact owned retained
resources. Pre-existing/shared resources and the CDK bootstrap are preserved;
`--nuke-bootstrap` is rejected. A resource-name prefix alone cannot authorize deletion.

KMS requires at least seven days before deletion. Pending key deletion is reported
separately from confirmed resource absence. Prerequisite, synthesis and image checks
also remain distinct from a successful managed experiment.

### Profiles (headless vs SSO)

Use a **headless** profile (credential-process / credential-helper — no browser) for automation:
it lets `setup.sh --yes` run unattended. An interactive **SSO** profile also works for a manual
run. In either case, the profile must map to the configured target account; `setup.sh` verifies
that identity before making any mutation.

The safety gate protects every profile choice: deployment aborts unless the live
account equals the configured account.

## 📁 Project layout

```
sample-ml-engineering-agent-swarm/     # repo root (distribution: agentic-ml-lifecycle-automation-surface)
├── pyproject.toml                 # build, dependencies, and console scripts
├── LICENSE  NOTICE                # MIT-0 + attribution
├── .agenticml.yaml.example        # project memory
├── .env.example                   # shared-base env + *_BACKEND selectors
├── .env.{agentcore,swarm,tui}.example   # per-layer overlays
├── bin/{amlas,amlas-ui}           # canonical CLI/TUI launchers
│
├── src/amlas/
│   ├── backends/                  # THE SEAM — ports & adapters
│   │   ├── ports.py               #   the 25 Protocol ports + ALL_PORTS + port_key
│   │   ├── types.py               #   domain value types (JobHandle, ModelRef, …)
│   │   ├── registry.py            #   BackendRegistry.resolve(port, steering, scale, budget)
│   │   └── adapters/              #   per-port adapters (sagemaker / local / bedrock_alt / …)
│   ├── control_plane/             # NL→tool-call control plane (11 modules)
│   │   ├── capability.py          #   invoke_capability — the single funnel
│   │   ├── approvals.py plan_mode.py guardrails.py   # safe-by-default gates
│   │   └── traceability.py cost_governance.py mcp_bridge.py …
│   ├── flows/                     # the 21 opinionated flows (flow_id → Flow + run_<id>)
│   ├── leads/  subagents/         # the 7 Lead agents + the specialist subagent swarm
│   ├── tools/                     # @tool fns (flat backend-neutral modules + 6 subpackages: data_quality/impute/synth/mlops/lake/research)
│   ├── arbiter/                   # the deterministic Arbiter (EvaluationBackend default)
│   ├── data_eng/                  # the SteeringProfile dial + data-engineering logic
│   ├── config/                    # externalized SSM/env config + back-compat seam
│   ├── cli/ tui/                  # the CLI (10 groups in cli/commands/) + the Textual TUI (9 panes)
│   └── pipeline/ evaluation/ knowledge/ memory/ telemetry/ cloud/ …
│
├── infra/                         # CDK IaC: app.py (the app entry) + cdk_stacks/ (lake, registry, runtime, per-backend IAM, …)
├── scripts/                       # setup/teardown entry points + private runtime helpers
├── prompts/                       # runtime agent prompts
├── specs/                         # one example ML specification
└── data/                          # local datasets (ignored; supply your own)
```

---

## 🔐 Security

Please **do not** open public issues for security vulnerabilities. See
[`SECURITY.md`](SECURITY.md) for reporting guidance and the project security posture.

See the [sample-code notice at the top of this README](#agentic-ml-engineering) for deployment responsibilities.

## 📄 License & attribution

Licensed under the **MIT No Attribution License (MIT-0)** — see [`LICENSE`](LICENSE). Attribution for prompt/pattern inspirations (no third-party source is vendored or redistributed) is recorded in [`NOTICE`](NOTICE), and includes ml-ralph (Apache-2.0), MetaGPT (MIT), AutoKaggle, mle-bench (MIT), and the research works R&D-Agent (arXiv:2505.14738), MLE-STAR (arXiv:2506.15692), and CAAFE (arXiv:2305.03403). Canonical AWS Strands + Bedrock AgentCore integration patterns follow the published aws-samples conventions.
