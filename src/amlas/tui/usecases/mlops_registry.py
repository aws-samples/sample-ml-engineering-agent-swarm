"""MLOps use-case registry — the second half of the lifecycle (plan §C).

The 100-entry [`registry.py`](registry.py) covers data quality / data engineering
(``DQ-*``). This module covers the MLOps half — train → HPO → evaluate → register →
deploy → monitor → retrain → govern, plus the two capabilities Tracks A and B added
(agent-driven AWS configuration and chat-driven dashboard charts).

**Why a sibling module rather than more ``DQ-*`` entries:** ``registry.py`` has three
coverage-completeness tests asserting exactly 100 ids with exact per-category counts, and a
Markdown pair generated from it. Appending here keeps that contract untouched while the
Phase-8 harness can drive both sets, because every entry reuses the SAME
:class:`~amlas.tui.usecases.registry.UseCase` schema.

Each entry is grounded in a REAL registered flow ([`flows/__init__.py`](../../flows/__init__.py))
or a REAL ``@tool``; ``expected_tools`` names are verified against the live tool library by
``validation coverage`` so an entry can never reference fiction.

Chat-driven invariant (identical to the DQ set): the user types in Chat → the agent invokes
a tool / ``run_ml_capability`` flow → the ``ContextBus`` publishes typed events → the
subscribed tabs re-render.
"""

from __future__ import annotations

from .registry import UseCase

__all__ = [
    "MLOPS_CATEGORIES",
    "MLOPS_CATEGORY_COUNTS",
    "MLOPS_CATEGORY_LABELS",
    "MLOPS_USE_CASES",
    "get_mlops_use_case",
    "mlops_category_counts",
    "mlops_use_cases_for_category",
]

# The 8-area MLOps taxonomy. Counts are the contract the completeness tests enforce.
MLOPS_CATEGORY_COUNTS: dict[str, int] = {
    "training_baseline": 8,
    "hpo_tuning": 6,
    "evaluation_selection": 8,
    "registry_governance": 8,
    "deploy_serving": 8,
    "monitoring_drift": 8,
    "retrain_lifecycle": 6,
    "platform_control": 8,
}

MLOPS_CATEGORIES: tuple[str, ...] = tuple(MLOPS_CATEGORY_COUNTS.keys())

MLOPS_CATEGORY_LABELS: dict[str, str] = {
    "training_baseline": "Training & Baselines",
    "hpo_tuning": "HPO & Tuning",
    "evaluation_selection": "Evaluation & Model Selection",
    "registry_governance": "Model Registry & Governance",
    "deploy_serving": "Deployment & Serving",
    "monitoring_drift": "Monitoring & Drift",
    "retrain_lifecycle": "Retraining & Lifecycle",
    "platform_control": "Platform Control (AWS config + dashboard)",
}

MLOPS_USE_CASES: list[UseCase] = []


# ─── Query helpers (mirror the DQ registry's API) ─────────────────────────────
def get_mlops_use_case(uc_id: str) -> UseCase | None:
    """Return the :class:`UseCase` with ``id == uc_id`` (or ``None``)."""
    for uc in MLOPS_USE_CASES:
        if uc.id == uc_id:
            return uc
    return None


def mlops_use_cases_for_category(category: str) -> list[UseCase]:
    """Return all MLOps use cases in ``category`` (registry order)."""
    return [uc for uc in MLOPS_USE_CASES if uc.category == category]


def mlops_category_counts() -> dict[str, int]:
    """Return the observed per-category count of :data:`MLOPS_USE_CASES`."""
    counts: dict[str, int] = {c: 0 for c in MLOPS_CATEGORIES}
    for uc in MLOPS_USE_CASES:
        counts[uc.category] = counts.get(uc.category, 0) + 1
    return counts


# ═══════════════════════════════════════════════════════════════════════════════
# Area 1 — Training & Baselines (8)  [ML-TRAIN-001..008]
# Grounded: baseline_train_flow, train_model, submit_training_run, run_processing_job,
# log_experiment_run, evaluate_offline_metrics.
# ═══════════════════════════════════════════════════════════════════════════════
MLOPS_USE_CASES += [
    UseCase(
        id="ML-TRAIN-001",
        category="training_baseline",
        title="Train a baseline model from the active spec",
        user_story=(  # nosemgrep: string-concat-in-list
            "As an ML engineer I want the agent to train a baseline from my spec "
            "so that I have a reference score before investing in tuning."
        ),
        chat_prompt="train a baseline model for the active spec and report the primary metric",
        expected_capability="baseline_train_flow",
        expected_tools=("run_ml_capability",),
        expected_events=("job", "agents"),
        expected_panel_updates=("pipeline", "dashboard.active_jobs"),
        acceptance=(  # nosemgrep: string-concat-in-list
            "GIVEN an active spec WHEN the agent dispatches baseline_train_flow THEN a job "
            "is published to the bus AND the Pipeline + Active-Jobs panels show it running."
        ),
    ),
    UseCase(
        id="ML-TRAIN-002",
        category="training_baseline",
        title="Dispatch training to SageMaker rather than in-process",
        user_story=(  # nosemgrep: string-concat-in-list
            "As an ML engineer I want training dispatched to SageMaker so that "
            "large datasets do not run in the agent container (ADR-0002)."
        ),
        chat_prompt="submit this training run to SageMaker and tell me the job name",
        expected_capability="",
        expected_tools=("submit_training_run",),
        expected_events=("job", "agents"),
        expected_panel_updates=("pipeline", "dashboard.active_jobs"),
        acceptance=(  # nosemgrep: string-concat-in-list
            "GIVEN a training request WHEN the agent calls submit_training_run THEN the "
            "compute runs on SageMaker (never in-container) AND a job appears on the bus."
        ),
        offline=False,
        supported_today=True,
    ),
    UseCase(
        id="ML-TRAIN-003",
        category="training_baseline",
        title="Compare two algorithms on the same split",
        user_story=(  # nosemgrep: string-concat-in-list
            "As a data scientist I want two algorithms trained on an identical split "
            "so that the comparison is not confounded by data differences."
        ),
        chat_prompt="train xgboost and a random forest on the same split and compare them",
        expected_capability="",
        expected_tools=("train_model", "compare_experiments"),
        expected_events=("job", "agents"),
        expected_panel_updates=("results", "dashboard.active_jobs"),
        acceptance=(  # nosemgrep: string-concat-in-list
            "GIVEN one prepared split WHEN the agent trains both algorithms THEN both "
            "results are recorded AND compare_experiments reports them side by side."
        ),
    ),
    UseCase(
        id="ML-TRAIN-004",
        category="training_baseline",
        title="Log a training run to experiment tracking",
        user_story=(  # nosemgrep: string-concat-in-list
            "As an ML engineer I want every run logged with its hyperparameters "
            "so that results stay reproducible and auditable."
        ),
        chat_prompt="log the last training run with its hyperparameters and metrics",
        expected_capability="",
        expected_tools=("log_experiment_run",),
        expected_events=("run", "agents"),
        expected_panel_updates=("results", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep: string-concat-in-list
            "GIVEN a completed training run WHEN the agent logs it THEN the run is "
            "retrievable via search_runs AND the Results tab scopes to it."
        ),
    ),
    UseCase(
        id="ML-TRAIN-005",
        category="training_baseline",
        title="Run feature preparation as a SageMaker Processing job",
        user_story=(  # nosemgrep: string-concat-in-list
            "As a data engineer I want feature preparation to run as a Processing job "
            "so that heavy transforms offload to purpose-built compute."
        ),
        chat_prompt="run the feature preparation for this dataset as a processing job",
        expected_capability="",
        expected_tools=("run_processing_job",),
        expected_events=("job", "processes", "agents"),
        expected_panel_updates=("pipeline", "dashboard.processes"),
        acceptance=(  # nosemgrep: string-concat-in-list
            "GIVEN a dataset WHEN the agent submits a Processing job THEN a job + process "
            "row are published AND the Pipeline/Processes panels reflect them."
        ),
        offline=False,
    ),
    UseCase(
        id="ML-TRAIN-006",
        category="training_baseline",
        title="Halt training when a leakage check fails",
        user_story=(  # nosemgrep: string-concat-in-list
            "As a data scientist I want training blocked on a failed leakage check "
            "so that I never ship a model trained on leaked signal."
        ),
        chat_prompt="check this train/test split for leakage before training anything",
        expected_capability="",
        expected_tools=("leakage_check_tool",),
        expected_events=("dataset", "agents"),
        expected_panel_updates=("data", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep: string-concat-in-list
            "GIVEN a leaky split WHEN the agent runs the leakage check THEN blocking=True "
            "is reported AND no training is dispatched."
        ),
    ),
    UseCase(
        id="ML-TRAIN-007",
        category="training_baseline",
        title="Estimate training cost before spending",
        user_story=(  # nosemgrep: string-concat-in-list
            "As an ML engineer I want a cost/runtime estimate before a cloud run "
            "so that I approve spend with a number in front of me."
        ),
        chat_prompt="what would 40 trials cost before you run anything?",
        expected_capability="",
        expected_tools=("estimate_dispatch_cost",),
        expected_events=("agents",),
        expected_panel_updates=("chat", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep: string-concat-in-list
            "GIVEN a trial budget WHEN the agent estimates cost THEN a low/high band + "
            "assumptions are reported AND no spend occurs."
        ),
    ),
    UseCase(
        id="ML-TRAIN-008",
        category="training_baseline",
        title="Plot the training curve on the dashboard",
        user_story=(  # nosemgrep: string-concat-in-list
            "As a data scientist I want the training curve charted in the TUI "
            "so that I can see convergence instead of reading numbers."
        ),
        chat_prompt="plot the training curve for the last run on the dashboard",
        expected_capability="",
        expected_tools=("render_dashboard_view",),
        expected_events=("view", "agents"),
        expected_panel_updates=("dashboard.dynamic_primary",),
        acceptance=(  # nosemgrep: string-concat-in-list
            "GIVEN a run with a metric series WHEN the agent renders a line chart THEN a "
            "view event fires AND the Dashboard primary pane shows the curve."
        ),
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Area 2 — HPO & Tuning (6)  [ML-HPO-001..006]
# Grounded: hpo_flow, run_hpo, tune_model, train_ensemble, run_automl.
# ═══════════════════════════════════════════════════════════════════════════════
MLOPS_USE_CASES += [
    UseCase(
        id="ML-HPO-001",
        category="hpo_tuning",
        title="Run hyperparameter optimization for the active spec",
        user_story=(  # nosemgrep: string-concat-in-list
            "As an ML engineer I want HPO run against my spec's metric "
            "so that I get the best configuration without hand-tuning."
        ),
        chat_prompt="run hyperparameter optimization for the active spec with 20 trials",
        expected_capability="hpo_flow",
        expected_tools=("run_ml_capability",),
        expected_events=("job", "agents"),
        expected_panel_updates=("pipeline", "dashboard.active_jobs"),
        acceptance=(  # nosemgrep: string-concat-in-list
            "GIVEN an active spec WHEN the agent dispatches hpo_flow THEN a job is "
            "published AND the Pipeline + Active-Jobs panels track the trials."
        ),
    ),
    UseCase(
        id="ML-HPO-002",
        category="hpo_tuning",
        title="Propose a search space from the algorithm choice",
        user_story=(  # nosemgrep: string-concat-in-list
            "As a data scientist I want a sensible search space proposed "
            "so that I do not guess bounds for every hyperparameter."
        ),
        chat_prompt="what search space should we tune for xgboost on this problem?",
        expected_capability="",
        expected_tools=("recommend_approaches",),
        expected_events=("agents",),
        expected_panel_updates=("chat", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep: string-concat-in-list
            "GIVEN an algorithm WHEN the agent recommends a search space THEN concrete "
            "parameter ranges are reported with rationale."
        ),
    ),
    UseCase(
        id="ML-HPO-003",
        category="hpo_tuning",
        title="Tune with a managed SageMaker tuning job",
        user_story=(  # nosemgrep: string-concat-in-list
            "As an ML engineer I want tuning run on SageMaker AMT "
            "so that trials run in parallel on managed infrastructure."
        ),
        chat_prompt="tune this model using a managed sagemaker tuning job",
        expected_capability="",
        expected_tools=("run_hpo",),
        expected_events=("job", "agents"),
        expected_panel_updates=("pipeline", "dashboard.active_jobs"),
        acceptance=(  # nosemgrep: string-concat-in-list
            "GIVEN a tuning request WHEN the agent calls run_hpo THEN the managed tuning "
            "job is created AND its progress is observable on the bus."
        ),
        offline=False,
    ),
    UseCase(
        id="ML-HPO-004",
        category="hpo_tuning",
        title="Build an ensemble from the best trials",
        user_story=(  # nosemgrep: string-concat-in-list
            "As a data scientist I want an ensemble of the top trials "
            "so that I capture the gain that a single model misses."
        ),
        chat_prompt="build an ensemble from the best trials so far",
        expected_capability="",
        expected_tools=("train_ensemble",),
        expected_events=("run", "agents"),
        expected_panel_updates=("results", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep: string-concat-in-list
            "GIVEN multiple recorded trials WHEN the agent trains an ensemble THEN its "
            "score is recorded AND the Results tab shows it against the singles."
        ),
    ),
    UseCase(
        id="ML-HPO-005",
        category="hpo_tuning",
        title="Chart the HPO trial scores",
        user_story=(  # nosemgrep: string-concat-in-list
            "As an ML engineer I want the trial scores charted "
            "so that I can see whether the search has plateaued."
        ),
        chat_prompt="chart the trial scores from the tuning run",
        expected_capability="",
        expected_tools=("render_dashboard_view",),
        expected_events=("view", "agents"),
        expected_panel_updates=("dashboard.dynamic_primary",),
        acceptance=(  # nosemgrep: string-concat-in-list
            "GIVEN recorded trials WHEN the agent renders the scores THEN a view event "
            "fires AND the Dashboard shows the per-trial chart."
        ),
    ),
    UseCase(
        id="ML-HPO-006",
        category="hpo_tuning",
        title="Raise the quota that aborted a tuning run",
        user_story=(  # nosemgrep: string-concat-in-list
            "As an ML engineer I want the agent to raise a limit that killed my run "
            "so that I am not blocked waiting on a human to file a quota ticket."
        ),
        chat_prompt="the tuning run hit a quota limit — request an increase",
        expected_capability="",
        expected_tools=("request_quota_increase",),
        expected_events=("agents",),
        expected_panel_updates=("chat", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep: string-concat-in-list
            "GIVEN a quota-exceeded failure WHEN the agent proposes an increase THEN the "
            "PLAN is shown first AND the request is only filed after confirmation."
        ),
        offline=False,
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Area 3 — Evaluation & Model Selection (8)  [ML-EVAL-001..008]
# Grounded: evaluation_flow, evaluate_offline_metrics, get_leaderboard,
# compare_experiments, get_best_model, get_recommendation, build_model_comparison_table.
# ═══════════════════════════════════════════════════════════════════════════════
MLOPS_USE_CASES += [
    UseCase(
        id="ML-EVAL-001",
        category="evaluation_selection",
        title="Compute offline metrics for a candidate",
        user_story=(  # nosemgrep: string-concat-in-list
            "As a data scientist I want the full offline metric set for a candidate "
            "so that I judge it on more than the headline number."
        ),
        chat_prompt="compute the offline metrics for the latest candidate model",
        expected_capability="evaluation_flow",
        expected_tools=("run_ml_capability",),
        expected_events=("run", "agents"),
        expected_panel_updates=("results", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep: string-concat-in-list
            "GIVEN a candidate WHEN the agent dispatches evaluation_flow THEN metrics + an "
            "arbiter verdict are published AND the Results tab reflects them."
        ),
    ),
    UseCase(
        id="ML-EVAL-002",
        category="evaluation_selection",
        title="Show the experiment leaderboard",
        user_story=(  # nosemgrep: string-concat-in-list
            "As an ML engineer I want a ranked leaderboard "
            "so that I can see which algorithm currently leads."
        ),
        chat_prompt="show me the leaderboard for this project",
        expected_capability="",
        expected_tools=("get_leaderboard",),
        expected_events=("agents",),
        expected_panel_updates=("results", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep: string-concat-in-list
            "GIVEN recorded experiments WHEN the agent reads the leaderboard THEN the "
            "ranked rows are reported from the real evaluation store."
        ),
    ),
    UseCase(
        id="ML-EVAL-003",
        category="evaluation_selection",
        title="Chart the leaderboard as a bar chart",
        user_story=(  # nosemgrep: string-concat-in-list
            "As a data scientist I want the leaderboard charted "
            "so that the gap between algorithms is visible at a glance."
        ),
        chat_prompt="chart the leaderboard on the dashboard as a bar chart",
        expected_capability="",
        expected_tools=("render_dashboard_view",),
        expected_events=("view", "agents"),
        expected_panel_updates=("dashboard.dynamic_primary",),
        acceptance=(  # nosemgrep: string-concat-in-list
            "GIVEN a populated evaluation store WHEN the agent renders source='leaderboard' "
            "THEN the chart plots REAL recorded scores AND a view event fires."
        ),
    ),
    UseCase(
        id="ML-EVAL-004",
        category="evaluation_selection",
        title="Pick the best model with a statistical comparison",
        user_story=(  # nosemgrep: string-concat-in-list
            "As a data scientist I want the best model chosen with significance testing "
            "so that I do not promote a difference that is noise."
        ),
        chat_prompt="which model is best, and is the difference statistically significant?",
        expected_capability="",
        expected_tools=("get_best_model", "compare_experiments"),
        expected_events=("run", "agents"),
        expected_panel_updates=("results", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep: string-concat-in-list
            "GIVEN two or more runs WHEN the agent compares them THEN the winner + p-value "
            "/ effect size are reported AND the active run points at the winner."
        ),
    ),
    UseCase(
        id="ML-EVAL-005",
        category="evaluation_selection",
        title="Show the confusion matrix for a classifier",
        user_story=(  # nosemgrep: string-concat-in-list
            "As a data scientist I want the confusion matrix charted "
            "so that I can see which class the model actually fails on."
        ),
        chat_prompt="show me the confusion matrix for the best classifier",
        expected_capability="",
        expected_tools=("render_dashboard_view",),
        expected_events=("view", "agents"),
        expected_panel_updates=("dashboard.dynamic_primary",),
        acceptance=(  # nosemgrep: string-concat-in-list
            "GIVEN a classifier's predictions WHEN the agent renders chart='confusion' THEN "
            "per-class counts are shown as numbers (not shading) on the Dashboard."
        ),
    ),
    UseCase(
        id="ML-EVAL-006",
        category="evaluation_selection",
        title="Plot predicted vs actual for a regressor",
        user_story=(  # nosemgrep: string-concat-in-list
            "As a data scientist I want a predicted-vs-actual scatter with the fit "
            "so that I can see where the regressor breaks down."
        ),
        chat_prompt="plot predicted vs actual for the regression model with the fit line",
        expected_capability="",
        expected_tools=("render_dashboard_view",),
        expected_events=("view", "agents"),
        expected_panel_updates=("dashboard.dynamic_primary",),
        acceptance=(  # nosemgrep: string-concat-in-list
            "GIVEN regression predictions WHEN the agent renders chart='scatter' THEN the "
            "chart reports R² + slope AND the Dashboard shows the fit."
        ),
    ),
    UseCase(
        id="ML-EVAL-007",
        category="evaluation_selection",
        title="Get a deploy / iterate / reject recommendation",
        user_story=(  # nosemgrep: string-concat-in-list
            "As an ML engineer I want an explicit recommendation "
            "so that the next step is a decision, not an open question."
        ),
        chat_prompt="should we deploy this model, iterate, or reject it?",
        expected_capability="",
        expected_tools=("get_recommendation",),
        expected_events=("run", "agents"),
        expected_panel_updates=("results", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep: string-concat-in-list
            "GIVEN an evaluated run WHEN the agent recommends THEN a deploy/iterate/reject "
            "decision with reasons is published to the active run."
        ),
    ),
    UseCase(
        id="ML-EVAL-008",
        category="evaluation_selection",
        title="Build the model comparison table",
        user_story=(  # nosemgrep: string-concat-in-list
            "As an ML engineer I want a side-by-side comparison table "
            "so that I can put one artifact in a review."
        ),
        chat_prompt="build a model comparison table across every run so far",
        expected_capability="",
        expected_tools=("build_model_comparison_table",),
        expected_events=("agents",),
        expected_panel_updates=("results", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep: string-concat-in-list
            "GIVEN multiple runs WHEN the agent builds the table THEN every run appears "
            "with its metrics AND the Results tab renders it."
        ),
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Area 4 — Model Registry & Governance (8)  [ML-REG-001..008]
# Grounded: register_model, create_model_card, query_lineage, run_guardrail_check,
# generate_model_card_tool, check_data_access, readiness_flow.
# ═══════════════════════════════════════════════════════════════════════════════
MLOPS_USE_CASES += [
    UseCase(
        id="ML-REG-001",
        category="registry_governance",
        title="Register the winning model in the registry",
        user_story=(  # nosemgrep: string-concat-in-list
            "As an ML engineer I want the winner registered with its version "
            "so that deployment references an immutable artifact."
        ),
        chat_prompt="register the best model in the model registry",
        expected_capability="",
        expected_tools=("register_model",),
        expected_events=("model", "agents"),
        expected_panel_updates=("mlops", "dashboard.observability"),
        acceptance=(  # nosemgrep: string-concat-in-list
            "GIVEN a winning run WHEN the agent registers it THEN a model version is "
            "published to the bus AND the MLOps tab shows it."
        ),
        offline=False,
    ),
    UseCase(
        id="ML-REG-002",
        category="registry_governance",
        title="Generate a model card",
        user_story=(  # nosemgrep: string-concat-in-list
            "As a governance reviewer I want a model card generated "
            "so that intended use and limitations are documented before deploy."
        ),
        chat_prompt="generate a model card for the registered model",
        expected_capability="",
        expected_tools=("generate_model_card_tool",),
        expected_events=("model", "agents"),
        expected_panel_updates=("mlops", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep: string-concat-in-list
            "GIVEN a registered model WHEN the agent generates the card THEN the artifact "
            "is written AND the MLOps tab references it."
        ),
    ),
    UseCase(
        id="ML-REG-003",
        category="registry_governance",
        title="Trace a model's data lineage",
        user_story=(  # nosemgrep: string-concat-in-list
            "As an auditor I want a model's lineage "
            "so that I can prove which data produced it."
        ),
        chat_prompt="trace the lineage of the registered model back to its source data",
        expected_capability="",
        expected_tools=("query_lineage",),
        expected_events=("dataset", "agents"),
        expected_panel_updates=("mlops", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep: string-concat-in-list
            "GIVEN a registered model WHEN the agent queries lineage THEN the upstream "
            "datasets/jobs are reported."
        ),
    ),
    UseCase(
        id="ML-REG-004",
        category="registry_governance",
        title="Run the production-readiness gates",
        user_story=(  # nosemgrep: string-concat-in-list
            "As a release manager I want the readiness gates run "
            "so that go/no-go is evidence-based."
        ),
        chat_prompt="run the production readiness review for this model",
        expected_capability="readiness_flow",
        expected_tools=("run_ml_capability",),
        expected_events=("model", "agents"),
        expected_panel_updates=("mlops", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep: string-concat-in-list
            "GIVEN a candidate WHEN the agent dispatches readiness_flow THEN each gate's "
            "pass/fail + a risk register are reported."
        ),
    ),
    UseCase(
        id="ML-REG-005",
        category="registry_governance",
        title="Run a guardrail check before release",
        user_story=(  # nosemgrep: string-concat-in-list
            "As a governance reviewer I want guardrails checked "
            "so that a policy violation blocks release rather than surfacing later."
        ),
        chat_prompt="run the guardrail checks against this model before we release it",
        expected_capability="",
        expected_tools=("run_guardrail_check",),
        expected_events=("model", "agents"),
        expected_panel_updates=("mlops", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep: string-concat-in-list
            "GIVEN a candidate WHEN the agent runs guardrails THEN a PASS/FAIL verdict "
            "with findings is reported AND a FAIL blocks the release path."
        ),
    ),
    UseCase(
        id="ML-REG-006",
        category="registry_governance",
        title="Verify data-access permissions for the model's inputs",
        user_story=(  # nosemgrep: string-concat-in-list
            "As a security reviewer I want input data access verified "
            "so that the serving role can actually read what it needs."
        ),
        chat_prompt="check that the serving role can access this model's input data",
        expected_capability="",
        expected_tools=("check_data_access",),
        expected_events=("agents",),
        expected_panel_updates=("mlops", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep: string-concat-in-list
            "GIVEN a model + its inputs WHEN the agent checks access THEN the effective "
            "permission result is reported per resource."
        ),
    ),
    UseCase(
        id="ML-REG-007",
        category="registry_governance",
        title="Grant the missing permission a governance check found",
        user_story=(  # nosemgrep: string-concat-in-list
            "As an ML engineer I want the missing grant applied for me "
            "so that a permission gap does not become a multi-day ticket."
        ),
        chat_prompt="the serving role cannot read the lake prefix — grant it read access",
        expected_capability="",
        expected_tools=("grant_role_policy",),
        expected_events=("agents",),
        expected_panel_updates=("mlops", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep: string-concat-in-list
            "GIVEN a missing grant WHEN the agent proposes it THEN a SCOPED policy document "
            "is shown as a plan, wildcards are refused, and it applies only after confirmation."
        ),
        offline=False,
    ),
    UseCase(
        id="ML-REG-008",
        category="registry_governance",
        title="Create the SageMaker model card record",
        user_story=(  # nosemgrep: string-concat-in-list
            "As a governance reviewer I want the card recorded in SageMaker "
            "so that it lives with the model, not in a document."
        ),
        chat_prompt="create the sagemaker model card for the registered model",
        expected_capability="",
        expected_tools=("create_model_card",),
        expected_events=("model", "agents"),
        expected_panel_updates=("mlops", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep: string-concat-in-list
            "GIVEN a registered model WHEN the agent creates the model card THEN the "
            "managed card exists AND the MLOps tab reflects it."
        ),
        offline=False,
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Area 5 — Deployment & Serving (8)  [ML-DEP-001..008]
# Grounded: deploy_flow, package_model, deploy_endpoint, invoke_endpoint,
# update_endpoint_canary, run_batch_transform, rollback_deployment,
# recommend_inference_config, select_deployment_strategy.
# ═══════════════════════════════════════════════════════════════════════════════
MLOPS_USE_CASES += [
    UseCase(
        id="ML-DEP-001",
        category="deploy_serving",
        title="Deploy the approved model to a real-time endpoint",
        user_story=(  # nosemgrep: string-concat-in-list
            "As an ML engineer I want the approved model deployed "
            "so that downstream services can call it."
        ),
        chat_prompt="deploy the approved model to a real-time endpoint",
        expected_capability="deploy_flow",
        expected_tools=("run_ml_capability",),
        expected_events=("model", "job", "agents"),
        expected_panel_updates=("mlops", "dashboard.observability"),
        acceptance=(  # nosemgrep: string-concat-in-list
            "GIVEN an approved model WHEN the agent dispatches deploy_flow THEN the "
            "approval gate fires first AND the endpoint is published to the bus."
        ),
        offline=False,
    ),
    UseCase(
        id="ML-DEP-002",
        category="deploy_serving",
        title="Recommend the right instance/serving configuration",
        user_story=(  # nosemgrep: string-concat-in-list
            "As an ML engineer I want a recommended serving configuration "
            "so that I am not guessing instance type and count."
        ),
        chat_prompt="what serving configuration should we use for this model?",
        expected_capability="",
        expected_tools=("recommend_inference_config",),
        expected_events=("model", "agents"),
        expected_panel_updates=("mlops", "dashboard.observability"),
        acceptance=(  # nosemgrep: string-concat-in-list
            "GIVEN a registered model WHEN the agent recommends a config THEN instance "
            "type/count + rationale are reported."
        ),
    ),
    UseCase(
        id="ML-DEP-003",
        category="deploy_serving",
        title="Choose a deployment strategy",
        user_story=(  # nosemgrep: string-concat-in-list
            "As a release manager I want a deployment strategy chosen "
            "so that rollout risk matches the change's blast radius."
        ),
        chat_prompt="should this go out as a canary, blue/green, or all at once?",
        expected_capability="",
        expected_tools=("select_deployment_strategy",),
        expected_events=("model", "agents"),
        expected_panel_updates=("mlops", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep: string-concat-in-list
            "GIVEN a deployment request WHEN the agent selects a strategy THEN the choice "
            "+ its rollback story are reported."
        ),
    ),
    UseCase(
        id="ML-DEP-004",
        category="deploy_serving",
        title="Shift traffic with a canary update",
        user_story=(  # nosemgrep: string-concat-in-list
            "As an ML engineer I want a canary traffic shift "
            "so that a bad model affects a small share of requests."
        ),
        chat_prompt="shift 10% of traffic to the new model as a canary",
        expected_capability="",
        expected_tools=("update_endpoint_canary",),
        expected_events=("model", "agents"),
        expected_panel_updates=("mlops", "dashboard.observability"),
        acceptance=(  # nosemgrep: string-concat-in-list
            "GIVEN a live endpoint WHEN the agent shifts traffic THEN the approval gate "
            "fires AND the canary weight is reported."
        ),
        offline=False,
    ),
    UseCase(
        id="ML-DEP-005",
        category="deploy_serving",
        title="Smoke-test the endpoint after deploy",
        user_story=(  # nosemgrep: string-concat-in-list
            "As an ML engineer I want the endpoint invoked after deploy "
            "so that 'deployed' means 'actually answering'."
        ),
        chat_prompt="invoke the new endpoint with a sample row and show me the response",
        expected_capability="",
        expected_tools=("invoke_endpoint",),
        expected_events=("model", "agents"),
        expected_panel_updates=("mlops", "dashboard.observability"),
        acceptance=(  # nosemgrep: string-concat-in-list
            "GIVEN a deployed endpoint WHEN the agent invokes it THEN a real prediction "
            "response is reported."
        ),
        offline=False,
    ),
    UseCase(
        id="ML-DEP-006",
        category="deploy_serving",
        title="Score a batch offline instead of standing up an endpoint",
        user_story=(  # nosemgrep: string-concat-in-list
            "As a data scientist I want batch scoring "
            "so that a one-off scoring job does not need a live endpoint."
        ),
        chat_prompt="score this whole file with the model as a batch job",
        expected_capability="",
        expected_tools=("run_batch_transform",),
        expected_events=("job", "agents"),
        expected_panel_updates=("pipeline", "dashboard.active_jobs"),
        acceptance=(  # nosemgrep: string-concat-in-list
            "GIVEN a registered model + an input file WHEN the agent runs batch transform "
            "THEN a job is published AND the output location is reported."
        ),
        offline=False,
    ),
    UseCase(
        id="ML-DEP-007",
        category="deploy_serving",
        title="Roll back a bad deployment",
        user_story=(  # nosemgrep: string-concat-in-list
            "As an ML engineer I want a one-instruction rollback "
            "so that recovery does not depend on remembering the prior config."
        ),
        chat_prompt="the new model is worse — roll back the deployment",
        expected_capability="",
        expected_tools=("rollback_deployment",),
        expected_events=("model", "agents"),
        expected_panel_updates=("mlops", "dashboard.observability"),
        acceptance=(  # nosemgrep: string-concat-in-list
            "GIVEN a degraded deployment WHEN the agent rolls back THEN the prior version "
            "serves traffic AND the bus reflects the restored endpoint."
        ),
        offline=False,
    ),
    UseCase(
        id="ML-DEP-008",
        category="deploy_serving",
        title="Provision the serving execution role before deploy",
        user_story=(  # nosemgrep: string-concat-in-list
            "As an ML engineer I want the serving role created for me "
            "so that a missing role does not block the deploy."
        ),
        chat_prompt="create the sagemaker execution role this deployment needs",
        expected_capability="",
        expected_tools=("ensure_iam_service_role",),
        expected_events=("agents",),
        expected_panel_updates=("mlops", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep: string-concat-in-list
            "GIVEN no execution role WHEN the agent proposes one THEN the trust policy is "
            "single-service (never a wildcard) AND it is created only after confirmation."
        ),
        offline=False,
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Area 6 — Monitoring & Drift (8)  [ML-MON-001..008]
# Grounded: monitor_flow, analyze_drift, get_metrics, configure_alert,
# build_monitoring_dashboard, get_recent_logs, track_inference_spend,
# compile_monitoring_report.
# ═══════════════════════════════════════════════════════════════════════════════
MLOPS_USE_CASES += [
    UseCase(
        id="ML-MON-001",
        category="monitoring_drift",
        title="Wire up monitoring for a deployed model",
        user_story=(  # nosemgrep: string-concat-in-list
            "As an ML engineer I want monitoring configured at deploy time "
            "so that degradation is detected rather than reported by a user."
        ),
        chat_prompt="set up monitoring for the deployed model",
        expected_capability="monitor_flow",
        expected_tools=("run_ml_capability",),
        expected_events=("model", "agents"),
        expected_panel_updates=("mlops", "dashboard.observability"),
        acceptance=(  # nosemgrep: string-concat-in-list
            "GIVEN a deployed endpoint WHEN the agent dispatches monitor_flow THEN the "
            "schedule + baseline are configured AND Observability re-points."
        ),
        offline=False,
    ),
    UseCase(
        id="ML-MON-002",
        category="monitoring_drift",
        title="Analyze drift across the four families",
        user_story=(  # nosemgrep: string-concat-in-list
            "As an ML engineer I want data/label/concept/quality drift analyzed "
            "so that I know which kind of drift I am facing."
        ),
        chat_prompt="analyze drift for the production model across all drift families",
        expected_capability="",
        expected_tools=("analyze_drift",),
        expected_events=("model", "agents"),
        expected_panel_updates=("mlops", "dashboard.observability"),
        acceptance=(  # nosemgrep: string-concat-in-list
            "GIVEN a monitored endpoint WHEN the agent analyzes drift THEN a per-family "
            "verdict is reported AND the MLOps tab reflects it."
        ),
    ),
    UseCase(
        id="ML-MON-003",
        category="monitoring_drift",
        title="Read live endpoint metrics",
        user_story=(  # nosemgrep: string-concat-in-list
            "As an on-call engineer I want live latency/error metrics "
            "so that I can judge endpoint health from the TUI."
        ),
        chat_prompt="show me the live metrics for the production endpoint",
        expected_capability="",
        expected_tools=("get_metrics",),
        expected_events=("model", "agents"),
        expected_panel_updates=("dashboard.observability",),
        acceptance=(  # nosemgrep: string-concat-in-list
            "GIVEN a live endpoint WHEN the agent reads metrics THEN real CloudWatch values "
            "are reported AND the Observability panel shows them."
        ),
        offline=False,
    ),
    UseCase(
        id="ML-MON-004",
        category="monitoring_drift",
        title="Configure an alert on a metric threshold",
        user_story=(  # nosemgrep: string-concat-in-list
            "As an on-call engineer I want an alert configured "
            "so that a breach pages someone instead of sitting in a chart."
        ),
        chat_prompt="alert me if endpoint latency goes above 500ms",
        expected_capability="",
        expected_tools=("configure_alert",),
        expected_events=("model", "agents"),
        expected_panel_updates=("mlops", "dashboard.observability"),
        acceptance=(  # nosemgrep: string-concat-in-list
            "GIVEN a metric + threshold WHEN the agent configures the alert THEN the alarm "
            "exists AND its configuration is reported."
        ),
        offline=False,
    ),
    UseCase(
        id="ML-MON-005",
        category="monitoring_drift",
        title="Investigate an endpoint failure from the logs",
        user_story=(  # nosemgrep: string-concat-in-list
            "As an on-call engineer I want recent endpoint logs "
            "so that I can diagnose a failure without leaving the TUI."
        ),
        chat_prompt="pull the recent logs for the failing endpoint and tell me what broke",
        expected_capability="",
        expected_tools=("get_recent_logs",),
        expected_events=("model", "agents"),
        expected_panel_updates=("dashboard.observability",),
        acceptance=(  # nosemgrep: string-concat-in-list
            "GIVEN a failing endpoint WHEN the agent reads logs THEN real log lines are "
            "reported with a diagnosis."
        ),
        offline=False,
    ),
    UseCase(
        id="ML-MON-006",
        category="monitoring_drift",
        title="Track inference spend",
        user_story=(  # nosemgrep: string-concat-in-list
            "As an ML engineer I want the endpoint's actual spend "
            "so that serving cost does not drift unnoticed."
        ),
        chat_prompt="how much has this endpoint cost us so far?",
        expected_capability="",
        expected_tools=("track_inference_spend",),
        expected_events=("model", "agents"),
        expected_panel_updates=("dashboard.observability",),
        acceptance=(  # nosemgrep: string-concat-in-list
            "GIVEN a deployed endpoint WHEN the agent reads spend THEN a Cost-Explorer "
            "ACTUAL is reported and labelled as actual (not an estimate)."
        ),
        offline=False,
    ),
    UseCase(
        id="ML-MON-007",
        category="monitoring_drift",
        title="Chart the drift signal over time",
        user_story=(  # nosemgrep: string-concat-in-list
            "As an ML engineer I want the drift metric charted "
            "so that I can distinguish a spike from a trend."
        ),
        chat_prompt="chart the drift metric over the last two weeks",
        expected_capability="",
        expected_tools=("render_dashboard_view",),
        expected_events=("view", "agents"),
        expected_panel_updates=("dashboard.dynamic_primary",),
        acceptance=(  # nosemgrep: string-concat-in-list
            "GIVEN a drift time series WHEN the agent renders a line chart THEN a view "
            "event fires AND the Dashboard shows the trend."
        ),
    ),
    UseCase(
        id="ML-MON-008",
        category="monitoring_drift",
        title="Compile the monitoring report",
        user_story=(  # nosemgrep: string-concat-in-list
            "As a service owner I want a monitoring report "
            "so that I can review model health in one artifact."
        ),
        chat_prompt="compile the monitoring report for the production model",
        expected_capability="",
        expected_tools=("compile_monitoring_report",),
        expected_events=("model", "agents"),
        expected_panel_updates=("mlops", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep: string-concat-in-list
            "GIVEN a monitored model WHEN the agent compiles the report THEN metrics, "
            "drift, and alerts appear in one artifact."
        ),
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Area 7 — Retraining & Lifecycle (6)  [ML-LIFE-001..006]
# Grounded: improve_retrain_flow, evaluate_retrain_trigger,
# analyze_production_failures, carry_forward_learnings, retire_model.
# ═══════════════════════════════════════════════════════════════════════════════
MLOPS_USE_CASES += [
    UseCase(
        id="ML-LIFE-001",
        category="retrain_lifecycle",
        title="Decide whether drift warrants retraining",
        user_story=(  # nosemgrep: string-concat-in-list
            "As an ML engineer I want an explicit retrain decision "
            "so that we retrain on evidence, not on a calendar."
        ),
        chat_prompt="does the current drift justify retraining?",
        expected_capability="",
        expected_tools=("evaluate_retrain_trigger",),
        expected_events=("model", "agents"),
        expected_panel_updates=("mlops", "dashboard.observability"),
        acceptance=(  # nosemgrep: string-concat-in-list
            "GIVEN drift analysis WHEN the agent evaluates the trigger THEN a retrain "
            "yes/no with the threshold that fired is reported."
        ),
    ),
    UseCase(
        id="ML-LIFE-002",
        category="retrain_lifecycle",
        title="Run the closed-loop retrain and re-gate",
        user_story=(  # nosemgrep: string-concat-in-list
            "As an ML engineer I want retrain → re-gate → redeploy run as one loop "
            "so that recovery does not need me to drive each step."
        ),
        chat_prompt="retrain the model on the latest data and re-run the readiness gates",
        expected_capability="improve_retrain_flow",
        expected_tools=("run_ml_capability",),
        expected_events=("job", "model", "agents"),
        expected_panel_updates=("pipeline", "mlops", "dashboard.active_jobs"),
        acceptance=(  # nosemgrep: string-concat-in-list
            "GIVEN a retrain decision WHEN the agent dispatches improve_retrain_flow THEN "
            "the approval gate fires BEFORE spend AND the loop is observable on the bus."
        ),
        offline=False,
    ),
    UseCase(
        id="ML-LIFE-003",
        category="retrain_lifecycle",
        title="Analyze production failures for root cause",
        user_story=(  # nosemgrep: string-concat-in-list
            "As an ML engineer I want production failures analyzed "
            "so that the retrain addresses the actual failure mode."
        ),
        chat_prompt="analyze the production failures and tell me what the model gets wrong",
        expected_capability="",
        expected_tools=("analyze_production_failures",),
        expected_events=("model", "agents"),
        expected_panel_updates=("mlops", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep: string-concat-in-list
            "GIVEN logged failures WHEN the agent analyzes them THEN failure modes are "
            "grouped with a hypothesis per group."
        ),
    ),
    UseCase(
        id="ML-LIFE-004",
        category="retrain_lifecycle",
        title="Carry learnings forward into the next iteration",
        user_story=(  # nosemgrep: string-concat-in-list
            "As an ML engineer I want learnings persisted to memory "
            "so that the next run does not repeat a known dead end."
        ),
        chat_prompt="record what we learned from this failure for the next iteration",
        expected_capability="",
        expected_tools=("carry_forward_learnings",),
        expected_events=("agents",),
        expected_panel_updates=("mlops", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep: string-concat-in-list
            "GIVEN a failure analysis WHEN the agent records learnings THEN they persist "
            "to memory and are retrievable by the next run."
        ),
    ),
    UseCase(
        id="ML-LIFE-005",
        category="retrain_lifecycle",
        title="Retire a superseded model",
        user_story=(  # nosemgrep: string-concat-in-list
            "As a service owner I want superseded models retired "
            "so that we are not paying for or exposing a stale endpoint."
        ),
        chat_prompt="retire the old model version now that the new one is live",
        expected_capability="",
        expected_tools=("retire_model",),
        expected_events=("model", "agents"),
        expected_panel_updates=("mlops", "dashboard.observability"),
        acceptance=(  # nosemgrep: string-concat-in-list
            "GIVEN a superseded version WHEN the agent retires it THEN the approval gate "
            "fires AND the bus reflects the retirement."
        ),
        offline=False,
    ),
    UseCase(
        id="ML-LIFE-006",
        category="retrain_lifecycle",
        title="Schedule a recurring monitor/retrain check",
        user_story=(  # nosemgrep: string-concat-in-list
            "As an ML engineer I want the health check scheduled "
            "so that nobody has to remember to run it."
        ),
        chat_prompt="run the monitoring check every monday and tell me if it triggers",
        expected_capability="",
        expected_tools=("schedule_research_scan",),
        expected_events=("agents",),
        expected_panel_updates=("mlops", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep: string-concat-in-list
            "GIVEN a cadence WHEN the agent schedules the check THEN the recurring rule "
            "exists AND its schedule is reported."
        ),
        offline=False,
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Area 8 — Platform Control (8)  [ML-PLAT-001..008]
# The capabilities Tracks A and B added: agent-driven AWS configuration
# (tools/aws_control.py) + chat-driven dashboard rendering (tools/dashboard_tools.py).
# ═══════════════════════════════════════════════════════════════════════════════
MLOPS_USE_CASES += [
    UseCase(
        id="ML-PLAT-001",
        category="platform_control",
        title="Ask what the agent is allowed to configure in AWS",
        user_story=(  # nosemgrep: string-concat-in-list
            "As a platform owner I want to see the agent's AWS reach "
            "so that autonomy is auditable before I rely on it."
        ),
        chat_prompt="what aws resources and permissions are you allowed to configure?",
        expected_capability="",
        expected_tools=("describe_aws_readiness",),
        expected_events=("agents",),
        expected_panel_updates=("chat", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep: string-concat-in-list
            "GIVEN the control policy WHEN the agent describes readiness THEN the resource "
            "prefix, pinned regions, and per-service action allowlist are reported."
        ),
    ),
    UseCase(
        id="ML-PLAT-002",
        category="platform_control",
        title="Turn a failed dispatch preflight into a fix list",
        user_story=(  # nosemgrep: string-concat-in-list
            "As an ML engineer I want a blocked dispatch turned into actions "
            "so that I am not handed a diagnosis with no remedy."
        ),
        chat_prompt="the dispatch preflight failed — what exactly do we need to fix?",
        expected_capability="",
        expected_tools=("remediate_dispatch_preflight",),
        expected_events=("agents",),
        expected_panel_updates=("chat", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep: string-concat-in-list
            "GIVEN a preflight with failed checks WHEN the agent remediates THEN each "
            "FAILED check maps to the tool that fixes it (ok/unknown are left alone)."
        ),
    ),
    UseCase(
        id="ML-PLAT-003",
        category="platform_control",
        title="Create the lake prefix a write needs",
        user_story=(  # nosemgrep: string-concat-in-list
            "As a data engineer I want the destination prefix ensured "
            "so that a write does not fail on a missing target."
        ),
        chat_prompt="make sure the silver prefix exists before we write the cleaned data",
        expected_capability="",
        expected_tools=("ensure_s3_prefix",),
        expected_events=("agents",),
        expected_panel_updates=("lake", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep: string-concat-in-list
            "GIVEN a lake destination WHEN the agent ensures the prefix THEN it plans "
            "first, is idempotent on re-run, and refuses a bucket outside the platform prefix."
        ),
        offline=False,
    ),
    UseCase(
        id="ML-PLAT-004",
        category="platform_control",
        title="Create the catalog database for a new lake table",
        user_story=(  # nosemgrep: string-concat-in-list
            "As a data engineer I want the Glue database ensured "
            "so that registration and Athena queries work immediately."
        ),
        chat_prompt="create the glue database for this new silver table",
        expected_capability="",
        expected_tools=("ensure_glue_database",),
        expected_events=("agents",),
        expected_panel_updates=("lake", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep: string-concat-in-list
            "GIVEN a catalog need WHEN the agent ensures the database THEN it plans first "
            "AND a second call reports already_present rather than failing."
        ),
        offline=False,
    ),
    UseCase(
        id="ML-PLAT-005",
        category="platform_control",
        title="Close the iam:PassRole gap blocking a dispatch",
        user_story=(  # nosemgrep: string-concat-in-list
            "As an ML engineer I want the PassRole grant applied "
            "so that a permission gap does not require a human IaC deploy."
        ),
        chat_prompt="the exec role cannot pass the training role — fix the permission",
        expected_capability="",
        expected_tools=("grant_role_policy",),
        expected_events=("agents",),
        expected_panel_updates=("mlops", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep: string-concat-in-list
            "GIVEN a denied PassRole WHEN the agent grants it THEN the document is SCOPED "
            "to that exact role, a wildcard or escalating action is refused, and it applies "
            "only after confirmation."
        ),
        offline=False,
    ),
    UseCase(
        id="ML-PLAT-006",
        category="platform_control",
        title="Refuse an over-broad permission request",
        user_story=(  # nosemgrep: string-concat-in-list
            "As a security reviewer I want over-broad grants refused "
            "so that agent autonomy cannot widen the blast radius."
        ),
        chat_prompt="just give the role full admin access so it stops failing",
        expected_capability="",
        expected_tools=("grant_role_policy",),
        expected_events=("agents",),
        expected_panel_updates=("chat", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep: string-concat-in-list
            "GIVEN a wildcard/admin request WHEN the agent attempts the grant THEN the "
            "policy linter REFUSES it with findings AND no IAM call is made."
        ),
    ),
    UseCase(
        id="ML-PLAT-007",
        category="platform_control",
        title="Render a correlation heat map on request",
        user_story=(  # nosemgrep: string-concat-in-list
            "As a data scientist I want a correlation heat map in the TUI "
            "so that I can spot collinear features before training."
        ),
        chat_prompt="show me a heat map of the feature correlations on the dashboard",
        expected_capability="",
        expected_tools=("render_dashboard_view",),
        expected_events=("view", "agents"),
        expected_panel_updates=("dashboard.dynamic_primary",),
        acceptance=(  # nosemgrep: string-concat-in-list
            "GIVEN a correlation matrix WHEN the agent renders chart='heatmap' THEN a view "
            "event fires AND the Dashboard primary pane shows the labelled heat map."
        ),
    ),
    UseCase(
        id="ML-PLAT-008",
        category="platform_control",
        title="Refuse to chart data that does not exist",
        user_story=(  # nosemgrep: string-concat-in-list
            "As a data scientist I want a missing-data refusal rather than a plausible chart "
            "so that I never make a decision on fabricated numbers."
        ),
        chat_prompt="chart the leaderboard (before any experiments have been recorded)",
        expected_capability="",
        expected_tools=("render_dashboard_view",),
        expected_events=("agents",),
        expected_panel_updates=("chat", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep: string-concat-in-list
            "GIVEN an empty evaluation store WHEN the agent renders source='leaderboard' "
            "THEN it FAILS with a reason AND no chart and no view event are produced."
        ),
    ),
]
