"""Top-100 data-quality & data-engineering use-case registry (machine-readable).

This module is the **single source of truth** for the 100 DQ/DE use cases that the
Phase-8 end-to-end harness drives through the TUI, following the chat-driven
invariant proven in Phase 2:

    user types in Chat  ->  ChatEngine routes to a tool / capability  ->
    ContextBus publishes typed event(s)  ->  every subscribed tab re-renders  ->
    the harness asserts (tools invoked, bus fields, panel text).

Every entry is grounded in a REAL ``@tool`` / flow discovered by inventorying
``src/amlas/tools/`` + ``src/amlas/flows/`` (see the master plan §A.6/§B.3). Where a
capability does not yet exist, the entry is marked ``supported_today=False`` with a
``gap`` note referencing the master-plan gap list (PII detection T-1, partitioning
T-2, SDV/SMOTE generators T-3). Phase 7 implements those gaps; Phase 8/9 test all 100.

The two companion Markdown documents share IDENTICAL ids/titles/prompts:
  * ``docs/use-cases/data-quality-engineering-100-stories.md``
  * ``docs/use-cases/data-quality-engineering-100-tdds.md``

Schema (master-plan §C.3) — the :class:`UseCase` dataclass:
  * ``id``                     stable slug, e.g. ``DQ-PROF-001``
  * ``category``               one of :data:`CATEGORIES`
  * ``title``                  short human title
  * ``user_story``             "As a … I want … so that …"
  * ``chat_prompt``            the natural-language message the harness sends to Chat
  * ``expected_capability``    ``run_ml_capability`` flow id (or "" for a direct tool)
  * ``expected_tools``         real tool name(s) the agent should invoke
  * ``expected_events``        ContextBus events that must fire (subset of CANONICAL_EVENTS)
  * ``expected_panel_updates`` TUI tab(s)/panel(s) that must reflect the result
  * ``acceptance``             given/when/then assertion the harness checks
  * ``supported_today``        False if a gap blocks end-to-end support today
  * ``gap``                    gap note (only when ``supported_today`` is False)
  * ``offline``                True if deterministic offline; False if gated (data-extra/cloud)
"""

# SECURITY NOTE (Security review, S3 bucket squatting): every ``s3://amlas/...``
# URI in the chat_prompt examples below is a LOGICAL lake alias, NOT a physical
# bucket. The lake chokepoint (amlas.tools.lake.resolve_lake_bucket) rewrites the
# alias to the adopter-configured physical bucket (env AMLAS_LAKE_BUCKET / SSM
# /amlas/lake-bucket), and live S3 calls now FAIL LOUD if the alias is unresolved
# (_guard_unresolved_alias) — so these examples can never hit a squattable
# literal 'amlas' bucket in a real deployment.

from __future__ import annotations

from dataclasses import dataclass

# ─── Canonical ContextBus event set (post-Phase-2 extension) ─────────────────
# Mirrors the event literal at ``src/amlas/tui/context/bus.py`` + the ``processes``
# event added by the Phase-2 spec (§2.f). ``expected_events`` values MUST be a
# subset of this tuple.
CANONICAL_EVENTS: tuple[str, ...] = (
    "spec",
    "dataset",
    "job",
    "run",
    "model",
    "problem",
    "steering",
    "agents",
    "selection",
    "processes",
    # Plan §B2 — the agent's dashboard RENDER REQUEST (``ContextBus.set_view``). Additive:
    # no existing DQ entry declares it, so every prior expectation is unchanged.
    "view",
)

# ─── The 12-category taxonomy (must total exactly 100) ───────────────────────
# category slug -> (human label, expected count). Sum of counts == 100.
CATEGORY_COUNTS: dict[str, int] = {
    "profiling_eda": 8,
    "validation_constraints": 12,
    "missingness_imputation": 12,
    "dedup_entity_resolution": 6,
    "outliers_distribution": 6,
    "schema_drift_lineage": 8,
    "leakage_pii_governance": 8,
    "ingestion_lakehouse": 10,
    "transformation_etl": 8,
    "feature_engineering_store": 8,
    "synthetic_augmentation": 6,
    "dq_pipeline_orchestration": 8,
}

CATEGORIES: tuple[str, ...] = tuple(CATEGORY_COUNTS.keys())

# Human-readable category labels (for docs / panels).
CATEGORY_LABELS: dict[str, str] = {
    "profiling_eda": "Profiling & EDA",
    "validation_constraints": "Validation & Constraints",
    "missingness_imputation": "Missingness Diagnosis & Imputation",
    "dedup_entity_resolution": "Deduplication & Entity Resolution",
    "outliers_distribution": "Outliers & Distribution",
    "schema_drift_lineage": "Schema & Era Drift / Lineage",
    "leakage_pii_governance": "Leakage & PII / Governance",
    "ingestion_lakehouse": "Ingestion & Lakehouse",
    "transformation_etl": "Transformation & ETL Jobs",
    "feature_engineering_store": "Feature Engineering & Store",
    "synthetic_augmentation": "Synthetic Data & Augmentation",
    "dq_pipeline_orchestration": "DQ Pipeline Orchestration & Monitoring",
}


@dataclass(frozen=True)
class UseCase:
    """One declarative, harness-drivable DQ/DE use case (master-plan §C.3)."""

    id: str
    category: str
    title: str
    user_story: str
    chat_prompt: str
    expected_capability: str
    expected_tools: tuple[str, ...]
    expected_events: tuple[str, ...]
    expected_panel_updates: tuple[str, ...]
    acceptance: str
    supported_today: bool = True
    gap: str = ""
    offline: bool = True


# The 100-entry registry. Populated category-by-category below (each block appends
# to this list). Assembled in taxonomy order so ``USE_CASES[:8]`` are Profiling, etc.
USE_CASES: list[UseCase] = []


# ─── Query helpers ───────────────────────────────────────────────────────────
def get_use_case(uc_id: str) -> UseCase | None:
    """Return the :class:`UseCase` with ``id == uc_id`` (or ``None``)."""
    for uc in USE_CASES:
        if uc.id == uc_id:
            return uc
    return None


def use_cases_for_category(category: str) -> list[UseCase]:
    """Return all use cases in ``category`` (registry order)."""
    return [uc for uc in USE_CASES if uc.category == category]


def category_counts() -> dict[str, int]:
    """Return the observed per-category count of :data:`USE_CASES`."""
    counts: dict[str, int] = {c: 0 for c in CATEGORIES}
    for uc in USE_CASES:
        counts[uc.category] = counts.get(uc.category, 0) + 1
    return counts


# ═══════════════════════════════════════════════════════════════════════════════
# Category 1 — Profiling & EDA (8)  [DQ-PROF-001..008]
# Grounded: describe_dataset, generate_eda_report, run_profile, load_dataset,
# score_data_quality, score_data_quality_6dim (all READ_ONLY safe tools).
# ═══════════════════════════════════════════════════════════════════════════════
USE_CASES += [
    UseCase(
        id="DQ-PROF-001",
        category="profiling_eda",
        title="Infer schema and column dtypes",
        user_story=(  # nosemgrep
            "As a data engineer I want the agent to infer the schema and per-column "
            "dtypes of a raw dataset so that I can confirm the table shape before modeling."
        ),
        chat_prompt="describe the dataset at data/bogie.csv and show me its schema and column types",
        expected_capability="",
        expected_tools=("describe_dataset",),
        expected_events=("dataset", "agents"),
        expected_panel_updates=("data", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep
            "GIVEN a CSV at the referenced path WHEN the agent runs describe_dataset "
            "THEN AgentTurn.executions includes 'describe_dataset' AND context_bus.state().dataset "
            "reflects the inferred rows/cols AND the Data tab repaints the active dataset."
        ),
    ),
    UseCase(
        id="DQ-PROF-002",
        category="profiling_eda",
        title="Cardinality and uniqueness profile",
        user_story=(  # nosemgrep
            "As a data engineer I want per-column cardinality and uniqueness ratios so that "
            "I can spot identifier-like and constant columns early."
        ),
        chat_prompt="profile the cardinality and uniqueness of every column in data/bogie.csv",
        expected_capability="",
        expected_tools=("describe_dataset",),
        expected_events=("dataset", "agents"),
        expected_panel_updates=("data", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep
            "GIVEN a loaded dataset WHEN the agent profiles it THEN 'describe_dataset' appears "
            "in AgentTurn.executions AND the result payload carries per-column cardinality AND "
            "the Data tab shows the active dataset."
        ),
    ),
    UseCase(
        id="DQ-PROF-003",
        category="profiling_eda",
        title="Summary statistics (numeric + categorical)",
        user_story=(  # nosemgrep
            "As a data scientist I want summary statistics (mean/std/min/max, top categories) "
            "so that I understand value ranges before feature engineering."
        ),
        chat_prompt="give me summary statistics for the numeric and categorical columns in data/bogie.csv",
        expected_capability="",
        expected_tools=("describe_dataset",),
        expected_events=("dataset", "agents"),
        expected_panel_updates=("data", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep
            "GIVEN a dataset WHEN summary stats are requested THEN 'describe_dataset' is invoked "
            "AND the response includes numeric describe() + categorical top-values AND the Data "
            "tab reflects the active dataset."
        ),
    ),
    UseCase(
        id="DQ-PROF-004",
        category="profiling_eda",
        title="Correlation analysis across features",
        user_story=(  # nosemgrep
            "As a data scientist I want a feature-correlation report so that I can find "
            "redundant or collinear columns before training."
        ),
        chat_prompt="generate an EDA report with the feature correlation matrix for data/bogie.csv",
        expected_capability="",
        expected_tools=("generate_eda_report",),
        expected_events=("dataset", "agents"),
        expected_panel_updates=("data", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep
            "GIVEN a dataset WHEN an EDA report is requested THEN 'generate_eda_report' is invoked "
            "AND the report contains a correlation section AND the agent-activity panel shows "
            "last_tool == 'generate_eda_report'."
        ),
    ),
    UseCase(
        id="DQ-PROF-005",
        category="profiling_eda",
        title="Target profile and class balance",
        user_story=(  # nosemgrep
            "As a data scientist I want a target-column profile with class balance so that "
            "I can plan for imbalance before choosing a metric."
        ),
        chat_prompt="profile the target column machine_failure in data/bogie.csv and report its class balance",
        expected_capability="",
        expected_tools=("describe_dataset",),
        expected_events=("dataset", "problem", "agents"),
        expected_panel_updates=("data", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep
            "GIVEN a dataset with a target WHEN the target profile is requested THEN "
            "'describe_dataset' is invoked with target_column AND the response reports class "
            "counts/balance AND the Data tab reflects the active dataset."
        ),
    ),
    UseCase(
        id="DQ-PROF-006",
        category="profiling_eda",
        title="Full auto EDA report",
        user_story=(  # nosemgrep
            "As an analyst I want a one-shot EDA report so that I can review distributions, "
            "missingness, and correlations in a single artifact."
        ),
        chat_prompt="run a full EDA report on data/bogie.csv and summarize the key findings",
        expected_capability="",
        expected_tools=("generate_eda_report",),
        expected_events=("dataset", "agents"),
        expected_panel_updates=("data", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep
            "GIVEN a dataset WHEN a full EDA is requested THEN 'generate_eda_report' is invoked "
            "AND the report includes distributions + missingness + correlation sections AND the "
            "agent-activity panel records the tool."
        ),
    ),
    UseCase(
        id="DQ-PROF-007",
        category="profiling_eda",
        title="Scale-aware profiling for large tables",
        user_story=(  # nosemgrep
            "As a data engineer I want scale-aware profiling of a large lakehouse table so that "
            "I can profile datasets too big for a single-node pandas pass."
        ),
        chat_prompt="run a profile on the dataset uri s3://amlas/silver/bogie and pick an appropriate scale",
        expected_capability="",
        expected_tools=("run_profile",),
        expected_events=("dataset", "agents"),
        expected_panel_updates=("data", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep
            "GIVEN a dataset uri WHEN profiling at scale is requested THEN 'run_profile' is invoked "
            "AND the result carries a profile summary AND the Data tab reflects the profiled dataset."
        ),
    ),
    UseCase(
        id="DQ-PROF-008",
        category="profiling_eda",
        title="Six-dimension data-quality score",
        user_story=(  # nosemgrep
            "As a data steward I want a six-dimension quality score (completeness, validity, "
            "uniqueness, consistency, accuracy, timeliness) so that I get a single quality headline."
        ),
        chat_prompt="score the data quality of data/bogie.csv across all six dimensions with target machine_failure",
        expected_capability="",
        expected_tools=("score_data_quality_6dim",),
        expected_events=("dataset", "agents"),
        expected_panel_updates=("data", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep
            "GIVEN a dataset WHEN a 6-dimension score is requested THEN 'score_data_quality_6dim' "
            "is invoked AND context_bus.state().dataset.dq_score is set (>=0) AND the Data tab's "
            "Quality Score repaints."
        ),
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Category 2 — Validation & Constraints (12)  [DQ-VAL-001..012]
# Grounded: scan_dq_patterns (VAL-*/CON-*), check_expectations (pandera gate),
# recommend_dq_rules, evaluate_dq (Glue-DQ/DQDL), validate_spec, data_quality_flow.
# ═══════════════════════════════════════════════════════════════════════════════
USE_CASES += [
    UseCase(
        id="DQ-VAL-001",
        category="validation_constraints",
        title="Detect mixed-type columns (VAL-TYPE)",
        user_story=(  # nosemgrep
            "As a data engineer I want to detect columns that mix numeric and text values so "
            "that I can coerce or split them before training."
        ),
        chat_prompt="scan data/bogie.csv for mixed-type columns and tell me which ones need coercion",
        expected_capability="",
        expected_tools=("scan_dq_patterns",),
        expected_events=("dataset", "agents"),
        expected_panel_updates=("data", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep
            "GIVEN a dataset with an object column mixing types WHEN scanned THEN "
            "'scan_dq_patterns' is invoked AND the findings include pattern_id 'VAL-TYPE' AND the "
            "Data tab reflects the dataset."
        ),
    ),
    UseCase(
        id="DQ-VAL-002",
        category="validation_constraints",
        title="Flag out-of-range numeric values (VAL-RANGE)",
        user_story=(  # nosemgrep
            "As a data steward I want out-of-range numeric values flagged so that physically "
            "impossible readings (e.g. negative age) are quarantined."
        ),
        chat_prompt="check data/bogie.csv for out-of-range or physically impossible numeric values",
        expected_capability="",
        expected_tools=("scan_dq_patterns",),
        expected_events=("dataset", "agents"),
        expected_panel_updates=("data", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep
            "GIVEN numeric columns with implausible values WHEN scanned THEN 'scan_dq_patterns' "
            "reports pattern_id 'VAL-RANGE' AND the Data tab reflects the dataset."
        ),
    ),
    UseCase(
        id="DQ-VAL-003",
        category="validation_constraints",
        title="Detect infinite values (VAL-INF)",
        user_story=(  # nosemgrep
            "As a data engineer I want infinite values detected so that overflow/divide-by-zero "
            "artifacts are replaced before modeling."
        ),
        chat_prompt="scan data/bogie.csv for infinite values and report which columns contain them",
        expected_capability="",
        expected_tools=("scan_dq_patterns",),
        expected_events=("dataset", "agents"),
        expected_panel_updates=("data", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep
            "GIVEN a column with +/-inf WHEN scanned THEN 'scan_dq_patterns' reports pattern_id "
            "'VAL-INF' AND the agent-activity panel records the tool."
        ),
    ),
    UseCase(
        id="DQ-VAL-004",
        category="validation_constraints",
        title="Enforce format / enum violations (VAL-FORMAT)",
        user_story=(  # nosemgrep
            "As a data steward I want date/regex/enum format violations detected so that malformed "
            "categorical values are normalized."
        ),
        chat_prompt="check data/bogie.csv for date and enum format violations in the string columns",
        expected_capability="",
        expected_tools=("scan_dq_patterns",),
        expected_events=("dataset", "agents"),
        expected_panel_updates=("data", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep
            "GIVEN string columns violating an expected format WHEN scanned THEN 'scan_dq_patterns' "
            "reports pattern_id 'VAL-FORMAT' AND the Data tab reflects the dataset."
        ),
    ),
    UseCase(
        id="DQ-VAL-005",
        category="validation_constraints",
        title="Detect unit drift in numeric columns (VAL-UNIT)",
        user_story=(  # nosemgrep
            "As a data engineer I want unit-drift detection (imperial vs metric mixing) so that "
            "multi-modal scale columns are normalized to one unit."
        ),
        chat_prompt="scan data/bogie.csv for unit drift where a column mixes imperial and metric scales",
        expected_capability="",
        expected_tools=("scan_dq_patterns",),
        expected_events=("dataset", "agents"),
        expected_panel_updates=("data", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep
            "GIVEN a numeric column with multi-modal scale WHEN scanned THEN 'scan_dq_patterns' "
            "reports pattern_id 'VAL-UNIT' AND the agent-activity panel records the tool."
        ),
    ),
    UseCase(
        id="DQ-VAL-006",
        category="validation_constraints",
        title="Validate cross-field business rules (CON-REF)",
        user_story=(  # nosemgrep
            "As a domain expert I want deterministic cross-column rules validated (e.g. start<=end) "
            "so that referential relationships hold."
        ),
        chat_prompt="validate the cross-field rule that torque must be <= max_torque across data/bogie.csv",
        expected_capability="",
        expected_tools=("scan_dq_patterns",),
        expected_events=("dataset", "agents"),
        expected_panel_updates=("data", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep
            "GIVEN a cross-column relationship WHEN validated THEN 'scan_dq_patterns' reports "
            "pattern_id 'CON-REF' for the violated rows AND the Data tab reflects the dataset."
        ),
    ),
    UseCase(
        id="DQ-VAL-007",
        category="validation_constraints",
        title="Detect temporal inconsistency (CON-TEMP)",
        user_story=(  # nosemgrep
            "As a data engineer I want future-dated or non-monotonic timestamps flagged so that "
            "time-series integrity is preserved."
        ),
        chat_prompt="check data/bogie.csv for future-dated or non-monotonic timestamps in the time column",
        expected_capability="",
        expected_tools=("scan_dq_patterns",),
        expected_events=("dataset", "agents"),
        expected_panel_updates=("data", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep
            "GIVEN a time column with future/non-monotonic values WHEN scanned THEN "
            "'scan_dq_patterns' reports pattern_id 'CON-TEMP' AND the agent-activity panel records it."
        ),
    ),
    UseCase(
        id="DQ-VAL-008",
        category="validation_constraints",
        title="Run pandera/expectations suite (check_expectations)",
        user_story=(  # nosemgrep
            "As a data steward I want a pandera expectations gate run so that a dataset passes or "
            "fails a declared contract before it enters the pipeline."
        ),
        chat_prompt="run the data-quality expectations gate on data/bogie.csv and tell me if it passes",
        expected_capability="",
        expected_tools=("check_expectations",),
        expected_events=("dataset", "agents"),
        expected_panel_updates=("data", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep
            "GIVEN a dataset and a suite WHEN the gate runs THEN 'check_expectations' is invoked "
            "AND the result reports pass/fail per expectation AND the Data tab reflects the dataset."
        ),
    ),
    UseCase(
        id="DQ-VAL-009",
        category="validation_constraints",
        title="Recommend a validation ruleset (recommend_dq_rules)",
        user_story=(  # nosemgrep
            "As a data engineer I want the agent to recommend a DQ ruleset from a dataset so that "
            "I don't have to hand-write every constraint."
        ),
        chat_prompt="recommend a set of data-quality rules for the dataset uri s3://amlas/silver/bogie",
        expected_capability="",
        expected_tools=("recommend_dq_rules",),
        expected_events=("dataset", "agents"),
        expected_panel_updates=("data", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep
            "GIVEN a dataset uri WHEN rules are requested THEN 'recommend_dq_rules' is invoked AND "
            "the response contains a candidate ruleset AND the agent-activity panel records the tool."
        ),
    ),
    UseCase(
        id="DQ-VAL-010",
        category="validation_constraints",
        title="Evaluate a dataset against a ruleset (evaluate_dq / Glue DQ)",
        user_story=(  # nosemgrep
            "As a data engineer I want to evaluate a dataset against a DQDL/Glue-DQ ruleset so that "
            "I get a pass rate and per-rule breakdown."
        ),
        chat_prompt="evaluate the dataset uri s3://amlas/silver/bogie against the recommended DQ ruleset",
        expected_capability="",
        expected_tools=("evaluate_dq",),
        expected_events=("dataset", "agents"),
        expected_panel_updates=("data", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep
            "GIVEN a dataset uri and ruleset WHEN evaluated THEN 'evaluate_dq' is invoked AND the "
            "result reports an overall pass rate + per-rule outcomes AND the Data tab reflects it."
        ),
    ),
    UseCase(
        id="DQ-VAL-011",
        category="validation_constraints",
        title="Validate an ML spec contract (validate_spec)",
        user_story=(  # nosemgrep
            "As an ML engineer I want a spec YAML validated so that a malformed problem contract is "
            "caught before a run is launched."
        ),
        chat_prompt="validate the spec I have open and tell me if the contract is well-formed",
        expected_capability="",
        expected_tools=("validate_spec",),
        expected_events=("spec", "agents"),
        expected_panel_updates=("specs", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep
            "GIVEN a spec YAML WHEN validated THEN 'validate_spec' is invoked AND the result reports "
            "valid/invalid with errors AND the Specs tab reflects the active spec."
        ),
    ),
    UseCase(
        id="DQ-VAL-012",
        category="validation_constraints",
        title="Full validation gate via data_quality_flow",
        user_story=(  # nosemgrep
            "As a data steward I want a full validation gate (scan + expectations + score) run as "
            "one capability so that the dataset is certified before modeling."
        ),
        chat_prompt="run the data quality flow on data/bogie.csv and give me a go/no-go gate decision",
        expected_capability="data_quality_flow",
        expected_tools=("run_ml_capability",),
        expected_events=("job", "dataset", "agents"),
        expected_panel_updates=("data", "pipeline", "dashboard.active_jobs"),
        acceptance=(  # nosemgrep
            "GIVEN a dataset WHEN data_quality_flow is dispatched THEN run_ml_capability routes to "
            "'data_quality_flow' AND a job event fires AND the Pipeline/Active-Jobs panels show the "
            "gate job with a go/no-go result."
        ),
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Category 3 — Missingness Diagnosis & Imputation (12)  [DQ-MISS-001..012]
# Grounded: diagnose_missingness (MCAR/MAR/MNAR + MISS-RUN/BLOCK/SENTINEL),
# recommend_imputation_plan, apply_imputation_plan, impute_numeric/categorical/
# timeseries, llm_extrapolate_values, evaluate_imputation, imputation_flow.
# ═══════════════════════════════════════════════════════════════════════════════
USE_CASES += [
    UseCase(
        id="DQ-MISS-001",
        category="missingness_imputation",
        title="Diagnose MCAR missingness",
        user_story=(  # nosemgrep
            "As a data scientist I want to confirm whether missingness is completely at random so "
            "that I can safely use a simple imputer."
        ),
        chat_prompt="diagnose the missingness mechanism on data/mcar_sample.csv and tell me if it is MCAR",
        expected_capability="",
        expected_tools=("diagnose_missingness",),
        expected_events=("dataset", "agents"),
        expected_panel_updates=("data", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep
            "GIVEN a dataset with random nulls WHEN diagnosed THEN 'diagnose_missingness' is invoked "
            "AND the mechanism is reported MCAR (pattern MISS-MCAR) AND the Data tab reflects the dataset."
        ),
    ),
    UseCase(
        id="DQ-MISS-002",
        category="missingness_imputation",
        title="Diagnose MAR missingness",
        user_story=(  # nosemgrep
            "As a data scientist I want to detect when missingness is predictable from other "
            "columns so that I choose a model-based imputer (MICE/missForest/kNN)."
        ),
        chat_prompt="diagnose missingness on data/mar_sample.csv and tell me if the null mask is predictable",
        expected_capability="",
        expected_tools=("diagnose_missingness",),
        expected_events=("dataset", "agents"),
        expected_panel_updates=("data", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep
            "GIVEN a dataset whose null mask is predictable WHEN diagnosed THEN 'diagnose_missingness' "
            "reports mechanism MAR (mask-AUC>0.6, pattern MISS-MAR) AND the Data tab reflects it."
        ),
    ),
    UseCase(
        id="DQ-MISS-003",
        category="missingness_imputation",
        title="Diagnose MNAR then impute with pattern-mixture + indicator",
        user_story=(  # nosemgrep
            "As a data scientist I want value-dependent (MNAR) missingness diagnosed and imputed "
            "with a pattern-mixture plus indicator strategy so that the bias is modeled explicitly."
        ),
        chat_prompt="diagnose missingness on data/mnar_sample.csv then impute the MNAR columns with a pattern-mixture strategy",
        expected_capability="",
        expected_tools=("diagnose_missingness", "recommend_imputation_plan", "apply_imputation_plan"),
        expected_events=("dataset", "job", "agents"),
        expected_panel_updates=("data", "pipeline", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep
            "GIVEN a dataset with value-dependent gaps WHEN diagnosed+imputed THEN "
            "'diagnose_missingness' reports MNAR AND 'apply_imputation_plan' runs the "
            "pattern_mixture_plus_indicator strategy AND the Data tab dq_score updates."
        ),
    ),
    UseCase(
        id="DQ-MISS-004",
        category="missingness_imputation",
        title="Detect missing runs over time (MISS-RUN)",
        user_story=(  # nosemgrep
            "As a data engineer I want long contiguous null runs in a time-ordered column detected "
            "so that sensor dropout is interpolated and flagged."
        ),
        chat_prompt="scan data/sensor_ts.csv for long missing runs in the time-ordered sensor column",
        expected_capability="",
        expected_tools=("diagnose_missingness",),
        expected_events=("dataset", "agents"),
        expected_panel_updates=("data", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep
            "GIVEN a time-ordered column with contiguous nulls WHEN diagnosed THEN pattern "
            "'MISS-RUN' is reported with a time_interp_plus_flag suggestion AND the Data tab reflects it."
        ),
    ),
    UseCase(
        id="DQ-MISS-005",
        category="missingness_imputation",
        title="Detect block/era missingness (MISS-BLOCK)",
        user_story=(  # nosemgrep
            "As a data steward I want null density concentrated in a contiguous era detected so "
            "that per-era imputation is applied rather than a global fill."
        ),
        chat_prompt="check data/lineage_60yr.csv for block or era missingness where a whole period never recorded a column",
        expected_capability="",
        expected_tools=("diagnose_missingness",),
        expected_events=("dataset", "agents"),
        expected_panel_updates=("data", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep
            "GIVEN a dataset with block nulls WHEN diagnosed THEN pattern 'MISS-BLOCK' is reported "
            "with a per_era_impute suggestion AND the agent-activity panel records the tool."
        ),
    ),
    UseCase(
        id="DQ-MISS-006",
        category="missingness_imputation",
        title="Detect disguised/sentinel missing values (MISS-SENTINEL)",
        user_story=(  # nosemgrep
            "As a data engineer I want sentinel-encoded missing values (-999, 'N/A', 'unknown') "
            "detected so that they are normalized to NaN before imputation."
        ),
        chat_prompt="find disguised missing values like -999 or 'N/A' in data/bogie.csv and normalize them",
        expected_capability="",
        expected_tools=("diagnose_missingness",),
        expected_events=("dataset", "agents"),
        expected_panel_updates=("data", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep
            "GIVEN sentinel-encoded nulls WHEN diagnosed THEN pattern 'MISS-SENTINEL' is reported "
            "with a normalize_then_impute suggestion AND the Data tab reflects the dataset."
        ),
    ),
    UseCase(
        id="DQ-MISS-007",
        category="missingness_imputation",
        title="Recommend an imputation plan",
        user_story=(  # nosemgrep
            "As a data scientist I want a recommended multi-strategy imputation plan so that each "
            "column gets a mechanism-appropriate imputer."
        ),
        chat_prompt="recommend an imputation plan for data/bogie.csv based on each column's missingness mechanism",
        expected_capability="",
        expected_tools=("recommend_imputation_plan",),
        expected_events=("dataset", "agents"),
        expected_panel_updates=("data", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep
            "GIVEN a dataset with mixed missingness WHEN a plan is requested THEN "
            "'recommend_imputation_plan' is invoked AND the plan maps columns to strategies AND the "
            "agent-activity panel records the tool."
        ),
    ),
    UseCase(
        id="DQ-MISS-008",
        category="missingness_imputation",
        title="Impute numeric columns",
        user_story=(  # nosemgrep
            "As a data engineer I want numeric columns imputed with a chosen numeric strategy so "
            "that downstream models receive a complete matrix."
        ),
        chat_prompt="impute the missing numeric values in data/bogie.csv using a median strategy",
        expected_capability="",
        expected_tools=("impute_numeric",),
        expected_events=("dataset", "agents"),
        expected_panel_updates=("data", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep
            "GIVEN numeric nulls WHEN imputed THEN 'impute_numeric' is invoked AND the returned "
            "dataset has no numeric nulls AND the Data tab dq_score repaints."
        ),
    ),
    UseCase(
        id="DQ-MISS-009",
        category="missingness_imputation",
        title="Impute categorical columns",
        user_story=(  # nosemgrep
            "As a data engineer I want categorical columns imputed with a mode/'missing' category "
            "so that string features are complete."
        ),
        chat_prompt="impute the missing categorical values in data/bogie.csv with the most-frequent category",
        expected_capability="",
        expected_tools=("impute_categorical",),
        expected_events=("dataset", "agents"),
        expected_panel_updates=("data", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep
            "GIVEN categorical nulls WHEN imputed THEN 'impute_categorical' is invoked AND the "
            "returned dataset has no categorical nulls AND the Data tab reflects it."
        ),
    ),
    UseCase(
        id="DQ-MISS-010",
        category="missingness_imputation",
        title="Impute time-series gaps",
        user_story=(  # nosemgrep
            "As a data engineer I want time-series gaps interpolated so that sensor streams remain "
            "temporally coherent."
        ),
        chat_prompt="impute the time-series gaps in data/sensor_ts.csv with interpolation and a filled flag",
        expected_capability="",
        expected_tools=("impute_timeseries",),
        expected_events=("dataset", "agents"),
        expected_panel_updates=("data", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep
            "GIVEN a time-series with gaps WHEN imputed THEN 'impute_timeseries' is invoked AND the "
            "series is interpolated with a fill indicator AND the Data tab reflects it."
        ),
    ),
    UseCase(
        id="DQ-MISS-011",
        category="missingness_imputation",
        title="Self-evaluate imputation quality",
        user_story=(  # nosemgrep
            "As a data scientist I want the imputation self-evaluated (holdout error, distribution "
            "drift) so that I trust the filled values before training."
        ),
        chat_prompt="evaluate the quality of the imputation I just ran on data/bogie.csv against a holdout",
        expected_capability="",
        expected_tools=("evaluate_imputation",),
        expected_events=("dataset", "agents"),
        expected_panel_updates=("data", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep
            "GIVEN an imputed dataset WHEN evaluated THEN 'evaluate_imputation' is invoked AND the "
            "report includes a holdout error + distribution-preservation check AND the panel records it."
        ),
    ),
    UseCase(
        id="DQ-MISS-012",
        category="missingness_imputation",
        title="End-to-end imputation via imputation_flow",
        user_story=(  # nosemgrep
            "As a data scientist I want the full diagnose->recommend->apply->evaluate imputation "
            "pipeline run as one capability so that the dataset is model-ready in one step."
        ),
        chat_prompt="run the imputation flow on data/bogie.csv end to end and report the before/after quality",
        expected_capability="imputation_flow",
        expected_tools=("run_ml_capability",),
        expected_events=("job", "dataset", "agents"),
        expected_panel_updates=("data", "pipeline", "dashboard.active_jobs"),
        acceptance=(  # nosemgrep
            "GIVEN a dataset with missingness WHEN imputation_flow is dispatched THEN "
            "run_ml_capability routes to 'imputation_flow' AND a job event fires AND the "
            "Pipeline/Active-Jobs panels show the job AND the Data dq_score improves."
        ),
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Category 4 — Deduplication & Entity Resolution (6)  [DQ-DEDUP-001..006]
# Grounded: dedupe_dataset (exact/fuzzy), scan_dq_patterns (UNQ-DUP/UNQ-KEY/
# UNQ-FUZZY/CON-REF).
# ═══════════════════════════════════════════════════════════════════════════════
USE_CASES += [
    UseCase(
        id="DQ-DEDUP-001",
        category="dedup_entity_resolution",
        title="Detect exact duplicate rows (UNQ-DUP)",
        user_story=(  # nosemgrep
            "As a data engineer I want fully-duplicated rows detected so that training counts are "
            "not inflated by copies."
        ),
        chat_prompt="scan data/bogie.csv for exact duplicate rows and tell me how many there are",
        expected_capability="",
        expected_tools=("scan_dq_patterns",),
        expected_events=("dataset", "agents"),
        expected_panel_updates=("data", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep
            "GIVEN a dataset with duplicate rows WHEN scanned THEN 'scan_dq_patterns' reports "
            "pattern_id 'UNQ-DUP' with a duplicate count AND the Data tab reflects the dataset."
        ),
    ),
    UseCase(
        id="DQ-DEDUP-002",
        category="dedup_entity_resolution",
        title="Drop exact duplicate rows",
        user_story=(  # nosemgrep
            "As a data engineer I want exact duplicate rows removed so that the deduplicated "
            "dataset becomes the active one."
        ),
        chat_prompt="deduplicate data/bogie.csv by dropping exact duplicate rows",
        expected_capability="",
        expected_tools=("dedupe_dataset",),
        expected_events=("dataset", "agents"),
        expected_panel_updates=("data", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep
            "GIVEN a dataset with duplicates WHEN deduped THEN 'dedupe_dataset' runs with "
            "strategy='exact' AND the returned row count drops AND the Data tab reflects the new dataset."
        ),
    ),
    UseCase(
        id="DQ-DEDUP-003",
        category="dedup_entity_resolution",
        title="Detect duplicate primary key (UNQ-KEY)",
        user_story=(  # nosemgrep
            "As a data steward I want repeated values in a declared key column detected so that "
            "broken uniqueness is surfaced."
        ),
        chat_prompt="check data/bogie.csv for duplicate values in the primary key column unit_id",
        expected_capability="",
        expected_tools=("scan_dq_patterns",),
        expected_events=("dataset", "agents"),
        expected_panel_updates=("data", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep
            "GIVEN a key column with repeats WHEN scanned THEN 'scan_dq_patterns' reports pattern_id "
            "'UNQ-KEY' AND the Data tab reflects the dataset."
        ),
    ),
    UseCase(
        id="DQ-DEDUP-004",
        category="dedup_entity_resolution",
        title="Resolve key collisions on dedup",
        user_story=(  # nosemgrep
            "As a data engineer I want key collisions resolved during dedup so that one canonical "
            "row survives per key."
        ),
        chat_prompt="deduplicate data/bogie.csv keyed on unit_id keeping the most recent row per key",
        expected_capability="",
        expected_tools=("dedupe_dataset",),
        expected_events=("dataset", "agents"),
        expected_panel_updates=("data", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep
            "GIVEN a dataset with key collisions WHEN deduped by key THEN 'dedupe_dataset' runs with "
            "keys=['unit_id'] AND each key appears once AND the Data tab reflects the deduped dataset."
        ),
    ),
    UseCase(
        id="DQ-DEDUP-005",
        category="dedup_entity_resolution",
        title="Detect near-duplicate strings (UNQ-FUZZY)",
        user_story=(  # nosemgrep
            "As a data steward I want high-similarity category strings ('ACME' vs 'ACME.') detected "
            "so that the same entity is not split across labels."
        ),
        chat_prompt="find near-duplicate category strings in the manufacturer column of data/bogie.csv",
        expected_capability="",
        expected_tools=("scan_dq_patterns",),
        expected_events=("dataset", "agents"),
        expected_panel_updates=("data", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep
            "GIVEN near-duplicate categories WHEN scanned THEN 'scan_dq_patterns' reports pattern_id "
            "'UNQ-FUZZY' with candidate merges AND the Data tab reflects the dataset."
        ),
    ),
    UseCase(
        id="DQ-DEDUP-006",
        category="dedup_entity_resolution",
        title="Fuzzy entity-resolution merge",
        user_story=(  # nosemgrep
            "As a data engineer I want fuzzy near-duplicate rows merged so that entity records are "
            "consolidated by string similarity."
        ),
        chat_prompt="deduplicate data/bogie.csv using a fuzzy strategy to merge near-duplicate entity rows",
        expected_capability="",
        expected_tools=("dedupe_dataset",),
        expected_events=("dataset", "agents"),
        expected_panel_updates=("data", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep
            "GIVEN near-duplicate rows WHEN deduped with strategy='fuzzy' THEN 'dedupe_dataset' "
            "merges them AND the returned row count drops AND the Data tab reflects the dataset."
        ),
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Category 5 — Outliers & Distribution (6)  [DQ-OUT-001..006]
# Grounded: detect_anomalies (DST-OUT), scan_dq_patterns (DST-IMB/DST-OUT),
# augment_imbalance (SMOTE — data-extra gated), score_data_quality.
# ═══════════════════════════════════════════════════════════════════════════════
USE_CASES += [
    UseCase(
        id="DQ-OUT-001",
        category="outliers_distribution",
        title="Detect distributional outliers (DST-OUT)",
        user_story=(  # nosemgrep
            "As a data scientist I want extreme values far from the bulk distribution detected so "
            "that I can cap or remove them before training."
        ),
        chat_prompt="detect distributional outliers in the numeric columns of data/bogie.csv",
        expected_capability="",
        expected_tools=("detect_anomalies",),
        expected_events=("dataset", "agents"),
        expected_panel_updates=("data", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep
            "GIVEN numeric columns with extreme values WHEN scanned THEN 'detect_anomalies' is "
            "invoked AND outlier rows/indices are returned AND the Data tab reflects the dataset."
        ),
    ),
    UseCase(
        id="DQ-OUT-002",
        category="outliers_distribution",
        title="Cap or remove outliers",
        user_story=(  # nosemgrep
            "As a data engineer I want detected outliers capped or removed so that the cleaned "
            "dataset is robust to extreme tails."
        ),
        chat_prompt="cap the distributional outliers in data/bogie.csv at the 1st/99th percentile",
        expected_capability="",
        expected_tools=("detect_anomalies", "apply_transform_flow"),
        expected_events=("dataset", "agents"),
        expected_panel_updates=("data", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep
            "GIVEN outliers detected WHEN capped THEN 'detect_anomalies' then a transform runs AND "
            "the returned distribution is winsorized AND the Data tab reflects the dataset."
        ),
    ),
    UseCase(
        id="DQ-OUT-003",
        category="outliers_distribution",
        title="Detect class imbalance (DST-IMB)",
        user_story=(  # nosemgrep
            "As a data scientist I want severe target class imbalance detected so that I pick the "
            "right metric and rebalancing strategy."
        ),
        chat_prompt="check the class balance of the target machine_failure in data/bogie.csv for severe imbalance",
        expected_capability="",
        expected_tools=("scan_dq_patterns",),
        expected_events=("dataset", "agents"),
        expected_panel_updates=("data", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep
            "GIVEN an imbalanced target WHEN scanned THEN 'scan_dq_patterns' reports pattern_id "
            "'DST-IMB' with the minority rate AND the Data tab reflects the dataset."
        ),
    ),
    UseCase(
        id="DQ-OUT-004",
        category="outliers_distribution",
        title="Rebalance minority class with SMOTE",
        user_story=(  # nosemgrep
            "As a data scientist I want the minority class oversampled with SMOTE so that recall on "
            "rare failures improves."
        ),
        chat_prompt="rebalance the minority class in data/bogie.csv using SMOTE oversampling",
        expected_capability="",
        expected_tools=("augment_imbalance",),
        expected_events=("dataset", "agents"),
        expected_panel_updates=("data", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep
            "GIVEN an imbalanced dataset WHEN SMOTE is requested THEN 'augment_imbalance' is invoked "
            "AND (when the data extra is installed) the minority rate rises; without the extra it "
            "returns a clear capability-unavailable error (graceful degrade, Phase-7 gate T-3)."
        ),
        # Phase 7 (gap T-3): augment_imbalance is a real @tool with lazy imblearn import
        # and a clear extra-required fail-soft; supported end-to-end (extra installed) or
        # gracefully degraded (extra absent). offline stays False — real oversampling
        # needs the optional 'data' extra.
        offline=False,
    ),
    UseCase(
        id="DQ-OUT-005",
        category="outliers_distribution",
        title="Anomaly detection scorecard",
        user_story=(  # nosemgrep
            "As a data steward I want an anomaly-detection scorecard so that I get a per-column "
            "outlier rate summary in one pass."
        ),
        chat_prompt="run anomaly detection on data/bogie.csv and give me a per-column outlier-rate scorecard",
        expected_capability="",
        expected_tools=("detect_anomalies",),
        expected_events=("dataset", "agents"),
        expected_panel_updates=("data", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep
            "GIVEN a dataset WHEN anomaly detection runs THEN 'detect_anomalies' is invoked AND a "
            "per-column outlier-rate summary is returned AND the agent-activity panel records it."
        ),
    ),
    UseCase(
        id="DQ-OUT-006",
        category="outliers_distribution",
        title="Distribution health score",
        user_story=(  # nosemgrep
            "As a data steward I want a distribution-health quality score so that skew/outlier load "
            "is folded into the overall dq score."
        ),
        chat_prompt="score the distribution health of data/bogie.csv with target machine_failure",
        expected_capability="",
        expected_tools=("score_data_quality",),
        expected_events=("dataset", "agents"),
        expected_panel_updates=("data", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep
            "GIVEN a dataset WHEN scored THEN 'score_data_quality' is invoked AND a distribution "
            "component contributes to dq_score AND the Data tab's Quality Score repaints."
        ),
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Category 6 — Schema & Era Drift / Lineage (8)  [DQ-DRIFT-001..008]
# Grounded: detect_ocr_artifacts (OCR-CONF/NOISE/CASE), scan_dq_patterns
# (LIN-SCHEMA/LIN-ENCODE/DST-DRIFT), analyze_drift, query_lineage.
# ═══════════════════════════════════════════════════════════════════════════════
USE_CASES += [
    UseCase(
        id="DQ-DRIFT-001",
        category="schema_drift_lineage",
        title="Detect schema drift across eras (LIN-SCHEMA)",
        user_story=(  # nosemgrep
            "As a data engineer I want columns that exist only in a contiguous era detected so that "
            "60-year lineage schema changes are mapped."
        ),
        chat_prompt="scan data/lineage_60yr.csv for schema drift where a column exists only in some eras",
        expected_capability="",
        expected_tools=("scan_dq_patterns",),
        expected_events=("dataset", "agents"),
        expected_panel_updates=("data", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep
            "GIVEN a dataset spanning eras WHEN scanned THEN 'scan_dq_patterns' reports pattern_id "
            "'LIN-SCHEMA' AND the Data tab reflects the dataset."
        ),
    ),
    UseCase(
        id="DQ-DRIFT-002",
        category="schema_drift_lineage",
        title="Detect encoding inconsistency (LIN-ENCODE)",
        user_story=(  # nosemgrep
            "As a data engineer I want mojibake / mixed-encoding artifacts detected so that decades "
            "of digitization noise are normalized."
        ),
        chat_prompt="check data/lineage_60yr.csv for encoding inconsistencies and mojibake in the text columns",
        expected_capability="",
        expected_tools=("scan_dq_patterns",),
        expected_events=("dataset", "agents"),
        expected_panel_updates=("data", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep
            "GIVEN text with mixed encodings WHEN scanned THEN 'scan_dq_patterns' reports pattern_id "
            "'LIN-ENCODE' AND the agent-activity panel records the tool."
        ),
    ),
    UseCase(
        id="DQ-DRIFT-003",
        category="schema_drift_lineage",
        title="Detect era distribution drift (DST-DRIFT)",
        user_story=(  # nosemgrep
            "As a data scientist I want feature distribution drift across era bins (KS/PSI) detected "
            "so that era-aware modeling is triggered."
        ),
        chat_prompt="check data/lineage_60yr.csv for feature distribution drift across the era bins",
        expected_capability="",
        expected_tools=("scan_dq_patterns",),
        expected_events=("dataset", "agents"),
        expected_panel_updates=("data", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep
            "GIVEN features shifting across eras WHEN scanned THEN 'scan_dq_patterns' reports "
            "pattern_id 'DST-DRIFT' with KS/PSI evidence AND the Data tab reflects the dataset."
        ),
    ),
    UseCase(
        id="DQ-DRIFT-004",
        category="schema_drift_lineage",
        title="Detect OCR character confusion (OCR-CONF)",
        user_story=(  # nosemgrep
            "As a data steward I want classic OCR confusions (O<->0, l<->1, S<->5) detected in "
            "scanned IDs so that they are corrected or flagged."
        ),
        chat_prompt="scan data/ocr_scanned.csv for OCR character-confusion artifacts in the id columns",
        expected_capability="",
        expected_tools=("detect_ocr_artifacts",),
        expected_events=("dataset", "agents"),
        expected_panel_updates=("data", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep
            "GIVEN scanned text with OCR confusions WHEN scanned THEN 'detect_ocr_artifacts' is "
            "invoked AND pattern 'OCR-CONF' candidates are returned AND the Data tab reflects it."
        ),
    ),
    UseCase(
        id="DQ-DRIFT-005",
        category="schema_drift_lineage",
        title="Detect OCR stray noise (OCR-NOISE)",
        user_story=(  # nosemgrep
            "As a data steward I want stray punctuation / broken decimals from scanning detected so "
            "that OCR noise is cleaned."
        ),
        chat_prompt="check data/ocr_scanned.csv for OCR stray noise like broken decimals or stray punctuation",
        expected_capability="",
        expected_tools=("detect_ocr_artifacts",),
        expected_events=("dataset", "agents"),
        expected_panel_updates=("data", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep
            "GIVEN scanned text with stray noise WHEN scanned THEN 'detect_ocr_artifacts' reports "
            "pattern 'OCR-NOISE' AND the agent-activity panel records the tool."
        ),
    ),
    UseCase(
        id="DQ-DRIFT-006",
        category="schema_drift_lineage",
        title="Detect casing/whitespace variants (OCR-CASE)",
        user_story=(  # nosemgrep
            "As a data engineer I want mixed-casing / stray-whitespace category variants detected "
            "so that categories are canonicalized."
        ),
        chat_prompt="find casing and whitespace variants of the same category in data/ocr_scanned.csv",
        expected_capability="",
        expected_tools=("detect_ocr_artifacts",),
        expected_events=("dataset", "agents"),
        expected_panel_updates=("data", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep
            "GIVEN categories differing only by case/whitespace WHEN scanned THEN "
            "'detect_ocr_artifacts' reports pattern 'OCR-CASE' AND the Data tab reflects the dataset."
        ),
    ),
    UseCase(
        id="DQ-DRIFT-007",
        category="schema_drift_lineage",
        title="Analyze production feature drift",
        user_story=(  # nosemgrep
            "As an ML engineer I want production feature drift analyzed against a baseline so that "
            "I know when a deployed model's inputs have shifted."
        ),
        chat_prompt="analyze drift between the training baseline and the latest production data for the bogie model",
        expected_capability="",
        expected_tools=("analyze_drift",),
        expected_events=("model", "agents"),
        expected_panel_updates=("mlops", "dashboard.observability"),
        acceptance=(  # nosemgrep
            "GIVEN a baseline and current data WHEN drift is analyzed THEN 'analyze_drift' is invoked "
            "AND a drift score/verdict is returned AND the MLOps/Observability panel reflects it."
        ),
    ),
    UseCase(
        id="DQ-DRIFT-008",
        category="schema_drift_lineage",
        title="Query dataset lineage",
        user_story=(  # nosemgrep
            "As a data steward I want to query a dataset's lineage so that I can trace which "
            "upstream sources and transforms produced it."
        ),
        chat_prompt="query the lineage for the dataset reference s3://amlas/silver/bogie",
        expected_capability="",
        expected_tools=("query_lineage",),
        expected_events=("dataset", "agents"),
        expected_panel_updates=("data", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep
            "GIVEN a dataset ref WHEN lineage is queried THEN 'query_lineage' is invoked AND the "
            "upstream sources/transforms are returned AND the agent-activity panel records the tool."
        ),
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Category 7 — Leakage & PII / Governance (8)  [DQ-GOV-001..008]
# Grounded: leakage_check_tool (LEAK-CARD), check_data_access, query_lineage,
# run_guardrail_check, create_model_card. PII detection = GAP T-1 (3 cases).
# ═══════════════════════════════════════════════════════════════════════════════
USE_CASES += [
    UseCase(
        id="DQ-GOV-001",
        category="leakage_pii_governance",
        title="Detect identifier/leakage columns (LEAK-CARD)",
        user_story=(  # nosemgrep
            "As a data scientist I want identifier-like or near-unique columns that leak the target "
            "detected so that I drop or quarantine them before training."
        ),
        chat_prompt="check data/bogie.csv for leakage or identifier columns that could leak the target machine_failure",
        expected_capability="",
        expected_tools=("leakage_check_tool",),
        expected_events=("dataset", "agents"),
        expected_panel_updates=("data", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep
            "GIVEN a dataset with an identifier column WHEN checked THEN 'leakage_check_tool' is "
            "invoked AND the leaking columns are listed (pattern LEAK-CARD) AND the Data tab reflects it."
        ),
    ),
    UseCase(
        id="DQ-GOV-002",
        category="leakage_pii_governance",
        title="Detect train/test leakage across a split",
        user_story=(  # nosemgrep
            "As an ML engineer I want train/test leakage detected so that rows or features shared "
            "across the split are caught before evaluation."
        ),
        chat_prompt="check for train/test leakage between the split files for data/bogie.csv",
        expected_capability="",
        expected_tools=("leakage_check_tool",),
        expected_events=("dataset", "agents"),
        expected_panel_updates=("data", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep
            "GIVEN a train/test split WHEN checked THEN 'leakage_check_tool' is invoked AND any "
            "cross-split leakage is reported AND the agent-activity panel records the tool."
        ),
    ),
    UseCase(
        id="DQ-GOV-003",
        category="leakage_pii_governance",
        title="Verify data-access authorization",
        user_story=(  # nosemgrep
            "As a governance officer I want to verify a principal's access to a resource so that "
            "unauthorized data use is blocked."
        ),
        chat_prompt="check whether the principal analyst-role is allowed to access s3://amlas/silver/bogie",
        expected_capability="",
        expected_tools=("check_data_access",),
        expected_events=("agents",),
        expected_panel_updates=("dashboard.agent_activity",),
        acceptance=(  # nosemgrep
            "GIVEN a principal and resource WHEN checked THEN 'check_data_access' is invoked AND an "
            "allow/deny decision is returned AND the agent-activity panel records the tool."
        ),
    ),
    UseCase(
        id="DQ-GOV-004",
        category="leakage_pii_governance",
        title="Query governance lineage for a resource",
        user_story=(  # nosemgrep
            "As a governance officer I want a resource's lineage queried so that I can audit its "
            "provenance for compliance."
        ),
        chat_prompt="query the governance lineage for the model artifact ref bogie-classifier-v3",
        expected_capability="",
        expected_tools=("query_lineage",),
        expected_events=("agents",),
        expected_panel_updates=("dashboard.agent_activity",),
        acceptance=(  # nosemgrep
            "GIVEN a resource ref WHEN lineage is queried THEN 'query_lineage' is invoked AND the "
            "provenance chain is returned AND the agent-activity panel records the tool."
        ),
    ),
    UseCase(
        id="DQ-GOV-005",
        category="leakage_pii_governance",
        title="Run a governance guardrail check",
        user_story=(  # nosemgrep
            "As a governance officer I want a guardrail check run so that policy violations are "
            "caught before a model is promoted."
        ),
        chat_prompt="run a governance guardrail check on the bogie model before approval",
        expected_capability="",
        expected_tools=("run_guardrail_check",),
        expected_events=("model", "agents"),
        expected_panel_updates=("mlops", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep
            "GIVEN a model WHEN a guardrail check runs THEN 'run_guardrail_check' is invoked AND a "
            "pass/fail with findings is returned AND the MLOps panel reflects it."
        ),
    ),
    UseCase(
        id="DQ-GOV-006",
        category="leakage_pii_governance",
        title="Detect PII columns (emails/SSN/phone/names)",
        user_story=(  # nosemgrep
            "As a governance officer I want PII columns (emails, SSNs, phone numbers, names) "
            "detected so that regulated fields are identified before the data is used."
        ),
        chat_prompt="scan data/customers.csv for PII columns such as emails, SSNs, phone numbers, and names",
        expected_capability="",
        expected_tools=("detect_pii",),
        expected_events=("dataset", "agents"),
        expected_panel_updates=("data", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep
            "GIVEN a dataset with PII WHEN scanned THEN the 'detect_pii' tool reports the PII columns "
            "and types (masked samples) AND the Data tab reflects the dataset."
        ),
        # Phase 7 (gap T-1 CLOSED): tools/pii_tools.py::detect_pii is a stdlib-regex
        # @tool bound in SAFE_LLM_TOOLS (read-only) — deterministic, offline.
    ),
    UseCase(
        id="DQ-GOV-007",
        category="leakage_pii_governance",
        title="Redact/mask PII columns",
        user_story=(  # nosemgrep
            "As a governance officer I want detected PII masked/redacted so that a de-identified "
            "dataset can be safely shared for modeling."
        ),
        chat_prompt="redact the PII columns in data/customers.csv and produce a de-identified copy",
        expected_capability="",
        expected_tools=("redact_pii",),
        expected_events=("dataset", "agents"),
        expected_panel_updates=("data", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep
            "GIVEN a dataset with PII WHEN redaction runs THEN the 'redact_pii' tool masks the PII "
            "columns AND writes a de-identified CSV artifact (confirm-gated, state-changing)."
        ),
        # Phase 7 (gap T-1 CLOSED): tools/pii_tools.py::redact_pii is a @tool bound in
        # SAFE_LLM_TOOLS (state-changing / confirm-gated) — deterministic, offline.
    ),
    UseCase(
        id="DQ-GOV-008",
        category="leakage_pii_governance",
        title="PII compliance gate before sharing",
        user_story=(  # nosemgrep
            "As a governance officer I want a PII compliance gate that blocks sharing when "
            "unredacted PII remains so that regulated data never leaves de-identified."
        ),
        chat_prompt="run a PII compliance gate on data/customers.csv and block sharing if any unredacted PII remains",
        expected_capability="",
        expected_tools=("pii_compliance_gate", "detect_pii"),
        expected_events=("dataset", "agents"),
        expected_panel_updates=("data", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep
            "GIVEN a dataset WHEN the PII gate runs THEN 'pii_compliance_gate' (grounded in "
            "'detect_pii') returns action=BLOCK if un-redacted PII remains (action=ALLOW when clean) "
            "AND the Data/agent panels reflect the verdict."
        ),
        # Phase 7 (gap T-1 CLOSED): tools/pii_tools.py::pii_compliance_gate runs the
        # detect_pii core and emits a run_guardrail_check-shaped {action,passed,violations}
        # verdict — bound in SAFE_LLM_TOOLS (read-only), deterministic, offline.
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Category 8 — Ingestion & Lakehouse (10)  [DQ-LAKE-001..010]
# Grounded: read_lake_slice, write_lake_dataset, list_lake_catalog, run_crawler,
# run_sql (Athena), register_dataset_version. Partitioning = GAP T-2 (2 cases).
# ═══════════════════════════════════════════════════════════════════════════════
USE_CASES += [
    UseCase(
        id="DQ-LAKE-001",
        category="ingestion_lakehouse",
        title="Read a medallion slice into the active dataset",
        user_story=(  # nosemgrep
            "As a data engineer I want to read a silver/gold medallion slice so that it becomes the "
            "active dataset for downstream work."
        ),
        chat_prompt="read a slice of the silver table s3://amlas/silver/bogie and make it the active dataset",
        expected_capability="",
        expected_tools=("read_lake_slice",),
        expected_events=("dataset", "selection", "agents"),
        expected_panel_updates=("lake", "data"),
        acceptance=(  # nosemgrep
            "GIVEN a medallion table WHEN a slice is read THEN 'read_lake_slice' is invoked AND "
            "context_bus.state().dataset.source == 'lake_tab' AND the Data tab reflects the slice."
        ),
    ),
    UseCase(
        id="DQ-LAKE-002",
        category="ingestion_lakehouse",
        title="Write a dataset back to the silver/gold tier",
        user_story=(  # nosemgrep
            "As a data engineer I want to write a cleaned dataset back to silver/gold so that the "
            "curated slice becomes the active dataset (bronze rejected)."
        ),
        chat_prompt="write the current cleaned dataset back to the silver tier at s3://amlas/silver/bogie_clean",
        expected_capability="",
        expected_tools=("write_lake_dataset",),
        expected_events=("dataset", "selection", "agents"),
        expected_panel_updates=("lake", "data"),
        acceptance=(  # nosemgrep
            "GIVEN a cleaned dataset WHEN written to silver/gold THEN 'write_lake_dataset' is invoked "
            "AND the written slice becomes the active dataset (source == 'lake_tab') AND the Lake/Data "
            "tabs reflect it."
        ),
    ),
    UseCase(
        id="DQ-LAKE-003",
        category="ingestion_lakehouse",
        title="List the lake/Glue catalog",
        user_story=(  # nosemgrep
            "As a data engineer I want to list the lake/Glue catalog so that I can discover "
            "available tables and tiers."
        ),
        chat_prompt="list the tables in the lake catalog so I can see what silver and gold tables exist",
        expected_capability="",
        expected_tools=("list_lake_catalog",),
        expected_events=("agents",),
        expected_panel_updates=("lake", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep
            "GIVEN a catalog WHEN listed THEN 'list_lake_catalog' is invoked AND the available "
            "tables are returned AND the Lake tab reflects the catalog (empty placeholder offline)."
        ),
    ),
    UseCase(
        id="DQ-LAKE-004",
        category="ingestion_lakehouse",
        title="Run a Glue crawler to catalog new data",
        user_story=(  # nosemgrep
            "As a data engineer I want to run a Glue crawler on an S3 target so that newly-landed "
            "data is cataloged and queryable."
        ),
        chat_prompt="run a glue crawler over s3://amlas/raw/bogie to catalog the newly landed data",
        expected_capability="",
        expected_tools=("run_crawler",),
        expected_events=("job", "agents"),
        expected_panel_updates=("lake", "dashboard.active_jobs"),
        acceptance=(  # nosemgrep
            "GIVEN an S3 target WHEN a crawler runs THEN 'run_crawler' is invoked AND a job event "
            "fires AND the Active-Jobs/Lake panels reflect the crawl."
        ),
    ),
    UseCase(
        id="DQ-LAKE-005",
        category="ingestion_lakehouse",
        title="Query the lakehouse with Athena SQL",
        user_story=(  # nosemgrep
            "As a data analyst I want to run an Athena SQL query so that I can slice the lakehouse "
            "without downloading it."
        ),
        chat_prompt="run the SQL query SELECT count(*) FROM bogie WHERE machine_failure = 1 against the lakehouse",
        expected_capability="",
        expected_tools=("run_sql",),
        expected_events=("dataset", "agents"),
        expected_panel_updates=("lake", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep
            "GIVEN a SQL query WHEN executed THEN 'run_sql' is invoked AND a result set is returned "
            "AND the Lake tab reflects the query output."
        ),
    ),
    UseCase(
        id="DQ-LAKE-006",
        category="ingestion_lakehouse",
        title="Register a versioned dataset",
        user_story=(  # nosemgrep
            "As a data engineer I want to register a versioned dataset so that a promoted slice is "
            "tracked with an immutable version."
        ),
        chat_prompt="register data/bogie_clean.csv as a new silver dataset version at s3://amlas/silver/bogie_clean",
        expected_capability="",
        expected_tools=("register_dataset_version",),
        expected_events=("dataset", "agents"),
        expected_panel_updates=("lake", "data"),
        acceptance=(  # nosemgrep
            "GIVEN a local path and destination WHEN registered THEN 'register_dataset_version' is "
            "invoked AND a version id is returned AND the Lake/Data tabs reflect the registered version."
        ),
    ),
    UseCase(
        id="DQ-LAKE-007",
        category="ingestion_lakehouse",
        title="Ingest raw file into bronze tier",
        user_story=(  # nosemgrep
            "As a data engineer I want to ingest a raw file into the bronze tier so that raw data "
            "lands in the medallion architecture."
        ),
        chat_prompt="ingest the raw file data/bogie_raw.csv into the bronze tier of the lake",
        expected_capability="",
        expected_tools=("register_dataset_version",),
        expected_events=("dataset", "agents"),
        expected_panel_updates=("lake", "data"),
        acceptance=(  # nosemgrep
            "GIVEN a raw file WHEN ingested to bronze THEN 'register_dataset_version' is invoked with "
            "tier='bronze' AND the landed dataset is tracked AND the Lake tab reflects it."
        ),
    ),
    UseCase(
        id="DQ-LAKE-008",
        category="ingestion_lakehouse",
        title="Profile a lakehouse table at scale",
        user_story=(  # nosemgrep
            "As a data engineer I want to profile a lakehouse table at scale so that I understand a "
            "large table before querying it."
        ),
        chat_prompt="profile the lakehouse table s3://amlas/gold/bogie at scale and summarize the columns",
        expected_capability="",
        expected_tools=("run_profile",),
        expected_events=("dataset", "agents"),
        expected_panel_updates=("lake", "data"),
        acceptance=(  # nosemgrep
            "GIVEN a lakehouse table uri WHEN profiled THEN 'run_profile' is invoked AND a scaled "
            "profile summary is returned AND the Lake/Data tabs reflect the profiled table."
        ),
    ),
    UseCase(
        id="DQ-LAKE-009",
        category="ingestion_lakehouse",
        title="Partition a dataset by a Hive key",
        user_story=(  # nosemgrep
            "As a data engineer I want to write a dataset partitioned by a Hive key (e.g. by era) "
            "so that downstream queries prune partitions efficiently."
        ),
        chat_prompt="write s3://amlas/silver/bogie partitioned by era so that Athena can prune partitions",
        expected_capability="",
        expected_tools=("partition_dataset",),
        expected_events=("dataset", "agents"),
        expected_panel_updates=("lake", "data"),
        acceptance=(  # nosemgrep
            "GIVEN a dataset and a partition key WHEN partitioned THEN the 'partition_dataset' tool "
            "writes a Hive-style key=value/ partition tree (no data loss) AND the partitioned layout "
            "is reported AND the Lake tab reflects it."
        ),
        # Phase 7 (gap T-2 CLOSED): tools/lake/partition.py::partition_dataset is a @tool
        # bound in SAFE_LLM_TOOLS (state-changing / confirm-gated) — offline temp-dir write.
    ),
    UseCase(
        id="DQ-LAKE-010",
        category="ingestion_lakehouse",
        title="Recommend a partitioning strategy",
        user_story=(  # nosemgrep
            "As a data engineer I want a recommended partitioning strategy so that I choose "
            "partition keys that balance file size and query pruning."
        ),
        chat_prompt="recommend a partitioning strategy for the large table s3://amlas/gold/bogie",
        expected_capability="",
        expected_tools=("recommend_partition_strategy",),
        expected_events=("dataset", "agents"),
        expected_panel_updates=("lake", "data"),
        acceptance=(  # nosemgrep
            "GIVEN a large table WHEN a partitioning strategy is requested THEN "
            "'recommend_partition_strategy' returns recommended keys + per-column cardinality + a "
            "rationale AND the Lake tab reflects the recommendation."
        ),
        # Phase 7 (gap T-2 CLOSED): tools/lake/partition.py::recommend_partition_strategy
        # is a read-only cardinality-profile advisor bound in SAFE_LLM_TOOLS — offline.
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Category 9 — Transformation & ETL Jobs (8)  [DQ-ETL-001..008]
# Grounded: auto_preprocess, apply_transform_flow, run_glue_job, run_processing_job,
# run_batch_transform, split_dataset, feature_engineering_flow.
# ═══════════════════════════════════════════════════════════════════════════════
USE_CASES += [
    UseCase(
        id="DQ-ETL-001",
        category="transformation_etl",
        title="Auto-preprocess a dataset",
        user_story=(  # nosemgrep
            "As a data scientist I want a dataset auto-preprocessed (encode/scale/impute) so that a "
            "model-ready matrix and manifest are produced in one step."
        ),
        chat_prompt="auto-preprocess data/bogie.csv for target machine_failure and save the transformed dataset",
        expected_capability="",
        expected_tools=("auto_preprocess",),
        expected_events=("dataset", "agents"),
        expected_panel_updates=("data", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep
            "GIVEN a raw dataset WHEN auto-preprocessed THEN 'auto_preprocess' is invoked (confirm-"
            "gated) AND a transformed dataset + manifest are produced AND the Data tab reflects it."
        ),
    ),
    UseCase(
        id="DQ-ETL-002",
        category="transformation_etl",
        title="Apply a declarative transform flow",
        user_story=(  # nosemgrep
            "As a data engineer I want to apply a declarative transform spec so that a repeatable "
            "ETL step produces a curated slice."
        ),
        chat_prompt="apply a transform flow to s3://amlas/silver/bogie that casts types and drops the id column",
        expected_capability="",
        expected_tools=("apply_transform_flow",),
        expected_events=("dataset", "agents"),
        expected_panel_updates=("data", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep
            "GIVEN a dataset uri and transform spec WHEN applied THEN 'apply_transform_flow' is "
            "invoked AND the transformed output uri is returned AND the Data tab reflects it."
        ),
    ),
    UseCase(
        id="DQ-ETL-003",
        category="transformation_etl",
        title="Run a Glue ETL job",
        user_story=(  # nosemgrep
            "As a data engineer I want to run a Glue ETL job so that a large transformation runs "
            "server-side on the lakehouse."
        ),
        chat_prompt="run a glue job to transform s3://amlas/bronze/bogie into a cleaned silver table",
        expected_capability="",
        expected_tools=("run_glue_job",),
        expected_events=("job", "agents"),
        expected_panel_updates=("pipeline", "dashboard.active_jobs"),
        acceptance=(  # nosemgrep
            "GIVEN a Glue job config WHEN run THEN 'run_glue_job' is invoked AND a job event fires "
            "AND the Active-Jobs/Pipeline panels reflect the running Glue job."
        ),
    ),
    UseCase(
        id="DQ-ETL-004",
        category="transformation_etl",
        title="Run a SageMaker Processing job",
        user_story=(  # nosemgrep
            "As a data engineer I want to run a Processing job so that a containerized preprocessing "
            "step scales beyond a single node."
        ),
        chat_prompt="run a processing job to featurize s3://amlas/silver/bogie into a training-ready dataset",
        expected_capability="",
        expected_tools=("run_processing_job",),
        expected_events=("job", "agents", "processes"),
        expected_panel_updates=("pipeline", "dashboard.active_jobs", "dashboard.processes"),
        acceptance=(  # nosemgrep
            "GIVEN a processing config WHEN run THEN 'run_processing_job' is invoked AND a job (and "
            "local submit-poll process) event fires AND the Active-Jobs/Processes panels reflect it."
        ),
    ),
    UseCase(
        id="DQ-ETL-005",
        category="transformation_etl",
        title="Run a batch transform over a dataset",
        user_story=(  # nosemgrep
            "As an ML engineer I want to run a batch transform so that a model scores a whole "
            "dataset offline."
        ),
        chat_prompt="run a batch transform of the bogie model over s3://amlas/silver/bogie and write the scored output",
        expected_capability="",
        expected_tools=("run_batch_transform",),
        expected_events=("job", "agents"),
        expected_panel_updates=("pipeline", "dashboard.active_jobs"),
        acceptance=(  # nosemgrep
            "GIVEN a model and input dataset WHEN a batch transform runs THEN 'run_batch_transform' "
            "is invoked AND a job event fires AND the Active-Jobs panel reflects the scoring job."
        ),
    ),
    UseCase(
        id="DQ-ETL-006",
        category="transformation_etl",
        title="Split a dataset into train/test",
        user_story=(  # nosemgrep
            "As a data scientist I want a dataset split into train/test (stratified) so that "
            "evaluation uses a held-out set."
        ),
        chat_prompt="split data/bogie.csv into stratified train and test sets on target machine_failure",
        expected_capability="",
        expected_tools=("split_dataset",),
        expected_events=("dataset", "agents"),
        expected_panel_updates=("data", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep
            "GIVEN a dataset WHEN split THEN 'split_dataset' is invoked (confirm-gated) AND train/"
            "test files are produced AND the Data tab reflects the split."
        ),
    ),
    UseCase(
        id="DQ-ETL-007",
        category="transformation_etl",
        title="Normalize/canonicalize categorical values",
        user_story=(  # nosemgrep
            "As a data engineer I want categorical values canonicalized via a transform so that "
            "casing/whitespace variants collapse to one label."
        ),
        chat_prompt="apply a transform flow to data/bogie.csv that canonicalizes the manufacturer casing and whitespace",
        expected_capability="",
        expected_tools=("apply_transform_flow",),
        expected_events=("dataset", "agents"),
        expected_panel_updates=("data", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep
            "GIVEN a dataset with categorical variants WHEN a canonicalizing transform is applied "
            "THEN 'apply_transform_flow' is invoked AND labels collapse AND the Data tab reflects it."
        ),
    ),
    UseCase(
        id="DQ-ETL-008",
        category="transformation_etl",
        title="End-to-end feature ETL via feature_engineering_flow",
        user_story=(  # nosemgrep
            "As a data scientist I want the full preprocessing+feature ETL run as one capability so "
            "that a model-ready feature set is produced end to end."
        ),
        chat_prompt="run the feature engineering flow on data/bogie.csv and produce a model-ready feature set",
        expected_capability="feature_engineering_flow",
        expected_tools=("run_ml_capability",),
        expected_events=("job", "dataset", "agents"),
        expected_panel_updates=("pipeline", "data", "dashboard.active_jobs"),
        acceptance=(  # nosemgrep
            "GIVEN a dataset WHEN feature_engineering_flow is dispatched THEN run_ml_capability routes "
            "to 'feature_engineering_flow' AND a job event fires AND the Pipeline/Data/Active-Jobs "
            "panels reflect the feature ETL."
        ),
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Category 10 — Feature Engineering & Store (8)  [DQ-FEAT-001..008]
# Grounded: feature_engineering_flow (interaction/polynomial/target-encode via
# fe_advanced_tools), feature_store_ingest/get, feature_pit_dataset,
# save_feature_manifest.
# ═══════════════════════════════════════════════════════════════════════════════
USE_CASES += [
    UseCase(
        id="DQ-FEAT-001",
        category="feature_engineering_store",
        title="Create interaction features",
        user_story=(  # nosemgrep
            "As a data scientist I want interaction features generated so that non-linear feature "
            "combinations are available to the model."
        ),
        chat_prompt="engineer interaction features between torque and rotational_speed for data/bogie.csv",
        expected_capability="feature_engineering_flow",
        expected_tools=("run_ml_capability",),
        expected_events=("job", "dataset", "agents"),
        expected_panel_updates=("data", "pipeline", "dashboard.active_jobs"),
        acceptance=(  # nosemgrep
            "GIVEN a dataset WHEN interaction features are requested THEN feature_engineering_flow is "
            "dispatched (create_interaction_features) AND a job event fires AND the Data/Pipeline "
            "panels reflect the enriched feature set."
        ),
    ),
    UseCase(
        id="DQ-FEAT-002",
        category="feature_engineering_store",
        title="Create polynomial features",
        user_story=(  # nosemgrep
            "As a data scientist I want polynomial features generated so that curvature in numeric "
            "predictors is captured."
        ),
        chat_prompt="add polynomial features up to degree 2 for the numeric columns of data/bogie.csv",
        expected_capability="feature_engineering_flow",
        expected_tools=("run_ml_capability",),
        expected_events=("job", "dataset", "agents"),
        expected_panel_updates=("data", "pipeline", "dashboard.active_jobs"),
        acceptance=(  # nosemgrep
            "GIVEN a dataset WHEN polynomial features are requested THEN feature_engineering_flow is "
            "dispatched (create_polynomial_features) AND a job event fires AND the Data/Pipeline "
            "panels reflect the expanded features."
        ),
    ),
    UseCase(
        id="DQ-FEAT-003",
        category="feature_engineering_store",
        title="Target-encode high-cardinality categoricals",
        user_story=(  # nosemgrep
            "As a data scientist I want high-cardinality categoricals target-encoded so that they "
            "become model-usable without exploding dimensionality."
        ),
        chat_prompt="target-encode the high-cardinality categorical columns in data/bogie.csv against machine_failure",
        expected_capability="feature_engineering_flow",
        expected_tools=("run_ml_capability",),
        expected_events=("job", "dataset", "agents"),
        expected_panel_updates=("data", "pipeline", "dashboard.active_jobs"),
        acceptance=(  # nosemgrep
            "GIVEN high-cardinality categoricals WHEN target-encoded THEN feature_engineering_flow is "
            "dispatched (target_encode) AND a job event fires AND the Data panel reflects the encoded "
            "features."
        ),
    ),
    UseCase(
        id="DQ-FEAT-004",
        category="feature_engineering_store",
        title="Ingest features into the feature store",
        user_story=(  # nosemgrep
            "As an ML engineer I want features ingested into the feature store so that they are "
            "reusable across training and serving."
        ),
        chat_prompt="ingest the engineered bogie features into the feature store group bogie_features",
        expected_capability="",
        expected_tools=("feature_store_ingest",),
        expected_events=("dataset", "agents"),
        expected_panel_updates=("mlops", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep
            "GIVEN a feature set WHEN ingested THEN 'feature_store_ingest' is invoked AND the group/"
            "record ids are returned AND the MLOps/agent panels reflect the ingest."
        ),
    ),
    UseCase(
        id="DQ-FEAT-005",
        category="feature_engineering_store",
        title="Retrieve features from the store",
        user_story=(  # nosemgrep
            "As an ML engineer I want to retrieve a record's features from the store so that online "
            "serving reads consistent features."
        ),
        chat_prompt="get the online features for record unit-42 from the feature store group bogie_features",
        expected_capability="",
        expected_tools=("feature_store_get",),
        expected_events=("agents",),
        expected_panel_updates=("mlops", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep
            "GIVEN a group and record id WHEN retrieved THEN 'feature_store_get' is invoked AND the "
            "feature record is returned AND the agent-activity panel records the tool."
        ),
    ),
    UseCase(
        id="DQ-FEAT-006",
        category="feature_engineering_store",
        title="Build a point-in-time (no-leakage) dataset",
        user_story=(  # nosemgrep
            "As an ML engineer I want a point-in-time correct dataset built so that no future "
            "information leaks into training features."
        ),
        chat_prompt="build a point-in-time feature dataset for bogie_features as of the training cutoff so there is no leakage",
        expected_capability="",
        expected_tools=("feature_pit_dataset",),
        expected_events=("dataset", "agents"),
        expected_panel_updates=("data", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep
            "GIVEN a feature group and cutoff WHEN a PIT dataset is built THEN 'feature_pit_dataset' "
            "is invoked AND a leakage-free dataset is returned AND the Data tab reflects it."
        ),
    ),
    UseCase(
        id="DQ-FEAT-007",
        category="feature_engineering_store",
        title="Save a feature manifest",
        user_story=(  # nosemgrep
            "As an ML engineer I want a feature manifest saved so that the feature definitions are "
            "versioned and reproducible."
        ),
        chat_prompt="save a feature manifest for the engineered bogie feature set",
        expected_capability="",
        expected_tools=("save_feature_manifest",),
        expected_events=("dataset", "agents"),
        expected_panel_updates=("data", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep
            "GIVEN an engineered feature set WHEN a manifest is saved THEN 'save_feature_manifest' is "
            "invoked AND a manifest path is returned AND the agent-activity panel records the tool."
        ),
    ),
    UseCase(
        id="DQ-FEAT-008",
        category="feature_engineering_store",
        title="CAAFE-style iterative feature discovery",
        user_story=(  # nosemgrep
            "As a data scientist I want an LLM-guided iterative feature-discovery loop so that "
            "candidate features are proposed, evaluated, and kept if they help."
        ),
        chat_prompt="run an iterative feature-discovery loop on data/bogie.csv to propose and evaluate new features",
        expected_capability="feature_engineering_flow",
        expected_tools=("run_ml_capability",),
        expected_events=("job", "dataset", "agents"),
        expected_panel_updates=("data", "pipeline", "dashboard.active_jobs"),
        acceptance=(  # nosemgrep
            "GIVEN a dataset WHEN a feature-discovery loop runs THEN feature_engineering_flow is "
            "dispatched (advanced_preprocess/CAAFE path) AND a job event fires AND the Data/Pipeline "
            "panels reflect the discovered features."
        ),
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Category 11 — Synthetic Data & Augmentation (6)  [DQ-SYN-001..006]
# Grounded: build_synthesis_spec, llm_synthesize_rows, validate_synthetic (offline
# 3-pillar gate). synthesize_tabular (SDV) + augment_imbalance (SMOTE) = GAP T-3.
# ═══════════════════════════════════════════════════════════════════════════════
USE_CASES += [
    UseCase(
        id="DQ-SYN-001",
        category="synthetic_augmentation",
        title="Build a synthesis spec from a requirement",
        user_story=(  # nosemgrep
            "As a data scientist I want a synthesis spec built from a natural-language requirement "
            "so that synthetic generation is grounded in the real schema."
        ),
        chat_prompt="build a synthesis spec for 500 extra bogie rows that preserve the failure-mode distribution",
        expected_capability="",
        expected_tools=("build_synthesis_spec",),
        expected_events=("dataset", "agents"),
        expected_panel_updates=("data", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep
            "GIVEN a requirement and source schema WHEN a spec is built THEN 'build_synthesis_spec' "
            "is invoked AND a synthesis spec is returned AND the agent-activity panel records the tool."
        ),
    ),
    UseCase(
        id="DQ-SYN-002",
        category="synthetic_augmentation",
        title="Generate synthetic rows with an LLM",
        user_story=(  # nosemgrep
            "As a data scientist I want LLM-generated synthetic rows so that I can augment a small "
            "table without the heavy SDV dependency."
        ),
        chat_prompt="generate 50 synthetic bogie rows with the LLM that match the schema of data/bogie.csv",
        expected_capability="",
        expected_tools=("llm_synthesize_rows",),
        expected_events=("dataset", "agents"),
        expected_panel_updates=("data", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep
            "GIVEN a source schema WHEN LLM rows are requested THEN 'llm_synthesize_rows' is invoked "
            "AND schema-conformant rows are returned AND the Data tab reflects the augmented dataset."
        ),
    ),
    UseCase(
        id="DQ-SYN-003",
        category="synthetic_augmentation",
        title="Validate synthetic data (3-pillar gate)",
        user_story=(  # nosemgrep
            "As a data scientist I want synthetic data validated on fidelity, utility/detection, and "
            "privacy so that a memorized copy is rejected even at perfect fidelity."
        ),
        chat_prompt="validate the synthetic bogie dataset against the real one on fidelity, utility, and privacy",
        expected_capability="",
        expected_tools=("validate_synthetic",),
        expected_events=("dataset", "agents"),
        expected_panel_updates=("data", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep
            "GIVEN real+synthetic datasets WHEN validated THEN 'validate_synthetic' runs the 3-pillar "
            "AND-gate AND an exact-copy fails on privacy AND the Data tab reflects the verdict."
        ),
    ),
    UseCase(
        id="DQ-SYN-004",
        category="synthetic_augmentation",
        title="Generate synthetic tabular data with SDV",
        user_story=(  # nosemgrep
            "As a data scientist I want SDV-generated synthetic tabular data so that I can produce a "
            "high-fidelity synthetic copy of a table."
        ),
        chat_prompt="synthesize 1000 rows of tabular data with SDV matching the distribution of data/bogie.csv",
        expected_capability="",
        expected_tools=("synthesize_tabular",),
        expected_events=("dataset", "agents"),
        expected_panel_updates=("data", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep
            "GIVEN a source dataset WHEN SDV synthesis is requested THEN 'synthesize_tabular' is "
            "invoked AND (with the data extra) synthetic rows are produced; without the extra it "
            "returns a clear capability-unavailable error (graceful degrade, Phase-7 gate T-3)."
        ),
        # Phase 7 (gap T-3): synthesize_tabular is a real @tool with lazy sdv import + a
        # clear extra-required fail-soft; supported end-to-end (extra installed) or
        # gracefully degraded (extra absent). offline stays False — SDV needs the 'data' extra.
        offline=False,
    ),
    UseCase(
        id="DQ-SYN-005",
        category="synthetic_augmentation",
        title="Balance classes with SMOTE augmentation",
        user_story=(  # nosemgrep
            "As a data scientist I want SMOTE augmentation so that the minority class is synthetically "
            "oversampled for a balanced training set."
        ),
        chat_prompt="augment data/bogie.csv with SMOTE so the failure class is balanced against the majority",
        expected_capability="",
        expected_tools=("augment_imbalance",),
        expected_events=("dataset", "agents"),
        expected_panel_updates=("data", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep
            "GIVEN an imbalanced dataset WHEN SMOTE augmentation is requested THEN 'augment_imbalance' "
            "is invoked AND (with the data extra) balances the classes; without the extra it returns "
            "a clear capability-unavailable error (graceful degrade, Phase-7 gate T-3)."
        ),
        # Phase 7 (gap T-3): augment_imbalance is a real @tool with lazy imblearn import +
        # a clear extra-required fail-soft; supported end-to-end (extra installed) or
        # gracefully degraded (extra absent). offline stays False — SMOTE needs the extra.
        offline=False,
    ),
    UseCase(
        id="DQ-SYN-006",
        category="synthetic_augmentation",
        title="Requirement-to-synthesis end-to-end (spec->generate->validate)",
        user_story=(  # nosemgrep
            "As a data scientist I want a full requirement->generate->validate synthesis pipeline so "
            "that augmentation is produced and quality-gated in one flow."
        ),
        chat_prompt="from the requirement 'balance the failure class', synthesize rows and validate them end to end",
        expected_capability="",
        expected_tools=("build_synthesis_spec", "synthesize_tabular", "validate_synthetic"),
        expected_events=("dataset", "agents"),
        expected_panel_updates=("data", "dashboard.agent_activity"),
        acceptance=(  # nosemgrep
            "GIVEN a requirement WHEN the synthesis pipeline runs THEN spec->generate->validate "
            "executes AND the accepted synthetic set becomes active; build_synthesis_spec + "
            "validate_synthetic run offline, and the generate step degrades gracefully without the "
            "'data' extra (clear capability-unavailable at exactly that seam)."
        ),
        # Phase 7 (gap T-3): the DQ-SYN-006 pipeline composes real @tools —
        # build_synthesis_spec + validate_synthetic (offline) and synthesize_tabular
        # (data-extra gated with a clear fail-soft). End-to-end supported with the extra;
        # gracefully degraded without it. offline stays False — generation needs the extra.
        offline=False,
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Category 12 — DQ Pipeline Orchestration & Monitoring (8)  [DQ-ORCH-001..008]
# Grounded: data_quality_flow, analyze_drift, mlops_kpi_scorecard, configure_alert,
# evaluate_retrain_trigger, setup_model_monitor, run_clarify.
# ═══════════════════════════════════════════════════════════════════════════════
USE_CASES += [
    UseCase(
        id="DQ-ORCH-001",
        category="dq_pipeline_orchestration",
        title="Run the DQ flow gate as a pipeline",
        user_story=(  # nosemgrep
            "As a data steward I want the DQ flow run as an orchestrated gate so that a dataset is "
            "certified (scan+expectations+score) before it enters modeling."
        ),
        chat_prompt="orchestrate the data quality flow on data/bogie.csv and give me a certified go/no-go gate",
        expected_capability="data_quality_flow",
        expected_tools=("run_ml_capability",),
        expected_events=("job", "dataset", "agents"),
        expected_panel_updates=("pipeline", "data", "dashboard.active_jobs"),
        acceptance=(  # nosemgrep
            "GIVEN a dataset WHEN data_quality_flow is orchestrated THEN run_ml_capability routes to "
            "'data_quality_flow' AND a job event fires AND the Pipeline/Active-Jobs panels show the "
            "gate result."
        ),
    ),
    UseCase(
        id="DQ-ORCH-002",
        category="dq_pipeline_orchestration",
        title="Produce a DQ scorecard / KPIs",
        user_story=(  # nosemgrep
            "As a data steward I want a DQ scorecard with KPIs so that quality trends are summarized "
            "for stakeholders."
        ),
        chat_prompt="produce a data-quality KPI scorecard from the latest DQ run events",
        expected_capability="",
        expected_tools=("mlops_kpi_scorecard",),
        expected_events=("agents",),
        expected_panel_updates=("mlops", "dashboard.observability"),
        acceptance=(  # nosemgrep
            "GIVEN DQ/pipeline events WHEN a scorecard is requested THEN 'mlops_kpi_scorecard' is "
            "invoked AND KPI values are returned AND the MLOps/Observability panel reflects them."
        ),
    ),
    UseCase(
        id="DQ-ORCH-003",
        category="dq_pipeline_orchestration",
        title="Monitor data drift on a deployed model",
        user_story=(  # nosemgrep
            "As an ML engineer I want data drift monitored on a deployed model so that input shift "
            "is detected in production."
        ),
        chat_prompt="analyze data drift for the deployed bogie endpoint against its training baseline",
        expected_capability="",
        expected_tools=("analyze_drift",),
        expected_events=("model", "agents"),
        expected_panel_updates=("mlops", "dashboard.observability"),
        acceptance=(  # nosemgrep
            "GIVEN a deployed model WHEN drift is analyzed THEN 'analyze_drift' is invoked AND a "
            "drift verdict is returned AND the Observability panel reflects it."
        ),
    ),
    UseCase(
        id="DQ-ORCH-004",
        category="dq_pipeline_orchestration",
        title="Configure a data-drift alarm",
        user_story=(  # nosemgrep
            "As an ML engineer I want a data-drift CloudWatch alarm configured so that I am paged "
            "when drift crosses a threshold."
        ),
        chat_prompt="configure an alarm that fires when the bogie model's data-drift metric exceeds 0.3",
        expected_capability="",
        expected_tools=("configure_alert",),
        expected_events=("model", "agents"),
        expected_panel_updates=("mlops", "dashboard.observability"),
        acceptance=(  # nosemgrep
            "GIVEN a metric and threshold WHEN an alarm is configured THEN 'configure_alert' is "
            "invoked AND the alarm config is returned AND the Observability panel reflects it."
        ),
    ),
    UseCase(
        id="DQ-ORCH-005",
        category="dq_pipeline_orchestration",
        title="Evaluate a retrain trigger on drift",
        user_story=(  # nosemgrep
            "As an ML engineer I want a retrain trigger evaluated on drift so that retraining fires "
            "only when data drift justifies it."
        ),
        chat_prompt="evaluate whether the bogie model should be retrained given the current data drift",
        expected_capability="",
        expected_tools=("evaluate_retrain_trigger",),
        expected_events=("model", "agents"),
        expected_panel_updates=("mlops", "dashboard.observability"),
        acceptance=(  # nosemgrep
            "GIVEN drift metrics WHEN a retrain trigger is evaluated THEN 'evaluate_retrain_trigger' "
            "is invoked AND a retrain/no-retrain decision is returned AND the MLOps panel reflects it."
        ),
    ),
    UseCase(
        id="DQ-ORCH-006",
        category="dq_pipeline_orchestration",
        title="Set up a model monitor with a DQ baseline",
        user_story=(  # nosemgrep
            "As an ML engineer I want a model monitor set up with a data-quality baseline so that "
            "production inputs are continuously checked against it."
        ),
        chat_prompt="set up a model monitor for the bogie endpoint using a data-quality baseline",
        expected_capability="",
        expected_tools=("setup_model_monitor",),
        expected_events=("model", "agents"),
        expected_panel_updates=("mlops", "dashboard.observability"),
        acceptance=(  # nosemgrep
            "GIVEN a deployed model WHEN a monitor is set up THEN 'setup_model_monitor' is invoked "
            "AND a monitoring schedule with a DQ baseline is returned AND the MLOps panel reflects it."
        ),
    ),
    UseCase(
        id="DQ-ORCH-007",
        category="dq_pipeline_orchestration",
        title="Run a bias/quality Clarify check",
        user_story=(  # nosemgrep
            "As a governance officer I want a Clarify bias/quality check run so that data-quality "
            "and bias metrics are produced for a model."
        ),
        chat_prompt="run a clarify bias and data-quality check on the bogie training dataset and model",
        expected_capability="",
        expected_tools=("run_clarify",),
        expected_events=("model", "agents"),
        expected_panel_updates=("mlops", "dashboard.observability"),
        acceptance=(  # nosemgrep
            "GIVEN a dataset and model WHEN Clarify runs THEN 'run_clarify' is invoked AND bias/"
            "quality metrics are returned AND the MLOps/Observability panel reflects them."
        ),
    ),
    UseCase(
        id="DQ-ORCH-008",
        category="dq_pipeline_orchestration",
        title="Continuous DQ monitoring dashboard",
        user_story=(  # nosemgrep
            "As a data steward I want a continuous DQ monitoring surface so that scorecard + drift + "
            "alarms are visible in one operations console."
        ),
        chat_prompt="show me a continuous data-quality monitoring summary with the latest scorecard and drift status",
        expected_capability="",
        expected_tools=("mlops_kpi_scorecard", "analyze_drift"),
        expected_events=("model", "agents"),
        expected_panel_updates=("mlops", "dashboard.observability"),
        acceptance=(  # nosemgrep
            "GIVEN prior DQ runs WHEN a monitoring summary is requested THEN scorecard+drift tools "
            "are invoked AND a combined DQ status is returned AND the Observability panel reflects it."
        ),
    ),
]


# ── END OF REGISTRY (append new category blocks above this line) ──
