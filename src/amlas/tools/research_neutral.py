"""Stage 0–4 front-end tools — research / business-goal / framing / survey / labeling+query (M6).

The thin front of the lifecycle (RECON §10.3) the M6 milestone fills (doc 06 Stage 0–4 +
roadmap M6). Most Stage 0–2 tools are **pure-reasoning** (doc 06 §2.7 — no managed backend;
they structure inputs into problem cards / KPI specs / shortlists); the research-corpus tools
use the RetrievalBackend; solution-survey uses the ModelHubBackend; labeling uses the
LabelingBackend; SQL uses the QueryBackend.

Tool families (doc 06; EXACT T-IDs):
  Stage 0 research-intelligence: ``scan_research`` (evolves ``web_retrieve`` + research_search,
    T-PR-01..04), ``triage_reproducibility`` (T-PR-05), ``generate_research_digest`` (T-PR-11),
    ``schedule_research_scan`` (T-ORCH-04).
  Stage 1 business-goal: ``capture_business_problem`` (T-PR-14), ``define_business_kpi`` (T-PR-16),
    ``assess_ml_necessity`` (T-PR-17), ``business_go_no_go`` (T-PR-26).
  Stage 2 framing: ``build_problem_card`` (T-PR-27), ``select_problem_type`` (T-PR-31),
    ``define_success_metrics`` (T-PR-35), ``check_stage_exit`` (T-PR-67, parameterized).
  Stage 3 solution-survey: ``list_jumpstart_models`` (T-HUB-01), ``survey_managed_services``
    (T-PR-42), ``apply_research_gate`` (T-PR-44), ``build_candidate_shortlist`` (T-PR-46).
  Stage 4 labeling/query: ``create_labeling_job`` (T-LBL-01), ``get_labeling_progress`` (T-LBL-02),
    ``run_sql`` (T-QRY-01).
"""

from __future__ import annotations

from typing import Any

from strands import tool

from amlas.config._env_compat import env_with_compat

from ..backends.ports import LabelingBackend, ModelHubBackend, QueryBackend
from ..backends.registry import REGISTRY
from ..errors import tool_error_boundary
from .spec_heuristics import GENERATION_TASKS, resolve_task_modality

__all__ = [
    "apply_research_gate",
    "assess_ml_necessity",
    "build_candidate_shortlist",
    # Stage 2 — framing
    "build_problem_card",
    "business_go_no_go",
    # Stage 1 — business goal
    "capture_business_problem",
    "check_stage_exit",
    # Stage 4 — labeling + query
    "create_labeling_job",
    "define_business_kpi",
    "define_success_metrics",
    "generate_research_digest",
    "get_labeling_progress",
    # Stage 3 — solution survey
    "list_jumpstart_models",
    "run_sql",
    # Stage 0 — research intelligence
    "scan_research",
    "schedule_research_scan",
    "select_problem_type",
    "survey_managed_services",
    "triage_reproducibility",
]


# ══════════════════════════════════════════════════════════════════════════════
# Stage 0 — research intelligence (T-PR-01..11, T-ORCH-04)
# ══════════════════════════════════════════════════════════════════════════════
@tool
@tool_error_boundary(source="tools.research_neutral")
def scan_research(query: str, max_results: int = 5, index_name: str = "research") -> dict[str, Any]:
    """Scan external + indexed research for ``query`` (T-PR-01..04; evolves ``web_retrieve``).

    Fans out to the live research aggregator (arxiv/Semantic Scholar/Tavily when configured)
    AND the indexed research corpus (RetrievalBackend), merging both into one ranked findings
    list. Offline-safe: if the live aggregator yields nothing, the corpus retrieval still works.
    Returns ``{findings, source_count, sources}``.
    """
    findings: list[dict[str, Any]] = []
    sources: list[str] = []

    # 1) live research aggregator (the real web_retrieve evolution) — best-effort.
    # An explicit offline flag skips the live web fan-out; indexed retrieval below
    # remains available.
    _explicit_offline = str(
        env_with_compat("AMLAS_OFFLINE") or env_with_compat("AMLAS_BACKENDS_OFFLINE") or ""
    ).strip().lower() in ("1", "true", "yes")
    if not _explicit_offline:
        try:
            from .research.strands_tools import research_search

            raw = getattr(research_search, "_tool_func", research_search)(query, max_results)
            for r in raw.get("results", []):
                findings.append({"title": r.get("title", ""), "url": r.get("url", ""),
                                 "snippet": r.get("snippet", ""), "origin": "web"})
            sources.extend(raw.get("queried_sources", []))
        except Exception:  # nosec B110
            pass

    # 2) indexed research corpus (RetrievalBackend) — always available offline
    try:
        from .retrieval_neutral import query_research_corpus

        corpus = getattr(query_research_corpus, "_tool_func", query_research_corpus)(
            index_name=index_name, query=query, top_k=max_results)
        for r in corpus.get("results", []):
            findings.append({"title": r.get("source", ""), "snippet": r.get("text", "")[:240],
                             "score": r.get("score", 0.0), "origin": "corpus"})
        if corpus.get("results"):
            sources.append("research_corpus")
    except Exception:  # nosec B110
        pass

    return {"query": query, "findings": findings[: max_results * 2], "source_count": len(set(sources)),
            "sources": sorted(set(sources))}


@tool
@tool_error_boundary(source="tools.research_neutral")
def triage_reproducibility(findings: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Triage research findings by reproducibility signals (T-PR-05, pure reasoning).

    Scores each finding on cheap reproducibility heuristics — presence of code/benchmark/
    dataset keywords + a source-mix bonus — into tiers ``high|medium|low``. Returns the triaged
    findings + a recommended backlog of the high-reproducibility items to deep-dive.
    """
    findings = findings or []
    code_kw = ("github", "code", "implementation", "open-source", "open source", "repo")
    bench_kw = ("benchmark", "dataset", "sota", "leaderboard", "results", "accuracy", "f1")
    triaged: list[dict[str, Any]] = []
    for f in findings:
        text = f"{f.get('title','')} {f.get('snippet','')} {f.get('url','')}".lower()
        score = 0
        if any(k in text for k in code_kw):
            score += 2
        if any(k in text for k in bench_kw):
            score += 1
        if f.get("origin") == "corpus":
            score += 1  # curated corpus is more trustworthy
        tier = "high" if score >= 3 else ("medium" if score >= 1 else "low")
        triaged.append({**f, "reproducibility_score": score, "reproducibility_tier": tier})
    backlog = [t for t in triaged if t["reproducibility_tier"] == "high"]
    return {"triaged": triaged, "backlog": backlog, "high_count": len(backlog),
            "total": len(triaged)}


@tool
@tool_error_boundary(source="tools.research_neutral")
def generate_research_digest(findings: list[dict[str, Any]] | None = None, theme: str = "") -> dict[str, Any]:
    """Compile a structured research digest from findings (T-PR-11, pure reasoning).

    Groups findings by origin, surfaces the top-ranked items, and emits a short digest the
    research_curator persists to ``memory-bank/research/``. Returns ``{digest, item_count, themes}``.
    """
    findings = findings or []
    by_origin: dict[str, int] = {}
    for f in findings:
        by_origin[f.get("origin", "web")] = by_origin.get(f.get("origin", "web"), 0) + 1
    top = sorted(findings, key=lambda f: f.get("score", 0.0) or len(f.get("snippet", "")), reverse=True)[:5]
    digest_lines = [f"# Research digest{f': {theme}' if theme else ''}", ""]
    digest_lines.append(f"- {len(findings)} findings across {len(by_origin)} origins ({by_origin}).")
    for i, t in enumerate(top, 1):
        digest_lines.append(f"{i}. {t.get('title','(untitled)')} — {t.get('snippet','')[:120]}")
    return {"digest": "\n".join(digest_lines), "item_count": len(findings),
            "themes": [theme] if theme else [], "by_origin": by_origin}


@tool
@tool_error_boundary(source="tools.research_neutral")
def schedule_research_scan(query: str, cadence: str = "weekly") -> dict[str, Any]:
    """Schedule a recurring research scan (T-ORCH-04, pure reasoning descriptor).

    Produces a schedule descriptor the Research Lead registers (EventBridge in prod). Offline it
    returns the descriptor only (no AWS). Returns ``{scan_id, query, cadence, enabled}``.
    """
    cadence = (cadence or "weekly").strip().lower()
    valid = {"hourly", "daily", "weekly", "monthly"}
    if cadence not in valid:
        cadence = "weekly"
    scan_id = f"research-scan-{abs(hash((query, cadence))) % 100000}"
    return {"scan_id": scan_id, "query": query, "cadence": cadence, "enabled": True,
            "note": "descriptor only offline; EventBridge rule is the managed scheduler"}


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1 — business goal (T-PR-14..26)
# ══════════════════════════════════════════════════════════════════════════════
@tool
@tool_error_boundary(source="tools.research_neutral")
def capture_business_problem(statement: str, stakeholders: list[str] | None = None,
                             current_process: str = "", pain_points: list[str] | None = None) -> dict[str, Any]:
    """Capture a business problem into a structured record (T-PR-14, pure reasoning).

    Distills a free-text business problem into ``{statement, stakeholders, current_process,
    pain_points, clarity_score}`` — the Stage-1 starting artifact. ``clarity_score`` flags whether
    the statement is specific enough to proceed (more detail → higher clarity).
    """
    stakeholders = stakeholders or []
    pain_points = pain_points or []
    clarity = 0.0
    clarity += 0.3 if len(statement.split()) >= 8 else 0.1
    clarity += 0.3 if stakeholders else 0.0
    clarity += 0.2 if current_process else 0.0
    clarity += 0.2 if pain_points else 0.0
    return {"statement": statement, "stakeholders": stakeholders, "current_process": current_process,
            "pain_points": pain_points, "clarity_score": round(min(clarity, 1.0), 2),
            "ready": clarity >= 0.5}


@tool
@tool_error_boundary(source="tools.research_neutral")
def define_business_kpi(objective: str, metric_name: str, baseline: float | None = None,
                        target: float | None = None, unit: str = "") -> dict[str, Any]:
    """Define a measurable business KPI for the objective (T-PR-16, pure reasoning).

    Builds a ``{objective, kpi, baseline, target, unit, improvement_pct, measurable}`` spec. The
    KPI is "measurable" only when both a baseline and target are supplied (so success is verifiable).
    """
    improvement = None
    if isinstance(baseline, (int, float)) and isinstance(target, (int, float)) and baseline:
        improvement = round(100.0 * (target - baseline) / abs(baseline), 1)
    return {"objective": objective, "kpi": metric_name, "baseline": baseline, "target": target,
            "unit": unit, "improvement_pct": improvement,
            "measurable": baseline is not None and target is not None}


@tool
@tool_error_boundary(source="tools.research_neutral")
def assess_ml_necessity(problem: str, has_pattern: bool = True, has_data: bool = True,
                        rules_sufficient: bool = False, tolerance_for_error: str = "medium") -> dict[str, Any]:
    """Assess whether ML is necessary vs a non-ML alternative (T-PR-17, pure reasoning).

    Applies the standard ML-necessity rubric: ML is warranted when there's a learnable pattern +
    data + rules are NOT sufficient. Returns ``{ml_recommended, rationale, alternative}`` so a
    problem solvable with deterministic rules is steered away from ML (avoiding over-engineering).
    """
    reasons: list[str] = []
    if not has_pattern:
        reasons.append("no clear learnable pattern")
    if not has_data:
        reasons.append("insufficient data")
    if rules_sufficient:
        reasons.append("deterministic rules are sufficient")
    ml_recommended = has_pattern and has_data and not rules_sufficient
    alternative = "" if ml_recommended else (
        "rule-based system / heuristics" if rules_sufficient else "collect data + re-assess")
    rationale = ("ML warranted: learnable pattern + data, rules insufficient"
                 if ml_recommended else f"ML not recommended: {'; '.join(reasons)}")
    return {"problem": problem, "ml_recommended": ml_recommended, "rationale": rationale,
            "alternative": alternative, "tolerance_for_error": tolerance_for_error}


@tool
@tool_error_boundary(source="tools.research_neutral")
def business_go_no_go(clarity_score: float = 0.0, kpi_measurable: bool = False,
                      ml_recommended: bool = False, value_estimate: float = 0.0) -> dict[str, Any]:
    """Stage-1 business go/no-go decision (T-PR-26, pure reasoning gate).

    Combines the Stage-1 signals into a go/no-go: proceed only when the problem is clear, the KPI
    is measurable, and ML is recommended. Returns ``{decision, blockers, score}``.
    """
    blockers: list[str] = []
    if clarity_score < 0.5:
        blockers.append("problem statement not clear enough")
    if not kpi_measurable:
        blockers.append("KPI not measurable (need baseline + target)")
    if not ml_recommended:
        blockers.append("ML not recommended for this problem")
    decision = "go" if not blockers else "no-go"
    score = round((clarity_score + (0.3 if kpi_measurable else 0) + (0.3 if ml_recommended else 0)) / 1.6, 2)
    return {"decision": decision, "blockers": blockers, "score": min(score, 1.0),
            "value_estimate": value_estimate}


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2 — framing (T-PR-27..41, T-PR-67)
# ══════════════════════════════════════════════════════════════════════════════
_PROBLEM_TYPES = {
    "binary_classification", "multiclass_classification", "regression",
    "time_series_forecasting", "ranking", "clustering", "anomaly_detection",
    "recommendation", "nlp_classification", "object_detection", "generation",
}


@tool
@tool_error_boundary(source="tools.research_neutral")
def select_problem_type(target_description: str, output_kind: str = "", num_classes: int = 0) -> dict[str, Any]:
    """Select the ML problem type from the target description (T-PR-31, pure reasoning).

    Maps the target/output to one of the canonical problem types. Returns
    ``{problem_type, output_type, rationale}``.
    """
    desc = (target_description + " " + output_kind).lower()
    if any(k in desc for k in ("forecast", "time series", "next value", "future")):
        ptype, otype = "time_series_forecasting", "continuous"
    elif any(k in desc for k in ("anomaly", "outlier", "fraud", "defect detection")):
        ptype, otype = "anomaly_detection", "binary"
    elif any(k in desc for k in ("rank", "ordering", "relevance")):
        ptype, otype = "ranking", "ordinal"
    elif any(k in desc for k in ("price", "amount", "value", "rul", "remaining useful", "continuous", "regression")):
        ptype, otype = "regression", "continuous"
    elif num_classes > 2 or "multiclass" in desc or "categor" in desc:
        ptype, otype = "multiclass_classification", "categorical"
    else:
        ptype, otype = "binary_classification", "binary"
    return {"problem_type": ptype, "output_type": otype,
            "rationale": f"derived from target description: {target_description[:80]!r}"}


@tool
@tool_error_boundary(source="tools.research_neutral")
def define_success_metrics(problem_type: str, primary: str = "", threshold: float | None = None,
                           guardrail_metrics: list[str] | None = None) -> dict[str, Any]:
    """Define the success metric(s) for the problem type (T-PR-35, pure reasoning).

    Picks a sensible primary metric per problem type when not supplied + guardrail metrics
    (latency/fairness). Returns ``{primary_metric, threshold, guardrail_metrics, direction}``.
    """
    pt = (problem_type or "").lower()
    defaults = {
        "binary_classification": ("f1", "maximize"),
        "multiclass_classification": ("f1_macro", "maximize"),
        "regression": ("rmse", "minimize"),
        "time_series_forecasting": ("mape", "minimize"),
        "anomaly_detection": ("pr_auc", "maximize"),
        "ranking": ("ndcg", "maximize"),
    }
    metric, direction = defaults.get(pt, ("f1", "maximize"))
    primary_metric = primary or metric
    return {"primary_metric": primary_metric, "threshold": threshold, "direction": direction,
            "guardrail_metrics": guardrail_metrics or ["latency_ms", "demographic_parity"]}


@tool
@tool_error_boundary(source="tools.research_neutral")
def build_problem_card(business_problem: str, problem_type: str = "", target: str = "",
                       inputs: list[str] | None = None, output_type: str = "",
                       success_metric: str = "", kpi: dict[str, Any] | None = None,
                       constraints: list[str] | None = None) -> dict[str, Any]:
    """Assemble the canonical ``problem_card.json`` (T-PR-27, pure reasoning — the Stage-2 artifact).

    Combines the framing decisions into the single structured problem card the downstream
    Research/FE/ML leads consume. Returns ``{problem_card, complete, missing}`` where ``complete``
    flags whether all required fields are present (so framing can gate before solutioning).
    """
    card = {
        "business_problem": business_problem,
        "problem_type": problem_type,
        "target": target,
        "inputs": inputs or [],
        "output_type": output_type,
        "success_metric": success_metric,
        "kpi": kpi or {},
        "constraints": constraints or [],
    }
    required = ["business_problem", "problem_type", "target", "output_type", "success_metric"]
    missing = [k for k in required if not card.get(k)]
    return {"problem_card": card, "complete": not missing, "missing": missing}


# ── stage-exit gate metadata (the 12-stage parameterized check; T-PR-67) ──────
_STAGE_EXITS: dict[str, list[str]] = {
    "0": ["findings", "digest"],
    "1": ["statement", "kpi", "decision"],
    "2": ["problem_card", "success_metric"],
    "3": ["shortlist"],
    "4": ["dataset", "labeling_or_query"],
}


@tool
@tool_error_boundary(source="tools.research_neutral")
def check_stage_exit(stage: str, artifacts: dict[str, Any] | None = None) -> dict[str, Any]:
    """Parameterized stage-exit gate across the 12 lifecycle stages (T-PR-67, pure reasoning).

    Given a ``stage`` id and the ``artifacts`` produced so far, checks the stage's exit criteria
    (the required artifacts present + truthy). Returns ``{stage, passed, missing, present}`` so the
    orchestrator only advances a stage when its exit criteria are met.
    """
    artifacts = artifacts or {}
    required = _STAGE_EXITS.get(str(stage), [])
    present = [k for k in required if artifacts.get(k)]
    missing = [k for k in required if not artifacts.get(k)]
    # stage 4 is satisfied by EITHER a labeling job OR a query coverage artifact
    if str(stage) == "4" and ("labeling" in artifacts or "query" in artifacts) and "labeling_or_query" in missing:
        missing.remove("labeling_or_query")
        present.append("labeling_or_query")
    return {"stage": str(stage), "passed": not missing, "missing": missing, "present": present,
            "required": required}


# ══════════════════════════════════════════════════════════════════════════════
# Stage 3 — solution survey (T-HUB-01, T-PR-42..46)
# ══════════════════════════════════════════════════════════════════════════════
@tool
@tool_error_boundary(source="tools.research_neutral")
def list_jumpstart_models(task: str | None = None, query: str | None = None) -> dict[str, Any]:
    """List JumpStart / Bedrock FM model cards via :class:`ModelHubBackend` (T-HUB-01)."""
    try:
        resolved = REGISTRY.resolve(ModelHubBackend)
        models = resolved.adapter.list_models(task=task, query=query)
        return {"models": models, "model_count": len(models), "adapter": resolved.adapter_name}
    except Exception as e:
        return {"success": False, "error": f"list_jumpstart_models failed: {e}", "models": [], "model_count": 0}


@tool
@tool_error_boundary(source="tools.research_neutral")
def survey_managed_services(problem_type: str = "", modality: str | None = None) -> dict[str, Any]:
    """Survey the AWS managed-service options for a problem type/modality (T-PR-42, pure reasoning).

    Maps the framing to the candidate AWS managed services (Autopilot/JumpStart/Bedrock/etc.) +
    a custom-build option. Returns ``{managed_options, build_option, recommendation}``.
    """
    modality = resolve_task_modality(problem_type, modality)
    task = problem_type.strip().lower().replace("-", "_")
    options: list[dict[str, str]] = []
    if modality == "tabular":
        options = [
            {"service": "SageMaker Autopilot", "fit": "high", "why": "tabular AutoML baseline"},
            {"service": "JumpStart (XGBoost/TabTransformer)", "fit": "high", "why": "fine-tunable tabular models"},
        ]
    elif modality == "text" and task in GENERATION_TASKS:
        options = [
            {"service": "SageMaker custom training", "fit": "high",
             "why": "language-model SFT with a verified model-aware trainer and measured resource fit"},
            {"service": "JumpStart generative models", "fit": "medium",
             "why": "verify selected model availability and supported fine-tuning recipes"},
        ]
    elif modality == "text":
        options = [
            {"service": "Bedrock FM (Titan/Claude)", "fit": "high", "why": "managed text generation/extraction"},
            {"service": "JumpStart (BERT)", "fit": "medium", "why": "fine-tunable text classifier"},
            {"service": "Comprehend", "fit": "medium", "why": "managed NLP for standard tasks"},
        ]
    elif modality == "image":
        options = [
            {"service": "JumpStart (ResNet/ViT)", "fit": "high", "why": "fine-tunable vision models"},
            {"service": "Rekognition", "fit": "medium", "why": "managed vision for standard tasks"},
        ]
    else:
        options = [{"service": "SageMaker (custom)", "fit": "medium", "why": "general custom training"}]
    return {"managed_options": options, "build_option": {"service": "SageMaker custom training",
            "fit": "always-available", "why": "full control when managed services don't fit"},
            "recommendation": options[0]["service"] if options else "SageMaker custom",
            "problem_type": problem_type, "modality": modality}


@tool
@tool_error_boundary(source="tools.research_neutral")
def apply_research_gate(candidate: dict[str, Any] | None = None,
                        reproducibility_findings: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Gate a candidate approach against the research evidence (T-PR-44, pure reasoning).

    A candidate passes the research gate when there is at least one high-reproducibility finding
    supporting it (or it is a well-established managed service). Returns ``{passed, evidence_count,
    rationale}`` so unproven approaches are flagged before they enter the shortlist.
    """
    candidate = candidate or {}
    findings = reproducibility_findings or []
    high = [f for f in findings if f.get("reproducibility_tier") == "high"]
    established = any(s in str(candidate.get("approach", "")).lower()
                      for s in ("xgboost", "autopilot", "jumpstart", "random forest", "lightgbm"))
    passed = bool(high) or established
    rationale = ("backed by high-reproducibility research" if high else
                 "well-established approach" if established else
                 "no strong research evidence — flag for prototype validation")
    return {"passed": passed, "evidence_count": len(high), "rationale": rationale,
            "candidate": candidate.get("approach", "")}


@tool
@tool_error_boundary(source="tools.research_neutral")
def build_candidate_shortlist(candidates: list[dict[str, Any]] | None = None, top_n: int = 3) -> dict[str, Any]:
    """Build a ranked candidate shortlist for the solution survey (T-PR-46, pure reasoning).

    Scores each candidate on fit + research-gate + cost, ranks them, and returns the top-N
    shortlist the prototype stage will evaluate. Returns ``{shortlist, ranked, count}``.
    """
    candidates = candidates or []
    fit_rank = {"high": 3, "medium": 2, "low": 1, "": 1}
    scored: list[dict[str, Any]] = []
    for c in candidates:
        score = fit_rank.get(str(c.get("fit", "")).lower(), 1)
        if c.get("research_gate_passed"):
            score += 2
        if str(c.get("cost", "")).lower() in ("low", "cheap"):
            score += 1
        scored.append({**c, "shortlist_score": score})
    ranked = sorted(scored, key=lambda c: c["shortlist_score"], reverse=True)
    shortlist = ranked[: max(1, int(top_n))]
    return {"shortlist": shortlist, "ranked": ranked, "count": len(shortlist),
            "approaches": [c.get("approach", c.get("service", "")) for c in shortlist]}


# ══════════════════════════════════════════════════════════════════════════════
# Stage 4 — labeling + query (T-LBL-01/02, T-QRY-01)
# ══════════════════════════════════════════════════════════════════════════════
@tool
@tool_error_boundary(source="tools.research_neutral")
def create_labeling_job(input_manifest: str, task_type: str = "text-classification",
                        categories: list[str] | None = None, workteam: str = "private-team",
                        auto_label: bool = True) -> dict[str, Any]:
    """Create a data-labeling job via :class:`LabelingBackend` (T-LBL-01; Ground Truth managed).

    IRREVERSIBLE/spend when managed (Ground Truth incurs cost) — gated via the approval policy when
    invoked through ``data_collection_flow``. Returns the job handle id + task config.
    """
    try:
        resolved = REGISTRY.resolve(LabelingBackend)
        handle = resolved.adapter.create_job(
            input_manifest=input_manifest, task_type=task_type,
            categories=categories or ["positive", "negative"], workteam=workteam, auto_label=auto_label)
        return {"job_name": handle.id, "backend": handle.backend, "task_type": task_type,
                "categories": categories or ["positive", "negative"], "adapter": resolved.adapter_name}
    except Exception as e:
        return {"success": False, "error": f"create_labeling_job failed: {e}", "job_name": ""}


@tool
@tool_error_boundary(source="tools.research_neutral")
def get_labeling_progress(job_name: str) -> dict[str, Any]:
    """Get the progress of a labeling job via :class:`LabelingBackend` (T-LBL-02)."""
    try:
        from ..backends.types import JobHandle

        resolved = REGISTRY.resolve(LabelingBackend)
        out: dict[str, Any] = resolved.adapter.progress(JobHandle(id=job_name, backend=resolved.adapter_name))
        out.setdefault("adapter", resolved.adapter_name)
        return out
    except Exception as e:
        return {"success": False, "error": f"get_labeling_progress failed: {e}", "status": "Unknown"}


@tool
@tool_error_boundary(source="tools.research_neutral")
def run_sql(query: str, database: str | None = None) -> dict[str, Any]:
    """Run NL→SQL over the lake via :class:`QueryBackend` (T-QRY-01; DuckDB offline, Athena managed)."""
    try:
        resolved = REGISTRY.resolve(QueryBackend)
        out: dict[str, Any] = resolved.adapter.run_sql(query=query, database=database)
        out.setdefault("adapter", resolved.adapter_name)
        return out
    except Exception as e:
        return {"success": False, "error": f"run_sql failed: {e}", "rows": [], "row_count": 0}
