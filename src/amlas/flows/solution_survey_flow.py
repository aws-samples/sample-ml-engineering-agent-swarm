"""``solution_survey_flow`` — the Stage-3 solution-survey sequence (roadmap M6).

The invariant Stage-3 sequence (doc 06 §5.3 ``solution_survey_flow``): **list JumpStart / Bedrock
FM candidates (ModelHub) → survey the AWS managed-service options → apply the research gate
against the reproducibility evidence → build a ranked candidate shortlist**. All CHEAP_READ
(read/reasoning only; the ModelHub list is a catalog read, the rest is pure reasoning), so the
public entrypoint runs without an approval prompt. Composes the Stage-3 tools
(``tools/research_neutral.py``) over the ModelHubBackend port — turns a framed ``problem_card``
into a ranked solution shortlist the prototype stage evaluates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..tools import research_neutral as R
from ..tools.spec_heuristics import GENERATION_TASKS, resolve_task_modality
from ._util import raw

__all__ = ["SolutionSurveyFlowResult", "run_solution_survey_flow"]


@dataclass
class SolutionSurveyFlowResult:
    """The invariant output contract of one ``solution_survey_flow`` run."""

    status: str  # "completed" | "failed"
    problem_type: str = ""
    modality: str = ""
    hub_models: list[dict[str, Any]] = field(default_factory=list)
    managed_options: list[dict[str, Any]] = field(default_factory=list)
    gated_candidates: list[dict[str, Any]] = field(default_factory=list)
    shortlist: list[dict[str, Any]] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)


def run_solution_survey_flow(
    *,
    problem_type: str,
    modality: str | None = None,
    task: str | None = None,
    query: str | None = None,
    reproducibility_findings: list[dict[str, Any]] | None = None,
    top_n: int = 3,
) -> SolutionSurveyFlowResult:
    """Run the Stage-3 solution-survey sequence; return the ranked candidate shortlist.

    Fixed steps (all CHEAP_READ — ModelHub catalog read + pure reasoning):
      1. **list** JumpStart / Bedrock FM model cards (T-HUB-01, ModelHubBackend);
      2. **survey** the AWS managed-service options for the problem/modality (T-PR-42);
      3. **gate** each candidate against the reproducibility evidence (T-PR-44);
      4. **shortlist** the gated candidates, ranked by fit + research-gate + cost (T-PR-46).
    """
    try:
        modality = resolve_task_modality(problem_type, modality)
    except ValueError as error:
        return SolutionSurveyFlowResult(
            status="failed", problem_type=problem_type, modality=modality or "",
            detail={"error": str(error)},
        )
    findings = reproducibility_findings or []

    # ── 1. list JumpStart / Bedrock FM model cards (ModelHubBackend) ────────────
    generation = problem_type.strip().lower().replace("-", "_") in GENERATION_TASKS
    hub_task = task or ("text-generation" if generation else None)
    hub = raw(R.list_jumpstart_models)(task=hub_task, query=query)
    hub_models = hub.get("models", [])
    if generation:
        hub_models = [
            model for model in hub_models
            if str(model.get("task", "")).lower().replace("_", "-") == "text-generation"
        ]

    # ── 2. survey the AWS managed-service options ───────────────────────────────
    survey = raw(R.survey_managed_services)(problem_type=problem_type, modality=modality)
    managed_options = survey.get("managed_options", [])

    # ── 3. assemble candidates (managed options + hub models) + research gate ───
    candidates: list[dict[str, Any]] = []
    for opt in managed_options:
        candidates.append({"approach": opt.get("service", ""), "fit": opt.get("fit", "medium"),
                           "why": opt.get("why", ""), "kind": "managed"})
    for m in hub_models:
        candidates.append({"approach": m.get("model_id", m.get("task", "")),
                           "fit": "high" if m.get("fine_tunable") else "medium",
                           "why": m.get("description", ""), "kind": "hub"})

    gated: list[dict[str, Any]] = []
    for c in candidates:
        gate = raw(R.apply_research_gate)(candidate=c, reproducibility_findings=findings)
        gated.append({**c, "research_gate_passed": gate.get("passed", False),
                      "gate_rationale": gate.get("rationale", ""),
                      "evidence_count": gate.get("evidence_count", 0)})

    # ── 4. build the ranked candidate shortlist ─────────────────────────────────
    short = raw(R.build_candidate_shortlist)(candidates=gated, top_n=top_n)

    return SolutionSurveyFlowResult(
        status="completed", problem_type=problem_type, modality=modality,
        hub_models=hub_models, managed_options=managed_options, gated_candidates=gated,
        shortlist=short.get("shortlist", []),
        detail={"recommendation": survey.get("recommendation", ""),
                "approaches": short.get("approaches", []),
                "hub_adapter": hub.get("adapter", ""),
                "candidate_count": len(candidates)},
    )
