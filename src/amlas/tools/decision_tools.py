"""Decision Support Tools — @tool-decorated functions for agent use.

Tools:
  - get_recommendation: Analyze experiment results → produce Decision
  - generate_model_card_tool: Produce a model card for the best model
  - generate_next_steps_report: Full report with rationale and action items
"""

from __future__ import annotations

import json
from typing import Any

from strands import tool

from ..decisions.engine import DecisionEngine
from ..decisions.model_card import generate_model_card
from ..errors import tool_error_boundary
from ..evaluation.store import EvaluationStore


@tool
@tool_error_boundary(source="tools.decision_tools")
def get_recommendation(
    store_path: str,
    spec_id: str,
    spec_json: str,
) -> dict[str, Any]:
    """Analyze experiment results and produce a deployment recommendation.

    Runs the rules-based Decision Engine on all successful experiments
    for the given spec_id and returns a structured recommendation.

    Args:
        store_path: Path to the evaluation store JSON file.
        spec_id: Which spec/project to analyze.
        spec_json: JSON string of spec dict with threshold, baseline_score, etc.

    Returns:
        Dict with success flag, recommendation, confidence, rationale, next_steps, risk_level.
    """
    try:
        spec = json.loads(spec_json)
    except (json.JSONDecodeError, TypeError) as e:
        return {"success": False, "error": f"Invalid spec_json: {e}"}

    try:
        store = EvaluationStore(store_path=store_path)
    except Exception as e:
        return {"success": False, "error": f"Failed to load store: {e}"}

    results = store.get_by_spec(spec_id)
    successful = [r for r in results if r.status == "success"]

    engine = DecisionEngine()
    decision = engine.analyze(results=successful, spec=spec)

    return {
        "success": True,
        "recommendation": decision.recommendation,
        "confidence": decision.confidence,
        "rationale": decision.rationale,
        "next_steps": decision.next_steps,
        "risk_level": decision.risk_level,
    }


@tool
@tool_error_boundary(source="tools.decision_tools")
def generate_model_card_tool(
    store_path: str,
    spec_id: str,
    spec_json: str,
) -> dict[str, Any]:
    """Generate a model card for the best model in the evaluation store.

    Finds the best performing model for the given spec_id and produces
    a standard model card in markdown format.

    Args:
        store_path: Path to the evaluation store JSON file.
        spec_id: Which spec/project to get best model for.
        spec_json: JSON string of spec dict with project details.

    Returns:
        Dict with success flag and model_card (markdown string).
    """
    try:
        spec = json.loads(spec_json)
    except (json.JSONDecodeError, TypeError) as e:
        return {"success": False, "error": f"Invalid spec_json: {e}"}

    try:
        store = EvaluationStore(store_path=store_path)
    except Exception as e:
        return {"success": False, "error": f"Failed to load store: {e}"}

    results = store.get_by_spec(spec_id)
    successful = [r for r in results if r.status == "success"]

    if not successful:
        return {
            "success": False,
            "error": f"No successful results found for spec_id: {spec_id}",
        }

    # Find best result
    best = max(successful, key=lambda r: r.primary_score)

    # Generate model card
    card = generate_model_card(result=best, spec=spec)

    return {
        "success": True,
        "model_card": card,
        "best_algorithm": best.algorithm,
        "best_score": best.primary_score,
    }


@tool
@tool_error_boundary(source="tools.decision_tools")
def generate_next_steps_report(
    store_path: str,
    spec_id: str,
    spec_json: str,
) -> dict[str, Any]:
    """Generate a comprehensive next-steps report with rationale and action items.

    Combines the decision recommendation with detailed analysis of all
    experiment results to produce an actionable report.

    Args:
        store_path: Path to the evaluation store JSON file.
        spec_id: Which spec/project to analyze.
        spec_json: JSON string of spec dict with threshold, baseline_score, etc.

    Returns:
        Dict with success flag and report (markdown string).
    """
    try:
        spec = json.loads(spec_json)
    except (json.JSONDecodeError, TypeError) as e:
        return {"success": False, "error": f"Invalid spec_json: {e}"}

    try:
        store = EvaluationStore(store_path=store_path)
    except Exception as e:
        return {"success": False, "error": f"Failed to load store: {e}"}

    results = store.get_by_spec(spec_id)
    successful = [r for r in results if r.status == "success"]
    failed = [r for r in results if r.status != "success"]

    engine = DecisionEngine()
    decision = engine.analyze(results=successful, spec=spec)

    # Build comprehensive report
    lines: list[str] = []

    lines.append("# Decision Support Report")
    lines.append("")
    lines.append(f"**Project**: {spec.get('project_id', spec_id)}")
    lines.append(f"**Spec ID**: {spec_id}")
    lines.append(f"**Total Experiments**: {len(results)}")
    lines.append(f"**Successful**: {len(successful)}")
    lines.append(f"**Failed**: {len(failed)}")
    lines.append("")

    # Decision summary
    lines.append("## Recommendation")
    lines.append("")
    lines.append(f"**Decision**: {decision.recommendation.upper()}")
    lines.append(f"**Confidence**: {decision.confidence}")
    lines.append(f"**Risk Level**: {decision.risk_level}")
    lines.append("")
    lines.append("### Rationale")
    lines.append("")
    lines.append(decision.rationale)
    lines.append("")

    # Next steps
    lines.append("## Next Steps")
    lines.append("")
    for i, step in enumerate(decision.next_steps, 1):
        lines.append(f"{i}. {step}")
    lines.append("")

    # Experiment Summary
    if successful:
        lines.append("## Experiment Results Summary")
        lines.append("")
        lines.append("| Algorithm | Score | Duration (s) |")
        lines.append("|-----------|-------|--------------|")
        sorted_results = sorted(successful, key=lambda r: r.primary_score, reverse=True)
        for r in sorted_results:
            lines.append(f"| {r.algorithm} | {r.primary_score:.4f} | {r.duration_seconds:.1f} |")
        lines.append("")

        # Best model highlight
        best = sorted_results[0]
        lines.append(f"**Best Model**: {best.algorithm} ({best.primary_metric}={best.primary_score:.4f})")
        lines.append("")

    # Threshold analysis
    threshold = spec.get("threshold")
    baseline_score = spec.get("baseline_score")
    if threshold and successful:
        lines.append("## Threshold Analysis")
        lines.append("")
        lines.append(f"- **Target Threshold**: {threshold}")
        lines.append(f"- **Baseline Score**: {baseline_score}")
        best_score = max(r.primary_score for r in successful)
        lines.append(f"- **Best Score Achieved**: {best_score:.4f}")
        gap = threshold - best_score
        if gap > 0:
            lines.append(f"- **Gap to Threshold**: {gap:.4f}")
        else:
            lines.append(f"- **Margin Above Threshold**: {abs(gap):.4f}")
        lines.append("")

    report = "\n".join(lines)

    return {
        "success": True,
        "report": report,
        "recommendation": decision.recommendation,
        "confidence": decision.confidence,
    }
