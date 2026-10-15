"""Model Card Generator — produces standard ML model cards.

Generates markdown model cards following community standards:
  - Model Details (name, version, type)
  - Training Data description
  - Performance Metrics
  - Intended Use
  - Limitations
  - Ethical Considerations
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from ..evaluation.store import ExperimentResult


def generate_model_card(
    result: ExperimentResult,
    spec: dict[str, Any],
) -> str:
    """Generate a standard model card in markdown format.

    Args:
        result: The ExperimentResult for the model to document.
        spec: Spec dict with project_id, task_type, target, etc.

    Returns:
        Markdown-formatted model card string.
    """
    project_id = spec.get("project_id", "unknown_project")
    task_type = spec.get("task_type", "classification")
    target = spec.get("target", "unknown")
    primary_metric = spec.get("primary_metric", result.primary_metric)
    threshold = spec.get("threshold", "N/A")
    baseline_score = spec.get("baseline_score", "N/A")

    # Format algorithm name for display
    algo_display = result.algorithm.replace("_", " ").title()

    sections: list[str] = []

    # ── Title ────────────────────────────────────────────────────────────
    sections.append(f"# Model Card: {algo_display}")
    sections.append("")
    sections.append(f"**Project**: {project_id}")
    sections.append(f"**Generated**: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}")
    sections.append("")

    # ── Model Details ────────────────────────────────────────────────────
    sections.append("## Model Details")
    sections.append("")
    sections.append(f"- **Model Name**: {algo_display}")
    sections.append(f"- **Model Type**: {task_type}")
    sections.append(f"- **Algorithm**: {result.algorithm}")
    sections.append("- **Version**: 1.0 (initial training)")
    sections.append(f"- **Run ID**: {result.run_id}")
    sections.append(f"- **Training Duration**: {result.duration_seconds:.1f} seconds")
    if result.hyperparameters:
        params_str = ", ".join(f"{k}={v}" for k, v in result.hyperparameters.items())
        sections.append(f"- **Hyperparameters**: {params_str}")
    sections.append("")

    # ── Training Data ────────────────────────────────────────────────────
    sections.append("## Training Data")
    sections.append("")
    sections.append(f"- **Dataset**: {project_id}")
    sections.append(f"- **Target Variable**: {target}")
    sections.append(f"- **Task Type**: {task_type}")
    if result.artifact_path:
        sections.append(f"- **Artifact Path**: {result.artifact_path}")
    sections.append("")

    # ── Performance Metrics ──────────────────────────────────────────────
    sections.append("## Performance Metrics")
    sections.append("")
    sections.append("| Metric | Score |")
    sections.append("|--------|-------|")
    sections.append(f"| **{primary_metric}** (primary) | **{result.primary_score:.4f}** |")

    for metric_name, score in sorted(result.metrics.items()):
        if metric_name != primary_metric:
            sections.append(f"| {metric_name} | {score:.4f} |")

    sections.append("")
    sections.append(f"- **Threshold**: {threshold}")
    sections.append(f"- **Baseline Score**: {baseline_score}")
    if isinstance(threshold, (int, float)) and isinstance(baseline_score, (int, float)):
        delta = result.primary_score - baseline_score
        sections.append(f"- **Delta vs Baseline**: +{delta:.4f}")
    sections.append("")

    # ── Intended Use ─────────────────────────────────────────────────────
    sections.append("## Intended Use")
    sections.append("")
    sections.append(
        f"This model is intended for **{task_type}** tasks, specifically predicting "
        f"the `{target}` variable for the {project_id} use case."
    )
    sections.append("")
    sections.append("**Primary Use Cases**:")
    sections.append(f"- Automated {task_type} of {target}")
    sections.append("- Batch prediction on new incoming data")
    sections.append("- Integration into ML pipeline for continuous prediction")
    sections.append("")

    # ── Limitations ──────────────────────────────────────────────────────
    sections.append("## Limitations")
    sections.append("")
    sections.append("**Known Limitations**:")
    sections.append("")
    sections.append(
        "- Model trained on a single train/test split; performance may vary "
        "with different data distributions."
    )
    sections.append(
        "- No subgroup fairness analysis performed in this version; "
        "demographic parity not guaranteed."
    )
    sections.append(
        "- Model assumes input data follows same distribution as training data; "
        "data drift may degrade performance."
    )
    sections.append(
        "- Inference latency and resource requirements not benchmarked "
        "for production deployment."
    )
    if result.primary_score < 0.90:
        sections.append(
            f"- Primary metric ({result.primary_score:.4f}) indicates room for "
            f"improvement; not suitable for safety-critical applications without "
            f"additional validation."
        )
    sections.append("")

    # ── Ethical Considerations ───────────────────────────────────────────
    sections.append("## Ethical Considerations")
    sections.append("")
    sections.append(
        "- Ensure input features do not encode protected attributes "
        "(race, gender, age) unless explicitly justified."
    )
    sections.append(
        "- Monitor for disparate impact across subgroups in production."
    )
    sections.append(
        "- Human oversight recommended for high-stakes decisions based on model output."
    )
    sections.append(
        "- Regular model retraining recommended to prevent staleness and drift."
    )
    sections.append("")

    # ── Maintenance ──────────────────────────────────────────────────────
    sections.append("## Maintenance")
    sections.append("")
    sections.append(f"- **Last Trained**: {result.timestamp}")
    sections.append("- **Retraining Cadence**: TBD based on data drift monitoring")
    sections.append("- **Owner**: ML Engineering Team")
    sections.append("")

    return "\n".join(sections)
