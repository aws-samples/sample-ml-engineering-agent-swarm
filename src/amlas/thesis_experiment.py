"""Thesis experiment runner — research → full agent pipeline → arbiter, per dataset.

A "thesis experiment" is the headline E2E proof that the system works on Amlas's
actual problem shape:

    For each dataset case (spec + data + target):
        1. Search arxiv + Semantic Scholar (+ Tavily) for problem-relevant findings.
        2. Persist findings to <run_dir>/research_findings.json.
        3. Run the full agent pipeline (Framing → FeatureHyp → AlgoHyp → Data
           Integrity → CI sandbox FE+training → Arbiter), with the research
           findings injected as additional context for the algo-hypothesis prompt.
        4. Capture arbiter score + token usage.
    Aggregate everything into <out_dir>/thesis_report.md.

Pure-Python orchestrator — every external call (research, agent pipeline, grader)
is injected so offline validation can run without AWS or the LLM.
"""

from __future__ import annotations

import json
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

ResearchFn = Callable[..., dict[str, Any]]
PipelineFn = Callable[..., dict[str, Any]]
GraderFn = Callable[..., Any]
SpecLoader = Callable[[Path], Any]


@dataclass(frozen=True)
class DatasetCase:
    name: str
    spec_path: Path
    data_path: Path
    target_col: str
    research_query: str


@dataclass
class CaseResult:
    case_name: str
    run_dir: Path
    research_n_results: int = 0
    research_sources: list[str] = field(default_factory=list)
    arbiter_overall: float | None = None
    arbiter_baseline: float | None = None
    arbiter_delta: float | None = None
    arbiter_passes: bool = False
    arbiter_reason: str = ""
    feature_count_kept: int = 0
    winner: str = ""
    tokens: int = 0
    error: str | None = None
    elapsed_sec: float = 0.0


@dataclass
class ExperimentResult:
    per_case: list[CaseResult]
    out_dir: Path
    report_path: Path

    @property
    def all_passed(self) -> bool:
        return all(c.arbiter_passes for c in self.per_case)


def run_thesis_experiment(
    cases: list[DatasetCase],
    *,
    out_dir: Path,
    research_fn: ResearchFn,
    pipeline_fn: PipelineFn,
    grader_fn: GraderFn,
    spec_loader: SpecLoader,
    max_research_results: int = 5,
) -> ExperimentResult:
    """Run the experiment end-to-end across `cases`.

    Args:
        cases: list of `DatasetCase` describing each dataset to process.
        out_dir: top-level directory; per-case subdirs are created under it.
        research_fn: callable `(query, max_results=...) -> {results, errors, queried_sources}`.
            In production wire `tools.research.strands_tools.research_search._tool_func`.
        pipeline_fn: callable `(*, spec_path, data_path, target_col, run_dir,
            research_findings) -> {predictions, y_true, feature_manifest, winner, tokens}`.
        grader_fn: callable `(*, spec, y_true, y_pred) -> arbiter_report` with `model_dump()`.
        spec_loader: callable `(spec_path: Path) -> spec_object` consumed by `grader_fn`.
        max_research_results: cap on results per case.

    Returns:
        ExperimentResult: per-case outcomes + aggregated markdown report on disk.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    per_case: list[CaseResult] = []

    for case in cases:
        run_dir = out_dir / case.name
        run_dir.mkdir(parents=True, exist_ok=True)
        result = CaseResult(case_name=case.name, run_dir=run_dir)
        t0 = time.monotonic()

        try:
            findings = research_fn(case.research_query, max_results=max_research_results)
            (run_dir / "research_findings.json").write_text(
                json.dumps({"query": case.research_query, **findings}, indent=2)
            )
            result.research_n_results = len(findings.get("results", []))
            result.research_sources = list(findings.get("queried_sources", []))

            outcome = pipeline_fn(
                spec_path=case.spec_path,
                data_path=case.data_path,
                target_col=case.target_col,
                run_dir=run_dir,
                research_findings=findings,
            )
            preds = np.asarray(outcome["predictions"])
            y_true = np.asarray(outcome["y_true"])
            result.feature_count_kept = sum(
                1 for f in outcome.get("feature_manifest", [])
                if f.get("decision") == "keep"
            )
            result.winner = str(outcome.get("winner", ""))
            result.tokens = int(outcome.get("tokens", 0))

            spec = spec_loader(case.spec_path)
            report = grader_fn(spec=spec, y_true=y_true, y_pred=preds)
            (run_dir / "arbiter_report.json").write_text(
                json.dumps(report.model_dump(), indent=2)
                if hasattr(report, "model_dump") else json.dumps(_to_dict(report), indent=2)
            )
            result.arbiter_overall = float(getattr(report, "overall_score", 0.0))
            result.arbiter_baseline = float(getattr(report, "baseline_score", 0.0))
            result.arbiter_delta = float(getattr(report, "delta_vs_baseline", 0.0))
            result.arbiter_passes = bool(getattr(report, "passes_acceptance", False))
            result.arbiter_reason = str(getattr(report, "acceptance_reason", ""))
        except Exception as e:
            result.error = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=5)}"
            result.arbiter_passes = False

        result.elapsed_sec = round(time.monotonic() - t0, 2)
        per_case.append(result)

    report_path = out_dir / "thesis_report.md"
    report_path.write_text(_render_report(per_case, out_dir))
    return ExperimentResult(per_case=per_case, out_dir=out_dir, report_path=report_path)


def _to_dict(obj: Any) -> dict[str, Any]:
    return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}


def _render_report(per_case: list[CaseResult], out_dir: Path) -> str:
    ts = datetime.now(UTC).isoformat()
    lines = [
        "# Thesis Experiment Report",
        "",
        f"**Generated:** {ts}",
        f"**Cases:** {len(per_case)}",
        f"**All passed:** {all(c.arbiter_passes for c in per_case)}",
        "",
        "## Per-case results",
        "",
        "| case | arbiter overall | baseline | Δ | passes | features kept | winner | tokens | sec |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for c in per_case:
        if c.error:
            lines.append(
                f"| {c.case_name} | — | — | — | ❌ ERROR | — | — | — | {c.elapsed_sec} |"
            )
        else:
            lines.append(
                f"| {c.case_name} | {c.arbiter_overall:.4f} | {c.arbiter_baseline:.4f} | "
                f"{c.arbiter_delta:+.4f} | {'✅' if c.arbiter_passes else '❌'} | "
                f"{c.feature_count_kept} | {c.winner} | {c.tokens} | {c.elapsed_sec} |"
            )
    lines.append("")
    lines.append("## Detail")
    lines.append("")
    for c in per_case:
        lines.append(f"### {c.case_name}")
        lines.append(f"- run_dir: `{c.run_dir}`")
        lines.append(f"- research sources: {', '.join(c.research_sources) or '—'}")
        lines.append(f"- research results found: {c.research_n_results}")
        if c.error:
            lines.append("")
            lines.append("**Error:**")
            lines.append("```")
            lines.append(c.error)
            lines.append("```")
        else:
            lines.append(f"- arbiter reason: {c.arbiter_reason}")
        lines.append("")
    return "\n".join(lines)
