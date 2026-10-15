"""The agent's dashboard render verb — "show me X as a Y" (plan §B3).

**The gap this closes.** The TUI agent could publish DATA to the ``ContextBus`` (a dataset,
a job, a run) but had no way to express a VIEW: ``ContextEvent`` had no view-control member
and the Dashboard's primary pane chose its content from a job-kind keyword table
(``choose_dynamic_view``). So "plot the training curve", "show me the correlation heat map",
and "chart the leaderboard" had nowhere to land.

:func:`render_dashboard_view` is that verb. The agent names a chart and either supplies the
data or names a ``source`` to resolve it from; the tool renders it with the pure renderers
in [`tui/widgets/charts.py`](../tui/widgets/charts.py) and publishes an
:class:`~amlas.tui.context.ActiveView` so the pane repaints.

**Two invariants worth stating, because they are what make this trustworthy:**

* **Real data only.** A ``source`` reads the actual ``EvaluationStore`` / run
  ``arbiter_report.json``. When the requested data is absent the tool FAILS with a reason —
  it never substitutes sample or interpolated data, because a fabricated chart is worse
  than no chart (an ML engineer would act on it).
* **Read-only.** Rendering a view mutates no ML state and spends nothing, so it is
  auto-executing (``READ_ONLY_TOOLS``) — the agent can illustrate an answer without a
  confirmation round-trip.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from strands import tool

from ..errors import tool_error_boundary
from ..tui.widgets.charts import (
    render_bar,
    render_confusion_matrix,
    render_heatmap,
    render_histogram,
    render_line,
    render_scatter,
)

__all__ = ["SUPPORTED_CHARTS", "SUPPORTED_SOURCES", "render_dashboard_view"]

# The chart kinds the agent may request. Naming them explicitly (rather than accepting any
# string) means an unknown chart is an actionable error listing the real options, instead of
# a silent fallback to something the user did not ask for.
SUPPORTED_CHARTS: tuple[str, ...] = (
    "line",
    "bar",
    "heatmap",
    "scatter",
    "histogram",
    "confusion",
)

# Where the tool may resolve data from when the agent does not pass it inline.
SUPPORTED_SOURCES: tuple[str, ...] = ("inline", "leaderboard", "arbiter_report")

# Default canvas. Comfortably inside a standard 80-column terminal so the pane can render
# it without horizontal scrolling; the renderers clamp anything smaller.
_DEFAULT_WIDTH = 64
_DEFAULT_HEIGHT = 12


def _resolve_store_path(store_path: str | None) -> Path | None:
    """Resolve the EvaluationStore path (explicit arg, else the conventional location)."""
    if store_path:
        candidate = Path(store_path)
        return candidate if candidate.exists() else None
    for candidate in (Path("output/evaluation_store.json"), Path("output/store.json")):
        if candidate.exists():
            return candidate
    return None


def _leaderboard_values(store_path: str | None) -> tuple[dict[str, float], str]:
    """Read ``{algorithm: primary_score}`` from the REAL EvaluationStore.

    Returns ``({}, reason)`` when there is nothing to plot, so the caller can fail with a
    truthful message rather than charting an empty frame.
    """
    path = _resolve_store_path(store_path)
    if path is None:
        return {}, f"no evaluation store found (looked for {store_path or 'output/evaluation_store.json'})"
    try:
        from ..evaluation.store import EvaluationStore

        results = EvaluationStore(path).get_all()
    except Exception as exc:
        return {}, f"could not read the evaluation store: {type(exc).__name__}"
    if not results:
        return {}, "the evaluation store has no recorded experiments yet"

    # Best score per algorithm — the leaderboard question is "which algorithm won", so a
    # repeated algorithm collapses to its best run rather than plotting duplicates.
    best: dict[str, float] = {}
    for result in results:
        algo = str(getattr(result, "algorithm", "") or "unknown")
        try:
            score = float(getattr(result, "primary_score", 0.0))
        except (TypeError, ValueError):
            continue
        if algo not in best or score > best[algo]:
            best[algo] = score
    if not best:
        return {}, "the evaluation store has experiments but no numeric scores"
    return best, f"{len(results)} run(s) from {path}"


def _arbiter_values(run_dir: str | None) -> tuple[dict[str, float], str]:
    """Read the score-vs-baseline pair from a run's ``arbiter_report.json`` (real data)."""
    if not run_dir:
        return {}, "run_dir is required for source='arbiter_report'"
    report = Path(run_dir) / "arbiter_report.json"
    if not report.exists():
        return {}, f"no arbiter_report.json under {run_dir}"
    try:
        data = json.loads(report.read_text())
    except Exception as exc:
        return {}, f"could not parse {report}: {type(exc).__name__}"

    values: dict[str, float] = {}
    for key, label in (("overall_score", "achieved"), ("baseline_score", "baseline")):
        raw = data.get(key)
        if isinstance(raw, (int, float)):
            values[f"{label} ({data.get('primary_metric', 'score')})"] = float(raw)
    if not values:
        return {}, f"{report} has no numeric overall/baseline score"
    return values, f"arbiter report {report}"


@tool
@tool_error_boundary("tools.dashboard_tools")
def render_dashboard_view(
    chart: str,
    title: str = "",
    *,
    source: str = "inline",
    series: dict[str, list[float]] | None = None,
    values: dict[str, float] | None = None,
    matrix: list[list[float]] | None = None,
    x: list[float] | None = None,
    y: list[float] | None = None,
    labels: list[str] | None = None,
    view_id: str = "",
    target: str = "",
    store_path: str | None = None,
    run_dir: str | None = None,
    width: int = _DEFAULT_WIDTH,
    height: int = _DEFAULT_HEIGHT,
    bus: Any | None = None,
) -> dict[str, Any]:
    """Render a chart on a TUI tab (read-only; publishes a view request).

    Call this to SHOW something rather than describe it: a training curve, a
    predicted-vs-actual regression scatter with R², a feature-correlation heat map, a
    confusion matrix, a leaderboard bar chart, or a value distribution. The chart is
    rendered and published to the shared context bus, so the addressed tab repaints.

    Pass ``target`` to choose WHICH tab the chart lands on — put the leaderboard on
    ``results`` next to the run it describes, the pipeline throughput on ``pipeline``, the
    endpoint metrics on ``mlops``. Omit it for the Dashboard.

    Args:
        chart: One of :data:`SUPPORTED_CHARTS` — ``line`` (metric curves), ``bar``
            (leaderboard / feature importance), ``heatmap`` (correlation), ``scatter``
            (predicted-vs-actual + regression fit), ``histogram`` (distribution),
            ``confusion`` (confusion matrix).
        title: The header shown above the chart (e.g. ``"Training Curve — f1_macro"``).
        source: Where the data comes from. ``inline`` (default) uses the arguments below;
            ``leaderboard`` reads the real EvaluationStore; ``arbiter_report`` reads a run's
            ``arbiter_report.json``.
        series: For ``line`` — ``{series_name: [values...]}``.
        values: For ``bar`` — ``{label: value}``.
        matrix: For ``heatmap`` / ``confusion`` — row-major 2-D values.
        x: For ``scatter`` — x values (e.g. actual). For ``histogram`` — the raw values.
        y: For ``scatter`` — y values (e.g. predicted); same length as ``x``.
        labels: Axis / class labels for ``heatmap`` and ``confusion``.
        view_id: Optional logical view name (defaults to the chart kind).
        target: Which tab renders it — ``dashboard`` (default), ``results``, ``pipeline``,
            ``mlops``, ``data``, or ``lake``. An unknown target is refused (it names the
            real options) rather than silently landing somewhere the user did not ask for.
        store_path: Explicit EvaluationStore path for ``source="leaderboard"``.
        run_dir: Run directory for ``source="arbiter_report"``.
        width: Chart width in characters.
        height: Chart height in rows (line/scatter only).
        bus: Injected ``ContextBus`` (validation coverage); defaults to the active session bus.

    Returns:
        ``{"success", "chart", "title", "rendered", "published", "detail"}`` on success, or
        ``{"success": False, "error": ...}`` when the chart kind is unknown or the required
        data is missing. It NEVER substitutes placeholder data for missing data.
    """
    kind = (chart or "").strip().lower()
    if kind not in SUPPORTED_CHARTS:
        return {
            "success": False,
            "error": (
                f"unknown chart {chart!r} — supported charts are: "
                f"{', '.join(SUPPORTED_CHARTS)}"
            ),
        }

    # Validate against the tabs that EXIST in this config, not a static list. A disabled
    # tab composes no ``ChartViewPanel``, so accepting it returned success while the chart
    # rendered nowhere — the tool lied. The shipped surface enables only chat + dashboard.
    from ..tui.widgets.chart_view import CHART_HOSTABLE_TABS, chart_target_tabs

    available = chart_target_tabs()
    tab = (target or "").strip().lower()
    if tab and tab not in available:
        known = tab in CHART_HOSTABLE_TABS
        reason = (
            f"the {tab!r} tab is disabled in this configuration"
            if known
            else f"unknown target tab {target!r}"
        )
        hint = (
            f" (enable it with AMLAS_TUI_TAB_{tab.upper()}=1 or AMLAS_TUI_ENABLE_ALL_TABS=1)"
            if known
            else ""
        )
        return {
            "success": False,
            "error": (
                f"{reason} — chartable tabs available now: {', '.join(available)}{hint}"
            ),
        }

    src = (source or "inline").strip().lower()
    if src not in SUPPORTED_SOURCES:
        return {
            "success": False,
            "error": f"unknown source {source!r} — supported: {', '.join(SUPPORTED_SOURCES)}",
        }

    bar_values = dict(values or {})
    detail = ""
    if src == "leaderboard":
        bar_values, detail = _leaderboard_values(store_path)
        if not bar_values:
            return {"success": False, "error": detail}
    elif src == "arbiter_report":
        bar_values, detail = _arbiter_values(run_dir)
        if not bar_values:
            return {"success": False, "error": detail}

    heading = (title or "").strip() or kind.title()

    # Render, validating that the data the chosen chart NEEDS is actually present. Each
    # branch fails with the specific missing argument rather than drawing an empty frame.
    if kind == "line":
        if not (series and any(series.values())):
            return {"success": False, "error": "chart='line' requires a non-empty `series`"}
        rendered = render_line(series, title=heading, width=width, height=height)
    elif kind == "bar":
        if not bar_values:
            return {
                "success": False,
                "error": "chart='bar' requires `values` (or source='leaderboard')",
            }
        rendered = render_bar(bar_values, title=heading, width=width)
    elif kind == "heatmap":
        if not matrix:
            return {"success": False, "error": "chart='heatmap' requires a `matrix`"}
        rendered = render_heatmap(matrix, labels=labels, title=heading, width=width)
    elif kind == "confusion":
        if not matrix:
            return {"success": False, "error": "chart='confusion' requires a `matrix`"}
        rendered = render_confusion_matrix(matrix, labels=labels, title=heading, width=width)
    elif kind == "scatter":
        if not (x and y):
            return {"success": False, "error": "chart='scatter' requires both `x` and `y`"}
        rendered = render_scatter(x, y, title=heading, width=width, height=height)
    else:  # histogram
        if not x:
            return {"success": False, "error": "chart='histogram' requires `x` (the values)"}
        rendered = render_histogram(x, bins=10, title=heading, width=width)

    published = _publish(
        bus,
        view_id=(view_id or kind),
        chart=kind,
        title=heading,
        series=series,
        values=bar_values,
        matrix=matrix,
        x=x,
        y=y,
        labels=labels,
        detail=detail,
        target=tab,
    )

    return {
        "success": True,
        "chart": kind,
        "title": heading,
        "target": tab,
        "rendered": rendered,
        "published": published,
        "detail": detail or "inline data",
        "note": (
            f"The {tab or 'dashboard'} tab now shows this chart."
            if published
            else "Rendered, but no active TUI session to publish to."
        ),
    }


def _publish(bus: Any | None, **fields: Any) -> bool:
    """Publish the view to the bus; return whether it landed.

    Best-effort by design: the tool is useful (it still returns the rendered chart) when
    called headlessly from the CLI or a test with no TUI session, so a missing bus is a
    ``False`` return rather than an error.
    """
    try:
        from ..tui.context import ActiveView, get_active_bus

        target = bus if bus is not None else get_active_bus()
        if target is None:
            return False
        target.set_view(
            ActiveView(
                view_id=fields.get("view_id"),
                chart=fields.get("chart", ""),
                title=fields.get("title", ""),
                series={k: list(v) for k, v in (fields.get("series") or {}).items()},
                values=dict(fields.get("values") or {}),
                matrix=[list(r) for r in (fields.get("matrix") or [])],
                x=list(fields.get("x") or []),
                y=list(fields.get("y") or []),
                labels=[str(v) for v in (fields.get("labels") or [])],
                source="agent",
                detail=str(fields.get("detail") or ""),
                target=str(fields.get("target") or ""),
            )
        )
        return True
    except Exception:
        return False
