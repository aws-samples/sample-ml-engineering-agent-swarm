"""``DynamicPrimaryPane``: the dashboard's agent-chosen primary view.

When the user selects an :class:`~amlas.tui.widgets.active_jobs.ActiveJobsPanel`
row, the primary
left-hand pane updates with a job-appropriate, dynamic view — "the TUI agent's
decision of what data should be displayed": a data-science chart (a training curve /
scores) for a training or pipeline-run job, an inference/scores view, a data-quality
view, etc. — "something dynamic, topical, important about that particular job".

**The "TUI agent's opinion" is a pure, deterministic policy** — :func:`choose_dynamic_view`
maps a job ``kind`` to a ``(view_id, human_title)`` pair. Keeping the decision policy a
pure function (rather than a live LLM call on the render path) makes it directly verifiable
headless, instant, and drift-proof, while still being the single place "what matters
for this job type" is decided — the exact seam a smarter agent-driven selector can
later replace without touching the pane or the layout.

**Data source (no fake data):** the pane subscribes to the bus ``job`` + ``selection``
events. It records the REAL observed ``ActiveJob.progress`` series per ``job_id`` (a
genuine, monotonically-appended signal the JobManager already publishes) and drives a
built-in Textual :class:`~textual.widgets.Sparkline` from it for the chart views. When
a view has no numeric series to plot (e.g. a data-quality scan), it renders a
structured text/log fallback — never a fabricated series.

**Discipline (mirrors the Phase-4/5 panes):** a ``set_job`` / ``set_selection`` may fire
from a background job thread, so the repaint marshals via ``app.call_from_thread`` with
a direct-call fallback; every ``query_one`` is guarded so a headless / pre-mount
subscriber never crashes.
"""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Sparkline, Static

from ..context import AppContextState, ContextEvent
from ..markup import escape_markup

__all__ = ["DynamicPrimaryPane", "choose_dynamic_view"]

# ── The "TUI agent's opinion" — the view-selection policy ──────────────────────
#
# Each entry maps a lower-cased job-kind SUBSTRING → (view_id, human_title). The
# first matching keyword wins (checked in this order), so the more specific/critical
# job families are listed first. This is the single source of truth for "what view
# matters for this job type" — the DS chart / scores / training / inference / DQ
# decision policy.
_TRAINING_KEYS = ("train", "tune", "hpo", "optuna", "sft", "dpo", "grpo", "finetune", "fine_tune")
_INFERENCE_KEYS = ("infer", "predict", "deploy", "serve", "endpoint", "batch_transform", "transform")
_DQ_KEYS = (
    "data_quality", "dqscan", "dq", "quality", "profil", "imput", "imbalance",
    "drift", "validation", "dedup", "outlier", "synthes", "synthetic",
)
_SCORES_KEYS = ("run", "regression", "classif", "score", "eval", "leaderboard", "ensemble")

# Human titles per view id — the label the pane renders as its self-title header.
_VIEW_TITLES: dict[str, str] = {
    "training": "Training Curve",
    "inference": "Inference & Scores",
    "data_quality": "Data Quality",
    "scores": "Model Scores",
    "overview": "Job Overview",
}


def choose_dynamic_view(kind: str | None) -> tuple[str, str]:
    """Return ``(view_id, human_title)`` — the TUI agent's opinion for a job ``kind``.

    Pure and deterministic (no I/O, no LLM on the render path) so it is directly verifiable
    and can never drift. Matches a lower-cased keyword against the job kind; unknown /
    empty kinds fall back to the neutral ``"overview"`` view (never raises).
    """
    k = (kind or "").strip().lower()
    if not k:
        return "overview", _VIEW_TITLES["overview"]
    # Order matters: DQ and training/inference families are more specific than the
    # generic "run"/scores family, so they are checked first.
    for view_id, keys in (
        ("data_quality", _DQ_KEYS),
        ("training", _TRAINING_KEYS),
        ("inference", _INFERENCE_KEYS),
        ("scores", _SCORES_KEYS),
    ):
        if any(key in k for key in keys):
            return view_id, _VIEW_TITLES[view_id]
    return "overview", _VIEW_TITLES["overview"]


# View ids whose primary content is a numeric chart (driven by the Sparkline).
_CHART_VIEWS = frozenset({"training", "inference", "scores"})


class DynamicPrimaryPane(Static):
    """Agent-chosen, per-job primary view — swaps content on Active-Jobs selection."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Real observed per-job progress series (job_id → [progress, ...]); the chart
        # is driven from THIS (no fabricated data). Appended on each ``job`` event.
        self._series: dict[str, list[float]] = {}
        # The currently-selected job id (set by a dashboard ``selection`` event).
        self._selected_job_id: str | None = None
        # The last rendered content body (inspection seam — avoids reaching into
        # the Static's private ``renderable``).
        self._last_text: str = self._placeholder()
        # Set when an agent-requested chart was rendered while this pane was NOT visible
        # (e.g. the request arrives from Chat, before the user switches to the Dashboard).
        # ``scroll_visible`` cannot work on a hidden tab, so the scroll is DEFERRED to
        # :meth:`on_show`. Without this, a chart requested from Chat renders below the fold
        # and the user sees an unchanged dashboard.
        self._pending_scroll: bool = False

    def compose(self) -> ComposeResult:
        with Vertical():
            # markup=False keeps dynamic job ids and kinds literal.
            yield Static(self._placeholder(), id="dynamic-primary-content", markup=False)
            # The built-in Textual chart for numeric views. Starts empty/hidden; the
            # selected chart view fills + reveals it.
            spark = Sparkline([], id="dynamic-primary-chart")
            spark.display = False
            yield spark

    # ─── ContextBus wiring (mirrors the Phase-4/5 panes) ────────────────────────

    def on_mount(self) -> None:
        """Subscribe to the bus and prime from current state. Null-safe (no bus → no-op)."""
        bus = self._bus()
        if bus is None:
            return
        try:
            bus.subscribe(self._on_context)
            self._prime(bus.state())
        except Exception:  # nosec B110
            pass

    def on_show(self) -> None:
        """Complete a deferred scroll once this pane actually becomes visible.

        Textual fires this when the containing tab is shown. An agent chart requested from
        Chat is rendered while the Dashboard is hidden, so the scroll is deferred to here —
        this is what makes "show me a heat map" land in view when the user arrives.
        """
        if not self._pending_scroll:
            return
        self._pending_scroll = False
        try:
            self.scroll_visible(animate=False)
        except Exception:  # nosec B110
            pass

    def _bus(self) -> Any | None:
        try:
            return getattr(self.app, "context_bus", None)
        except Exception:
            return None

    def _on_context(self, event: ContextEvent, state: AppContextState) -> None:
        """Record ``job`` progress + repaint on ``job``/``selection``/``view``. Never raises.

        Plan §B4 adds ``view``: the agent's explicit render request (published by
        ``render_dashboard_view``) must repaint this pane, otherwise every "show me a heat
        map" would be silently dropped.
        """
        if event not in ("job", "selection", "view"):
            return
        # Always fold the observed progress series (both event kinds carry the snapshot).
        self._record_series(state)
        try:
            app = self.app
        except Exception:
            app = None
        if app is None:
            self._apply_selection(state)
            return
        try:
            app.call_from_thread(self._apply_selection, state)
        except Exception:
            try:
                self._apply_selection(state)
            except Exception:  # nosec B110
                pass

    # ─── Series capture (REAL data only) ────────────────────────────────────────

    def _record_series(self, state: AppContextState) -> None:
        """Append every job's observed progress to its series (deduping repeats)."""
        jobs = state.jobs or {}
        for job_id, job in jobs.items():
            try:
                pct = float(job.progress or 0.0)
            except Exception:  # nosec B112
                continue
            series = self._series.setdefault(job_id, [])
            if not series or series[-1] != pct:
                series.append(pct)

    def series_for(self, job_id: str) -> list[float]:
        """Return the observed progress series for a job (test/inspection seam)."""
        return list(self._series.get(job_id, []))

    def _prime(self, state: AppContextState) -> None:
        """Seed series + selection from the current snapshot (on-mount / re-enter)."""
        self._record_series(state)
        self._apply_selection(state)

    # ─── Selection → view swap ───────────────────────────────────────────────────

    def _apply_selection(self, state: AppContextState) -> None:
        """Swap the primary content to the requested view, else the job-kind default.

        Precedence (plan §B4): an EXPLICIT agent view request wins over the keyword policy.
        Before this, ``choose_dynamic_view`` was the only decider — its docstring called
        itself "the TUI agent's opinion" while actually being a job-kind substring match the
        agent could not influence. Now the agent can genuinely ask, and the keyword policy
        remains the default for when it hasn't (so all prior behaviour is preserved).
        """
        # ``getattr(state, "view", None)`` keeps this null-safe for a legacy snapshot built
        # before ``AppContextState.view`` existed; the chart is read off the record directly
        # (a 3-arg ``getattr`` default here would be typed against the field, not a str).
        requested = getattr(state, "view", None)
        if requested is not None and str(getattr(requested, "chart", None) or ""):
            self._render_requested_view(requested)
            return

        selection = state.selection or {}
        # Only dashboard selections drive this pane (spec/data/lake selections are for
        # other tabs); fall back to the focused job when no dashboard selection exists.
        job_id = None
        if selection.get("tab") == "dashboard":
            job_id = selection.get("job_id")
        if not job_id:
            job_id = state.job.job_id
        self._selected_job_id = job_id

        job = None
        if job_id and state.jobs:
            job = state.jobs.get(job_id)
        if job is None:
            job = state.job

        kind = getattr(job, "kind", None)
        view_id, title = choose_dynamic_view(kind)
        self._render_view(view_id, title, job_id, job, state)

    def _render_requested_view(self, view: Any) -> None:
        """Render an agent-requested chart (plan §B4) into the content Static.

        Delegates to the pure renderers in :mod:`amlas.tui.widgets.charts`, so the pane
        holds no plotting logic and a chart can never raise on the render path. The
        built-in ``Sparkline`` is hidden: these charts are self-contained text, and leaving
        a progress sparkline underneath a heat map would misattribute the data.
        """
        from .charts import (
            render_bar,
            render_confusion_matrix,
            render_heatmap,
            render_histogram,
            render_line,
            render_scatter,
        )

        chart = str(getattr(view, "chart", "") or "").lower()
        title = str(getattr(view, "title", "") or "Dynamic View")
        labels = list(getattr(view, "labels", None) or []) or None

        try:
            if chart == "line":
                body = render_line(dict(getattr(view, "series", None) or {}), title=title)
            elif chart == "bar":
                body = render_bar(dict(getattr(view, "values", None) or {}), title=title)
            elif chart == "heatmap":
                body = render_heatmap(
                    list(getattr(view, "matrix", None) or []), labels=labels, title=title
                )
            elif chart == "confusion":
                body = render_confusion_matrix(
                    list(getattr(view, "matrix", None) or []), labels=labels, title=title
                )
            elif chart == "scatter":
                body = render_scatter(
                    list(getattr(view, "x", None) or []),
                    list(getattr(view, "y", None) or []),
                    title=title,
                )
            elif chart == "histogram":
                body = render_histogram(list(getattr(view, "x", None) or []), title=title)
            else:
                body = f"{title}\n  (unsupported chart {chart!r})"
        except Exception:
            body = f"{title}\n  (chart could not be rendered)"

        detail = str(getattr(view, "detail", "") or "")
        if detail:
            body = f"{body}\n  source: {escape_markup(detail)}"

        self._last_text = body
        try:
            spark = self.query_one("#dynamic-primary-chart", Sparkline)
            spark.data = []
            spark.display = False
        except Exception:  # nosec B110
            pass
        try:
            self.query_one("#dynamic-primary-content", Static).update(body)
        except Exception:  # nosec B110
            pass

        # Scroll the pane into view. The Dashboard's left column is taller than the
        # viewport (health → jobs → dynamic → progress → observability → processes → log),
        # so this pane sits BELOW THE FOLD: a mandated-path E2E screenshot showed the chart
        # rendered correctly and still invisible until scrolled. A user who asks "show me a
        # heat map" must SEE it without hunting.
        #
        # A request usually arrives from CHAT, while the Dashboard tab is still hidden — and
        # ``scroll_visible`` silently no-ops on a hidden widget. So scroll now when we can,
        # and otherwise defer to :meth:`on_show` (fired when the tab becomes visible).
        self._pending_scroll = True
        try:
            if self.is_on_screen:  # nosemgrep: is-function-without-parentheses
                self.scroll_visible(animate=False)
                self._pending_scroll = False
        except Exception:  # nosec B110
            pass

    def _render_view(
        self,
        view_id: str,
        title: str,
        job_id: str | None,
        job: Any,
        state: AppContextState,
    ) -> None:
        """Render the chosen view: chart-backed views drive the Sparkline; DQ/overview text."""
        series = self._series.get(job_id or "", []) if job_id else []
        is_chart = view_id in _CHART_VIEWS and len(series) >= 1

        # Drive (or hide) the built-in Sparkline from the REAL observed series.
        try:
            spark = self.query_one("#dynamic-primary-chart", Sparkline)
            if is_chart:
                spark.data = list(series)
                spark.display = True
            else:
                spark.data = []
                spark.display = False
        except Exception:  # nosec B110
            pass

        text = self._view_text(view_id, title, job_id, job, series, state)
        self._last_text = text
        try:
            content = self.query_one("#dynamic-primary-content", Static)
            content.update(text)
        except Exception:  # nosec B110
            pass

    def _view_text(
        self,
        view_id: str,
        title: str,
        job_id: str | None,
        job: Any,
        series: list[float],
        state: AppContextState,
    ) -> str:
        """Build the human text body for a view (a pure string — directly verifiable)."""
        header = [f"🎛️ Dynamic View — {title}", "━━━━━━━━━━━━━━━━━━━━━━━━━━"]
        if not job_id:
            return "\n".join(
                [*header, "  Select a job row above to see a job-specific view."]
            )

        kind = escape_markup(str(getattr(job, "kind", None) or "—"))
        status = escape_markup(str(getattr(job, "status", None) or "idle"))
        stage = escape_markup(str(getattr(job, "stage", None) or "—"))
        lines = list(header)
        lines.append(f"  Job {escape_markup(str(job_id))}  ·  kind={kind}  ·  {status}  ·  stage={stage}")

        if view_id in _CHART_VIEWS:
            if series:
                last = series[-1]
                lines.append(f"  Progress series ({len(series)} pts): latest {last:.0f}%")
                lines.append("  (live chart below — driven by observed job progress)")
            else:
                lines.append("  No numeric series yet — chart appears as progress is reported.")
            if view_id == "training":
                lines.append("  Showing: training/tuning progress curve.")
            elif view_id == "inference":
                lines.append("  Showing: inference throughput / scores.")
            else:  # scores
                lines.append("  Showing: model scores across the run.")
            # Surface the completed run's best score when present (real ActiveRun data).
            run = state.run
            if run and isinstance(run.best_score, (int, float)):
                metric = escape_markup(str(run.primary_metric or "score"))
                lines.append(f"  Best {metric}: {run.best_score:.3f}")
        elif view_id == "data_quality":
            lines.append("  Showing: data-quality summary for this job.")
            ds = state.dataset
            if ds and ds.path:
                lines.append(f"  Dataset: {escape_markup(str(ds.path))}")
            if ds and isinstance(ds.dq_score, (int, float)):
                lines.append(f"  DQ score: {ds.dq_score:.3f}")
            else:
                lines.append("  DQ score: — (run a /dqscan to populate)")
        else:  # overview
            lines.append("  Showing: job overview (no specialized view for this kind).")
            lines.append("  Select a training / inference / data-quality job for a chart.")

        return "\n".join(lines)

    # ─── Render helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _placeholder() -> str:
        return (
            "🎛️ Dynamic View\n━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "  Select an Active Jobs row to see a job-specific view."
        )

    def _render_text(self) -> str:
        """Return the current content body (test/inspection seam)."""
        return self._last_text
