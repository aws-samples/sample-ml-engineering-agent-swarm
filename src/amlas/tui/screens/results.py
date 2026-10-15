"""Results & Leaderboard pane — view ranked model performance.

Phase 6 (TUI context integration, design §E.3 / §F Phase 6, fixes GT-4): the Results
tab is now a first-class **consumer** of the shared ``ContextBus``. Previously its
``BestModelCard`` was hardcoded static text and the pane never subscribed to any event,
so a completed/selected run was invisible here (design §D GT-4, use case #11
``evaluation_flow``). Now the pane SUBSCRIBES to the ``run`` event and repaints the
best-model card from ``state.run`` (algorithm / metric / score / decision) whenever a
run becomes active in ANY tab or via the agent — using the App's ``call_from_thread``
marshalling discipline. Everything is additive and null-safe: with no bus (headless /
pre-mount) the card shows its placeholder and never crashes.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Static

from ..context import ActiveRun, AppContextState, ContextEvent
from ..widgets.chart_view import ChartViewPanel
from ..widgets.leaderboard import LeaderboardWidget


def _find_store_path() -> Path:
    """Locate the evaluation store JSON (cwd ``output/`` first, then search)."""
    store_path = Path("./output/evaluation_store.json")
    if not store_path.exists():
        found = list(Path(".").rglob("evaluation_store.json"))
        if found:
            store_path = found[0]
    return store_path


class ResultsPane(Static):
    """Results & leaderboard screen with sortable metrics table."""

    def compose(self) -> ComposeResult:
        # : redundant per-tab title removed (tab-nav already labels the tab).
        with Vertical(classes="results-content"):
            with Horizontal(classes="results-top"):
                with Vertical(classes="leaderboard-panel"):
                    yield Static("Leaderboard", classes="section-title")
                    yield LeaderboardWidget()
                with Vertical(classes="best-model-panel"):
                    yield Static("🥇 Best Model", classes="section-title")
                    yield BestModelCard()
            with Horizontal(classes="results-actions"):
                yield Button("📊 Compare", id="btn-compare", variant="primary")
                yield Button("📥 Export CSV", id="btn-export", variant="default")
                yield Button("🔄 Refresh", id="btn-refresh", variant="warning")
            # Agent-addressable chart surface: "chart the leaderboard on results" /
            # "plot predicted vs actual" lands HERE, next to the run it describes, instead
            # of only on the Dashboard. Mounted AFTER the action row on purpose — placing it
            # above pushed the buttons off the visible screen region, which broke every
            # Pilot ``click`` on this tab.
            yield ChartViewPanel(tab="results", classes="chart-view-section")

    # ─── ContextBus wiring (design §E.3, fixes GT-4) ─────────────────────────

    def on_mount(self) -> None:
        """Subscribe to the shared ``ContextBus`` and prime the card once.

        Null-safe: with no ``app.context_bus`` (bare-pane / headless) this is a no-op
        so the pane still mounts cleanly.
        """
        bus = self._bus()
        if bus is None:
            return
        try:
            bus.subscribe(self._on_context)
            self._apply_context(bus.state())
        except Exception:  # nosec B110
            pass

    def _bus(self) -> Any | None:
        """Return the App-owned ``ContextBus`` or ``None`` (headless / pre-mount)."""
        try:
            return getattr(self.app, "context_bus", None)
        except Exception:
            return None

    def _on_context(self, event: ContextEvent, state: AppContextState) -> None:
        """Bus subscriber — repaint the best-model card on the ``run`` event.

        Mirrors the App's ``_on_context_event`` discipline: a ``ContextBus`` mutation
        may originate on a background thread (a completed run publishing ``set_run``),
        so widget updates go through ``call_from_thread``. Only ``run`` events matter to
        this pane. Never raises (a bad refresh must not break the mutation).
        """
        if event != "run":
            return
        try:
            app = self.app
        except Exception:
            return
        try:
            app.call_from_thread(self._apply_context, state)
        except Exception:
            try:
                self._apply_context(state)
            except Exception:  # nosec B110
                pass

    def _apply_context(self, state: AppContextState) -> None:
        """Repaint the BestModelCard from the active run (guarded pre-mount).

        The "is there anything to show" test must match
        :meth:`BestModelCard.update_from_run`'s — an earlier narrower guard here
        (``run_id or best_algorithm``) dropped the event BEFORE the card could render a
        decision-only run, so a real ``get_recommendation`` answer never reached the tab.
        """
        run = state.run
        if not (
            run.run_id
            or run.best_algorithm
            or run.decision
            or run.primary_metric
            or isinstance(run.best_score, (int, float))
        ):
            return
        try:
            card = self.query_one(BestModelCard)
        except Exception:
            return
        try:
            card.update_from_run(run)
        except Exception:  # nosec B110
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle results action buttons."""
        if event.button.id == "btn-compare":
            self._compare()
        elif event.button.id == "btn-export":
            self._export()
        elif event.button.id == "btn-refresh":
            self._refresh()

    def _refresh(self) -> None:
        """Re-render the best-model card from the current bus run (Phase 6, was a no-op).

        The previous Refresh only emitted a notification. It now re-applies the active
        run so a user who clicks Refresh sees the live cross-tab run — null-safe.
        """
        bus = self._bus()
        if bus is not None:
            try:
                self._apply_context(bus.state())
            except Exception:  # nosec B110
                pass
        self.notify("🔄 Leaderboard refreshed", severity="information")

    def _compare(self) -> None:
        """Compare recorded runs via the REAL compare_experiments tool (G6)."""
        store_path = _find_store_path()
        if not store_path.exists():
            self.notify("📊 No experiments recorded yet — nothing to compare.", severity="warning")
            return
        try:
            from ...evaluation.store import EvaluationStore
            from ...tools.evaluation_tools import compare_experiments

            store = EvaluationStore(store_path=store_path)
            run_ids = [r.run_id for r in store.get_all()]
            if len(run_ids) < 2:
                self.notify(
                    "📊 Need at least 2 runs to compare.", severity="warning"
                )
                return
            result = compare_experiments(
                store_path=str(store_path),
                run_ids=json.dumps(run_ids),
            )
            n = result.get("n_experiments", len(run_ids)) if isinstance(result, dict) else len(run_ids)
            self.notify(f"📊 Compared {n} experiments.", severity="information")
        except Exception as e:
            self.notify(f"❌ Compare failed: {e}", severity="error")

    def _export(self) -> None:
        """Export the best run's model artifact to ``output/export/`` (G4)."""
        store_path = _find_store_path()
        if not store_path.exists():
            self.notify("📥 No experiments recorded yet — nothing to export.", severity="warning")
            return
        try:
            from ...evaluation.store import EvaluationStore

            store = EvaluationStore(store_path=store_path)
            results = store.get_all()
            # Prefer a successful run that has an on-disk artifact.
            candidates = [
                r for r in results
                if r.artifact_path and Path(r.artifact_path).exists()
            ]
            if not candidates:
                self.notify(
                    "📥 No model artifact available to export.", severity="warning"
                )
                return
            source = candidates[0].artifact_path
            dest_dir = Path("./output/export")
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / Path(source).name
            shutil.copy2(source, dest)
            self.notify(f"📥 Exported artifact to {dest}", severity="information")
        except Exception as e:
            self.notify(f"❌ Export failed: {e}", severity="error")


class BestModelCard(Static):
    """Shows details of the best-performing model.

    Phase 6: the placeholder text is the fallback; :meth:`update_from_run` repaints it
    from the active :class:`~amlas.tui.context.models.ActiveRun` on the shared bus (fixes
    GT-4 — the card was previously hardcoded and never scoped to the active run).
    """

    _PLACEHOLDER = (
        "  No active run yet.\n"
        "  Start one with `/run <spec> <data>` in Chat (or the Pipeline tab),\n"
        "  or select a run, and its best model will appear here."
    )

    def compose(self) -> ComposeResult:
        yield Static(self._PLACEHOLDER, id="best-model-details")

    def update_from_run(self, run: ActiveRun) -> None:
        """Repaint the card from the active run (guarded pre-mount, null-safe).

        Renders the run id, best algorithm, primary metric + score, and the
        deploy/iterate/reject decision — the real cross-tab run, not hardcoded text.
        """
        try:
            details = self.query_one("#best-model-details", Static)
        except Exception:
            return
        # "Has something to say" is broader than "has an id or an algorithm": a real
        # ``get_recommendation`` returns a deploy/iterate/reject DECISION with no run
        # identity, and an evaluation flow can report a metric+score before a run id
        # exists. Discarding those to the placeholder threw away the exact answer the
        # user asked for (ML-EVAL-001/007), so any populated field renders the card.
        if not (
            run.run_id
            or run.best_algorithm
            or run.decision
            or run.primary_metric
            or isinstance(run.best_score, (int, float))
        ):
            try:
                details.update(self._PLACEHOLDER)
            except Exception:  # nosec B110
                pass
            return
        lines: list[str] = []
        # Phase B1: surface the run's lifecycle status prominently — a FAILED run
        # must read as a failure, not an empty/ambiguous card.
        status = getattr(run, "status", None)
        if status:
            label = "❌ FAILED" if status == "failed" else f"Status:     {status.upper()}"
            lines.append(f"  {label}")
        if run.run_id:
            lines.append(f"  Run:        {run.run_id}")
        if run.spec_id:
            lines.append(f"  Spec:       {run.spec_id}")
        if run.best_algorithm:
            lines.append(f"  Algorithm:  {run.best_algorithm}")
        if run.primary_metric is not None:
            score = (
                f"{run.best_score:.4f}"
                if isinstance(run.best_score, (int, float))
                else str(run.best_score)
            )
            lines.append(f"  {run.primary_metric}:  {score}")
        elif isinstance(run.best_score, (int, float)):
            lines.append(f"  Score:      {run.best_score:.4f}")
        if run.decision:
            lines.append(f"  Decision:   {run.decision.upper()}")
        if run.artifacts_dir:
            lines.append(f"  Artifacts:  {run.artifacts_dir}")
        try:
            details.update("\n".join(lines) if lines else self._PLACEHOLDER)
        except Exception:  # nosec B110
            pass
