"""MLOps pane — agentic-over-SageMaker dashboard (master plan §10, §8 D1-D7).

Surfaces the Model Registry (versions + approval), a monitor/drift panel, and the
MLOps KPI scorecard. The KPI refresh action invokes the REAL ``compute_mlops_kpis``
scorecard (pure, no AWS), proving the screen drives a Phase-10 tool rather than
showing static text.

Phase 6 (TUI context integration, design §E.3 / §F Phase 6, fixes GT-7): the MLOps tab
is now a first-class **producer AND consumer** of the shared ``ContextBus`` for the
active MODEL (use cases #12 readiness / #13 deploy / #14 monitor). Selecting a registry
row publishes ``set_model(...)``; the previously-unhandled **Approve** button publishes
``set_model(approval="Approved")``; and the pane SUBSCRIBES to the ``model`` event to
repaint a live "active model" line — so a model selected/approved here is visible to the
Dashboard, the agent (``get_tui_context``), and every other tab. Additive + null-safe.
"""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, DataTable, Static

from ...tools.mlops import compute_mlops_kpis
from ...tools.mlops.registry import list_model_packages_sm
from ..context import AppContextState, ContextEvent
from ..widgets.chart_view import ChartViewPanel


class MLOpsPane(Static):
    """MLOps dashboard — registry table, monitor panel, KPI scorecard."""

    def compose(self) -> ComposeResult:
        # : redundant per-tab title removed (tab-nav already labels the tab).
        with Vertical(classes="mlops-content"):
            with Horizontal(classes="mlops-top"):
                with Vertical(classes="mlops-registry-panel"):
                    yield Static("Model Registry", classes="section-title")
                    yield RegistryTable()
                with Vertical(classes="mlops-kpi-panel"):
                    yield Static("MLOps KPIs", classes="section-title")
                    yield Static("Press Refresh KPIs to compute the scorecard.", id="mlops-kpis")
            yield Static("No model selected.", id="mlops-active-model")
            with Horizontal(classes="mlops-actions"):
                yield Button("📊 Refresh KPIs", id="mlops-status-btn", variant="primary")
                yield Button("✅ Approve Selected", id="mlops-approve-btn", variant="success")
                yield Button("🛰️ Monitor", id="mlops-monitor-btn", variant="default")
            # Agent-addressable chart surface: "chart the endpoint latency on mlops" /
            # "show the drift trend" lands HERE, beside the model it describes. Mounted
            # AFTER the action row so it cannot push the buttons out of the visible screen
            # region (that broke every Pilot ``click`` on this tab).
            yield ChartViewPanel(tab="mlops", classes="chart-view-section")

    # ─── ContextBus wiring (design §E.3, fixes GT-7) ─────────────────────────

    def on_mount(self) -> None:
        """Subscribe to the shared ``ContextBus`` and prime the active-model line once.

        Null-safe: with no ``app.context_bus`` (bare-pane / headless) this is a no-op so
        the pane still mounts cleanly.
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
        """Bus subscriber — repaint the active-model line on the ``model`` event.

        Same discipline as the App's ``_on_context_event`` / the Phase-3 panes: a
        ``model`` mutation may originate off the UI thread, so the repaint is marshalled
        via ``call_from_thread``. Never raises.
        """
        if event != "model":
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
        """Repaint the active-model line from ``state.model`` (guarded pre-mount).

        Phase 2 §3c (X-3): ALSO highlight the registry row matching the active model
        so the dynamic listing reflects the active model cross-tab.
        """
        # X-3: highlight the active model row in the (dynamic) registry table.
        # NB: query the RegistryTable WRAPPER (which owns ``highlight_active`` + the
        # (model,version)→row map), not the inner ``#mlops-registry-table`` DataTable.
        try:
            self.query_one(RegistryTable).highlight_active(
                state.model.model_ref, state.model.version
            )
        except Exception:  # nosec B110
            pass
        model = state.model
        if not (model.model_ref or model.endpoint):
            return
        try:
            line = self.query_one("#mlops-active-model", Static)
        except Exception:
            return
        parts = [f"Active model: {model.model_ref or model.endpoint}"]
        if model.version:
            parts.append(f"version={model.version}")
        if model.approval:
            parts.append(f"approval={model.approval}")
        if model.endpoint:
            parts.append(f"endpoint={model.endpoint}")
        try:
            line.update(" · ".join(parts))
        except Exception:  # nosec B110
            pass

    def _publish_model(self, *, model_ref: str, version: str | None = None,
                       approval: str | None = None) -> None:
        """Publish the selected/approved model to the shared bus (fixes GT-7). Null-safe."""
        bus = self._bus()
        if bus is None:
            return
        try:
            bus.set_model(model_ref=model_ref, version=version, approval=approval)
        except Exception:  # nosec B110
            pass
        try:
            bus.set_selection({"tab": "mlops", "row_id": f"{model_ref}:{version or ''}"})
        except Exception:  # nosec B110
            pass

    def _selected_row(self) -> tuple[str, str | None, str | None] | None:
        """Return ``(model_ref, version, approval)`` for the highlighted registry row."""
        try:
            table = self.query_one("#mlops-registry-table", DataTable)
            row = table.get_row_at(table.cursor_row)
        except Exception:
            return None
        if not row:
            return None
        model_ref = str(row[0])
        version = str(row[1]) if len(row) > 1 else None
        approval = str(row[2]) if len(row) > 2 else None
        return model_ref, version, approval

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Selecting a registry row publishes the active model on the bus (fixes GT-7)."""
        try:
            table = event.data_table
            if table.id != "mlops-registry-table":
                return
            row = table.get_row_at(event.cursor_row)
        except Exception:
            return
        if not row:
            return
        model_ref = str(row[0])
        version = str(row[1]) if len(row) > 1 else None
        approval = str(row[2]) if len(row) > 2 else None
        self._publish_model(model_ref=model_ref, version=version, approval=approval)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "mlops-status-btn":
            self._refresh_kpis()
        elif event.button.id == "mlops-approve-btn":
            self._approve_selected()
        elif event.button.id == "mlops-monitor-btn":
            self._monitor_selected()

    def _approve_selected(self) -> None:
        """Approve the highlighted registry row → ``set_model(approval='Approved')``.

        Retires the GT-7 "MLOps Approve unhandled" gap: the approval is published to the
        shared bus so the Dashboard/agent see the model as Approved. Guarded/null-safe.
        """
        selected = self._selected_row()
        if selected is None:
            self.notify("Select a model in the registry first.", severity="warning")
            return
        model_ref, version, _ = selected
        self._publish_model(model_ref=model_ref, version=version, approval="Approved")
        self.notify(f"✅ Approved {model_ref} {version or ''}", severity="information")

    def _monitor_selected(self) -> None:
        """Monitor the highlighted model → publish it as the active model (bus re-point).

        Retires the GT-7 "MLOps Monitor unhandled" gap. Setting the active model re-points
        the Dashboard's ObservabilityPanel to this model/endpoint (design §E.7). Null-safe.
        """
        selected = self._selected_row()
        if selected is None:
            self.notify("Select a model in the registry first.", severity="warning")
            return
        model_ref, version, approval = selected
        self._publish_model(model_ref=model_ref, version=version, approval=approval)
        self.notify(f"🛰️ Monitoring {model_ref} {version or ''}", severity="information")

    def _refresh_kpis(self) -> None:
        """Invoke compute_mlops_kpis (real) and render the scorecard summary."""
        scorecard = compute_mlops_kpis({})
        overall = scorecard.get("overall", {})
        panel = self.query_one("#mlops-kpis", Static)
        panel.update(
            f"Maturity: {overall.get('maturity_level', 'L0')}\n"
            f"Composite health: {overall.get('composite_health', 0.0):.2f}\n"
            f"Pipelines: {scorecard.get('pipeline', {}).get('total', 0)}"
        )


class RegistryTable(Static):
    """Model registry table (X-3: real ``list_model_packages_sm`` lister, spec §3c).

    Phase 2 §3c: replaces the hardcoded ``amlas_models v3/v2/v1`` sample rows with the
    REAL SageMaker Model Registry listing for the configured Model Package Group. Each
    row is a live model-package version + its approval status. Offline-safe: an AWS/
    lister failure degrades to a SINGLE placeholder row naming the group — mirroring
    ``ProjectTable.on_mount`` (which surfaces a status row when the backend is
    unavailable). NO live AWS in CI — the lister is patched/injected in tests. The
    active model row is highlighted via :meth:`highlight_active`.
    """

    def compose(self) -> ComposeResult:
        yield DataTable(id="mlops-registry-table")

    def _group_name(self) -> str:
        """Resolve the configured Model Package Group name (default ``amlas_models``)."""
        try:
            from ...config.settings import Settings

            return str(Settings().mlops.model_package_group or "amlas_models")
        except Exception:
            return "amlas_models"

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("Model", "Version", "Approval", "F1")
        table.cursor_type = "row"
        group = self._group_name()
        # Track (model, version) → row-index for active-state highlighting (X-3).
        self._row_keys: list[tuple[str, str]] = []
        try:
            result = list_model_packages_sm._tool_func(group)
        except Exception:
            result = {"success": False}

        versions = (
            result.get("versions", [])
            if isinstance(result, dict) and result.get("success") is not False
            else []
        )
        if not versions:
            # Offline/empty placeholder — never crash, never show stale sample rows.
            table.add_row(group, "—", "(offline)", "—")
            return

        for v in versions:
            version = str(v.get("version", "—"))
            approval = str(v.get("approval_status", "—"))
            table.add_row(group, version, approval, "—")
            self._row_keys.append((group, version))

    def highlight_active(self, model_ref: str | None, version: str | None) -> None:
        """Move the table cursor to the row matching ``(model_ref, version)`` (X-3)."""
        if not model_ref:
            return
        keys = getattr(self, "_row_keys", [])
        want_version = str(version) if version is not None else None
        idx: int | None = None
        for i, (m, ver) in enumerate(keys):
            if m == model_ref and (want_version is None or ver == want_version):
                idx = i
                break
        if idx is None:
            return
        try:
            table = self.query_one(DataTable)
            table.move_cursor(row=idx)
        except Exception:  # nosec B110
            pass
