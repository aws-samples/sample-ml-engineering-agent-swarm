"""Lake pane — medallion lakehouse browser (master plan §10, §8 D6).

A catalog tree (bronze/silver/gold), a slice-read form, and a write-back form.
The Read action invokes the REAL ``read_lake_slice`` tool (S3/Glue under real AWS in
tests) — proving the screen drives the Phase-10 lake adapter rather than showing
static text. Writes target silver/gold only (bronze/raw is rejected at the tool).
"""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, DataTable, Input, Static

from ...tools.lake import list_lake_catalog, read_lake_slice, write_lake_dataset
from ..context import AppContextState, ContextEvent
from ..widgets.chart_view import ChartViewPanel


class LakePane(Static):
    """Medallion lakehouse browser — catalog, slice read, write-back."""

    # ─── ContextBus wiring (Phase 9 — agent-driven Lake reaction) ────────────
    #
    # The Lake pane was a PRODUCER-only widget: it painted ``#lake-read-result`` only on
    # a Read/Write BUTTON click, so an AGENT-driven lake read/write/register (folded to a
    # ``dataset``/``selection`` event by the chat engine) never repainted the Lake tab
    # (the DQ-LAKE-* Phase-8 gap). This adds the CONSUMER half: the pane SUBSCRIBES to the
    # bus and repaints ``#lake-read-result`` when a lake-sourced dataset (or a lake
    # selection) becomes active — so a chat/agent lake action is visible on the Lake tab
    # exactly like a button click. Additive/null-safe (mirrors the DataPane discipline).

    def on_mount(self) -> None:
        """Subscribe to the shared ``ContextBus`` (null-safe: no bus → no-op)."""
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
        """Repaint the lake result line on a lake-sourced ``dataset``/``selection`` event."""
        if event not in ("dataset", "selection"):
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
        """Reflect a lake-sourced active dataset / lake selection into ``#lake-read-result``.

        Only reacts to lake-originated context (``dataset.source == 'lake_tab'`` or a
        ``selection`` whose ``tab == 'lake'``) so a plain Data-tab dataset does not
        repaint the Lake pane. Guarded pre-mount; never raises.
        """
        ds = state.dataset
        selection = state.selection or {}
        is_lake_dataset = ds.path and ds.source == "lake_tab"
        is_lake_selection = selection.get("tab") == "lake"
        if not (is_lake_dataset or is_lake_selection):
            return
        try:
            panel = self.query_one("#lake-read-result", Static)
        except Exception:
            return
        path = ds.path or selection.get("row_id") or ""
        rows = ds.rows
        line = f"✓ active lake slice → {path}"
        if isinstance(rows, int):
            line = f"✓ {rows} rows → {path}"
        try:
            panel.update(line)
        except Exception:  # nosec B110
            pass

    def compose(self) -> ComposeResult:
        # : redundant per-tab title removed (tab-nav already labels the tab).
        with Vertical(classes="lake-content"):
            with Horizontal(classes="lake-top"):
                with Vertical(classes="lake-catalog-panel"):
                    yield Static("Catalog (bronze / silver / gold)", classes="section-title")
                    yield CatalogTable()
                with Vertical(classes="lake-slice-panel"):
                    yield Static("Read a silver/gold slice", classes="section-title")
                    yield Input(placeholder="silver://db.table or s3://bucket/key.csv", id="lake-uri-input")
                    yield Input(placeholder="columns (comma-sep, optional)", id="lake-columns-input")
                    yield Static("", id="lake-read-result")
            with Horizontal(classes="lake-actions"):
                yield Button("📥 Read Slice", id="lake-read-btn", variant="primary")
                yield Button("📤 Write Back", id="lake-write-btn", variant="success")
            # Agent-addressable chart surface (e.g. "chart the partition row counts on lake").
            # Mounted AFTER the action row so it cannot push the buttons out of the
            # visible screen region (that breaks Pilot clicks — see widgets/AGENTS.md).
            yield ChartViewPanel(tab="lake", classes="chart-view-section")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "lake-read-btn":
            self._read_slice()
        elif event.button.id == "lake-write-btn":
            self._write_back()

    def _read_slice(self) -> None:
        """Invoke read_lake_slice (real tool) for the URI in the form.

        Phase 3 (design §E.3): a successfully-read slice now becomes the active
        dataset on the shared ``ContextBus`` (``set_dataset(source="lake_tab")``) so
        a lake read is visible to the Data/Pipeline tabs and the agent (closes the
        GA-4 "a lake read never becomes the active dataset" gap). Additive/null-safe.
        """
        uri = self.query_one("#lake-uri-input", Input).value.strip()
        result_panel = self.query_one("#lake-read-result", Static)
        if not uri:
            result_panel.update("Enter a silver://db.table or s3:// URI first.")
            return
        cols_raw = self.query_one("#lake-columns-input", Input).value.strip()
        columns = [c.strip() for c in cols_raw.split(",") if c.strip()] or None
        result = read_lake_slice._tool_func(uri, columns=columns)
        if result.get("success") is False:
            result_panel.update(f"✗ {result.get('error', 'read failed')}")
            self.notify("Lake read failed", severity="error")
            return
        result_panel.update(
            f"✓ {result.get('rows', 0)} rows (tier={result.get('tier', '?')}) → "
            f"{result.get('local_path', '?')}"
        )
        self._activate_slice(uri, result)
        self.notify("Lake slice read", severity="information")

    def _write_back(self) -> None:
        """Invoke write_lake_dataset (real tool) to push the active artifact back.

        Phase 2 (spec §3a, fixes X-1): ``on_button_pressed`` previously handled ONLY
        ``lake-read-btn`` — the ``lake-write-btn`` had NO branch, so a write-back never
        ran and never became the active dataset. This medallion write-back uploads the
        currently-active (cleaned/imputed/synthetic) artifact BACK to the silver/gold
        destination in ``#lake-uri-input`` (bronze/raw rejected at the tool). On success
        it propagates the written slice through the SAME ``_activate_slice`` seam the
        read path uses (``set_dataset(source="lake_tab")``) so a write-back is visible to
        the Data/Pipeline tabs and the agent exactly like a read. Additive/null-safe.
        """
        dest_uri = self.query_one("#lake-uri-input", Input).value.strip()
        result_panel = self.query_one("#lake-read-result", Static)
        if not dest_uri:
            result_panel.update("Enter a silver://db.table or s3:// destination URI first.")
            return
        # The write SOURCE is the currently-active dataset artifact (a cleaned/imputed
        # slice produced by chat/agent/Data-tab work). Null-safe: no active artifact →
        # guide the user rather than crash.
        local_path = self._active_artifact_path()
        if not local_path:
            result_panel.update("No active dataset artifact to write back — load/clean one first.")
            self.notify("No active dataset to write back", severity="warning")
            return
        result = write_lake_dataset._tool_func(local_path, dest_uri)
        if result.get("success") is False:
            result_panel.update(f"✗ {result.get('error', 'write-back failed')}")
            self.notify("Lake write-back failed", severity="error")
            return
        written = result.get("written_uri", dest_uri)
        result_panel.update(
            f"✓ wrote {result.get('rows', 0)} rows (tier={result.get('tier', '?')}) → {written}"
        )
        # Propagate the written slice as the active dataset (same seam as the read path).
        self._activate_slice(str(written), result)
        self.notify("Lake write-back complete", severity="information")

    def _active_artifact_path(self) -> str | None:
        """Return the active dataset's local artifact path from the bus (null-safe).

        The write-back source is whatever slice/dataset is currently active on the
        shared ``ContextBus`` (a chat/agent-cleaned artifact). Returns ``None`` headless
        / when nothing is active so ``_write_back`` can guide instead of raising.
        """
        try:
            bus = getattr(self.app, "context_bus", None)
        except Exception:
            bus = None
        if bus is None:
            return None
        try:
            path: str | None = bus.state().dataset.path
            return path
        except Exception:
            return None

    def _activate_slice(self, uri: str, result: dict[str, Any]) -> None:
        """Publish the read slice as the active dataset on the shared bus.

        Uses the tool's ``local_path`` (the materialized CSV) when present, else the
        source URI. Never raises — a bus write must not break the Read Slice action.
        """
        try:
            bus = getattr(self.app, "context_bus", None)
        except Exception:
            bus = None
        if bus is None:
            return
        path = result.get("local_path") or uri
        rows = result.get("rows")
        try:
            bus.set_dataset(
                path=str(path),
                rows=int(rows) if isinstance(rows, int) else None,
                source="lake_tab",
            )
        except Exception:  # nosec B110
            pass
        try:
            bus.set_selection({"tab": "lake", "row_id": uri})
        except Exception:  # nosec B110
            pass


class CatalogTable(Static):
    """Glue-style catalog table grouped by medallion tier (X-3: real lister, spec §3c).

    Phase 2 §3c: replaces the hardcoded ``locomotive_1957`` / ``wheel_wear`` /
    ``fault_features`` sample rows with the REAL Glue Data Catalog listing
    (``list_lake_catalog``). Each row is a live catalog dataset + its tier + column
    count. Offline-safe: an AWS/Glue failure (or no catalog) degrades to an EMPTY table
    (no rows) — never the static sample, never a crash — mirroring
    ``ProjectTable.on_mount``'s defensive fallback. NO live AWS in CI (the lister is
    patched/injected in tests).
    """

    def compose(self) -> ComposeResult:
        yield DataTable(id="lake-catalog-table")

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("Dataset", "Tier", "Rows", "Columns")
        table.cursor_type = "row"
        try:
            result = list_lake_catalog._tool_func()
        except Exception:
            result = {"success": False}

        datasets = (
            result.get("datasets", [])
            if isinstance(result, dict) and result.get("success") is not False
            else []
        )
        if not datasets:
            # Offline / empty catalog → a single placeholder row (never the static
            # locomotive_1957 sample, never a crash) — mirrors ProjectTable.on_mount's
            # status-row fallback and keeps the on-mount "≥1 row" contract.
            table.add_row("(catalog unavailable)", "—", "—", "—")
            return
        for ds in datasets:
            columns = ds.get("columns") or []
            table.add_row(
                str(ds.get("name", "?")),
                str(ds.get("tier", "—")),
                "—",  # row counts require a scan; not surfaced by the catalog listing
                str(len(columns)) if columns else "—",
            )
