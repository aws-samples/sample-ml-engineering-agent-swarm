"""Data Explorer pane — browse datasets, quality scores, and EDA summaries.

Phase 11 extends this pane (master plan §10) with a **DQ pattern-scan** trigger:
an input for the dataset path + a button that invokes the real Phase-6
``scan_dq_patterns`` tool (the 25-pattern catalog) and renders the findings. This
is the Data screen's "Patterns" agentic flow — a genuine tool call, not a cosmetic
button.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, DataTable, Input, Static

from ...config._env_compat import env_with_compat
from ...config.aws_region import resolve_aws_region
from ...tools.data_quality import scan_dq_patterns
from ..context import AppContextState, ContextEvent
from ..widgets.chart_view import ChartViewPanel


def _parse_int(value: str) -> int | None:
    """Parse a table-cell integer like ``"12,450"`` → ``12450`` (``None`` if N/A)."""
    if value is None:
        return None
    cleaned = str(value).replace(",", "").strip()
    if not cleaned or cleaned in {"—", "-", "?"}:
        return None
    try:
        return int(cleaned)
    except ValueError:
        return None


def _dataset_scan_roots() -> list[Path]:
    """Return the directories the dataset scan globs (CWD-relative + repo-anchored).

    Mirrors :meth:`DataPane._load_data`'s ``data/`` / ``repos/`` roots but ALSO anchors
    to the project root derived from this module's path (``src/amlas/tui/screens/data.py``
    → ``parents[4]``), so the scan finds the tracked fixtures regardless of the process
    current working directory (the same ``__file__``-anchored discipline ``spec_tools.SPECS_DIR``
    uses). De-duplicated, existence-checked.
    """
    roots: list[Path] = [Path("data"), Path("repos")]
    try:
        repo_root = Path(__file__).resolve().parents[4]
        roots.extend([repo_root / "data", repo_root / "repos"])
    except Exception:  # nosec B110
        pass
    seen: dict[str, Path] = {}
    for r in roots:
        try:
            if r.exists():
                seen.setdefault(str(r.resolve()), r)
        except Exception:  # nosec B112
            continue
    return list(seen.values())


def _scan_local_datasets(limit: int = 50) -> list[str]:
    """Scan common data directories for CSV/parquet files (X-3, spec §3c).

    Mirrors the local scan already in :meth:`DataPane._load_data` (globs ``data/`` /
    ``repos/`` for ``*.csv`` / ``*.parquet``), but robust to CWD via
    :func:`_dataset_scan_roots`. Returns a de-duplicated, sorted list of discovered
    paths made relative to the project root when possible (capped at ``limit`` for a
    bounded table). Module-level so the dynamic :class:`DatasetTable` and tests share
    ONE discovery seam. Never raises.
    """
    try:
        repo_root = Path(__file__).resolve().parents[4]
    except Exception:
        repo_root = None
    found: set[str] = set()
    try:
        for d in _dataset_scan_roots():
            for pattern in ("*.csv", "*.parquet"):
                for f in d.rglob(pattern):
                    rel: str
                    if repo_root is not None:
                        try:
                            rel = str(f.resolve().relative_to(repo_root))
                        except Exception:
                            rel = str(f)
                    else:
                        rel = str(f)
                    found.add(rel)
    except Exception:
        return []
    # Deterministic, de-duplicated, bounded.
    return sorted(found)[:limit]


def _list_all_objects(client: Any, bucket: str, *, prefix: str | None = None) -> list[dict[str, Any]]:
    """Enumerate ALL objects in ``bucket`` via ``list_objects_v2`` pagination (PERF-13s).

    A single ``list_objects_v2`` call returns at most 1000 keys and silently truncates
    the rest (no ``NextContinuationToken`` follow-up) — so a bucket with >1000 objects
    under-reports. This loops on ``NextContinuationToken`` until ``IsTruncated`` is False,
    returning the full ``Contents`` list. Same API (``list_objects_v2``), correct result.
    """
    contents: list[dict[str, Any]] = []
    kwargs: dict[str, Any] = {"Bucket": bucket}
    if prefix:
        kwargs["Prefix"] = prefix
    while True:
        resp = client.list_objects_v2(**kwargs)
        if not isinstance(resp, dict):
            break
        contents.extend(resp.get("Contents", []) or [])
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
        if not token:
            break
        kwargs["ContinuationToken"] = token
    return contents


class DataPane(Static):
    """Data explorer screen for dataset management + DQ pattern scan."""

    def compose(self) -> ComposeResult:
        # : redundant per-tab title removed (tab-nav already labels the tab).
        with Vertical(classes="data-content"):
            with Horizontal(classes="data-top"):
                with Vertical(classes="dataset-list-panel"):
                    yield Static("Datasets", classes="section-title")
                    yield DatasetTable()
                with Vertical(classes="data-details-panel"):
                    yield Static("Quality & EDA", classes="section-title")
                    yield DataQualityPanel()
            with Vertical(classes="data-dq"):
                yield Static("DQ Pattern Scan (25-pattern catalog)", classes="section-title")
                yield Input(placeholder="dataset path (e.g. data/locomotive_1957.csv)", id="data-dq-path")
                yield Static("", id="data-dq-result")
            with Horizontal(classes="data-actions"):
                yield Button("📂 Load Dataset", id="btn-load-data", variant="primary")
                yield Button("🔍 Run EDA", id="btn-run-eda", variant="success")
                yield Button("🩺 DQ Scan", id="data-dqscan-btn", variant="warning")
                yield Button("☁️ S3 Browse", id="btn-s3-browse", variant="default")
                yield Button("⬆️ S3 Upload", id="btn-s3-upload", variant="warning")
                yield Button("⬇️ S3 Download", id="btn-s3-download", variant="warning")
            # Agent-addressable chart surface (e.g. "chart the tool_wear distribution on data" / a feature correlation heat map).
            # Mounted AFTER the action row so it cannot push the buttons out of the
            # visible screen region (that breaks Pilot clicks — see widgets/AGENTS.md).
            yield ChartViewPanel(tab="data", classes="chart-view-section")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle ALL data explorer button presses.

        BUGFIX (2026-06-04): this class previously declared ``on_button_pressed``
        TWICE — a second definition (handling only ``data-dqscan-btn``) shadowed
        this one in the class body, so the Load/EDA/S3 buttons became DEAD code
        (their handlers were never reached). That broke the G7/G8/G9/G36 wiring
        tests (``test_data_pane``). The two handlers are now merged into this single
        dispatch so every button — including the DQ-scan — routes to its action.
        """
        if event.button.id == "btn-load-data":
            self._load_data()
        elif event.button.id == "btn-run-eda":
            self._run_eda()
        elif event.button.id == "btn-s3-browse":
            self._s3_browse()
        elif event.button.id == "btn-s3-upload":
            self._s3_upload()
        elif event.button.id == "btn-s3-download":
            self._s3_download()
        elif event.button.id == "data-dqscan-btn":
            self._run_dqscan()

    # ─── ContextBus wiring (design §E.3) ─────────────────────────────────────

    def on_mount(self) -> None:
        """Subscribe to the shared ``ContextBus`` and prime the DQ panel once.

        Null-safe: with no ``app.context_bus`` (bare-pane / headless) this is a
        no-op so the pane still mounts cleanly.
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
        """Bus subscriber — marshal the DQ-panel re-render onto the UI thread.

        Same discipline as the App's ``_on_context_event``: only the ``dataset``
        event repaints the Quality/EDA panel. Never raises (a bad refresh must not
        break the mutation that triggered it).
        """
        if event != "dataset":
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
        """Repaint the Quality/EDA panel from the active dataset (guarded pre-mount).

        Phase 2 §3c (X-3): ALSO highlight the Data table row matching the active
        dataset path so the dynamic listing reflects the active dataset cross-tab.
        """
        # X-3: highlight the active dataset row in the (dynamic) table (best-effort).
        # NB: query the DatasetTable WRAPPER (which owns ``highlight_active`` + the
        # path→row map), not the inner ``#dataset-table`` DataTable.
        try:
            self.query_one(DatasetTable).highlight_active(state.dataset.path)
        except Exception:  # nosec B110
            pass
        ds = state.dataset
        if not ds.path:
            return
        try:
            panel = self.query_one("#data-quality-details", Static)
        except Exception:
            return
        lines = [f"📈 Dataset: {ds.path}", "━━━━━━━━━━━━━━━━━━━━━━━━━━"]
        if ds.rows is not None:
            lines.append(f"  Rows:           {ds.rows:,}")
        if ds.cols is not None:
            lines.append(f"  Columns:        {ds.cols}")
        if ds.dq_score is not None:
            lines.append(f"  Quality Score:  {ds.dq_score:.2f}")
        if ds.columns:
            preview = ", ".join(ds.columns[:8])
            lines.append("")
            lines.append(f"Columns: {preview}")
        if ds.source:
            lines.append("")
            lines.append(f"(active — set via {ds.source})")
        try:
            panel.update("\n".join(lines))
        except Exception:  # nosec B110
            pass

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Selecting a dataset row makes it the active dataset (fixes GA-2).

        The DatasetTable ``RowSelected`` message bubbles here; we parse the row's
        name/rows/cols cells and publish them via ``_activate_dataset``. Guarded so
        only the dataset table drives this (other tables ignored).
        """
        try:
            table = event.data_table
            if table.id != "dataset-table":
                return
            row = table.get_row_at(event.cursor_row)
        except Exception:
            return
        if not row:
            return
        name = str(row[0])
        rows = _parse_int(row[2]) if len(row) > 2 else None
        cols = _parse_int(row[3]) if len(row) > 3 else None
        self._activate_dataset(name, rows=rows, cols=cols)

    def _activate_dataset(
        self, path: str, *, rows: int | None = None, cols: int | None = None
    ) -> None:
        """Publish the selected dataset to the shared bus + record the selection.

        Fixes GA-2: the Data tab finally has a "set active dataset" write-path.
        Sets ``set_dataset(source="data_tab")`` and a transient ``set_selection``
        marker so other tabs (and the agent) see the tab-side selection. Null-safe.
        """
        bus = self._bus()
        if bus is None:
            return
        try:
            bus.set_dataset(path=path, rows=rows, cols=cols, source="data_tab")
        except Exception:  # nosec B110
            pass
        try:
            bus.set_selection({"tab": "data", "row_id": path})
        except Exception:  # nosec B110
            pass

    def _load_data(self) -> None:
        """Load a dataset from local directory scan."""
        # Scan common data directories for CSV/parquet files
        data_dirs = [Path("data"), Path("repos"), Path(".")]
        found_files: list[str] = []
        for d in data_dirs:
            if d.exists():
                found_files.extend(str(f) for f in d.rglob("*.csv"))
                found_files.extend(str(f) for f in d.rglob("*.parquet"))

        if found_files:
            count = len(found_files)
            sample = ", ".join(found_files[:3])
            self.notify(
                f"📂 Found {count} data file(s): {sample}... "
                f"Use /data <path> in chat to load one.",
                severity="information",
            )
        else:
            self.notify(
                "📂 No CSV/parquet files found in data/ or repos/ directories. "
                "Use /attach <path> in chat to load a file by path.",
                severity="warning",
            )

    def _s3_client(self) -> Any:
        """Return a cached ``boto3.client("s3")`` — built once, reused per pane (PERF-03).

        The S3 browse/upload/download buttons previously each built a fresh
        ``boto3.client("s3")`` per click, re-resolving creds + endpoint every
        time. This lazy singleton builds the client on first use and reuses it
        for the lifetime of the pane. Same API calls / same behavior — only the
        client construction is amortized.
        """
        client = getattr(self, "_s3_client_cache", None)
        if client is None:
            import boto3

            client = boto3.client("s3", region_name=resolve_aws_region())
            self._s3_client_cache = client
        return client

    def reset_client_cache(self) -> None:
        """Drop the cached boto3 S3 client so the NEXT S3 op rebuilds it (Phase D1).

        A mode switch may change creds/profile/offline posture; the per-pane S3
        singleton (``_s3_client_cache``) was built once and would otherwise reuse
        stale config. Clearing it forces a fresh ``boto3.client("s3")`` on next use
        (which re-resolves the profile/endpoint). Null-safe / never raises.
        """
        self._s3_client_cache = None

    def _active_dataset(self) -> dict[str, Any] | None:
        """Read the active dataset summary from the Chat pane's engine context."""
        try:
            from .chat import ChatPane

            engine = self.app.query_one(ChatPane)._engine
            summary = engine.context.get("data_summary")
            return summary if isinstance(summary, dict) else None
        except Exception:
            return None

    def _run_eda(self) -> None:
        """Run the REAL EDA report tool on the active dataset (G7)."""
        summary = self._active_dataset()
        data_path = summary.get("path") if summary else None
        if not data_path:
            self.notify(
                "🔍 No active dataset. Load one with /data <path> in the Chat tab first.",
                severity="warning",
            )
            return

        # Infer target column from the summary (fallbacks handled by default).
        target = "target"
        cols = summary.get("columns") if summary else None
        if cols:
            for cand in ("target", "label", "y", "class"):
                if cand in cols:
                    target = cand
                    break
            else:
                target = cols[-1]

        try:
            from ...tools.data_quality_tools import generate_eda_report

            result = generate_eda_report(data_path=str(data_path), target_column=target)
            summary_text = (
                result.get("summary", "EDA complete.")
                if isinstance(result, dict)
                else "EDA complete."
            )
            self.notify(f"🔍 EDA complete: {summary_text}", severity="information")
        except Exception as e:
            self.notify(f"❌ EDA failed: {e}", severity="error")

    def _s3_browse(self) -> None:
        """List S3 objects via boto3 when a bucket is configured (G8)."""

        bucket = env_with_compat("AMLAS_S3_BUCKET")
        if not bucket:
            self.notify(
                "☁️ S3 browsing requires a bucket. Set AGENTICML_S3_BUCKET "
                "and AWS credentials (AWS_PROFILE/.env) to enable.",
                severity="warning",
            )
            return

        try:
            client = self._s3_client()
            # PERF-13s: paginate so buckets with >1000 objects are fully counted.
            contents = _list_all_objects(client, bucket)
            self.notify(
                f"☁️ S3 `{bucket}`: {len(contents)} object(s) found.",
                severity="information",
            )
        except Exception as e:
            self.notify(f"❌ S3 browse failed: {e}", severity="error")

    def _s3_upload(self) -> None:
        """Upload the active dataset to S3 via boto3 (G36 — creds-gated).

        Safety default: cloud side-effect surfaced via an explicit button only.
        Uses the active dataset from the ChatPane engine context as the source;
        the destination key mirrors the file name.
        """

        bucket = env_with_compat("AMLAS_S3_BUCKET")
        if not bucket:
            self.notify(
                "☁️ S3 upload requires a bucket. Set AGENTICML_S3_BUCKET and AWS "
                "credentials (AWS_PROFILE/.env) to enable.",
                severity="warning",
            )
            return
        summary = self._active_dataset()
        data_path = summary.get("path") if summary else None
        if not data_path:
            self.notify(
                "⬆️ No active dataset. Load one with /data <path> in the Chat tab first.",
                severity="warning",
            )
            return
        try:
            key = f"uploads/{Path(data_path).name}"
            client = self._s3_client()
            client.upload_file(str(data_path), bucket, key)
            self.notify(
                f"⬆️ Uploaded `{data_path}` → s3://{bucket}/{key}",
                severity="information",
            )
        except Exception as e:
            self.notify(f"❌ S3 upload failed: {e}", severity="error")

    def _s3_download(self) -> None:
        """Download the first S3 object into ``output/downloads/`` via boto3 (G36).

        Lists the bucket, then downloads the first object as a concrete,
        creds-gated demonstration of the data-input path (no tool equivalent).
        """

        bucket = env_with_compat("AMLAS_S3_BUCKET")
        if not bucket:
            self.notify(
                "☁️ S3 download requires a bucket. Set AGENTICML_S3_BUCKET and AWS "
                "credentials (AWS_PROFILE/.env) to enable.",
                severity="warning",
            )
            return
        try:
            client = self._s3_client()
            # PERF-13s: paginate so the first object is found even past key 1000.
            contents = _list_all_objects(client, bucket)
            if not contents:
                self.notify(f"⬇️ S3 `{bucket}` is empty — nothing to download.", severity="warning")
                return
            key = contents[0]["Key"]
            dest = Path("output/downloads") / Path(key).name
            dest.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, key, str(dest))
            self.notify(
                f"⬇️ Downloaded s3://{bucket}/{key} → {dest}",
                severity="information",
            )
        except Exception as e:
            self.notify(f"❌ S3 download failed: {e}", severity="error")

    def _run_dqscan(self) -> None:
        """Invoke scan_dq_patterns (real Phase-6 tool) for the dataset path."""
        path = self.query_one("#data-dq-path", Input).value.strip()
        result_panel = self.query_one("#data-dq-result", Static)
        if not path:
            result_panel.update("Enter a dataset path first.")
            return
        result = scan_dq_patterns._tool_func(data_path=path)
        if result.get("success") is False:
            result_panel.update(f"✗ {result.get('error', 'scan failed')}")
            return
        detected = result.get("patterns_detected", [])
        ids = ", ".join(sorted({f.get("pattern_id", "") for f in detected})) or "none"
        result_panel.update(
            f"✓ score {result.get('overall_score', 0):.1f}/100 · "
            f"{len(detected)} finding(s): {ids}"
        )
        self.notify("DQ pattern scan complete", severity="information")


class DatasetTable(Static):
    """Table listing available datasets (X-3: real local scan, spec §3c).

    Phase 2 §3c: replaces the hardcoded ``adsa/train.csv`` / ``brake-data/raw.parquet``
    / … sample rows with a REAL scan of the local ``data/`` / ``repos/`` directories
    (:func:`_scan_local_datasets`). Rows/Cols are unknown from a filename scan and shown
    as ``—`` (the selection handler's ``_parse_int`` treats ``—`` as ``None``). Offline /
    empty-scan safe: a placeholder row communicates the empty state — mirroring
    ``ProjectTable.on_mount``. The active dataset row is highlighted via
    :meth:`highlight_active`.
    """

    def compose(self) -> ComposeResult:
        yield DataTable(id="dataset-table")

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("Dataset", "Source", "Rows", "Cols", "Quality Score")
        table.cursor_type = "row"
        # Track the dataset-path → row-index map for active-state highlighting (X-3).
        self._row_paths: list[str] = []
        try:
            discovered = _scan_local_datasets()
        except Exception:
            discovered = []

        if not discovered:
            # Offline/empty placeholder — never crash, never show stale sample rows.
            table.add_row("(no local datasets found)", "—", "—", "—", "—")
            return

        for path in discovered:
            source = "local" if path.startswith(("data/", "repos/")) else "—"
            table.add_row(path, source, "—", "—", "—")
            self._row_paths.append(path)

    def highlight_active(self, dataset_path: str | None) -> None:
        """Move the table cursor to the row whose path matches ``dataset_path`` (X-3)."""
        if not dataset_path:
            return
        paths = getattr(self, "_row_paths", [])
        try:
            idx = paths.index(dataset_path)
        except ValueError:
            return
        try:
            table = self.query_one(DataTable)
            table.move_cursor(row=idx)
        except Exception:  # nosec B110
            pass


class DataQualityPanel(Static):
    """Shows data quality details for the ACTIVE dataset.

    Phase B1 (fixes stuck mock data): the ``compose`` fallback was hardcoded
    ``adsa/train.csv / 12,450 / 24 / 0.92 / speed_avg↔fuel_rate 0.82`` — fabricated
    numbers that persisted until a ``dataset`` event fired. It is now a NEUTRAL empty
    placeholder; :meth:`DataPane._apply_context` repaints the ``#data-quality-details``
    Static from ``state.dataset`` (real path / rows / cols / dq_score) on the
    ``"dataset"`` bus event, so only genuinely-available metrics are shown.
    """

    _PLACEHOLDER = (
        "📈 No dataset loaded\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "  Load a spec (Specs tab) or run `/run <spec> <data>`,\n"
        "  or select a dataset above, and its quality & EDA\n"
        "  summary will appear here."
    )

    def compose(self) -> ComposeResult:
        yield Static(self._PLACEHOLDER, id="data-quality-details")
