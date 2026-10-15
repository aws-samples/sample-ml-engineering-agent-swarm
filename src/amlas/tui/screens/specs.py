"""Specs pane — list, preview, and manage ML specs.

Phase 3 (TUI context integration, design §E.3 / §F): the Specs tab is now a
first-class **producer AND consumer** of the shared ``ContextBus`` (fixes GA-1 /
GT-1 — "when I load a spec it should configure the pipeline"). Loading, validating,
or selecting a spec here calls ``context_bus.set_spec(...)`` so the active spec
propagates to the Pipeline/Config pane and the agent's ``get_tui_context``. The pane
also SUBSCRIBES to the ``spec`` event and repaints its :class:`SpecViewerWidget`
(finally mounting the built-but-unused widget, GT-8) whenever the active spec changes
in any tab — using the App's ``call_from_thread`` marshalling discipline so widget
mutations always land on the UI thread. Everything is additive and null-safe: with no
bus (headless / pre-mount) the pane behaves exactly as before.
"""

from __future__ import annotations

from typing import Any, Literal

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, DataTable, Static, TextArea

from ..context import AppContextState, ContextEvent
from ..widgets.spec_viewer import SpecViewerWidget


class SpecsPane(Static):
    """Spec management screen with list and preview."""

    def compose(self) -> ComposeResult:
        # : redundant per-tab title removed (tab-nav already labels the tab).
        with Horizontal(classes="specs-layout"):
            with Vertical(classes="specs-list-panel"):
                yield Static("Available Specs", classes="section-title")
                yield SpecTable()
                with Horizontal(classes="spec-actions"):
                    yield Button("📂 Load", id="btn-load-spec", variant="primary")
                    yield Button("➕ New", id="btn-new-spec", variant="success")
                    yield Button("✓ Validate", id="btn-validate-spec", variant="warning")
            with Vertical(classes="spec-preview-panel"):
                yield Static("Spec Preview", classes="section-title")
                yield SpecPreview()
                # GT-8: the SpecViewerWidget was built but never mounted. It now
                # renders the active spec (from the shared bus) as a formatted card.
                yield SpecViewerWidget(id="spec-viewer")

    # ─── ContextBus wiring (design §E.3) ─────────────────────────────────────

    def on_mount(self) -> None:
        """Subscribe to the shared ``ContextBus`` and prime the viewer once.

        Null-safe: with no ``app.context_bus`` (a bare-pane offline validation / headless
        harness) this is a no-op, so the pane still mounts cleanly.
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
        """Bus subscriber — marshal the re-render onto the UI thread.

        Mirrors the App's ``_on_context_event`` discipline: a ``ContextBus`` mutation
        may originate on a background thread, so widget updates go through
        ``call_from_thread``. When already on the UI thread ``call_from_thread``
        raises and we update directly. Only ``spec`` events matter to this pane.
        Never raises (a bad refresh must not break the mutation that triggered it).
        """
        if event != "spec":
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
        """Repaint the SpecViewer from the active spec (guarded pre-mount).

        Phase 2 §3c (X-3): ALSO highlight the Specs table row matching the active
        spec's path (so the dynamic listing reflects which spec is active cross-tab).
        """
        # X-3: highlight the active spec row in the (dynamic) table (best-effort).
        # NB: query the SpecTable WRAPPER (which owns ``highlight_active`` + the
        # name→row map), not the inner ``#spec-table`` DataTable.
        try:
            self.query_one(SpecTable).highlight_active(state.spec.path)
        except Exception:  # nosec B110
            pass
        spec = state.spec
        if not (spec.path or spec.project_id or spec.spec_dict):
            return
        try:
            viewer = self.query_one("#spec-viewer", SpecViewerWidget)
        except Exception:
            return
        # Prefer the parsed front-matter dict; fall back to the summary fields.
        detail: dict[str, Any] = dict(spec.spec_dict) if spec.spec_dict else {}
        if not detail:
            for key, value in (
                ("project_id", spec.project_id),
                ("task_type", spec.task_type),
                ("target", spec.target),
                ("primary_metric", spec.primary_metric),
                ("golden_dataset", spec.golden_dataset),
            ):
                if value is not None:
                    detail[key] = value
        if spec.path:
            detail.setdefault("path", spec.path)
        try:
            viewer.update_spec(detail)
        except Exception:  # nosec B110
            pass

    def _publish_spec(self, *, path: str | None, content: str, valid: bool | None = None) -> None:
        """Parse ``content`` and publish it to the shared bus as the active spec.

        Best-effort: parse the YAML front-matter into ``SpecModel`` for the typed
        fields; if parsing fails we still publish the path so the Pipeline pane can
        prefill. Never raises — a bus write must not break a spec Load/Validate.

        Phase B1.5: a loaded spec now ALSO drives the data-oriented tabs BEFORE any
        run — :meth:`_publish_spec_dataset_and_validate` resolves the spec's local
        dataset (via the SAME ``resolve_run_data_path`` logic ``/run`` uses) and
        publishes it (``set_dataset(source="specs_tab")``), and validates the spec's
        ``target`` against that dataset's columns, warning via ``notify`` on a
        mismatch — catching the AI4I-class bug pre-run.
        """
        bus = self._bus()
        if bus is None:
            return
        spec_dict: dict[str, Any] = {}
        fields: dict[str, Any] = {}
        try:
            from ...spec import load_spec_from_text

            model = load_spec_from_text(content)
            spec_dict = model.model_dump()
            fields = {
                "project_id": spec_dict.get("project_id"),
                "task_type": spec_dict.get("task_type"),
                "target": spec_dict.get("target"),
                "primary_metric": spec_dict.get("primary_metric"),
                "golden_dataset": spec_dict.get("golden_dataset"),
            }
            if valid is None:
                valid = True
        except Exception:
            # Unparseable preview text — still surface the path/selection.
            if valid is None:
                valid = False
        try:
            bus.set_spec(
                path=path,
                spec_dict=spec_dict,
                valid=valid,
                source="specs_tab",
                **fields,
            )
        except Exception:  # nosec B110
            pass
        # B1.5: drive the data tabs + validate target — only when we parsed a real
        # spec dict (a resolvable local dataset + a declared target).
        if spec_dict:
            self._publish_spec_dataset_and_validate(bus, spec_dict)

    def _publish_spec_dataset_and_validate(self, bus: Any, spec_dict: dict[str, Any]) -> None:
        """Resolve the spec's LOCAL dataset → bus, and validate its target (B1.5).

        Uses the SAME ``resolve_run_data_path`` precedence as chat ``/run`` (an
        explicit path wins, else search ``data/`` subdirs, else a LOCAL
        ``golden_dataset``) so the Specs tab and chat feed ONE universal context. An
        ``s3://``-only golden_dataset can't resolve locally → we simply don't publish
        a dataset (no fabrication). When a local CSV resolves, its target is checked
        with ``resolve_target_column`` (case/space-insensitive, same as ``/run``);
        a mismatch surfaces a clear ``notify`` warning that names the bad target +
        the dataset's real columns — the pre-run catch for the AI4I-class bug.

        Fully null-safe / headless-safe: guards ``bus is None`` and wraps every step
        in try/except so a preview/Load never breaks.
        """
        if bus is None:
            return
        # 1. Resolve the effective LOCAL dataset (no explicit path → falls back to a
        #    local golden_dataset; s3:// or missing → RunDataPathError, handled).
        resolved: str | None = None
        try:
            from ..chat.engine import RunDataPathError, resolve_run_data_path

            try:
                resolved = resolve_run_data_path("", spec_dict)
            except RunDataPathError:
                resolved = None
        except Exception:
            resolved = None
        if not resolved:
            return
        # 2. Publish the resolved dataset so data-oriented tabs reflect the spec's
        #    dataset before any run — consistent with how /run publishes.
        try:
            bus.set_dataset(path=str(resolved), source="specs_tab")
        except Exception:  # nosec B110
            pass
        # 3. Validate the declared target against the resolved dataset's columns.
        target = spec_dict.get("target")
        if not target or not str(resolved).lower().endswith(".csv"):
            return
        try:
            import pandas as pd

            from ...tools.training_tools import resolve_target_column

            columns = list(pd.read_csv(resolved, nrows=0).columns)
        except Exception:
            return  # unreadable header → defer to the pipeline's own error
        try:
            resolve_target_column(columns, target)
        except ValueError:
            self._notify_safe(
                f"⚠️ Spec target `{target}` is not a column in the resolved dataset "
                f"`{resolved}`. Columns: {columns}. Fix the spec `target:` before /run.",
                severity="warning",
            )

    def _notify_safe(
        self,
        message: str,
        *,
        severity: Literal["information", "warning", "error"] = "information",
    ) -> None:
        """Best-effort ``self.notify`` that never raises headless (no running App).

        The App-level ``notify`` override already markup-escapes the
        message, so bracket-shaped dynamic text is safe here.
        """
        try:
            self.notify(message, severity=severity)
        except Exception:  # nosec B110
            pass

    def _capture_and_notify(self, exc: BaseException, *, source: str) -> None:
        """Funnel an exception to the centralized error log + a safe error toast.

        The single place the Specs pane routes an UNEXPECTED handler exception:
        record it in the process-wide :data:`~amlas.telemetry.error_log.ERROR_LOG`
        ( funnel — so it is inspectable across all runtime modes) and
        surface a plain error toast. The toast text is markup-safe because the App
        ``notify`` override escapes it; ``capture_error`` and ``notify`` are each
        individually guarded so the safety net itself can never raise.
        """
        try:
            from ...telemetry.error_log import capture_error

            capture_error(exc, source=source)
        except Exception:  # nosec B110
            pass
        self._notify_safe(f"❌ {type(exc).__name__}: {exc}", severity="error")

    def _spec_path_for(self, name: str) -> str | None:
        """Reconstruct the on-disk path for an example spec ``name`` (stem)."""
        try:
            from ...tools.spec_tools import SPECS_DIR

            return str(SPECS_DIR / f"{name}.md")
        except Exception:
            return None

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle spec action buttons (bubbled up from children).

         hardening: each handler is dispatched inside a guard so an
        unexpected exception in ANY spec action surfaces as a safe (escaped) error
        toast + a centralized ``capture_error`` record instead of tearing down the
        TUI. The Validate path additionally has its own internal guard (see
        :meth:`_validate_spec`); this is the outer belt-and-suspenders net.
        """
        handlers = {
            "btn-load-spec": self._load_spec,
            "btn-new-spec": self._new_spec,
            "btn-validate-spec": self._validate_spec,
        }
        handler = handlers.get(event.button.id or "")
        if handler is None:
            return
        try:
            handler()
        except Exception as exc:  # never let a button handler crash the app
            self._capture_and_notify(exc, source=f"tui.specs.{event.button.id}")

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Selecting a spec row publishes a lightweight selection to the bus.

        A row click/Enter records the transient selection (``selection`` event) and
        publishes the spec name + reconstructed path so other tabs can react even
        before the full Load parse. Guarded: only the spec table drives this.
        """
        try:
            table = event.data_table
            if table.id != "spec-table":
                return
            row = table.get_row_at(event.cursor_row)
        except Exception:
            return
        if not row:
            return
        name = str(row[0])
        bus = self._bus()
        if bus is None:
            return
        try:
            bus.set_selection({"tab": "specs", "row_id": name})
        except Exception:  # nosec B110
            pass
        path = self._spec_path_for(name)
        if path:
            try:
                from pathlib import Path

                content = Path(path).read_text(encoding="utf-8") if Path(path).exists() else ""
            except Exception:
                content = ""
            if content:
                # Reflect the selected spec into the preview + bus (full parse).
                try:
                    preview = self.app.query_one("#spec-preview-area", TextArea)
                    preview.load_text(content)
                except Exception:  # nosec B110
                    pass
                self._publish_spec(path=path, content=content)

    def _load_spec(self) -> None:
        """List example specs via the REAL ``list_example_specs`` tool (G33).

        Populates the preview with the first available spec's content, publishes it
        to the shared ``ContextBus`` (GA-1/GT-1 — so the Pipeline/Config pane and the
        agent see the active spec), and notifies how many specs were discovered.
        """
        try:
            from ...tools.spec_tools import list_example_specs

            result = list_example_specs()
        except Exception as e:
            self.notify(f"❌ Spec listing failed: {e}", severity="error")
            return

        specs = result.get("specs", []) if isinstance(result, dict) else []
        if not specs:
            self.notify(
                "📂 No example specs found in the specs/ directory.",
                severity="warning",
            )
            return

        first = specs[0]
        content = first.get("content", "")
        preview = self.app.query_one("#spec-preview-area", TextArea)
        preview.load_text(content)

        # Phase 3: propagate the loaded spec across tabs via the shared bus (GA-1).
        self._publish_spec(path=self._spec_path_for(first.get("name", "")), content=content)

        names = ", ".join(s.get("name", "?") for s in specs[:5])
        self.notify(
            f"📂 {len(specs)} spec(s): {names}. Loaded `{first.get('name')}` into preview.",
            severity="information",
        )

    def _new_spec(self) -> None:
        """Create a new blank spec template."""
        preview = self.app.query_one("#spec-preview-area", TextArea)
        template = (
            "---\n"
            "project_id: new-project\n"
            "task_type: classification\n"
            "target: target_column\n"
            "primary_metric: f1_macro\n"
            "threshold: 0.80\n"
            "golden_dataset: data/train.csv\n"
            "---\n"
            "\n"
            "# New ML Spec\n"
            "\n"
            "Edit the YAML fields above to define your ML task.\n"
        )
        preview.load_text(template)
        self.notify("✨ New spec template created — edit and save", severity="information")

    def _validate_spec(self) -> None:
        """Validate the preview content via the REAL validate_spec tool (G9).

        Phase 3: a valid spec is ALSO published to the shared bus (with ``valid``
        set) so validating in the Specs tab updates the active cross-tab spec.

        : the ENTIRE handler is wrapped so a validation error carrying a
        pydantic-style bracket token (``[type=..., input_value='…', input_type=str]``)
        — or any other unexpected exception — can never tear down the TUI. Error
        toasts are markup-safe (the App ``notify`` override escapes them) and any
        unexpected failure is funnelled to the centralized error log via
        :meth:`_capture_and_notify`.
        """
        try:
            preview = self.app.query_one("#spec-preview-area", TextArea)
            content = preview.text
            if not content.strip():
                self.notify("⚠️ Spec is empty — nothing to validate", severity="warning")
                return

            try:
                from ...tools.spec_tools import validate_spec

                result = validate_spec(content)
            except Exception as e:
                # Expected/known validate failures render as a (now escaped) toast
                # AND are captured centrally so the surface stays inspectable.
                self._capture_and_notify(e, source="tui.specs.validate")
                return

            if isinstance(result, dict) and result.get("valid"):
                # Publish the validated spec to the bus (path unknown for edited text).
                self._publish_spec(path=None, content=content, valid=True)
                self.notify("✅ Spec is valid", severity="information")
            else:
                errors = result.get("errors", []) if isinstance(result, dict) else []
                detail = "; ".join(str(x) for x in errors) if errors else "see schema requirements"
                self.notify(f"❌ Spec invalid: {detail}", severity="error")
        except Exception as exc:  # last-resort guard — never crash the TUI
            self._capture_and_notify(exc, source="tui.specs.validate")


class SpecTable(Static):
    """Scrollable table of available specs (X-3: real ``list_example_specs`` lister).

    Phase 2 §3c: replaces the hardcoded ``adsa-classifier`` / ``brake-anomaly`` /
    ``fuel-predictor`` sample rows with the REAL example specs under ``specs/``. Each
    row's Task Type / Target / Metric / Threshold come from parsing the spec's YAML
    front-matter (best-effort). Offline-safe: a lister/parse failure degrades to an
    empty table with a single placeholder row — mirroring ``ProjectTable.on_mount``.
    """

    def compose(self) -> ComposeResult:
        yield DataTable(id="spec-table")

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("Name", "Task Type", "Target", "Metric", "Threshold")
        table.cursor_type = "row"
        # Track the spec-name → row-index map for active-state highlighting (X-3).
        self._row_names: list[str] = []
        try:
            from ...tools.spec_tools import list_example_specs

            result = list_example_specs()
            specs = result.get("specs", []) if isinstance(result, dict) else []
        except Exception:
            specs = []

        if not specs:
            # Offline/empty placeholder — never crash, never show stale sample rows.
            table.add_row("(no example specs found)", "—", "—", "—", "—")
            return

        for entry in specs:
            name = str(entry.get("name", "?"))
            fields = self._parse_spec_fields(entry.get("content", ""))
            table.add_row(
                name,
                fields.get("task_type", "—"),
                fields.get("target", "—"),
                fields.get("primary_metric", "—"),
                fields.get("threshold", "—"),
            )
            self._row_names.append(name)

    def _parse_spec_fields(self, content: str) -> dict[str, str]:
        """Parse a spec's typed fields from its YAML front-matter (best-effort)."""
        try:
            from ...spec import load_spec_from_text

            model = load_spec_from_text(content)
            data = model.model_dump()
            return {
                "task_type": str(data.get("task_type") or "—"),
                "target": str(data.get("target") or "—"),
                "primary_metric": str(data.get("primary_metric") or "—"),
                "threshold": str(data.get("threshold") if data.get("threshold") is not None else "—"),
            }
        except Exception:
            return {}

    def highlight_active(self, spec_path: str | None) -> None:
        """Move the table cursor to the row whose spec stem matches ``spec_path`` (X-3)."""
        if not spec_path:
            return
        try:
            from pathlib import Path

            stem = Path(spec_path).stem
        except Exception:
            return
        names = getattr(self, "_row_names", [])
        try:
            idx = names.index(stem)
        except ValueError:
            return
        try:
            table = self.query_one(DataTable)
            table.move_cursor(row=idx)
        except Exception:  # nosec B110
            pass


class SpecPreview(Static):
    """Shows YAML/JSON preview of the selected spec.

    Phase B1 (fixes stuck mock data): the ``compose`` fallback baked in a fake
    ``adsa-classifier`` sample YAML (``target: trip_type`` / ``repos/adsa/train.csv``)
    that persisted until a spec was loaded. It now starts as a NEUTRAL empty
    ``TextArea`` (a placeholder comment); :class:`SpecsPane` loads the REAL selected
    spec's content into ``#spec-preview-area`` via ``.load_text(...)`` on Load /
    row-select / New, and the SpecViewer repaints from the ``"spec"`` bus event.
    """

    _EMPTY_PLACEHOLDER = (
        "# No spec selected — press 📂 Load, pick one from the list, or ➕ New.\n"
    )

    def compose(self) -> ComposeResult:
        yield TextArea(
            self._EMPTY_PLACEHOLDER,
            language="yaml",
            read_only=False,
            id="spec-preview-area",
        )
