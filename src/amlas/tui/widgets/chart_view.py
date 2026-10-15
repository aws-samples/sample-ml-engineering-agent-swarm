"""``ChartViewPanel`` — a per-tab, agent-addressable chart surface (goal item #3).

ADR-0006 made the Dashboard's primary pane agent-renderable, but a chart request that can
only ever land on ONE pane does not deliver "re-render the tab based on the user's chat
request — pipelines, experiments, specs + results". This widget generalizes it: the SAME
panel is hosted on Results / Pipeline / MLOps (and reusable anywhere), each instance bound
to its own ``tab``, and each renders ONLY a view addressed to that tab.

**Why a target rather than broadcasting.** One ``view`` event painting the same chart onto
every surface would make the tabs indistinguishable and destroy whatever each tab was
already showing. Addressing means "plot the leaderboard on Results" puts it on Results and
leaves Pipeline alone. An UNTARGETED request belongs to the Dashboard, so ADR-0006's
existing behaviour (and every test of it) is unchanged.

**Rendering** delegates entirely to the pure functions in [`charts.py`](charts.py) — this
widget holds no plotting logic, so a chart can never raise on a repaint path. Split into
:meth:`apply_view` (pure state → text, directly verifiable with no Textual app) and the Textual
mount/subscribe wiring, mirroring the discipline of the Phase-4/5 panes.
"""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from ..context import AppContextState, ContextEvent

__all__ = ["CHART_HOSTABLE_TABS", "CHART_TARGET_TABS", "ChartViewPanel", "chart_target_tabs"]

# Every tab that CAN host a chart panel — i.e. where a ``ChartViewPanel`` is composed.
# This is the static capability; what is actually OFFERED to the agent is the subset whose
# feature flag is on (see :func:`chart_target_tabs`).
CHART_HOSTABLE_TABS: tuple[str, ...] = ("dashboard", "results", "pipeline", "mlops", "data", "lake")

# Back-compat alias for the pre-gating name. Prefer :func:`chart_target_tabs`, which
# reflects the tabs that actually exist in the running config.
CHART_TARGET_TABS: tuple[str, ...] = CHART_HOSTABLE_TABS

# The tab an untargeted request belongs to (ADR-0006 back-compat).
_DEFAULT_TARGET = "dashboard"


def chart_target_tabs() -> tuple[str, ...]:
    """Return the tabs a chart may be addressed to IN THIS CONFIG (Dashboard first).

    Operator direction 2026-07-30: the Dashboard is the single intelligent visualization
    surface; the other tabs are feature-flagged developer surfaces. The shipped default
    enables only ``chat`` + ``dashboard``, so this returns ``("dashboard",)`` there and the
    full set under ``AMLAS_TUI_ENABLE_ALL_TABS``.

    **Why this is derived rather than a constant.** A static list let the render tool accept
    ``target="results"`` while no ``ChartViewPanel`` was mounted (that tab is not composed
    when disabled) — the tool reported success and the chart rendered NOWHERE. Deriving the
    offer from the same flags ``app.py`` composes from makes that state unreachable.

    The Dashboard is always included even if its flag were somehow off: a chart must always
    have a destination, and ``dashboard`` is the documented home of an untargeted request.
    """
    from ..feature_flags import is_tab_enabled

    enabled = [t for t in CHART_HOSTABLE_TABS if _tab_on(is_tab_enabled, t)]  # nosemgrep: is-function-without-parentheses
    if _DEFAULT_TARGET not in enabled:
        enabled.insert(0, _DEFAULT_TARGET)
    return tuple(enabled)


def _tab_on(is_tab_enabled: Any, tab: str) -> bool:
    """``is_tab_enabled`` guarded — a flag-lookup failure must not break chart routing."""
    try:
        return bool(is_tab_enabled(tab))
    except Exception:
        return False


class ChartViewPanel(Static):
    """Renders an agent-requested chart addressed to ``tab`` (and ignores others)."""

    def __init__(self, *args: Any, tab: str = _DEFAULT_TARGET, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._tab = (tab or _DEFAULT_TARGET).strip().lower()
        self._text: str = self.placeholder(self._tab)

    @staticmethod
    def placeholder(tab: str) -> str:
        """The idle body for ``tab`` — states HOW to populate it, not just that it is empty."""
        return (
            f"📊 Agent Charts — {tab}\n━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f'  Ask in Chat, e.g. "chart the leaderboard on {tab}".'
        )

    @property
    def tab(self) -> str:
        """The tab this panel instance is bound to."""
        return self._tab

    def render_text(self) -> str:
        """Current body (test/inspection seam — no reliance on Textual internals)."""
        return self._text

    # ─── state → text (pure; no Textual app required) ────────────────────────

    def claims(self, view: Any) -> bool:
        """True when ``view`` is addressed to THIS panel's tab.

        An empty ``target`` is claimed by the Dashboard only, so an ADR-0006-era request
        (no target) behaves exactly as before instead of appearing on every tab.
        """
        chart = str(getattr(view, "chart", "") or "").strip()
        if not chart:
            return False
        target = str(getattr(view, "target", "") or "").strip().lower() or _DEFAULT_TARGET
        return target == self._tab

    def apply_view(self, state: AppContextState) -> None:
        """Re-render from ``state`` — the chart when claimed, else this tab's placeholder."""
        view = getattr(state, "view", None)
        if view is None or not self.claims(view):
            self._set_text(self.placeholder(self._tab))
            return
        self._set_text(self._body_for(view))

    def _body_for(self, view: Any) -> str:
        """Render ``view`` via the pure chart renderers (never raises)."""
        from .charts import (
            render_bar,
            render_confusion_matrix,
            render_heatmap,
            render_histogram,
            render_line,
            render_scatter,
        )

        chart = str(getattr(view, "chart", "") or "").lower()
        title = str(getattr(view, "title", "") or "Agent Chart")
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
                # Name the unsupported kind rather than rendering an empty frame, so the
                # operator can see WHY nothing was plotted.
                body = f"{title}\n  (unsupported chart {chart!r})"
        except Exception:
            body = f"{title}\n  (chart could not be rendered)"

        detail = str(getattr(view, "detail", "") or "")
        return f"{body}\n  source: {detail}" if detail else body

    def _set_text(self, text: str) -> None:
        """Store + push the body to the child ``Static`` (guarded pre-mount/headless)."""
        self._text = text
        try:
            self.query_one(f"#{self._content_id()}", Static).update(text)
        except Exception:  # nosec B110
            pass

    def _content_id(self) -> str:
        return f"chart-view-{self._tab}"

    # ─── Textual wiring (mirrors the Phase-4/5 pane discipline) ──────────────

    def compose(self) -> ComposeResult:
        with Vertical():
            # markup=False → chart glyphs and dynamic titles render literally.
            yield Static(self._text, id=self._content_id(), markup=False)

    def on_mount(self) -> None:
        """Subscribe to the bus and prime. Null-safe: no bus (headless) → no-op."""
        bus = self._bus()
        if bus is None:
            return
        try:
            bus.subscribe(self._on_context)
            self.apply_view(bus.state())
        except Exception:  # nosec B110
            pass

    def _bus(self) -> Any | None:
        try:
            return getattr(self.app, "context_bus", None)
        except Exception:
            return None

    def _on_context(self, event: ContextEvent, state: AppContextState) -> None:
        """Repaint on a ``view`` event, marshalled to the UI thread. Never raises."""
        if event != "view":
            return
        try:
            app = self.app
        except Exception:
            app = None
        if app is None:
            self.apply_view(state)
            return
        try:
            app.call_from_thread(self.apply_view, state)
        except Exception:
            try:
                self.apply_view(state)
            except Exception:  # nosec B110
                pass
