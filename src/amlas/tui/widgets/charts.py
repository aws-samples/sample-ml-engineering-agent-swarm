"""Terminal ML chart renderers — the standard chart family, as pure functions (plan §B1).

**The gap this closes.** Before this module the TUI had exactly ONE chart: a Textual
``Sparkline`` in [`dynamic_primary.py`](dynamic_primary.py) driven by ``ActiveJob.progress``
— i.e. *job completion percentage*, not a model metric. Meanwhile the real numbers sat
unplotted in ``EvaluationStore.metrics`` and each run's ``arbiter_report.json``. A data
scientist asking "show me the regression fit" or "show me the correlation heat map" had
nowhere for that to land.

**Design choices, and why:**

* **Pure functions returning ``str``.** No Textual import, no widget state, no I/O. That
  makes every chart directly directly verifiable (see ``validation coverage``) and safe to
  call from a widget's render path — a chart can never crash a repaint.
* **No new dependency.** ``plotext`` / ``textual-plotext`` are not installed and
  ``matplotlib`` (which is) cannot render into a terminal. Everything here is stdlib +
  Unicode block/braille glyphs, so it works in any terminal the TUI already supports.
* **Never raises, never lies.** Non-finite values (NaN/inf) are dropped rather than
  plotted as zero; an empty series renders an explicit "no data" line instead of an empty
  frame that would read as "the metric is zero". Degenerate inputs (single point, flat
  series, identical x values, ragged matrix, 1×1 canvas) all return a string.

Chart family: :func:`render_line` (metric curves), :func:`render_scatter` (predicted-vs-
actual + least-squares fit and R²), :func:`render_heatmap` (correlation matrices),
:func:`render_confusion_matrix`, :func:`render_bar` (leaderboard / feature importance),
:func:`render_histogram` (distributions).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

__all__ = [
    "render_bar",
    "render_confusion_matrix",
    "render_heatmap",
    "render_histogram",
    "render_line",
    "render_scatter",
]

# Per-series plot marks, in assignment order. Visually distinct in a terminal font so
# overlaid series stay readable without colour (which a plain string cannot carry).
_SERIES_MARKS = ("•", "×", "▲", "◆", "○", "+")

# Heat-map shading ramp, lowest → highest magnitude. Using distinct glyph *shapes* (not
# just density) keeps the ramp legible on terminals with unusual block metrics.
_SHADE_RAMP = (" ", "·", "░", "▒", "▓", "█")

# The empty-state marker. Tests assert on "no data", and it must be unmistakable in the UI
# so an empty chart is never misread as a zero-valued metric.
_NO_DATA = "  (no data yet)"


def _finite(values: Iterable[object]) -> list[float]:
    """Coerce an iterable to the finite floats within it (dropping NaN/inf/non-numerics).

    Dropping rather than zero-filling is deliberate: plotting a NaN as 0.0 would invent a
    data point that the model never produced.
    """
    out: list[float] = []
    for value in values or []:
        try:
            number = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            out.append(number)
    return out


def _fmt(value: float) -> str:
    """Format an axis/value label compactly (integers without a decimal tail)."""
    if not math.isfinite(value):
        return "—"
    if abs(value - round(value)) < 1e-9 and abs(value) < 1e6:
        return str(round(value))
    if abs(value) >= 1000:
        return f"{value:.0f}"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _header(title: str, width: int) -> list[str]:
    """Build the chart's title line, truncated to ``width``."""
    text = (title or "").strip() or "Chart"
    return [text[: max(1, width)]]


def _clamp_dims(width: int, height: int) -> tuple[int, int]:
    """Clamp a requested canvas to something plottable (callers may pass 1×1)."""
    return max(4, int(width or 0)), max(2, int(height or 0))


def render_line(
    series: Mapping[str, Sequence[float] | None] | None,
    *,
    title: str = "",
    width: int = 60,
    height: int = 12,
) -> str:
    """Render one or more metric curves as an ASCII line chart.

    The chart for "show me the training curve" / "plot f1 across trials". Each series gets
    its own mark from :data:`_SERIES_MARKS` and is named in the legend, so multiple metrics
    (train vs val) overlay readably without colour.

    Args:
        series: ``{name: [values...]}``. Empty/None series are skipped; a series with a
            single point renders as one mark.
        title: Chart title (always rendered, even for the empty state).
        width: Total character width — no output line exceeds it.
        height: Plot-body height in rows.

    Returns:
        A multi-line string: title, plot body with y-axis labels, then the legend.
    """
    width, height = _clamp_dims(width, height)
    lines = _header(title, width)

    cleaned = {
        str(name): _finite(values or [])
        for name, values in (series or {}).items()
        if _finite(values or [])
    }
    if not cleaned:
        return "\n".join([*lines, _NO_DATA])

    all_values = [v for values in cleaned.values() for v in values]
    y_min, y_max = min(all_values), max(all_values)
    if y_max - y_min < 1e-12:  # flat series → give the band a visible thickness
        y_max = y_min + 1.0

    # Reserve the left gutter for y labels, leaving the rest for the plot body.
    gutter = min(max(len(_fmt(y_min)), len(_fmt(y_max))) + 1, max(1, width - 3))
    plot_w = max(1, width - gutter - 1)
    longest = max(len(values) for values in cleaned.values())

    grid = [[" "] * plot_w for _ in range(height)]
    for index, (_, values) in enumerate(sorted(cleaned.items())):
        mark = _SERIES_MARKS[index % len(_SERIES_MARKS)]
        for position, value in enumerate(values):
            col = 0 if longest == 1 else int(position * (plot_w - 1) / max(1, longest - 1))
            frac = (value - y_min) / (y_max - y_min)
            row = height - 1 - round(frac * (height - 1))
            grid[min(max(row, 0), height - 1)][min(max(col, 0), plot_w - 1)] = mark

    # NB: distinct loop name — ``row`` above is an int grid index, not the row list.
    for row_index, grid_row in enumerate(grid):
        if row_index == 0:
            label = _fmt(y_max)
        elif row_index == height - 1:
            label = _fmt(y_min)
        else:
            label = ""
        lines.append(f"{label:>{gutter}}│{''.join(grid_row)}"[:width])

    legend_marks = [
        f"{_SERIES_MARKS[i % len(_SERIES_MARKS)]} {name}"
        for i, (name, _) in enumerate(sorted(cleaned.items()))
    ]
    lines.append(("  " + "  ".join(legend_marks))[:width])
    return "\n".join(lines)


def _least_squares(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    """Return ``(slope, intercept, r_squared)`` for a least-squares fit.

    Degenerate inputs (fewer than 2 points, zero x-variance, zero y-variance) return a
    horizontal fit at the mean with ``r_squared = 0.0`` rather than raising — a vertical
    or constant relationship has no meaningful linear fit to report.
    """
    n = len(xs)
    if n < 2:
        return 0.0, (ys[0] if ys else 0.0), 0.0
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    var_x = sum((x - mean_x) ** 2 for x in xs)
    if var_x < 1e-12:
        return 0.0, mean_y, 0.0
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)) / var_x
    intercept = mean_y - slope * mean_x
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    if ss_tot < 1e-12:
        return slope, intercept, 0.0
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys, strict=True))
    return slope, intercept, max(0.0, min(1.0, 1.0 - ss_res / ss_tot))


def render_scatter(
    xs: Sequence[float] | None,
    ys: Sequence[float] | None,
    *,
    title: str = "",
    width: int = 60,
    height: int = 12,
) -> str:
    """Render a scatter plot with a least-squares regression line and R².

    The "regression chart" a data scientist asks for after a regression run: predicted vs
    actual, with the fit quality quantified rather than eyeballed. Points are ``•`` and the
    fitted line is ``─``; where they coincide the point wins (real data over the model).

    Args:
        xs: X values (e.g. actual).
        ys: Y values (e.g. predicted). Must be the same length as ``xs``.
        title: Chart title.
        width: Total character width.
        height: Plot-body height in rows.

    Returns:
        A multi-line string; the footer carries ``R²``, slope, and intercept. A
        length mismatch returns an explicit "length mismatch" line rather than truncating
        (silently dropping points would misstate the fit).
    """
    width, height = _clamp_dims(width, height)
    lines = _header(title, width)

    raw_x, raw_y = list(xs or []), list(ys or [])
    if len(raw_x) != len(raw_y):
        return "\n".join([*lines, f"  (length mismatch: {len(raw_x)} x vs {len(raw_y)} y)"])

    pairs = [
        (float(x), float(y))
        for x, y in zip(raw_x, raw_y, strict=True)
        if _finite([x]) and _finite([y])
    ]
    if not pairs:
        return "\n".join([*lines, _NO_DATA])

    px = [p[0] for p in pairs]
    py = [p[1] for p in pairs]
    slope, intercept, r2 = _least_squares(px, py)

    x_min, x_max = min(px), max(px)
    y_min, y_max = min(py), max(py)
    if x_max - x_min < 1e-12:
        x_max = x_min + 1.0
    if y_max - y_min < 1e-12:
        y_max = y_min + 1.0

    gutter = min(max(len(_fmt(y_min)), len(_fmt(y_max))) + 1, max(1, width - 3))
    plot_w = max(1, width - gutter - 1)
    grid = [[" "] * plot_w for _ in range(height)]

    def _cell(x: float, y: float) -> tuple[int, int]:
        col = int((x - x_min) / (x_max - x_min) * (plot_w - 1))
        row = height - 1 - round((y - y_min) / (y_max - y_min) * (height - 1))
        return min(max(row, 0), height - 1), min(max(col, 0), plot_w - 1)

    # Fit line first so real observations render on top of the model's guess.
    for col in range(plot_w):
        x = x_min + (x_max - x_min) * (col / max(1, plot_w - 1))
        row, _ = _cell(x, slope * x + intercept)
        grid[row][col] = "─"
    for x, y in pairs:
        row, col = _cell(x, y)
        grid[row][col] = "•"

    # NB: distinct loop name — ``row`` above holds an int grid index from ``_cell``, and
    # reusing it here for the row LIST silently changes the variable's type.
    for row_index, grid_row in enumerate(grid):
        if row_index == 0:
            label = _fmt(y_max)
        elif row_index == height - 1:
            label = _fmt(y_min)
        else:
            label = ""
        lines.append(f"{label:>{gutter}}│{''.join(grid_row)}"[:width])

    lines.append(f"  R²={r2:.2f}  slope={_fmt(slope)}  intercept={_fmt(intercept)}"[:width])
    return "\n".join(lines)


def _shade(value: float, low: float, high: float) -> str:
    """Map ``value`` into the :data:`_SHADE_RAMP` band for its magnitude."""
    if high - low < 1e-12:
        return _SHADE_RAMP[len(_SHADE_RAMP) // 2]
    frac = (value - low) / (high - low)
    index = round(max(0.0, min(1.0, frac)) * (len(_SHADE_RAMP) - 1))
    return _SHADE_RAMP[index]


def render_heatmap(
    matrix: Sequence[Sequence[float]] | None,
    *,
    labels: Sequence[str] | None = None,
    title: str = "",
    width: int = 60,
) -> str:
    """Render a matrix as a shaded heat map with row/column labels and a scale legend.

    The chart for "show me the correlation matrix" — the standard way to spot collinear
    features before training. Ragged rows are padded so a partially-computed matrix still
    renders instead of raising.

    Args:
        matrix: Row-major 2-D values (e.g. a correlation matrix).
        labels: Optional axis labels; generated as ``c0..cN`` when omitted, and padded or
            truncated to the matrix size so a mismatch cannot raise.
        title: Chart title.
        width: Total character width.

    Returns:
        A multi-line string: title, a column-label header, one row per matrix row, then a
        legend giving the value range the shading spans.
    """
    width, _ = _clamp_dims(width, 2)
    lines = _header(title, width)

    rows = [list(row or []) for row in (matrix or [])]
    rows = [row for row in rows if row]
    if not rows:
        return "\n".join([*lines, _NO_DATA])

    n_cols = max(len(row) for row in rows)
    grid = [[*_pad_row(row, n_cols)] for row in rows]

    names = [str(n) for n in (labels or [])]
    if len(names) < max(len(grid), n_cols):
        names += [f"c{i}" for i in range(len(names), max(len(grid), n_cols))]

    flat = _finite([v for row in grid for v in row])
    low, high = (min(flat), max(flat)) if flat else (0.0, 1.0)

    # Row labels share one gutter so the columns line up under the header.
    gutter = min(max((len(n) for n in names[: len(grid)]), default=1), max(3, width // 4))
    header_cells = "".join(str(names[c])[:1] for c in range(n_cols))
    lines.append(f"{'':>{gutter}} {header_cells}"[:width])

    for r, row in enumerate(grid):
        cells = "".join(
            _shade(v, low, high) if (v is not None and math.isfinite(v)) else "?" for v in row
        )
        lines.append(f"{str(names[r])[:gutter]:>{gutter}} {cells}"[:width])

    lines.append(f"  scale {_fmt(low)} {_SHADE_RAMP[1]}→{_SHADE_RAMP[-1]} {_fmt(high)}"[:width])
    # Column names in full, so a one-character header is still decodable.
    lines.append(("  cols: " + ", ".join(names[:n_cols]))[:width])
    return "\n".join(lines)


def _pad_row(row: Sequence[object], size: int) -> list[float | None]:
    """Pad/convert a matrix row to exactly ``size`` floats (``None`` where unusable)."""
    out: list[float | None] = []
    for index in range(size):
        if index >= len(row):
            out.append(None)
            continue
        try:
            out.append(float(row[index]))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            out.append(None)
    return out


def render_confusion_matrix(
    matrix: Sequence[Sequence[float]] | None,
    *,
    labels: Sequence[str] | None = None,
    title: str = "",
    width: int = 60,
) -> str:
    """Render a confusion matrix as a labelled count table (actual rows × predicted cols).

    Counts are shown as NUMBERS, not shading: for a confusion matrix the exact cell values
    are the point (how many failures were missed), so shading would destroy the signal.

    Args:
        matrix: Row-major counts; row = actual class, column = predicted class.
        labels: Class names; generated as ``c0..cN`` when omitted.
        title: Chart title.
        width: Total character width.

    Returns:
        A multi-line string with explicit ``actual`` / ``pred`` axis annotations.
    """
    width, _ = _clamp_dims(width, 2)
    lines = _header(title, width)

    rows = [list(row or []) for row in (matrix or [])]
    rows = [row for row in rows if row]
    if not rows:
        return "\n".join([*lines, _NO_DATA])

    n_cols = max(len(row) for row in rows)
    names = [str(n) for n in (labels or [])]
    if len(names) < max(len(rows), n_cols):
        names += [f"c{i}" for i in range(len(names), max(len(rows), n_cols))]

    cell_w = 6
    gutter = max((len(n) for n in names[: len(rows)]), default=3) + 8  # room for "actual "
    lines.append("  pred →")
    lines.append(f"{'':>{gutter}}" + "".join(f"{names[c][:cell_w]:>{cell_w}}" for c in range(n_cols)))
    for r, row in enumerate(rows):
        values = _pad_row(row, n_cols)
        cells = "".join(
            f"{(_fmt(v) if v is not None else '—'):>{cell_w}}" for v in values
        )
        lines.append(f"{('actual ' + names[r]):>{gutter}}{cells}")
    return "\n".join(lines)


def render_bar(
    values: dict[str, float] | None,
    *,
    title: str = "",
    width: int = 60,
) -> str:
    """Render a horizontal bar chart, sorted highest-first, with numeric labels.

    The leaderboard / feature-importance chart. Bars are scaled by absolute magnitude so a
    negative value (e.g. a SHAP contribution) still renders a proportional bar, with the
    signed number shown alongside.

    Args:
        values: ``{label: value}``.
        title: Chart title.
        width: Total character width — no output line exceeds it.

    Returns:
        A multi-line string, one row per entry, ordered descending by value.
    """
    width, _ = _clamp_dims(width, 2)
    lines = _header(title, width)

    cleaned: dict[str, float] = {}
    for name, value in (values or {}).items():
        finite = _finite([value])
        if finite:
            cleaned[str(name)] = finite[0]
    if not cleaned:
        return "\n".join([*lines, _NO_DATA])

    ordered = sorted(cleaned.items(), key=lambda kv: kv[1], reverse=True)
    label_w = min(max(len(name) for name, _ in ordered), max(3, width // 3))
    value_w = max(len(_fmt(v)) for _, v in ordered)
    bar_w = max(1, width - label_w - value_w - 3)
    scale = max(abs(v) for _, v in ordered) or 1.0

    for name, value in ordered:
        filled = round(abs(value) / scale * bar_w)
        bar = "█" * min(filled, bar_w)
        lines.append(f"{name[:label_w]:<{label_w}} {bar:<{bar_w}} {_fmt(value):>{value_w}}"[:width])
    return "\n".join(lines)


def render_histogram(
    values: Sequence[float] | None,
    *,
    bins: int = 10,
    title: str = "",
    width: int = 60,
) -> str:
    """Render a value distribution as a binned horizontal histogram.

    The distribution chart behind "is this feature skewed?" / "show me the target
    distribution". Bin edges are labelled with real values so the shape is interpretable.

    Args:
        values: The raw values (non-finite entries are dropped).
        bins: Requested bin count (clamped to ``[1, 50]``).
        title: Chart title.
        width: Total character width.

    Returns:
        A multi-line string: title, one row per bin (edge label + bar + count), then the
        observed min/max range.
    """
    width, _ = _clamp_dims(width, 2)
    lines = _header(title, width)

    data = _finite(values or [])
    if not data:
        return "\n".join([*lines, _NO_DATA])

    n_bins = max(1, min(int(bins or 1), 50))
    low, high = min(data), max(data)
    span = high - low
    counts = [0] * n_bins
    for value in data:
        # An all-identical dataset has zero span → everything lands in bin 0.
        index = 0 if span < 1e-12 else min(int((value - low) / span * n_bins), n_bins - 1)
        counts[index] += 1

    edge_labels = [
        _fmt(low + (span * i / n_bins)) if span >= 1e-12 else _fmt(low) for i in range(n_bins)
    ]
    label_w = min(max(len(e) for e in edge_labels), max(3, width // 4))
    count_w = max(len(str(c)) for c in counts)
    bar_w = max(1, width - label_w - count_w - 3)
    peak = max(counts) or 1

    for edge, count in zip(edge_labels, counts, strict=True):
        bar = "█" * round(count / peak * bar_w)
        lines.append(f"{edge[:label_w]:>{label_w}} {bar:<{bar_w}} {count:>{count_w}}"[:width])

    lines.append(f"  n={len(data)}  range {_fmt(low)}..{_fmt(high)}"[:width])
    return "\n".join(lines)
