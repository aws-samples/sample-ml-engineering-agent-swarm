"""Conservative cost presentation — the ``CostBand`` formatter (doc 05 §4/§5; design §6).

**The problem this closes.** Cost estimation across the surface is a static price book of
round-number guesses (:mod:`amlas.control_plane.plan_mode` ``_FLOW_TEMPLATES``,
:mod:`amlas.control_plane.cost_governance` ``_PRICE_BOOK`` + the ``_DEFAULT_USD = 0.05``
fallback). Those single-point numbers were surfaced as confident precise dollar figures — a
``$0.05`` guess reads as a fact. The user's directive was blunt: *"be cautious with cost
estimates we seem to suck at it."*

**What this module does (and, importantly, does NOT do).** It is a pure, deterministic
PRESENTATION formatter that turns any point estimate into a caveated, **upward-biased range**
with a confidence + source label. It changes ONLY what a human SEES. It does **not** touch any
gate math: the approval gate (:func:`~amlas.control_plane.approvals.request_approval`) and the
budget ceiling (:meth:`~amlas.control_plane.cost_governance.LocalCostAdvisor.check_budget`)
still consume the RAW point estimate, so the safety posture is byte-identical (the T-CC-07
regression guards this). The band is additive metadata alongside the raw figure, never a
replacement for it.

**The rules** (:func:`band_for`):

* ``source in {"cost_explorer", "aggregate"}`` → ``confidence="actual"``: a REAL figure
  (Cost-Explorer spend or a recorded-ledger roll-up), shown precisely (``low==high==point``).
* ``source="estimate"`` (or ``"sizing"``) and ``point == _DEFAULT_USD`` → ``confidence="unknown"``:
  the fallback guess. Rendered as "unknown, could be materially higher" — NEVER as ``$0.05``.
* ``source="estimate"`` otherwise → ``confidence="rough"``: a price-book guess, widened
  UPWARD (``low = 0.5×point``, ``high = 4×point``) so we under-promise / over-deliver.
* a genuine ``0.0`` estimate (a read-only / monitor flow) renders as "no expected spend" — not
  a misleading upward range.

:meth:`CostBand.human` for any non-actual, non-zero band always contains a range separator and
a caveat word — never a lone ``"$X.XX"``. :meth:`CostBand.to_dict` is the JSON-safe shape the
``@tool`` results embed (the chat agent serializes it to the model).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "ACTUAL_CAVEAT",
    "DEFAULT_HIGH_MULT",
    "DEFAULT_LOW_MULT",
    "ROUGH_CAVEAT",
    "UNKNOWN_CAVEAT",
    "CostBand",
    "band_for",
]

# ── multipliers: bias the band UPWARD, never downward (design decision §9) ────
# The user said estimates are unreliable; under-promising is the safe failure mode. The high
# bound is a deliberate 4× so the presented ceiling over-covers a wrong-low guess. The gate math
# still uses the point estimate, so this never loosens a budget ceiling.
DEFAULT_LOW_MULT = 0.5      # low bound = 0.5× point (bias down only a little)
DEFAULT_HIGH_MULT = 4.0     # high bound = 4× point (bias UP a lot — under-promise / over-deliver)

# ── human caveats (the explicit uncertainty wording) ──────────────────────────
UNKNOWN_CAVEAT = "No reliable estimate — treat as unknown; could be materially higher."
ROUGH_CAVEAT = "Rough estimate only; actual AWS cost may be several times higher."
ACTUAL_CAVEAT = "Actual spend from Cost Explorer for the window shown."

# Sources that represent a REAL measured figure (not a price-book guess).
_ACTUAL_SOURCES = frozenset({"cost_explorer", "aggregate"})

# The price-book fallback constant that means "we have no idea" — imported lazily to avoid a
# hard import cycle (cost_governance imports nothing from here, but keep it defensive/cheap).
try:  # pragma: no cover - trivial import guard
    from .cost_governance import _DEFAULT_USD
except Exception:  # pragma: no cover - defensive; keep the design's confirmed value
    _DEFAULT_USD = 0.05


def _fmt(usd: float) -> str:
    """Format a USD figure the way the surfaces do (2 dp, leading ``$``)."""
    return f"${usd:.2f}"


@dataclass(frozen=True)
class CostBand:
    """A caveated, source-labeled cost band — the human-facing view of a point estimate.

    Attributes:
        low: The lower bound of the presented range (== ``point`` for an actual).
        high: The upper (upward-biased) bound (== ``point`` for an actual).
        point: The RAW point estimate the gate math uses (unchanged, echoed for provenance).
        confidence: ``"actual"`` (real spend) | ``"rough"`` (price-book guess) |
            ``"unknown"`` (the ``_DEFAULT_USD`` fallback).
        source: The originating source label (``"cost_explorer"`` | ``"aggregate"`` |
            ``"price_book_estimate"`` | ``"default_fallback"`` | ``"sizing"`` | …).
        caveat: The human uncertainty sentence (one of the module ``*_CAVEAT`` constants).
    """

    low: float
    high: float
    point: float
    confidence: str
    source: str
    caveat: str

    def human(self) -> str:
        """Render the band as a single human-facing sentence.

        * ``actual`` → a precise figure labeled "actual (Cost Explorer)".
        * a genuine ``0.0`` estimate → "no expected spend" (never a fake range).
        * otherwise → an upward-biased RANGE + the caveat word — never a lone ``$X.XX``.
        """
        if self.confidence == "actual":
            return f"{_fmt(self.point)} actual (Cost Explorer) — {self.caveat}"
        # A genuinely-free flow (read-only / monitor): no spend to caveat.
        if self.point == 0.0 and self.low == 0.0 and self.high == 0.0:
            return "no expected spend ($0.00)"
        if self.confidence == "unknown":
            # Deliberately DO NOT print the misleadingly-precise fallback figure.
            return f"cost unknown (rough guess ~{_fmt(self.high)} or less) — {self.caveat}"
        # rough price-book estimate → an explicit upward range + caveat.
        return f"~{_fmt(self.low)}–{_fmt(self.high)} ({self.caveat})"

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict (the shape a ``@tool`` result embeds; ``human`` pre-rendered).

        The chat agent serializes tool results to the model, so the band must be plain JSON —
        never a frozen dataclass. The pre-rendered ``human`` string is what surfaces quote.
        """
        return {
            "low": self.low,
            "high": self.high,
            "point": self.point,
            "confidence": self.confidence,
            "source": self.source,
            "caveat": self.caveat,
            "human": self.human(),
        }


def band_for(estimate: Any, *, source: str | None = None, confidence: str | None = None) -> CostBand:
    """Turn a point estimate into a caveated, upward-biased :class:`CostBand`.

    Accepts EITHER a :class:`~amlas.control_plane.cost_governance.CostEstimate` (``.usd`` +
    ``.source``) OR a bare numeric point with an explicit ``source=`` kwarg (the render-layer
    convenience). ``confidence`` may be forced but is normally derived from ``source`` + whether
    the point is the ``_DEFAULT_USD`` fallback.

    Args:
        estimate: A ``CostEstimate`` (uses its ``.usd``/``.source``) or a bare ``float`` point.
        source: The source label when ``estimate`` is a bare number (ignored when ``estimate``
            carries its own ``.source``, unless that is empty).
        confidence: Optional explicit override of the derived confidence.

    Returns:
        A :class:`CostBand`. Actuals are precise (``low==high==point``); rough estimates widen
        upward (``0.5×``–``4×``); the ``_DEFAULT_USD`` fallback is ``unknown``.
    """
    # Resolve the raw point + source from either shape.
    point_val = getattr(estimate, "usd", estimate)
    try:
        point = round(float(point_val), 4)
    except (TypeError, ValueError):
        point = 0.0
    src = (getattr(estimate, "source", None) or source or "estimate")
    src = str(src).strip().lower()

    # ── actuals: a real measured figure — shown precisely, labeled "actual" ──
    if src in _ACTUAL_SOURCES:
        return CostBand(
            low=point, high=point, point=point,
            confidence=confidence or "actual",
            source=src, caveat=ACTUAL_CAVEAT,
        )

    # ── the _DEFAULT_USD fallback guess → unknown (never render "$0.05") ─────
    is_fallback = abs(point - float(_DEFAULT_USD)) < 1e-9
    if is_fallback:
        return CostBand(
            low=round(point * DEFAULT_LOW_MULT, 4),
            high=round(point * DEFAULT_HIGH_MULT, 4),
            point=point,
            confidence=confidence or "unknown",
            source="default_fallback",
            caveat=UNKNOWN_CAVEAT,
        )

    # ── a genuine zero-spend flow (read-only / monitor) → no range at all ────
    if point == 0.0:
        return CostBand(
            low=0.0, high=0.0, point=0.0,
            confidence=confidence or "rough",
            source="price_book_estimate",
            caveat=ROUGH_CAVEAT,
        )

    # ── the common case: a rough price-book estimate → widen UPWARD ──────────
    return CostBand(
        low=round(point * DEFAULT_LOW_MULT, 4),
        high=round(point * DEFAULT_HIGH_MULT, 4),
        point=point,
        confidence=confidence or "rough",
        source="price_book_estimate",
        caveat=ROUGH_CAVEAT,
    )
