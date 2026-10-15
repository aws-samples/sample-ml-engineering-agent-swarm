"""``SteeringProfile`` — the domain-steering "dial" (master plan §3.1, §11; spec 1.1).

A Pydantic model (``extra="forbid"``) loaded from a YAML file or the spec
``imputation.steering`` block. It carries three classes of steering signal:

* **Hard constraints** — ``forbid_methods``, ``pin_methods`` (per column *or*
  dtype), ``mechanism_hints``, ``max_llm_fraction``, ``require_evaluation``.
  Enforced *deterministically in the tools* (e.g. ``apply_imputation_plan``);
  the agent cannot override them.
* **Soft preferences** — ``preferred_methods`` (ordering hint), ``domain_notes``
  (free text injected into the Strategist prompt), ``novelty_budget`` (how many
  novel agent-proposed strategies are allowed beyond the defaults).
* **Acceptance thresholds** — ``thresholds`` (recovery_rmse, recovery_ks,
  fidelity, utility_delta, detection_auroc, privacy_dcr) consumed by the
  deterministic graders.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SteeringProfile(BaseModel):
    """Domain-steering profile for the imputation / synthetic engines."""

    model_config = ConfigDict(extra="forbid")

    # ── hard constraints (enforced in tools) ─────────────────────────────────
    forbid_methods: list[str] = Field(default_factory=list)
    pin_methods: dict[str, str] = Field(default_factory=dict)  # {column_or_dtype: method}
    mechanism_hints: dict[str, str] = Field(default_factory=dict)  # {column: MCAR|MAR|MNAR}
    max_llm_fraction: float = 0.25  # cap LLM-filled cells (0..1)
    require_evaluation: bool = True

    # ── soft preferences (flow into prompts) ─────────────────────────────────
    preferred_methods: list[str] = Field(default_factory=list)
    domain_notes: str = ""
    novelty_budget: int = 2

    # ── acceptance thresholds (feed the graders) ─────────────────────────────
    thresholds: dict[str, float] = Field(default_factory=dict)

    # ── hard-constraint helpers ───────────────────────────────────────────────
    def is_forbidden(self, method: str) -> bool:
        """True if ``method`` is on the hard ``forbid_methods`` list (case-insensitive)."""
        m = str(method).strip().lower()
        return any(m == str(f).strip().lower() for f in self.forbid_methods)

    def pin_for(self, column: str, dtype: str | None = None) -> str | None:
        """Return the pinned method for ``column`` (column pin beats dtype pin).

        Args:
            column: The column name.
            dtype: Optional dtype string (e.g. ``"float"``, ``"datetime64[ns]"``).

        Returns:
            The pinned method name, or ``None`` when neither a column nor dtype pin
            applies. A column-specific pin always wins over a dtype-level pin.
        """
        if column in self.pin_methods:
            return self.pin_methods[column]
        if dtype is not None:
            # exact dtype key first, then a coarse family key (e.g. "float").
            if dtype in self.pin_methods:
                return self.pin_methods[dtype]
            family = _dtype_family(dtype)
            if family in self.pin_methods:
                return self.pin_methods[family]
        return None

    def threshold(self, name: str, default: float) -> float:
        """Return ``thresholds[name]`` if present, else ``default``."""
        v = self.thresholds.get(name)
        return float(v) if v is not None else float(default)

    def mechanism_hint(self, column: str) -> str | None:
        """Return an explicit MCAR/MAR/MNAR hint for ``column`` if pinned."""
        h = self.mechanism_hints.get(column)
        return str(h).upper() if h is not None else None

    # ── loaders / serializers ─────────────────────────────────────────────────
    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> SteeringProfile:
        """Build a profile from a plain dict; ``None`` yields an empty profile."""
        if not data:
            return cls()
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str) -> SteeringProfile:
        """Load a profile from a YAML file (raises ``FileNotFoundError`` if absent)."""
        import yaml

        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Steering profile not found: {path}")
        data = yaml.safe_load(p.read_text()) or {}
        return cls(**data)

    def as_profile_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict the ``@tool`` functions consume as ``profile=``."""
        return self.model_dump()

    def as_prompt_block(self) -> str:
        """Render the SOFT half of the dial for prompt injection (master plan §3.1).

        The HARD constraints (``forbid_methods`` / ``pin_methods`` / ``thresholds``)
        are enforced deterministically *in the tools*, so they are listed here only
        for the agent's awareness; the SOFT preferences (``preferred_methods``,
        ``domain_notes``, ``novelty_budget``) are what actually steer the agent's
        search. Returns an empty string when the profile carries no signal so the
        Lead prompt stays clean for the no-steering path.
        """
        lines: list[str] = []
        if self.domain_notes:
            lines.append(f"- Domain notes (literal): {self.domain_notes}")
        if self.preferred_methods:
            lines.append(f"- Preferred methods (try first): {', '.join(self.preferred_methods)}")
        if self.forbid_methods:
            lines.append(
                f"- Forbidden methods (HARD — enforced in tools): {', '.join(self.forbid_methods)}"
            )
        if self.pin_methods:
            pins = ", ".join(f"{k}→{v}" for k, v in self.pin_methods.items())
            lines.append(f"- Pinned methods (HARD — enforced in tools): {pins}")
        if self.mechanism_hints:
            hints = ", ".join(f"{k}={v}" for k, v in self.mechanism_hints.items())
            lines.append(f"- Mechanism hints (HARD — honored in scan): {hints}")
        if self.thresholds:
            thr = ", ".join(f"{k}={v}" for k, v in self.thresholds.items())
            lines.append(f"- Acceptance thresholds (HARD — feed the graders): {thr}")
        if not lines:
            # No active steering signal — keep the Lead prompt clean (no-dial path).
            return ""
        # novelty_budget is always meaningful once *some* steering is supplied.
        lines.append(
            f"- Novelty budget (novel strategies allowed beyond defaults): {self.novelty_budget}"
        )
        return "## Active steering for this dataset\n\n" + "\n".join(lines)


def _dtype_family(dtype: str) -> str:
    """Map a concrete dtype string to a coarse pin-family key.

    ``"float64"`` -> ``"float"``, ``"int32"`` -> ``"int"``,
    ``"datetime64[ns]"`` -> ``"datetime"``, ``"object"`` -> ``"object"``.
    """
    d = str(dtype).lower()
    if d.startswith("float"):
        return "float"
    if d.startswith("int") or d.startswith("uint"):
        return "int"
    if d.startswith("datetime") or d.startswith("date"):
        return "datetime"
    if d.startswith("bool"):
        return "bool"
    return d
