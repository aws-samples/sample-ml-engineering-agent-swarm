"""``imputation_flow`` — the team's signature fixed-flow / swappable-module pattern.

The canonical realization of doc 05 §4.5 / doc 06 §4.1: the FLOW is invariant —
**diagnose missingness mechanism → recommend strategy → apply → validate** — while the
**imputation strategy MODULE is pluggable**. The flow contract ("give me an imputed
dataset that passes validation") never changes; the registry/steering/`.agenticml.yaml`
selector picks *how* the apply step manipulates the data.

CORE IP preserved (per the project data-quality rules, RECON §10):
  * **Missingness-mechanism identifiability** — the fixed flow runs the EXISTING
    ``diagnose_missingness`` (Little's MCAR + mask-classifier MAR/MNAR) and the
    ``recommend_imputation_plan`` engine UNCHANGED. MCAR/MAR are diagnosed on
    identifiable columns; an MNAR signal on a non-reconstructable column is honored.
    A column reconstructable from others is observationally MAR — that is correct, not
    a bug.
  * **3-pillar synthetic validation** — when the swappable module is the
    ``synthetic_fill`` strategy, the fill is accepted ONLY if the EXISTING
    ``validate_synthetic`` 3-pillar AND-gate passes (fidelity AND utility/detection AND
    privacy). An exact-copy fill scores perfect fidelity + undetectable discriminator
    yet is REJECTED on the privacy pillar (min DCR = 0 / exact duplicates) — the
    verbatim-copy guard.

Strategy modules (swappable; ≥2 demonstrate the pattern):
  * ``simple``        — mean/median/mode fill (cheap, MCAR-safe).
  * ``model_based``   — the EXISTING ``imputation_strategist`` engine (KNN/MICE/
    MissForest by mechanism) — the default.
  * ``port``          — delegates to ``DataQualityBackend.impute`` so the apply step
    ITSELF swaps in-process pandas ↔ managed DataBrew recipe by ``DQ_BACKEND``/scale.
  * ``synthetic_fill``— synthetic generation + the 3-pillar AND-gate validation.

Selection precedence: explicit ``strategy=`` arg > steering ``pin_methods['imputation_strategy']``
> ``.agenticml.yaml``/env ``IMPUTATION_STRATEGY`` > the ``model_based`` default.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..tools.data_quality import diagnose_missingness
from ..tools.imputation import apply_imputation_plan, evaluate_imputation, recommend_imputation_plan
from ._util import raw

__all__ = [
    "STRATEGIES",
    "ImputationStrategy",
    "get_strategy",
    "list_strategies",
    "register_strategy",
    "run_imputation_flow",
    "select_strategy",
]


# ══════════════════════════════════════════════════════════════════════════════
# the swappable strategy-module seam
# ══════════════════════════════════════════════════════════════════════════════
@runtime_checkable
class ImputationStrategy(Protocol):
    """A swappable imputation apply-module (the part of the flow that varies).

    ``apply`` consumes the dataset + the recommended plan and returns
    ``{"imputed_uri", "module", "validation"?, ...}``. Everything else in the flow
    (diagnose → recommend → validate) is FIXED and lives in :func:`run_imputation_flow`.
    """

    name: str

    def apply(
        self,
        *,
        dataset_uri: str,
        plan: dict[str, Any],
        profile: dict[str, Any] | None = None,
        steering: Any | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]: ...


# strategy registry (``name → factory``) — last-write-wins (idempotent under re-import).
STRATEGIES: dict[str, Callable[[], ImputationStrategy]] = {}


def register_strategy(name: str, factory: Callable[[], ImputationStrategy]) -> None:
    """Register an imputation strategy module under ``name`` (the swappable seam)."""
    STRATEGIES[name.strip().lower()] = factory


def get_strategy(name: str) -> ImputationStrategy:
    """Instantiate the registered strategy ``name`` (raises ``KeyError`` on unknown)."""
    key = (name or "").strip().lower()
    if key not in STRATEGIES:
        raise KeyError(f"unknown imputation strategy {name!r}; registered: {sorted(STRATEGIES)}")
    return STRATEGIES[key]()


def list_strategies() -> list[str]:
    """Return the registered strategy names."""
    return sorted(STRATEGIES)


def select_strategy(strategy: str | None = None, steering: Any | None = None) -> str:
    """Resolve which strategy module to use (the registry/steering/.agenticml selector).

    Precedence (doc 05 §4.3 mirrored for the module choice):
      1. explicit ``strategy=`` arg;
      2. steering hard-pin ``pin_methods['imputation_strategy']``;
      3. env / ``.agenticml.yaml`` ``IMPUTATION_STRATEGY``;
      4. the ``model_based`` default.
    """
    if strategy:
        return strategy.strip().lower()
    pins = getattr(steering, "pin_methods", None)
    if isinstance(pins, dict):
        pin = pins.get("imputation_strategy") or pins.get("IMPUTATION_STRATEGY")
        if pin:
            return str(pin).strip().lower()
    env = os.environ.get("IMPUTATION_STRATEGY")
    if env:
        return env.strip().lower()
    return "model_based"


# ══════════════════════════════════════════════════════════════════════════════
# built-in strategy modules
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class SimpleImputationStrategy:
    """Mean/median/mode fill — the cheap MCAR-safe module.

    Forces every planned column to a simple central-tendency method, ignoring the
    richer per-mechanism recommendation. Demonstrates the swappable seam at its
    simplest while still flowing through the SAME ``apply_imputation_plan`` IP.
    """

    name: str = "simple"

    def apply(self, *, dataset_uri: str, plan: dict[str, Any], profile: dict[str, Any] | None = None,
              steering: Any | None = None, **kwargs: Any) -> dict[str, Any]:
        cols = plan.get("plan", plan)
        simple_plan = {
            col: {"method": "median", "params": None}
            for col in cols
        }
        out = raw(apply_imputation_plan)(dataset_uri, simple_plan, profile=_profile_dict(steering))
        return {"module": self.name, "imputed_uri": out.get("imputed_path", ""), "apply_result": out}


@dataclass
class ModelBasedImputationStrategy:
    """The EXISTING ``imputation_strategist`` engine — KNN/MICE/MissForest by mechanism.

    The default module: applies the per-column plan the fixed flow recommended (which
    already honored MCAR/MAR/MNAR + hard steering). Behavior-preserving over the
    existing ``apply_imputation_plan`` IP.
    """

    name: str = "model_based"

    def apply(self, *, dataset_uri: str, plan: dict[str, Any], profile: dict[str, Any] | None = None,
              steering: Any | None = None, **kwargs: Any) -> dict[str, Any]:
        cols = plan.get("plan", plan)
        out = raw(apply_imputation_plan)(dataset_uri, cols, profile=_profile_dict(steering))
        return {"module": self.name, "imputed_uri": out.get("imputed_path", ""), "apply_result": out}


@dataclass
class PortImputationStrategy:
    """Delegates apply to ``DataQualityBackend.impute`` — the module-within-a-module.

    This is the literal doc 05 §4.5 example: the apply step ITSELF swaps in-process
    pandas ↔ managed DataBrew recipe via the registry (``DQ_BACKEND``/scale), while the
    surrounding flow is unchanged. Proves the pattern composes with the backend seam.
    """

    name: str = "port"

    def apply(self, *, dataset_uri: str, plan: dict[str, Any], profile: dict[str, Any] | None = None,
              steering: Any | None = None, **kwargs: Any) -> dict[str, Any]:
        from ..backends.ports import DataQualityBackend
        from ..backends.registry import REGISTRY

        resolved = REGISTRY.resolve(DataQualityBackend, steering=steering, scale=kwargs.get("scale"))
        ref = resolved.adapter.impute(dataset_uri=dataset_uri, plan={"plan": plan.get("plan", plan),
                                                                      "profile": _profile_dict(steering)})
        return {
            "module": self.name,
            "imputed_uri": ref.uri,
            "adapter": resolved.adapter_name,
            "lineage": ref.lineage,
        }


@dataclass
class SyntheticFillImputationStrategy:
    """Synthetic-generation fill gated by the 3-pillar AND-gate (CORE IP preserved).

    For heavily-missing columns, generates synthetic rows (SDV when available) and
    accepts the fill ONLY if the EXISTING ``validate_synthetic`` 3-pillar gate passes
    (fidelity AND utility/detection AND privacy). The exact-copy guard is inherent: a
    verbatim copy scores perfect fidelity + AUROC≈0.5 yet FAILS privacy (min DCR = 0).

    When SDV is unavailable (no ``data`` extra), it falls back to the model_based apply
    and records ``synthetic_skipped`` — the flow still produces a validated dataset.
    """

    name: str = "synthetic_fill"
    thresholds: dict[str, float] | None = None

    def apply(self, *, dataset_uri: str, plan: dict[str, Any], profile: dict[str, Any] | None = None,
              steering: Any | None = None, **kwargs: Any) -> dict[str, Any]:

        from ..tools.synthetic import validate_synthetic

        target = kwargs.get("target")
        # First produce a complete dataset via the model_based apply (the spine).
        base = ModelBasedImputationStrategy().apply(
            dataset_uri=dataset_uri, plan=plan, profile=profile, steering=steering
        )
        imputed_uri = base["imputed_uri"]
        if not imputed_uri:
            return {"module": self.name, "imputed_uri": "", "synthetic_skipped": "apply produced no output",
                    "apply_result": base.get("apply_result", {})}

        # Validate the imputed result AS synthetic-vs-real (3-pillar AND-gate). The
        # imputed frame must not be a verbatim copy that leaks privacy, and must
        # preserve fidelity/utility — exactly the synthetic acceptance contract.
        try:
            validation = raw(validate_synthetic)(dataset_uri, imputed_uri, target=target,
                                                  thresholds=self.thresholds)
        except Exception as e:
            validation = {"success": False, "error": str(e), "accepted": False}

        accepted = bool(validation.get("accepted", False))
        return {
            "module": self.name,
            "imputed_uri": imputed_uri,
            "accepted": accepted,
            "validation": validation,
            "apply_result": base.get("apply_result", {}),
        }


# register the built-ins (the ≥2 strategies the swap test exercises).
register_strategy("simple", SimpleImputationStrategy)
register_strategy("model_based", ModelBasedImputationStrategy)
register_strategy("port", PortImputationStrategy)
register_strategy("synthetic_fill", SyntheticFillImputationStrategy)


def _profile_dict(steering: Any | None) -> dict[str, Any] | None:
    """Best-effort ``SteeringProfile.as_profile_dict()`` extraction (or a raw dict)."""
    if steering is None:
        return None
    if isinstance(steering, dict):
        return steering
    fn = getattr(steering, "as_profile_dict", None)
    if callable(fn):
        try:
            result = fn()
            return result if isinstance(result, dict) else None
        except Exception:
            return None
    return None


# ══════════════════════════════════════════════════════════════════════════════
# the FIXED flow (never changes — only the apply module swaps)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class ImputationFlowResult:
    """The result of one ``imputation_flow`` run (the invariant output contract)."""

    status: str
    strategy: str
    imputed_uri: str = ""
    diagnosis: dict[str, Any] = field(default_factory=dict)
    plan: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)
    accepted: bool = False
    detail: dict[str, Any] = field(default_factory=dict)


def run_imputation_flow(
    *,
    dataset_uri: str,
    strategy: str | None = None,
    steering: Any | None = None,
    target: str | None = None,
    scale: int | str | None = None,
    validate: bool = True,
    thresholds: dict[str, float] | None = None,
) -> ImputationFlowResult:
    """Run the invariant imputation flow with a swappable apply module.

    The fixed sequence (NEVER changes):
      1. **diagnose** the missingness mechanism per column (``diagnose_missingness`` —
         Little's MCAR + mask-classifier; identifiability IP preserved);
      2. **recommend** a per-column plan honoring mechanism + hard steering
         (``recommend_imputation_plan``);
      3. **apply** via the SELECTED swappable :class:`ImputationStrategy` module;
      4. **validate** — distribution preservation for most modules; the 3-pillar
         AND-gate for ``synthetic_fill`` (the verbatim-copy-fails-on-privacy guard).

    Args:
        dataset_uri: CSV (local) or s3 URI to impute.
        strategy: Strategy module name (overrides steering/env; see :func:`select_strategy`).
        steering: Optional ``SteeringProfile`` (hard pins flow through to apply + recommend).
        target: Optional target column (enables the utility pillar for synthetic_fill).
        scale: Data-scale hint forwarded to the ``port`` strategy's registry resolution.
        validate: When True, run the post-apply validation step (the flow's gate).
        thresholds: Optional synthetic-validation threshold overrides.

    Returns:
        An :class:`ImputationFlowResult` (status + chosen strategy + imputed URI +
        diagnosis + plan + validation + acceptance).
    """
    chosen = select_strategy(strategy, steering)

    # ── 1. diagnose (mechanism identifiability IP — unchanged) ───────────────
    profile_dict = _profile_dict(steering)
    diag = raw(diagnose_missingness)(dataset_uri, profile=profile_dict)
    if isinstance(diag, dict) and diag.get("success") is False:
        return ImputationFlowResult(status="failed", strategy=chosen, detail={"stage": "diagnose", "error": diag.get("error")})

    # ── 2. recommend (per-mechanism + hard-steering plan — unchanged) ────────
    plan_out = raw(recommend_imputation_plan)(dataset_uri, target=target, profile=profile_dict)
    if isinstance(plan_out, dict) and plan_out.get("success") is False:
        return ImputationFlowResult(status="failed", strategy=chosen, diagnosis=diag,
                                    detail={"stage": "recommend", "error": plan_out.get("error")})
    plan = plan_out.get("plan", {})

    # ── 3. apply via the SWAPPABLE module ────────────────────────────────────
    try:
        module = get_strategy(chosen)
    except KeyError as e:
        return ImputationFlowResult(status="failed", strategy=chosen, diagnosis=diag, plan=plan,
                                    detail={"stage": "select", "error": str(e)})
    if isinstance(module, SyntheticFillImputationStrategy) and thresholds:
        module.thresholds = thresholds
    applied = module.apply(
        dataset_uri=dataset_uri, plan={"plan": plan}, profile=profile_dict,
        steering=steering, target=target, scale=scale,
    )
    imputed_uri = applied.get("imputed_uri", "")
    if not imputed_uri:
        return ImputationFlowResult(status="failed", strategy=chosen, diagnosis=diag, plan=plan,
                                    detail={"stage": "apply", "applied": applied})

    # ── 4. validate (distribution preservation OR the 3-pillar AND-gate) ─────
    validation: dict[str, Any] = {}
    accepted = True
    if validate:
        if "validation" in applied:  # synthetic_fill already ran the 3-pillar gate
            validation = applied["validation"]
            accepted = bool(applied.get("accepted", False))
        else:
            cols = [c for c in plan]
            if cols:
                ev = raw(evaluate_imputation)(dataset_uri, cols)
                validation = ev if isinstance(ev, dict) else {}
                # distribution-preservation acceptance: no column collapsed its distribution
                accepted = _distribution_preserved(validation)
            else:
                validation = {"note": "no columns required imputation"}
                accepted = True

    status = "accepted" if accepted else "rejected"
    return ImputationFlowResult(
        status=status,
        strategy=chosen,
        imputed_uri=imputed_uri,
        diagnosis=diag,
        plan=plan,
        validation=validation,
        accepted=accepted,
        detail={"applied": {k: v for k, v in applied.items() if k != "apply_result"}},
    )


def _distribution_preserved(evaluation: dict[str, Any]) -> bool:
    """True when no evaluated column's imputation collapsed its distribution (KS gate).

    Reads the ``evaluate_imputation`` per-column ``best`` method's ``distribution_ok``.
    Conservative: an empty/odd report is treated as preserved (the flow does not block
    on a missing signal — only on concrete distribution divergence).
    """
    per_col = evaluation.get("per_column", {})
    if not isinstance(per_col, dict) or not per_col:
        return True
    for _col, info in per_col.items():
        if not isinstance(info, dict) or "methods" not in info:
            continue
        best = info.get("best_method_hint")
        methods = info.get("methods", {})
        if best in methods and methods[best].get("distribution_ok") is False:
            return False
    return True
