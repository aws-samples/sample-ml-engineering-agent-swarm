"""The single registry of the seven Leads — declaration + public builder, paired.

There was previously no lead registry: [`app.py`](../app.py:191) hand-wires each
``build_*_lead()`` call individually, so nothing could enumerate "the 7 Leads"
programmatically. ADR-0007 phase 1 needs exactly that enumeration in two places at
once — the harness registry (which leads are harness-hosted) and the parity checks
(which leads must match their local peer) — and T-HARN-02.4 / T-HARN-15.3 require it
to be ONE registry, not two lists that can drift.

Import direction: each lead module imports :mod:`amlas.leads.declarations`; this
registry imports the lead modules. Nothing under ``subagents/`` or ``tools/`` imports
``leads``, so there is no cycle.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from strands import Agent

from .data_eng import build_data_engineering_lead
from .declarations import (
    DATA_ENGINEERING_LEAD,
    DEPLOY_SERVING_LEAD,
    FE_LEAD,
    ML_EXECUTION_LEAD,
    MLOPS_LEAD,
    MONITORING_LEAD,
    RESEARCH_LEAD,
    LeadDeclaration,
)
from .deploy_serving_lead import build_deploy_serving_lead
from .feature_eng import build_fe_lead
from .ml_execution import build_ml_execution_lead
from .mlops_lead import build_mlops_lead
from .monitoring_lead import build_monitoring_lead
from .research_lead import build_research_lead


@dataclass(frozen=True)
class LeadEntry:
    """One Lead: its offline declaration plus the public builder that realises it.

    Attributes:
        declaration: The tool/prompt declaration — readable with no ``Agent`` built.
        builder: The public ``build_*_lead`` factory (the local-construction peer,
            retained per T-HARN-04.3 as the equivalence oracle).
    """

    declaration: LeadDeclaration
    builder: Callable[..., Agent]

    @property
    def key(self) -> str:
        return self.declaration.key


LEADS: tuple[LeadEntry, ...] = (
    LeadEntry(RESEARCH_LEAD, build_research_lead),
    LeadEntry(DATA_ENGINEERING_LEAD, build_data_engineering_lead),
    LeadEntry(FE_LEAD, build_fe_lead),
    LeadEntry(ML_EXECUTION_LEAD, build_ml_execution_lead),
    LeadEntry(DEPLOY_SERVING_LEAD, build_deploy_serving_lead),
    LeadEntry(MLOPS_LEAD, build_mlops_lead),
    LeadEntry(MONITORING_LEAD, build_monitoring_lead),
)

#: ``LEADS`` keyed by lead key, for direct lookup.
LEADS_BY_KEY: dict[str, LeadEntry] = {entry.key: entry for entry in LEADS}


def lead_entry(key: str) -> LeadEntry:
    """Return the :class:`LeadEntry` for ``key``, raising ``KeyError`` with valid keys."""
    try:
        return LEADS_BY_KEY[key]
    except KeyError:
        raise KeyError(f"unknown lead {key!r}; known leads: {sorted(LEADS_BY_KEY)}") from None


def lead_keys() -> tuple[str, ...]:
    """The seven lead keys, in registry order."""
    return tuple(entry.key for entry in LEADS)


__all__ = [
    "LEADS",
    "LEADS_BY_KEY",
    "LeadEntry",
    "lead_entry",
    "lead_keys",
]
