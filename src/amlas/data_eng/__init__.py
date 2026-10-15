"""Data-Engineering domain-steering primitives (master plan §3.1, §11).

This subpackage holds the **steering "dial"** — the mechanism that lets a domain
expert (the Amlas lab tech, transcript "give them a way to dial that") impose
HARD constraints and SOFT preferences on the imputation / synthetic engines.

The :class:`SteeringProfile` is a Pydantic model loaded from a YAML file or the
``imputation.steering`` spec-block. Its HARD constraints (``forbid_methods``,
``pin_methods``, ``mechanism_hints``, ``max_llm_fraction``, ``require_evaluation``)
are enforced *deterministically inside the tools* so the agent can never violate
them; its SOFT preferences (``preferred_methods``, ``domain_notes``,
``novelty_budget``) flow into agent prompts; its ``thresholds`` feed the
deterministic graders (Arbiter-style acceptance).
"""

from __future__ import annotations

from .steering import SteeringProfile

__all__ = ["SteeringProfile"]
