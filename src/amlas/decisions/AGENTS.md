# src/amlas/decisions

## Purpose

Produce rules-based experiment recommendations, next-step decisions and model-card content.

## Ownership

Inherits [../AGENTS.md](../AGENTS.md); this document owns the files directly in this directory.

## Local Contracts

- `engine.py:Decision` — Structured recommendation from the Decision Engine.
- `engine.py:DecisionEngine` — Rules-based decision engine for ML experiment analysis.
- `model_card.py:generate_model_card` — Generate a standard model card in markdown format.
- `rules.py:check_target_met` — Check if the best result meets or exceeds the target threshold.
- `rules.py:check_diminishing_returns` — Check if improvements are getting smaller across iterations.
- Owned files: `__init__.py`, `engine.py`, `model_card.py`, `rules.py`.

## Work Guidance

Update this contract when owned interfaces, configuration, side effects or boundaries change. Read indexed child contracts before editing their files.

## Verification

Use the root verification contract and the owning regressions for changed behavior.

## Child DOX Index

None.
