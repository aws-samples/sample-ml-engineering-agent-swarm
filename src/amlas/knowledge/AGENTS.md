# src/amlas/knowledge

## Purpose

Provide algorithm knowledge and persisted learning used by recommendations.

## Ownership

Inherits [../AGENTS.md](../AGENTS.md); this document owns the files directly in this directory.

## Local Contracts

- `algo_learning.py:AlgorithmLearner` — Learns from past experiment results to improve recommendations.
- `algorithms.py:get_algorithms_by_task` — Filter algorithms by task type (classification or regression).
- `algorithms.py:get_algorithms_by_data_type` — Filter algorithms by data type (tabular, text, image, time-series).
- Owned files: `__init__.py`, `algo_learning.py`, `algorithms.py`.

## Work Guidance

Update this contract when owned interfaces, configuration, side effects or boundaries change. Read indexed child contracts before editing their files.

## Verification

Use the root verification contract and the owning regressions for changed behavior.

## Child DOX Index

None.
