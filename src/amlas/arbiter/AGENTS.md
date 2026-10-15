# src/amlas/arbiter

## Purpose

Grade predictive results deterministically and produce acceptance reports without delegating the decision to an LLM.

## Ownership

Inherits [../AGENTS.md](../AGENTS.md); this document owns the files directly in this directory.

## Local Contracts

- Keep grading deterministic; evaluate supplied labels/predictions against the selected specification, metric and thresholds.
- Register new grading tasks through `registry.py`; return the declared report/schema instead of inventing an accepted result.
- `arbiter.py:grade` — Score y_pred vs y_true per the spec's task_type and emit an ArbiterReport.
- `registry.py:register` — Register a grader. Duplicate task_type registrations raise ValueError.
- `registry.py:get_grader` — Return the registered grader for a task_type; raises KeyError if missing.
- `schema.py:InvalidSubmissionError` — Raised by a grader when the agent's submission is malformed.
- `schema.py:Grader` — Binding of task_type → scoring function.
- Owned files: `__init__.py`, `arbiter.py`, `registry.py`, `schema.py`.

## Work Guidance

Update this contract when owned interfaces, configuration, side effects or boundaries change. Read indexed child contracts before editing their files.

## Verification

Use the root verification contract and the owning regressions for changed behavior.

## Child DOX Index

| Path | Covers |
| --- | --- |
| `graders/AGENTS.md` | Register the classification, regression, detection and segmentation grading implementations. |
