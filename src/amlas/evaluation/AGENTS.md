# src/amlas/evaluation

## Purpose

Store and compare experiment results, leaderboards and statistical evidence.

## Ownership

Inherits [../AGENTS.md](../AGENTS.md); this document owns the files directly in this directory.

## Local Contracts

- `judge.py:JudgeVerdict` — The structured result of an LLM-as-judge evaluation.
- `judge.py:Judge` — An agent-output judge — evaluate a (prompt, response) pair against criteria.
- `BedrockLLMJudge` resolves the shared reviewer model unless explicitly overridden.
  The separate reviewer default is GPT-6.1 Sol; execution agents use Sonnet 5.5.
  Filter sampling fields through the shared model compatibility helper and parse only
  visible text blocks; adaptive-thinking blocks are never evaluation output.
- `leaderboard.py:Leaderboard` — Provides ranking, comparison, and historical trend analysis over experiment results.
- `statistics.py:ModelComparator` — Statistical significance testing for model comparisons.
- `store.py:ExperimentResult` — Canonical schema for a single experiment run result.
- `store.py:EvaluationStore` — Thread-safe JSON-based experiment results store.
- Owned files: `__init__.py`, `judge.py`, `leaderboard.py`, `statistics.py`, `store.py`.

## Work Guidance

Update this contract when owned interfaces, configuration, side effects or boundaries change. Read indexed child contracts before editing their files.

## Verification

Use the root verification contract and the owning regressions for changed behavior.

## Child DOX Index

None.
