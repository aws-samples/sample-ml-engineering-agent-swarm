# src/amlas/tui

## Purpose

Provide the Textual application, feature flags and shared capability presentation.

## Ownership

Inherits [../AGENTS.md](../AGENTS.md); this document owns the files directly in this directory.

## Local Contracts

- `app.py:AgenticMLApp` — Agentic-ML Terminal UI — Interactive ML Engineering Platform.
- `app.py:run` — Launch the Agentic-ML TUI application.
- `capability_surface.py:CapabilityRow` — One render-ready capability row for a TUI table/pane (registry-derived).
- `feature_flags.py:is_enabled` — Return whether a generic named feature is enabled.
- `feature_flags.py:is_tab_enabled` — Return whether the given tab is enabled under the current env.
- `markup.py:escape_markup` — Escape Rich/Textual console-markup metacharacters in ``text``.
- Owned files: `__init__.py`, `__main__.py`, `app.py`, `capability_surface.py`, `feature_flags.py`, `markup.py`, `styles.css`.

## Work Guidance

Update this contract when owned interfaces, configuration, side effects or boundaries change. Read indexed child contracts before editing their files.

## Verification

Use the root verification contract and the owning regressions for changed behavior.

## Child DOX Index

| Path | Covers |
| --- | --- |
| `chat/AGENTS.md` | Own conversational turns, deterministic command handling, tool binding, dispatch and visible thought/tool traces. |
| `context/AGENTS.md` | Share validated UI context and project/session/trace state through the context bus. |
| `jobs/AGENTS.md` | Manage background UI execution, job lifecycle and cancellation. |
| `screens/AGENTS.md` | Present the application tabs and route their actions to the existing tools/capability layer. |
| `usecases/AGENTS.md` | Define machine-readable data/ML use-case registries used by the interface. |
| `widgets/AGENTS.md` | Provide reusable UI progress, jobs, charts, logs, specification and observability components. |
