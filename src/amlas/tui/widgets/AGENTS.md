# src/amlas/tui/widgets

## Purpose

Provide reusable UI progress, jobs, charts, logs, specification and observability components.

## Ownership

Inherits [../AGENTS.md](../AGENTS.md); this document owns the files directly in this directory.

## Local Contracts

- `active_jobs.py:ActiveJobsPanel` — Live table of ALL concurrent jobs sourced from ``state.jobs`` (§2a).
- `agent_activity.py:AgentActivityPanel` — Live table of per-agent activity sourced from ``state.agents`` (§E.8).
- `agent_logs.py:AgentLogsPane` — Right-hand pane: the 7 ML-automation leads' logs — summaries + errors/warnings.
- `chart_view.py:chart_target_tabs` — Return the tabs a chart may be addressed to IN THIS CONFIG (Dashboard first).
- `chart_view.py:ChartViewPanel` — Renders an agent-requested chart addressed to ``tab`` (and ignores others).
- `charts.py:render_line` — Render one or more metric curves as an ASCII line chart.
- `charts.py:render_scatter` — Render a scatter plot with a least-squares regression line and R².
- Owned files: `__init__.py`, `active_jobs.py`, `agent_activity.py`, `agent_logs.py`, `chart_view.py`, `charts.py`, `dynamic_primary.py`, `jobs_indicator.py`, `leaderboard.py`, `observability.py`, `processes.py`, `progress.py`, `running_log.py`, `spec_viewer.py`, `status_bar.py`, `thought_ticker.py`.

## Work Guidance

Update this contract when owned interfaces, configuration, side effects or boundaries change. Read indexed child contracts before editing their files.

## Verification

Use the root verification contract and the owning regressions for changed behavior.

## Child DOX Index

None.
