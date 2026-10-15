"""Leaderboard — ranking, comparison, and improvement tracking.

Pure deterministic Python. Operates on EvaluationStore data.
"""

from __future__ import annotations

from typing import Any

from .store import EvaluationStore, ExperimentResult


class Leaderboard:
    """Provides ranking, comparison, and historical trend analysis over experiment results."""

    def __init__(self, store: EvaluationStore) -> None:
        self._store = store

    def get_leaderboard(
        self,
        spec_id: str,
        metric: str = "f1_macro",
        direction: str = "maximize",
        top_n: int | None = None,
    ) -> list[ExperimentResult]:
        """Get ranked list of best runs for a spec.

        Args:
            spec_id: Filter to this spec.
            metric: Metric to sort by (uses primary_score).
            direction: "maximize" (higher first) or "minimize" (lower first).
            top_n: Return only top N results. None for all.

        Returns:
            Sorted list of ExperimentResult, best first.
        """
        results = [r for r in self._store.get_by_spec(spec_id) if r.status == "success"]

        if direction == "minimize":
            results.sort(key=lambda r: r.primary_score)
        else:
            results.sort(key=lambda r: r.primary_score, reverse=True)

        if top_n is not None:
            results = results[:top_n]

        return results

    def get_comparison_table(self, spec_id: str, direction: str = "maximize") -> list[dict[str, Any]]:
        """Side-by-side comparison of all algorithms for a spec.

        Returns list of dicts, each with algorithm, run_id, and all metrics.

        Args:
            spec_id: Filter to this spec.
            direction: "maximize" (higher first, default) or "minimize" (lower
                first). Threaded through like ``get_leaderboard`` /
                ``get_improvement_history`` so minimize metrics (RMSE/MAE) rank
                best-first instead of worst-first (LB-01).
        """
        results = [r for r in self._store.get_by_spec(spec_id) if r.status == "success"]

        table: list[dict[str, Any]] = []
        for r in results:
            entry = {
                "run_id": r.run_id,
                "algorithm": r.algorithm,
                "primary_score": r.primary_score,
                "duration_seconds": r.duration_seconds,
                "timestamp": r.timestamp,
            }
            # Flatten metrics into the entry
            for k, v in r.metrics.items():
                entry[k] = v
            table.append(entry)

        # Sort by primary score, honoring optimization direction (LB-01).
        table.sort(key=lambda e: e["primary_score"], reverse=(direction != "minimize"))
        return table

    def get_improvement_history(
        self,
        spec_id: str,
        direction: str = "maximize",
    ) -> list[dict[str, Any]]:
        """Chronological best-score progression.

        Returns list of dicts with timestamp, run_id, algorithm, best_score.
        Only includes entries where a new best was achieved.
        """
        results = [r for r in self._store.get_by_spec(spec_id) if r.status == "success"]

        # Sort by timestamp (chronological)
        results.sort(key=lambda r: r.timestamp)

        history = []
        current_best: float | None = None

        for r in results:
            is_new_best = False
            if current_best is None:
                is_new_best = True
            elif direction == "maximize" and r.primary_score > current_best:
                is_new_best = True
            elif direction == "minimize" and r.primary_score < current_best:
                is_new_best = True

            if is_new_best:
                current_best = r.primary_score
                history.append({
                    "run_id": r.run_id,
                    "algorithm": r.algorithm,
                    "best_score": r.primary_score,
                    "timestamp": r.timestamp,
                })

        return history

    def format_leaderboard_markdown(
        self,
        spec_id: str,
        metric: str = "f1_macro",
        direction: str = "maximize",
        top_n: int | None = None,
    ) -> str:
        """Format leaderboard as a markdown table.

        Returns formatted markdown string with ranking table.
        """
        ranked = self.get_leaderboard(spec_id, metric=metric, direction=direction, top_n=top_n)

        if not ranked:
            return f"## Leaderboard: {spec_id}\n\nNo results available."

        lines = [
            f"## Leaderboard: {spec_id}",
            "",
            f"| Rank | Algorithm | {metric} | Duration (s) | Run ID |",
            "|------|-----------|" + "-" * (len(metric) + 2) + "|--------------|--------|",
        ]

        for i, r in enumerate(ranked, 1):
            score = f"{r.primary_score:.4f}"
            duration = f"{r.duration_seconds:.1f}"
            short_id = r.run_id[:8]
            lines.append(f"| {i} | {r.algorithm} | {score} | {duration} | {short_id}... |")

        lines.append("")
        lines.append(f"_Total experiments: {len(ranked)}_")
        return "\n".join(lines)
