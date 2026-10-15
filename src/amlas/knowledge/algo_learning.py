"""Algorithm Learning — learns from past experiment results to improve recommendations.

Combines historical performance data from EvaluationStore with the static algorithm
knowledge base to produce evidence-based algorithm recommendations.
"""

from __future__ import annotations

import statistics as stats_mod
from collections import defaultdict
from typing import Any

from ..evaluation.store import EvaluationStore, ExperimentResult
from ..knowledge.algorithms import rank_algorithms_for_task


class AlgorithmLearner:
    """Learns from past experiment results to improve recommendations."""

    def __init__(self, store: EvaluationStore) -> None:
        self.store = store

    def get_algorithm_performance_history(self, algorithm: str) -> dict[str, Any]:
        """Get historical performance of an algorithm across all specs.

        Returns:
            {"avg_score": float, "std_score": float, "n_runs": int,
             "best_score": float, "worst_score": float, "win_rate": float}
        """
        results = [
            r for r in self.store.get_all()
            if r.algorithm == algorithm and r.status == "success"
        ]

        if not results:
            return {
                "avg_score": 0.0,
                "std_score": 0.0,
                "n_runs": 0,
                "best_score": 0.0,
                "worst_score": 0.0,
                "win_rate": 0.0,
            }

        scores = [r.primary_score for r in results]
        n_runs = len(scores)

        # Calculate win rate: how often this algorithm has the best score in its spec
        wins = 0
        spec_ids = {r.spec_id for r in results}
        for spec_id in spec_ids:
            spec_results = [
                r for r in self.store.get_all()
                if r.spec_id == spec_id and r.status == "success"
            ]
            if spec_results:
                best_score = max(r.primary_score for r in spec_results)
                algo_best = max(
                    (r.primary_score for r in spec_results if r.algorithm == algorithm),
                    default=0.0,
                )
                if algo_best >= best_score and best_score > 0:
                    wins += 1

        win_rate = wins / len(spec_ids) if spec_ids else 0.0

        return {
            "avg_score": stats_mod.mean(scores),
            "std_score": stats_mod.stdev(scores) if n_runs > 1 else 0.0,
            "n_runs": n_runs,
            "best_score": max(scores),
            "worst_score": min(scores),
            "win_rate": win_rate,
        }

    def get_best_algorithms_for_task_type(self, task_type: str, top_n: int = 5) -> list[dict[str, Any]]:
        """Based on ACTUAL past results, rank algorithms for a task type.

        Filters results by task_type annotation in notes, then ranks by average score.

        Returns:
            Ranked list with historical evidence.
        """
        all_results = [r for r in self.store.get_all() if r.status == "success"]

        # Filter by task_type (stored in notes as "task_type=classification")
        task_results = [
            r for r in all_results
            if f"task_type={task_type}" in r.notes
        ]

        if not task_results:
            return []

        # Group by algorithm
        algo_scores: dict[str, list[float]] = defaultdict(list)
        for r in task_results:
            algo_scores[r.algorithm].append(r.primary_score)

        # Rank by average score
        ranked: list[dict[str, Any]] = []
        for algo_name, scores in algo_scores.items():
            ranked.append({
                "algorithm": algo_name,
                "avg_score": stats_mod.mean(scores),
                "std_score": stats_mod.stdev(scores) if len(scores) > 1 else 0.0,
                "n_runs": len(scores),
                "best_score": max(scores),
            })

        ranked.sort(key=lambda x: -x["avg_score"])
        return ranked[:top_n]

    def get_algorithm_rankings(self) -> dict[str, dict[str, Any]]:
        """Overall rankings: which algorithms tend to win most often.

        Returns:
            {algorithm: {"wins": int, "total": int, "avg_rank": float}}
        """
        all_results = [r for r in self.store.get_all() if r.status == "success"]

        if not all_results:
            return {}

        # Group results by spec_id
        spec_groups: dict[str, list[ExperimentResult]] = defaultdict(list)
        for r in all_results:
            spec_groups[r.spec_id].append(r)

        # For each spec, rank algorithms by primary_score
        algo_stats: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"wins": 0, "total": 0, "ranks": []}
        )

        for _spec_id, results in spec_groups.items():
            # Sort by score descending
            sorted_results = sorted(results, key=lambda r: -r.primary_score)

            # Assign ranks (best score = rank 1)
            seen_algos: set[str] = set()
            for rank, r in enumerate(sorted_results, 1):
                if r.algorithm not in seen_algos:
                    seen_algos.add(r.algorithm)
                    algo_stats[r.algorithm]["ranks"].append(rank)
                    algo_stats[r.algorithm]["total"] += 1
                    if rank == 1:
                        algo_stats[r.algorithm]["wins"] += 1

        # Compute avg_rank
        rankings = {}
        for algo, data in algo_stats.items():
            ranks = data["ranks"]
            rankings[algo] = {
                "wins": data["wins"],
                "total": data["total"],
                "avg_rank": stats_mod.mean(ranks) if ranks else 0.0,
            }

        return rankings

    def recommend_with_learning(self, spec_dict: dict[str, Any], top_n: int = 5) -> list[dict[str, Any]]:
        """Enhanced recommendations that combine KB knowledge + historical performance.

        Weights: 60% historical performance, 40% KB suitability.
        Falls back to KB-only if no history available.

        Args:
            spec_dict: Dict with at least task_type, optionally target, primary_metric.
            top_n: Number of recommendations to return.

        Returns:
            List of recommendation dicts with algorithm, combined_score, etc.
        """
        task_type = spec_dict.get("task_type", "classification")
        data_type = "tabular"
        data_size = "medium"

        # Get KB rankings (static knowledge)
        kb_ranked = rank_algorithms_for_task(
            task_type=task_type,
            data_type=data_type,
            data_size=data_size,
        )

        # Normalize KB scores: best gets 1.0, worst gets 0.0
        if kb_ranked:
            max_rank = len(kb_ranked)
            kb_scores = {
                algo["name"]: 1.0 - ((algo["rank"] - 1) / max_rank)
                for algo in kb_ranked
            }
        else:
            kb_scores = {}

        # Get historical rankings
        history = self.get_best_algorithms_for_task_type(task_type, top_n=20)

        has_history = len(history) > 0

        if has_history:
            # Normalize historical scores: best avg_score → 1.0, worst → 0.0
            hist_scores_raw = {h["algorithm"]: h["avg_score"] for h in history}
            max_hist = max(hist_scores_raw.values()) if hist_scores_raw else 1.0
            min_hist = min(hist_scores_raw.values()) if hist_scores_raw else 0.0
            hist_range = max_hist - min_hist if max_hist != min_hist else 1.0

            hist_scores = {
                algo: (score - min_hist) / hist_range
                for algo, score in hist_scores_raw.items()
            }
        else:
            hist_scores = {}

        # Combine: 60% history + 40% KB (if history available)
        # Else: 100% KB
        all_algos = set(kb_scores.keys()) | set(hist_scores.keys())

        combined = []
        for algo_name in all_algos:
            kb_score = kb_scores.get(algo_name, 0.0)

            if has_history:
                hist_score = hist_scores.get(algo_name, 0.0)
                combined_score = 0.6 * hist_score + 0.4 * kb_score
            else:
                combined_score = kb_score

            combined.append({
                "algorithm": algo_name,
                "combined_score": combined_score,
                "kb_score": kb_score,
                "historical_score": hist_scores.get(algo_name, 0.0) if has_history else None,
            })

        # Sort by combined score descending
        combined.sort(key=lambda x: -x["combined_score"])
        return combined[:top_n]
