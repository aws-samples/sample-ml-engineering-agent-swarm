"""Statistical significance testing for model comparisons.

Provides:
  - ModelComparator: Compare models with Wilcoxon, t-test, bootstrap CI
  - Bonferroni correction for multiple comparisons
  - Significance report generation (markdown)
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Any

import numpy as np
from scipy import stats


class ModelComparator:
    """Statistical significance testing for model comparisons."""

    def compare_models(
        self,
        scores_a: list[float],
        scores_b: list[float],
        test: str = "wilcoxon",
        alpha: float = 0.05,
    ) -> dict[str, Any]:
        """Compare two models using statistical test.

        Tests available:
        - "wilcoxon": Wilcoxon signed-rank (non-parametric, paired)
        - "ttest": Paired t-test (parametric)

        Returns:
            {
                "statistic": float,
                "p_value": float,
                "significant": bool,  # p < alpha
                "effect_size": float,  # Cohen's d
                "confidence_interval": (float, float),  # 95% CI of difference
                "interpretation": str
            }
        """
        a = np.array(scores_a, dtype=float)
        b = np.array(scores_b, dtype=float)
        diff = a - b

        # Perform the statistical test
        if test == "wilcoxon":
            # Wilcoxon signed-rank test (non-parametric, paired)
            try:
                statistic, p_value = stats.wilcoxon(diff)
            except ValueError:
                # All differences are zero
                statistic, p_value = 0.0, 1.0
        elif test == "ttest":
            # Paired t-test (parametric)
            statistic, p_value = stats.ttest_rel(a, b)
        else:
            raise ValueError(f"Unknown test: {test}. Use 'wilcoxon' or 'ttest'.")

        # Cohen's d (effect size)
        effect_size = self._cohens_d(a, b)

        # 95% confidence interval of the difference
        ci = self._paired_ci(diff, alpha=alpha)

        # Determine significance
        significant = bool(p_value < alpha)

        # Interpretation
        if significant:
            mean_diff = float(np.mean(diff))
            if mean_diff > 0:
                interpretation = "Model A significantly better than Model B"
            else:
                interpretation = "Model B significantly better than Model A"
        else:
            interpretation = "No significant difference between models"

        return {
            "statistic": float(statistic),
            "p_value": float(p_value),
            "significant": significant,
            "effect_size": float(effect_size),
            "confidence_interval": ci,
            "interpretation": interpretation,
        }

    def cross_validation_compare(
        self,
        model_a_cv_scores: list[float],
        model_b_cv_scores: list[float],
        alpha: float = 0.05,
    ) -> dict[str, Any]:
        """Compare models using k-fold CV scores.

        Uses corrected paired t-test (Nadeau & Bengio, 2003).
        The correction accounts for the non-independence of CV folds.

        Args:
            model_a_cv_scores: k-fold CV scores for model A.
            model_b_cv_scores: k-fold CV scores for model B.
            alpha: Significance level.

        Returns:
            Dict with statistic, p_value, significant, effect_size.
        """
        a = np.array(model_a_cv_scores, dtype=float)
        b = np.array(model_b_cv_scores, dtype=float)
        diff = a - b

        k = len(diff)
        mean_diff = float(np.mean(diff))
        var_diff = float(np.var(diff, ddof=1))

        # Corrected variance (Nadeau & Bengio, 2003)
        # Assumes test_size = 1/k (standard k-fold)
        test_train_ratio = 1.0 / (k - 1)  # approximation for k-fold
        corrected_var = var_diff * (1.0 / k + test_train_ratio)

        if corrected_var <= 0:
            return {
                "statistic": 0.0,
                "p_value": 1.0,
                "significant": False,
                "effect_size": 0.0,
            }

        # Corrected t-statistic
        t_stat = mean_diff / math.sqrt(corrected_var)
        # Degrees of freedom = k - 1
        df = k - 1
        p_value = float(2 * stats.t.sf(abs(t_stat), df))

        effect_size = self._cohens_d(a, b)

        return {
            "statistic": float(t_stat),
            "p_value": p_value,
            "significant": p_value < alpha,
            "effect_size": float(effect_size),
        }

    def bootstrap_confidence_interval(
        self,
        scores: list[float],
        n_bootstrap: int = 1000,
        ci: float = 0.95,
    ) -> tuple[float, float]:
        """Bootstrap confidence interval for a metric.

        Args:
            scores: List of metric scores.
            n_bootstrap: Number of bootstrap samples.
            ci: Confidence level (default 0.95).

        Returns:
            (lower_bound, upper_bound)
        """
        rng = np.random.default_rng(seed=42)
        arr = np.array(scores, dtype=float)
        n = len(arr)

        bootstrap_means = np.empty(n_bootstrap)
        for i in range(n_bootstrap):
            sample = rng.choice(arr, size=n, replace=True)
            bootstrap_means[i] = np.mean(sample)

        alpha = 1.0 - ci
        lower = float(np.percentile(bootstrap_means, 100 * alpha / 2))
        upper = float(np.percentile(bootstrap_means, 100 * (1 - alpha / 2)))

        return (lower, upper)

    def generate_significance_report(self, results: list[dict[str, Any]]) -> str:
        """Generate markdown report comparing all model pairs.

        Includes: pairwise comparisons, adjusted p-values (Bonferroni),
        and clear winner determination.

        Args:
            results: List of {"name": str, "scores": list[float]}

        Returns:
            Markdown formatted significance report.
        """
        if len(results) < 2:
            return "## Significance Report\n\nNeed at least 2 models to compare."

        n_models = len(results)
        pairs = list(combinations(range(n_models), 2))
        n_comparisons = len(pairs)

        lines = [
            "## Statistical Significance Report",
            "",
            f"**Models compared**: {n_models}",
            f"**Pairwise comparisons**: {n_comparisons}",
            f"**Bonferroni-corrected alpha**: {0.05 / n_comparisons:.4f}",
            "",
            "### Pairwise Comparisons",
            "",
            "| Model A | Model B | Mean Diff | p-value | Adjusted p | Significant | Effect Size |",
            "|---------|---------|-----------|---------|------------|-------------|-------------|",
        ]

        0.05 / n_comparisons
        comparison_results = []

        for i, j in pairs:
            name_a = results[i]["name"]
            name_b = results[j]["name"]
            scores_a = results[i]["scores"]
            scores_b = results[j]["scores"]

            result = self.compare_models(scores_a, scores_b, test="wilcoxon")

            adjusted_p = min(result["p_value"] * n_comparisons, 1.0)
            significant_adjusted = adjusted_p < 0.05

            mean_diff = float(np.mean(np.array(scores_a) - np.array(scores_b)))

            lines.append(
                f"| {name_a} | {name_b} | {mean_diff:+.4f} | "
                f"{result['p_value']:.4f} | {adjusted_p:.4f} | "
                f"{'✓' if significant_adjusted else '✗'} | "
                f"{result['effect_size']:.3f} |"
            )

            comparison_results.append({
                "model_a": name_a,
                "model_b": name_b,
                "p_value": result["p_value"],
                "adjusted_p": adjusted_p,
                "significant": significant_adjusted,
                "mean_diff": mean_diff,
            })

        # Determine overall winner
        lines.extend(["", "### Summary", ""])

        # Score each model by wins
        wins = {r["name"]: 0 for r in results}
        for comp in comparison_results:
            if comp["significant"]:
                if comp["mean_diff"] > 0:
                    wins[comp["model_a"]] += 1
                else:
                    wins[comp["model_b"]] += 1

        best_model = max(wins, key=lambda k: wins[k])
        lines.append(
            f"**Winner (most significant wins)**: {best_model} "
            f"({wins[best_model]}/{n_comparisons} comparisons won)"
        )
        lines.append("")
        lines.append(
            f"_Note: p-values Bonferroni-corrected for {n_comparisons} comparisons._"
        )

        return "\n".join(lines)

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
        """Compute Cohen's d effect size (pooled standard deviation)."""
        n_a, n_b = len(a), len(b)
        var_a = float(np.var(a, ddof=1))
        var_b = float(np.var(b, ddof=1))

        # Pooled standard deviation
        pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
        pooled_std = math.sqrt(pooled_var) if pooled_var > 0 else 1e-10

        return abs(float(np.mean(a) - np.mean(b))) / pooled_std

    @staticmethod
    def _paired_ci(diff: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
        """Compute confidence interval for paired differences."""
        n = len(diff)
        mean_diff = float(np.mean(diff))
        se = float(np.std(diff, ddof=1)) / math.sqrt(n)

        # t critical value
        t_crit = float(stats.t.ppf(1 - alpha / 2, df=n - 1))

        lower = mean_diff - t_crit * se
        upper = mean_diff + t_crit * se
        return (lower, upper)
