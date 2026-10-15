"""Structured knowledge base of classical ML algorithms.

Each algorithm entry includes:
  - name, category (tree-based, linear, ensemble, neural)
  - best_for: (tabular, time-series, image, text)
  - task_types: (classification, regression, both)
  - strengths, weaknesses
  - hyperparameters: typical parameters to tune
  - complexity: training time (fast/medium/slow)
  - data_requirements: min samples, handles_missing, handles_categorical
  - sklearn_class: fully qualified class name for implementation
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Algorithm Knowledge Base — structured list of classical ML algorithms
# ---------------------------------------------------------------------------

ALGORITHM_KB: list[dict[str, Any]] = [
    {
        "name": "logistic_regression",
        "display_name": "Logistic Regression",
        "category": "linear",
        "best_for": ["tabular"],
        "task_types": ["classification"],
        "strengths": [
            "Fast training and inference",
            "Interpretable coefficients",
            "Works well with linearly separable data",
            "Low variance, less prone to overfitting with regularization",
            "Outputs calibrated probabilities",
        ],
        "weaknesses": [
            "Cannot capture non-linear relationships without feature engineering",
            "Sensitive to feature scaling",
            "Assumes independence of features",
            "Poor performance on complex decision boundaries",
        ],
        "hyperparameters": [
            {"name": "C", "type": "float", "range": [0.001, 100], "log_scale": True},
            {"name": "penalty", "type": "categorical", "values": ["l1", "l2", "elasticnet"]},
            {"name": "solver", "type": "categorical", "values": ["lbfgs", "liblinear", "saga"]},
            {"name": "max_iter", "type": "int", "range": [100, 1000]},
        ],
        "complexity": "fast",
        "data_requirements": {
            "min_samples": 50,
            "handles_missing": False,
            "handles_categorical": False,
            "prefers_scaled": True,
        },
        "sklearn_class": "sklearn.linear_model.LogisticRegression",
    },
    {
        "name": "random_forest",
        "display_name": "Random Forest",
        "category": "ensemble",
        "best_for": ["tabular"],
        "task_types": ["classification", "regression"],
        "strengths": [
            "Robust to outliers and noisy data",
            "Handles mixed feature types well",
            "Feature importance built-in",
            "Minimal hyperparameter tuning needed for decent results",
            "Parallelizable training",
            "Low risk of overfitting with enough trees",
        ],
        "weaknesses": [
            "Large model size in memory",
            "Slower inference than linear models",
            "Can struggle with very high-dimensional sparse data",
            "Less interpretable than single decision tree",
            "Biased toward features with more levels",
        ],
        "hyperparameters": [
            {"name": "n_estimators", "type": "int", "range": [50, 500]},
            {"name": "max_depth", "type": "int", "range": [3, 30]},
            {"name": "min_samples_split", "type": "int", "range": [2, 20]},
            {"name": "min_samples_leaf", "type": "int", "range": [1, 10]},
            {"name": "max_features", "type": "categorical", "values": ["sqrt", "log2", 0.5, 0.8]},
        ],
        "complexity": "medium",
        "data_requirements": {
            "min_samples": 100,
            "handles_missing": False,
            "handles_categorical": False,
            "prefers_scaled": False,
        },
        "sklearn_class": "sklearn.ensemble.RandomForestClassifier",
        "sklearn_class_regression": "sklearn.ensemble.RandomForestRegressor",
    },
    {
        "name": "gradient_boosting",
        "display_name": "Gradient Boosting (XGBoost/LightGBM/CatBoost)",
        "category": "ensemble",
        "best_for": ["tabular"],
        "task_types": ["classification", "regression"],
        "strengths": [
            "State-of-the-art performance on tabular data",
            "Handles mixed feature types",
            "Built-in handling of missing values (XGBoost, LightGBM, CatBoost)",
            "Feature importance and SHAP compatibility",
            "Excellent with moderate-to-large datasets",
            "Regularization built-in to prevent overfitting",
        ],
        "weaknesses": [
            "Requires more careful hyperparameter tuning",
            "Sequential training (slower than RF to parallelize)",
            "Can overfit on very small datasets",
            "More complex to interpret than linear models",
            "Sensitive to noisy labels",
        ],
        "hyperparameters": [
            {"name": "n_estimators", "type": "int", "range": [50, 1000]},
            {"name": "learning_rate", "type": "float", "range": [0.01, 0.3], "log_scale": True},
            {"name": "max_depth", "type": "int", "range": [3, 10]},
            {"name": "subsample", "type": "float", "range": [0.6, 1.0]},
            {"name": "colsample_bytree", "type": "float", "range": [0.6, 1.0]},
            {"name": "min_child_weight", "type": "int", "range": [1, 10]},
            {"name": "reg_alpha", "type": "float", "range": [0, 10]},
            {"name": "reg_lambda", "type": "float", "range": [0, 10]},
        ],
        "complexity": "medium",
        "data_requirements": {
            "min_samples": 200,
            "handles_missing": True,
            "handles_categorical": True,
            "prefers_scaled": False,
        },
        "sklearn_class": "sklearn.ensemble.GradientBoostingClassifier",
        "sklearn_class_regression": "sklearn.ensemble.GradientBoostingRegressor",
        "alternatives": [
            "xgboost.XGBClassifier",
            "lightgbm.LGBMClassifier",
            "catboost.CatBoostClassifier",
        ],
    },
    {
        "name": "svm",
        "display_name": "Support Vector Machine",
        "category": "kernel",
        "best_for": ["tabular", "text"],
        "task_types": ["classification", "regression"],
        "strengths": [
            "Effective in high-dimensional spaces",
            "Memory efficient (uses support vectors subset)",
            "Versatile through different kernels (linear, rbf, poly)",
            "Strong theoretical guarantees (margin maximization)",
        ],
        "weaknesses": [
            "Does not scale well beyond ~10K samples (O(n^2) to O(n^3))",
            "Sensitive to feature scaling",
            "No probabilistic output by default (requires calibration)",
            "Does not handle missing values",
            "Kernel selection can be tricky",
        ],
        "hyperparameters": [
            {"name": "C", "type": "float", "range": [0.01, 100], "log_scale": True},
            {"name": "kernel", "type": "categorical", "values": ["rbf", "linear", "poly"]},
            {"name": "gamma", "type": "float", "range": [0.001, 10], "log_scale": True},
            {"name": "degree", "type": "int", "range": [2, 5]},
        ],
        "complexity": "slow",
        "data_requirements": {
            "min_samples": 50,
            "handles_missing": False,
            "handles_categorical": False,
            "prefers_scaled": True,
        },
        "sklearn_class": "sklearn.svm.SVC",
        "sklearn_class_regression": "sklearn.svm.SVR",
    },
    {
        "name": "knn",
        "display_name": "k-Nearest Neighbors",
        "category": "instance_based",
        "best_for": ["tabular"],
        "task_types": ["classification", "regression"],
        "strengths": [
            "Simple and intuitive",
            "No training phase (lazy learner)",
            "Adapts to any decision boundary shape",
            "Good for small to medium datasets",
        ],
        "weaknesses": [
            "Slow inference on large datasets (O(n) per query)",
            "Very sensitive to feature scaling",
            "Curse of dimensionality in high dimensions",
            "No feature importance",
            "Does not handle missing values",
            "Memory-intensive (stores all training data)",
        ],
        "hyperparameters": [
            {"name": "n_neighbors", "type": "int", "range": [3, 30]},
            {"name": "weights", "type": "categorical", "values": ["uniform", "distance"]},
            {"name": "metric", "type": "categorical", "values": ["euclidean", "manhattan", "minkowski"]},
            {"name": "p", "type": "int", "range": [1, 3]},
        ],
        "complexity": "fast",
        "data_requirements": {
            "min_samples": 30,
            "handles_missing": False,
            "handles_categorical": False,
            "prefers_scaled": True,
        },
        "sklearn_class": "sklearn.neighbors.KNeighborsClassifier",
        "sklearn_class_regression": "sklearn.neighbors.KNeighborsRegressor",
    },
    {
        "name": "decision_tree",
        "display_name": "Decision Tree",
        "category": "tree_based",
        "best_for": ["tabular"],
        "task_types": ["classification", "regression"],
        "strengths": [
            "Highly interpretable (visual tree structure)",
            "Handles mixed feature types naturally",
            "No feature scaling required",
            "Captures non-linear relationships",
            "Fast training and inference",
        ],
        "weaknesses": [
            "Prone to overfitting without pruning",
            "High variance (unstable to small data changes)",
            "Biased toward features with more levels",
            "Greedy splitting may miss global optima",
            "Generally outperformed by ensembles",
        ],
        "hyperparameters": [
            {"name": "max_depth", "type": "int", "range": [3, 20]},
            {"name": "min_samples_split", "type": "int", "range": [2, 20]},
            {"name": "min_samples_leaf", "type": "int", "range": [1, 10]},
            {"name": "criterion", "type": "categorical", "values": ["gini", "entropy"]},
            {"name": "max_features", "type": "categorical", "values": ["sqrt", "log2", None]},
        ],
        "complexity": "fast",
        "data_requirements": {
            "min_samples": 30,
            "handles_missing": False,
            "handles_categorical": False,
            "prefers_scaled": False,
        },
        "sklearn_class": "sklearn.tree.DecisionTreeClassifier",
        "sklearn_class_regression": "sklearn.tree.DecisionTreeRegressor",
    },
    {
        "name": "adaboost",
        "display_name": "AdaBoost",
        "category": "ensemble",
        "best_for": ["tabular"],
        "task_types": ["classification", "regression"],
        "strengths": [
            "Simple boosting approach with few hyperparameters",
            "Less prone to overfitting than single trees",
            "Adapts to difficult examples",
            "Works well with weak learners",
        ],
        "weaknesses": [
            "Sensitive to noisy data and outliers",
            "Generally outperformed by gradient boosting",
            "Sequential training (not parallelizable)",
            "Performance degrades with insufficient data",
        ],
        "hyperparameters": [
            {"name": "n_estimators", "type": "int", "range": [50, 500]},
            {"name": "learning_rate", "type": "float", "range": [0.01, 1.0], "log_scale": True},
            {"name": "algorithm", "type": "categorical", "values": ["SAMME", "SAMME.R"]},
        ],
        "complexity": "medium",
        "data_requirements": {
            "min_samples": 100,
            "handles_missing": False,
            "handles_categorical": False,
            "prefers_scaled": False,
        },
        "sklearn_class": "sklearn.ensemble.AdaBoostClassifier",
        "sklearn_class_regression": "sklearn.ensemble.AdaBoostRegressor",
    },
    {
        "name": "elastic_net",
        "display_name": "Elastic Net",
        "category": "linear",
        "best_for": ["tabular"],
        "task_types": ["regression"],
        "strengths": [
            "Combines L1 (sparsity) and L2 (stability) regularization",
            "Feature selection built-in via L1 component",
            "Handles multicollinearity better than Lasso alone",
            "Fast training",
            "Interpretable coefficients",
        ],
        "weaknesses": [
            "Linear model — cannot capture non-linear relationships",
            "Requires feature scaling",
            "Two hyperparameters to tune (alpha + l1_ratio)",
            "Does not handle missing values",
        ],
        "hyperparameters": [
            {"name": "alpha", "type": "float", "range": [0.0001, 10], "log_scale": True},
            {"name": "l1_ratio", "type": "float", "range": [0.0, 1.0]},
            {"name": "max_iter", "type": "int", "range": [1000, 10000]},
        ],
        "complexity": "fast",
        "data_requirements": {
            "min_samples": 30,
            "handles_missing": False,
            "handles_categorical": False,
            "prefers_scaled": True,
        },
        "sklearn_class": "sklearn.linear_model.ElasticNet",
    },
    {
        "name": "ridge_regression",
        "display_name": "Ridge Regression",
        "category": "linear",
        "best_for": ["tabular"],
        "task_types": ["regression"],
        "strengths": [
            "Handles multicollinearity via L2 regularization",
            "Closed-form solution (very fast)",
            "Stable with correlated features",
            "Interpretable",
            "Works well when most features are relevant",
        ],
        "weaknesses": [
            "Cannot perform feature selection (all coefficients non-zero)",
            "Linear model limitations",
            "Requires feature scaling",
            "Does not handle missing values",
        ],
        "hyperparameters": [
            {"name": "alpha", "type": "float", "range": [0.001, 100], "log_scale": True},
        ],
        "complexity": "fast",
        "data_requirements": {
            "min_samples": 20,
            "handles_missing": False,
            "handles_categorical": False,
            "prefers_scaled": True,
        },
        "sklearn_class": "sklearn.linear_model.Ridge",
    },
    {
        "name": "lasso_regression",
        "display_name": "Lasso Regression",
        "category": "linear",
        "best_for": ["tabular"],
        "task_types": ["regression"],
        "strengths": [
            "Automatic feature selection via L1 regularization",
            "Sparse solutions (many zero coefficients)",
            "Fast training",
            "Interpretable — identifies important features",
            "Good for high-dimensional data with few relevant features",
        ],
        "weaknesses": [
            "Selects at most n features when p > n",
            "Unstable with correlated features (picks one arbitrarily)",
            "Linear model limitations",
            "Requires feature scaling",
        ],
        "hyperparameters": [
            {"name": "alpha", "type": "float", "range": [0.0001, 10], "log_scale": True},
            {"name": "max_iter", "type": "int", "range": [1000, 10000]},
        ],
        "complexity": "fast",
        "data_requirements": {
            "min_samples": 20,
            "handles_missing": False,
            "handles_categorical": False,
            "prefers_scaled": True,
        },
        "sklearn_class": "sklearn.linear_model.Lasso",
    },
    {
        "name": "naive_bayes",
        "display_name": "Naive Bayes",
        "category": "probabilistic",
        "best_for": ["tabular", "text"],
        "task_types": ["classification"],
        "strengths": [
            "Extremely fast training and inference",
            "Works well with high-dimensional data (text classification)",
            "Good with small training sets",
            "Outputs calibrated probabilities",
            "Simple to implement and interpret",
        ],
        "weaknesses": [
            "Strong independence assumption (rarely holds in practice)",
            "Cannot learn feature interactions",
            "Zero frequency problem requires smoothing",
            "Generally outperformed by more complex models on tabular data",
        ],
        "hyperparameters": [
            {"name": "var_smoothing", "type": "float", "range": [1e-12, 1e-6], "log_scale": True},
        ],
        "complexity": "fast",
        "data_requirements": {
            "min_samples": 20,
            "handles_missing": False,
            "handles_categorical": True,
            "prefers_scaled": False,
        },
        "sklearn_class": "sklearn.naive_bayes.GaussianNB",
        "alternatives": [
            "sklearn.naive_bayes.MultinomialNB",
            "sklearn.naive_bayes.BernoulliNB",
        ],
    },
]


# ---------------------------------------------------------------------------
# Query Functions
# ---------------------------------------------------------------------------


def get_algorithms_by_task(task_type: str) -> list[dict[str, Any]]:
    """Filter algorithms by task type (classification or regression).

    Args:
        task_type: One of 'classification' or 'regression'.

    Returns:
        List of algorithm entries supporting the given task type.
    """
    return [algo for algo in ALGORITHM_KB if task_type in algo["task_types"]]


def get_algorithms_by_data_type(data_type: str) -> list[dict[str, Any]]:
    """Filter algorithms by data type (tabular, text, image, time-series).

    Args:
        data_type: One of 'tabular', 'text', 'image', 'time-series'.

    Returns:
        List of algorithm entries suitable for the given data type.
    """
    return [algo for algo in ALGORITHM_KB if data_type in algo["best_for"]]


def get_algorithms_by_category(category: str) -> list[dict[str, Any]]:
    """Filter algorithms by category (linear, ensemble, tree_based, etc.).

    Args:
        category: Algorithm category string.

    Returns:
        List of algorithm entries in the given category.
    """
    return [algo for algo in ALGORITHM_KB if algo["category"] == category]


def rank_algorithms_for_task(
    task_type: str,
    data_type: str = "tabular",
    data_size: str = "medium",
) -> list[dict[str, Any]]:
    """Return algorithms ranked by suitability for a given task configuration.

    Ranking heuristic:
      - Ensemble methods ranked highest for tabular data
      - Linear models ranked higher for small data
      - Neural approaches deprioritized for small/medium tabular data

    Args:
        task_type: 'classification' or 'regression'
        data_type: 'tabular', 'text', 'image', 'time-series'
        data_size: 'small' (<500), 'medium' (500-10K), 'large' (>10K)

    Returns:
        Ranked list of algorithm dicts with added 'rank' and 'rationale' fields.
    """
    candidates = [
        algo for algo in ALGORITHM_KB
        if task_type in algo["task_types"] and data_type in algo["best_for"]
    ]

    # Scoring heuristic
    def _score(algo: dict[str, Any]) -> float:
        score = 0.0
        cat = algo["category"]

        if data_type == "tabular":
            # Ensemble methods excel on tabular
            if cat == "ensemble":
                score += 10.0
                # Gradient boosting is SOTA for tabular
                if "gradient" in algo["name"]:
                    score += 5.0
            elif cat == "tree_based":
                score += 5.0
            elif cat == "linear":
                score += 3.0
            elif cat == "probabilistic":
                score += 2.0
            elif cat == "instance_based":
                score += 1.0
            elif cat == "kernel":
                score += 4.0

        # Prefer fast models for small data (they generalize better)
        if data_size == "small":
            if algo["complexity"] == "fast":
                score += 3.0
            elif algo["complexity"] == "medium":
                score += 1.0
            # Penalize methods that need lots of data
            min_samples = algo["data_requirements"].get("min_samples", 0)
            if min_samples > 100:
                score -= 2.0

        # For large data, prefer scalable methods
        if data_size == "large":
            if cat in ("ensemble", "linear"):
                score += 2.0
            if cat == "instance_based":
                score -= 5.0  # kNN is O(n) per query
            if cat == "kernel":
                score -= 3.0  # SVM is O(n^2)

        return score

    # Sort by score descending
    scored = [(algo, _score(algo)) for algo in candidates]
    scored.sort(key=lambda x: -x[1])

    # Add rank and rationale
    ranked = []
    for i, (algo, _score_val) in enumerate(scored, 1):
        entry = {**algo, "rank": i}
        # Generate rationale
        rationale_parts = []
        if algo["category"] == "ensemble":
            rationale_parts.append("ensemble method with strong tabular performance")
        if algo["data_requirements"].get("handles_missing"):
            rationale_parts.append("handles missing values natively")
        if data_size == "small" and algo["complexity"] == "fast":
            rationale_parts.append("fast training suitable for small datasets")
        if "gradient" in algo["name"]:
            rationale_parts.append("state-of-the-art for tabular classification/regression")
        if algo["category"] == "linear":
            rationale_parts.append("fast baseline with interpretable coefficients")

        entry["rationale"] = "; ".join(rationale_parts) if rationale_parts else (
            f"{algo['display_name']} — suitable for {task_type} on {data_type} data"
        )
        ranked.append(entry)

    return ranked
