"""Default hyperparameter grids for supported algorithms.

Each algorithm has multiple configurations to enable comparison
of different hyperparameter settings without manual tuning.
"""

from __future__ import annotations

from typing import Any

# Default hyperparameter configurations for each supported algorithm.
# Each entry is a list of dicts — one experiment per dict.
DEFAULT_CONFIGS: dict[str, list[dict[str, Any]]] = {
    "logistic_regression": [
        {"C": 0.1},
        {"C": 1.0},
        {"C": 10.0},
    ],
    "random_forest": [
        {"n_estimators": 100, "max_depth": 10},
        {"n_estimators": 200, "max_depth": 20},
        {"n_estimators": 100, "max_depth": None},
    ],
    "gradient_boosting": [
        {"n_estimators": 100, "learning_rate": 0.1},
        {"n_estimators": 200, "learning_rate": 0.05},
        {"n_estimators": 100, "learning_rate": 0.2, "max_depth": 5},
    ],
    "xgboost": [
        {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 6},
        {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 8},
        {"n_estimators": 100, "learning_rate": 0.3, "max_depth": 4},
    ],
}
