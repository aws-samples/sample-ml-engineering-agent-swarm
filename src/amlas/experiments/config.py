"""Experiment configuration dataclasses.

Provides ExperimentConfig (single experiment) and ExperimentPlan (collection).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run.

    Captures all information needed to reproduce an experiment:
    algorithm, hyperparameters, data source, and evaluation criteria.
    """

    experiment_id: str
    spec_id: str
    algorithm: str  # e.g., "random_forest"
    hyperparameters: dict[str, Any]
    data_path: str
    target_column: str
    task_type: str  # "classification" | "regression"
    primary_metric: str  # "f1_macro" | "rmse" | "accuracy"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentConfig:
        """Deserialize from dictionary."""
        return cls(**data)


@dataclass
class ExperimentPlan:
    """A plan containing multiple experiment configurations to run.

    Created from research recommendations — each recommended algorithm
    gets one or more configs with different hyperparameter settings.
    """

    plan_id: str
    spec_id: str
    configs: list[ExperimentConfig]
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return {
            "plan_id": self.plan_id,
            "spec_id": self.spec_id,
            "configs": [c.to_dict() for c in self.configs],
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentPlan:
        """Deserialize from dictionary."""
        configs = [ExperimentConfig.from_dict(c) for c in data["configs"]]
        return cls(
            plan_id=data["plan_id"],
            spec_id=data["spec_id"],
            configs=configs,
            created_at=data["created_at"],
        )
