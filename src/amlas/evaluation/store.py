"""Evaluation Store — JSON-based experiment results persistence.

Provides:
  - ExperimentResult: dataclass for a single experiment outcome
  - EvaluationStore: thread-safe JSON file store for results
  - training_result_to_experiment: convert training tool output → ExperimentResult
  - arbiter_report_to_experiment: convert ArbiterReport → ExperimentResult
"""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class ExperimentResult:
    """Canonical schema for a single experiment run result.

    Designed for JSON serialization and cross-experiment comparison.
    """

    run_id: str
    spec_id: str
    algorithm: str
    hyperparameters: dict[str, Any]
    metrics: dict[str, float]
    primary_metric: str
    primary_score: float
    duration_seconds: float
    timestamp: str
    artifact_path: str
    status: str  # "success" | "failed" | "error"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentResult:
        """Deserialize from dictionary."""
        return cls(**data)


class EvaluationStore:
    """Thread-safe JSON-based experiment results store.

    Persists experiment results to a local JSON file. Designed for S3
    compatibility — just swap the file path for an S3 URI in production.
    """

    def __init__(self, store_path: Path | str) -> None:
        self._path = Path(store_path)
        self._lock = threading.Lock()
        self._results: list[ExperimentResult] = []
        self._load()

    def _load(self) -> None:
        """Load existing results from JSON file."""
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text())
                self._results = [ExperimentResult.from_dict(d) for d in data]
            except (json.JSONDecodeError, KeyError, TypeError):
                self._results = []
        else:
            self._results = []

    def _save(self) -> None:
        """Persist current results to JSON file."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = [r.to_dict() for r in self._results]
        # Atomic write via temp file + rename
        tmp_path = self._path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(data, indent=2))
        tmp_path.rename(self._path)

    def record(self, result: ExperimentResult) -> None:
        """Record an experiment result (thread-safe)."""
        with self._lock:
            self._results.append(result)
            self._save()

    def get_all(self) -> list[ExperimentResult]:
        """Return all recorded results."""
        return list(self._results)

    def get_by_spec(self, spec_id: str) -> list[ExperimentResult]:
        """Filter results by spec_id."""
        return [r for r in self._results if r.spec_id == spec_id]

    def get_by_algorithm(self, algorithm: str) -> list[ExperimentResult]:
        """Filter results by algorithm name."""
        return [r for r in self._results if r.algorithm == algorithm]

    def get_best_run(
        self,
        spec_id: str,
        direction: str = "maximize",
    ) -> ExperimentResult | None:
        """Get the best run for a spec based on primary_score.

        Args:
            spec_id: Filter to this spec.
            direction: "maximize" (higher is better) or "minimize" (lower is better).
        """
        candidates = [r for r in self._results if r.spec_id == spec_id and r.status == "success"]
        if not candidates:
            return None
        if direction == "minimize":
            return min(candidates, key=lambda r: r.primary_score)
        return max(candidates, key=lambda r: r.primary_score)

    def get_by_run_ids(self, run_ids: list[str]) -> list[ExperimentResult]:
        """Get specific runs by their IDs."""
        id_set = set(run_ids)
        return [r for r in self._results if r.run_id in id_set]


def training_result_to_experiment(
    *,
    training_result: dict[str, Any],
    spec_id: str,
    primary_metric: str,
) -> ExperimentResult:
    """Convert training tool output dict → ExperimentResult.

    Handles both successful and failed training runs.
    """
    success = training_result.get("success", False)
    metrics = training_result.get("metrics", {})
    primary_score = metrics.get(primary_metric, 0.0) if metrics else 0.0
    algorithm = training_result.get("algorithm", "unknown")

    return ExperimentResult(
        run_id=str(uuid.uuid4()),
        spec_id=spec_id,
        algorithm=algorithm,
        hyperparameters=training_result.get("params", {}),
        metrics=metrics,
        primary_metric=primary_metric,
        primary_score=float(primary_score),
        duration_seconds=float(training_result.get("duration_seconds", 0.0)),
        timestamp=datetime.now(UTC).isoformat(),
        artifact_path=training_result.get("model_path", ""),
        status="success" if success else "failed",
        notes=training_result.get("error", ""),
    )


def arbiter_report_to_experiment(
    *,
    report: Any,  # ArbiterReport — import avoided to prevent circular deps
    spec_id: str,
    algorithm: str,
    hyperparameters: dict[str, Any],
    artifact_path: str = "",
    duration_seconds: float = 0.0,
) -> ExperimentResult:
    """Convert an ArbiterReport → ExperimentResult.

    Merges slices into metrics alongside the overall score.
    """
    # Build metrics dict: primary metric score + all slices
    metrics: dict[str, float] = {report.primary_metric: report.overall_score}
    if report.slices:
        metrics.update(report.slices)

    return ExperimentResult(
        run_id=str(uuid.uuid4()),
        spec_id=spec_id,
        algorithm=algorithm,
        hyperparameters=hyperparameters,
        metrics=metrics,
        primary_metric=report.primary_metric,
        primary_score=float(report.overall_score),
        duration_seconds=duration_seconds,
        timestamp=datetime.now(UTC).isoformat(),
        artifact_path=artifact_path,
        status="success" if report.passes_acceptance else "failed",
        notes=report.acceptance_reason,
    )
