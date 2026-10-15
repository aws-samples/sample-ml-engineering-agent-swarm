"""Local experiment-tracking adapter.

* ``LocalLeaderboardStore`` — **MVP local default**; wraps the EXISTING
  ``evaluation/store.py`` ``EvaluationStore`` (the JSON experiment store the leaderboard
  reads). Behavior-preserving (same store + ``ExperimentResult`` schema). M3 adds a
  ``build_comparison_table`` that delegates to the EXISTING
  ``evaluation/leaderboard.py`` ``Leaderboard.get_comparison_table`` IP (T-EXP-05/07).
``log_run`` records an experiment result; ``search`` returns the stored runs;
``build_comparison_table`` ranks runs for a model-comparison view (the M3 evaluate flow).
"""

from __future__ import annotations

import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..ports import ExperimentTrackingBackend
from ..registry import register_adapter


class LocalLeaderboardStore:
    """MVP local ExperimentTrackingBackend — wraps EXISTING ``evaluation/store.py``."""

    def __init__(self, *, store_path: str | Path | None = None) -> None:
        self._path = Path(store_path) if store_path else Path(tempfile.gettempdir()) / "amlas_experiments.json"

    def log_run(
        self, *, experiment: str, params: dict[str, Any], metrics: dict[str, Any],
        artifacts: dict[str, Any] | None = None,
    ) -> str:
        """Record a run into the EXISTING ``EvaluationStore`` and return its run id."""
        from ...evaluation.store import EvaluationStore, ExperimentResult

        store = EvaluationStore(self._path)
        primary_metric = next(iter(metrics), "")
        primary_score = float(metrics.get(primary_metric, 0.0)) if metrics else 0.0
        run_id = str(uuid.uuid4())
        store.record(
            ExperimentResult(
                run_id=run_id,
                spec_id=experiment,
                algorithm=str(params.get("algorithm", "unknown")),
                hyperparameters=dict(params),
                metrics={k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
                primary_metric=primary_metric,
                primary_score=primary_score,
                duration_seconds=float(params.get("duration_seconds", 0.0)),
                timestamp=datetime.now(UTC).isoformat(),
                artifact_path=str((artifacts or {}).get("model_path", "")),
                status="success",
            )
        )
        return run_id

    def search(self, *, experiment: str, filter: str | None = None, order_by: str | None = None) -> list[dict[str, Any]]:
        """Return the recorded runs for ``experiment`` from the EXISTING store."""
        from ...evaluation.store import EvaluationStore

        store = EvaluationStore(self._path)
        return [r.to_dict() for r in store.get_by_spec(experiment)]

    def build_comparison_table(self, *, experiment: str, direction: str = "maximize") -> list[dict[str, Any]]:
        """Ranked model-comparison table for ``experiment`` (T-EXP-05/07).

        Delegates to the EXISTING ``evaluation/leaderboard.py``
        ``Leaderboard.get_comparison_table`` so the comparison/ranking IP is preserved
        unchanged — the M3 ``evaluation_flow`` renders this alongside the Arbiter verdict.
        """
        from ...evaluation.leaderboard import Leaderboard
        from ...evaluation.store import EvaluationStore

        store = EvaluationStore(self._path)
        return Leaderboard(store).get_comparison_table(experiment, direction=direction)


# Local leaderboard is the supported default and local adapter.
register_adapter(ExperimentTrackingBackend, "local", lambda: LocalLeaderboardStore(), is_default=True, is_local=True)
