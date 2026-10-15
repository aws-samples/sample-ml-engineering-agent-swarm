"""Kanban run state — ml-ralph pattern.

Schema ported from `ml-ralph/src/domain/types/kanban.ts`. Three first-class lanes:
    - active: current iteration work
    - completed: successful passes with their scores
    - abandoned: failed experiments with abandoned_reason — the retreat ledger

Persisted to `runs/<run_id>/kanban.json` after every phase transition.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Phase = Literal["R1_FRAMING", "R2_FEATURE_HYP", "R3_ALGO_HYP", "D1_FE", "D2_ML_EXEC", "D3_ARBITER"]


class KanbanTask(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    phase: Phase
    iteration: int
    title: str
    started_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class CompletedTask(KanbanTask):
    completed_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    score: float | None = None
    metric: str | None = None


class AbandonedTask(KanbanTask):
    abandoned_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    abandoned_reason: str
    learnings: list[str] = Field(default_factory=list)


class RunKanban(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str
    project_id: str
    current_phase: Phase = "R1_FRAMING"
    current_iteration: int = 1
    active: list[KanbanTask] = Field(default_factory=list)
    completed: list[CompletedTask] = Field(default_factory=list)
    abandoned: list[AbandonedTask] = Field(default_factory=list)

    def start(self, task: KanbanTask) -> None:
        self.active.append(task)

    def complete(self, task_id: str, *, score: float | None = None, metric: str | None = None) -> None:
        t = self._pop_active(task_id)
        self.completed.append(
            CompletedTask(**t.model_dump(), score=score, metric=metric)
        )

    def abandon(self, task_id: str, *, reason: str, learnings: list[str] | None = None) -> None:
        t = self._pop_active(task_id)
        self.abandoned.append(
            AbandonedTask(**t.model_dump(), abandoned_reason=reason, learnings=list(learnings or []))
        )

    def advance_phase(self, next_phase: Phase) -> None:
        self.current_phase = next_phase

    def retreat(self, next_iteration: int) -> None:
        self.current_iteration = next_iteration
        self.current_phase = "R1_FRAMING"

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.model_dump(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> RunKanban:
        return cls.model_validate(json.loads(Path(path).read_text()))

    # --- internals ---
    def _pop_active(self, task_id: str) -> KanbanTask:
        for i, t in enumerate(self.active):
            if t.id == task_id:
                return self.active.pop(i)
        raise KeyError(f"no active task with id={task_id!r}")
