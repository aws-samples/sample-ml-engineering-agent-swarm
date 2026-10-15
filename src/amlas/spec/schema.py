"""Pydantic schema for task_spec.md front-matter.

See PLAN.md §8 for the canonical YAML shape. The body of task_spec.md (free-form
domain notes) is preserved as `SpecModel.domain_notes` and injected into Lead system
prompts at init time.
"""

from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

TaskType = Literal["classification", "regression", "segmentation", "detection", "text_generation"]
SplitStrategy = Literal["random", "stratified", "time_based", "group"]
# Medallion write-back tiers — silver/gold only (the "never write raw" rule of the
# lakehouse, mirrored at the schema layer; matches tools/lake.WRITABLE_TIERS).
WritableTier = Literal["silver", "gold"]

# AgentCore naming constraint — Runtime / Memory / CodeInterpreter names reject hyphens
PROJECT_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_]+$")


class Constraints(BaseModel):
    model_config = ConfigDict(extra="forbid")
    no_pii_features: bool = False
    inference_latency_ms: Annotated[int | None, Field(ge=0)] = None
    max_model_size_mb: Annotated[int | None, Field(ge=0)] = None


class Splits(BaseModel):
    model_config = ConfigDict(extra="forbid")
    strategy: SplitStrategy = "stratified"
    cutoff: str | None = None  # ISO date for time_based
    group_col: str | None = None  # column name for group strategy

    @field_validator("cutoff", mode="before")
    @classmethod
    def _coerce_cutoff(cls, v: object) -> object:
        # YAML parses unquoted dates into datetime.date; coerce to ISO string
        import datetime as _dt

        if isinstance(v, (_dt.date, _dt.datetime)):
            return v.isoformat()
        return v


class Baseline(BaseModel):
    model_config = ConfigDict(extra="allow")  # metric name varies; values are numeric
    type: str


class MedallionSpec(BaseModel):
    """Optional medallion read/write-back block (master plan §11; Phase 10 MLOps Lead).

    Drives the MLOps Lead's read→write-back loop: ``read`` points at the silver
    slice to pull (``silver://db.table`` or ``s3://...``); ``write_back`` is the
    destination tier for the cleaned/inferred artifact and is constrained to
    ``silver``/``gold`` — the lakehouse "never write raw" rule at the schema layer.
    """

    model_config = ConfigDict(extra="forbid")
    read: str | None = None  # e.g. "silver://db.table" or "s3://.../silver/x.csv"
    write_back: WritableTier = "silver"


class ComputeBudget(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # FIX-26 (operator directive: NO arbitrary caps on the work). 0 means UNBOUNDED for the
    # cap-like fields — the run/HPO/retreat loop goes to natural completion — so ``ge=0`` (not
    # ``ge=1``) and NO upper bound (the old ``le=10`` on retreats was an arbitrary ceiling). The
    # defaults are unchanged, so nothing that omits these fields changes behavior; a spec/operator
    # may now set 0 to remove the cap entirely, or any positive value to bound a specific run.
    gpu_hours: Annotated[float, Field(ge=0)] = 0
    trials: Annotated[int, Field(ge=0)] = 50            # 0 ⇒ unbounded HPO trials
    wall_clock_minutes: Annotated[int, Field(ge=0)] = 30  # 0 ⇒ unbounded wall-clock
    max_retreats: Annotated[int, Field(ge=0)] = 3        # no upper ceiling


class AcceptanceCriteria(BaseModel):
    model_config = ConfigDict(extra="allow")
    # Delta-vs-baseline expressed as a comparison string, e.g. ">= 0.005" or "<= -0.05"
    primary_metric_delta_vs_baseline: str
    no_subgroup_regression: bool = True

    @field_validator("primary_metric_delta_vs_baseline")
    @classmethod
    def _validate_delta(cls, v: str) -> str:
        if not re.match(r"^\s*(>=|<=|>|<|==)\s*-?\d+(\.\d+)?\s*$", v):
            raise ValueError(
                f"primary_metric_delta_vs_baseline must be a comparison like '>= 0.005', got: {v!r}"
            )
        return v.strip()


class SpecModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_id: str
    task_type: TaskType
    target: str
    primary_metric: str
    golden_dataset: str
    constraints: Constraints = Field(default_factory=Constraints)
    splits: Splits = Field(default_factory=Splits)
    baseline: Baseline
    compute_budget: ComputeBudget = Field(default_factory=ComputeBudget)
    acceptance: AcceptanceCriteria

    # Optional Phase 10 medallion read/write-back block (master plan §11). Defaults
    # to None so every pre-Phase-10 spec keeps validating — backward-compatible.
    medallion: MedallionSpec | None = None

    # Body content (free-form domain notes from task_spec.md) — injected into system prompts
    domain_notes: str = ""

    @field_validator("project_id")
    @classmethod
    def _validate_project_id(cls, v: str) -> str:
        if not PROJECT_ID_PATTERN.match(v):
            raise ValueError(
                f"project_id must match [a-zA-Z0-9_]+ (AgentCore naming constraint); got: {v!r}. "
                "Use underscores, not hyphens."
            )
        if len(v) > 48:
            raise ValueError(f"project_id too long ({len(v)} chars); max 48")
        return v

    @field_validator("golden_dataset")
    @classmethod
    def _validate_golden_dataset(cls, v: str) -> str:
        if not (v.startswith("s3://") or v.startswith("/") or v.startswith("./")):
            raise ValueError(
                f"golden_dataset must be s3:// or an absolute/relative filesystem path; got: {v!r}"
            )
        return v
