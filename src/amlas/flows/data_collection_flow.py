"""``data_collection_flow`` — the Stage-4 data-collection + labeling sequence (roadmap M6).

The invariant Stage-4 sequence (doc 06 §5.3 ``data_collection_flow``): **inventory the available
data (Query/run_sql, CHEAP_READ) → decide the labeling strategy (pure reasoning) → create a
labeling job (LabelingBackend; the IRREVERSIBLE/spend step — APPROVAL-GATED) → poll progress**.

Safe-by-default (ADR-014): the inventory + strategy steps are CHEAP_READ and run UNGATED (they
provision nothing). Only the **labeling-job creation** spends money (Ground Truth incurs cost),
so that step is WRAPPED with the M1 ``plan_approve_execute`` — plan-preview → approval gate →
execute, ABORTING **before any labeling job** on denial (``status="denied"``, ``approved=False`` —
no spend). When ``create_labeling`` is False the flow stops after strategy (``status="planned"``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..control_plane.approvals import ApprovalPolicy
from ..control_plane.plan_mode import plan_approve_execute
from ..tools import research_neutral as R
from ._util import raw

__all__ = [
    "DataCollectionFlowResult",
    "decide_labeling_strategy",
    "execute_labeling_job",
    "run_data_collection_flow",
]


@dataclass
class DataCollectionFlowResult:
    """The invariant output contract of one ``data_collection_flow`` run."""

    status: str  # "completed" | "planned" | "denied" | "failed"
    approved: bool = True
    inventory: dict[str, Any] = field(default_factory=dict)
    strategy: dict[str, Any] = field(default_factory=dict)
    labeling_job: dict[str, Any] = field(default_factory=dict)
    progress: dict[str, Any] = field(default_factory=dict)
    plan: dict[str, Any] = field(default_factory=dict)
    detail: dict[str, Any] = field(default_factory=dict)


def decide_labeling_strategy(
    *,
    available_labels: int = 0,
    total_records: int = 0,
    task_type: str = "text-classification",
    budget: str = "medium",
) -> dict[str, Any]:
    """Decide the data-labeling strategy (pure reasoning; the labeling_strategist's core).

    Picks ``none`` (already labeled), ``auto`` (active-learning auto-label when budget is tight),
    or ``manual`` (human workteam) based on the labeled-coverage ratio + budget. Returns
    ``{strategy, needs_labeling, coverage, auto_label, rationale}``.
    """
    coverage = round(available_labels / total_records, 3) if total_records else 0.0
    if coverage >= 0.95:
        strategy, needs, auto = "none", False, False
        rationale = f"coverage {coverage:.0%} ≥ 95% — dataset already labeled, no job needed"
    elif budget == "low" or total_records > 50_000:
        strategy, needs, auto = "auto", True, True
        rationale = "tight budget / large corpus → Ground Truth active-learning auto-labeling"
    else:
        strategy, needs, auto = "manual", True, False
        rationale = "moderate corpus + budget → human workteam labeling for quality"
    return {"strategy": strategy, "needs_labeling": needs, "coverage": coverage,
            "auto_label": auto, "rationale": rationale, "task_type": task_type}


def execute_labeling_job(
    *,
    input_manifest: str,
    task_type: str = "text-classification",
    categories: list[str] | None = None,
    workteam: str = "private-team",
    auto_label: bool = True,
    poll: bool = True,
    steering: Any | None = None,
) -> dict[str, Any]:
    """The INNER executor (post-approval): create the labeling job + poll progress (T-LBL-01/02).

    Called by :func:`run_data_collection_flow` only AFTER the approval gate passes (Ground Truth
    incurs cost). Creates the job via the ``LabelingBackend`` port, then reads back its progress.
    """
    job = raw(R.create_labeling_job)(
        input_manifest=input_manifest, task_type=task_type,
        categories=categories or ["positive", "negative"], workteam=workteam, auto_label=auto_label)
    progress: dict[str, Any] = {}
    if poll and job.get("job_name"):
        progress = raw(R.get_labeling_progress)(job_name=job["job_name"])
    return {"labeling_job": job, "progress": progress}


def run_data_collection_flow(
    *,
    dataset: str = "",
    query: str | None = None,
    database: str | None = None,
    input_manifest: str = "",
    task_type: str = "text-classification",
    categories: list[str] | None = None,
    workteam: str = "private-team",
    available_labels: int = 0,
    total_records: int = 0,
    budget: str = "medium",
    create_labeling: bool = True,
    poll: bool = True,
    policy: ApprovalPolicy | None = None,
    steering: Any | None = None,
) -> DataCollectionFlowResult:
    """Run the Stage-4 data-collection + labeling sequence; the labeling job is APPROVAL-GATED.

    Fixed steps:
      1. **inventory** the available data (T-QRY-01, CHEAP_READ — runs only if ``query`` given);
      2. **decide** the labeling strategy (pure reasoning) — ``none``/``auto``/``manual``;
      3. if labeling is needed AND requested, **create the labeling job** (T-LBL-01) — the
         IRREVERSIBLE/spend step, WRAPPED with ``plan_approve_execute`` (aborts before any job on
         denial → ``status="denied"``); then **poll** progress (T-LBL-02).

    Returns ``status="planned"`` when no labeling is needed/requested (no spend), ``"denied"`` when
    the approval gate refuses (no job), or ``"completed"`` when the job is created.
    """
    # ── 1. inventory the available data (CHEAP_READ; only if a query is supplied) ─
    inventory: dict[str, Any] = {}
    if query:
        inventory = raw(R.run_sql)(query=query, database=database)

    # infer record/label counts from the inventory when not explicitly provided
    if not total_records and inventory.get("row_count"):
        total_records = int(inventory.get("row_count", 0))

    # ── 2. decide the labeling strategy (pure reasoning) ────────────────────────
    strategy = decide_labeling_strategy(
        available_labels=available_labels, total_records=total_records,
        task_type=task_type, budget=budget)

    # ── no labeling needed/requested → planned (no spend) ───────────────────────
    if not create_labeling or not strategy.get("needs_labeling", True):
        return DataCollectionFlowResult(
            status="planned", inventory=inventory, strategy=strategy,
            detail={"reason": "no labeling job created (already labeled or not requested)"})

    # ── 3. create the labeling job — APPROVAL-GATED (Ground Truth spend, ADR-014) ─
    outcome = plan_approve_execute(
        flow_id="data_collection_flow",
        params={
            # Squatting guard: invalid-by-construction placeholder — adopters must
            # pass input_manifest/dataset; a live call with this default fails loud.
            "input_manifest": input_manifest or dataset or "s3://<your-lake-bucket>/unlabeled/",
            "task_type": task_type, "categories": categories,
            "workteam": workteam, "auto_label": strategy.get("auto_label", True),
            "poll": poll, "steering": steering,
        },
        executor=execute_labeling_job,
        policy=policy,
        steering=steering,
    )
    if not outcome["approved"]:
        return DataCollectionFlowResult(
            status="denied", approved=False, inventory=inventory, strategy=strategy,
            plan=outcome["plan"],
            detail={"decision": outcome["decision"], "reason": "approval gate denied — no labeling job"})

    result = outcome["result"]
    return DataCollectionFlowResult(
        status="completed", approved=True, inventory=inventory, strategy=strategy,
        labeling_job=result.get("labeling_job", {}), progress=result.get("progress", {}),
        plan=outcome["plan"], detail={"strategy": strategy.get("strategy", "")},
    )
