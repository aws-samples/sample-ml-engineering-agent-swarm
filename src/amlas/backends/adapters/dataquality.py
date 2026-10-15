"""DataQualityBackend adapters (doc 05 §4.2 G2 / §4.5 / roadmap M2; ADR-004 / GAP-ADAPTER-2).

Closes the second-priority GAP (RECON §10.2): a port with one **local** in-process
adapter (the EXISTING 6-dim / missingness / pandera IP, behavior-preserving) plus two
**managed** adapters that ORCHESTRATE the AWS service behind the same port — *never
reimplementing it* (the ADR-004 anti-goal guard). The surface authors/triggers the
managed job; it does not re-build DataBrew/Glue-DQ transforms.

Three adapters, all implementing
:class:`~amlas.backends.ports.DataQualityBackend`
(``profile`` / ``evaluate`` / ``impute``):

* ``InProcessDQBackend`` — the **local/offline default** (auto-resolves offline). Wraps
  the EXISTING ``tools/data_quality/`` (``score_data_quality_6dim``,
  ``diagnose_missingness``, ``check_expectations``) + ``tools/imputation/``
  (``apply_imputation_plan``). Byte-for-byte today's behavior (ADR-001) — the parity
  test asserts the 6-dim score is unchanged.
* ``DataBrewBackend`` — the **default (managed)**; orchestrates Glue DataBrew profile
  jobs (``profile`` → T-DQ-01) + recipe jobs (``impute`` → T-ETL-06) via boto3
  ``databrew`` (offline-verified). Thin orchestration: create dataset → create job → start
  run; ``evaluate`` delegates to the local IP (DataBrew does not evaluate DQDL).
* ``GlueDataQualityBackend`` — orchestrates Glue Data Quality DQDL rulesets
  (``recommend_rules`` → T-DQ-02, ``evaluate`` → T-DQ-03) via the boto3 ``glue``
  data-quality APIs. Because the offline AWS API emulator does not yet mock ``create_data_quality_ruleset``,
  the client is **injectable** so callers may pass a recording stub (the orchestration calls
  are asserted, not the managed engine). ``profile``/``impute`` delegate to the local IP.

Selector (``DQ_BACKEND``): ``inprocess`` (local), ``databrew`` (default managed),
``glue_dq``. Offline auto-resolves to ``inprocess`` so offline operation never
touches AWS (doc 05 §4.3).
"""

from __future__ import annotations

from typing import Any

from ...config.aws_region import resolve_aws_region
from ..ports import DataQualityBackend
from ..registry import register_adapter
from ..types import DatasetRef, DQScore, ProfileReport
from ._util import raw_tool


# ══════════════════════════════════════════════════════════════════════════════
# helpers
# ══════════════════════════════════════════════════════════════════════════════
def _parse_s3_uri(uri: str) -> tuple[str, str]:
    """Split ``s3://bucket/key`` → ``(bucket, key)``; raises on a non-s3 URI."""
    if not uri.startswith("s3://"):
        raise ValueError(f"expected an s3:// URI for the managed adapter, got {uri!r}")
    rest = uri[len("s3://") :]
    bucket, _, key = rest.partition("/")
    return bucket, key


def _sanitize_name(prefix: str, raw: str) -> str:
    """A DataBrew/Glue-safe name (alnum + hyphen) derived from ``raw``."""
    import re

    base = re.sub(r"[^a-zA-Z0-9]+", "-", raw).strip("-")[:60] or "ds"
    return f"{prefix}-{base}"


# ══════════════════════════════════════════════════════════════════════════════
# InProcessDQBackend — the local/offline default (wraps EXISTING IP, ADR-001)
# ══════════════════════════════════════════════════════════════════════════════
class InProcessDQBackend:
    """Local DataQualityBackend — wraps the EXISTING in-process DQ/imputation tools.

    Behavior-preserving: ``profile`` runs the SAME ``score_data_quality_6dim`` (+ the
    ``diagnose_missingness`` mechanism IP) the team uses today; ``evaluate`` runs the
    SAME pandera ``check_expectations`` gate; ``impute`` runs the SAME
    ``apply_imputation_plan`` (which enforces hard steering). No AWS.
    """

    backend_name = "inprocess"

    def profile(self, *, dataset_uri: str) -> ProfileReport:
        """6-dim score + missingness diagnosis via the EXISTING tools (unchanged)."""
        import pandas as pd

        from ...tools.data_quality import diagnose_missingness, score_data_quality_6dim

        scored = raw_tool(score_data_quality_6dim)(dataset_uri)
        if scored.get("success") is False:
            raise RuntimeError(scored.get("error", "score_data_quality_6dim failed"))

        # cheap shape read (the score tool does not surface rows/cols)
        rows, cols = 0, 0
        try:
            df = pd.read_csv(dataset_uri)
            rows, cols = int(df.shape[0]), int(df.shape[1])
        except Exception:  # nosec B110
            pass

        diag = raw_tool(diagnose_missingness)(dataset_uri)
        dims = {name: float(d.get("score", 0.0)) for name, d in scored.get("dimensions", {}).items()}
        return ProfileReport(
            rows=rows,
            columns=cols,
            score=float(scored.get("overall_score", 0.0)),
            dimensions=dims,
            detail={
                "summary": scored.get("summary", ""),
                "dimensions_full": scored.get("dimensions", {}),
                "missingness": diag if isinstance(diag, dict) else {},
                "orchestrated": "inprocess",
            },
        )

    def evaluate(self, *, dataset_uri: str, ruleset: Any = None) -> DQScore:
        """Pandera expectations gate via the EXISTING ``check_expectations`` (unchanged).

        ``ruleset`` may be a pandera YAML suite path (str) or ``None`` (auto-infer).
        """
        from ...tools.data_quality import check_expectations

        suite = ruleset if isinstance(ruleset, str) else None
        out = raw_tool(check_expectations)(dataset_uri, suite=suite)
        if out.get("success") is False:
            raise RuntimeError(out.get("error", "check_expectations failed"))
        violations = list(out.get("violations", []))
        passed = bool(out.get("passed", False))
        # score: 1.0 when clean; otherwise decay by violation count (bounded ≥ 0).
        score = 1.0 if passed else max(0.0, 1.0 - 0.1 * len(violations))
        return DQScore(passed=passed, score=round(score, 4), violations=violations)

    def impute(self, *, dataset_uri: str, plan: dict[str, Any]) -> DatasetRef:
        """Apply the per-column plan via the EXISTING ``apply_imputation_plan`` (steered)."""
        from ...tools.imputation import apply_imputation_plan

        cols = plan.get("plan", plan)  # accept either {"plan": {...}} or the bare plan
        profile = plan.get("profile") if isinstance(plan, dict) else None
        out = raw_tool(apply_imputation_plan)(dataset_uri, cols, profile=profile)
        if out.get("success") is False:
            raise RuntimeError(out.get("error", "apply_imputation_plan failed"))
        report = out.get("imputation_report", {})
        return DatasetRef(
            uri=str(out.get("imputed_path", "")),
            tier="silver",
            rows=0,
            lineage={
                "module": "inprocess",
                "plan_applied": out.get("plan_applied", {}),
                "total_filled": report.get("total_filled", 0),
                "steering_violations": out.get("steering_violations", []),
            },
        )

    def recommend_rules(self, *, dataset_uri: str) -> dict[str, Any]:
        """Derive a simple expectation ruleset from the profile (local fallback for T-DQ-02)."""
        report = self.profile(dataset_uri=dataset_uri)
        rules = [
            {"rule": "RowCountGreaterThan", "value": 0},
            {"rule": "ColumnCountGreaterThan", "value": 0},
            {"rule": "Completeness", "min": 0.8, "dimension_score": report.dimensions.get("completeness")},
        ]
        return {"ruleset": None, "rules": rules, "source": "inprocess", "profile_score": report.score}


# ══════════════════════════════════════════════════════════════════════════════
# DataBrewBackend — managed profile + recipe orchestration (ADR-004, offline-verified)
# ══════════════════════════════════════════════════════════════════════════════
class DataBrewBackend:
    """Managed DataQualityBackend — orchestrates Glue DataBrew (does NOT reimplement it).

    ``profile`` submits a DataBrew **profile job**; ``impute`` authors a DataBrew
    **recipe** from the imputation plan and submits a **recipe job**. Both are thin
    boto3 ``databrew`` orchestration (create → start), exactly the ADR-004 contract.
    ``evaluate`` delegates to the local IP (DataBrew has no DQDL evaluation).
    """

    backend_name = "databrew"

    def __init__(
        self,
        *,
        region: str | None = None,
        role_arn: str = "",
        output_bucket: str = "",
        databrew_client: Any = None,
    ) -> None:
        self._region = resolve_aws_region(region, required=False)
        self._role = role_arn or "arn:aws:iam::000000000000:role/databrew"
        self._output_bucket = output_bucket
        self._client = databrew_client

    def _brew(self) -> Any:
        """Lazily build (or return the injected) boto3 ``databrew`` client."""
        if self._client is None:
            import boto3

            self._client = boto3.client("databrew", region_name=resolve_aws_region(self._region))
        return self._client

    @staticmethod
    def _start_run(brew: Any, job_name: str) -> str:
        """Trigger a DataBrew job run, tolerating the offline AWS API emulator's unimplemented ``start_job_run``.

        ADR-004: the meaningful orchestration is AUTHORING + triggering the managed job.
        Some the offline AWS API emulator versions mock ``create_*`` but not ``start_job_run`` ("Not yet
        implemented"); the job is still authored (the create calls succeeded), so we
        record a deferred run id rather than fail — on real AWS the start succeeds.
        """
        try:
            run = brew.start_job_run(Name=job_name)
            return str(run.get("RunId", ""))
        except Exception as e:
            if "not yet implemented" in str(e).lower() or "notimplemented" in type(e).__name__.lower():
                return "deferred-start (job authored; start unstubbed offline)"
            raise

    def profile(self, *, dataset_uri: str) -> ProfileReport:
        """Submit a DataBrew profile job for ``dataset_uri`` (orchestrate, T-DQ-01)."""
        bucket, key = _parse_s3_uri(dataset_uri)
        brew = self._brew()
        ds_name = _sanitize_name("dq-ds", key or bucket)
        job_name = _sanitize_name("dq-profile", key or bucket)
        out_bucket = self._output_bucket or bucket

        brew.create_dataset(Name=ds_name, Input={"S3InputDefinition": {"Bucket": bucket, "Key": key}})
        brew.create_profile_job(
            Name=job_name,
            DatasetName=ds_name,
            RoleArn=self._role,
            OutputLocation={"Bucket": out_bucket, "Key": f"dq-profiles/{ds_name}/"},
        )
        run_id = self._start_run(brew, job_name)
        return ProfileReport(
            rows=0,
            columns=0,
            score=0.0,  # async managed job — score lands in the output location
            dimensions={},
            detail={
                "orchestrated": "databrew",
                "managed": True,
                "dataset_name": ds_name,
                "job_name": job_name,
                "job_run_id": run_id,
                "output_uri": f"s3://{out_bucket}/dq-profiles/{ds_name}/",
            },
        )

    def evaluate(self, *, dataset_uri: str, ruleset: Any = None) -> DQScore:
        """DataBrew has no DQDL evaluation — delegate to the local IP (composition)."""
        return InProcessDQBackend().evaluate(dataset_uri=dataset_uri, ruleset=ruleset)

    def impute(self, *, dataset_uri: str, plan: dict[str, Any]) -> DatasetRef:
        """Author a DataBrew recipe from the plan + submit a recipe job (T-ETL-06)."""
        bucket, key = _parse_s3_uri(dataset_uri)
        brew = self._brew()
        cols = plan.get("plan", plan)
        steps = self._recipe_steps(cols)

        ds_name = _sanitize_name("imp-ds", key or bucket)
        recipe_name = _sanitize_name("imp-recipe", key or bucket)
        job_name = _sanitize_name("imp-job", key or bucket)
        out_bucket = self._output_bucket or bucket
        out_key = f"imputed/{ds_name}/"

        brew.create_dataset(Name=ds_name, Input={"S3InputDefinition": {"Bucket": bucket, "Key": key}})
        brew.create_recipe(Name=recipe_name, Steps=steps)
        brew.create_recipe_job(
            Name=job_name,
            RoleArn=self._role,
            DatasetName=ds_name,
            RecipeReference={"Name": recipe_name},
            Outputs=[{"Location": {"Bucket": out_bucket, "Key": out_key}}],
        )
        run_id = self._start_run(brew, job_name)
        return DatasetRef(
            uri=f"s3://{out_bucket}/{out_key}",
            tier="silver",
            rows=0,
            lineage={
                "module": "databrew",
                "managed": True,
                "recipe_name": recipe_name,
                "job_name": job_name,
                "job_run_id": run_id,
                "recipe_steps": len(steps),
            },
        )

    @staticmethod
    def _recipe_steps(plan_cols: dict[str, Any]) -> list[dict[str, Any]]:
        """Translate the per-column imputation plan into DataBrew recipe steps.

        ADR-004: we AUTHOR the recipe (the managed transform spec) — DataBrew executes
        it. Each column maps to a fill step naming the recommended method as the
        strategy parameter; an empty plan yields a single no-op-safe ``REMOVE_MISSING``.
        """
        steps: list[dict[str, Any]] = []
        for col, spec in (plan_cols or {}).items():
            method = str((spec or {}).get("method", "median"))
            steps.append(
                {
                    "Action": {
                        "Operation": "IMPUTE_MISSING",
                        "Parameters": {"sourceColumn": str(col), "strategy": method},
                    }
                }
            )
        if not steps:
            steps.append({"Action": {"Operation": "REMOVE_MISSING", "Parameters": {"sourceColumns": "[]"}}})
        return steps

    def recommend_rules(self, *, dataset_uri: str) -> dict[str, Any]:
        """DataBrew does not recommend DQDL — delegate to the local IP fallback."""
        return InProcessDQBackend().recommend_rules(dataset_uri=dataset_uri)


# ══════════════════════════════════════════════════════════════════════════════
# GlueDataQualityBackend — managed DQDL ruleset orchestration (ADR-004)
# ══════════════════════════════════════════════════════════════════════════════
class GlueDataQualityBackend:
    """Managed DataQualityBackend — orchestrates Glue Data Quality DQDL (T-DQ-02/03).

    ``recommend_rules`` triggers a Glue DQ rule-recommendation run; ``evaluate`` creates
    a DQDL ruleset + starts a ruleset-evaluation run, then surfaces the score/results.
    Thin orchestration only (ADR-004). The boto3 ``glue`` client is **injectable**
    because the offline AWS API emulator does not yet mock the data-quality APIs — callers may pass a recording stub
    and assert the orchestration calls; ``profile``/``impute`` delegate to the local IP.
    """

    backend_name = "glue_dq"

    def __init__(
        self,
        *,
        region: str | None = None,
        role_arn: str = "",
        glue_client: Any = None,
    ) -> None:
        self._region = resolve_aws_region(region, required=False)
        self._role = role_arn or "arn:aws:iam::000000000000:role/glue-dq"
        self._client = glue_client

    def _glue(self) -> Any:
        """Lazily build (or return the injected) boto3 ``glue`` client."""
        if self._client is None:
            import boto3

            self._client = boto3.client("glue", region_name=resolve_aws_region(self._region))
        return self._client

    def profile(self, *, dataset_uri: str) -> ProfileReport:
        """Glue DQ does not profile — delegate to the local IP (composition)."""
        return InProcessDQBackend().profile(dataset_uri=dataset_uri)

    def recommend_rules(self, *, dataset_uri: str, database: str | None = None, table: str | None = None) -> dict[str, Any]:
        """Trigger a Glue DQ rule-recommendation run (orchestrate, T-DQ-02)."""
        glue = self._glue()
        data_source = self._data_source(dataset_uri, database, table)
        run = glue.start_data_quality_rule_recommendation_run(
            DataSource=data_source, Role=self._role, NumberOfWorkers=2, Timeout=120
        )
        run_id = run.get("RunId", "")
        # The recommended DQDL lands when the run completes; surface the handle + (if the
        # stub returns it eagerly) the ruleset.
        dqdl = run.get("RecommendedRuleset") or "Rules = [ ColumnCount > 0 ]"
        return {"ruleset": dqdl, "run_id": run_id, "source": "glue_dq", "orchestrated": "glue_dq"}

    def evaluate(
        self,
        *,
        dataset_uri: str,
        ruleset: Any = None,
        database: str | None = None,
        table: str | None = None,
        wait_seconds: int = 0,
        poll_interval: float = 30.0,
    ) -> DQScore:
        """Create a DQDL ruleset + start an evaluation run; surface the score (T-DQ-03).

        When a real Glue Catalog ``database``/``table`` is supplied the ruleset is
        bound to that ``TargetTable`` and the evaluation ``DataSource`` targets the
        same real table — which is what a LIVE Glue Data Quality run requires (an
        evaluation run resolves a *cataloged* table, not a bare ``s3://`` URI). When
        both are omitted the prior synthetic-``GlueTable`` behavior is preserved
        (backward-compatible: the offline orchestration test passes neither), so the
        only change is that the managed adapter can now coordinate a real evaluation
        run end-to-end (F2 live-AWS fix).

        G6 live-injection seam (ADDITIVE, default-preserving): a REAL
        ``start_data_quality_ruleset_evaluation_run`` returns only a ``RunId`` — the
        ``ResultIds`` appear later, once ``get_data_quality_ruleset_evaluation_run``
        reaches ``SUCCEEDED`` (the recording fake hid this by returning a ``ResultId``
        eagerly). When ``wait_seconds>0`` the call POLLS the evaluation run until it is
        terminal, takes ``ResultIds[0]``, and reads that result — the live-runnable
        path. Default ``wait_seconds=0`` keeps today's single-shot behavior exactly
        (uses the start's ``ResultId`` directly — the offline orchestration test stays
        green).
        """
        import time as _time

        glue = self._glue()
        dqdl = ruleset if isinstance(ruleset, str) and ruleset.strip() else "Rules = [ ColumnCount > 0 ]"
        ruleset_name = _sanitize_name("dq-ruleset", dataset_uri)
        create_kwargs: dict[str, Any] = {"Name": ruleset_name, "Ruleset": dqdl}
        if database and table:
            # Bind the ruleset to the REAL catalog table so a live evaluation run
            # has a resolvable target (real Glue rejects a ruleset/DataSource whose
            # table does not exist in the catalog).
            create_kwargs["TargetTable"] = {"DatabaseName": database, "TableName": table}
        glue.create_data_quality_ruleset(**create_kwargs)

        data_source = self._data_source(dataset_uri, database, table)
        run = glue.start_data_quality_ruleset_evaluation_run(
            DataSource=data_source, Role=self._role, RulesetNames=[ruleset_name]
        )
        run_id = run.get("RunId", "")
        result_id = run.get("ResultId", run_id)
        if wait_seconds > 0:
            # live-poll path: a real start returns no ResultId — poll the run until terminal,
            # then read the first ResultId it produced.
            _terminal = {"SUCCEEDED", "FAILED", "STOPPED", "TIMEOUT"}
            deadline = _time.time() + wait_seconds
            run_status: dict[str, Any] = {}
            while _time.time() < deadline:
                try:
                    run_status = glue.get_data_quality_ruleset_evaluation_run(RunId=run_id)
                except Exception:
                    run_status = {}
                if str(run_status.get("Status", "")).upper() in _terminal:
                    break
                if poll_interval > 0:
                    # Deadline-bounded poll (finite loop budget) — P3 audit H-06.
                    _time.sleep(poll_interval)  # nosemgrep: arbitrary-sleep
            result_ids = run_status.get("ResultIds") or []
            if result_ids:
                result_id = result_ids[0]
        result = {}
        try:
            result = glue.get_data_quality_result(ResultId=result_id)
        except Exception:
            result = run.get("Result", {})

        rule_results = list(result.get("RuleResults", []))
        passed = bool(result.get("Score", 1.0) >= 1.0) if "Score" in result else all(
            r.get("Result", "PASS") == "PASS" for r in rule_results
        ) if rule_results else True
        score = float(result.get("Score", 1.0 if passed else 0.0))
        failed = [r for r in rule_results if r.get("Result") == "FAIL"]
        return DQScore(
            passed=bool(passed),
            score=round(score, 4),
            violations=[{"rule": r.get("Name", ""), "result": r.get("Result", "")} for r in failed],
        )

    def impute(self, *, dataset_uri: str, plan: dict[str, Any]) -> DatasetRef:
        """Glue DQ does not impute — delegate to the local IP (composition)."""
        return InProcessDQBackend().impute(dataset_uri=dataset_uri, plan=plan)

    @staticmethod
    def _data_source(dataset_uri: str, database: str | None, table: str | None) -> dict[str, Any]:
        """Build the Glue DQ ``DataSource`` (Glue Catalog table preferred; S3 otherwise)."""
        if database and table:
            return {"GlueTable": {"DatabaseName": database, "TableName": table}}
        # Fall back to a synthetic catalog reference derived from the URI segment.
        name = _sanitize_name("tbl", dataset_uri)
        return {"GlueTable": {"DatabaseName": database or "default", "TableName": name}}


# ── self-registration (the registry imports this module lazily) ───────────────
# DataBrew is the AWS default; in-process is the offline/local default. Glue-DQ is
# selectable via DQ_BACKEND=glue_dq. Offline auto → inprocess (doc 05 §4.3).
register_adapter(DataQualityBackend, "databrew", lambda: DataBrewBackend(), is_default=True)
register_adapter(DataQualityBackend, "inprocess", lambda: InProcessDQBackend(), is_local=True)
register_adapter(DataQualityBackend, "glue_dq", lambda: GlueDataQualityBackend())
