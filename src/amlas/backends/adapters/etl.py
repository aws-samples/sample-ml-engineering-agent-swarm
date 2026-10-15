"""ETLBackend adapters (doc 05 §4.2 G2 / roadmap M2; ADR-004 / GAP-ADAPTER-2).

The transform/crawl half of the GAP-ADAPTER-2 close: one **local** in-process pandas
adapter (wraps the EXISTING ``auto_preprocess`` + adds the dedup/clean/encode/normalize/
split transforms) plus two **managed** adapters that ORCHESTRATE SageMaker Processing
and AWS Glue — *never reimplementing the engine* (ADR-004). The surface authors/triggers
the managed job; it does not re-build Glue/Processing transforms.

Three adapters, all implementing
:class:`~amlas.backends.ports.ETLBackend` (``run_transform`` / ``run_crawl``):

* ``InProcessPandasBackend`` — the **local/offline default** (auto-resolves offline).
  Wraps EXISTING ``auto_preprocess`` (T-ETL-14) and runs in-process pandas transforms —
  ``dedupe`` (T-ETL-07), ``clean_text`` (T-ETL-08), ``normalize``, ``encode``, ``split``
  — selected by ``transform_spec["op"]``. Behavior-preserving for the wrapped
  ``auto_preprocess`` (ADR-001 parity test).
* ``ProcessingJobBackend`` — the **default (managed)**; orchestrates a SageMaker
  Processing job (T-ETL-01) via boto3 ``sagemaker`` ``create_processing_job``
  (offline-verified). Thin orchestration: build the ProcessingInputs/Outputs from the spec,
  submit, return the job handle + status.
* ``GlueETLBackend`` — orchestrates AWS Glue jobs (T-ETL-02) + crawlers (T-ETL-03) via
  boto3 ``glue`` ``create_job``/``start_job_run`` and ``create_crawler``/``start_crawler``
  (offline-verified).

Selector (``ETL_BACKEND``): ``inprocess`` (local), ``processing`` (default managed),
``glue``. Offline auto-resolves to ``inprocess`` (doc 05 §4.3).
"""

from __future__ import annotations

import os
import uuid
from typing import Any

from ...config.aws_region import resolve_aws_client_region, resolve_aws_region
from ...config.deployment import deployment_job_tags, resolve_deployment_prefix
from ...config.sagemaker_images import sklearn_processing_image, validate_image_region
from ..ports import ETLBackend
from ..registry import register_adapter
from ._util import raw_tool


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    """Split ``s3://bucket/key`` → ``(bucket, key)``; raises on a non-s3 URI."""
    if not uri.startswith("s3://"):
        raise ValueError(f"expected an s3:// URI for the managed adapter, got {uri!r}")
    rest = uri[len("s3://") :]
    bucket, _, key = rest.partition("/")
    return bucket, key


def _sanitize_name(prefix: str, raw: str) -> str:
    """A SageMaker/Glue-safe job name (alnum + hyphen, no underscores per the rules)."""
    import re
    import uuid

    name_budget = min(40, 63 - len(prefix) - 10)
    base = re.sub(r"[^a-zA-Z0-9]+", "-", raw).strip("-")[:name_budget].rstrip("-") or "job"
    return f"{prefix}-{base}-{uuid.uuid4().hex[:8]}"


def _out_path(input_uri: str, op: str) -> str:
    """A deterministic local output path next to the input for in-process transforms."""
    from pathlib import Path

    p = Path(input_uri)
    return str(p.with_name(f"{p.stem}.{op}{p.suffix or '.csv'}"))


# ══════════════════════════════════════════════════════════════════════════════
# InProcessPandasBackend — local/offline default (wraps EXISTING auto_preprocess)
# ══════════════════════════════════════════════════════════════════════════════
class InProcessPandasBackend:
    """Local ETLBackend — in-process pandas transforms + the EXISTING ``auto_preprocess``.

    ``run_transform`` dispatches on ``transform_spec["op"]``:
      * ``auto_preprocess`` → EXISTING ``fe_tools.auto_preprocess`` (T-ETL-14, unchanged);
      * ``dedupe`` → exact / key-based row de-duplication (T-ETL-07);
      * ``clean_text`` → strip/whitespace/lowercase text ops (T-ETL-08);
      * ``normalize`` → min-max / z-score scaling of numeric columns;
      * ``encode`` → one-hot encoding of categorical columns;
      * ``split`` → deterministic train/test split (writes two outputs).
    No AWS — runs entirely in pandas.
    """

    backend_name = "inprocess"

    def run_transform(
        self,
        *,
        input_uri: str,
        transform_spec: dict[str, Any],
        output: str,
        resources: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run an in-process pandas transform selected by ``transform_spec['op']``."""
        op = str((transform_spec or {}).get("op", "auto_preprocess")).strip().lower()
        dispatch = {
            "auto_preprocess": self._auto_preprocess,
            "dedupe": self._dedupe,
            "dedup": self._dedupe,
            "clean_text": self._clean_text,
            "normalize": self._normalize,
            "encode": self._encode,
            "split": self._split,
        }
        fn = dispatch.get(op)
        if fn is None:
            return {"success": False, "error": f"unknown in-process transform op {op!r}", "op": op}
        result = fn(input_uri, transform_spec, output)
        result.setdefault("op", op)
        result.setdefault("orchestrated", "inprocess")
        return result

    def run_crawl(self, *, s3_target: str) -> dict[str, Any]:
        """In-process 'crawl' = a cheap local schema inventory (no Glue catalog)."""
        import pandas as pd

        try:
            df = pd.read_csv(s3_target)
        except Exception as e:
            return {"success": False, "error": f"failed to read {s3_target}: {e}"}
        schema = {c: str(df[c].dtype) for c in df.columns}
        return {
            "success": True,
            "orchestrated": "inprocess",
            "tables_updated": 1,
            "schema": schema,
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
        }

    # ── transform ops ───────────────────────────────────────────────────────
    def _auto_preprocess(self, input_uri: str, spec: dict[str, Any], output: str) -> dict[str, Any]:
        from ...tools.fe_tools import auto_preprocess

        target = str(spec.get("target_column", spec.get("target", "")))
        out_dir = spec.get("output_dir") or (output if output and not output.endswith(".csv") else None)
        out = raw_tool(auto_preprocess)(
            path=input_uri,
            target_column=target,
            output_dir=out_dir,
            drop_threshold=float(spec.get("drop_threshold", 0.5)),
        )
        return dict(out)

    def _dedupe(self, input_uri: str, spec: dict[str, Any], output: str) -> dict[str, Any]:
        import pandas as pd

        df = pd.read_csv(input_uri)
        before = len(df)
        keys = spec.get("keys")
        strategy = str(spec.get("strategy", "exact")).lower()
        if strategy == "key" and keys:
            deduped = df.drop_duplicates(subset=list(keys), keep="first")
        else:
            deduped = df.drop_duplicates(keep="first")
        removed = before - len(deduped)
        out_path = output if output.endswith(".csv") else _out_path(input_uri, "dedupe")
        deduped.to_csv(out_path, index=False)
        return {
            "success": True,
            "output_uri": out_path,
            "removed_count": int(removed),
            "rows_before": int(before),
            "rows_after": len(deduped),
        }

    def _clean_text(self, input_uri: str, spec: dict[str, Any], output: str) -> dict[str, Any]:
        import pandas as pd

        df = pd.read_csv(input_uri)
        cols = spec.get("columns") or df.select_dtypes(include=["object"]).columns.tolist()
        ops = [str(o).lower() for o in (spec.get("ops") or ["strip", "collapse_ws"])]
        cleaned = 0
        for c in cols:
            if c not in df.columns:
                continue
            s = df[c].astype(str)
            if "strip" in ops:
                s = s.str.strip()
            if "lower" in ops or "lowercase" in ops:
                s = s.str.lower()
            if "collapse_ws" in ops:
                s = s.str.replace(r"\s+", " ", regex=True)
            df[c] = s
            cleaned += 1
        out_path = output if output.endswith(".csv") else _out_path(input_uri, "clean_text")
        df.to_csv(out_path, index=False)
        return {"success": True, "output_uri": out_path, "columns_cleaned": cleaned, "removed_artifacts": 0}

    def _normalize(self, input_uri: str, spec: dict[str, Any], output: str) -> dict[str, Any]:
        import pandas as pd

        df = pd.read_csv(input_uri)
        method = str(spec.get("method", "zscore")).lower()
        target = spec.get("target_column") or spec.get("target")
        num_cols = [c for c in df.select_dtypes(include=["number"]).columns if c != target]
        for c in num_cols:
            col = df[c].astype(float)
            if method in ("minmax", "min_max"):
                lo, hi = col.min(), col.max()
                df[c] = (col - lo) / (hi - lo) if hi > lo else 0.0
            else:  # zscore
                mu, sd = col.mean(), col.std()
                df[c] = (col - mu) / sd if sd > 0 else 0.0
        out_path = output if output.endswith(".csv") else _out_path(input_uri, "normalize")
        df.to_csv(out_path, index=False)
        return {
            "success": True,
            "output_uri": out_path,
            "method": method,
            "columns_normalized": len(num_cols),
        }

    def _encode(self, input_uri: str, spec: dict[str, Any], output: str) -> dict[str, Any]:
        import pandas as pd

        df = pd.read_csv(input_uri)
        target = spec.get("target_column") or spec.get("target")
        cat_cols = [c for c in df.select_dtypes(include=["object", "bool"]).columns if c != target]
        encoded = pd.get_dummies(df, columns=cat_cols, dummy_na=False)
        out_path = output if output.endswith(".csv") else _out_path(input_uri, "encode")
        encoded.to_csv(out_path, index=False)
        return {
            "success": True,
            "output_uri": out_path,
            "columns_encoded": len(cat_cols),
            "n_features_after": int(encoded.shape[1]),
        }

    def _split(self, input_uri: str, spec: dict[str, Any], output: str) -> dict[str, Any]:
        import pandas as pd
        from sklearn.model_selection import train_test_split

        df = pd.read_csv(input_uri)
        test_size = float(spec.get("test_size", 0.2))
        seed = int(spec.get("seed", 42))
        target = spec.get("target_column") or spec.get("target")
        strat = (
            df[target]
            if (target in df.columns and df[target].nunique() > 1 and df[target].value_counts().min() >= 2)
            else None
        )
        try:
            train, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=strat)
        except ValueError:
            train, test = train_test_split(df, test_size=test_size, random_state=seed)
        train_path = _out_path(input_uri, "train")
        test_path = _out_path(input_uri, "test")
        train.to_csv(train_path, index=False)
        test.to_csv(test_path, index=False)
        return {
            "success": True,
            "output_uri": train_path,
            "train_uri": train_path,
            "test_uri": test_path,
            "train_rows": len(train),
            "test_rows": len(test),
        }


# ══════════════════════════════════════════════════════════════════════════════
# ProcessingJobBackend — managed SageMaker Processing orchestration (ADR-004)
# ══════════════════════════════════════════════════════════════════════════════
class ProcessingJobBackend:
    """Managed ETLBackend — orchestrates a SageMaker Processing job (T-ETL-01).

    Thin boto3 ``sagemaker`` orchestration: build ProcessingInputs/Outputs from the
    spec, ``create_processing_job``, return the handle + status. The Processing
    *container* runs the transform — we never reimplement it (ADR-004). Crawl is N/A
    for Processing (delegates to Glue/local). The client is injectable for the offline AWS API emulator.
    """

    backend_name = "processing"

    def __init__(
        self,
        *,
        region: str | None = None,
        role_arn: str = "",
        default_image: str = "",
        instance_type: str = "ml.m5.large",
        sagemaker_client: Any = None,
    ) -> None:
        self._region = region
        # Resolve the SageMaker exec role: explicit arg → SEC-01 env
        # (SAGEMAKER_EXECUTION_ROLE_ARN / AMLAS_SAGEMAKER_ROLE_ARN /
        # AMLAS_SAGEMAKER_ROLE_ARN) → the placeholder. Mirrors the deployment +
        # explainability adapters so a managed Processing job passes a REAL role
        # (a 000000000000 placeholder is rejected with "Cross-account pass role
        # is not allowed"). NB: the role's trust policy must allow
        # ``sagemaker.amazonaws.com`` — an Admin role does NOT, so the purpose-built
        # ``amlas-sagemaker-exec`` role is the correct execution role here.
        import os as _os

        self._role = (
            role_arn
            or _os.environ.get("SAGEMAKER_EXECUTION_ROLE_ARN", "")
            or _os.environ.get("AMLAS_SAGEMAKER_ROLE_ARN", "")
            or _os.environ.get("AMLAS_SAGEMAKER_ROLE_ARN", "")
            or "arn:aws:iam::000000000000:role/processing"
        )
        # ``config.sagemaker_images`` owns the URI. This line previously hardcoded BOTH the tag
        # and ``us-west-2`` + the us-* DLC account, so it composed an unpullable image in any
        # other region.
        self._default_image = default_image
        self._instance_type = instance_type
        self._sm = sagemaker_client

    def _client(self) -> Any:
        """Lazily build (or return the injected) boto3 ``sagemaker`` client."""
        if self._sm is None:
            import boto3

            self._sm = boto3.client("sagemaker", region_name=resolve_aws_region(self._region))
        return self._sm

    def run_transform(
        self,
        *,
        input_uri: str,
        transform_spec: dict[str, Any],
        output: str,
        resources: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Submit a SageMaker Processing job that runs ``transform_spec`` (T-ETL-01)."""
        res = dict(resources or {})
        runtime_session_id = str(
            res.get("runtime_session_id")
            or os.environ.get("RUNTIME_SESSION_ID")
            or f"processing-{uuid.uuid4()}"
        )
        job_tags = deployment_job_tags(runtime_session_id)
        sm = self._client()
        job_name = _sanitize_name(
            f"{resolve_deployment_prefix()}-etl-proc", str(transform_spec.get("op", "transform"))
        )
        override = res.get("image") or transform_spec.get("image") or self._default_image
        region = resolve_aws_client_region(sm, self._region, required=not bool(override))
        image = override or sklearn_processing_image(region)
        image = validate_image_region(str(image), region)
        script_ref = transform_spec.get("script_ref", "")

        app_spec: dict[str, Any] = {"ImageUri": image}
        if script_ref:
            app_spec["ContainerEntrypoint"] = ["python3", script_ref]
        args = transform_spec.get("arguments")
        if args:
            app_spec["ContainerArguments"] = [str(a) for a in args]

        cluster = {
            "InstanceCount": int(res.get("instance_count", transform_spec.get("instance_count", 1))),
            "InstanceType": res.get("instance_type", self._instance_type),
            "VolumeSizeInGB": int(res.get("volume_size_gb", 20)),
        }
        kwargs: dict[str, Any] = {
            "ProcessingJobName": job_name,
            "RoleArn": self._role,
            "AppSpecification": app_spec,
            "ProcessingResources": {"ClusterConfig": cluster},
            "ProcessingInputs": [
                {
                    "InputName": "input",
                    "S3Input": {
                        "S3Uri": input_uri,
                        "LocalPath": "/opt/ml/processing/input",
                        "S3DataType": "S3Prefix",
                        "S3InputMode": "File",
                    },
                }
            ],
            "ProcessingOutputConfig": {
                "Outputs": [
                    {
                        "OutputName": "output",
                        "S3Output": {
                            "S3Uri": output,
                            "LocalPath": "/opt/ml/processing/output",
                            "S3UploadMode": "EndOfJob",
                        },
                    }
                ]
            },
        }
        if job_tags:
            kwargs["Tags"] = job_tags
        accepted = sm.create_processing_job(**kwargs)
        accepted_arn = accepted.get("ProcessingJobArn", "")
        observation_error = ""
        try:
            desc = sm.describe_processing_job(ProcessingJobName=job_name)
        except Exception as error:
            # Native acceptance and atomic tags remain authoritative when a read fails.
            desc = {"ProcessingJobStatus": "Submitted", "ProcessingJobArn": accepted_arn}
            observation_error = type(error).__name__
        return {
            "success": True,
            "orchestrated": "processing",
            "managed": True,
            "job_name": job_name,
            "status": desc.get("ProcessingJobStatus", "InProgress"),
            "arn": accepted_arn or desc.get("ProcessingJobArn", ""),
            "accepted": True,
            "runtime_session_id": runtime_session_id,
            "observation_error": observation_error,
            "output_uris": [output],
            "op": transform_spec.get("op", "transform"),
        }

    def run_crawl(self, *, s3_target: str) -> dict[str, Any]:
        """Processing does not crawl — defer to the Glue adapter / local inventory."""
        return GlueETLBackend(region=self._region).run_crawl(s3_target=s3_target)


# ══════════════════════════════════════════════════════════════════════════════
# GlueETLBackend — managed Glue job + crawler orchestration (ADR-004, offline-verified)
# ══════════════════════════════════════════════════════════════════════════════
class GlueETLBackend:
    """Managed ETLBackend — orchestrates AWS Glue jobs (T-ETL-02) + crawlers (T-ETL-03).

    Thin boto3 ``glue`` orchestration: ``run_transform`` ensures a Glue job exists then
    ``start_job_run``; ``run_crawl`` ensures a crawler exists then ``start_crawler``. The
    Glue Spark job runs the transform — we never reimplement it (ADR-004). The client is
    injectable for the offline AWS API emulator.
    """

    backend_name = "glue"

    def __init__(
        self,
        *,
        region: str | None = None,
        role_arn: str = "",
        glue_client: Any = None,
    ) -> None:
        self._region = resolve_aws_region(region, required=False)
        self._role = role_arn or "arn:aws:iam::000000000000:role/glue"
        self._glue = glue_client

    def _client(self) -> Any:
        """Lazily build (or return the injected) boto3 ``glue`` client."""
        if self._glue is None:
            import boto3

            self._glue = boto3.client("glue", region_name=resolve_aws_region(self._region))
        return self._glue

    def run_transform(
        self,
        *,
        input_uri: str,
        transform_spec: dict[str, Any],
        output: str,
        resources: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Ensure a Glue job exists + start a run (T-ETL-02)."""
        res = dict(resources or {})
        glue = self._client()
        job_name = str(
            transform_spec.get("job_name")
            or _sanitize_name("etl-glue", str(transform_spec.get("op", "transform")))
        )
        script_location = transform_spec.get("script_ref") or f"{output.rstrip('/')}/glue_script.py"

        # Ensure the job exists (idempotent create; ignore AlreadyExists).
        try:
            glue.create_job(
                Name=job_name,
                Role=self._role,
                Command={"Name": "glueetl", "ScriptLocation": script_location, "PythonVersion": "3"},
                GlueVersion=res.get("glue_version", "4.0"),
            )
        except Exception:  # nosec B110
            pass

        run_args = {
            "--input": input_uri,
            "--output": output,
            **{str(k): str(v) for k, v in (transform_spec.get("arguments") or {}).items()},
        }
        run = glue.start_job_run(
            JobName=job_name,
            Arguments=run_args,
            WorkerType=res.get("worker_type", "G.1X"),
            NumberOfWorkers=int(res.get("num_workers", 2)),
        )
        return {
            "success": True,
            "orchestrated": "glue",
            "managed": True,
            "job_name": job_name,
            "run_id": run.get("JobRunId", ""),
            "status": "RUNNING",
            "output_uris": [output],
            "op": transform_spec.get("op", "transform"),
        }

    def run_crawl(self, *, s3_target: str) -> dict[str, Any]:
        """Ensure a Glue crawler exists + start it to catalog ``s3_target`` (T-ETL-03)."""
        glue = self._client()
        crawler_name = _sanitize_name("crawl", s3_target)
        db_name = "agenticml_catalog"
        try:
            glue.create_database(DatabaseInput={"Name": db_name})
        except Exception:  # nosec B110
            pass
        try:
            glue.create_crawler(
                Name=crawler_name,
                Role=self._role,
                DatabaseName=db_name,
                Targets={"S3Targets": [{"Path": s3_target}]},
            )
        except Exception:  # nosec B110
            pass
        try:
            glue.start_crawler(Name=crawler_name)
            started = True
        except Exception:
            started = False
        return {
            "success": True,
            "orchestrated": "glue",
            "managed": True,
            "crawler_run": crawler_name,
            "started": started,
            "database": db_name,
            "tables_updated": 0,  # populated asynchronously by the crawler
        }


# ── self-registration (the registry imports this module lazily) ───────────────
# Processing is the AWS default; in-process pandas is the offline/local default. Glue is
# selectable via ETL_BACKEND=glue. Offline auto → inprocess (doc 05 §4.3).
register_adapter(ETLBackend, "processing", lambda: ProcessingJobBackend(), is_default=True)
register_adapter(ETLBackend, "inprocess", lambda: InProcessPandasBackend(), is_local=True)
register_adapter(ETLBackend, "glue", lambda: GlueETLBackend())
