"""Data management commands: describe, quality, upload, download, list.

Phase 11 extends this group (master plan §9) with the agentic Data-Engineering
surfaces: ``quality --patterns`` (the 25-pattern DQ scan), ``diagnose-missing``
(MCAR/MAR/MNAR), ``expectations`` (pandera gate), ``impute`` (multi-strategy with
a ``--steer`` domain dial), ``synth`` (SDV / SMOTE / requirement→spec / validate),
and ``lake`` (medallion read-slice / write-back / catalog). Each new command calls
the corresponding Phase 6–10 ``@tool`` via its ``._tool_func`` (the plain dict
form) and supports ``--json`` for machine output.

The Phase 6–10 tools are imported at module scope so they degrade lazily (heavy
deps like sdv/boto3 are imported inside the tool functions) and so the CLI tests
can override the tool layer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click

from ...config.aws_region import resolve_aws_region

# Phase 6–10 tools (module-scope so callers can override + so dispatch is
# explicit). The @tool wrappers themselves are cheap to import — sdv / boto3 /
# pandera / imblearn are all imported lazily *inside* the tool functions.
from ...tools.data_quality import (
    check_expectations,
    diagnose_missingness,
    scan_dq_patterns,
)
from ...tools.imputation import (
    apply_imputation_plan,
    evaluate_imputation,
    recommend_imputation_plan,
)
from ...tools.lake import list_lake_catalog, read_lake_slice, write_lake_dataset
from ...tools.synthetic import (
    augment_imbalance,
    build_synthesis_spec,
    synthesize_tabular,
    validate_synthetic,
)
from ..display import (
    console,
    print_error,
    print_json_output,
    print_panel,
    print_success,
    print_table,
    print_warning,
)


def _load_steering_profile(steer_path: str | None) -> dict[str, Any] | None:
    """Load a ``SteeringProfile`` from a YAML path and return its profile dict.

    Returns ``None`` when no path is given so the no-dial path stays clean. The
    profile dict is what the tools consume as ``profile=`` (HARD constraints
    enforced deterministically in ``apply_imputation_plan`` / ``write_lake_dataset``).
    """
    if not steer_path:
        return None
    from ...data_eng.steering import SteeringProfile

    return SteeringProfile.from_yaml(steer_path).as_profile_dict()


@click.group()
def data() -> None:
    """Data management — describe, quality, impute, synth, lake, and more."""


@data.command()
@click.argument("path")
@click.option("--target", default=None, help="Target column name.")
@click.option("--json", "as_json", is_flag=True, default=False, help="Output as JSON.")
def describe(path: str, target: str | None, as_json: bool) -> None:
    """Describe a dataset (EDA summary).

    PATH is the filesystem path to a CSV file.
    """
    from pathlib import Path as P

    file_path = P(path)
    if not file_path.exists():
        print_error(f"File not found: {path}")
        raise SystemExit(1)

    import pandas as pd

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print_error(f"Failed to read CSV: {e}")
        raise SystemExit(1) from e

    n_rows, n_cols = df.shape
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    missing_total = int(df.isnull().sum().sum())
    missing_pct = missing_total / (n_rows * n_cols) * 100 if (n_rows * n_cols) > 0 else 0

    result = {
        "shape": {"rows": n_rows, "cols": n_cols},
        "numeric_features": len(numeric_cols),
        "categorical_features": len(categorical_cols),
        "missing_values": missing_total,
        "missing_pct": round(missing_pct, 2),
        "columns": list(df.columns),
    }

    if target and target in df.columns:
        value_counts = df[target].value_counts()
        result["target_distribution"] = {str(k): int(v) for k, v in value_counts.items()}

    if as_json:
        print_json_output(result)
        return

    # Rich display
    print_panel(
        f"Dataset: {file_path.name}",
        f"Rows: {n_rows:,} | Columns: {n_cols}\n"
        f"Numeric: {len(numeric_cols)} | Categorical: {len(categorical_cols)}\n"
        f"Missing: {missing_total:,} ({missing_pct:.1f}%)",
        style="blue",
    )

    # Column list table
    rows_display = []
    for col in df.columns[:20]:  # First 20 columns
        dtype = str(df[col].dtype)
        n_missing = int(df[col].isnull().sum())
        n_unique = int(df[col].nunique())
        rows_display.append([col, dtype, str(n_unique), str(n_missing)])

    print_table(
        title="Columns (first 20)",
        columns=[("Name", "cyan"), ("Type", ""), ("Unique", ""), ("Missing", "yellow")],
        rows=rows_display,
    )

    if target and target in df.columns:
        console.print(f"\n[bold]Target Distribution ({target}):[/bold]")
        for val, count in df[target].value_counts().head(10).items():
            pct = count / n_rows * 100
            # Bound the label rendering: truncate long user-entered category values
            # so free-text labels cannot dump arbitrary content to the console.
            console.print(f"  {str(val)[:60]}: {count} ({pct:.1f}%)")


@data.command()
@click.argument("path")
@click.option("--target", default="target", help="Target column name.")
@click.option(
    "--patterns",
    is_flag=True,
    default=False,
    help="Run the 25-pattern DQ catalog scan (master plan §5) instead of the score-only path.",
)
@click.option("--json", "as_json", is_flag=True, default=False, help="Output as JSON.")
def quality(path: str, target: str, patterns: bool, as_json: bool) -> None:
    """Run data quality assessment.

    PATH is the filesystem path to a CSV file. With ``--patterns`` this runs the
    25-pattern DQ catalog scan (``scan_dq_patterns``) printing findings by
    pattern + severity + suggested strategy; otherwise the legacy 6-dimension
    score path (``score_data_quality``) is used.
    """
    from pathlib import Path as P

    file_path = P(path)
    if not file_path.exists():
        print_error(f"File not found: {path}")
        raise SystemExit(1)

    if patterns:
        result = scan_dq_patterns._tool_func(data_path=path, target=target)
        if as_json:
            print_json_output(result)
            return
        if result.get("success") is False:
            print_error(result.get("error", "pattern scan failed"))
            raise SystemExit(1)
        overall = result.get("overall_score", 0)
        console.print(f"\n[bold]DQ Pattern Scan[/bold] — overall score: {overall:.1f}/100\n")
        detected = result.get("patterns_detected", [])
        if not detected:
            print_success("No DQ patterns detected (clean).")
            return
        rows = []
        for f in detected[:40]:
            sev = f.get("severity", "")
            color = {"high": "red", "medium": "yellow", "low": "blue"}.get(sev, "white")
            rows.append(
                [
                    f.get("pattern_id", ""),
                    f.get("column", ""),
                    f"[{color}]{sev}[/{color}]",
                    str(f.get("suggested_strategy", ""))[:40],
                ]
            )
        print_table(
            title=f"Patterns Detected ({len(detected)})",
            columns=[("Pattern", "cyan"), ("Column", ""), ("Severity", ""), ("Suggested Strategy", "dim")],
            rows=rows,
        )
        return

    from ...tools.data_quality_tools import score_data_quality

    result = score_data_quality(data_path=path, target_column=target)

    if as_json:
        print_json_output(result)
        return

    overall = result.get("overall_score", 0)
    score_color = "green" if overall >= 80 else ("yellow" if overall >= 60 else "red")

    console.print(f"\n[bold]Data Quality Score:[/bold] [{score_color}]{overall:.1f}/100[/{score_color}]")
    console.print()

    # Dimensions table
    dimensions = result.get("dimensions", {})
    dim_rows = []
    for name, dim_data in dimensions.items():
        score = dim_data.get("score", 0)
        details = dim_data.get("details", "")
        color = "green" if score >= 80 else ("yellow" if score >= 60 else "red")
        dim_rows.append([name.capitalize(), f"[{color}]{score:.1f}[/{color}]", details[:60]])

    print_table(
        title="Quality Dimensions",
        columns=[("Dimension", "bold"), ("Score", ""), ("Details", "dim")],
        rows=dim_rows,
    )

    # Issues
    issues = result.get("issues", [])
    if issues:
        console.print(f"\n[bold yellow]Issues Found: {len(issues)}[/bold yellow]")
        for issue in issues[:10]:
            severity_color = {"high": "red", "medium": "yellow", "low": "blue"}.get(
                issue.get("severity", ""), "white"
            )
            console.print(
                f"  [{severity_color}]●[/{severity_color}] "
                f"[bold]{issue.get('column', '')}[/bold]: {issue.get('issue', '')}"
            )
            if issue.get("recommendation"):
                console.print(f"    → {issue['recommendation']}")


@data.command("list")
@click.option("--dir", "data_dir", default="./data", help="Directory to search.")
@click.option("--s3", is_flag=True, default=False, help="List S3 datasets (requires AWS creds).")
def list_data(data_dir: str, s3: bool) -> None:
    """List available datasets."""
    if s3:
        print_warning("S3 listing requires AWS credentials. Showing local files only.")

    dir_path = Path(data_dir)
    if not dir_path.exists():
        console.print(f"[dim]No data directory at {data_dir}[/dim]")
        return

    csv_files = list(dir_path.rglob("*.csv"))
    parquet_files = list(dir_path.rglob("*.parquet"))
    all_files = csv_files + parquet_files

    if not all_files:
        console.print("[dim]No data files found.[/dim]")
        return

    rows = []
    for f in sorted(all_files):
        size_mb = f.stat().st_size / (1024 * 1024)
        rows.append([str(f.relative_to(dir_path)), f.suffix, f"{size_mb:.2f} MB"])

    print_table(
        title=f"Data Files in {data_dir}",
        columns=[("File", "cyan"), ("Type", ""), ("Size", "green")],
        rows=rows,
    )


@data.command()
@click.argument("local_path")
@click.argument("s3_path")
def upload(local_path: str, s3_path: str) -> None:
    """Upload local file to S3.

    LOCAL_PATH is the local file path.
    S3_PATH is the S3 destination (e.g., s3://bucket/key).
    """
    local = Path(local_path)
    if not local.exists():
        print_error(f"Local file not found: {local_path}")
        raise SystemExit(1)

    try:
        import boto3

        # Parse S3 path
        if s3_path.startswith("s3://"):
            parts = s3_path[5:].split("/", 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else local.name
        else:
            print_error("S3 path must start with s3://")
            raise SystemExit(1)

        s3_client = boto3.client("s3", region_name=resolve_aws_region())
        file_size = local.stat().st_size

        from ..display import create_progress

        with create_progress() as progress:
            task = progress.add_task(f"Uploading {local.name}", total=file_size)

            def callback(bytes_transferred: int) -> None:
                progress.update(task, advance=bytes_transferred)

            s3_client.upload_file(str(local), bucket, key, Callback=callback)

        print_success(f"Uploaded {local.name} → s3://{bucket}/{key}")

    except ImportError:
        print_error("boto3 is unavailable. Rerun scripts/setup.sh.")
        raise SystemExit(1) from None
    except Exception as e:
        print_error(f"Upload failed: {e}")
        raise SystemExit(1) from e


@data.command()
@click.argument("s3_path")
@click.argument("local_path")
def download(s3_path: str, local_path: str) -> None:
    """Download from S3 to local.

    S3_PATH is the S3 source (e.g., s3://bucket/key).
    LOCAL_PATH is the local destination path.
    """
    try:
        import boto3

        if s3_path.startswith("s3://"):
            parts = s3_path[5:].split("/", 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ""
        else:
            print_error("S3 path must start with s3://")
            raise SystemExit(1)

        s3_client = boto3.client("s3", region_name=resolve_aws_region())

        # Get file size for progress
        try:
            head = s3_client.head_object(Bucket=bucket, Key=key)
            file_size = head["ContentLength"]
        except Exception:
            file_size = 0

        local = Path(local_path)
        local.parent.mkdir(parents=True, exist_ok=True)

        from ..display import create_progress

        with create_progress() as progress:
            task = progress.add_task(f"Downloading {key.split('/')[-1]}", total=file_size or None)

            def callback(bytes_transferred: int) -> None:
                progress.update(task, advance=bytes_transferred)

            s3_client.download_file(bucket, key, str(local), Callback=callback)

        print_success(f"Downloaded s3://{bucket}/{key} → {local_path}")

    except ImportError:
        print_error("boto3 is unavailable. Rerun scripts/setup.sh.")
        raise SystemExit(1) from None
    except Exception as e:
        print_error(f"Download failed: {e}")
        raise SystemExit(1) from e


# ─── Phase 11: DQ / imputation / synthetic / lake subcommands (master plan §9) ──


@data.command("diagnose-missing")
@click.argument("path")
@click.option("--columns", multiple=True, help="Subset of columns to diagnose (repeatable).")
@click.option("--json", "as_json", is_flag=True, default=False, help="Output as JSON.")
def diagnose_missing(path: str, columns: tuple[str, ...], as_json: bool) -> None:
    """Diagnose the missingness mechanism (MCAR / MAR / MNAR) per column.

    PATH is a CSV file. Calls ``diagnose_missingness`` (Little's MCAR test +
    per-column mask classifier) and prints the per-column mechanism + evidence.
    """
    if not Path(path).exists():
        print_error(f"File not found: {path}")
        raise SystemExit(1)

    cols = list(columns) or None
    result = diagnose_missingness._tool_func(data_path=path, columns=cols)

    if as_json:
        print_json_output(result)
        return
    if result.get("success") is False:
        print_error(result.get("error", "diagnosis failed"))
        raise SystemExit(1)

    console.print(
        f"\n[bold]Missingness Diagnosis[/bold] — overall: "
        f"{result.get('overall_mechanism', 'unknown')}\n"
    )
    rows = []
    for col, info in result.get("per_column", {}).items():
        rows.append(
            [
                col,
                str(info.get("mechanism", "")),
                f"{info.get('mask_auc', 0):.3f}" if info.get("mask_auc") is not None else "—",
                str(info.get("evidence", ""))[:48],
            ]
        )
    print_table(
        title="Per-Column Mechanism",
        columns=[("Column", "cyan"), ("Mechanism", ""), ("Mask AUC", ""), ("Evidence", "dim")],
        rows=rows,
    )


@data.command()
@click.argument("path")
@click.option("--suite", default=None, help="Path to a pandera YAML suite (auto-infer if omitted).")
@click.option("--json", "as_json", is_flag=True, default=False, help="Output as JSON.")
def expectations(path: str, suite: str | None, as_json: bool) -> None:
    """Run a declarative pandera expectations gate (the silver gate).

    PATH is a CSV file. Calls ``check_expectations`` (auto-infers a schema when
    no ``--suite`` is given) and reports pass/fail + violations.
    """
    if not Path(path).exists():
        print_error(f"File not found: {path}")
        raise SystemExit(1)

    result = check_expectations._tool_func(data_path=path, suite=suite)

    if as_json:
        print_json_output(result)
        return
    if result.get("success") is False:
        print_error(result.get("error", "expectations check failed"))
        raise SystemExit(1)

    if result.get("passed"):
        print_success(f"Expectations passed (suite: {result.get('suite_used', '?')}).")
        return
    violations = result.get("violations", [])
    print_warning(f"Expectations FAILED — {len(violations)} violation(s):")
    for v in violations[:20]:
        console.print(f"  • {v.get('column', '?')}: {v.get('check', '?')} (n_failed={v.get('n_failed', '?')})")


@data.command()
@click.argument("path")
@click.option("--columns", multiple=True, help="Columns to impute (repeatable; default: auto-detect).")
@click.option(
    "--method",
    default="median",
    help="Imputation method (median/knn/iterative/missforest/...) or 'auto' to recommend per column.",
)
@click.option("--time-col", "time_col", default=None, help="Datetime column for time-series methods.")
@click.option("--allow-llm", "allow_llm", is_flag=True, default=False, help="Allow LLM extrapolation (B2).")
@click.option("--evaluate", "do_evaluate", is_flag=True, default=False, help="Run mask-and-score self-evaluation.")
@click.option("--steer", "steer_path", default=None, help="Steering profile YAML (HARD constraints — the dial).")
@click.option("--write", "write_uri", default=None, help="Write the imputed dataset back to a lake URI.")
@click.option("--target", default=None, help="Target column (excluded from auto-imputation candidates).")
@click.option("--json", "as_json", is_flag=True, default=False, help="Output as JSON.")
def impute(
    path: str,
    columns: tuple[str, ...],
    method: str,
    time_col: str | None,
    allow_llm: bool,
    do_evaluate: bool,
    steer_path: str | None,
    write_uri: str | None,
    target: str | None,
    as_json: bool,
) -> None:
    """Impute missing values with a steerable, multi-strategy engine (master plan §6).

    PATH is a CSV file. With ``--method auto`` the engine diagnoses the mechanism
    and recommends a per-column method (``recommend_imputation_plan``); otherwise
    the chosen ``--method`` is applied to ``--columns``. In both cases the plan is
    executed through ``apply_imputation_plan``, which enforces the HARD steering
    constraints from ``--steer`` deterministically (a forbidden method is rejected;
    a pinned method overrides the proposal). ``--evaluate`` runs the mask-and-score
    harness; ``--write`` writes the imputed dataset back to a silver/gold lake URI.
    """
    if not Path(path).exists():
        print_error(f"File not found: {path}")
        raise SystemExit(1)

    profile = _load_steering_profile(steer_path)

    # Build the per-column plan.
    if method.strip().lower() == "auto":
        rec = recommend_imputation_plan._tool_func(data_path=path, target=target, profile=profile)
        if rec.get("success") is False:
            print_error(rec.get("error", "recommendation failed"))
            raise SystemExit(1)
        plan = {col: {k: v for k, v in spec.items() if k in ("method", "params")} for col, spec in rec.get("plan", {}).items()}
        if not plan:
            print_warning("No columns required imputation (no missing values detected).")
            if as_json:
                print_json_output({"plan_applied": {}, "imputed_path": path, "diagnosis": rec.get("diagnosis", {})})
            return
    else:
        cols = list(columns)
        if not cols:
            import pandas as pd

            df = pd.read_csv(path)
            cols = [c for c in df.columns if df[c].isnull().any() and c != target]
        plan = {}
        for c in cols:
            spec: dict[str, Any] = {"method": method}
            if time_col:
                spec["time_col"] = time_col
            plan[c] = spec

    result = apply_imputation_plan._tool_func(data_path=path, plan=plan, profile=profile)
    if result.get("success") is False:
        if as_json:
            print_json_output(result)
            return
        print_error(result.get("error", "imputation rejected"))
        for v in result.get("steering_violations", []):
            console.print(f"  • {v.get('column', '?')}: {v.get('reason', '')}")
        raise SystemExit(1)

    # Optional self-evaluation (the "prove it works" harness).
    if do_evaluate:
        eval_cols = list(plan.keys())
        evaluation = evaluate_imputation._tool_func(original_path=path, columns=eval_cols)
        result["evaluation"] = evaluation

    # Optional write-back to the medallion lake.
    if write_uri:
        wb = write_lake_dataset._tool_func(
            local_path=result["imputed_path"], dest_uri=write_uri, profile=profile
        )
        result["written_back"] = wb

    if as_json:
        print_json_output(result)
        return

    print_success(f"Imputed dataset → {result['imputed_path']}")
    rows = [
        [col, info.get("method", ""), info.get("kind", ""), str(info.get("n_filled", 0))]
        for col, info in result.get("plan_applied", {}).items()
    ]
    print_table(
        title="Imputation Plan Applied",
        columns=[("Column", "cyan"), ("Method", ""), ("Kind", ""), ("Filled", "green")],
        rows=rows,
    )
    for v in result.get("steering_violations", []):
        if v.get("enforced"):
            print_warning(f"Steering enforced on {v.get('column', '?')}: {v.get('reason', '')}")


@data.command()
@click.argument("path")
@click.option("--rows", "n_rows", default=1000, type=int, help="Number of synthetic rows to generate.")
@click.option(
    "--generator",
    default="gaussian_copula",
    help="Generator: gaussian_copula | ctgan | tvae (ignored for --augment / --from).",
)
@click.option("--from", "requirement", default=None, help="Requirement → SynthesisSpec (NL, capability C3).")
@click.option("--augment", is_flag=True, default=False, help="Imbalance augmentation (SMOTE family, C2).")
@click.option("--balance", is_flag=True, default=False, help="Balance classes when augmenting.")
@click.option("--target", default=None, help="Target column (required for --augment / --validate utility).")
@click.option("--validate", "do_validate", is_flag=True, default=False, help="Run 3-pillar synthetic validation.")
@click.option("--steer", "steer_path", default=None, help="Steering profile YAML (validation thresholds — the dial).")
@click.option("--write", "write_uri", default=None, help="Write the synthetic dataset back to a lake URI.")
@click.option("--json", "as_json", is_flag=True, default=False, help="Output as JSON.")
def synth(
    path: str,
    n_rows: int,
    generator: str,
    requirement: str | None,
    augment: bool,
    balance: bool,
    target: str | None,
    do_validate: bool,
    steer_path: str | None,
    write_uri: str | None,
    as_json: bool,
) -> None:
    """Generate / augment synthetic data with 3-pillar validation (master plan §7).

    PATH is a reference CSV. ``--from "requirement"`` builds a SynthesisSpec
    (``build_synthesis_spec``); ``--augment [--balance]`` oversamples the minority
    class (``augment_imbalance``); otherwise ``synthesize_tabular`` generates
    ``--rows`` realistic rows via the chosen ``--generator``. ``--validate`` runs the
    fidelity/utility/privacy AND-gate (``validate_synthetic``); a ``--steer`` profile
    supplies the acceptance thresholds. ``--write`` writes the result back to a lake URI.
    """
    if not Path(path).exists():
        print_error(f"File not found: {path}")
        raise SystemExit(1)

    profile = _load_steering_profile(steer_path)

    # 1. Requirement → SynthesisSpec (no generation).
    if requirement:
        result = build_synthesis_spec._tool_func(requirement=requirement, seed_path=path, volume=n_rows)
        if as_json:
            print_json_output(result)
            return
        if result.get("success") is False:
            print_error(result.get("error", "spec build failed"))
            raise SystemExit(1)
        spec = result.get("synthesis_spec", {})
        print_success("SynthesisSpec built from requirement.")
        console.print(
            f"  generator={spec.get('generator', '?')} volume={spec.get('volume', '?')} "
            f"conditions={spec.get('conditions', {})}"
        )
        return

    # 2. Imbalance augmentation.
    if augment:
        if not target:
            print_error("--augment requires --target (the class label column).")
            raise SystemExit(1)
        result = augment_imbalance._tool_func(data_path=path, target=target, strategy="smote")
        synth_path = result.get("augmented_path")
    else:
        # 3. Classical SDV generation.
        result = synthesize_tabular._tool_func(data_path=path, rows=n_rows, generator=generator)
        synth_path = result.get("synthetic_path")

    if result.get("success") is False:
        if as_json:
            print_json_output(result)
            return
        print_error(result.get("error", "synthesis failed"))
        raise SystemExit(1)

    # 4. Optional 3-pillar validation (steerable thresholds).
    if do_validate and synth_path:
        thresholds = None
        if profile and profile.get("thresholds"):
            thresholds = profile["thresholds"]
        validation = validate_synthetic._tool_func(
            real_path=path, synthetic_path=synth_path, target=target, thresholds=thresholds
        )
        result["validation"] = validation

    # 5. Optional write-back.
    if write_uri and synth_path:
        result["written_back"] = write_lake_dataset._tool_func(
            local_path=synth_path, dest_uri=write_uri, profile=profile
        )

    if as_json:
        print_json_output(result)
        return

    if augment:
        print_success(f"Augmented dataset → {synth_path}")
        console.print(f"  before={result.get('class_counts_before')} after={result.get('class_counts_after')}")
    else:
        print_success(f"Synthetic dataset → {synth_path} ({result.get('rows', '?')} rows, {result.get('generator', '?')})")
    if "validation" in result:
        acc = result["validation"].get("accepted")
        console.print(f"  validation accepted: {acc}")


@data.command()
@click.argument("action", type=click.Choice(["read", "write", "catalog"]))
@click.argument("args", nargs=-1)
@click.option("--tier", default="silver", type=click.Choice(["silver", "gold"]), help="Write-back tier.")
@click.option("--columns", multiple=True, help="Column projection for read (repeatable).")
@click.option("--limit", default=None, type=int, help="Row cap for read.")
@click.option("--database", default=None, help="Glue database for catalog listing.")
@click.option("--steer", "steer_path", default=None, help="Steering profile YAML (HARD write-tier dial).")
@click.option("--json", "as_json", is_flag=True, default=False, help="Output as JSON.")
def lake(
    action: str,
    args: tuple[str, ...],
    tier: str,
    columns: tuple[str, ...],
    limit: int | None,
    database: str | None,
    steer_path: str | None,
    as_json: bool,
) -> None:
    """Medallion lakehouse read-slice / write-back / catalog (master plan §8 D6).

    \b
    Usage:
      data lake read  <uri> [--columns C --limit N]
      data lake write <local_path> <dest_uri> [--tier silver|gold] [--steer FILE]
      data lake catalog [--database DB]

    Reads a tier-scoped subset (``read_lake_slice``), writes a cleaned/imputed
    artifact back to silver/gold with lineage (``write_lake_dataset`` — rejects
    bronze/raw), or lists Glue-cataloged datasets (``list_lake_catalog``). The
    ``--steer`` profile HARD-pins/forbids the write tier deterministically.
    """
    profile = _load_steering_profile(steer_path)

    if action == "read":
        if len(args) < 1:
            print_error("lake read requires a URI (e.g. silver://db.table or s3://...).")
            raise SystemExit(1)
        result = read_lake_slice._tool_func(
            args[0], columns=list(columns) or None, limit=limit
        )
    elif action == "write":
        if len(args) < 2:
            print_error("lake write requires LOCAL_PATH and DEST_URI.")
            raise SystemExit(1)
        result = write_lake_dataset._tool_func(
            local_path=args[0], dest_uri=args[1], tier=tier, profile=profile
        )
    else:  # catalog
        result = list_lake_catalog._tool_func(database=database)

    if as_json:
        print_json_output(result)
        return
    if result.get("success") is False:
        print_error(result.get("error", f"lake {action} failed"))
        raise SystemExit(1)

    if action == "read":
        print_success(f"Read slice → {result.get('local_path')} ({result.get('rows')} rows, tier={result.get('tier')})")
    elif action == "write":
        print_success(f"Wrote → {result.get('written_uri')} (tier={result.get('tier')}, lineage={result.get('lineage_id')})")
    else:
        console.print(f"\n[bold]Lake Catalog[/bold] — {result.get('count', 0)} dataset(s)\n")
        rows = [
            [d.get("name", ""), d.get("database", ""), d.get("tier", "")]
            for d in result.get("datasets", [])
        ]
        print_table(
            title="Datasets",
            columns=[("Name", "cyan"), ("Database", ""), ("Tier", "green")],
            rows=rows,
        )
