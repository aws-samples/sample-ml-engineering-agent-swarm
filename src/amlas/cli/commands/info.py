"""Info commands: runtime, version, health."""

from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Any

import click

from ...config.aws_region import resolve_aws_region
from ..display import console, print_json_output, print_table


@click.group()
def info() -> None:
    """System info — runtime, version, health checks."""


@info.command()
@click.option("--json", "as_json", is_flag=True, default=False, help="Output as JSON.")
def version(as_json: bool) -> None:
    """Show the platform version."""
    ver = "0.1.0"

    if as_json:
        print_json_output({"version": ver, "name": "amlas"})
        return

    console.print(f"[bold]amlas[/bold] v{ver}")


@info.command()
@click.option("--json", "as_json", is_flag=True, default=False, help="Output as JSON.")
def runtime(as_json: bool) -> None:
    """Show runtime info (Python, OS, paths, dependencies)."""
    data: dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cwd": str(Path.cwd()),
        "executable": sys.executable,
    }

    # Check key dependencies
    deps = {}
    for pkg in ["pandas", "numpy", "scikit-learn", "xgboost", "optuna", "click", "rich", "yaml"]:
        try:
            if pkg == "scikit-learn":
                import sklearn
                deps[pkg] = sklearn.__version__
            elif pkg == "yaml":
                import yaml as _yaml
                deps[pkg] = _yaml.__version__
            else:
                mod = __import__(pkg)
                deps[pkg] = getattr(mod, "__version__", "installed")
        except ImportError:
            deps[pkg] = "NOT INSTALLED"

    data["dependencies"] = deps

    # Check AWS
    try:
        import boto3
        data["aws_boto3_version"] = boto3.__version__
        # Try to get identity
        try:
            sts = boto3.client("sts", region_name=resolve_aws_region())
            identity = sts.get_caller_identity()
            data["aws_account"] = identity.get("Account", "unknown")
            data["aws_available"] = True
        except Exception:
            data["aws_available"] = False
    except ImportError:
        data["aws_boto3_version"] = "NOT INSTALLED"
        data["aws_available"] = False

    if as_json:
        print_json_output(data)
        return

    console.print("\n[bold]Runtime Information[/bold]\n")

    rows = [
        ["Python", data["python_version"]],
        ["Platform", data["platform"]],
        ["Machine", data["machine"]],
        ["CWD", data["cwd"]],
        ["Executable", data["executable"]],
    ]

    if "aws_boto3_version" in data:
        rows.append(["AWS boto3", data["aws_boto3_version"]])
        rows.append(["AWS Available", "✓" if data.get("aws_available") else "✗"])

    print_table(
        title="System",
        columns=[("Property", "cyan"), ("Value", "")],
        rows=rows,
    )

    # Dependencies
    dep_rows = []
    for pkg, ver in sorted(deps.items()):
        style = "" if ver != "NOT INSTALLED" else "[red]"
        end_style = "" if ver != "NOT INSTALLED" else "[/red]"
        dep_rows.append([pkg, f"{style}{ver}{end_style}"])

    print_table(
        title="Dependencies",
        columns=[("Package", ""), ("Version", "green")],
        rows=dep_rows,
    )


@info.command()
@click.option("--json", "as_json", is_flag=True, default=False, help="Output as JSON.")
def health(as_json: bool) -> None:
    """Run health checks (S3, Bedrock, local paths)."""
    checks: dict[str, dict[str, Any]] = {}

    # Check 1: Python version
    py_ver = platform.python_version()
    py_ok = sys.version_info >= (3, 11)
    checks["python"] = {
        "status": "ok" if py_ok else "warning",
        "detail": f"Python {py_ver}" + ("" if py_ok else " (3.11+ recommended)"),
    }

    # Check 2: Core dependencies
    core_deps = ["pandas", "numpy", "scikit-learn", "click", "rich"]
    missing_deps = []
    for pkg in core_deps:
        try:
            if pkg == "scikit-learn":
                __import__("sklearn")
            else:
                __import__(pkg)
        except ImportError:
            missing_deps.append(pkg)

    checks["dependencies"] = {
        "status": "ok" if not missing_deps else "error",
        "detail": "All core deps installed" if not missing_deps else f"Missing: {', '.join(missing_deps)}",
    }

    # Check 3: Output directory writable
    output_dir = Path("./output")
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        test_file = output_dir / ".health_check"
        test_file.write_text("ok")
        test_file.unlink()
        checks["output_dir"] = {"status": "ok", "detail": str(output_dir.resolve())}
    except Exception as e:
        checks["output_dir"] = {"status": "error", "detail": f"Cannot write to output: {e}"}

    # Check 4: AWS S3 connectivity
    try:
        import boto3

        s3 = boto3.client("s3", region_name=resolve_aws_region())
        s3.list_buckets()
        checks["aws_s3"] = {"status": "ok", "detail": "S3 accessible"}
    except ImportError:
        checks["aws_s3"] = {"status": "skip", "detail": "boto3 not installed (optional)"}
    except Exception as e:
        checks["aws_s3"] = {"status": "skip", "detail": f"S3 not available: {type(e).__name__}"}

    # Check 5: AWS Bedrock
    try:
        import boto3

        # Client construction is the probe — success means boto3 accepts the service.
        boto3.client("bedrock-runtime", region_name=resolve_aws_region())
        checks["aws_bedrock"] = {"status": "ok", "detail": "Bedrock accessible"}
    except ImportError:
        checks["aws_bedrock"] = {"status": "skip", "detail": "boto3 not installed"}
    except Exception as e:
        checks["aws_bedrock"] = {"status": "skip", "detail": f"Bedrock not available: {type(e).__name__}"}

    if as_json:
        print_json_output(checks)
        return

    # Rich display
    console.print("\n[bold]Health Check[/bold]\n")

    status_icons = {"ok": "[green]✓[/green]", "warning": "[yellow]⚠[/yellow]", "error": "[red]✗[/red]", "skip": "[dim]○[/dim]"}

    for name, check in checks.items():
        icon = status_icons.get(check["status"], "?")
        console.print(f"  {icon} [bold]{name}[/bold]: {check['detail']}")

    # Summary
    all_ok = all(c["status"] in ("ok", "skip") for c in checks.values())
    if all_ok:
        console.print("\n[green]All health checks passed.[/green]")
    else:
        errors = [n for n, c in checks.items() if c["status"] == "error"]
        console.print(f"\n[red]Issues found: {', '.join(errors)}[/red]")
