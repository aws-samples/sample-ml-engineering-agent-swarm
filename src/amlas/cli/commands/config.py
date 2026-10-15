"""Config commands: show, set, init, health."""

from __future__ import annotations

import click

from ..display import console, print_json_output, print_success, print_table


@click.group()
def config() -> None:
    """Configuration — show, set, initialize workspace, check health."""


@config.command()
@click.option("--config-file", default=None, help="Path to config file.")
@click.option("--json", "as_json", is_flag=True, default=False, help="Output as JSON.")
def show(config_file: str | None, as_json: bool) -> None:
    """Show current configuration."""
    from ..config import CLIConfig

    cfg = CLIConfig(config_path=config_file)

    if as_json:
        print_json_output(cfg.all)
        return

    console.print(f"\n[bold]Configuration[/bold] ({cfg.path})\n")

    rows = []
    for key, value in sorted(cfg.all.items()):
        rows.append([key, str(value)])

    print_table(
        title="Settings",
        columns=[("Key", "cyan"), ("Value", "green")],
        rows=rows,
    )


@config.command("set")
@click.argument("key")
@click.argument("value")
@click.option("--config-file", default=None, help="Path to config file.")
def set_value(key: str, value: str, config_file: str | None) -> None:
    """Set a configuration value.

    KEY is the config key (e.g., output_dir).
    VALUE is the new value.
    """
    from ..config import CLIConfig

    cfg = CLIConfig(config_path=config_file)
    cfg.set(key, value)
    print_success(f"Set {key} = {value}")


@config.command()
@click.option("--dir", "workspace_dir", default=".", help="Workspace directory to initialize.")
def init(workspace_dir: str) -> None:
    """Initialize a new workspace with config files.

    Creates directory structure, .agenticml.yaml, and ~/.agenticml/config.yaml.
    """
    from ..config import init_workspace

    cfg = init_workspace(workspace_dir)
    print_success(f"Workspace initialized at {workspace_dir}")
    console.print(f"  [dim]Config: {cfg.path}[/dim]")
    console.print("  [dim]Created: specs/, data/, output/, .agenticml.yaml[/dim]")
    console.print("  [dim]User config: ~/.agenticml/config.yaml[/dim]")


@config.command()
@click.option("--json", "as_json", is_flag=True, default=False, help="Output as JSON.")
def health(as_json: bool) -> None:
    """Check AWS connectivity and service access.

    Uses boto3 standard credential chain to verify:
    - AWS credentials (STS get_caller_identity)
    - Bedrock runtime access
    - S3 bucket access (if configured)
    """
    from amlas.config.health import check_aws_health
    from amlas.config.settings import Settings

    settings = Settings.load()
    status = check_aws_health(settings)

    if as_json:
        import json

        result = {
            "aws_connected": status.aws_connected,
            "account_id": status.account_id,
            "profile": status.profile_used,
            "region": status.region,
            "bedrock_available": status.bedrock_available,
            "bedrock_region": status.bedrock_region,
            "s3_available": status.s3_available,
            "s3_bucket": settings.aws.s3_bucket,
            "s3_bucket_accessible": status.s3_bucket_accessible,
            "error": status.error,
            "errors": status.errors,
        }
        console.print_json(json.dumps(result))
        return

    console.print("\n[bold]AWS Health Check[/bold]\n")

    # AWS Credentials
    if status.aws_connected:
        console.print("  [green]✓[/green] AWS Connected")
        console.print(f"    Account: [cyan]{status.account_id}[/cyan]")
        console.print(f"    Profile: [cyan]{status.profile_used}[/cyan]")
        console.print(f"    Region:  [cyan]{status.region}[/cyan]")
        console.print(f"    ARN:     [dim]{status.arn}[/dim]")
    else:
        console.print(f"  [red]✗[/red] AWS Not Connected: {status.error}")
        console.print("\n  [dim]Tip: Set AWS_PROFILE env var or configure ~/.aws/credentials[/dim]")
        return

    console.print()

    # Bedrock
    if status.bedrock_available:
        console.print(f"  [green]✓[/green] Bedrock: Available ({status.bedrock_region})")
    else:
        err = status.errors.get("bedrock", "unknown error")
        console.print(f"  [red]✗[/red] Bedrock: {err}")

    # S3
    if status.s3_available:
        if settings.aws.s3_bucket:
            if status.s3_bucket_accessible:
                console.print(f"  [green]✓[/green] S3 Bucket: {settings.aws.s3_bucket}")
            else:
                console.print(
                    f"  [yellow]![/yellow] S3 Bucket: {settings.aws.s3_bucket} (not accessible)"
                )
        else:
            console.print("  [yellow]![/yellow] S3: No bucket configured")
    else:
        console.print("  [red]✗[/red] S3: Access error")

    console.print()
