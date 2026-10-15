"""System health status widget — shows real AWS connectivity and resource status."""

from __future__ import annotations

import shutil
from typing import Any

from textual import work
from textual.app import ComposeResult
from textual.widgets import Static


class SystemHealthWidget(Static):
    """Displays system health indicators for AWS and local resources."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._health_lines: list[str] = [
            "  ⏳ AWS: Checking...",
            "  ⏳ Bedrock: Checking...",
            "  ⏳ S3: Checking...",
            "  🟢 Local Environment: Ready",
        ]

    def compose(self) -> ComposeResult:
        yield Static(self._get_health_display(), id="health-content")

    def on_mount(self) -> None:
        """Start async health check when widget mounts."""
        self._run_health_check()

    @work(thread=True, exclusive=True)
    def _run_health_check(self) -> None:
        """Run AWS health check in background thread (avoids blocking UI)."""
        from amlas.config.health import check_aws_health
        from amlas.config.settings import Settings

        settings = Settings.load()
        status = check_aws_health(settings)

        lines: list[str] = []

        # AWS connectivity
        if status.aws_connected:
            profile_info = f" ({status.profile_used})" if status.profile_used != "default chain" else ""
            lines.append(
                f"  🟢 AWS: Connected{profile_info} — {status.account_id}"
            )
        else:
            lines.append(f"  🔴 AWS: {status.error}")

        # Bedrock
        if status.aws_connected and status.bedrock_available:
            lines.append(f"  🟢 Bedrock: Available ({status.bedrock_region})")
        elif status.aws_connected:
            err = status.errors.get("bedrock", "unavailable")
            lines.append(f"  🔴 Bedrock: {err}")
        else:
            lines.append("  🟡 Bedrock: No credentials")

        # S3
        if status.aws_connected and status.s3_available:
            if settings.aws.s3_bucket:
                if status.s3_bucket_accessible:
                    lines.append(f"  🟢 S3: {settings.aws.s3_bucket}")
                else:
                    lines.append(f"  🟡 S3: Bucket not accessible ({settings.aws.s3_bucket})")
            else:
                lines.append("  🟡 S3: No bucket configured")
        elif status.aws_connected:
            lines.append("  🔴 S3: Access denied")
        else:
            lines.append("  🟡 S3: No credentials")

        # Local resources
        disk = shutil.disk_usage(".")
        disk_free_gb = disk.free / (1024**3)
        lines.append(f"  🟢 Disk: {disk_free_gb:.0f} GB free")

        self._health_lines = lines
        try:
            self.app.call_from_thread(self._update_display)
        except (AttributeError, RuntimeError):
            # Widget not yet mounted or app not running (e.g. in tests)
            pass

    def _update_display(self) -> None:
        """Update the display with current health lines (must run on main thread)."""
        try:
            content = self.query_one("#health-content", Static)
            content.update(self._get_health_display())
        except Exception:  # nosec B110
            pass

    def _get_health_display(self) -> str:
        """Build system health status text from current lines."""
        return "\n".join(self._health_lines)

    def refresh_health(self) -> None:
        """Trigger a fresh health check."""
        self._health_lines = [
            "  ⏳ AWS: Checking...",
            "  ⏳ Bedrock: Checking...",
            "  ⏳ S3: Checking...",
            "  🟢 Local Environment: Ready",
        ]
        self._update_display()
        self._run_health_check()
