"""CLI configuration management.

Handles reading/writing config from ~/.agenticml/config.yaml or .agenticml.yaml.
Integrates with the new Settings system while maintaining backward compatibility.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ..config.settings import (
    CONFIG_TEMPLATE,
    PROJECT_CONFIG_FILE,
    USER_CONFIG_DIR,
    USER_CONFIG_FILE,
)

# --- Legacy compat: old flat config format ---
DEFAULT_CONFIG_DIR = USER_CONFIG_DIR
DEFAULT_CONFIG_FILE = USER_CONFIG_FILE

# Default configuration values (legacy flat format)
DEFAULT_CONFIG: dict[str, Any] = {
    "workspace_dir": ".",
    "s3_bucket": "",
    "aws_profile": "",
    "output_dir": "./output",
    "specs_dir": "./specs",
    "data_dir": "./data",
}


class CLIConfig:
    """CLI configuration manager.

    Reads from YAML config file with sensible defaults.
    Now wraps the Settings system for YAML-based config.
    """

    def __init__(self, config_path: str | Path | None = None) -> None:
        self._path = Path(config_path) if config_path else DEFAULT_CONFIG_FILE
        self._config: dict[str, Any] = dict(DEFAULT_CONFIG)
        self._load()

    def _load(self) -> None:
        """Load config from file if it exists."""
        if self._path.exists():
            try:
                data = yaml.safe_load(self._path.read_text())
                if isinstance(data, dict):
                    # Support both flat and nested formats
                    if "aws" in data and isinstance(data["aws"], dict):
                        # New nested format — flatten for backward compat
                        aws = data["aws"]
                        self._config["aws_profile"] = aws.get("profile", "")
                        self._config["s3_bucket"] = aws.get("s3_bucket", "")
                        self._config["aws_region"] = aws.get("region", "")
                        self._config["bedrock_region"] = aws.get("bedrock_region", "")
                        self._config["agentcore_endpoint"] = aws.get("agentcore_endpoint", "")
                    if "local" in data and isinstance(data["local"], dict):
                        local = data["local"]
                        self._config["workspace_dir"] = local.get("workspace_dir", ".")
                        self._config["output_dir"] = local.get("output_dir", "./output")
                        self._config["specs_dir"] = local.get("specs_dir", "./specs")
                        self._config["data_dir"] = local.get("data_dir", "./data")
                    else:
                        # Legacy flat format
                        self._config.update(data)
            except (yaml.YAMLError, OSError):
                pass  # Use defaults

    def save(self) -> None:
        """Persist current config to file in new nested format."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        nested = {
            "aws": {
                "profile": self._config.get("aws_profile", ""),
                "region": self._config.get("aws_region", ""),
                "s3_bucket": self._config.get("s3_bucket", ""),
                "bedrock_region": self._config.get("bedrock_region", ""),
                "agentcore_endpoint": self._config.get("agentcore_endpoint", ""),
            },
            "local": {
                "workspace_dir": self._config.get("workspace_dir", "."),
                "output_dir": self._config.get("output_dir", "./output"),
                "specs_dir": self._config.get("specs_dir", "./specs"),
                "data_dir": self._config.get("data_dir", "./data"),
            },
        }
        self._path.write_text(yaml.dump(nested, default_flow_style=False, sort_keys=False))

    def get(self, key: str, default: Any = None) -> Any:
        """Get a config value."""
        return self._config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a config value and persist."""
        self._config[key] = value
        self.save()

    @property
    def all(self) -> dict[str, Any]:
        """Get all config values."""
        return dict(self._config)

    @property
    def path(self) -> Path:
        """Config file path."""
        return self._path

    @property
    def workspace_dir(self) -> Path:
        """Resolved workspace directory."""
        return Path(self._config.get("workspace_dir", ".")).resolve()

    @property
    def output_dir(self) -> Path:
        """Resolved output directory."""
        return Path(self._config.get("output_dir", "./output")).resolve()

    @property
    def specs_dir(self) -> Path:
        """Resolved specs directory."""
        return Path(self._config.get("specs_dir", "./specs")).resolve()

    @property
    def data_dir(self) -> Path:
        """Resolved data directory."""
        return Path(self._config.get("data_dir", "./data")).resolve()

    @property
    def s3_bucket(self) -> str:
        """S3 bucket name."""
        return str(self._config.get("s3_bucket", ""))

    @property
    def aws_profile(self) -> str:
        """AWS profile name."""
        return str(self._config.get("aws_profile", ""))


def init_workspace(workspace_dir: str | Path) -> CLIConfig:
    """Initialize a new workspace with default directory structure.

    Creates:
        workspace_dir/
        ├── specs/
        ├── data/
        ├── output/
        └── .agenticml.yaml
    """
    ws = Path(workspace_dir)
    ws.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (ws / "specs").mkdir(exist_ok=True)
    (ws / "data").mkdir(exist_ok=True)
    (ws / "output").mkdir(exist_ok=True)

    # Create project config file
    project_config = ws / PROJECT_CONFIG_FILE
    if not project_config.exists():
        project_config.write_text(CONFIG_TEMPLATE)

    # Also create user config dir if it doesn't exist
    USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if not USER_CONFIG_FILE.exists():
        USER_CONFIG_FILE.write_text(CONFIG_TEMPLATE)

    config = CLIConfig(project_config)
    config.set("workspace_dir", str(ws))
    config.set("specs_dir", str(ws / "specs"))
    config.set("data_dir", str(ws / "data"))
    config.set("output_dir", str(ws / "output"))
    config.save()

    return config
