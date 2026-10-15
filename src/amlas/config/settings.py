"""Agentic-ML Configuration System.

Config resolution order (highest priority first):
1. Environment variables (AMLAS_* and standard AWS_*; generic AGENTICML_*
   aliases remain supported for compatibility)
2. Project config file (.agenticml.yaml in working directory)
3. User config file (~/.agenticml/config.yaml)
4. Default values

Supports:
- AWS profile (respects AWS CLI config and credential chain)
- S3 bucket for artifacts
- Bedrock model IDs and region
- AgentCore endpoint
- Local data/output directories
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Single env compatibility seam: canonical ``AMLAS_*`` with generic
# ``AGENTICML_*`` fallback.
from ._env_compat import env_with_compat
from .aws_region import AWS_REGION_ENV_KEYS, resolve_aws_profile, resolve_aws_region

# Single source of truth for the default AWS region. The Bedrock inference
# profiles (Opus/Sonnet/Haiku) are verified live in this region; keeping the
# Settings layer aligned with config.runtime avoids the CFG-01 region split.

# --- Config file locations ---
USER_CONFIG_DIR = Path.home() / ".agenticml"
USER_CONFIG_FILE = USER_CONFIG_DIR / "config.yaml"
PROJECT_CONFIG_FILE = ".agenticml.yaml"


@dataclass
class AWSConfig:
    """AWS connectivity configuration."""

    profile: str = ""  # AWS CLI profile name (empty = default credential chain)
    region: str | None = None  # consistent with config.runtime (CFG-01)
    s3_bucket: str = ""  # S3 bucket for artifacts
    bedrock_region: str = ""  # Region for Bedrock (defaults to region if empty)
    agentcore_endpoint: str = ""  # AgentCore endpoint URL
    # SEC-01: cloud-dispatch targets — empty = must be supplied via env/CLI.
    # NEVER default to a live account/bucket (the hardcoded values were removed).
    agentcore_runtime_arn: str = ""  # env AGENTCORE_RUNTIME_ARN
    session_bucket: str = ""  # env AMLAS_SESSION_BUCKET (falls back to s3_bucket if empty)

    @property
    def effective_session_bucket(self) -> str:
        """Session S3 bucket for cloud dispatch, falling back to s3_bucket."""
        return self.session_bucket or self.s3_bucket

    @property
    def effective_bedrock_region(self) -> str:
        """Bedrock region, falling back to primary region."""
        return self.bedrock_region or self.region

    @property
    def effective_profile(self) -> str | None:
        """Profile for boto3 Session (None triggers default chain)."""
        return self.profile if self.profile else None


@dataclass
class LakeConfig:
    """Medallion lakehouse configuration (master plan §11; tools/lake adapter).

    SEC-01: ``silver_prefix`` / ``gold_prefix`` default to empty — NEVER hardcode a
    live bucket. They are supplied via ``.agenticml.yaml`` or ``AGENTICML_LAKE_*``
    env vars. ``catalog`` selects the data-catalog backend (``glue`` | ``none``).
    """

    catalog: str = "glue"  # glue | none
    silver_prefix: str = ""  # e.g. s3://<bucket>/silver/
    gold_prefix: str = ""  # e.g. s3://<bucket>/gold/


@dataclass
class MLOpsConfig:
    """SageMaker MLOps configuration (master plan §11; tools/mlops agents).

    Names follow the §11 sample: a Model Registry package group, a Feature Store
    database, and a default Model Monitor schedule cadence.
    """

    model_package_group: str = "amlas_models"
    feature_store_db: str = "amlas_features"
    monitor_schedule: str = "hourly"  # hourly | daily | weekly | cron(...)


@dataclass
class LocalConfig:
    """Local workspace directory configuration."""

    workspace_dir: str = "."
    output_dir: str = "./output"
    specs_dir: str = "./specs"
    data_dir: str = "./data"

    @property
    def resolved_workspace(self) -> Path:
        return Path(self.workspace_dir).resolve()

    @property
    def resolved_output(self) -> Path:
        return Path(self.output_dir).resolve()

    @property
    def resolved_specs(self) -> Path:
        return Path(self.specs_dir).resolve()

    @property
    def resolved_data(self) -> Path:
        return Path(self.data_dir).resolve()


@dataclass
class Settings:
    """Top-level settings container with layered config resolution."""

    aws: AWSConfig = field(default_factory=AWSConfig)
    local: LocalConfig = field(default_factory=LocalConfig)
    lake: LakeConfig = field(default_factory=LakeConfig)
    mlops: MLOpsConfig = field(default_factory=MLOpsConfig)

    @classmethod
    def load(cls) -> Settings:
        """Load settings from config files + env vars (highest priority last)."""
        settings = cls()

        # Layer 1: User config (~/.agenticml/config.yaml)
        user_config = USER_CONFIG_FILE
        if user_config.exists():
            settings._merge_yaml(user_config)

        # Layer 2: Project config (.agenticml.yaml in cwd)
        project_config = Path.cwd() / PROJECT_CONFIG_FILE
        if project_config.exists():
            settings._merge_yaml(project_config)

        # Layer 3: Environment variables (highest priority)
        settings._apply_env_vars()

        return settings

    @classmethod
    def from_yaml(cls, path: Path) -> Settings:
        """Load settings from a specific YAML file only."""
        settings = cls()
        settings._merge_yaml(path)
        return settings

    def _merge_yaml(self, path: Path) -> None:
        """Merge a YAML config file into current settings."""
        try:
            data = yaml.safe_load(path.read_text())
            if not isinstance(data, dict):
                return
        except (yaml.YAMLError, OSError):
            return

        # Merge aws section
        if "aws" in data and isinstance(data["aws"], dict):
            aws_data = data["aws"]
            if "profile" in aws_data:
                self.aws.profile = str(aws_data["profile"] or "")
            if "region" in aws_data:
                self.aws.region = str(aws_data["region"])
            if "s3_bucket" in aws_data:
                self.aws.s3_bucket = str(aws_data["s3_bucket"] or "")
            if "bedrock_region" in aws_data:
                self.aws.bedrock_region = str(aws_data["bedrock_region"] or "")
            if "agentcore_endpoint" in aws_data:
                self.aws.agentcore_endpoint = str(aws_data["agentcore_endpoint"] or "")
            if "agentcore_runtime_arn" in aws_data:
                self.aws.agentcore_runtime_arn = str(aws_data["agentcore_runtime_arn"] or "")
            if "session_bucket" in aws_data:
                self.aws.session_bucket = str(aws_data["session_bucket"] or "")

        # Merge local section
        if "local" in data and isinstance(data["local"], dict):
            local_data = data["local"]
            if "workspace_dir" in local_data:
                self.local.workspace_dir = str(local_data["workspace_dir"])
            if "output_dir" in local_data:
                self.local.output_dir = str(local_data["output_dir"])
            if "specs_dir" in local_data:
                self.local.specs_dir = str(local_data["specs_dir"])
            if "data_dir" in local_data:
                self.local.data_dir = str(local_data["data_dir"])

        # Merge lake section (Phase 10 — medallion adapter)
        if "lake" in data and isinstance(data["lake"], dict):
            lake_data = data["lake"]
            if "catalog" in lake_data:
                self.lake.catalog = str(lake_data["catalog"] or "")
            if "silver_prefix" in lake_data:
                self.lake.silver_prefix = str(lake_data["silver_prefix"] or "")
            if "gold_prefix" in lake_data:
                self.lake.gold_prefix = str(lake_data["gold_prefix"] or "")

        # Merge mlops section (Phase 10 — SageMaker MLOps)
        if "mlops" in data and isinstance(data["mlops"], dict):
            mlops_data = data["mlops"]
            if "model_package_group" in mlops_data:
                self.mlops.model_package_group = str(mlops_data["model_package_group"] or "")
            if "feature_store_db" in mlops_data:
                self.mlops.feature_store_db = str(mlops_data["feature_store_db"] or "")
            if "monitor_schedule" in mlops_data:
                self.mlops.monitor_schedule = str(mlops_data["monitor_schedule"] or "")

    def _apply_env_vars(self) -> None:
        """Override settings from environment variables.

        Env keys use the canonical ``AMLAS_*`` prefix and are read through
        :func:`env_with_compat` so a generic ``AGENTICML_*`` name still
        resolves with a one-time deprecation. The standard AWS_*
        names are still honored as the lowest env-tier fallback; ``env_with_compat``
        preserves canonical-over-legacy precedence.
        """
        # AWS standard env vars (lowest env tier; AMLAS_/legacy names below win).
        if os.environ.get("AWS_PROFILE"):
            self.aws.profile = os.environ["AWS_PROFILE"]
        # One target-region contract; a standard AWS_REGION selection wins aliases.
        if (profile := resolve_aws_profile()):
            self.aws.profile = profile
        env_region = next((os.environ[key].strip() for key in AWS_REGION_ENV_KEYS
                           if os.environ.get(key, "").strip()), None)
        self.aws.region = resolve_aws_region(env_region or self.aws.region,
                                            profile_name=self.aws.effective_profile, required=False)
        if (s3_bucket := env_with_compat("AMLAS_S3_BUCKET")):
            self.aws.s3_bucket = s3_bucket
        if (bedrock_region := env_with_compat("AMLAS_BEDROCK_REGION")):
            self.aws.bedrock_region = bedrock_region
        if (agentcore_endpoint := env_with_compat("AMLAS_AGENTCORE_ENDPOINT")):
            self.aws.agentcore_endpoint = agentcore_endpoint
        # SEC-01: cloud-dispatch targets from env (also the AgentCore convention
        # in async_callback/lambda_handler.py + the deployed runtime env).
        if os.environ.get("AGENTCORE_RUNTIME_ARN"):
            self.aws.agentcore_runtime_arn = os.environ["AGENTCORE_RUNTIME_ARN"]
        if (session_bucket := env_with_compat("AMLAS_SESSION_BUCKET")):
            self.aws.session_bucket = session_bucket
        elif os.environ.get("SESSION_BUCKET"):
            self.aws.session_bucket = os.environ["SESSION_BUCKET"]

        # Local dirs
        if (workspace_dir := env_with_compat("AMLAS_WORKSPACE_DIR")):
            self.local.workspace_dir = workspace_dir
        if (output_dir := env_with_compat("AMLAS_OUTPUT_DIR")):
            self.local.output_dir = output_dir

        # Lake (medallion) env vars (Phase 10)
        if (lake_catalog := env_with_compat("AMLAS_LAKE_CATALOG")):
            self.lake.catalog = lake_catalog
        if (lake_silver := env_with_compat("AMLAS_LAKE_SILVER_PREFIX")):
            self.lake.silver_prefix = lake_silver
        if (lake_gold := env_with_compat("AMLAS_LAKE_GOLD_PREFIX")):
            self.lake.gold_prefix = lake_gold

        # MLOps env vars (Phase 10)
        if (mpg := env_with_compat("AMLAS_MODEL_PACKAGE_GROUP")):
            self.mlops.model_package_group = mpg
        if (fsdb := env_with_compat("AMLAS_FEATURE_STORE_DB")):
            self.mlops.feature_store_db = fsdb
        if (monitor := env_with_compat("AMLAS_MONITOR_SCHEDULE")):
            self.mlops.monitor_schedule = monitor

    def to_dict(self) -> dict[str, Any]:
        """Serialize settings to a dict (for YAML export)."""
        return {
            "aws": {
                "profile": self.aws.profile,
                "region": self.aws.region,
                "s3_bucket": self.aws.s3_bucket,
                "bedrock_region": self.aws.bedrock_region,
                "agentcore_endpoint": self.aws.agentcore_endpoint,
                "agentcore_runtime_arn": self.aws.agentcore_runtime_arn,
                "session_bucket": self.aws.session_bucket,
            },
            "local": {
                "workspace_dir": self.local.workspace_dir,
                "output_dir": self.local.output_dir,
                "specs_dir": self.local.specs_dir,
                "data_dir": self.local.data_dir,
            },
            "lake": {
                "catalog": self.lake.catalog,
                "silver_prefix": self.lake.silver_prefix,
                "gold_prefix": self.lake.gold_prefix,
            },
            "mlops": {
                "model_package_group": self.mlops.model_package_group,
                "feature_store_db": self.mlops.feature_store_db,
                "monitor_schedule": self.mlops.monitor_schedule,
            },
        }

    def save(self, path: Path | None = None) -> Path:
        """Save settings to a YAML file."""
        target = path or USER_CONFIG_FILE
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False))
        return target


# --- Default config template content ---
CONFIG_TEMPLATE = """\
# Agentic-ML Configuration
# Copy to ~/.agenticml/config.yaml or .agenticml.yaml in project root
#
# Config resolution order (highest priority first):
#   1. Environment variables (AGENTICML_*, AWS_PROFILE, AWS_REGION)
#   2. Project config (.agenticml.yaml in working directory)
#   3. User config (~/.agenticml/config.yaml)
#   4. Defaults

aws:
  profile: ""              # AWS CLI profile (leave empty for default credential chain)
  region: ""              # Select a region in setup, environment or the chosen AWS profile
  s3_bucket: ""           # S3 bucket for model artifacts
  bedrock_region: ""      # Bedrock region (defaults to aws.region if empty)
  agentcore_endpoint: ""  # AgentCore runtime endpoint
  agentcore_runtime_arn: ""  # SEC-01: AgentCore runtime ARN for cloud dispatch (env AGENTCORE_RUNTIME_ARN)
  session_bucket: ""      # Session S3 bucket for cloud dispatch (env AMLAS_SESSION_BUCKET; falls back to s3_bucket)

local:
  workspace_dir: "."
  output_dir: "./output"
  specs_dir: "./specs"
  data_dir: "./data"

lake:                       # Phase 10 — medallion lakehouse (tools/lake adapter)
  catalog: "glue"           # glue | none (Glue Data Catalog backend)
  silver_prefix: ""         # SEC-01: e.g. s3://<bucket>/silver/ (env AGENTICML_LAKE_SILVER_PREFIX)
  gold_prefix: ""           # SEC-01: e.g. s3://<bucket>/gold/   (env AGENTICML_LAKE_GOLD_PREFIX)

mlops:                      # Phase 10 — SageMaker MLOps (tools/mlops agents)
  model_package_group: "amlas_models"   # Model Registry group (env AGENTICML_MODEL_PACKAGE_GROUP)
  feature_store_db: "amlas_features"    # Feature Store database (env AGENTICML_FEATURE_STORE_DB)
  monitor_schedule: "hourly"             # Model Monitor cadence (env AGENTICML_MONITOR_SCHEDULE)
"""
