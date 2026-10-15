"""Resolve AWS target regions from explicit inputs and local configuration only.

Region discovery never reads credentials or calls AWS. Offline callers may ask for
``required=False``; an AWS operation must require a target region before creating
its client. No built-in target region or import-time environment snapshot exists.
"""

from __future__ import annotations

import configparser
import os
import shlex
from pathlib import Path

AWS_REGION_ENV_KEYS = ("AWS_REGION", "AMLAS_AWS_REGION", "AGENTICML_AWS_REGION", "AWS_DEFAULT_REGION")


class MissingRegionError(ValueError):
    """An AWS operation has no configured target region."""


def _clean(value: str | None) -> str | None:
    text = (value or "").strip()
    return text if text and not ("<" in text and ">" in text) else None


def resolve_aws_profile(explicit: str | None = None) -> str | None:
    """Use the selected profile, without loading credentials."""
    return _clean(explicit) or next(
        (
            _clean(os.environ.get(key))
            for key in ("AWS_PROFILE", "AMLAS_AWS_PROFILE", "AGENTICML_AWS_PROFILE", "AWS_DEFAULT_PROFILE")
            if _clean(os.environ.get(key))
        ),
        None,
    )


def resolve_aws_region(
    explicit: str | None = None, *, profile_name: str | None = None, required: bool = True
) -> str | None:
    """Explicit -> AWS_REGION -> application alias -> AWS_DEFAULT_REGION -> profile.

    The AWS config INI file holds region metadata; the shared credentials file is
    never read. Unknown/future region names are passed through for service preflight
    to validate, rather than restricted by a stale package-wide region allowlist.
    """
    region = _clean(explicit)
    if region is None:
        region = next(
            (_clean(os.environ.get(key)) for key in AWS_REGION_ENV_KEYS if _clean(os.environ.get(key))), None
        )
    profile = resolve_aws_profile(profile_name) or "default"
    if region is None:
        config_path = Path(os.environ.get("AWS_CONFIG_FILE") or "~/.aws/config").expanduser()
        config = configparser.RawConfigParser()
        try:
            config.read(config_path, encoding="utf-8")
        except (OSError, configparser.Error, UnicodeError) as exc:
            if not required:
                # Offline callers do not need readable/valid AWS metadata.
                return None
            # Parser errors may include raw configuration lines. Keep diagnostics to
            # metadata and action, and suppress the parser's original traceback.
            raise MissingRegionError(
                f"Cannot read local AWS region configuration {config_path} ({type(exc).__name__}); "
                "check the AWS config file format, set AWS_REGION, or rerun scripts/setup.sh."
            ) from None
        # Match botocore's profile section grammar without loading credentials.
        # Quoting (or escaping) permits spaces; unquoted multiword names are not profiles.
        profiles: dict[str, str] = {}
        for section in config.sections():
            if section == "default":
                profiles["default"] = section
            elif section.startswith("profile"):
                try:
                    parts = shlex.split(section)
                except ValueError:
                    continue
                if len(parts) == 2:
                    profiles[parts[1]] = section
        section = profiles.get(profile)
        region = _clean(config.get(section, "region", fallback=None)) if section is not None else None
    if region is None and required:
        raise MissingRegionError(
            "No AWS region selected. Run scripts/setup.sh, set AWS_REGION, "
            "pass an explicit region, or configure region in the selected AWS profile."
        )
    return region


def resolve_aws_client_region(
    client: object | None, explicit: str | None = None, *, required: bool = True
) -> str | None:
    """Use an explicit selection or concrete SDK endpoint metadata before ambient config.

    An injected client's endpoint is the request target. Ambient configuration must
    not move its default images to a different region; an explicit disagreement is
    rejected before any service write. Metadata-free substitutes may use an explicit
    image with ``required=False``. No client construction or credential lookup occurs.
    """
    selected = _clean(explicit)
    metadata = getattr(getattr(client, "meta", None), "region_name", None)
    client_region = _clean(metadata) if isinstance(metadata, str) else None
    if selected and client_region and selected != client_region:
        raise ValueError(
            f"Explicit region {selected!r} differs from client region {client_region!r}; "
            "select a matching region or client."
        )
    return selected or client_region or resolve_aws_region(required=required)
