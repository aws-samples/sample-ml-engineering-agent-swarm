"""Write a provider machine secret from JSON stdin without exposing its value."""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import boto3


def write_secret(operation: str, payload: dict[str, Any]) -> None:
    """Use the selected profile or default credential chain and explicit region."""
    if operation not in ("create", "put") or not isinstance(payload, dict):
        raise ValueError("invalid secret operation or payload")
    region = os.environ.get("AWS_REGION", "")
    if not region:
        raise ValueError("explicit AWS_REGION is required")
    profile = os.environ.get("AWS_PROFILE") or None
    if profile is None:
        # An empty selector is not the native SDK default credential chain.
        os.environ.pop("AWS_PROFILE", None)
    session = boto3.Session(profile_name=profile, region_name=region)
    client = session.client("secretsmanager")
    if operation == "create":
        client.create_secret(**payload)
    else:
        client.put_secret_value(**payload)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    try:
        if len(args) != 1 or args[0] not in ("create", "put"):
            raise ValueError("invalid secret operation")
        payload = json.load(sys.stdin)
        write_secret(args[0], payload)
    except Exception:
        # SDK and parser exceptions may contain request values or untrusted text.
        # Emit no exception body, payload, response, credentials, or traceback.
        print("Research secret write failed.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
