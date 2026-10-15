"""The single owner of the managed SageMaker scikit-learn image URIs.

There were THREE hardcoded copies of this decision — [`tools/sagemaker.py`](../tools/sagemaker.py:1)
(training), [`backends/adapters/etl.py`](../backends/adapters/etl.py:1) (processing) and
[`tools/mlops/pipelines.py`](../tools/mlops/pipelines.py:1) (pipeline steps) — and they had already
drifted in two ways:

* only the pipelines copy carried a per-region DLC account map, so the other two composed
  ``246618743249`` (a ``us-*`` registry) into every region;
* the training copy pinned a Python 3.9 image while the code it runs needs 3.11+.

This module is dependency-free (no ``amlas`` imports, no boto3) so every caller can use it
without an import cycle.

Deliberately TWO tags, not one
------------------------------
``SKLEARN_TRAINING_TAG`` and ``SKLEARN_PROCESSING_TAG`` are separate because only the TRAINING
container executes ``amlas`` code. Collapsing them would silently move processing onto a new
scikit-learn minor for no reason; keeping them apart makes each version an explicit decision.
"""

from __future__ import annotations

import re

from ._sagemaker_registry import (
    CLARIFY_DLC_ACCOUNTS,
    MODEL_MONITOR_DLC_ACCOUNTS,
    PYTORCH_DLC_ACCOUNTS,
    SKLEARN_DLC_ACCOUNTS,
    XGBOOST_DLC_ACCOUNTS,
)

#: The AWS-owned scikit-learn DLC registry account, per region. ``246618743249`` is the ``us-*``
#: registry for that specific region; the others differ and a wrong publisher yields an
#: unpullable image URI.
from .aws_region import resolve_aws_region

#: TRAINING image tag. MUST be a Python >= 3.11 family, because this container runs
#: [`sagemaker_entrypoints/tabular_train.py`](../sagemaker_entrypoints/tabular_train.py:1), which
#: imports ``tools.training_tools`` → ``errors`` → ``errors.taxonomy``. On the previous
#: ``1.2-1-cpu-py3`` (Python 3.9) EVERY dispatched training job died at import with
#: ``ImportError: cannot import name 'StrEnum' from 'enum'``, which no offline test
#: could see because the repo's own interpreter is 3.11. ``taxonomy`` now carries a py<3.11
#: fallback too — belt and braces, since the container's Python is not ours to control — but the
#: image should still be a modern one rather than relying on the shim.
SKLEARN_TRAINING_TAG = "1.4-2-py312"

#: The Python versions the training tag is allowed to be built on. Asserted by the unit suite so
#: a future edit cannot quietly reintroduce a py3.9 image.
ALLOWED_TRAINING_PY_TAGS = ("py311", "py312", "py313")

#: PROCESSING image tag. Unchanged at scikit-learn 1.2-1: processing jobs run inline scripts and
#: do NOT import ``amlas``, so they are unaffected by the training constraint above. Bump this
#: only for a processing-side reason.
SKLEARN_PROCESSING_TAG = "1.2-1"


def sklearn_dlc_account(region: str) -> str:
    """The AWS scikit-learn DLC registry account for ``region`` (``us-*`` default)."""
    try:
        return SKLEARN_DLC_ACCOUNTS[region]
    except KeyError as exc:
        raise ValueError(
            f"No verified SageMaker scikit-learn registry for region {region!r}; "
            "supply an explicit image_uri/default_image or select a supported image region."
        ) from exc


def sklearn_image(region: str, tag: str) -> str:
    """Compose the region-correct managed scikit-learn image URI for ``tag``."""
    return regional_image(SKLEARN_DLC_ACCOUNTS, region, "sagemaker-scikit-learn", tag)


def sklearn_training_image(region: str) -> str:
    """The managed scikit-learn TRAINING image — Python >= 3.11 (see module docstring)."""
    return sklearn_image(region, SKLEARN_TRAINING_TAG)


def sklearn_processing_image(region: str) -> str:
    """The managed scikit-learn PROCESSING image."""
    return sklearn_image(region, SKLEARN_PROCESSING_TAG)


__all__ = [
    "ALLOWED_TRAINING_PY_TAGS",
    "SKLEARN_DLC_ACCOUNTS",
    "SKLEARN_PROCESSING_TAG",
    "SKLEARN_TRAINING_TAG",
    "sklearn_dlc_account",
    "sklearn_image",
    "sklearn_processing_image",
    "sklearn_training_image",
]


def regional_image(accounts: dict[str, str], region: str | None, repository: str, tag: str = "") -> str:
    """Compose only a catalog-listed image; unknown partitions require an override."""
    selected = resolve_aws_region(region)
    if selected.startswith(("us-iso-", "us-isob-", "us-isof-", "eu-isoe-")):
        raise ValueError(
            f"No verified ECR domain for image region {selected!r}; supply an explicit image URI."
        )
    try:
        account = accounts[selected]
    except KeyError as exc:
        raise ValueError(
            f"No verified {repository} image registry for region {selected!r}; supply an explicit image URI."
        ) from exc
    suffix = "amazonaws.com.cn" if selected.startswith("cn-") else "amazonaws.com"
    version = f":{tag}" if tag else ""
    return f"{account}.dkr.ecr.{selected}.{suffix}/{repository}{version}"


def xgboost_image(region: str | None) -> str:
    return regional_image(XGBOOST_DLC_ACCOUNTS, region, "sagemaker-xgboost", "1.5-1")


def clarify_image(region: str | None) -> str:
    return regional_image(CLARIFY_DLC_ACCOUNTS, region, "sagemaker-clarify-processing", "1.0")


def model_monitor_image(region: str | None) -> str:
    return regional_image(MODEL_MONITOR_DLC_ACCOUNTS, region, "sagemaker-model-monitor-analyzer")


def pytorch_training_image(region: str | None, *, gpu: bool) -> str:
    return regional_image(
        PYTORCH_DLC_ACCOUNTS, region, "pytorch-training", "2.1.0-gpu-py310" if gpu else "2.1.0-cpu-py310"
    )


def validate_image_region(image: str, region: str | None) -> str:
    """Reject regional ECR overrides that target a different SageMaker region."""
    match = re.match(r"^(?:https?://)?[0-9]{12}\.dkr\.ecr\.([a-z0-9-]+)\.", image)
    selected = resolve_aws_region(region, required=False)
    if match and selected and match.group(1) != selected:
        raise ValueError(
            f"Image region {match.group(1)!r} differs from target region {selected!r}; supply a same-region image."
        )
    return image
