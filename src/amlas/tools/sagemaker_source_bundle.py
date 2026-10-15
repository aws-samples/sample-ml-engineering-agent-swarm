"""FIX-34 (ADR-0002): build + upload the SageMaker SCRIPT-MODE source tarball for tabular training.

The stock SageMaker SKLearn framework container runs in *script mode*: it downloads a
``sourcedir.tar.gz`` from S3 (``sagemaker_submit_directory``), ``pip install``s any bundled
``requirements.txt``, then runs the script named by ``sagemaker_program``. The prior dispatch set
neither, so the container started with ``program=None`` and crashed at ``name.endswith('.sh')``.

This module packages, at DISPATCH time, exactly what that container needs — with NO new Docker
image to build/push (managed-services-first): a tarball containing

  * ``train.py``      — a thin root wrapper the container runs; it calls
    :func:`amlas.sagemaker_entrypoints.tabular_train.main`.
  * the ``amlas`` package (copied from the running dispatcher's installed source tree) so the
    reused, tested fit logic (``_build_model`` / ``_preprocess_for_training``) is byte-identical
    off-box. ``@tool`` degrades to identity when ``strands`` is absent (FIX-34 import guard), so
    the package imports cleanly in the strands-free training container.
  * ``requirements.txt`` — the handful of libs the stock SKLearn container may lack
    (xgboost / lightgbm / skops / pyyaml). scikit-learn/numpy/pandas are already present.

Everything is filesystem + boto3; no SageMaker SDK dependency (kept out of the runtime image).
"""

from __future__ import annotations

import io
import os
import tarfile
from pathlib import Path
from typing import Any

from ..config.aws_region import resolve_aws_region

# The root wrapper script the framework container executes (sagemaker_program). Kept tiny: it only
# delegates to the tested entrypoint module bundled alongside it.
_ROOT_TRAIN_PY = '''\
"""SageMaker script-mode entry (FIX-34). Runs inside the SKLearn framework container."""
from amlas.sagemaker_entrypoints.tabular_train import main

if __name__ == "__main__":
    main()
'''

# Extra libs the SKLearn framework container lacks (xgboost / lightgbm / skops). The scientific
# stack (numpy/scipy/pandas/scikit-learn/joblib) is ALREADY in the container.
#
# CRITICAL — PIN THE BASE STACK TO THE DLC's EXACT SHIPPED VERSIONS (FIX-34b→34c, live-discovered
# in two steps). The DLC ships C-extension libs (numpy/scipy/pandas/sklearn) compiled against ONE
# specific ABI. ANY pip install that perturbs one of them breaks the others at import:
#   * unpinned  → pip pulled numpy 2.x → "numpy.dtype size changed (96→88)" on ``import pandas`` (34b)
#   * ``numpy<2`` only → pip pulled a numpy != the DLC's build → the DLC's prebuilt SCIPY then failed
#     with a ``dfitpack_int`` TypeError on ``import scipy`` (34c).
# The robust, universal fix (managed-services-first): pin numpy/scipy/pandas/scikit-learn/joblib to
# the DLC's EXACT versions (read from the container as ground truth) so pip sees them as already
# satisfied and NEVER rebuilds the base stack — only xgboost/lightgbm/skops are added.
#
# ── Updated for the py312 DLC ─────────────────────────────────────────────────────
# The training image moved from ``1.2-1-cpu-py3`` (Python 3.9) to ``1.4-2-py312``, because the
# entrypoint imports ``amlas`` and Python 3.9 lacks ``enum.StrEnum``. That change made these pins
# WRONG in a new way: ``numpy==1.24.1`` has no cp312 wheel, so pip built it from source, and the
# sdist's vendored setuptools calls ``pkgutil.ImpImporter`` — removed in 3.12 — so every job died
# with ``InstallRequirementsError``. Pinning is only safe when the pins ARE the container's build.
#
# HOW TO RE-READ THEM (this image is x86_64, so it cannot be run on an arm64 dev host — export its
# filesystem instead of executing it):
#
#   aws ecr get-login-password --region us-west-2 \
#     | docker login --username AWS --password-stdin 246618743249.dkr.ecr.us-west-2.amazonaws.com
#   docker pull --platform linux/amd64 <image>
#   CID=$(docker create --platform linux/amd64 <image> /bin/true)
#   docker export "$CID" | tar -tf - | grep -oE '[^/]+\.dist-info/$' | sort -u
#   docker rm -f "$CID"
#
# Ground truth read that way from ``1.4-2-py312`` on 2026-08-28 (81 distributions):
#   numpy 2.1.0 · scipy 1.18.0 · pandas 2.3.2 · scikit-learn 1.4.2 · joblib 1.5.3 · PyYAML 6.0.1
# PyYAML is already present, so it needs no line; xgboost / lightgbm / skops are genuinely absent.
#
# The extras are RANGES, not exact pins, and that asymmetry is deliberate: the base stack must be
# pinned so pip leaves the ABI alone, while the extras must be free to resolve a build compatible
# with numpy 2.x (the old ``xgboost==1.7.6`` / ``lightgbm==4.1.0`` predate numpy-2 support). Lower
# bounds are the first releases that support numpy 2.
_DLC_BASE_STACK = (
    # Read from the container — see the docstring above before touching these.
    "numpy==2.1.0",
    "scipy==1.18.0",
    "pandas==2.3.2",
    "scikit-learn==1.4.2",
    "joblib==1.5.3",
)

_REQUIREMENTS_TXT = "\n".join(
    [
        *_DLC_BASE_STACK,
        # extras the container lacks — ranges so they resolve against numpy 2.x:
        "xgboost>=2.1,<4",
        "lightgbm>=4.5,<5",
        "skops>=0.11,<1",
        "",
    ]
)

_PROGRAM_NAME = "train.py"


def _amlas_package_dir() -> Path:
    """Locate the installed ``amlas`` package source dir (the running dispatcher's own tree)."""
    import amlas

    pkg = Path(amlas.__file__).resolve().parent
    if not (pkg / "__init__.py").is_file():  # pragma: no cover - defensive
        raise FileNotFoundError(f"amlas package dir not found at {pkg}")
    return pkg


def _should_skip(rel_posix: str) -> bool:
    """Prune bytecode / caches / bulky non-runtime dirs from the tarball."""
    parts = rel_posix.split("/")
    if any(p in {"__pycache__", ".mypy_cache", ".ruff_cache"} for p in parts):
        return True
    return rel_posix.endswith((".pyc", ".pyo"))


def build_source_tarball() -> bytes:
    """Build the script-mode ``sourcedir.tar.gz`` in memory and return its bytes.

    Layout inside the tar (all at the archive root, per SageMaker script-mode convention):
      train.py, requirements.txt, amlas/**  (the package, minus caches/bytecode)
    """
    pkg_dir = _amlas_package_dir()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        # Root wrapper script (the sagemaker_program).
        info = tarfile.TarInfo(name=_PROGRAM_NAME)
        data = _ROOT_TRAIN_PY.encode("utf-8")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
        # requirements.txt (installed by the framework container before running the program).
        rinfo = tarfile.TarInfo(name="requirements.txt")
        rdata = _REQUIREMENTS_TXT.encode("utf-8")
        rinfo.size = len(rdata)
        tar.addfile(rinfo, io.BytesIO(rdata))
        # The amlas package itself, at archive root as ``amlas/...``.
        for root, _dirs, files in os.walk(pkg_dir):
            for fn in files:
                abspath = Path(root) / fn
                rel = abspath.relative_to(pkg_dir.parent).as_posix()  # -> amlas/...
                if _should_skip(rel):
                    continue
                tar.add(str(abspath), arcname=rel, recursive=False)
    return buf.getvalue()


def upload_source_tarball(
    *,
    session_bucket: str,
    job_name: str,
    region: str | None = None,
    s3_client: Any = None,
    key_prefix: str = "sourcedir",
) -> str:
    """Build + upload the tarball to S3; return its ``s3://`` URI (sagemaker_submit_directory)."""
    import boto3

    tar_bytes = build_source_tarball()
    s3 = s3_client or boto3.client("s3", region_name=resolve_aws_region(region))
    key = f"{key_prefix.rstrip('/')}/{job_name}/sourcedir.tar.gz"
    s3.put_object(Bucket=session_bucket, Key=key, Body=tar_bytes)
    return f"s3://{session_bucket}/{key}"


def script_mode_hyperparameters(submit_dir_s3_uri: str) -> dict[str, str]:
    """The reserved script-mode hyperparameters the framework container reads.

    ``sagemaker_program`` names the script to run; ``sagemaker_submit_directory`` is the S3 tarball
    to download + unpack. SageMaker JSON-encodes hyperparameter values, so these are strings.
    """
    return {
        "sagemaker_program": _PROGRAM_NAME,
        "sagemaker_submit_directory": submit_dir_s3_uri,
    }
