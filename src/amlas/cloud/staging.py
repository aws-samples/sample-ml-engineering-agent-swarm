"""N1 + N2 — runtime-side dataset staging + y_true materialization.

Phase 14a proved the cloud round-trip works but the run took a *correct* data-
integrity retreat at D1 because:
    N1 — no dataset existed on the container filesystem and the problem card
         carried no train/validation paths, so the agents guessed `data/ai4i/train.csv`
         and found nothing.
    N2 — `y_true_path` was a bare local path the Arbiter would `np.load` directly;
         nothing materialized it.

This module fixes BOTH at the source, server-side, from the SAME split so the
y_true the Arbiter grades is guaranteed 1:1 aligned with the held-out test rows
the ML Lead predicts on.

Pure boto3 + pandas + numpy — NO strands / bedrock_agentcore imports, so it is
fully directly verifiable with real AWS. `app.py` calls `stage_dataset(...)` at startup
and threads the returned concrete paths into the Orchestrator's problem card.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config.aws_region import resolve_aws_region

# SEC-01: region default sourced from the single canonical region constant
# (config.runtime.DEFAULT_REGION) rather than a re-declared "us-west-2" literal.
from ..golden.ai4i import (
    AI4I_TARGET,
    load_ai4i,
)
from ..spec.schema import SpecModel

# AI4I raw failure-mode flag columns — their presence signals the AI4I golden
# collapse path (5 binary flags → one multi-class `failure_mode`).
_AI4I_MODE_COLS = {"TWF", "HDF", "PWF", "OSF", "RNF"}


@dataclass(frozen=True)
class StagedData:
    """Result of staging a spec's golden_dataset onto the container filesystem.

    Every path is a concrete, existing local file the data subagents + Arbiter
    can open directly — no guessing, no S3 round-trips mid-run.
    """

    train_path: str
    test_path: str
    y_true_path: str
    target: str
    n_train: int
    n_test: int
    source: str  # "local" | "s3"


class StagingError(RuntimeError):
    """Raised when a spec's golden_dataset cannot be resolved or staged."""


# ---------------------------------------------------------------------------
# S3 URI helpers
# ---------------------------------------------------------------------------


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Split an ``s3://bucket/key`` URI into ``(bucket, key)``.

    Raises:
        ValueError: if ``uri`` does not start with ``s3://`` or has no key.
    """
    if not uri.startswith("s3://"):
        raise ValueError(f"not an s3:// uri: {uri!r}")
    rest = uri[len("s3://") :]
    if "/" not in rest:
        raise ValueError(f"s3 uri missing key: {uri!r}")
    bucket, key = rest.split("/", 1)
    if not bucket or not key:
        raise ValueError(f"s3 uri missing bucket or key: {uri!r}")
    return bucket, key


def _make_s3_client(region: str, s3_client: Any | None) -> Any:
    if s3_client is not None:
        return s3_client
    import boto3  # local import — keeps module import cheap + boto-optional in pure tests

    return boto3.client("s3", region_name=resolve_aws_region(region))


def _download_s3_to(
    uri: str, dest: Path, *, s3_client: Any | None, region: str
) -> Path:
    bucket, key = parse_s3_uri(uri)
    client = _make_s3_client(region, s3_client)
    dest.parent.mkdir(parents=True, exist_ok=True)
    client.download_file(bucket, key, str(dest))
    return dest


def _s3_object_exists(uri: str, *, s3_client: Any | None, region: str) -> bool:
    bucket, key = parse_s3_uri(uri)
    client = _make_s3_client(region, s3_client)
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except Exception:
        return False


def resolve_sagemaker_output_artifact(
    y_pred_uri: str, *, dest: Path, s3_client: Any | None = None, region: str | None = None
) -> Path | None:
    """Materialize a SageMaker artifact (e.g. ``y_pred_test.npy``) to ``dest`` — FIX-36.

    The agent reports a y_pred path like ``s3://…/<job>/output/y_pred_test.npy``, but the SageMaker
    FRAMEWORK container (SKLearn/PyTorch script mode) does NOT write loose files there — it TARS the
    ``SM_OUTPUT_DATA_DIR`` contents into ``…/output/output.tar.gz``. So the loose object 404s and the
    Arbiter can't grade. This resolver bridges BOTH layouts, universally (ADR-0002 Phase 6):

      1. If the loose object exists at ``y_pred_uri`` → download it to ``dest`` (custom-container /
         legacy layout).
      2. Else download the sibling ``output/output.tar.gz`` and EXTRACT the requested basename from
         it (framework-container script-mode layout). Also handles ``model.tar.gz`` as a fallback.

    Returns the local ``dest`` path on success, or ``None`` if neither layout yields the file (the
    caller then raises the usual FileNotFoundError).
    """
    if not y_pred_uri.startswith("s3://"):
        return None
    filename = Path(parse_s3_uri(y_pred_uri)[1]).name
    # 1) loose object (custom-container / already-extracted layout)
    if _s3_object_exists(y_pred_uri, s3_client=s3_client, region=region):
        _download_s3_to(y_pred_uri, dest, s3_client=s3_client, region=region)
        return dest
    # 2) framework-container layout: <…>/output/output.tar.gz holds SM_OUTPUT_DATA_DIR contents.
    import tarfile

    bucket, key = parse_s3_uri(y_pred_uri)
    output_dir_key = key.rsplit("/", 1)[0]  # strip the y_pred filename → the 'output/' prefix
    for tar_name in ("output.tar.gz", "model.tar.gz"):
        tar_uri = f"s3://{bucket}/{output_dir_key}/{tar_name}"
        if not _s3_object_exists(tar_uri, s3_client=s3_client, region=region):
            continue
        with tempfile.TemporaryDirectory(prefix="amlas-sm-artifact-") as td:
            local_tar = Path(td) / tar_name
            _download_s3_to(tar_uri, local_tar, s3_client=s3_client, region=region)
            with tarfile.open(local_tar, mode="r:gz") as tf:
                member = next(
                    (m for m in tf.getmembers() if Path(m.name).name == filename), None
                )
                if member is None:
                    continue
                extracted = tf.extractfile(member)
                if extracted is None:
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(extracted.read())
                return dest
    return None


# ---------------------------------------------------------------------------
# y_true loader (N2 contract — used by app.py)
# ---------------------------------------------------------------------------


def load_y_true_npy(
    path_or_uri: str,
    *,
    s3_client: Any | None = None,
    region: str | None = None,
    download_dir: str | Path | None = None,
) -> np.ndarray:
    """Load a ground-truth ``.npy`` from a local path OR an ``s3://`` URI.

    This is the N2 loader the Runtime entrypoint uses for ``y_true_path``:
    when the operator passes an ``s3://`` URI we download it first, then load.

    FIX-19b (Subtask-13): a CLASSIFICATION target of STRING class labels (e.g. string-label classification's
    ``fault_class`` = normal/mixed_fault_1/…) is staged as an OBJECT-dtype ``.npy`` (staging
    writes ``test[target].to_numpy()`` — string labels → object array). A plain
    ``np.load(allow_pickle=False)`` REFUSES object arrays, so the Arbiter's y_true load raised
    ``ValueError: Object arrays cannot be loaded when allow_pickle=False`` BEFORE grading
    (string-label classification run #5 on v7: terminal status=failed, iterations=0 — y_true loads before y_pred,
    so it fired first). We route through the SAME SECURITY-restricted ``_load_npy_secure`` used
    for y_pred (FIX-19): non-object arrays load with allow_pickle=False; object-dtype label
    vectors are recovered via a restricted unpickler (numpy reconstruction + Python scalars
    ONLY) — arbitrary pickle is NEVER executed.
    """
    from .sandbox_staging import _load_npy_secure  # lazy: avoid import cycle at module load

    if path_or_uri.startswith("s3://"):
        filename = Path(parse_s3_uri(path_or_uri)[1]).name
        if download_dir is not None:
            dest = Path(download_dir) / filename
            _download_s3_to(path_or_uri, dest, s3_client=s3_client, region=region)
            return np.asarray(_load_npy_secure(dest))

        with tempfile.TemporaryDirectory(prefix="amlas-y-true-") as temp_dir:
            dest = Path(temp_dir) / filename
            _download_s3_to(path_or_uri, dest, s3_client=s3_client, region=region)
            return np.asarray(_load_npy_secure(dest))
    return np.asarray(_load_npy_secure(Path(path_or_uri)))


# ---------------------------------------------------------------------------
# Dataset staging (N1 + N2)
# ---------------------------------------------------------------------------


def _load_frame(local_csv: Path, target: str) -> tuple[pd.DataFrame, str]:
    """Load a CSV into a frame with a concrete target column present.

    Two cases:
      * AI4I golden — the raw failure-mode flag columns are present; we collapse
        them into the single multi-class ``failure_mode`` target via the shared
        golden handler (so the Arbiter's known baseline applies).
      * Direct target — the target column already exists in the CSV; we normalise
        headers but otherwise keep the column as-is.
    """
    raw_cols = set(pd.read_csv(local_csv, nrows=0).columns)
    if _AI4I_MODE_COLS & raw_cols:
        # AI4I path: load_ai4i normalises headers + collapses modes → failure_mode.
        df = load_ai4i(local_csv)
        return df, AI4I_TARGET

    df = pd.read_csv(local_csv)
    # Normalise headers the same way load_ai4i does so the resolved target matches.
    df.columns = [
        c.strip().lower().replace(" ", "_").replace("[", "").replace("]", "")
        for c in df.columns
    ]
    norm_target = target.strip().lower().replace(" ", "_").replace("[", "").replace("]", "")
    if norm_target not in df.columns:
        raise StagingError(
            f"target column {target!r} (normalised {norm_target!r}) not found in "
            f"{local_csv}; available: {sorted(df.columns)}"
        )
    return df, norm_target


def _resolve_local_csv(
    golden: str,
    *,
    dest_dir: Path,
    s3_client: Any | None,
    region: str,
) -> tuple[Path, str]:
    """Return a concrete local CSV path for the spec's golden_dataset + its source tag."""
    if golden.startswith("s3://"):
        local = dest_dir / "golden_source.csv"
        _download_s3_to(golden, local, s3_client=s3_client, region=region)
        return local, "s3"

    # Local path: support absolute, ./relative, and bare relative.
    p = Path(golden[2:]) if golden.startswith("./") else Path(golden)
    if not p.exists():
        raise StagingError(f"local golden_dataset not found: {golden} (resolved {p})")
    return p, "local"


def stage_dataset(
    spec: SpecModel,
    *,
    dest_dir: str | Path | None = None,
    s3_client: Any | None = None,
    region: str | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    max_rows: int | None = None,
) -> StagedData:
    """Resolve ``spec.golden_dataset`` to concrete train/test files + y_true.npy.

    Steps:
      1. Resolve the golden_dataset to a local CSV (download from S3 if needed).
      2. Load the frame, materializing the concrete target column (AI4I collapse
         or direct target).
      3. Optionally cap to ``max_rows`` for a cheap verification slice.
      4. Stratified train/test split (deterministic) — the SAME split that
         produces both the test CSV and the aligned y_true.npy.
      5. Write ``train.csv``, ``test.csv``, ``y_true.npy`` into ``dest_dir``.

    When ``dest_dir`` is omitted, a unique mode-0700 directory is allocated for
    the runtime session. Explicit destinations remain supported for controlled
    callers and tests.

    Returns the concrete paths + counts so the caller (``app.py``) can thread
    them into the Orchestrator problem card (N1) and use y_true for the
    Arbiter (N2).
    """
    dest = (
        Path(dest_dir)
        if dest_dir is not None
        else Path(tempfile.mkdtemp(prefix="amlas-data-"))
    )
    dest.mkdir(parents=True, exist_ok=True)

    local_csv, source = _resolve_local_csv(
        spec.golden_dataset, dest_dir=dest, s3_client=s3_client, region=region
    )
    df, target = _load_frame(local_csv, spec.target)

    if max_rows is not None and len(df) > max_rows:
        # Deterministic head slice keeps class structure roughly intact for a
        # tiny smoke run while capping spend; stratify still applies below.
        df = df.iloc[:max_rows].reset_index(drop=True)

    train, test = _split(
        df, target=target, test_size=test_size, random_state=random_state
    )

    train_path = dest / "train.csv"
    test_path = dest / "test.csv"
    y_true_path = dest / "y_true.npy"

    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    # N2: y_true is taken from the SAME held-out test split → 1:1 aligned with
    # the rows the ML Lead will predict on.
    np.save(y_true_path, test[target].to_numpy())

    return StagedData(
        train_path=str(train_path),
        test_path=str(test_path),
        y_true_path=str(y_true_path),
        target=target,
        n_train=len(train),
        n_test=len(test),
        source=source,
    )


def verify_staged_test_split(test_path: str, y_true_path: str) -> dict[str, Any]:
    """Is the test CSV still 1:1 with the ``y_true`` staged beside it?

    ``y_true`` is materialised ONCE, from ``test[target]`` at stage time, and never revisited. So
    any later write to ``test.csv`` silently invalidates it — and the agent DOES rewrite it: on a
    live run the FE step replaced the staged pair in place, taking the test split from 17,964 to
    17,939 rows while ``y_true`` kept 17,964. Nothing noticed until grading, which then reported a
    shape mismatch against the training entrypoint — blaming the component that had faithfully
    predicted every row of the file it was handed.

    Read-only and never raises: returns ``ok=False`` with both counts and the path so a caller can
    fail EARLY and accurately. Missing/unreadable inputs give ``ok=None`` (unknown), because "we
    could not check" must not masquerade as "we checked and it is fine".
    """
    out: dict[str, Any] = {
        "ok": None,
        "test_path": test_path,
        "y_true_path": y_true_path,
        "n_test_rows": None,
        "n_y_true": None,
        "detail": "",
    }
    try:
        n_y_true = len(np.load(y_true_path, allow_pickle=True))
        # ``nrows``-free count: read the column-free row count without materialising the frame.
        n_test = int(sum(1 for _ in open(test_path, encoding="utf-8")) - 1)
    except (OSError, ValueError) as e:
        out["detail"] = f"could not verify: {type(e).__name__}: {e}"
        return out

    out["n_test_rows"] = n_test
    out["n_y_true"] = n_y_true
    out["ok"] = n_test == n_y_true
    if out["ok"]:
        out["detail"] = f"test split and y_true agree at {n_test} rows"
    else:
        out["detail"] = (
            f"the staged test split has been rewritten: {test_path} now has {n_test} rows but "
            f"y_true ({y_true_path}) has {n_y_true}. y_true is materialised once at stage time and "
            "never regenerated, so grading would fail with an opaque shape mismatch. Write "
            "engineered features to NEW paths instead of overwriting the staged pair."
        )
    return out


def _split(
    df: pd.DataFrame, *, target: str, test_size: float, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified split when every class has >=2 members, else a plain split.

    Tiny verification slices can have singleton classes that break stratify;
    we fall back to a non-stratified split rather than fail the whole run.
    """
    from sklearn.model_selection import train_test_split

    stratify = df[target]
    can_stratify = stratify.value_counts().min() >= 2 and stratify.nunique() >= 2
    train, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify if can_stratify else None,
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)
