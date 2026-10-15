"""N1.b — bridge staged data into the AgentCore Code Interpreter sandbox.

The first real Phase-14b invoke revealed two filesystems in the runtime:
  * the runtime PROCESS fs — where `data_integrity`'s pandas tools read
    (cloud.staging writes here → D1 data-integrity PASSED), and
  * the Code Interpreter SANDBOX fs — where `execute_python_tool` runs for
    tabular_fe / trainer_tuner (isolated, no network).

cloud.staging only writes the process fs, so the sandbox saw "no data file".
This module copies the staged CSVs INTO the sandbox at the SAME absolute paths
by running base64-embedded writes through the HOISTED Code Interpreter session
([`code_interpreter.execute_python`]); because that session persists, every
later `execute_python_tool` call by the agents sees the files.

Imports `tools.code_interpreter` (strands-free) only — directly verifiable via its
`configure_session_factory` fake-kernel hook.
"""

from __future__ import annotations

import base64
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

# SEC-01: region defaults sourced from the single canonical region constant
# (config.runtime.DEFAULT_REGION) rather than re-declared "us-west-2" literals.
from .staging import parse_s3_uri

# Sentinel markers wrapping the base64 payload printed by the sandbox so the
# process side can robustly slice it out of (possibly noisy) stdout. N5 egress.
EGRESS_BEGIN = "AMLAS_EGRESS_BEGIN"
EGRESS_END = "AMLAS_EGRESS_END"

# Header tokens that indicate a CSV's first single-column row is a column name
# (NOT a class label), so load_predictions can drop it. Class labels like
# "TWF"/"HDF"/"none" are deliberately excluded — only generic prediction-column
# names appear here.
_PRED_HEADER_TOKENS = frozenset(
    {
        "prediction",
        "predictions",
        "pred",
        "preds",
        "y_pred",
        "ypred",
        "label",
        "labels",
        "target",
        "class",
        "classes",
        "yhat",
        "y_hat",
        "output",
    }
)


def build_write_file_code(target_path: str, content: bytes) -> str:
    """Return Python that writes ``content`` to ``target_path`` inside the sandbox.

    The bytes are base64-embedded so arbitrary CSV content (commas, quotes,
    newlines) survives transport through the executeCode payload unharmed.
    """
    b64 = base64.b64encode(content).decode("ascii")
    # Keep it dependency-free (os + base64 are always available in the kernel).
    return (
        "import os, base64\n"
        f"_p = {target_path!r}\n"
        "os.makedirs(os.path.dirname(_p), exist_ok=True)\n"
        f"_data = base64.b64decode({b64!r})\n"
        "with open(_p, 'wb') as _f:\n"
        "    _f.write(_data)\n"
        "print('WROTE', _p, len(_data))\n"
    )


def stage_files_into_sandbox(
    targets: dict[str, str],
    *,
    region: str | None = None,
) -> dict[str, bool]:
    """Write each ``{sandbox_target_path: local_source_path}`` into the sandbox.

    Uses the hoisted Code Interpreter session so all writes (and the agents'
    subsequent code) share one kernel/filesystem. Returns a per-target
    success map; a missing local source is marked ``False`` (never raises, so a
    partial-stage failure degrades into a normal D1 retreat rather than a crash).
    """
    from ..tools.code_interpreter import execute_python, register_staged_file  # strands-free

    results: dict[str, bool] = {}
    for target_path, source_path in targets.items():
        src = Path(source_path)
        if not src.exists():
            results[target_path] = False
            continue
        content = src.read_bytes()
        code = build_write_file_code(target_path, content)
        res: Any = execute_python(
            code, description=f"stage {target_path} into sandbox", region=region
        )
        ok = not bool(getattr(res, "is_error", False))
        results[target_path] = ok
        if ok:
            # FIX-22: remember this input so ``_invoke_with_reprovision`` can re-stage it into a
            # FRESH kernel after a mid-run reprovision (whose /tmp is empty) — otherwise the
            # trainer's later ``pd.read_csv`` FileNotFounds and no y_pred is ever produced.
            register_staged_file(target_path, str(src))
    return results


# ---------------------------------------------------------------------------
# N5 — EGRESS: bridge files OUT of the sandbox back to the process fs.
#
# The reverse of N1.b. The ML Lead's trainer_tuner writes y_pred INSIDE the
# Code Interpreter SANDBOX fs (via execute_python_tool); the Orchestrator's
# Arbiter step reads y_pred from the runtime PROCESS fs. Two filesystems →
# FileNotFoundError (CW 2026-05-29 14:04:44 /tmp/amlas/y_pred_test.csv). We
# read the sandbox file through the SAME hoisted session (so it sees what the
# agents wrote), base64-encode it to stdout between sentinels, then decode +
# write it to the process fs at the requested path.
# ---------------------------------------------------------------------------


def build_read_file_code(sandbox_path: str) -> str:
    """Return Python that base64-prints ``sandbox_path``'s bytes between sentinels.

    Run inside the hoisted Code Interpreter session so it reads what the agents
    actually wrote. If the file is absent the ``open`` raises → CodeExecResult
    ``is_error=True`` and the caller returns ``False`` (graceful, never crashes
    the run).
    """
    return (
        "import base64\n"
        f"_p = {sandbox_path!r}\n"
        "with open(_p, 'rb') as _f:\n"
        "    _b = _f.read()\n"
        f"print({EGRESS_BEGIN!r})\n"
        "print(base64.b64encode(_b).decode('ascii'))\n"
        f"print({EGRESS_END!r})\n"
    )


def _slice_between(text: str, begin: str, end: str) -> str | None:
    """Return the whitespace-stripped payload between ``begin`` and ``end``."""
    i = text.find(begin)
    if i == -1:
        return None
    j = text.find(end, i + len(begin))
    if j == -1:
        return None
    payload = text[i + len(begin) : j]
    # base64 from b64encode().decode() is single-line, but strip ALL whitespace
    # to survive any kernel-added newlines/indentation.
    return "".join(payload.split())


def egress_file_from_sandbox(
    sandbox_path: str,
    dest_path: str,
    *,
    region: str | None = None,
) -> bool:
    """Copy ``sandbox_path`` (Code Interpreter fs) → ``dest_path`` (process fs).

    Returns ``True`` on success, ``False`` if the sandbox file is missing, the
    execute failed, or no base64 payload was recovered. Never raises — a failed
    egress degrades into a normal "missing y_pred" retreat at the Arbiter.
    """
    from ..tools.code_interpreter import execute_python  # strands-free

    code = build_read_file_code(sandbox_path)
    res: Any = execute_python(
        code, description=f"egress {sandbox_path} from sandbox", region=region
    )
    if bool(getattr(res, "is_error", False)):
        return False
    # Prefer the structured stdout; fall back to joined content text.
    haystack = getattr(res, "stdout", "") or getattr(res, "text", "") or ""
    b64 = _slice_between(haystack, EGRESS_BEGIN, EGRESS_END)
    if b64 is None:
        return False
    try:
        data = base64.b64decode(b64)
    except Exception:
        return False
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return True


def _coerce_labels(values: list[str]) -> np.ndarray:
    """Coerce string cells to int → float → str (np.array), matching np.load shape."""
    try:
        return np.array([int(v) for v in values])
    except ValueError:
        pass
    try:
        return np.array([float(v) for v in values])
    except ValueError:
        pass
    return np.array(values)


def _load_csv_predictions(path: Path) -> np.ndarray:
    """Parse a predictions CSV robustly across the shapes agents emit.

    Handles: single column with a generic header ("prediction\\nTWF\\n..."),
    single column with no header ("0\\n1\\n..."), and the pandas
    ``Series.to_csv()`` index,value form (",0\\n0,TWF\\n...").
    """
    raw = [ln for ln in path.read_text().splitlines() if ln.strip() != ""]
    if not raw:
        return np.array([])
    rows = [ln.split(",") for ln in raw]
    ncols = max(len(r) for r in rows)
    if ncols >= 2:
        # index,value (pandas Series.to_csv): first row is the header → drop it,
        # take the value (last) column.
        values = [r[-1].strip() for r in rows[1:]]
    else:
        first = rows[0][0].strip()
        if first.lower() in _PRED_HEADER_TOKENS:
            values = [r[0].strip() for r in rows[1:]]
        else:
            values = [r[0].strip() for r in rows]
    return _coerce_labels(values)


def load_predictions(
    path: str,
    *,
    egress: bool = True,
    region: str | None = None,
) -> np.ndarray:
    """Load y_pred for the Arbiter — format-agnostic, with sandbox egress (N5).

    1. If ``path`` is absent on the PROCESS fs and ``egress`` is True, bridge it
       OUT of the Code Interpreter SANDBOX to the same path first (N5).
    2. Load ``.npy`` via ``np.load(allow_pickle=False)``; ``.csv``/``.txt`` via
       the robust CSV parser; otherwise try ``.npy`` then fall back to CSV.

    Raises ``FileNotFoundError`` if the file cannot be obtained.
    """
    # FIX-36 (ADR-0002 Phase 6): an ``s3://`` y_pred URI is the OFF-BOX dispatch path (SageMaker).
    # The agent reports ``s3://…/<job>/output/y_pred_test.npy``, but the SageMaker FRAMEWORK
    # container tars SM_OUTPUT_DATA_DIR into ``…/output/output.tar.gz`` — so the loose object 404s.
    # ``resolve_sagemaker_output_artifact`` bridges BOTH layouts (loose object OR extract from the
    # output tarball) into a local temp file the Arbiter can load. This is what closes the graded
    # round-trip for the dispatch architecture (previously: FileNotFoundError → no arbiter_report).
    if path.startswith("s3://"):
        from .staging import resolve_sagemaker_output_artifact

        with tempfile.TemporaryDirectory(prefix="amlas-y-pred-") as _td:
            local = Path(_td) / Path(parse_s3_uri(path)[1]).name
            resolved = resolve_sagemaker_output_artifact(path, dest=local, region=region)
            if resolved is None or not resolved.exists():
                raise FileNotFoundError(path)
            suffix = resolved.suffix.lower()
            if suffix == ".npy":
                return _load_npy_secure(resolved)
            if suffix in (".csv", ".txt"):
                return _load_csv_predictions(resolved)
            try:
                return _load_npy_secure(resolved)
            except Exception:
                return _load_csv_predictions(resolved)

    p = Path(path)
    if not p.exists():
        if egress and egress_file_from_sandbox(path, path, region=region) and p.exists():
            pass
        else:
            raise FileNotFoundError(path)
    suffix = p.suffix.lower()
    if suffix == ".npy":
        return _load_npy_secure(p)
    if suffix in (".csv", ".txt"):
        return _load_csv_predictions(p)
    try:
        return _load_npy_secure(p)
    except Exception:
        return _load_csv_predictions(p)


# ---------------------------------------------------------------------------
# FIX-19 (Subtask-13) — secure recovery of object-dtype STRING-LABEL y_pred .npy.
#
# The hard 7-class run trained a real model (CV f1_macro=0.80) and wrote y_pred as an
# OBJECT-dtype .npy of string class labels ("normal"/"mixed_fault_1"/...). numpy stores
# object arrays via pickle, so ``np.load(allow_pickle=False)`` raises "Object arrays
# cannot be loaded when allow_pickle=False" — the Arbiter could not load y_pred and the
# run failed with no grade (G-METRIC miss). Prior missions used integer labels so this
# latent defect never fired. We must recover such predictions WITHOUT enabling arbitrary
# pickle (SEC-A15 / model-artifact posture): a RESTRICTED unpickler that resolves ONLY
# numpy's array-reconstruction primitives + Python scalar types refuses any other class
# (e.g. os.system) → hostile payloads raise, they never execute.
# ---------------------------------------------------------------------------

# The exact set of globals a legitimate numpy object-array pickle needs to rebuild a 1-D
# array of Python str/int/float/bool. Anything else (os.system, subprocess, builtins.eval,
# arbitrary classes) is REFUSED by _RestrictedUnpickler.find_class.
_ALLOWED_PICKLE_GLOBALS: frozenset[tuple[str, str]] = frozenset(
    {
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy._core.multiarray", "_reconstruct"),  # numpy>=2 module path
        ("numpy._core.multiarray", "scalar"),
        ("builtins", "str"),
        ("builtins", "int"),
        ("builtins", "float"),
        ("builtins", "bool"),
        ("builtins", "bytes"),
        ("builtins", "list"),
        ("builtins", "tuple"),
    }
)


def _load_npy_secure(p: Path) -> np.ndarray:
    """Load a ``.npy`` with ``allow_pickle=False``; on an object-array ValueError, recover
    the label vector via a SECURITY-RESTRICTED unpickler (numpy reconstruction + scalars
    ONLY). Never enables arbitrary pickle — hostile payloads are refused, not executed."""
    try:
        return np.asarray(np.load(p, allow_pickle=False))
    except ValueError as e:
        if "allow_pickle" not in str(e) and "Object arrays" not in str(e):
            raise
        # Object-dtype array (string/mixed labels). Recover via a restricted unpickler that
        # ONLY resolves numpy array reconstruction + Python scalars. This is the same
        # trust-narrowing pattern skops uses for model artifacts — pickle is NEVER opened
        # to arbitrary classes.
        import pickle  # nosec B403

        class _RestrictedUnpickler(pickle.Unpickler):
            def find_class(self, module: str, name: str) -> Any:
                if (module, name) in _ALLOWED_PICKLE_GLOBALS:
                    return super().find_class(module, name)
                raise pickle.UnpicklingError(  # nosemgrep: avoid-pickle
                    f"refused disallowed global for y_pred recovery: {module}.{name}"
                )

        # numpy .npy layout: 6-byte magic \x93NUMPY, 2-byte version, header-len, header,
        # then the pickled object-array body. numpy's own reader parses the (untrusted-safe)
        # header; we only feed the pickle BODY to the restricted unpickler. Using numpy's
        # format module keeps header parsing correct across .npy versions (v1.0/v2.0 use
        # different header readers).
        from numpy.lib import format as _npformat

        with open(p, "rb") as f:
            major, minor = _npformat.read_magic(f)
            if (major, minor) >= (2, 0):
                shape, _fortran, dtype = _npformat.read_array_header_2_0(f)
            else:
                shape, _fortran, dtype = _npformat.read_array_header_1_0(f)
            if not dtype.hasobject:
                # Non-object dtype that still tripped allow_pickle (shouldn't happen) — re-raise.
                raise
            # The remainder of the file is the pickled object array body.
            arr = _RestrictedUnpickler(f).load()
            out = np.asarray(arr)
            # Defensive: enforce the header shape (a mismatch means a malformed file).
            if out.shape != shape and out.size == int(np.prod(shape or (out.size,))):
                out = out.reshape(shape)
            return out
