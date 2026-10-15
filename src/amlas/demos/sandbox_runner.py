"""Run AI4I FE + training inside the AgentCore Code Interpreter sandbox.

The live demo previously executed the training half inline in the host Python
process. Per ADR-0001 (managed-first), the actual compute belongs in the
sandbox. This module ships the train/test frames + a self-contained training
script over to the sandbox via `execute_python`, then parses the JSON the
script emits on stdout.

Runner is injectable so offline validation don't need AWS credentials.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd


class _RunnerResult(Protocol):
    """Shape returned by execute_python_tool — only the keys we read."""

    is_error: bool
    stdout: str
    stderr: str


Runner = Callable[..., dict[str, Any]]


class SandboxRunFailed(RuntimeError):
    """Sandbox returned an error or unparseable output."""


@dataclass(frozen=True)
class SandboxResult:
    winner: str
    candidates: list[dict[str, Any]]
    feature_manifest: list[dict[str, Any]]
    predictions: np.ndarray


_SENTINEL = "__SANDBOX_RESULT_JSON__"


_TRAINING_SCRIPT = f"""\
import base64, io, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

def _frame_from_b64(encoded):
    payload = base64.b64decode(encoded).decode("utf-8")
    return pd.read_json(io.StringIO(payload), orient="table")

train = _frame_from_b64(_TRAIN_B64)
test = _frame_from_b64(_TEST_B64)
target = _TARGET_COL

# Remap target labels to contiguous 0..K-1 (XGBoost multi:softprob requires it; the
# raw AI4I `failure_mode` set is {0,1,2,3,4,5} but a train slice may skip classes).
_unique = sorted(train[target].unique().tolist())
_label_map = {{v: i for i, v in enumerate(_unique)}}
_inv_label_map = {{i: v for v, i in _label_map.items()}}
train[target] = train[target].map(_label_map)
num_class = len(_unique)

# Feature engineering — CAAFE-style: propose, validate via 3-fold CV, keep if Δ ≥ 0.003.
def _cv_macro_f1(X, y):
    Xn = X.select_dtypes(include=[np.number]).fillna(0.0)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    for tr, va in cv.split(Xn, y):
        sc = StandardScaler()
        Xtr = sc.fit_transform(Xn.iloc[tr])
        Xva = sc.transform(Xn.iloc[va])
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        clf.fit(Xtr, y.iloc[tr])
        scores.append(f1_score(y.iloc[va], clf.predict(Xva), average="macro", zero_division=0))
    return float(np.mean(scores))

proposals = [
    ("wear_per_torque", lambda d: d["tool_wear_min"] / (d["torque_nm"].abs() + 1e-6)),
    ("temp_delta", lambda d: d["process_temperature_k"] - d["air_temperature_k"]),
    ("power_proxy", lambda d: d["torque_nm"] * d["rotational_speed_rpm"] / 9550.0),
    ("log_rpm", lambda d: np.log1p(d["rotational_speed_rpm"])),
    ("wear_sq", lambda d: d["tool_wear_min"] ** 2),
    ("type_onehot_H", lambda d: (d["type"] == "H").astype(int)),
    ("type_onehot_L", lambda d: (d["type"] == "L").astype(int)),
]

baseline_cols = [c for c in train.select_dtypes(include=[np.number]).columns if c != target]
kept = list(baseline_cols)
manifest = []
working = train.copy()
baseline = _cv_macro_f1(working[baseline_cols], working[target])
for name, expr in proposals:
    try:
        working[name] = expr(working).astype(float)
    except Exception as e:
        manifest.append({{"name": name, "decision": "drop", "reason": f"derivation_error: {{e}}"}})
        continue
    new_cols = kept + [name]
    new_score = _cv_macro_f1(working[new_cols], working[target])
    delta = new_score - baseline
    decision = "keep" if delta >= 0.003 else "drop"
    manifest.append({{"name": name, "decision": decision, "cv_delta": round(delta, 6), "cv_after": round(new_score, 4)}})
    if decision == "keep":
        kept.append(name); baseline = new_score
    else:
        working = working.drop(columns=[name])

# Materialize same features on test (only kept ones).
for rec in manifest:
    if rec["decision"] == "keep":
        for n, e in proposals:
            if n == rec["name"]:
                test[n] = e(test).astype(float)
                break

X_tr = working[kept]
y_tr = working[target]
X_te = test[kept]

# Shortlist — LogReg, RF, XGB (or GB fallback).
candidates = []
sc = StandardScaler().fit(X_tr.select_dtypes(include=[np.number]).fillna(0.0))
lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42).fit(
    sc.transform(X_tr.select_dtypes(include=[np.number]).fillna(0.0)), y_tr
)
candidates.append({{"name": "logistic_regression", "cv": _cv_macro_f1(X_tr, y_tr)}})
rf = RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample", random_state=42, n_jobs=-1)
rf.fit(X_tr.select_dtypes(include=[np.number]).fillna(0.0), y_tr)
rf_cv = _cv_macro_f1(X_tr, y_tr)
candidates.append({{"name": "random_forest", "cv": rf_cv}})

xgb_model = None
xgb_cv = -1.0
if HAS_XGB:
    _xgb_kwargs = dict(
        max_depth=6, learning_rate=0.1, n_estimators=200,
        random_state=42, n_jobs=-1, verbosity=0,
    )
    if num_class == 2:
        _xgb_kwargs.update(objective="binary:logistic", eval_metric="logloss")
    else:
        _xgb_kwargs.update(objective="multi:softprob", eval_metric="mlogloss", num_class=num_class)
    xgb = XGBClassifier(**_xgb_kwargs)
    xgb.fit(X_tr.select_dtypes(include=[np.number]).fillna(0.0), y_tr)
    xgb_cv = _cv_macro_f1(X_tr, y_tr)
    candidates.append({{"name": "xgboost", "cv": xgb_cv}})
    xgb_model = xgb

candidates.sort(key=lambda c: c["cv"], reverse=True)
winner = candidates[0]["name"]

X_te_num = X_te.select_dtypes(include=[np.number]).fillna(0.0)
if winner == "logistic_regression":
    y_pred = lr.predict(sc.transform(X_te_num))
elif winner == "random_forest":
    y_pred = rf.predict(X_te_num)
else:
    y_pred = xgb_model.predict(X_te_num)

y_pred = [_inv_label_map.get(int(v), int(v)) for v in y_pred]

result = {{
    "winner": winner,
    "candidates": candidates,
    "feature_manifest": manifest,
    "predictions": [int(v) for v in y_pred],
}}
print("{_SENTINEL}" + json.dumps(result))
"""


def _build_payload(train: pd.DataFrame, test: pd.DataFrame, target_col: str) -> str:
    """Encode frames with non-executable table-oriented JSON for sandbox transfer."""
    train_json = train.to_json(orient="table", date_format="iso").encode("utf-8")
    test_json = test.to_json(orient="table", date_format="iso").encode("utf-8")
    train_b64 = base64.b64encode(train_json).decode("ascii")
    test_b64 = base64.b64encode(test_json).decode("ascii")
    header = (
        f'_TRAIN_B64 = "{train_b64}"\n'
        f'_TEST_B64 = "{test_b64}"\n'
        f'_TARGET_COL = {target_col!r}\n'
    )
    return header + _TRAINING_SCRIPT


def _default_runner(code: str, *, description: str = "", region: str | None = None) -> dict[str, Any]:
    from ..tools.code_interpreter import execute_python

    r = execute_python(code, description=description, region=region)
    return {
        "is_error": r.is_error,  # nosemgrep
        "stdout": r.stdout,
        "stderr": r.stderr,
        "exit_code": r.exit_code,
        "execution_time": r.execution_time,
        "text": r.text,
    }


def train_ai4i_in_sandbox(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    target_col: str,
    runner: Runner | None = None,
    region: str | None = None,
) -> SandboxResult:
    """Run FE + training in the AgentCore CI sandbox; return a typed result.

    Args:
        train, test: AI4I-shaped DataFrames (columns already lowercased + snake_cased).
        target_col: target column name (e.g. "machine_failure").
        runner: callable matching `execute_python_tool` shape — takes (code, description, region)
            and returns a dict with keys {is_error, stdout, stderr, ...}. Default uses the
            real AgentCore sandbox; callers may inject a fake.
        region: AWS region for the sandbox session.

    Raises:
        SandboxRunFailed: sandbox errored, returned non-JSON output, or returned a
            predictions array of unexpected length.
    """
    runner = runner or _default_runner
    code = _build_payload(train, test, target_col)
    response = runner(code, description="ai4i FE + shortlist training", region=region)

    # Diagnostics are BOUNDED excerpts (500/300 chars) of sandbox output from the
    # demo's own generated code — synthetic demo data only, never customer data.
    if response.get("is_error"):
        raise SandboxRunFailed(
            f"sandbox returned is_error=True; stderr={response.get('stderr', '')[:500]}"
        )

    stdout = response.get("stdout", "")
    line = next((ln for ln in stdout.splitlines() if ln.startswith(_SENTINEL)), None)
    if line is None:
        raise SandboxRunFailed(
            f"could not parse sandbox stdout; expected line starting with {_SENTINEL!r}; "
            f"got: {stdout[:300]!r}"
        )
    try:
        payload = json.loads(line[len(_SENTINEL):])
    except json.JSONDecodeError as e:
        raise SandboxRunFailed(f"could not parse sandbox JSON: {e}") from e

    preds = np.asarray(payload["predictions"])
    if preds.shape != (len(test),):
        raise SandboxRunFailed(
            f"predictions length {preds.shape[0]} does not match test rows {len(test)}"
        )
    return SandboxResult(
        winner=payload["winner"],
        candidates=payload.get("candidates", []),
        feature_manifest=payload.get("feature_manifest", []),
        predictions=preds,
    )
