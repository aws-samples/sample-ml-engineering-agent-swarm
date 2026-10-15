"""Pure heuristic NL-parsing helpers for spec generation — STRANDS-FREE.

These functions were originally inline in `spec_generation_tools.py`, which
imports `from strands import tool`. Phase 14b's client-side cloud-dispatch needs
the SAME heuristics on a plain operator machine (no `agents` extra installed),
so the pure logic lives here and `spec_generation_tools.py` re-imports it.

No `strands` / `bedrock_agentcore` imports — only stdlib + (optional) pandas at
call time. Keep this module dependency-light so both the Strands @tool wrapper
and the boto3-only cloud client can share one source of truth (DRY).
"""

from __future__ import annotations

import re

GENERATION_TASKS = frozenset({
    "text_generation", "code_generation", "language_modeling", "causal_lm",
})


def resolve_task_modality(problem_type: str, modality: str | None = None) -> str:
    """Infer an omitted modality while rejecting explicit generation contradictions."""
    task = problem_type.strip().lower().replace("-", "_")
    explicit = (modality or "").strip().lower()
    if task in GENERATION_TASKS:
        if explicit and explicit != "text":
            raise ValueError(f"{problem_type} requires text modality, got {modality!r}")
        return "text"
    return explicit or "tabular"


def _infer_task_type(text: str) -> str:
    """Infer task type from description keywords by DOMINANT signal (not first-match).

    Previously this returned ``regression`` on the FIRST regression keyword seen, so a single
    stray word could flip the whole spec — e.g. the session-history ledger line
    ``[tool] estimate_dispatch_cost`` (FIX-25's cost estimator) matched the regression keyword
    "estimate" and mis-typed a 7-class f1_macro CLASSIFICATION task as regression, which staged
    the wrong grader on the deployed runtime. We now COUNT matches from each family and pick the
    stronger; ties and no-signal default to classification (the safe/common default). Regression
    keywords are matched at WORD BOUNDARIES so a substring inside a tool name cannot trip them.
    """
    lower = text.lower()
    # Exclusions describe what the objective is not; they must not become
    # positive classification/regression signals.
    lower = re.sub(
        r"\b(?:not|never|without|rather\s+than)\s+(?:(?:a|an)\s+)?"
        r"(?:classification|classifier|regression|regressor)\b",
        "", lower,
    )
    generation = re.search(
        r"\btext[\s_-]+generation\b|\bcode[\s_-]+generation\b|"
        r"\blanguage[\s_-]+model(?:s|ing)?\b|\bcoding[\s_-]+models?\b",
        lower,
    )
    explicit_classification = re.search(r"\bclassif(?:y|ication|ier)\b", lower)
    explicit_regression = re.search(
        r"\b(?:forecast(?:ing)?|regression|rmse|mae)\b|"
        r"\bpredict\b.*?\b(?:value|price|amount)s?\b",
        lower,
    )
    if generation and not explicit_classification and not explicit_regression:
        return "text_generation"

    regression_keywords = [
        "forecast", "estimate", "predict.*value", "predict.*price",
        "predict.*amount", "regression", "continuous", "rmse", "mae",
        "mean squared", "mean absolute",
    ]
    classification_keywords = [
        "classify", "classification", "predict.*whether",
        "predict.*if", "predict.*which", "binary", "multiclass",
        "transported", "churn", "fraud", "spam", "label",
        "category", "f1", "accuracy", "precision", "recall",
        "auc", "roc",
    ]

    def _count(keywords: list[str]) -> int:
        n = 0
        for kw in keywords:
            # Bare-word keywords (no regex metachars) match at word boundaries so a substring
            # inside another token (e.g. "estimate" in "estimate_dispatch_cost") does not count.
            pattern = kw if any(c in kw for c in ".*+[](){}^$") else rf"\b{re.escape(kw)}\b"
            n += len(re.findall(pattern, lower))
        return n

    reg = _count(regression_keywords)
    clf = _count(classification_keywords)
    if reg > clf:
        return "regression"
    return "classification"


def _extract_target_column(text: str) -> str:
    """Extract target column name from description."""
    patterns = [
        r"target\s+(?:column\s+)?(?:is|variable|:)\s+['\"]([^'\"]+)['\"]",
        r"target\s*[:=]\s*['\"]([^'\"]+)['\"]",
        r"['\"]([^'\"]+)['\"]\s+(?:column|is the target|is our target)",
        r"target\s+(?:column\s+)?(?:is|:)\s+(\w+)",
        r"predict\s+(?:the\s+)?['\"]([^'\"]+)['\"]",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    quoted = re.findall(r"['\"]([A-Za-z_]\w*)['\"]", text)
    if quoted:
        return str(quoted[0])

    return "target"


def _extract_metric(text: str) -> str:
    """Extract primary metric from description."""
    lower = text.lower()

    metric_patterns = [
        (r"\bpass(?:@|[\s_-]*at[\s_-]*)1\b", "pass_at_1"),
        (r"\bperplexity\b", "perplexity"),
        (r"\bresolved[\s_-]*rate\b", "resolved_rate"),
        (r"f1[\s_-]*macro", "f1_macro"),
        (r"f1[\s_-]*weighted", "f1_weighted"),
        (r"f1[\s_-]*score|f1", "f1_macro"),
        (r"accuracy", "accuracy"),
        (r"auc|roc[\s_-]*auc", "roc_auc"),
        (r"precision", "precision"),
        (r"recall", "recall"),
        (r"rmse|root[\s_-]*mean[\s_-]*squared", "rmse"),
        (r"mae|mean[\s_-]*absolute", "mae"),
        (r"r2|r[\s_-]*squared|r_squared", "r2"),
        (r"mse|mean[\s_-]*squared[\s_-]*error", "mse"),
    ]

    for pattern, metric_name in metric_patterns:
        if re.search(pattern, lower):
            return metric_name

    task = _infer_task_type(text)
    if task == "text_generation":
        return ""
    return "f1_macro" if task == "classification" else "rmse"


def _extract_threshold(text: str) -> float | None:
    """Extract success threshold from description."""
    lower = text.lower()

    patterns = [
        r"(?:at\s+least|need|achieve|require|above|exceed|>=?)\s+(\d+(?:\.\d+)?)\s*%",
        r"(\d+(?:\.\d+)?)\s*%",
        r"(?:at\s+least|need|threshold|>=?)\s+0\.(\d+)",
        r"(?:below|under|less\s+than|<=?)\s+(\d+(?:\.\d+)?)",
    ]

    for pattern in patterns:
        match = re.search(pattern, lower)
        if match:
            value = float(match.group(1))
            if value > 1.0:
                return value / 100.0
            return value

    task = _infer_task_type(text)
    if task == "text_generation":
        return None
    return 0.80 if task == "classification" else 10.0


def _extract_title(text: str) -> str:
    """Generate a title from the description."""
    first_sentence = text.split(".")[0].strip()
    if len(first_sentence) > 80:
        first_sentence = first_sentence[:77] + "..."
    return first_sentence


def _validate_column_in_data(data_path: str, column: str) -> bool:
    """Check if a column exists in a CSV file."""
    try:
        import pandas as pd

        df = pd.read_csv(data_path, nrows=0)
        return column in df.columns
    except Exception:
        return False
