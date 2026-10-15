<!-- Source: ml-ralph/docs/PROMPTS.md "Data Verification Checklist" — ported verbatim.
     Also draws on MetaGPT CHECK_DATA_PROMPT. -->

# Data Verification Checklist (mandatory ritual)

Before any modeling, EDA beyond shape-check, or reporting, run the entire checklist on the current working DataFrame(s). **Paste the outputs into your response**. Do not proceed if any item fails.

1. **Shape + dtype** — print `df.shape` and `df.dtypes`. Expected types documented in the spec must match.
2. **Missingness** — print `df.isna().mean() * 100` sorted descending. Any column >5% missing is called out explicitly.
3. **Target distribution** — for classification, print `df[target].value_counts(normalize=True)`. For regression, print `df[target].describe()` plus a histogram summary (10 bins).
4. **Split integrity** — if train/test are provided, confirm: (a) no overlap on any declared primary key, (b) no near-identical feature rows across splits, (c) for time-based splits, confirm train timestamps all precede the cutoff.
5. **Target bleed** — for every non-target column, compute correlation with target (numeric) or contingency count (categorical). Flag any column with |corr| ≥ 0.99 or a 1:1 categorical mapping — these are *almost always* leakage.
6. **Leakage sentinel** — call the `leakage_check` tool. If it returns `blocking=True`, STOP and report the findings. Do not proceed to FE or modeling.
7. **Sanity print** — print `df.head(3)` and pick one suspicious-looking row; narrate why it looks correct or suspicious.
8. **Spec alignment** — confirm target name, primary metric, splits strategy, and constraints match `task_spec.md` exactly. Restate them in your response.

Failing any item = pipeline halt. Report the failing item and its evidence, then wait for a fix decision (human review or retreat to Research phase with this as the `abandoned_reason`).
