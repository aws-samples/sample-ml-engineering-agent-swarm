---
project_id: amlas_ai4i_binary
task_type: classification
target: machine_failure
primary_metric: f1_macro
golden_dataset: s3://amlas-golden-${AWS_ACCOUNT_ID}-${AWS_REGION}/golden/ai4i_2020_v1.parquet
constraints:
  no_pii_features: true
  inference_latency_ms: 50
splits:
  strategy: stratified
baseline:
  # Binary "any failure vs no failure" is a much easier problem than 6-class failure-mode.
  # A class-weighted LogReg on raw numeric features hits ~0.78 F1-macro;
  # GBDT-default hits ~0.85. Acceptance gate is "beat 0.85 by ≥ 0.02".
  type: gradient_boosting_default
  f1_macro: 0.85
compute_budget:
  gpu_hours: 0
  trials: 50
  wall_clock_minutes: 30
  max_retreats: 3
acceptance:
  primary_metric_delta_vs_baseline: ">= 0.02"
  no_subgroup_regression: false
---

# Amlas Predictive Maintenance — Binary Failure Detector (AI4I 2020)

## Domain context

Same UCI AI4I 2020 dataset as the multiclass failure-mode project, but reframed
as a **binary classifier**: `machine_failure ∈ {0, 1}`. The goal is fleet-level
"any-failure-imminent" alerting before drilling down into specific failure
modes (TWF/HDF/PWF/OSF/RNF), so a single high-recall binary model is the right
deployment target for the first triage layer.

## Known pitfalls

- **Class imbalance**: ~3.4% positive rate. Stratified CV is mandatory; raw accuracy
  would let a constant-zero classifier hit 96.6% — meaningless.
- **`Product ID` leakage**: same as the multiclass spec — UDI is a row identifier
  and must be dropped.
- **Pseudo-failure noise**: `RNF` (random failure) is by construction unpredictable
  and contributes ~0.4% of positives. The model can't beat random on those rows;
  acceptance is graded on the rest.
- **Threshold choice**: F1-macro is the gate, but the deployment threshold should be
  picked from the precision–recall curve to hit the operations team's tolerable
  false-alarm rate (~1/day fleet-wide). Out of scope here; surface in the model card.

## Stakeholder notes

Same locomotive PdM team as the multiclass project. This binary classifier is the
first-line "page operations" model; the multiclass one is the second-line
"which mode is it" follow-on. Both ship together.

## Acceptance

A model is accepted iff `f1_macro` on the stratified holdout exceeds `0.85 + 0.02 = 0.87`,
and no subgroup (by `Type`) regresses below baseline once subgroup tracking is wired in v2.
