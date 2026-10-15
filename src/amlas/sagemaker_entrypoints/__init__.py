"""SageMaker-side training/processing entrypoints (ADR-0002).

These modules run INSIDE SageMaker Training/Processing jobs — the off-box compute the
orchestrator agents DISPATCH to. They are the data-plane counterpart to the control-plane
agents: the agent decides algo/params/dataset and launches the job; the entrypoint here does
the actual fit/predict and writes the canonical artifacts (e.g. ``y_pred_test.npy``) the Arbiter
grades. Kept import-light and directly verifiable (pure ``run_training``; a thin SageMaker env shim).
"""
