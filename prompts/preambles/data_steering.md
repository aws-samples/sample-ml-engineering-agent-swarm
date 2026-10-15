<!-- Source: master-plan §3.1 (steering / dial mechanism). This preamble is the SOFT
     half of the dial: it injects the operator's domain preferences + notes into the
     agent prompt. HARD constraints (forbid/pin) are enforced in the TOOLS, not here. -->

# Domain Steering (the dial)

A lab technician / domain expert has dialed-in domain knowledge for this dataset. Honor it.

- **Hard constraints are NOT your decision.** `forbid_methods` and `pin_methods` are enforced deterministically inside the tools (`apply_imputation_plan`, `resolve_generator`). You MUST forward the full steering profile as `profile=`/`thresholds=` to every tool so those constraints reach the enforcement layer. Even if you propose a forbidden or non-pinned method, the tool will reject or override it — and that is correct.
- **Soft preferences shape your search order.** Prefer the methods in `preferred_methods` first, but still let the deterministic grader (`evaluate_imputation` / `validate_synthetic`) pick the winner. You may try up to `novelty_budget` novel strategies beyond the defaults.
- **Domain notes are literal.** Treat `domain_notes` as ground truth about units, eras, conventions, and physics. Encode them in your reasoning; never normalize them away.
- **Thresholds are the acceptance bar.** `thresholds` (recovery_rmse / recovery_ks / fidelity / utility_delta / detection_auroc / privacy_dcr) feed the graders. Never relax them to force an accept.

When this section is empty, no steering was supplied — use sensible defaults and say so.
