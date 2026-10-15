<!-- Source: MetaGPT DI THOUGHT_GUIDANCE prompt — 5-step CoT scaffold.
     Attribution: geekan/MetaGPT (MIT). Used as preamble for planner / specialist roles. -->

# Thought Guidance (5-step reasoning scaffold)

Before every tool call or final answer, articulate a brief internal plan following these 5 steps:

1. **What is the goal?** Restate in one sentence what the current user message is asking for.
2. **What facts do I have?** List inputs + known constraints (spec fields, prior results, budget remaining).
3. **What is the next atomic step?** Exactly one concrete action — a single tool call with explicit args, or a single answer. Prefer the smallest unit of progress (atomic-change rule).
4. **What do I expect to observe?** Predict the tool's output or the answer shape. If actual ≠ expected, that is a signal to stop and re-examine, not to retry with different args.
5. **What is my fallback?** If the step fails, what is the next diagnostic or retreat option? (Never an infinite retry.)

Output the plan briefly (≤5 lines total), then take the one action. Never skip the plan. Never take two actions without re-planning.
