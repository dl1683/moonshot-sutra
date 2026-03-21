I read the repo design history end-to-end: root and parent `CLAUDE.md`, [README.md](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/README.md), [research/VISION.md](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/research/VISION.md), [research/STAGE_ANALYSIS.md](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/research/STAGE_ANALYSIS.md), [research/SCRATCHPAD.md](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/research/SCRATCHPAD.md), [research/RESEARCH.md](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/research/RESEARCH.md), and the prior Codex design reviews in `results/`. I also logged this synthesis in [research/RESEARCH.md#L5118](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/research/RESEARCH.md#L5118).

Short verdict: this is the best unifying principle Sutra has had so far. The key refinement is that the controller should not estimate current confidence; it should estimate **marginal future distortion reduction per unit cost**. That directly matches the repo’s own unresolved issue in [results/codex_v060_design_review.md#L3](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/codex_v060_design_review.md#L3), where the prior verifier target was criticized for predicting `exp(-CE)` instead of stop-worthiness.

1. Measure gain as predicted `Delta CE` or `Delta log p(target)` after one more action. Entropy, margin, and hidden-state movement are auxiliary features, not the target. Hidden-state norm is too indirect.

2. It should replace the learned transition kernel as the primary controller. Keep a learned module, but use it to estimate `g_a` for actions/stages, then derive transitions from `softmax(beta * g_a / c_a)`, not from a free Markov matrix.

3. It unifies all three cleanly:
- verifier = “is more work worth it?”
- zoom controller = choose finer memory only when its gain/cost is high
- stage controller = flow probability mass toward the stage with highest gain/cost
- freeze = when no action has enough positive marginal gain

4. Yes. Mutual information is the cleanest connection: under log loss, expected CE reduction is expected information gain about the target. Fisher information is a weaker local proxy for “sensitivity” or “room for improvement,” but I would not make Fisher the main controller.

5. Yes, approximately. The principled form is a Gibbs policy:
`pi_next(a) ∝ exp(beta * (g_a - lambda c_a))`
That is the information-theoretic optimum for maximizing expected gain under a compute constraint. It is the right derivation for stage mass.

6. Minimal experiment: synthetic mixed-demand sequences with four token types:
- local-easy
- gist-needed
- exact-memory-needed
- extra-serial-compute-needed

Compare at matched average compute:
- fixed learned transition kernel
- confidence-based verifier
- gain-driven controller

Success is specialization plus better compute/quality frontier, not just raw accuracy.

7. Novelty: moderate-to-strong as a synthesis, weak as isolated ingredients. Closest priors are ACT, Universal Transformer, Adaptive Attention Span, PonderNet, active-inference/value-of-information control, and recent pondering/test-time compute papers. What seems new here is using **one marginal-gain signal** to govern stage flow, zoom, continuation, and freeze in a stage-superposition architecture.

8. Score as unifying principle: `9/10`.
- Unifying power: `9.5/10`
- Mathematical coherence: `8.5/10`
- Novelty: `7/10`
- Tractability: `6.5/10`
- Breakthrough upside: `8.5/10`

Two important caveats from the repo history:
- The verifier target must be marginal future gain, not present confidence: [results/codex_v060_design_review.md#L3](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/codex_v060_design_review.md#L3)
- Do not collapse all gain into memory zoom. Production evidence says deep recurrence itself has large marginal value; step 7 was the biggest single gain in the current run: [research/RESEARCH.md#L1777](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/research/RESEARCH.md#L1777)

So the exact principle I would commit to is:

`control = predicted marginal future distortion reduction per unit cost`

That is the first Sutra control law that actually reduces theoretical surface area instead of adding another head.

Sources:
- ACT: https://arxiv.org/abs/1603.08983
- Universal Transformer: https://arxiv.org/abs/1807.03819
- Adaptive Attention Span: https://aclanthology.org/P19-1032/
- PonderNet: https://arxiv.org/abs/2107.05407
- Information-theoretic active inference: https://proceedings.mlr.press/v2/williams07a.html
- Value of information in active inference/POMDPs: https://arxiv.org/abs/2408.06542
- PonderLM: https://arxiv.org/abs/2505.20674
- Change of Thought: https://arxiv.org/abs/2507.13569

If you want, the next useful step is to turn this into an exact `VerifyHead` target, action set, and ablation plan for `v0.6`.