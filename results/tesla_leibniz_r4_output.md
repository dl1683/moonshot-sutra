**1. Assumption Updates**

- `Precise <-> general retrieval is now mandatory.` For: the token-type audit shows a 5.1x CE gap from whitespace to acronyms, with numbers/proper nouns/acronyms at 8.7%-18.8% top-1. Against: too many memory paths can fragment a small model. Lean: keep one controller and two learned paths in training, `scratchpad + ARMT`, with the exact episodic buffer as a zero-param runtime path. Confidence: `10/10`.

- `The scratchpad is not the memory solution.` For: it gives real BPT lift and remains useful for diffuse discourse state. Against: it is an 8-slot EMA workspace, not exact recall, and the failure classes are exactly the ones it should fail on. Lean: keep it as the general/workspace end of the spectrum only. Confidence: `9/10`.

- `Collapse is real, not just a metric artifact.` For: earlier “late-pass degradation” was a sampled-CE artifact. Against that correction: the new audit shows pass disagreement is effectively zero across all token classes, so the model is still collapsing even though late passes are valuable. Lean: the metric story changed; the collapse story did not. Confidence: `9/10`.

- `Random-depth is P0, not a nice-to-have.` For: current training gives almost all meaningful gradient to the last pass; random-depth fixes the gradient structure directly and is warm-start compatible. Against: an abrupt switch can shock a 12-pass checkpoint. Lean: do it first, but stage it with a deep-only warm start. Confidence: `9/10`.

- `12 passes is not the final doctrine.` For: truncation proves recurrence is doing real work in the current model. Against: the target should be `Dmax=8`; `12` is mostly a teacher/warm-start depth, not the production depth. Lean: keep `12` only for the current line, target `8` for the shared-core architecture. Confidence: `8/10`.

- `Explicit multi-mode processing survives, but only as cheap control.` For: the token audit and the survey both argue that different tokens need different processing regimes. Against: heavy per-mode compute at 68M wastes capacity. Lean: one shared content block, 4 explicit modes, top-2 routing, no multi-bank stage stack. Confidence: `9/10`.

- `ARMT is now the first precise parametric memory to build.` For: it is the best surveyed tradeoff on overhead, warm-startability, and sub-200M evidence. Against: it may help recall more than perplexity and can devolve into “latest fact wins.” Lean: use ARMT as the semi-precise associative sidecar, with an exact episodic buffer beside it. Confidence: `8/10`.

- `Benchmarks are not the early gate.` For: they matter at the end. Against: at this near-random scale, benchmark movement is too quantized to diagnose architecture; token-type recall and generation are more informative right now. Lean: early gates are `full-vocab BPT + token-type recall + generation quality`; benchmarks come after memory lands. Confidence: `8/10`.

- `The current front-end is an interim constraint, not a final design.` For: keeping GPT-2 vocab/embeddings preserves warm-start compatibility. Against: embeddings + learned positions consume 58.8% of the model. Lean: do not touch the front-end in the warm-start line; switch to factored embeddings + RoPE in the clean line only. Confidence: `9/10`.

**2. Research/Probe Requests For Claude To Execute**

- `Gap:` actual ARMT runtime cost on the 5090 is unknown. `Ask:` benchmark ARMT causal read/write at `H in {768,1024}`, `B in {1,4}`, `T in {256,512}`, bf16 and fp32 accumulators. `Decision:` keep ARMT if end-to-end overhead is `<15%` versus workspace-only.

- `Gap:` the best bootstrap path from 7 stages to 4 modes is unknown. `Ask:` map `{1,2,3}->compose`, `4->route`, `5->retrieve`, `{6,7}->verify`, then measure mutual information between that pseudo-label and token class/pass index on saved `pi_hist`. `Decision:` if MI is nontrivial, use pseudo-supervised controller warm-start.

- `Gap:` the exact-memory ceiling is still unmeasured. `Ask:` run an inference-only `kNN-LM` or FAISS sidecar on SciQ and LAMBADA with the current checkpoint. `Decision:` if exact external memory moves those tasks sharply, memory is the bottleneck and ARMT/episodic work is justified.

- `Gap:` episodic exact buffer design is underspecified. `Ask:` compare shared-vs-separate `Q/K/V` projections for the episodic buffer, and compare `M in {64,128,256}` entries. `Decision:` choose the smallest `M` that still lifts exact-token recall.

- `Gap:` mode collapse risk under top-2 routing is unknown. `Ask:` run a small-scale controller-only probe with and without `L_MI = H(mean pi) - E[H(pi)]`. `Decision:` keep MI only if it materially improves mode usage without hurting BPT.

- `Gap:` factorized front-end risk is unknown under recurrence. `Ask:` run a from-scratch canary with `32K BPE + E=192` and `32K BPE + E=256` on the winning core-lite family. `Decision:` promote `E=192` only if rare-token recall does not regress.

**3. Implementation Specs**

`3A. Pattern Findings`

- `Residual correction is the rhyme.` ARMT delta-rule memory, predictive coding, Titans-style surprise writing, and energy-based iterative refinement all say the same thing: do not rewrite belief; write only prediction error.

- `Zero-init + ramp is the growth law.` It appears in adaLN-Zero, function-preserving growth, DiT-style module insertion, and it is the correct way to add controller, ARMT, and shared-core replacements without destroying a working checkpoint.

- `Separate content from control.` The shared block should carry language competence. The controller should decide `what context to read` and `how to process`, not own a second copy of the content capacity.

- `Free energy / residual gain is the best unifier.` Retrieval, mode choice, and halting can all be phrased as “which action most reduces expected future error?” That is more implementable here than optimal transport or pure information geometry.

- `Biology rhymes too.` Scratchpad = cortical workspace, episodic buffer = hippocampal exact replay, ARMT = associative cortex, control simplex = basal ganglia / repertoire selection, halting = prediction-error quenching.

`3B. Target Architecture`

- Per token, keep `h_t^d`, `lambda_t^d`, and `pi_t^d in Delta^4`, where the 4 modes are `compose, route, retrieve, verify`. `lambda` remains a bounded update inertia term only; it is no longer the halting signal.

- One shared recurrent content block is repeated for `Dmax` passes. The block is `SwiGLU` plus pass+mode-conditioned `adaLN`; the controller is outside the block.

- Retrieval is a continuous blend:
```text
m_precise = alpha_epi * m_epi + (1 - alpha_epi) * m_armt
m_retr    = (1 - rho_precise) * m_workspace + rho_precise * m_precise
x_t^d     = h_t^d + pi_route * m_local + pi_retrieve * m_retr
```

- The conditioning vector is:
```text
c_t^d = pass_emb[d] + sum_k pi_tk^d * mode_emb[k] + b_seq^d
u_t^d = adaLN(x_t^d ; c_t^d)
y_t^d = SharedSwiGLU(u_t^d)
```

- Update state with the existing bounded writer first; later replace only if earned:
```text
h_t^{d+1}, lambda_t^{d+1} = BayesianWrite(h_t^d, lambda_t^d, y_t^d, g_write,t^d)
g_write,eff = g_write * (1 - 0.5 * pi_verify)
```
That preserves warm-start compatibility and already splits halting from writing.

`3C. Parameter Budgets`

- `Warm-start current-front-end line, recommended:` `V=50257`, `H=768`, learned positions kept, `F=4096`, current router kept initially. Counts:
  `embeddings 38.60M`, `positions 1.57M`, `shared core 9.45M`, `local router 4.72M`, `scratchpad 2.37M`, `BayesianWrite 2.36M`, `control simplex 2.47M`, `ARMT 2.36M`, `init 1.18M`, `halting head 0.10M`, `readout tied ~0M`. Total: `65.19M`. I would not force-fill the remaining ~`3.1M`; the current bottleneck is training geometry, not raw parameter count.

- `Control simplex math at H=768:` router `768->128->8` is `99,464` params; `pass_emb` is `9,216`; `mode_emb` is `3,072`; mode+pass adaLN generator `768->768->2304` is `2,362,368`. Total: `2,474,120`.

- `ARMT math at H=768:` `W_Q, W_K, W_V, W_beta`, each `768x768 + 768`, total `2,362,368`. Runtime state `A` is `768x768 = 589,824` numbers, about `1.18MB` per sequence in bf16.

- `Clean-line compute-matched recommendation:` `32K BPE`, `E=256`, `H=1024`, `RoPE`, `F=6144`, `Dmax=8`. Counts:
  `token table 8.19M`, `input proj 0.26M`, `shared core 18.89M`, `local mixer 4.21M`, `scratchpad 4.21M`, `BayesianWrite 4.20M`, `control simplex 4.35M`, `ARMT 4.20M`, `halting 0.13M`, `readout proj 0.26M`, `init 2.10M`. Total: `51.00M`. This is the safe first clean line because its train compute is close to the current 68M/12-pass model.

- `Clean-line full-budget stretch, only after the gate passes:` `E=256`, `H=1152`, `F=8064`, `Dmax=8`, `RoPE`. Total: `66.24M`.

`3D. Random-Depth Training`

- Use one sampled depth per microbatch, not per token. Training-time simplicity matters more than matching inference exactly.

- `Warm-start distribution on the current 12-pass line:` for the first `500` steps after the switch, sample `D in {8,9,10,11,12}` with `P(D=d) proportional to d^2`. After that, sample `D in {1..12}` with the same law. The averages are `10.39` then `9.36` passes, so the checkpoint is not shocked.

- `Target distribution on the final 8-pass line:` `P(D=d) = d^2 / sum_{j=1}^8 j^2`, average depth `6.35`.

- `Loss at P0:` `L = CE(logits_D, y)`. Set the current sampled `L_step` to zero; it is structurally biased because its negatives come from final-pass logits. Do not reintroduce intermediate full-vocab CE; that direction already collapsed the late-pass gains.

- `Loss after P0 works:` `L = CE_D + 0.02 * L_gain + lambda_compute * D / Dmax`, with `lambda_compute` ramped from `0` to `0.01` over `2K` steps. Train `L_gain` from occasional calibration batches that run one extra pass and target `||h^{d+1} - h^d||_2`.

- `Warm-start recipe:` load weights, reset optimizer moments, `lr=1e-4`, `300-500` warmup steps, `3K-5K` probe length.

`3E. Retrieval Spectrum And ARMT`

- Use three backends behind one controller interface:
  `workspace scratchpad` for diffuse state,
  `exact episodic buffer` for lossless in-sequence recall,
  `ARMT` for compact associative memory.

- The episodic buffer should be zero-param at first. Reuse the ARMT projections, store only top-scoring precise writes in a causal ring buffer. Recommended start: `M=128` entries, `k_dim=128`, value = hidden snapshot. Runtime cost at `H=768` is about `224KB` per sequence in bf16.

- ARMT should be `one sequence-level memory state`, persistent across passes and positions, reset at sequence boundary. Do not create separate ARMT states per pass.

- Read/write every pass:
```text
q = phi(W_Q h) , k = phi(W_K h) , v = W_V h , beta = sigmoid(W_beta h)
m_armt = A q / (z^T q + eps)
A <- A + g_armt * beta * (v - m_armt) outer k
z <- z + g_armt * beta * k
```
`g_armt = pi_retrieve * rho_precise * sigmoid(w_g^T h)`.

- Exact buffer write rule: write only if `pi_retrieve * rho_precise * novelty > tau`, where novelty can be `1 - cosine(h, m_armt)` or top-token entropy. Keep the highest-score `M` entries causally.

- Sequence order matters: `read memory -> shared block -> bounded writer -> memory write`. That lets early passes write and late passes exploit what was written.

`3F. Control Simplex`

- The controller is `per token, per pass`. A sequence-level prior `b_seq^d` is allowed, but only as an additive bias on token logits.

- Implementation:
```text
l_t^d   = W2 * SiLU(W1 * RMSNorm(h_t^d)) + b_seq^d
pi_t^d  = Top2(softmax(l_t^d / tau)), tau = 1.0
rho_t^d = sigmoid(w_rho^T RMSNorm(h_t^d))
alpha_epi,t^d = sigmoid(w_epi^T RMSNorm(h_t^d))
```

- Use deterministic top-2 projection, not straight-through, in the first implementation. It is simpler, lower-variance, and matches the existing `top2_project` logic. Only reach for ST-Gumbel if mode starvation persists after MI regularization.

- Pass identity should enter only through `pass_emb[d]` in the conditioning vector. Do not encode pass index as a second positional axis.

- Add `L_MI = -lambda_MI * (H(mean pi) - E[H(pi)])` only after `500` stabilization steps, with `lambda_MI=0.01`. The goal is “all modes used overall, each token decisive locally.”

- Mode semantics:
  `compose` = no extra context, just transform;
  `route` = inject local mixer;
  `retrieve` = inject retrieval blend;
  `verify` = same shared block, but with lower write tendency and stronger pressure for another pass if residual gain stays high.

`3G. Recurrence Gate Experiment Design`

- The current `v0.6.0a` truncation result is useful but not decisive. It proves recurrence helps in the current 7-stage model; it does not prove recurrence survives simplification.

- The real gate must be run on a `core-lite family` with the same front-end, the same retrieval modules, and the same controller ABI. Only `Dmax` and shared-block width change.

- Compare three siblings:
  `D=1` wide single-pass,
  `Dmax=4` random-depth recurrent,
  `Dmax=8` random-depth recurrent.

- Match compute by measured profiler output on the actual RTX 5090 laptop GPU, not by hand formulas. Tune the single-pass branch width until forward+backward wall time per token matches the recurrent branch within `+-5%`.

- Train all three from the same teacher snapshot or the same distilled initialization. Use the same tokenizer, data slice, optimizer, schedule, and evaluation cadence.

- Gate metrics:
  `full-vocab held-out BPT`,
  `token-type recall` on numbers/proper nouns/acronyms,
  `generation quality` with repetition-aware decoding,
  `average inference depth` once halting is enabled.

- Pass rule: recurrence survives only if the best `D>1` branch beats `D=1` by at least `0.10` BPT and at least `2pp` on exact-token recall at matched average compute. If it ties, recurrence has not earned the tax at this scale.

**4. Per-Outcome Confidence**

- Outcome 1 (Intelligence): `7/10` — the main failure mode is now concrete, the retrieval spectrum is directly supported by probes, and the shared-core/controller/memory spec is implementable; the end-to-end win is still unproven.

- Outcome 2 (Improvability): `8/10` — the split is now real: shared core, controller, workspace memory, exact memory, halting. Each can be added, swapped, or ablated surgically.

- Outcome 3 (Democratization): `7/10` — the ABI is much clearer now, especially around memory backends and controller behavior, but it is not frozen yet.

- Outcome 4 (Data Efficiency): `7/10` — module-local distillation, exact memory, and factorized embeddings are now concrete levers instead of vague hopes.

- Outcome 5 (Inference Efficiency): `6/10` — random-depth and gain-based halting have a clean recipe, but token-level retirement is not yet validated on a non-collapsed model.

**5. What Would Raise My Confidence**

- Outcome 1 (Intelligence): exact-token recall improves materially once ARMT + episodic memory land, and generation coherence improves without a BPT tradeoff.

- Outcome 2 (Improvability): a memory-only swap fixes number/entity recall without harming general text quality.

- Outcome 3 (Democratization): a domain-specific memory backend plugs in behind the same controller and composes cleanly.

- Outcome 4 (Data Efficiency): a narrow distillation path or exact-memory path reaches the same held-out quality in fewer tokens.

- Outcome 5 (Inference Efficiency): at least `40%` of tokens retire by pass `<=3` with negligible quality loss, and hard tokens still use the tail profitably.

**6. What Would Lower My Confidence**

- Outcome 1 (Intelligence): ARMT + episodic memory fail to move the exact-token buckets, or the shared core loses generation quality when the StageBank is removed.

- Outcome 2 (Improvability): gains only appear when retraining the whole model, and module-local interventions do not isolate failure modes.

- Outcome 3 (Democratization): the controller/memory ABI keeps changing, or different module improvements do not compose.

- Outcome 4 (Data Efficiency): module-local distillation acts as noise again, or exact memory helps recall but not useful downstream behavior.

- Outcome 5 (Inference Efficiency): easy tokens still need near-max depth, or memory reads erase the compute savings.

**7. Warm-Start Evolution Roadmap**

1. `v0.6.0b-rd` — trainer-only random-depth on the current checkpoint. Freeze: none. Steps: `3K-5K`. Success: final BPT stays within `0.05`, early-pass usefulness rises, late-pass share of total improvement drops below `70%`.

2. `v0.6.1-ctrl` — add the 4-mode controller + pass/mode adaLN with zero-init outputs. Freeze: old backbone for `500-1K` steps, then unfreeze all for `2K`. Bootstrap from the old 7-stage `pi` mapping. Success: each mode gets used, controller correlates with token class, BPT holds.

3. `v0.6.2-core` — add one shared `SwiGLU` block in parallel with `StageBank` behind a gate `alpha_core=0`. Freeze: old model, train only new block on hidden-state mimicry for `1K`; then ramp `alpha_core 0->1` over `2K`; then prune `StageBank`. Temporary params can rise to ~`80M`; that is acceptable. Success: StageBank can be deleted with minimal loss and better throughput.

4. `v0.6.3-route` — add a local-only mixer in parallel with the current router, with `alpha_route=0`. Train new mixer `1K`, ramp `1K-2K`, then delete pheromone and the full `T x T` path. Success: same or better quality with lower latency.

5. `v0.6.4-memory` — add ARMT and the exact episodic buffer zero-gated. Freeze: new memory params only for `1K`, then unfreeze all for `2K-3K`. Success: numbers/proper nouns/acronyms improve first; SciQ/LAMBADA move next.

6. `v0.6.5-halt` — add the gain head and compute penalty. Freeze: none. Steps: `2K`. Success: easy tokens begin retiring early under dynamic inference, with no meaningful quality loss.

7. `v0.7.0-gate` — run the decisive `D=1` vs `D=4` vs `D=8` core-lite family at matched measured compute. Use the same teacher snapshot or distilled init. Success: a recurrent branch wins on both BPT and exact-token recall.

8. `v0.8.0-clean` — only after step 7 passes, restart from scratch with `32K BPE + factored embeddings + RoPE` on the winning core. Start with the compute-matched `51M` clean line, not the full-budget stretch. Success: it reaches the warm-start line’s quality in fewer tokens and with a saner parameter allocation.