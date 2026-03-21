**Unified Design**

The clean resolution is: make budget a **service rate**, not a hard stop. In Sutra, the thing that spends service is not “more identical recurrent steps”; it is **finer spectrum-memory zoom** plus extra reroute/write passes. That makes the unified design:

\[
\text{easy token} \Rightarrow \text{coarse read} \Rightarrow \text{cheap, fast}
\]
\[
\text{hard token} \Rightarrow \text{verify fails} \Rightarrow \text{zoom finer / reroute again} \Rightarrow \text{slower, not worse}
\]

This fits the repo state directly: [launch_v054.py](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/launch_v054.py) still uses a fixed recurrent loop and returns zero compute cost, while [spectrum_memory.py](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/spectrum_memory.py) already exposes the right continuous zoom variable \(s_i\), and [research/STAGE_ANALYSIS.md](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/research/STAGE_ANALYSIS.md) already wants Stage 7 failure to loop back to Stage 4.

1. **Mathematical system**

For token \(i\) at refinement step \(t\), maintain
\[
z_{i,t}=(\mu_{i,t}, \lambda_{i,t}, \pi_{i,t}, s_{i,t}, q_{i,t})
\]
with \(s_{i,t}\in[0,1]\) the spectrum-memory zoom, where larger \(s\) means coarser/gist.

Per-step cost:
\[
c_{i,t}=c_{\text{base}}+c_{\text{verify}}+c_{\text{route}}\psi(k_{i,t})+c_{\text{zoom}}\phi(s_{i,t})
\]
with \(\phi(s)\) decreasing, e.g.
\[
\phi(s)=(1-s)^2
\]
or the tree version from your prior spec:
\[
\phi(s)=\sum_l \kappa_l(s)\,2^{-l}
\]

Verifier:
\[
q_{i,t}=\sigma(f_{\text{verify}}(\mu,\log \mathrm{var},\pi,s,\text{entropy},\text{margin}))
\approx e^{-CE_{i,t}}
\]

Acceptance threshold:
\[
\tau(G)=\tau_{\min}+(\tau_{\max}-\tau_{\min})\left(\frac{G}{G_{\text{ref}}}\right)^\gamma
\]

Control law:
\[
q_{i,t}\ge\tau(G)\Rightarrow \text{emit/freeze}
\]
\[
q_{i,t}<\tau(G),\ G\ge c_{\min}\Rightarrow \text{refine now}
\]
\[
q_{i,t}<\tau(G),\ G<c_{\min}\Rightarrow \text{wait/refill, then refine}
\]

Refine means:
\[
s_{i,t+1}=s_{i,t}-\Delta s_i,\quad \pi_{i,t+1}\to \text{Stage 4},\quad k_{i,t+1}\uparrow
\]

2. **Why this avoids both failures**

It avoids uniform mediocrity because easy tokens stop at coarse \(s\) and cheap compute.

It avoids the hard ceiling because “budget exhausted” no longer means “emit garbage”; it means “pause until more credits exist, then keep refining.” The penalty is latency.

3. **Graceful-degradation proof sketch**

There is one subtlety: with a **fixed finite sequence budget**, literal “unbounded per-token compute” is impossible. The best you can get is “no fixed local cap; a hard token may consume up to the whole sequence budget.”

To get **bounded total compute and truly unbounded per-token compute**, budget must be a **rate-limited tank**:

\[
G(\tau+\Delta \tau)=\min(G_{\max},\,G(\tau)+r\Delta\tau-\sum_i u_i(\tau)\Delta\tau)
\]

Then by construction:
\[
C(\tau)\le G(0)+r\tau
\]

So total compute up to wall-clock time \(\tau\) is bounded, but any one token can consume arbitrary extra compute if you allow arbitrary wait time. That is exactly “cost is latency, not failure.”

My recommendation is a 2-level design:
- Short-term elastic pool inside the sequence.
- Time-based refill fallback when verify still says “not enough.”

4. **Why spectrum memory is the elastic mechanism**

This is the key unification: compute is not a separate halting module sitting beside memory. Compute is **how far down the memory spectrum you go**.

- Coarse gist: scratchpad/root, cheap.
- Mid-scale: chunk/span summaries, moderate.
- Fine exact retrieval: leaves, expensive.

So Stage 7 failure should not just say “do another generic recurrent pass.” It should say:
\[
\text{not enough} \Rightarrow s_i \downarrow
\]
That makes the budget set the **default zoom**, not the maximum zoom.

5. **Stage 6 / Stage 7 changes**

Current contract in [research/STAGE_ANALYSIS.md](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/research/STAGE_ANALYSIS.md) says Stage 6 outputs `continue_mask` and Stage 7 outputs `verify_score`. That is too weak for this design.

Revised Stage 6 output:
\[
\text{control}_i=(\text{active}, s_i^{\text{target}}, k_i^{\text{target}}, \text{credit\_needed})
\]

Revised Stage 7 output:
\[
\text{verify}_i=(q_i,\ \text{fail\_type},\ \Delta s_i)
\]

Stage 7 should estimate residual distortion only. Stage 6 should apply the deterministic controller. That preserves the good part of your previous budget note while removing its bad failure mode: budget exhaustion should no longer force emission.

6. **Novelty, as of March 21, 2026**

Nearest priors I found are ACT (2016), Universal Transformer with dynamic halting (2018), Adaptive Attention Span (2019), PonderNet (2021), prompt-level compute-optimal test-time scaling (2024-2025), and PHOTON’s hierarchical multi-resolution LM (Dec 2025 / Jan 2026).

What they miss relative to this design:
- ACT / PonderNet / UT learn **halting depth**. Your design learns **quality**, then uses a deterministic priced controller.
- Adaptive Attention Span learns context span, but not a verify-triggered elastic refill law.
- Snell 2024 / DORA 2025 allocate compute across prompts or rollouts, not inside a token’s recurrent memory-resolution loop.
- PHOTON is multi-resolution memory, but not Stage-7-driven per-token “zoom finer until verified.”

My inference: the **pieces are not novel**, but the **combination is probably novel enough to be publishable** if you show:
- rate-distortion framing,
- verifier-calibrated elastic refill,
- per-token specialization of zoom/compute,
- better quality-compute frontier than fixed-depth or hard-budget baselines.

7. **Minimal experiment**

Do not start with open-ended language modeling. Start with a synthetic mixed-resolution next-token task where token difficulty is known.

Use three token classes in the same sequence:
- local-easy tokens: predictable from last 1-2 tokens
- gist tokens: require chunk/topic summary
- exact tokens: require copying a precise symbol/string 100-300 tokens back

Train a warm-started [launch_v054.py](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/launch_v054.py) variant with:
- per-step logits
- `VerifyHead`
- token freezing
- budget tank
- budget-aware [spectrum_memory.py](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/spectrum_memory.py)

Measure:
- accuracy by token class
- spent compute \(C_i\) by token class
- chosen scale \(s_i\) by token class
- \(I(\text{token type}; C_i)\)
- \(I(\text{token type}; s_i)\)
- matched-average-budget comparison vs fixed 8-step baseline

Success is not just higher accuracy. Success is **specialization**:
- easy tokens stay coarse and cheap
- exact tokens zoom finer and spend more
- average compute stays flat or nearly flat

8. **Score**

- Coherence: `9/10`
- Novelty: `7.5/10`
- Tractability: `7/10`
- Breakthrough upside: `9/10`
- Overall: `8.2/10`

The main risk is verifier calibration. If \(q\) is wrong, the whole controller misprices compute. That is the first thing to test.

Sources for novelty check:
- ACT (2016): https://arxiv.org/abs/1603.08983
- Universal Transformer (2018): https://arxiv.org/abs/1807.03819
- Adaptive Attention Span (2019): https://aclanthology.org/P19-1032/
- PonderNet (2021): https://arxiv.org/abs/2107.05407
- Snell et al. test-time compute (2024): https://arxiv.org/abs/2408.03314
- DORA / Every Rollout Counts (NeurIPS 2025): https://arxiv.org/abs/2506.15707
- PHOTON (v2, Jan 8 2026): https://arxiv.org/abs/2512.20687

If you want, I can turn this into an exact Stage 6/7 contract rewrite and a concrete patch plan against [launch_v054.py](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/launch_v054.py) and [spectrum_memory.py](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/spectrum_memory.py).