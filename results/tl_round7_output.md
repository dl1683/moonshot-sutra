**1. Assumption Challenges**

Inherited paradigm audit first:

- `24x512` transformer budget is approximately `90.2M`: embeddings `8.19M` (`9.1%`), attention `25.17M` (`27.9%`), FFN `56.62M` (`62.7%`), exits/norms `0.26M` (`0.3%`).
- `24x512` projected hybrid budget is approximately `93.4M`: embeddings `8.19M` (`8.8%`), branch cores `15.75M` (`16.9%`), projection plumbing `12.58M` (`13.5%`), FFN `56.62M` (`60.6%`), exits/norms `0.26M` (`0.3%`).
- If “actual novel intelligence work” means the local+global mechanism, only about `16.9%` of the projected hybrid budget is doing it. `60.6%` is still inherited FFN mass, and `13.5%` is plumbing needed only because the branches were squeezed to `256->512`.
- Inherited decisions not re-derived for this scale: decoder-only NTP, BPE + dense embedding matrix, tied embeddings, uniform `24x512` stack, `3x` SwiGLU FFN, RoPE, `head_dim=64`, `4Q/2KV` GQA, real-valued Euclidean states, AdamW+WSD, fixed thirds exits. Strong local justification exists only for “small tokenizer beats GPT-2 50K” and “tied embeddings save params.”
- Top 3 inherited assumptions to change: `3x` FFN dominance, every-layer `256->512` branch compression, and the discrete tokenizer/embedding interface that saves params but blocks teacher transfer.
- Components consuming `>20%` of params: FFN only. That is the main redesign reservoir. Approximate code arithmetic says `24x512` full-dim hybrid is budget-viable if FFN is cut: full-dim `P` with `ff=1024` is about `90.3M`; full-dim `G` with `ff=1024` is about `99.7M`; full-dim `G` with `ff=1536` needs about `20` layers to stay near `100M`.

1. Projected branches remain the right `100M` family. Obvious: three hybrids showed early gains, so keep fixing fusion. Alternatives: the gain is real but unsustainable because projection is the failure channel; hybrid helps only early layers; `100M` is too small for every-layer projected branches. Evidence: `R5-G`, `R6-F`, and `R6-S` all fail at `100M`, while the stable hybrid is the no-projection `42M` P-block. Lean: projected every-layer hybrids are effectively dead at `100M` unless `R7-P` passes. Confidence: `9/10`.

2. Post-fusion normalization deserves one decisive probe. Obvious: it directly attacks fused-RMS collapse. Alternatives: it may only amplify cancellation residue; it may improve stability metrics while hurting signal. Evidence: telemetry shows branch magnitudes grow while fused RMS shrinks; `R7PostFusionHybridBlock` is already implemented in [dense_baseline.py](C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-sutra/code/dense_baseline.py#L791) and the probe is already specified in [q5_r7_postfusion.json](C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-sutra/code/q5_r7_postfusion.json). Lean: run it. Confidence: `7/10`.

3. Full-dim branches are too expensive at `100M`. Obvious: going back to full-dim blows the budget. Alternatives: the budget only blows up because `60%+` is trapped in the inherited FFN; branch capacity can be funded by slimming FFN or using fewer hybrid layers. Evidence: the stable `42M` hybrid was full-dim; approximate parameter arithmetic from current modules shows full-dim branches fit if FFN is questioned. Lean: “full-dim is impossible” is false. Confidence: `8/10`.

4. The current `3x` FFN is the right place to spend `~60%` of params. Obvious: FFN is generic capacity and small models need it. Alternatives: at this scale FFN is crowding out the very branch capacity the hybrid needs; the model may be overpaying for a generic transformer slab and underpaying for routing/mixing. Evidence: all projected variants fail while the only stable hybrid is the no-projection block; the FFN is the only component above `20%` of params. Lean: this is the biggest inherited assumption to attack next. Confidence: `9/10`.

5. Weight norm or spectral norm on projections should be the next fix. Obvious: projection magnitude grows with depth, so constrain the weights. Alternatives: the dominant issue is directional cancellation, not just operator norm; this adds machinery without restoring fused contribution. Evidence: `R6-S` bounded mixing still failed, and telemetry’s strongest signal is fused shrinkage, not merely branch explosion. Lean: not first-line; only consider if `R7-P` is close but not enough. Confidence: `7/10`.

6. Mixed schedules should wait until fusion is solved. Obvious: mixed schedules hide the root cause. Alternatives: after telemetry, mixed schedules are now a root-cause test, because the failure compounds with depth. Evidence: `6G18A` is already queued in the `R7` config and directly tests whether hybrid benefit is front-loaded. Lean: run mixed now, not later. Confidence: `8/10`.

7. The pure transformer should become the production default immediately. Obvious: it is the only stable `100M` model and it wins final BPT (`4.6669` vs `4.7768/4.7661`). Alternatives: one last `5K` probe is cheaper than a long wrong production run; full-dim/slim-FFN hybrids still have an untested path. Evidence: all current projected hybrids fail, but `R7-P` is a direct discriminant and already implemented. Lean: if production training must start now, ship the pure transformer; for architecture research, allow exactly one more probe. Confidence: `8/10`.

8. The next test should be at larger scale. Obvious: perhaps projections only work at `200M+`. Alternatives: scaling up a known loser violates the single-GPU efficiency rule; the relevant question is not “does more capacity rescue it?” but “is the mechanism sound at all?” Evidence: three projected `100M` variants already failed. Lean: do not scale projected hybrids up before a cheap `100M` discriminant wins. Confidence: `9/10`.

9. O4 can wait until architecture is locked. Obvious: backbone first, data later. Alternatives: O4 is already the least developed pillar and MiniPLM is validated enough to run in parallel. Evidence: `500`-window MiniPLM reproduced the pilot: diff `0.2681 ± 0.1045`, with Gutenberg/Wikipedia clearly above WildChat/ELI5. Lean: keep O4 moving now. Confidence: `9/10`.

**2. Research Requests For Claude**

1. Pull exact sub-`200M` evidence on full-dim hybrid blocks that trade FFN width for branch capacity. The key question is not “are hybrids good,” but “what is the best branch/FFN budget split below `200M`?”
2. Pull primary-source methods for heterogeneous multi-teacher alignment: decoder hidden states, encoder sentence embeddings, CLIP-text/diffusion text embeddings, and domain/STEM models mapped into one student space. Prioritize neural stitching, translation networks, CKA/CCA-based teacher selection, and family-rotation curricula.
3. Pull methods for extracting domain priors from non-text or non-autoregressive teachers into a text-only student. The question is how to steal structure from heterogeneous models without requiring the student to become multimodal.

**3. Experiment / Probe Requests For Claude**

1. Run the existing `R7` gate exactly as specified in [q5_r7_postfusion.json](C:/Users/devan/OneDrive/Desktop/Projects/AI Moonshots/moonshot-sutra/code/q5_r7_postfusion.json): transformer vs `24G` vs `6G18A`, `5K` steps. Success remains `>= +0.05` BPT, `kurtosis <= 2.1`, `max_act <= 52`.
2. If both `R7` arms fail, run a budget-reallocation mini-gate, not another projected variant: transformer `24x512 ff1536` vs full-dim `P 24x512 ff1024` vs full-dim `G 24x512 ff1024` or `G 20x512 ff1536`. This directly tests whether the problem was projection compression rather than hybridization.
3. Run the first real O4 decision probe: pure transformer `24x512` trained on raw data vs MiniPLM top-50% filtered/reweighted data at equal token and wall-clock budget.
4. Multi-teacher probe A: offline complementarity map on `5K-10K` windows. Use one decoder LM teacher, two embedding teachers, and one code/math/domain teacher; compute pooled-state CKA/CCA and pick the most complementary set, not the largest set.
5. Multi-teacher probe B: Universal Teacher Interface pilot. Cache a `128d` teacher slot per teacher family offline; train the student to predict one family per batch after step `1000`; compare single-teacher vs three-teacher rotation. Do not average teachers in the same batch.

**4. Per-Outcome Confidence**

- Outcome 1 (Intelligence): `4/10` — lowered from `6/10`. New project data since Round 6 is negative: `R6-F` and `R6-S` both lose to the transformer by step `5000` (`4.7768/4.7661` vs `4.6669`) and both miss stability badly. The positive evidence that remains is only partial: the `42M` full-dim P-block was stable and the `100M` hybrids learn much faster early, so hybrid signal is real, but not production-ready.
- Outcome 2 (Improvability): `5/10` — lowered from `6/10`. New positive evidence exists for diagnosability, not fixability: telemetry isolated projection growth plus directional cancellation, and it falsified the Round-6 beta hypothesis. But no local block change has yet restored stability, so improvability is not proven in the stronger surgical sense.
- Outcome 3 (Democratization): `4/10` — lowered from `5/10`. The code still exposes branches, exits, memory, and auxiliary ports, but there is still zero empirical evidence that independently improved modules compose without retraining the trunk.
- Outcome 4 (Data Efficiency): `4/10` — unchanged. New data exists, but it is still infrastructure evidence, not end-task gain: `500`-window MiniPLM reproduced the pilot and sharpened source ranking. There is still no filtered-training win and no multi-teacher win.
- Outcome 5 (Inference Efficiency): `5/10` — unchanged. Fixed exits remain the only live path and historical elastic-depth evidence still matters, but there is no new exit-ordering, PTQ, or latency result in this round.

**5. What Would Raise Confidence**

- O1: `R7-P` or a full-dim/slim-FFN hybrid beats the transformer at `100M` and stays healthy through `8K-10K`.
- O2: a branch-local budget change or fusion change fixes the model without rewriting the trunk.
- O3: one new module or teacher family is added and improves behavior without retraining everything.
- O4: MiniPLM-filtered training beats raw at equal compute, and a rotated multi-teacher pilot beats single-teacher.
- O5: fixed exits remain ordered after PTQ and produce measured latency savings.

**6. What Would Lower Confidence**

- O1: `R7-P` and the mixed `6G18A` arm both fail, and a full-dim/slim-FFN hybrid also fails.
- O2: every useful fix still requires a new full-model training run with no clean local diagnosis.
- O3: module swaps or teacher ports consistently break the base model.
- O4: filtered data is neutral/worse and multi-teacher rotation adds noise.
- O5: exits collapse under quantization or provide no real latency gain.

**7. Intuitions Worth Testing**

- Best `100M` hybrid will steal budget from FFN, not from branches. Trigger: the current projected hybrid spends `60.6%` on FFN and only `16.9%` on the novel mechanism. Conviction: high. Probe: full-dim branch gate with `ff=1024`.
- `6G18A` has a better chance than `24G`. Trigger: every projected hybrid shows strong early gains and late depth-compounding collapse. Conviction: medium-high. Probe: run the existing mixed `R7` arm.
- The first big O4 win will come from teacher-budget routing, not raw KD weight. Trigger: MiniPLM already says some sources are much more valuable, and purification literature says naive multi-teacher averaging hurts. Conviction: high. Probe: route teacher families by source/domain and difficulty.
- Heterogeneous teachers should meet the student in a shared latent slot space, not in logits. Trigger: tokenizer mismatch is a hard local constraint and the project is text-only, but representation alignment is architecture-agnostic. Conviction: high. Probe: UTI pilot with pooled-state targets.

**8. Design Proposal Or Need More Data**

Need more data before promoting any new hybrid. The operational decision is conditional:

1. If a long production run must begin now, use the pure transformer `24x512` with the current `16K` tokenizer, `SS-RMSNorm`, and fixed exits. It is the only validated stable `100M` path on March 26, 2026.
2. Before killing hybridization entirely, run exactly one decisive probe: `R7-P` all-`G` plus `6G18A`.
3. If both `R7` arms fail, declare the projected-branch family dead at `100M`. Do not spend another cycle on weight norm, soft constraints, or larger-scale projected reruns.
4. If hybrid research continues after that, pivot hard: full-dim branches funded by FFN cuts or by fewer hybrid layers. The boring inherited budget assumption to attack is the `3x` FFN, not the branch width.