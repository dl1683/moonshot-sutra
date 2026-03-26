Operational note: on March 26, 2026 at 03:31 EDT, `nvidia-smi` showed the RTX 5090 idle at `0/24463 MiB`. `results/` had no active checkpoint directories; the newest completed probe was `trunk_choice` at 03:27 EDT.

**1. Assumption Challenges**
1. `42M is still only the scout scale, not the final judge.` Obvious reading: the completed V3 trunk-choice result means the architecture question is settled. Alternatives: the ranking could flip at 100M; or the 12x512 probe may reward one very specific inter-layer schedule. Distinguish with a matched 100M gate. Lean: keep the two-scale rule, and do not wait for anything else because V3 is already complete as of March 26, 2026 03:27 EDT. Confidence: 9/10.
2. `Hybrid is now the mainline trunk hypothesis.` Obvious reading: hybrid_3to1 won locally at 42M/5k with `4.7218` BPT versus `4.9131` for pure transformer and `5.6147` for pure conv, with better activation health and fewer params. Alternatives: the gain may be specific to inter-layer alternation; or it may mostly be a parameter-allocation effect rather than a universal hybrid win. Distinguish with a 100M pure-transformer vs intra-layer-parallel match. Lean: yes, pivot now. Confidence: 8/10.
3. `The non-attention branch should be a small-k local mixer, not k=64 everywhere.` Obvious reading: Falcon-H1’s tiny local mixer means our `k=64` GatedConv is the wrong within-block choice. Alternatives: `k=64` may be the very reason the sparse inter-layer probe worked; or kernel size may matter less than simply having any local branch. Distinguish with intra-layer `k=4` vs `k=16` vs `k=64`. Lean: use a small-k depthwise GatedConv surrogate, default `k=16`, not Mamba yet and not `k=64` in every block. Confidence: 6/10.
4. `Fusion should be scaled concat-then-project, not mean or head-interleaving.` Obvious reading: Falcon-H1’s concat path is the better template because branch roles are unequal. Alternatives: Hymba’s mean may be more stable; head-interleaving may regularize better at small scale. Distinguish with a short fusion ablation on the same block. Lean: scaled concat plus two branch scalars; this is the light version of the Falcon balancing trick. Confidence: 7/10.
5. `Depth should dominate width for the 100M scout.` Obvious reading: Falcon-H1, Hymba, and the local 12x512 win all point toward “go deeper, not wider.” Alternatives: our conv surrogate is weaker than Mamba, so extra width may matter more; or 512 may underpower attention. Distinguish with `24x512` vs `18x640` at matched steps. Lean: `24x512` for the immediate gate, and `26x512` for the full scout if the gate passes. Confidence: 7/10.
6. `Conv-only losing does not kill the conv-heavy thesis, but it does argue for more attention than Falcon-H1 while the branch is only GatedConv.` Obvious reading: move away from Round 3’s `1:2` attention:conv ratio. Alternatives: sparse attention is still enough, because the inter-layer winner used only 3 attention blocks out of 12; or ratio matters less than fusion quality. Distinguish with `1:1` vs `2:3` vs `1:2` within the same intra-layer block. Lean: `1:1` branch dims for the GatedConv scout, then revisit more conv-heavy ratios only after a stronger SSM branch exists. Confidence: 6/10.
7. `AdamW stays default, but NorMuon belongs in the 100M optimizer probe.` Obvious reading: local AdamW already won, so stop there. Alternatives: Muon failed because of LR/horizon, not concept; NorMuon directly fixes the exact late-stage spike pattern we saw. Distinguish with a 500-step NorMuon smoke test and then a 100M four-way optimizer probe. Lean: AdamW mainline, NorMuon side probe. Confidence: 8/10 for AdamW default, 7/10 for adding NorMuon.
8. `MiniPLM should use a truly weak same-tokenizer reference.` Obvious reading: follow the paper and train a ~100M/5B reference. Alternatives: Qwen3-0.6B may be “good enough” and far simpler; or reference strength may matter less than family/tokenizer match. Distinguish with score-dispersion and downstream pilot comparisons. Lean: train the ~100M Qwen-tokenizer reference; use Qwen3-0.6B only as a quick pipeline pilot. Confidence: 7/10.
9. `The first 100M scout should remain brutally simple online.` Obvious reading: keep online loss to plain NTP plus fixed exits, because every local extra loss hurt. Alternatives: the stronger hybrid trunk may now tolerate a light representation loss; or exits may need self-distillation sooner. Distinguish only after the plain scout is positive. Lean: no memory, no online teachers, no TOP, no MTP, no learned halting in the first scout. Confidence: 9/10.

**2. Research Requests**
1. Determine whether MiniPLM-style difference sampling is still valid with cross-tokenizer teacher/reference pairs after per-byte normalization, or whether same-tokenizer teacher/reference should be treated as mandatory.
2. Find the best small embedding teachers under 1B for semantic transfer into decoder students, including which layer and pooling strategy are most stable for alignment.
3. Find the lightest practical branch-balancing recipes for two-path hybrid decoders at ~100M: scalar gates, residual rescaling, or partial μP, with emphasis on open implementations.

**3. Experiment / Probe Requests**
1. Run the 100M architecture gate: `24x512` pure transformer vs `24x512` intra-layer hybrid, same tokenizer, AdamW, SS-RMSNorm, fixed exits, plain NTP, no memory/KD/TOP. Success condition: hybrid wins final BPT by at least `0.10` and keeps `max_act <=` transformer.
2. Run a local-mixer microprobe on the intra-layer block: `k=4`, `k=16`, `k=64`, all else matched. This resolves Falcon-style tiny local mixing vs Hyena-style large kernel.
3. Run a fusion/ratio microprobe: mean `1:1`, scaled-concat `1:1`, scaled-concat `2:3`. Log final BPT, branch RMS, kurtosis, and max activation.
4. Run the 100M optimizer probe on the winning architecture: AdamW, Muon `0.01`, Muon `0.005`, NorMuon `3.6e-4`. First do a 500-step NorMuon numerical smoke test.
5. Run a MiniPLM reference probe on 10% of the corpus: Qwen3-1.7B teacher with Qwen3-0.6B reference vs custom ~100M Qwen-tokenizer reference. Compare score spread, top-50% overlap, and downstream pilot training.
6. After the plain scout is positive, run the first O4 multi-source pilot: one decoder teacher plus one embedding teacher, one family per batch, pooled-state alignment at weight `0.05` after step `1000`.
7. Run the quantized-exit fidelity probe after the 100M scout: PTQ/NVFP4 full-depth quality, exit ordering after PTQ, and real latency savings.

**4. Per-Outcome Confidence**
1. Outcome 1 (Intelligence): 5/10. Up from 4/10 because the March 26, 2026 trunk-choice probe produced the first project-local positive trunk result: hybrid `4.7218` BPT vs transformer `4.9131`, with lower kurtosis and lower max activation. It is still only 42M, only 5k steps, and not yet the intended intra-layer form.
2. Outcome 2 (Improvability): 6/10. Unchanged. The architecture story is still good because attention branch, local branch, exits, and later teacher ports are named modules, but there is still no local proof that a module-local change improves behavior without collateral damage.
3. Outcome 3 (Democratization): 5/10. Unchanged. The hybrid pivot actually helps because it creates a cleaner “swap the local branch” interface, but this is still design intent, not an executed contributor-style extension.
4. Outcome 4 (Data Efficiency): 3/10. Unchanged. MiniPLM and filtering are concrete, but there is still no local positive O4 result, and multi-source pretraining remains a plan rather than evidence.
5. Outcome 5 (Inference Efficiency): 5/10. Unchanged. Fixed exits are still the only live path; the healthier hybrid activations are good news, but there is still no new exit-calibration, PTQ, or latency evidence, so the score cannot rise.

**5. What Would Raise Confidence**
1. O1: a 100M intra-layer hybrid beats the matched pure transformer and produces visibly better generations.
2. O2: a frozen-trunk experiment improves behavior by changing only one module.
3. O3: a new branch or teacher family is added without retraining the whole model and the gain composes.
4. O4: quality filtering plus MiniPLM plus one multi-source teacher pair beats raw-data training at equal compute.
5. O5: exits stay ordered after PTQ and deliver real measured latency savings.

**6. What Would Lower Confidence**
1. O1: the 100M intra-layer hybrid fails to beat the pure transformer, or only “wins” by instability/noise.
2. O2: every useful improvement still requires touching the whole model.
3. O3: swapping a branch or teacher family breaks the model badly.
4. O4: MiniPLM, filtering, and multi-source pilots are all neutral or negative locally.
5. O5: exit heads collapse, PTQ breaks exit ordering, or speculation acceptance is poor.

**7. Intuitions**
1. The hybrid win is probably conservative because it was achieved with plain RMSNorm, while a separate local probe already showed SS-RMSNorm beats RMSNorm. Conviction: medium-high. Validate by rerunning the 100M architecture gate with SS-RMSNorm from step 0.
2. Small-k intra-layer conv will beat `k=64` once the local branch exists in every block, because the sparse inter-layer probe likely used `k=64` to compensate for infrequent local mixing. Conviction: medium. Validate with the kernel microprobe.
3. NorMuon will look mediocre early and then separate late, because our local Muon run was strong early and bad late, and NorMuon’s row-wise normalization is aimed exactly at that failure mode. Conviction: medium. Validate with the 100M optimizer probe.
4. The first real O4 win will be offline MiniPLM plus one-family-per-batch online representation transfer, not simultaneous multi-teacher averaging. Conviction: high. Validate with the deferred decoder+encoder pilot.
5. `24x512` is the right immediate gate, but `26x512` is the real sweet spot if the gate passes, because it spends the full budget while keeping the same proven width. Conviction: medium. Validate by depth-extending the winning gate geometry without changing anything else.

**8. Design Proposal**
Ready enough to propose a design: `HEMD-R4-S`.

Architecture: real-valued decoder-only LM, `26` blocks, hidden dim `512`, FFN dim `1536`, tied `16K` embeddings, context `512` first then `1024`, fixed exits after blocks `9`, `18`, and `26`. If you want the cheaper promotion gate first, run the exact same block at `24x512` with exits `8/16/24`.

Block math:
```text
u = g1 * h / sqrt(mean(h^2) + eps)                              # SS-RMSNorm
a = Attn(RoPE(Wq u), RoPE(Wk u), Wv u)                          # d_att = 256, 4 Q heads, 2 KV heads, head_dim = 64
c = Wco( DWConv1D_k=16(Wcv u) * sigmoid(Wcg u) )               # d_conv = 256, depthwise causal gated conv
m = Wmix([ga * a ; gc * c])                                     # scaled concat-project, ga=gc=1.0 init
r = h + m
n = g2 * r / sqrt(mean(r^2) + eps)                              # SS-RMSNorm
h_next = r + Wdown( silu(Wgate n) * Wup n )                     # SwiGLU FFN
```

Number system, activations, norm, init, precision: real-valued activations, SwiGLU FFN, SS-RMSNorm everywhere, `N(0, 0.02)` init for linear/embedding weights, branch scalars initialized to `1.0`, BF16 forward/backward, FP32 optimizer states. Do not introduce complex states, Mamba kernels, DyT, TOP, MTP, learned halting, memory, or online teacher losses into this first scout.

Optimizer and loss: AdamW `lr=3e-4`, betas `(0.9, 0.95)`, weight decay `0.1`, WSD schedule, `200` warmup steps, `1e-5` min LR, gradient clip `1.0`. Loss is `L = CE_26 + 0.20 * CE_9 + 0.35 * CE_18`. NorMuon is probe-only, not default.

Data pipeline: keep the offline O4 stack and nothing else online. First filter the corpus: drop the bottom `10%` of shards and half-weight the next `20%`. Then run MiniPLM difference sampling with Qwen3-1.7B as teacher and a custom ~100M Qwen-tokenizer reference trained on `5B` tokens; select the top `50%` by score. Feed those weights into the existing shard-weight path. Online training stays plain NTP plus fixed exits.

Parameter and VRAM estimate: with `26x512`, `d_att=256`, `d_conv=256`, `ff=1536`, `k=16`, the model is about `98.6M` params before tiny scalar/bias terms. Rough breakdown: embeddings `8.2M`, each hybrid block `3.48M`, total trunk `90.4M`. Training memory at seq `512`, microbatch `16`, BF16 should land roughly in the `13-16 GB` range including optimizer state; seq `1024` should require batch reduction or more accumulation.

Why this is the best current path: it is the lowest-risk extrapolation of the first locally positive trunk result, it preserves the strongest proven width (`512`), it makes the local/global split explicit for Outcomes 2 and 3, it keeps O4 offline where the project’s positive evidence actually is, and it preserves fixed exits as the only still-live O5 mechanism.