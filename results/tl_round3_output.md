Live state on March 26, 2026 at 02:46 ET: `nvidia-smi` showed `6088 MiB / 24463 MiB`, `94%` GPU utilization, active `python.exe` plus `ollama.exe`. The freshest log indicates the running job is the trunk-choice probe, currently in the `pure_transformer` arm around step `2600`, so local trunk evidence is still incomplete.

**1. Assumption Challenges**
- `42M is the right probe scale.` For: it already killed bad drop-ins cheaply and clearly: DyT lost by `+0.75` BPT, TOP lost by `+4.61` BPT with kurtosis exploding to `99.3`, prior MTP lost by `+0.13`, prior halting lost by `+0.15`. Against: several mechanisms in the literature only pay off above this regime: TOP was validated at `340M+`, MoR underperforms at `135M`, MiniPLM gains get larger at `500M` and `1.2B`. Resolve with a dual gate: `42M` for catastrophic screening, `100M` for promotion. Lean: keep `42M`, but only as a falsification floor, not as the final selection scale. Confidence: `9/10`.
- `All failed auxiliaries are universally dead.` For: three different auxiliary losses have now hurt locally, and TOP was not marginally bad but catastrophically unstable. Against: the same data also supports a capacity-threshold interpretation, because TOP and token-adaptive compute succeed in the literature at larger scales. Distinguish by severity: catastrophic failures at `42M` mean “not for the scout”; mild negatives mean “retest at `100M`.” Lean: TOP, classic MTP, and learned halting are dead for the `100M` scout mainline, not necessarily dead forever. Confidence: `8/10`.
- `HEMD survives unchanged from Round 2.` For: the broad thesis still lives: local-global mixing is plausible, fixed exits still fit the mission, and the project still needs a modular edge model. Against: the specific Round 2 embodiment is outdated. The field has moved from inter-layer alternation to intra-layer parallel hybrid blocks, and Muon no longer deserves default status. Lean: keep the family, but rewrite it: Hybrid = intra-layer parallel, Elastic = fixed exits only, Memory = deferred, Decoder = retained. Confidence: `7/10`.
- `Inter-layer hybrid is the right hybrid.` For: it is simpler, already represented in code, and the live `42M` trunk probe can still tell us whether non-attention mixing helps at all. Against: March 2026 evidence is convergent: Falcon-H1 and Hymba both use parallel attention plus SSM inside every block; the systematic analysis says intra-layer beats inter-layer; inter-layer is now the older, weaker baseline. Lean: pivot the architecture hypothesis to intra-layer parallel hybrid. Use the running inter-layer probe only as a lower bound. Confidence: `7/10`.
- `Muon should still be the scout optimizer.` For: at `42M`, Muon converged much faster early and clearly has real signal up to about step `3000`; literature says the efficiency story is real. Against: the only completed local optimizer verdict is negative. `AdamW + SS-RMSNorm` finished best at `4.91` BPT; `Muon + SS-RMSNorm` finished worse at `5.01`, with `2.7x` larger max activations and visible regressions around steps `2000` and `3500-4000`. Lean: AdamW for the scout, lower-LR Muon or NorMuon only as side probes. Confidence: `8/10`.
- `The scout should still carry memory, teacher ports, or planning losses on day 1.` For: they serve Outcomes 4 and 5 in theory. Against: the project’s actual probe pattern is that extra moving parts hurt before the base optimizer, norm, and trunk are stable. The only clearly validated Round 3 additions are SS-RMSNorm and architectural simplification. Lean: the first scout should be brutally simple: data shaping offline, plain NTP online, fixed exits, nothing else. Confidence: `9/10`.

**2. Research Requests**
- Research NorMuon implementation details for small dense decoders, especially whether its neuron-wise normalization fixes the exact instability pattern seen in local Muon runs.
- Research the smallest successful intra-layer parallel hybrid blocks that use causal convolution instead of Mamba, because the repo already has GatedConv code and does not yet have a production-grade Mamba branch.
- Research multi-teacher purification for pretraining, because the next O4 step should not be naive averaging; the Knowledge Purification result says teacher conflict is real.
- Research MiniPLM reference-model choice for same-family teacher/reference pairs, because `Qwen3-1.7B -> Qwen3-0.6B` is likely cleaner than cross-family scoring for the first pass.

**3. Experiment / Probe Requests**
- Adopt a two-scale protocol. `42M / 5k` stays the catastrophic-screen. `100M / 5-10k` becomes the promotion gate for anything capacity-sensitive.
- Finish the live trunk-choice probe, but do not let it decide the final hybrid style. If inter-layer hybrid wins, that supports “mixing helps.” If it loses, intra-layer is still alive because it is a different, stronger design.
- Run a matched `100M` trunk probe: pure transformer vs intra-layer parallel hybrid. Same tokenizer, AdamW, SS-RMSNorm, plain NTP, fixed exits, no memory, no TOP, no KD.
- Run a `100M` optimizer probe: `AdamW + SS-RMSNorm` vs Muon with `lr=0.01` and `lr=0.005`. Promote Muon only if it beats AdamW at final BPT and keeps max activation under `2x` the AdamW baseline.
- Run a `100M` data-shaping probe: raw data vs quality filter only vs MiniPLM only vs quality filter plus MiniPLM. This is the real O4 gate.
- Run one multi-source probe only after the plain scout is healthy: pooled-state alignment to one decoder teacher and one embedding teacher, one family per batch, no simultaneous multi-teacher averaging.

**4. Per-Outcome Confidence**
- Outcome 1 (Intelligence): `4/10`. No increase from Round 2. New data simplified the recipe by validating SS-RMSNorm and rejecting TOP and Muon-as-default, but there is still no integrated scout win over the dense baseline.
- Outcome 2 (Improvability): `6/10`. No change. The architecture can still expose named modules, but there is still no module-isolation proof that local changes compose cleanly.
- Outcome 3 (Democratization): `5/10`. No change. The best current path is still modular ports and swappable blocks, but that is still architectural intent rather than an operational ecosystem.
- Outcome 4 (Data Efficiency): `3/10`. No change. MiniPLM and quality filtering are concrete, but still unvalidated locally. The new probe evidence mostly killed online auxiliaries rather than proving multi-source learning.
- Outcome 5 (Inference Efficiency): `5/10`. Down from `6/10`. New data killed TOP at `42M`, prior learned halting was already negative, and Muon did not validate as the hoped-for quantization-friendly default. Fixed exits remain the only live path.

**5. What Would Raise Confidence**
- O1: a `100M` scout that beats the best `42M` control by at least `0.25-0.35` BPT on the same eval cache and shows cleaner generation.
- O2: a post-scout experiment where only one module is changed and the improvement sticks without retraining the whole trunk.
- O3: one successful contributor-style extension, such as adding a new teacher port or replacing one block family without breaking the rest.
- O4: a local win from `quality filter + MiniPLM` at equal compute, followed by one non-destructive multi-source representation probe.
- O5: fixed exits that stay well-ordered after PTQ and deliver real latency savings without noticeable BPT drift.

**6. What Would Lower Confidence**
- O1: if the `100M` scout still cannot beat a plain transformer control after removing the dead mechanisms.
- O2: if every meaningful improvement still requires touching the whole model.
- O3: if the architecture can only be improved by monolithic retraining.
- O4: if MiniPLM and quality filtering are neutral or negative locally, because that would remove the cleanest O4 lever.
- O5: if even fixed exits do not calibrate well after training, because then no validated adaptive-compute mechanism remains below `200M`.

**7. Intuitions**
- The scout should get most of its early gains from better data, not clever losses. Trigger: every extra online loss has hurt locally, while the literature says quality filtering and MiniPLM are low-risk and scale well. Conviction: high. Validation: the `raw / filter / MiniPLM / both` probe.
- Muon’s local failure looks like a learning-rate or horizon problem, not a concept failure. Trigger: it wins hard early, then destabilizes late. Conviction: medium. Validation: `100M` Muon LR sweep or NorMuon.
- Intra-layer hybrid is the right architectural pivot, but GatedConv is the right first SSM surrogate because it already exists in the repo. Trigger: Falcon-H1 and Hymba validate the parallel principle, while the codebase already supports conv blocks. Conviction: high. Validation: one `100M` parallel-hybrid vs pure-transformer probe.
- The project should stop asking the scout to solve O4 and O5 with the same mechanism. Trigger: TOP, MTP, and halting all tried to do both and all failed. Conviction: high. Validation: keep O4 offline first, O5 as fixed exits first.

**8. Design Proposal**
- `Probe-scale policy:` `42M` remains the numerical-stability and gross-regression screen. Promotion decisions move to `100M`. Catastrophic `42M` failures like TOP stay dead for the scout. Mild negatives like Muon get one `100M` retry.
- `Scout name:` `HEMD-R3-S`, but it is not Round 2 HEMD. It is an intra-layer parallel-hybrid decoder with fixed exits and no day-1 extras.
- `Tokenizer / sequence:` keep the `16K` tokenizer. Train at context `512` first, then `1024` after the base loss is stable. Keep dimensions multiples of `64`.
- `Backbone:` `14` blocks, hidden size `768`, FFN size `2048`, tied embeddings, exits after blocks `5`, `10`, and `14`. This stays near the `100M` budget while moving slightly deeper than the old `12x768` scout.
- `Parallel hybrid block:` use real-valued activations and SS-RMSNorm. For each block,
```text
u = g * h / sqrt(mean(h^2) + eps)
a = Attn(Wa u)                         with d_att = 256, GQA 4Q / 2KV, head_dim = 64
c = ConvOut(DepthwiseConv1D(Wv u) * sigmoid(Wg u))   with d_conv = 512, kernel = 64
r = h + Wmix[a ; c]
n = g' * r / sqrt(mean(r^2) + eps)
h_next = r + Wdown(silu(Wgate n) * Wup n)
```
This keeps the Falcon-H1/Hymba principle, but uses causal convolution instead of Mamba for implementation realism.
- `Number system / activations / norm / precision:` real-valued trunk, SwiGLU FFN, SS-RMSNorm everywhere, BF16 training, FP32 optimizer states. Do not introduce complex states, hyperbolic trunk geometry, or DyT into the scout.
- `Optimizer:` AdamW, `lr = 3e-4`, `betas = (0.9, 0.95)`, `weight_decay = 0.1`, WSD schedule, warmup `200`, min LR `1e-5`, clip `1.0`. Lower-LR Muon and NorMuon are probes, not defaults.
- `Loss:` `L = CE_14 + 0.2 * CE_5 + 0.35 * CE_10`. No TOP, no MTP, no learned halting, no memory, no online KD in the first scout. If exit quality is weak, test self-distillation later as a small follow-up, not on day 1.
- `Data pipeline:` Stage `0A` quality filter first. Drop the bottom `10%` of shards after spot-checking, half-weight the next `20%`. Stage `0B` MiniPLM second using `Qwen3-1.7B` as teacher and `Qwen3-0.6B` as reference on raw text windows, then feed weights into the existing shard-weight path. Do not start with online teacher losses.
- `Memory / multi-source learning:` external memory and teacher ports move out of the first scout. Once the plain scout is positive, the first multi-source extension should be one decoder teacher plus one semantic embedding teacher with one family active per batch.
- `Realistic targets:` for the short `100M` promotion run, require `<= 4.6` BPT on the current deterministic eval cache and a clear improvement over the best `42M` control. For the full scout, require roughly `1-2B` training tokens, internal BPT near `<= 4.0`, and benchmark targets in the Pythia-160M band: HellaSwag `30-35%`, ARC `28-33%`, PIQA `60-64%`. Stretch goals near SmolLM2-135M are not fair first gates because SmolLM2 used `2T` tokens versus our `22.9B`.