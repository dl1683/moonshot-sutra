Proposal: implement the unified system as a staged extension of the current core in [launch_v054.py](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/launch_v054.py), not a rewrite. `v0.6.0` should add gain estimation, shadow supervision, in-loop verify, and freezing on top of the existing 8-step loop. `v0.6.1` should let gain drive stage mass and shared budget. `v0.6.2` should add causal spectrum memory. `v0.6.3` should add teacher absorption. That is the only path that preserves warm-start from `results/checkpoints_v054/step_15000.pt`.

**1. Gain Estimator**
Implement one small shared module `GainEstimator` inside the recurrent loop.

Exact token inputs at step `p`:
- `mu_i ∈ R^768`
- `log_var_i = -log(lam_i.clamp_min(1e-6)) ∈ R^768`
- `pi_i ∈ R^7`
- `entropy_i` from draft logits, scalar
- `margin_i` = top1 minus top2 logit, scalar
- `pass_frac_i = p / (max_steps_ceiling - 1)`, scalar
- `budget_local_frac_i`, scalar
- `budget_shared_frac_i`, scalar
- `delta_mu_rms_i = ||mu_i - mu_{i,p-1}||_2 / sqrt(D)`, scalar
- `pheromone_i`, scalar
- `write_mass_i = pi_i[4]`, scalar

Architecture:
- `LayerNorm(768)` on `mu`, `log_var`
- `Linear(768,128)` for `mu`
- `Linear(768,64)` for `log_var`
- concat projected features with `pi` and 8 scalars: total input `207`
- MLP `207 -> 256 -> 128` with `SiLU`
- heads:
  - `stage_head: 128 -> 5` for stages `{3,4,5,6,7}`
  - `zoom_head: 128 -> 3` for `{coarser, hold, finer}`
  - `resid_head: 128 -> 1` for “best remaining positive gain if we continue”
- new params: about `238k`

Exact outputs:
- `g_stage ∈ R^{B×T×5}`: predicted per-token per-stage future CE reduction
- `g_zoom ∈ R^{B×T×3}`: predicted gain for `Δs ∈ {+0.15, 0, -0.15}`
- `g_resid ∈ R^{B×T}`: predicted `max` achievable remaining gain before freezing

Training targets:
- Draft CE at each step: `ce_p[i] = CE(logits_p[i], y_i)` with `reduction='none'`
- Residual-gain target:
```text
target_resid_p[i] = max(0, ce_p[i] - min_{q>p} ce_q[i])
```
This is the exact stop-worthiness target. It uses future steps from the same forward pass, so store only `ce_hist[B,T,P]`, not logits.

- Stage target for the executed next stage:
```text
target_stage_p[i] = ce_p[i] - ce_{p+1}[i] + gamma * target_resid_{p+1}[i]
gamma = 0.9
```
Apply this only to the dominant chosen stage `argmax(pi_{p+1})`. Mask others at first.

- Zoom target:
  - do not train from full counterfactuals initially
  - on a 10% token subsample, run one no-grad extra read with `Δs ∈ {+0.15,0,-0.15}` and label with the same one-step target
  - otherwise no zoom loss in `v0.6.0`

Bootstrap:
1. `0-1k` steps after warm-start: train `resid_head` only in shadow mode, no control effect.
2. `1k-3k`: train `stage_head` on executed-stage labels, still shadow mode.
3. `3k-5k`: enable kernel blend at `alpha=0.1`.
4. After verifier AUROC and residual-gain AUROC both exceed `0.75`, ramp control influence.

**2. Gain-Driven Stage Transitions**
Keep [sutra_v05_ssm.py](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/sutra_v05_ssm.py) `SwitchingKernel2` as the warm-start prior.

Use:
```text
pi_kernel = pi @ K(mu)
u_stage = g_stage / c_stage
pi_gain = masked_softmax(beta * u_stage, allowed_next_stages)
pi_mix = normalize((1 - alpha) * pi_kernel + alpha * pi_gain)
pi_next = top2_project(normalize(pi_mix * softmax(evidence / temp)))
```

Exact initial stage costs, fixed not learned:
- `c_local(stage3)=1.00`
- `c_route(stage4)=1.25`
- `c_write(stage5)=1.20`
- `c_control(stage6)=1.00`
- `c_verify(stage7)=1.10`

`beta`:
- fixed `4.0` in `v0.6.0` and `v0.6.1`
- optional later upgrade to one learned global scalar, clamped to `[1,8]`

Warm-start coexistence:
- new scalar gate `alpha_kernel_to_gain`
- initialize `alpha=-7.0`, so `sigmoid(alpha)≈0.001`
- with gate closed, model reproduces `v0.5.4`

Collapse prevention:
- retain `top2_project`
- keep a floor prior: `pi_mix = 0.02/K + 0.98*pi_mix`
- add `KL(pi_gain || pi_kernel)` penalty during ramp-up
- use `epsilon=0.05` forced stage exploration on 5% of tokens during training
- never let `alpha` exceed `0.5` before matched-compute eval passes

**3. Gain-Driven Outer Loop**
Round 1 should not add a second outer PyTorch loop. Use the existing 8-step loop in [launch_v054.py](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/launch_v054.py) as the outer loop. Verify happens every recurrent step.

Budget should modify the stop threshold, not the gain. Use a shadow-price threshold:
```text
lambda_budget = lambda0 + lambda1 * (1 - budget_shared_frac)^2 + lambda2 * 1[budget_local<=0]
u(a) = g(a)/c(a) - lambda_budget
```

Exact freeze rule:
```text
freeze if max(u_stage, u_zoom) <= 0
```

Exact reroute rule:
```text
reroute to argmax over positive {stage4, stage5, stage6}
```

Exact shared-budget design:
- local stipend per token: `B_local = 2.0`
- shared pool: `B_shared = 6.0 * T`
- total average budget remains `8.0` units/token, matching current 8-step regime

Do not use “budget exhausted => accept.” If local stipend is empty and shared pool is empty, token stops only because all utilities are non-positive.

OOM fix:
- do not store `pass_logits`
- compute per-step `ce_p`, `entropy_p`, `margin_p`, then discard logits
- store only `ce_hist`, `g_pred_hist`, `active_hist`, each shape `B×T×P`

Exact online CE accumulation:
```text
L_ce = (sum_p w_p * sum_i active_{i,p} * ce_{i,p}) / (sum_p w_p * sum_i active_{i,p})
w = [0.5,0.6,0.7,0.8,0.9,1.0,1.0,1.0] for 8 steps
```

Exact total training loss:
```text
L = L_ce
  + 0.2 * L_resid
  + 0.1 * L_stage
  + 0.02 * L_budget
  + 0.01 * KL(pi_gain || pi_kernel)
```
where
```text
L_resid = mean_smooth_l1(g_resid, target_resid)
L_stage = masked_mean_smooth_l1(g_stage_taken, target_stage)
L_budget = mean(relu(spent_seq - 8*T))
```

**4. Gain-Driven Memory Zoom**
Do not let `scale_net` in [spectrum_memory.py](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/spectrum_memory.py) directly choose absolute `s_i`. Make `s_i` stateful and let gain choose relative moves.

Exact state:
- initialize `s_i = 0.85` for all tokens on pass 0
- update by action:
```text
Δs ∈ {+0.15, 0.0, -0.15}
s_{i,p+1} = clamp(s_{i,p} + Δs*, 0.05, 0.95)
```
Interpretation:
- larger `s` = coarser
- smaller `s` = finer

Exact zoom cost:
```text
c_zoom(s) = 0.2 + 0.8 * (1 - s)^2
```
This makes fine retrieval expensive enough to stop “always fine” collapse.

Implementation order:
- `v0.6.0`: keep [scratchpad.py](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/scratchpad.py)
- `v0.6.2`: fix `ContinuousSpectrumMemory.write_scratch()` to be prefix-causal before enabling it
- only then let `zoom_head` control `s`

Collapse prevention:
- coarse init `s=0.85`
- positive cost for finer zoom
- occupancy regularizer in early training: target mean `s` in `[0.70,0.85]`
- matched-compute ablation required before rollout

**5. Gain-Driven Teacher Absorption**
Use only teachers allowed by `../models/MODEL_DIRECTORY.md`.

Round-1 allowed teachers:
- autoregressive: `Pythia-160M`, `Pythia-410M`, `Qwen3-0.6B`, `SmolLM2-360M`
- embedding: `EmbeddingGemma-300M`, `bge-base-en-v1.5`, `e5-base-v2`, `Qwen3-Embedding-0.6B`

Not allowed yet:
- `GPT-2`, `BERT`, `DeBERTa`
Reason: they are not in the model directory. Add them there first if you want them.

Stage mapping:
- AR teachers -> Stage 7 readout
- embedding teachers -> Stage 5 memory geometry
- no Stage 4 encoder-teacher path until a prefix-causal encoder teacher is approved

Exact teacher utility:
```text
u_{i,j,k} = g_{i,j,k} / c_{j,k}
```
Pick only the top positive teacher-stage pair per token.

Stored artifacts, not live teacher inference:
- AR teacher cache: top-8 token ids + top-8 logprobs per selected token, fp16
- embedding cache: chunk embedding compressed to 256 dims, fp16, L2-normalized
- teacher metadata: `teacher_id`, `stage_contract`, `chunk_id`, `valid_mask`

Teacher is “fully absorbed” when all three hold on held-out data:
- `95%` of eligible tokens have `u_{i,j,k} < 0.01`
- dropping that teacher changes validation BPT by `< 0.01`
- dropping that teacher changes stage-specific probe metric by `< 0.5%`
Then remove its loss entirely.

**6. Warm-Start Path: `v0.5.4 step_15000 -> v0.6.x`**
Load from `results/checkpoints_v054/step_15000.pt`.

Exact modules loaded unchanged:
- `emb`, `pos_emb`
- `init_mu`, `init_lam`
- `transition`
- `stage_bank`
- `router`
- `writer`
- `scratchpad`
- all six `GatedLayerNorm`s
- final `ln`

New in `v0.6.0`:
- `GainEstimator`
- `done_mask` logic
- budget accountant tensors
- optional `pass_emb` or `pass_frac`
- `alpha_kernel_to_gain` gate
- `alpha_soft_freeze` gate

Initial values:
- `alpha_kernel_to_gain = -7.0`
- `alpha_soft_freeze = -7.0`
- `max_steps_ceiling = 8`
- `B_local=2.0`, `B_shared=6.0`
- `beta=4.0`
- no spectrum memory yet

Optimizer:
- do not load `v0.5.4` optimizer into `v0.6.0`
- create fresh AdamW
- rewarm for 500 steps exactly like [train_v054.py](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/train_v054.py)

Lossless warm-start verification:
1. Instantiate `v0.6.0` with all new gates closed.
2. Load `step_15000.pt`.
3. Run the same input through `v0.5.4` and `v0.6.0`.
4. Require:
   - max logit diff `< 1e-6` in fp32
   - aux values identical except new diagnostics
5. Save `v0.6.0` at step 10, resume, require next 3 losses match bitwise in fp32 smoke test.

Version schedule:
- `v0.6.0`: gain shadowing + soft freeze + budget observer
- `v0.6.1`: gain-driven stage mix + hard freeze + shared budget
- `v0.6.2`: causal spectrum memory + gain-driven zoom
- `v0.6.3`: teacher absorption + teacher dropout + teacher-free consolidation

**7. Experiments Required Before Full Rollout**
Run these before enabling each controller.

1. `Per-step convergence probe`
- data: current validation shards, 100k tokens
- metric: per-token `ce_p` trajectory under current 8-step model
- pass: bottom 30% easiest tokens improve `< 0.02` CE after step 4 while top 30% hardest still improve `> 0.10` by step 8
- fail: no separation, meaning freezing is not worth building

2. `Residual-gain label calibration`
- data: same 100k-token val slice
- metric: AUROC of `g_resid` vs `target_resid > 0.05`
- pass: AUROC `>= 0.75`, AUPRC `>= 0.55`
- fail: estimator is not usable for freezing

3. `Lossless warm-start parity`
- data: 4 fixed mini-batches
- metric: max logit diff and 3-step resumed loss match
- pass: `<1e-6` fp32 parity, identical resumed losses
- fail: version transition is not safe

4. `Soft freeze at matched compute`
- data: training for 2k warm-started steps, same corpus
- metric: validation BPT and average active steps
- pass: either `>=10%` drop in active steps at `ΔBPT <= 0.02`, or `>=0.03` BPT gain at same average steps
- fail: keep freezing disabled

5. `Gain vs kernel routing`
- data: synthetic mixed-demand sequences with local-easy / gist / exact-copy / extra-serial classes
- metric: class accuracy, MI(token_class; chosen_stage), matched compute
- pass: hard classes improve `>=3` points and MI `>=0.20 bits`
- fail: keep kernel as primary router

6. `Causal spectrum memory test`
- data: unit causality test plus synthetic exact-recall task
- metric: earlier-position outputs unaffected by future-token perturbation
- pass: max earlier-token diff `<1e-6` fp32 and exact-recall accuracy `>=+5` points over scratchpad at matched compute
- fail: do not enable spectrum memory

7. `Teacher absorption sparsity`
- data: 5M-token curated subset
- metric: fraction of tokens receiving teacher loss, teacher-drop regression
- pass: teacher loss naturally sparsifies below `35%` of tokens by 5k steps and teacher-drop regression after consolidation is `<0.02` BPT
- fail: teacher routing is not selective enough

Bottom line: Round 1 should implement a shared `GainEstimator` trained on exact hindsight residual-gain labels from the current 8-step loop, blend it with the existing switching kernel behind a closed gate, freeze only when all gain-per-cost utilities are non-positive, and defer spectrum memory and teacher routing until causality and warm-start parity are proven. That is the concrete path from `v0.5.4` to a true unified controller without breaking the checkpoint lineage.