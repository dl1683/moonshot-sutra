# v0.6.0a Final Buildable Spec

## CONVENTIONS (Round 6 Fix #1)

- Pass index: `p in {0, 1, ..., 11}` (0-based, 12 passes total)
- "Pass 8" in human terms = `p=7`
- Final pass = `p=11`
- `mu_hist` shape: `(B, T, 12, D)`
- `pi_hist` shape: `(B, T, 12, 7)`
- `probe_pred` shape: `(B, T, 12)`
- `residual_after_8 = CE[p=7] - min(CE[p=8], CE[p=9], ..., CE[p=11])`

## SAMPLED MARGIN MATH (Round 6 Fix #2)

- `candidate_ids` are built ONCE from final-pass (`p=11`) logits after the full forward pass
- For each token: target_id + K=32 hard negatives (top-32 non-target from final logits)
- Per-pass sampled scores: `scores_p = F.linear(LN(mu_p), emb[candidate_ids]) / sqrt(dim)` — shape `(B, T, 33)`
- `sampled_ce_p = -log_softmax(scores_p)[..., 0]` (CE over the 33-way classification)
- `sampled_margin_p = scores_p[..., 0] - max(scores_p[..., 1:])` (target score minus best negative)
- `sampled_margin_slope_p = sampled_margin_p - sampled_margin_{p-1}` (zero at p=0)
- Candidates exclude target; random-table fill excludes already-selected negatives

## FREEZE/CACHE SEMANTICS (Round 6 Fix #3)

- `force_freeze_after_pass=N` means tokens update through `p=N-1` and are frozen starting `p=N`
- `force_freeze_mask` shape: `(B, T)` boolean
- Cache reads for query position `t` may use ONLY frozen positions `j < t` (prefix-causal)
- "Later-token BPT delta" = BPT computed only on tokens with at least one frozen earlier position
- FrozenPrefixCache stores `(B, T, 192)` K and `(B, T, 192)` V — NOT `(B, T, T, 192)`

## EVALUATION PROTOCOL (Round 6 Fix #4)

- Easy/hard buckets: computed once from pass-1 (`p=0`) full CE on fixed 50K-token eval slice, held fixed
- `residual_after_8 = CE[p=7] - min(CE[p=8..11])`
- Top-decile recall: over all evaluated tokens on fixed slice
- Logistic baseline: fit on first half of eval slice, scored on second half, features=[margin, margin_slope]
- Result JSON schema: `{"experiment": str, "step": int, "n_tokens": int, "metrics": {...}, "thresholds": {...}, "status": "pass"|"fail"}`

## Decision

Build `v0.6.0a` as a **probe-only dense-12 training scaffold**, not a controller.

Goal: answer one question before any full controller work:

> If the v0.5.4 core is retrained **from scratch** at `12` recurrent passes on the diverse `17B+` token corpus, with **inter-step supervision from pass 0**, does it naturally develop convergence separation that the uniform-8-step `v0.5.4` model does not have?

If the answer is no, stop the controller branch.

## What v0.6.0a Is

`v0.6.0a` keeps the `v0.5.4` core dynamics and changes only what is required to measure/train stepwise convergence safely:

- `12` recurrent passes from the start
- inter-step supervision from pass `0`
- one scalar residual-gain probe head in shadow mode
- one optional differentiable frozen-prefix cache used only for ablations
- full controller features explicitly excluded

## What v0.6.0a Is Not

Do **not** build any of this in `v0.6.0a`:

- no acting stop/continue controller
- no stage-control head
- no budget controller
- no graph migration (`STAGE_GRAPH` stays unchanged)
- no active-set compaction
- no spectrum memory
- no teacher routing or KD
- no warm-started production training from `v0.5.4` weights

The only warm-start allowed is **code-level reuse** of the `v0.5.4` architecture and an optional parity harness that proves the shared core still reproduces `v0.5.4` when probe features are disabled.

## Exact Build

### 1. New file: `code/launch_v060a.py`

Create a new versioned launch file by copying the `v0.5.4` model structure and adding probe-only instrumentation.

Implement these symbols:

#### `class ResidualGainProbe(nn.Module)`

Purpose: predict future residual improvement from the current pass state.

Inputs per token at pass `p`:

- `mu_p` from the live recurrent state
- `pi_p`
- sampled margin at pass `p`
- sampled margin slope: `margin_p - margin_{p-1}` with zero at pass `0`
- `delta_mu_rms_p = ||mu_p - mu_{p-1}|| / sqrt(D)` with zero at pass `0`
- pass fraction `p / 11`

Do **not** use raw `lam` or `log_var`.

Exact architecture:

- `Linear(dim, 128)` on `mu`
- concat projected `mu` with `pi` and 4 scalars
- `Linear(128 + 7 + 4, 128) -> SiLU -> Linear(128, 1)`

Output:

- `r_hat_p[b, t]`: predicted residual sampled-loss gain from pass `p` onward

This head is shadow-only in `v0.6.0a`; it does not modify `mu`, `pi`, routing, or stopping.

#### `class FrozenPrefixCache(nn.Module)`

Purpose: test whether frozen tokens can remain useful context without updating their state.

Exact constraints:

- projected cache size `mem_dim = 192`
- store **per-token** `K` and `V` only: shapes `(B, T, 192)` and `(B, T, 192)`
- never materialize `(B, T, T, 192)` cache tensors
- causal reads may use score tensors of shape `(B, T, T)`, but stored cache state must remain `O(B*T*192)`

Implement methods:

- `init_state(batch, seq, device, dtype)`
- `write(mu, freeze_mask, state)`
- `read(mu, state)`

Behavior:

- `write()` captures `K/V` only for tokens frozen on the current pass
- `read()` lets active later tokens attend causally to cached frozen tokens
- base training path keeps this module off
- only forced-freeze ablations use it

#### `def build_negative_set(final_logits, targets, k=32)`

Purpose: get cheap per-pass supervision without per-pass full-vocab logits.

Exact rule:

- take final-pass top-48 ids
- remove target id
- keep first `32` unique negatives
- if fewer than `32`, fill from a fixed precomputed random negative table

Return shape:

- `(B, T, 33)` candidate ids with target in slot `0`

#### `def sampled_pass_ce(mu_hist, emb_weight, targets, candidate_ids)`

Purpose: compute cheap sampled CE for all passes.

Exact rule:

- for each pass `p`, normalize `mu_p` with the same final `LayerNorm`
- score only the `33` candidate ids via dot product with tied embedding rows
- compute cross-entropy where target is slot `0`

Return:

- `sampled_ce_hist` with shape `(B, T, 12)`
- `sampled_sampled_margin_hist` with shape `(B, T, 12)`

#### `class SutraV060a(nn.Module)`

Start from `SutraV054`.

Exact changes:

- default `max_steps=12`
- keep the recurrent core identical to `v0.5.4`
- keep `STAGE_GRAPH` identical
- keep scratchpad path identical
- keep pheromone logic identical except generalized to 12 passes by using pass index directly
- collect `mu` after every pass into `mu_hist`
- collect `pi` after every pass into `pi_hist`
- collect sampled margins into `sampled_margin_hist`
- compute final full logits only once, after pass `12`
- build sampled candidate set from final logits
- compute probe predictions for all passes
- expose optional `force_freeze_after_pass` and `force_freeze_mask` arguments for ablations only
- when forced freeze is active:
  - frozen tokens keep their last `mu`
  - frozen tokens skip further writer updates
  - active tokens may read frozen-token cache if `use_frozen_cache=True`
  - no sparse compaction; loop stays dense

Forward return:

- `final_logits`
- `aux` dict containing:
  - `mu_hist`
  - `pi_hist`
  - `sampled_ce_hist`
  - `sampled_sampled_margin_hist`
  - `probe_pred`
  - `compute_cost`
  - `avg_steps`

Exact accounting:

- `compute_cost = 12.0` for normal dense training
- for forced-freeze ablations, also log `active_token_pass_frac`, but do not claim FLOP savings

#### `def create_v060a(...)`

Versioned constructor, same pattern as `create_v054`.

#### `def warmstart_v060a_from_v054(checkpoint_path, ...)`

Purpose: parity harness only.

Behavior:

- load all matching shared-core weights from a `v0.5.4` checkpoint
- leave `ResidualGainProbe` and `FrozenPrefixCache` fresh
- never use this helper for production `v0.6.0a` training

#### `if __name__ == "__main__":`

Add self-tests:

- forward pass works at `max_steps=12`
- backward pass works
- `--parity_v054 CHECKPOINT` mode:
  - instantiate `v0.6.0a` with `max_steps=8`
  - disable probe loss path
  - disable forced freeze and cache
  - load shared weights from checkpoint
  - compare logits to `v0.5.4`
  - pass only if max diff `< 1e-6` in fp32

### 2. New file: `code/train_v060a.py`

Create a new versioned trainer from `train_v054.py`.

Exact training configuration:

- `MAX_STEPS_PER_POSITION = 12`
- same batch/seq defaults as `v0.5.4` unless memory forces a one-step reduction in batch
- same optimizer family: `AdamW`
- training starts from scratch on the diverse `17B+` token corpus
- no optimizer/state warm-start from `v0.5.4`

Implement these functions:

#### `def compute_v060a_losses(model, x, y)`

Inputs:

- `final_logits, aux = model(x, y=y)`

Compute three losses:

1. `L_final`

- full-vocab CE on final pass only

2. `L_step`

- weighted sampled CE over all 12 passes
- exact weights:
  - `[0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.00]`

3. `L_probe`

- target:
  - `target_p = max(0, sampled_ce_p - min_{q>p}(sampled_ce_q))`
  - final pass target is `0`
  - clamp target to `[0, 4]`
- loss:
  - `SmoothL1Loss(probe_pred, target)`

Exact total loss:

```text
L = L_final + 0.50 * L_step + 0.20 * L_probe
```

Return:

- total loss
- scalar metrics for logging

#### `def evaluate_final_bpt(model, dataset, n_batches=20)`

Same role as `v0.5.4`: final-pass validation BPT.

#### `def evaluate_pass_curve(model, dataset, n_tokens=50000)`

Purpose: measure real dense-12 behavior using **full-vocab** CE at every pass on a fixed validation slice.

Exact outputs:

- mean CE by pass
- easy/hard bucket CE by pass
- easy/hard bucket residual-after-8
- hard-token `8 -> 12` gain

Bucket definition:

- easy = bottom 30% tokens by pass-1 full CE
- hard = top 30% tokens by pass-1 full CE

#### `def evaluate_probe_calibration(model, dataset, n_tokens=50000)`

Compare `probe_pred` to full-CE residual target:

- positive label = `full_residual_gain > 0.05`
- metrics:
  - AUROC
  - AUPRC
  - Spearman correlation to full residual gain

Baseline:

- logistic regressor built from sampled margin and margin slope only

Proceed requires the probe to beat this baseline materially.

#### `def evaluate_sampled_fidelity(model, dataset, n_tokens=50000)`

Purpose: validate that sampled supervision tracks real CE well enough.

Metrics:

- Spearman correlation between sampled residual gain and full residual gain
- top-decile recall of sampled ranking against full residual-gain ranking

#### `def evaluate_forced_freeze_ablation(model, dataset, n_tokens=50000)`

Purpose: test frozen-context viability before any controller exists.

Exact protocol:

1. Run dense-12 eval and compute oracle-easy mask:
   - token is oracle-easy-at-4 if `full_ce_4 - min(full_ce_5..12) <= 0.02`
2. Rerun with `force_freeze_after_pass=4`
3. Compare three modes:
   - baseline dense-12
   - forced freeze without frozen cache
   - forced freeze with frozen cache

Metrics:

- final BPT delta vs baseline
- later-token BPT delta on tokens after frozen positions

#### `def evaluate_negative_drift(model, dataset, n_tokens=50000)`

Purpose: check whether final-pass negatives are stable enough for sampled training.

Metric:

- Jaccard overlap of top-32 wrong-token sets between pass `12` and passes `1..11`

This is diagnostic, not a primary proceed gate.

#### `def smoke_test_v060a(model, dataset, n_steps=500)`

Mandatory pre-training gate for the new version.

Pass conditions:

- no NaN/Inf
- final loss at step 500 lower than step 50
- pass-12 sampled CE <= pass-8 sampled CE on the held-out smoke slice

#### `def main()`

Training loop requirements:

- same resume/checkpoint pattern as `v0.5.4`, but under `results/checkpoints_v060a/`
- log:
  - `loss_final`
  - `loss_step`
  - `loss_probe`
  - `final_bpt`
  - `sampled_residual_mean`
  - `probe_auroc`
  - `hard_gain_8_to_12`

Evaluation cadence:

- every `1000` steps: final BPT
- every `5000` steps: pass curve, probe calibration, sampled fidelity, forced-freeze ablation

### 3. No changes in `v0.6.0a`

Do not modify these files in this phase:

- `code/sutra_v05_ssm.py`
- `code/scratchpad.py`
- `code/spectrum_memory.py`

Reason:

- graph migration is deferred
- scratchpad is already the stable baseline
- spectrum memory is a later branch

### 4. Results files to write after building

The implementation should produce these canonical artifacts:

- `results/experiment_v060a_smoke.json`
- `results/experiment_v060a_pass_curve.json`
- `results/experiment_v060a_probe_calibration.json`
- `results/experiment_v060a_sampled_fidelity.json`
- `results/experiment_v060a_forced_freeze.json`

## Exact Test Plan After Build

Run the following in order.

### Test 1. `v0.5.4` parity harness

Purpose: prove the new scaffold did not break the old core.

Method:

- load a `v0.5.4` checkpoint into `warmstart_v060a_from_v054`
- run both models at `8` passes in fp32 with all new paths disabled

Pass:

- max logit diff `< 1e-6`

Fail means:

- do not start `v0.6.0a` training until parity is fixed

### Test 2. 500-step dense-12 smoke test

Purpose: prove the 12-pass training path is numerically stable.

Pass:

- no NaN/Inf
- loss decreases
- pass-12 sampled CE beats pass-8 sampled CE on the smoke slice

Fail means:

- stop and fix training dynamics before long runs

### Test 3. Sampled-target fidelity

Purpose: validate the cheap training target.

Proceed threshold:

- Spearman `>= 0.70`
- top-decile recall `>= 0.80`

Stop threshold:

- Spearman `< 0.60` or top-decile recall `< 0.70`

If this fails, the readout-cost constraint blocks the whole branch.

### Test 4. Dense-12 stability and headroom

Purpose: verify that training-from-scratch removes the `8 -> 12` divergence.

Proceed threshold:

- hard-token mean CE at pass `12` is lower than pass `8`
- hard-token absolute gain `8 -> 12 >= 0.20`

Stop threshold:

- pass `12` CE >= pass `8` CE on hard tokens

If this fails, there is no reason to build a controller above 8 passes.

### Test 5. Convergence separation

Purpose: verify that easy and hard tokens separate under inter-step supervision.

Proceed threshold:

- easy-token residual-after-8 `<= 0.02`
- hard-token residual-after-8 `>= 0.10`
- gap between hard and easy residual-after-8 `>= 0.08`

Stop threshold:

- easy/hard residual-after-8 gap `< 0.04`

If this fails, the model is still converging uniformly and freezing will not help.

### Test 6. Probe calibration

Purpose: verify that the shadow probe is worth turning into a controller later.

Proceed threshold:

- AUROC `>= 0.78`
- AUPRC `>= 0.55`
- AUROC at least `+0.03` above the margin-only baseline

Stop threshold:

- AUROC `< 0.72`
- or AUROC improvement over baseline `< 0.01`

If this fails, do not build an acting controller.

### Test 7. Forced-freeze cache ablation

Purpose: verify that frozen tokens can remain useful context.

Proceed threshold:

- forced freeze **with** cache worsens final BPT by `<= 0.02` vs dense-12 baseline
- forced freeze with cache beats forced freeze without cache by at least `0.03` BPT

Stop threshold:

- forced freeze with cache worsens BPT by `> 0.05`

If this fails, do not build freeze/continue control until memory is redesigned.

## Proceed vs Stop

### Proceed to full controller only if all five core claims are true

1. sampled supervision is faithful enough to train on
2. dense-12 training is stable after pass 8
3. easy and hard tokens separate by residual-after-8
4. residual-gain probe beats trivial uncertainty baselines
5. frozen tokens remain useful context with the cache

### Stop the controller branch if any of these are false

- sampled target cannot track full residual CE
- 12-pass from-scratch model still diverges after pass 8
- convergence remains uniform
- probe adds no real signal over margin heuristics
- forced freezing breaks downstream context even with the cache

In that stop case, the conclusion is:

> The current Sutra recurrent core does not naturally support elastic compute control, even with inter-step supervision. Do not build the full controller yet.

## Implementation Time

Engineering time for the code only:

- `launch_v060a.py`: `3-4` hours
- `train_v060a.py`: `3-4` hours
- parity + smoke + eval wiring: `2` hours

Total implementation time:

- about `8-10` hours in one coding session

First useful empirical readout after implementation:

- parity + 500-step smoke: under `1` hour
- first `5k`-step signal: about `2-3` additional GPU hours at current throughput

## Warm-Start Answer

### We can warm-start the code, but not the training run

Use `v0.5.4` as the **implementation base**:

- same core modules
- same trainer skeleton
- same checkpoint/eval structure

Do **not** warm-start model weights for the real `v0.6.0a` experiment.

Reason:

- `v0.5.4` was trained for uniform `8` passes
- the dense-12 probe already showed `9-12` are out-of-distribution and diverge
- the scientific question is specifically whether **from-scratch inter-step supervision** induces natural separation

So the correct answer is:

- **code warm-start:** yes
- **weight warm-start for production v0.6.0a training:** no

## Final Definition of Success

`v0.6.0a` is successful if it gives a clear decision on whether `v0.6.x` controller work is worth doing.

That means success is not "lower BPT."

Success is one of these two outcomes:

- **Proceed:** dense-12 + inter-step training creates real easy/hard separation and a usable residual-gain probe
- **Stop:** even with the correct training setup, the core still converges uniformly or freezing remains harmful

Either outcome is valuable. The only bad outcome is building the controller before this probe is run.
