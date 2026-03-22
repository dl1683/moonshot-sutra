Minimal version: keep the current Bayesian proposal, but stop letting late passes irreversibly collapse it straight into `mu`. Add one local per-token correction trace `c`, plus a correction gate `z` and anti-correction gate `a`, both late-only and zero at birth.

1. **Math**

Current writer in [sutra_v05_ssm.py#L157](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/sutra_v05_ssm.py#L157) computes
```text
mu_prop = (lam * mu + alpha * m) / (lam + alpha)
alpha = pi_write * softplus(gain(...))
```
and overwrites `mu` directly. That is the destructive step.

Minimal reversible sidecar:
```text
delta_p   = mu_prop_p - mu_p

z_p       = late_p * pi_write_p * sigmoid(Wz[mu_p, msg_p] + b_z)
a_p       = late_p * sigmoid(Wa[mu_p, msg_p, c_p] + b_a)

c_{p+1}   = (1 - a_p) * c_p + z_p * delta_p
mu_{p+1}  = mu_prop_p - a_p * c_{p+1}
lam_{p+1} = lam_p + (1 - a_p) * z_p * alpha_p
```

Interpretation:
- `delta` is the writer’s proposed change.
- `c` stores explicit recent write mass instead of letting it vanish inside `mu`.
- `a` subtracts that stored write mass when later evidence says “don’t overwrite this token.”
- If `z=a=0`, behavior is exactly the current model.
- If `z=a=1` and `c=0`, then `mu_{p+1}=mu_p`: solved token gets no write.
- This is not exact whole-model invertibility. It is local no-erasure writing in Stage 5, which is the right minimum.

2. **Integration**

Use the current `BayesianWrite` as the proposal engine in [sutra_v05_ssm.py#L145](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/sutra_v05_ssm.py#L145). Add only:
- one extra token state `c` with shape `(B,T,D)`
- one scalar `z` head
- one scalar `a` head
- write stats: `delta_rms`, `trace_rms`, `z_mean`, `a_mean`, `committed_gain`

In [launch_v060a.py#L333](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/launch_v060a.py#L333), change the writer call from `(mu_new, lam)` to `(mu_new, lam, c, stats)`. Do not touch router, scratchpad, or stage bank. Keep `c` local to Stage 5 only. That preserves system coherence.

3. **Zero Influence At Birth**

This must obey the warm-start invariants in [RESEARCH.md#L5443](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/research/RESEARCH.md#L5443) and [RESEARCH.md#L5524](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/research/RESEARCH.md#L5524).

Use:
- `c = 0` at init
- `b_z = -7`, `b_a = -7`
- `late_p = 1[p >= 3]` to match current late-pass geometry in [RESEARCH.md#L6113](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/research/RESEARCH.md#L6113)

Before training, verify checkpoint equivalence:
- max logit diff old vs new `< 1e-6`

4. **Elastic Controller Connection**

This is where the design becomes clean.

The repo already wants “frozen tokens remain readable context but no longer consume active compute” in [RESEARCH.md#L5601](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/research/RESEARCH.md#L5601), and `launch_v060a.py` already has forced freeze plus frozen cache at [launch_v060a.py#L289](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/launch_v060a.py#L289) and [launch_v060a.py#L338](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/launch_v060a.py#L338).

Rule:
- if token freezes, set write to zero
- keep `mu` fixed
- zero `c`
- write frozen `mu` into the existing `FrozenPrefixCache`

So:
```text
if frozen_i:
    mu_i <- mu_i
    c_i  <- 0
    z_i  <- 0
```
Solved tokens stop paying recurrence. Unsolved tokens still get reversible refinement.

5. **CPU Probe At `dim=768`**

Minimal canary:
- start from current `v0.6.0a` checkpoint
- CPU only
- `dim=768`, `ff=1536`, `passes=12`, `bs=1`, `seq=96`
- fixed exact-indexed train/eval batches
- train only the new `z/a` heads for 300-500 steps
- keep all existing weights frozen

Arms:
1. control: current writer
2. correction-only: `z` trainable, `a=0`
3. correction+anti: `z,a` trainable

Metrics:
- `easy_drift = E[CE_final - CE_p | late, low future_gain]`
- `hard_gain = E[CE_p - min_{q>p} CE_q | late, high future_gain]`
- final BPT
- forced-freeze penalty using existing ablation path
- gate selectivity: `z_hard > z_easy`, `a_easy > a_hard`

This is the right canary because [results/workshop_v060_pretests.json#L4](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/results/workshop_v060_pretests.json#L4) already says late steps may overwrite good intermediates, and [RESEARCH.md#L5916](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/research/RESEARCH.md#L5916) says attached-history helps hard late tokens. The reversible writer should reduce the former without killing the latter.

6. **Kill Criteria**

Kill the branch if any of these hold:
1. zero-gate checkpoint equivalence fails
2. `easy_drift` is not reduced by at least 20% vs control
3. `hard_gain` drops by more than 5%
4. final BPT regresses by more than 0.2%
5. forced-freeze penalty is not better than control
6. no gate selectivity emerges: `z_hard <= z_easy` and `a_easy <= a_hard`

7. **Prior Art**

Inference from primary sources: borrow the bounded reversible-update principle locally in Stage 5, not full-network exact invertibility.

- RevNet: reversible additive coupling, good precedent for explicit recoverable updates  
  https://papers.nips.cc/paper/6816-the-reversible-residual-network-backpropagation-without-storing-activations
- i-ResNet: invertible residual maps via constrained residual blocks  
  https://proceedings.mlr.press/v97/behrmann19a.html
- Neural ODE: continuous-depth invertible flow framing, but too global/heavy for this first test  
  https://papers.nips.cc/paper/7892-neural-ordinary-differential-equations
- Hamiltonian Neural Networks: conservative/reversible dynamics as inspiration for storing change in an auxiliary state instead of erasing it  
  https://papers.nips.cc/paper/9672-hamiltonian-neural-networks
- Caveat source: exact invertible families have expressivity limits unless augmented, so do not make all of Sutra invertible  
  https://proceedings.mlr.press/v119/zhang20h.html

Net: Stage 5 gets one extra trace tensor and two tiny gates. No new routing path, no new controller head, no new memory bank. That is the right minimum.

I couldn’t append this into [RESEARCH.md](/C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/research/RESEARCH.md) because this session is read-only.