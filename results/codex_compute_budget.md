The right design is a hard resource accountant wrapped around the existing recurrent core, not another learned halting head. That fits the repo state: [launch_v054.py](C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/launch_v054.py#L126) still runs a fixed `max_steps` loop, [launch_v054.py](C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/launch_v054.py#L184) still returns `compute_cost=0`, [train_v054.py](C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/train_v054.py#L211) already has a trainer hook for compute cost, and [spectrum_memory.py](C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/spectrum_memory.py#L57) already has the continuous-scale read path plus a warm-start gate at [spectrum_memory.py](C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/spectrum_memory.py#L72).

**Answers**
1. Parameterize budget as a hybrid: fixed sequence budget plus per-token local stipend. Use `B_total(T) = T * B_local + B_shared * T`, with each token starting with `b_i = B_local` and one shared pool `R = B_shared * T`. This is better than pure per-token because easy tokens can finish early and hard tokens can consume reclaimed budget. Do not make budget learned at first. Use curriculum: warm-start with average budget equal to current v0.5.4 compute, then tighten later. Concretely, start with ceiling `max_steps_ceiling=8`, average budget `8.0` units/token; after verifier stabilizes, raise ceiling to `10-12` while keeping average budget `8.0`, so redistribution becomes real without raising average FLOPs.

2. Costs should be deterministic and convex for expensive actions. Initial cost schedule:
   - Base recurrent cycle: `c_base = 1.0`
   - Verify attempt: `c_verify = 0.25`
   - Extra routing width: `c_width = 0.25 * max(k_eff / k0 - 1, 0)`
   - Fine memory zoom: `c_zoom = alpha * sum_l kappa_l * 2^{-l}` for tree levels, because level `l=0` is fine and expensive while coarse levels are cheap
   - If you want a simpler first pass before tree-cost accounting, use `c_zoom = alpha * (1 - s)^2`
   Keep costs fixed, not learned. Learned costs will become another controller and recreate the halting failure mode.

3. Stage 7 should predict quality only; the stop/continue decision should be deterministic from quality and remaining budget. Add a verifier head `q_hat_i = sigmoid(f_verify(...))` trained to predict a smooth target like `exp(-CE_i_t)`. Then set an acceptance threshold that depends on remaining budget:
   `tau(b) = tau_lo + (tau_hi - tau_lo) * (b / b_ref)^gamma`
   If `q_hat >= tau(b)`, emit/freeze. If `q_hat < tau(b)` and enough budget remains, reroute to Stage 4. If budget is exhausted, emit whatever is there. More budget means stricter acceptance; low budget means tolerance rises. That is the hard budget controller. Also, use entropy and logit margin as verifier inputs; do not trust `lambda` alone, because repo notes already show lambda was anti-calibrated.

4. Continuous-spectrum memory fits naturally if budget enters the scale predictor. Extend the current scale net to
   `s_i = sigmoid(f_scale(mu_i, log_var_i, pi_i, budget_frac_i, verify_fail_i))`.
   Then:
   - high budget + low quality -> smaller `s_i` -> finer zoom -> higher `c_zoom`
   - low budget or near-accept -> larger `s_i` -> coarse gist read
   The nice part is that the “always fine” collapse is now dominated by budget cost, not just loss shaping. The tree already gives the right geometry for this.

5. Warm-start from v0.5.4, but do it in phases. Phase 1 should keep the existing [launch_v054.py](C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/launch_v054.py#L57) backbone, scratchpad, Peri-LN, and pheromone intact, and only add `BudgetState` plus `VerifyHead`. Phase 2 can swap in [spectrum_memory.py](C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/spectrum_memory.py#L34) using the existing gated residual warm-start design. I would not train the full budgeted-spectrum-reroute system from scratch unless you are also changing Stage 4 routing. The budget controller itself is a clean warm-startable refinement.

6. Yes, this is directly rate-distortion. Budget is the rate/compute constraint, token error is distortion, and the memory tree is a successive-refinement codebook. The ideal control law is:
   `continue if expected distortion drop / marginal cost > shadow_price_of_budget`
   Your practical implementation is a discrete approximation of that. Stage 7 becomes the local rate controller. Fine zoom is literally spending more rate to reduce distortion on hard positions.

7. Minimal implementation to test:
   - Keep `v0.5.4` architecture.
   - Add per-step draft logits and inter-step CE loss.
   - Add `VerifyHead`.
   - Add `BudgetAccountant`.
   - Allow per-token freezing when verify passes.
   - Keep `max_steps=8` at first so warm-start is gentle.
   - Return real `aux["compute_cost"]`, `aux["avg_steps"]`, `aux["verify_pass_rate"]`, `aux["budget_spent"]`.
   Train with:
   `L = L_final + lambda_step * L_intermediate + lambda_verify * BCE(q_hat, exp(-CE).detach()) + lambda_cost * mean(spent / B_total)`
   The inter-step loss is mandatory. Repo evidence already says post-hoc freezing hurts if the model was trained expecting all 8 steps.

8. This changes the adaptive-compute story completely relative to PonderNet/ACT. PonderNet learns a halting policy and regularizes it with a prior; that is meta-control. Your proposal does not learn when to stop. It learns only state quality, while stopping is a consequence of budget depletion plus a deterministic threshold. It is also broader than PonderNet because the same budget prices recurrence depth, routing width, and memory resolution. That is much closer to an anytime constrained inference system than a halting network.

**PyTorch Spec**
```python
# new state
budget = torch.full((B, T), B_local, device=x.device)
shared = torch.full((B,), B_shared * T, device=x.device)
done = torch.zeros(B, T, dtype=torch.bool, device=x.device)

for t in range(max_steps_ceiling):
    active = ~done
    if not active.any():
        break

    # existing stage evolution
    K = self.transition(mu)
    pi_evolved = ...
    stage_out, evidence = self.stage_bank(self.pre_bank_ln(mu), pi)
    pi = top2_project(normalize(pi_evolved * softmax(evidence / temp)))

    # budget-aware memory scale
    budget_frac = (budget / B_ref).clamp(0, 1)
    log_var = -torch.log(lam.clamp_min(1e-6))
    s = torch.sigmoid(self.scale_net(torch.cat([mu, log_var, pi, budget_frac.unsqueeze(-1)], dim=-1)))

    # route + write
    mem_ctx, kappa = self.memory.read(mu, lam, pi, tree, scratch, s)
    messages = self.router(mu) * pi[:, :, 3:4] + mem_ctx
    mu, lam = self.writer(mu, lam, messages, pi[:, :, 4:5])
    mu = mu + 0.1 * stage_out

    # draft readout + verify
    logits_t = F.linear(self.ln(mu), self.emb.weight) / math.sqrt(self.dim)
    entropy_t = categorical_entropy(logits_t.detach())
    margin_t = top1_minus_top2(logits_t.detach())
    q_hat = torch.sigmoid(self.verify_head(
        torch.cat([mu, log_var, pi, budget_frac.unsqueeze(-1),
                   entropy_t.unsqueeze(-1), margin_t.unsqueeze(-1)], dim=-1)
    )).squeeze(-1)

    # spend budget
    zoom_cost = alpha * (kappa * level_cost.view(1, 1, -1)).sum(dim=-1)
    step_cost = active.float() * (c_base + c_verify + zoom_cost)
    budget, shared = spend_budget(budget, shared, step_cost, active)

    # deterministic accept / reroute
    tau = tau_lo + (tau_hi - tau_lo) * budget_frac.pow(gamma)
    accept = active & ((q_hat >= tau) | (budget <= c_min_continue))
    done = done | accept
    pi = reroute_bias(pi, accept, q_hat, budget_frac)  # accepted -> stage 7, failed -> stage 4
```

**Implementation Order**
1. Patch [launch_v054.py](C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/launch_v054.py) to add `VerifyHead`, per-step logits, budget tensors, and real `compute_cost`.
2. Patch [train_v054.py](C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/train_v054.py) to add inter-step CE and verify calibration loss.
3. Only after that is stable, patch [spectrum_memory.py](C:/Users/devan/OneDrive/Desktop/Projects/AI%20Moonshots/moonshot-sutra/code/spectrum_memory.py) so `scale_net` consumes `budget_frac` and its read path reports zoom cost.
4. First experiment: warm-start from v0.5.4, keep average compute equal, measure whether easy-token freezing reduces spent budget without hurting BPT.
5. Second experiment: keep same average budget, raise step ceiling above 8, test whether hard tokens actually consume reclaimed compute and improve.

If you want, I can turn this into an exact patch plan against `launch_v054.py` and `train_v054.py` next.