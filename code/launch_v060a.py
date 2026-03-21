"""Launch v0.6.0a: Probe-only dense-12 training scaffold.

NOT a controller. Tests one question:
  Does a model trained from scratch at 12 recurrent passes with inter-step
  supervision naturally develop convergence separation?

Built on v0.5.4 core. Changes:
  - 12 passes (not 8)
  - Per-pass mu/pi history collection
  - Sampled inter-step CE (K=32 negatives from final logits)
  - Shadow ResidualGainProbe (predicts future improvement, doesn't act)
  - Optional FrozenPrefixCache for ablations only

Conventions (from 8 Chrome rounds):
  - Pass index: p in {0..11}, 0-based
  - mu_hist shape: (B, T, 12, D)
  - sampled_ce_hist shape: (B, T, 12)
  - "pass 8" = p=7, final pass = p=11

Approved for implementation after 8 design review rounds.
"""

import sys, math
from pathlib import Path
REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "code"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from sutra_v05_ssm import (N_STAGES, STAGE_GRAPH, top2_project,
                            SwitchingKernel2, StageBank, BayesianWrite,
                            LocalRouter)
from scratchpad import Scratchpad
from launch_v054 import GatedLayerNorm


class ResidualGainProbe(nn.Module):
    """Shadow probe: predicts future residual improvement. Does NOT act.

    Inputs per token at pass p:
      mu_p (projected to 128), pi_p (7), sampled_margin (1),
      margin_slope (1), delta_mu_rms (1), pass_fraction (1)

    Output: r_hat_p (scalar) = predicted residual gain from pass p onward.
    """

    def __init__(self, dim):
        super().__init__()
        self.mu_proj = nn.Linear(dim, 128)
        # 128 (mu) + 7 (pi) + 4 (margin, slope, delta_mu, pass_frac) = 139
        self.net = nn.Sequential(
            nn.Linear(139, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )

    def forward(self, mu, pi, margin, margin_slope, delta_mu_rms, pass_frac):
        """All inputs: (B, T) or (B, T, D). Returns: (B, T)."""
        mu_feat = self.mu_proj(mu)  # (B, T, 128)
        scalars = torch.stack([margin, margin_slope, delta_mu_rms, pass_frac], dim=-1)  # (B, T, 4)
        x = torch.cat([mu_feat, pi, scalars], dim=-1)  # (B, T, 139)
        return self.net(x).squeeze(-1)  # (B, T)


class FrozenPrefixCache(nn.Module):
    """Test whether frozen tokens can remain useful context.

    Stores projected K/V per token: (B, T, mem_dim) each.
    Causal reads: query position t attends only to frozen positions j < t.
    Used ONLY in forced-freeze ablations.
    """

    def __init__(self, dim, mem_dim=192):
        super().__init__()
        self.mem_dim = mem_dim
        self.k_proj = nn.Linear(dim, mem_dim)
        self.v_proj = nn.Linear(dim, mem_dim)
        self.q_proj = nn.Linear(dim, mem_dim)
        self.out_proj = nn.Linear(mem_dim, dim)

    def init_state(self, B, T, device, dtype):
        return {
            "k": torch.zeros(B, T, self.mem_dim, device=device, dtype=dtype),
            "v": torch.zeros(B, T, self.mem_dim, device=device, dtype=dtype),
            "frozen": torch.zeros(B, T, dtype=torch.bool, device=device),
        }

    def write(self, mu, freeze_mask, state):
        """Capture K/V for newly frozen tokens."""
        newly_frozen = freeze_mask & ~state["frozen"]
        if newly_frozen.any():
            k = self.k_proj(mu)
            v = self.v_proj(mu)
            state["k"] = torch.where(newly_frozen.unsqueeze(-1), k, state["k"])
            state["v"] = torch.where(newly_frozen.unsqueeze(-1), v, state["v"])
            state["frozen"] = state["frozen"] | freeze_mask
        return state

    def read(self, mu, state):
        """Active tokens attend causally to frozen tokens."""
        B, T, D = mu.shape
        q = self.q_proj(mu)  # (B, T, mem_dim)
        scores = torch.bmm(q, state["k"].transpose(1, 2)) / math.sqrt(self.mem_dim)

        # Causal mask: can only read frozen positions j < t
        causal = torch.triu(torch.ones(T, T, device=mu.device) * float("-inf"), diagonal=1)
        frozen_mask = (~state["frozen"]).unsqueeze(1).expand(B, T, T).float() * float("-inf")
        scores = scores + causal.unsqueeze(0) + frozen_mask

        attn = F.softmax(scores, dim=-1)
        ctx = torch.bmm(attn, state["v"])
        return self.out_proj(ctx)


def build_negative_set(final_logits, targets, k=32):
    """Get cheap per-pass supervision without per-pass full-vocab logits.

    Returns: (B, T, 33) candidate ids with target in slot 0.
    Fully vectorized — no Python loops over B*T.
    """
    B, T, V = final_logits.shape
    device = final_logits.device

    # Top-(k+8) from final logits (extra buffer to filter targets)
    topk_n = min(k + 8, V)  # Guard against small vocab
    top_ids = final_logits.topk(topk_n, dim=-1).indices  # (B, T, topk_n)

    # Mask out the target from top_ids
    target_expanded = targets.unsqueeze(-1)  # (B, T, 1)
    not_target = top_ids != target_expanded  # (B, T, k+8)

    # Gather first k non-target indices per position
    # Use cumsum on the mask to find the first k valid entries
    valid_cumsum = not_target.long().cumsum(dim=-1)  # (B, T, k+8)
    # valid_cumsum[b,t,i] = how many non-target entries in top_ids[b,t,:i+1]
    # We want entries where valid_cumsum <= k AND not_target is True
    take_mask = (valid_cumsum <= k) & not_target  # (B, T, k+8)

    # Extract the selected negative ids
    # Flatten and use masked_select, then reshape
    neg_ids = top_ids.masked_fill(~take_mask, 0)  # zero out non-selected

    # Pack into exactly k negatives per position using a gather approach
    # Count how many we got from top-k filtering
    n_selected = take_mask.sum(dim=-1)  # (B, T)

    # Build candidates: slot 0 = target, slots 1..k = negatives
    candidates = torch.zeros(B, T, k + 1, dtype=torch.long, device=device)
    candidates[:, :, 0] = targets

    # For each position, gather the first k masked entries
    # Use argsort of ~take_mask to push True entries to the front
    sort_idx = (~take_mask).long().argsort(dim=-1, stable=True)  # True (selected) first
    sorted_ids = top_ids.gather(-1, sort_idx)
    candidates[:, :, 1:] = sorted_ids[:, :, :k]

    # Fill any remaining slots (if top-k didn't have k non-target entries) with random
    n_short = (k - n_selected).clamp(min=0)  # (B, T)
    if n_short.any():
        max_short = n_short.max().item()
        if max_short > 0:
            random_fill = torch.randint(0, V, (B, T, max_short), device=device)
            for i in range(max_short):
                fill_pos = k - max_short + i + 1  # position in candidates
                mask = i < n_short  # (B, T) which positions need filling
                if mask.any() and fill_pos <= k:
                    candidates[:, :, fill_pos] = torch.where(mask, random_fill[:, :, i], candidates[:, :, fill_pos])

    return candidates


def sampled_pass_ce(mu_hist, ln, emb_weight, dim, candidates):
    """Compute sampled CE and margin for all passes.

    Args:
        mu_hist: (B, T, P, D) — hidden states at each pass
        ln: LayerNorm module
        emb_weight: (V, D) embedding weight matrix
        dim: model dimension
        candidates: (B, T, 33) candidate ids

    Returns:
        sampled_ce_hist: (B, T, P)
        sampled_margin_hist: (B, T, P)
    """
    B, T, P, D = mu_hist.shape
    K_plus_1 = candidates.size(-1)  # 33

    # Gather embedding rows for candidates: (B, T, 33, D)
    cand_emb = emb_weight[candidates.view(-1)].view(B, T, K_plus_1, D)

    ce_list = []
    margin_list = []
    for p in range(P):
        mu_p = ln(mu_hist[:, :, p, :])  # (B, T, D)
        # Score against candidates: (B, T, 33)
        scores = torch.einsum("btd,btcd->btc", mu_p, cand_emb) / math.sqrt(dim)
        # CE where target is slot 0
        ce = F.cross_entropy(scores.reshape(-1, K_plus_1),
                             torch.zeros(B * T, dtype=torch.long, device=mu_p.device),
                             reduction="none").reshape(B, T)
        # Margin: target score minus best negative
        margin = scores[:, :, 0] - scores[:, :, 1:].max(dim=-1).values
        ce_list.append(ce)
        margin_list.append(margin)

    return torch.stack(ce_list, dim=2), torch.stack(margin_list, dim=2)


class SutraV060a(nn.Module):
    """v0.6.0a: Probe-only dense-12 scaffold on v0.5.4 core.

    12 recurrent passes. Collects per-pass history. Shadow gain probe.
    Optional forced-freeze ablation. No acting controller.
    """

    def __init__(self, vocab_size=50257, dim=768, ff_dim=1536,
                 max_steps=12, window=4, k_retrieval=8, n_scratch_slots=8,
                 pheromone_rho=0.90):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.max_steps = max_steps
        self.pheromone_rho = pheromone_rho

        # v0.5.4 core (unchanged)
        self.emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(2048, dim)
        self.init_mu = nn.Linear(dim, dim)
        self.init_lam = nn.Linear(dim, dim)
        self.transition = SwitchingKernel2(dim)
        self.stage_bank = StageBank(dim, ff_dim)
        self.router = LocalRouter(dim, window=window, k=k_retrieval)
        self.writer = BayesianWrite(dim)
        self.scratchpad = Scratchpad(dim, n_slots=n_scratch_slots)

        # Gated Peri-LN (from v0.5.4)
        self.pre_bank_ln = GatedLayerNorm(dim)
        self.post_bank_ln = GatedLayerNorm(dim)
        self.pre_route_ln = GatedLayerNorm(dim)
        self.post_route_ln = GatedLayerNorm(dim)
        self.pre_write_ln = GatedLayerNorm(dim)
        self.post_write_ln = GatedLayerNorm(dim)

        # Output
        self.ln = nn.LayerNorm(dim)

        # v0.6.0a additions
        self.gain_probe = ResidualGainProbe(dim)
        self.frozen_cache = FrozenPrefixCache(dim)

    def forward(self, x, y=None, force_freeze_after_pass=None,
                force_freeze_mask=None, use_frozen_cache=False,
                collect_history=None):
        B, T = x.shape
        device = x.device

        # Auto-detect: collect history only when training with targets
        if collect_history is None:
            collect_history = y is not None and self.training

        # Init (same as v0.5.4)
        h = self.emb(x) + self.pos_emb(torch.arange(T, device=device))
        mu = self.init_mu(h)
        lam = F.softplus(self.init_lam(h)) + 0.1
        pi = torch.zeros(B, T, N_STAGES, device=device, dtype=h.dtype)
        pi[:, :, 2] = 1.0
        mem = self.scratchpad.init_memory(B, device)
        pheromone = torch.zeros(B, T, device=device, dtype=h.dtype)
        mu_prev = mu.detach()

        # History collection (skip during eval to save ~36MB)
        if collect_history:
            mu_hist = torch.zeros(B, T, self.max_steps, self.dim, device=device, dtype=h.dtype)
            pi_hist = torch.zeros(B, T, self.max_steps, N_STAGES, device=device, dtype=h.dtype)
        else:
            mu_hist = None
            pi_hist = None

        # Frozen state (for ablations only)
        frozen = torch.zeros(B, T, dtype=torch.bool, device=device)
        cache_state = self.frozen_cache.init_state(B, T, device, h.dtype) if use_frozen_cache else None

        for p in range(self.max_steps):
            late_gate = 1.0 if p >= 3 else 0.0

            # Check forced freeze
            if force_freeze_after_pass is not None and p >= force_freeze_after_pass:
                if force_freeze_mask is not None:
                    newly_frozen = force_freeze_mask & ~frozen
                    frozen = frozen | force_freeze_mask
                    if use_frozen_cache and cache_state is not None:
                        cache_state = self.frozen_cache.write(mu.detach(), frozen, cache_state)

            # Stage transitions
            K = self.transition(mu)
            pi_ev = torch.bmm(
                pi.view(B * T, 1, N_STAGES),
                K.view(B * T, N_STAGES, N_STAGES)
            ).view(B, T, N_STAGES)

            # Peri-LN stage bank
            mu_bank = self.pre_bank_ln(mu)
            stage_out, evidence = self.stage_bank(mu_bank, pi)
            stage_out = self.post_bank_ln(stage_out)

            pi_new = pi_ev * F.softmax(evidence / 0.5, dim=-1)
            pi_new = pi_new / pi_new.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            pi = top2_project(pi_new)

            # Peri-LN routing
            mu_route = self.pre_route_ln(mu)
            messages = self.router(mu_route) * pi[:, :, 3:4]
            messages = self.post_route_ln(messages)

            # Scratchpad injection
            mem_ctx = self.scratchpad.read(mu, mem)
            messages = messages + 0.1 * mem_ctx * pi[:, :, 3:4]

            # Pheromone (generalized to variable passes)
            if late_gate > 0:
                phero_gate = torch.sigmoid(pheromone).unsqueeze(-1)
                messages = messages + 0.05 * phero_gate * mem_ctx * pi[:, :, 3:4]

            # Frozen cache read (ablation only)
            if use_frozen_cache and cache_state is not None and frozen.any():
                cache_ctx = self.frozen_cache.read(mu, cache_state)
                messages = messages + 0.05 * cache_ctx * pi[:, :, 3:4]

            # Peri-LN writer
            mu_write = self.pre_write_ln(mu)
            mu_new, lam = self.writer(mu_write, lam, messages, pi[:, :, 4:5])
            mu_new = self.post_write_ln(mu_new)
            mu_new = mu_new + stage_out * 0.1

            # Apply freeze: frozen tokens keep their state
            if frozen.any():
                mu = torch.where(frozen.unsqueeze(-1), mu, mu_new)
            else:
                mu = mu_new

            # Scratchpad update
            mem = self.scratchpad.write(mu.detach(), mem.detach(), pi[:, :, 4:5].detach())

            # Pheromone update
            with torch.no_grad():
                delta_mag = (mu - mu_prev).pow(2).mean(dim=-1).sqrt()
                deposit = pi[:, :, 4].detach() * torch.tanh(delta_mag)
                pheromone = self.pheromone_rho * pheromone + late_gate * deposit
            mu_prev = mu.detach()

            # Collect history (skip during eval)
            if collect_history:
                # Keep history attached during training so inter-step losses
                # can backprop into the recurrent core.
                mu_hist[:, :, p, :] = mu
                pi_hist[:, :, p, :] = pi

        # Final output (full logits once)
        final = self.ln(mu)
        logits = F.linear(final, self.emb.weight) / math.sqrt(self.dim)

        # Build sampled targets from final logits
        aux = {
            "mu_hist": mu_hist,
            "pi_hist": pi_hist,
            "compute_cost": torch.tensor(float(self.max_steps), device=device),
            "avg_steps": self.max_steps,
        }

        if y is not None and collect_history:
            candidates = build_negative_set(logits.detach(), y)
            s_ce, s_margin = sampled_pass_ce(mu_hist, self.ln, self.emb.weight, self.dim, candidates)
            aux["sampled_ce_hist"] = s_ce
            aux["sampled_margin_hist"] = s_margin

            # Probe predictions for all passes
            probe_preds = []
            prev_margin = torch.zeros(B, T, device=device)
            prev_mu_norm = torch.zeros(B, T, device=device)
            for p in range(self.max_steps):
                margin_p = s_margin[:, :, p].detach()
                margin_slope = margin_p - prev_margin if p > 0 else torch.zeros_like(margin_p)
                # Detach probe inputs so L_probe is truly shadow-only (doesn't train recurrent core)
                mu_p_det = mu_hist[:, :, p].detach()
                pi_p_det = pi_hist[:, :, p].detach()
                mu_prev_det = (mu_hist[:, :, p-1] if p > 0 else mu_hist[:, :, 0]).detach()
                delta_mu = (mu_p_det - mu_prev_det).pow(2).mean(-1).sqrt()
                pass_frac = torch.full((B, T), p / (self.max_steps - 1), device=device)
                pred = self.gain_probe(mu_p_det, pi_p_det,
                                       margin_p, margin_slope, delta_mu, pass_frac)
                probe_preds.append(pred)
                prev_margin = margin_p
            aux["probe_pred"] = torch.stack(probe_preds, dim=2)  # (B, T, 12)

        return logits, aux

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


def create_v060a(**kwargs):
    defaults = dict(vocab_size=50257, dim=768, ff_dim=1536, max_steps=12,
                    window=4, k_retrieval=8)
    defaults.update(kwargs)
    return SutraV060a(**defaults)


if __name__ == "__main__":
    print("Testing v0.6.0a...")
    m = create_v060a(dim=64, ff_dim=128, max_steps=12, window=2, k_retrieval=4)
    print(f"Params: {m.count_params():,}")

    x = torch.randint(0, 50257, (2, 32))
    y = torch.randint(0, 50257, (2, 32))
    logits, aux = m(x, y=y)
    print(f"Forward: logits={logits.shape}")
    print(f"  mu_hist: {aux['mu_hist'].shape}")
    print(f"  sampled_ce_hist: {aux['sampled_ce_hist'].shape}")
    print(f"  sampled_margin_hist: {aux['sampled_margin_hist'].shape}")
    print(f"  probe_pred: {aux['probe_pred'].shape}")

    # Backward
    loss = F.cross_entropy(logits.reshape(-1, 50257), y.reshape(-1))
    loss.backward()
    print(f"Backward OK, loss={loss.item():.4f}")
    print(f"NaN: {any(torch.isnan(p.grad).any() for p in m.parameters() if p.grad is not None)}")

    # Causality
    xa = torch.randint(0, 50257, (1, 16))
    xb = xa.clone(); xb[0, 8] = (xb[0, 8] + 1) % 50257
    with torch.no_grad():
        la, _ = m(xa); lb, _ = m(xb)
    diff = (la[0, :8] - lb[0, :8]).abs().max().item()
    print(f"Causality: diff={diff:.6f} {'PASS' if diff < 0.001 else 'INVESTIGATE'}")
    print("v0.6.0a OK")
