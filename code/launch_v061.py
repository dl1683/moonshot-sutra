"""Launch v0.6.1: Pass-Conditioned Normalization (adaLN).

First warm-start step toward v0.7.0. ONE mechanism change from v0.6.0a:
  - 6x GatedLayerNorm replaced by PassConditionedNorm (adaLN wrapping
    existing GatedLayerNorm with pass-dependent scale/shift)

All other architecture unchanged. PassConditionedNorm initializes to
identity (gamma=1, beta=0), so at step 0 after warm-start this model
is functionally identical to v0.6.0a.

Hypothesis: Pass conditioning breaks the shared-parameter fixed-point
collapse that causes passes 0-7 to contribute almost nothing. TMLT
(2024) proves pass conditioning is mathematically necessary for
shared-weight recurrent models.

Warm-start: ADDITIVE — new parameters (pass_emb, to_gamma_beta)
initialized, existing weights map 1:1 from v0.6.0a.
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
from launch_v060a import (build_negative_set, sampled_pass_ce,
                           ResidualGainProbe, FrozenPrefixCache)


class PassConditionedNorm(nn.Module):
    """adaLN: GatedLayerNorm + pass-dependent scale/shift.

    Wraps an existing GatedLayerNorm for weight transfer. A tiny MLP
    maps pass_index -> (gamma, beta). Init: gamma=1, beta=0 (identity).
    """

    def __init__(self, dim, max_passes=12, base_norm=None):
        super().__init__()
        self.norm = base_norm if base_norm is not None else GatedLayerNorm(dim)
        self.pass_emb = nn.Embedding(max_passes, 64)
        self.to_gamma_beta = nn.Sequential(
            nn.Linear(64, 128), nn.SiLU(), nn.Linear(128, dim * 2)
        )
        # Init to identity: gamma=1, beta=0
        nn.init.zeros_(self.to_gamma_beta[-1].weight)
        nn.init.zeros_(self.to_gamma_beta[-1].bias)
        with torch.no_grad():
            self.to_gamma_beta[-1].bias[:dim] = 1.0

    def forward(self, x, pass_idx):
        """x: (B, T, D), pass_idx: int."""
        h = self.norm(x)
        p_emb = self.pass_emb(torch.tensor(pass_idx, device=x.device))
        gamma_beta = self.to_gamma_beta(p_emb)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return gamma * h + beta


class SutraV061(nn.Module):
    """v0.6.1: adaLN pass conditioning on v0.6.0a core.

    Same as v0.6.0a except 6 GatedLayerNorm -> PassConditionedNorm.
    Pass index p flows through every normalization layer.
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

        # v0.6.1: PassConditionedNorm (adaLN) instead of GatedLayerNorm
        self.pre_bank_pcn = PassConditionedNorm(dim, max_steps)
        self.post_bank_pcn = PassConditionedNorm(dim, max_steps)
        self.pre_route_pcn = PassConditionedNorm(dim, max_steps)
        self.post_route_pcn = PassConditionedNorm(dim, max_steps)
        self.pre_write_pcn = PassConditionedNorm(dim, max_steps)
        self.post_write_pcn = PassConditionedNorm(dim, max_steps)

        # Output
        self.ln = nn.LayerNorm(dim)

        # v0.6.0a additions (unchanged)
        self.gain_probe = ResidualGainProbe(dim)
        self.frozen_cache = FrozenPrefixCache(dim)

    def forward(self, x, y=None, force_freeze_after_pass=None,
                force_freeze_mask=None, use_frozen_cache=False,
                collect_history=None):
        B, T = x.shape
        device = x.device

        if collect_history is None:
            collect_history = y is not None and self.training

        # Init
        h = self.emb(x) + self.pos_emb(torch.arange(T, device=device))
        mu = self.init_mu(h)
        lam = F.softplus(self.init_lam(h)) + 0.1
        pi = torch.zeros(B, T, N_STAGES, device=device, dtype=h.dtype)
        pi[:, :, 2] = 1.0
        mem = self.scratchpad.init_memory(B, device)
        pheromone = torch.zeros(B, T, device=device, dtype=h.dtype)
        mu_prev = mu.detach()

        if collect_history:
            mu_hist = torch.zeros(B, T, self.max_steps, self.dim, device=device, dtype=h.dtype)
            pi_hist = torch.zeros(B, T, self.max_steps, N_STAGES, device=device, dtype=h.dtype)
        else:
            mu_hist = None
            pi_hist = None

        frozen = torch.zeros(B, T, dtype=torch.bool, device=device)
        cache_state = self.frozen_cache.init_state(B, T, device, h.dtype) if use_frozen_cache else None

        for p in range(self.max_steps):
            late_gate = 1.0 if p >= 3 else 0.0

            if force_freeze_after_pass is not None and p >= force_freeze_after_pass:
                if force_freeze_mask is not None:
                    frozen = frozen | force_freeze_mask
                    if use_frozen_cache and cache_state is not None:
                        cache_state = self.frozen_cache.write(mu.detach(), frozen, cache_state)

            # Stage transitions
            K = self.transition(mu)
            pi_ev = torch.bmm(
                pi.view(B * T, 1, N_STAGES),
                K.view(B * T, N_STAGES, N_STAGES)
            ).view(B, T, N_STAGES)

            # Peri-LN stage bank (PASS-CONDITIONED)
            mu_bank = self.pre_bank_pcn(mu, p)
            stage_out, evidence = self.stage_bank(mu_bank, pi)
            stage_out = self.post_bank_pcn(stage_out, p)

            pi_new = pi_ev * F.softmax(evidence / 0.5, dim=-1)
            pi_new = pi_new / pi_new.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            pi = top2_project(pi_new)

            # Peri-LN routing (PASS-CONDITIONED)
            mu_route = self.pre_route_pcn(mu, p)
            messages = self.router(mu_route) * pi[:, :, 3:4]
            messages = self.post_route_pcn(messages, p)

            # Scratchpad
            mem_ctx = self.scratchpad.read(mu, mem)
            messages = messages + 0.1 * mem_ctx * pi[:, :, 3:4]

            # Pheromone (still present — removal is v0.6.2)
            if late_gate > 0:
                phero_gate = torch.sigmoid(pheromone).unsqueeze(-1)
                messages = messages + 0.05 * phero_gate * mem_ctx * pi[:, :, 3:4]

            # Frozen cache (ablation only)
            if use_frozen_cache and cache_state is not None and frozen.any():
                cache_ctx = self.frozen_cache.read(mu, cache_state)
                messages = messages + 0.05 * cache_ctx * pi[:, :, 3:4]

            # Peri-LN writer (PASS-CONDITIONED)
            mu_write = self.pre_write_pcn(mu, p)
            mu_new, lam = self.writer(mu_write, lam, messages, pi[:, :, 4:5])
            mu_new = self.post_write_pcn(mu_new, p)
            mu_new = mu_new + stage_out * 0.1

            # Freeze
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

            if collect_history:
                mu_hist[:, :, p, :] = mu
                pi_hist[:, :, p, :] = pi

        # Final output
        final = self.ln(mu)
        logits = F.linear(final, self.emb.weight) / math.sqrt(self.dim)

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

            probe_preds = []
            prev_margin = torch.zeros(B, T, device=device)
            for pp in range(self.max_steps):
                margin_p = s_margin[:, :, pp].detach()
                margin_slope = margin_p - prev_margin if pp > 0 else torch.zeros_like(margin_p)
                mu_p_det = mu_hist[:, :, pp].detach()
                pi_p_det = pi_hist[:, :, pp].detach()
                mu_prev_det = (mu_hist[:, :, pp-1] if pp > 0 else mu_hist[:, :, 0]).detach()
                delta_mu = (mu_p_det - mu_prev_det).pow(2).mean(-1).sqrt()
                pass_frac = torch.full((B, T), pp / (self.max_steps - 1), device=device)
                pred = self.gain_probe(mu_p_det, pi_p_det,
                                       margin_p, margin_slope, delta_mu, pass_frac)
                probe_preds.append(pred)
                prev_margin = margin_p
            aux["probe_pred"] = torch.stack(probe_preds, dim=2)

        return logits, aux

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


def create_v061(**kwargs):
    defaults = dict(vocab_size=50257, dim=768, ff_dim=1536, max_steps=12,
                    window=4, k_retrieval=8)
    defaults.update(kwargs)
    return SutraV061(**defaults)


def warm_start_from_v060a(model_061, ckpt_path):
    """Load v0.6.0a checkpoint into v0.6.1 model.

    Weight mapping:
      v0.6.0a pre_bank_ln.* -> v0.6.1 pre_bank_pcn.norm.*
      (same for all 6 norms)
    New adaLN params (pass_emb, to_gamma_beta) keep identity init.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_060a = ckpt["model"] if "model" in ckpt else ckpt

    # Build remapped state dict
    new_state = {}
    norm_map = {
        "pre_bank_ln": "pre_bank_pcn.norm",
        "post_bank_ln": "post_bank_pcn.norm",
        "pre_route_ln": "pre_route_pcn.norm",
        "post_route_ln": "post_route_pcn.norm",
        "pre_write_ln": "pre_write_pcn.norm",
        "post_write_ln": "post_write_pcn.norm",
    }

    mapped = 0
    skipped = 0
    for key, val in state_060a.items():
        new_key = key
        for old_prefix, new_prefix in norm_map.items():
            if key.startswith(old_prefix + "."):
                suffix = key[len(old_prefix):]
                new_key = new_prefix + suffix
                break
        new_state[new_key] = val
        mapped += 1

    # Load with strict=False to allow new adaLN params to keep init values
    missing, unexpected = model_061.load_state_dict(new_state, strict=False)

    # Verify: missing should be ONLY the new adaLN params
    adaLN_params = {"pass_emb", "to_gamma_beta"}
    truly_missing = [k for k in missing
                     if not any(a in k for a in adaLN_params)]

    print(f"Warm-start from v0.6.0a:")
    print(f"  Mapped: {mapped} params")
    print(f"  New adaLN params (expected missing): {len(missing)}")
    if truly_missing:
        print(f"  WARNING: Unexpected missing keys: {truly_missing}")
    if unexpected:
        print(f"  WARNING: Unexpected keys: {unexpected}")

    step = ckpt.get("step", 0)
    best_bpt = ckpt.get("best_bpt", float("inf"))
    print(f"  Source checkpoint: step {step}, best BPT {best_bpt:.4f}")

    return {
        "step": step,
        "best_bpt": best_bpt,
        "optimizer": ckpt.get("optimizer", None),
        "metrics": ckpt.get("metrics", []),
    }


if __name__ == "__main__":
    print("Testing v0.6.1...")
    m = create_v061(dim=64, ff_dim=128, max_steps=12, window=2, k_retrieval=4)
    print(f"Params: {m.count_params():,}")

    # Param count comparison
    from launch_v060a import create_v060a
    m0 = create_v060a(dim=64, ff_dim=128, max_steps=12, window=2, k_retrieval=4)
    delta = m.count_params() - m0.count_params()
    print(f"v0.6.0a: {m0.count_params():,}, v0.6.1: {m.count_params():,}, delta: +{delta:,}")

    x = torch.randint(0, 50257, (2, 32))
    y = torch.randint(0, 50257, (2, 32))
    logits, aux = m(x, y=y)
    print(f"Forward: logits={logits.shape}")
    print(f"  mu_hist: {aux['mu_hist'].shape}")
    print(f"  sampled_ce_hist: {aux['sampled_ce_hist'].shape}")

    loss = F.cross_entropy(logits.reshape(-1, 50257), y.reshape(-1))
    loss.backward()
    print(f"Backward OK, loss={loss.item():.4f}")

    # Causality
    xa = torch.randint(0, 50257, (1, 16))
    xb = xa.clone(); xb[0, 8] = (xb[0, 8] + 1) % 50257
    with torch.no_grad():
        la, _ = m(xa); lb, _ = m(xb)
    diff = (la[0, :8] - lb[0, :8]).abs().max().item()
    print(f"Causality: diff={diff:.6f} {'PASS' if diff < 0.001 else 'INVESTIGATE'}")
    print("v0.6.1 OK")
