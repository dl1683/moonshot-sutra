"""Launch v0.6.1: Controller-Local Canary.

Warm-start from v0.6.0b step 3000. Tests one hypothesis:
  Does replacing the pass-global mode_gate with a tokenwise factorized
  controller + zero-init pass adapters produce content-dependent routing?

Changes from v0.6.0a/b:
  - ControlledTransition replaces SwitchingKernel2 (removes mode_gate + mode_bias)
  - TokenwiseController: per-token, per-pass with bilinear content x pass interaction
  - 12 rank-8 zero-init PassAdapters (one per pass)
  - Everything else unchanged from parent

Per R8+R9 spec (Tesla+Leibniz rounds 8-9):
  - Controller input: [LN(mu); stopgrad(emb[x]); mean(log lam); std(log lam); ||dmu||; p/(P-1)]
  - Bilinear interaction: content x pass (the core novelty)
  - 4 modes map to 7 stage-bias logits via W_ms (4->7)
  - Pass adapters: g_p * W_up[p] * W_down[p] * LN(mu), rank 8, zero-init

Supersedes the earlier adaLN design — R7/R8 identified mode_gate as root cause,
not LayerNorm. +329K params (+0.48%).

Success criteria (500 steps):
  - BPT <= parent + 0.05
  - MI(mode, token_class) >= 0.10
  - late_verify_share(passes 8-11) <= 0.75
  - MI(mode, pass_index) <= 0.40
  - D7-D9 frontier within 0.03 BPT of parent
"""

import sys, math
from pathlib import Path
REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "code"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from sutra_v05_ssm import (N_STAGES, STAGE_GRAPH, top2_project,
                            StageBank, BayesianWrite, LocalRouter)
from scratchpad import Scratchpad
from launch_v054 import GatedLayerNorm
from launch_v060a import (ResidualGainProbe, FrozenPrefixCache,
                           build_negative_set, sampled_pass_ce)

N_MODES = 4  # compose, route, retrieve, verify
PASS_EMBED_DIM = 32
CTRL_HIDDEN = 128


class TokenwiseController(nn.Module):
    """Per-token, per-pass factorized controller with bilinear content x pass.

    Replaces SwitchingKernel2's sequence-global mode_gate(h.mean(dim=1)).
    Root cause fix: the old controller washed out per-token content by mean-pooling.

    Architecture (R8 spec):
      r_t^p = [LN(mu); stopgrad(emb[x]); mean(log lam); std(log lam); ||dmu||; p/(P-1)]
      c_t^p = W2 SiLU(W1 r_t^p)         # input_dim -> 128 -> 128
      u_p   = E_pass[p]                  # 32-dim pass embedding

      ell_{t,p,m} = a_m^T c + b_m^T u + c^T U_m u + beta_m    (m in {0..3})
      alpha_{t,p} = softmax(ell / tau)   # 4-mode distribution

      stage_bias = alpha @ W_ms          # (4,) -> (7,) stage bias logits
    """

    def __init__(self, dim, max_passes=12):
        super().__init__()
        self.dim = dim
        self.max_passes = max_passes

        # Input: LN(mu)[dim] + stopgrad(emb[x])[dim] + 4 scalars = 2*dim+4
        self.input_ln = nn.LayerNorm(dim)
        input_dim = dim * 2 + 4  # mu + emb + (mean_log_lam, std_log_lam, delta_mu_rms, pass_frac)

        # Content encoder
        self.content_mlp = nn.Sequential(
            nn.Linear(input_dim, CTRL_HIDDEN),
            nn.SiLU(),
            nn.Linear(CTRL_HIDDEN, CTRL_HIDDEN),
        )

        # Pass embedding
        self.pass_embed = nn.Embedding(max_passes, PASS_EMBED_DIM)

        # Mode logit components
        self.content_head = nn.Linear(CTRL_HIDDEN, N_MODES)       # a_m^T c
        self.pass_head = nn.Linear(PASS_EMBED_DIM, N_MODES)       # b_m^T u
        # Bilinear: c^T U_m u — implemented as c -> (PASS_EMBED_DIM * N_MODES) then dot with u
        self.bilinear_proj = nn.Linear(CTRL_HIDDEN, PASS_EMBED_DIM * N_MODES, bias=False)
        self.mode_bias = nn.Parameter(torch.zeros(N_MODES))       # beta_m

        # Mode -> stage bias (4 -> 7)
        self.mode_to_stage = nn.Linear(N_MODES, N_STAGES, bias=False)

        self._init_weights()

    def _init_weights(self):
        """Near-uniform init: minimal disruption at step 0."""
        nn.init.zeros_(self.bilinear_proj.weight)
        nn.init.normal_(self.content_head.weight, std=0.01)
        nn.init.zeros_(self.content_head.bias)
        nn.init.normal_(self.pass_head.weight, std=0.01)
        nn.init.zeros_(self.pass_head.bias)
        nn.init.normal_(self.mode_to_stage.weight, std=0.01)

    def forward(self, mu, emb_x, lam, mu_prev, pass_idx):
        """
        Args:
            mu: (B, T, D) current hidden state
            emb_x: (B, T, D) stopgrad token embeddings
            lam: (B, T, D) precision
            mu_prev: (B, T, D) previous pass state
            pass_idx: int (0-indexed)
        Returns:
            alpha: (B, T, N_MODES) mode distribution
            stage_bias: (B, T, N_STAGES) stage bias logits
        """
        B, T, D = mu.shape

        mu_normed = self.input_ln(mu)

        # Scalar features
        log_lam = torch.log(lam.clamp(min=1e-6))
        mean_log_lam = log_lam.mean(dim=-1, keepdim=True)
        std_log_lam = log_lam.std(dim=-1, keepdim=True).clamp(min=1e-6)
        delta_mu_rms = ((mu - mu_prev).pow(2).mean(dim=-1, keepdim=True) + 1e-8).sqrt()
        pass_frac = torch.full((B, T, 1), pass_idx / max(self.max_passes - 1, 1),
                               device=mu.device, dtype=mu.dtype)

        # Assemble input
        r = torch.cat([mu_normed, emb_x.detach(),
                        mean_log_lam, std_log_lam, delta_mu_rms, pass_frac], dim=-1)
        c = self.content_mlp(r)  # (B, T, 128)

        # Pass embedding
        u = self.pass_embed(torch.tensor(pass_idx, device=mu.device, dtype=torch.long))  # (32,)

        # Mode logits
        content_logits = self.content_head(c)       # (B, T, 4)
        pass_logits = self.pass_head(u)             # (4,)
        bilinear = self.bilinear_proj(c).view(B, T, N_MODES, PASS_EMBED_DIM)
        bilinear_logits = (bilinear * u).sum(dim=-1)  # (B, T, 4)

        ell = content_logits + pass_logits + bilinear_logits + self.mode_bias
        alpha = F.softmax(ell, dim=-1)  # (B, T, 4)

        stage_bias = self.mode_to_stage(alpha)  # (B, T, 7)
        return alpha, stage_bias


class ControlledTransition(nn.Module):
    """Content-dependent transition kernel with tokenwise controller.

    Replaces SwitchingKernel2. Inherits base MLP, replaces mode_gate.
    """

    def __init__(self, dim, hidden=256):
        super().__init__()
        self.base = nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(),
                                   nn.Linear(hidden, N_STAGES * N_STAGES))
        self.controller = TokenwiseController(dim)

    def forward(self, h, emb_x, lam, mu_prev, pass_idx):
        """Returns K: (B,T,7,7), alpha: (B,T,4)."""
        B, T, D = h.shape
        base = self.base(h).view(B, T, N_STAGES, N_STAGES)
        alpha, stage_bias = self.controller(h, emb_x, lam, mu_prev, pass_idx)
        raw = base + stage_bias.unsqueeze(2)  # (B,T,1,7) broadcast over rows
        mask = STAGE_GRAPH.to(h.device).unsqueeze(0).unsqueeze(0)
        K = F.softmax(raw.masked_fill(mask == 0, float('-inf')), dim=-1)
        return K, alpha


class PassAdapter(nn.Module):
    """Zero-init rank-8 per-pass adapter. No-op at step 0.

    adapter_p(x) = g_p * W_up * W_down * LN(x)
    g_p=0, W_up=0 at init -> exact identity.
    """

    def __init__(self, dim, rank=8):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        self.gate = nn.Parameter(torch.zeros(1))
        # NOTE: up.weight must NOT be zero-init when gate is zero-init.
        # Both zero -> dead gradient (neither gets signal because the other is zero).
        # Small random up + zero gate = small random output initially, but gate grad flows.
        nn.init.normal_(self.up.weight, std=0.001)
        nn.init.normal_(self.down.weight, std=0.02)

    def forward(self, x):
        return self.gate * self.up(self.down(self.ln(x)))


class SutraV061(nn.Module):
    """v0.6.1: Controller-Local Canary on v0.6.0a/b core.

    Changes: ControlledTransition + PassAdapters. Everything else unchanged.
    """

    def __init__(self, vocab_size=50257, dim=768, ff_dim=1536,
                 max_steps=12, window=4, k_retrieval=8, n_scratch_slots=8,
                 pheromone_rho=0.90):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.max_steps = max_steps
        self.pheromone_rho = pheromone_rho

        # Core (from v0.5.4, unchanged)
        self.emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(2048, dim)
        self.init_mu = nn.Linear(dim, dim)
        self.init_lam = nn.Linear(dim, dim)
        self.stage_bank = StageBank(dim, ff_dim)
        self.router = LocalRouter(dim, window=window, k=k_retrieval)
        self.writer = BayesianWrite(dim)
        self.scratchpad = Scratchpad(dim, n_slots=n_scratch_slots)

        # v0.6.1: NEW — ControlledTransition replaces SwitchingKernel2
        self.transition = ControlledTransition(dim)

        # v0.6.1: NEW — per-pass adapters
        self.pass_adapters = nn.ModuleList([PassAdapter(dim) for _ in range(max_steps)])

        # Gated Peri-LN (from v0.5.4, unchanged)
        self.pre_bank_ln = GatedLayerNorm(dim)
        self.post_bank_ln = GatedLayerNorm(dim)
        self.pre_route_ln = GatedLayerNorm(dim)
        self.post_route_ln = GatedLayerNorm(dim)
        self.pre_write_ln = GatedLayerNorm(dim)
        self.post_write_ln = GatedLayerNorm(dim)

        # Output
        self.ln = nn.LayerNorm(dim)

        # v0.6.0a additions (kept)
        self.gain_probe = ResidualGainProbe(dim)
        self.frozen_cache = FrozenPrefixCache(dim)

    def forward(self, x, y=None, n_steps=None, collect_history=None):
        B, T = x.shape
        device = x.device

        if n_steps is None:
            n_steps = self.max_steps
        if collect_history is None:
            collect_history = y is not None and self.training

        # Init
        h = self.emb(x) + self.pos_emb(torch.arange(T, device=device))
        emb_x = self.emb(x).detach()  # stopgrad embeddings for controller

        mu = self.init_mu(h)
        lam = F.softplus(self.init_lam(h)) + 0.1
        pi = torch.zeros(B, T, N_STAGES, device=device, dtype=h.dtype)
        pi[:, :, 2] = 1.0
        mem = self.scratchpad.init_memory(B, device)
        pheromone = torch.zeros(B, T, device=device, dtype=h.dtype)
        mu_prev = mu.detach()

        # History
        if collect_history:
            mu_hist = torch.zeros(B, T, n_steps, self.dim, device=device, dtype=h.dtype)
            pi_hist = torch.zeros(B, T, n_steps, N_STAGES, device=device, dtype=h.dtype)
        else:
            mu_hist = pi_hist = None

        # Alpha history (for L_collapse and diagnostics)
        alpha_hist = [] if self.training or collect_history else None

        for p in range(n_steps):
            late_gate = 1.0 if p >= 3 else 0.0

            # v0.6.1: Controlled transition
            K, alpha = self.transition(mu, emb_x, lam, mu_prev, p)
            pi_ev = torch.bmm(
                pi.view(B * T, 1, N_STAGES),
                K.view(B * T, N_STAGES, N_STAGES)
            ).view(B, T, N_STAGES)

            if alpha_hist is not None:
                alpha_hist.append(alpha)

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

            # Scratchpad
            mem_ctx = self.scratchpad.read(mu, mem)
            messages = messages + 0.1 * mem_ctx * pi[:, :, 3:4]

            # Pheromone
            if late_gate > 0:
                phero_gate = torch.sigmoid(pheromone).unsqueeze(-1)
                messages = messages + 0.05 * phero_gate * mem_ctx * pi[:, :, 3:4]

            # Peri-LN writer
            mu_write = self.pre_write_ln(mu)
            mu_new, lam = self.writer(mu_write, lam, messages, pi[:, :, 4:5])
            mu_new = self.post_write_ln(mu_new)
            mu_new = mu_new + stage_out * 0.1

            # v0.6.1: Pass adapter (residual, zero-init)
            mu_new = mu_new + self.pass_adapters[p](mu_new)

            mu = mu_new

            # Scratchpad update
            mem = self.scratchpad.write(mu.detach(), mem.detach(), pi[:, :, 4:5].detach())

            # Pheromone
            with torch.no_grad():
                delta_mag = (mu - mu_prev).pow(2).mean(dim=-1).sqrt()
                deposit = pi[:, :, 4].detach() * torch.tanh(delta_mag)
                pheromone = self.pheromone_rho * pheromone + late_gate * deposit
            mu_prev = mu.detach()

            if collect_history:
                mu_hist[:, :, p, :] = mu
                pi_hist[:, :, p, :] = pi

        # Output
        final = self.ln(mu)
        logits = F.linear(final, self.emb.weight) / math.sqrt(self.dim)

        aux = {
            "mu_hist": mu_hist, "pi_hist": pi_hist,
            "compute_cost": torch.tensor(float(n_steps), device=device),
            "avg_steps": n_steps,
        }

        if alpha_hist is not None:
            aux["alpha_hist"] = torch.stack(alpha_hist, dim=2)  # (B, T, P, 4)

        if y is not None and collect_history:
            candidates = build_negative_set(logits.detach(), y)
            s_ce, s_margin = sampled_pass_ce(mu_hist, self.ln, self.emb.weight, self.dim, candidates)
            aux["sampled_ce_hist"] = s_ce
            aux["sampled_margin_hist"] = s_margin

            probe_preds = []
            prev_margin = torch.zeros(B, T, device=device)
            for p_idx in range(n_steps):
                margin_p = s_margin[:, :, p_idx].detach()
                margin_slope = margin_p - prev_margin if p_idx > 0 else torch.zeros_like(margin_p)
                mu_p_det = mu_hist[:, :, p_idx].detach()
                pi_p_det = pi_hist[:, :, p_idx].detach()
                mu_prev_det = (mu_hist[:, :, p_idx-1] if p_idx > 0 else mu_hist[:, :, 0]).detach()
                delta_mu = (mu_p_det - mu_prev_det).pow(2).mean(-1).sqrt()
                pass_frac = torch.full((B, T), p_idx / max(n_steps - 1, 1), device=device)
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


def load_v061_from_v060b(model, parent_path, device="cpu"):
    """Load v0.6.0b weights into v0.6.1 with surgical mapping.

    1. All parent weights load where names match
    2. transition.base -> transition.base (direct)
    3. Old mode_gate + mode_bias -> SKIPPED (removed in v0.6.1)
    4. New controller + pass adapters keep their init values
    """
    parent = torch.load(parent_path, weights_only=False, map_location=device)
    parent_sd = parent["model"] if "model" in parent else parent

    model_sd = model.state_dict()
    loaded, skipped_keys, new_keys = [], [], []

    for name, param in model_sd.items():
        if name in parent_sd and parent_sd[name].shape == param.shape:
            model_sd[name] = parent_sd[name]
            loaded.append(name)
        else:
            new_keys.append(name)

    # Identify parent keys we intentionally skip
    dropped = [k for k in parent_sd if k not in model_sd]

    model.load_state_dict(model_sd)

    print(f"Warm-start v0.6.1 from v0.6.0b:")
    print(f"  Loaded: {len(loaded)} params")
    print(f"  New (init preserved): {len(new_keys)} params")
    print(f"  Dropped from parent: {len(dropped)} keys")
    for k in dropped:
        print(f"    - {k}")

    return {
        "loaded": len(loaded),
        "new": new_keys,
        "dropped": dropped,
        "parent_step": parent.get("step", 0),
        "parent_bpt": parent.get("best_bpt", None),
    }


if __name__ == "__main__":
    print("Testing v0.6.1 (Controller-Local Canary)...")
    m = create_v061(dim=64, ff_dim=128, max_steps=12, window=2, k_retrieval=4)
    total = m.count_params()
    print(f"Params: {total:,}")

    # Compare to parent
    from launch_v060a import create_v060a
    parent = create_v060a(dim=64, ff_dim=128, max_steps=12, window=2, k_retrieval=4)
    parent_total = parent.count_params()
    delta = total - parent_total
    print(f"Parent: {parent_total:,}, v0.6.1: {total:,}, delta: +{delta:,} ({delta/parent_total*100:.2f}%)")

    x = torch.randint(0, 50257, (2, 32))
    y = torch.randint(0, 50257, (2, 32))
    logits, aux = m(x, y=y)
    print(f"Forward: logits={logits.shape}")
    print(f"  mu_hist: {aux['mu_hist'].shape}")
    print(f"  alpha_hist: {aux['alpha_hist'].shape}")
    print(f"  sampled_ce_hist: {aux['sampled_ce_hist'].shape}")
    print(f"  probe_pred: {aux['probe_pred'].shape}")

    loss = F.cross_entropy(logits.reshape(-1, 50257), y.reshape(-1))
    loss.backward()
    print(f"Backward OK, loss={loss.item():.4f}")
    nan_grads = any(torch.isnan(p.grad).any() for p in m.parameters() if p.grad is not None)
    print(f"NaN grads: {nan_grads}")

    # Causality check
    xa = torch.randint(0, 50257, (1, 16))
    xb = xa.clone(); xb[0, 8] = (xb[0, 8] + 1) % 50257
    with torch.no_grad():
        la, _ = m(xa); lb, _ = m(xb)
    diff = (la[0, :8] - lb[0, :8]).abs().max().item()
    print(f"Causality: diff={diff:.6f} {'PASS' if diff < 0.001 else 'INVESTIGATE'}")

    # Verify pass adapter gates start at 0
    gates = [m.pass_adapters[i].gate.item() for i in range(12)]
    print(f"Adapter gates: {gates}")

    print("v0.6.1 OK")
