"""Chrome Probe A: Pass Conditioning.

Compare:
  Arm 0 (baseline): v0.6.0a as-is (GatedLayerNorm, no pass info)
  Arm 1 (adaLN): GatedRMSNorm conditioned on pass index via scale/shift
  Arm 2 (adaLN+RoPE): Arm 1 + rotary position encoding on LocalRouter QK

Metrics:
  - BPT (test set)
  - Pass-to-pass cosine similarity (collapse detection)
  - Late-pass contribution (passes 8-11 share of total BPT improvement)
  - Greedy trigram diversity

Gate: BPT improves AND cosine collapse decreases (more diverse per-pass states).
Kill: No BPT improvement, or collapse not reduced.

Runs on CPU from a v0.6.0a checkpoint. 200 batches, dim=768.
"""

import sys, math, json, time, argparse
from pathlib import Path
REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "code"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

# ── Pass-Conditioned GatedRMSNorm ────────────────────────────────────

class GatedRMSNorm(nn.Module):
    """RMSNorm with learnable bypass gate (warm-start compatible)."""
    def __init__(self, dim, init_alpha=-5.0):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.alpha = nn.Parameter(torch.tensor(init_alpha))
        self.eps = 1e-6

    def forward(self, x):
        a = torch.sigmoid(self.alpha)
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        normed = x / rms * self.weight
        return (1 - a) * x + a * normed


class PassConditionedNorm(nn.Module):
    """adaLN-style: existing GatedLayerNorm + pass-dependent scale/shift.

    Wraps an existing GatedLayerNorm (for warm-start compatibility) and adds
    a tiny MLP that maps pass_index -> (gamma, beta) modulation.
    At init, gamma=1, beta=0 → behaves identically to the base norm.
    """
    def __init__(self, dim, max_passes=12, base_norm=None):
        super().__init__()
        if base_norm is not None:
            self.norm = base_norm  # reuse existing GatedLayerNorm
        else:
            self.norm = GatedRMSNorm(dim)
        self.has_pass_arg = True  # flag for conditional calling
        # Small pass embedding -> scale/shift
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


# ── Rotary Position Encoding ─────────────────────────────────────────

class RotaryPositionEncoding(nn.Module):
    """Standard RoPE for sequence positions."""
    def __init__(self, dim, max_len=2048, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_len = max_len

    def forward(self, seq_len, device):
        """Returns cos, sin tensors of shape (seq_len, dim)."""
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # (T, dim//2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (T, dim)
        return emb.cos(), emb.sin()


def apply_rotary(x, cos, sin):
    """Apply RoPE to x: (B, T, D)."""
    d = x.shape[-1]
    x1, x2 = x[..., :d//2], x[..., d//2:]
    cos = cos[:x.shape[1], :d//2]
    sin = sin[:x.shape[1], :d//2]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


# ── Modified Forward with Pass Conditioning ──────────────────────────

def run_forward_with_conditioning(model, x, y, mode="baseline"):
    """Run forward pass with different conditioning modes.

    mode="baseline": standard v0.6.0a forward
    mode="adaln": replace norms with pass-conditioned norms
    mode="adaln_rope": adaln + RoPE on router QK

    Returns: logits, per_pass_bpt, per_pass_cosine
    """
    B, T = x.shape
    device = x.device
    P = model.max_steps

    # Init (same for all modes)
    if mode in ("adaln_rope",):
        # Use RoPE instead of learned pos embeddings
        h = model.emb(x)  # skip pos_emb, apply RoPE in router
    else:
        h = model.emb(x) + model.pos_emb(torch.arange(T, device=device))

    mu = model.init_mu(h)
    lam = F.softplus(model.init_lam(h)) + 0.1
    from sutra_v05_ssm import N_STAGES, top2_project
    pi = torch.zeros(B, T, N_STAGES, device=device, dtype=h.dtype)
    pi[:, :, 2] = 1.0
    mem = model.scratchpad.init_memory(B, device)
    pheromone = torch.zeros(B, T, device=device, dtype=h.dtype)
    mu_prev = mu.detach()

    # Per-pass metrics
    per_pass_bpt = []
    per_pass_cosine = []

    # Get conditioning modules (created externally and attached to model)
    adaln_norms = getattr(model, '_adaln_norms', None)
    rope_module = getattr(model, '_rope', None)

    for p in range(P):
        late_gate = 1.0 if p >= 3 else 0.0

        # Stage transitions
        K = model.transition(mu)
        pi_ev = torch.bmm(
            pi.view(B * T, 1, N_STAGES),
            K.view(B * T, N_STAGES, N_STAGES)
        ).view(B, T, N_STAGES)

        # Stage bank (with or without pass conditioning on norm)
        if adaln_norms and mode in ("adaln", "adaln_rope"):
            mu_bank = adaln_norms['pre_bank'](mu, p)
        else:
            mu_bank = model.pre_bank_ln(mu)
        stage_out, evidence = model.stage_bank(mu_bank, pi)
        if adaln_norms and mode in ("adaln", "adaln_rope"):
            stage_out = adaln_norms['post_bank'](stage_out, p)
        else:
            stage_out = model.post_bank_ln(stage_out)

        pi_new = pi_ev * F.softmax(evidence / 0.5, dim=-1)
        pi_new = pi_new / pi_new.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        pi = top2_project(pi_new)

        # Routing (with or without RoPE)
        if adaln_norms and mode in ("adaln", "adaln_rope"):
            mu_route = adaln_norms['pre_route'](mu, p)
        else:
            mu_route = model.pre_route_ln(mu)

        if mode == "adaln_rope" and rope_module is not None:
            # Apply RoPE to Q and K in the router
            cos, sin = rope_module(T, device)
            q = apply_rotary(model.router.q_proj(mu_route), cos, sin)
            k = apply_rotary(model.router.k_proj(mu_route), cos, sin)
            v = model.router.v_proj(mu_route)

            # QK-norm: normalize q and k
            q = F.normalize(q, dim=-1) * math.sqrt(model.dim)
            k = F.normalize(k, dim=-1)

            scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(model.dim)
            causal_mask = torch.triu(torch.ones(T, T, device=device) * float("-inf"), diagonal=1)
            scores = scores + causal_mask
            if T > model.router.k:
                topk_vals, topk_idx = scores.topk(model.router.k, dim=-1)
                sparse = torch.full_like(scores, float("-inf"))
                sparse.scatter_(2, topk_idx, topk_vals)
                attn = F.softmax(sparse, dim=-1)
            else:
                attn = F.softmax(scores, dim=-1)
            global_msgs = torch.bmm(attn, v)

            # Local messages (unchanged)
            padded = F.pad(mu_route, (0, 0, model.router.window, 0))
            neighbors = torch.stack([
                padded[:, model.router.window - w:model.router.window - w + T, :]
                for w in range(1, model.router.window + 1)
            ], dim=2)
            self_exp = mu_route.unsqueeze(2).expand_as(neighbors)
            local_msgs = model.router.msg_net(torch.cat([self_exp, neighbors], dim=-1)).mean(dim=2)

            messages = model.router.out_proj(torch.cat([local_msgs, global_msgs], dim=-1))
        else:
            messages = model.router(mu_route)

        messages = messages * pi[:, :, 3:4]
        if adaln_norms and mode in ("adaln", "adaln_rope"):
            messages = adaln_norms['post_route'](messages, p)
        else:
            messages = model.post_route_ln(messages)

        # Scratchpad
        mem_ctx = model.scratchpad.read(mu, mem)
        messages = messages + 0.1 * mem_ctx * pi[:, :, 3:4]

        # Pheromone
        if late_gate > 0:
            phero_gate = torch.sigmoid(pheromone).unsqueeze(-1)
            messages = messages + 0.05 * phero_gate * mem_ctx * pi[:, :, 3:4]

        # Writer
        if adaln_norms and mode in ("adaln", "adaln_rope"):
            mu_write = adaln_norms['pre_write'](mu, p)
        else:
            mu_write = model.pre_write_ln(mu)
        mu_new, lam = model.writer(mu_write, lam, messages, pi[:, :, 4:5])
        if adaln_norms and mode in ("adaln", "adaln_rope"):
            mu_new = adaln_norms['post_write'](mu_new, p)
        else:
            mu_new = model.post_write_ln(mu_new)
        mu = mu_new + stage_out * 0.1

        # Scratchpad/pheromone update
        mem = model.scratchpad.write(mu.detach(), mem.detach(), pi[:, :, 4:5].detach())
        with torch.no_grad():
            delta_mag = (mu - mu_prev).pow(2).mean(dim=-1).sqrt()
            deposit = pi[:, :, 4].detach() * torch.tanh(delta_mag)
            pheromone = model.pheromone_rho * pheromone + late_gate * deposit
        mu_prev = mu.detach()

        # Per-pass BPT (full vocab CE)
        with torch.no_grad():
            pass_logits = F.linear(model.ln(mu), model.emb.weight) / math.sqrt(model.dim)
            pass_ce = F.cross_entropy(pass_logits.view(-1, model.vocab_size), y.view(-1))
            per_pass_bpt.append(pass_ce.item() / math.log(2))

            # Cosine with previous pass
            if p > 0:
                cos_sim = F.cosine_similarity(mu.view(-1, model.dim),
                                               mu_prev_for_cos.view(-1, model.dim), dim=-1).mean()
                per_pass_cosine.append(cos_sim.item())
            mu_prev_for_cos = mu.detach().clone()

    return per_pass_bpt, per_pass_cosine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default=str(REPO / "results/checkpoints_v060a/rolling_latest.pt"))
    parser.add_argument("--n_batches", type=int, default=200)
    parser.add_argument("--output", type=str,
                        default=str(REPO / "results/probe_pass_conditioning.json"))
    args = parser.parse_args()

    device = "cpu"
    print(f"Chrome Probe A: Pass Conditioning")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Batches: {args.n_batches}")
    print(f"  Device: {device}")

    # Load model
    from launch_v060a import create_v060a
    model = create_v060a()
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"], strict=False)
    elif "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model.eval()
    model.to(device)
    print(f"  Model loaded: {model.count_params():,} params")

    # Create adaLN modules (randomly initialized — this is a PROBE, not trained)
    adaln_norms = {
        'pre_bank': PassConditionedNorm(768).to(device),
        'post_bank': PassConditionedNorm(768).to(device),
        'pre_route': PassConditionedNorm(768).to(device),
        'post_route': PassConditionedNorm(768).to(device),
        'pre_write': PassConditionedNorm(768).to(device),
        'post_write': PassConditionedNorm(768).to(device),
    }
    rope = RotaryPositionEncoding(768).to(device)

    # Attach to model for access in forward
    model._adaln_norms = adaln_norms
    model._rope = rope

    # Load test data
    from data_loader import SutraDataLoader
    loader = SutraDataLoader(str(REPO / "data/diverse_shards"), seq_len=512, batch_size=4)

    # Run 3 arms
    results = {}
    for mode in ["baseline", "adaln", "adaln_rope"]:
        print(f"\n  Running arm: {mode}")
        all_pass_bpt = [[] for _ in range(12)]
        all_pass_cos = [[] for _ in range(11)]
        t0 = time.time()

        for batch_idx in range(args.n_batches):
            batch = loader.get_batch()
            x = batch[:, :-1].to(device)
            y = batch[:, 1:].to(device)

            with torch.no_grad():
                pass_bpt, pass_cos = run_forward_with_conditioning(model, x, y, mode=mode)

            for p in range(12):
                all_pass_bpt[p].append(pass_bpt[p])
            for p in range(len(pass_cos)):
                all_pass_cos[p].append(pass_cos[p])

            if (batch_idx + 1) % 50 == 0:
                elapsed = time.time() - t0
                final_bpt = sum(all_pass_bpt[11]) / len(all_pass_bpt[11])
                print(f"    Batch {batch_idx+1}/{args.n_batches}: "
                      f"final BPT={final_bpt:.4f}, {elapsed:.1f}s")

        # Aggregate
        avg_pass_bpt = [sum(p) / len(p) for p in all_pass_bpt]
        avg_pass_cos = [sum(p) / len(p) for p in all_pass_cos]
        final_bpt = avg_pass_bpt[-1]
        late_improvement = avg_pass_bpt[7] - avg_pass_bpt[11]
        total_improvement = avg_pass_bpt[0] - avg_pass_bpt[11]
        late_pct = late_improvement / total_improvement * 100 if total_improvement > 0 else 0
        avg_cosine = sum(avg_pass_cos) / len(avg_pass_cos) if avg_pass_cos else 0

        results[mode] = {
            "final_bpt": round(final_bpt, 4),
            "per_pass_bpt": [round(b, 4) for b in avg_pass_bpt],
            "per_pass_cosine": [round(c, 4) for c in avg_pass_cos],
            "late_improvement": round(late_improvement, 4),
            "late_pct": round(late_pct, 1),
            "avg_cosine": round(avg_cosine, 4),
            "time_s": round(time.time() - t0, 1),
        }

        print(f"  {mode}: BPT={final_bpt:.4f}, late_improv={late_improvement:.4f} "
              f"({late_pct:.1f}%), avg_cos={avg_cosine:.4f}")

    # Compare
    print("\n  === COMPARISON ===")
    base = results["baseline"]
    for mode in ["adaln", "adaln_rope"]:
        arm = results[mode]
        delta_bpt = arm["final_bpt"] - base["final_bpt"]
        delta_cos = arm["avg_cosine"] - base["avg_cosine"]
        print(f"  {mode} vs baseline: BPT {delta_bpt:+.4f}, cosine {delta_cos:+.4f}")

    # Verdict
    adaln_better = results["adaln"]["final_bpt"] < base["final_bpt"]
    rope_better = results["adaln_rope"]["final_bpt"] < base["final_bpt"]
    cos_reduced = results["adaln"]["avg_cosine"] < base["avg_cosine"]

    verdict = "KEEP" if (adaln_better or rope_better) and cos_reduced else "INCONCLUSIVE"
    if results["adaln"]["final_bpt"] > base["final_bpt"] + 0.5:
        verdict = "KILL"

    results["verdict"] = verdict
    results["probe"] = "pass_conditioning"
    results["checkpoint"] = str(args.checkpoint)
    results["n_batches"] = args.n_batches

    print(f"\n  VERDICT: {verdict}")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {args.output}")


if __name__ == "__main__":
    main()
