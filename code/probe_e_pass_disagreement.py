"""Probe E: Pass Disagreement as Intrinsic Uncertainty Signal.

Hypothesis: tokens where intermediate passes DISAGREE (high entropy spread,
high KL between consecutive passes, high CE spread) are the "hard" tokens
that benefit most from additional passes. If pass disagreement predicts
token difficulty, it's a free uncertainty signal for elastic compute.

Method:
  - Load v0.6.0a checkpoint, run on CPU
  - For each batch, collect per-pass hidden states (mu_hist)
  - Compute FULL-VOCAB logits at each pass
  - Measure 4 disagreement features across passes
  - Correlate with 3 difficulty signals

CPU-only. No GPU needed.
"""

import json, math, os, sys, time
from pathlib import Path

# Force CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "16"

import torch
import torch.nn.functional as F
import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "code"))

from launch_v060a import create_v060a
from data_loader import ShardedDataset

# Config
BATCH_SIZE = 2
SEQ_LEN = 512
N_BATCHES = 20
DEVICE = torch.device("cpu")
CKPT_PATH = REPO / "results" / "checkpoints_v060a" / "rolling_latest.pt"


def load_model():
    """Load v0.6.0a from checkpoint."""
    print(f"Loading checkpoint: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    model = create_v060a(vocab_size=50257, dim=768, ff_dim=1536, max_steps=12,
                         window=4, k_retrieval=8)
    model.load_state_dict(ckpt["model"])
    model.eval()
    model.to(DEVICE)
    step = ckpt.get("step", "?")
    print(f"Loaded model at step {step}, {model.count_params():,} params")
    return model, step


def compute_per_pass_logits(model, mu_hist):
    """Compute full-vocab logits at each pass from mu_hist.

    mu_hist: (B, T, P, D) hidden states at each pass.
    Returns: (B, T, P, V) logits.
    """
    B, T, P, D = mu_hist.shape
    logits_all = []
    for p in range(P):
        mu_p = model.ln(mu_hist[:, :, p, :])  # (B, T, D)
        logits_p = F.linear(mu_p, model.emb.weight) / math.sqrt(model.dim)  # (B, T, V)
        logits_all.append(logits_p)
    return torch.stack(logits_all, dim=2)  # (B, T, P, V)


def compute_disagreement_features(logits_all, mu_hist, targets):
    """Compute 4 disagreement features and 3 difficulty signals.

    Args:
        logits_all: (B, T, P, V) full-vocab logits at each pass
        mu_hist: (B, T, P, D) hidden states
        targets: (B, T) ground truth token ids

    Returns:
        features: dict of (B, T) tensors for each disagreement feature
        difficulty: dict of (B, T) tensors for each difficulty signal
    """
    B, T, P, V = logits_all.shape

    # Convert logits to probabilities per pass
    probs_all = F.softmax(logits_all, dim=-1)  # (B, T, P, V)

    # --- Feature 1: Entropy spread ---
    # Per-pass entropy, then std across passes
    log_probs = torch.log(probs_all.clamp(min=1e-10))
    entropy_per_pass = -(probs_all * log_probs).sum(dim=-1)  # (B, T, P)
    entropy_spread = entropy_per_pass.std(dim=-1)  # (B, T)

    # --- Feature 2: Logit KL spread ---
    # Avg KL divergence between consecutive pass distributions
    kl_list = []
    for p in range(1, P):
        # KL(pass_p || pass_{p-1})
        kl = F.kl_div(
            log_probs[:, :, p-1, :],  # log of Q
            probs_all[:, :, p, :],     # P
            reduction='none',
            log_target=False
        ).sum(dim=-1)  # (B, T)
        kl_list.append(kl)
    kl_stack = torch.stack(kl_list, dim=-1)  # (B, T, P-1)
    logit_kl_spread = kl_stack.mean(dim=-1)  # (B, T)

    # --- Feature 3: Passwise CE spread ---
    # Per-pass cross-entropy with ground truth, then std
    # Gather the log-prob of the target token at each pass
    target_expanded = targets.unsqueeze(-1).unsqueeze(-1).expand(B, T, P, 1)  # (B, T, P, 1)
    target_log_probs = log_probs.gather(dim=-1, index=target_expanded).squeeze(-1)  # (B, T, P)
    ce_per_pass = -target_log_probs  # (B, T, P)
    ce_spread = ce_per_pass.std(dim=-1)  # (B, T)

    # --- Feature 4: State cosine spread ---
    # Cosine similarity between consecutive hidden states, then std
    cos_list = []
    for p in range(1, P):
        cos = F.cosine_similarity(mu_hist[:, :, p, :], mu_hist[:, :, p-1, :], dim=-1)  # (B, T)
        cos_list.append(cos)
    cos_stack = torch.stack(cos_list, dim=-1)  # (B, T, P-1)
    state_cosine_spread = cos_stack.std(dim=-1)  # (B, T)

    # --- Difficulty signal 1: Future gain ---
    # How much CE improves from pass 0 to final pass (high = hard token that benefits from passes)
    future_gain = ce_per_pass[:, :, 0] - ce_per_pass[:, :, -1]  # (B, T)

    # --- Difficulty signal 2: Hard-token identification ---
    # Final-pass CE (high = still hard after all passes)
    final_ce = ce_per_pass[:, :, -1]  # (B, T)

    # --- Difficulty signal 3: Repetition onset ---
    # Detect repetitive sequences: token matches any of the previous 5 tokens
    rep_mask = torch.zeros(B, T, dtype=torch.float32)
    for offset in range(1, 6):
        if offset < T:
            shifted = torch.zeros_like(targets)
            shifted[:, offset:] = targets[:, :-offset]
            match = (targets == shifted).float()
            match[:, :offset] = 0  # no comparison possible
            rep_mask = rep_mask + match
    rep_mask = (rep_mask > 0).float()  # (B, T) binary: is this token a repeat?

    features = {
        "entropy_spread": entropy_spread,
        "logit_kl_spread": logit_kl_spread,
        "ce_spread": ce_spread,
        "state_cosine_spread": state_cosine_spread,
    }
    difficulty = {
        "future_gain": future_gain,
        "final_ce": final_ce,
        "repetition": rep_mask,
    }
    # Also export raw per-pass data for deeper analysis
    extras = {
        "entropy_per_pass_mean": entropy_per_pass.mean(dim=(0, 1)),  # (P,)
        "ce_per_pass_mean": ce_per_pass.mean(dim=(0, 1)),  # (P,)
        "cos_per_pass_mean": cos_stack.mean(dim=(0, 1)),  # (P-1,)
        "kl_per_pass_mean": kl_stack.mean(dim=(0, 1)),  # (P-1,)
    }
    return features, difficulty, extras


def pearson_corr(x, y):
    """Pearson correlation between flattened tensors."""
    x = x.flatten().float()
    y = y.flatten().float()
    # Remove NaN/Inf
    mask = torch.isfinite(x) & torch.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 10:
        return float('nan')
    xm = x - x.mean()
    ym = y - y.mean()
    num = (xm * ym).sum()
    den = (xm.pow(2).sum() * ym.pow(2).sum()).sqrt()
    if den < 1e-10:
        return float('nan')
    return (num / den).item()


def run_probe():
    """Run the full probe."""
    t0 = time.time()

    model, step = load_model()
    print(f"\nLoading data...")
    dataset = ShardedDataset()

    # Accumulators for all features/difficulty signals
    all_features = {k: [] for k in ["entropy_spread", "logit_kl_spread", "ce_spread", "state_cosine_spread"]}
    all_difficulty = {k: [] for k in ["future_gain", "final_ce", "repetition"]}
    all_extras = []

    print(f"\nRunning {N_BATCHES} batches (B={BATCH_SIZE}, T={SEQ_LEN}) on CPU...")
    for batch_idx in range(N_BATCHES):
        bt = time.time()
        x, y = dataset.sample_batch(BATCH_SIZE, SEQ_LEN, device=DEVICE, split='train')

        with torch.no_grad():
            # Run forward with history collection forced on
            logits, aux = model(x, y=None, collect_history=True)
            mu_hist = aux["mu_hist"]  # (B, T, 12, D)

            # Compute full-vocab logits at each pass
            # Do in chunks of 3 passes to limit peak memory (12 * B * T * V is huge)
            P = model.max_steps
            logits_passes = []
            for p in range(P):
                mu_p = model.ln(mu_hist[:, :, p, :])
                logits_p = F.linear(mu_p, model.emb.weight) / math.sqrt(model.dim)
                logits_passes.append(logits_p)
            logits_all = torch.stack(logits_passes, dim=2)  # (B, T, P, V)

            # Compute features
            features, difficulty, extras = compute_disagreement_features(logits_all, mu_hist, y)

            # Free large tensors immediately
            del logits_all, logits_passes, mu_hist, logits

        for k in all_features:
            all_features[k].append(features[k])
        for k in all_difficulty:
            all_difficulty[k].append(difficulty[k])
        all_extras.append(extras)

        elapsed = time.time() - bt
        print(f"  Batch {batch_idx+1}/{N_BATCHES}: {elapsed:.1f}s")

    # Concatenate all batches
    for k in all_features:
        all_features[k] = torch.cat(all_features[k], dim=0)  # (N*B, T)
    for k in all_difficulty:
        all_difficulty[k] = torch.cat(all_difficulty[k], dim=0)

    # Compute correlations: each feature x each difficulty signal
    print(f"\n=== CORRELATIONS ===")
    corr_matrix = {}
    for feat_name, feat_vals in all_features.items():
        corr_matrix[feat_name] = {}
        for diff_name, diff_vals in all_difficulty.items():
            r = pearson_corr(feat_vals, diff_vals)
            corr_matrix[feat_name][diff_name] = r
            print(f"  {feat_name:25s} x {diff_name:15s} = {r:+.4f}")

    # Aggregate per-pass statistics
    n_extra = len(all_extras)
    entropy_per_pass = torch.stack([e["entropy_per_pass_mean"] for e in all_extras]).mean(dim=0)
    ce_per_pass = torch.stack([e["ce_per_pass_mean"] for e in all_extras]).mean(dim=0)
    cos_per_pass = torch.stack([e["cos_per_pass_mean"] for e in all_extras]).mean(dim=0)
    kl_per_pass = torch.stack([e["kl_per_pass_mean"] for e in all_extras]).mean(dim=0)

    print(f"\n=== PER-PASS STATISTICS ===")
    print(f"  Pass | Entropy  | CE(target) | Cos(p,p-1) | KL(p||p-1)")
    print(f"  -----+----------+------------+------------+-----------")
    for p in range(12):
        e = entropy_per_pass[p].item()
        c = ce_per_pass[p].item()
        cos_val = cos_per_pass[p-1].item() if p > 0 else float('nan')
        kl_val = kl_per_pass[p-1].item() if p > 0 else float('nan')
        print(f"  {p:4d} | {e:8.3f} | {c:10.3f} | {cos_val:10.4f} | {kl_val:9.4f}")

    # Summary statistics
    total_tokens = all_features["entropy_spread"].numel()
    feature_stats = {}
    for k, v in all_features.items():
        feature_stats[k] = {
            "mean": v.mean().item(),
            "std": v.std().item(),
            "median": v.median().item(),
            "min": v.min().item(),
            "max": v.max().item(),
        }

    difficulty_stats = {}
    for k, v in all_difficulty.items():
        difficulty_stats[k] = {
            "mean": v.mean().item(),
            "std": v.std().item(),
        }

    # Top-quartile analysis: do high-disagreement tokens have higher future gain?
    print(f"\n=== QUARTILE ANALYSIS ===")
    quartile_analysis = {}
    for feat_name in all_features:
        feat = all_features[feat_name].flatten()
        q75 = feat.quantile(0.75).item()
        q25 = feat.quantile(0.25).item()
        high_mask = feat >= q75
        low_mask = feat <= q25

        fg = all_difficulty["future_gain"].flatten()
        fc = all_difficulty["final_ce"].flatten()

        fg_high = fg[high_mask].mean().item()
        fg_low = fg[low_mask].mean().item()
        fc_high = fc[high_mask].mean().item()
        fc_low = fc[low_mask].mean().item()

        quartile_analysis[feat_name] = {
            "q75_threshold": q75,
            "q25_threshold": q25,
            "future_gain_q4_mean": fg_high,
            "future_gain_q1_mean": fg_low,
            "future_gain_q4_vs_q1_ratio": fg_high / max(fg_low, 1e-6),
            "final_ce_q4_mean": fc_high,
            "final_ce_q1_mean": fc_low,
        }
        print(f"  {feat_name}:")
        print(f"    High-disagreement (Q4) future_gain: {fg_high:.4f}")
        print(f"    Low-disagreement (Q1) future_gain:  {fg_low:.4f}")
        print(f"    Ratio: {fg_high / max(fg_low, 1e-6):.2f}x")
        print(f"    High-disagreement (Q4) final_ce:    {fc_high:.4f}")
        print(f"    Low-disagreement (Q1) final_ce:     {fc_low:.4f}")

    elapsed = time.time() - t0
    print(f"\n=== DONE === ({elapsed:.0f}s total, {total_tokens:,} tokens)")

    # Determine verdict
    # Pass disagreement is useful if features correlate positively with future_gain
    # (high disagreement = token benefits from more passes)
    best_corr_feat = max(corr_matrix.keys(),
                         key=lambda k: abs(corr_matrix[k]["future_gain"]))
    best_r = corr_matrix[best_corr_feat]["future_gain"]
    verdict = (
        "STRONG" if abs(best_r) > 0.3 else
        "MODERATE" if abs(best_r) > 0.15 else
        "WEAK" if abs(best_r) > 0.05 else
        "NONE"
    )
    print(f"\nVERDICT: {verdict} signal (best: {best_corr_feat}, r={best_r:+.4f})")

    # Save results
    results = {
        "probe": "E_pass_disagreement",
        "model": "v0.6.0a",
        "checkpoint_step": step,
        "config": {
            "batch_size": BATCH_SIZE,
            "seq_len": SEQ_LEN,
            "n_batches": N_BATCHES,
            "total_tokens": total_tokens,
            "device": "cpu",
            "n_passes": 12,
        },
        "correlations": corr_matrix,
        "per_pass_stats": {
            "entropy": entropy_per_pass.tolist(),
            "ce_target": ce_per_pass.tolist(),
            "cosine_consecutive": cos_per_pass.tolist(),
            "kl_consecutive": kl_per_pass.tolist(),
        },
        "feature_stats": feature_stats,
        "difficulty_stats": difficulty_stats,
        "quartile_analysis": quartile_analysis,
        "verdict": verdict,
        "best_feature": best_corr_feat,
        "best_correlation": best_r,
        "elapsed_seconds": elapsed,
    }

    out_path = REPO / "results" / "probe_e_pass_disagreement.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_path}")

    return results


if __name__ == "__main__":
    run_probe()
