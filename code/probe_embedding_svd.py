"""Probe: Embedding Factorization Feasibility (CPU)

Computes randomized SVD of the current tied embedding matrix and evaluates
reconstruction quality at ranks 64, 128, and 256.

Measures:
- Reconstruction error (Frobenius norm ratio)
- Nearest-neighbor preservation (top-10 neighbors before/after)
- Logit drift on a fixed eval slice
- Variance explained per rank

Requested by Tesla+Leibniz Round 2, Probe #6.
"""

import sys, json, time, math
from pathlib import Path
REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "code"))

import torch
import torch.nn.functional as F
import numpy as np

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def load_embedding_matrix(ckpt_path):
    """Load the tied embedding matrix from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    if isinstance(state, dict) and any(k.startswith("module.") for k in state):
        state = {k.replace("module.", ""): v for k, v in state.items()}

    emb_key = "emb.weight"
    if emb_key not in state:
        for k in state:
            if "emb" in k and "weight" in k:
                emb_key = k
                break

    W = state[emb_key].float()  # (V, H)
    print(f"Embedding matrix: {W.shape} = {W.numel():,} params")
    return W


def compute_svd_analysis(W, ranks=[64, 128, 256]):
    """SVD analysis of embedding matrix."""
    V, H = W.shape
    print(f"Computing SVD of {V}x{H} matrix...")

    t0 = time.time()
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    svd_time = time.time() - t0
    print(f"SVD computed in {svd_time:.1f}s")

    # Variance explained
    total_var = (S ** 2).sum().item()
    cumvar = (S ** 2).cumsum(0) / total_var

    results = {
        "total_singular_values": len(S),
        "top_10_singular_values": S[:10].tolist(),
        "variance_at_ranks": {},
    }

    for r in [32, 64, 128, 256, 384, 512]:
        if r <= len(S):
            results["variance_at_ranks"][str(r)] = cumvar[r - 1].item()

    # Reconstruction at each rank
    for rank in ranks:
        print(f"\n--- Rank {rank} ---")
        W_approx = U[:, :rank] @ torch.diag(S[:rank]) @ Vh[:rank, :]

        # Frobenius error
        frob_err = (W - W_approx).norm().item()
        frob_orig = W.norm().item()
        rel_err = frob_err / frob_orig

        # Per-token reconstruction error
        per_token_err = (W - W_approx).norm(dim=1)
        mean_token_err = per_token_err.mean().item()
        max_token_err = per_token_err.max().item()

        print(f"  Frobenius error: {frob_err:.2f} / {frob_orig:.2f} = {rel_err:.4f}")
        print(f"  Per-token error: mean={mean_token_err:.4f}, max={max_token_err:.4f}")
        print(f"  Variance explained: {cumvar[rank-1].item():.4f}")

        # Nearest-neighbor preservation
        nn_preserved = check_nn_preservation(W, W_approx, n_queries=500, k=10)
        print(f"  NN preservation (top-10): {nn_preserved:.3f}")

        # Parameter savings
        factored_params = V * rank + rank * H
        original_params = V * H
        savings = 1.0 - factored_params / original_params
        print(f"  Params: {factored_params:,} vs {original_params:,} (savings: {savings:.1%})")

        results[f"rank_{rank}"] = {
            "frobenius_relative_error": rel_err,
            "variance_explained": cumvar[rank - 1].item(),
            "per_token_error_mean": mean_token_err,
            "per_token_error_max": max_token_err,
            "nn_preservation_top10": nn_preserved,
            "factored_params": factored_params,
            "original_params": original_params,
            "param_savings_pct": savings * 100,
        }

    return results, U, S, Vh


def check_nn_preservation(W_orig, W_approx, n_queries=500, k=10):
    """Check if nearest neighbors are preserved after factorization."""
    V = W_orig.shape[0]
    # Sample random query tokens
    indices = torch.randperm(V)[:n_queries]

    preserved_count = 0
    total_count = 0

    for idx in indices:
        # Original neighbors
        q_orig = W_orig[idx].unsqueeze(0)
        dists_orig = torch.cdist(q_orig, W_orig).squeeze(0)
        nn_orig = dists_orig.topk(k + 1, largest=False).indices[1:]  # exclude self

        # Approx neighbors
        q_approx = W_approx[idx].unsqueeze(0)
        dists_approx = torch.cdist(q_approx, W_approx).squeeze(0)
        nn_approx = dists_approx.topk(k + 1, largest=False).indices[1:]

        # Count overlap
        overlap = len(set(nn_orig.tolist()) & set(nn_approx.tolist()))
        preserved_count += overlap
        total_count += k

    return preserved_count / total_count


def eval_logit_drift(ckpt_path, W_orig, W_approx, n_batches=5, seq_len=256):
    """Measure how much logits change when using factored embeddings."""
    from launch_v060a import SutraV060a
    from datasets import load_dataset
    import tiktoken

    tokenizer = tiktoken.get_encoding("gpt2")

    # Load data
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    text = "\n".join([t for t in ds["text"] if len(t) > 50])
    tokens = tokenizer.encode(text)

    # Load model
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = SutraV060a(
        vocab_size=50257, dim=768, ff_dim=1536,
        max_steps=12, window=4, k_retrieval=4, n_scratch_slots=8,
    )
    state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    if isinstance(state, dict) and any(k.startswith("module.") for k in state):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()

    kl_divs = []
    ce_diffs = []

    for i in range(n_batches):
        s = i * seq_len
        if s + seq_len + 1 > len(tokens):
            break
        x = torch.tensor([tokens[s:s + seq_len]])
        y = torch.tensor([tokens[s + 1:s + seq_len + 1]])

        with torch.no_grad():
            # Original logits
            logits_orig, _ = model(x, collect_history=False)

            # Swap embedding and get new logits
            orig_emb_weight = model.emb.weight.data.clone()
            # For tied weights, replacing emb.weight also changes the output projection
            # This simulates what factored embeddings would do
            model.emb.weight.data.copy_(W_approx)
            logits_approx, _ = model(x, collect_history=False)
            model.emb.weight.data.copy_(orig_emb_weight)

            # KL divergence
            p = F.log_softmax(logits_orig[0], dim=-1)
            q = F.softmax(logits_approx[0], dim=-1)
            kl = F.kl_div(p, q, reduction='batchmean', log_target=False).item()
            kl_divs.append(kl)

            # CE difference
            ce_orig = F.cross_entropy(logits_orig.view(-1, logits_orig.size(-1)), y.view(-1)).item()
            ce_approx = F.cross_entropy(logits_approx.view(-1, logits_approx.size(-1)), y.view(-1)).item()
            ce_diffs.append(ce_approx - ce_orig)

    return {
        "kl_divergence_mean": np.mean(kl_divs),
        "kl_divergence_std": np.std(kl_divs),
        "ce_diff_mean": np.mean(ce_diffs),
        "ce_diff_std": np.std(ce_diffs),
        "n_batches": len(kl_divs),
    }


def main():
    ckpt_path = REPO / "results" / "checkpoints_v060a" / "rolling_latest.pt"
    print(f"Loading embedding from {ckpt_path}")

    W = load_embedding_matrix(str(ckpt_path))

    ranks = [64, 128, 256]
    results, U, S, Vh = compute_svd_analysis(W, ranks)

    # Logit drift for each rank
    print("\n\n=== LOGIT DRIFT ANALYSIS ===")
    for rank in ranks:
        print(f"\n--- Rank {rank} logit drift ---")
        W_approx = U[:, :rank] @ torch.diag(S[:rank]) @ Vh[:rank, :]
        drift = eval_logit_drift(str(ckpt_path), W, W_approx, n_batches=5, seq_len=256)
        results[f"rank_{rank}"]["logit_drift"] = drift
        print(f"  KL divergence: {drift['kl_divergence_mean']:.4f} +/- {drift['kl_divergence_std']:.4f}")
        print(f"  CE difference: {drift['ce_diff_mean']:+.4f} +/- {drift['ce_diff_std']:.4f}")

    # Summary
    print("\n\n" + "=" * 60)
    print("EMBEDDING FACTORIZATION FEASIBILITY SUMMARY")
    print("=" * 60)
    print(f"{'Rank':<8} {'Var Exp':>8} {'Frob Err':>10} {'NN Pres':>8} {'Savings':>8} {'KL Drift':>10} {'CE Delta':>10}")
    print("-" * 68)
    for rank in ranks:
        r = results[f"rank_{rank}"]
        d = r.get("logit_drift", {})
        print(f"{rank:<8} {r['variance_explained']:>8.4f} {r['frobenius_relative_error']:>10.4f} "
              f"{r['nn_preservation_top10']:>8.3f} {r['param_savings_pct']:>7.1f}% "
              f"{d.get('kl_divergence_mean', 0):>10.4f} {d.get('ce_diff_mean', 0):>+10.4f}")

    # Verdict
    rank_128 = results["rank_128"]
    if rank_128["variance_explained"] > 0.95 and rank_128["nn_preservation_top10"] > 0.7:
        verdict = "FEASIBLE"
    elif rank_128["variance_explained"] > 0.90:
        verdict = "MARGINAL"
    else:
        verdict = "RISKY"

    results["verdict"] = verdict
    results["probe"] = "embedding_factorization_feasibility"

    out_path = REPO / "results" / "probe_embedding_svd.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nVerdict: {verdict}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
