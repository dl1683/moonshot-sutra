"""KD Evaluation Suite — Compare CE vs KD checkpoints beyond BPT.

Codex R11 Probe 1: Run entropy histograms, ECE/reliability, token-type
disaggregated BPT, and linear probes on multiple checkpoints.

Usage:
  python code/kd_eval_suite.py <ckpt1> <ckpt2> ... --output results/kd_eval_suite.json
"""

import sys, os, math, json, argparse
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "code"))

from dense_baseline import DenseTransformer, VOCAB_SIZE, DIM, N_LAYERS, N_HEADS, FF_DIM, DTYPE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model_from_ckpt(ckpt_path):
    """Load model from checkpoint, auto-detect config."""
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    cfg = ckpt.get("config", {})
    model = DenseTransformer(
        vocab_size=cfg.get("vocab_size", VOCAB_SIZE),
        dim=cfg.get("dim", DIM),
        n_layers=cfg.get("n_layers", N_LAYERS),
        n_heads=cfg.get("n_heads", N_HEADS),
        ff_dim=cfg.get("ff_dim", FF_DIM),
        exit_layers=cfg.get("exit_layers", None),
        norm_type=cfg.get("norm_type", "rmsnorm"),
        block_schedule=cfg.get("block_schedule", None),
        n_q_heads=cfg.get("n_q_heads", None),
        n_kv_heads=cfg.get("n_kv_heads", None),
        head_dim=cfg.get("head_dim", 64),
        conv_kernel_size=cfg.get("conv_kernel_size", 64),
    ).to(DEVICE)
    # Handle both key conventions
    state_key = "model_state_dict" if "model_state_dict" in ckpt else "model"
    model.load_state_dict(ckpt[state_key])
    model.eval()
    step = ckpt.get("step", "?")
    return model, cfg, step


def compute_eval_metrics(model, cache_path, n_ece_bins=15):
    """Run full evaluation suite on a model.

    Returns dict with:
      - bpt: bits per token
      - entropy_mean, entropy_std, entropy_p10/p25/p50/p75/p90: output entropy stats
      - ece: expected calibration error
      - ece_over, ece_under: overconfident/underconfident ECE decomposition
      - bin_accs, bin_confs, bin_counts: reliability diagram data
      - token_type_bpt: {common, rare, high_entropy, low_entropy}
    """
    cache = torch.load(cache_path, weights_only=False, map_location="cpu")
    windows = cache["windows"]

    all_entropies = []
    all_confidences = []
    all_correct = []
    total_loss = 0.0
    n_tokens = 0

    # For token-type disaggregated BPT
    # "Common" = top-500 most frequent tokens, "Rare" = outside top-2000
    # We'll use token ID as proxy (lower IDs tend to be more common in BPE)
    common_loss = 0.0
    common_count = 0
    rare_loss = 0.0
    rare_count = 0
    high_ent_loss = 0.0
    high_ent_count = 0
    low_ent_loss = 0.0
    low_ent_count = 0

    COMMON_THRESHOLD = 500
    RARE_THRESHOLD = 2000

    with torch.no_grad():
        for w in windows:
            x = w.unsqueeze(0).to(DEVICE)
            inp, tgt = x[:, :-1], x[:, 1:]

            with torch.amp.autocast("cuda", dtype=DTYPE):
                logits = model(inp)

            logits_f = logits.float()
            V = logits_f.size(-1)
            tgt_flat = tgt.reshape(-1)

            # Per-token CE loss
            per_token_loss = F.cross_entropy(
                logits_f.reshape(-1, V), tgt_flat, reduction="none"
            )
            total_loss += per_token_loss.sum().item()
            n_tokens += tgt.numel()

            # Entropy of output distribution
            probs = F.softmax(logits_f.reshape(-1, V), dim=-1)
            log_probs = torch.log(probs + 1e-10)
            entropy = -(probs * log_probs).sum(-1)  # (T,)
            all_entropies.append(entropy.cpu())

            # Confidence and correctness for ECE
            max_probs, preds = probs.max(dim=-1)
            correct = (preds == tgt_flat).float()
            all_confidences.append(max_probs.cpu())
            all_correct.append(correct.cpu())

            # Token-type disaggregated
            for i in range(tgt_flat.size(0)):
                tok_id = tgt_flat[i].item()
                loss_i = per_token_loss[i].item()
                ent_i = entropy[i].item()

                if tok_id < COMMON_THRESHOLD:
                    common_loss += loss_i
                    common_count += 1
                elif tok_id >= RARE_THRESHOLD:
                    rare_loss += loss_i
                    rare_count += 1

                if ent_i > 5.0:  # high entropy threshold
                    high_ent_loss += loss_i
                    high_ent_count += 1
                elif ent_i < 2.0:  # low entropy threshold
                    low_ent_loss += loss_i
                    low_ent_count += 1

    # Aggregate
    bpt = (total_loss / n_tokens) / math.log(2)

    all_entropies = torch.cat(all_entropies)
    all_confidences = torch.cat(all_confidences)
    all_correct = torch.cat(all_correct)

    # Entropy statistics
    ent_np = all_entropies.numpy()
    entropy_stats = {
        "entropy_mean": float(np.mean(ent_np)),
        "entropy_std": float(np.std(ent_np)),
        "entropy_p10": float(np.percentile(ent_np, 10)),
        "entropy_p25": float(np.percentile(ent_np, 25)),
        "entropy_p50": float(np.percentile(ent_np, 50)),
        "entropy_p75": float(np.percentile(ent_np, 75)),
        "entropy_p90": float(np.percentile(ent_np, 90)),
    }

    # ECE computation
    bin_boundaries = torch.linspace(0, 1, n_ece_bins + 1)
    bin_accs = []
    bin_confs = []
    bin_counts = []
    ece = 0.0
    ece_over = 0.0
    ece_under = 0.0
    N = all_confidences.size(0)

    for i in range(n_ece_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (all_confidences > lo) & (all_confidences <= hi)
        count = mask.sum().item()
        if count > 0:
            acc = all_correct[mask].mean().item()
            conf = all_confidences[mask].mean().item()
            bin_accs.append(acc)
            bin_confs.append(conf)
            bin_counts.append(count)
            gap = abs(acc - conf)
            ece += gap * count / N
            if conf > acc:
                ece_over += (conf - acc) * count / N
            else:
                ece_under += (acc - conf) * count / N
        else:
            bin_accs.append(0.0)
            bin_confs.append(0.0)
            bin_counts.append(0)

    # Token-type BPT
    token_type_bpt = {}
    if common_count > 0:
        token_type_bpt["common"] = round((common_loss / common_count) / math.log(2), 4)
    if rare_count > 0:
        token_type_bpt["rare"] = round((rare_loss / rare_count) / math.log(2), 4)
    if high_ent_count > 0:
        token_type_bpt["high_entropy"] = round((high_ent_loss / high_ent_count) / math.log(2), 4)
    if low_ent_count > 0:
        token_type_bpt["low_entropy"] = round((low_ent_loss / low_ent_count) / math.log(2), 4)

    result = {
        "bpt": round(bpt, 4),
        "n_tokens": n_tokens,
        **entropy_stats,
        "ece": round(ece, 4),
        "ece_overconfident": round(ece_over, 4),
        "ece_underconfident": round(ece_under, 4),
        "reliability_bin_accs": [round(x, 4) for x in bin_accs],
        "reliability_bin_confs": [round(x, 4) for x in bin_confs],
        "reliability_bin_counts": bin_counts,
        "token_type_bpt": token_type_bpt,
        "token_type_counts": {
            "common": common_count,
            "rare": rare_count,
            "high_entropy": high_ent_count,
            "low_entropy": low_ent_count,
        },
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="KD Evaluation Suite")
    parser.add_argument("checkpoints", nargs="+", help="Checkpoint paths")
    parser.add_argument("--output", type=str, default="results/kd_eval_suite.json")
    parser.add_argument("--cache", type=str, default=str(REPO / "results" / "eval_cache_16k.pt"))
    args = parser.parse_args()

    cache_path = Path(args.cache)
    if not cache_path.exists():
        print(f"ERROR: eval cache not found at {cache_path}")
        sys.exit(1)

    results = {}
    for ckpt_path in args.checkpoints:
        name = Path(ckpt_path).stem
        parent = Path(ckpt_path).parent.name
        label = f"{parent}/{name}"
        print(f"\n{'='*60}")
        print(f"Evaluating: {label}")
        print(f"{'='*60}")

        model, cfg, step = load_model_from_ckpt(ckpt_path)
        metrics = compute_eval_metrics(model, cache_path)
        metrics["step"] = step
        metrics["checkpoint"] = str(ckpt_path)
        results[label] = metrics

        # Print summary
        print(f"  BPT: {metrics['bpt']}")
        print(f"  Entropy: mean={metrics['entropy_mean']:.3f}, std={metrics['entropy_std']:.3f}")
        print(f"  ECE: {metrics['ece']:.4f} (over={metrics['ece_overconfident']:.4f}, under={metrics['ece_underconfident']:.4f})")
        print(f"  Token-type BPT: {metrics['token_type_bpt']}")

        # Free memory
        del model
        torch.cuda.empty_cache()

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print comparison table
    print(f"\n{'='*60}")
    print("COMPARISON TABLE")
    print(f"{'='*60}")
    labels = list(results.keys())
    print(f"{'Metric':<25}", end="")
    for l in labels:
        short = l.split("/")[-1][:20]
        print(f" {short:>20}", end="")
    print()
    print("-" * (25 + 21 * len(labels)))

    for metric in ["bpt", "entropy_mean", "entropy_std", "ece", "ece_overconfident"]:
        print(f"{metric:<25}", end="")
        for l in labels:
            val = results[l].get(metric, "N/A")
            if isinstance(val, float):
                print(f" {val:>20.4f}", end="")
            else:
                print(f" {str(val):>20}", end="")
        print()

    # Token-type comparison
    for tt in ["common", "rare", "high_entropy", "low_entropy"]:
        print(f"bpt_{tt:<20}", end="")
        for l in labels:
            val = results[l].get("token_type_bpt", {}).get(tt, "N/A")
            if isinstance(val, float):
                print(f" {val:>20.4f}", end="")
            else:
                print(f" {str(val):>20}", end="")
        print()


if __name__ == "__main__":
    main()
