"""Committee-regret window scoring probe — Codex T+L KM-R4 top-priority upside probe.

Scores a corpus of candidate training windows by mean positive committee regret:
    R(w) = (1/T) · Σ_t max_m [log P_m(y_t) - log P_s(y_t)]_+

Windows with high R(w) are those where at least one teacher significantly
advantages the student on a majority of positions. Used as a window-weighted
sampler (50% uniform + 50% regret-weighted) for a follow-up training run.

Codex R4 hypothesis: "The utility distribution is so heavy-tailed that the
decisive gain is more likely to come from window selection than from any
further per-position KL refinement."

Usage:
    python code/probe_committee_regret.py [--checkpoint PATH] [--n-windows 512]

Output:
    results/probe_committee_regret.json
        Overall regret distribution statistics
    results/committee_regret_scores.npz
        Per-window regret scores + shard+offset metadata
        (for downstream use by regret-weighted sampler)

Requires GPU for teacher forwards (SmolLM2-1.7B + Pythia-1.4B at 4-bit).
"""
import os
import sys
import json
import math
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "code"))

from sutra_dyad import (
    SutraDyadS1, SEQ_BYTES,
    _build_covering_tables,
    _get_teacher_targets_covering_batched,
)
from data_loader import ByteShardedDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
KD_TEMP = 1.0

TEACHERS_CFG = [
    {"id": "HuggingFaceTB/SmolLM2-1.7B", "role": "anchor"},
    {"id": "EleutherAI/pythia-1.4b", "role": "aux"},
]


def seed_all(s):
    import random
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def load_teacher(tid):
    print(f"  Loading {tid} (4-bit)...")
    tok = AutoTokenizer.from_pretrained(tid)
    bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                  bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        tid, quantization_config=bnb_cfg, device_map={"": DEVICE})
    model.eval()
    # Use logits dimension (model output width), not tokenizer vocab — they can differ
    # (Pythia: tokenizer 50277, logits 50304 for GPU-efficient dimensions).
    vocab_size = model.get_output_embeddings().weight.shape[0]
    cov = _build_covering_tables(tok, vocab_size=vocab_size)
    print(f"  -> tokenizer vocab: {len(tok)}, logits vocab: {vocab_size}")
    return {"id": tid, "tokenizer": tok, "model": model, "covering": cov, "vocab": vocab_size}


def main(args):
    seed_all(SEED)

    out_json = REPO / "results" / "probe_committee_regret.json"
    out_npz = REPO / "results" / "committee_regret_scores.npz"

    # Load student
    ckpt_path = Path(args.checkpoint)
    print(f"Loading student: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    student = SutraDyadS1(max_seq_bytes=SEQ_BYTES)
    student.load_state_dict(ckpt["model"], strict=False)
    student = student.to(DEVICE).eval()
    print(f"  -> step={ckpt.get('step', '?')}, eval_loss={ckpt.get('eval_loss', '?')}")

    # Sample windows (store shard/offset metadata for downstream sampler use)
    print(f"Sampling {args.n_windows} windows (seed={SEED})...")
    dataset = ByteShardedDataset()

    # Use multiple calls to get diverse windows; track (shard_idx, offset) if possible
    chunk = 64  # batch for teacher forward (memory-bounded)
    n_batches = (args.n_windows + chunk - 1) // chunk

    all_regrets = []       # per-window scalar regret R(w)
    all_regret_h4 = []     # horizon-4 regret (avg over 4-byte windows)
    all_metadata = []      # placeholder for shard/offset info
    all_byte_probs_top = []  # for later use
    all_x_windows = []     # raw x bytes per window (for downstream sampler)
    all_y_windows = []     # raw y bytes per window

    print("Loading teachers...")
    teachers = [load_teacher(cfg["id"]) for cfg in TEACHERS_CFG]

    processed = 0
    for bi in range(n_batches):
        n_this = min(chunk, args.n_windows - processed)
        if n_this <= 0:
            break

        x_batch, y_batch = dataset.sample_batch(n_this, SEQ_BYTES, device=DEVICE, split='train')
        batch_raw = [x_batch[b].tolist() for b in range(n_this)]

        # Student forward
        with torch.no_grad():
            ce_loss, internals = student(x_batch, y_batch, return_internals=True)
            s_logits = internals["logits"].float()
            T_logits = s_logits.shape[1]
            s_logp = F.log_softmax(s_logits, dim=-1)

        # Get per-teacher byte probs
        per_t_logp_at_y = []
        per_t_mask = []
        for t_idx, t in enumerate(teachers):
            targets = _get_teacher_targets_covering_batched(
                t["model"], t["tokenizer"], t["covering"],
                batch_raw, DEVICE, temperature=KD_TEMP, extract_hidden=False, max_depth=8,
            )
            bp = torch.stack([targets[b]["byte_probs"][:T_logits] for b in range(n_this)]).float()
            bm = torch.stack([targets[b]["byte_mask"][:T_logits] for b in range(n_this)])
            # log p_m at actual label y
            y_sl = y_batch[:, :T_logits]
            t_logp_at_y = torch.log(bp.gather(-1, y_sl.unsqueeze(-1)).squeeze(-1).clamp_min(1e-10))
            per_t_logp_at_y.append(t_logp_at_y)
            per_t_mask.append(bm)
            del bp

        # Student log p at actual label
        y_sl = y_batch[:, :T_logits]
        s_logp_at_y = s_logp.gather(-1, y_sl.unsqueeze(-1)).squeeze(-1)

        # Per-teacher advantage at label (no clamp yet — we want signed for flexibility, but positive for regret)
        t_logp_stack = torch.stack(per_t_logp_at_y)  # (M, B, T)
        t_mask_stack = torch.stack(per_t_mask)       # (M, B, T)
        # Mask out invalid positions with very negative value before max
        t_logp_stack_masked = torch.where(t_mask_stack, t_logp_stack, torch.full_like(t_logp_stack, -1e9))
        max_teacher_logp_at_y, _ = t_logp_stack_masked.max(dim=0)  # (B, T)
        # Valid where any teacher has supervision
        any_valid = t_mask_stack.any(dim=0)
        # Regret per position: max(0, max_m t_logp - s_logp)
        regret_pos = (max_teacher_logp_at_y - s_logp_at_y).clamp_min(0.0) * any_valid.float()

        # Per-window scalar: mean regret over valid positions
        valid_counts = any_valid.float().sum(dim=-1).clamp_min(1.0)
        regret_per_window = regret_pos.sum(dim=-1) / valid_counts  # (B,)

        # Horizon-4 regret: chunk positions into 4-byte windows, max regret per chunk, mean over chunks
        # (Approximation of Codex R4 Probe #6 "long-horizon regret" at h=4)
        # Reshape (B, T) → (B, T//4, 4) where T//4 chunks
        T_trunc = (T_logits // 4) * 4
        if T_trunc > 0:
            regret_4 = regret_pos[:, :T_trunc].reshape(-1, T_trunc // 4, 4).max(dim=-1).values.mean(dim=-1)
        else:
            regret_4 = torch.zeros(n_this, device=DEVICE)

        all_regrets.extend(regret_per_window.cpu().tolist())
        all_regret_h4.extend(regret_4.cpu().tolist())
        all_metadata.extend([{"batch_idx": bi, "item_idx": b} for b in range(n_this)])
        # Save raw windows (uint8) for downstream sampler use
        all_x_windows.append(x_batch.cpu().to(torch.uint8))
        all_y_windows.append(y_batch.cpu().to(torch.uint8))

        processed += n_this
        if processed % (chunk * 2) == 0 or processed == args.n_windows:
            print(f"  processed {processed}/{args.n_windows} (mean_regret_so_far={np.mean(all_regrets):.3f})")

    regrets = np.array(all_regrets, dtype=np.float32)
    regrets_h4 = np.array(all_regret_h4, dtype=np.float32)

    # Analysis
    print(f"\n--- Committee Regret Distribution (n={len(regrets)}) ---")
    print(f"Mean R(w):       {regrets.mean():.4f}")
    print(f"Median R(w):     {np.median(regrets):.4f}")
    print(f"Std R(w):        {regrets.std():.4f}")
    print(f"P90:             {np.percentile(regrets, 90):.4f}")
    print(f"P99:             {np.percentile(regrets, 99):.4f}")
    print(f"Max:             {regrets.max():.4f}")
    print(f"Min:             {regrets.min():.4f}")

    # Heavy-tail check
    top10_mean = regrets[regrets >= np.percentile(regrets, 90)].mean()
    top1_mean = regrets[regrets >= np.percentile(regrets, 99)].mean()
    bottom50_mean = regrets[regrets < np.percentile(regrets, 50)].mean()
    concentration_ratio = top10_mean / max(bottom50_mean, 1e-6)
    print(f"Top-10% mean / bottom-50% mean: {concentration_ratio:.2f}x")

    # Decision
    if concentration_ratio >= 3.0:
        decision = "STRONG_HEAVY_TAIL"
        interp = f"Top-10% of windows have {concentration_ratio:.1f}x the regret of bottom-50%. Regret-weighted sampling should meaningfully concentrate useful signal."
    elif concentration_ratio >= 1.5:
        decision = "MODERATE_HEAVY_TAIL"
        interp = f"Modest heavy tail ({concentration_ratio:.1f}x). Regret-weighted sampling may help but not guaranteed."
    else:
        decision = "UNIFORM"
        interp = f"Regret distribution is roughly uniform ({concentration_ratio:.1f}x). Regret-weighted sampling unlikely to help much."

    # Correlation between h=1 and h=4 regret (do they agree?)
    if len(regrets) >= 2 and regrets.std() > 1e-6 and regrets_h4.std() > 1e-6:
        corr_h1_h4 = float(np.corrcoef(regrets, regrets_h4)[0, 1])
    else:
        corr_h1_h4 = float('nan')
    print(f"Correlation R(w) vs R_h4(w): {corr_h1_h4:.3f}")

    results = {
        "probe": "committee_regret_scoring",
        "checkpoint": str(ckpt_path),
        "checkpoint_step": ckpt.get("step", None),
        "n_windows": int(len(regrets)),
        "teachers": [c["id"] for c in TEACHERS_CFG],
        "regret_h1_stats": {
            "mean": float(regrets.mean()),
            "median": float(np.median(regrets)),
            "std": float(regrets.std()),
            "min": float(regrets.min()),
            "max": float(regrets.max()),
            "p90": float(np.percentile(regrets, 90)),
            "p99": float(np.percentile(regrets, 99)),
        },
        "regret_h4_stats": {
            "mean": float(regrets_h4.mean()),
            "median": float(np.median(regrets_h4)),
            "std": float(regrets_h4.std()),
            "p90": float(np.percentile(regrets_h4, 90)),
        },
        "concentration": {
            "top10_vs_bottom50_ratio": float(concentration_ratio),
            "corr_h1_vs_h4": corr_h1_h4,
        },
        "decision": decision,
        "interpretation": interp,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote: {out_json}")

    # Also save per-window scores + raw windows for later use by sampler
    x_all = torch.cat(all_x_windows, dim=0).numpy()  # (N, 1536) uint8
    y_all = torch.cat(all_y_windows, dim=0).numpy()  # (N, 1536) uint8
    np.savez(out_npz,
             regrets_h1=regrets,
             regrets_h4=regrets_h4,
             x_windows=x_all,
             y_windows=y_all)
    print(f"Wrote: {out_npz}  (x_windows shape: {x_all.shape}, dtype: {x_all.dtype})")

    print(f"\nDecision: {decision}")
    print(f"Interpretation: {interp}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default=str(REPO / "results" / "checkpoints_ekalavya_iter5_full_6k" / "best.pt"))
    parser.add_argument("--n-windows", type=int, default=256,
                        help="Number of windows to score (each 1536 bytes). 256 is quick, 2048 is thorough.")
    args = parser.parse_args()
    main(args)
