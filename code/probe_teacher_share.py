"""Teacher-share probe — Codex T+L KM-R3 Probe #3.

Measures what fraction of RRDSD-selected positions Pythia would win under
competitive argmax routing. Offline, no training. Uses iter5 best.pt.

Decision (per Codex R3):
- If Pythia wins <10% of retained positions: SKIP +multi-teacher fork (live Pythia cost
  not justified by its contribution)
- If Pythia wins 10-25%: launch +multi-teacher fork as planned
- If Pythia wins >25%: Pythia is a real co-equal teacher, strong mandate for MT

Usage:
    python code/probe_teacher_share.py [--checkpoint PATH] [--n-windows 32]

Output:
    results/probe_teacher_share.json
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
    SutraDyadS1, SEQ_BYTES, PATCH_SIZE,
    _build_covering_tables,
    _get_teacher_targets_covering_batched,
)
from data_loader import ByteShardedDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
SEED = 42

RRDSD_ZPD_Q_LO = 0.45
RRDSD_ZPD_Q_HI = 0.98
RRDSD_UTIL_MASS = 0.65
RRDSD_DENSITY_MIN = 0.10
RRDSD_DENSITY_MAX = 0.20

TEACHERS_CFG = [
    {"id": "HuggingFaceTB/SmolLM2-1.7B", "role": "anchor"},
    {"id": "EleutherAI/pythia-1.4b", "role": "aux"},
]
KD_TEMP = 1.0


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
        tid, quantization_config=bnb_cfg, device_map={"": DEVICE},
    )
    model.eval()
    cov = _build_covering_tables(tok, max_depth=8)
    return {"id": tid, "tokenizer": tok, "model": model, "covering": cov}


def main(args):
    seed_all(SEED)

    out_path = REPO / "results" / "probe_teacher_share.json"

    # Load student
    ckpt_path = Path(args.checkpoint)
    print(f"Loading student: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    student = SutraDyadS1(max_seq_bytes=SEQ_BYTES)
    student.load_state_dict(ckpt["model"], strict=False)
    student = student.to(DEVICE).eval()
    print(f"  -> step={ckpt.get('step', '?')}, eval_loss={ckpt.get('eval_loss', '?')}")

    # Sample windows
    print(f"Sampling {args.n_windows} windows (seed={SEED})...")
    dataset = ByteShardedDataset()
    x_batch, y_batch = dataset.sample_batch(args.n_windows, SEQ_BYTES, device=DEVICE, split='test')
    print(f"  -> x shape: {tuple(x_batch.shape)}")

    # Student forward → logits
    print("Running student forward...")
    with torch.no_grad():
        ce_loss, internals = student(x_batch, y_batch, return_internals=True)
        s_logits = internals["logits"].float()  # (B, T, 256)
        T_logits = s_logits.shape[1]
        s_logp = F.log_softmax(s_logits, dim=-1)
        s_probs = s_logp.exp()
        s_H = -(s_probs * s_logp).sum(dim=-1)  # (B, T) student entropy

    # Load teachers and get per-teacher byte probs
    print("Loading teachers...")
    teachers = [load_teacher(cfg["id"]) for cfg in TEACHERS_CFG]
    batch_raw = [x_batch[b].tolist() for b in range(args.n_windows)]

    print("\nBuilding per-teacher byte probs...")
    per_t_probs_list = []
    per_t_mask_list = []
    for i, t in enumerate(teachers):
        print(f"  Teacher {i}: {t['id']}...")
        targets = _get_teacher_targets_covering_batched(
            t["model"], t["tokenizer"], t["covering"],
            batch_raw, DEVICE, temperature=KD_TEMP, extract_hidden=False, max_depth=8,
        )
        bp = torch.stack([targets[b]["byte_probs"][:T_logits] for b in range(args.n_windows)]).float()
        bm = torch.stack([targets[b]["byte_mask"][:T_logits] for b in range(args.n_windows)])
        per_t_probs_list.append(bp)
        per_t_mask_list.append(bm)
        # Free teacher GPU
        del t["model"]
        torch.cuda.empty_cache()

    per_t_probs = torch.stack(per_t_probs_list)  # (M, B, T, 256)
    per_t_mask = torch.stack(per_t_mask_list)    # (M, B, T)
    M = per_t_probs.shape[0]
    print(f"  -> per_t_probs shape: {tuple(per_t_probs.shape)}")

    # Compute per-teacher utility A_m × sqrt(D_m + eps)
    print("\nComputing per-teacher utility...")
    with torch.no_grad():
        y_sl = y_batch[:, :T_logits]
        # Advantage
        y_bcast = y_sl.unsqueeze(0).unsqueeze(-1).expand(M, -1, -1, 1)
        t_logp_at_y = torch.log(per_t_probs.gather(-1, y_bcast).squeeze(-1).clamp_min(1e-10))
        s_logp_at_y = s_logp.gather(-1, y_sl.unsqueeze(-1)).squeeze(-1).unsqueeze(0)
        A_m = (t_logp_at_y - s_logp_at_y).clamp_min(0.0) * per_t_mask.float()  # (M, B, T)

        # JSD per teacher
        m_jsd = 0.5 * (per_t_probs + s_probs.unsqueeze(0))
        log_m_jsd = torch.log(m_jsd.clamp_min(1e-10))
        kl_t = (per_t_probs * (torch.log(per_t_probs.clamp_min(1e-10)) - log_m_jsd)).sum(-1)
        kl_s = (s_probs.unsqueeze(0) * (s_logp.unsqueeze(0) - log_m_jsd)).sum(-1)
        D_m = (0.5 * (kl_t + kl_s)).clamp_min(0.0)

        U_m = A_m * torch.sqrt(D_m + 1e-8)  # (M, B, T)
        U_max, m_star = U_m.max(dim=0)      # (B, T)
        joint_mask = per_t_mask.any(dim=0)

    # Apply RRDSD selection (ZPD + cumulative utility mass + density clamp)
    print("Applying RRDSD selection criteria...")
    mask_any = joint_mask
    H_valid = s_H[mask_any]
    q_lo = torch.quantile(H_valid, RRDSD_ZPD_Q_LO)
    q_hi = torch.quantile(H_valid, RRDSD_ZPD_Q_HI)
    zpd_cand = (s_H >= q_lo) & (s_H <= q_hi) & mask_any & (U_max > 0)

    n_valid = mask_any.sum().item()
    density_min = max(1, int(RRDSD_DENSITY_MIN * n_valid))
    density_max = max(density_min + 1, int(RRDSD_DENSITY_MAX * n_valid))

    n_cand = zpd_cand.sum().item()
    if n_cand > 0:
        U_cand = U_max[zpd_cand]
        U_sorted, _ = torch.sort(U_cand, descending=True)
        U_total = U_sorted.sum().item()
        cumsum = torch.cumsum(U_sorted, dim=0)
        mass_idx = int((cumsum >= RRDSD_UTIL_MASS * U_total).float().argmax().item())
        n_keep = max(density_min, min(density_max, mass_idx + 1))
        n_keep = min(n_keep, n_cand)
        flat_U = U_max.flatten()
        flat_cand = zpd_cand.flatten()
        U_masked = torch.where(flat_cand, flat_U, torch.full_like(flat_U, float('-inf')))
        topk_idx = torch.topk(U_masked, n_keep).indices
        final_mask_flat = torch.zeros_like(flat_U, dtype=torch.bool)
        final_mask_flat[topk_idx] = True
        final_mask = final_mask_flat.reshape(U_max.shape)
    else:
        final_mask = torch.zeros_like(mask_any)

    # Compute per-teacher win fractions on FINAL_MASK positions
    print("\nTeacher share analysis...")
    n_selected = final_mask.sum().item()
    per_teacher_wins = {}
    for m_idx, cfg in enumerate(TEACHERS_CFG):
        wins = ((m_star == m_idx) & final_mask).sum().item()
        frac = wins / max(n_selected, 1)
        per_teacher_wins[cfg["id"]] = {
            "wins": wins,
            "fraction": frac,
            "role": cfg["role"],
        }
        print(f"  {cfg['id']} ({cfg['role']}): {wins}/{n_selected} = {frac:.1%}")

    # Decision
    pythia_frac = per_teacher_wins["EleutherAI/pythia-1.4b"]["fraction"]
    if pythia_frac < 0.10:
        decision = "SKIP_MULTI_TEACHER_FORK"
        interp = f"Pythia wins only {pythia_frac:.1%} of RRDSD-selected positions. Live cost not justified. Skip +multi-teacher fork, focus on +CKA and lite-control."
    elif pythia_frac <= 0.25:
        decision = "LAUNCH_MULTI_TEACHER_FORK"
        interp = f"Pythia wins {pythia_frac:.1%}. Moderate contribution. Launch fork as planned."
    else:
        decision = "PYTHIA_STRONG_CO_EQUAL"
        interp = f"Pythia wins {pythia_frac:.1%} — strong complementary signal. Multi-teacher is high-priority next step."

    # Summary stats
    stats = {
        "probe": "teacher_share",
        "checkpoint": str(ckpt_path),
        "checkpoint_step": ckpt.get("step", None),
        "n_windows": args.n_windows,
        "n_valid_positions": int(n_valid),
        "n_zpd_candidates": int(n_cand),
        "n_selected": int(n_selected),
        "density_fraction": n_selected / max(n_valid, 1),
        "rrdsd_config": {
            "zpd_q_lo": RRDSD_ZPD_Q_LO,
            "zpd_q_hi": RRDSD_ZPD_Q_HI,
            "utility_mass": RRDSD_UTIL_MASS,
            "density_min": RRDSD_DENSITY_MIN,
            "density_max": RRDSD_DENSITY_MAX,
        },
        "per_teacher_wins": per_teacher_wins,
        "decision": decision,
        "interpretation": interp,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nWrote: {out_path}")
    print(f"\nDecision: {decision}")
    print(f"Interpretation: {interp}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default=str(REPO / "results" / "checkpoints_ekalavya_iter5_full_6k" / "best.pt"))
    parser.add_argument("--n-windows", type=int, default=32)
    args = parser.parse_args()
    main(args)
