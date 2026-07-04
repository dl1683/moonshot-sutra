"""Sutra Energy Probe: Is the LM readout bottleneck real in production?

Designed by Codex R57. Tests whether frozen Sutra S0 already contains
benchmark-relevant features that byte likelihood cannot read.

Approach:
  1. Load frozen S0 checkpoint
  2. For each MCQ example, run context+candidate through model
  3. Extract h_context_last and h_candidate_pool from hidden states
  4. Train a small MLP energy head: score = MLP([h_ctx, h_cand, h_ctx*h_cand])
  5. Compare energy head accuracy vs LM likelihood accuracy

If frozen energy head gives +5-10pp with little data, the readout bottleneck
is real in Sutra. If not, the toy finding is a toy artifact.

Critical claim boundary (Codex R57):
  Energy head trained on HellaSwag labels is a supervised reranker.
  Do NOT market as "Sutra zero-shot improved."
  The question is: does frozen Sutra CONTAIN the features?
"""

import json
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from s0_architecture import SutraS0, S0Config
from s0_training import TrainConfig
from benchmark_harness import load_hellaswag, load_piqa, load_arc

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = Path("C:/sutra_fast/checkpoints/s0_full/s0_step10000.pt")
RESULTS_DIR = Path("C:/sutra_fast/energy_probe")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --- Energy Head ---

class EnergyHead(nn.Module):
    """MLP energy scorer over frozen hidden states.

    score(context, candidate) = MLP([h_ctx, h_cand, h_ctx * h_cand])
    """
    def __init__(self, d_model, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model * 3, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h_ctx, h_cand):
        combined = torch.cat([h_ctx, h_cand, h_ctx * h_cand], dim=-1)
        return self.net(combined).squeeze(-1)


# --- Hidden State Extraction ---

def extract_hidden_for_choice(model, context_bytes, choice_bytes):
    """Run context+choice through frozen model, extract h_ctx_last and h_cand_pool.

    Returns:
        h_ctx_last: (d_model,) hidden state at last context patch
        h_cand_pool: (d_model,) mean-pooled hidden over candidate patches
        lm_score: float, byte-normalized NLL score (existing method)
    """
    P = model.cfg.patch_size
    max_bytes = model.cfg.max_seq_len * P

    ctx_len = len(context_bytes)
    choice_len = len(choice_bytes)

    if choice_len == 0:
        return None, None, float("inf")

    if choice_len > max_bytes - 2 * P:
        choice_bytes = choice_bytes[:max_bytes - 2 * P]
        choice_len = len(choice_bytes)

    budget = max_bytes - choice_len
    if ctx_len > budget:
        context_bytes = context_bytes[ctx_len - budget:]
        ctx_len = len(context_bytes)

    ctx_pad = (P - (ctx_len % P)) % P
    padded_context = [0] * ctx_pad + context_bytes
    full_seq = padded_context + choice_bytes
    end_pad = (P - (len(full_seq) % P)) % P
    full_seq = full_seq + [0] * end_pad

    if len(full_seq) < 2 * P:
        full_seq = [0] * (2 * P - len(full_seq)) + full_seq

    byte_ids = torch.tensor([full_seq], dtype=torch.long, device=DEVICE)
    B, T = byte_ids.shape
    N = T // P

    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
        with torch.no_grad():
            out = model(byte_ids, return_aux=False)

    hidden = out["hidden"]  # (1, N, d_model)
    logits = out["logits"]  # (1, N-1, P, 256)

    # Context boundary: padded_context ends, choice begins
    ctx_patches = len(padded_context) // P
    # h_ctx_last: last patch of context (this conditions first candidate patch prediction)
    ctx_last_idx = ctx_patches - 1
    h_ctx_last = hidden[0, ctx_last_idx].float()

    # h_cand_pool: mean of hidden states over candidate patch span
    cand_start_patch = ctx_patches
    cand_end_patch = ctx_patches + (choice_len + P - 1) // P
    cand_end_patch = min(cand_end_patch, N)
    if cand_start_patch >= cand_end_patch:
        h_cand_pool = hidden[0, ctx_last_idx].float()
    else:
        h_cand_pool = hidden[0, cand_start_patch:cand_end_patch].float().mean(dim=0)

    # LM score (BPB on candidate bytes)
    targets = byte_ids.reshape(1, N, P)[:, 1:]
    completion_start = len(padded_context)
    completion_end = completion_start + choice_len

    logit_rows = []
    target_list = []
    for flat_pos in range(completion_start, completion_end):
        patch_idx = flat_pos // P
        byte_in_patch = flat_pos % P
        target_patch_idx = patch_idx - 1
        if target_patch_idx < 0 or target_patch_idx >= logits.shape[1]:
            continue
        logit_rows.append(logits[0, target_patch_idx, byte_in_patch])
        target_list.append(targets[0, target_patch_idx, byte_in_patch])

    if len(logit_rows) == 0:
        lm_bpb = float("inf")
    else:
        all_logits = torch.stack(logit_rows).float()
        all_targets = torch.stack(target_list)
        total_nll = F.cross_entropy(all_logits, all_targets, reduction="sum").item()
        lm_bpb = total_nll / (len(logit_rows) * math.log(2))

    return h_ctx_last, h_cand_pool, lm_bpb


# --- Dataset Preparation ---

def prepare_energy_dataset(model, examples, max_examples=0):
    """Extract hidden states for all choices in all examples.

    Returns list of dicts with h_ctx, h_cands (list), lm_scores (list), label.
    """
    dataset = []
    n = len(examples) if max_examples == 0 else min(max_examples, len(examples))

    for i, ex in enumerate(examples[:n]):
        if (i + 1) % 100 == 0:
            print(f"  Processing {i+1}/{n}...", flush=True)

        ctx_bytes = list(ex["context"].encode("utf-8"))
        h_ctxs = []
        h_cands = []
        lm_scores = []

        for choice in ex["choices"]:
            choice_bytes = list(choice.encode("utf-8"))
            h_ctx, h_cand, lm_bpb = extract_hidden_for_choice(model, ctx_bytes, choice_bytes)
            if h_ctx is None:
                h_ctxs.append(torch.zeros(model.cfg.d_model, device=DEVICE))
                h_cands.append(torch.zeros(model.cfg.d_model, device=DEVICE))
                lm_scores.append(999.0)
            else:
                h_ctxs.append(h_ctx)
                h_cands.append(h_cand)
                lm_scores.append(lm_bpb)

        dataset.append({
            "h_ctx": torch.stack(h_ctxs),      # (n_choices, d_model)
            "h_cand": torch.stack(h_cands),    # (n_choices, d_model)
            "lm_scores": lm_scores,
            "label": ex["label"],
        })

    return dataset


# --- Training ---

def train_energy_head(energy_head, train_data, val_data, steps=3000, lr=3e-4, wd=1e-3):
    """Train energy head on MCQ ranking task."""
    opt = torch.optim.AdamW(energy_head.parameters(), lr=lr, weight_decay=wd)
    n_train = len(train_data)
    best_val_acc = 0.0
    best_state = None

    for step in range(1, steps + 1):
        energy_head.train()
        idx = random.randint(0, n_train - 1)
        ex = train_data[idx]

        h_ctx = ex["h_ctx"]    # (n_choices, d_model)
        h_cand = ex["h_cand"]  # (n_choices, d_model)
        label = ex["label"]

        scores = energy_head(h_ctx, h_cand)  # (n_choices,)
        target = torch.tensor(label, dtype=torch.long, device=DEVICE)
        loss = F.cross_entropy(scores.unsqueeze(0), target.unsqueeze(0))

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 500 == 0:
            val_acc = evaluate_energy_head(energy_head, val_data)
            print(f"    Step {step}: val_acc={val_acc*100:.1f}%", flush=True)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.clone() for k, v in energy_head.state_dict().items()}

    if best_state is not None:
        energy_head.load_state_dict(best_state)
    return best_val_acc


def evaluate_energy_head(energy_head, data):
    """Evaluate energy head accuracy on MCQ data."""
    energy_head.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for ex in data:
            scores = energy_head(ex["h_ctx"], ex["h_cand"])
            pred = scores.argmax().item()
            if pred == ex["label"]:
                correct += 1
            total += 1
    return correct / max(total, 1)


def evaluate_lm_baseline(data):
    """Evaluate LM likelihood baseline accuracy."""
    correct = 0
    total = 0
    for ex in data:
        pred = min(range(len(ex["lm_scores"])), key=lambda i: ex["lm_scores"][i])
        if pred == ex["label"]:
            correct += 1
        total += 1
    return correct / max(total, 1)


# --- Main ---

def main():
    print("="*60)
    print("  Sutra Energy Probe: Frozen Readout Bottleneck Test")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")

    # Load model
    print("\nLoading S0 checkpoint...")
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model_cfg = ckpt["model_cfg"]
    model = SutraS0(model_cfg).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    step = ckpt.get("step", "?")
    d_model = model_cfg.d_model
    print(f"Loaded step {step}, d_model={d_model}, eval_bpb={ckpt.get('eval_bpb', '?')}")

    # Load benchmarks
    print("\nLoading HellaSwag...")
    hellaswag = load_hellaswag("validation")
    random.seed(42)
    random.shuffle(hellaswag)

    # Split: train 2000, val 1000, test 1000
    train_size = 2000
    val_size = 1000
    test_size = 1000

    hs_train_raw = hellaswag[:train_size]
    hs_val_raw = hellaswag[train_size:train_size + val_size]
    hs_test_raw = hellaswag[train_size + val_size:train_size + val_size + test_size]

    # Extract hidden states
    print(f"\nExtracting hidden states (train={train_size})...")
    train_data = prepare_energy_dataset(model, hs_train_raw, max_examples=train_size)
    print(f"Extracting hidden states (val={val_size})...")
    val_data = prepare_energy_dataset(model, hs_val_raw, max_examples=val_size)
    print(f"Extracting hidden states (test={test_size})...")
    test_data = prepare_energy_dataset(model, hs_test_raw, max_examples=test_size)

    # LM baseline
    print("\nLM Baseline (byte-normalized BPB):")
    lm_train_acc = evaluate_lm_baseline(train_data)
    lm_val_acc = evaluate_lm_baseline(val_data)
    lm_test_acc = evaluate_lm_baseline(test_data)
    print(f"  Train: {lm_train_acc*100:.1f}%")
    print(f"  Val:   {lm_val_acc*100:.1f}%")
    print(f"  Test:  {lm_test_acc*100:.1f}%")

    results = {
        "lm_baseline": {"train": lm_train_acc, "val": lm_val_acc, "test": lm_test_acc},
    }

    # Train energy head
    print("\n--- Training Energy Head (full 2000 examples) ---")
    energy_head = EnergyHead(d_model, hidden_dim=256).to(DEVICE)
    best_val = train_energy_head(energy_head, train_data, val_data, steps=5000)
    test_acc = evaluate_energy_head(energy_head, test_data)
    print(f"  Energy Head Test: {test_acc*100:.1f}% (best val: {best_val*100:.1f}%)")
    results["energy_full"] = {"val": best_val, "test": test_acc}

    # Sample efficiency curve
    print("\n--- Sample Efficiency ---")
    sizes = [100, 250, 500, 1000, 2000]
    eff_results = []
    for n in sizes:
        subset = train_data[:n]
        head = EnergyHead(d_model, hidden_dim=256).to(DEVICE)
        best = train_energy_head(head, subset, val_data, steps=3000)
        test_n = evaluate_energy_head(head, test_data)
        print(f"  n={n}: val={best*100:.1f}%, test={test_n*100:.1f}%")
        eff_results.append({"n": n, "val": best, "test": test_n})
    results["sample_efficiency"] = eff_results

    # Shuffled label control
    print("\n--- Shuffled Label Control ---")
    shuffled_train = []
    labels = [ex["label"] for ex in train_data]
    random.shuffle(labels)
    for ex, lbl in zip(train_data, labels):
        shuffled_train.append({**ex, "label": lbl})
    head_shuffled = EnergyHead(d_model, hidden_dim=256).to(DEVICE)
    shuffled_val = train_energy_head(head_shuffled, shuffled_train, val_data, steps=3000)
    print(f"  Shuffled control: val={shuffled_val*100:.1f}% (chance=25%)")
    results["shuffled_control"] = shuffled_val

    # Summary
    print("\n" + "="*60)
    print("  FINAL SUMMARY")
    print("="*60)
    print(f"  LM Baseline (test):     {lm_test_acc*100:.1f}%")
    print(f"  Energy Head (test):     {test_acc*100:.1f}%")
    print(f"  Gain:                   {(test_acc - lm_test_acc)*100:+.1f}pp")
    print(f"  Shuffled control:       {shuffled_val*100:.1f}%")
    print()

    gap = test_acc - lm_test_acc
    if gap >= 0.05:
        print("  VERDICT: READOUT BOTTLENECK IS REAL IN SUTRA")
        print(f"  Frozen energy head gains +{gap*100:.1f}pp over LM likelihood.")
        print("  The model contains benchmark-relevant features that byte scoring cannot read.")
    elif gap >= 0.02:
        print("  VERDICT: MODEST READOUT GAP")
        print(f"  Energy head gains +{gap*100:.1f}pp. Some features are hidden but gap is not dramatic.")
    else:
        print("  VERDICT: NO SIGNIFICANT READOUT BOTTLENECK")
        print("  The toy finding does not transfer to real Sutra at this scale.")
        print("  LM likelihood is an adequate readout for current representations.")

    results["verdict"] = {
        "gap_pp": gap * 100,
        "significant": gap >= 0.05,
    }

    # Save
    with open(RESULTS_DIR / "energy_probe_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_DIR / 'energy_probe_results.json'}")


if __name__ == "__main__":
    main()
